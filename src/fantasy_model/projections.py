"""Point-in-time, all-player preseason projection layer.

This module deliberately stays simple enough to backtest.  Veteran estimates use
only the three seasons before ``target_season`` and the league's exact scoring.
Rookies use the separately validated draft-capital model, joined by ESPN id.
ADP is carried as market context but is not used to manufacture projected points.

The distribution is a first-pass marginal distribution, not a claim that weekly
scores are normal or independent.  See ``ProjectionAssumptions`` for the explicit
conversion from active-game production to full-season moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .backtest import POSITIONS, add_league_points


MFL_POSITIONS = (*POSITIONS, "PK", "Def")
RECENCY_WEIGHTS = {1: 1.0, 2: 0.55, 3: 0.30}


@dataclass(frozen=True)
class ProjectionAssumptions:
    """Transparent constants used by the bounded baseline.

    ``season_weeks`` is the number of NFL team games whose fantasy production is
    represented.  Rookie draft-capital predictions were trained on weeks 1-14,
    so their season totals are extrapolated from that fixed window.

    Veteran PPG is shrunk by ``production_prior_games`` toward the prior-season
    positional median. Expected games is shrunk by ``availability_prior_seasons``
    toward a conservative positional availability assumption.  Season standard
    deviation uses an independent-week square-root-of-time approximation; a
    downstream simulator should replace that approximation with explicit injury
    and role scenarios when those inputs are available.
    """

    season_weeks: int = 17
    rookie_baseline_weeks: int = 14
    production_prior_games: float = 6.0
    availability_prior_seasons: float = 1.0
    qb_expected_games: float = 15.0
    rb_expected_games: float = 14.0
    wr_expected_games: float = 14.5
    te_expected_games: float = 14.5
    pk_expected_games: float = 16.0
    defense_weekly_mean: float = 6.0
    defense_weekly_sd: float = 4.5

    def expected_games_prior(self, position: str) -> float:
        return {
            "QB": self.qb_expected_games,
            "RB": self.rb_expected_games,
            "WR": self.wr_expected_games,
            "TE": self.te_expected_games,
            "PK": self.pk_expected_games,
            "Def": float(self.season_weeks),
        }[position]


def _stat(frame: pl.DataFrame, name: str) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name).fill_null(0).cast(pl.Float64)
    return pl.lit(0.0)


def add_kicker_points(frame: pl.DataFrame) -> pl.DataFrame:
    """Add this league's distance-aware kicker points to nflverse weekly rows."""
    points = (
        (
            _stat(frame, "fg_made_0_19")
            + _stat(frame, "fg_made_20_29")
            + _stat(frame, "fg_made_30_39")
        )
        * 3.0
        + _stat(frame, "fg_made_40_49") * 4.0
        + _stat(frame, "fg_made_50_59") * 5.0
        + _stat(frame, "fg_made_60_") * 6.0
        - (
            _stat(frame, "fg_missed_0_19")
            + _stat(frame, "fg_missed_20_29")
            + _stat(frame, "fg_missed_30_39")
        )
        * 3.0
        - _stat(frame, "fg_missed_40_49") * 2.0
        - _stat(frame, "fg_missed_50_59")
        + _stat(frame, "pat_made")
    )
    return frame.with_columns(points.round(2).alias("kicker_points"))


def add_projection_points(frame: pl.DataFrame) -> pl.DataFrame:
    """Score offensive and kicker rows under the exact league rules."""
    scored = add_kicker_points(add_league_points(frame))
    normalized_position = (
        pl.when(pl.col("position") == "K")
        .then(pl.lit("PK"))
        .otherwise(pl.col("position"))
    )
    return scored.with_columns(
        normalized_position.alias("position"),
        pl.when(pl.col("position") == "K")
        .then(pl.col("kicker_points"))
        .otherwise(pl.col("league_points"))
        .alias("projection_points"),
    )


def _validate_unique_nonempty(frame: pl.DataFrame, column: str, label: str) -> None:
    duplicate = (
        frame.filter(pl.col(column).is_not_null() & (pl.col(column) != ""))
        .group_by(column)
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate.height:
        raise ValueError(
            f"{label} contains duplicate {column} values: "
            f"{duplicate[column].head(10).to_list()}"
        )


def _historical_projection_inputs(
    stats: pl.DataFrame,
    *,
    target_season: int,
    assumptions: ProjectionAssumptions,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return veteran moments and positional priors using only prior seasons."""
    history = (
        stats.filter(
            (pl.col("season") < target_season)
            & (pl.col("season") >= target_season - 3)
            & (pl.col("season_type") == "REG")
            & pl.col("position").is_in((*POSITIONS, "K", "PK"))
        )
        .pipe(add_projection_points)
        .with_columns(
            (target_season - pl.col("season")).alias("years_ago")
        )
        .with_columns(
            pl.col("years_ago")
            .replace_strict(RECENCY_WEIGHTS)
            .cast(pl.Float64)
            .alias("recency_weight")
        )
    )
    if history.is_empty():
        return (
            pl.DataFrame(
                schema={
                    "nflverse_player_id": pl.String,
                    "historical_active_ppg": pl.Float64,
                    "historical_active_sd": pl.Float64,
                    "expected_games": pl.Float64,
                }
            ),
            pl.DataFrame(
                schema={
                    "position": pl.String,
                    "position_prior_ppg": pl.Float64,
                    "position_prior_sd": pl.Float64,
                }
            ),
        )

    player_seasons = history.group_by("player_id", "position", "season").agg(
        pl.len().alias("games"),
        pl.col("projection_points").mean().alias("ppg"),
        pl.col("projection_points").std().fill_null(0.0).alias("weekly_sd"),
    )
    latest_season = player_seasons.filter(pl.col("season") == target_season - 1)
    priors = latest_season.group_by("position").agg(
        pl.col("ppg").median().alias("position_prior_ppg"),
        pl.col("weekly_sd").median().alias("position_prior_sd"),
    )

    # Active-game production moments. Recency is attached per game so a full
    # recent season receives more evidence than a cameo or an older season.
    moments = history.group_by("player_id", "position").agg(
        (pl.col("projection_points") * pl.col("recency_weight"))
        .sum()
        .alias("weighted_points"),
        (pl.col("projection_points").pow(2) * pl.col("recency_weight"))
        .sum()
        .alias("weighted_squares"),
        pl.col("recency_weight").sum().alias("weighted_games"),
    )
    availability = (
        player_seasons.with_columns(
            (target_season - pl.col("season")).alias("years_ago")
        )
        .with_columns(
            pl.col("years_ago")
            .replace_strict(RECENCY_WEIGHTS)
            .cast(pl.Float64)
            .alias("recency_weight")
        )
        .group_by("player_id")
        .agg(
            (pl.col("games") * pl.col("recency_weight"))
            .sum()
            .alias("weighted_season_games"),
            pl.col("recency_weight").sum().alias("weighted_seasons"),
        )
    )

    position_game_priors = pl.DataFrame(
        {
            "position": ["QB", "RB", "WR", "TE", "PK"],
            "expected_games_prior": [
                assumptions.expected_games_prior(position)
                for position in ("QB", "RB", "WR", "TE", "PK")
            ],
        }
    )
    projected = (
        moments.join(priors, on="position", how="left", validate="m:1")
        .join(position_game_priors, on="position", how="left", validate="m:1")
        .join(availability, on="player_id", how="left", validate="1:1")
        .with_columns(
            pl.col("position_prior_ppg").fill_null(0.0),
            pl.col("position_prior_sd").fill_null(0.0),
        )
        .with_columns(
            (
                (
                    pl.col("weighted_points")
                    + pl.col("position_prior_ppg")
                    * assumptions.production_prior_games
                )
                /
                (pl.col("weighted_games") + assumptions.production_prior_games)
            ).alias("historical_active_ppg"),
            (
                (
                    pl.col("weighted_squares")
                    + (
                        pl.col("position_prior_sd").pow(2)
                        + pl.col("position_prior_ppg").pow(2)
                    )
                    * assumptions.production_prior_games
                )
                /
                (pl.col("weighted_games") + assumptions.production_prior_games)
            ).alias("historical_second_moment"),
            (
                (
                    pl.col("weighted_season_games")
                    + pl.col("expected_games_prior")
                    * assumptions.availability_prior_seasons
                )
                /
                (
                    pl.col("weighted_seasons")
                    + assumptions.availability_prior_seasons
                )
            )
            .clip(0.0, float(assumptions.season_weeks))
            .alias("expected_games"),
        )
        .with_columns(
            (
                pl.col("historical_second_moment")
                - pl.col("historical_active_ppg").pow(2)
            )
            .clip(0.0, None)
            .sqrt()
            .alias("historical_active_sd")
        )
        .select(
            pl.col("player_id").alias("nflverse_player_id"),
            "historical_active_ppg",
            "historical_active_sd",
            "expected_games",
        )
    )
    return projected, priors


def build_all_player_projections(
    mfl_players: pl.DataFrame,
    mfl_adp: pl.DataFrame,
    historical_stats: pl.DataFrame,
    nflverse_players: pl.DataFrame,
    rookie_rankings: pl.DataFrame,
    *,
    target_season: int = 2026,
    assumptions: ProjectionAssumptions | None = None,
) -> pl.DataFrame:
    """Build one conservative distribution row for every passed MFL player.

    Stable ESPN identifiers are mandatory for rookie and veteran joins. Players
    lacking a stable match receive a zero-production prior (except defenses,
    whose individual history is absent from nflverse player stats and therefore
    receives a clearly labelled neutral team-defense prior).
    """
    assumptions = assumptions or ProjectionAssumptions()
    required_mfl = {
        "mfl_player_id",
        "espn_id",
        "mfl_display_name",
        "mfl_position",
        "mfl_team",
    }
    missing = required_mfl - set(mfl_players.columns)
    if missing:
        raise ValueError(f"MFL players frame is missing columns: {sorted(missing)}")
    if assumptions.season_weeks <= 0 or assumptions.rookie_baseline_weeks <= 0:
        raise ValueError("Projection week counts must be positive")

    universe = (
        mfl_players.filter(pl.col("mfl_position").is_in(MFL_POSITIONS))
        .select(*required_mfl)
        .with_columns(
            pl.col("mfl_player_id").cast(pl.String),
            pl.col("espn_id").cast(pl.String).fill_null(""),
        )
    )
    _validate_unique_nonempty(universe, "mfl_player_id", "MFL universe")

    identity = (
        nflverse_players.select(
            pl.col("gsis_id").cast(pl.String).alias("nflverse_player_id"),
            pl.col("espn_id").cast(pl.String).fill_null("").alias("espn_id"),
        )
        .filter(pl.col("nflverse_player_id").is_not_null())
    )
    _validate_unique_nonempty(identity, "espn_id", "nflverse player identity")
    identity = identity.filter(pl.col("espn_id") != "")

    rookie = rookie_rankings.select(
        pl.col("espn_id").cast(pl.String).fill_null("").alias("espn_id"),
        pl.col("draft_capital").cast(pl.Float64).alias("rookie_baseline_points"),
    ).filter(pl.col("espn_id") != "")
    _validate_unique_nonempty(rookie, "espn_id", "rookie rankings")

    veteran, priors = _historical_projection_inputs(
        historical_stats,
        target_season=target_season,
        assumptions=assumptions,
    )
    prior_sd = {
        row["position"]: float(row["position_prior_sd"] or 0.0)
        for row in priors.iter_rows(named=True)
    }
    position_prior_sd = pl.col("mfl_position").replace_strict(
        prior_sd, default=0.0, return_dtype=pl.Float64
    )

    adp_columns = ["mfl_player_id"]
    for candidate in ("average_pick", "adp_rank"):
        if candidate in mfl_adp.columns:
            adp_columns.append(candidate)
    adp = mfl_adp.select(adp_columns).with_columns(
        pl.col("mfl_player_id").cast(pl.String)
    )
    _validate_unique_nonempty(adp, "mfl_player_id", "MFL ADP")

    joined = (
        universe.join(identity, on="espn_id", how="left", validate="m:1")
        .join(veteran, on="nflverse_player_id", how="left", validate="m:1")
        .join(rookie, on="espn_id", how="left", validate="m:1")
        .join(adp, on="mfl_player_id", how="left", validate="1:1")
    )
    if "average_pick" not in joined.columns:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("average_pick"))
    if "adp_rank" not in joined.columns:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Int64).alias("adp_rank"))

    is_rookie = pl.col("rookie_baseline_points").is_not_null()
    has_history = pl.col("historical_active_ppg").is_not_null()
    is_defense = pl.col("mfl_position") == "Def"

    # Rookies' predicted weeks 1-14 points already include role and availability.
    # Veteran weekly mean is made unconditional by multiplying active PPG by the
    # projected availability rate. Unknowns stay at zero rather than receiving an
    # unjustified sleeper boost.
    mean_weekly = (
        pl.when(is_rookie)
        .then(pl.col("rookie_baseline_points") / assumptions.rookie_baseline_weeks)
        .when(has_history)
        .then(
            pl.col("historical_active_ppg")
            * pl.col("expected_games")
            / assumptions.season_weeks
        )
        .when(is_defense)
        .then(assumptions.defense_weekly_mean)
        .otherwise(0.0)
    )
    weekly_sd = (
        pl.when(is_rookie)
        .then(
            pl.max_horizontal(
                position_prior_sd,
                mean_weekly
                * pl.col("mfl_position").replace_strict(
                    {"QB": 0.40, "RB": 0.80, "WR": 0.85, "TE": 0.85},
                    default=0.75,
                    return_dtype=pl.Float64,
                ),
            )
        )
        .when(has_history)
        .then(
            (
                (pl.col("expected_games") / assumptions.season_weeks)
                * (
                    pl.col("historical_active_sd").pow(2)
                    + pl.col("historical_active_ppg").pow(2)
                )
                - mean_weekly.pow(2)
            )
            .clip(0.0, None)
            .sqrt()
        )
        .when(is_defense)
        .then(assumptions.defense_weekly_sd)
        .otherwise(0.0)
    )

    return (
        joined.with_columns(
            mean_weekly.round(4).alias("mean_weekly_points"),
            weekly_sd.round(4).alias("sd_weekly_points"),
            pl.when(is_rookie)
            .then(pl.lit("rookie_draft_capital"))
            .when(has_history)
            .then(pl.lit("veteran_recency_shrink"))
            .when(is_defense)
            .then(pl.lit("defense_neutral_prior"))
            .otherwise(pl.lit("unmatched_zero_prior"))
            .alias("source"),
        )
        .with_columns(
            (pl.col("mean_weekly_points") * assumptions.season_weeks)
            .round(2)
            .alias("mean_season_points"),
            (pl.col("sd_weekly_points") * assumptions.season_weeks**0.5)
            .round(2)
            .alias("sd_season_points"),
        )
        .select(
            "mfl_player_id",
            pl.col("mfl_display_name").alias("name"),
            pl.col("mfl_position").alias("position"),
            pl.col("mfl_team").alias("team"),
            "mean_weekly_points",
            "mean_season_points",
            "sd_weekly_points",
            "sd_season_points",
            "source",
            pl.col("average_pick").cast(pl.Float64).alias("adp"),
            "adp_rank",
            pl.col("expected_games").round(2),
            "espn_id",
            "nflverse_player_id",
        )
        .sort(
            ["mean_season_points", "adp"],
            descending=[True, False],
            nulls_last=True,
        )
    )
