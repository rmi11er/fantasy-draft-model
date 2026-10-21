import polars as pl

from fantasy_model.projections import (
    ProjectionAssumptions,
    add_projection_points,
    build_all_player_projections,
)


def _mfl_players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mfl_player_id": ["v1", "r1", "k1", "d1", "u1"],
            "espn_id": ["101", "202", "303", "", "404"],
            "mfl_display_name": [
                "Veteran QB",
                "Rookie Runner",
                "Kicker One",
                "Team Defense",
                "Unknown Receiver",
            ],
            "mfl_position": ["QB", "RB", "PK", "Def", "WR"],
            "mfl_team": ["NE", "ARI", "KC", "SF", "FA"],
        }
    )


def _identities() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["g-vet", "g-rook", "g-kick", "g-unknown"],
            "espn_id": ["101", "202", "303", "404"],
        }
    )


def _stats(*, include_future: bool = False) -> pl.DataFrame:
    rows = []
    for season, passing_yards in ((2023, 200), (2024, 250), (2025, 300)):
        for week in (1, 2):
            rows.append(
                {
                    "player_id": "g-vet",
                    "position": "QB",
                    "season": season,
                    "season_type": "REG",
                    "week": week,
                    "passing_yards": passing_yards,
                    "passing_tds": 2,
                }
            )
    for season in (2024, 2025):
        rows.append(
            {
                "player_id": "g-kick",
                "position": "K",
                "season": season,
                "season_type": "REG",
                "week": 1,
                "fg_made_40_49": 1,
                "fg_made_50_59": 1,
                "fg_missed_0_19": 1,
                "pat_made": 2,
            }
        )
    if include_future:
        rows.append(
            {
                "player_id": "g-vet",
                "position": "QB",
                "season": 2026,
                "season_type": "REG",
                "week": 1,
                "passing_yards": 999,
                "passing_tds": 10,
            }
        )
    return pl.DataFrame(rows)


def _rookies() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "espn_id": ["202"],
            # Validated rookie model predicts total points over weeks 1-14.
            "draft_capital": [140.0],
        }
    )


def _adp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mfl_player_id": ["r1", "v1"],
            "average_pick": [3.5, 22.0],
            "adp_rank": [4, 20],
        }
    )


def _project(stats: pl.DataFrame) -> pl.DataFrame:
    return build_all_player_projections(
        _mfl_players(),
        _adp(),
        stats,
        _identities(),
        _rookies(),
        target_season=2026,
        assumptions=ProjectionAssumptions(),
    )


def test_exact_kicker_distance_scoring_is_used() -> None:
    scored = add_projection_points(
        pl.DataFrame(
            {
                "position": ["K"],
                "fg_made_40_49": [1],
                "fg_made_50_59": [1],
                "fg_missed_0_19": [1],
                "pat_made": [2],
            }
        )
    )
    assert scored["position"].item() == "PK"
    assert scored["projection_points"].item() == 8.0


def test_all_player_projection_sources_and_distribution_contract() -> None:
    projected = _project(_stats())
    assert projected.height == 5
    assert {
        "mfl_player_id",
        "name",
        "position",
        "team",
        "mean_weekly_points",
        "mean_season_points",
        "sd_weekly_points",
        "sd_season_points",
        "season_uncertainty_method",
        "source",
        "adp",
    } <= set(projected.columns)

    rookie = projected.filter(pl.col("mfl_player_id") == "r1").row(0, named=True)
    assert rookie["source"] == "rookie_draft_capital"
    assert rookie["mean_weekly_points"] == 10.0
    assert rookie["mean_season_points"] == 170.0
    assert rookie["sd_season_points"] == 65.07
    assert rookie["season_uncertainty_method"] == "rookie_walk_forward_residual_band"
    assert rookie["adp"] == 3.5

    veteran = projected.filter(pl.col("mfl_player_id") == "v1").row(0, named=True)
    kicker = projected.filter(pl.col("mfl_player_id") == "k1").row(0, named=True)
    defense = projected.filter(pl.col("mfl_player_id") == "d1").row(0, named=True)
    unknown = projected.filter(pl.col("mfl_player_id") == "u1").row(0, named=True)
    assert veteran["source"] == "veteran_recency_shrink"
    assert veteran["mean_season_points"] > 0
    assert veteran["season_uncertainty_method"] == "veteran_conformal_2020_2025"
    assert kicker["source"] == "veteran_recency_shrink"
    assert kicker["mean_season_points"] > 0
    assert defense["source"] == "defense_neutral_prior"
    assert defense["mean_weekly_points"] == 6.0
    assert unknown["source"] == "unmatched_zero_prior"
    assert unknown["mean_season_points"] == 0.0


def test_target_season_stats_are_never_used() -> None:
    baseline = _project(_stats(include_future=False))
    with_future = _project(_stats(include_future=True))
    columns = [
        "mfl_player_id",
        "mean_weekly_points",
        "mean_season_points",
        "sd_weekly_points",
        "sd_season_points",
    ]
    assert baseline.select(columns).sort("mfl_player_id").equals(
        with_future.select(columns).sort("mfl_player_id")
    )


def test_rookie_join_requires_unambiguous_stable_id() -> None:
    duplicate_rookies = pl.DataFrame(
        {"espn_id": ["202", "202"], "draft_capital": [100.0, 120.0]}
    )
    try:
        build_all_player_projections(
            _mfl_players(),
            _adp(),
            _stats(),
            _identities(),
            duplicate_rookies,
        )
    except ValueError as error:
        assert "duplicate espn_id" in str(error)
    else:
        raise AssertionError("ambiguous rookie mapping should be rejected")
