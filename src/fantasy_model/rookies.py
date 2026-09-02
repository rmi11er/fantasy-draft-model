from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .backtest import (
    POSITION_STARTER_CUTOFF,
    POSITIONS,
    add_league_points,
    seasonal_player_table,
)
from .data import load_player_stats, load_players


CAPITAL_FEATURES = ["log_draft_pick"]
CONTEXT_FEATURES = [
    "log_draft_pick",
    "age_at_draft",
    "height",
    "weight",
    "prior_team_position_points",
]


def _rookie_bio(players: pl.DataFrame) -> pl.DataFrame:
    return (
        players.filter(
            pl.col("position").is_in(POSITIONS)
            & pl.col("draft_year").is_not_null()
            & pl.col("draft_pick").is_not_null()
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            "espn_id",
            "display_name",
            "position",
            "draft_year",
            "draft_pick",
            "draft_team",
            "birth_date",
            "height",
            "weight",
        )
        .with_columns(
            pl.col("birth_date").str.to_date(strict=False).alias("birth_date_parsed"),
            pl.col("draft_pick").cast(pl.Float64).log().alias("log_draft_pick"),
        )
        .with_columns(
            (
                (
                    pl.date(pl.col("draft_year"), pl.lit(9), pl.lit(1))
                    - pl.col("birth_date_parsed")
                ).dt.total_days()
                / 365.25
            ).alias("age_at_draft")
        )
    )


def _team_context(stats: pl.DataFrame) -> pl.DataFrame:
    regular = stats.filter(
        pl.col("position").is_in(POSITIONS)
        & (pl.col("season_type") == "REG")
        & (pl.col("week") <= 14)
    )
    return (
        add_league_points(regular)
        .group_by("season", "team", "position")
        .agg(pl.col("league_points").sum().alias("prior_team_position_points"))
        .with_columns((pl.col("season") + 1).alias("draft_year"))
        .rename({"team": "draft_team"})
        .select(
            "draft_year", "draft_team", "position", "prior_team_position_points"
        )
    )


def rookie_outcomes(players: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    bios = _rookie_bio(players)
    seasonal = seasonal_player_table(stats)
    actual_ranks = seasonal.with_columns(
        pl.col("points")
        .rank("ordinal", descending=True)
        .over("season", "position")
        .alias("actual_position_rank")
    ).select(
        "season",
        "player_id",
        "points",
        "ppg",
        "games",
        "weekly_sd",
        "actual_position_rank",
    )
    return (
        bios.join(
            actual_ranks,
            left_on=["draft_year", "player_id"],
            right_on=["season", "player_id"],
            how="left",
        )
        .join(
            _team_context(stats),
            on=["draft_year", "draft_team", "position"],
            how="left",
        )
        .with_columns(
            pl.col("points").fill_null(0).alias("actual_points"),
            pl.col("ppg").fill_null(0).alias("actual_ppg"),
            pl.col("games").fill_null(0).alias("actual_games"),
            pl.col("weekly_sd").fill_null(0).alias("actual_weekly_sd"),
            pl.col("actual_position_rank").fill_null(999),
            pl.col("prior_team_position_points").fill_null(
                pl.col("prior_team_position_points").median().over("position")
            ),
        )
        .with_columns(
            (
                pl.col("actual_position_rank")
                <= pl.col("position").replace_strict(POSITION_STARTER_CUTOFF)
            ).alias("starter_hit")
        )
    )


def rookie_features(
    players: pl.DataFrame, stats: pl.DataFrame, target_season: int
) -> pl.DataFrame:
    return (
        _rookie_bio(players)
        .filter(pl.col("draft_year") == target_season)
        .join(
            _team_context(stats),
            on=["draft_year", "draft_team", "position"],
            how="left",
        )
        .with_columns(
            pl.col("prior_team_position_points").fill_null(
                pl.col("prior_team_position_points").median().over("position")
            )
        )
    )


def _pipeline(kind: str) -> Pipeline:
    numeric_features = CAPITAL_FEATURES if kind == "draft_capital" else CONTEXT_FEATURES
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    transform = ColumnTransformer(
        [
            ("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["position"]),
            ("numeric", numeric, numeric_features),
        ]
    )
    estimator = (
        Ridge(alpha=18.0)
        if kind == "draft_capital"
        else HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=250,
            max_depth=3,
            min_samples_leaf=18,
            l2_regularization=12.0,
            random_state=50473,
        )
    )
    return Pipeline([("features", transform), ("model", estimator)])


def _to_model_frame(frame: pl.DataFrame):
    return frame.select(
        "position", *CONTEXT_FEATURES
    ).to_pandas(use_pyarrow_extension_array=False)


@dataclass(frozen=True)
class RookieMetric:
    model: str
    season: int
    rookies: int
    mae: float
    rmse: float
    correlation: float
    top12_recall: float
    starter_hit_recall: float


def _evaluate(
    test: pl.DataFrame, predictions: np.ndarray, model_name: str, season: int
) -> RookieMetric:
    actual = test["actual_points"].to_numpy()
    error = predictions - actual
    top_count = min(12, len(actual))
    actual_top = set(np.argsort(actual)[-top_count:])
    predicted_top = set(np.argsort(predictions)[-top_count:])
    hits = test["starter_hit"].to_numpy().astype(bool)
    hit_count = int(hits.sum())
    predicted_hits = set(np.argsort(predictions)[-hit_count:]) if hit_count else set()
    actual_hits = set(np.flatnonzero(hits))
    return RookieMetric(
        model=model_name,
        season=season,
        rookies=len(actual),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        correlation=float(np.corrcoef(predictions, actual)[0, 1]),
        top12_recall=len(actual_top & predicted_top) / max(top_count, 1),
        starter_hit_recall=len(actual_hits & predicted_hits) / max(hit_count, 1),
    )


def walk_forward_rookie_backtest(
    outcomes: pl.DataFrame, start_season: int, end_season: int
) -> tuple[list[RookieMetric], pl.DataFrame]:
    metrics: list[RookieMetric] = []
    rows: list[pl.DataFrame] = []
    for season in range(start_season, end_season + 1):
        train = outcomes.filter(pl.col("draft_year") < season)
        test = outcomes.filter(pl.col("draft_year") == season)
        if test.is_empty():
            raise ValueError(f"No rookie outcomes found for {season}")
        x_train = _to_model_frame(train)
        x_test = _to_model_frame(test)
        y_train = train["actual_points"].to_numpy()
        season_result = test.select(
            "player_id", "display_name", "position", "draft_year", "actual_points"
        )
        for kind in ("draft_capital", "context_model"):
            model = _pipeline(kind)
            model.fit(x_train, y_train)
            prediction = np.maximum(0.0, model.predict(x_test))
            metrics.append(_evaluate(test, prediction, kind, season))
            season_result = season_result.with_columns(
                pl.Series(f"{kind}_prediction", prediction)
            )
        rows.append(season_result)
    return metrics, pl.concat(rows, how="vertical")


def rank_current_rookies(
    outcomes: pl.DataFrame,
    current: pl.DataFrame,
    target_season: int,
) -> pl.DataFrame:
    train = outcomes.filter(pl.col("draft_year") < target_season)
    x_train = _to_model_frame(train)
    x_current = _to_model_frame(current)
    y_train = train["actual_points"].to_numpy()
    result = current.select(
        "player_id",
        "espn_id",
        "display_name",
        "position",
        "draft_pick",
        "draft_team",
        "age_at_draft",
        "prior_team_position_points",
    )
    for kind in ("draft_capital", "context_model"):
        model = _pipeline(kind)
        model.fit(x_train, y_train)
        result = result.with_columns(
            pl.Series(kind, np.maximum(0.0, model.predict(x_current)))
        )
    return (
        result.with_columns(
            (pl.col("context_model") - pl.col("draft_capital")).alias(
                "context_delta"
            )
        )
        .sort("draft_capital", descending=True)
        .with_row_index("rookie_rank", offset=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward rookie model and current rookie rankings"
    )
    parser.add_argument("--train-start", type=int, default=2006)
    parser.add_argument("--backtest-start", type=int, default=2020)
    parser.add_argument("--backtest-end", type=int, default=2025)
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/backtests")
    )
    args = parser.parse_args()

    stats = load_player_stats(
        args.cache_dir, range(args.train_start, args.backtest_end + 1)
    )
    players = load_players(args.cache_dir)
    outcomes = rookie_outcomes(players, stats).filter(
        pl.col("draft_year") >= args.train_start
    )
    metrics, predictions = walk_forward_rookie_backtest(
        outcomes, args.backtest_start, args.backtest_end
    )
    current = rookie_features(players, stats, args.target_season)
    rankings = rank_current_rookies(outcomes, current, args.target_season)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([metric.__dict__ for metric in metrics]).write_csv(
        args.output_dir / "rookie_metrics.csv"
    )
    predictions.write_parquet(args.output_dir / "rookie_predictions.parquet")
    rankings.write_csv(args.output_dir / f"rookie_rankings_{args.target_season}.csv")

    print("model          season rookies mae   rmse  corr top12 starter_hit")
    for metric in metrics:
        print(
            f"{metric.model:<14} {metric.season} {metric.rookies:>7} "
            f"{metric.mae:>5.1f} {metric.rmse:>5.1f} {metric.correlation:>5.2f} "
            f"{metric.top12_recall:>5.1%} {metric.starter_hit_recall:>10.1%}"
        )
    print(
        f"\nTop {args.target_season} offensive rookies "
        "(validated draft-capital baseline; context delta is experimental):"
    )
    print(
        rankings.select(
            "rookie_rank",
            "display_name",
            "position",
            "draft_pick",
            "draft_team",
            pl.col("draft_capital").round(1).alias("baseline_points"),
            pl.col("context_delta").round(1).alias("experimental_context_delta"),
        ).head(25)
    )


if __name__ == "__main__":
    main()
