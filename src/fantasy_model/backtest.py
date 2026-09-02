from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .data import load_player_stats


POSITIONS = ("QB", "RB", "WR", "TE")
POSITION_STARTER_CUTOFF = {"QB": 10, "RB": 30, "WR": 40, "TE": 10}


def _stat(frame: pl.DataFrame, *names: str) -> pl.Expr:
    for name in names:
        if name in frame.columns:
            return pl.col(name).fill_null(0)
    return pl.lit(0.0)


def add_league_points(frame: pl.DataFrame) -> pl.DataFrame:
    points = (
        _stat(frame, "passing_tds") * 6
        + _stat(frame, "passing_yards") * 0.04
        - _stat(frame, "passing_interceptions", "interceptions") * 2
        + _stat(frame, "passing_2pt_conversions") * 2
        + _stat(frame, "rushing_tds") * 6
        + _stat(frame, "rushing_yards") * 0.10
        + _stat(frame, "rushing_2pt_conversions") * 2
        + _stat(frame, "receiving_tds") * 6
        + _stat(frame, "receiving_yards") * 0.10
        + _stat(frame, "receptions") * 0.5
        + _stat(frame, "receiving_2pt_conversions") * 2
        - _stat(frame, "fumbles_lost_total", "fumbles_lost") * 2
    )
    return frame.with_columns(points.round(2).alias("league_points"))


def seasonal_player_table(frame: pl.DataFrame) -> pl.DataFrame:
    player_id = "player_id" if "player_id" in frame.columns else "player_id_name"
    player_name = (
        "player_display_name" if "player_display_name" in frame.columns else "player_name"
    )
    regular = frame.filter(
        pl.col("position").is_in(POSITIONS)
        & (pl.col("season_type") == "REG")
        & (pl.col("week") <= 14)
    )
    return (
        add_league_points(regular)
        .group_by("season", player_id, player_name, "position")
        .agg(
            pl.len().alias("games"),
            pl.col("league_points").sum().alias("points"),
            pl.col("league_points").mean().alias("ppg"),
            pl.col("league_points").std().fill_null(0).alias("weekly_sd"),
        )
        .rename({player_id: "player_id", player_name: "player_name"})
    )


def preseason_projection(seasonal: pl.DataFrame, target_season: int) -> pl.DataFrame:
    history = seasonal.filter(
        (pl.col("season") < target_season)
        & (pl.col("season") >= target_season - 3)
    ).with_columns(
        pl.when(pl.col("season") == target_season - 1)
        .then(1.0)
        .when(pl.col("season") == target_season - 2)
        .then(0.55)
        .otherwise(0.30)
        .alias("recency_weight")
    )
    prior = (
        seasonal.filter(pl.col("season") == target_season - 1)
        .group_by("position")
        .agg(pl.col("ppg").median().alias("position_prior"))
    )
    history_projection = (
        history.group_by("player_id")
        .agg(
            (pl.col("ppg") * pl.col("games") * pl.col("recency_weight"))
            .sum()
            .alias("weighted_points"),
            (pl.col("games") * pl.col("recency_weight"))
            .sum()
            .alias("weighted_games"),
            pl.col("weekly_sd").mean().alias("projected_sd"),
        )
    )
    actual = seasonal.filter(pl.col("season") == target_season).select(
        "player_id",
        "player_name",
        "position",
        pl.col("ppg").alias("actual_ppg"),
        pl.col("points").alias("actual_points"),
        pl.col("games").alias("actual_games"),
    )
    previous = seasonal.filter(pl.col("season") == target_season - 1).select(
        "player_id",
        pl.col("ppg").alias("last_year_ppg"),
        pl.col("games").alias("last_year_games"),
    )
    return (
        actual.join(history_projection, on="player_id", how="left")
        .join(previous, on="player_id", how="left")
        .join(prior, on="position", how="left")
        .with_columns(
            pl.col("weighted_points").fill_null(0),
            pl.col("weighted_games").fill_null(0),
            pl.col("projected_sd").fill_null(0),
            pl.col("last_year_games").fill_null(0),
            pl.col("last_year_ppg").fill_null(pl.col("position_prior")),
        )
        .with_columns(
            (
                (pl.col("weighted_points") + pl.col("position_prior") * 6)
                / (pl.col("weighted_games") + 6)
            ).alias("projected_ppg"),
            (pl.col("weighted_games") == 0).alias("no_nfl_history"),
            pl.lit(target_season).alias("target_season"),
        )
    )


@dataclass(frozen=True)
class BacktestMetric:
    model: str
    season: int
    players: int
    mae: float
    rmse: float
    correlation: float
    starter_recall: float
    new_player_count: int
    new_player_mae: float


def evaluate(
    projections: pl.DataFrame, season: int, model: str, prediction_column: str
) -> BacktestMetric:
    projected = projections[prediction_column].to_numpy()
    actual = projections["actual_ppg"].to_numpy()
    error = projected - actual

    ranked = projections.with_columns(
        pl.col(prediction_column)
        .rank("ordinal", descending=True)
        .over("position")
        .alias("projected_rank"),
        pl.col("actual_points")
        .rank("ordinal", descending=True)
        .over("position")
        .alias("actual_rank"),
    ).with_columns(
        pl.col("position")
        .replace_strict(POSITION_STARTER_CUTOFF)
        .alias("starter_cutoff")
    )
    actual_starters = ranked.filter(pl.col("actual_rank") <= pl.col("starter_cutoff"))
    starter_recall = actual_starters.filter(
        pl.col("projected_rank") <= pl.col("starter_cutoff")
    ).height / max(actual_starters.height, 1)
    new_players = projections.filter(pl.col("no_nfl_history"))
    new_player_error = (
        new_players[prediction_column].to_numpy()
        - new_players["actual_ppg"].to_numpy()
    )

    correlation = float(np.corrcoef(projected, actual)[0, 1])
    return BacktestMetric(
        model=model,
        season=season,
        players=projections.height,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        correlation=correlation,
        starter_recall=starter_recall,
        new_player_count=new_players.height,
        new_player_mae=float(np.mean(np.abs(new_player_error))),
    )


def run_backtest(
    *,
    cache_dir: Path,
    output_dir: Path,
    start_season: int,
    end_season: int,
) -> tuple[list[BacktestMetric], pl.DataFrame]:
    stats = load_player_stats(cache_dir, range(start_season - 3, end_season + 1))
    seasonal = seasonal_player_table(stats)
    available_seasons = set(seasonal["season"].unique().to_list())
    requested_seasons = list(range(start_season, end_season + 1))
    missing = [season for season in requested_seasons if season not in available_seasons]
    if missing:
        raise ValueError(f"Historical outcomes are missing for seasons: {missing}")
    predictions = pl.concat(
        [preseason_projection(seasonal, season) for season in requested_seasons],
        how="vertical",
    )
    metrics = []
    for season in requested_seasons:
        season_predictions = predictions.filter(pl.col("target_season") == season)
        metrics.extend(
            [
                evaluate(
                    season_predictions,
                    season,
                    "recency_shrink",
                    "projected_ppg",
                ),
                evaluate(
                    season_predictions,
                    season,
                    "last_year_ppg",
                    "last_year_ppg",
                ),
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "baseline_predictions.parquet")
    pl.DataFrame([metric.__dict__ for metric in metrics]).write_csv(
        output_dir / "baseline_metrics.csv"
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run point-in-time baseline backtests")
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/backtests")
    )
    args = parser.parse_args()
    metrics, _ = run_backtest(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    print("model          season players mae rmse corr starter_recall new_n new_mae")
    for metric in metrics:
        print(
            f"{metric.model:<14} {metric.season} {metric.players:>7} {metric.mae:>5.2f} "
            f"{metric.rmse:>5.2f} {metric.correlation:>5.2f} "
            f"{metric.starter_recall:>6.1%} {metric.new_player_count:>5} "
            f"{metric.new_player_mae:>7.2f}"
        )


if __name__ == "__main__":
    main()
