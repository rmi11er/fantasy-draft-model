"""Walk-forward calibration for season-level player uncertainty.

The projection model's historical weekly standard deviation captures in-game
volatility, but not preseason role, injury, and availability uncertainty.  The
constants are calibrated from point-in-time 2020-2025 residuals under this
league's scoring. Veteran means also receive a simple age correction whose
shape is validated with expanding-year holdouts. Rookie bands come from the
existing walk-forward rookie predictions over the same seasons.

This module deliberately keeps calibration separate from player means: it does
not manufacture an upside bonus.  It widens or floors the distribution that is
subsequently evaluated under the league's 70/20/10 payout utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .data import load_player_stats, load_players
from .distribution_constants import (
    CALIBRATION_SEASONS,
    ROOKIE_SEASON_SD_BANDS,
    VETERAN_MEAN_AGE_MODEL,
    VETERAN_SEASON_SD_MULTIPLIER,
    calibrated_rookie_sd,
    calibrated_veteran_mean_adjustment,
)
from .projections import (
    ProjectionAssumptions,
    _historical_projection_inputs,
    add_projection_points,
)


def veteran_calibration_rows(
    stats: pl.DataFrame,
    nflverse_players: pl.DataFrame,
    *,
    start_season: int = CALIBRATION_SEASONS[0],
    end_season: int = CALIBRATION_SEASONS[-1],
    assumptions: ProjectionAssumptions | None = None,
) -> pl.DataFrame:
    """Build strictly point-in-time veteran residuals for calibration audit."""
    assumptions = assumptions or ProjectionAssumptions()
    rows: list[pl.DataFrame] = []
    positions = ("QB", "RB", "WR", "TE", "K", "PK")
    for target in range(start_season, end_season + 1):
        veteran, _ = _historical_projection_inputs(
            stats, target_season=target, assumptions=assumptions
        )
        actual = (
            stats.filter(
                (pl.col("season") == target)
                & (pl.col("season_type") == "REG")
                & pl.col("position").is_in(positions)
            )
            .pipe(add_projection_points)
            .group_by("player_id")
            .agg(
                pl.col("projection_points").sum().alias("actual_points"),
                pl.len().alias("actual_games"),
                pl.col("position").last().alias("position"),
            )
            .with_columns(
                pl.when(pl.col("position") == "K")
                .then(pl.lit("PK"))
                .otherwise(pl.col("position"))
                .alias("position")
            )
        )
        target_rows = (
            veteran.join(
                actual,
                left_on="nflverse_player_id",
                right_on="player_id",
                how="inner",
                validate="1:1",
            )
            .with_columns(
                (
                    pl.col("historical_active_ppg") * pl.col("expected_games")
                ).alias("projected_points"),
                (pl.col("expected_games") / assumptions.season_weeks).alias(
                    "availability_rate"
                ),
            )
            .with_columns(
                (
                    assumptions.season_weeks
                    * (
                        pl.col("availability_rate")
                        * (
                            pl.col("historical_active_sd").pow(2)
                            + pl.col("historical_active_ppg").pow(2)
                        )
                        - (
                            pl.col("availability_rate")
                            * pl.col("historical_active_ppg")
                        ).pow(2)
                    )
                )
                .clip(0.0, None)
                .sqrt()
                .alias("raw_season_sd"),
                pl.lit(target).alias("target_season"),
            )
            .with_columns(
                (pl.col("actual_points") - pl.col("projected_points")).alias(
                    "unadjusted_residual"
                )
            )
            .select(
                "target_season",
                "nflverse_player_id",
                "position",
                "projected_points",
                "actual_points",
                "actual_games",
                "raw_season_sd",
                "unadjusted_residual",
            )
        )
        rows.append(target_rows)
    birth_dates = nflverse_players.select(
        pl.col("gsis_id").alias("nflverse_player_id"),
        pl.col("birth_date").cast(pl.String),
    ).unique("nflverse_player_id")
    return (
        pl.concat(rows, how="vertical")
        .join(birth_dates, on="nflverse_player_id", how="left", validate="m:1")
        .with_columns(
            (
                pl.col("target_season")
                - pl.col("birth_date").str.to_date(strict=False).dt.year()
            )
            .cast(pl.Float64)
            .alias("player_age")
        )
        .with_columns(
            pl.struct("position", "player_age")
            .map_elements(
                lambda row: calibrated_veteran_mean_adjustment(
                    row["position"], row["player_age"]
                ),
                return_dtype=pl.Float64,
            )
            .alias("mean_adjustment_points")
        )
        .with_columns(
            (pl.col("projected_points") + pl.col("mean_adjustment_points"))
            .clip(0.0, None)
            .alias("projected_points")
        )
        .with_columns(
            (pl.col("actual_points") - pl.col("projected_points")).alias(
                "residual"
            ),
            (
                pl.col("raw_season_sd")
                * pl.col("position").replace_strict(
                    VETERAN_SEASON_SD_MULTIPLIER
                )
            ).alias("calibrated_season_sd"),
        )
        .drop("birth_date")
    )


def _coverage(frame: pl.DataFrame, sd_column: str, z: float) -> float:
    return frame.filter(
        pl.col("residual").abs() <= pl.col(sd_column) * z
    ).height / max(frame.height, 1)


def veteran_calibration_summary(rows: pl.DataFrame) -> pl.DataFrame:
    summaries = []
    for position in ("QB", "RB", "WR", "TE", "PK"):
        frame = rows.filter(pl.col("position") == position)
        residual = frame["residual"].to_numpy()
        summaries.append(
            {
                "group": f"veteran_{position.lower()}",
                "observations": frame.height,
                "bias": float(residual.mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "raw_68_coverage": _coverage(frame, "raw_season_sd", 1.0),
                "calibrated_68_coverage": _coverage(
                    frame, "calibrated_season_sd", 1.0
                ),
                "raw_80_coverage": _coverage(frame, "raw_season_sd", 1.28155),
                "calibrated_80_coverage": _coverage(
                    frame, "calibrated_season_sd", 1.28155
                ),
                "sd_multiplier": VETERAN_SEASON_SD_MULTIPLIER[position],
            }
        )
    return pl.DataFrame(summaries)


def veteran_age_walk_forward_summary(rows: pl.DataFrame) -> pl.DataFrame:
    """Validate the age-curve architecture on expanding future-season folds."""
    records: list[dict[str, Any]] = []
    for target_season in CALIBRATION_SEASONS[2:]:
        season_records: list[tuple[np.ndarray, np.ndarray]] = []
        for position, model in VETERAN_MEAN_AGE_MODEL.items():
            training = rows.filter(
                (pl.col("target_season") < target_season)
                & (pl.col("position") == position)
            )
            test = rows.filter(
                (pl.col("target_season") == target_season)
                & (pl.col("position") == position)
            )
            training_age = training["player_age"].to_numpy()
            training_residual = training["unadjusted_residual"].to_numpy()
            centered_age = training_age - model["center_age"]
            design = np.column_stack([np.ones(training.height), centered_age])
            penalty = np.diag([0.0, 25.0])
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ training_residual,
            )
            test_age = test["player_age"].to_numpy() - model["center_age"]
            adjustment = np.clip(
                coefficients[0] + coefficients[1] * test_age,
                -60.0,
                30.0,
            )
            season_records.append(
                (test["unadjusted_residual"].to_numpy(), adjustment)
            )
        residual = np.concatenate([item[0] for item in season_records])
        adjustment = np.concatenate([item[1] for item in season_records])
        for method, error in (
            ("unadjusted", residual),
            ("age_curve", residual - adjustment),
        ):
            records.append(
                {
                    "target_season": target_season,
                    "method": method,
                    "observations": len(error),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
    return pl.DataFrame(records)


def rookie_calibration_summary(predictions: pl.DataFrame) -> pl.DataFrame:
    scaled = predictions.with_columns(
        (pl.col("draft_capital_prediction") * 17 / 14).alias("projected_points"),
        (pl.col("actual_points") * 17 / 14).alias("actual_points_scaled"),
    ).with_columns(
        (pl.col("actual_points_scaled") - pl.col("projected_points")).alias(
            "residual"
        )
    )
    summaries = []
    for position in ("QB", "RB", "WR", "TE"):
        frame = scaled.filter(pl.col("position") == position).with_columns(
            pl.col("projected_points")
            .map_elements(
                lambda mean: calibrated_rookie_sd(position, mean),
                return_dtype=pl.Float64,
            )
            .alias("calibrated_season_sd")
        )
        residual = frame["residual"].to_numpy()
        summaries.append(
            {
                "group": f"rookie_{position.lower()}",
                "observations": frame.height,
                "bias": float(residual.mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "raw_68_coverage": None,
                "calibrated_68_coverage": _coverage(
                    frame, "calibrated_season_sd", 1.0
                ),
                "raw_80_coverage": None,
                "calibrated_80_coverage": _coverage(
                    frame, "calibrated_season_sd", 1.28155
                ),
                "sd_multiplier": None,
            }
        )
    return pl.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit season-distribution calibration from local data"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--rookie-predictions",
        type=Path,
        default=Path("outputs/backtests/rookie_predictions.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/backtests")
    )
    args = parser.parse_args()
    stats = load_player_stats(args.cache_dir, range(2017, 2026))
    veteran_rows = veteran_calibration_rows(stats, load_players(args.cache_dir))
    summary = pl.concat(
        [
            veteran_calibration_summary(veteran_rows),
            rookie_calibration_summary(pl.read_parquet(args.rookie_predictions)),
        ],
        how="diagonal_relaxed",
    )
    age_validation = veteran_age_walk_forward_summary(veteran_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    veteran_rows.write_parquet(
        args.output_dir / "distribution_calibration_rows.parquet"
    )
    summary.write_csv(args.output_dir / "distribution_calibration_metrics.csv")
    age_validation.write_csv(
        args.output_dir / "distribution_age_walk_forward_metrics.csv"
    )
    metadata = {
        "calibration_seasons": list(CALIBRATION_SEASONS),
        "veteran_season_sd_multiplier": VETERAN_SEASON_SD_MULTIPLIER,
        "veteran_mean_age_model": VETERAN_MEAN_AGE_MODEL,
        "rookie_season_sd_bands": {
            position: [
                {
                    "upper_bound": upper_bound if np.isfinite(upper_bound) else None,
                    "season_sd": season_sd,
                }
                for upper_bound, season_sd in bands
            ]
            for position, bands in ROOKIE_SEASON_SD_BANDS.items()
        },
        "notes": [
            "Player means are point-in-time; age-curve shape is expanding-year validated.",
            "Final age coefficients and uncertainty scales are frozen on 2020-2025 residuals.",
            "Historical active-player filtering is approximate because point-in-time MFL universes are unavailable.",
            "Calibration widens uncertainty; it does not add an upside preference bonus.",
        ],
    }
    (args.output_dir / "distribution_calibration.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(summary)
    print(age_validation)


if __name__ == "__main__":
    main()
