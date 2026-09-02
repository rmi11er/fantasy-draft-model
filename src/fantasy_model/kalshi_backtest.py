from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .backtest import preseason_projection, seasonal_player_table
from .data import load_player_stats
from .kalshi import (
    KalshiPublicClient,
    POSITION_WINNER_SERIES,
    candle_book_probability,
    normalize_player_name,
    select_candle_as_of,
    settlement_scoring,
)


HISTORICAL_POSITION_SERIES = tuple(POSITION_WINNER_SERIES)


def _timestamp_seconds(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _cached_candles(
    client: KalshiPublicClient,
    market: dict[str, Any],
    *,
    as_of_seconds: int,
    cache_dir: Path,
) -> dict[str, Any]:
    ticker = str(market["ticker"])
    path = cache_dir / f"{ticker}.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {
            "ticker": ticker,
            "as_of_seconds": as_of_seconds,
            "candlesticks": client.candlesticks(
                ticker,
                start_seconds=_timestamp_seconds(str(market["open_time"])),
                end_seconds=as_of_seconds,
                period_minutes=1440,
                historical=True,
            ),
        }
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)
    if int(payload.get("as_of_seconds", 0)) != as_of_seconds:
        raise ValueError(f"Cached Kalshi cutoff mismatch for {ticker}")
    return payload


def historical_position_snapshot(
    client: KalshiPublicClient,
    *,
    as_of: datetime,
    cache_dir: Path,
    workers: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch the last daily book boundary before a fixed preseason cutoff."""

    if as_of.tzinfo is None:
        raise ValueError("Kalshi backtest cutoff must be timezone-aware")
    as_of_seconds = int(as_of.timestamp())
    cache_dir.mkdir(parents=True, exist_ok=True)
    markets: list[dict[str, Any]] = []
    for series in HISTORICAL_POSITION_SERIES:
        position = POSITION_WINNER_SERIES[series].removeprefix("top_").upper()
        for raw in client.list_markets(series, historical=True):
            markets.append({**raw, "_series_ticker": series, "_position": position})

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        {
            "market_ticker": str(market.get("ticker", "")),
            "player_name": str(market.get("yes_sub_title", "")),
            "position": market["_position"],
            "failure": "market_not_open_by_cutoff",
        }
        for market in markets
        if _timestamp_seconds(str(market["open_time"])) >= as_of_seconds
    ]
    markets = [
        market
        for market in markets
        if _timestamp_seconds(str(market["open_time"])) < as_of_seconds
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_markets = {
            pool.submit(
                _cached_candles,
                client,
                market,
                as_of_seconds=as_of_seconds,
                cache_dir=cache_dir,
            ): market
            for market in markets
        }
        for future in as_completed(future_markets):
            market = future_markets[future]
            try:
                payload = future.result()
                candle = select_candle_as_of(payload["candlesticks"], as_of_seconds)
                if candle is None:
                    raise ValueError("no candle at or before cutoff")
                book = candle_book_probability(candle)
                if book.probability is None:
                    raise ValueError("no usable point-in-time price")
                rows.append(
                    {
                        "series_ticker": market["_series_ticker"],
                        "event_ticker": str(market.get("event_ticker", "")),
                        "market_ticker": str(market["ticker"]),
                        "player_name": str(market.get("yes_sub_title", "")),
                        "normalized_name": normalize_player_name(
                            str(market.get("yes_sub_title", ""))
                        ),
                        "position": market["_position"],
                        "settlement_scoring": settlement_scoring(market),
                        "candle_end_seconds": int(candle["end_period_ts"]),
                        "as_of_seconds": as_of_seconds,
                        "yes_bid": book.bid,
                        "yes_ask": book.ask,
                        "market_probability": book.probability,
                        "quote_spread": book.spread,
                        "quote_quality": book.quality,
                    }
                )
            except Exception as error:  # preserve coverage failures in the audit
                failures.append(
                    {
                        "market_ticker": str(market.get("ticker", "")),
                        "player_name": str(market.get("yes_sub_title", "")),
                        "position": market["_position"],
                        "failure": str(error),
                    }
                )
    frame = pl.DataFrame(rows).with_columns(
        pl.col("market_probability")
        .sum()
        .over("event_ticker")
        .alias("event_probability_sum")
    ).with_columns(
        (pl.col("market_probability") / pl.col("event_probability_sum")).alias(
            "event_normalized_probability"
        )
    )
    audit = pl.DataFrame(failures) if failures else pl.DataFrame(
        schema={
            "market_ticker": pl.String,
            "player_name": pl.String,
            "position": pl.String,
            "failure": pl.String,
        }
    )
    return frame, audit


def _score_percentile(column: str) -> pl.Expr:
    count = pl.len().over("position")
    rank = pl.col(column).rank("average", descending=True).over("position")
    return pl.when(count <= 1).then(1.0).otherwise(
        1 - (rank - 1) / (count - 1)
    )


def join_snapshot_outcomes(
    snapshot: pl.DataFrame, seasonal: pl.DataFrame, season: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    projection = preseason_projection(seasonal, season).with_columns(
        pl.col("player_name")
        .map_elements(normalize_player_name, return_dtype=pl.String)
        .alias("normalized_name")
    )
    duplicate = (
        projection.group_by("normalized_name", "position")
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate.height:
        raise ValueError(
            "Ambiguous nflverse player names in Kalshi validation: "
            f"{duplicate.select('normalized_name', 'position').rows()}"
        )
    joined = snapshot.join(
        projection.select(
            "normalized_name",
            "position",
            "player_id",
            "actual_points",
            "actual_ppg",
            "projected_ppg",
        ),
        on=["normalized_name", "position"],
        how="left",
        validate="m:1",
    )
    audit = joined.select(
        "market_ticker",
        "player_name",
        "position",
        pl.when(pl.col("player_id").is_not_null())
        .then(pl.lit("matched"))
        .otherwise(pl.lit("unmatched"))
        .alias("mapping_status"),
    )
    matched = joined.filter(pl.col("player_id").is_not_null()).with_columns(
        _score_percentile("actual_points").alias("actual_percentile"),
        _score_percentile("projected_ppg").alias("baseline_percentile"),
        _score_percentile("event_normalized_probability").alias(
            "kalshi_percentile"
        ),
    )
    for weight in (0.10, 0.20, 0.30):
        matched = matched.with_columns(
            (
                pl.col("baseline_percentile") * (1 - weight)
                + pl.col("kalshi_percentile") * weight
            ).alias(f"blend_{int(weight * 100)}pct_market")
        )
    return matched, audit


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pl.Series(left).rank("average").to_numpy()
    right_rank = pl.Series(right).rank("average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _partial_correlation(actual: np.ndarray, market: np.ndarray, baseline: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(baseline)), baseline))
    actual_residual = actual - design @ np.linalg.lstsq(design, actual, rcond=None)[0]
    market_residual = market - design @ np.linalg.lstsq(design, market, rcond=None)[0]
    return float(np.corrcoef(actual_residual, market_residual)[0, 1])


def _top_recall(frame: pl.DataFrame, score_column: str, top_n: int = 5) -> float:
    if frame["position"].n_unique() > 1:
        return float(
            np.mean(
                [
                    _top_recall(
                        frame.filter(pl.col("position") == position),
                        score_column,
                        top_n,
                    )
                    for position in frame["position"].unique().to_list()
                ]
            )
        )
    count = min(top_n, frame.height)
    actual = set(
        frame.sort("actual_percentile", descending=True)
        .head(count)["player_id"]
        .to_list()
    )
    predicted = set(
        frame.sort(score_column, descending=True).head(count)["player_id"].to_list()
    )
    return len(actual & predicted) / max(count, 1)


def validation_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("ALL", frame)] + [
        (position, frame.filter(pl.col("position") == position))
        for position in ("QB", "RB", "WR", "TE")
    ]
    for position, group in groups:
        if group.height < 3:
            continue
        actual = group["actual_percentile"].to_numpy()
        baseline = group["baseline_percentile"].to_numpy()
        market = group["kalshi_percentile"].to_numpy()
        row: dict[str, Any] = {
            "position": position,
            "players": group.height,
            "baseline_spearman": _spearman(actual, baseline),
            "kalshi_spearman": _spearman(actual, market),
            "kalshi_partial_correlation_after_baseline": _partial_correlation(
                actual, market, baseline
            ),
            "baseline_top5_recall": _top_recall(group, "projected_ppg"),
            "kalshi_top5_recall": _top_recall(
                group, "event_normalized_probability"
            ),
        }
        for weight in (10, 20, 30):
            column = f"blend_{weight}pct_market"
            row[f"blend_{weight}_spearman"] = _spearman(
                actual, group[column].to_numpy()
            )
            row[f"blend_{weight}_top5_recall"] = _top_recall(group, column)
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate point-in-time Kalshi fantasy rankings on available history"
    )
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--as-of", default="2025-09-02T00:00:00+00:00")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/backtests")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    stats = load_player_stats(
        args.cache_dir, range(args.season - 3, args.season + 1)
    )
    seasonal = seasonal_player_table(stats)
    with KalshiPublicClient() as client:
        snapshot, fetch_audit = historical_position_snapshot(
            client,
            as_of=as_of,
            cache_dir=args.cache_dir / "kalshi" / str(args.season),
            workers=args.workers,
        )
    joined, mapping_audit = join_snapshot_outcomes(snapshot, seasonal, args.season)
    metrics = validation_metrics(joined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.write_csv(args.output_dir / f"kalshi_snapshot_{args.season}.csv")
    joined.write_csv(args.output_dir / f"kalshi_validation_rows_{args.season}.csv")
    metrics.write_csv(args.output_dir / f"kalshi_validation_metrics_{args.season}.csv")
    pl.concat([fetch_audit, mapping_audit], how="diagonal_relaxed").write_csv(
        args.output_dir / f"kalshi_validation_audit_{args.season}.csv"
    )
    print(metrics)


if __name__ == "__main__":
    main()
