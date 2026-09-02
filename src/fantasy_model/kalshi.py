from __future__ import annotations

import argparse
import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import polars as pl


KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# These are direct fantasy outcomes, not generic sports-betting proxies.
POSITION_WINNER_SERIES = {
    "KXRANKLISTFFQB": "top_qb",
    "KXRANKLISTFFRB": "top_rb",
    "KXRANKLISTFFWR": "top_wr",
    "KXRANKLISTFFTE": "top_te",
}
ROOKIE_LEADER_SERIES = "KXNFLFFLEADER"
CURRENT_MARKET_SERIES = (*POSITION_WINNER_SERIES, ROOKIE_LEADER_SERIES)


class KalshiReadError(RuntimeError):
    """A public, read-only Kalshi request failed."""


class MappingAmbiguityError(ValueError):
    """A supposedly exact player mapping has more than one possible target."""


@dataclass(frozen=True)
class BookProbability:
    bid: float | None
    ask: float | None
    probability: float | None
    spread: float | None
    source: str
    quality: str


def normalize_player_name(value: str) -> str:
    """Conservative join key; it normalizes formatting but never guesses a player."""

    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", ascii_name.casefold())
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def _probability_field(raw: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = raw.get(name)
        if value in (None, ""):
            continue
        number = float(value)
        if not name.endswith("_dollars") and number > 1:
            number /= 100.0
        if not 0 <= number <= 1:
            raise ValueError(f"Invalid Kalshi probability {name}={value!r}")
        return number
    return None


def book_probability(raw: dict[str, Any]) -> BookProbability:
    """Return a transparent book midpoint and a coarse usability label."""

    bid = _probability_field(raw, "yes_bid_dollars", "yes_bid")
    ask = _probability_field(raw, "yes_ask_dollars", "yes_ask")
    last = _probability_field(raw, "last_price_dollars", "last_price")

    probability: float | None = None
    spread: float | None = None
    source = "none"
    if bid is not None and ask is not None and ask >= bid and (bid > 0 or ask > 0):
        probability = (bid + ask) / 2
        spread = ask - bid
        source = "book_midpoint"
    elif last is not None and last > 0:
        probability = last
        source = "last_trade_fallback"

    volume = float(raw.get("volume_fp", raw.get("volume", 0)) or 0)
    if probability is None:
        quality = "no_quote"
    elif source == "last_trade_fallback":
        quality = "last_only"
    elif spread is not None and spread <= 0.05 and volume >= 25:
        quality = "tight"
    elif spread is not None and spread <= 0.15:
        quality = "usable"
    else:
        quality = "wide"
    return BookProbability(bid, ask, probability, spread, source, quality)


def _close_field(container: Any) -> Any:
    if not isinstance(container, dict):
        return None
    return container.get("close_dollars", container.get("close"))


def select_candle_as_of(
    candles: Iterable[dict[str, Any]], as_of_seconds: int
) -> dict[str, Any] | None:
    """Select only information whose candle boundary is at or before ``as_of``."""

    eligible = [
        candle
        for candle in candles
        if int(candle.get("end_period_ts", 0)) <= as_of_seconds
    ]
    return max(eligible, key=lambda candle: int(candle["end_period_ts"]), default=None)


def candle_book_probability(candle: dict[str, Any]) -> BookProbability:
    raw = {
        "yes_bid_dollars": _close_field(candle.get("yes_bid")),
        "yes_ask_dollars": _close_field(candle.get("yes_ask")),
        "last_price_dollars": _close_field(candle.get("price")),
        "volume_fp": candle.get("volume_fp", candle.get("volume", 0)),
    }
    return book_probability(raw)


class KalshiPublicClient:
    """Small GET-only client adapted from ``../noncompete/kalshi-mm`` rails."""

    def __init__(
        self,
        *,
        base_url: str = KALSHI_BASE_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        max_read_requests_per_second: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_read_requests_per_second = max_read_requests_per_second
        self._pace_lock = threading.Lock()
        self._next_request_monotonic = 0.0
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "old-mission-fantasy-research/0.1"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KalshiPublicClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not path.startswith("/") or ".." in path:
            raise ValueError("Kalshi path must be absolute and normalized")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.max_read_requests_per_second > 0:
                    with self._pace_lock:
                        now = time.monotonic()
                        if now < self._next_request_monotonic:
                            time.sleep(self._next_request_monotonic - now)
                            now = time.monotonic()
                        self._next_request_monotonic = now + (
                            1.0 / self.max_read_requests_per_second
                        )
                response = self._client.get(self.base_url + path, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code >= 400:
                    raise KalshiReadError(
                        f"Kalshi GET {path} failed ({response.status_code}): "
                        f"{response.text[:500]}"
                    )
                payload = response.json()
                if not isinstance(payload, dict):
                    raise KalshiReadError(f"Kalshi GET {path} returned non-object JSON")
                return payload
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                time.sleep(0.25 * (2**attempt))
        raise KalshiReadError(f"Kalshi GET {path} failed after retries: {last_error}")

    def list_markets(
        self, series_ticker: str, *, historical: bool = False
    ) -> list[dict[str, Any]]:
        path = "/historical/markets" if historical else "/markets"
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": series_ticker,
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self.get(path, params)
            batch = payload.get("markets", [])
            if not isinstance(batch, list):
                raise KalshiReadError("Kalshi markets payload is not a list")
            rows.extend(batch)
            cursor = payload.get("cursor") or None
            if not cursor or not batch:
                return rows

    def candlesticks(
        self,
        ticker: str,
        *,
        start_seconds: int,
        end_seconds: int,
        period_minutes: int = 1440,
        historical: bool = True,
        series_ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        if period_minutes not in (1, 60, 1440):
            raise ValueError("Kalshi candle interval must be 1, 60, or 1440")
        if historical:
            path = f"/historical/markets/{ticker}/candlesticks"
        elif series_ticker:
            path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        else:
            raise ValueError("Current-tier candles require series_ticker")
        payload = self.get(
            path,
            {
                "start_ts": start_seconds,
                "end_ts": end_seconds,
                "period_interval": period_minutes,
            },
        )
        candles = payload.get("candlesticks", [])
        if not isinstance(candles, list):
            raise KalshiReadError("Kalshi candlesticks payload is not a list")
        return candles


def _family(series_ticker: str, event_ticker: str) -> str | None:
    if series_ticker in POSITION_WINNER_SERIES:
        return POSITION_WINNER_SERIES[series_ticker]
    if series_ticker != ROOKIE_LEADER_SERIES:
        return None
    suffix = event_ticker.rsplit("-", 1)[-1]
    if suffix.endswith("ROOK"):
        return "rookie_overall"
    for position in ("QB", "RB", "WR", "TE"):
        if suffix.endswith(f"ROOK{position}"):
            return f"rookie_{position.lower()}"
    return None


def settlement_scoring(raw: dict[str, Any]) -> str:
    rules = str(raw.get("rules_secondary") or "").casefold()
    if "espn" in rules and "ppr" in rules:
        return "espn_full_ppr"
    if "sleeper" in rules and "ppr" in rules:
        return "sleeper_full_ppr"
    return "unknown"


def normalize_market_snapshot(
    markets: Iterable[dict[str, Any]], *, snapshot_time: str
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in markets:
        series = str(raw.get("_series_ticker", ""))
        event = str(raw.get("event_ticker", ""))
        family = _family(series, event)
        player_name = str(raw.get("yes_sub_title") or "").strip()
        if family is None or not player_name:
            continue
        book = book_probability(raw)
        rows.append(
            {
                "series_ticker": series,
                "event_ticker": event,
                "market_ticker": str(raw.get("ticker", "")),
                "family": family,
                "player_name": player_name,
                "normalized_name": normalize_player_name(player_name),
                "kalshi_player_id": str(
                    (raw.get("custom_strike") or {}).get("football_player", "")
                ),
                "yes_bid": book.bid,
                "yes_ask": book.ask,
                "market_probability": book.probability,
                "quote_spread": book.spread,
                "probability_source": book.source,
                "quote_quality": book.quality,
                "volume": float(raw.get("volume_fp", raw.get("volume", 0)) or 0),
                "open_interest": float(
                    raw.get("open_interest_fp", raw.get("open_interest", 0)) or 0
                ),
                "settlement_scoring": settlement_scoring(raw),
                "rules_primary": str(raw.get("rules_primary") or ""),
                "rules_secondary": str(raw.get("rules_secondary") or ""),
                "snapshot_time": snapshot_time,
            }
        )
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows).with_columns(
        pl.col("market_probability")
        .sum()
        .over("event_ticker")
        .alias("event_probability_sum")
    )
    return frame.with_columns(
        pl.when(
            pl.col("market_probability").is_not_null()
            & (pl.col("event_probability_sum") > 0)
        )
        .then(pl.col("market_probability") / pl.col("event_probability_sum"))
        .otherwise(None)
        .alias("event_normalized_probability"),
        pl.col("market_probability")
        .rank("average", descending=True)
        .over("event_ticker")
        .alias("event_market_rank"),
        pl.len().over("event_ticker").alias("event_market_count"),
    ).with_columns(
        pl.when(pl.col("event_market_count") <= 1)
        .then(1.0)
        .otherwise(
            1
            - (pl.col("event_market_rank") - 1)
            / (pl.col("event_market_count") - 1)
        )
        .alias("event_percentile")
    )


def fetch_current_market_snapshot(
    client: KalshiPublicClient,
) -> pl.DataFrame:
    snapshot_time = datetime.now(timezone.utc).isoformat()
    markets: list[dict[str, Any]] = []
    for series in CURRENT_MARKET_SERIES:
        for raw in client.list_markets(series):
            markets.append({**raw, "_series_ticker": series})
    return normalize_market_snapshot(markets, snapshot_time=snapshot_time)


def _single_value_feature(
    family_pattern: str, source_column: str, output_name: str
) -> pl.Expr:
    return (
        pl.col(source_column)
        .filter(pl.col("family").str.contains(family_pattern))
        .max()
        .alias(output_name)
    )


def rookie_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "normalized_name": pl.String,
                "kalshi_player_name": pl.String,
                "kalshi_rookie_overall_probability": pl.Float64,
                "kalshi_rookie_position_probability": pl.Float64,
                "kalshi_top_position_probability": pl.Float64,
                "kalshi_rookie_overall_percentile": pl.Float64,
                "kalshi_rookie_position_percentile": pl.Float64,
                "kalshi_top_position_percentile": pl.Float64,
                "kalshi_best_spread": pl.Float64,
                "kalshi_max_volume": pl.Float64,
                "kalshi_market_count": pl.UInt32,
            }
        )
    usable = frame.filter(pl.col("quote_quality").is_in(["tight", "usable"]))
    if usable.is_empty():
        return rookie_market_features(pl.DataFrame())
    usable = usable.with_columns(
        pl.col("market_probability")
        .rank("average", descending=True)
        .over("event_ticker")
        .alias("usable_event_rank"),
        pl.len().over("event_ticker").alias("usable_event_count"),
    ).with_columns(
        pl.when(pl.col("usable_event_count") <= 1)
        .then(1.0)
        .otherwise(
            1
            - (pl.col("usable_event_rank") - 1)
            / (pl.col("usable_event_count") - 1)
        )
        .alias("usable_event_percentile")
    )
    return usable.group_by("normalized_name").agg(
        pl.col("player_name").first().alias("kalshi_player_name"),
        _single_value_feature(
            r"^rookie_overall$",
            "market_probability",
            "kalshi_rookie_overall_probability",
        ),
        _single_value_feature(
            r"^rookie_(qb|rb|wr|te)$",
            "market_probability",
            "kalshi_rookie_position_probability",
        ),
        _single_value_feature(
            r"^top_(qb|rb|wr|te)$",
            "market_probability",
            "kalshi_top_position_probability",
        ),
        pl.col("usable_event_percentile")
        .filter(pl.col("family") == "rookie_overall")
        .max()
        .alias("kalshi_rookie_overall_percentile"),
        pl.col("usable_event_percentile")
        .filter(pl.col("family").str.contains(r"^rookie_(qb|rb|wr|te)$"))
        .max()
        .alias("kalshi_rookie_position_percentile"),
        pl.col("usable_event_percentile")
        .filter(pl.col("family").str.contains(r"^top_(qb|rb|wr|te)$"))
        .max()
        .alias("kalshi_top_position_percentile"),
        pl.col("quote_spread").min().alias("kalshi_best_spread"),
        pl.col("volume").max().alias("kalshi_max_volume"),
        pl.len().alias("kalshi_market_count"),
    )


def join_rookie_market_overlay(
    rankings: pl.DataFrame, market_snapshot: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ranked = rankings.with_columns(
        pl.col("display_name")
        .map_elements(normalize_player_name, return_dtype=pl.String)
        .alias("normalized_name")
    )
    duplicates = (
        ranked.group_by("normalized_name")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("normalized_name")
        .to_list()
    )
    if duplicates:
        raise MappingAmbiguityError(
            f"Duplicate normalized rookie names require an explicit mapping: {duplicates}"
        )

    features = rookie_market_features(market_snapshot)
    merged = ranked.join(features, on="normalized_name", how="left")
    probability_columns = [
        "kalshi_rookie_overall_percentile",
        "kalshi_rookie_position_percentile",
        "kalshi_top_position_percentile",
    ]
    weights = [0.50, 0.35, 0.15]
    numerator = pl.lit(0.0)
    denominator = pl.lit(0.0)
    for column, weight in zip(probability_columns, weights, strict=True):
        numerator += pl.when(pl.col(column).is_not_null()).then(
            pl.col(column) * weight
        ).otherwise(0.0)
        denominator += pl.when(pl.col(column).is_not_null()).then(weight).otherwise(0.0)
    merged = merged.with_columns(
        pl.col("kalshi_market_count").fill_null(0),
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(None)
        .alias("experimental_kalshi_signal"),
    ).with_columns(
        pl.col("experimental_kalshi_signal")
        .rank("ordinal", descending=True)
        .alias("kalshi_overlay_rank")
    )

    rookie_keys = set(ranked["normalized_name"].to_list())
    market_names = (
        market_snapshot.select("player_name", "normalized_name")
        .unique()
        .sort("player_name")
    )
    audit_rows = [
        {
            "source": "kalshi",
            "source_name": row["player_name"],
            "normalized_name": row["normalized_name"],
            "mapping_status": (
                "matched" if row["normalized_name"] in rookie_keys else "unmatched"
            ),
        }
        for row in market_names.iter_rows(named=True)
    ]
    covered = set(market_names["normalized_name"].to_list())
    audit_rows.extend(
        {
            "source": "rookie_model",
            "source_name": row["display_name"],
            "normalized_name": row["normalized_name"],
            "mapping_status": "no_kalshi_market",
        }
        for row in ranked.select("display_name", "normalized_name").iter_rows(
            named=True
        )
        if row["normalized_name"] not in covered
    )
    return merged, pl.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a read-only, point-in-time Kalshi fantasy market snapshot"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/backtests/kalshi_market_snapshot.csv")
    )
    args = parser.parse_args()

    with KalshiPublicClient() as client:
        snapshot = fetch_current_market_snapshot(client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_csv(args.output)
    print(
        json.dumps(
            {
                "rows": snapshot.height,
                "events": snapshot["event_ticker"].n_unique(),
                "players": snapshot["normalized_name"].n_unique(),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
