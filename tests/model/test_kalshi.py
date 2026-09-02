import polars as pl
import pytest

from fantasy_model.kalshi import (
    MappingAmbiguityError,
    book_probability,
    join_rookie_market_overlay,
    normalize_market_snapshot,
    normalize_player_name,
    select_candle_as_of,
)


def test_book_probability_preserves_price_and_quality() -> None:
    tight = book_probability(
        {
            "yes_bid_dollars": "0.12",
            "yes_ask_dollars": "0.18",
            "last_price_dollars": "0.19",
            "volume_fp": "200",
        }
    )
    assert tight.probability == pytest.approx(0.15)
    assert tight.spread == pytest.approx(0.06)
    assert tight.quality == "usable"

    one_cent = book_probability(
        {"yes_bid_dollars": "0", "yes_ask_dollars": "0.01"}
    )
    assert one_cent.probability == pytest.approx(0.005)


def test_candle_selector_cannot_see_future_prices() -> None:
    candles = [
        {"end_period_ts": 100, "yes_bid": {"close": "0.10"}},
        {"end_period_ts": 200, "yes_bid": {"close": "0.90"}},
    ]
    assert select_candle_as_of(candles, 150)["end_period_ts"] == 100
    assert select_candle_as_of(candles, 99) is None


def test_market_probabilities_are_normalized_within_event() -> None:
    raw = [
        {
            "_series_ticker": "KXNFLFFLEADER",
            "event_ticker": "KXNFLFFLEADER-27ROOK",
            "ticker": "A",
            "yes_sub_title": "Player One",
            "yes_bid_dollars": "0.10",
            "yes_ask_dollars": "0.20",
            "rules_secondary": "Based on ESPN Points Per Reception (PPR) scoring",
        },
        {
            "_series_ticker": "KXNFLFFLEADER",
            "event_ticker": "KXNFLFFLEADER-27ROOK",
            "ticker": "B",
            "yes_sub_title": "Player Two",
            "yes_bid_dollars": "0.20",
            "yes_ask_dollars": "0.30",
        },
    ]
    frame = normalize_market_snapshot(raw, snapshot_time="2026-09-01T00:00:00Z")
    assert frame["event_normalized_probability"].sum() == pytest.approx(1)
    assert frame.sort("market_ticker")["event_market_rank"].to_list() == [2, 1]
    assert frame.sort("market_ticker")["settlement_scoring"][0] == "espn_full_ppr"


def test_rookie_join_is_exact_and_rejects_ambiguity() -> None:
    rankings = pl.DataFrame(
        {
            "display_name": ["Omar Cooper"],
            "rookie_rank": [1],
        }
    )
    raw = [
        {
            "_series_ticker": "KXNFLFFLEADER",
            "event_ticker": "KXNFLFFLEADER-27ROOKWR",
            "ticker": "A",
            "yes_sub_title": "Omar Cooper Jr.",
            "yes_bid_dollars": "0.10",
            "yes_ask_dollars": "0.12",
        }
    ]
    snapshot = normalize_market_snapshot(raw, snapshot_time="now")
    merged, audit = join_rookie_market_overlay(rankings, snapshot)
    assert merged["kalshi_market_count"].item() == 1
    assert audit.filter(pl.col("mapping_status") == "matched").height == 1
    assert normalize_player_name("Omar Cooper Jr.") == normalize_player_name(
        "Omar Cooper"
    )

    duplicate = pl.DataFrame(
        {"display_name": ["John Doe", "John Doe Jr."], "rookie_rank": [1, 2]}
    )
    with pytest.raises(MappingAmbiguityError):
        join_rookie_market_overlay(duplicate, snapshot)
