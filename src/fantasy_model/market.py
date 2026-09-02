from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from .kalshi import (
    KalshiPublicClient,
    fetch_current_market_snapshot,
    join_rookie_market_overlay,
)
from .league import LeagueSnapshot, MFLClient


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def mfl_display_name(value: str) -> str:
    if "," not in value:
        return value.strip()
    last, first = value.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def mfl_players_frame(payload: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for raw in _as_list(payload.get("players", {}).get("player")):
        if raw.get("position") not in {"QB", "RB", "WR", "TE", "PK", "Def"}:
            continue
        rows.append(
            {
                "mfl_player_id": str(raw["id"]),
                "espn_id": str(raw.get("espn_id") or ""),
                "mfl_name": str(raw.get("name") or ""),
                "mfl_display_name": mfl_display_name(str(raw.get("name") or "")),
                "mfl_position": str(raw.get("position") or ""),
                "mfl_team": str(raw.get("team") or "FA"),
                "mfl_status": str(raw.get("status") or ""),
            }
        )
    return pl.DataFrame(rows)


def mfl_adp_frame(payload: dict[str, Any]) -> pl.DataFrame:
    rows = [
        {
            "mfl_player_id": str(raw["id"]),
            "adp_rank": int(raw["rank"]),
            "average_pick": float(raw["averagePick"]),
            "adp_min_pick": int(raw["minPick"]),
            "adp_max_pick": int(raw["maxPick"]),
            "adp_drafts": int(raw["draftsSelectedIn"]),
            "adp_selection_pct": float(raw["draftSelPct"]),
        }
        for raw in _as_list(payload.get("adp", {}).get("player"))
    ]
    return pl.DataFrame(rows)


def build_rookie_market_board(
    rankings: pl.DataFrame,
    players: pl.DataFrame,
    adp: pl.DataFrame,
    league: LeagueSnapshot,
    kalshi_snapshot: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if "espn_id" not in rankings.columns:
        raise ValueError(
            "Rookie rankings lack espn_id; rerun fantasy-rookies before market sync"
        )
    rookies = rankings.with_columns(pl.col("espn_id").cast(pl.String))
    identified_players = players.filter(pl.col("espn_id") != "")
    duplicate_mfl_ids = (
        identified_players
        .group_by("espn_id")
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_mfl_ids.height:
        raise ValueError(
            "MFL contains duplicate ESPN identifiers: "
            f"{duplicate_mfl_ids['espn_id'].to_list()}"
        )

    board = rookies.join(
        identified_players, on="espn_id", how="left", validate="m:1"
    )
    board = board.join(adp, on="mfl_player_id", how="left", validate="m:1")
    board, kalshi_audit = join_rookie_market_overlay(board, kalshi_snapshot)
    drafted_ids = league.drafted_player_ids
    board = board.with_columns(
        pl.col("mfl_player_id")
        .map_elements(
            lambda player_id: bool(player_id and player_id in drafted_ids),
            return_dtype=pl.Boolean,
        )
        .alias("drafted_in_conference")
    ).with_columns((~pl.col("drafted_in_conference")).alias("available"))

    mfl_audit = board.select(
        pl.lit("mfl").alias("source"),
        pl.col("display_name").alias("source_name"),
        pl.col("espn_id").alias("normalized_name"),
        pl.when(pl.col("mfl_player_id").is_not_null())
        .then(pl.lit("matched"))
        .otherwise(pl.lit("unmatched"))
        .alias("mapping_status"),
    )
    return board.sort("rookie_rank"), pl.concat(
        [mfl_audit, kalshi_audit], how="vertical_relaxed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the live MFL + Kalshi rookie draft board"
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--league-id", default="50473")
    parser.add_argument("--franchise", default="Levered Leopolds")
    parser.add_argument(
        "--rookies",
        type=Path,
        default=Path("outputs/backtests/rookie_rankings_2026.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/backtests/rookie_market_2026.csv"),
    )
    args = parser.parse_args()

    mfl = MFLClient(args.year, args.league_id)
    league = mfl.snapshot(args.franchise)
    players = mfl_players_frame(mfl.players())
    adp = mfl_adp_frame(mfl.adp())
    with KalshiPublicClient() as client:
        kalshi_snapshot = fetch_current_market_snapshot(client)

    board, audit = build_rookie_market_board(
        pl.read_csv(args.rookies), players, adp, league, kalshi_snapshot
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.write_csv(args.output)
    kalshi_snapshot.write_csv(args.output.with_name("kalshi_market_snapshot.csv"))
    audit.write_csv(args.output.with_name("player_mapping_audit.csv"))

    print(
        board.select(
            "rookie_rank",
            "display_name",
            "position",
            "draft_team",
            pl.col("draft_capital").round(1).alias("baseline_points"),
            "average_pick",
            pl.col("kalshi_rookie_overall_probability")
            .round(3)
            .alias("kalshi_best_rookie_probability"),
            pl.col("kalshi_rookie_position_probability")
            .round(3)
            .alias("kalshi_position_rookie_probability"),
            pl.col("experimental_kalshi_signal")
            .round(3)
            .alias("market_signal"),
            "available",
        ).head(30)
    )


if __name__ == "__main__":
    main()
