"""Export a human-readable pre-draft cheat sheet from frozen model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


POSITION_DEMAND = {"QB": 10, "RB": 30, "WR": 40, "TE": 10, "PK": 10, "DEF": 10}


def _position(value: str) -> str:
    return "DEF" if value == "Def" else value


def _number(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def export_rankings(
    projections_path: Path,
    board_path: Path,
    output_path: Path,
    *,
    limit: int = 1000,
) -> int:
    projections = pl.read_parquet(projections_path).to_dicts()
    board = json.loads(board_path.read_text())
    recommendations = board["report"]["recommendations"]
    exact_by_id = {str(row["player_id"]): row for row in recommendations}

    draftable = [
        row
        for row in projections
        if row["source"] != "unmatched_zero_prior"
        and _number(row["mean_season_points"]) > 0
    ]
    replacement: dict[str, float] = {}
    for position, demand in POSITION_DEMAND.items():
        values = sorted(
            (
                _number(row["mean_season_points"])
                for row in draftable
                if _position(str(row["position"])) == position
            ),
            reverse=True,
        )
        replacement[position] = values[min(demand, len(values)) - 1] if values else 0.0

    ranked: list[dict[str, Any]] = []
    for row in projections:
        player_id = str(row["mfl_player_id"])
        position = _position(str(row["position"]))
        mean = _number(row["mean_season_points"])
        sd = _number(row["sd_season_points"])
        adp = _number(row.get("adp"), _number(row.get("adp_rank"), 9999.0))
        exact = exact_by_id.get(player_id)
        is_draftable = row["source"] != "unmatched_zero_prior" and mean > 0
        vorp = mean - replacement.get(position, 0.0) if is_draftable else float("-inf")
        if exact is not None:
            sort_key = (0, int(exact["rank"]), 0.0, 0.0, 0.0, player_id)
        elif is_draftable:
            sort_key = (1, 0, -vorp, -mean, adp, player_id)
        else:
            sort_key = (2, 0, 0.0, 0.0, adp, player_id)
        ranked.append(
            {
                "sort_key": sort_key,
                "id": player_id,
                "name": str(row["name"]),
                "position": position,
                "team": str(row.get("team") or "FA"),
                "mean": mean,
                "sd": sd,
                "adp": adp,
                "vorp": vorp,
                "source": str(row["source"]),
                "exact": exact,
                "draftable": is_draftable,
            }
        )

    ranked.sort(key=lambda row: row["sort_key"])
    ranked = ranked[:limit]
    lines = [
        "LEVERED LEOPOLDS — PRE-DRAFT TOP 1,000",
        f"Frozen model: {board['generated_at']}",
        "Scoring: league 50473 | Prize utility: 70% / 20% / 10%",
        "",
        "READ THIS:",
        "  Rows 1–8 have paired, full-draft simulated Prize EV at pick 1.02.",
        "  Rows 9–1,000 use the model's baseline draft priority: projected season",
        "  points above position replacement (VORP). Their Prize EV is deliberately",
        "  left blank rather than fabricated; recompute EV for the live shortlist.",
        "  UNMATCHED rows have no defensible projection and are marked DO NOT DRAFT.",
        "",
        "RANK  PLAYER                              POS TEAM   PRIZE_EV  dEV_PP   AVAIL  PROJ_MEAN  PROJ_SD   VORP    ADP      BASIS",
        "----  ----------------------------------  --- ----   --------  -------  -----  ---------  --------  ------  -------  ----------------",
    ]
    for rank, row in enumerate(ranked, start=1):
        exact = row["exact"]
        if exact is not None:
            prize_ev = f"{100 * _number(exact['payout_equity']):7.2f}%"
            marginal = f"{100 * _number(exact['marginal_ev']):+6.2f}"
            available = f"{100 * _number(exact['availability_probability']):4.0f}%"
            basis = "SIMULATED EV"
        else:
            prize_ev = "       —"
            marginal = "      —"
            available = "    —"
            basis = "VORP PRIORITY" if row["draftable"] else "DO NOT DRAFT"
        vorp = f"{row['vorp']:6.1f}" if row["draftable"] else "     —"
        adp = f"{row['adp']:7.1f}" if row["adp"] < 9999 else "      —"
        lines.append(
            f"{rank:4d}  {row['name'][:34]:34}  {row['position']:>3} "
            f"{row['team'][:4]:4}  {prize_ev:>8}  {marginal:>7}  {available:>5}  "
            f"{row['mean']:9.1f}  {row['sd']:8.1f}  {vorp:>6}  {adp:>7}  {basis}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return len(ranked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections", type=Path, default=Path("outputs/draft_day/projections.parquet")
    )
    parser.add_argument("--board", type=Path, default=Path("public/draft-board.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/draft_day/top_1000_pre_draft_rankings.txt"),
    )
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    count = export_rankings(args.projections, args.board, args.output, limit=args.limit)
    print(f"Wrote {count} players to {args.output}")


if __name__ == "__main__":
    main()
