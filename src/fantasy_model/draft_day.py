"""Read-only draft-day decision command for Old Mission vol '26.

The command refreshes authoritative MFL state, builds point-in-time player
distributions, simulates the remainder of the conference draft, and ranks the
user's next-pick policies by expected share of a 70/20/10 prize pool.  It never
submits a pick to MFL.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .data import load_player_stats, load_players
from .draft_simulator import DraftEVRanking, DraftSimulationConfig, rank_draft_candidates
from .league import LeagueSnapshot, MFLClient
from .market import mfl_adp_frame, mfl_players_frame
from .projections import build_all_player_projections


DEFAULT_CACHE = Path("data/cache")
DEFAULT_ROOKIES = Path("outputs/backtests/rookie_rankings_2026.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/draft_day")
DEFAULT_XOUTS = Path("data/draft_day/xouts.json")


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _fetch_mfl_inputs(
    client: MFLClient,
    franchise: str,
    cache_dir: Path,
    *,
    offline: bool = False,
) -> tuple[LeagueSnapshot, pl.DataFrame, pl.DataFrame, bool, list[str]]:
    """Fetch all MFL inputs, falling back only to a complete last-good cache."""
    names = ("league", "draftResults", "rosters", "players", "adp")
    warnings: list[str] = []
    stale = False
    if offline:
        missing = [name for name in names if not (cache_dir / f"{name}.json").exists()]
        if missing:
            raise RuntimeError(
                "Offline mode needs one successful live sync first; missing cached "
                + ", ".join(missing)
            )
        payloads = {
            name: _read_json(cache_dir / f"{name}.json") for name in names
        }
        snapshot = LeagueSnapshot.from_payloads(
            payloads["league"],
            payloads["draftResults"],
            franchise,
            roster_payload=payloads["rosters"],
        )
        players = mfl_players_frame(payloads["players"])
        adp = mfl_adp_frame(payloads["adp"])
        warnings.append("OFFLINE: using cached MFL player/ADP data plus manual picks")
        return snapshot, players, adp, False, warnings
    try:
        payloads = {
            "league": client.export("league"),
            "draftResults": client.export("draftResults"),
            "rosters": client.export("rosters"),
            "players": client.players(),
            "adp": client.adp(),
        }
        # Parse and validate before replacing the last-good cache.
        snapshot = LeagueSnapshot.from_payloads(
            payloads["league"],
            payloads["draftResults"],
            franchise,
            roster_payload=payloads["rosters"],
        )
        players = mfl_players_frame(payloads["players"])
        adp = mfl_adp_frame(payloads["adp"])
        if players.is_empty() or adp.is_empty():
            raise ValueError("MFL returned an empty player or ADP table")
        for name, payload in payloads.items():
            _write_json(cache_dir / f"{name}.json", payload)
    except Exception as error:
        missing = [name for name in names if not (cache_dir / f"{name}.json").exists()]
        if missing:
            raise RuntimeError(
                f"MFL refresh failed and no complete last-good cache exists ({error})"
            ) from error
        stale = True
        warnings.append(f"STALE MFL snapshot: live refresh failed ({error})")
        payloads = {
            name: _read_json(cache_dir / f"{name}.json") for name in names
        }
        snapshot = LeagueSnapshot.from_payloads(
            payloads["league"],
            payloads["draftResults"],
            franchise,
            roster_payload=payloads["rosters"],
        )
        players = mfl_players_frame(payloads["players"])
        adp = mfl_adp_frame(payloads["adp"])
    return snapshot, players, adp, stale, warnings


def apply_manual_picks(
    snapshot: LeagueSnapshot,
    projections: pl.DataFrame,
    path: Path,
    records: Iterable[str] = (),
    undos: Iterable[int] = (),
) -> LeagueSnapshot:
    """Overlay a durable, site-independent manual draft ledger."""
    records = tuple(records)
    undos = tuple(undos)
    saved = _read_json(path) if path.exists() else {"picks": {}}
    manual = {int(key): str(value) for key, value in saved.get("picks", {}).items()}
    valid_overalls = {pick.overall for pick in snapshot.picks}
    for overall in undos:
        if overall not in valid_overalls:
            raise ValueError(f"Unknown overall pick {overall}")
        manual.pop(overall, None)
    for record in records:
        if "=" not in record:
            raise ValueError(
                f"Manual pick {record!r} must be OVERALL=EXACT_PLAYER_NAME_OR_ID"
            )
        overall_text, raw_player = record.split("=", 1)
        try:
            overall = int(overall_text)
        except ValueError as error:
            raise ValueError(f"Invalid overall pick {overall_text!r}") from error
        if overall not in valid_overalls:
            raise ValueError(f"Unknown overall pick {overall}")
        player_id = next(
            iter(_resolve_values([raw_player], projections, label="manual pick"))
        )
        used_at = next(
            (pick for pick, existing in manual.items() if existing == player_id and pick != overall),
            None,
        )
        if used_at is not None:
            raise ValueError(f"Player {raw_player!r} is already recorded at pick {used_at}")
        manual[overall] = player_id
    if records or undos or not path.exists():
        _write_json(
            path,
            {
                "league_id": snapshot.league_id,
                "conference_id": snapshot.conference_id,
                "picks": {str(key): value for key, value in sorted(manual.items())},
            },
        )

    actual_by_overall = {pick.overall: pick.player_id for pick in snapshot.picks}
    conflicts = [
        overall
        for overall, player_id in manual.items()
        if actual_by_overall.get(overall)
        and actual_by_overall[overall] != player_id
    ]
    if conflicts:
        raise ValueError(
            "Manual ledger conflicts with live/cached MFL picks at overall "
            + ", ".join(map(str, conflicts))
        )
    picks = tuple(
        replace(
            pick,
            player_id=pick.player_id or manual.get(pick.overall),
            timestamp=pick.timestamp or ("manual" if pick.overall in manual else None),
        )
        for pick in snapshot.picks
    )
    return replace(snapshot, picks=picks)


def draft_state(snapshot: LeagueSnapshot, *, stale: bool = False) -> str:
    if stale:
        return "STALE"
    if snapshot.current_pick is None:
        return "COMPLETE"
    if not any(pick.processed for pick in snapshot.picks):
        return "PRE_DRAFT"
    return "ON_CLOCK" if snapshot.on_clock else "WAITING"


def _resolve_values(
    values: Iterable[str],
    projections: pl.DataFrame,
    *,
    label: str,
) -> set[str]:
    rows = projections.select("mfl_player_id", "name").to_dicts()
    by_id = {str(row["mfl_player_id"]): str(row["mfl_player_id"]) for row in rows}
    by_name: dict[str, list[str]] = {}
    for row in rows:
        by_name.setdefault(_normalized_name(str(row["name"])), []).append(
            str(row["mfl_player_id"])
        )
    resolved: set[str] = set()
    for raw in values:
        value = raw.strip()
        if value in by_id:
            resolved.add(by_id[value])
            continue
        matches = by_name.get(_normalized_name(value), [])
        if len(matches) == 1:
            resolved.add(matches[0])
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous {label} name {raw!r}; use an MFL player id")
        else:
            raise ValueError(f"Unknown {label} {raw!r}; use an exact name or MFL player id")
    return resolved


def update_xouts(
    path: Path,
    projections: pl.DataFrame,
    additions: Iterable[str],
    removals: Iterable[str],
) -> set[str]:
    additions = tuple(additions)
    removals = tuple(removals)
    saved = set(_read_json(path).get("mfl_player_ids", [])) if path.exists() else set()
    saved |= _resolve_values(additions, projections, label="X-out")
    saved -= _resolve_values(removals, projections, label="restore")
    if additions or removals:
        _write_json(path, {"mfl_player_ids": sorted(saved)})
    return saved


def _simulation_board(
    projections: pl.DataFrame,
    snapshot: LeagueSnapshot,
    xouts: set[str],
    required_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    """Make a bounded player universe while preserving every conditioned pick."""
    all_rows = projections.with_columns(
        pl.coalesce("adp", "adp_rank", pl.lit(9999.0)).cast(pl.Float64).alias("sim_adp")
    ).to_dicts()
    by_id = {str(row["mfl_player_id"]): row for row in all_rows}
    drafted = snapshot.drafted_player_ids
    missing_drafted = sorted(drafted - set(by_id))
    if missing_drafted:
        raise ValueError(
            "Draft state contains players absent from the scoring universe: "
            + ", ".join(missing_drafted)
        )

    valid = [
        row
        for row in all_rows
        if row["source"] != "unmatched_zero_prior"
        and float(row["mean_season_points"] or 0) > 0
    ]
    market = sorted(valid, key=lambda row: (float(row["sim_adp"]), row["mfl_player_id"]))
    selected_ids = {str(row["mfl_player_id"]) for row in market[:260]}
    for position in ("QB", "RB", "WR", "TE", "PK", "Def"):
        position_rows = sorted(
            (row for row in valid if row["position"] == position),
            key=lambda row: (-float(row["mean_season_points"]), float(row["sim_adp"])),
        )
        selected_ids.update(str(row["mfl_player_id"]) for row in position_rows[:60])
    selected_ids |= drafted
    selected_ids |= set(required_ids)

    board: list[dict[str, Any]] = []
    for player_id in selected_ids:
        row = by_id[player_id]
        board.append(
            {
                "id": player_id,
                "name": row["name"],
                "position": row["position"],
                "adp": float(row["sim_adp"]),
                "season_mean": float(row["mean_season_points"]),
                "season_sd": float(row["sd_season_points"]),
            }
        )

    # A rostered/keeper player is unavailable even if MFL draftResults does not
    # list it. Current live rosters are empty; its value is not yet conditioned
    # into the simulator, so surface that approximation explicitly.
    unavailable = set(snapshot.rostered_player_ids) | xouts
    warnings: list[str] = []
    if snapshot.rostered_player_ids:
        warnings.append(
            "Keeper/roster players are excluded, but their points are not yet "
            "conditioned into simulated roster strength"
        )
    unmatched = {
        str(row["mfl_player_id"])
        for row in all_rows
        if row["source"] == "unmatched_zero_prior"
    }
    unavailable |= unmatched
    return board, unavailable, warnings


def _draft_slots(snapshot: LeagueSnapshot) -> list[dict[str, Any]]:
    if snapshot.current_pick is None:
        return []
    current_overall = snapshot.current_pick.overall
    # Processed empty slots are MFL skips/timeouts and must not be replayed.
    return [
        {
            "overall": pick.overall,
            "franchise_id": pick.franchise_id,
            "player_id": pick.player_id,
        }
        for pick in snapshot.picks
        if pick.player_id or (not pick.processed and pick.overall >= current_overall)
    ]


def _market_context(player_id: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pl.read_csv(path, infer_schema_length=1000)
    if "mfl_player_id" not in frame.columns:
        return {}
    match = frame.filter(pl.col("mfl_player_id").cast(pl.String) == player_id)
    if match.is_empty():
        return {}
    row = match.row(0, named=True)
    return {
        key: row.get(key)
        for key in (
            "kalshi_rookie_overall_probability",
            "kalshi_rookie_position_probability",
            "experimental_kalshi_signal",
        )
        if row.get(key) is not None
    }


def _format_report(
    ranking: DraftEVRanking,
    projections: pl.DataFrame,
    snapshot: LeagueSnapshot,
    state: str,
    warnings: list[str],
    top: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    projection_by_id = {
        str(row["mfl_player_id"]): row for row in projections.to_dicts()
    }
    reference_name = projection_by_id.get(ranking.reference_player_id, {}).get(
        "name", ranking.reference_player_id
    )
    records: list[dict[str, Any]] = []
    for candidate in ranking.candidates[:top]:
        projection = projection_by_id[candidate.player_id]
        variance = (
            0.49 * candidate.first_probability
            + 0.04 * candidate.second_probability
            + 0.01 * candidate.third_probability
            - candidate.payout_equity**2
        )
        market = _market_context(
            candidate.player_id, Path("outputs/backtests/rookie_market_2026.csv")
        )
        explanation = (
            f"{projection['source']}; {projection['mean_season_points']:.1f}±"
            f"{projection['sd_season_points']:.1f} projected points; ADP "
            f"{candidate.adp:.1f}; variance is priced through top-three payout odds"
        )
        if market:
            explanation += "; Kalshi rookie signal is experimental (one validated season)"
        records.append(
            {
                **asdict(candidate),
                "team": projection["team"],
                "mean_season_points": projection["mean_season_points"],
                "sd_season_points": projection["sd_season_points"],
                "projection_source": projection["source"],
                "mc_standard_error": sqrt(max(variance, 0.0) / ranking.simulations),
                "why": explanation,
                **market,
            }
        )
    current = snapshot.current_pick
    next_mine = snapshot.next_my_pick
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "league": snapshot.league_name,
        "conference": snapshot.conference_name,
        "franchise": snapshot.franchise_name,
        "current_overall": current.overall if current else None,
        "current_franchise_id": current.franchise_id if current else None,
        "next_my_pick": next_mine.overall if next_mine else None,
        "picks_until_mine": (
            next_mine.overall - current.overall if current and next_mine else None
        ),
        "my_roster_player_ids": list(snapshot.my_player_ids),
        "simulations": ranking.simulations,
        "seed": ranking.seed,
        "payout_assumption": {"first": 0.70, "second": 0.20, "third": 0.10},
        "payout_field_size": ranking.payout_field_size,
        "field_approximation": ranking.field_approximation,
        "reference_policy_player": reference_name,
        "reference_payout_equity": ranking.reference_payout_equity,
        "warnings": warnings,
        "recommendations": records,
    }
    return report, records


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"{report['state']} | {report['franchise']} | current "
        f"{report['current_overall']} | next pick {report['next_my_pick']} "
        f"({report['picks_until_mine']} picks away)"
    )
    print(
        "EV is expected share of the prize pool under 70/20/10 payouts; "
        "multiply by the final pool dollars."
    )
    print(
        f"Reference policy: {report['reference_policy_player']} = "
        f"{100 * report['reference_payout_equity']:.2f}% payout equity"
    )
    print()
    print(" #  Player                    Pos Team  ADP  Avail   EV     ΔEV   1st/2nd/3rd")
    for row in report["recommendations"]:
        print(
            f"{row['rank']:>2}  {row['name'][:25]:<25} "
            f"{row['position']:<3} {str(row['team'] or 'FA'):<4} "
            f"{row['adp']:>5.1f} {100 * row['availability_probability']:>5.1f}% "
            f"{100 * row['payout_equity']:>6.2f}% "
            f"{100 * row['marginal_ev']:>+6.2f}pp "
            f"{100 * row['first_probability']:>4.1f}/"
            f"{100 * row['second_probability']:>4.1f}/"
            f"{100 * row['third_probability']:>4.1f}"
        )
    if report["recommendations"]:
        print(f"\nWhy #1: {report['recommendations'][0]['why']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank the next Levered Leopolds pick by marginal payout EV"
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--league", default="50473")
    parser.add_argument("--franchise", default="Levered Leopolds")
    parser.add_argument("--simulations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=50_473)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rookies", type=Path, default=DEFAULT_ROOKIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--web-output",
        type=Path,
        default=Path("public/draft-board.json"),
        help="Static data contract consumed by the emergency manual board",
    )
    parser.add_argument("--xouts", type=Path, default=DEFAULT_XOUTS)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never contact MFL; use cached league data and the manual pick ledger",
    )
    parser.add_argument(
        "--manual-state",
        type=Path,
        default=Path("data/draft_day/manual_state.json"),
    )
    parser.add_argument(
        "--record-pick",
        action="append",
        default=[],
        metavar="OVERALL=PLAYER",
        help="Record/replace a manual conference pick using an exact name or MFL id",
    )
    parser.add_argument("--undo-pick", action="append", default=[], type=int)
    args = parser.parse_args()

    client = MFLClient(args.year, args.league)
    mfl_cache = args.cache / f"mfl_{args.year}_{args.league}"
    snapshot, mfl_players, mfl_adp, stale, warnings = _fetch_mfl_inputs(
        client, args.franchise, mfl_cache, offline=args.offline
    )
    projections = build_all_player_projections(
        mfl_players,
        mfl_adp,
        load_player_stats(args.cache, range(args.year - 3, args.year)),
        load_players(args.cache),
        pl.read_csv(args.rookies, infer_schema_length=1000),
        target_season=args.year,
    )
    if args.offline or args.record_pick or args.undo_pick:
        snapshot = apply_manual_picks(
            snapshot,
            projections,
            args.manual_state,
            records=args.record_pick,
            undos=args.undo_pick,
        )
        warnings.append(f"Manual draft ledger: {args.manual_state}")
    state = draft_state(snapshot, stale=stale)
    if snapshot.current_pick is None or snapshot.next_my_pick is None:
        raise SystemExit("Draft complete: Levered Leopolds has no remaining pick")
    xouts = update_xouts(
        args.xouts, projections, tuple(args.exclude), tuple(args.include)
    )
    candidates = (
        sorted(_resolve_values(args.candidate, projections, label="candidate"))
        if args.candidate
        else None
    )
    board, unavailable, board_warnings = _simulation_board(
        projections, snapshot, xouts, required_ids=candidates or ()
    )
    warnings.extend(board_warnings)
    if stale:
        warnings.append("Fail-closed mode: verify MFL before acting on this board")
    if candidates and len(candidates) > args.max_candidates:
        raise ValueError("Explicit candidate count exceeds --max-candidates")
    ranking = rank_draft_candidates(
        board,
        _draft_slots(snapshot),
        snapshot.conference_franchise_ids,
        snapshot.franchise_id,
        exclusions=unavailable,
        candidate_ids=candidates,
        config=DraftSimulationConfig(
            simulations=args.simulations,
            seed=args.seed,
            max_candidates=args.max_candidates,
            payout_field_size=20,
        ),
    )
    report, records = _format_report(
        ranking, projections, snapshot, state, warnings, args.top
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "recommendations.json", report)
    pl.DataFrame(records).write_csv(args.output_dir / "recommendations.csv")
    projections.write_parquet(args.output_dir / "projections.parquet")
    web_players = (
        projections.filter(
            (pl.col("source") != "unmatched_zero_prior")
            & (pl.col("mean_season_points") > 0)
        )
        .with_columns(
            pl.coalesce("adp", "adp_rank", pl.lit(9999.0))
            .cast(pl.Float64)
            .alias("display_adp")
        )
        .sort("display_adp")
        .select(
            "mfl_player_id",
            "name",
            "position",
            "team",
            "display_adp",
            "mean_season_points",
            "sd_season_points",
            "source",
        )
        .to_dicts()
    )
    _write_json(
        args.web_output,
        {
            "generated_at": report["as_of"],
            "report": report,
            "draft_order": [
                {
                    "overall": pick.overall,
                    "round": pick.round,
                    "slot": pick.slot,
                    "franchise_id": pick.franchise_id,
                    "player_id": pick.player_id,
                }
                for pick in snapshot.picks
            ],
            "players": web_players,
            "xouts": sorted(xouts),
        },
    )
    _print_report(report)
    print(f"\nArtifacts: {args.output_dir / 'recommendations.json'}")


if __name__ == "__main__":
    main()
