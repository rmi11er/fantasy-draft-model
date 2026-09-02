from pathlib import Path

import polars as pl
import pytest

from fantasy_model.draft_day import apply_manual_picks, draft_state, update_xouts
from fantasy_model.league import LeagueSnapshot
from test_league import _draft_payload, _league_payload


def _projections() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mfl_player_id": ["p1", "p2"],
            "name": ["Jahmyr Gibbs", "Drake Maye"],
        }
    )


def test_manual_pick_ledger_advances_offline_state_and_can_undo(tmp_path: Path) -> None:
    payload = _draft_payload()
    payload["draftResults"]["draftUnit"][0]["draftPick"][0]["player"] = ""
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), payload, "Levered Leopolds"
    )
    path = tmp_path / "manual.json"

    picked = apply_manual_picks(
        snapshot, _projections(), path, records=["1=Jahmyr Gibbs"]
    )
    assert picked.current_pick.overall == 2
    assert picked.drafted_player_ids == frozenset({"p1"})
    assert draft_state(picked) == "ON_CLOCK"

    undone = apply_manual_picks(snapshot, _projections(), path, undos=[1])
    assert undone.current_pick.overall == 1
    assert draft_state(undone) == "PRE_DRAFT"


def test_manual_ledger_rejects_duplicate_players(tmp_path: Path) -> None:
    payload = _draft_payload()
    payload["draftResults"]["draftUnit"][0]["draftPick"][0]["player"] = ""
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), payload, "Levered Leopolds"
    )
    with pytest.raises(ValueError, match="already recorded"):
        apply_manual_picks(
            snapshot,
            _projections(),
            tmp_path / "manual.json",
            records=["1=Jahmyr Gibbs", "2=Jahmyr Gibbs"],
        )


def test_xouts_require_exact_names_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "xouts.json"
    assert update_xouts(path, _projections(), ["Drake Maye"], []) == {"p2"}
    assert update_xouts(path, _projections(), [], ["p2"]) == set()
    with pytest.raises(ValueError, match="Unknown X-out"):
        update_xouts(path, _projections(), ["Maye"], [])
