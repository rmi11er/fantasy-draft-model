from __future__ import annotations

import numpy as np
import pytest

from fantasy_model.draft_simulator import (
    DraftSimulationConfig,
    DraftSimulator,
    rank_draft_candidates,
)


def _board() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specifications = {
        "QB": (30, 250.0),
        "RB": (100, 190.0),
        "WR": (70, 180.0),
        "TE": (30, 125.0),
        "PK": (30, 105.0),
        "DEF": (30, 100.0),
    }
    adp = 1
    for position, (count, starting_mean) in specifications.items():
        for index in range(count):
            mean = max(35.0, starting_mean - index * 3.0)
            rows.append(
                {
                    "id": f"{position}_{index}",
                    "name": f"{position} Player {index}",
                    "position": position,
                    "adp": float(adp),
                    "season_mean": mean,
                    "season_sd": 15.0,
                }
            )
            adp += 1
    # The two explicit choices are not conflated with ADP order.
    elite = next(row for row in rows if row["id"] == "RB_0")
    replacement = next(row for row in rows if row["id"] == "RB_1")
    elite["season_mean"], elite["adp"] = 800.0, 1.0
    replacement["season_mean"], replacement["adp"] = 120.0, 2.0
    return rows


def _slots() -> list[dict[str, object]]:
    franchises = [f"F{index}" for index in range(10)]
    slots: list[dict[str, object]] = []
    overall = 1
    for round_number in range(18):
        order = franchises if round_number % 2 == 0 else list(reversed(franchises))
        for franchise in order:
            slots.append({"overall": overall, "franchise_id": franchise})
            overall += 1
    # The user (F1) is on the clock at pick 2 and pick 1 is authoritative state.
    slots[0]["player_id"] = "QB_0"
    return slots


def _external_slots() -> list[dict[str, object]]:
    return [
        {
            **slot,
            "franchise_id": str(slot["franchise_id"]).replace("F", "E"),
        }
        for slot in _slots()
    ]


def _config(seed: int = 123) -> DraftSimulationConfig:
    return DraftSimulationConfig(simulations=20, seed=seed, max_candidates=4)


def test_candidate_ranking_is_reproducible_and_uses_payout_equity() -> None:
    kwargs = dict(
        player_board=_board(),
        draft_slots=_slots(),
        franchises=[f"F{index}" for index in range(10)],
        user_franchise="F1",
        candidate_ids=["RB_0", "RB_1"],
        config=_config(),
    )
    first = rank_draft_candidates(**kwargs)
    second = rank_draft_candidates(**kwargs)

    assert first == second
    assert first.current_overall == 2
    assert first.seed == 123
    assert first.candidates[0].player_id == "RB_0"
    assert first.candidates[0].payout_equity > first.candidates[1].payout_equity
    assert first.candidates[0].statistically_tied_for_best
    assert first.candidates[0].best_ev_gap_standard_error == 0
    assert all(candidate.marginal_ev_standard_error >= 0 for candidate in first.candidates)
    assert all(
        0 <= candidate.first_probability + candidate.second_probability + candidate.third_probability <= 1
        for candidate in first.candidates
    )


def test_existing_pick_and_x_out_make_player_unselectable() -> None:
    simulator = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        exclusions={"RB_1"},
        config=_config(),
    )
    with pytest.raises(ValueError, match="not selectable"):
        simulator.evaluate_forced_candidate("QB_0")
    with pytest.raises(ValueError, match="not selectable"):
        simulator.evaluate_forced_candidate("RB_1")

    personal_xout = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        user_exclusions={"RB_1"},
        config=_config(),
    )
    with pytest.raises(ValueError, match="not selectable"):
        personal_xout.evaluate_forced_candidate("RB_1")
    assert "RB_1" in {
        player.player_id
        for player in personal_xout._available(set(), [], 17, for_user=False)
    }


def test_forced_candidate_reports_marginal_ev_against_reference_policy() -> None:
    simulator = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        config=_config(seed=77),
    )
    candidate = simulator.evaluate_forced_candidate("RB_0")

    assert candidate.payout_equity - candidate.marginal_ev == pytest.approx(
        simulator.rank_candidates(["RB_0"]).reference_payout_equity
    )
    assert not hasattr(candidate, "upside_bonus")


def test_pre_draft_simulates_intervening_picks_and_bounds_candidate_batch() -> None:
    slots = _slots()
    slots[0].pop("player_id")
    pre_draft = DraftSimulator(
        _board(), slots, [f"F{index}" for index in range(10)], "F1", config=_config()
    )
    result = pre_draft.rank_candidates(["RB_0"])
    assert result.current_overall == 2
    assert 0 <= result.candidates[0].availability_probability <= 1
    assert result.payout_field_size == 20

    simulator = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        config=_config(),
    )
    with pytest.raises(ValueError, match="At most 4"):
        simulator.rank_candidates(["RB_0", "RB_1", "RB_2", "RB_3", "RB_4"])


def test_scenarios_use_antithetic_pairs_to_reduce_monte_carlo_noise() -> None:
    simulator = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        config=DraftSimulationConfig(simulations=3, seed=9, max_candidates=2),
    )
    first, second, _unpaired = simulator._scenarios

    np.testing.assert_array_equal(first.draft_noise, -second.draft_noise)
    np.testing.assert_array_equal(first.outcome_z, -second.outcome_z)


def test_roster_capacity_must_support_all_eleven_starters() -> None:
    with pytest.raises(ValueError, match="needs 11 positions"):
        DraftSimulator(
            _board(),
            _slots()[:100],
            [f"F{index}" for index in range(10)],
            "F1",
            config=_config(),
        )


def test_exact_external_conference_shares_player_outcome_scenarios() -> None:
    simulator = DraftSimulator(
        _board(),
        _slots(),
        [f"F{index}" for index in range(10)],
        "F1",
        external_draft_slots=_external_slots(),
        external_franchises=[f"E{index}" for index in range(10)],
        config=_config(),
    )
    ranking = simulator.rank_candidates(["RB_0"])

    assert simulator._conditioned_rosters["F0"] == ["QB_0"]
    assert simulator._external_conditioned_rosters["E0"] == ["QB_0"]
    assert ranking.field_approximation == (
        "exact_two_conference_drafts_shared_player_outcomes"
    )
