import numpy as np
import pytest

from fantasy_model.utility import (
    PayoutStructure,
    PlayerDistribution,
    optimal_week_score,
    payout_equity,
)


def test_flexible_lineup_optimizer_uses_eight_skill_players() -> None:
    players = [
        ("QB", 20),
        ("PK", 8),
        ("Def", 7),
        *(('RB', score) for score in [18, 17, 16, 1]),
        *(('WR', score) for score in [15, 14, 13, 12, 11]),
        *(('TE', score) for score in [10, 2, 1]),
    ]
    assert optimal_week_score(players) == 150


def test_payout_equity_uses_70_20_10_utility() -> None:
    our_scores = np.array([100.0, 90.0, 80.0, 70.0])
    opponents = np.array(
        [
            [99, 98, 97],
            [100, 89, 88],
            [100, 90, 79],
            [100, 90, 80],
        ],
        dtype=float,
    )
    assert payout_equity(our_scores, opponents, PayoutStructure()) == pytest.approx(0.25)


def test_player_distribution_keeps_upside_as_distribution_not_bonus() -> None:
    rookie = PlayerDistribution("r", "Rookie", "WR", 8, 9, 0.9)
    assert rookie.weekly_sd == 9
    assert not hasattr(rookie, "upside_bonus")
