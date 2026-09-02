from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PayoutStructure:
    first: float = 0.70
    second: float = 0.20
    third: float = 0.10

    def __post_init__(self) -> None:
        shares = (self.first, self.second, self.third)
        if any(share < 0 for share in shares):
            raise ValueError("Payout shares cannot be negative")
        if not np.isclose(sum(shares), 1.0):
            raise ValueError("Payout shares must sum to 1")

    def for_rank(self, rank: int) -> float:
        return (self.first, self.second, self.third)[rank - 1] if rank <= 3 else 0.0


@dataclass(frozen=True)
class PlayerDistribution:
    player_id: str
    name: str
    position: str
    weekly_mean: float
    weekly_sd: float
    active_probability: float = 1.0


SKILL_POSITION_COUNTS = tuple(
    counts
    for counts in product(range(2, 5), range(3, 6), range(1, 4))
    if sum(counts) == 8
)


def optimal_week_score(players: Sequence[tuple[str, float]]) -> float:
    """Optimize one weekly lineup under the exact flexible starter limits."""
    scores: dict[str, list[float]] = {position: [] for position in ("QB", "RB", "WR", "TE", "PK", "Def")}
    for position, score in players:
        normalized = "Def" if position.casefold() in {"def", "dst"} else position.upper()
        if normalized in scores:
            scores[normalized].append(float(score))
    for values in scores.values():
        values.sort(reverse=True)

    fixed = sum(scores["QB"][:1]) + sum(scores["PK"][:1]) + sum(scores["Def"][:1])
    skill_totals: list[float] = []
    for rb_count, wr_count, te_count in SKILL_POSITION_COUNTS:
        if (
            len(scores["RB"]) >= rb_count
            and len(scores["WR"]) >= wr_count
            and len(scores["TE"]) >= te_count
        ):
            skill_totals.append(
                sum(scores["RB"][:rb_count])
                + sum(scores["WR"][:wr_count])
                + sum(scores["TE"][:te_count])
            )
    return fixed + (max(skill_totals) if skill_totals else 0.0)


def simulate_season_scores(
    roster: Sequence[PlayerDistribution],
    *,
    simulations: int = 2_000,
    weeks: int = 14,
    seed: int = 50473,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    totals = np.zeros(simulations, dtype=float)
    for simulation in range(simulations):
        for _week in range(weeks):
            weekly: list[tuple[str, float]] = []
            for player in roster:
                active = rng.random() <= player.active_probability
                score = (
                    max(-3.0, rng.normal(player.weekly_mean, player.weekly_sd))
                    if active
                    else 0.0
                )
                weekly.append((player.position, score))
            totals[simulation] += optimal_week_score(weekly)
    return totals


def payout_equity(
    our_scores: np.ndarray,
    opponent_scores: np.ndarray,
    payout: PayoutStructure = PayoutStructure(),
) -> float:
    if opponent_scores.ndim != 2 or opponent_scores.shape[0] != our_scores.shape[0]:
        raise ValueError("Opponent scores must be shaped [simulations, opponents]")
    ranks = 1 + np.sum(opponent_scores > our_scores[:, None], axis=1)
    utilities = np.select(
        [ranks == 1, ranks == 2, ranks == 3],
        [payout.first, payout.second, payout.third],
        default=0.0,
    )
    return float(np.mean(utilities))


def marginal_payout_ev(
    roster: Sequence[PlayerDistribution],
    candidate: PlayerDistribution,
    alternative: PlayerDistribution,
    opponent_scores: np.ndarray,
    *,
    payout: PayoutStructure = PayoutStructure(),
    weeks: int = 14,
    seed: int = 50473,
) -> float:
    simulations = opponent_scores.shape[0]
    candidate_scores = simulate_season_scores(
        [*roster, candidate], simulations=simulations, weeks=weeks, seed=seed
    )
    alternative_scores = simulate_season_scores(
        [*roster, alternative], simulations=simulations, weeks=weeks, seed=seed
    )
    return payout_equity(candidate_scores, opponent_scores, payout) - payout_equity(
        alternative_scores, opponent_scores, payout
    )

