"""Bounded, reproducible draft simulation for marginal payout-EV decisions.

The simulator is deliberately a policy evaluator, not another projection model.
It consumes point-in-time player distributions, completes the conference draft,
and measures the payout equity caused by forcing a player at the current pick.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite, log, log1p, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .utility import PayoutStructure, optimal_week_score


POSITIONS = ("QB", "RB", "WR", "TE", "PK", "DEF")
SKILL_POSITIONS = ("RB", "WR", "TE")
STARTING_SKILL_PLAYERS = 8


def _position(value: Any) -> str:
    normalized = str(value).strip().upper().replace("/", "")
    aliases = {"K": "PK", "DST": "DEF", "D-ST": "DEF", "D/ST": "DEF"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in POSITIONS:
        raise ValueError(f"Unsupported position: {value!r}")
    return normalized


@dataclass(frozen=True)
class DraftPlayer:
    player_id: str
    name: str
    position: str
    adp: float
    season_mean: float
    season_sd: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "DraftPlayer":
        try:
            player = cls(
                player_id=str(row["id"]),
                name=str(row["name"]),
                position=_position(row["position"]),
                adp=float(row["adp"]),
                season_mean=float(row["season_mean"]),
                season_sd=float(row["season_sd"]),
            )
        except KeyError as error:
            raise ValueError(f"Player board is missing required column {error.args[0]!r}") from error
        if not player.player_id:
            raise ValueError("Player id cannot be empty")
        values = (player.adp, player.season_mean, player.season_sd)
        if not all(isfinite(value) for value in values):
            raise ValueError(f"Player {player.player_id} has a non-finite projection")
        if player.adp <= 0 or player.season_mean < 0 or player.season_sd < 0:
            raise ValueError(f"Player {player.player_id} has an invalid ADP/distribution")
        if player.season_mean == 0 and player.season_sd > 0:
            raise ValueError(
                f"Player {player.player_id} cannot have positive variance with zero mean"
            )
        return player


@dataclass(frozen=True)
class DraftSlot:
    overall: int
    franchise_id: str
    player_id: str | None = None

    @classmethod
    def from_value(cls, value: "DraftSlot | Mapping[str, Any]") -> "DraftSlot":
        if isinstance(value, cls):
            return value
        player_id = value.get("player_id", value.get("player"))
        return cls(
            overall=int(value["overall"]),
            franchise_id=str(value.get("franchise_id", value.get("franchise"))),
            player_id=str(player_id) if player_id else None,
        )


@dataclass(frozen=True)
class RosterRules:
    """League minimums plus practical draft-policy caps, not league limits."""

    minimums: Mapping[str, int] = field(
        default_factory=lambda: {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "PK": 1,
            "DEF": 1,
        }
    )
    policy_caps: Mapping[str, int] = field(
        default_factory=lambda: {
            "QB": 2,
            "RB": 8,
            "WR": 8,
            "TE": 4,
            "PK": 2,
            "DEF": 2,
        }
    )

    def __post_init__(self) -> None:
        for position in POSITIONS:
            minimum = int(self.minimums.get(position, 0))
            cap = int(self.policy_caps.get(position, 0))
            if minimum < 0 or cap < minimum:
                raise ValueError(f"Invalid roster bounds for {position}")


@dataclass(frozen=True)
class DraftSimulationConfig:
    simulations: int = 400
    seed: int = 50_473
    adp_noise: float = 0.38
    max_candidates: int = 16
    payout_field_size: int = 20
    antithetic_sampling: bool = True
    payout: PayoutStructure = field(default_factory=PayoutStructure)
    roster_rules: RosterRules = field(default_factory=RosterRules)

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.adp_noise < 0:
            raise ValueError("adp_noise cannot be negative")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.payout_field_size < 10:
            raise ValueError("payout_field_size cannot be smaller than the conference")


@dataclass(frozen=True)
class CandidateEV:
    rank: int
    player_id: str
    name: str
    position: str
    adp: float
    payout_equity: float
    marginal_ev: float
    first_probability: float
    second_probability: float
    third_probability: float
    availability_probability: float
    marginal_ev_standard_error: float
    best_ev_gap_standard_error: float
    statistically_tied_for_best: bool


@dataclass(frozen=True)
class DraftEVRanking:
    seed: int
    simulations: int
    current_overall: int
    reference_player_id: str
    reference_payout_equity: float
    payout_field_size: int
    field_approximation: str
    candidates: tuple[CandidateEV, ...]


@dataclass(frozen=True)
class _Scenario:
    # Each simulated opponent gets a stable noisy board; every candidate is
    # evaluated against the same noise and same player outcome shocks.
    draft_noise: np.ndarray
    outcome_z: np.ndarray
    external_scenario_indices: np.ndarray
    external_opponent_indices: np.ndarray


@dataclass(frozen=True)
class _Equity:
    payout: float
    first: float
    second: float
    third: float
    availability: float
    most_common_player_id: str
    payout_samples: tuple[float, ...]


def _paired_standard_error(first: _Equity, second: _Equity) -> float:
    """Monte Carlo error for a difference under common random numbers."""
    differences = np.asarray(first.payout_samples) - np.asarray(second.payout_samples)
    if differences.size < 2:
        return 0.0
    return float(np.std(differences, ddof=1) / sqrt(differences.size))


def _board_rows(board: Any) -> list[Mapping[str, Any]]:
    if hasattr(board, "to_dicts"):  # Polars
        return list(board.to_dicts())
    if hasattr(board, "to_dict"):  # Pandas
        try:
            return list(board.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(board, Mapping):
        raise ValueError("Player board must be a table or a sequence of rows")
    return list(board)


class DraftSimulator:
    """Evaluate current-pick choices in a ten-franchise conference draft."""

    def __init__(
        self,
        player_board: Any,
        draft_slots: Sequence[DraftSlot | Mapping[str, Any]],
        franchises: Sequence[str],
        user_franchise: str,
        *,
        exclusions: Iterable[str] = (),
        user_exclusions: Iterable[str] = (),
        external_draft_slots: Sequence[DraftSlot | Mapping[str, Any]] = (),
        external_franchises: Sequence[str] = (),
        external_exclusions: Iterable[str] = (),
        config: DraftSimulationConfig = DraftSimulationConfig(),
    ) -> None:
        self.players = tuple(DraftPlayer.from_mapping(row) for row in _board_rows(player_board))
        self.player_by_id = {player.player_id: player for player in self.players}
        if len(self.player_by_id) != len(self.players):
            raise ValueError("Player board contains duplicate ids")

        self.franchises = tuple(str(franchise) for franchise in franchises)
        if len(self.franchises) != 10 or len(set(self.franchises)) != 10:
            raise ValueError("This engine requires exactly ten unique conference franchises")
        self.user_franchise = str(user_franchise)
        if self.user_franchise not in self.franchises:
            raise ValueError("User franchise is not in the conference")

        self.slots = tuple(
            sorted((DraftSlot.from_value(slot) for slot in draft_slots), key=lambda slot: slot.overall)
        )
        if not self.slots or len({slot.overall for slot in self.slots}) != len(self.slots):
            raise ValueError("Draft order must contain unique overall picks")
        if any(slot.franchise_id not in self.franchises for slot in self.slots):
            raise ValueError("Draft order contains an unknown franchise")
        if any(slot.overall <= 0 for slot in self.slots):
            raise ValueError("Overall picks must be positive")

        self.external_franchises = tuple(
            str(franchise) for franchise in external_franchises
        )
        self.external_slots = tuple(
            sorted(
                (DraftSlot.from_value(slot) for slot in external_draft_slots),
                key=lambda slot: slot.overall,
            )
        )
        if bool(self.external_franchises) != bool(self.external_slots):
            raise ValueError(
                "External conference franchises and draft slots must be supplied together"
            )
        if self.external_franchises:
            expected_external = config.payout_field_size - len(self.franchises)
            if (
                len(self.external_franchises) != expected_external
                or len(set(self.external_franchises)) != expected_external
                or set(self.external_franchises) & set(self.franchises)
            ):
                raise ValueError("External conference franchises must complete the payout field")
            if len({slot.overall for slot in self.external_slots}) != len(
                self.external_slots
            ):
                raise ValueError("External draft order must contain unique overall picks")
            if any(
                slot.franchise_id not in self.external_franchises
                for slot in self.external_slots
            ):
                raise ValueError("External draft order contains an unknown franchise")

        self.config = config
        self.exclusions = frozenset(str(player_id) for player_id in exclusions)
        self.user_exclusions = frozenset(
            str(player_id) for player_id in user_exclusions
        )
        self.external_exclusions = frozenset(
            str(player_id) for player_id in external_exclusions
        )
        self._player_index = {
            player.player_id: index for index, player in enumerate(self.players)
        }
        all_franchises = (*self.franchises, *self.external_franchises)
        self._franchise_index = {
            franchise: index for index, franchise in enumerate(all_franchises)
        }
        self._conditioned_rosters, self._conditioned_ids = self._condition_picks(
            self.slots, self.franchises
        )
        (
            self._external_conditioned_rosters,
            self._external_conditioned_ids,
        ) = self._condition_picks(self.external_slots, self.external_franchises)
        # PRE_DRAFT and WAITING are valid states: opponents ahead of the user's
        # next pick are simulated rather than requiring the user to be on clock.
        self.current_slot = next(
            (
                slot
                for slot in self.slots
                if slot.player_id is None and slot.franchise_id == self.user_franchise
            ),
            None,
        )
        if self.current_slot is None:
            raise ValueError("The user franchise has no unfilled picks")

        self._unfilled_after = self._count_unfilled_after(self.slots)
        self._external_unfilled_after = self._count_unfilled_after(
            self.external_slots
        )
        self._validate_roster_capacity(
            self.franchises,
            self.slots,
            self._conditioned_rosters,
        )
        if self.external_franchises:
            self._validate_roster_capacity(
                self.external_franchises,
                self.external_slots,
                self._external_conditioned_rosters,
            )
        self._replacement_points = self._calculate_replacement_points()
        self._scenarios = self._make_scenarios()
        self._equity_cache: dict[str | None, _Equity] = {}
        self._external_score_rows: tuple[dict[str, float], ...] | None = None

    def _condition_picks(
        self,
        slots: Sequence[DraftSlot],
        franchises: Sequence[str],
    ) -> tuple[dict[str, list[str]], frozenset[str]]:
        rosters = {franchise: [] for franchise in franchises}
        drafted: set[str] = set()
        for slot in slots:
            if slot.player_id is None:
                continue
            if slot.player_id not in self.player_by_id:
                raise ValueError(f"Drafted player {slot.player_id!r} is missing from the board")
            if slot.player_id in drafted:
                raise ValueError(f"Player {slot.player_id!r} appears in multiple draft slots")
            rosters[slot.franchise_id].append(slot.player_id)
            drafted.add(slot.player_id)
        return rosters, frozenset(drafted)

    def _count_unfilled_after(
        self, slots: Sequence[DraftSlot]
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        for slot in slots:
            result[slot.overall] = sum(
                later.player_id is None
                and later.overall > slot.overall
                and later.franchise_id == slot.franchise_id
                for later in slots
            )
        return result

    def _validate_roster_capacity(
        self,
        franchises: Sequence[str],
        slots: Sequence[DraftSlot],
        conditioned_rosters: Mapping[str, Sequence[str]],
    ) -> None:
        minimum_total = STARTING_SKILL_PLAYERS + sum(
            self.config.roster_rules.minimums[position]
            for position in ("QB", "PK", "DEF")
        )
        for franchise in franchises:
            total_slots = sum(slot.franchise_id == franchise for slot in slots)
            if total_slots < minimum_total:
                raise ValueError(
                    f"Franchise {franchise} has {total_slots} picks but needs {minimum_total} positions"
                )
            existing = Counter(
                self.player_by_id[player_id].position
                for player_id in conditioned_rosters[franchise]
            )
            remaining = sum(
                slot.player_id is None and slot.franchise_id == franchise for slot in slots
            )
            deficit = self._minimum_picks_needed(existing)
            if deficit > remaining:
                raise ValueError(f"Existing picks make {franchise}'s roster infeasible")

    def _minimum_picks_needed(self, counts: Counter[str]) -> int:
        fixed_deficit = sum(
            max(0, self.config.roster_rules.minimums[position] - counts[position])
            for position in ("QB", "PK", "DEF")
        )
        skill_position_deficit = sum(
            max(0, self.config.roster_rules.minimums[position] - counts[position])
            for position in SKILL_POSITIONS
        )
        skill_total_deficit = max(
            0,
            STARTING_SKILL_PLAYERS
            - sum(counts[position] for position in SKILL_POSITIONS),
        )
        return fixed_deficit + max(skill_position_deficit, skill_total_deficit)

    def _calculate_replacement_points(self) -> dict[str, float]:
        # Expected starter demand in the league's flexible lineup: a neutral
        # 3 RB / 4 WR / 1 TE allocation for the eight skill slots.
        demand_per_team = {"QB": 1, "RB": 3, "WR": 4, "TE": 1, "PK": 1, "DEF": 1}
        replacement: dict[str, float] = {}
        for position in POSITIONS:
            values = sorted(
                (
                    player.season_mean
                    for player in self.players
                    if player.position == position
                ),
                reverse=True,
            )
            target = demand_per_team[position] * len(self.franchises)
            replacement[position] = values[min(target, len(values)) - 1] if values else 0.0
        return replacement

    def _make_scenarios(self) -> tuple[_Scenario, ...]:
        rng = np.random.default_rng(self.config.seed)
        scenarios: list[_Scenario] = []

        def external_field_indices(scenario_index: int) -> tuple[np.ndarray, np.ndarray]:
            count = self.config.payout_field_size - len(self.franchises)
            # Evenly rotate through independent scenario/team combinations.
            # This stratifies the unknown conference rather than adding another
            # noisy bootstrap draw to an already finite Monte Carlo estimate.
            sources = np.asarray(
                [
                    (
                        scenario_index
                        + ((field_index + 1) * self.config.simulations) // (count + 1)
                    )
                    % self.config.simulations
                    for field_index in range(count)
                ],
                dtype=int,
            )
            opponents = np.asarray(
                [
                    (scenario_index + field_index) % (len(self.franchises) - 1)
                    for field_index in range(count)
                ],
                dtype=int,
            )
            return sources, opponents

        while len(scenarios) < self.config.simulations:
            draft_noise = rng.normal(
                0.0,
                self.config.adp_noise,
                size=(len(self._franchise_index), len(self.players)),
            )
            outcome_z = rng.standard_normal(len(self.players))
            external_scenario_indices, external_opponent_indices = (
                external_field_indices(len(scenarios))
            )
            scenarios.append(
                _Scenario(
                    draft_noise=draft_noise,
                    outcome_z=outcome_z,
                    external_scenario_indices=external_scenario_indices,
                    external_opponent_indices=external_opponent_indices,
                )
            )
            if self.config.antithetic_sampling and len(scenarios) < self.config.simulations:
                external_scenario_indices, external_opponent_indices = (
                    external_field_indices(len(scenarios))
                )
                scenarios.append(
                    _Scenario(
                        draft_noise=-draft_noise,
                        outcome_z=-outcome_z,
                        external_scenario_indices=external_scenario_indices,
                        external_opponent_indices=external_opponent_indices,
                    )
                )
        return tuple(scenarios)

    def _feasible(
        self,
        player: DraftPlayer,
        roster_ids: Sequence[str],
        picks_after: int,
    ) -> bool:
        counts = Counter(self.player_by_id[player_id].position for player_id in roster_ids)
        return self._feasible_with_counts(player.position, counts, picks_after)

    def _feasible_with_counts(
        self,
        position: str,
        counts: Counter[str],
        picks_after: int,
    ) -> bool:
        if counts[position] >= self.config.roster_rules.policy_caps[position]:
            return False
        counts = counts.copy()
        counts[position] += 1
        deficit = self._minimum_picks_needed(counts)
        return deficit <= picks_after

    def _available(
        self,
        selected_ids: set[str],
        roster_ids: Sequence[str],
        picks_after: int,
        *,
        for_user: bool = False,
        exclusions: frozenset[str] | None = None,
    ) -> list[DraftPlayer]:
        counts = Counter(self.player_by_id[player_id].position for player_id in roster_ids)
        blocked = self.exclusions if exclusions is None else exclusions
        return [
            player
            for player in self.players
            if player.player_id not in selected_ids
            and player.player_id not in blocked
            and (not for_user or player.player_id not in self.user_exclusions)
            and self._feasible_with_counts(player.position, counts, picks_after)
        ]

    def _draft_value(self, player: DraftPlayer) -> tuple[float, float, float, str]:
        # Projections determine value; ADP only breaks economically equivalent
        # choices. Variance is reserved for the payout simulation.
        vorp = player.season_mean - self._replacement_points[player.position]
        return (vorp, player.season_mean, -player.adp, player.player_id)

    def _choose_user(
        self,
        selected_ids: set[str],
        roster_ids: Sequence[str],
        picks_after: int,
    ) -> DraftPlayer:
        available = self._available(
            selected_ids, roster_ids, picks_after, for_user=True
        )
        if not available:
            raise RuntimeError("No feasible player remains for the user roster")
        return max(available, key=self._draft_value)

    def _choose_opponent(
        self,
        franchise: str,
        selected_ids: set[str],
        roster_ids: Sequence[str],
        picks_after: int,
        scenario: _Scenario,
        *,
        exclusions: frozenset[str] | None = None,
    ) -> DraftPlayer:
        available = self._available(
            selected_ids,
            roster_ids,
            picks_after,
            exclusions=exclusions,
        )
        if not available:
            raise RuntimeError(f"No feasible player remains for {franchise}")
        franchise_index = self._franchise_index[franchise]
        return min(
            available,
            key=lambda player: (
                log(player.adp)
                + scenario.draft_noise[franchise_index, self._player_index[player.player_id]],
                player.adp,
                player.player_id,
            ),
        )

    def _draft_one(
        self, scenario: _Scenario, forced_candidate_id: str | None
    ) -> tuple[dict[str, list[str]], str]:
        rosters = {
            franchise: list(player_ids)
            for franchise, player_ids in self._conditioned_rosters.items()
        }
        selected_ids = set(self._conditioned_ids)

        for slot in self.slots:
            if slot.player_id is not None:
                continue
            roster = rosters[slot.franchise_id]
            picks_after = self._unfilled_after[slot.overall]
            if (
                slot.overall == self.current_slot.overall
                and forced_candidate_id is not None
                and forced_candidate_id not in selected_ids
                and forced_candidate_id not in self.exclusions
                and forced_candidate_id not in self.user_exclusions
            ):
                player = self.player_by_id[forced_candidate_id]
                if not self._feasible(player, roster, picks_after):
                    raise ValueError(f"Forced candidate {forced_candidate_id!r} is not selectable")
            elif slot.franchise_id == self.user_franchise:
                player = self._choose_user(selected_ids, roster, picks_after)
            else:
                player = self._choose_opponent(
                    slot.franchise_id,
                    selected_ids,
                    roster,
                    picks_after,
                    scenario,
                )
            roster.append(player.player_id)
            selected_ids.add(player.player_id)
            if slot.overall == self.current_slot.overall:
                current_selection = player.player_id
        return rosters, current_selection

    def _draft_external(self, scenario: _Scenario) -> dict[str, list[str]]:
        rosters = {
            franchise: list(player_ids)
            for franchise, player_ids in self._external_conditioned_rosters.items()
        }
        selected_ids = set(self._external_conditioned_ids)
        for slot in self.external_slots:
            if slot.player_id is not None:
                continue
            roster = rosters[slot.franchise_id]
            player = self._choose_opponent(
                slot.franchise_id,
                selected_ids,
                roster,
                self._external_unfilled_after[slot.overall],
                scenario,
                exclusions=self.external_exclusions,
            )
            roster.append(player.player_id)
            selected_ids.add(player.player_id)
        return rosters

    @staticmethod
    def _sample_player(player: DraftPlayer, z_score: float) -> float:
        if player.season_sd == 0 or player.season_mean == 0:
            return player.season_mean
        variance_ratio = (player.season_sd / player.season_mean) ** 2
        log_variance = log1p(variance_ratio)
        log_sigma = sqrt(log_variance)
        log_mean = log(player.season_mean) - log_variance / 2
        return float(np.exp(log_mean + log_sigma * z_score))

    def _roster_score(
        self, roster_ids: Sequence[str], outcome_z: np.ndarray
    ) -> float:
        outcomes = [
            (
                self.player_by_id[player_id].position,
                self._sample_player(
                    self.player_by_id[player_id],
                    outcome_z[self._player_index[player_id]],
                ),
            )
            for player_id in roster_ids
        ]
        # This is a managed-lineup proxy: it credits the best legal collection
        # of season outcomes, applied identically to every franchise.
        return optimal_week_score(outcomes)

    def _exact_external_scores(self) -> tuple[dict[str, float], ...]:
        if not self.external_franchises:
            return ()
        if self._external_score_rows is None:
            self._external_score_rows = tuple(
                {
                    franchise: self._roster_score(roster, scenario.outcome_z)
                    for franchise, roster in self._draft_external(scenario).items()
                }
                for scenario in self._scenarios
            )
        return self._external_score_rows

    def _equity(self, forced_candidate_id: str | None) -> _Equity:
        if forced_candidate_id in self._equity_cache:
            return self._equity_cache[forced_candidate_id]
        first_total = second_total = third_total = payout_total = available_total = 0.0
        current_selections: Counter[str] = Counter()
        score_rows: list[dict[str, float]] = []
        for scenario in self._scenarios:
            rosters, current_selection = self._draft_one(scenario, forced_candidate_id)
            current_selections[current_selection] += 1
            if forced_candidate_id is None or current_selection == forced_candidate_id:
                available_total += 1
            score_rows.append(
                {
                    franchise: self._roster_score(roster, scenario.outcome_z)
                    for franchise, roster in rosters.items()
                }
            )

        opponent_franchises = tuple(
            franchise for franchise in self.franchises if franchise != self.user_franchise
        )
        exact_external_scores = self._exact_external_scores()
        payout_samples: list[float] = []
        for index, scenario in enumerate(self._scenarios):
            scores = score_rows[index]
            our_score = scores[self.user_franchise]
            conference_opponents = [scores[franchise] for franchise in opponent_franchises]
            if exact_external_scores:
                external_opponents = [
                    exact_external_scores[index][franchise]
                    for franchise in self.external_franchises
                ]
            else:
                external_opponents = [
                    score_rows[int(source_index)][
                        opponent_franchises[int(opponent_index)]
                    ]
                    for source_index, opponent_index in zip(
                        scenario.external_scenario_indices,
                        scenario.external_opponent_indices,
                        strict=True,
                    )
                ]
            opponents = [*conference_opponents, *external_opponents]
            # Exact ties split each occupied finishing place fairly. This is
            # mostly relevant to deterministic test projections.
            greater = sum(score > our_score for score in opponents)
            tied = 1 + sum(score == our_score for score in opponents)
            occupied = range(greater + 1, greater + tied + 1)
            first_share = (1.0 / tied) if 1 in occupied else 0.0
            second_share = (1.0 / tied) if 2 in occupied else 0.0
            third_share = (1.0 / tied) if 3 in occupied else 0.0
            first_total += first_share
            second_total += second_share
            third_total += third_share
            scenario_payout = (
                first_share * self.config.payout.first
                + second_share * self.config.payout.second
                + third_share * self.config.payout.third
            )
            payout_total += scenario_payout
            payout_samples.append(scenario_payout)
        denominator = self.config.simulations
        result = _Equity(
            payout=payout_total / denominator,
            first=first_total / denominator,
            second=second_total / denominator,
            third=third_total / denominator,
            availability=available_total / denominator,
            most_common_player_id=current_selections.most_common(1)[0][0],
            payout_samples=tuple(payout_samples),
        )
        self._equity_cache[forced_candidate_id] = result
        return result

    def _initial_selectable(self) -> list[DraftPlayer]:
        roster = self._conditioned_rosters[self.user_franchise]
        picks_after = self._unfilled_after[self.current_slot.overall]
        return self._available(
            set(self._conditioned_ids), roster, picks_after, for_user=True
        )

    def reference_player(self) -> DraftPlayer:
        """Player selected now by the projection/VORP completion policy."""
        return self._choose_user(
            set(self._conditioned_ids),
            self._conditioned_rosters[self.user_franchise],
            self._unfilled_after[self.current_slot.overall],
        )

    def evaluate_forced_candidate(self, player_id: str) -> CandidateEV:
        """Evaluate one selectable player against the unforced reference policy."""
        player_id = str(player_id)
        if player_id not in self.player_by_id:
            raise ValueError(f"Candidate {player_id!r} is missing from the player board")
        if player_id not in {player.player_id for player in self._initial_selectable()}:
            raise ValueError(f"Candidate {player_id!r} is not selectable")
        reference = self._equity(None)
        equity = self._equity(player_id)
        player = self.player_by_id[player_id]
        marginal_se = _paired_standard_error(equity, reference)
        return CandidateEV(
            rank=1,
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            adp=player.adp,
            payout_equity=equity.payout,
            marginal_ev=equity.payout - reference.payout,
            first_probability=equity.first,
            second_probability=equity.second,
            third_probability=equity.third,
            availability_probability=equity.availability,
            marginal_ev_standard_error=marginal_se,
            best_ev_gap_standard_error=0.0,
            statistically_tied_for_best=True,
        )

    def rank_candidates(
        self, candidate_ids: Sequence[str] | None = None
    ) -> DraftEVRanking:
        """Rank a bounded candidate set by conditional expected payout share.

        ``marginal_ev`` is measured against the same simulated draft with the
        current pick chosen by the projection/VORP reference policy.
        """
        selectable = self._initial_selectable()
        selectable_ids = {player.player_id for player in selectable}
        if candidate_ids is not None:
            normalized_ids = tuple(dict.fromkeys(str(player_id) for player_id in candidate_ids))
            if len(normalized_ids) > self.config.max_candidates:
                raise ValueError(
                    f"At most {self.config.max_candidates} candidates may be evaluated at once"
                )
            missing = [player_id for player_id in normalized_ids if player_id not in selectable_ids]
            if missing:
                raise ValueError(f"Candidates are not selectable: {', '.join(missing)}")
            candidates = [self.player_by_id[player_id] for player_id in normalized_ids]
        else:
            # Split the bounded default pool between market price and model VORP
            # so a falling projection value is not hidden by ADP alone.
            market = sorted(selectable, key=lambda player: (player.adp, player.player_id))
            model = sorted(selectable, key=self._draft_value, reverse=True)
            candidates = []
            half = max(1, self.config.max_candidates // 2)
            for player in [*market[:half], *model]:
                if player not in candidates:
                    candidates.append(player)
                if len(candidates) == self.config.max_candidates:
                    break

        if not candidates:
            raise ValueError("No candidates were supplied for evaluation")

        reference = self._equity(None)
        raw: list[tuple[DraftPlayer, _Equity]] = [
            (player, self._equity(player.player_id)) for player in candidates
        ]
        raw.sort(key=lambda item: (-item[1].payout, item[0].adp, item[0].player_id))
        best_equity = raw[0][1]
        ranked = tuple(
            CandidateEV(
                rank=rank,
                player_id=player.player_id,
                name=player.name,
                position=player.position,
                adp=player.adp,
                payout_equity=equity.payout,
                marginal_ev=equity.payout - reference.payout,
                first_probability=equity.first,
                second_probability=equity.second,
                third_probability=equity.third,
                availability_probability=equity.availability,
                marginal_ev_standard_error=_paired_standard_error(equity, reference),
                best_ev_gap_standard_error=(
                    best_gap_se := _paired_standard_error(equity, best_equity)
                ),
                statistically_tied_for_best=(
                    best_equity.payout - equity.payout <= 1.96 * best_gap_se
                ),
            )
            for rank, (player, equity) in enumerate(raw, start=1)
        )
        return DraftEVRanking(
            seed=self.config.seed,
            simulations=self.config.simulations,
            current_overall=self.current_slot.overall,
            reference_player_id=reference.most_common_player_id,
            reference_payout_equity=reference.payout,
            payout_field_size=self.config.payout_field_size,
            field_approximation=(
                "exact_two_conference_drafts_shared_player_outcomes"
                if self.external_franchises
                else "stratified_external_teams_from_conference_opponents"
            ),
            candidates=ranked,
        )


def rank_draft_candidates(
    player_board: Any,
    draft_slots: Sequence[DraftSlot | Mapping[str, Any]],
    franchises: Sequence[str],
    user_franchise: str,
    *,
    exclusions: Iterable[str] = (),
    user_exclusions: Iterable[str] = (),
    external_draft_slots: Sequence[DraftSlot | Mapping[str, Any]] = (),
    external_franchises: Sequence[str] = (),
    external_exclusions: Iterable[str] = (),
    candidate_ids: Sequence[str] | None = None,
    config: DraftSimulationConfig = DraftSimulationConfig(),
) -> DraftEVRanking:
    """Functional interface for a draft-day command or application."""
    return DraftSimulator(
        player_board,
        draft_slots,
        franchises,
        user_franchise,
        exclusions=exclusions,
        user_exclusions=user_exclusions,
        external_draft_slots=external_draft_slots,
        external_franchises=external_franchises,
        external_exclusions=external_exclusions,
        config=config,
    ).rank_candidates(candidate_ids)
