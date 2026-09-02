from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


def _value(stats: Mapping[str, float | int | None], key: str) -> float:
    value = stats.get(key, 0)
    return 0.0 if value is None else float(value)


@dataclass(frozen=True)
class LeagueScoring:
    passing_td: float = 6.0
    passing_yard: float = 0.04
    interception: float = -2.0
    two_point_conversion: float = 2.0
    rushing_td: float = 6.0
    rushing_yard: float = 0.10
    receiving_td: float = 6.0
    receiving_yard: float = 0.10
    reception: float = 0.5
    fumble_lost: float = -2.0

    def score_offense(self, stats: Mapping[str, float | int | None]) -> float:
        """Score a normalized offensive stat line under the league rules."""
        return round(
            _value(stats, "passing_tds") * self.passing_td
            + _value(stats, "passing_yards") * self.passing_yard
            + _value(stats, "interceptions") * self.interception
            + _value(stats, "passing_2pt_conversions")
            * self.two_point_conversion
            + _value(stats, "rushing_tds") * self.rushing_td
            + _value(stats, "rushing_yards") * self.rushing_yard
            + _value(stats, "rushing_2pt_conversions")
            * self.two_point_conversion
            + _value(stats, "receiving_tds") * self.receiving_td
            + _value(stats, "receiving_yards") * self.receiving_yard
            + _value(stats, "receptions") * self.reception
            + _value(stats, "receiving_2pt_conversions")
            * self.two_point_conversion
            + _value(stats, "fumbles_lost") * self.fumble_lost,
            2,
        )

    @staticmethod
    def made_field_goal_points(distance: int) -> float:
        if distance < 40:
            return 3.0
        if distance < 50:
            return 4.0
        if distance < 60:
            return 5.0
        return 6.0

    @staticmethod
    def missed_field_goal_points(distance: int) -> float:
        if distance < 40:
            return -3.0
        if distance < 50:
            return -2.0
        if distance < 60:
            return -1.0
        return 0.0

    def score_kicker(
        self,
        made_distances: Sequence[int],
        missed_distances: Sequence[int],
        extra_points_made: int = 0,
    ) -> float:
        return round(
            sum(self.made_field_goal_points(distance) for distance in made_distances)
            + sum(
                self.missed_field_goal_points(distance)
                for distance in missed_distances
            )
            + extra_points_made,
            2,
        )

    @staticmethod
    def points_allowed_score(points_allowed: int) -> float:
        if points_allowed == 0:
            return 10.0
        if points_allowed <= 6:
            return 7.0
        if points_allowed <= 14:
            return 3.0
        if points_allowed <= 20:
            return 0.0
        return -3.0

    def score_defense(self, stats: Mapping[str, float | int | None]) -> float:
        return round(
            _value(stats, "fumble_recoveries") * 2
            + _value(stats, "interceptions") * 2
            + _value(stats, "sacks")
            + _value(stats, "safeties") * 2
            + _value(stats, "defensive_special_teams_tds") * 6
            + self.points_allowed_score(int(_value(stats, "points_allowed"))),
            2,
        )


OLD_MISSION_SCORING = LeagueScoring()

