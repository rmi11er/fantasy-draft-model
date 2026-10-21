from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import httpx


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@dataclass(frozen=True)
class DraftPick:
    overall: int
    round: int
    slot: int
    franchise_id: str
    player_id: str | None
    timestamp: str | None = None

    @property
    def processed(self) -> bool:
        """MFL records a timeout/skip with a timestamp but no player."""
        return bool(self.player_id or self.timestamp)


@dataclass(frozen=True)
class LeagueSnapshot:
    league_id: str
    league_name: str
    franchise_id: str
    franchise_name: str
    conference_id: str
    conference_name: str
    roster_size: int
    draft_rounds: int
    conference_franchise_ids: tuple[str, ...]
    picks: tuple[DraftPick, ...]
    rostered_player_ids: frozenset[str] = frozenset()
    my_rostered_player_ids: tuple[str, ...] = ()
    external_conference_franchise_ids: tuple[str, ...] = ()
    external_picks: tuple[DraftPick, ...] = ()
    external_rostered_player_ids: frozenset[str] = frozenset()

    @property
    def my_picks(self) -> tuple[DraftPick, ...]:
        return tuple(pick for pick in self.picks if pick.franchise_id == self.franchise_id)

    @property
    def completed_picks(self) -> tuple[DraftPick, ...]:
        return tuple(pick for pick in self.picks if pick.player_id)

    @property
    def current_pick(self) -> DraftPick | None:
        processed_indexes = [
            index for index, pick in enumerate(self.picks) if pick.processed
        ]
        next_index = max(processed_indexes, default=-1) + 1
        return self.picks[next_index] if next_index < len(self.picks) else None

    @property
    def next_my_pick(self) -> DraftPick | None:
        current_overall = self.current_pick.overall if self.current_pick else None
        return next(
            (
                pick
                for pick in self.my_picks
                if not pick.processed
                and current_overall is not None
                and pick.overall >= current_overall
            ),
            None,
        )

    @property
    def my_drafted_player_ids(self) -> tuple[str, ...]:
        return tuple(
            pick.player_id
            for pick in self.my_picks
            if pick.player_id is not None
        )

    @property
    def on_clock(self) -> bool:
        return bool(
            self.current_pick
            and self.current_pick.franchise_id == self.franchise_id
        )

    @property
    def drafted_player_ids(self) -> frozenset[str]:
        return frozenset(pick.player_id for pick in self.picks if pick.player_id)

    @property
    def unavailable_player_ids(self) -> frozenset[str]:
        return self.drafted_player_ids | self.rostered_player_ids

    @property
    def my_player_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.my_rostered_player_ids, *self.my_drafted_player_ids)))

    def is_available(self, player_id: str) -> bool:
        """Availability is scoped to the user's conference draft unit."""
        return player_id not in self.unavailable_player_ids

    @classmethod
    def from_payloads(
        cls,
        league_payload: dict[str, Any],
        draft_payload: dict[str, Any],
        franchise_name: str,
        roster_payload: dict[str, Any] | None = None,
    ) -> "LeagueSnapshot":
        league = league_payload["league"]
        franchises = _as_list(league["franchises"]["franchise"])
        franchise = next(
            item
            for item in franchises
            if item["name"].casefold() == franchise_name.casefold()
        )

        divisions = {
            item["id"]: item for item in _as_list(league["divisions"]["division"])
        }
        division = divisions[franchise["division"]]
        conference_id = division["conference"]
        conferences = {
            item["id"]: item
            for item in _as_list(league["conferences"]["conference"])
        }

        units = _as_list(draft_payload["draftResults"]["draftUnit"])

        def conference_franchises(selected_conference_id: str) -> tuple[str, ...]:
            return tuple(
                item["id"]
                for item in franchises
                if divisions[item["division"]]["conference"]
                == selected_conference_id
            )

        def conference_picks(selected_conference_id: str) -> tuple[DraftPick, ...]:
            unit = next(
                item
                for item in units
                if item["unit"] == f"CONFERENCE{selected_conference_id}"
            )
            team_count = len(conference_franchises(selected_conference_id))
            return tuple(
                sorted(
                    (
                        DraftPick(
                            overall=(int(item["round"]) - 1) * team_count
                            + int(item["pick"]),
                            round=int(item["round"]),
                            slot=int(item["pick"]),
                            franchise_id=item["franchise"],
                            player_id=item.get("player") or None,
                            timestamp=item.get("timestamp") or None,
                        )
                        for item in _as_list(unit.get("draftPick"))
                    ),
                    key=lambda pick: pick.overall,
                )
            )

        conference_franchise_ids = conference_franchises(conference_id)
        picks = conference_picks(conference_id)
        external_conference_ids = tuple(
            selected_id for selected_id in conferences if selected_id != conference_id
        )
        external_conference_id = (
            external_conference_ids[0] if len(external_conference_ids) == 1 else None
        )
        external_conference_franchise_ids = (
            conference_franchises(external_conference_id)
            if external_conference_id is not None
            else ()
        )
        external_picks = (
            conference_picks(external_conference_id)
            if external_conference_id is not None
            else ()
        )

        rostered_by_franchise: dict[str, tuple[str, ...]] = {}
        if roster_payload:
            for raw_roster in _as_list(
                roster_payload.get("rosters", {}).get("franchise")
            ):
                rostered_by_franchise[str(raw_roster["id"])] = tuple(
                    str(player["id"])
                    for player in _as_list(raw_roster.get("player"))
                    if player.get("id")
                )
        conference_rostered = frozenset(
            player_id
            for roster_franchise, player_ids in rostered_by_franchise.items()
            if roster_franchise in conference_franchise_ids
            for player_id in player_ids
        )
        external_rostered = frozenset(
            player_id
            for roster_franchise, player_ids in rostered_by_franchise.items()
            if roster_franchise in external_conference_franchise_ids
            for player_id in player_ids
        )

        return cls(
            league_id=league["id"],
            league_name=league["name"],
            franchise_id=franchise["id"],
            franchise_name=franchise["name"],
            conference_id=conference_id,
            conference_name=conferences[conference_id]["name"],
            roster_size=int(league["rosterSize"]),
            draft_rounds=max((pick.round for pick in picks), default=0),
            conference_franchise_ids=conference_franchise_ids,
            picks=picks,
            rostered_player_ids=conference_rostered,
            my_rostered_player_ids=rostered_by_franchise.get(franchise["id"], ()),
            external_conference_franchise_ids=external_conference_franchise_ids,
            external_picks=external_picks,
            external_rostered_player_ids=external_rostered,
        )


class MFLClient:
    def __init__(
        self,
        year: int,
        league_id: str,
        *,
        api_key: str | None = None,
        user_cookie: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.year = year
        self.league_id = league_id
        self.api_key = api_key
        self.user_cookie = user_cookie
        self.timeout_seconds = timeout_seconds
        self.base_url = f"https://api.myfantasyleague.com/{year}/export"

    def export(self, endpoint: str, **params: str | int) -> dict[str, Any]:
        query: dict[str, str | int] = {
            "TYPE": endpoint,
            "L": self.league_id,
            "JSON": 1,
            **params,
        }
        return self._request(endpoint, query)

    def _request(
        self, endpoint: str, query: dict[str, str | int]
    ) -> dict[str, Any]:
        if self.api_key:
            query["APIKEY"] = self.api_key
        cookies = {"MFL_USER_ID": self.user_cookie} if self.user_cookie else None
        response = httpx.get(
            self.base_url,
            params=query,
            cookies=cookies,
            follow_redirects=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            message = payload["error"].get("$t", str(payload["error"]))
            raise RuntimeError(f"MFL {endpoint} export failed: {message}")
        return payload

    def snapshot(self, franchise_name: str) -> LeagueSnapshot:
        league = self.export("league")
        draft = self.export("draftResults")
        rosters = self.export("rosters")
        return LeagueSnapshot.from_payloads(
            league, draft, franchise_name, roster_payload=rosters
        )

    def players(self, player_ids: Iterable[str] | None = None) -> dict[str, Any]:
        params: dict[str, str | int] = {"DETAILS": 1}
        if player_ids:
            params["PLAYERS"] = ",".join(player_ids)
        return self.export("players", **params)

    def adp(self) -> dict[str, Any]:
        """Comparable 10-team reception-scoring redraft market prices."""
        # ADP is a global export. MFL rejects it when a league ID is supplied.
        return self._request(
            "adp",
            {
                "TYPE": "adp",
                "JSON": 1,
                "PERIOD": "RECENT",
                "FCOUNT": 10,
                "IS_PPR": 1,
                "IS_KEEPER": "N",
                "IS_MOCK": -1,
                "CUTOFF": 5,
                "DETAILS": 1,
            },
        )
