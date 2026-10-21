from fantasy_model.league import LeagueSnapshot


def _league_payload() -> dict:
    return {
        "league": {
            "id": "50473",
            "name": "Old Mission vol '26",
            "rosterSize": "18",
            "franchises": {
                "franchise": [
                    {"id": "0005", "name": "Franchise 5", "division": "00"},
                    {"id": "0016", "name": "Levered Leopolds", "division": "01"},
                    {"id": "0017", "name": "JH/JS TBD", "division": "02"},
                    {"id": "0013", "name": "Franchise 13", "division": "03"},
                ]
            },
            "divisions": {
                "division": [
                    {"id": "00", "conference": "00", "name": "Friday Night Lights"},
                    {"id": "01", "conference": "00", "name": "The Waterboy"},
                    {"id": "02", "conference": "01", "name": "Varsity Blues"},
                    {"id": "03", "conference": "01", "name": "The Longest Yard"},
                ]
            },
            "conferences": {
                "conference": [
                    {"id": "00", "name": "Old Mission Madness"},
                    {"id": "01", "name": "Old Mission Mayhem"},
                ]
            },
        }
    }


def _draft_payload() -> dict:
    picks = []
    order = ["0005", "0016"]
    for round_number in range(1, 17):
        round_order = order if round_number % 2 else list(reversed(order))
        for slot, franchise in enumerate(round_order, start=1):
            picks.append(
                {
                    "round": f"{round_number:02}",
                    "pick": f"{slot:02}",
                    "franchise": franchise,
                    "player": "16162" if round_number == 1 and slot == 1 else "",
                }
            )
    return {
        "draftResults": {
            "draftUnit": [
                {"unit": "CONFERENCE00", "draftPick": picks},
                {
                    "unit": "CONFERENCE01",
                    "draftPick": [
                        {
                            "round": "01",
                            "pick": "01",
                            "franchise": "0017",
                            "player": "16580",
                        }
                    ],
                },
            ]
        }
    }


def test_snapshot_scopes_drafted_players_to_my_conference() -> None:
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), _draft_payload(), "Levered Leopolds"
    )
    assert snapshot.franchise_id == "0016"
    assert snapshot.conference_id == "00"
    assert snapshot.draft_rounds == 16
    assert "16162" in snapshot.drafted_player_ids
    assert "16580" not in snapshot.drafted_player_ids
    assert snapshot.is_available("16580")
    assert snapshot.external_conference_franchise_ids == ("0017", "0013")
    assert snapshot.external_picks[0].player_id == "16580"


def test_snake_pick_numbers_are_derived_from_draft_unit() -> None:
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), _draft_payload(), "Levered Leopolds"
    )
    assert [pick.overall for pick in snapshot.my_picks[:5]] == [2, 3, 6, 7, 10]


def test_live_turn_state_is_derived_from_first_unfilled_pick() -> None:
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), _draft_payload(), "Levered Leopolds"
    )
    assert snapshot.current_pick.overall == 2
    assert snapshot.next_my_pick.overall == 2
    assert snapshot.on_clock
    assert snapshot.my_drafted_player_ids == ()


def test_rostered_players_in_my_conference_are_unavailable() -> None:
    rosters = {
        "rosters": {
            "franchise": [
                {"id": "0005", "player": {"id": "keeper-a"}},
                {"id": "0016", "player": [{"id": "keeper-b"}]},
                {"id": "0017", "player": {"id": "other-conference"}},
            ]
        }
    }
    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), _draft_payload(), "Levered Leopolds", rosters
    )

    assert not snapshot.is_available("keeper-a")
    assert not snapshot.is_available("keeper-b")
    assert snapshot.is_available("other-conference")
    assert snapshot.my_player_ids == ("keeper-b",)
    assert snapshot.external_rostered_player_ids == frozenset({"other-conference"})


def test_skipped_empty_pick_does_not_reopen_after_later_selection() -> None:
    payload = _draft_payload()
    picks = payload["draftResults"]["draftUnit"][0]["draftPick"]
    picks[0]["player"] = ""
    picks[0]["timestamp"] = "1720000000"
    picks[1]["player"] = "player-at-two"

    snapshot = LeagueSnapshot.from_payloads(
        _league_payload(), payload, "Levered Leopolds"
    )

    assert snapshot.current_pick.overall == 3
    assert snapshot.next_my_pick.overall == 3
