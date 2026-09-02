from fantasy_model.market import mfl_adp_frame, mfl_display_name, mfl_players_frame


def test_mfl_players_and_adp_use_stable_identifiers() -> None:
    players = mfl_players_frame(
        {
            "players": {
                "player": [
                    {
                        "id": "16580",
                        "espn_id": "4431452",
                        "name": "Maye, Drake",
                        "position": "QB",
                        "team": "NEP",
                    }
                ]
            }
        }
    )
    adp = mfl_adp_frame(
        {
            "adp": {
                "player": [
                    {
                        "id": "16580",
                        "rank": "28",
                        "averagePick": "27.98",
                        "minPick": "15",
                        "maxPick": "39",
                        "draftsSelectedIn": "100",
                        "draftSelPct": "80",
                    }
                ]
            }
        }
    )
    assert players["espn_id"].item() == "4431452"
    assert players["mfl_display_name"].item() == "Drake Maye"
    assert adp["average_pick"].item() == 27.98
    assert mfl_display_name("Maye, Drake") == "Drake Maye"
