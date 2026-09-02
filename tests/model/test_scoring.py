from fantasy_model.scoring import OLD_MISSION_SCORING


def test_offensive_scoring_matches_league_rules() -> None:
    assert OLD_MISSION_SCORING.score_offense(
        {
            "passing_yards": 300,
            "passing_tds": 2,
            "interceptions": 1,
            "rushing_yards": 20,
            "fumbles_lost": 1,
        }
    ) == 22.0
    assert OLD_MISSION_SCORING.score_offense(
        {
            "rushing_yards": 100,
            "rushing_tds": 1,
            "receptions": 4,
            "receiving_yards": 40,
        }
    ) == 22.0


def test_kicker_distance_bonuses_and_miss_penalties() -> None:
    assert OLD_MISSION_SCORING.score_kicker(
        made_distances=[35, 45, 55, 65],
        missed_distances=[35, 45, 55],
        extra_points_made=2,
    ) == 14.0


def test_defense_points_allowed_buckets() -> None:
    expected = {0: 10, 1: 7, 6: 7, 7: 3, 14: 3, 15: 0, 20: 0, 21: -3}
    for points_allowed, score in expected.items():
        assert OLD_MISSION_SCORING.points_allowed_score(points_allowed) == score

