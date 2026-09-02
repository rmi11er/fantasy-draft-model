import polars as pl

from fantasy_model.rookies import _rookie_bio


def test_rookie_features_use_only_preseason_bio_fields() -> None:
    players = pl.DataFrame(
        {
            "gsis_id": ["00-1"],
            "espn_id": ["123"],
            "display_name": ["Prospect"],
            "position": ["WR"],
            "draft_year": [2026],
            "draft_pick": [32],
            "draft_team": ["NE"],
            "birth_date": ["2003-04-01"],
            "height": [73],
            "weight": [205],
        }
    )
    feature = _rookie_bio(players).row(0, named=True)
    assert feature["player_id"] == "00-1"
    assert feature["log_draft_pick"] > 3
    assert 23 < feature["age_at_draft"] < 24
