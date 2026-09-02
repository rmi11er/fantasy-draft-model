import polars as pl

from fantasy_model.backtest import add_league_points


def test_nflverse_column_aliases_apply_turnover_penalties() -> None:
    frame = pl.DataFrame(
        {
            "passing_yards": [300],
            "passing_tds": [2],
            "passing_interceptions": [1],
            "rushing_yards": [20],
            "fumbles_lost_total": [1],
        }
    )
    assert add_league_points(frame)["league_points"].item() == 22.0
