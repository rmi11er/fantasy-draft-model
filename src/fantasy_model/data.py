from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl


PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)
PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/"
    "players.parquet"
)


def ensure_player_stats(cache_dir: Path, season: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"stats_player_week_{season}.parquet"
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(".parquet.part")
    with httpx.stream(
        "GET",
        PLAYER_STATS_URL.format(season=season),
        follow_redirects=True,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    temporary.replace(destination)
    return destination


def load_player_stats(cache_dir: Path, seasons: range) -> pl.DataFrame:
    frames = [
        pl.read_parquet(ensure_player_stats(cache_dir, season)) for season in seasons
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def ensure_players(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "players.parquet"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(".parquet.part")
    with httpx.stream(
        "GET", PLAYERS_URL, follow_redirects=True, timeout=120.0
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    temporary.replace(destination)
    return destination


def load_players(cache_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(ensure_players(cache_dir))
