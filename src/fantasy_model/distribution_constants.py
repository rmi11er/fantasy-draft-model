"""Frozen season-uncertainty constants from the local calibration audit."""

from __future__ import annotations


CALIBRATION_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)

VETERAN_SEASON_SD_MULTIPLIER = {
    "QB": 2.171,
    "RB": 2.285,
    "WR": 2.029,
    "TE": 1.844,
    "PK": 2.464,
}

# Ridge-regularized residual-on-age fits over point-in-time 2020-2025 veteran
# projections.  The intercept is the correction at center_age and the slope is
# the change for each year of age.  Expanding-year validation is produced by
# ``fantasy-calibrate``; corrections are bounded to avoid extrapolating a thin
# tail of very young or old players.
VETERAN_MEAN_AGE_MODEL = {
    "QB": {"center_age": 28.0, "intercept": -25.6095, "slope": -1.4026},
    "RB": {"center_age": 26.0, "intercept": -7.5998, "slope": -4.8052},
    "WR": {"center_age": 27.0, "intercept": -9.6330, "slope": -4.6760},
    "TE": {"center_age": 27.0, "intercept": -2.2073, "slope": -2.0118},
    "PK": {"center_age": 30.0, "intercept": -5.9773, "slope": -0.2465},
}

# Season standard deviations by predicted-production band.  Each value is the
# 80th-percentile absolute walk-forward residual divided by the standard-normal
# 80% interval (1.28155).  Bands preserve the observed fact that an early-role
# rookie QB has far more unresolved season value than a late-round reserve,
# without applying an arbitrary "rookie upside" bonus to the mean.
ROOKIE_SEASON_SD_BANDS = {
    "QB": ((19.80, 10.93), (72.00, 44.03), (float("inf"), 110.38)),
    "RB": ((34.23, 23.64), (53.23, 36.23), (float("inf"), 65.07)),
    "WR": ((15.47, 11.99), (53.60, 30.45), (float("inf"), 49.00)),
    "TE": ((8.64, 9.24), (28.41, 27.78), (float("inf"), 30.47)),
}


def calibrated_veteran_sd(position: str, raw_season_sd: float) -> float:
    return max(0.0, float(raw_season_sd)) * VETERAN_SEASON_SD_MULTIPLIER.get(
        position, 1.0
    )


def calibrated_veteran_mean_adjustment(
    position: str | None, age: float | None
) -> float:
    if position is None or age is None:
        return 0.0
    model = VETERAN_MEAN_AGE_MODEL.get(position)
    if model is None:
        return 0.0
    adjustment = model["intercept"] + model["slope"] * (
        float(age) - model["center_age"]
    )
    return min(30.0, max(-60.0, adjustment))


def calibrated_rookie_sd(
    position: str | None, mean_season_points: float | None
) -> float:
    if position is None or mean_season_points is None:
        return 0.0
    mean = max(0.0, float(mean_season_points))
    for upper_bound, season_sd in ROOKIE_SEASON_SD_BANDS.get(position, ()):
        if mean <= upper_bound:
            return season_sd
    return 0.0
