import pytest

from fantasy_model.distribution_constants import (
    calibrated_rookie_sd,
    calibrated_veteran_mean_adjustment,
    calibrated_veteran_sd,
)


def test_veteran_season_uncertainty_scales_role_and_injury_error() -> None:
    assert calibrated_veteran_sd("QB", 40.0) == pytest.approx(86.84)
    assert calibrated_veteran_sd("RB", 40.0) == pytest.approx(91.4)
    assert calibrated_veteran_sd("Def", 40.0) == pytest.approx(40.0)


def test_veteran_age_curve_is_position_specific_and_bounded() -> None:
    assert calibrated_veteran_mean_adjustment("RB", 24) == pytest.approx(2.0106)
    assert calibrated_veteran_mean_adjustment("RB", 30) == pytest.approx(-26.8206)
    assert calibrated_veteran_mean_adjustment("WR", 40) == -60.0
    assert calibrated_veteran_mean_adjustment("Def", 30) == 0.0
    assert calibrated_veteran_mean_adjustment("QB", None) == 0.0


def test_rookie_uncertainty_uses_walk_forward_production_bands() -> None:
    assert calibrated_rookie_sd("RB", 30.0) == pytest.approx(23.64)
    assert calibrated_rookie_sd("RB", 40.0) == pytest.approx(36.23)
    assert calibrated_rookie_sd("RB", 200.0) == pytest.approx(65.07)
    assert calibrated_rookie_sd("WR", 100.0) == pytest.approx(49.0)
