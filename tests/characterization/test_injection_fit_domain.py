"""A recovery log slope needs positive measured excess at distinct powers."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pilot_proxy.characterization.injection import (
    signal_dominated_log_slope,
    weighted_linear_fit,
)


def test_signed_recovery_below_fitted_floor_has_no_log_slope():
    x = np.array([0.0, 1.0, 2.0, 4.0])
    y = np.array([0.1, -0.1, 2.1, 4.1])
    sem = np.array([0.01, 1.0, 0.01, 0.01])
    fit = weighted_linear_fit(x, y, sem)
    assert fit["gain_per_lsb2"] == pytest.approx(1.0, rel=1e-4)
    assert x[1] * fit["gain_per_lsb2"] > 3.0 * abs(fit["floor"])

    with np.errstate(invalid="raise", divide="raise"):
        slope = signal_dominated_log_slope(x, y, fit)

    assert slope is None
    assert json.loads(json.dumps({"fit": fit, "slope": slope}, allow_nan=False))["slope"] is None


@pytest.mark.parametrize("above_floor", [0.0, -0.01, np.nan, np.inf, -np.inf])
def test_undefined_selected_response_does_not_drop_a_ladder_point(above_floor):
    x = np.array([0.0, 1.0, 2.0, 4.0])
    fit = {"floor": 0.1, "gain_per_lsb2": 1.0}
    y = fit["floor"] + x
    y[1] = fit["floor"] + above_floor

    with np.errstate(invalid="raise", divide="raise"):
        assert signal_dominated_log_slope(x, y, fit) is None


def test_repeated_selected_power_cannot_define_a_log_slope():
    x = np.array([0.0, 1.0, 1.0])
    fit = {"floor": 0.1, "gain_per_lsb2": 1.0}
    assert signal_dominated_log_slope(x, fit["floor"] + x, fit) is None


def test_valid_log_slope_preserves_selection_and_measures_nonunit_response():
    x = np.array([0.0, 0.01, 1.0, 4.0, 16.0])
    fit = {"floor": 0.2, "gain_per_lsb2": 1.0}
    y = fit["floor"] + 3.0 * x**1.25
    y[:2] = [-0.1, fit["floor"]]
    assert signal_dominated_log_slope(x, y, fit) == pytest.approx(1.25)
