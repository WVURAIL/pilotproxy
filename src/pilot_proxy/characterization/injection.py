"""Recovery statistics for truth-labelled injections: detection probability at
a matched false-alarm rate, and the recovery-linearity fit.

An injection ladder puts a known tone into stored data at several amplitudes,
with one ``a = 0`` control point that is byte-identical to the source. The
statistics here need only per-frame statistics and the ladder's truth labels:

- :func:`matched_thresholds`: for each requested false-alarm rate, the
  empirical ``(1 - P_fa)`` quantile of each statistic on the control, used
  only where the control holds at least
  :data:`MIN_FRAMES_PER_FALSE_ALARM` / ``P_fa`` valid frames (the detector
  register's ``detection.minimum_frames_per_false_alarm``);
- :func:`detection_rates`: the detected fraction of a point at each threshold,
  with Wilson 95% intervals;
- :func:`weighted_linear_fit`: ``rho_hat = floor + gain * a^2`` with
  ``1 / sem^2`` weights;
- :func:`signal_dominated_log_slope`: the slope-one check in log space over the
  points where the injected term dominates the floor.

The orchestration (reading ladder points, writing the report and figures)
stays with the instrument reader (``pilot_proxy.chime.injection_recovery``).
"""
from __future__ import annotations

import math
import operator
from typing import Any, Sequence

import numpy as np

from pilot_proxy.config.project import default_project
from pilot_proxy.testbench.binomial import wilson_interval

# An empirical quantile at P_fa needs enough H0 frames to be meaningful: at
# least this many expected false alarms (detector register entry
# detection.minimum_frames_per_false_alarm).
MIN_FRAMES_PER_FALSE_ALARM = default_project().register.value(
    "detection.minimum_frames_per_false_alarm"
)


def weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, sem: np.ndarray
) -> dict[str, float]:
    """Weighted least squares y = floor + gain * x with 1/sem^2 weights.

    Report formal coefficient errors for the supplied SEMs, without a
    residual-variance rescaling. Undefined or unrepresentable fits are refused.
    """
    if any(np.iscomplexobj(values) for values in (x, y, sem)):
        raise ValueError("fit inputs must be real")
    x, y, sem = (np.asarray(values, dtype=np.longdouble) for values in (x, y, sem))
    if any(values.ndim != 1 for values in (x, y, sem)):
        raise ValueError("fit inputs must be one-dimensional")
    if x.shape != y.shape or x.shape != sem.shape:
        raise ValueError("fit inputs must have the same shape")
    if x.size < 2:
        raise ValueError("a fit needs at least two points")
    if any(not np.isfinite(values).all() for values in (x, y, sem)):
        raise ValueError("fit inputs must be finite")
    if np.any(sem <= 0):
        raise ValueError("fit SEMs must be positive")

    # Center before scaling to preserve small separations at large offsets.
    # Relative weights avoid squaring an unscaled tiny or huge SEM. Extended
    # intermediates retain very unequal weights where the platform supports it.
    anchor = int(np.argmin(sem))
    x0, y0, sem_scale = x[anchor], y[anchor], sem[anchor]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        dx, dy = x - x0, y - y0
        x_scale = np.max(np.abs(dx))
        if x_scale == 0:
            raise ValueError("a fit needs distinct x values")
        y_scale = np.max(np.abs(dy))
        if y_scale == 0:
            y_scale = np.longdouble(1)
        u, v = dx / x_scale, dy / y_scale
        weights = np.square(sem_scale / sem)
        total = weights.sum()
        u_mean = (weights * u).sum() / total
        v_mean = (weights * v).sum() / total
        u_residual = u - u_mean
        xx = (weights * u_residual**2).sum()
        if not np.isfinite(xx) or xx <= 0:
            raise ValueError("weighted x values do not resolve a distinct fit")
        slope = (weights * u_residual * (v - v_mean)).sum() / xx
        gain = slope * y_scale / x_scale
        floor = y0 + (v_mean - slope * u_mean) * y_scale - gain * x0
        gain_err = sem_scale / x_scale / np.sqrt(xx)
        floor_err = sem_scale * np.sqrt(1 / total + (x0 / x_scale + u_mean)**2 / xx)
    result = {"floor": floor, "gain_per_lsb2": gain, "floor_err": floor_err, "gain_err": gain_err}
    converted = {name: float(value) for name, value in result.items()}
    if (not all(math.isfinite(value) for value in converted.values())
            or any(result[name] != 0 and value == 0 for name, value in converted.items())
            or converted["floor_err"] <= 0 or converted["gain_err"] <= 0):
        raise ValueError("fit coefficients or standard errors are not representable")
    return converted


def false_alarm_rate_key(pfa: float) -> str:
    """Keep conventional rate keys compact without merging distinct floats."""
    value = float(pfa)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("false-alarm rates must be finite and strictly between 0 and 1")
    compact = f"{value:g}"
    return compact if float(compact) == value else repr(value)


def _population_statistics(point: dict[str, Any], statistics: dict[str, str]) -> tuple[int, dict[str, np.ndarray]]:
    """Validate the common finite population without silently dropping samples."""
    count = point["n_valid"]
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n_valid must be an exact nonnegative integer")
    try:
        count = operator.index(count)
    except TypeError as exc:
        raise ValueError("n_valid must be an exact nonnegative integer") from exc
    if count < 0:
        raise ValueError("n_valid must be an exact nonnegative integer")
    arrays = {}
    for name, source in statistics.items():
        if np.iscomplexobj(point[source]):
            raise ValueError(f"statistic {source!r} must contain real samples")
        values = np.asarray(point[source], dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"statistic {source!r} must contain n_valid finite samples")
        arrays[name] = values
    return count, arrays


def matched_thresholds(control: dict[str, Any], false_alarm_rates: Sequence[float], *,
                       statistics: dict[str, str]) -> tuple[dict[str, dict[str, float]], list[float]]:
    """Thresholds at matched P_fa from the control's empirical quantiles, and the rates it supports.

    ``statistics`` maps an output name to the control's key for that
    statistic (for example ``{"power_ratio": "power_ratio", "radiometer": "power"}``).
    """
    rates = sorted(set(float(p) for p in false_alarm_rates), reverse=True)
    keys = {pfa: false_alarm_rate_key(pfa) for pfa in rates}
    count, arrays = _population_statistics(control, statistics)
    ordered = {name: np.sort(values) for name, values in arrays.items()}
    thresholds: dict[str, dict[str, float]] = {name: {} for name in statistics}
    usable_pfa: list[float] = []
    for pfa in rates:
        if count < MIN_FRAMES_PER_FALSE_ALARM / pfa:
            continue
        usable_pfa.append(pfa)
        position = (count - 1) * (1.0 - pfa)
        lower, upper = math.floor(position), math.ceil(position)
        fraction = position - lower
        for name, values in ordered.items():
            a, b = float(values[lower]), float(values[upper])
            # This is the same linear empirical quantile as NumPy's default.
            # Opposite extreme endpoints cannot safely form the difference b-a.
            difference = b - a
            if math.isfinite(difference):
                # Match NumPy's endpoint-side arithmetic for ordinary inputs.
                value = (a + fraction * difference if fraction < 0.5
                         else b - (1 - fraction) * difference)
            else:
                value = (1 - fraction) * a + fraction * b
            thresholds[name][keys[pfa]] = value
    return thresholds, usable_pfa


def detection_rates(point: dict[str, Any], thresholds: dict[str, dict[str, float]], usable_pfa: Sequence[float], *,
                    statistics: dict[str, str]) -> dict[str, float]:
    """The detected fraction of one point at each matched threshold, with Wilson 95% intervals."""
    count, arrays = _population_statistics(point, statistics)
    if count == 0:
        raise ValueError("detection rates need a positive n_valid population")
    row: dict[str, float] = {}
    for pfa in usable_pfa:
        key = false_alarm_rate_key(pfa)
        for name, stat in arrays.items():
            threshold = float(thresholds[name][key])
            if not math.isfinite(threshold):
                raise ValueError(f"threshold for {name!r} at false-alarm rate {key} must be finite")
            detected = int(np.count_nonzero(stat > threshold))
            lo, hi = wilson_interval(detected, count)
            row[f"pd_{name}_pfa{key}"] = detected / count
            row[f"pd_{name}_pfa{key}_wilson95_lo"] = lo
            row[f"pd_{name}_pfa{key}_wilson95_hi"] = hi
    return row


def signal_dominated_log_slope(x: np.ndarray, y: np.ndarray, fit: dict[str, float]) -> float | None:
    """Log slope over the signal-dominated points, or None if undefined.

    The selected powers must contain two distinct positive values, and all
    selected measured responses must be finite and above the fitted floor.
    """
    dominated = x * fit["gain_per_lsb2"] > 3.0 * abs(fit["floor"])
    power = x[dominated]
    response = y[dominated] - fit["floor"]
    if (
        not np.isfinite(power).all()
        or not np.isfinite(response).all()
        or np.any(power <= 0)
        or np.any(response <= 0)
        or np.unique(power).size < 2
    ):
        # Dropping nonpositive responses would change the selected population.
        return None
    return float(np.polyfit(np.log10(power), np.log10(response), 1)[0])


__all__ = ["MIN_FRAMES_PER_FALSE_ALARM", "detection_rates", "false_alarm_rate_key", "matched_thresholds", "signal_dominated_log_slope",
           "weighted_linear_fit"]
