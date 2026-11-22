# coding=utf-8
"""Power-limit word of a channel from the fine powers of a run and of its control.

The acceptance test keeps the most frames whose upper residual power meets the checked limit,

    f * power_scale * (x + 2 s)^2 <= limit,    x = max(C_s e(u) - o - delta(u), 0).

The kept mean excess e(u) and its clustered error come from the channel. The null o and the
selection shift delta(u) come from the control, read under the words of the channel. Under v3 the
excess divides by T_rho over the even bins clear of each channel's own pilot (`shelf_reference`),
so the control is read in its own window.
The response table, the checked limits and the coupling fraction f are inputs. The test is in-sample,
so its word is a candidate until it is tested on independent data.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# frames reduced together, about 0.15 GB of float temporaries
EXCESS_CHUNK = 65536
# fine bins (381 Hz) about the nominal pilot that the v3 shelf reference leaves out
SHELF_HALF_WIDTH = 32
MARGIN_STANDARD_ERRORS = 2.0
SELECTION = "maximum kept fraction subject to upper power <= checked limit"
EVALUATION = "in-sample; independent evaluation required"


def shelf_reference(nominal_bin: int, half_width: int = SHELF_HALF_WIDTH) -> tuple[np.ndarray, int]:
    """Return the bins and median rank of T_rho for the v3 shelf estimate.

    The v3 decision divides by the median reference of all 128 even bins. The shelf estimate divides the target
    by the reference bin by bin, and the target holds the pilot and its nearby lines, so T_rho takes the even bins
    farther than `half_width` fine bins, circularly, from the nominal pilot bin: 95 bins at an even nominal bin, 96
    at an odd one, rank 47. With all 128, the pilot raises T_rho and biases the kept estimate low (the
    dissertation's scripts/shelf_reference.py).
    """
    b = np.arange(256)
    distance = np.minimum((b - nominal_bin) % 256, (nominal_bin - b) % 256)
    bins = (b % 2 == 0) & (distance > half_width)
    return bins, (int(bins.sum()) - 1) // 2


def median_excess(fines: np.ndarray, bulk: np.ndarray, rank: int) -> np.ndarray:
    """Return e' = R_0 / T_rank - 1 of each [3, 256] frame, NaN where the reference has no support.

    R_0 is the ratio of the frame powers and T_rank the fine ratio of the given rank over the
    reference bins, the same background the decision divides by.
    """
    bins = np.flatnonzero(bulk)
    out = np.empty(len(fines))
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo in range(0, len(fines), EXCESS_CHUNK):
            f = np.asarray(fines[lo:lo + EXCESS_CHUNK], dtype=np.float64)
            power = f.sum(axis=2)
            background = np.sort(f[:, 0, bins] / (f[:, 1, bins] + f[:, 2, bins]), axis=1)[:, rank]
            out[lo:lo + len(f)] = power[:, 0] / (power[:, 1] + power[:, 2]) / background - 1.0
    return np.where(np.isfinite(out), out, np.nan)


def control_corrections(words: np.ndarray, excess: np.ndarray, capture: np.ndarray, at: np.ndarray) -> dict[str, Any]:
    """Return the null of the control and its selection shift at each word of `at`, with clustered errors.

    The null is the mean excess of the control. The shift at a word is the mean over the control
    frames that the word keeps, less the mean over all of them. Values are in units of the excess.
    """
    n = excess.size
    null = float(excess.mean())
    e = excess - null
    _, unit = np.unique(capture, return_inverse=True)
    total = np.bincount(unit, e)
    tt = float(total @ total)
    order = np.argsort(words, kind="stable")
    ws, es, us = words[order], e[order], unit[order]
    # the sum and count of each capture before each frame, in word order
    grouped = np.argsort(us, kind="stable")
    g = us[grouped]
    first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    start = np.repeat(first, np.diff(np.r_[first, n]))
    running = np.cumsum(es[grouped])
    before, count = np.empty(n), np.empty(n)
    before[grouped] = running - es[grouped] - np.r_[0.0, running][start]
    count[grouped] = np.arange(n) - start

    k = np.searchsorted(ws, at, side="right")

    def prefix(step: np.ndarray) -> np.ndarray:
        return np.r_[0.0, np.cumsum(step)][k]

    inside = (k > 0) & (k < n)
    kk = np.where(inside, k, 1)
    mean = prefix(es) / kk
    # sum over captures of (E_u - mean n_u)^2 and of (E_u - mean n_u) D_u, for the kept sums E_u, n_u
    square = prefix(2.0 * before * es + es * es) - 2.0 * mean * prefix(before + es * count + es) \
        + mean * mean * prefix(2.0 * count + 1.0)
    cross = prefix(es * total[us]) - mean * prefix(total[us])
    variance = square / kk**2 - 2.0 * cross / (kk * n) + tt / n**2
    return dict(null=null, null_se=math.sqrt(tt) / n, frames=int(n), captures=int(total.size),
                shift=np.where(inside, mean, 0.0), shift_se=np.where(inside, np.sqrt(np.maximum(variance, 0.0)), 0.0))


def limit_test(sweep: dict[str, np.ndarray], *, shelf: float, power_scale: float, power_limit: float,
             coupling: float = 1.0, offset: float = 0.0, shift: np.ndarray | float = 0.0, margin: np.ndarray | float = 0.0,
             min_captures: int = 30) -> dict[str, Any]:
    """Return the outcome of the acceptance test over the words of a sweep.

    `sweep` is threshold_rule.sweep of the channel. `offset`, `shift` and `margin` are in shelf units,
    and `margin` is added to two standard errors of the kept estimate.
    """
    for name, value in (("power_scale", power_scale), ("power_limit", power_limit), ("coupling", coupling)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if coupling > 1:
        raise ValueError("the coupling fraction is at most 1, its worst case")
    x = np.maximum(shelf * sweep["mean_excess"] - shift - offset, 0.0)
    upper = x + MARGIN_STANDARD_ERRORS * sweep["se"] + margin
    ratio = coupling * power_scale * upper**2 / power_limit
    eligible = (sweep["kept_events"] >= min_captures) & (sweep["kappa"] > 0)
    passing = eligible & (ratio <= 1.0)
    result: dict[str, Any] = dict(outcome="useful" if passing.any() else "not useful", power_limit=float(power_limit),
                                  power_scale=float(power_scale), coupling=float(coupling), offset=float(offset),
                                  eligible_words=int(eligible.sum()), passing_words=int(passing.sum()))
    if not eligible.any():
        return dict(result, reason=f"no word keeps {min_captures} captures")
    best = np.flatnonzero(eligible)[np.argmin(ratio[eligible])]
    result.update(least_ratio=float(ratio[best]), least_ratio_word=int(sweep["word"][best]),
                  least_ratio_kappa=float(sweep["kappa"][best]))
    if not passing.any():
        return dict(result, reason="no supported threshold meets tolerance")
    # the kept fraction never falls with the word, so the largest passing word keeps the most
    i = int(np.flatnonzero(passing)[-1])
    return dict(result, reason=None, word=int(sweep["word"][i]), kappa=float(sweep["kappa"][i]), ratio=float(ratio[i]),
                x=float(x[i]), upper_shelf=float(upper[i]), kept_frames=int(sweep["kept_frames"][i]),
                kept_captures=int(sweep["kept_events"][i]))
