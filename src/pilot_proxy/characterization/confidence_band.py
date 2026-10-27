"""Joint event-pairs bootstrap upper bands for a retained, signed proxy mean.

The band describes the sampling error for the supplied words and windows and
does not bound the physical contamination. The estimand, selection
assumptions, support rules, and treatment of failed replicates are described
in docs/RETAINED_PROXY_CONFIDENCE_BAND.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .threshold_rule import SHELF, ChannelFrames


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, *, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if value <= lower or (upper is not None and value >= upper):
        raise ValueError(f"{name} must be > {lower}" + (f" and < {upper}" if upper is not None else ""))
    return float(value)


def _vector(value, name, length, kind=None):
    value = np.asarray(value)
    if value.ndim != 1 or (length is not None and len(value) != length):
        raise ValueError(f"{name} must be a one-dimensional vector of length {length}")
    if kind is not None and value.dtype.kind not in kind:
        raise TypeError(f"{name} has an invalid dtype")
    return value


def _prefix_by_event(values, event):
    """Return cumulative sums restarted at each contiguous event group, so one group cannot cancel another."""
    result = np.empty_like(values)
    starts = np.r_[0, np.flatnonzero(event[1:] != event[:-1]) + 1, len(event)]
    for left, right in zip(starts[:-1], starts[1:]):
        np.cumsum(values[left:right], out=result[left:right])
    return result


@dataclass
class _Window:
    event: np.ndarray
    bucket: np.ndarray
    increments: tuple[np.ndarray, ...]
    center: float
    points: int
    kept_frames: np.ndarray
    kept_events: np.ndarray

    def sums(self, multiplicity):
        """Return the cumulative window sums with each event weighted by its multiplicity m.

        Each drawn copy of an event is its own cluster, so the event's squared
        prefix enters the sums m times.
        """
        m = multiplicity[self.event]
        return tuple(np.cumsum(np.bincount(self.bucket, weights=m * delta, minlength=self.points))
                     for delta in self.increments)


def _prepare_window(required, weight, excess, event, selected, words):
    """Return the sparse event and word increments of one window.

    The memory grows linearly with the number of selected frames and words.
    """
    rows = np.flatnonzero(selected & (required <= words[-1]))
    points = len(words)
    if not len(rows):
        empty = np.empty(0, dtype=np.float64)
        return _Window(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                       (empty,) * 5, 0., points, np.zeros(points, dtype=np.int64),
                       np.zeros(points, dtype=np.int64))
    bucket = np.searchsorted(words, required[rows], side="left")
    order = np.lexsort((bucket, event[rows]))
    rows, bucket = rows[order], bucket[order]
    ev = event[rows]
    center = float(np.median(excess[rows]))
    # The frames of an event that enter at one grid point are summed before the
    # squared-prefix increments are formed. No event-by-threshold matrix is built.
    starts = np.r_[0, np.flatnonzero((ev[1:] != ev[:-1]) | (bucket[1:] != bucket[:-1])) + 1]
    a = np.add.reduceat(weight[rows] * (excess[rows] - center), starts)
    b = np.add.reduceat(weight[rows], starts)
    counts = np.diff(np.r_[starts, len(rows)])
    ev, bucket = ev[starts], bucket[starts]
    a_old, b_old = _prefix_by_event(a, ev) - a, _prefix_by_event(b, ev) - b
    increments = (a, b, 2 * a_old * a + a * a,
                  a_old * b + b_old * a + a * b, 2 * b_old * b + b * b)
    if not all(np.isfinite(x).all() for x in increments):
        raise ValueError("input values exceed the numerical range of event-prefix statistics")
    first = np.r_[True, ev[1:] != ev[:-1]]
    kept_events = np.cumsum(np.bincount(bucket[first], minlength=points))
    kept_frames = np.cumsum(np.bincount(bucket, weights=counts, minlength=points)).astype(np.int64)
    return _Window(ev, bucket, increments, center, points, kept_frames, kept_events)


def _statistics(sums, center, shelf):
    a, b, aa, ab, bb = sums
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        centered_mean = a / b
        variance = aa - 2 * centered_mean * ab + centered_mean * centered_mean * bb
        magnitude = np.abs(aa) + 2 * np.abs(centered_mean * ab) + np.abs(centered_mean * centered_mean * bb)
        # A variance dominated by cancellation cannot support a narrow band, so both
        # negative and spuriously positive residues at the rounding scale become NaN.
        rounding = 64 * np.finfo(float).eps * magnitude
        bad = (variance < -rounding) | ((magnitude > 0) & (variance <= rounding))
        variance = np.where(bad, np.nan, np.maximum(variance, 0.))
        mean = shelf * (centered_mean + center)
        se = shelf * np.sqrt(variance) / b
        effective = b * b / bb
    mean = np.where(b > 0, mean, np.nan)
    se = np.where(b > 0, se, np.nan)
    return mean, se, effective


def simultaneous_upper_band(
    frames: ChannelFrames,
    words: np.ndarray,
    *,
    windows: Mapping[str, np.ndarray] | None = None,
    cluster_labels: np.ndarray | None = None,
    confidence: float = .95,
    bootstrap_replicates: int = 999,
    seed: int = 20261016,
    min_kept_events: int = 30,
    min_effective_events: float = 20.,
    batch_size: int = 16,
    shelf: float = SHELF,
) -> dict:
    """Return a one-sided bootstrap-t upper band taken jointly over the supplied Q16 words and windows.

    The original frame weights are preserved. Each draw resamples every
    original event (or supplied cluster), even an event with no retained
    frames, and the same multiplicities are used in every window. A cluster
    cannot split an original event. The family of eligible points is set once
    by the support thresholds on the original data. An ineligible point has an
    infinite upper limit, so it never passes.

    The upper pivot is (original_mean - bootstrap_mean) / bootstrap_SE. An
    eligible point with an undefined or zero SE sets that replicate's maximum
    to +inf, and the replicate is still counted in the critical-value
    quantile. Nominal coverage needs the sampling assumptions and a validation
    of the supplied grid and selection procedure.
    """
    confidence = _real(confidence, "confidence", lower=0, upper=1)
    shelf = _real(shelf, "shelf", lower=0)
    bootstrap_replicates = _integer(bootstrap_replicates, "bootstrap_replicates", 2)
    seed = _integer(seed, "seed", 0)
    min_kept_events = _integer(min_kept_events, "min_kept_events", 2)
    min_effective_events = _real(min_effective_events, "min_effective_events", lower=1)
    batch_size = _integer(batch_size, "batch_size", 1)
    words = _vector(words, "words", None)
    if words.dtype != np.dtype(np.uint64):
        raise TypeError("words must have dtype uint64")
    if not len(words) or words[0] == 0 or np.any(words[1:] <= words[:-1]):
        raise ValueError("words must be nonempty, positive, unique, and increasing")
    required = _vector(frames.required, "required", None)
    if required.dtype != np.dtype(np.uint64):
        raise TypeError("required must have dtype uint64")
    n = len(required)
    if n == 0:
        raise ValueError("frames must be nonempty")
    supported = _vector(frames.supported, "supported", n, "b")
    weight = _vector(frames.weight, "weight", n, "fiu").astype(float)
    excess = _vector(frames.excess, "excess", n, "fiu").astype(float)
    original_event = _vector(frames.event, "event", n, "iu")
    if not np.isfinite(weight).all() or np.any(weight <= 0) or not np.isfinite(weight.sum()):
        raise ValueError("all original frame weights must be finite and strictly positive")
    if not np.isfinite(excess[supported]).all():
        raise ValueError("supported-frame excesses must be finite; no observations are silently removed")
    labels = original_event if cluster_labels is None else _vector(cluster_labels, "cluster_labels", n, "iu")
    cluster_ids, event = np.unique(labels, return_inverse=True)
    original_ids, original_index = np.unique(original_event, return_inverse=True)
    order = np.argsort(original_index, kind="stable")
    if np.any((original_index[order][1:] == original_index[order][:-1]) &
              (event[order][1:] != event[order][:-1])):
        raise ValueError("cluster_labels must not split an original event")
    if windows is None:
        windows = {"full": np.ones(n, dtype=bool)}
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("windows must be a nonempty mapping")
    masks = {}
    for name, mask in windows.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("window names must be nonempty strings")
        masks[name] = _vector(mask, f"windows[{name!r}]", n, "b")
    weight_sum = float(weight.sum())
    # One common normalization leaves the ratios and errors unchanged. No window
    # or resampling cluster is reweighted to equal mass.
    weight = weight / weight_sum
    prepared = {name: _prepare_window(required, weight, excess, event, supported & mask, words)
                for name, mask in masks.items()}
    output = {}
    ones = np.ones(len(cluster_ids), dtype=np.int64)
    for name, window in prepared.items():
        sums = window.sums(ones)
        mean, se, effective = _statistics(sums, window.center, shelf)
        status = np.full(len(words), "eligible", dtype="U48")
        status[~np.isfinite(mean) | ~np.isfinite(se)] = "nonfinite_statistics"
        status[se == 0] = "zero_standard_error"
        status[effective < min_effective_events] = "insufficient_effective_events"
        status[window.kept_events < min_kept_events] = "insufficient_kept_events"
        status[sums[1] <= 0] = "no_kept_weight"
        eligible = status == "eligible"
        output[name] = {
            "signed_mean": mean, "se": se, "eligible": eligible, "status": status,
            "signed_upper": np.full(len(words), np.inf), "upper": np.full(len(words), np.inf),
            "kept_frames": window.kept_frames, "kept_events": window.kept_events,
            "kept_clusters": window.kept_events.copy(), "effective_events": effective,
            "kept_weight": sums[1] * weight_sum,
        }
    family_size = sum(int(v["eligible"].sum()) for v in output.values())
    maxima = np.full(bootstrap_replicates, np.inf)
    failed_by_window = {name: 0 for name in windows}
    failed_points_by_window = {name: 0 for name in windows}
    completed = 0
    if family_size:
        rng = np.random.default_rng(seed)
        probabilities = np.full(len(cluster_ids), 1 / len(cluster_ids))
        for start in range(0, bootstrap_replicates, batch_size):
            count = min(batch_size, bootstrap_replicates - start)
            draws = rng.multinomial(len(cluster_ids), probabilities, size=count)
            for offset, multiplicity in enumerate(draws):
                maximum = -np.inf
                for name, window in prepared.items():
                    eligible = output[name]["eligible"]
                    if not eligible.any():
                        continue
                    boot_mean, boot_se, _ = _statistics(window.sums(multiplicity), window.center, shelf)
                    bad = eligible & (~np.isfinite(boot_mean) | ~np.isfinite(boot_se) | (boot_se <= 0))
                    if bad.any():
                        failed_by_window[name] += 1
                        failed_points_by_window[name] += int(bad.sum())
                        maximum = np.inf
                    else:
                        pivot = (output[name]["signed_mean"][eligible] - boot_mean[eligible]) / boot_se[eligible]
                        if not np.isfinite(pivot).all():
                            failed_by_window[name] += 1
                            failed_points_by_window[name] += int((~np.isfinite(pivot)).sum())
                            maximum = np.inf
                        else:
                            maximum = max(maximum, float(pivot.max()))
                maxima[start + offset] = maximum
                completed += 1
    quantile_index = int(np.ceil(confidence * (bootstrap_replicates - 1)))
    critical = float(np.partition(maxima, quantile_index)[quantile_index])
    available = bool(family_size and np.isfinite(critical))
    for result in output.values():
        eligible = result["eligible"]
        if available:
            result["signed_upper"][eligible] = result["signed_mean"][eligible] + critical * result["se"][eligible]
            result["upper"][eligible] = np.maximum(result["signed_upper"][eligible], 0.)
        elif family_size:
            result["status"][eligible] = "bootstrap_unavailable"
    return {
        "schema": "pilotproxy_retained_proxy_confidence_band_v1",
        "status": "available" if available else ("unavailable: no eligible points" if not family_size
                                                  else "unavailable: nonfinite bootstrap critical value"),
        "words": words.copy(), "windows": output, "confidence": confidence, "critical": critical,
        "bootstrap_max_pivots": maxima, "seed": seed,
        "original_event_count": len(original_ids), "resampling_cluster_count": len(cluster_ids),
        "eligible_family_size": family_size,
        "diagnostics": {
            "requested_replicates": bootstrap_replicates, "completed_replicates": completed,
            "failed_replicates": int(np.isposinf(maxima).sum()) if completed else 0,
            "failed_by_window": failed_by_window, "failed_points_by_window": failed_points_by_window,
            "critical_order_index_zero_based": quantile_index,
            "quantile_method": "higher; failed replicates remain positive infinity",
        },
        "settings": {"min_kept_events": min_kept_events, "min_effective_events": min_effective_events,
                     "bootstrap_replicates": bootstrap_replicates, "batch_size": batch_size, "shelf": shelf},
        "assumptions": [
            "The estimand is the event-weighted conditional signed proxy mean, not true physical shelf power.",
            "Original frame weights are preserved; clusters may have unequal mass and are not re-equalized.",
            "Resampling clusters are independent/exchangeable for this pairs bootstrap; temporal blocks are a sensitivity analysis.",
            "Words, windows, era, anchors, profiles and weighting rules are frozen; data-dependent grid/selection requires matching validation.",
            "The family comprises eligible supplied words and windows only, not other channels or unobserved conditions.",
            "Support floors and bootstrap-t calibration do not alone prove nominal coverage; relevant repeated-sample validation is required.",
        ],
    }


def widen_upper_band(band, multiplier, *, minimum_critical=1.):
    """Widen a band by a multiplier calibrated on declared models.

    Every original statistic and resampling draw is kept. The calibration
    models are not checked against the caller's data. The critical value is
    floored at ``minimum_critical`` before it is multiplied, so the widened
    margin stays conservative even when the original upper-tail bootstrap
    quantile is negative.
    """
    multiplier = _real(multiplier, "multiplier", lower=0)
    minimum_critical = _real(minimum_critical, "minimum_critical", lower=0)
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1")
    result = deepcopy(band)
    original = float(band["critical"])
    critical = multiplier * max(original, minimum_critical)
    result["bootstrap_critical"] = original
    result["critical"] = critical
    result["model_widening"] = {"multiplier": multiplier, "minimum_critical": minimum_critical,
        "scope": "Separately calibrated declared-model margin; no universal or physical coverage claim."}
    if band["status"] == "available" and not np.isfinite(critical):
        raise ValueError("widened critical value is not finite")
    if band["status"] == "available":
        for window in result["windows"].values():
            eligible = window["eligible"]
            window["signed_upper"][eligible] = window["signed_mean"][eligible] + critical * window["se"][eligible]
            window["upper"][eligible] = np.maximum(window["signed_upper"][eligible], 0.)
    return result


__all__ = ["simultaneous_upper_band", "widen_upper_band"]
