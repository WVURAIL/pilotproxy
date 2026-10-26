"""Signed comparisons of capture powers on the same frames, with no physical truth model."""
from __future__ import annotations

import numpy as np


def signed_proxy(powers, target_norm: int, reference_norm: int, shelf: float):
    """Return the signed proxy of exact [frame, term, ...] powers, or NaN where the reference sum is zero."""
    p = np.asarray(powers)
    if p.dtype != np.uint64 or p.ndim < 2 or p.shape[1] != 3:
        raise ValueError("powers must be uint64 [frame, 3, ...]")
    if target_norm <= 0 or reference_norm <= 0 or not shelf > 0:
        raise ValueError("positive norms and shelf coefficient required")
    denominator = p[:, 1].astype(float) + p[:, 2].astype(float)
    ratio = np.divide(p[:, 0], denominator, out=np.full(denominator.shape, np.nan),
                      where=denominator > 0)
    return shelf * (ratio * reference_norm / target_norm - 1)


def exact_frame_join(left, right):
    """Return the row indices of the FPGA frame origins shared by two strictly increasing vectors.

    Frames are matched on the exact origin, so a dropped frame in one capture
    cannot shift the pairing of the frames after it.
    """
    a, b = np.asarray(left), np.asarray(right)
    for v in (a, b):
        if (v.ndim != 1 or v.dtype.kind not in 'iu' or np.any(v < 0)
                or np.any(v > np.iinfo(np.int64).max) or np.any(v[1:] <= v[:-1])):
            raise ValueError("frame origins must be strictly increasing integers")
    a, b = a.astype(np.int64), b.astype(np.int64)
    _, i, j = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    return i, j


def event_weights(events):
    """Return frame weights that give each event equal mass and share it equally among its frames."""
    e = np.asarray(events)
    if e.ndim != 1:
        raise ValueError("events must be one dimensional")
    _, inverse, counts = np.unique(e, return_inverse=True, return_counts=True)
    if not len(counts):
        raise ValueError("empty population")
    return 1 / (len(counts) * counts[inverse])


def fit_training_transfer(x, ratio, events, live_floor=0.5, auto=None, control=None):
    """Fit a weighted line per input, using only the supplied training frames.

    The response is the fractional change of a cross-frequency power ratio from
    its baseline on the training frames. Any power in the target and reference
    terms moves this ratio, so the response does not isolate the DTV power. The
    inputs are diagnostic coordinates and are not independent validation
    replicates.
    """
    x, ratio = np.asarray(x, float), np.asarray(ratio, float)
    if x.shape != ratio.shape or x.ndim != 2:
        raise ValueError("x and ratio must share [frame, input] shape")
    if np.asarray(events).shape != (len(x),):
        raise ValueError("events must match the original frame count")
    for value in (auto, control):
        if value is not None and np.asarray(value).shape != x.shape:
            raise ValueError("auto/control must match predictor shape")
    w = event_weights(events)[:, None]
    valid = np.isfinite(x).all(0) & np.isfinite(ratio).all(0) & (ratio > 0).all(0)
    if auto is not None:
        valid &= (np.asarray(auto) > live_floor).all(0)
    if control is not None:
        valid &= (np.asarray(control) > live_floor).all(0)
    baseline = np.sum(w * ratio, axis=0)
    y = ratio / baseline - 1
    x0 = np.sum(w * x, axis=0)
    y0 = np.sum(w * y, axis=0)
    variance = np.sum(w * (x - x0) ** 2, axis=0)
    valid &= np.isfinite(variance) & (variance > 1e-30)
    slope = np.divide(np.sum(w * (x - x0) * (y - y0), axis=0), variance,
                      out=np.full_like(variance, np.nan), where=valid)
    return dict(baseline=baseline, x0=x0, y0=y0, slope=slope, live=valid)


def transfer_residuals(model, x, ratio):
    """Return the response, the prediction of a frozen training model, and their residual.

    The live inputs are those of the training fit, so no input is selected on
    the holdout frames.
    """
    y = np.asarray(ratio, float) / model['baseline'] - 1
    prediction = model['y0'] + model['slope'] * (np.asarray(x) - model['x0'])
    return y, prediction, y - prediction


def lag_moments(values, origins, keep, max_lag=8, frame_samples=16384):
    """Return the signed lag moments, centered covariances, and pair counts of one capture.

    Both frames of a pair must be kept and spaced exactly
    ``lag * frame_samples`` samples apart. The mean is taken over the kept rows of
    this capture. The sample moments and the biased finite-record covariance
    describe this capture only and are not a long-timescale noise model.
    """
    v = np.asarray(values)
    t = np.asarray(origins)
    k = np.asarray(keep, bool)
    exact_frame_join(t, t)
    t = t.astype(np.int64)
    if max_lag < 0 or frame_samples <= 0 or (t.size and int(t[-1]) > np.iinfo(np.int64).max - max_lag * frame_samples):
        raise ValueError("invalid lag or FPGA origin overflow")
    if v.shape[0] != t.size or k.shape != t.shape:
        raise ValueError("time/selection/value shape mismatch")
    if not np.all(np.isfinite(v)):
        raise ValueError("lag values must be finite on the fixed coordinates")
    mean = v[k].mean(axis=0) if k.any() else np.full(v.shape[1:], np.nan)
    moments, covariances, counts = [], [], []
    for lag in range(max_lag + 1):
        i, j = exact_frame_join(t + lag * frame_samples, t)
        take = k[i] & k[j]
        i, j = i[take], j[take]
        counts.append(len(i))
        moments.append(np.mean(v[i].conj() * v[j], axis=0) if len(i)
                       else np.full(v.shape[1:], np.nan))
        covariances.append(np.mean((v[i] - mean).conj() * (v[j] - mean), axis=0)
                           if len(i) else np.full(v.shape[1:], np.nan))
    return np.asarray(moments), np.asarray(covariances), np.asarray(counts)


def coherent_cross_moment(values, keep):
    """Return the off-diagonal real cross moment of the kept rows.

    A negative result is kept as it is, and no square root is taken.
    """
    v = np.asarray(values)[np.asarray(keep, bool)]
    n = len(v)
    if n < 2:
        return np.full(v.shape[1:], np.nan)
    return (np.abs(v.sum(axis=0)) ** 2 - np.sum(np.abs(v) ** 2, axis=0)) / (n * (n - 1))
