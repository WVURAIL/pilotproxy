"""Select maximum retention within the linear criterion and checked finite limit.

The signed pilot-to-shelf proxy and clustered sampling error are measured from
the selected population. The simple model assigns temperature Tsys*x and
distributes its squared amplitude uniformly over the supported forecast modes.
Thus a = power_scale*x**2, in mK^2. The channel table supplies power_scale;
it is a model normalization, not an empirical sky calibration.
A required power_limit caps the residual at the checked finite-model limit.
Threshold selection must be evaluated on independent data before reporting
out-of-sample performance; the two-standard-error margin is not a simultaneous
confidence guarantee over a searched threshold grid.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from pilot_proxy.config.project import default_project
from pilot_proxy.product_contract import pilot_excess_from_fine_powers

Q16 = 65536
_TEMPLATE = default_project().interference
FINE_BIN_HZ = 390625 / 128
SHELF = 10 ** (_TEMPLATE.marker.marker_to_band_db / 10) * FINE_BIN_HZ / _TEMPLATE.band_width_hz
FRAME_SAMPLES = 16384
GUARD_STANDARD_ERRORS = 2.0
TABLE_SCHEMA = "rfisher-channel-table-v3"

def load_table(path=None):
    source = resources.files("pilot_proxy.characterization").joinpath("data/tolerance_table.json") if path is None else Path(path)
    table = json.loads(source.read_text())
    if (table.get("schema") != TABLE_SCHEMA
            or table.get("inputs", {}).get("residual") != "flat-additive-power-v1"
            or table["inputs"].get("power_units") != "mK^2"):
        raise ValueError("expected a unit-additive-power table in mK^2 (schema v3)")
    return table

def power_scale(table, channel):
    row = next(row for row in table["channels"] if row["channel"] == channel)
    norm = row.get("normalization", {})
    if norm.get("model") != "flat-temperature-variance-v1":
        raise ValueError("table lacks the simple temperature-variance normalization")
    value = float(norm["power_scale"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("power_scale must be finite and positive")
    return value


def response(table, channel):
    return abs(next(row["response_dv"] for row in table["channels"] if row["channel"] == channel))

@dataclass
class ChannelFrames:
    channel: int
    required: np.ndarray
    weight: np.ndarray
    supported: np.ndarray
    event: np.ndarray
    excess: np.ndarray
    current_word: int = 1
    current_keep: np.ndarray | None = None
    time: np.ndarray | None = None

def validate_frames(f):
    n = len(f.required)
    if f.required.dtype != np.dtype("uint64") or f.required.ndim != 1:
        raise ValueError("required words must be a uint64 vector")
    if any(np.shape(x) != (n,) for x in (f.weight, f.supported, f.event, f.excess)):
        raise ValueError("frame arrays must have matching shapes")
    if f.supported.dtype != np.dtype("bool") or f.event.dtype.kind not in "iu" or np.any(f.event < 0):
        raise ValueError("invalid support or event IDs")
    if not np.isfinite(f.weight).all() or np.any(f.weight <= 0):
        raise ValueError("weights must be finite and positive")
    if not np.isfinite(f.excess[f.supported]).all():
        raise ValueError("supported frames must have finite excess")
    # Compact IDs prevent untrusted/noncontiguous event numbers from huge allocations.
    _, f.event = np.unique(f.event, return_inverse=True)

def load_frames(channel, scores_dir):
    from .score_cache import sha
    scores_dir = Path(scores_dir)
    meta = json.loads((scores_dir / f"ch{channel}.json").read_text())
    product = meta["original_product_identity"]
    score_path = scores_dir / f"ch{channel}.npz"
    if sha(score_path) != meta["npz_sha256"]:
        raise ValueError("score cache differs from its metadata")
    product_path = (scores_dir / product["path"]).resolve()
    if product_path.stat().st_size != product["bytes"] or sha(product_path) != product["sha256"]:
        raise ValueError("source product differs from its metadata")
    with np.load(score_path, allow_pickle=False) as z:
        rows = z["source_row_index"]
        f = ChannelFrames(channel, z["required_q16"], z["probability_weight"],
                          z["supported"], z["frame_event_index"], np.empty(len(rows)),
                          int(meta["current_threshold"]["eta_q16"]), z["current_keep"],
                          z["frame_time"] if "frame_time" in z.files else None)
    with np.load(product_path, allow_pickle=False) as z:
        f.excess = pilot_excess_from_fine_powers(z, rows)
    validate_frames(f)
    if not np.array_equal(keep(f, f.current_word), f.current_keep):
        raise ValueError("the keep rule does not reproduce the accepted mask")
    return f

def keep(frames, word):
    if isinstance(word, bool) or not isinstance(word, (int, np.integer)) or not 1 <= word < 2**64:
        raise ValueError("threshold must be a positive uint64 word")
    return frames.supported & (frames.required <= np.uint64(word))

def _binary_integers(values):
    """Represent finite binary floating-point values on one exact integer scale."""
    ratios = [float(value).as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    return [numerator * (scale // denominator) for numerator, denominator in ratios], scale


def _sqrt_ratio(numerator, denominator):
    """Return sqrt(numerator) / denominator without floating-point intermediates outside range."""
    if numerator == 0:
        return 0.0
    root_shift = (numerator.bit_length() - 1) // 2
    denominator_shift = denominator.bit_length() - 1
    root = math.sqrt(numerator / (1 << (2 * root_shift)))
    scaled_denominator = denominator / (1 << denominator_shift)
    return math.ldexp(root / scaled_denominator, root_shift - denominator_shift)


def sweep(frames: ChannelFrames, shelf: float = SHELF) -> dict:
    """kappa, x and its event-clustered standard error at every distinct supported word."""
    validate_frames(frames)
    idx = np.flatnonzero(frames.supported)
    idx = idx[np.argsort(frames.required[idx], kind="stable")]
    words = frames.required[idx]
    if not len(words):
        return dict(word=np.empty(0, dtype=np.uint64),
                    **{key: np.empty(0, dtype=float) for key in ("kappa", "x", "se", "mean_excess")},
                    kept_frames=np.empty(0, dtype=np.int64), kept_events=np.empty(0, dtype=np.int64))
    # Binary floating-point inputs have exact integer numerators over powers
    # of two. Common scales cancel from retention and the cluster equation.
    # Integer moments therefore avoid both cancellation and prefix rebuilds.
    weights, _ = _binary_integers(frames.weight)
    total = sum(weights)
    w = [weights[i] for i in idx]
    q, excess_scale = _binary_integers(frames.excess[idx])
    e = frames.event[idx]
    A = [0] * (int(frames.event.max()) + 1)  # per-event sum of integer w*q
    B = [0] * len(A)                         # per-event sum of integer w
    saa = sab = sbb = W = Q = events = 0
    last = np.r_[np.flatnonzero(np.diff(words)), len(words) - 1]
    out = {k: np.empty(len(last)) for k in ("kappa", "mean", "se")}
    kept_frames = np.empty(len(last), dtype=np.int64)
    kept_events = np.empty(len(last), dtype=np.int64)
    j = 0
    for i in range(len(words)):
        ei = e[i]
        a, b = A[ei], B[ei]
        if b == 0:
            events += 1
        saa -= a * a
        sab -= a * b
        sbb -= b * b
        delta = w[i] * q[i]
        a += delta
        b += w[i]
        A[ei], B[ei] = a, b
        saa += a * a
        sab += a * b
        sbb += b * b
        W += w[i]
        Q += delta
        if i == last[j]:
            # sum_e (A_e - (Q/W) B_e)^2 / W^2, before removing
            # the common excess scale. Every term remains exact here.
            numerator = saa * W * W - 2 * Q * W * sab + Q * Q * sbb
            error = _sqrt_ratio(numerator, W * W * excess_scale)
            out["kappa"][j] = W / total
            out["mean"][j] = Q / (W * excess_scale)
            out["se"][j] = error
            kept_frames[j] = i + 1
            kept_events[j] = events
            j += 1
    return dict(word=words[last], kappa=out["kappa"], x=np.maximum(shelf * out["mean"], 0.0),
                se=shelf * out["se"], mean_excess=out["mean"], kept_frames=kept_frames, kept_events=kept_events)


def bias_over_sigma(s, R, *, power_scale, guard=GUARD_STANDARD_ERRORS):
    if not np.isfinite(power_scale) or power_scale <= 0:
        raise ValueError("power_scale must be a finite positive mK^2 model coefficient")
    if not np.isfinite(R) or R < 0 or not np.isfinite(guard) or guard < 0:
        raise ValueError("response and sampling margin must be finite and nonnegative")
    return (s["x"]**2 * power_scale * R,
            np.maximum(s["x"] + guard * s["se"], 0)**2 * power_scale * R)

def decide(s, R, *, power_scale, power_limit, exclude=None):
    """Maximize retention subject to the linear criterion and checked power range."""
    if not np.isfinite(power_limit) or power_limit <= 0:
        raise ValueError("power_limit must be a finite positive checked bound in mK^2")
    measured, upper = bias_over_sigma(s, R, power_scale=power_scale)
    upper_power = np.maximum(s["x"] + GUARD_STANDARD_ERRORS * s["se"], 0)**2 * power_scale
    valid = (np.isfinite(upper) & (upper <= 1) & (upper_power <= power_limit) & (s["kappa"] > 0)
             & (s["kept_events"] >= 2) & (s["word"] >= 1))
    ix = np.flatnonzero(valid)
    point = None
    if len(ix) and exclude is None:
        i = int(ix[np.argmax(s["kappa"][ix])])
        point = dict(word=int(s["word"][i]), eta=float(s["word"][i]) / Q16,
                     kappa=float(s["kappa"][i]), time_years=1 / float(s["kappa"][i]),
                     x=float(s["x"][i]), se=float(s["se"][i]),
                     a=float(s["x"][i]**2 * power_scale), a_hi=float(upper_power[i]),
                     r=float(measured[i]), r_hi=float(upper[i]),
                     kept_frames=int(s["kept_frames"][i]), kept_events=int(s["kept_events"][i]))
    return dict(outcome="useful" if point else "not useful", power_limit=float(power_limit),
                reason=None if point else (exclude or "no supported threshold meets tolerance"),
                deployed=point, deployed_word=point["word"] if point else None,
                deployed_at="maximum retention" if point else None)
