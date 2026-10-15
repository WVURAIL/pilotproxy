"""Each band's detector word, from the science tolerance table.

For every distinct keep word ``w`` of a band's current era (``eta = w / 2**16``):

    kept(w)   supported frames whose required word is at most w
    kappa(w)  kept probability; every event has equal mass and unsupported
              frames stay in the denominator
    x(w)      interference power per input relative to the noise, over the
              band: SHELF times the kept-weighted mean marker excess, floored at 0
    se(w)     the event-clustered standard error of that mean, times SHELF
    a(w)      (c x)^2 N G, the residual strength the tolerance table is written
              in: c the coherent fraction of the power, N the samples in one
              frame, G the frames it stays coherent, priced at the admissible
              minimum the measured c allows, one value for every band (``coherent_gain``)
    r(w)      a(w) R, the bias in units of the clean sigma; R is the band's
              response from the table (the combined distance D_V)
    r_hi(w)   r at x + 2 se, the value the rule tests

The decision starts at the band's knee, the detector's own operating point. A
threshold is usable when it meets the tolerance, ``r_hi <= 1``, within
``MAX_CALENDAR_YEARS``, so that its kept fraction is at least
``1 / MAX_CALENDAR_YEARS``. A band whose knee is usable is deployed at its knee.
Otherwise ``below_knee`` is the loosest threshold at or below the knee that meets
the tolerance (None if there is none); it keeps the most frames of those that
do, and the band is deployed there if it is usable. A band with no usable
threshold is not useful for the science the table describes and is not
deployed. Its ``reason`` is "out of time" when the tolerance is met but only
beyond ``MAX_CALENDAR_YEARS``, and "above tolerance" when it is not met at all.
A band can also be excluded with a stated reason (``exclude``), for example when
the detector does not see its marker; it is then not deployed whatever the sweep
says, and the sweep is still recorded.
The calendar time to collect the one retained year the table assumes is
``CALENDAR_YEARS_PER_RETAINED_YEAR / kappa``: a retained year is 8,760 h on the sky,
observed at the science-quality duty ``ON_SKY_DUTY``.

The last-year guard (``decide_guarded``) protects the tolerance against a recent
change the full era dilutes. The full-era decision stands; a deployed band's
word is then tested on the curves of the era's last ``LAST_YEAR_DAYS`` of frames
(``last_year``: kappa, x and its standard error from those frames, the same G
and R). Twelve months is one full seasonal cycle, so the recent state is judged
over every season and no season weighs twice. If r_hi there is at most 1 the
decision is unchanged. Otherwise the band moves to the loosest word at or below
its word whose r_hi is at most 1 on both the full-era and the last-year curves
and whose full-era kappa meets ``MAX_CALENDAR_YEARS``; with no such word it is
not deployed, "fails the tolerance on its last year". No knee is computed on
the last year. Kappa, x, r and calendar years are always read on the full era.

The table (``data/tolerance_table.json``) is the science side's only input
here. RFIsher writes it once per forecast configuration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from pilot_proxy.config.project import default_project

Q16 = 65536
_TEMPLATE = default_project().interference
FINE_BIN_HZ = 390625 / 128                 # 3051.76 Hz, the bin the marker excess is read in
# Marker excess to band power: the marker sits marker_to_band_db below the band's total power.
SHELF = 10 ** (_TEMPLATE.marker.marker_to_band_db / 10) * FINE_BIN_HZ / _TEMPLATE.band_width_hz
FRAME_SAMPLES = 16384                      # samples per frame
FRAME_SECONDS = FRAME_SAMPLES / 390625     # 41.94 ms
# Coherent fraction: the share of the per-input interference amplitude that survives in a
# calibrated visibility, measured in the matched captures (data/coherent_fraction.json).
# Both measurements are CHIME-derived; the code-only cut withholds them (CHIME_DERIVED.md), and
# then the module still imports but the rule refuses to price a band.


def _measurement(name: str) -> dict | None:
    path = resources.files("pilot_proxy.characterization").joinpath(f"data/{name}")
    return json.loads(path.read_text()) if path.is_file() else None


def _require(data: dict | None, name: str) -> dict:
    if data is None:
        raise FileNotFoundError(f"data/{name} is withheld from this copy (CHIME_DERIVED.md); "
                                "the threshold rule needs it")
    return data


_COHERENT = _measurement("coherent_fraction.json")
COHERENT_FRACTION_BY_BAND = {int(k): float(v) for k, v in (_COHERENT or {"by_band": {}})["by_band"].items()}
COHERENT_FRACTION = float(_COHERENT["value"]) if _COHERENT else float("nan")
COHERENT_FRACTION_SCALE = _COHERENT["scale"] if _COHERENT else None
# G: the frames the residual keeps its phase. The coherent fraction is measured from products of
# distinct frames, so it cannot be priced below the admissible minimum (data/coherent_gain.json).
# One coherent fraction serves every band, so one G does too: that of the dumps that measured it.
_GAIN = _measurement("coherent_gain.json")
GAIN_FRAMES = int(_GAIN["frames"]) if _GAIN else None
GUARD_STANDARD_ERRORS = 2.0
MAX_CALENDAR_YEARS = 30.0                  # a threshold needing longer than this is not usable
ON_SKY_DUTY = 0.152                        # science-quality duty: night only, 94 of 309 days retained
CALENDAR_YEARS_PER_RETAINED_YEAR = 8760 / (ON_SKY_DUTY * 8766)   # 6.57
LAST_YEAR_DAYS = 365.25                    # the last-year guard's window: one full seasonal cycle
TABLE_SCHEMA = "rfisher-channel-table-v2"


def load_table(path=None) -> dict:
    """The science tolerance table (the packaged copy unless ``path`` is given)."""
    if path is None:
        text = resources.files("pilot_proxy.characterization").joinpath("data/tolerance_table.json").read_text()
    else:
        text = Path(path).read_text()
    table = json.loads(text)
    if table.get("schema") != TABLE_SCHEMA:
        raise ValueError(f"not a {TABLE_SCHEMA} table")
    return table


def response(table: dict, channel: int) -> float:
    """The band's response: bias / sigma of the combined distance per unit strength."""
    row = next(r for r in table["channels"] if r["channel"] == channel)
    return abs(row["response_dv"])


@dataclass
class ChannelFrames:
    """One band's current-era frames, as the rule needs them."""
    channel: int
    required: np.ndarray       # uint64 Q16 word each frame needs to be kept
    weight: np.ndarray         # probability mass of each frame
    supported: np.ndarray      # frame has a usable score
    event: np.ndarray          # event index of each frame
    excess: np.ndarray         # marker excess Q - 1 of each frame
    knee_word: int
    current_word: int
    current_keep: np.ndarray | None = None
    time: np.ndarray | None = None           # frame time, s (the last-year guard)


def load_frames(channel: int, scores_dir) -> ChannelFrames:
    """Read ``ch<N>.json`` and ``ch<N>.npz`` of the score cache and the marker
    excess from the archive product they name, and check the keep rule
    reproduces the accepted mask at the current word."""
    scores_dir = Path(scores_dir)
    meta = json.loads((scores_dir / f"ch{channel}.json").read_text())
    product = meta["original_product_identity"]
    if Path(product["path"]).stat().st_size != product["bytes"]:
        raise ValueError(f"channel {channel}: {product['path']} is not the product the scores were built on")
    with np.load(scores_dir / f"ch{channel}.npz", allow_pickle=False) as z:
        rows = z["source_row_index"]
        frames = ChannelFrames(channel=channel, required=z["required_q16"], weight=z["probability_weight"],
                               supported=z["supported"], event=z["frame_event_index"], excess=np.empty(0),
                               knee_word=int(meta["knee_threshold"]["eta_q16"]),
                               current_word=int(meta["current_threshold"]["eta_q16"]),
                               current_keep=z["current_keep"],
                               time=z["frame_time"] if "frame_time" in z.files else None)
    with np.load(product["path"], allow_pickle=False) as p:
        frames.excess = p["normalized_pilot_excess"][:, 0][rows]
    if not np.array_equal(keep(frames, frames.current_word), frames.current_keep):
        raise ValueError(f"channel {channel}: the keep rule does not reproduce the accepted mask")
    return frames


def last_year(frames: ChannelFrames) -> ChannelFrames:
    """The frames of the era's last ``LAST_YEAR_DAYS`` (one full seasonal cycle). Frames keep their era
    weights, so kappa is relative to the subset; the knee is the full era's (none is computed here)."""
    if frames.time is None:
        raise ValueError(f"channel {frames.channel}: the cache lacks frame_time")
    sel = frames.time >= frames.time.max() - LAST_YEAR_DAYS * 86400
    return ChannelFrames(channel=frames.channel, required=frames.required[sel], weight=frames.weight[sel],
                         supported=frames.supported[sel], event=frames.event[sel], excess=frames.excess[sel],
                         knee_word=frames.knee_word, current_word=frames.current_word, time=frames.time[sel])


def keep(frames: ChannelFrames, word: int) -> np.ndarray:
    return frames.supported & (frames.required <= np.uint64(word))


def sweep(frames: ChannelFrames, shelf: float = SHELF) -> dict:
    """kappa, x and its event-clustered standard error at every distinct supported word."""
    idx = np.flatnonzero(frames.supported)
    idx = idx[np.argsort(frames.required[idx], kind="stable")]
    words = frames.required[idx]
    w, q, e = frames.weight[idx], frames.excess[idx], frames.event[idx]
    total = frames.weight.sum()
    A = np.zeros(int(frames.event.max()) + 1)   # per-event sum of w q over kept frames
    B = np.zeros_like(A)                         # per-event sum of w over kept frames
    saa = sab = sbb = 0.0
    W = Q = 0.0
    events = 0
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
        a += w[i] * q[i]
        b += w[i]
        A[ei], B[ei] = a, b
        saa += a * a
        sab += a * b
        sbb += b * b
        W += w[i]
        Q += w[i] * q[i]
        if i == last[j]:
            m = Q / W
            out["kappa"][j] = W / total
            out["mean"][j] = m
            out["se"][j] = np.sqrt(max(saa - 2 * m * sab + m * m * sbb, 0.0)) / W
            kept_frames[j] = i + 1
            kept_events[j] = events
            j += 1
    return dict(word=words[last], kappa=out["kappa"], x=np.maximum(shelf * out["mean"], 0.0),
                se=shelf * out["se"], mean_excess=out["mean"], kept_frames=kept_frames, kept_events=kept_events)


def gain_of(tau, model: str):
    """G(tau): the residual's power in a long average over a white residual's, the sum of rho(d)
    over every lag d (tau in frame intervals). coth(1 / 2 tau) for the exponential rho = exp(-d / tau),
    about 2 tau; the exact lag sum of the triangular rho = max(1 - d / tau, 0), about tau."""
    tau = np.asarray(tau, dtype=float)
    if model == "exp":
        return 1 / np.tanh(1 / (2 * tau))
    m = np.ceil(tau) - 1                                       # lags 1..m have d < tau
    return 1 + 2 * m - m * (m + 1) / tau


@lru_cache
def admissible_min_gain(frames: int) -> float:
    """The least G(tau) / rho_bar_n(tau) over the coherence time tau for an n-frame dump, the lower
    of an exponential and a triangular frame-to-frame correlation, with G(tau) the lag sum
    (``gain_of``). 6.75 for 4 frames, 15.23 for 11, 40.00 for 33; the triangular sets each."""
    d = np.arange(1, frames)[:, None]
    tau = np.geomspace(0.05, 1e4, 20001)                      # in frame intervals
    best = np.inf
    for model, rho in (("exp", np.exp(-d / tau)), ("tri", np.clip(1 - d / tau, 0, None))):
        rbar = (2 * (frames - d) * rho).sum(axis=0) / (frames * (frames - 1))
        ok = rbar > 0
        best = min(best, float(np.min(gain_of(tau[ok], model) / rbar[ok])))
    return best


def coherent_gain(channel: int) -> float:
    """G the decision prices a band at: the admissible minimum of the dumps that measured the
    coherent fraction, the same for every band."""
    _require(_GAIN, "coherent_gain.json")
    return admissible_min_gain(GAIN_FRAMES)


def strength(x, gain: float, c: float = COHERENT_FRACTION):
    """Residual strength a = (c x)^2 N G."""
    if c is COHERENT_FRACTION:
        _require(_COHERENT, "coherent_fraction.json")
    return (c * np.asarray(x)) ** 2 * FRAME_SAMPLES * gain


def bias_over_sigma(s: dict, R: float, gain: float,
                    guard: float = GUARD_STANDARD_ERRORS) -> tuple[np.ndarray, np.ndarray]:
    """(r, r_hi): bias / sigma at the measured power and at its upper end."""
    return strength(s["x"], gain) * R, strength(np.maximum(s["x"] + guard * s["se"], 0.0), gain) * R


def _at(s: dict, word: int) -> int | None:
    """Index of the largest swept word not above ``word`` (None if none)."""
    i = int(np.searchsorted(s["word"], np.uint64(word), side="right")) - 1
    return i if i >= 0 else None


def _point(s: dict, word: int, R: float, gain: float) -> dict:
    """The sweep read at ``word``: kappa, x, r and calendar years of the frames it keeps."""
    i = _at(s, word)
    if i is None:
        return dict(word=int(word), eta=word / Q16, kappa=0.0, time_years=float("inf"), x=None, se=None, a=None,
                    r=None, r_hi=None, kept_frames=0, kept_events=0, coherence_limit_s=None)
    r, r_hi = (float(v[i]) for v in bias_over_sigma(s, R, gain))
    kappa = float(s["kappa"][i])
    return dict(word=int(word), eta=word / Q16, kappa=kappa,
                time_years=CALENDAR_YEARS_PER_RETAINED_YEAR / kappa if kappa > 0 else float("inf"),
                x=float(s["x"][i]), se=float(s["se"][i]), a=float(strength(s["x"][i], gain)),
                r=r, r_hi=r_hi, kept_frames=int(s["kept_frames"][i]), kept_events=int(s["kept_events"][i]),
                coherence_limit_s=FRAME_SECONDS * gain / r_hi if r_hi > 0 else float("inf"))


def decide(s: dict, knee_word: int, R: float, *, gain: float, max_years: float = MAX_CALENDAR_YEARS,
           exclude: str | None = None) -> dict:
    """Where the band is deployed (its knee, or the loosest usable threshold below it), or why it is not.
    ``gain`` is G, the band's coherence in frames (``coherent_gain``)."""
    r, r_hi = bias_over_sigma(s, R, gain)

    def usable(p):
        return p is not None and p["r_hi"] is not None and p["r_hi"] <= 1 and p["time_years"] <= max_years

    knee = _point(s, knee_word, R, gain)
    below = None
    if usable(knee):
        deployed, at = knee, "knee"
    else:
        idx = np.flatnonzero((s["word"] <= np.uint64(knee_word)) & (r_hi <= 1) & (s["kappa"] > 0))
        if len(idx):
            below = _point(s, int(s["word"][int(idx[-1])]), R, gain)
        deployed, at = (below, "below the knee") if usable(below) else (None, None)
    reason = None if deployed else ("out of time" if below is not None else "above tolerance")
    if exclude:
        deployed, at, reason = None, None, exclude
    return dict(knee=knee, below_knee=below, outcome="useful" if deployed else "not useful", reason=reason,
                deployed=deployed, deployed_at=at, deployed_word=deployed["word"] if deployed else None)


GUARD_FAILED = "fails the tolerance on its last year"


def decide_guarded(full: dict, knee_word: int, last: dict, R: float, *, gain: float,
                   max_years: float = MAX_CALENDAR_YEARS, exclude: str | None = None) -> dict:
    """``decide`` on the full era (sweep ``full``) with the last-year guard (sweep ``last``, from
    ``last_year``). Adds ``full_era_word``, ``last_year_r_hi`` (at the full-era word) and ``guard_acted``."""
    d = decide(full, knee_word, R, gain=gain, max_years=max_years, exclude=exclude)
    out = dict(d, full_era_word=d["deployed_word"], last_year_r_hi=None, guard_acted=False)
    if d["deployed"] is None:
        return out
    word = d["deployed_word"]
    out["last_year_r_hi"] = _point(last, word, R, gain)["r_hi"]
    if out["last_year_r_hi"] is not None and out["last_year_r_hi"] <= 1:
        return out
    # Both curves are constant between the full era's swept words (the last year's words are among them).
    _, r_hi = bias_over_sigma(full, R, gain)
    _, r_hi_last = bias_over_sigma(last, R, gain)
    at_last = np.searchsorted(last["word"], full["word"], side="right") - 1
    last_ok = (at_last >= 0) & (np.where(at_last >= 0, r_hi_last[np.maximum(at_last, 0)], np.inf) <= 1)
    idx = np.flatnonzero((full["word"] <= np.uint64(word)) & (r_hi <= 1) & last_ok & (full["kappa"] > 0))
    moved = _point(full, int(full["word"][idx[-1]]), R, gain) if len(idx) else None
    out["guard_acted"] = True
    if moved is None or moved["time_years"] > max_years:
        return dict(out, outcome="not useful", reason=GUARD_FAILED, deployed=None, deployed_at=None, deployed_word=None)
    return dict(out, deployed=moved, deployed_word=moved["word"], deployed_at="below the knee")
