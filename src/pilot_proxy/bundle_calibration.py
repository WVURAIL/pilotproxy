# coding=utf-8
"""Calibrate a runtime bundle from captures and test a bundle on other captures.

The two entry points are the calibrate and test steps of docs/RUNNING_THE_DETECTOR.md.
Fine powers are computed with the deployed integer path, so a word found here is
the word the stage applies to the same voltages.
"""
from __future__ import annotations

import glob
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from pilot_proxy.capture_reference import expected_matched_filter_row_projections
from pilot_proxy.detector_contract import (
    null_power_ratio_from_weight_norms,
    weight_term_norms_sq,
)
from pilot_proxy.detectors.narrowband_marker.anchors import WINDOW_HALF_WIDTH, bulk_mask
from pilot_proxy.fine_decision import (
    CALIBRATION_DECISION_VERSION,
    FINE_DECISION_V2_VERSION,
    FINE_DECISION_V3_VERSION,
    FINE_DECISION_VERSION,
    FINE_DECISION_VERSIONS,
    calibration_geometry,
    decision_functions,
    pack_bulk_mask,
    v3_geometry,
    v3_word,
)
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.power_limit_word import (
    EVALUATION,
    SELECTION,
    control_corrections,
    limit_test,
    ratio_excess,
)
from pilot_proxy.runtime_bundle import (
    CONTROL_PHYSICAL_CHANNEL,
    _write_sha256sums,
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)
from pilot_proxy.sun import is_night

FINE_BINS = 256
WINDOWS = 128
PAD_FACTOR = 2
GUARD_FINE_BINS = 1
Q16 = 65536
# 65 designated bins, the search of record
SEARCH_HALF_WIDTH = 32
# above this share of frames with a pilot, the knee has no clean frames to bend from
ALWAYS_ON_SHARE = 0.9
# the fewest kept captures the archive rule accepts
MIN_KNEE_CAPTURES = 30
BOOTSTRAP_REPLICATES = 200
# chance of the hits under the null, kept small for the search over the window
MAX_LINE_TAIL = 1e-4
MIN_SIMULATED_FRAMES = 2000
MAX_SCORE = (1 << 63) - 1
# frames scored together, about 0.2 GB of float temporaries
WORD_CHUNK = 65536
# control frames scored together, 33 kB each in the pseudo-search peaks
NULL_CHUNK = 4096
# relative float gap below which a frame is scored exactly, far above the 1e-15 error of the float word
FLOAT_MARGIN = 1e-12
BUNDLE_FILES = ("detector_contract.json", "pilot_profiles.json", "weights.bin", "weights.manifest.json")
# the knee (v1, v2) or the scaled control word (v3) is the default word of a channel, and the power-limit word keeps
# the most frames under a checked residual power
THRESHOLDS = ("knee", "power_limit")
DECISIONS_SCHEMA = "pilotproxy-power-decisions-v4"
# reported only for a channel that passes
PASSING_FIELDS = ("word", "kappa", "ratio", "x", "upper_shelf", "kept_frames", "kept_captures")
# The power-limit word of a v3 channel with no passing word. The smallest word keeps a frame only when it has no
# target power, so DtvRfiMask and the relay, which apply the power-limit word where a bundle has one, mask the
# channel whole without a permanent list.
REJECT_WORD_Q16 = 1
# The reason of a channel whose calibration captures never showed its pilot, so its estimate reads only noise. It is
# not the threshold rule's "DTV without a matching pilot", which the operating policy compiles at the bundle word.
NO_PILOT_REASON = "pilot not seen in the calibration captures"


def iter_blocks(path: str | Path, capture_format: str, channel_id: int, block_samples: int,
                max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield uint8 [block_samples, streams] blocks of one coarse channel, on the block grid."""
    import h5py

    with h5py.File(path, "r") as handle:
        if capture_format == "chime_baseband":
            if int(handle.attrs["freq_id"]) != channel_id:
                raise ValueError(f"{path} holds freq_id {handle.attrs['freq_id']}, not {channel_id}")
            data = handle["baseband"]
            start = (-int(handle.attrs["time0_fpga_count"])) % block_samples

            def read(lo: int, hi: int) -> np.ndarray:
                return data[lo:hi]
        elif capture_format == "chord_baseband":
            data = handle["baseband_subset"]
            coarse = [int(c) for c in data.attrs["coarse_freq"]]
            if channel_id not in coarse:
                raise ValueError(f"{path} does not hold coarse_freq {channel_id}")
            index = coarse.index(channel_id)
            # rfi antennas never enter the detector sum
            labels = [l.decode() if isinstance(l, bytes) else str(l) for l in data.attrs.get("dish_label", [])]
            dishes = [j for j in range(data.shape[3]) if not (labels and labels[j].startswith("RFI"))]
            start = (-int(data.attrs["fpga_seq_num"])) % block_samples

            def read(lo: int, hi: int) -> np.ndarray:
                return data[lo:hi, index][:, :, dishes].reshape(hi - lo, -1)
        else:
            raise ValueError(f"unknown capture_format {capture_format!r}")
        frames = (data.shape[0] - start) // block_samples
        for frame in range(frames if max_frames is None else min(frames, max_frames)):
            lo = start + frame * block_samples
            yield read(lo, lo + block_samples)


def iter_run_frames(path: str | Path, channel_id: int, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield the uint64 [3, 256] fine powers of one channel from a dtv_fine_powers file of a detect-only run."""
    yield from run_rows(path, channel_id, max_frames)


def run_rows(path: str | Path, channel_id: int, max_frames: int | None = None) -> np.ndarray:
    """Return the uint64 [N, 3, 256] tested fine powers of one channel from a dtv_fine_powers file."""
    from pilot_proxy.run_files import read_rows

    rows, _ = read_rows(path, channel_id, max_frames)
    if rows is None:
        return np.empty((0, 3, FINE_BINS), np.uint64)
    # zeros were not tested, and 0xFF fills a block missing from the concat window
    missing = np.iinfo(np.uint64).max
    return rows[rows.any(axis=(1, 2)) & ~(rows == missing).all(axis=(1, 2))]


def fine_powers(block: np.ndarray, weights: bytes, k: int, time_reverse: bool) -> np.ndarray:
    """Return the exact uint64 fine powers [3, 256] of one block."""
    if block.shape[0] != k * WINDOWS:
        raise ValueError(f"a block needs {k * WINDOWS} samples, got {block.shape[0]}")
    rows = np.ascontiguousarray(block.T).reshape(-1, k)
    if time_reverse:
        rows = rows[:, ::-1]
    rows = rows.astype(np.int16)
    projections = expected_matched_filter_row_projections(((rows >> 4) & 15) - 8, (rows & 15) - 8, weights)
    return fine_power_fx(projections, num_streams=block.shape[1])


def fine_ratio(fine: np.ndarray) -> np.ndarray:
    """Return F2[b] = 2 S_t / (S_l + S_u) for [..., 3, 256] powers, zero on an empty reference."""
    fine = np.asarray(fine, dtype=np.float64)
    den = fine[..., 1, :] + fine[..., 2, :]
    return np.where(den > 0, 2.0 * fine[..., 0, :] / np.where(den > 0, den, 1.0), 0.0)


def nominal_fine_bin(pilot_hz: float, centre_hz: float, cell_hz: float, descending: bool) -> int:
    """Return the fine bin of an on-frequency pilot in its target cell."""
    cells = (pilot_hz - centre_hz) / cell_hz
    residual = cells - round(cells)
    return int(round((-residual if descending else residual) * FINE_BINS)) % FINE_BINS


def default_rank(bulk: np.ndarray) -> int:
    """Return the zero-based median rank of the reference bins."""
    return (int(np.count_nonzero(bulk)) - 1) // 2


def _exact_word(fine: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int, decision_version: str) -> int:
    _, required = decision_functions(decision_version)
    result = required(
        np.ascontiguousarray(fine), anchor_bin=anchor_bin, designated_half_width=SEARCH_HALF_WIDTH,
        bulk_mask=bulk, cfar_rank=rank)
    return -1 if not result.valid or result.multiplier_q16 is None else min(int(result.multiplier_q16), MAX_SCORE)


def _float_words(fines: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int,
                 decision_version: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the keep words of a decision version from float powers, and the frames too close to call."""
    unchecked = {FINE_DECISION_V2_VERSION: _float_words_v2_unchecked,
                 FINE_DECISION_V3_VERSION: _float_words_v3_unchecked}.get(decision_version, _float_words_unchecked)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return unchecked(fines, anchor_bin=anchor_bin, bulk=bulk, rank=rank)


def _float_words_unchecked(fines: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    num = 2.0 * fines[:, 0].astype(np.float64)
    den = fines[:, 1].astype(np.float64) + fines[:, 2].astype(np.float64)
    usable = den > 0
    ratio = np.where(usable, num / np.where(usable, den, 1.0), np.inf)
    background = np.sort(ratio[:, np.flatnonzero(bulk)], axis=1)
    count = np.count_nonzero(usable[:, bulk], axis=1)
    valid = rank < count
    r = background[:, rank]
    # a neighbour within the margin could be the exact rank value
    close = np.zeros(len(fines), bool)
    if rank > 0:
        close |= background[:, rank] - background[:, rank - 1] <= FLOAT_MARGIN * r
    if rank + 1 < background.shape[1]:
        close |= background[:, rank + 1] - r <= FLOAT_MARGIN * r

    search = (anchor_bin + np.arange(-SEARCH_HALF_WIDTH, SEARCH_HALF_WIDTH + 1)) % FINE_BINS
    fires = usable[:, search] & (num[:, search] > 0)
    x = np.where(fires, ratio[:, search], 0.0).max(axis=1) * Q16 / r
    always = fires.any(axis=1) & (r == 0)
    # every word from 2**63 up is reported as MAX_SCORE
    huge = always | (x > 2.0**63 * (1 + FLOAT_MARGIN))
    x = np.where(huge | ~fires.any(axis=1), 1.0, x)
    # the ceiling of a value this close to an integer needs the exact products
    close |= ~huge & fires.any(axis=1) & (np.abs(x - np.rint(x)) <= FLOAT_MARGIN * x)
    words = np.maximum(np.ceil(x), 1.0)
    huge |= words >= 2.0**63
    out = np.where(huge, MAX_SCORE, np.where(huge, 1.0, words).astype(np.int64))
    out = np.where(valid, out, -1)
    return out, close & valid


def _float_words_v2_unchecked(fines: np.ndarray, *, anchor_bin: int, bulk: np.ndarray,
                              rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Version 2: the largest target in the search over the reference of rank `rank` in the background."""
    num = 2.0 * fines[:, 0].astype(np.float64)
    den = fines[:, 1].astype(np.float64) + fines[:, 2].astype(np.float64)
    usable = den > 0
    # a background bin with no reference power holds no rank
    background = np.sort(np.where(usable, den, np.inf)[:, np.flatnonzero(bulk)], axis=1)
    valid = rank < np.count_nonzero(usable[:, bulk], axis=1)
    r = background[:, rank]
    close = np.zeros(len(fines), bool)
    if rank > 0:
        close |= background[:, rank] - background[:, rank - 1] <= FLOAT_MARGIN * r
    if rank + 1 < background.shape[1]:
        close |= background[:, rank + 1] - r <= FLOAT_MARGIN * r

    search = (anchor_bin + np.arange(-SEARCH_HALF_WIDTH, SEARCH_HALF_WIDTH + 1)) % FINE_BINS
    x = num[:, search].max(axis=1) * Q16 / np.where(valid, r, 1.0)
    huge = x > 2.0**63 * (1 + FLOAT_MARGIN)
    # the ceiling of a value this close to an integer needs the exact products
    close |= ~huge & (np.abs(x - np.rint(x)) <= FLOAT_MARGIN * x)
    words = np.maximum(np.ceil(np.where(huge, 1.0, x)), 1.0)
    huge |= words >= 2.0**63
    out = np.where(huge, MAX_SCORE, np.where(huge, 1.0, words).astype(np.int64))
    out = np.where(valid, out, -1)
    return out, close & valid


def _float_words_v3_unchecked(fines: np.ndarray, *, anchor_bin: int, bulk: np.ndarray,
                              rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Version 3: the largest target of all 256 bins over the median reference of the even bins.

    Every bin is searched and the median takes the even bins at rank 63, zeros included, so `anchor_bin`,
    `bulk` and `rank` are not read. A frame whose median reference is zero has no decision.
    """
    even, rank = v3_geometry()
    num = 2.0 * fines[:, 0].astype(np.float64)
    den = fines[:, 1].astype(np.float64) + fines[:, 2].astype(np.float64)
    background = np.sort(den[:, np.flatnonzero(even)], axis=1)
    r = background[:, rank]
    valid = r > 0
    close = (background[:, rank] - background[:, rank - 1] <= FLOAT_MARGIN * r)
    close |= background[:, rank + 1] - r <= FLOAT_MARGIN * r

    x = num.max(axis=1) * Q16 / np.where(valid, r, 1.0)
    huge = x > 2.0**63 * (1 + FLOAT_MARGIN)
    # the ceiling of a value this close to an integer needs the exact products
    close |= ~huge & (np.abs(x - np.rint(x)) <= FLOAT_MARGIN * x)
    words = np.maximum(np.ceil(np.where(huge, 1.0, x)), 1.0)
    huge |= words >= 2.0**63
    out = np.where(huge, MAX_SCORE, np.where(huge, 1.0, words).astype(np.int64))
    out = np.where(valid, out, -1)
    return out, close & valid


def required_words(fines: Sequence[np.ndarray] | np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int,
                   decision_version: str) -> np.ndarray:
    """Return each frame's smallest keeping word, -1 where the rank has too little support.

    Equal to the exact keep boundary of the decision version on every frame. Float powers score
    most frames, and a frame whose rank value or word falls within FLOAT_MARGIN of a tie or an
    integer is scored exactly.
    """
    decision_functions(decision_version)
    bulk = np.asarray(bulk, dtype=bool)
    words = np.empty(len(fines), dtype=np.int64)
    for lo in range(0, len(fines), WORD_CHUNK):
        chunk = np.asarray(fines[lo:lo + WORD_CHUNK], dtype=np.uint64).reshape(-1, 3, FINE_BINS)
        part, close = _float_words(chunk, anchor_bin=anchor_bin, bulk=bulk, rank=rank,
                                   decision_version=decision_version)
        for i in np.flatnonzero(close):
            part[i] = _exact_word(chunk[i], anchor_bin=anchor_bin, bulk=bulk, rank=rank,
                                  decision_version=decision_version)
        words[lo:lo + len(chunk)] = part
    return words


def capture_weights(captures: np.ndarray) -> np.ndarray:
    """Return frame weights that give every capture equal total weight."""
    _, inverse, counts = np.unique(captures, return_inverse=True, return_counts=True)
    return 1.0 / (counts.size * counts[inverse])


def knee_word(words: np.ndarray, weights: np.ndarray) -> int:
    """Return the keep word of largest height above the chord of the kept fraction on a log word axis."""
    words = np.maximum(np.asarray(words, dtype=np.float64), 1.0)
    order = np.argsort(words, kind="stable")
    words = words[order]
    kept = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    kept /= kept[-1]
    last = np.r_[words[1:] != words[:-1], True]
    words, kept = words[last], kept[last]
    if words.size == 1:
        return int(words[0])
    height = kept - np.log(words / words[0]) / np.log(words[-1] / words[0])
    # the larger word wins a tie
    return int(words[words.size - 1 - int(np.argmax(height[::-1]))])


def bootstrap_captures(words: np.ndarray, captures: np.ndarray, seed: int = 20261106) -> dict[str, list[float]]:
    """Return the 68% knee interval over resampled captures."""
    rng = np.random.default_rng(seed)
    groups = [words[captures == c] for c in np.unique(captures)]
    knees = []
    for _ in range(BOOTSTRAP_REPLICATES):
        picks = rng.integers(len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in picks])
        ids = np.concatenate([np.full(groups[i].size, n) for n, i in enumerate(picks)])
        knees.append(knee_word(sample, capture_weights(ids)))
    return dict(knee_interval_q16=[float(q) for q in np.quantile(knees, [0.16, 0.84])])


def hourly_masked(masked: np.ndarray, bins: np.ndarray) -> dict[str, float | int]:
    """Return the frame-weighted mean and the standard deviation of the masked fractions of the bins.

    `masked` is True for each frame with a decision that the word masks, and `bins` holds the bin of each frame,
    its UTC hour or its capture. The standard deviation is that of the bins' fractions (population, ddof 0).
    """
    masked = np.asarray(masked, dtype=np.float64)
    if not masked.size:
        raise ValueError("no frame has a decision")
    _, inverse, counts = np.unique(np.asarray(bins), return_inverse=True, return_counts=True)
    fractions = np.bincount(inverse, weights=masked, minlength=counts.size) / counts
    return dict(mean=float(np.average(fractions, weights=counts)), std=float(fractions.std()), bins=int(counts.size))


def bias_within_std(calibration: dict[str, Any], test_mean: float) -> tuple[bool, str]:
    """Return whether a test masked fraction agrees with its calibration, and why.

    `calibration` is a record of hourly_masked. bias = test mean - calibration mean, and the check passes when
    |bias| < std, the hour-to-hour spread of the calibration, the rule of the dissertation's BAO test. With std 0
    it passes only on a bias of 0. A calibration of fewer than two bins has no spread, so the check fails.
    """
    unit = "hour" if calibration.get("bin", "hour") == "hour" else "capture"
    count, mean, std = int(calibration["bins"]), float(calibration["mean"]), float(calibration["std"])
    if count < 2:
        return False, f"calibration has {_count(count, unit)}, so no hour-to-hour spread; calibrate on more hours"
    bias = test_mean - mean
    ratio = abs(bias) / std if std > 0 else (0.0 if bias == 0 else math.inf)
    return (abs(bias) < std or bias == 0,
            f"calibration {mean:.4f}, std {std:.4f} over {_count(count, unit)}; bias/std {ratio:.2f}")


def _count(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


def _masked_record(masked: np.ndarray, bins: np.ndarray, word: int, binning: str) -> dict[str, Any]:
    """Return the hourly masked fractions at a word, as calibrate-bundle records them for test-bundle."""
    return dict(word_q16=int(word), **hourly_masked(masked, bins), bin=binning)


def _agreement(masked: np.ndarray, bins: np.ndarray, binning: str, calibration: dict[str, Any], word: int,
               label: str) -> tuple[bool, str]:
    """Return the check of a test's masked fraction at a word against the calibration's hourly fractions."""
    if int(calibration["word_q16"]) != word:
        return False, f"calibration gave its masked fractions at word {calibration['word_q16']}, not {word}; recalibrate"
    if not masked.size:
        return False, "no test frame has a decision"
    test = hourly_masked(masked, bins)
    passed, verdict = bias_within_std(calibration, test["mean"])
    unit = "hour" if binning == "hour" else "capture"
    return passed, f"masked {test['mean']:.4f} of {masked.size} {label} over {_count(test['bins'], 'test ' + unit)}; {verdict}"


def null_scores(fines: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int,
                decision_version: str) -> np.ndarray:
    """Return Q16 null scores at pseudo-searches over the bins clear of the real search of each frame.

    Version 1 scores the bin ratio against the ratio of rank `rank` in the background, and version 2
    scores the target alone against the reference of that rank. Version 3 searches every bin, so it
    gives one score per frame, the largest target over the median reference of the even bins, and reads
    neither `anchor_bin`, `bulk` nor `rank`.
    """
    decision_functions(decision_version)
    if decision_version == FINE_DECISION_V3_VERSION:
        bulk, rank = v3_geometry()
    bins = np.flatnonzero(bulk)
    if decision_version == FINE_DECISION_V3_VERSION:
        fines = np.asarray(fines, dtype=np.float64)
        ratios = 2.0 * fines[:, 0]
        reference = np.sort((fines[:, 1] + fines[:, 2])[:, bins], axis=1)[:, rank]
        # v3 searches every bin, so each frame gives one null score; one with a zero median gives none
        good = reference > 0
        return ratios[good].max(axis=1) / reference[good] * Q16
    elif decision_version == FINE_DECISION_V2_VERSION:
        fines = np.asarray(fines, dtype=np.float64)
        ratios = 2.0 * fines[:, 0]
        den = fines[:, 1] + fines[:, 2]
        reference = np.sort(np.where(den > 0, den, np.inf)[:, bins], axis=1)[:, rank]
    else:
        ratios = fine_ratio(fines)
        reference = np.sort(ratios[:, bins], axis=1)[:, rank]
    width = 2 * SEARCH_HALF_WIDTH + 1
    # overlapping, so a short control fills the tail, but only clear_searches() of them are independent
    markers = (anchor_bin + np.arange(width, FINE_BINS - width + 1, 2)) % FINE_BINS
    near = np.arange(-SEARCH_HALF_WIDTH, SEARCH_HALF_WIDTH + 1)
    peaks = ratios[:, (markers[:, None] + near[None, :]) % FINE_BINS].max(axis=2)
    return (peaks / reference[:, None] * Q16).ravel()


def clear_searches() -> int:
    """Return the independent pseudo-searches in one frame."""
    return FINE_BINS // (2 * SEARCH_HALF_WIDTH + 1) - 1


def simulated_null_scores(num_inputs: int, frames: int = MIN_SIMULATED_FRAMES, seed: int = 20261105, *,
                          decision_version: str) -> np.ndarray:
    """Return null scores for white noise summed over num_inputs streams."""
    rng = np.random.default_rng(seed)
    bulk, rank = _geometry(0, (), decision_version)
    powers = np.empty((frames, 3, FINE_BINS))
    for frame in range(frames):
        z = rng.standard_normal((3, num_inputs, WINDOWS, 2), dtype=np.float32).view(np.complex64)[..., 0]
        powers[frame] = (np.abs(np.fft.fft(z, FINE_BINS, axis=2)) ** 2).sum(axis=1)
    return null_scores(powers, anchor_bin=0, bulk=bulk, rank=rank, decision_version=decision_version)


def _geometry(anchor: int, excluded: Sequence[int], decision_version: str) -> tuple[np.ndarray, int]:
    """Return the reference bins and rank: the guarded background for v1 and v2, the even bins for v3."""
    if decision_version == FINE_DECISION_V3_VERSION:
        return v3_geometry()
    bulk = _bulk(anchor, excluded)
    return bulk, default_rank(bulk)


def _bulk(anchor: int, excluded: Sequence[int] = ()) -> np.ndarray:
    return bulk_mask(anchor, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS, census_excluded_bins=excluded,
                     designated_half_width=SEARCH_HALF_WIDTH)


def _weights(row: dict[str, Any], bank: bytes) -> bytes:
    offset = int(row["weight_bank_offset_bytes"])
    return bank[offset:offset + int(row["weight_bank_nbytes"])]


def weight_norms(weights: bytes) -> tuple[int, int]:
    """Return the exact squared norms (|w_t|^2, |w_lo|^2 + |w_up|^2) of a channel's packed weights.

    Their null level, mu_0 = 2 |w_t|^2 / (|w_lo|^2 + |w_up|^2), is null_power_ratio_from_weight_norms(*norms).
    """
    target, lower, upper = weight_term_norms_sq(np.frombuffer(weights, dtype=np.int8).reshape(3, -1))
    return target, lower + upper


def control_quantile_q16(words: np.ndarray, quantile: float) -> float:
    """Return the quantile of the control's exact keep words, the v3 control word before it is rounded up.

    Only frames with a decision count: a word of -1 (no decision) or MAX_SCORE (masked at every word) is left out.
    """
    words = np.asarray(words, dtype=np.int64)
    return float(np.quantile(words[(words >= 0) & (words < MAX_SCORE)].astype(np.float64), quantile))


def scaled_word(quantile_q16: float, scale: float) -> int:
    """Return the default word of fine decision v3: the control quantile times the channel scale, rounded up to Q16."""
    return int(math.ceil(quantile_q16 / Q16 * scale * Q16))


def control_words_at(words: np.ndarray, scale: float) -> np.ndarray:
    """Return the control keep word that a channel word maps to, floor(u / lambda_c), as the v3 default word scales it.

    A control frame is kept at the channel word u when its keep word is at most floor(u / lambda_c), so the default
    word and the control corrections of the power-limit test map the control onto the channel the same way.
    """
    words = np.asarray(words, dtype=np.int64)
    if scale == 1.0:
        return words
    return np.floor(words.astype(np.float64) / scale).astype(np.int64)


def _passband_factors(plan: dict[str, Any]) -> dict[int, float]:
    """Return the passband factors g = 1 + o/C of the plan's "default_word" block, by physical channel."""
    given = plan.get("default_word", {})
    unknown = set(given) - {"passband_factor", "night"}
    if unknown:
        raise ValueError(f"unknown default_word keys {sorted(unknown)}")
    factors = {int(c): float(v) for c, v in given.get("passband_factor", {}).items()}
    if any(not np.isfinite(v) or v <= 0 for v in factors.values()):
        raise ValueError("every passband factor must be finite and positive")
    return factors


# the coarse-channel sample of the CHIME F-engine, 1/390625 s, which steps the frames of a baseband capture
BASEBAND_SAMPLE_SECONDS = 1.0 / 390625.0
# test-bundle compares masked fractions over bins of one UTC hour of frame time
HOUR_SECONDS = 3600


class _NoFrameTime(ValueError):
    """A capture that records no frame time."""


def _night_site(plan: dict[str, Any]) -> tuple[float, float] | None:
    """Return the site (latitude, east longitude in degrees) of the plan's "default_word": {"night": ...}, or None.

    With it, the control quantile of the default word is taken over the control frames recorded with the Sun below
    the horizon at the site (sun.py), the data that CHIME's 21 cm analyses use.
    """
    night = plan.get("default_word", {}).get("night")
    if night is None:
        return None
    if not isinstance(night, dict) or set(night) != {"latitude_deg", "longitude_deg_east"}:
        raise ValueError('default_word night takes {"latitude_deg": ..., "longitude_deg_east": ...}')
    latitude, longitude = float(night["latitude_deg"]), float(night["longitude_deg_east"])
    if not (np.isfinite(latitude) and -90.0 <= latitude <= 90.0):
        raise ValueError("the night latitude must lie between -90 and 90 degrees")
    if not (np.isfinite(longitude) and -180.0 <= longitude <= 360.0):
        raise ValueError("the night longitude must lie between -180 and 360 degrees east")
    return latitude, longitude


def _frame_times(paths: Sequence[str], capture_format: str, block_samples: int, counts: np.ndarray) -> np.ndarray:
    """Return the Unix time in seconds of each frame that _load_fines returns for the same paths.

    counts[i] is the number of frames _load_fines took from the i-th expanded capture. A dtv_fine_powers window
    gives all its frames the start time of the window (fpga_seq_time_nsec), within 43 s of each; a chime_baseband
    capture steps from time0_ctime by its block. A chord_baseband capture records no time, so it is refused, as is
    a capture without its time attribute (_NoFrameTime).
    """
    import h5py

    from pilot_proxy.run_files import read_attrs

    times = []
    for index, path in enumerate(_expand(paths)):
        n = int(counts[index]) if index < len(counts) else 0
        if not n:
            continue
        if capture_format == "dtv_fine_powers":
            start_ns = int(read_attrs(path).get("fpga_seq_time_nsec", -1))
            if start_ns < 0:
                raise _NoFrameTime(f"{path} records no fpga_seq_time_nsec, so its frames have no time")
            times.append(np.full(n, start_ns / 1e9))
        elif capture_format == "chime_baseband":
            with h5py.File(path, "r") as handle:
                if "time0_ctime" not in handle.attrs:
                    raise _NoFrameTime(f"{path} records no time0_ctime, so its frames have no time")
                t0 = float(handle.attrs["time0_ctime"])
                start = (-int(handle.attrs["time0_fpga_count"])) % block_samples
            times.append(t0 + (start + np.arange(n) * block_samples) * BASEBAND_SAMPLE_SECONDS)
        else:
            raise _NoFrameTime(f"{capture_format} captures record no frame time")
    return np.concatenate(times) if times else np.empty(0)


def _capture_times(paths: Sequence[str], capture_format: str, block_samples: int, captures: np.ndarray) -> np.ndarray:
    """Return the Unix time of each frame of _load_fines(paths), whose capture indices are `captures`."""
    counts = np.bincount(captures, minlength=len(_expand(paths)))
    times = _frame_times(paths, capture_format, block_samples, counts)
    if len(times) != len(captures):
        raise ValueError("the frame times do not match the frames read")
    return times


def _night_frames(paths: Sequence[str], capture_format: str, block_samples: int, captures: np.ndarray,
                  site: tuple[float, float]) -> np.ndarray:
    """Return True for each frame of _load_fines(paths) recorded with the Sun below the horizon at the site."""
    return is_night(_capture_times(paths, capture_format, block_samples, captures), *site)


def _hour_bins(paths: Sequence[str], capture_format: str, block_samples: int,
               captures: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the bin of each frame of _load_fines(paths) for the masked fractions of test-bundle, and its kind.

    A bin is the UTC hour of the frame's time ("hour"). Where the captures record no frame time, each capture is
    a bin, and the kind says why.
    """
    try:
        times = _capture_times(paths, capture_format, block_samples, captures)
    except _NoFrameTime as missing:
        return np.asarray(captures), f"capture: {missing}"
    return np.floor(times / HOUR_SECONDS).astype(np.int64), "hour"


def _expand(paths: Sequence[str]) -> list[str]:
    """Return the paths with any glob patterns expanded, in sorted order, and combined run files as their windows."""
    from pilot_proxy.run_files import windows

    found = []
    for path in paths:
        for match in (sorted(glob.glob(path, recursive=True)) if glob.has_magic(path) else [path]):
            found += windows(match)
    return found


# bytes read from each end of a capture file for its identity
FINGERPRINT_BYTES = 1 << 20


def capture_identity(spec: str, capture_format: str | None) -> str:
    """Return the identity of one capture, so the same data under another name or in another layout is one capture.

    A recorded window (dtv_fine_powers) is named by the time and sequence number of its first block and by its
    frequencies, which a per-window, a combined and a several-frame file give alike. Any other capture, or a window
    with no sequence number, is named by its file's size and the SHA-256 of the file's first and last MiB (and its
    window index), so a copy or a hard link has the identity of its source.
    """
    import hashlib

    from pilot_proxy.run_files import read_attrs, split

    file, index = split(spec)
    if capture_format == "dtv_fine_powers":
        attrs = read_attrs(spec)
        if "fpga_seq_num" in attrs:
            freqs = ",".join(str(int(c)) for c in np.ravel(attrs.get("coarse_freq", [])))
            return f"window {int(attrs.get('fpga_seq_time_nsec', -1))} {int(attrs['fpga_seq_num'])} {freqs}"
    path = Path(file)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(FINGERPRINT_BYTES))
        if size > FINGERPRINT_BYTES:
            handle.seek(max(FINGERPRINT_BYTES, size - FINGERPRINT_BYTES))
            digest.update(handle.read())
    return f"file {size} {digest.hexdigest()}" + ("" if index is None else f" #{index}")


def _load_fines(paths: Sequence[str], capture_format: str, row: dict[str, Any], bank: bytes, k: int,
                time_reverse: bool, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the uint64 [N, 3, 256] fine powers of every frame and the capture index of each."""
    weights = _weights(row, bank)
    channel_id = int(row["receiver_channel_id"])
    parts, captures = [], []
    for index, path in enumerate(_expand(paths)):
        if capture_format == "dtv_fine_powers":
            frames = run_rows(path, channel_id, max_frames)
        else:
            frames = np.asarray([fine_powers(block, weights, k, time_reverse)
                                 for block in iter_blocks(path, capture_format, channel_id, k * WINDOWS, max_frames)],
                                dtype=np.uint64).reshape(-1, 3, FINE_BINS)
        parts.append(frames)
        captures.append(np.full(len(frames), index))
    fines = np.concatenate(parts) if parts else np.empty((0, 3, FINE_BINS), np.uint64)
    if not len(fines):
        raise ValueError(f"no complete blocks for channel {row['physical_channel']}")
    return fines, np.concatenate(captures)


def _export(plan: dict[str, Any], base: Path, channels: Sequence[int], out: Path) -> tuple[dict, bytes]:
    export_runtime_weight_bundle(
        receiver_profile_path=base / plan["receiver_profile"],
        detector_core_profile_path=base / plan["detector_core_profile"],
        physical_channels=list(channels),
        weight_coordinate_system=plan.get("weight_coordinate_system", "post_nyquist_zone_normalization"),
        output_dir=out,
    )
    return json.loads((out / "pilot_profiles.json").read_text()), (out / "weights.bin").read_bytes()


def _receiver(plan: dict[str, Any], base: Path) -> tuple[int, float, bool]:
    profile = json.loads((base / plan["receiver_profile"]).read_text())
    adapter = profile["detector_adapter"]
    order = profile["channelizer"]["frequency_axis"]["order"]
    return int(adapter["detector_window_samples"]), float(adapter["fine_bin_enbw_hz"]), order == "descending_rf"


def _rows(bundle: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["physical_channel"]): row for row in bundle["profiles"]}


def _bin_list(bulk: np.ndarray) -> list[int]:
    return [int(b) for b in np.flatnonzero(bulk)]


def _fine_values(channel: int, anchor: int, rank: int, bulk: np.ndarray, word: int, plan: dict[str, Any], quantile: float,
                 note: str, decision_version: str, power_limit_word: int | None = None) -> dict[str, Any]:
    # the control is decided and recorded, and never reaches a mask
    status = "control" if channel == CONTROL_PHYSICAL_CHANNEL else "calibrated"
    provenance = dict(epochs=[str(plan.get("epoch", "unstated"))], source_product_sha256=[],
                      null_quantile=quantile, note=f"{note} Detect-only, not a calibration.")
    # the word holds for this set of summed inputs, which Kotekan compares with the live set
    if "num_inputs" in plan:
        provenance["summed_inputs"] = int(plan["num_inputs"])
    if decision_version == FINE_DECISION_V3_VERSION:
        # v3 has no geometry (every bin searched, the even bins' median), and carries the power-limit word
        values = dict(status=status, decision_version=decision_version, cfar_multiplier_q16=word,
                      provenance=provenance)
        if power_limit_word is not None:
            values["power_limit_multiplier_q16"] = int(power_limit_word)
        return values
    return dict(status=status, decision_version=decision_version, cfar_multiplier_q16=word,
                anchor_bin=anchor, designated_half_width=SEARCH_HALF_WIDTH, cfar_rank=rank,
                bulk_mask_words_hex=[f"0x{w:016x}" for w in pack_bulk_mask(bulk)], provenance=provenance)


def _shelf(plan: dict[str, Any], base: Path) -> float:
    """Return the shelf power in one fine bin over the pilot power, C_s."""
    adapter = json.loads((base / plan["receiver_profile"]).read_text())["detector_adapter"]
    return 10 ** (float(adapter["pilot_below_data_db"]) / 10) * float(adapter["fine_bin_enbw_hz"]) / float(adapter["dtv_bandwidth_hz"])


def _limit_inputs(plan: dict[str, Any], base: Path) -> dict[str, Any]:
    """Return the response table, the checked limits and the optional corrections of a power-limit plan."""
    from pilot_proxy.characterization import threshold_rule

    given = plan.get("power_limit")
    if not isinstance(given, dict) or not {"table", "limits"} <= set(given):
        raise ValueError('the power_limit threshold needs "power_limit": {"table": ..., "limits": ...} in the plan')
    unknown = set(given) - {"table", "limits", "coupling", "curvature", "margin", "min_captures"}
    if unknown:
        raise ValueError(f"unknown power_limit keys {sorted(unknown)}")
    if "words" in plan or "word_q16" in plan:
        raise ValueError("a power_limit plan takes no words, because the test sets them")
    # no defaults: the offset difference is negative on some channels, and where an estimate clips at zero only the
    # error terms remain, so leaving them out reads a channel low, the direction that passes it
    missing = sorted({"curvature", "margin"} - set(given))
    if missing:
        raise ValueError(f"a power_limit plan needs {missing}: the offset difference (curvature) of each channel with its standard error, "
                         "and the margin, in units of the shelf estimate (0 where measured as none)")
    control_channel = {int(plan["control"]["channel"])} if plan.get("control") else set()
    uncovered = sorted({int(c) for c in plan["channels"]} - control_channel - {int(c) for c in given["curvature"]})
    if uncovered:
        raise ValueError(f"the power_limit curvature has no entry for channels {uncovered}")
    table_path, limits_path = (base / given["table"]).resolve(), (base / given["limits"]).resolve()
    table = threshold_rule.load_table(table_path)
    rows = {int(row["channel"]): row for row in table["channels"]}
    margin = float(given["margin"])
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("the margin must be finite and nonnegative")
    curvature = {int(c): (float(v[0]), float(v[1])) for c, v in given["curvature"].items()}
    return dict(
        table=table, rows=rows, limits={int(c): float(v) for c, v in json.loads(limits_path.read_text()).items()},
        coupling={int(c): float(v) for c, v in given.get("coupling", {}).items()}, curvature=curvature, margin=margin,
        min_captures=int(given.get("min_captures", MIN_KNEE_CAPTURES)), shelf=_shelf(plan, base),
        record=dict(table=str(table_path), table_sha256=_sha256_text(table_path.read_bytes()),
                    experiment=table["inputs"].get("experiment"), limits=str(limits_path),
                    limits_sha256=_sha256_text(limits_path.read_bytes()), margin=margin,
                    selection=SELECTION, evaluation=EVALUATION))


def _limit_channel(channel: int, fines: np.ndarray, captures: np.ndarray, seen: np.ndarray, control, *, anchor: int,
                 bulk: np.ndarray, rank: int, inputs: dict[str, Any], decision_version: str,
                 norms: tuple[int, int], control_scale: float = 1.0) -> dict[str, Any]:
    """Return the power-limit test of one channel. The control is read with no corrections, as `control=None`.

    `control` holds the fine powers, captures and weight norms of the control channel, and `norms` are the weight
    norms of this one. The excess of each is the ratio F / mu_0 - 1 at its own null level. Under v3 the control is
    read at floor(u / lambda_c), where lambda_c = `control_scale` is the scale of the default word.
    """
    from pilot_proxy.characterization import threshold_rule

    if channel not in inputs["rows"] or channel not in inputs["limits"]:
        raise ValueError(f"channel {channel} needs a row in the response table and a checked power limit")
    shelf = inputs["shelf"]
    excess = ratio_excess(fines, norms)
    good = (seen >= 0) & np.isfinite(excess)
    sweep = threshold_rule.sweep(threshold_rule.ChannelFrames(
        channel, np.where(good, seen, 0).astype(np.uint64), capture_weights(captures), good, captures,
        np.where(good, excess, 0.0)), shelf=shelf)
    offset, shift, margin, measured = 0.0, 0.0, 0.0, None
    if control is not None:
        control_fines, control_captures, control_norms = control
        words = required_words(control_fines, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision_version)
        control_excess = ratio_excess(control_fines, control_norms)
        # decided frames only, as in the dissertation: a frame masked at every word (MAX_SCORE) is no null sample
        good = (words >= 0) & (words < MAX_SCORE) & np.isfinite(control_excess)
        if not good.any():
            raise ValueError(f"the control has no supported frames under the geometry of channel {channel}")
        at = control_words_at(sweep["word"].astype(np.int64), control_scale)
        measured = control_corrections(words[good], control_excess[good], control_captures[good], at)
        curvature, curvature_se = inputs["curvature"].get(channel, (0.0, 0.0))
        offset = shelf * measured["null"] + curvature
        shift = shelf * measured["shift"]
        margin = 2.0 * (shelf * (measured["shift_se"] + measured["null_se"]) + curvature_se) + inputs["margin"]
    coupling = inputs["coupling"].get(channel, 1.0)
    result = limit_test(sweep, shelf=shelf, power_scale=threshold_rule.power_scale(inputs["table"], channel),
                      power_limit=inputs["limits"][channel], coupling=coupling, offset=offset, shift=shift, margin=margin,
                      min_captures=inputs["min_captures"])
    result.update(estimate="ratio, F / mu_0 - 1", mu0=null_power_ratio_from_weight_norms(*norms),
                  coupling_source="plan" if channel in inputs["coupling"] else "worst case",
                  corrections="none, the control itself" if measured is None else dict(
                      null=shelf * measured["null"], null_se=shelf * measured["null_se"],
                      curvature=inputs["curvature"].get(channel, (0.0, 0.0))[0],
                      curvature_source="plan" if channel in inputs["curvature"] else "not given",
                      control_frames=measured["frames"], control_captures=measured["captures"],
                      control_mu0=null_power_ratio_from_weight_norms(*control_norms), control_word_scale=control_scale))
    if measured is not None and result["outcome"] == "useful":
        at = int(np.searchsorted(sweep["word"], np.uint64(result["word"])))
        result["corrections"].update(selection_shift=float(shift[at]), selection_shift_se=float(shelf * measured["shift_se"][at]))
    return result


def calibrate_bundle(*, plan_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write a detect-only commissioning bundle from a plan and return its report."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir} exists; a bundle is never changed in place")
    k, cell_hz, descending = _receiver(plan, base)
    quantile = float(plan.get("null_quantile", 0.99))
    if not 0.0 < quantile < 1.0:
        raise ValueError("null_quantile must lie between 0 and 1")
    channels = sorted(int(c) for c in plan["channels"])
    control = plan.get("control")
    if control and int(control["channel"]) not in channels:
        raise ValueError(f"the control channel {control['channel']} must also be listed under channels, "
                         "with its captures, so the bundle carries it")
    if control is None and "num_inputs" not in plan:
        raise ValueError("a plan needs a control channel or num_inputs for a simulated first word")
    threshold = plan.get("threshold", "knee")
    if threshold not in THRESHOLDS:
        raise ValueError(f"threshold must be one of {THRESHOLDS}, got {threshold!r}")
    if threshold == "power_limit" and control is None:
        raise ValueError("the power_limit threshold needs a control channel, which measures its corrections")
    inputs = _limit_inputs(plan, base) if threshold == "power_limit" else None
    fmt = plan.get("capture_format")
    max_frames = plan.get("max_frames_per_capture")
    decision = plan.get("decision_version", CALIBRATION_DECISION_VERSION)
    if decision not in FINE_DECISION_VERSIONS:
        raise ValueError(f"decision_version must be one of {FINE_DECISION_VERSIONS}, got {decision!r}")

    night_site = _night_site(plan)
    if night_site is not None and not control:
        raise ValueError("the night option of default_word takes its quantile from a control channel, and the plan has none")

    with tempfile.TemporaryDirectory() as temp:
        wanted = sorted({*channels, *([int(control["channel"])] if control else [])})
        bundle, bank = _export(plan, base, wanted, Path(temp) / "all")
        rows = _rows(bundle)
        reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])

        if control:
            row = rows[int(control["channel"])]
            anchor = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            bulk, control_rank = _geometry(anchor, (), decision)
            fines, control_captures = _load_fines(control["captures"], fmt, row, bank, k, reverse, max_frames)
            # the quantile reads the night frames only, when the plan asks; the corrections read every frame
            night = (_night_frames(control["captures"], fmt, k * WINDOWS, control_captures, night_site)
                     if night_site is not None else np.ones(len(fines), dtype=bool))
            if not night.any():
                raise ValueError("no control frame was recorded with the Sun below the horizon at the night site")
            quiet = fines[night]
            # test-bundle compares the control's masked fraction hour by hour, over the same frames
            control_bins, control_binning = _hour_bins(control["captures"], fmt, k * WINDOWS, control_captures)
            control_bins = control_bins[night]
            control_words = required_words(quiet, anchor_bin=anchor, bulk=bulk, rank=control_rank,
                                           decision_version=decision)
            control_weights = _weights(row, bank)
            control_norms = weight_norms(control_weights)
            # the corrections of the power-limit test read the control again under the words of each channel
            control_frames = (fines, control_captures, control_norms) if inputs else None
            # in chunks, so the float ratios of a long control never sit in memory at once
            scores = np.concatenate([
                null_scores(quiet[lo:lo + NULL_CHUNK], anchor_bin=anchor, bulk=bulk, rank=control_rank,
                            decision_version=decision)
                for lo in range(0, len(quiet), NULL_CHUNK)])
            control_record = dict(channel=int(control["channel"]), receiver_channel_id=int(row["receiver_channel_id"]),
                                  anchor_bin=anchor, bulk_bins=_bin_list(bulk), cfar_rank=control_rank,
                                  weights_hex=control_weights.hex(), frames=len(fines),
                                  captures=[str(Path(p).resolve()) for p in _expand(control["captures"])],
                                  capture_ids=sorted({capture_identity(p, fmt) for p in _expand(control["captures"])}))
            if night_site is not None:
                control_record["night"] = dict(latitude_deg=night_site[0], longitude_deg_east=night_site[1],
                                               frames=int(night.sum()))
            source = "control channel" if night_site is None else "control channel, night frames"
        else:
            scores = simulated_null_scores(int(plan["num_inputs"]), decision_version=decision)
            control_record = None
            source = f"white noise, {int(plan['num_inputs'])} inputs"
        # v3 searches every bin, so a frame holds one independent null score, not clear_searches()
        per_frame = 1 if decision == FINE_DECISION_V3_VERSION else clear_searches()
        independent = (len(quiet) if control else MIN_SIMULATED_FRAMES) * per_frame
        if independent * (1.0 - quantile) < 1:
            raise ValueError(f"{independent} independent null scores are too few for quantile {quantile}")
        v3 = decision == FINE_DECISION_V3_VERSION
        factors = _passband_factors(plan) if v3 else {}
        if v3 and control:
            # the quantile of the exact keep words of the control frames that have a word
            quantile_q16 = control_quantile_q16(control_words, quantile)
            reference_level = null_power_ratio_from_weight_norms(*control_norms) * factors.get(int(control["channel"]), 1.0)
        elif v3:
            # white noise through ideal weights has the null level 1
            quantile_q16, reference_level = float(np.quantile(scores, quantile)), 1.0
        measured_word = int(math.ceil(quantile_q16)) if v3 else int(math.ceil(float(np.quantile(scores, quantile))))
        floor = int(plan.get("word_q16", measured_word))
        if "word_q16" in plan:
            source = "the plan"
        if control:
            decided = control_words >= 0
            control_record["masked_by_hour"] = _masked_record(control_words[decided] > floor, control_bins[decided],
                                                              floor, control_binning)
        min_captures = int(plan.get("min_knee_captures", MIN_KNEE_CAPTURES))

        report_channels = {}
        values = {}
        for channel in channels:
            row = rows[channel]
            norms = weight_norms(_weights(row, bank))
            nominal = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            paths = plan["channels"][str(channel)]
            excluded = plan.get("exclude_bins", {}).get(str(channel), [])
            scaling = {}
            if v3:
                # v3's default word: the control's 1% word scaled to the null level of this channel
                level = null_power_ratio_from_weight_norms(*norms)
                factor = factors.get(channel, 1.0)
                scale = level * factor / reference_level
                default = floor if "word_q16" in plan else scaled_word(quantile_q16, scale)
                scaling = dict(mu0=level, passband_factor=factor,
                               passband_factor_source="plan" if channel in factors else "not given", scale=scale)
            if not paths:
                # a first bundle, so the detector can record the data for the next one
                anchor = int(plan.get("anchors", {}).get(str(channel), nominal))
                bulk, rank = _geometry(anchor, excluded, decision)
                given_word = plan.get("words", {}).get(str(channel))
                if v3:
                    # the v3 default word needs no captures of the channel
                    word = int(default if given_word is None else given_word)
                    word_source = "plan" if given_word is not None or "word_q16" in plan else "control word x scale"
                else:
                    word, word_source = floor, "floor, no captures"
                note = (f"Word from {word_source}, control word from {source}." if v3
                        else f"Word from the floor, no captures, floor from {source}.")
                is_control = bool(control) and channel == int(control["channel"])
                # no captures, so no passing word: the decision masks the channel whole, and under v3 so does the
                # bundle, through the reject word
                masked_whole = bool(inputs) and not is_control and channel in inputs["rows"] and channel in inputs["limits"]
                if masked_whole and v3:
                    note += f" Reject word {REJECT_WORD_Q16}, masked whole (power limit: no captures)."
                values[channel] = _fine_values(channel, anchor, rank, bulk, word, plan, quantile, note, decision,
                                               power_limit_word=REJECT_WORD_Q16 if masked_whole and v3 else None)
                report_channels[str(channel)] = dict(
                    anchor_bin=anchor, anchor_source="nominal", nominal_bin=nominal, frames=0, captures=0,
                    word_q16=word, eta=word / Q16, word_source=word_source, **scaling)
                if masked_whole:
                    from pilot_proxy.characterization import threshold_rule
                    report_channels[str(channel)].update(
                        power_limit=dict(outcome="not useful", reason="no captures",
                                         power_limit=inputs["limits"][channel],
                                         power_scale=threshold_rule.power_scale(inputs["table"], channel),
                                         coupling=inputs["coupling"].get(channel, 1.0)),
                        **({"power_limit_word_q16": REJECT_WORD_Q16,
                            "power_limit_word_source": "reject, power limit: no captures"} if v3 else {}))
                continue
            fines, captures = _load_fines(paths, fmt, row, bank, k, reverse, max_frames)
            window = (nominal + np.arange(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)) % FINE_BINS
            # only the window, so the ratios stay small
            upper = np.quantile(fine_ratio(fines[:, :, window]), 0.9, axis=0)
            anchor = int(window[np.argmax(upper)])
            given = plan.get("anchors", {}).get(str(channel))
            bulk, rank = _geometry(anchor, excluded, decision)
            seen = required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision)
            # the pilot is seen when more frames than the null allows pass the channel's own null word: the floor,
            # or under v3 the default word, the control word scaled to this channel. A v3 keep word takes every bin,
            # so a line anywhere in the channel counts as well.
            gate = default if v3 else floor
            hits = int(np.count_nonzero(seen > gate))
            fraction = hits / seen.size
            measured = binomial_upper_tail(hits, seen.size, 1.0 - quantile) < MAX_LINE_TAIL
            peak = float(np.median(fine_ratio(fines[:, :, [anchor]])))
            origin = "measured" if measured else "nominal"
            if given is not None or not measured:
                # noise peak in the window, so fall back to the expected bin
                anchor = nominal if given is None else int(given)
                origin = "nominal" if given is None else "plan"
                bulk, rank = _geometry(anchor, excluded, decision)
                if not v3:  # v3 words do not depend on the anchor
                    seen = required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision)

            supported = seen >= 0
            if not supported.any():
                raise ValueError(f"channel {channel} has no supported frames")
            knee = None if v3 else knee_word(seen[supported], capture_weights(captures)[supported])
            weights = capture_weights(captures[supported])
            at_floor = float(np.sum(weights[seen[supported] > floor]))
            # share of frames with a pilot, less the clean frames the channel's null word masks
            at_gate = at_floor if gate == floor else float(np.sum(weights[seen[supported] > gate]))
            share = (at_gate - (1.0 - quantile)) / quantile
            given_word = plan.get("words", {}).get(str(channel))
            count = int(np.unique(captures).size)
            is_control = bool(control) and channel == int(control["channel"])
            tested = None
            if inputs and (not is_control or (channel in inputs["rows"] and channel in inputs["limits"])):
                tested = _limit_channel(channel, fines, captures, seen, None if is_control else control_frames,
                                      anchor=anchor, bulk=bulk, rank=rank, inputs=inputs, decision_version=decision,
                                      norms=norms, control_scale=scaling.get("scale", 1.0))
                if origin == "nominal" and not is_control:
                    # with no pilot in the run the estimate reads only noise, which is no pass
                    tested = dict({key: value for key, value in tested.items() if key not in PASSING_FIELDS},
                                  outcome="not useful", reason=NO_PILOT_REASON)
            limited = None
            if decision == FINE_DECISION_V3_VERSION and tested is not None and not is_control:
                # v3 keeps the power-limit word apart, and a channel with no passing word carries the reject word,
                # so every consumer masks it whole, as the operating policy does
                limited = ((tested["word"], "power limit") if tested["outcome"] == "useful"
                           else (REJECT_WORD_Q16, f"reject, power limit: {tested['reason']}"))
            if given_word is not None or "word_q16" in plan:
                word, word_source = int(floor if given_word is None else given_word), "plan"
            elif v3:
                word, word_source = default, "control word x scale"
            elif tested is not None and not is_control and limited is None:
                # a channel with no passing word is masked whole by the operating policy, and holds the floor here
                word, word_source = (tested["word"], "power limit") if tested["outcome"] == "useful" else (floor, f"floor, power limit: {tested['reason']}")
            elif origin == "nominal":
                word, word_source = floor, "floor, no pilot"
            elif count < min_captures:
                word, word_source = floor, f"floor, {count} captures of {min_captures}"
            elif share > ALWAYS_ON_SHARE:
                # no clean frames for the curve to bend from, so the knee would sit in the pilot's own spread
                word, word_source = floor, "floor, pilot in nearly every frame"
            elif knee < floor:
                word, word_source = floor, "floor, knee below it"
            else:
                word, word_source = knee, "knee"
            masked = float(np.sum(weights[seen[supported] > word]))
            # the knee interval is a v1 and v2 diagnostic
            intervals = {} if v3 else bootstrap_captures(seen[supported], captures[supported])
            bins, binning = _hour_bins(paths, fmt, k * WINDOWS, captures)
            intervals["masked_by_hour"] = _masked_record(seen[supported] > word, bins[supported], word, binning)
            if limited is not None and limited[0] != word:
                # Kotekan and the relay apply the power-limit word, so test-bundle tests that word
                intervals["applied_masked_by_hour"] = _masked_record(seen[supported] > limited[0], bins[supported],
                                                                     limited[0], binning)
            note = f"Word from {word_source}, {'control word' if v3 else 'floor'} from {source}."
            if limited is not None:
                note += (f" Power-limit word from {limited[1]}." if tested["outcome"] == "useful"
                         else f" Reject word {REJECT_WORD_Q16}, masked whole ({limited[1]}).")
            values[channel] = _fine_values(channel, anchor, rank, bulk, word, plan, quantile, note, decision,
                                           power_limit_word=None if limited is None else limited[0])
            report_channels[str(channel)] = dict(
                anchor_bin=anchor, anchor_source=origin, nominal_bin=nominal, frames=len(fines), captures=count,
                word_q16=word, eta=word / Q16, word_source=word_source, **scaling,
                **({} if v3 else {"knee_word_q16": knee}),
                **({"power_limit_word_q16": limited[0], "power_limit_word_source": limited[1]} if limited else {}),
                masked_fraction=masked, masked_at_floor=at_floor, pilot_share=share,
                pilot_in_nearly_every_frame=bool(share > ALWAYS_ON_SHARE),
                strongest_bin_median_ratio=peak, strongest_bin_masked_fraction=fraction, **intervals,
                **({"power_limit": tested} if tested is not None else {}))

    bundle, _ = _export(plan, base, channels, output_dir)
    for row in bundle["profiles"]:
        calibration = values[int(row["physical_channel"])]
        calibration["provenance"]["source_product_sha256"] = [
            _sha256_text(plan_path.read_bytes())]
        row["fine_calibration"] = calibration
    (output_dir / "pilot_profiles.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    _write_sha256sums(output_dir, BUNDLE_FILES)
    validation = validate_runtime_weight_bundle(bundle_dir=output_dir)
    if validation.get("errors"):
        shutil.rmtree(output_dir)
        raise ValueError(f"bundle failed validation: {validation['errors']}")

    report = dict(threshold=threshold, decision_version=decision, floor_word_q16=floor, floor_eta=floor / Q16, measured_floor_q16=measured_word,
                  **(dict(default_word_rule="control quantile x mu0_c g_c / (mu0_ref g_ref), rounded up to Q16",
                          control_quantile_q16=quantile_q16, reference_level=reference_level) if v3 else {}),
                  null_quantile=quantile, floor_source=source, search_half_width=SEARCH_HALF_WIDTH,
                  null_scores=int(independent), channels=report_channels, control=control_record,
                  calibration_captures=sorted(str(Path(p).resolve()) for ps in plan["channels"].values()
                                             for p in _expand(ps)),
                  # test-bundle compares these, so a copy or another layout of the same data is not new test data
                  calibration_capture_ids=sorted({capture_identity(p, fmt) for ps in plan["channels"].values()
                                                  for p in _expand(ps)}))
    if inputs:
        decided = {int(c): row["power_limit"] for c, row in report_channels.items()
                   if "power_limit" in row and int(c) != int(control["channel"])}
        report["power_limit"] = dict(inputs["record"], shelf=inputs["shelf"], min_captures=inputs["min_captures"],
                             no_passing_word=sorted(c for c, t in decided.items() if t["outcome"] != "useful"))
        # the record the operating-policy compiler reads: a word, or the reason a channel has none
        write_json_strict(output_dir.with_name(output_dir.name + ".decisions.json"), dict(
            schema=DECISIONS_SCHEMA, inputs=dict(inputs["record"], plan_sha256=_sha256_text(plan_path.read_bytes())),
            channels=[dict(channel=c, outcome=t["outcome"], reason=t["reason"], deployed_word=t.get("word"),
                           kappa=t.get("kappa"), ratio=t.get("ratio"), power_limit=t["power_limit"],
                           power_scale=t["power_scale"], coupling=t["coupling"]) for c, t in sorted(decided.items())]))
    write_json_strict(output_dir.with_name(output_dir.name + ".report.json"), report)
    return report


def _sha256_text(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _binomial_log_pmf(count: int, trials: int, probability: float) -> float:
    return (math.lgamma(trials + 1) - math.lgamma(count + 1) - math.lgamma(trials - count + 1)
            + count * math.log(probability) + (trials - count) * math.log1p(-probability))


def binomial_upper_tail(hits: int, trials: int, probability: float) -> float:
    """Return P(X >= hits) for X ~ Binomial(trials, probability)."""
    if hits <= 0 or probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return 0.0
    return min(sum(math.exp(_binomial_log_pmf(c, trials, probability)) for c in range(hits, trials + 1)), 1.0)


def binomial_lower_tail(hits: int, trials: int, probability: float) -> float:
    """Return P(X <= hits) for X ~ Binomial(trials, probability)."""
    return 1.0 if hits >= trials else 1.0 - binomial_upper_tail(hits + 1, trials, probability)


def test_bundle(*, bundle_dir: Path, plan_path: Path, output_json: Path | None = None) -> dict[str, Any]:
    """Run the acceptance checks of a bundle on captures it was not made from."""
    bundle_dir = Path(bundle_dir)
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    report_path = bundle_dir.with_name(bundle_dir.name + ".report.json")
    calibration = json.loads(report_path.read_text()) if report_path.exists() else {}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(dict(check=name, passed=bool(passed), detail=detail))

    validation = validate_runtime_weight_bundle(bundle_dir=bundle_dir)
    errors = validation.get("errors", [])
    check("structure and digests", not errors, f"{len(errors)} errors")
    if errors:
        return _finish(checks, output_json)
    digest = bundle_digest(bundle_dir)

    bundle = json.loads((bundle_dir / "pilot_profiles.json").read_text())
    bank = (bundle_dir / "weights.bin").read_bytes()
    k = int(json.loads((bundle_dir / "detector_contract.json").read_text())["detector_window_samples"])
    reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])
    rows = _rows(bundle)
    fmt = plan.get("capture_format")
    max_frames = plan.get("max_frames_per_capture")
    uncalibrated = [c for c, row in rows.items() if row["fine_calibration"]["status"] not in ("calibrated", "control")]
    check("every bound channel has values", not uncalibrated, f"pending: {uncalibrated}")

    group = plan.get("channels", {})
    test_specs = [p for ps in group.values() for p in _expand(ps)] + _expand(plan.get("control", []))
    calibration_control = calibration.get("control") or {}
    paths = set(calibration.get("calibration_captures", [])) | set(calibration_control.get("captures", []))
    # a report from before the identities were recorded is compared by path alone
    ids = set(calibration.get("calibration_capture_ids", [])) | set(calibration_control.get("capture_ids", []))
    shared = {p for p in test_specs if str(Path(p).resolve()) in paths or (ids and capture_identity(p, fmt) in ids)}
    check("test data differ from calibration data", not shared, f"{len(shared)} shared captures")

    predicted = calibration.get("channels", {})
    for channel, paths in group.items():
        channel = int(channel)
        if channel not in rows:
            check(f"channel {channel}", False, "channel is not in the bundle")
            continue
        fine_cal = rows[channel]["fine_calibration"]
        bulk, rank = calibration_geometry(fine_cal)
        fines, test_captures = _load_fines(paths, fmt, rows[channel], bank, k, reverse, max_frames)
        words = required_words(fines, anchor_bin=fine_cal.get("anchor_bin", 0), bulk=bulk, rank=rank,
                               decision_version=fine_cal["decision_version"])
        valid = words >= 0
        check(f"rank support on channel {channel}", valid.all(), f"{int(valid.sum())} of {words.size} frames")
        if str(channel) not in predicted:
            check(f"channel {channel}", False, f"{report_path} has no prediction")
            continue
        if not predicted[str(channel)].get("frames"):
            # a channel calibrated with no captures has a word but no masked fraction to test against
            check(f"masked fraction on channel {channel}", False,
                  "calibration had no captures of this channel, so it predicts no masked fraction")
            continue
        # the word the stage applies: a v3 channel's power-limit word where calibration wrote one
        applied = (v3_word(fine_cal) if fine_cal["decision_version"] == FINE_DECISION_V3_VERSION
                   else int(fine_cal["cfar_multiplier_q16"]))
        limit_word = applied != int(fine_cal["cfar_multiplier_q16"])
        key = "applied_masked_by_hour" if limit_word else "masked_by_hour"
        label = f"frames at the {'power-limit' if limit_word else 'default'} word"
        if key not in predicted[str(channel)]:
            check(f"masked fraction on channel {channel}", False,
                  f"calibration predicts no hourly masked fraction for the {label}; recalibrate")
            continue
        # passes when |test mean - calibration mean| < the calibration's hour-to-hour standard deviation
        bins, binning = _hour_bins(paths, fmt, k * WINDOWS, test_captures)
        check(f"masked fraction on channel {channel}",
              *_agreement(words[valid] > applied, bins[valid], binning, predicted[str(channel)][key], applied, label))

    record = calibration.get("control")
    if plan.get("control") and record and "masked_by_hour" not in record:
        check("false alarms on the control channel", False,
              f"{report_path} has no hourly masked fraction of the control; recalibrate")
    elif plan.get("control") and record:
        row = dict(receiver_channel_id=record["receiver_channel_id"], physical_channel=record["channel"],
                   weight_bank_offset_bytes=0, weight_bank_nbytes=len(record["weights_hex"]) // 2)
        bulk = np.zeros(FINE_BINS, dtype=bool)
        bulk[record["bulk_bins"]] = True
        fines, test_captures = _load_fines(plan["control"], fmt, row, bytes.fromhex(record["weights_hex"]), k, reverse,
                                           max_frames)
        bins, binning = _hour_bins(plan["control"], fmt, k * WINDOWS, test_captures)
        label = "frames"
        night = record.get("night")
        if night:
            # a bundle calibrated on the night control is tested on the night control
            keep = _night_frames(plan["control"], fmt, k * WINDOWS, test_captures,
                                 (night["latitude_deg"], night["longitude_deg_east"]))
            fines, bins = fines[keep], bins[keep]
            label = "night frames"
        if not len(fines):
            check("false alarms on the control channel", False,
                  "the test control has no frame recorded with the Sun below the horizon, and the bundle was calibrated at night")
        else:
            # a report written before version 2 calibrated version 1
            words = required_words(fines, anchor_bin=record["anchor_bin"], bulk=bulk, rank=record["cfar_rank"],
                                   decision_version=calibration.get("decision_version", FINE_DECISION_VERSION))
            word = int(calibration["floor_word_q16"])
            decided = words >= 0
            check("false alarms on the control channel",
                  *_agreement(words[decided] > word, bins[decided], binning, record["masked_by_hour"], word, label))
    elif plan.get("control"):
        # the report may exist with no control record, when calibration simulated the control
        check("false alarms on the control channel", False,
              f"{report_path} has no control record, since calibration simulated it" if Path(report_path).exists()
              else f"{report_path} is missing")
    else:
        # without the control only the masked fractions are tested, and they pass a floor set too low
        check("false alarms on the control channel", False, "the test plan has no control")
    return _finish(checks, output_json, bundle_sha256sums=digest)


test_bundle.__test__ = False


def _finish(checks: list[dict[str, Any]], output_json: Path | None,
            bundle_sha256sums: str | None = None) -> dict[str, Any]:
    report = dict(passed=all(c["passed"] for c in checks), checks=checks)
    if bundle_sha256sums is not None:
        report["bundle_sha256sums"] = bundle_sha256sums
    if output_json is not None:
        write_json_strict(Path(output_json), report)
    return report


def bundle_digest(bundle_dir: Path) -> str:
    """Return the SHA256 of a bundle's sha256sums.txt, which names the bundle."""
    return _sha256_text((Path(bundle_dir) / "sha256sums.txt").read_bytes())


def qualify_bundle(*, bundle_dir: Path, test_report: Path, reviewer: str, output_dir: Path) -> dict[str, Any]:
    """Write a copy of a bundle whose channels are qualified for masking, and return its summary.

    A person runs this after reading a passing test report of the bundle. Each
    channel other than the control gets a mask_qualification record that names
    the report and the reviewer. Kotekan masks only with a bundle in which every
    such channel has the record.
    """
    bundle_dir, test_report, output_dir = Path(bundle_dir), Path(test_report), Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir} exists; a bundle is never changed in place")
    if not reviewer.strip():
        raise ValueError("a reviewer must be named")
    report_bytes = test_report.read_bytes()
    report = json.loads(report_bytes)
    if report.get("bundle_sha256sums") != bundle_digest(bundle_dir):
        raise ValueError(f"{test_report} is not a test of {bundle_dir}")
    if not report.get("passed"):
        failed = [c["check"] for c in report.get("checks", []) if not c["passed"]]
        raise ValueError(f"{test_report} did not pass: {failed}")
    if not any(c["check"] == "false alarms on the control channel" and c["passed"] for c in report["checks"]):
        raise ValueError(f"{test_report} has no passing control check")
    validation = validate_runtime_weight_bundle(bundle_dir=bundle_dir)
    if validation.get("errors"):
        raise ValueError(f"{bundle_dir} failed validation: {validation['errors']}")
    bundle = json.loads((bundle_dir / "pilot_profiles.json").read_text())
    record = dict(test_report_sha256=_sha256_text(report_bytes), reviewer=reviewer.strip())
    tested = {c["check"] for c in report["checks"] if c["passed"]}
    qualified = []
    for row in bundle["profiles"]:
        calibration = row["fine_calibration"]
        if calibration["status"] == "control":
            continue
        if calibration["status"] != "calibrated":
            raise ValueError(f"channel {row['physical_channel']} has status {calibration['status']}")
        # a channel the test plan left out was never tested, so it is not qualified
        if f"masked fraction on channel {row['physical_channel']}" not in tested:
            raise ValueError(f"{test_report} has no passing masked-fraction check of channel {row['physical_channel']}; "
                             "test it on captures of its own, or leave it out of the bundle")
        calibration["provenance"]["mask_qualification"] = dict(record)
        qualified.append(int(row["physical_channel"]))
    output_dir.mkdir(parents=True)
    for name in BUNDLE_FILES:
        if name != "pilot_profiles.json":
            shutil.copyfile(bundle_dir / name, output_dir / name)
    (output_dir / "pilot_profiles.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    _write_sha256sums(output_dir, BUNDLE_FILES)
    validation = validate_runtime_weight_bundle(bundle_dir=output_dir)
    if validation.get("errors"):
        shutil.rmtree(output_dir)
        raise ValueError(f"qualified bundle failed validation: {validation['errors']}")
    return dict(bundle_dir=str(output_dir), source_bundle_sha256sums=report["bundle_sha256sums"],
                bundle_sha256sums=bundle_digest(output_dir), qualified_channels=qualified, **record)
