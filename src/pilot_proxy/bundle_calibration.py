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
from pilot_proxy.detectors.narrowband_marker.anchors import WINDOW_HALF_WIDTH, bulk_mask
from pilot_proxy.fine_decision import (
    FINE_DECISION_VERSION,
    fine_required_multiplier_q16,
    pack_bulk_mask,
    unpack_bulk_mask,
)
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.runtime_bundle import (
    CONTROL_PHYSICAL_CHANNEL,
    _write_sha256sums,
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

FINE_BINS = 256
WINDOWS = 128
PAD_FACTOR = 2
GUARD_FINE_BINS = 1
Q16 = 65536
# 65 designated bins, the search of record
SEARCH_HALF_WIDTH = 32
# the fewest kept captures the archive rule accepts
MIN_KNEE_CAPTURES = 30
BOOTSTRAP_REPLICATES = 200
# chance of the hits under the null, kept small for the search over the window
MAX_LINE_TAIL = 1e-4
MIN_TAIL = 1e-3
MIN_SIMULATED_FRAMES = 2000
MAX_SCORE = (1 << 63) - 1
BUNDLE_FILES = ("detector_contract.json", "pilot_profiles.json", "weights.bin", "weights.manifest.json")


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
            labels = data.attrs.get("dish_label", [])
            dishes = [j for j in range(data.shape[3]) if not (len(labels) and str(labels[j]).startswith("RFI"))]
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
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)
    except ImportError:
        pass
    with h5py.File(path, "r") as handle:
        data = handle[next(name for name in handle if name.startswith("dtv_fine_powers"))]
        # the attribute is dropped when every row is unused (-1)
        coarse = [int(c) for c in data.attrs.get("coarse_freq", [])]
        if channel_id not in coarse:
            return
        rows = data[:max_frames, coarse.index(channel_id)]
    # zeros were not tested, and 0xFF fills a block missing from the concat window
    missing = np.iinfo(np.uint64).max
    yield from (row for row in rows if row.any() and not (row == missing).all())


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


def required_words(fines: Sequence[np.ndarray], *, anchor_bin: int, bulk: np.ndarray, rank: int) -> np.ndarray:
    """Return each frame's smallest keeping word, -1 where the rank has too little support."""
    words = []
    for fine in fines:
        result = fine_required_multiplier_q16(
            np.ascontiguousarray(fine), anchor_bin=anchor_bin, designated_half_width=SEARCH_HALF_WIDTH,
            bulk_mask=bulk, cfar_rank=rank)
        words.append(-1 if not result.valid or result.multiplier_q16 is None
                     else min(int(result.multiplier_q16), MAX_SCORE))
    return np.asarray(words, dtype=np.int64)


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


def bootstrap_captures(words: np.ndarray, captures: np.ndarray, word: int, seed: int = 20261106) -> dict[str, list[float]]:
    """Return the 68% knee interval and 95% masked-fraction interval over resampled captures."""
    rng = np.random.default_rng(seed)
    groups = [words[captures == c] for c in np.unique(captures)]
    knees, masked = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        picks = rng.integers(len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in picks])
        ids = np.concatenate([np.full(groups[i].size, n) for n, i in enumerate(picks)])
        knees.append(knee_word(sample, capture_weights(ids)))
        masked.append(float(np.sum(capture_weights(ids)[sample > word])))
    return dict(knee_interval_q16=[float(q) for q in np.quantile(knees, [0.16, 0.84])],
                masked_interval=[float(q) for q in np.quantile(masked, [0.025, 0.975])])


def null_scores(ratios: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int) -> np.ndarray:
    """Return Q16 null scores at pseudo-searches over the bins clear of the real search of each frame."""
    bins = np.flatnonzero(bulk)
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


def simulated_null_scores(num_inputs: int, frames: int = MIN_SIMULATED_FRAMES, seed: int = 20261105) -> np.ndarray:
    """Return null scores for white noise summed over num_inputs streams."""
    rng = np.random.default_rng(seed)
    bulk = _bulk(0)
    rank = default_rank(bulk)
    ratios = np.empty((frames, FINE_BINS))
    for frame in range(frames):
        z = rng.standard_normal((3, num_inputs, WINDOWS, 2), dtype=np.float32).view(np.complex64)[..., 0]
        power = np.abs(np.fft.fft(z, FINE_BINS, axis=2)) ** 2
        ratios[frame] = fine_ratio(power.sum(axis=1))
    return null_scores(ratios, anchor_bin=0, bulk=bulk, rank=rank)


def _bulk(anchor: int, excluded: Sequence[int] = ()) -> np.ndarray:
    return bulk_mask(anchor, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS, census_excluded_bins=excluded,
                     designated_half_width=SEARCH_HALF_WIDTH)


def _weights(row: dict[str, Any], bank: bytes) -> bytes:
    offset = int(row["weight_bank_offset_bytes"])
    return bank[offset:offset + int(row["weight_bank_nbytes"])]


def _expand(paths: Sequence[str]) -> list[str]:
    """Return the paths with any glob patterns expanded, in sorted order."""
    found = []
    for path in paths:
        found += sorted(glob.glob(path, recursive=True)) if glob.has_magic(path) else [path]
    return found


def _load_fines(paths: Sequence[str], capture_format: str, row: dict[str, Any], bank: bytes, k: int,
                time_reverse: bool, max_frames: int | None) -> tuple[list[np.ndarray], np.ndarray]:
    """Return the fine powers of every frame and the capture index of each."""
    weights = _weights(row, bank)
    channel_id = int(row["receiver_channel_id"])
    fines, captures = [], []
    for index, path in enumerate(_expand(paths)):
        if capture_format == "dtv_fine_powers":
            frames = iter_run_frames(path, channel_id, max_frames)
        else:
            frames = (fine_powers(block, weights, k, time_reverse)
                      for block in iter_blocks(path, capture_format, channel_id, k * WINDOWS, max_frames))
        for frame in frames:
            fines.append(frame)
            captures.append(index)
    if not fines:
        raise ValueError(f"no complete blocks for channel {row['physical_channel']}")
    return fines, np.asarray(captures)


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
                 note: str) -> dict[str, Any]:
    # the control is decided and recorded, and never reaches a mask
    status = "control" if channel == CONTROL_PHYSICAL_CHANNEL else "calibrated"
    return dict(status=status, decision_version=FINE_DECISION_VERSION, cfar_multiplier_q16=word,
                anchor_bin=anchor, designated_half_width=SEARCH_HALF_WIDTH, cfar_rank=rank,
                bulk_mask_words_hex=[f"0x{w:016x}" for w in pack_bulk_mask(bulk)],
                provenance=dict(epochs=[str(plan.get("epoch", "unstated"))], source_product_sha256=[],
                                null_quantile=quantile, note=f"{note} Detect-only, not a calibration."))


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
    if control is None and "num_inputs" not in plan:
        raise ValueError("a plan needs a control channel or num_inputs for a simulated first word")
    fmt = plan.get("capture_format")
    max_frames = plan.get("max_frames_per_capture")

    with tempfile.TemporaryDirectory() as temp:
        wanted = sorted({*channels, *([int(control["channel"])] if control else [])})
        bundle, bank = _export(plan, base, wanted, Path(temp) / "all")
        rows = _rows(bundle)
        reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])

        if control:
            row = rows[int(control["channel"])]
            anchor = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            bulk = _bulk(anchor)
            fines, _ = _load_fines(control["captures"], fmt, row, bank, k, reverse, max_frames)
            scores = null_scores(fine_ratio(np.stack(fines)), anchor_bin=anchor, bulk=bulk, rank=default_rank(bulk))
            control_record = dict(channel=int(control["channel"]), receiver_channel_id=int(row["receiver_channel_id"]),
                                  anchor_bin=anchor, bulk_bins=_bin_list(bulk), cfar_rank=default_rank(bulk),
                                  weights_hex=_weights(row, bank).hex(), frames=len(fines),
                                  captures=[str(Path(p).resolve()) for p in _expand(control["captures"])])
            source = "control channel"
        else:
            scores = simulated_null_scores(int(plan["num_inputs"]))
            control_record = None
            source = f"white noise, {int(plan['num_inputs'])} inputs"
        independent = (len(fines) if control else MIN_SIMULATED_FRAMES) * clear_searches()
        if independent * (1.0 - quantile) < 1:
            raise ValueError(f"{independent} independent null scores are too few for quantile {quantile}")
        measured_word = int(math.ceil(float(np.quantile(scores, quantile))))
        floor = int(plan.get("word_q16", measured_word))
        if "word_q16" in plan:
            source = "the plan"
        min_captures = int(plan.get("min_knee_captures", MIN_KNEE_CAPTURES))

        report_channels = {}
        values = {}
        for channel in channels:
            row = rows[channel]
            nominal = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            paths = plan["channels"][str(channel)]
            excluded = plan.get("exclude_bins", {}).get(str(channel), [])
            if not paths:
                # a first bundle, so the detector can record the data for the next one
                anchor = int(plan.get("anchors", {}).get(str(channel), nominal))
                bulk = _bulk(anchor, excluded)
                values[channel] = _fine_values(channel, anchor, default_rank(bulk), bulk, floor, plan, quantile,
                                               f"Word from the floor, no captures, floor from {source}.")
                report_channels[str(channel)] = dict(
                    anchor_bin=anchor, anchor_source="nominal", nominal_bin=nominal, frames=0, captures=0,
                    word_q16=floor, eta=floor / Q16, word_source="floor, no captures")
                continue
            fines, captures = _load_fines(paths, fmt, row, bank, k, reverse, max_frames)
            ratios = fine_ratio(np.stack(fines))
            window = (nominal + np.arange(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)) % FINE_BINS
            upper = np.quantile(ratios, 0.9, axis=0)
            anchor = int(window[np.argmax(upper[window])])
            given = plan.get("anchors", {}).get(str(channel))
            bulk = _bulk(anchor, excluded)
            rank = default_rank(bulk)
            seen = required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank)
            hits = int(np.count_nonzero(seen > floor))
            fraction = hits / seen.size
            measured = binomial_upper_tail(hits, seen.size, 1.0 - quantile) < MAX_LINE_TAIL
            peak = float(np.median(ratios[:, anchor]))
            origin = "measured" if measured else "nominal"
            if given is not None or not measured:
                # noise peak in the window, so fall back to the expected bin
                anchor = nominal if given is None else int(given)
                origin = "nominal" if given is None else "plan"
                bulk = _bulk(anchor, excluded)
                rank = default_rank(bulk)
                seen = required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank)

            supported = seen >= 0
            if not supported.any():
                raise ValueError(f"channel {channel} has no supported frames")
            knee = knee_word(seen[supported], capture_weights(captures)[supported])
            given_word = plan.get("words", {}).get(str(channel))
            count = int(np.unique(captures).size)
            if given_word is not None or "word_q16" in plan:
                word, word_source = int(floor if given_word is None else given_word), "plan"
            elif origin == "nominal":
                word, word_source = floor, "floor, no pilot"
            elif count < min_captures:
                word, word_source = floor, f"floor, {count} captures of {min_captures}"
            elif knee < floor:
                word, word_source = floor, "floor, knee below it"
            else:
                word, word_source = knee, "knee"
            weights = capture_weights(captures[supported])
            masked = float(np.sum(weights[seen[supported] > word]))
            intervals = bootstrap_captures(seen[supported], captures[supported], word)
            values[channel] = _fine_values(channel, anchor, rank, bulk, word, plan, quantile,
                                           f"Word from {word_source}, floor from {source}.")
            report_channels[str(channel)] = dict(
                anchor_bin=anchor, anchor_source=origin, nominal_bin=nominal, frames=len(fines), captures=count,
                word_q16=word, eta=word / Q16, word_source=word_source, knee_word_q16=knee,
                masked_fraction=masked, masked_at_floor=float(np.sum(weights[seen[supported] > floor])),
                pilot_in_most_frames=bool(np.median(seen[supported]) > floor),
                strongest_bin_median_ratio=peak, strongest_bin_masked_fraction=fraction, **intervals)

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

    report = dict(floor_word_q16=floor, floor_eta=floor / Q16, measured_floor_q16=measured_word,
                  null_quantile=quantile, floor_source=source, search_half_width=SEARCH_HALF_WIDTH,
                  null_scores=int(independent), channels=report_channels, control=control_record,
                  calibration_captures=sorted(str(Path(p).resolve()) for ps in plan["channels"].values()
                                             for p in _expand(ps)))
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
    used = {str(Path(p).resolve()) for ps in group.values() for p in _expand(ps)}
    used |= {str(Path(p).resolve()) for p in _expand(plan.get("control", []))}
    reused = used & set(calibration.get("calibration_captures", []))
    reused |= used & set((calibration.get("control") or {}).get("captures", []))
    check("test data differ from calibration data", not reused, f"{len(reused)} shared captures")

    predicted = calibration.get("channels", {})
    for channel, paths in group.items():
        channel = int(channel)
        if channel not in rows:
            check(f"channel {channel}", False, "channel is not in the bundle")
            continue
        fine_cal = rows[channel]["fine_calibration"]
        bulk = unpack_bulk_mask([int(w, 16) for w in fine_cal["bulk_mask_words_hex"]])
        fines, _ = _load_fines(paths, fmt, rows[channel], bank, k, reverse, max_frames)
        words = required_words(fines, anchor_bin=fine_cal["anchor_bin"], bulk=bulk, rank=fine_cal["cfar_rank"])
        valid = words >= 0
        check(f"rank support on channel {channel}", valid.all(), f"{int(valid.sum())} of {words.size} frames")
        if str(channel) not in predicted:
            check(f"channel {channel}", False, f"{report_path} has no prediction")
            continue
        hits = int(np.count_nonzero(words[valid] > fine_cal["cfar_multiplier_q16"]))
        trials = int(valid.sum())
        low, high = predicted[str(channel)]["masked_interval"]
        # consistent if the count could come from a rate inside the calibration interval
        agrees = binomial_upper_tail(hits, trials, high) >= MIN_TAIL and binomial_lower_tail(hits, trials, low) >= MIN_TAIL
        check(f"masked fraction on channel {channel}", agrees,
              f"masked {hits / trials:.3f}, calibration gave {low:.3f} to {high:.3f}")

    record = calibration.get("control")
    if plan.get("control") and record:
        row = dict(receiver_channel_id=record["receiver_channel_id"], physical_channel=record["channel"],
                   weight_bank_offset_bytes=0, weight_bank_nbytes=len(record["weights_hex"]) // 2)
        bulk = np.zeros(FINE_BINS, dtype=bool)
        bulk[record["bulk_bins"]] = True
        fines, _ = _load_fines(plan["control"], fmt, row, bytes.fromhex(record["weights_hex"]), k, reverse, max_frames)
        words = required_words(fines, anchor_bin=record["anchor_bin"], bulk=bulk, rank=record["cfar_rank"])
        word = int(calibration["floor_word_q16"])
        rate = 1.0 - float(calibration["null_quantile"])
        hits = int(np.count_nonzero(words > word))
        tail = binomial_upper_tail(hits, words.size, rate)
        check("false alarms on the control channel", tail >= MIN_TAIL,
              f"{hits} of {words.size} frames masked, target {rate:.3f}, tail probability {tail:.2g}")
    elif plan.get("control"):
        check("false alarms on the control channel", False, f"{report_path} is missing")
    return _finish(checks, output_json)


test_bundle.__test__ = False


def _finish(checks: list[dict[str, Any]], output_json: Path | None) -> dict[str, Any]:
    report = dict(passed=all(c["passed"] for c in checks), checks=checks)
    if output_json is not None:
        write_json_strict(Path(output_json), report)
    return report
