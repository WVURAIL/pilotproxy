# coding=utf-8
"""Calibrate a runtime bundle from captures and test a bundle on other captures.

The two entry points are the calibrate and test steps of docs/RUNNING_THE_DETECTOR.md.
Fine powers are computed with the deployed integer path, so a word found here is
the word the stage applies to the same voltages.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from pilot_proxy.capture_reference import expected_matched_filter_row_projections
from pilot_proxy.detectors.narrowband_marker.anchors import (
    DESIGNATED_HALF_WIDTH,
    WINDOW_HALF_WIDTH,
    bulk_mask,
)
from pilot_proxy.fine_decision import (
    FINE_DECISION_VERSION,
    fine_required_multiplier_q16,
    pack_bulk_mask,
    unpack_bulk_mask,
)
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.runtime_bundle import (
    _write_sha256sums,
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

FINE_BINS = 256
WINDOWS = 128
PAD_FACTOR = 2
GUARD_FINE_BINS = 1
Q16 = 65536
# chance of the hits under the null, kept small for the search over the window
MAX_LINE_TAIL = 1e-4
MIN_SIMULATED_FRAMES = 2000
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
            np.ascontiguousarray(fine), anchor_bin=anchor_bin, designated_half_width=DESIGNATED_HALF_WIDTH,
            bulk_mask=bulk, cfar_rank=rank)
        words.append(int(result.multiplier_q16) if result.valid else -1)
    return np.asarray(words, dtype=np.int64)


def null_scores(ratios: np.ndarray, *, anchor_bin: int, bulk: np.ndarray, rank: int) -> np.ndarray:
    """Return Q16 null scores at pseudo-markers tiled over the reference bins of each frame."""
    bins = np.flatnonzero(bulk)
    reference = np.sort(ratios[:, bins], axis=1)[:, rank]
    width = 2 * DESIGNATED_HALF_WIDTH + 1
    markers = (anchor_bin + width * np.arange(1, FINE_BINS // width)) % FINE_BINS
    near = np.arange(-DESIGNATED_HALF_WIDTH, DESIGNATED_HALF_WIDTH + 1)
    peaks = ratios[:, (markers[:, None] + near[None, :]) % FINE_BINS].max(axis=2)
    return (peaks / reference[:, None] * Q16).ravel()


def simulated_null_scores(num_inputs: int, frames: int = MIN_SIMULATED_FRAMES, seed: int = 20261105) -> np.ndarray:
    """Return null scores for white noise summed over num_inputs streams."""
    rng = np.random.default_rng(seed)
    bulk = bulk_mask(0, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS)
    rank = default_rank(bulk)
    ratios = np.empty((frames, FINE_BINS))
    for frame in range(frames):
        z = rng.standard_normal((3, num_inputs, WINDOWS, 2), dtype=np.float32).view(np.complex64)[..., 0]
        power = np.abs(np.fft.fft(z, FINE_BINS, axis=2)) ** 2
        ratios[frame] = fine_ratio(power.sum(axis=1))
    return null_scores(ratios, anchor_bin=0, bulk=bulk, rank=rank)


def _weights(row: dict[str, Any], bank: bytes) -> bytes:
    offset = int(row["weight_bank_offset_bytes"])
    return bank[offset:offset + int(row["weight_bank_nbytes"])]


def _load_fines(paths: Sequence[str], capture_format: str, row: dict[str, Any], bank: bytes, k: int,
                time_reverse: bool, max_frames: int | None) -> list[np.ndarray]:
    weights = _weights(row, bank)
    channel_id = int(row["receiver_channel_id"])
    fines = []
    for path in paths:
        for block in iter_blocks(path, capture_format, channel_id, k * WINDOWS, max_frames):
            fines.append(fine_powers(block, weights, k, time_reverse))
    if not fines:
        raise ValueError(f"no complete blocks for channel {row['physical_channel']}")
    return fines


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
        wanted = channels + ([int(control["channel"])] if control else [])
        bundle, bank = _export(plan, base, wanted, Path(temp) / "all")
        rows = _rows(bundle)
        reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])

        if control:
            row = rows[int(control["channel"])]
            anchor = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            bulk = bulk_mask(anchor, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS)
            fines = _load_fines(control["captures"], fmt, row, bank, k, reverse, max_frames)
            scores = null_scores(fine_ratio(np.stack(fines)), anchor_bin=anchor, bulk=bulk, rank=default_rank(bulk))
            control_record = dict(channel=int(control["channel"]), receiver_channel_id=int(row["receiver_channel_id"]),
                                  anchor_bin=anchor, bulk_bins=_bin_list(bulk), cfar_rank=default_rank(bulk),
                                  weights_hex=_weights(row, bank).hex(), frames=len(fines),
                                  captures=[str(Path(p).resolve()) for p in control["captures"]])
            source = "control channel"
        else:
            scores = simulated_null_scores(int(plan["num_inputs"]))
            control_record = None
            source = f"white noise, {int(plan['num_inputs'])} inputs"
        if scores.size * (1.0 - quantile) < 10:
            raise ValueError(f"{scores.size} null scores are too few for quantile {quantile}")
        measured_word = int(math.ceil(float(np.quantile(scores, quantile))))
        word = int(plan.get("word_q16", measured_word))
        if "word_q16" in plan:
            source = "the plan"

        report_channels = {}
        values = {}
        for channel in channels:
            row = rows[channel]
            nominal = nominal_fine_bin(row["pilot_frequency_hz"], row["coarse_channel_center_hz"], cell_hz, descending)
            paths = plan["channels"][str(channel)]
            fines = _load_fines(paths, fmt, row, bank, k, reverse, max_frames)
            window = (nominal + np.arange(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH + 1)) % FINE_BINS
            upper = np.quantile(fine_ratio(np.stack(fines)), 0.9, axis=0)
            anchor = int(window[np.argmax(upper[window])])
            given = plan.get("anchors", {}).get(str(channel))
            excluded = plan.get("exclude_bins", {}).get(str(channel), [])
            bulk = bulk_mask(anchor, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS,
                             census_excluded_bins=excluded)
            rank = default_rank(bulk)
            seen = required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank)
            hits = int(np.count_nonzero(seen > word))
            fraction = hits / seen.size
            measured = binomial_upper_tail(hits, seen.size, 1.0 - quantile) < MAX_LINE_TAIL
            peak = float(np.median(fine_ratio(np.stack(fines))[:, anchor]))
            origin = "measured" if measured else "nominal"
            if given is not None or not measured:
                # noise peak in the window, so fall back to the expected bin
                anchor = nominal if given is None else int(given)
                origin = "nominal" if given is None else "plan"
                bulk = bulk_mask(anchor, pad_factor=PAD_FACTOR, guard_fine_bins=GUARD_FINE_BINS,
                                 census_excluded_bins=excluded)
                rank = default_rank(bulk)
            values[channel] = dict(
                status="calibrated", decision_version=FINE_DECISION_VERSION, cfar_multiplier_q16=word,
                anchor_bin=anchor, designated_half_width=DESIGNATED_HALF_WIDTH, cfar_rank=rank,
                bulk_mask_words_hex=[f"0x{w:016x}" for w in pack_bulk_mask(bulk)],
                provenance=dict(epochs=[str(plan.get("epoch", "unstated"))], source_product_sha256=[],
                                null_quantile=quantile,
                                note=f"Commissioning word from {source}. Detect-only, not a calibration."))
            report_channels[str(channel)] = dict(
                anchor_bin=anchor, anchor_source=origin, nominal_bin=nominal, frames=len(fines),
                strongest_bin_median_ratio=peak, strongest_bin_masked_fraction=fraction)

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

    report = dict(word_q16=word, eta=word / Q16, measured_word_q16=measured_word, null_quantile=quantile,
                  word_source=source,
                  null_scores=int(scores.size), channels=report_channels, control=control_record,
                  calibration_captures=sorted(str(Path(p).resolve()) for ps in plan["channels"].values() for p in ps))
    write_json_strict(output_dir.with_name(output_dir.name + ".report.json"), report)
    return report


def _sha256_text(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def binomial_upper_tail(hits: int, trials: int, probability: float) -> float:
    """Return P(X >= hits) for X ~ Binomial(trials, probability)."""
    total = 0.0
    for count in range(hits, trials + 1):
        total += math.comb(trials, count) * probability ** count * (1.0 - probability) ** (trials - count)
    return min(total, 1.0)


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
    uncalibrated = [c for c, row in rows.items() if row["fine_calibration"]["status"] != "calibrated"]
    check("every bound channel has values", not uncalibrated, f"pending: {uncalibrated}")

    groups = [("strong", plan.get("strong", {})), ("quiet", plan.get("quiet", {}))]
    used = {str(Path(p).resolve()) for _, group in groups for ps in group.values() for p in ps}
    used |= {str(Path(p).resolve()) for p in plan.get("control", [])}
    reused = used & set(calibration.get("calibration_captures", []))
    reused |= used & set((calibration.get("control") or {}).get("captures", []))
    check("test data differ from calibration data", not reused, f"{len(reused)} shared captures")

    for kind, group in groups:
        for channel, paths in group.items():
            channel = int(channel)
            if channel not in rows:
                check(f"{kind} channel {channel}", False, "channel is not in the bundle")
                continue
            fine_cal = rows[channel]["fine_calibration"]
            bulk = unpack_bulk_mask([int(w, 16) for w in fine_cal["bulk_mask_words_hex"]])
            words = required_words(_load_fines(paths, fmt, rows[channel], bank, k, reverse, max_frames),
                                   anchor_bin=fine_cal["anchor_bin"], bulk=bulk, rank=fine_cal["cfar_rank"])
            valid = words >= 0
            check(f"rank support on channel {channel}", valid.all(), f"{int(valid.sum())} of {words.size} frames")
            fraction = float(np.mean(words[valid] > fine_cal["cfar_multiplier_q16"]))
            if kind == "strong":
                limit = float(plan.get("min_strong_masked", 0.9))
                check(f"strong channel {channel}", fraction >= limit, f"masked {fraction:.3f}, needs at least {limit}")
            else:
                limit = float(plan.get("max_quiet_masked", 0.2))
                check(f"quiet channel {channel}", fraction <= limit, f"masked {fraction:.3f}, limit {limit}")

    record = calibration.get("control")
    if plan.get("control") and record:
        row = dict(receiver_channel_id=record["receiver_channel_id"], physical_channel=record["channel"],
                   weight_bank_offset_bytes=0, weight_bank_nbytes=len(record["weights_hex"]) // 2)
        bulk = np.zeros(FINE_BINS, dtype=bool)
        bulk[record["bulk_bins"]] = True
        fines = _load_fines(plan["control"], fmt, row, bytes.fromhex(record["weights_hex"]), k, reverse, max_frames)
        words = required_words(fines, anchor_bin=record["anchor_bin"], bulk=bulk, rank=record["cfar_rank"])
        word = int(calibration["word_q16"])
        rate = 1.0 - float(calibration["null_quantile"])
        hits = int(np.count_nonzero(words > word))
        tail = binomial_upper_tail(hits, words.size, rate)
        check("false alarms on the control channel", tail >= 1e-3,
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
