#!/usr/bin/env python3
# coding=utf-8
"""Compare the fused fine-mask entry point with the staged path on CHIME frames.

The staged path is the library's staged entry points
(``PilotProxy_Compute_RowSums_I32`` and ``PilotProxy_Compute_FinePowers_U64``, called by
``detect_packed_detector_input``), whose frame powers are the host's exact
uint64 all-bin sums of the staged fine powers (``P_r = sum_f S_r[f]``), with the fine decision made in Python by
``fine_decision.fine_mask_decision``.
kotekan instead calls the fused entry point,
``PilotProxy_Compute_FusedFineMaskWithSupport_U64`` (or
``PilotProxy_Compute_FusedFineMask_U64`` when no support buffer is bound). This tool
replays CHIME baseband files through both and compares every output word:

* the fused mask, ``rank_valid`` and ``n_bulk`` against ``fine_mask_decision``
  on the staged fine powers, at each channel's working word and rank;
* the fused fine-power tap against the staged fine powers, and the fused
  frame-power tap (the device's all-bin sums) against the host sums of the
  staged fine powers;
* the fused entry point without support outputs or taps, against the same decision;
* optionally, a boundary probe at the frame's exact required multiplier
  (``fine_required_multiplier_q16``) and one below it, so both mask values are
  exercised on every valid frame;
* optionally, the staged fine powers against the stored ``fine_power_u64`` of
  a per-pilot product, and the staged frame powers against the sums of those
  stored fine powers (``product_contract.frame_powers_from_fine``), which ties
  the frames to the products the results were computed from. A schema v5
  product's stored ``p_*_u64`` are the retired row-domain powers and are not
  compared; a v6 product's equal the fine sums by its invariant.

Frames are packed exactly as the archive analyzer packs them (native
offset-binary int4, lossless, ``pilot_proxy.archive.detector``). Geometry and
words come from a working-threshold file (``channels[*].score_geometry`` and
``channels[*].threshold.eta_q16``). The tool writes one JSON record and stops at
the first difference; it never edits its inputs. Example::

    python tools/fused_real_receipt.py \\
        --thresholds WORKING_THRESHOLDS.json.gz \\
        --product /path/products/_per_pilot/521.npz \\
        --output receipt.json /path/canfar_pilots_10s/ch0521
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pilot_proxy.fine_decision import (  # noqa: E402
    ALWAYS_MASKED_Q16,
    FINE_BINS,
    MAX_MULTIPLIER_Q16,
    fine_mask_decision,
    fine_required_multiplier_q16,
    pack_bulk_mask,
)
from pilot_proxy.product_contract import frame_powers_from_fine  # noqa: E402

SCHEMA = "pilot-proxy-fused-real-receipt-v2"
TERMS = 3
GEOMETRY_KEYS = ("channel", "freq_id", "anchor_bin", "designated_half_width",
                 "bulk_bins", "cfar_rank_zero_based", "multiplier_q16")
COUNT_KEYS = ("files", "frames", "mask_ones", "rank_invalid", "probe_frames",
              "product_frames_matched")


@dataclass(frozen=True)
class ChannelGeometry:
    """One channel's fine decision constants, as the bundle would carry them."""

    channel: int
    freq_id: int
    anchor_bin: int
    designated_half_width: int
    bulk_mask: np.ndarray
    cfar_rank: int
    multiplier_q16: int

    def summary(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "freq_id": self.freq_id,
            "anchor_bin": self.anchor_bin,
            "designated_half_width": self.designated_half_width,
            "bulk_bins": int(np.count_nonzero(self.bulk_mask)),
            "cfar_rank_zero_based": self.cfar_rank,
            "multiplier_q16": str(self.multiplier_q16),
        }


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_geometry(path: Path) -> dict[int, ChannelGeometry]:
    """Read per-channel geometry and finite working words, keyed by freq_id."""
    data = _read_json(path)
    out: dict[int, ChannelGeometry] = {}
    for entry in data["channels"]:
        geometry = entry["score_geometry"]
        threshold = entry["threshold"]
        if threshold.get("kind") != "finite":
            raise ValueError(
                f"channel {entry['channel']}: threshold kind "
                f"{threshold.get('kind')!r} has no finite word to compare"
            )
        rank = int(geometry["cfar_rank_zero_based"])
        if "rho_one_based" in geometry and int(geometry["rho_one_based"]) != rank + 1:
            raise ValueError(f"channel {entry['channel']}: rho and rank disagree")
        bins = [int(b) for b in geometry["bulk_bins"]]
        if any(not 0 <= b < FINE_BINS for b in bins) or len(set(bins)) != len(bins):
            raise ValueError(f"channel {entry['channel']}: bad bulk bins")
        mask = np.zeros(FINE_BINS, dtype=bool)
        mask[bins] = True
        word = int(threshold["eta_q16"])
        if not 0 < word <= MAX_MULTIPLIER_Q16:
            raise ValueError(f"channel {entry['channel']}: word out of range")
        freq_id = int(entry["pilot_frequency_id"])
        if freq_id in out:
            raise ValueError(f"duplicate freq_id {freq_id} in {path}")
        out[freq_id] = ChannelGeometry(
            channel=int(entry["channel"]),
            freq_id=freq_id,
            anchor_bin=int(geometry["anchor_bin"]),
            designated_half_width=int(geometry["designated_half_width"]),
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=word,
        )
    return out


def boundary_multipliers(
    fine: np.ndarray, geometry: ChannelGeometry
) -> tuple[int, int] | None:
    """Return (required - 1, required) when both are deployable words."""
    required = fine_required_multiplier_q16(
        fine,
        anchor_bin=geometry.anchor_bin,
        designated_half_width=geometry.designated_half_width,
        bulk_mask=geometry.bulk_mask,
        cfar_rank=geometry.cfar_rank,
    )
    word = required.multiplier_q16
    if not required.valid or word is None or word == ALWAYS_MASKED_Q16 or word < 2:
        return None
    return int(word) - 1, int(word)


def compare_frame(
    *,
    staged_fine: np.ndarray,
    staged_powers: Sequence[int],
    fused: Mapping[str, Any],
    geometry: ChannelGeometry,
    multiplier_q16: int,
    with_support: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Compare one fused launch with the reference on the staged powers.

    Returns (differences, counted words by field, reference mask bit).
    ``fused`` holds ``mask`` and optionally ``rank_valid``, ``n_bulk``,
    ``fine`` ([3, 256] uint64) and ``powers`` ([3] uint64, the device's frame
    powers). ``staged_powers`` are the staged frame powers, the host's exact
    all-bin sums of ``staged_fine``.
    """
    staged_fine = np.asarray(staged_fine)
    if staged_fine.shape != (TERMS, FINE_BINS) or staged_fine.dtype != np.uint64:
        raise ValueError("staged fine powers must be uint64 [3, 256]")
    reference = fine_mask_decision(
        staged_fine,
        anchor_bin=geometry.anchor_bin,
        designated_half_width=geometry.designated_half_width,
        bulk_mask=geometry.bulk_mask,
        cfar_rank=geometry.cfar_rank,
        multiplier_q16=int(multiplier_q16),
    )
    expected: dict[str, Any] = {"mask": int(reference.mask)}
    if with_support:
        expected["rank_valid"] = 1 if reference.valid else 0
        expected["n_bulk"] = int(reference.n_bulk)
    differences: list[dict[str, Any]] = []
    counted: dict[str, int] = {}
    for field, want in expected.items():
        got = int(fused[field])
        counted[field] = counted.get(field, 0) + 1
        if got != want:
            differences.append({"field": field, "fused": got, "reference": want})
    if fused.get("fine") is not None:
        got_fine = np.asarray(fused["fine"], dtype=np.uint64).reshape(TERMS, FINE_BINS)
        counted["fine_powers"] = int(got_fine.size)
        bad = np.argwhere(got_fine != staged_fine)
        if bad.size:
            term, fine_bin = (int(v) for v in bad[0])
            differences.append({
                "field": "fine_powers",
                "count": int(len(bad)),
                "first_term": term,
                "first_bin": fine_bin,
                "fused": str(int(got_fine[term, fine_bin])),
                "reference": str(int(staged_fine[term, fine_bin])),
            })
    if fused.get("powers") is not None:
        got_powers = [int(v) for v in np.asarray(fused["powers"]).reshape(-1)]
        want_powers = [int(v) for v in staged_powers]
        counted["frame_powers"] = len(want_powers)
        if got_powers != want_powers:
            differences.append({
                "field": "frame_powers",
                "fused": [str(v) for v in got_powers],
                "reference": [str(v) for v in want_powers],
            })
    return differences, counted, int(reference.mask)


class ProductIndex:
    """Stored exact fine powers of one per-pilot product, keyed by (event, frame).

    ``powers`` are the frame powers summed from the stored fine powers
    (``frame_powers_from_fine``), never the stored ``p_*_u64``: in a schema v5
    product those are the retired row-domain powers.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.freq_id = int(np.asarray(data["freq_id"]).reshape(-1)[0])
            unit_event = np.asarray(data["unit_event_id"], dtype=np.int64)
            frame_unit = np.asarray(data["frame_unit_index"], dtype=np.int64)
            frame_in_unit = np.asarray(data["frame_in_unit"], dtype=np.int64)
            self.fine = np.asarray(data["fine_power_u64"])
            self.powers = np.stack(
                frame_powers_from_fine({"fine_power_u64": self.fine}), axis=1
            )
            self.weights_hash = str(np.asarray(data["weights_hash"]).reshape(-1)[0])
        if len(np.unique(unit_event)) != len(unit_event):
            raise ValueError(f"{self.path}: event ids repeat across units")
        if self.fine.shape[1:] != (TERMS, FINE_BINS):
            raise ValueError(f"{self.path}: fine_power_u64 is not [frames, 3, 256]")
        self._rows = {
            (int(unit_event[u]), int(f)): row
            for row, (u, f) in enumerate(zip(frame_unit, frame_in_unit))
        }

    def row(self, event_id: int | None, frame: int) -> int | None:
        if event_id is None:
            return None
        return self._rows.get((int(event_id), int(frame)))


def expand_inputs(paths: Iterable[str], max_files_per_dir: int | None) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = sorted(path.glob("*.h5"))
            if max_files_per_dir is not None:
                files = files[: int(max_files_per_dir)]
            out.extend(files)
        else:
            out.append(path)
    if not out:
        raise SystemExit("no input files")
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty_paths": [line for line in run("status", "--porcelain").splitlines() if line],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_paths": None}


class FusedRunner:
    """Launch the fused entry points on one packed frame already on the device."""

    def __init__(self, kernel: Any, cp: Any) -> None:
        self.kernel = kernel
        self.cp = cp

    def run(self, handle: Any, weights: np.ndarray, geometry: ChannelGeometry,
            multiplier_q16: int, *, with_support: bool, with_taps: bool) -> dict[str, Any]:
        cp = self.cp
        d_fine = cp.zeros((1, TERMS, FINE_BINS), dtype=cp.uint64)
        d_mask = cp.full(1, 77, dtype=cp.int32)
        d_powers = cp.zeros((1, TERMS), dtype=cp.uint64) if with_taps else None
        args = [
            handle,
            weights.ctypes.data,
            geometry.anchor_bin,
            geometry.designated_half_width,
            pack_bulk_mask(geometry.bulk_mask),
            geometry.cfar_rank,
            int(multiplier_q16),
            int(d_fine.data.ptr),
            int(d_mask.data.ptr),
        ]
        if with_support:
            d_rank_valid = cp.full(1, 77, dtype=cp.int32)
            d_n_bulk = cp.full(1, 77, dtype=cp.int32)
            args += [int(d_rank_valid.data.ptr), int(d_n_bulk.data.ptr)]
            compute = self.kernel.compute_fused_fine_mask_with_support_u64
        else:
            compute = self.kernel.compute_fused_fine_mask_u64
        args += [int(d_powers.data.ptr) if d_powers is not None else 0, 0]
        compute(*args)
        cp.cuda.Device().synchronize()
        out: dict[str, Any] = {
            "mask": int(cp.asnumpy(d_mask)[0]),
            "fine": cp.asnumpy(d_fine)[0],
            "powers": cp.asnumpy(d_powers)[0] if d_powers is not None else None,
        }
        if with_support:
            out["rank_valid"] = int(cp.asnumpy(d_rank_valid)[0])
            out["n_bulk"] = int(cp.asnumpy(d_n_bulk)[0])
        return out


def run_receipt(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz
    from pilot_proxy.archive.interfaces import UnreadableUnitError
    from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
    from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
    from pilot_proxy.chime.hdf5_input import (
        CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        nearest_atsc_physical_channel,
    )
    from pilot_proxy.config.instrument import load_instrument
    from pilot_proxy.detect import detect_packed_detector_input
    from pilot_proxy.detector_weights import DetectorWeightBank
    from pilot_proxy.kernel import PilotProxyKernel

    started = time.perf_counter()
    timers = dict.fromkeys(("hash", "read", "pack", "staged", "reference", "fused", "product"), 0.0)

    instrument = load_instrument(args.instrument)
    nfft = int(instrument.nfft)
    nyquist_zone = int(instrument.nyquist_zone)
    kernel = PilotProxyKernel(str(args.lib))
    if not kernel.supports_fused_fine_mask_with_support():
        raise SystemExit("library has no PilotProxy_Compute_FusedFineMaskWithSupport_U64")
    window = int(kernel.specs.K)
    bank = DetectorWeightBank(explicit_path=args.weights, expected_kernel=kernel.specs)
    geometry_by_freq = load_geometry(args.thresholds)
    products = {p.freq_id: p for p in (ProductIndex(path) for path in args.product)}
    runner = FusedRunner(kernel, cp)
    reader = ChimeBasebandPackedReader()
    ctx = SimpleNamespace(instrument=instrument)
    files = expand_inputs(args.inputs, args.max_files_per_dir)

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.executable).name, *sys.argv],
        "tool": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
            "git": _git_identity(),
        },
        "library": {
            "path": str(Path(args.lib).resolve()),
            "sha256": _sha256(Path(args.lib)),
            "core_version": f"{kernel.version.major}.{kernel.version.minor}.{kernel.version.patch}",
            "K": window,
        },
        "host": {
            "gpu": properties["name"].decode() if isinstance(properties["name"], bytes) else str(properties["name"]),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
        },
        "weights": {"path": str(Path(args.weights).resolve()), "sha256": _sha256(Path(args.weights))},
        "thresholds": {"path": str(Path(args.thresholds).resolve()), "sha256": _sha256(Path(args.thresholds))},
        "instrument": {"name": args.instrument, "nfft": nfft, "nyquist_zone": nyquist_zone},
        "launches": {
            "staged": "detect_packed_detector_input(emit_row_projections=True) "
                      "+ fine_decision.fine_mask_decision",
            "fused_with_support": "PilotProxy_Compute_FusedFineMaskWithSupport_U64, frame-power tap on, row tap off",
            "fused_legacy": "PilotProxy_Compute_FusedFineMask_U64, no taps",
            "boundary_probe": bool(args.boundary_probe),
        },
        "products": {
            str(freq): {"path": str(p.path.resolve()), "weights_hash": p.weights_hash}
            for freq, p in products.items()
        },
        "files": [],
        "skipped": [],
        "channels": {},
        "first_difference": None,
    }
    words: dict[str, int] = {}
    frames_total = 0
    launches = 0
    stop = False
    weights_cache: dict[int, np.ndarray] = {}

    for path in files:
        t0 = time.perf_counter()
        file_sha = _sha256(path)
        timers["hash"] += time.perf_counter() - t0
        try:
            meta = reader.probe(str(path))
        except UnreadableUnitError as exc:
            # The archive quarantines these units (sub-frame or unreadable) too.
            record["skipped"].append({"path": str(path.resolve()), "sha256": file_sha,
                                      "bytes": path.stat().st_size, "reason": str(exc)})
            continue
        f_center_hz = float(meta["f_center_hz"])
        freq_id = int(chime_freq_id_from_hz(f_center_hz, float(instrument.f0_mhz) * 1e6))
        channel = nearest_atsc_physical_channel(f_center_hz, tolerance_hz=3.0e6)
        if freq_id not in geometry_by_freq:
            raise SystemExit(f"{path}: freq_id {freq_id} has no working word")
        geometry = geometry_by_freq[freq_id]
        if channel != geometry.channel:
            raise SystemExit(f"{path}: channel {channel} != geometry channel {geometry.channel}")
        if channel not in weights_cache:
            weights, valid = bank.get_weights_for_physical_channel(int(channel))
            if weights is None or not valid:
                raise SystemExit(f"no valid weights for channel {channel}")
            weights_cache[channel] = np.ascontiguousarray(weights)
        weights = weights_cache[channel]
        product = products.get(freq_id)
        entry = {
            "path": str(path.resolve()),
            "sha256": file_sha,
            "bytes": path.stat().st_size,
            "channel": int(channel),
            "freq_id": freq_id,
            "event_id": meta.get("event_id"),
            "num_input_streams": int(meta["num_input_streams"]),
            "frames": 0,
            "product_frames_matched": 0 if product is not None else None,
        }
        stats = record["channels"].setdefault(
            str(channel), {**geometry.summary(), **dict.fromkeys(COUNT_KEYS, 0)}
        )
        stats["files"] += 1
        t_read = time.perf_counter()
        frames_iter = iter(enumerate(reader.iter_arrays(str(path), ctx)))
        while True:
            try:
                frame_index, chunk = next(frames_iter)
            except StopIteration:
                break
            except UnreadableUnitError as exc:
                record["skipped"].append({"path": str(path.resolve()), "sha256": file_sha,
                                          "bytes": path.stat().st_size, "reason": str(exc),
                                          "frames_compared_before": entry["frames"]})
                break
            if args.max_frames_per_file is not None and frame_index >= args.max_frames_per_file:
                break
            arr = np.asarray(chunk)
            timers["read"] += time.perf_counter() - t_read
            if arr.ndim != 2 or arr.shape[0] != nfft or arr.dtype != np.uint8:
                raise SystemExit(f"{path}: unexpected frame {arr.shape} {arr.dtype}")
            t1 = time.perf_counter()
            packed = pack_chime_block_for_detector(
                np.ascontiguousarray(arr.T)[:, np.newaxis, :],
                frame_size_samples=nfft,
                detector_window_samples=window,
                nyquist_zone=nyquist_zone,
                frames_in_chunk=1,
                sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
                selected_coarse_channel=0,
                physical_channel=int(channel),
            ).packed
            t2 = time.perf_counter()
            timers["pack"] += t2 - t1
            staged = detect_packed_detector_input(
                packed=packed, weights=weights, kernel=kernel, emit_row_projections=True,
            )
            if staged.get("fine_powers_u64") is None:
                raise SystemExit("staged path emitted no fine powers")
            staged_fine = np.asarray(staged["fine_powers_u64"])[0]
            row = staged["results"][0]
            staged_powers = [row["p_target_u64"], row["p_ref_lower_u64"], row["p_ref_upper_u64"]]
            t3 = time.perf_counter()
            timers["staged"] += t3 - t2
            context = {"file": str(path), "frame_in_file": frame_index, "channel": int(channel)}
            if product is not None:
                product_row = product.row(meta.get("event_id"), frame_index)
                if product_row is not None:
                    want = product.fine[product_row]
                    want_powers = [int(v) for v in product.powers[product_row]]
                    words["product_fine_powers"] = words.get("product_fine_powers", 0) + int(want.size)
                    words["product_fine_power_sums"] = words.get("product_fine_power_sums", 0) + 3
                    if not np.array_equal(np.asarray(want, dtype=np.uint64), staged_fine) or want_powers != [int(v) for v in staged_powers]:
                        record["first_difference"] = {
                            **context, "launch": "stored product vs staged",
                            "product_row": int(product_row),
                            "fine_mismatches": int(np.count_nonzero(np.asarray(want, dtype=np.uint64) != staged_fine)),
                            "product_fine_power_sums": [str(v) for v in want_powers],
                            "staged_powers": [str(int(v)) for v in staged_powers],
                        }
                        stop = True
                    entry["product_frames_matched"] += 1
                    stats["product_frames_matched"] += 1
            timers["product"] += time.perf_counter() - t3
            if stop:
                break

            cases = [("fused_with_support", geometry.multiplier_q16, True, True),
                     ("fused_legacy", geometry.multiplier_q16, False, False)]
            t4 = time.perf_counter()
            probe = boundary_multipliers(staged_fine, geometry) if args.boundary_probe else None
            timers["reference"] += time.perf_counter() - t4
            if probe is not None:
                cases += [("probe_below_required", probe[0], True, False),
                          ("probe_at_required", probe[1], True, False)]
            d_in = cp.asarray(np.ascontiguousarray(packed))
            if packed.ndim == 3 and packed.shape[0] != 1:
                raise SystemExit(f"{path}: expected one frame per packed block")
            rows = int(packed.shape[-2]) if packed.ndim == 3 else int(packed.shape[0])
            handle = kernel.create_raw(rows, d_in.data.ptr)
            try:
                for name, multiplier, with_support, with_taps in cases:
                    t5 = time.perf_counter()
                    fused = runner.run(handle, weights, geometry, multiplier,
                                       with_support=with_support, with_taps=with_taps)
                    t6 = time.perf_counter()
                    timers["fused"] += t6 - t5
                    launches += 1
                    differences, counted, reference_mask = compare_frame(
                        staged_fine=staged_fine, staged_powers=staged_powers, fused=fused,
                        geometry=geometry, multiplier_q16=multiplier, with_support=with_support,
                    )
                    timers["reference"] += time.perf_counter() - t6
                    for field, count in counted.items():
                        key = f"{name}.{field}"
                        words[key] = words.get(key, 0) + count
                    if name.startswith("probe") and reference_mask != (1 if name == "probe_below_required" else 0):
                        differences.append({"field": "probe_reference", "reference": reference_mask})
                    if differences:
                        record["first_difference"] = {
                            **context, "launch": name, "multiplier_q16": str(multiplier),
                            "differences": differences,
                        }
                        stop = True
                        break
                    if name == "fused_with_support":
                        stats["mask_ones"] += int(fused["mask"])
                        stats["rank_invalid"] += 1 - int(fused["rank_valid"])
                    if name == "probe_at_required":
                        stats["probe_frames"] += 1
            finally:
                kernel.destroy(handle)
            entry["frames"] += 1
            stats["frames"] += 1
            frames_total += 1
            if stop:
                break
            t_read = time.perf_counter()
        record["files"].append(entry)
        if stop:
            break

    record["totals"] = {
        "files": len(record["files"]),
        "skipped_files": len(record["skipped"]),
        "channels": len(record["channels"]),
        "frames": frames_total,
        "fused_launches": launches,
        "words_compared": dict(sorted(words.items())),
        "words_compared_total": int(sum(words.values())),
    }
    record["result"] = "DIFFERENCE" if record["first_difference"] else "PASS"
    record["timings_s"] = {key: round(value, 3) for key, value in timers.items()}
    record["timings_s"]["wall"] = round(time.perf_counter() - started, 3)
    return record


def merge_receipts(paths: Sequence[Path]) -> dict[str, Any]:
    """Combine receipts of disjoint inputs made with one tool, library and word set."""
    parts = [(Path(p), json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]
    if not parts:
        raise SystemExit("nothing to merge")
    first = parts[0][1]
    shared = ("schema", "library", "weights", "thresholds", "instrument", "launches")
    for path, part in parts[1:]:
        for key in shared:
            if part[key] != first[key]:
                raise SystemExit(f"{path}: {key} differs from {parts[0][0]}")
        if part["tool"]["sha256"] != first["tool"]["sha256"]:
            raise SystemExit(f"{path}: made with a different tool")
    seen: set[str] = set()
    channels: dict[str, dict[str, Any]] = {}
    words: dict[str, int] = {}
    timings: dict[str, float] = {}
    for path, part in parts:
        for entry in part["files"]:
            if entry["path"] in seen:
                raise SystemExit(f"{entry['path']} appears in more than one part")
            seen.add(entry["path"])
        for name, stats in part["channels"].items():
            if name not in channels:
                channels[name] = dict(stats)
                continue
            merged = channels[name]
            if any(merged[key] != stats[key] for key in GEOMETRY_KEYS):
                raise SystemExit(f"{path}: channel {name} geometry differs")
            for key in COUNT_KEYS:
                merged[key] += stats[key]
        for key, value in part["totals"]["words_compared"].items():
            words[key] = words.get(key, 0) + int(value)
        for key, value in part["timings_s"].items():
            timings[key] = round(timings.get(key, 0.0) + float(value), 3)
    differences = [part["first_difference"] for _, part in parts if part["first_difference"]]
    record = {key: first[key] for key in ("schema", "tool", "library", "host", "weights",
                                          "thresholds", "instrument", "launches")}
    record.update({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.executable).name, *sys.argv],
        "parts": [{"path": str(path.resolve()), "sha256": _sha256(path),
                   "command": part["command"], "result": part["result"],
                   "wall_s": part["timings_s"]["wall"]} for path, part in parts],
        "products": {k: v for _, part in parts for k, v in part["products"].items()},
        "files": [entry for _, part in parts for entry in part["files"]],
        "skipped": [entry for _, part in parts for entry in part.get("skipped", [])],
        "channels": dict(sorted(channels.items(), key=lambda item: int(item[0]))),
        "first_difference": differences[0] if differences else None,
    })
    record["totals"] = {
        "files": len(record["files"]),
        "skipped_files": len(record["skipped"]),
        "channels": len(channels),
        "frames": sum(part["totals"]["frames"] for _, part in parts),
        "fused_launches": sum(part["totals"]["fused_launches"] for _, part in parts),
        "words_compared": dict(sorted(words.items())),
        "words_compared_total": int(sum(words.values())),
    }
    record["result"] = "DIFFERENCE" if differences else "PASS"
    record["timings_s"] = dict(timings, note="summed over parts; wall is the sum of part walls")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH

    parser.add_argument("inputs", nargs="+",
                        help="CHIME baseband .h5 files or directories of them "
                             "(receipt JSON files with --merge)")
    parser.add_argument("--merge", action="store_true",
                        help="combine receipts of disjoint inputs instead of running")
    parser.add_argument("--thresholds", type=Path,
                        help="working-threshold JSON (or .json.gz) with per-channel geometry")
    parser.add_argument("--output", type=Path, required=True, help="receipt JSON to write")
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--instrument", default="chime")
    parser.add_argument("--product", type=Path, action="append", default=[],
                        help="per-pilot product .npz to compare stored powers with")
    parser.add_argument("--max-files-per-dir", type=int, default=None)
    parser.add_argument("--max-frames-per-file", type=int, default=None)
    parser.add_argument("--no-boundary-probe", dest="boundary_probe", action="store_false",
                        help="skip the two launches at the exact required multiplier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merge:
        record = merge_receipts([Path(p) for p in args.inputs])
    elif args.thresholds is None:
        parser.error("--thresholds is required unless --merge is given")
    else:
        record = run_receipt(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    totals = record["totals"]
    print(f"{record['result']}: {totals['frames']} frames, {totals['channels']} channels, "
          f"{totals['fused_launches']} fused launches, "
          f"{totals['words_compared_total']} words compared -> {args.output}")
    if record["first_difference"]:
        print(json.dumps(record["first_difference"], indent=1))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
