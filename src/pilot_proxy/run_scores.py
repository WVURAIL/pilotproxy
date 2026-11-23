"""Score cache of a detect-only run, for the threshold rule.

Each frame of a bound channel gets its smallest keeping word, its support, its
time and its event (a chunk of ``event_seconds``). The product beside it holds the
three frame powers of each frame as ``fine_power_u64`` of shape (N, 3, 1), which
gives the same pilot excess as the full fine powers, at 1/256 of the size.
"""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.bundle_calibration import _rows, required_words
from pilot_proxy.characterization.score_cache import sha, write_json
from pilot_proxy.fine_decision import (
    FINE_DECISION_V3_VERSION,
    calibration_geometry,
    v3_word,
)
from pilot_proxy.run_plans import ordered_run_files

# Pathfinder timing, used when a file does not state its own
BLOCK_SAMPLES = 8192
SAMPLE_NS = 5120
MISSING = np.iinfo(np.uint64).max


def _file_scores(path: str, channel_id: int, anchor: int, bulk: np.ndarray, rank: int,
                 decision_version: str) -> dict[str, np.ndarray]:
    """Return the words, frame powers and times of one channel's frames in one file."""
    from pilot_proxy.run_files import read_rows

    rows, attrs = read_rows(path, channel_id)
    if rows is None:
        return {}
    # Kotekan writes the sample period with the telescope metadata and the samples per frame
    # as the first dim scaling: 2560 ns and 16384 on CHIME, 5120 ns and 8192 on the Pathfinder
    sample_ns = int(attrs.get("seq_length_nsec", SAMPLE_NS))
    scalings = attrs.get("dim_scalings")
    block_samples = int(scalings[0]) if scalings is not None and len(scalings) else BLOCK_SAMPLES
    start_ns = int(attrs.get("fpga_seq_time_nsec", int(attrs["fpga_seq_num"]) * sample_ns))
    # zeros were not tested, and 0xFF fills a block missing from the concat window
    present = np.flatnonzero(rows.any(axis=(1, 2)) & ~(rows == MISSING).all(axis=(1, 2)))
    rows = rows[present]
    return dict(words=required_words(rows, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision_version),
                powers=rows.sum(axis=2, dtype=np.uint64),
                time=(start_ns + present * block_samples * sample_ns) * 1e-9)


def build_run_scores(*, bundle_dir: Path, runs: Sequence[str], channel: int, output_dir: Path,
                     event_seconds: float = 10.0, jobs: int = 1) -> dict[str, Any]:
    """Write ch<channel>.npz, ch<channel>.json and product_ch<channel>.npz under output_dir."""
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    rows = _rows(bundle)
    if channel not in rows:
        raise ValueError(f"channel {channel} is not in the bundle")
    row = rows[channel]
    calibration = row["fine_calibration"]
    # v3 searches every bin and has no anchor, and its current word is the one the stage applies, the power-limit
    # word where calibration wrote one
    v3 = calibration["decision_version"] == FINE_DECISION_V3_VERSION
    anchor = int(calibration.get("anchor_bin", 0))
    word = v3_word(calibration) if v3 else int(calibration["cfar_multiplier_q16"])
    bulk, rank = calibration_geometry(calibration)
    # recording order, as a window after an F-engine restart may record no time
    paths = [path for _, _, path in ordered_run_files(runs)]
    decision = calibration["decision_version"]
    args = [(p, int(row["receiver_channel_id"]), anchor, bulk, rank, decision) for p in paths]
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_file_scores, *zip(*args)))
    else:
        parts = [_file_scores(*a) for a in args]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"no recorded frames for channel {channel}")
    words = np.concatenate([p["words"] for p in parts])
    powers = np.concatenate([p["powers"] for p in parts])
    time = np.concatenate([p["time"] for p in parts])

    supported = words >= 0
    required = np.where(supported, words, 0).astype(np.uint64)
    _, event = np.unique(np.floor((time - time[0]) / event_seconds).astype(np.int64), return_inverse=True)
    output_dir = Path(output_dir)
    target = output_dir / f"ch{channel}"
    product_path = output_dir / f"product_ch{channel}.npz"
    if target.with_suffix(".npz").exists() or target.with_suffix(".json").exists() or product_path.exists():
        raise ValueError(f"a score cache for channel {channel} already exists under {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(product_path, fine_power_u64=powers[:, :, None],
                        target_norm_sq=np.int64(row["target_norm_sq"]),
                        reference_norm_sum_sq=np.int64(row["reference_norm_sum_sq"]))
    np.savez_compressed(target.with_suffix(".npz"), source_row_index=np.arange(len(words)), required_q16=required,
                        supported=supported, probability_weight=np.full(len(words), 1.0 / len(words)),
                        frame_event_index=event, current_keep=supported & (required <= np.uint64(word)),
                        frame_time=time)
    result = dict(schema="pilotproxy-score-cache-v2", channel=channel,
                  original_product_identity=dict(path=product_path.name, bytes=product_path.stat().st_size,
                                                 sha256=sha(product_path)),
                  score_geometry=dict(anchor_bin=anchor, cfar_rank=rank, bulk_bins=int(bulk.sum())),
                  fine_decision=decision,
                  current_threshold={"kind": "finite", "eta_q16": str(word)},
                  frames=len(words), events=int(event.max()) + 1, event_seconds=event_seconds,
                  supported_frames=int(supported.sum()), masked_percent=round(
                      100.0 * (1.0 - float(np.mean(supported & (required <= np.uint64(word))))), 3),
                  physical_science_qualified=False, threshold_activated=False,
                  npz_sha256=sha(target.with_suffix(".npz")),
                  source_runs=[str(Path(p).resolve()) for p in paths])
    write_json(target.with_suffix(".json"), result)
    return result
