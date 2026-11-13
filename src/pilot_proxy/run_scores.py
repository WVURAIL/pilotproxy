"""Score cache of a detect-only CHORD run, for the threshold rule.

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

from pilot_proxy.bundle_calibration import _expand, _rows, required_words
from pilot_proxy.characterization.score_cache import sha, write_json
from pilot_proxy.fine_decision import unpack_bulk_mask

BLOCK_SAMPLES = 8192
SAMPLE_NS = 5120
MISSING = np.iinfo(np.uint64).max


def _file_scores(path: str, channel_id: int, anchor: int, bulk: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    """Return the words, frame powers and times of one channel's frames in one file."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)
    except ImportError:
        pass
    with h5py.File(path, "r") as handle:
        data = handle[next(name for name in handle if name.startswith("dtv_fine_powers"))]
        coarse = [int(c) for c in data.attrs.get("coarse_freq", [])]
        if channel_id not in coarse:
            return {}
        rows = data[:, coarse.index(channel_id)]
        start_ns = int(data.attrs.get("fpga_seq_time_nsec", int(data.attrs["fpga_seq_num"]) * SAMPLE_NS))
    # zeros were not tested, and 0xFF fills a block missing from the concat window
    present = np.flatnonzero(rows.any(axis=(1, 2)) & ~(rows == MISSING).all(axis=(1, 2)))
    rows = rows[present]
    return dict(words=required_words(rows, anchor_bin=anchor, bulk=bulk, rank=rank),
                powers=rows.sum(axis=2, dtype=np.uint64),
                time=(start_ns + present * BLOCK_SAMPLES * SAMPLE_NS) * 1e-9)


def build_run_scores(*, bundle_dir: Path, runs: Sequence[str], channel: int, output_dir: Path,
                     event_seconds: float = 10.0, jobs: int = 1) -> dict[str, Any]:
    """Write ch<channel>.npz, ch<channel>.json and product_ch<channel>.npz under output_dir."""
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    rows = _rows(bundle)
    if channel not in rows:
        raise ValueError(f"channel {channel} is not in the bundle")
    row = rows[channel]
    calibration = row["fine_calibration"]
    anchor, word = int(calibration["anchor_bin"]), int(calibration["cfar_multiplier_q16"])
    bulk = unpack_bulk_mask([int(w, 16) for w in calibration["bulk_mask_words_hex"]])
    rank = int(calibration["cfar_rank"])
    paths = _expand(runs)
    args = [(p, int(row["receiver_channel_id"]), anchor, bulk, rank) for p in paths]
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
    order = np.argsort(time, kind="stable")
    words, powers, time = words[order], powers[order], time[order]

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
                  current_threshold={"kind": "finite", "eta_q16": str(word)},
                  frames=len(words), events=int(event.max()) + 1, event_seconds=event_seconds,
                  supported_frames=int(supported.sum()), masked_percent=round(
                      100.0 * (1.0 - float(np.mean(supported & (required <= np.uint64(word))))), 3),
                  physical_science_qualified=False, threshold_activated=False,
                  npz_sha256=sha(target.with_suffix(".npz")),
                  source_runs=[str(Path(p).resolve()) for p in paths])
    write_json(target.with_suffix(".json"), result)
    return result
