"""Good-input counts of the Pathfinder detector, from the bad feed mask in the visibility files.

Kotekan's bufferBadInputs writes one int8 mask element per frame, [polarization, dish] per stream (one
stream per NUMA half), 1 for a good input. The detector packs an input whose element is 0 as zero, so
the inputs it sums are the nonzero elements; -1, written before the first bad-input update of an
acquisition, counts as summed, as it does in the detector. The visibility files keep the mask under
bad_feed_mask/ (mask [time, stream, pol, dish], fpga_seq_num [time], stream_freq_id [stream, freq]),
and frame0_unix_ns and fpga_seq_length_ns turn its fpga_seq_num into times.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pilot_proxy.bundle_calibration import _expand


def read_good_inputs(vis_paths: Iterable[str], freq_ids: Iterable[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {freq_id: (time_ns, good inputs)} over the visibility files, in time order.

    A frequency counts the inputs of the stream that carries it. Rows that two files share are kept once.
    """
    import h5py

    wanted = [int(f) for f in freq_ids]
    rows: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {f: [] for f in wanted}
    for path in _expand(list(vis_paths)):
        try:
            handle = h5py.File(path, "r")
        except OSError:            # a partial file left by a stopped writer
            continue
        with handle:
            if "bad_feed_mask" not in handle:
                continue
            group = handle["bad_feed_mask"]
            seq = group["fpga_seq_num"][()].astype(np.int64)
            time_ns = int(handle.attrs["frame0_unix_ns"]) + seq * int(handle.attrs["fpga_seq_length_ns"])
            streams = group["stream_freq_id"][()]
            for freq_id in wanted:
                hit = np.flatnonzero((streams == freq_id).any(axis=1))
                if not hit.size:
                    continue
                mask = group["mask"][:, int(hit[0])]
                rows[freq_id].append((time_ns, np.count_nonzero(mask.reshape(len(seq), -1), axis=1)))
    out = {}
    for freq_id, parts in rows.items():
        if not parts:
            continue
        times = np.concatenate([t for t, _ in parts])
        good = np.concatenate([g for _, g in parts])
        times, first = np.unique(times, return_index=True)
        out[freq_id] = (times, good[first].astype(np.int32))
    return out


def save_good_inputs(counts: dict[int, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    np.savez_compressed(path, **{f"t{f}": t for f, (t, _) in counts.items()}, **{f"g{f}": g for f, (_, g) in counts.items()})


def load_good_inputs(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = np.load(path)
    return {int(key[1:]): (data[key], data["g" + key[1:]]) for key in data.files if key.startswith("t")}


def changes(times: np.ndarray, good: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (time_ns, count before, count after) at each change of the count."""
    at = np.flatnonzero(np.diff(good) != 0) + 1
    return [(int(times[i]), int(good[i - 1]), int(good[i])) for i in at]


def span_counts(times: np.ndarray, good: np.ndarray, start_ns: int, stop_ns: int) -> tuple[int, int, int] | None:
    """Return the (min, median, max) good inputs over [start_ns, stop_ns), or None with no mask rows there."""
    lo, hi = np.searchsorted(times, [start_ns, stop_ns])
    if hi <= lo:
        return None
    part = good[lo:hi]
    return int(part.min()), int(np.median(part)), int(part.max())
