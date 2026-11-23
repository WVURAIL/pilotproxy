"""Windows of a Pathfinder detect-only run, from Kotekan's recv files or from combined files.

Each X-engine half sends one frame of 1024 blocks (about 43 s) per window for each DTV product. recv writes them
in one of two layouts:

- one frame per file, with the window's fpga_seq_num, fpga_seq_time_nsec and coarse_freq as attributes of its one
  dataset. A run of several days is about 10^5 files, and opening each costs about 25 ms over NFS;
- several frames per file in arrival order, from every half, with each frame's metadata in the group
  frame_metadata (kotekan#1925, record_per_frame_metadata). The dataset attributes describe the first frame only.

combine() stacks the one-frame files of each half and product of an acquisition into one file,
<product>.<half>.combined.h5, with the per-window attributes as datasets. A window of a combined or a
several-frame file is named "<file>#<index>", so plans and records list it as they list a one-frame file.
"""
from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np

COMBINED = ".combined.h5"
SCHEMA = "pilotproxy-combined-dtv-v1"
WINDOW_ATTRS = ("fpga_seq_num", "fpga_seq_time_nsec")
FRAMES = "frame_metadata"
# the dataset attributes of a several-frame file that hold its first frame's values
FRAME_ATTRS = ("fpga_seq_num", "fpga_seq_time_nsec", "coarse_freq", "time_downsampling_fpga")


def _h5py():
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the bitshuffle filter of Kotekan's files)
    except ImportError:
        pass
    return h5py


def _codec() -> dict[str, Any]:
    """Byte shuffle and zstd, or gzip where hdf5plugin is not installed; Kotekan's files need it to be read.

    bitshuffle, as in Kotekan's files, cannot write under HDF5 2.0 with hdf5plugin 7.1 (cx50).
    """
    try:
        import hdf5plugin
    except ImportError:
        return dict(shuffle=True, compression="gzip", compression_opts=1)
    return dict(shuffle=True, **hdf5plugin.Zstd())


def _open(path: str | Path):
    """Open a run file for reading, in SWMR mode.

    Kotekan writes a several-frame file in SWMR mode, and while it is open, or after Kotekan was killed, the
    file opens only so (or after h5clear -s). Other files open in this mode as usual.
    """
    return _h5py().File(path, "r", swmr=True)


def _product(handle) -> str:
    return next(name for name in handle if name.startswith("dtv_"))


def split(spec: str | Path) -> tuple[str, int | None]:
    """Return the file and window index of a window name; the index is None for a per-window file."""
    path, mark, index = str(spec).rpartition("#")
    if mark and path.endswith(".h5") and index.isdigit():
        return path, int(index)
    return str(spec), None


def _stamp(path: str) -> tuple[int, int]:
    status = os.stat(path)
    return status.st_mtime_ns, status.st_size


@lru_cache(maxsize=None)
def _probe(path: str, stamp: tuple[int, int]) -> dict[str, Any] | None:
    """Return the dataset attributes of a one-frame file, or None for a file of several frames.

    Cached per file and modification, as planning and calibration both read a file's attributes.
    """
    with _open(path) as handle:
        if FRAMES in handle:
            return None
        return dict(handle[_product(handle)].attrs)


def _single(path: str) -> dict[str, Any] | None:
    return _probe(path, _stamp(path))


@lru_cache(maxsize=64)
def _frames(path: str, stamp: tuple[int, int]):
    """Keep a several-frame file open, with each frame's first row, row count and attributes.

    Frame n covers dim[n, 0] rows of the dataset, after the rows of the frames before it. frame_metadata holds
    FPGA_SEQ_NUM but no time; the time of frame n is then the first frame's fpga_seq_time_nsec plus
    seq_length_nsec per FPGA sample after it. A frame more than the file's length before the first frame
    follows an F-engine restart, and its time is not known (-1).
    """
    handle = _open(path)
    data, meta = handle[_product(handle)], handle[FRAMES]
    rows = meta["dim"][:, 0].astype(np.int64)
    # a file still being written can hold a frame's metadata before all of its rows
    rows = rows[:int(np.searchsorted(np.cumsum(rows), data.shape[0], side="right"))]
    start = np.concatenate([[0], np.cumsum(rows)[:-1]]).astype(np.int64)
    attrs = dict(data.attrs)
    per_frame: dict[str, np.ndarray] = {}
    if "FPGA_SEQ_NUM" in meta:
        per_frame["fpga_seq_num"] = meta["FPGA_SEQ_NUM"][:len(rows)].astype(np.int64)
    if "TIME_DOWNSAMPLING_FPGA" in meta:
        per_frame["time_downsampling_fpga"] = meta["TIME_DOWNSAMPLING_FPGA"][:len(rows)].astype(np.int64)
    if "FPGA_SEQ_TIME_NSEC" in meta:
        per_frame["fpga_seq_time_nsec"] = meta["FPGA_SEQ_TIME_NSEC"][:len(rows)].astype(np.int64)
    elif {"fpga_seq_num", "fpga_seq_time_nsec", "seq_length_nsec"} <= set(attrs) and "fpga_seq_num" in per_frame:
        seq, seq0 = per_frame["fpga_seq_num"], int(attrs["fpga_seq_num"])
        span = int(rows.sum()) * int(np.ravel(attrs.get("dim_scalings", [1]))[0])
        times = int(attrs["fpga_seq_time_nsec"]) + (seq - seq0) * int(attrs["seq_length_nsec"])
        per_frame["fpga_seq_time_nsec"] = np.where(seq >= seq0 - span, times, -1)
    coarse = meta["COARSE_FREQ"][:len(rows)].astype(np.int64) if "COARSE_FREQ" in meta else None
    static = {k: v for k, v in attrs.items() if k not in FRAME_ATTRS}
    return handle, data, start, rows, static, per_frame, coarse


def _frame_file(path: str):
    return _frames(path, _stamp(path))


@lru_cache(maxsize=64)
def _combined_file(path: str, stamp: tuple[int, int]):
    """Keep a combined file open, with its dataset, static attributes and per-window attributes.

    Opening a file over NFS costs about 25 ms, so a combined file is opened once per process and modification, not
    per window. A file combined again is opened again.
    """
    handle = _open(path)
    data = handle[_product(handle)]
    return handle, data, dict(data.attrs), {name: handle[name][()] for name in WINDOW_ATTRS if name in handle}


def _combined(path: str):
    return _combined_file(path, _stamp(path))


def windows(path: str | Path) -> list[str]:
    """Return the window names of a combined or several-frame file, or the path itself for any other file.

    A window name is returned as it is. Only recv's files (dtv_<product>.<number>.h5) are opened to tell one
    frame from several, as captures of other formats pass through here too.
    """
    path = str(path)
    if split(path)[1] is not None:
        return [path]
    if path.endswith(COMBINED):
        count = _combined(path)[1].shape[0]
    elif not Path(path).name.startswith("dtv_") or _single(path) is not None:
        return [path]
    else:
        count = len(_frame_file(path)[3])
    return [f"{path}#{i}" for i in range(count)]


def read_attrs(spec: str | Path) -> dict[str, Any]:
    """Return the dataset attributes of one window, with its own fpga_seq_num and fpga_seq_time_nsec."""
    path, index = split(spec)
    if index is None:
        attrs = _single(path)
        if attrs is None:
            raise ValueError(f"{path} holds several frames; name one as {path}#<frame>")
        return dict(attrs)
    if path.endswith(COMBINED):
        _, _, static, per_window = _combined(path)
        coarse = None
    else:
        _, _, _, _, static, per_window, coarse = _frame_file(path)
    attrs = dict(static)
    for name, values in per_window.items():
        # -1 marks a window whose source recorded no such value
        if int(values[index]) >= 0:
            attrs[name] = int(values[index])
    # as in a one-frame file, a frame whose rows are all unused (-1) has no coarse_freq
    if coarse is not None and (coarse[index] >= 0).any():
        attrs["coarse_freq"] = coarse[index].astype(np.int32)
    return attrs


def read_rows(spec: str | Path, channel_id: int, max_frames: int | None = None) -> tuple[np.ndarray | None, dict]:
    """Return the per-block rows of one coarse channel in one window and the window's attributes.

    The rows are None when the window's half does not carry the channel.
    """
    path, index = split(spec)
    attrs = read_attrs(spec)
    # the attribute is dropped when every row is unused (-1)
    coarse = [int(c) for c in attrs.get("coarse_freq", [])]
    if channel_id not in coarse:
        return None, attrs
    if index is not None:
        return _window(path, index, max_frames, coarse.index(channel_id)), attrs
    with _open(path) as handle:
        rows = handle[_product(handle)][:max_frames, coarse.index(channel_id)]
    return rows, attrs


def _window(path: str, index: int, max_frames: int | None = None, column: int | None = None) -> np.ndarray:
    """Return the rows of one window of a combined or several-frame file, or of one of its columns."""
    if path.endswith(COMBINED):
        data = _combined(path)[1]
        return data[index, :max_frames] if column is None else data[index, :max_frames, column]
    _, data, start, rows, _, _, _ = _frame_file(path)
    stop = start[index] + (rows[index] if max_frames is None else min(rows[index], max_frames))
    return data[start[index]:stop] if column is None else data[start[index]:stop, column]


def iter_windows(path: str | Path) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Yield (attributes, data) of every window of a one-frame, several-frame or combined file."""
    for spec in windows(path):
        file, index = split(spec)
        if index is not None:
            block = _window(file, index)
        else:
            with _open(file) as handle:
                block = handle[_product(handle)][()]
        yield read_attrs(spec), block


def v3_words(bundle_dir: str | Path | None, *, mask_control: bool = False) -> dict[int, int]:
    """Return {freq_id: word} of the bundle's fine_decision_v3 channels, the word N2 applies.

    Kotekan's DtvRfiMask decides these channels from dtv_stat, so their dtv_mask rows hold 0. The control
    is included only with ``mask_control``, as in DtvRfiMask. An empty mapping without a bundle.
    """
    import json

    from pilot_proxy.fine_decision import FINE_DECISION_V3_VERSION, v3_word

    if not bundle_dir:
        return {}
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    words = {}
    for row in bundle["profiles"]:
        calibration = row["fine_calibration"]
        if calibration.get("decision_version") != FINE_DECISION_V3_VERSION:
            continue
        if calibration.get("status") == "calibrated" or (calibration.get("status") == "control" and mask_control):
            words[int(row["receiver_channel_id"])] = v3_word(calibration)
    return words


def _half(coarse: list[int]) -> str:
    """Name a half by its coarse channels: the bound ones for fine powers, the first and count for the rest."""
    used = [c for c in coarse if c >= 0]
    if not used:
        return "idle"
    return "-".join(map(str, used)) if len(coarse) <= 4 else f"{used[0]}x{len(coarse)}"


def combine(acq_dir: str | Path, out_dir: str | Path, product: str, verify: bool = True) -> list[dict[str, Any]]:
    """Stack the per-window files of one product of one acquisition into one combined file per half.

    Windows are ordered by fpga_seq_time_nsec, or fpga_seq_num where no time is recorded, and keep their
    attributes as datasets. A half with no bound channel (no coarse_freq) is left out, as it holds no rows. A
    several-frame file (kotekan#1925) needs no combining and is left out. With verify, every window is read back
    and compared with its source. Returns one summary per file written, and a count of each kind left out.
    """
    h5py = _h5py()
    acq_dir, out_dir = Path(acq_dir), Path(out_dir)
    sources = sorted(p for p in acq_dir.glob(f"dtv_{product}.*.h5") if not p.name.endswith(COMBINED))
    groups: dict[tuple[int, ...], list[tuple[int, int, Path]]] = defaultdict(list)
    skipped = several = 0
    for path in sources:
        attrs = _single(str(path))
        if attrs is None:
            several += 1
            continue
        coarse = tuple(int(c) for c in attrs.get("coarse_freq", []))
        if not coarse:
            skipped += 1
            continue
        seq = int(attrs.get("fpga_seq_num", -1))
        time_ns = int(attrs.get("fpga_seq_time_nsec", -1))
        groups[coarse].append((time_ns, seq, path))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for coarse, members in sorted(groups.items()):
        timed = all(t >= 0 for t, _, _ in members)
        members.sort(key=lambda m: (m[0], m[2].name) if timed else (m[1], m[2].name))
        target = out_dir / f"dtv_{product}.{_half(list(coarse))}{COMBINED}"
        tmp = target.with_suffix(".tmp")
        digest = hashlib.sha256()
        with _open(members[0][2]) as first:
            data0 = first[_product(first)]
            shape, dtype, static = data0.shape, data0.dtype, {k: v for k, v in data0.attrs.items() if k not in WINDOW_ATTRS}
        with h5py.File(tmp, "w") as out:
            # fine powers are read one row (bound channel) at a time, so each row of a window is its own chunk
            rows = (1, shape[0], 1, *shape[2:]) if product == "fine_powers" else (1, *shape)
            data = out.create_dataset(f"dtv_{product}", shape=(len(members), *shape), dtype=dtype,
                                      chunks=rows, **_codec())
            for key, value in static.items():
                data.attrs[key] = value
            out["fpga_seq_num"] = np.array([s for _, s, _ in members], np.int64)
            out["fpga_seq_time_nsec"] = np.array([t for t, _, _ in members], np.int64)
            out["source"] = np.array([m[2].name for m in members], dtype=h5py.string_dtype())
            out.attrs.update(schema=SCHEMA, product=product, acquisition=acq_dir.name, windows=len(members))
            for i, (_, _, path) in enumerate(members):
                with _open(path) as source:
                    block = source[_product(source)][()]
                if block.shape != shape or block.dtype != dtype:
                    raise ValueError(f"{path}: shape {block.shape} {block.dtype}, expected {shape} {dtype}")
                data[i] = block
                digest.update(np.ascontiguousarray(block).tobytes())
        if verify:
            check = hashlib.sha256()
            with h5py.File(tmp, "r") as out:
                for i in range(len(members)):
                    check.update(np.ascontiguousarray(out[f"dtv_{product}"][i]).tobytes())
            if check.hexdigest() != digest.hexdigest():
                raise ValueError(f"{tmp}: windows read back differ from their sources")
        tmp.rename(target)
        written.append(dict(file=str(target), product=product, coarse_freq=list(coarse)[:8], windows=len(members),
                            data_sha256=digest.hexdigest(), verified=verify))
    if skipped:
        written.append(dict(product=product, skipped_without_coarse_freq=skipped))
    if several:
        written.append(dict(product=product, skipped_several_frame_files=several))
    return written
