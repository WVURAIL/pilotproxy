"""Windows of a Pathfinder detect-only run, from Kotekan's per-window files or from combined files.

recv writes one file per X-engine half and window of 1024 blocks (about 43 s) for each DTV product, with the
window's fpga_seq_num, fpga_seq_time_nsec and coarse_freq as attributes of its one dataset. A run of several days
is then about 10^5 files, and opening each costs about 25 ms over NFS. combine() stacks the windows of each half and
product of an acquisition into one file, <product>.<half>.combined.h5, with the per-window attributes as datasets.
A window of a combined file is named "<file>#<index>", so plans and records list it as they list a per-window file.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

COMBINED = ".combined.h5"
SCHEMA = "pilotproxy-combined-dtv-v1"
WINDOW_ATTRS = ("fpga_seq_num", "fpga_seq_time_nsec")


def _h5py():
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers the bitshuffle filter of Kotekan's files)
    except ImportError:
        pass
    return h5py


def _product(handle) -> str:
    return next(name for name in handle if name.startswith("dtv_"))


def split(spec: str | Path) -> tuple[str, int | None]:
    """Return the file and window index of a window name; the index is None for a per-window file."""
    path, mark, index = str(spec).rpartition("#")
    if mark and path.endswith(COMBINED) and index.isdigit():
        return path, int(index)
    return str(spec), None


def windows(path: str | Path) -> list[str]:
    """Return the window names of a combined file, or the path itself for a per-window file."""
    path = str(path)
    if not path.endswith(COMBINED):
        return [path]
    with _h5py().File(path, "r") as handle:
        return [f"{path}#{i}" for i in range(handle[_product(handle)].shape[0])]


def read_attrs(spec: str | Path) -> dict[str, Any]:
    """Return the dataset attributes of one window, with its own fpga_seq_num and fpga_seq_time_nsec."""
    path, index = split(spec)
    with _h5py().File(path, "r") as handle:
        data = handle[_product(handle)]
        attrs = dict(data.attrs)
        if index is not None:
            for name in WINDOW_ATTRS:
                if name in handle:
                    value = int(handle[name][index])
                    # -1 marks a window whose source file recorded no such attribute
                    attrs.pop(name, None)
                    if value >= 0:
                        attrs[name] = value
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
    with _h5py().File(path, "r") as handle:
        data = handle[_product(handle)]
        rows = data[:max_frames, coarse.index(channel_id)] if index is None \
            else data[index, :max_frames, coarse.index(channel_id)]
    return rows, attrs


def iter_windows(path: str | Path) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """Yield (attributes, data) of every window of a per-window or combined file."""
    for spec in windows(path):
        file, index = split(spec)
        with _h5py().File(file, "r") as handle:
            data = handle[_product(handle)]
            block = data[()] if index is None else data[index]
        yield read_attrs(spec), block


def _half(coarse: list[int]) -> str:
    """Name a half by its coarse channels: the bound ones for fine powers, the first and count for the rest."""
    used = [c for c in coarse if c >= 0]
    if not used:
        return "idle"
    return "-".join(map(str, used)) if len(coarse) <= 4 else f"{used[0]}x{len(coarse)}"


def combine(acq_dir: str | Path, out_dir: str | Path, product: str, verify: bool = True) -> list[dict[str, Any]]:
    """Stack the per-window files of one product of one acquisition into one combined file per half.

    Windows are ordered by fpga_seq_time_nsec, or fpga_seq_num where no time is recorded, and keep their
    attributes as datasets. A half with no bound channel (no coarse_freq) is left out, as it holds no rows.
    With verify, every window is read back and compared with its source. Returns one summary per file written.
    """
    h5py = _h5py()
    import hdf5plugin

    acq_dir, out_dir = Path(acq_dir), Path(out_dir)
    sources = sorted(p for p in acq_dir.glob(f"dtv_{product}.*.h5") if not p.name.endswith(COMBINED))
    groups: dict[tuple[int, ...], list[tuple[int, int, Path]]] = defaultdict(list)
    skipped = 0
    for path in sources:
        attrs = read_attrs(path)
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
        with h5py.File(members[0][2], "r") as first:
            data0 = first[_product(first)]
            shape, dtype, static = data0.shape, data0.dtype, {k: v for k, v in data0.attrs.items() if k not in WINDOW_ATTRS}
        with h5py.File(tmp, "w") as out:
            data = out.create_dataset(f"dtv_{product}", shape=(len(members), *shape), dtype=dtype,
                                      chunks=(1, *shape), **hdf5plugin.Bitshuffle(cname="zstd"))
            for key, value in static.items():
                data.attrs[key] = value
            out["fpga_seq_num"] = np.array([s for _, s, _ in members], np.int64)
            out["fpga_seq_time_nsec"] = np.array([t for t, _, _ in members], np.int64)
            out["source"] = np.array([m[2].name for m in members], dtype=h5py.string_dtype())
            out.attrs.update(schema=SCHEMA, product=product, acquisition=acq_dir.name, windows=len(members))
            for i, (_, _, path) in enumerate(members):
                with h5py.File(path, "r") as source:
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
    return written
