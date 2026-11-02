# coding=utf-8
"""Read CHIME/outrigger baseband HDF5 as native packed bytes.

The complex reader unpacks each chunk to complex64. The local-reference
power-ratio detector instead needs the original 4-bit nibbles, so that they can
be repacked without loss into the two's-complement int4 layout of the kernel
(with ``pack_chime_block_for_detector``) and with no float requantization. This
reader reads the same layout and yields raw ``uint8 [nfft, n_feeds]`` blocks.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterator, Mapping

import h5py
import numpy as np

from pilot_proxy.archive.interfaces import READY, PluginInfo, Reader, RunContext
from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.chime.unreadable import unreadable_file

from .stream_kinds import STREAM_PACKED_COMPLEX_INT4_BASEBAND


def _dtype_signature(dtype: np.dtype) -> object:
    if dtype.names:
        return {
            "fields": [
                [name, _dtype_signature(dtype.fields[name][0]), dtype.fields[name][1]]
                for name in dtype.names
            ],
            "itemsize": dtype.itemsize,
        }
    if dtype.subdtype:
        base, shape = dtype.subdtype
        return {"base": _dtype_signature(base), "shape": list(shape)}
    return dtype.str


def _dataset_content_sha256(dataset: h5py.Dataset) -> str:
    """Hash an HDF5 dataset's typed values in a stable order."""
    values = np.asarray(dataset[...])
    if values.dtype.hasobject:
        raise ValueError("index_map/input must not contain object values")
    header = json.dumps(
        {"dtype": _dtype_signature(values.dtype), "shape": list(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _iter_packed_chunks(path: str, nfft: int) -> Iterator:
    """Yield raw uint8 [nfft, n_feeds] blocks, dropping the final partial frame.

    A file too short for one complete transform at this nfft raises
    UnreadableUnitError through unreadable_file(). When it is raised while the
    reader streams, it is the only iteration error that the engine quarantines.
    The unit is written to the quarantine ledger, and a rerun resumes without
    it. If the analyzer saw zero frames instead, the same bytes would raise a
    run-level ValueError that cannot be quarantined. The check is made here,
    at the nfft used for framing, so the reject and framing criteria always
    agree. A probe-time check at the format default cannot guarantee this.
    """
    with unreadable_file(), h5py.File(path, "r") as h:
        bb = h["baseband"]
        n_chunks = int(bb.shape[0]) // int(nfft)
        if n_chunks == 0:
            raise ValueError(
                "CHIME packed baseband is shorter than one transform: "
                f"{int(bb.shape[0])} time samples < nfft {int(nfft)}; "
                "it cannot yield a complete frame"
            )
        for c in range(n_chunks):
            yield bb[c * nfft:(c + 1) * nfft, :]


class ChimeBasebandPackedReader(Reader):
    info = PluginInfo(
        name="chime-baseband-packed",
        kind="reader",
        summary="CHIME/outrigger baseband HDF5 yielded as raw offset-binary 4+4-bit "
                "uint8 (no unpack); for the F-stat detector's lossless re-pack.",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("h5py", "pilot-proxy"),
        stream_kind=STREAM_PACKED_COMPLEX_INT4_BASEBAND,
        notes="Same on-disk format as 'chime-baseband'; yields uint8 [nfft, n_feeds] "
              "instead of complex64 so the kernel packing keeps the native int4 grid.",
    )

    def probe(self, path: str) -> Mapping[str, object]:
        # open and schema failures mark the unit unreadable (quarantine) and
        # the scan continues, as in the 'chime-baseband' reader
        with unreadable_file():
            return self._probe(path)

    @staticmethod
    def _probe(path: str) -> Mapping[str, object]:
        f_center_hz = fmt.channel_center_hz(path)
        meta: dict = {"f_center_hz": f_center_hz,
                      "f_center_mhz": f_center_hz / 1e6,
                      "fs_hz": fmt.FS,
                      "nfft": fmt.NFFT}
        # The absolute-time axis and provenance are read from the root
        # attributes of the file. The detector uses them to stamp a per-unit
        # time and to derive a per-frame time as
        # time0_ctime + frame_in_unit*nfft*delta_time. A real CHIME baseband
        # file carries the full set, and a missing attribute is returned as
        # NaN, 0, None, or "" so that the probe does not fail.
        with h5py.File(path, "r") as h:
            a = h.attrs
            baseband = h["baseband"]
            if baseband.ndim != 2:
                raise ValueError(
                    "CHIME packed baseband must have shape (time, input_stream); "
                    f"got {baseband.shape}."
                )
            meta["num_input_streams"] = int(baseband.shape[1])
            # A file too short to yield one complete transform is a property of
            # the staged bytes, so it is rejected here. unreadable_file() turns
            # the error into UnreadableUnitError, and the engine quarantines the
            # file while the run continues. The framing length is
            # ctx.instrument.nfft, and the engine calls probe without a ctx, so
            # this check can use only the format default fmt.NFFT. Every
            # shipped instrument frames at that value. An instrument configured
            # below it would have files quarantined here that could still be
            # framed. No such instrument exists, and the check is kept strict
            # for the unattended multi-week scan. _iter_packed_chunks checks the
            # per-run length again at the exact nfft used for framing, so for
            # an instrument nfft above the default the file is quarantined at
            # iteration time (a resumable stop) and the run is not aborted by
            # the analyzer.
            if int(baseband.shape[0]) < int(fmt.NFFT):
                raise ValueError(
                    "CHIME packed baseband is shorter than one transform: "
                    f"{int(baseband.shape[0])} time samples < nfft {int(fmt.NFFT)}; "
                    "it cannot yield a complete frame"
                )

            def _f(name: str) -> float:
                return float(a[name]) if name in a else float("nan")

            def _s(name: str) -> str:
                if name not in a:
                    return ""
                v = a[name]
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)

            meta["time0_ctime"] = _f("time0_ctime")
            meta["time0_ctime_offset"] = _f("time0_ctime_offset")
            meta["delta_time"] = _f("delta_time")
            meta["time0_fpga_count"] = int(a["time0_fpga_count"]) if "time0_fpga_count" in a else 0
            meta["event_id"] = int(a["event_id"]) if "event_id" in a else None
            meta["archive_version"] = _s("archive_version")
            meta["git_version_tag"] = _s("git_version_tag")
            meta["collection_server"] = _s("collection_server")
            meta["input_map_sha256"] = (
                _dataset_content_sha256(h["index_map/input"])
                if "index_map/input" in h
                else ""
            )
        return meta

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        nfft = int(getattr(ctx.instrument, "nfft", fmt.NFFT) or fmt.NFFT)
        return _iter_packed_chunks(path, nfft)


__all__ = ["ChimeBasebandPackedReader"]
