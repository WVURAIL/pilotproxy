"""
Reader for CHIME and outrigger baseband HDF5 files (offset-binary 4+4-bit).

The bundled example runs on this reader. The on-disk format is held in
`baseband_format`, which gives the exact dataset and attribute layout, and the
reader yields the [nfft, n_feeds] complex64 frames that an analyzer consumes.
CHIME and its outriggers (KKO, GBO and HCO) share this product, so one reader
serves all four. A different file format needs its own reader, and this file
is the worked example in docs/ADDING_A_READER.md.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from pilot_proxy.archive.interfaces import (
    READY,
    STREAM_COMPLEX_BASEBAND,
    PluginInfo,
    Reader,
    RunContext,
)
from pilot_proxy.archive.names import baseband_filename

from . import baseband_format as fmt
from .unreadable import unreadable_file

# A truncated baseband object can still be valid HDF5, but an archive object
# smaller than 1 MiB cannot hold a useful fraction of the native payload. The
# floor is set by this format and is kept out of the generic CADC source, since
# other readers may describe much smaller products.
MINIMUM_ARCHIVE_BYTES = 1 << 20


class ChimeBasebandReader(Reader):
    # In survey schema 2 the frame count derived from the file size is named
    # n_frames_estimate. The archive size of an HDF5 object counts the container
    # overhead and may reflect compression, so the count is for planning only.
    survey_schema = 2
    minimum_archive_bytes = MINIMUM_ARCHIVE_BYTES

    info = PluginInfo(
        name="chime-baseband",
        kind="reader",
        summary="CHIME/outrigger baseband HDF5 (offset-binary 4+4-bit, "
                "dataset 'baseband', attr 'freq' in MHz).",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("h5py",),
        stream_kind=STREAM_COMPLEX_BASEBAND,
        notes="Yields [nfft, n_feeds] complex64 frames. Shared by CHIME and all "
              "CHIME-compatible outriggers.",
    )

    def probe(self, path: str) -> Mapping[str, object]:
        with unreadable_file():
            f_center_hz = fmt.channel_center_hz(path)
        return {"f_center_hz": f_center_hz,
                "f_center_mhz": f_center_hz / fmt.HZ_PER_MHZ,
                "fs_hz": fmt.FS}

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        nfft = int(getattr(ctx.instrument, "nfft", fmt.NFFT) or fmt.NFFT)
        if nfft < 1:
            raise ValueError("reader nfft must be positive")
        with unreadable_file():
            yield from fmt.iter_frames(path, nfft=nfft)

    # archive file shape
    # A baseband event has one HDF5 file per freq_id, so the survey's resolved
    # freq_id list `selection` (see CadcDatatrailSource.survey) gives the files.
    def survey_files(self, event, common_path, selection,
                     ctx: RunContext) -> Iterable[tuple]:
        for ch in (selection or ()):
            yield baseband_filename(event, ch), {"freq_id": int(ch)}

    def annotate_row(self, row: dict, instrument) -> None:
        """Add the channel frequency and an approximate frame count to a row.

        The packed payload is one byte per feed per sample, but ``size_bytes``
        is the size of the whole HDF5 object. The frame count is therefore
        reported as a planning estimate in whole frames.
        """
        if instrument is None or row.get("freq_id") is None:
            return
        ch = int(row["freq_id"])
        bytes_per_frame = instrument.nfft * instrument.n_feeds
        row["freq_mhz"] = round(instrument.freq_of_freq_id(ch), 4)
        row["n_frames_estimate"] = int(
            row.get("size_bytes", 0) // bytes_per_frame)
