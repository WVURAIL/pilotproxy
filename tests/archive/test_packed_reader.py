# coding=utf-8
"""The native-packed reader yields the same samples as the bundled unpacking reader.

The samples stay in raw 4+4-bit form, so the kernel packing keeps the int4
grid.
"""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.archive.packed_reader import (
    ChimeBasebandPackedReader,
    _dataset_content_sha256,
)

NFFT = 16384
N_FRAMES = 3
N_FEEDS = 8


def test_packed_reader_matches_unpacked(tmp_path):
    synth = tmp_path / "chime_synth.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=3)

    ctx = RunContext(instrument=load_instrument("chime"))
    packed_reader = ChimeBasebandPackedReader()
    unpacked_reader = ChimeBasebandReader()

    # The probe of the packed reader has every channel and format key of the
    # bundled reader, plus a per-unit absolute-time axis and receiver-state
    # identifiers read from the root attrs of the file.
    unpacked_probe = unpacked_reader.probe(str(synth))
    packed_probe = packed_reader.probe(str(synth))
    for k, v in unpacked_probe.items():
        assert packed_probe[k] == v, k
    for k in (
        "time0_ctime",
        "delta_time",
        "time0_fpga_count",
        "event_id",
        "archive_version",
        "git_version_tag",
        "collection_server",
        "input_map_sha256",
    ):
        assert k in packed_probe, k

    packed_chunks = list(packed_reader.iter_arrays(str(synth), ctx))
    unpacked_chunks = list(unpacked_reader.iter_arrays(str(synth), ctx))
    assert len(packed_chunks) == len(unpacked_chunks) == N_FRAMES

    for raw, cplx in zip(packed_chunks, unpacked_chunks):
        assert raw.dtype == np.uint8
        assert raw.shape == (NFFT, N_FEEDS)
        # unpacking the raw bytes must reproduce the bundled reader's complex chunk
        assert np.array_equal(fmt.unpack_4bit(raw), cplx)


def test_probe_retains_receiver_provenance(tmp_path):
    synth = tmp_path / "receiver.h5"
    fmt.make_synth_file(
        str(synth),
        n_time=NFFT,
        n_feeds=N_FEEDS,
        f_center_mhz=470.3125,
        f_tone_bb=1500.0,
        seed=12,
        receiver_provenance=False,
    )
    input_map = np.asarray(
        [(2, b"input-2"), (5, b"input-5")],
        dtype=[("chan_id", "<u2"), ("correlator_input", "S16")],
    )
    with h5py.File(synth, "a") as h:
        h.attrs["git_version_tag"] = "receiver-build"
        h.attrs["collection_server"] = "host-a"
        h.create_dataset("index_map/input", data=input_map)
        expected = _dataset_content_sha256(h["index_map/input"])

    probe = ChimeBasebandPackedReader().probe(str(synth))

    assert probe["git_version_tag"] == "receiver-build"
    assert probe["collection_server"] == "host-a"
    assert probe["input_map_sha256"] == expected
    assert expected == "244b14115600631f9525d0c20495f4a8174ebe49d58339bf59e1cb9c890faf9a"


def test_input_map_hash_changes_with_content(tmp_path):
    paths = [tmp_path / "first.h5", tmp_path / "second.h5"]
    hashes = []
    for index, path in enumerate(paths):
        fmt.make_synth_file(
            str(path),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=470.3125,
            f_tone_bb=1500.0,
            seed=20 + index,
            receiver_provenance=False,
        )
        with h5py.File(path, "a") as h:
            h.create_dataset("index_map/input", data=np.asarray([index], dtype="<u2"))
        hashes.append(ChimeBasebandPackedReader().probe(str(path))["input_map_sha256"])

    assert hashes[0] != hashes[1]


def _ctx_with_nfft(nfft):
    """Return a RunContext with a per-run nfft override.

    The override sets ctx.instrument.nfft, the attribute set by the CLI --nfft
    option and read by iter_arrays.
    """
    inst = load_instrument("chime")
    inst.nfft = nfft
    return RunContext(instrument=inst)


def test_probe_rejects_short_file_as_unreadable(tmp_path):
    """A file shorter than one transform is quarantined and the run continues.

    In the 2026-08-23 pre-flight, a 3.9 MB stub in the CANFAR staging aborted a
    scan that had already completed eight channels. The analyzer raised on zero
    frames, and the engine treats analyzer exceptions as run-level errors by
    design. When the file is classified at probe time, the engine quarantines it
    and the run continues. The iteration-time check below covers the
    configurations that probe cannot see.
    """
    from pilot_proxy.archive.interfaces import UnreadableUnitError

    short = tmp_path / "too_short.h5"
    fmt.make_synth_file(str(short), n_time=NFFT - 1, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=5)
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        ChimeBasebandPackedReader().probe(str(short))


def test_nfft_above_default_quarantines_at_iteration(tmp_path):
    """With an instrument nfft above the format default, the file is quarantined at iteration.

    A file with one default-length frame has zero frames at the nfft of the
    run. iter_arrays must raise the quarantinable error type before the
    run-fatal zero-frame check of the analyzer is reached.
    """
    from pilot_proxy.archive.interfaces import UnreadableUnitError

    path = tmp_path / "one_default_frame.h5"
    fmt.make_synth_file(str(path), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=8)
    reader = ChimeBasebandPackedReader()
    reader.probe(str(path))
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        list(reader.iter_arrays(str(path), _ctx_with_nfft(2 * NFFT)))


def test_nfft_below_default_is_deliberately_conservative(tmp_path):
    """With an instrument nfft below the format default, probe still quarantines.

    Probe never sees the configured nfft, so its gate uses the format default
    that every shipped instrument frames at. A file that could frame at a
    smaller configured nfft is therefore quarantined at probe. This is the
    accepted cost of catching stubs without interrupting an unattended
    multi-week scan, and no shipped instrument sets nfft below the default.
    This test pins that trade, so a future instrument with a smaller nfft must
    revisit it.
    """
    from pilot_proxy.archive.interfaces import UnreadableUnitError

    small = NFFT // 4
    path = tmp_path / "two_small_frames.h5"
    fmt.make_synth_file(str(path), n_time=2 * small, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=9)
    reader = ChimeBasebandPackedReader()
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        reader.probe(str(path))
    chunks = list(reader.iter_arrays(str(path), _ctx_with_nfft(small)))
    assert len(chunks) == 2
    assert all(c.shape == (small, N_FEEDS) for c in chunks)


def test_probe_accepts_exactly_one_frame(tmp_path):
    """The boundary is inclusive, so exactly nfft samples make one usable frame."""
    exact = tmp_path / "exactly_one.h5"
    fmt.make_synth_file(str(exact), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=6)
    probe = ChimeBasebandPackedReader().probe(str(exact))
    assert probe["num_input_streams"] == N_FEEDS
    ctx = RunContext(instrument=load_instrument("chime"))
    assert len(list(ChimeBasebandPackedReader().iter_arrays(str(exact), ctx))) == 1


def test_unpack_4bit_reads_the_high_nibble_as_real():
    b = np.arange(256, dtype=np.uint8).reshape(16, 16)
    x = fmt.unpack_4bit(b)
    assert x.dtype == np.complex64 and x.shape == b.shape
    assert x[0, 0] == complex(-8, -8) and x[15, 15] == complex(7, 7)
    assert fmt.unpack_4bit(np.array([0x9A], np.uint8))[0] == complex(1, 2)    # high nibble real, low imaginary
