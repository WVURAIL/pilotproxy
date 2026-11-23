# coding=utf-8
"""End-to-end parity for the `chime-scan` path.

``run_chime_scan`` drives the archive engine (the local source, the per-pilot
fan-out and ``pipeline.run``) and then combines the products. This test checks
that its canonical detector powers equal those computed directly from the
files' bytes (``cpu_detector.reference_fine_powers``) for two channels. The
detector, stub kernel and weights are injected through ``analyzer_options``, so
no GPU or CUDA is needed.

Files are named ``*_<freq_id>.h5`` so the local source maps each file to its
CHIME coarse channel with its default ``_(\\d+)\\.h5$`` regex, as for a real
CADC baseband file. The analyzer derives the ATSC channel label from the center
frequency in the file's freq attribute. ``--select`` is therefore the freq_id
list, and the weights stay keyed on ATSC channel. With one pilot per ATSC
channel, the ATSC ordering of combine maps one to one onto the selected
freq_ids.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt

from pilot_proxy.archive.scan import run_chime_scan
from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz

from cpu_detector import exact_cpu_detector_fn, reference_fine_powers, stub_kernel  # noqa: E402

NFFT = 16384
K = 128
N_FRAMES = 2
N_FILES = 2
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.171875}  # coarse-channel center (MHz) by ATSC channel
# the same centers as CHIME freq_id, the namespace of the files and inventory
FREQ_IDS = {ch: chime_freq_id_from_hz(mhz * 1e6) for ch, mhz in CHANNELS.items()}


def test_chime_scan_matches_cpu_reference(tmp_path):
    rng = np.random.default_rng(3)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8) for ch in CHANNELS
    }

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    for ch, mhz in CHANNELS.items():
        for file_index in range(N_FILES):
            event = 1000 + file_index
            fmt.make_synth_file(
                str(input_dir / f"baseband_{event}_{FREQ_IDS[ch]}.h5"),
                n_time=NFFT * N_FRAMES,
                n_feeds=N_FEEDS,
                f_center_mhz=mhz,
                f_tone_bb=1300.0 + 7 * ch,
                seed=ch * 10 + file_index,
            )

    # chime-scan fan-out and combine with the CPU reference detector
    # --select is the freq_id list, and the analyzer derives the ATSC label per file
    scan_dir = tmp_path / "scan"
    freq_id_select = ",".join(str(f) for f in sorted(FREQ_IDS.values()))
    run_chime_scan(
        input_dir=input_dir, output_dir=scan_dir, source="local",
        analyzer="pilot-proxy-detector", select=freq_id_select, instrument="chime",
        download_workers=2, max_staged_files=2,
        analyzer_options={
            "detector_fn": exact_cpu_detector_fn,
            "kernel": stub_kernel(K),
            "weights_by_channel": weights_by_channel,
        },
        verbose=False,
    )

    got = np.load(scan_dir / "chime_detector_outputs.npz")
    for pilot, ch in enumerate(sorted(CHANNELS)):
        files = sorted(input_dir.glob(f"*_{FREQ_IDS[ch]}.h5"))
        powers = np.concatenate(
            [reference_fine_powers(f, weights_by_channel[ch]) for f in files]
        ).sum(axis=-1, dtype=np.uint64)
        for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
            np.testing.assert_array_equal(got[name][:, pilot], powers[:, term], err_msg=f"{ch} {name}")
    assert got["p_target_u64"].shape == (N_FRAMES * N_FILES, len(CHANNELS))
    for name in ("chime_spectrogram_cache.npz", "chime_reductions_10s.npz"):
        assert (scan_dir / name).is_file(), name
