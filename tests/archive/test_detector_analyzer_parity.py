# coding=utf-8
"""GPU-free parity test for the `pilot-proxy-detector` analyzer.

The CUDA kernel runs only on a GPU, so a CPU-reference ``detector_fn``, a
stub kernel, and explicit weights are injected, as in the PilotProxy runner
test. The reference is the batch runner ``run_chime_analysis``, and the
candidate is the archive analyzer streaming the same file through the same
injected pieces. Both must produce the same per-frame ``chime_detector_outputs``
arrays, which tests the packing, windowing, accumulation, and output schema of
the analyzer. Parity of the fixed-point kernel on real data is tested in its
own step on a CANFAR GPU.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.chime.runner import run_chime_analysis
from pilot_proxy.detector_geometry import MIRRORED_NYQUIST_ZONE
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader

from cpu_detector import exact_cpu_detector_fn, stub_kernel  # noqa: E402

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
PHYS_CH = 14
F_CENTER_MHZ = 470.3125
FREQ_ID = 844  # coarse channel for 470.3125 MHz (== chime_freq_id_from_hz)


def test_detector_analyzer_matches_runner(tmp_path):
    rng = np.random.default_rng(11)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)

    # synthetic native baseband (offset-binary 4+4-bit), one channel
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    synth = input_dir / f"baseband_evt_{FREQ_ID}.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=5)

    # inverted-sense receiver profile (matches the chime instrument's sense=-1)
    profile = dataclasses.replace(
        default_reference_receiver_profile(frame_size_samples=NFFT,
                                           num_input_streams=N_FEEDS),
        nyquist_zone=MIRRORED_NYQUIST_ZONE,
        stream_map_required=False,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    # reference run of the batch runner with the CPU-reference detector
    ref_dir = tmp_path / "ref"
    run_chime_analysis(
        input_dir=input_dir, output_dir=ref_dir,
        receiver_profile_path=profile_path, stream_map_path=None,
        physical_channels=[PHYS_CH], frame_size_samples=NFFT,
        detector_window_samples=K, frames_per_chunk=1, max_frames=N_FRAMES,
        kernel=stub_kernel(K), detector_fn=exact_cpu_detector_fn,
        weights_by_channel={PHYS_CH: weights},
    )
    ref = np.load(ref_dir / "chime_detector_outputs.npz")

    # candidate run of the archive analyzer on the same file and injected pieces
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": exact_cpu_detector_fn,
        "kernel": stub_kernel(K),
        "weights": weights,
    })
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = "synth:dtv14"
    meta["scope"] = "local"
    red = PilotProxyDetectorAnalyzer()
    red.begin(ctx, meta)
    red.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    out = tmp_path / "out" / "14.npz"
    red.save(str(out))
    got = np.load(out)

    assert int(got["physical_channel"][0]) == PHYS_CH
    assert int(got["freq_id"][0]) == FREQ_ID
    assert got["p_target_u64"].shape == ref["p_target_u64"].shape == (N_FRAMES, 1)

    # The integer arrays must match exactly. The canonical runner output keeps
    # the legacy name `mask` for the frame mask, and the per-channel product of
    # the analyzer names it `reject_mask`. The values are the same (1 = discard),
    # so these two are compared across the name change.
    for name in ("p_target_u64", "p_ref_sum_u64", "valid", "frame_index"):
        assert np.array_equal(np.asarray(ref[name]), np.asarray(got[name])), name
    assert np.array_equal(
        np.asarray(ref["mask"]), np.asarray(got["reject_mask"])
    ), "reject_mask"

    # float metrics
    for name in ("coarse_power_ratio", "normalized_coarse_power_ratio_db", "pilot_excess_db", "estimated_data_shelf_snr_db",
                 "pilot_frequency_hz", "chime_frequency_hz"):
        a = np.asarray(ref[name], dtype=np.float64)
        b = np.asarray(got[name], dtype=np.float64)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"{name}: max|abs|={np.nanmax(np.abs(a - b)):.3e}"
        )
