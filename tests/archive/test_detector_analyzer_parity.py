# coding=utf-8
"""GPU-free parity test for the `pilot-proxy-detector` analyzer.

The CUDA kernel is GPU-only, so this test does what PilotProxy's own runner test does:
inject a CPU-reference ``detector_fn`` + a stub kernel + explicit weights. The
*reference* is PilotProxy's ``run_chime_analysis`` (the batch runner); the *candidate*
is the archive analyzer streaming the same file through the same injected pieces.
They must produce the same per-frame ``chime_detector_outputs`` arrays, which
validates the analyzer's packing / windowing / accumulation / schema. The actual
fixed-point kernel + real-data parity is a separate CANFAR/GPU step.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.chime.runner import run_chime_analysis
from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED
from pilot_proxy.detector_reference import (
    INT4_COMPONENT_BITS,
    coarse_power_ratio_cpu_reference,
    unpack_packed_complex,
)
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
PHYS_CH = 14
F_CENTER_MHZ = 470.3125
FREQ_ID = 844  # coarse channel for 470.3125 MHz (== chime_freq_id_from_hz)


def _cpu_ref_detector_fn(*, packed, weights, kernel):
    """Input-dependent CPU reference: float F-stat power sums per block."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    w_packed = np.asarray(weights, dtype=np.int8)
    w = unpack_packed_complex(w_packed, INT4_COMPONENT_BITS)
    nt, nl, nu = weight_term_norms_sq(w_packed)
    nrs = int(nl + nu)
    results = []
    for b in range(int(pk.shape[0])):
        samples = unpack_packed_complex(pk[b], INT4_COMPONENT_BITS)
        _fstat, sums = coarse_power_ratio_cpu_reference(samples, w)
        num = int(round(float(sums[0])))
        lower = int(round(float(sums[1])))
        upper = int(round(float(sums[2])))
        den = lower + upper
        results.append({
            "block_index": b,
            "mask": normalized_positive_excess(
                num, den, target_norm_sq=nt, reference_norm_sum_sq=nrs
            ),
            "p_target_u64": num,
            "p_ref_lower_u64": lower,
            "p_ref_upper_u64": upper,
            "p_ref_sum_u64": den,
        })
    return {
        "batch": int(pk.shape[0]),
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "results": results,
    }


def _stub_kernel(detector_window_samples: int):
    specs = SimpleNamespace(
        K=detector_window_samples, N=3, bits=4, reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": detector_window_samples,
            "num_weight_terms": 3, "sample_bits_per_component": 4,
            "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


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
        spectral_sense=SPECTRAL_SENSE_INVERTED,
        stream_map_required=False,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    # reference: PilotProxy's own runner with the CPU-reference detector
    ref_dir = tmp_path / "ref"
    run_chime_analysis(
        input_dir=input_dir, output_dir=ref_dir,
        receiver_profile_path=profile_path, stream_map_path=None,
        physical_channels=[PHYS_CH], frame_size_samples=NFFT,
        detector_window_samples=K, frames_per_chunk=1, max_frames=N_FRAMES,
        kernel=_stub_kernel(K), detector_fn=_cpu_ref_detector_fn,
        weights_by_channel={PHYS_CH: weights},
    )
    ref = np.load(ref_dir / "chime_detector_outputs.npz")

    # Candidate: the archive analyzer with the same file and injected pieces.
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
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

    # exact-equal integer arrays. The runner's canonical output keeps the legacy
    # `mask` field; the analyzer's per-channel product renames it `reject_mask`
    # (same values, 1 = discard), so compare those two across the name change.
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
