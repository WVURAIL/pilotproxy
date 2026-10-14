# coding=utf-8
"""End-to-end parity for the `chime-scan` path.

``run_chime_scan`` drives the archive engine (local source -> per-pilot fan-out
-> ``pipeline.run``) and then combines. This checks its canonical detector
products match a single ``run_chime_analysis`` pass over the same two channels.
GPU-free: the detector / stub kernel / weights are injected via ``analyzer_options``
(the same hooks the runner exposes), so no CUDA is needed.

Files are named ``*_<freq_id>.h5`` so the local source maps each file to
its CHIME coarse channel (its default ``_(\\d+)\\.h5$`` regex), exactly like a real
CADC baseband file; the analyzer derives the ATSC channel label from the freq-attr
center. ``--select`` is therefore the freq_id list, while the weights and the
runner reference stay keyed on ATSC channel (one pilot per ATSC channel, so the
combine's ATSC ordering is a well-defined bijection with the selected freq_ids).
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt

from pilot_proxy.chime.runner import run_chime_analysis
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED
from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    INT4_COMPONENT_BITS, coarse_power_ratio_cpu_reference, unpack_packed_complex,
)
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
from pilot_proxy.archive.scan import run_chime_scan
from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz

NFFT = 16384
K = 128
N_FRAMES = 2
N_FILES = 2
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.3125}  # ATSC channel -> coarse-channel center (MHz)
# what those centers are as CHIME freq_id (the on-disk / inventory namespace)
FREQ_IDS = {ch: chime_freq_id_from_hz(mhz * 1e6) for ch, mhz in CHANNELS.items()}


def _cpu_ref_detector_fn(*, packed, weights, kernel):
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    w = unpack_packed_complex(np.asarray(weights, dtype=np.int8), INT4_COMPONENT_BITS)
    _nt, _nl, _nu = weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    _nrs = int(_nl + _nu)
    results = []
    for b in range(int(pk.shape[0])):
        samples = unpack_packed_complex(pk[b], INT4_COMPONENT_BITS)
        _f, sums = coarse_power_ratio_cpu_reference(samples, w)
        num = int(round(float(sums[0])))
        lower = int(round(float(sums[1])))
        upper = int(round(float(sums[2])))
        den = lower + upper
        results.append({"block_index": b, "mask": normalized_positive_excess(
                num, den, target_norm_sq=_nt, reference_norm_sum_sq=_nrs
            ),
                        "p_target_u64": num,
                        "p_ref_lower_u64": lower,
                        "p_ref_upper_u64": upper,
                        "p_ref_sum_u64": den})
    return {"batch": int(pk.shape[0]), "detector_rows_per_block": int(pk.shape[1]),
            "rational_overflow_count": 0, "results": results}


def _stub_kernel(k):
    specs = SimpleNamespace(K=k, N=3, bits=4, reference_offset_bins=2,
                            as_descriptive_dict=lambda: {"detector_window_samples": k,
                            "num_weight_terms": 3, "sample_bits_per_component": 4,
                            "reference_offset_bins": 2})
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _assert_npz_equal(ref_path, got_path):
    a = np.load(ref_path)
    b = np.load(got_path)
    assert set(a.files) == set(b.files), set(a.files) ^ set(b.files)
    for key in a.files:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.shape == y.shape, f"{key}: {x.shape} != {y.shape}"
        if x.dtype.kind in "fc":
            assert np.array_equal(x, y, equal_nan=True), f"{key} differs"
        else:
            assert np.array_equal(x, y), f"{key} differs"


def test_chime_scan_matches_runner(tmp_path):
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

    profile = dataclasses.replace(
        default_reference_receiver_profile(frame_size_samples=NFFT,
                                           num_input_streams=N_FEEDS),
        spectral_sense=SPECTRAL_SENSE_INVERTED,
        stream_map_required=False,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    # reference: single multi-pilot runner pass
    ref_dir = tmp_path / "ref"
    run_chime_analysis(
        input_dir=input_dir, output_dir=ref_dir, receiver_profile_path=profile_path,
        stream_map_path=None, physical_channels=sorted(CHANNELS),
        frame_size_samples=NFFT, detector_window_samples=K, frames_per_chunk=1,
        max_frames=N_FRAMES * N_FILES, kernel=_stub_kernel(K),
        detector_fn=_cpu_ref_detector_fn,
        weights_by_channel=weights_by_channel,
    )

    # Candidate: chime-scan fan-out and combine with the CPU reference.
    # --select is the freq_id list; the analyzer derives the ATSC label per file.
    scan_dir = tmp_path / "scan"
    freq_id_select = ",".join(str(f) for f in sorted(FREQ_IDS.values()))
    run_chime_scan(
        input_dir=input_dir, output_dir=scan_dir, source="local",
        analyzer="pilot-proxy-detector", select=freq_id_select, instrument="chime",
        download_workers=2, max_staged_files=2,
        analyzer_options={
            "detector_fn": _cpu_ref_detector_fn,
            "kernel": _stub_kernel(K),
            "weights_by_channel": weights_by_channel,
        },
        verbose=False,
    )

    for name in ("chime_detector_outputs.npz", "chime_spectrogram_cache.npz",
                 "chime_reductions_10s.npz"):
        _assert_npz_equal(ref_dir / name, scan_dir / name)
