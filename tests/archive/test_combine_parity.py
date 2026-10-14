# coding=utf-8
"""The combine step reproduces PilotProxy's multi-pilot detector products exactly.

Two synthetic channels are processed two ways: (1) PilotProxy's ``run_chime_analysis``
over both pilots at once (the reference), and (2) the archive detector analyzer
per pilot followed by ``combine_detector_products`` (the candidate). The combined
``chime_detector_outputs`` / ``chime_spectrogram_cache`` / ``chime_reductions_10s``
must match the single-process run byte-for-byte (same writers, same stacked
arrays). GPU-free via the same CPU-reference ``detector_fn`` pattern.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.chime.runner import run_chime_analysis
from pilot_proxy.chime.products import SCAN_INPUT_MANIFEST_SCHEMA_TOKEN
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED
from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    INT4_COMPONENT_BITS, coarse_power_ratio_cpu_reference, unpack_packed_complex,
)
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.archive.combine import combine_detector_products
from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.3125}  # ATSC channel -> coarse-center (MHz)
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
    assert set(a.files) == set(b.files), (set(a.files) ^ set(b.files))
    for key in a.files:
        x = np.asarray(a[key])
        y = np.asarray(b[key])
        assert x.shape == y.shape, f"{key}: {x.shape} != {y.shape}"
        if x.dtype.kind in "fc":
            assert np.array_equal(x, y, equal_nan=True), f"{key} differs"
        else:
            assert np.array_equal(x, y), f"{key} differs"


def test_combine_matches_multipilot_runner(tmp_path):
    rng = np.random.default_rng(7)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8) for ch in CHANNELS
    }

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    files = {}
    for ch, mhz in CHANNELS.items():
        p = input_dir / f"baseband_evt_{FREQ_IDS[ch]}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                            f_center_mhz=mhz, f_tone_bb=1200.0 + 10 * ch, seed=ch)
        with h5py.File(p, "a") as h:   # timed inputs -> combined sample_rate = 1/delta_time
            h.attrs["delta_time"] = 1.0 / fmt.FS
            h.attrs["time0_ctime"] = 1.0e9
        files[ch] = p

    profile = dataclasses.replace(
        default_reference_receiver_profile(frame_size_samples=NFFT,
                                           num_input_streams=N_FEEDS),
        spectral_sense=SPECTRAL_SENSE_INVERTED,
        stream_map_required=False,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    # reference: one runner pass over both pilots
    ref_dir = tmp_path / "ref"
    run_chime_analysis(
        input_dir=input_dir, output_dir=ref_dir, receiver_profile_path=profile_path,
        stream_map_path=None, physical_channels=sorted(CHANNELS),
        frame_size_samples=NFFT, detector_window_samples=K, frames_per_chunk=1,
        max_frames=N_FRAMES, kernel=_stub_kernel(K), detector_fn=_cpu_ref_detector_fn,
        weights_by_channel=weights_by_channel,
    )

    # candidate: analyzer per pilot, then combine
    inst = load_instrument("chime")
    reader = ChimeBasebandPackedReader()
    per_pilot = []
    for ch in sorted(CHANNELS):
        ctx = RunContext(instrument=inst, selection=[FREQ_IDS[ch]], options={
            "detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K),
            "weights": weights_by_channel[ch],
        })
        meta = dict(reader.probe(str(files[ch])))
        meta["unit_key"] = str(files[ch])  # real path -> shared event after freq_id strip
        meta["scope"] = "local"
        red = PilotProxyDetectorAnalyzer()
        red.begin(ctx, meta)
        red.consume_file(reader.iter_arrays(str(files[ch]), ctx), meta)
        out = tmp_path / "per_pilot" / f"{FREQ_IDS[ch]}.npz"
        red.save(str(out))
        per_pilot.append(out)

    out_dir = tmp_path / "combined"
    combine_detector_products(per_pilot, out_dir)

    for name in ("chime_detector_outputs.npz", "chime_spectrogram_cache.npz",
                 "chime_reductions_10s.npz"):
        _assert_npz_equal(ref_dir / name, out_dir / name)

    # F4: the new integrated_spectra stack matches each per-pilot product's spectra,
    # in the combined product's (sorted) channel order, with the freq axis metadata.
    spec = np.load(out_dir / "chime_integrated_spectra.npz")
    assert spec["integrated_spectrum_before_mask"].shape == (len(CHANNELS), NFFT)
    per_by_ch = {int(np.load(p)["physical_channel"][0]): np.load(p) for p in per_pilot}
    for i, ch in enumerate(np.asarray(spec["physical_channel"])):
        z = per_by_ch[int(ch)]
        assert np.array_equal(
            spec["integrated_spectrum_before_mask"][i],
            np.asarray(z["integrated_spectrum_before_mask"]).reshape(-1)), ch
        assert np.array_equal(
            spec["integrated_spectrum_after_mask"][i],
            np.asarray(z["integrated_spectrum_after_mask"]).reshape(-1)), ch
    assert int(spec["nfft"]) == NFFT
    assert float(spec["sample_rate_hz"]) == pytest.approx(fmt.FS)
    assert list(np.asarray(spec["freq_id"])) == [FREQ_IDS[ch] for ch in sorted(CHANNELS)]

    # F3: validate-products must accept the combined scan output unchanged.
    from pilot_proxy.chime.validate_products import validate_products
    report = validate_products(run_dir=out_dir)
    assert report["valid"], report["errors"]

    # prep for the deferred 6 MHz mask-expansion: each pilot's coarse-channel
    # freq_id is recorded in stats.json, in the same physical_channel order the
    # detector products are stacked (ch14 -> 844, ch15 -> 829).
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["freq_id_by_pilot"] == [FREQ_IDS[ch] for ch in sorted(CHANNELS)]
    input_manifest = json.loads(
        (out_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    assert input_manifest["schema_version"] == SCAN_INPUT_MANIFEST_SCHEMA_TOKEN
