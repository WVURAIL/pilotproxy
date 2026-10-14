# coding=utf-8
"""Current per-pilot product contents for the PilotProxy analyzer.

The product folds three derived-from-the-same-pass products into one ``.npz`` so
the week-long CANFAR run never needs a second pass:

  * two integrated power spectra (rectangular-window |FFT|^2 summed over feeds),
    one over every valid frame and one over kept (not-rejected) frames, so their
    difference is the spectrum the positive-excess mask removed;
  * a per-unit absolute-time axis (time0_ctime / delta_time / fpga / event_id,
    surfaced by the reader from the file root attrs) plus per-frame unit tags, so
    each frame's wall time is t = time0 + frame_in_unit*nfft*delta_time without a
    timestamp per frame;
  * run provenance (weights_hash / detector_version / mask_rule).

The spectrum FFT is xp-generic (cupy on a GPU node, numpy here); these tests pin
the numpy path against an independent FFT, which is the same arithmetic.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py  # noqa: E402

from pilot_proxy.chime import baseband_format as fmt  # noqa: E402
from pilot_proxy.archive.instruments import load_instrument  # noqa: E402
from pilot_proxy.archive.interfaces import RunContext  # noqa: E402

from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE  # noqa: E402
from pilot_proxy.product_contract import (  # noqa: E402
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    current_decision_contract,
)
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer  # noqa: E402
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader  # noqa: E402

NFFT = 16384
K = 128
N_FEEDS = 4
F_CENTER_MHZ = 470.3125   # -> freq_id 844, ATSC ch14 pilot in band
FREQ_ID = 844

# CHIME baseband root attrs (values from a real freq_id 844 file)
CTIME = 1772288626.643602
DELTA = 1.0 / fmt.FS          # 2.56e-06 s
FPGA = 156722907657
EVENT_ID = 1151515151
ARCHIVE = "NT_3.1.0"
GIT_VERSION_TAG = "receiver-build"
COLLECTION_SERVER = "host-a"


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


def _detector_fn_factory(reject):
    """CPU stand-in. `reject(frame_index)` -> bool drives the per-frame mask; all
    frames are valid (p_ref_sum > 0) so they enter the integrated spectra."""
    state = {"i": 0}

    def _fn(*, packed, weights, kernel):
        pk = np.asarray(packed)
        if pk.ndim == 2:
            pk = pk[None, ...]
        results = []
        for b in range(int(pk.shape[0])):
            should_reject = bool(reject(state["i"]))
            results.append({
                "block_index": b,
                "mask": int(should_reject),
                # Canonical reject bits are derived from powers + exact norms,
                # so make this stand-in unambiguously high or low.
                "p_target_u64": 1_000_000 if should_reject else 0,
                "p_ref_sum_u64": 100,
                "p_ref_lower_u64": 50,
                "p_ref_upper_u64": 50,
            })
            state["i"] += 1
        return {"batch": int(pk.shape[0]),
                "detector_rows_per_block": int(pk.shape[1]),
                "rational_overflow_count": 0, "results": results}
    return _fn


def _ctx(detector_fn, weights):
    return RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": detector_fn, "kernel": _stub_kernel(K), "weights": weights,
    })


def _weights(seed=1):
    return np.random.default_rng(seed).integers(-120, 121, size=(3, K)).astype(np.int8)


def _add_timing(path, *, ctime=CTIME, delta=DELTA, fpga=FPGA, event_id=EVENT_ID,
                archive=ARCHIVE, git_version_tag=GIT_VERSION_TAG,
                collection_server=COLLECTION_SERVER):
    with h5py.File(path, "a") as h:
        h.attrs["time0_ctime"] = ctime
        h.attrs["time0_ctime_offset"] = -1.44e-08
        h.attrs["delta_time"] = delta
        h.attrs["time0_fpga_count"] = fpga
        h.attrs["event_id"] = event_id
        h.attrs["archive_version"] = archive
        h.attrs["git_version_tag"] = git_version_tag
        h.attrs["collection_server"] = collection_server


def _run(path, ctx, reader=None):
    reader = reader or ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"
    a = PilotProxyDetectorAnalyzer()
    a.begin(ctx, meta)
    a.consume_file(reader.iter_arrays(str(path), ctx), meta)
    return a, reader


def _expected_spectrum(reader, path, ctx, keep=lambda i: True):
    """Independent numpy reference for the integrated spectrum."""
    acc = np.zeros(NFFT, dtype=np.float64)
    for i, fr in enumerate(reader.iter_arrays(str(path), ctx)):
        if not keep(i):
            continue
        X = np.fft.fft(fmt.unpack_4bit(fr), axis=0)
        acc += (np.abs(X) ** 2).sum(axis=1, dtype=np.float64)
    return acc


# -- integrated spectra -------------------------------------------------------

def test_integrated_spectrum_matches_independent_fft_and_peaks_at_tone(tmp_path):
    bin_k = 64
    f_tone = bin_k * fmt.FS / NFFT            # exactly on bin 64 -> no leakage
    synth = tmp_path / "tone.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * 3, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=f_tone,
                        tone_amp=7.0, noise_std=0.5, seed=2)

    ctx = _ctx(_detector_fn_factory(reject=lambda i: False), _weights())
    a, reader = _run(synth, ctx)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)

    before = np.asarray(got["integrated_spectrum_before_mask"], dtype=np.float64)
    after = np.asarray(got["integrated_spectrum_after_mask"], dtype=np.float64)
    assert before.shape == after.shape == (NFFT,)

    # cuFFT (GPU) and numpy's FFT are both single precision (complex64 in ->
    # complex64 out) but use different algorithms, so they agree to ~1e-6 relative,
    # not float64. Use a tolerance that admits that backend difference while still
    # catching real logic bugs (orders of magnitude off); argmax and the
    # before/after structural checks pin correctness independently.
    expected = _expected_spectrum(reader, synth, ctx)
    scale = float(before.max())
    assert np.allclose(before, expected, rtol=1e-4, atol=1e-3 * scale)
    # nothing rejected -> after == before (same backend, identical accumulation)
    assert np.array_equal(before, after)
    # the injected tone dominates the integrated spectrum at its bin
    assert int(np.argmax(before)) == bin_k


def test_spectrum_after_mask_drops_rejected_frames(tmp_path):
    synth = tmp_path / "two.h5"
    # 2 frames, distinct tones so the rejected frame's spectrum is identifiable
    fmt.make_synth_file(str(synth), n_time=NFFT * 2, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=64 * fmt.FS / NFFT,
                        tone_amp=7.0, noise_std=0.5, seed=5)

    reject_first = lambda i: i == 0
    ctx = _ctx(_detector_fn_factory(reject=reject_first), _weights())
    a, reader = _run(synth, ctx)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)

    before = np.asarray(got["integrated_spectrum_before_mask"], dtype=np.float64)
    after = np.asarray(got["integrated_spectrum_after_mask"], dtype=np.float64)
    reject = np.asarray(got["reject_mask"]).reshape(-1)
    assert list(reject) == [1, 0]                      # frame 0 rejected, frame 1 kept

    # before = both frames; after = kept frame only; difference = rejected frame.
    # Loose rtol/atol for the cuFFT-vs-numpy single-precision FFT difference (see the
    # matched-FFT test); the structural reject=[1,0] split is the real check.
    scale = float(before.max())
    tol = dict(rtol=1e-4, atol=1e-3 * scale)
    assert np.allclose(before, _expected_spectrum(reader, synth, ctx), **tol)
    assert np.allclose(after, _expected_spectrum(reader, synth, ctx, keep=lambda i: i == 1), **tol)
    dropped = _expected_spectrum(reader, synth, ctx, keep=lambda i: i == 0)
    assert np.allclose(before - after, dropped, **tol)


def test_out_of_band_channel_has_zero_spectra(tmp_path):
    # freq_id 400 (643.75 MHz) -> nearest ATSC pilot is out of band -> all-invalid;
    # no valid frames -> both spectra stay zero.
    synth = tmp_path / "oob.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * 2, n_feeds=N_FEEDS,
                        f_center_mhz=643.75, f_tone_bb=1500.0, seed=3)
    ctx = RunContext(instrument=load_instrument("chime"), selection=[400], options={
        "detector_fn": _detector_fn_factory(reject=lambda i: False),
        "kernel": _stub_kernel(K), "weights": _weights(),
    })
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = "synth:oob"
    meta["scope"] = "local"
    a = PilotProxyDetectorAnalyzer()
    with pytest.warns(RuntimeWarning, match="does not contain"):
        a.begin(ctx, meta)
    a.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    out = tmp_path / "400.npz"
    a.save(str(out))
    got = np.load(out)
    assert not np.any(np.asarray(got["integrated_spectrum_before_mask"]))
    assert not np.any(np.asarray(got["integrated_spectrum_after_mask"]))


# -- per-unit time axis + per-frame derivation --------------------------------

def test_time_axis_from_root_attrs_and_frame_derivation(tmp_path):
    synth = tmp_path / "timed.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * 4, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=4)
    _add_timing(synth)

    ctx = _ctx(_detector_fn_factory(reject=lambda i: False), _weights())
    a, _ = _run(synth, ctx)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)

    # per-unit axis (one unit), aligned to unit_order
    assert got["unit_time0_ctime"].shape == (1,)
    assert float(got["unit_time0_ctime"][0]) == pytest.approx(CTIME, abs=0)
    assert float(got["unit_delta_time"][0]) == pytest.approx(DELTA, abs=0)
    assert int(got["unit_time0_fpga"][0]) == FPGA
    assert int(got["unit_event_id"][0]) == EVENT_ID
    assert str(got["archive_version"][0]) == ARCHIVE
    assert str(got["unit_git_version_tag"][0]) == GIT_VERSION_TAG
    assert str(got["unit_collection_server"][0]) == COLLECTION_SERVER
    assert len(str(got["unit_input_map_sha256"][0])) == 64
    assert str(got["unit_scope"][0]) == "local"

    # per-frame tags: one unit, contiguous chunk positions
    n = int(got["frame_index"].shape[0])
    assert n == 4
    assert list(np.asarray(got["frame_unit_index"]).reshape(-1)) == [0, 0, 0, 0]
    assert list(np.asarray(got["frame_in_unit"]).reshape(-1)) == [0, 1, 2, 3]

    # documented per-frame absolute time
    fui = np.asarray(got["frame_unit_index"]).reshape(-1)
    fiu = np.asarray(got["frame_in_unit"]).reshape(-1)
    t0 = np.asarray(got["unit_time0_ctime"])[fui]
    dt = np.asarray(got["unit_delta_time"])[fui]
    t = t0 + fiu * NFFT * dt
    assert np.allclose(np.diff(t), NFFT * DELTA)
    assert t[0] == pytest.approx(CTIME, abs=0)


def test_missing_timing_attrs_degrade_to_nan(tmp_path):
    # Missing root state must use explicit sentinels instead of crashing.
    synth = tmp_path / "untimed.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * 2, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=6,
                        receiver_provenance=False)
    ctx = _ctx(_detector_fn_factory(reject=lambda i: False), _weights())
    a, _ = _run(synth, ctx)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)
    assert np.isnan(float(got["unit_time0_ctime"][0]))
    assert np.isnan(float(got["unit_delta_time"][0]))
    assert int(got["unit_time0_fpga"][0]) == 0
    assert int(got["unit_event_id"][0]) == -1
    assert str(got["archive_version"][0]) == ""
    assert str(got["unit_git_version_tag"][0]) == ""
    assert str(got["unit_collection_server"][0]) == ""
    assert str(got["unit_input_map_sha256"][0]) == ""


def test_two_units_have_distinct_time0_and_reset_frame_in_unit(tmp_path):
    s0 = tmp_path / "u0.h5"
    s1 = tmp_path / "u1.h5"
    fmt.make_synth_file(str(s0), n_time=NFFT * 2, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=7)
    fmt.make_synth_file(str(s1), n_time=NFFT * 3, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=8)
    _add_timing(s0, ctime=CTIME, event_id=111)
    _add_timing(s1, ctime=CTIME + 100.0, event_id=222)

    ctx = _ctx(_detector_fn_factory(reject=lambda i: False), _weights())
    reader = ChimeBasebandPackedReader()
    a = PilotProxyDetectorAnalyzer()
    m0 = dict(reader.probe(str(s0))); m0["unit_key"] = "u0"; m0["scope"] = "local"
    a.begin(ctx, m0)
    a.consume_file(reader.iter_arrays(str(s0), ctx), m0)
    m1 = dict(reader.probe(str(s1))); m1["unit_key"] = "u1"; m1["scope"] = "local"
    a.consume_file(reader.iter_arrays(str(s1), ctx), m1)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)

    assert list(np.asarray(got["unit_event_id"])) == [111, 222]
    assert float(got["unit_time0_ctime"][1]) == pytest.approx(CTIME + 100.0, abs=0)
    # frame tags: unit 0 (2 frames) then unit 1 (3 frames); frame_in_unit resets
    assert list(np.asarray(got["frame_unit_index"]).reshape(-1)) == [0, 0, 1, 1, 1]
    assert list(np.asarray(got["frame_in_unit"]).reshape(-1)) == [0, 1, 0, 1, 2]


# -- provenance + rename ------------------------------------------------------

def test_current_product_provenance_and_reject_mask(tmp_path):
    synth = tmp_path / "prov.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * 2, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=9)
    w = _weights(seed=11)
    ctx = _ctx(_detector_fn_factory(reject=lambda i: False), w)
    a, _ = _run(synth, ctx)
    out = tmp_path / "844.npz"
    a.save(str(out))
    got = np.load(out)

    # rename propagated: the per-channel product uses reject_mask rather than mask
    assert "reject_mask" in got.files
    assert "mask" not in got.files
    # The scan stores the exact terms and nothing derived from them.
    assert "fine_power_u64" in got.files
    assert "fine_null_bulk_exceedance_fraction" not in got.files
    assert "fine_power_ratio" not in got.files
    assert "fine_nd_flag_rate" not in got.files

    # provenance
    import hashlib
    assert str(got["mask_rule"]) == NORMALIZED_POSITIVE_EXCESS_MASK_RULE
    assert str(np.asarray(got["weights_hash"])) == hashlib.sha256(
        np.ascontiguousarray(w).tobytes()
    ).hexdigest()
    assert PER_PILOT_PRODUCT_SCHEMA_TOKEN in str(np.asarray(got["detector_version"]))
    assert str(np.asarray(got["schema_version"])) == PER_PILOT_PRODUCT_SCHEMA_TOKEN
    assert str(np.asarray(got["schema_name"])) == "pilotproxy_per_pilot_product"
    assert int(np.asarray(got["schema_revision"])) == PER_PILOT_PRODUCT_SCHEMA_REVISION
    assert str(np.asarray(got["source_event_key_schema_version"])) == (
        "pilotproxy_namespaced_source_event_key_v1"
    )
    stored_contract = json.loads(
        str(np.asarray(got["decision_contract_json"]).reshape(()).item())
    )
    assert stored_contract == current_decision_contract()
