# coding=utf-8
"""Resume / relaunch safety for the ``pilot-proxy-detector`` analyzer.

A multi-day CANFAR detector run outlives a single interactive session, so
correctness on relaunch is essential: a killed run must continue from its
checkpoint without starting over (scarce GPU) and without corrupting the
product. These
tests drive the analyzer and the real scan entry point with an injected CPU
detector (no GPU) and assert:

  * a stream interrupted mid-way, resumed from its checkpoint, yields a product
    byte-identical to one consumed in a single pass (no reprocessing, no drift);
  * relaunching a scan whose channel is already complete is a no-op rather
    than an error (the produced-check counts resumed units, not just new
    ones);
  * a product built with a per-file cap refuses to be silently "completed" by an
    uncapped run.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    INT4_COMPONENT_BITS,
    coarse_power_ratio_cpu_reference,
    unpack_packed_complex,
)
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.archive.scan import run_chime_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
F_CENTER_MHZ = 470.3125
FREQ_ID = 844

# arrays that must match exactly / to fp tolerance between a clean and resumed run
_EXACT = ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64",
          "p_ref_sum_u64", "fine_power_u64", "reject_mask", "valid", "frame_index",
          "unit_keys", "unit_order", "frame_unit_index", "frame_in_unit",
          "unit_time0_fpga", "unit_event_id", "archive_version",
          "unit_git_version_tag", "unit_input_map_sha256",
          "unit_collection_server", "unit_scope")
_CLOSE = ("coarse_power_ratio", "normalized_coarse_power_ratio_db", "pilot_excess_db", "estimated_data_shelf_snr_db",
          "baseband_power_linear", "integrated_spectrum_before_mask",
          "integrated_spectrum_after_mask", "unit_time0_ctime", "unit_delta_time")


def _cpu_ref_detector_fn(*, packed, weights, kernel):
    """Input-dependent CPU reference standing in for the GPU kernel."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    w = unpack_packed_complex(np.asarray(weights, dtype=np.int8), INT4_COMPONENT_BITS)
    _nt, _nl, _nu = weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    _nrs = int(_nl + _nu)
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
                num, den, target_norm_sq=_nt, reference_norm_sum_sq=_nrs
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


def _stub_kernel(detector_window_samples: int, reference_offset_bins: int = 2):
    specs = SimpleNamespace(
        K=detector_window_samples, N=3, bits=4,
        reference_offset_bins=reference_offset_bins,
        as_descriptive_dict=lambda: {
            "detector_window_samples": detector_window_samples,
            "num_weight_terms": 3, "sample_bits_per_component": 4,
            "reference_offset_bins": reference_offset_bins,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _fine_stub_kernel(
    detector_window_samples: int, reference_offset_bins: int = 2
):
    kernel = _stub_kernel(detector_window_samples, reference_offset_bins)
    kernel.supports_row_projections = lambda: True
    return kernel


def _make_files(input_dir: Path, n_events: int) -> dict:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i in range(n_events):
        p = input_dir / f"baseband_evt{i}_{FREQ_ID}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                            f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=100 + i)
        files[(f"evt{i}", FREQ_ID)] = p
    return files


def _write_inventory(inv: Path, n_events: int) -> None:
    with open(inv, "w") as fh:
        for i in range(n_events):
            fh.write(json.dumps({"common_path": "cadc:TEST", "event": f"evt{i}",
                                 "scope": "test.scope",
                                 "name": f"baseband_evt{i}_{FREQ_ID}.h5",
                                 "freq_id": FREQ_ID, "size_bytes": 1}) + "\n")


def _assert_products_equal(ref, got):
    for name in _EXACT:
        assert np.array_equal(np.asarray(ref[name]), np.asarray(got[name])), name
    for name in _CLOSE:
        assert np.allclose(np.asarray(ref[name], dtype=np.float64),
                           np.asarray(got[name], dtype=np.float64),
                           rtol=1e-9, atol=1e-9, equal_nan=True), name


# -- analyzer level: a mid-stream checkpoint resumes to the same product ------

def test_analyzer_resume_matches_uninterrupted(tmp_path):
    rng = np.random.default_rng(7)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=3)
    paths = [files[(f"evt{i}", FREQ_ID)] for i in range(3)]

    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _tone_matched_filter_row_projections_detector_fn,
        "kernel": _fine_stub_kernel(K),
        "weights": weights,
        "fine_products": "on",
    })
    reader = ChimeBasebandPackedReader()

    def _meta(i):
        m = dict(reader.probe(str(paths[i])))
        m["unit_key"] = f"synth:{i}"
        m["scope"] = "local"
        return m

    def _consume(analyzer, idxs):
        for i in idxs:
            analyzer.consume_file(reader.iter_arrays(str(paths[i]), ctx), _meta(i))

    # uninterrupted: all three files through one analyzer
    a_clean = PilotProxyDetectorAnalyzer()
    a_clean.begin(ctx, _meta(0))
    _consume(a_clean, [0, 1, 2])
    clean_path = tmp_path / "clean.npz"
    a_clean.save(str(clean_path))

    # interrupted: files 0,1 -> checkpoint, then a fresh analyzer resumes file 2
    a1 = PilotProxyDetectorAnalyzer()
    a1.begin(ctx, _meta(0))
    _consume(a1, [0, 1])
    ckpt = tmp_path / "resumed.npz"
    a1.save(str(ckpt))

    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    assert a2.processed_keys() == {"synth:0", "synth:1"}
    a2.begin(ctx, _meta(2))            # first NEW file
    _consume(a2, [2])
    a2.save(str(ckpt))

    clean = np.load(clean_path)
    got = np.load(ckpt)
    assert got["frame_index"].shape[0] == clean["frame_index"].shape[0] == 3 * N_FRAMES
    _assert_products_equal(clean, got)


def test_analyzer_resume_rejects_changed_fine_retention(tmp_path):
    rng = np.random.default_rng(17)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=2)
    paths = [files[(f"evt{i}", FREQ_ID)] for i in range(2)]
    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": _tone_matched_filter_row_projections_detector_fn,
            "kernel": _fine_stub_kernel(K),
            "weights": weights,
            "fine_products": "on",
        },
    )
    reader = ChimeBasebandPackedReader()

    def _meta(index):
        meta = dict(reader.probe(str(paths[index])))
        meta["unit_key"] = f"synth:{index}"
        meta["scope"] = "local"
        return meta

    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, _meta(0))
    first.consume_file(reader.iter_arrays(str(paths[0]), ctx), _meta(0))
    checkpoint = tmp_path / "fine-retention.npz"
    first.save(str(checkpoint))

    changed_ctx = replace(
        ctx,
        options={**ctx.options, "fine_products": "off"},
    )
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="fine-product retention mode differs"):
        resumed.begin(changed_ctx, _meta(1))


def test_analyzer_resume_absent_product_is_fresh(tmp_path):
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={})
    a = PilotProxyDetectorAnalyzer()
    assert a.resume(str(tmp_path / "nope.npz"), ctx) is False
    assert a.processed_keys() == set()


def test_consume_file_rolls_back_all_frames_when_later_chunk_fails(tmp_path):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    calls = 0

    def fail_second_chunk(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chunk failed")
        return _cpu_ref_detector_fn(**kwargs)

    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": fail_second_chunk,
            "kernel": _stub_kernel(K),
            "weights": weights,
        },
    )
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:transaction"
    meta["scope"] = "local"
    analyzer = PilotProxyDetectorAnalyzer()
    analyzer.begin(ctx, meta)

    with pytest.raises(RuntimeError, match="second chunk failed"):
        analyzer.consume_file(reader.iter_arrays(str(path), ctx), meta)

    assert analyzer._n_frames == 0
    assert analyzer._p_target == []
    assert analyzer._fine_power_u64 == []
    assert analyzer._frame_unit_index == []
    assert analyzer._unit_order == []
    assert analyzer.processed_keys() == set()
    before = getattr(analyzer._spec_before, "get", lambda: analyzer._spec_before)()
    after = getattr(analyzer._spec_after, "get", lambda: analyzer._spec_after)()
    assert np.count_nonzero(np.asarray(before)) == 0
    assert np.count_nonzero(np.asarray(after)) == 0


@pytest.mark.parametrize(
    "missing_field", ["unit_order", "unit_input_map_sha256", "nfft"]
)
def test_analyzer_resume_requires_current_checkpoint_fields(
    tmp_path, missing_field
):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": _cpu_ref_detector_fn,
            "kernel": _stub_kernel(K),
            "weights": weights,
        },
    )
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / f"missing-{missing_field}.npz"
    first.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as product:
        payload = {name: product[name] for name in product.files}
    payload.pop(missing_field)
    np.savez_compressed(checkpoint, **payload)

    with pytest.raises(
        SystemExit, match=rf"resume-critical fields.*{missing_field}"
    ):
        PilotProxyDetectorAnalyzer().resume(str(checkpoint), ctx)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("sample_rate_hz", np.asarray("bad")),
        ("unit_delta_time", np.asarray(["bad"])),
    ),
)
def test_analyzer_resume_reports_malformed_timing_metadata(
    tmp_path, field, invalid_value
):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": _cpu_ref_detector_fn,
            "kernel": _stub_kernel(K),
            "weights": weights,
        },
    )
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / f"malformed-{field}.npz"
    first.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as product:
        payload = {name: product[name] for name in product.files}
    payload[field] = invalid_value
    np.savez_compressed(checkpoint, **payload)

    with pytest.raises(SystemExit, match=rf"invalid.*{field}.*remove it and rebuild"):
        PilotProxyDetectorAnalyzer().resume(str(checkpoint), ctx)


def test_analyzer_resume_rejects_changed_weights(tmp_path):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    base_options = {
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
        "weights": weights,
    }
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options=base_options)
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / "weights.npz"
    first.save(str(checkpoint))

    changed = weights.copy()
    changed[0, 0] = 2
    changed_ctx = RunContext(
        instrument=load_instrument("chime"), selection=[FREQ_ID],
        options={**base_options, "weights": changed},
    )
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="weights_hash"):
        resumed.begin(changed_ctx, meta)


def test_analyzer_resume_rejects_changed_detector_contract(tmp_path):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _tone_matched_filter_row_projections_detector_fn,
        "kernel": _fine_stub_kernel(K),
        "weights": weights,
        "fine_products": "on",
    })
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / "contract.npz"
    first.save(str(checkpoint))

    changed_ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _tone_matched_filter_row_projections_detector_fn,
        "kernel": _fine_stub_kernel(K, reference_offset_bins=3),
        "weights": weights,
        "fine_products": "on",
    })
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="detector_contract"):
        resumed.begin(changed_ctx, meta)


def test_analyzer_resume_rejects_changed_sample_rate(tmp_path):
    files = _make_files(tmp_path / "data", n_events=2)
    first_path = files[("evt0", FREQ_ID)]
    next_path = files[("evt1", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    options = {
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
        "weights": weights,
    }
    original_instrument = load_instrument("chime")
    assert original_instrument.fs_hz == pytest.approx(390_625.0)
    original_ctx = RunContext(
        instrument=original_instrument,
        selection=[FREQ_ID],
        options=options,
    )
    first_meta = dict(reader.probe(str(first_path)))
    first_meta["unit_key"] = "synth:0"
    first_meta["scope"] = "local"
    first = PilotProxyDetectorAnalyzer()
    first.begin(original_ctx, first_meta)
    first.consume_file(reader.iter_arrays(str(first_path), original_ctx), first_meta)
    checkpoint = tmp_path / "sample-rate.npz"
    first.save(str(checkpoint))

    changed_instrument = replace(original_instrument, bandwidth_mhz=200.0)
    assert changed_instrument.fs_hz == pytest.approx(195_312.5)
    changed_ctx = RunContext(
        instrument=changed_instrument,
        selection=[FREQ_ID],
        options=options,
    )
    next_meta = dict(reader.probe(str(next_path)))
    next_meta["unit_key"] = "synth:1"
    next_meta["scope"] = "local"
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="sample rate"):
        resumed.begin(changed_ctx, next_meta)


def test_analyzer_resume_rejects_changed_python_source(tmp_path, monkeypatch):
    """One checkpoint may never contain frames from two source builds."""
    import pilot_proxy
    from pilot_proxy.archive import detector as detector_mod

    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    options = {
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
        "weights": weights,
    }
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID],
                     options=options)
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"

    monkeypatch.setattr(pilot_proxy, "__version__", "0.3.0.dev0")
    monkeypatch.setattr(detector_mod, "package_source_sha256", lambda *a, **k: "02066f1d" * 8)
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / "release_bump.npz"
    first.save(str(checkpoint))

    # The release: new label, new tree hash, same detector.
    monkeypatch.setattr(pilot_proxy, "__version__", "1.0.0")
    monkeypatch.setattr(detector_mod, "package_source_sha256", lambda *a, **k: "0c66af82" * 8)
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), ctx)
    with pytest.raises(SystemExit, match="detector_version.*source="):
        resumed.begin(ctx, meta)


def test_analyzer_resume_allows_changed_storage_service(tmp_path, monkeypatch):
    """A run may change which replica it fetches from and still resume.

    The storage service is transport: both replicas serve the same object, and
    fetch() checks every staged file against the inventory's recorded size. So
    a run started while the default locator is degraded must be able to move
    back to it later without rebuilding the product. This is the deliberate
    counterpart to the source-build guard above -- implementation identity is
    fingerprinted, the route is not.
    """
    from pilot_proxy.archive.sources.cadc import STORAGE_SERVICE_ENV

    files = _make_files(tmp_path / "data", n_events=2)
    paths = [files[(f"evt{i}", FREQ_ID)] for i in range(2)]
    reader = ChimeBasebandPackedReader()
    options = {
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
        "weights": np.ones((3, K), dtype=np.int8),
    }
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID],
                     options=options)

    def _meta(i):
        m = dict(reader.probe(str(paths[i])))
        m["unit_key"] = f"synth:{i}"
        m["scope"] = "local"
        return m

    # first pass: fetched through a pinned replica
    monkeypatch.setenv(STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/uvic/minoc")
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, _meta(0))
    first.consume_file(reader.iter_arrays(str(paths[0]), ctx), _meta(0))
    checkpoint = tmp_path / "route_change.npz"
    first.save(str(checkpoint))

    # the locator recovers: back to the library default, same build
    monkeypatch.delenv(STORAGE_SERVICE_ENV, raising=False)
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), ctx) is True
    assert resumed.processed_keys() == {"synth:0"}
    resumed.begin(ctx, _meta(1))          # must not raise: route is not identity
    resumed.consume_file(reader.iter_arrays(str(paths[1]), ctx), _meta(1))
    resumed.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as got:
        assert [str(k) for k in got["unit_order"]] == ["synth:0", "synth:1"]
        assert got["frame_index"].shape[0] == 2 * N_FRAMES


# -- scan level: relaunch through the real entry point ------------------------

def _fake_fetch_factory(files):
    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(files[(str(unit.meta["event"]), int(unit.meta["freq_id"]))], dest)
        return True, ""
    return _fake_fetch


def _fake_fetch_preflight(self, ctx):
    return True, [], []


def test_scan_resume_and_noop_relaunch(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rng = np.random.default_rng(9)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=3)
    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, n_events=3)

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource
    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch_factory(files))
    monkeypatch.setattr(
        CadcDatatrailSource, "fetch_preflight", _fake_fetch_preflight
    )

    inject = {"detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K),
              "weights": weights}

    def _scan(out_dir, **kw):
        run_chime_scan(output_dir=out_dir, source="cadc-datatrail", inventory=inv,
                       analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                       analyzer_options=inject, verbose=False, **kw)

    clean = tmp_path / "clean"
    _scan(clean)
    ref = np.load(clean / "_per_pilot" / f"{FREQ_ID}.npz")
    assert ref["frame_index"].shape[0] == 3 * N_FRAMES

    resumed = tmp_path / "resumed"
    _scan(resumed, max_files=2, allow_partial=True)               # process 2 of 3
    partial = np.load(resumed / "_per_pilot" / f"{FREQ_ID}.npz")
    assert partial["frame_index"].shape[0] == 2 * N_FRAMES        # checkpoint at 2
    _scan(resumed)                                                # resume -> 3rd file
    _scan(resumed)                                                # complete -> no-op (not an error)
    got = np.load(resumed / "_per_pilot" / f"{FREQ_ID}.npz")
    assert got["frame_index"].shape[0] == 3 * N_FRAMES
    _assert_products_equal(ref, got)


def test_scan_refuses_incompatible_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rng = np.random.default_rng(8)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=2)
    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, n_events=2)

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource
    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch_factory(files))
    monkeypatch.setattr(
        CadcDatatrailSource, "fetch_preflight", _fake_fetch_preflight
    )

    inject = {"detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K),
              "weights": weights}
    out = tmp_path / "capped"
    capped_args = {
        "output_dir": out,
        "source": "cadc-datatrail",
        "inventory": inv,
        "analyzer": "pilot-proxy-detector",
        "select": str(FREQ_ID),
        "analyzer_options": inject,
        "verbose": False,
        "max_chunks_per_file": 1,
    }
    # A finite chunk cap is conservatively partial: a persisted frame does not
    # prove that its multi-frame source unit was fully consumed, so publication
    # requires the caller's explicit acknowledgement.
    with pytest.raises(SystemExit, match="--allow-partial"):
        run_chime_scan(**capped_args)
    assert not (out / "chime_detector_outputs.npz").exists()
    scope = json.loads((out / "scan_scope.json").read_text())
    pilot = scope["pilots"][0]
    assert scope["complete"] is False
    assert pilot["status"] == "capped"
    assert pilot["completed"] == 0
    assert pilot["capped"] == 2
    assert pilot["unprocessed"] == 0

    # The same cap is resume-compatible and becomes a no-op, but the output is
    # still published only with explicit partial acceptance.
    run_chime_scan(**capped_args, allow_partial=True)
    assert (out / "chime_detector_outputs.npz").exists()
    # relaunch WITHOUT the cap must refuse rather than complete the capped product
    with pytest.raises(SystemExit, match="capped product cannot be completed"):
        run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                       analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                       analyzer_options=inject, verbose=False)
    incompatible_scope = json.loads((out / "scan_scope.json").read_text())
    incompatible_pilot = incompatible_scope["pilots"][0]
    assert incompatible_scope["complete"] is False
    assert incompatible_pilot["status"] == "aborted"
    assert incompatible_pilot["completed"] == 0
    assert incompatible_pilot["capped"] == 2


# -- fine-width invariants across resume --------------------------------------

def test_resume_normalizes_zero_fine_placeholders(tmp_path):
    weights = np.ones((3, K), dtype=np.int8)
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": _cpu_ref_detector_fn,
            "kernel": _stub_kernel(K),
            "weights": weights,
            "fine_products": "off",
        },
    )
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"

    checkpoint = tmp_path / "disabled.npz"
    original = PilotProxyDetectorAnalyzer()
    original.begin(ctx, meta)
    original.consume_file(reader.iter_arrays(str(path), ctx), meta)
    original.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as saved:
        fields = {name: saved[name] for name in saved.files}
    frame_count = int(np.asarray(fields["frame_index"]).size)
    fields["fine_num_bins"] = np.asarray(4, dtype=np.int64)
    fields["fine_power_u64"] = np.zeros(
        (frame_count, 3, 4), dtype=np.uint64
    )
    np.savez_compressed(checkpoint, **fields)

    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), ctx)
    resumed.begin(ctx, meta)
    resumed.save(str(checkpoint))
    with np.load(checkpoint, allow_pickle=False) as normalized:
        assert int(normalized["fine_num_bins"]) == 0
        assert normalized["fine_power_u64"].shape == (frame_count, 0, 0)

def test_zero_frame_checkpoint_resume_keeps_fine_width(tmp_path):
    rng = np.random.default_rng(11)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _tone_matched_filter_row_projections_detector_fn,
        "kernel": _fine_stub_kernel(K),
        "weights": weights,
        "fine_products": "on",
    })
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"

    # checkpoint before any frame fixes the fine width
    a1 = PilotProxyDetectorAnalyzer()
    a1.begin(ctx, meta)
    ckpt = tmp_path / "zero.npz"
    a1.save(str(ckpt))
    z0 = np.load(str(ckpt))
    assert z0["fine_power_u64"].shape[0] == 0

    # resume adopts the current width and continues cleanly
    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    a2.begin(ctx, meta)
    a2.consume_file(reader.iter_arrays(str(path), ctx), meta)
    a2.save(str(ckpt))
    z1 = np.load(str(ckpt))
    assert z1["fine_power_u64"].shape == (
        N_FRAMES,
        3,
        int(z1["fine_num_bins"]),
    )
    assert int(z1["fine_num_bins"]) > 0


def test_resume_refuses_fine_width_change(tmp_path):
    a = PilotProxyDetectorAnalyzer()
    a._fine_bins = 64
    with pytest.raises(SystemExit, match="fine dimensionality"):
        a._ensure_fine_width(256)
    a._fine_bins = 0
    a._ensure_fine_width(256)
    assert a._fine_bins == 256


def test_resume_refuses_fine_definition_change_end_to_end(tmp_path, monkeypatch):
    """Behavioral coverage for the width guard across a real save/resume.

    An nfft change between runs is already refused by the resumed-identity
    check in begin() (nfft is part of the identity tuple), so the width
    guard's distinct job is the same-identity case: a fine-reduction
    definition change between software epochs (e.g. a FINE_PAD_FACTOR bump)
    that alters the width while file geometry, contract, and weights all
    still match. Simulate that epoch change by rebinding the detector
    module's fine_bin_count, and assert the refusal comes from the guard,
    not from the identity or provenance checks.
    """
    rng = np.random.default_rng(13)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=2)
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _tone_matched_filter_row_projections_detector_fn,
        "kernel": _fine_stub_kernel(K),
        "weights": weights,
        "fine_products": "on",
    })
    reader = ChimeBasebandPackedReader()

    def _meta(i):
        m = dict(reader.probe(str(files[(f"evt{i}", FREQ_ID)])))
        m["unit_key"] = f"synth:{i}"
        m["scope"] = "local"
        return m

    # checkpoint file 0 under the current fine-reduction definition
    a1 = PilotProxyDetectorAnalyzer()
    a1.begin(ctx, _meta(0))
    a1.consume_file(reader.iter_arrays(str(files[("evt0", FREQ_ID)]), ctx), _meta(0))
    ckpt = tmp_path / "844.npz"
    a1.save(str(ckpt))
    fixed = int(np.load(str(ckpt))["fine_num_bins"])
    assert fixed > 0

    # "new software epoch": identical files and identity, doubled fine width
    import pilot_proxy.archive.detector as det_mod
    monkeypatch.setattr(det_mod, "fine_bin_count", lambda windows: 2 * fixed)

    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    a2.begin(ctx, _meta(1))  # identity and provenance both still match
    with pytest.raises(SystemExit, match="fine dimensionality"):
        a2.consume_file(reader.iter_arrays(str(files[("evt1", FREQ_ID)]), ctx), _meta(1))


# -- fine detection ragged list: frame labels must be global frame indices ----

def _tone_matched_filter_row_projections_detector_fn(*, packed, weights, kernel, emit_row_projections=False):
    """Injected detector emitting v2 row sums with a guaranteed fine tone.

    The target term carries a pure envelope tone (detected bin every frame);
    the references are small broadband integers. Kernel powers are computed
    from the same integer row sums, so the exact coarse marginal identity holds
    by construction.
    """
    from pilot_proxy.fine_reduction import exact_coarse_power_by_term

    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    rows = int(pk.shape[1])
    windows = NFFT // K
    streams = rows // windows
    n = np.arange(windows)
    tone = 1000.0 * np.exp(2j * np.pi * 5 * n / windows)
    z = np.zeros((3, streams, windows), dtype=np.complex128)
    z[0] = tone[None, :]
    rng = np.random.default_rng(rows)
    for term in (1, 2):
        z[term] = (rng.integers(-3, 4, (streams, windows))
                   + 1j * rng.integers(-3, 4, (streams, windows)))
    zi = np.stack([np.round(z.real), np.round(z.imag)], axis=-1)
    zi = zi.astype(np.int32).reshape(3, rows, 2)
    powers = exact_coarse_power_by_term(zi, num_weight_terms=3)
    result = {
        "block_index": 0,
        "mask": True,
        "p_target_u64": int(powers[0]),
        "p_ref_lower_u64": int(powers[1]),
        "p_ref_upper_u64": int(powers[2]),
        "p_ref_sum_u64": int(powers[1] + powers[2]),
    }
    out = {"batch": 1, "detector_rows_per_block": rows,
           "rational_overflow_count": 0, "results": [result]}
    if emit_row_projections:
        out["matched_filter_row_projections"] = [zi]
    return out
