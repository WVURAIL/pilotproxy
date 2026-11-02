# coding=utf-8
"""Resume and relaunch safety for the ``pilot-proxy-detector`` analyzer.

A multi-day CANFAR detector run outlives a single interactive session, and GPU
time is scarce. A killed run must therefore continue from its checkpoint and
leave the product intact. These tests drive the analyzer and the scan entry
point with an injected CPU detector and check that:

  * a stream interrupted midway and resumed from its checkpoint yields the
    same product as a single pass, with integer and string arrays exact and
    float arrays to 1e-9, so no frame is reprocessed and nothing drifts
  * relaunching a scan whose channel is already complete does nothing and
    raises no error, because the produced-check counts resumed units as well
    as new ones
  * a product built with a per-file cap cannot be completed by an uncapped run
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

from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.archive.scan import run_chime_scan
from cpu_detector import exact_cpu_detector_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
F_CENTER_MHZ = 470.3125
FREQ_ID = 844

# arrays that must match exactly, or to floating-point tolerance, between a
# clean and a resumed run
_EXACT = ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64",
          "p_ref_sum_u64", "fine_power_u64", "reject_mask", "valid", "frame_index",
          "unit_keys", "unit_order", "frame_unit_index", "frame_in_unit",
          "unit_time0_fpga", "unit_event_id", "archive_version",
          "unit_git_version_tag", "unit_input_map_sha256",
          "unit_collection_server", "unit_scope")
_CLOSE = ("coarse_power_ratio", "normalized_coarse_power_ratio_db", "pilot_excess_db", "estimated_data_shelf_snr_db",
          "baseband_power_linear", "integrated_spectrum_before_mask",
          "integrated_spectrum_after_mask", "unit_time0_ctime", "unit_delta_time")


def _cpu_ref_detector_fn(*, packed, weights, kernel, emit_row_projections=False):
    """Input-dependent CPU stand-in for the GPU kernel.

    It runs the exact kernel chain of row sums, fxfft256, and all-bin sums.
    """
    return exact_cpu_detector_fn(
        packed=packed,
        weights=weights,
        kernel=kernel,
        emit_row_projections=emit_row_projections,
    )


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


# analyzer level, where a mid-stream checkpoint must resume to the same product

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

    # uninterrupted run with all three files through one analyzer
    a_clean = PilotProxyDetectorAnalyzer()
    a_clean.begin(ctx, _meta(0))
    _consume(a_clean, [0, 1, 2])
    clean_path = tmp_path / "clean.npz"
    a_clean.save(str(clean_path))

    # interrupted run, where files 0 and 1 are checkpointed and a fresh
    # analyzer resumes with file 2
    a1 = PilotProxyDetectorAnalyzer()
    a1.begin(ctx, _meta(0))
    _consume(a1, [0, 1])
    ckpt = tmp_path / "resumed.npz"
    a1.save(str(ckpt))

    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    assert a2.processed_keys() == {"synth:0", "synth:1"}
    a2.begin(ctx, _meta(2))            # first new file
    _consume(a2, [2])
    a2.save(str(ckpt))

    clean = np.load(clean_path)
    got = np.load(ckpt)
    assert got["frame_index"].shape[0] == clean["frame_index"].shape[0] == 3 * N_FRAMES
    _assert_products_equal(clean, got)


def test_analyzer_resume_refuses_fine_products_off(tmp_path):
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
    # In product schema v6 the frame powers are the sums of the fine powers, so
    # the fine products have no retention mode to switch off.
    with pytest.raises(ValueError, match="fine products are mandatory"):
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

    # a release with a new label and tree hash and the same detector
    monkeypatch.setattr(pilot_proxy, "__version__", "1.0.0")
    monkeypatch.setattr(detector_mod, "package_source_sha256", lambda *a, **k: "0c66af82" * 8)
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), ctx)
    with pytest.raises(SystemExit, match="detector_version.*source="):
        resumed.begin(ctx, meta)


def test_analyzer_resume_allows_changed_storage_service(tmp_path, monkeypatch):
    """A run may change which replica it fetches from and still resume.

    The storage service is only the transport. Both replicas serve the same
    object, and fetch() checks every staged file against the size recorded in
    the inventory. A run started while the default locator is degraded must
    therefore be able to move back to it later without rebuilding the product.
    The fingerprint covers the implementation identity (see the source-build
    guard above) and does not cover the fetch route.
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

    # first pass fetches through a pinned replica
    monkeypatch.setenv(STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/uvic/minoc")
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, _meta(0))
    first.consume_file(reader.iter_arrays(str(paths[0]), ctx), _meta(0))
    checkpoint = tmp_path / "route_change.npz"
    first.save(str(checkpoint))

    # the locator recovers, so the run moves back to the library default with
    # the same build
    monkeypatch.delenv(STORAGE_SERVICE_ENV, raising=False)
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), ctx) is True
    assert resumed.processed_keys() == {"synth:0"}
    resumed.begin(ctx, _meta(1))          # must not raise, since the route is not part of the identity
    resumed.consume_file(reader.iter_arrays(str(paths[1]), ctx), _meta(1))
    resumed.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as got:
        assert [str(k) for k in got["unit_order"]] == ["synth:0", "synth:1"]
        assert got["frame_index"].shape[0] == 2 * N_FRAMES


# scan level, with a relaunch through the real entry point

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
    _scan(resumed)                                                # resume with the 3rd file
    _scan(resumed)                                                # complete, so a no-op with no error
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
    # A finite chunk cap is treated as partial, since a persisted frame does not
    # show that its multi-frame source unit was fully consumed. Publication
    # therefore requires explicit acknowledgment from the caller.
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

    # The same cap is resume-compatible, and the relaunch is a no-op. The
    # output, however, is published only with explicit partial acceptance.
    run_chime_scan(**capped_args, allow_partial=True)
    assert (out / "chime_detector_outputs.npz").exists()
    # a relaunch without the cap must refuse to complete the capped product
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


# fine-width invariants across resume

@pytest.mark.parametrize(
    "status", ["disabled_by_option", "not_applicable_pilot_out_of_band"]
)
def test_resume_refuses_an_in_band_checkpoint_without_fine_powers(tmp_path, status):
    # The v5 placeholders for a scan without fine terms have no meaning in v6,
    # where the frame powers of an in-band product are the sums of its fine
    # powers.
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
        },
    )
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    meta["scope"] = "local"

    checkpoint = tmp_path / "stripped.npz"
    original = PilotProxyDetectorAnalyzer()
    original.begin(ctx, meta)
    original.consume_file(reader.iter_arrays(str(path), ctx), meta)
    original.save(str(checkpoint))

    with np.load(checkpoint, allow_pickle=False) as saved:
        fields = {name: saved[name] for name in saved.files}
    frame_count = int(np.asarray(fields["frame_index"]).size)
    fields["fine_status"] = np.asarray(status)
    fields["fine_num_bins"] = np.asarray(0, dtype=np.int64)
    fields["fine_power_u64"] = np.zeros((frame_count, 0, 0), dtype=np.uint64)
    np.savez_compressed(checkpoint, **fields)

    with pytest.raises(SystemExit, match="remove it and rebuild"):
        PilotProxyDetectorAnalyzer().resume(str(checkpoint), ctx)


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

    # checkpoint taken before any frame has fixed the fine width
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
    """Test the fine-width guard across a real save and resume.

    A change of nfft between runs is already refused by the resumed-identity
    check in begin(), since nfft is part of the identity tuple. The width guard
    covers the case of an unchanged identity, where the fine-reduction
    definition changes between software epochs (such as a larger
    FINE_PAD_FACTOR) and alters the width while the file geometry, contract,
    and weights still match. The epoch change is simulated by rebinding
    fine_bin_count in the detector module. The identity and provenance checks
    must pass, and the refusal must come from the width guard.
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

    # a new software epoch with identical files and identity and a doubled
    # fine width
    import pilot_proxy.archive.detector as det_mod
    monkeypatch.setattr(det_mod, "fine_bin_count", lambda windows: 2 * fixed)

    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    a2.begin(ctx, _meta(1))  # identity and provenance both still match
    with pytest.raises(SystemExit, match="fine dimensionality"):
        a2.consume_file(reader.iter_arrays(str(files[("evt1", FREQ_ID)]), ctx), _meta(1))


# fine detection ragged list, whose frame labels must be global frame indices

def _tone_matched_filter_row_projections_detector_fn(*, packed, weights, kernel, emit_row_projections=False):
    """Injected detector that emits v2 row sums with a fine tone in every frame.

    The target term holds a pure envelope tone, so a bin is detected in every
    frame, and the references are small broadband integers. The frame powers
    are the all-bin sums of the fixed-point fine spectra of the same rows, so
    the v6 frame-power invariant holds by construction.
    """
    from pilot_proxy.fine_reduction import frame_powers_from_fine_u64

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
    powers = frame_powers_from_fine_u64(zi, num_streams=streams, windows_per_stream=windows)
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
