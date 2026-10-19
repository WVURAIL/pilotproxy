# coding=utf-8
"""Scan-orchestration guards.

Covers source/plumbing and failure handling:
  #3  cadc-datatrail source plumbing (--inventory / --source-root) + freq_id
      enumeration, exercised offline with a fake inventory and a mocked fetch;
  #4  a GPU/cupy preflight for pilot-proxy-detector before any staging;
  #6  an all-units-failed / all-quarantined scan is surfaced rather than silently turned
      into an absent/empty product fed to combine;
  terminal-combine soft-fail: any combine integrity refusal (ValueError family)
      preserves the per-pilot products, prints working guidance, and records
      the skip in scan_scope.json; other exceptions still propagate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from cpu_detector import cpu_detector_options as _cpu_detector_options
from cpu_detector import stub_detector_fn as _stub_detector_fn
from pilot_proxy.chime import baseband_format as fmt

from pilot_proxy.archive.scan import (
    _failed_current_unit_count,
    _read_existing_scope,
    run_chime_scan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
N_FEEDS = 4


def _fine_stub_detector_fn(
    *, packed, weights, kernel, emit_row_projections=False
):
    from pilot_proxy.fine_reduction import frame_powers_from_fine_u64

    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    rows = int(pk.shape[1])
    projections = np.ones((3, rows, 2), dtype=np.int32)
    projections[0, :, 0] = 2
    projections[1, :, 1] = 2
    projections[2, :, 0] = 3
    # The frame powers are the all-bin sums of these rows' fixed-point fine
    # spectra (per-pilot v6); the analyzer forms the spectra from the rows.
    powers = frame_powers_from_fine_u64(projections, num_streams=rows // 128)
    result = {
        "block_index": 0,
        "mask": 0,
        "p_target_u64": int(powers[0]),
        "p_ref_lower_u64": int(powers[1]),
        "p_ref_upper_u64": int(powers[2]),
        "p_ref_sum_u64": int(powers[1] + powers[2]),
    }
    output = {
        "batch": 1,
        "detector_rows_per_block": rows,
        "rational_overflow_count": 0,
        "results": [result],
    }
    if emit_row_projections:
        output["matched_filter_row_projections"] = [projections]
    return output


def _fine_detector_options():
    options = _cpu_detector_options()
    options["detector_fn"] = _fine_stub_detector_fn
    options["kernel"].supports_row_projections = lambda: True
    options["fine_products"] = "on"
    return options
# freq_id -> coarse-channel center (MHz)
CHAN_MHZ = {844: 470.3125, 829: 476.171875, 752: 506.171875}

_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _make_inventory(inv_path, *, common_path, event, channels):
    with open(inv_path, "w") as fh:
        for ch in channels:
            fh.write(json.dumps({"common_path": common_path, "event": event,
                                 "scope": "test.scope",
                                 "name": f"baseband_{event}_{int(ch)}.h5",
                                 "freq_id": int(ch), "size_bytes": 1}) + "\n")


# -- #3: CADC plumbing + freq_id enumeration (offline) -----------------------

def test_cadc_scan_enumerates_by_freq_id(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # receiver-profile default path resolves here
    data = tmp_path / "data"
    data.mkdir()
    synth = {}
    for ch, mhz in CHAN_MHZ.items():
        p = data / f"baseband_evt1_{ch}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * 2, n_feeds=N_FEEDS,
                            f_center_mhz=mhz, f_tone_bb=1300.0, seed=ch)
        synth[ch] = p
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1",
                    channels=sorted(CHAN_MHZ))  # all three listed

    # offline fetch: copy the local synth matching the unit's channel
    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource

    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(synth[int(unit.meta["freq_id"])], dest)
        return True, ""

    preflight_calls = []

    def _fake_fetch_preflight(self, ctx):
        preflight_calls.append(ctx)
        return True, [], []

    def _reject_survey_preflight(self, ctx):
        raise AssertionError("scan must not run survey preflight")

    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch)
    monkeypatch.setattr(
        CadcDatatrailSource, "fetch_preflight", _fake_fetch_preflight
    )
    monkeypatch.setattr(CadcDatatrailSource, "preflight", _reject_survey_preflight)

    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                   analyzer="pilot-proxy-detector", select="829,844",
                   analyzer_options=_cpu_detector_options(), verbose=False)

    work = out / "_per_pilot"
    assert (work / "829.npz").exists()
    assert (work / "844.npz").exists()
    assert not (work / "752.npz").exists()   # listed in inventory but not selected
    assert (out / "chime_detector_outputs.npz").exists()  # combined product
    assert len(preflight_calls) == 1
    # The injected backend always says mask=0. The canonical product instead
    # derives the policy it declares from exact powers and stored weight norms.
    with np.load(work / "829.npz", allow_pickle=False) as product:
        num = np.asarray(product["p_target_u64"]).reshape(-1)
        den = np.asarray(product["p_ref_sum_u64"]).reshape(-1)
        target_norm = int(np.asarray(product["target_norm_sq"]).reshape(-1)[0])
        reference_norm = int(
            np.asarray(product["reference_norm_sum_sq"]).reshape(-1)[0]
        )
        expected = [
            int(int(n) * reference_norm > target_norm * int(d)) if d else 0
            for n, d in zip(num, den)
        ]
        assert np.asarray(product["reject_mask"]).reshape(-1).tolist() == expected
    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["input"] == {
        "inventory_path": str(inv.resolve()),
        "inventory_sha256": hashlib.sha256(inv.read_bytes()).hexdigest(),
    }
    assert scope["max_files"] is None
    assert scope["execution_attempts"] == [scope["execution"]]
    assert scope["complete"] is True
    assert scope["totals"] == {
        "requested": 2,
        "enumerated": 2,
        "completed": 2,
        "capped": 0,
        "failed": 0,
        "quarantined": 0,
        "unprocessed": 0,
        "extra_completed": 0,
        "pilots_requested": 2,
    }
    assert scope["terminal_combine"] == {"status": "combined"}


# -- #3: explicit per-source option validation -------------------------------

def test_local_requires_input_dir(tmp_path):
    with pytest.raises(SystemExit, match="--input-dir"):
        run_chime_scan(output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


def test_cadc_requires_inventory_or_root(tmp_path):
    with pytest.raises(SystemExit, match="--inventory"):
        run_chime_scan(output_dir=tmp_path / "o", source="cadc-datatrail",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


def test_changed_inventory_is_rejected_before_scope_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[844])

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource
    import pilot_proxy.archive.pipeline as pipeline_mod

    monkeypatch.setattr(
        CadcDatatrailSource,
        "fetch_preflight",
        lambda self, ctx: (True, [], []),
    )

    class _Stop(Exception):
        pass

    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(pipeline_mod, "run", _stop)
    kwargs = {
        "output_dir": tmp_path / "out",
        "source": "cadc-datatrail",
        "inventory": inv,
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(_Stop):
        run_chime_scan(**kwargs)
    scope_path = kwargs["output_dir"] / "scan_scope.json"
    original_scope = scope_path.read_text()
    inv.write_text(inv.read_text() + "\n")

    with pytest.raises(SystemExit, match="inventory SHA-256 differs"):
        run_chime_scan(**kwargs)
    assert scope_path.read_text() == original_scope


def test_changed_selection_is_rejected_before_scope_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[844])

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource
    import pilot_proxy.archive.pipeline as pipeline_mod

    monkeypatch.setattr(
        CadcDatatrailSource,
        "fetch_preflight",
        lambda self, ctx: (True, [], []),
    )

    class _Stop(Exception):
        pass

    def _stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(pipeline_mod, "run", _stop)
    kwargs = {
        "output_dir": tmp_path / "out",
        "source": "cadc-datatrail",
        "inventory": inv,
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(_Stop):
        run_chime_scan(**kwargs)
    scope_path = kwargs["output_dir"] / "scan_scope.json"
    original_scope = scope_path.read_text()

    with pytest.raises(SystemExit, match="selection differs"):
        run_chime_scan(**{**kwargs, "select": "829"})
    assert scope_path.read_text() == original_scope


@pytest.mark.parametrize(
    "change", ["root", "glob", "freq", "event", "bytes", "added"]
)
def test_cross_pilot_resume_rejects_changed_local_input(
    tmp_path, monkeypatch, change
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    alternate = tmp_path / "alternate"
    data.mkdir()
    alternate.mkdir()
    for root in (data, alternate):
        for freq_id in (829, 844):
            fmt.make_synth_file(
                str(root / f"baseband_1_{freq_id}.h5"),
                n_time=NFFT,
                n_feeds=N_FEEDS,
                f_center_mhz=CHAN_MHZ[freq_id],
                f_tone_bb=1300.0,
                seed=freq_id,
            )

    import pilot_proxy.archive.pipeline as pipeline_mod

    real_run = pipeline_mod.run
    calls = 0

    class _Stop(Exception):
        pass

    def _stop_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _Stop
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "run", _stop_second)
    kwargs = {
        "input_dir": data,
        "output_dir": tmp_path / "out",
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "829,844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }
    with pytest.raises(_Stop):
        run_chime_scan(**kwargs)

    scope_path = kwargs["output_dir"] / "scan_scope.json"
    original_scope = scope_path.read_text()
    changed = {
        "root": {"input_dir": alternate},
        "glob": {"source_glob": "baseband_*.h5"},
        "freq": {"source_freq_id_regex": r"_((?:829|844))\.h5$"},
        "event": {"source_event_regex": r"baseband_([0-9]+)_"},
    }.get(change, {})
    if change == "bytes":
        path = data / "baseband_1_829.h5"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif change == "added":
        shutil.copy2(
            data / "baseband_1_829.h5",
            data / "baseband_2_829.h5",
        )

    def _reject_work(*args, **kwargs):
        raise AssertionError("pipeline work started before the scope check")

    monkeypatch.setattr(pipeline_mod, "run", _reject_work)
    with pytest.raises(SystemExit, match="local input differs"):
        run_chime_scan(**{**kwargs, **changed})
    assert scope_path.read_text() == original_scope


def test_cross_pilot_resume_refuses_fine_products_auto(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for freq_id in (829, 844):
        fmt.make_synth_file(
            str(data / f"baseband_1_{freq_id}.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[freq_id],
            f_tone_bb=1300.0,
            seed=freq_id,
        )

    import pilot_proxy.archive.pipeline as pipeline_mod

    real_run = pipeline_mod.run
    calls = 0

    class _Stop(Exception):
        pass

    def _stop_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _Stop
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "run", _stop_second)
    kwargs = {
        "input_dir": data,
        "output_dir": tmp_path / "out",
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "829,844",
        "analyzer_options": _fine_detector_options(),
        "verbose": False,
    }
    with pytest.raises(_Stop):
        run_chime_scan(**kwargs)

    first_product = kwargs["output_dir"] / "_per_pilot" / "829.npz"
    second_product = kwargs["output_dir"] / "_per_pilot" / "844.npz"
    assert first_product.exists()
    assert not second_product.exists()
    scope_path = kwargs["output_dir"] / "scan_scope.json"
    original_scope = scope_path.read_text()

    unsupported = _cpu_detector_options()
    unsupported["fine_products"] = "auto"

    def _reject_work(*args, **kwargs):
        raise AssertionError("pipeline work started before the scope check")

    monkeypatch.setattr(pipeline_mod, "run", _reject_work)
    # fine_products=auto was the coarse-only degradation; since product
    # schema v6 it is refused before any work, and the scope is untouched.
    with pytest.raises(SystemExit, match="fine products are mandatory"):
        run_chime_scan(**{**kwargs, "analyzer_options": unsupported})
    assert scope_path.read_text() == original_scope


def test_in_pilot_resume_refuses_fine_products_auto(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in (1, 2):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )

    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "max_files": 1,
        "allow_partial": True,
        "analyzer_options": _fine_detector_options(),
        "verbose": False,
    }
    run_chime_scan(**kwargs)
    product_path = out / "_per_pilot" / "844.npz"
    product_digest = hashlib.sha256(product_path.read_bytes()).hexdigest()
    scope_path = out / "scan_scope.json"
    original_scope = scope_path.read_text()

    unsupported = _cpu_detector_options()
    unsupported["fine_products"] = "auto"
    with pytest.raises(SystemExit, match="fine products are mandatory"):
        run_chime_scan(
            **{
                **kwargs,
                "max_files": None,
                "allow_partial": False,
                "analyzer_options": unsupported,
            }
        )
    assert hashlib.sha256(product_path.read_bytes()).hexdigest() == product_digest
    assert scope_path.read_text() == original_scope


def test_unreadable_existing_scope_is_not_overwritten(tmp_path):
    scope_path = tmp_path / "scan_scope.json"
    scope_path.write_text("not json")

    with pytest.raises(SystemExit, match="existing scan scope is unreadable"):
        _read_existing_scope(scope_path)

    assert scope_path.read_text() == "not json"


# -- #4: GPU preflight for the detector --------------------------------------

@pytest.mark.skipif(_HAS_CUPY, reason="cupy is installed; the preflight would pass")
def test_detector_preflight_requires_cupy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_844.h5"), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1300.0, seed=1)
    # no injected detector_fn -> the real CUDA kernel is required -> must preflight
    with pytest.raises(SystemExit, match="cupy"):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


# -- #6: all-units-failed/quarantined is surfaced ----------------------------

def test_all_units_failed_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    # a file the reader cannot open -> the only unit fails -> no product written
    (data / "baseband_e_844.h5").write_bytes(b"not a valid hdf5 file")
    with pytest.raises(SystemExit, match="no usable product"):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844",
                       analyzer_options=_cpu_detector_options(), verbose=False)
    scope = json.loads((tmp_path / "o" / "scan_scope.json").read_text())
    assert scope["pilots"][0]["enumerated"] == 1
    assert scope["pilots"][0]["completed"] == 0
    assert scope["pilots"][0]["quarantined"] == 1


def test_partial_scan_requires_explicit_allow_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in ("a", "b"):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT * 2,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=ord(event),
        )
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "max_files": 1,
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(SystemExit, match="--allow-partial"):
        run_chime_scan(**kwargs)

    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["pilots"][0]["enumerated"] == 2
    assert scope["pilots"][0]["completed"] == 1
    assert scope["pilots"][0]["unprocessed"] == 1
    assert scope["complete"] is False

    result = run_chime_scan(**kwargs, allow_partial=True)
    assert result["scan_scope"] == out / "scan_scope.json"
    assert (out / "chime_detector_outputs.npz").exists()


def test_chunk_cap_records_capped_units_and_requires_allow_partial(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(
        str(data / "baseband_a_844.h5"),
        n_time=NFFT * 2,
        n_feeds=N_FEEDS,
        f_center_mhz=CHAN_MHZ[844],
        f_tone_bb=1300.0,
        seed=1,
    )
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "max_chunks_per_file": 1,
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(SystemExit, match="--allow-partial"):
        run_chime_scan(**kwargs)

    assert not (out / "chime_detector_outputs.npz").exists()
    with np.load(out / "_per_pilot" / "844.npz", allow_pickle=False) as product:
        assert np.asarray(product["frame_index"]).shape == (1,)
    scope = json.loads((out / "scan_scope.json").read_text())
    pilot = scope["pilots"][0]
    assert scope["max_chunks_per_file"] == 1
    assert scope["complete"] is False
    assert pilot["status"] == "capped"
    assert pilot["completed"] == 0
    assert pilot["capped"] == 1
    assert len(pilot["capped_unit_keys"]) == 1
    assert pilot["unprocessed"] == 0

    result = run_chime_scan(**kwargs, allow_partial=True)
    assert result["scan_scope"] == out / "scan_scope.json"
    assert (out / "chime_detector_outputs.npz").exists()


def test_aborted_scan_counts_only_current_unit_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in (1, 2, 3):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )
    calls = 0

    def fail_second_unit(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-unit failure")
        return _stub_detector_fn(**kwargs)

    options = _cpu_detector_options()
    options["detector_fn"] = fail_second_unit
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="second-unit failure"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=options,
            verbose=False,
        )

    pilot = json.loads((out / "scan_scope.json").read_text())["pilots"][0]
    assert pilot["completed"] == 0
    assert pilot["non_durable_completed"] == 1
    assert pilot["failed"] == 1
    assert pilot["unprocessed"] == 2


def test_later_pilot_units_remain_accounted_when_earlier_pilot_aborts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for freq_id in (829, 844):
        fmt.make_synth_file(
            str(data / f"baseband_1_{freq_id}.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[freq_id],
            f_tone_bb=1300.0,
            seed=freq_id,
        )

    def fail_first_pilot(**kwargs):
        del kwargs
        raise RuntimeError("injected first-pilot failure")

    options = _cpu_detector_options()
    options["detector_fn"] = fail_first_pilot
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="first-pilot failure"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="829,844",
            analyzer_options=options,
            verbose=False,
        )

    entries = {
        tuple(entry["selection"]): entry
        for entry in json.loads((out / "scan_scope.json").read_text())["pilots"]
    }
    assert entries[(829,)]["failed"] == 1
    assert entries[(829,)]["unprocessed"] == 0
    assert entries[(844,)]["status"] == "pending"
    assert entries[(844,)]["unprocessed"] == 1


def test_shorter_than_nfft_unit_is_not_committed(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(
        str(data / "baseband_1_844.h5"),
        n_time=NFFT - 1,
        n_feeds=N_FEEDS,
        f_center_mhz=CHAN_MHZ[844],
        f_tone_bb=1300.0,
        seed=1,
    )
    out = tmp_path / "out"

    # The unit must never be committed -- but it is rejected at probe time as an
    # unreadable unit, not by the analyzer. A shorter-than-nfft file that opens
    # cleanly and holds its full declared extent is a genuinely short
    # acquisition, not a truncated fetch (a truncated fetch fails to open, or
    # falls short of its declared extent, and quarantines via OSError), so it can
    # never yield a frame however often it is re-fetched. Raising from the
    # analyzer instead made this a run-level error: the 2026-08-23 pre-flight lost
    # eight completed channels to one 3.9 MB stub.
    # This channel's only unit is unusable, so the scope gate refuses to publish
    # a partial run -- with the unit accounted as quarantined rather than the run
    # dying mid-stream. A channel with other good units keeps them and continues.
    with pytest.raises(SystemExit, match="quarantined=1"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=_cpu_detector_options(),
            verbose=False,
        )

    pilot = json.loads((out / "scan_scope.json").read_text())["pilots"][0]
    assert pilot["completed"] == 0
    assert not (out / "_per_pilot" / "844.npz").exists()


def test_resume_rejects_removed_local_input_before_scope_overwrite(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    paths = []
    for event in (1, 2):
        path = data / f"baseband_{event}_844.h5"
        fmt.make_synth_file(
            str(path),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )
        paths.append(path)
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }
    run_chime_scan(**kwargs)
    scope_path = out / "scan_scope.json"
    original_scope = scope_path.read_text()
    paths[1].unlink()

    with pytest.raises(SystemExit, match="local input differs"):
        run_chime_scan(**kwargs, allow_partial=True)
    assert scope_path.read_text() == original_scope


def test_enumerated_scope_is_persisted_before_pipeline_runs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in (1, 2):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )

    import pilot_proxy.archive.pipeline as _dpl

    class _Stop(Exception):
        pass

    def inspect_scope_before_work(*args, out_path, **kwargs):
        del args, kwargs
        scope_path = Path(out_path).parents[1] / "scan_scope.json"
        scope = json.loads(scope_path.read_text())
        pilot = scope["pilots"][0]
        assert pilot["requested"] == 2
        assert pilot["enumerated"] == 2
        assert pilot["unprocessed"] == 2
        assert scope["totals"]["requested"] == 2
        assert scope["totals"]["pilots_requested"] == 1
        raise _Stop

    monkeypatch.setattr(_dpl, "run", inspect_scope_before_work)

    with pytest.raises(_Stop):
        run_chime_scan(
            input_dir=data,
            output_dir=tmp_path / "out",
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=_cpu_detector_options(),
            verbose=False,
        )


def test_checkpoint_every_reaches_pipeline(tmp_path, monkeypatch):
    # --checkpoint-every must thread CLI -> run_chime_scan -> pipeline.run, and an
    # unset value must fall back to the engine default (50). Spy on pipeline.run and
    # stop before the real work, so the guard needs no GPU and no processable data.
    monkeypatch.chdir(REPO_ROOT)
    import pilot_proxy.archive.pipeline as _dpl
    seen: dict = {}

    class _Stop(Exception):
        pass

    def _spy(*a, checkpoint_every=None, **k):
        seen["ckpt"] = checkpoint_every
        seen["workers"] = k["download_workers"]
        seen["staged"] = k["max_staged_files"]
        seen["tmp_dir"] = k["tmp_dir"]
        raise _Stop

    monkeypatch.setattr(_dpl, "run", _spy)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_829.h5"), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=476.3125, f_tone_bb=1200.0, seed=1)

    with pytest.raises(_Stop):                       # explicit value threads through
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o1", source="local",
                       analyzer="pilot-proxy-detector", select="829", checkpoint_every=7,
                       download_workers=2, max_staged_files=4,
                       max_files=3,
                       staging_dir=tmp_path / "staging",
                       analyzer_options=_cpu_detector_options(), verbose=False)
    assert seen["ckpt"] == 7
    assert seen["workers"] == 2
    assert seen["staged"] == 4
    assert seen["tmp_dir"] == str((tmp_path / "staging" / "829").resolve())
    scope = json.loads((tmp_path / "o1" / "scan_scope.json").read_text())
    assert {
        key: value
        for key, value in scope["input"].items()
        if key != "file_manifest"
    } == {
        "root": str(data.resolve()),
        "glob": "*.h5",
        "freq_id_regex": r"_(\d+)\.h5$",
        "source_event_regex": r"baseband_(\d+)_",
    }
    manifest = scope["input"]["file_manifest"]
    assert manifest["schema_version"] == "pilotproxy_local_input_manifest_v1"
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] == (
        data / "baseband_e_829.h5"
    ).stat().st_size
    assert [record["path"] for record in manifest["files"]] == [
        "baseband_e_829.h5"
    ]
    assert scope["max_files"] == 3
    assert scope["execution"] == {
        "preserve_source_order": True,
        "download_workers": 2,
        "max_staged_files": 4,
        "checkpoint_every": 7,
        "staging_dir": str((tmp_path / "staging").resolve()),
        "storage_service": None,
    }
    assert scope["execution_attempts"] == [scope["execution"]]

    first_execution = dict(scope["execution"])
    with pytest.raises(_Stop):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o1", source="local",
                       analyzer="pilot-proxy-detector", select="829", checkpoint_every=9,
                       download_workers=1, max_staged_files=2,
                       staging_dir=tmp_path / "staging-resume",
                       analyzer_options=_cpu_detector_options(), verbose=False)
    scope = json.loads((tmp_path / "o1" / "scan_scope.json").read_text())
    assert scope["execution_attempts"] == [first_execution, scope["execution"]]
    assert scope["execution"]["checkpoint_every"] == 9

    with pytest.raises(_Stop):                       # unset -> engine default
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o2", source="local",
                       analyzer="pilot-proxy-detector", select="829",
                       analyzer_options=_cpu_detector_options(), verbose=False)
    assert seen["ckpt"] == 50


def test_streaming_counts_are_validated_before_preflight(tmp_path):
    with pytest.raises(SystemExit, match="max-staged-files"):
        run_chime_scan(
            input_dir=tmp_path / "missing",
            output_dir=tmp_path / "out",
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            download_workers=2,
            max_staged_files=1,
            verbose=False,
        )

    with pytest.raises(SystemExit, match="positive integer"):
        run_chime_scan(
            input_dir=tmp_path / "missing",
            output_dir=tmp_path / "out",
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            download_workers=0,
            verbose=False,
        )


def test_failed_count_checks_exception_chain_cycle_safely():
    unit = SimpleNamespace(key="unit-1", name="baseband_1_844.h5")
    cause = RuntimeError("ordered fetch stopped at baseband_1_844.h5")
    outer = RuntimeError("download workers still active")
    outer.__cause__ = cause
    cause.__context__ = outer

    assert _failed_current_unit_count(
        outer,
        [unit],
        completed_keys=set(),
        quarantined_keys=set(),
        max_files=None,
    ) == 1


def test_local_staging_must_be_outside_input_tree(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(
        str(data / "baseband_e_844.h5"),
        n_time=NFFT,
        n_feeds=N_FEEDS,
        f_center_mhz=CHAN_MHZ[844],
        f_tone_bb=1300.0,
        seed=1,
    )

    with pytest.raises(SystemExit, match="outside the local input tree"):
        run_chime_scan(
            input_dir=data,
            output_dir=tmp_path / "out",
            staging_dir=data / "staging",
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=_cpu_detector_options(),
            verbose=False,
        )

# -- terminal-combine soft-fail: messages + durable record --------------------

def _completed_scan_kwargs(tmp_path):
    """One local channel that scans to completion, so the terminal combine runs."""
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_844.h5"), n_time=NFFT * 2,
                        n_feeds=N_FEEDS, f_center_mhz=CHAN_MHZ[844],
                        f_tone_bb=1300.0, seed=1)
    return {
        "input_dir": data,
        "output_dir": tmp_path / "out",
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }


def test_generic_combine_valueerror_soft_fails_and_records_skip(
    tmp_path, monkeypatch, capsys
):
    """Untyped integrity refusals from combine's validation pass must not kill
    the run either: ValueError is the soft-fail net, not the two typed members."""
    monkeypatch.chdir(REPO_ROOT)
    kwargs = _completed_scan_kwargs(tmp_path)

    import pilot_proxy.archive.combine as combine_mod

    def _raise_plain(*a, **k):
        raise ValueError(
            "combine: frame identity length does not match frame_index length")

    monkeypatch.setattr(combine_mod, "combine_detector_products", _raise_plain)
    result = run_chime_scan(**kwargs)
    out = kwargs["output_dir"]
    # the scan succeeded and the per-pilot product survived ...
    assert (out / "_per_pilot" / "844.npz").exists()
    assert result["per_pilot_work_dir"] == out / "_per_pilot"
    assert not (out / "chime_detector_outputs.npz").exists()
    printed = capsys.readouterr().out
    assert "terminal combine skipped" in printed
    assert "frame identity length" in printed          # the exception text
    assert "chime-combine --report" in printed
    # ... and the skip outlives stdout
    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["terminal_combine"] == {
        "status": "skipped",
        "error": "ValueError",
        "message": "combine: frame identity length does not match "
                   "frame_index length",
    }


def test_duplicate_identity_message_names_working_escape(
    tmp_path, monkeypatch, capsys
):
    """The duplicate-identity guidance must describe what the error means and an
    escape that works: (event, frame) identity, unit_scope/--report inspection,
    and --drop -- not the disproven two-archive-scopes cause or a plain
    chime-combine rerun that re-raises the same error."""
    monkeypatch.chdir(REPO_ROOT)
    kwargs = _completed_scan_kwargs(tmp_path)

    import pilot_proxy.archive.combine as combine_mod
    from pilot_proxy.archive.combine import CombineDuplicateIdentityError

    def _raise_duplicate(*a, **k):
        raise CombineDuplicateIdentityError(
            "combine: ch14/freq_id 844 contains duplicate (event, frame) "
            "identities")

    monkeypatch.setattr(combine_mod, "combine_detector_products",
                        _raise_duplicate)
    result = run_chime_scan(**kwargs)
    out = kwargs["output_dir"]
    assert result["per_pilot_work_dir"] == out / "_per_pilot"
    assert not (out / "chime_detector_outputs.npz").exists()
    printed = capsys.readouterr().out
    assert "terminal combine skipped" in printed
    assert "data-integrity signal" in printed
    assert "two archive scopes" not in printed
    assert "(source_event_key, frame_in_unit)" in printed
    assert "unit_scope" in printed
    assert "chime-combine --report" in printed
    assert "--drop" in printed
    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["terminal_combine"]["status"] == "skipped"
    assert scope["terminal_combine"]["error"] == "CombineDuplicateIdentityError"


def test_non_valueerror_combine_failure_still_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    kwargs = _completed_scan_kwargs(tmp_path)

    import pilot_proxy.archive.combine as combine_mod

    def _raise_runtime(*a, **k):
        raise RuntimeError("injected combine crash")

    monkeypatch.setattr(combine_mod, "combine_detector_products", _raise_runtime)
    with pytest.raises(RuntimeError, match="injected combine crash"):
        run_chime_scan(**kwargs)
    scope = json.loads((kwargs["output_dir"] / "scan_scope.json").read_text())
    assert "terminal_combine" not in scope     # a crash is not a recorded skip


# -- --source-freq-id-regex reaches the local source ---------------------------

def test_local_scan_freq_id_regex_reaches_source(tmp_path, monkeypatch):
    """The flag must populate source_freq_id_regex (what the paired
    LocalDirectorySource reads)."""
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    # Default parser expects _<freq_id>.h5; this layout needs the custom regex.
    fmt.make_synth_file(str(data / "evt1.freq844.h5"), n_time=NFFT * 2,
                        n_feeds=N_FEEDS, f_center_mhz=CHAN_MHZ[844],
                        f_tone_bb=1300.0, seed=11)
    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="local", input_dir=data,
                   analyzer="pilot-proxy-detector", select="844",
                   source_freq_id_regex=r"freq(\d+)\.h5$",
                   analyzer_options=_cpu_detector_options(), verbose=False)
    assert (out / "_per_pilot" / "844.npz").exists()


def test_local_scan_set_regex_overrides_flag(tmp_path, monkeypatch):
    """An explicit --set source_freq_id_regex=... takes precedence over the flag."""
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "evt1.freq844.h5"), n_time=NFFT * 2,
                        n_feeds=N_FEEDS, f_center_mhz=CHAN_MHZ[844],
                        f_tone_bb=1300.0, seed=12)
    opts = _cpu_detector_options()
    opts["source_freq_id_regex"] = r"freq(\d+)\.h5$"     # the explicit --set
    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="local", input_dir=data,
                   analyzer="pilot-proxy-detector", select="844",
                   source_freq_id_regex=r"nomatch(\d+)$",  # flag must NOT win
                   analyzer_options=opts, verbose=False)
    assert (out / "_per_pilot" / "844.npz").exists()


def test_scan_scope_records_the_storage_service_override(tmp_path, monkeypatch):
    # The scope is the provenance record of how a product was produced. A run
    # routed straight at a replica (locator degraded) must say so, or two runs
    # with different data paths are indistinguishable after the fact.
    monkeypatch.chdir(REPO_ROOT)
    import pilot_proxy.archive.pipeline as _dpl
    from pilot_proxy.archive.sources.cadc import STORAGE_SERVICE_ENV

    class _Stop(Exception):
        pass

    def _spy(*a, **k):
        raise _Stop

    monkeypatch.setattr(_dpl, "run", _spy)
    monkeypatch.setenv(STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/uvic/minoc")
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_829.h5"), n_time=NFFT,
                        n_feeds=N_FEEDS, f_center_mhz=476.3125, f_tone_bb=1200.0,
                        seed=1)

    with pytest.raises(_Stop):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "ovr",
                       source="local", analyzer="pilot-proxy-detector",
                       select="829", analyzer_options=_cpu_detector_options(),
                       verbose=False)
    scope = json.loads((tmp_path / "ovr" / "scan_scope.json").read_text())
    assert scope["execution"]["storage_service"] == "ivo://cadc.nrc.ca/uvic/minoc"
    assert scope["execution_attempts"][-1]["storage_service"] == (
        "ivo://cadc.nrc.ca/uvic/minoc")
