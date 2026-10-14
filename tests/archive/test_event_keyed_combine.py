# coding=utf-8
"""Event-keyed combine: frames align by (event, frame-in-file) identity.

The archive is ragged (not every channel holds every event), so positional
frame stacking cannot work at production scale. These tests cover:

  * the alignment core: pass-through parity on fully aligned inputs, reference
    (lowest-channel) ordering on ragged inputs, partial-event frame drops,
    the typed empty-intersection error, duplicate-identity refusal, and the
    refusal of products without event-keyed identity;
  * the scan layer end-to-end (offline, CPU-stub detector): a ragged
    inventory combines over the intersection with drops recorded in
    stats.json and the frame-identity sidecar, and a disjoint inventory
    soft-fails the terminal combine while the scan itself succeeds;
  * `report_products` and the `drop_freq_ids` subset knob.
"""
from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt

from pilot_proxy.archive.combine import (
    CombineDuplicateIdentityError,
    CombineEmptyIntersectionError,
    _align_frames,
    combine_detector_products,
    report_products,
)
from pilot_proxy.archive.scan import run_chime_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
N_FEEDS = 4
K = 128

CHAN_MHZ = {844: 470.3125, 829: 476.171875, 752: 506.171875}


# -- alignment core -----------------------------------------------------------

def _mem_product(channel, freq_id, events_frames):
    """In-memory per-pilot product with just the fields alignment touches.

    ``events_frames``: sequence of (event, n_frames) in processing order.
    """
    ev, unit_idx, in_unit, values = [], [], [], []
    for u, (event, n) in enumerate(events_frames):
        ev.append(str(event))
        for f in range(n):
            unit_idx.append(u)
            in_unit.append(f)
            values.append(float(f"{channel}.{u}{f}"))
    n_frames = len(unit_idx)
    return {
        "physical_channel": np.asarray([channel], dtype=np.int32),
        "freq_id": np.asarray([freq_id], dtype=np.int64),
        "frame_index": np.arange(n_frames, dtype=np.int64),
        "source_event_keys": np.asarray(ev),
        "frame_unit_index": np.asarray(unit_idx, dtype=np.int32),
        "frame_in_unit": np.asarray(in_unit, dtype=np.int32),
        "coarse_power_ratio": np.asarray(values, dtype=np.float64).reshape(n_frames, 1),
    }


def test_align_fully_aligned_is_passthrough():
    a = _mem_product(14, 844, [("100", 2), ("200", 2)])
    b = _mem_product(15, 829, [("100", 2), ("200", 2)])
    aligned, frame_index, info = _align_frames([a, b])
    assert info["mode"] == "event_keyed"
    assert np.array_equal(frame_index, np.arange(4))
    for orig, out in zip((a, b), aligned):
        assert np.array_equal(out["coarse_power_ratio"], orig["coarse_power_ratio"])
    assert all(p["n_frames_dropped"] == 0 for p in info["by_pilot"])


def test_align_ragged_intersects_in_reference_order():
    a = _mem_product(14, 844, [("100", 1), ("200", 1), ("300", 1)])
    b = _mem_product(15, 829, [("200", 1), ("300", 1), ("400", 1)])
    aligned, frame_index, info = _align_frames([a, b])
    assert info["n_frames_common"] == 2 and info["n_events_common"] == 2
    assert info["frame_event_key"] == ["200", "300"]  # reference (a) order
    # b's frames gathered by identity rather than position
    assert np.array_equal(aligned[1]["coarse_power_ratio"].reshape(-1),
                          b["coarse_power_ratio"].reshape(-1)[[0, 1]])
    assert np.array_equal(aligned[0]["coarse_power_ratio"].reshape(-1),
                          a["coarse_power_ratio"].reshape(-1)[[1, 2]])
    drops = {p["freq_id"]: p for p in info["by_pilot"]}
    assert drops[844]["n_frames_dropped"] == 1 and drops[844]["n_events_dropped"] == 1
    assert drops[829]["n_frames_dropped"] == 1 and drops[829]["n_events_dropped"] == 1


def test_align_partial_event_keeps_common_frames():
    a = _mem_product(14, 844, [("100", 3)])
    b = _mem_product(15, 829, [("100", 2)])
    aligned, frame_index, info = _align_frames([a, b])
    assert info["n_frames_common"] == 2 and info["n_events_common"] == 1
    drops = {p["freq_id"]: p for p in info["by_pilot"]}
    assert drops[844]["n_frames_dropped"] == 1 and drops[844]["n_events_dropped"] == 0


def test_align_empty_intersection_raises_typed_error():
    a = _mem_product(14, 844, [("100", 1)])
    b = _mem_product(15, 829, [("200", 1)])
    with pytest.raises(CombineEmptyIntersectionError, match="--report"):
        _align_frames([a, b])


def test_align_duplicate_identity_raises_typed_error():
    a = _mem_product(14, 844, [("100", 1), ("100", 1)])
    b = _mem_product(15, 829, [("100", 1)])
    with pytest.raises(CombineDuplicateIdentityError, match="unit_scope"):
        _align_frames([a, b])
    # still a ValueError, so existing callers that catch broadly keep working
    assert issubclass(CombineDuplicateIdentityError, ValueError)


# -- scan layer end-to-end (offline) ------------------------------------------

def _stub_detector_fn(*, packed, weights, kernel):
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    n = int(pk.shape[0])
    return {
        "batch": n,
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "results": [
            {
                "block_index": b,
                "mask": 0,
                "p_target_u64": 10,
                "p_ref_lower_u64": 10,
                "p_ref_upper_u64": 10,
                "p_ref_sum_u64": 20,
            }
            for b in range(n)
        ],
    }


def _stub_kernel(k):
    specs = SimpleNamespace(
        K=k, N=3, bits=4, reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": k, "num_weight_terms": 3,
            "sample_bits_per_component": 4, "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _cpu_detector_options():
    # three identical weight rows => reference_norm_sum_sq == 2 * target_norm_sq, so
    # with the stub's p_target=10 / p_ref_sum=20 the positive-excess rule sits
    # exactly at equality and mask=0 is rule-consistent (validate-products
    # recomputes the mask from p's and norms and must agree).
    rng = np.random.default_rng(0)
    row = rng.integers(-120, 121, size=(1, K)).astype(np.int8)
    weights_by_channel = {ch: np.repeat(row, 3, axis=0) for ch in range(10, 41)}
    return {
        "detector_fn": _stub_detector_fn,
        "kernel": _stub_kernel(K),
        "weights_by_channel": weights_by_channel,
    }


def _ragged_archive(tmp_path, monkeypatch, events_by_channel):
    """Synth files + inventory + offline fetch for a per-channel event layout."""
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    synth = {}
    rows = []
    for ch, events in events_by_channel.items():
        for event in events:
            p = data / f"baseband_{event}_{ch}.h5"
            fmt.make_synth_file(str(p), n_time=NFFT * 2, n_feeds=N_FEEDS,
                                f_center_mhz=CHAN_MHZ[ch],
                                f_tone_bb=1300.0, seed=ch + int(event))
            synth[(str(event), ch)] = p
            rows.append({"common_path": "cadc:TEST", "event": str(event),
                         "scope": "test.scope",
                         "name": f"baseband_{event}_{int(ch)}.h5",
                         "freq_id": int(ch), "size_bytes": 1})
    inv = tmp_path / "inventory.jsonl"
    with open(inv, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource

    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(synth[(str(unit.meta["event"]),
                               int(unit.meta["freq_id"]))], dest)
        return True, ""

    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch)
    monkeypatch.setattr(
        CadcDatatrailSource,
        "fetch_preflight",
        lambda self, ctx: (True, [], []),
    )
    return inv


def test_scan_ragged_inventory_combines_intersection(tmp_path, monkeypatch):
    inv = _ragged_archive(tmp_path, monkeypatch, {
        844: ["100", "200"],
        829: ["100", "200"],
        752: ["100"],
    })
    out = tmp_path / "out"
    run_chime_scan(output_dir=out, inventory=inv,
                   analyzer="pilot-proxy-detector",
                   analyzer_options=_cpu_detector_options(), verbose=False)

    combined = out / "chime_detector_outputs.npz"
    assert combined.exists()
    with np.load(combined) as z:
        # one common event, NFFT*2 samples -> 2 frames, stacked over 3 pilots
        assert z["frame_index"].size == 2
        assert z["coarse_power_ratio"].shape == (2, 3)
    stats = json.loads((out / "stats.json").read_text())
    align = stats["combine_alignment"]
    assert align["mode"] == "event_keyed"
    assert align["n_events_common"] == 1 and align["n_frames_common"] == 2
    drops = {p["freq_id"]: p["n_frames_dropped"] for p in align["by_pilot"]}
    assert drops == {844: 2, 829: 2, 752: 0}
    with np.load(out / "chime_frame_identity.npz") as ident:
        # Full-schema rows key units by the source's logical unit key
        # (scope/event/name triple), freq_id token stripped per pilot.
        expected_key = "cadc-datatrail:" + json.dumps(
            ["test.scope", "100", "baseband_100.h5"], separators=(",", ":"))
        assert set(ident["frame_event_key"].tolist()) == {expected_key}

    # report + subset knob over the same work dir
    work = sorted((out / "_per_pilot").glob("*.npz"))
    report = report_products(work)
    assert "intersection of all 3 pilots: 1" in report
    out2 = tmp_path / "out2"
    combine_detector_products(work, out2, drop_freq_ids=[752])
    with np.load(out2 / "chime_detector_outputs.npz") as z:
        assert z["frame_index"].size == 4  # both events, two pilots
        assert z["coarse_power_ratio"].shape == (4, 2)

    # spectrogram figures must render with event-boundary ticks driven by
    # the frame-identity sidecar (out2 stitches two events -> one boundary)
    pytest.importorskip("matplotlib")
    from pilot_proxy.chime.plots import _event_boundaries, plot_mask_spectrogram
    change, n_events = _event_boundaries(out2)
    assert n_events == 2 and list(change) == [2]
    figs = plot_mask_spectrogram(out2)
    assert figs[0].exists()

    # validate-products must accept keyed-combine run dirs (frame-identity
    # sidecar file, combine_alignment stats block)
    import subprocess, sys
    for run in (out, out2):
        proc = subprocess.run(
            [sys.executable, "-m", "pilot_proxy.chime.validate_products",
             "--run-dir", str(run)],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin",
                 "MPLBACKEND": "Agg"},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_scan_disjoint_inventory_soft_fails_terminal_combine(tmp_path, monkeypatch,
                                                             capsys):
    inv = _ragged_archive(tmp_path, monkeypatch, {
        844: ["100"],
        829: ["200"],
        752: ["300"],
    })
    out = tmp_path / "out"
    result = run_chime_scan(output_dir=out, inventory=inv,
                            analyzer="pilot-proxy-detector",
                            analyzer_options=_cpu_detector_options(),
                            verbose=False)
    # the scan itself succeeded: all per-pilot products exist ...
    for ch in (844, 829, 752):
        assert (out / "_per_pilot" / f"{ch}.npz").exists()
    assert result["per_pilot_work_dir"] == out / "_per_pilot"
    # ... but no combined product was written, and the operator was told why
    assert not (out / "chime_detector_outputs.npz").exists()
    printed = capsys.readouterr().out
    assert "terminal combine skipped" in printed
    assert "chime-combine --report" in printed


def test_scan_duplicate_identity_soft_fails_terminal_combine(tmp_path, monkeypatch,
                                                             capsys):
    """A duplicate (event, frame) identity must not fail the whole scan.

    Weeks of per-pilot compute must survive a terminal-combine refusal. The
    duplicate is injected at the combine boundary rather than synthesized from
    an inventory: the conditions that can produce one are a property of the
    source keys, and this test is about the scan layer's response to it.
    """
    inv = _ragged_archive(tmp_path, monkeypatch, {844: ["100"], 829: ["100"]})
    out = tmp_path / "out"

    import pilot_proxy.archive.combine as combine_mod

    def _raise_duplicate(*a, **k):
        raise CombineDuplicateIdentityError(
            "combine: ch14/freq_id 844 contains duplicate (event, frame) "
            "identities; check unit_scope on that product")

    monkeypatch.setattr(combine_mod, "combine_detector_products", _raise_duplicate)
    result = run_chime_scan(output_dir=out, inventory=inv,
                            analyzer="pilot-proxy-detector",
                            analyzer_options=_cpu_detector_options(),
                            verbose=False)
    # the scan succeeded and every per-pilot product survived ...
    for ch in (844, 829):
        assert (out / "_per_pilot" / f"{ch}.npz").exists()
    assert result["per_pilot_work_dir"] == out / "_per_pilot"
    # ... but the stack was withheld and flagged as integrity, not routine
    assert not (out / "chime_detector_outputs.npz").exists()
    printed = capsys.readouterr().out
    assert "terminal combine skipped" in printed
    assert "data-integrity signal" in printed


def _ver(source, kernel="2548aef", version="0.2.0.dev0"):
    return (f"pilot-proxy/{version} source={source} kernel=1.0.0 "
            f"kernel_sha256={kernel} pilotproxy_per_pilot_product_v1 K=128")


def test_invariants_allow_mixed_source_builds_same_geometry():
    from pilot_proxy.archive.combine import _check_invariants
    a = {"detector_version": np.asarray([_ver("aaa111")]),
         "nfft": np.asarray([16384])}
    b = {"detector_version": np.asarray([_ver("bbb222")]),
         "nfft": np.asarray([16384])}
    notes = _check_invariants([a, b], ("nfft", "detector_version"), "geometry")
    assert len(notes["detector_versions"]) == 2


def test_invariants_refuse_mixed_kernel_geometry():
    from pilot_proxy.archive.combine import _check_invariants
    a = {"detector_version": np.asarray([_ver("aaa111", kernel="k1")])}
    b = {"detector_version": np.asarray([_ver("aaa111", kernel="k2")])}
    with pytest.raises(ValueError, match="geometry tokens"):
        _check_invariants([a, b], ("detector_version",), "detector geometry")


def test_invariants_identical_versions_add_no_note():
    from pilot_proxy.archive.combine import _check_invariants
    a = {"detector_version": np.asarray([_ver("aaa111")])}
    b = {"detector_version": np.asarray([_ver("aaa111")])}
    assert _check_invariants([a, b], ("detector_version",), "g") == {}


def test_invariants_allow_release_version_bump_same_geometry():
    """A mid-survey release bump (0.3.0.dev0 -> 1.0.0) must not split the
    stack. Pilots finished before the bump carry the old label; pilots
    finished after carry the new one, and combine sees both in one run."""
    from pilot_proxy.archive.combine import _check_invariants
    a = {"detector_version": np.asarray([_ver("02066f1d", version="0.3.0.dev0")]),
         "nfft": np.asarray([16384])}
    b = {"detector_version": np.asarray([_ver("0c66af82", version="1.0.0")]),
         "nfft": np.asarray([16384])}
    notes = _check_invariants([a, b], ("nfft", "detector_version"), "geometry")
    assert len(notes["detector_versions"]) == 2


def test_combine_provenance_token_policy():
    """Combine may retain distinct builds; resume is tested as exact elsewhere."""
    from pilot_proxy.provenance import detector_version_geometry

    a, b = _ver("aaa111"), _ver("bbb222")
    assert a != b and detector_version_geometry(a) == detector_version_geometry(b)
    c = _ver("aaa111", version="1.0.0")
    assert a != c and detector_version_geometry(a) == detector_version_geometry(c)
    d = _ver("aaa111", kernel="other")
    assert detector_version_geometry(a) != detector_version_geometry(d)
