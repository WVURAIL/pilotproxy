# coding=utf-8
"""Name-carries-the-metadata guards.

The artifact a flag names already records the facts, so flags become optional
(derived from the artifact), assertions (must agree with it), or conflicts
(hard errors, never silent ignores):

  * ``chime-scan --select`` defaults to every freq_id the inventory contains
    (echoed before any staging), and stays required for ``--source local``;
  * ``chime-scan --source`` is inferred from --inventory/--inventory-name;
    conflicting pairings error instead of silently ignoring flags;
  * ``--instrument`` must match the inventory sidecar's telescope;
  * ``detect`` resolves the pilot from quantize's metadata.json sidecar,
    asserts an explicit flag against it, and refuses a bare matrix rather
    than guessing channel 14.
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

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.archive.scan import (
    _default_selection_from_inventory,
    _freq_ids_in_inventory,
    _parse_freq_id_list,
    run_chime_scan,
)
from pilot_proxy.detect import _resolve_pilot_request

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
N_FEEDS = 4
K = 128

# freq_id -> coarse-channel center (MHz)
CHAN_MHZ = {844: 470.3125, 829: 476.171875, 752: 506.171875}


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
    rng = np.random.default_rng(0)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8)
        for ch in range(10, 41)
    }
    return {
        "detector_fn": _stub_detector_fn,
        "kernel": _stub_kernel(K),
        "weights_by_channel": weights_by_channel,
    }


def _make_inventory(inv_path, *, common_path, event, channels, extra_rows=()):
    with open(inv_path, "w") as fh:
        for ch in channels:
            fh.write(json.dumps({"common_path": common_path, "event": event,
                                 "scope": "test.scope",
                                 "name": f"baseband_{event}_{int(ch)}.h5",
                                 "freq_id": int(ch), "size_bytes": 1}) + "\n")
        for row in extra_rows:
            fh.write(json.dumps(row) + "\n")


def _write_meta(inv_path, **fields):
    metadata = {
        "datatrawl_inventory": 1,
        "telescope": "chime",
        "source": "cadc-datatrail",
        "reader": "chime-baseband",
    }
    metadata.update(fields)
    inv_path.with_suffix(".meta.json").write_text(json.dumps(metadata) + "\n")


# -- default selection: the inventory is the scope ----------------------------

def test_default_select_scans_every_inventory_freq_id(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
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
                    channels=sorted(CHAN_MHZ))
    _write_meta(inv, telescope="chime", freq_ids="752,829,844")

    from pilot_proxy.archive.sources.cadc import CadcDatatrailSource

    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(synth[int(unit.meta["freq_id"])], dest)
        return True, ""

    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch)
    monkeypatch.setattr(
        CadcDatatrailSource,
        "fetch_preflight",
        lambda self, ctx: (True, [], []),
    )

    out = tmp_path / "out"
    # no source, no select: source inferred from --inventory, select from rows
    run_chime_scan(output_dir=out, inventory=inv,
                   analyzer="pilot-proxy-detector",
                   analyzer_options=_cpu_detector_options(), verbose=True)

    work = out / "_per_pilot"
    for ch in CHAN_MHZ:
        assert (work / f"{ch}.npz").exists()
    assert (out / "chime_detector_outputs.npz").exists()

    printed = capsys.readouterr().out
    assert "source: cadc-datatrail (inferred from --inventory)" in printed
    assert "scanning all 3 freq_id(s)" in printed
    assert "752,829,844" in printed
    assert "note: survey requested" not in printed  # request == rows: no note


def test_default_select_notes_requested_vs_found(tmp_path, capsys):
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[829, 844])
    _write_meta(inv, telescope="chime", freq_ids="752,829,844")
    from pilot_proxy.archive.scan import _read_inventory_meta

    found = _default_selection_from_inventory(
        inv, label="chime-pilots", meta=_read_inventory_meta(inv), verbose=True)
    assert found == [829, 844]
    printed = capsys.readouterr().out
    assert "scanning all 2 freq_id(s) from inventory 'chime-pilots'" in printed
    assert "survey requested 3 freq_id(s); inventory rows cover 2" in printed
    assert "missing from inventory: 752" in printed


def test_empty_inventory_refuses_default_selection(tmp_path):
    inv = tmp_path / "inventory.jsonl"
    # companion-only inventory: rows exist, none carry a freq_id
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[],
                    extra_rows=[{"common_path": "cadc:TEST", "event": "evt1",
                                 "name": "gains.h5"}])
    with pytest.raises(SystemExit, match="no rows with a freq_id"):
        _default_selection_from_inventory(inv, label="x", meta=None, verbose=False)


def test_freq_ids_in_inventory_skips_companion_and_malformed_rows(tmp_path):
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(
        inv, common_path="cadc:TEST", event="evt1", channels=[844, 829],
        extra_rows=[
            {"common_path": "cadc:TEST", "event": "evt1", "name": "gains.h5"},
            {"common_path": "cadc:TEST", "event": "evt1", "freq_id": None},
        ])
    with open(inv, "a") as fh:
        fh.write("not json\n")
    assert _freq_ids_in_inventory(inv) == [829, 844]
    with pytest.raises(SystemExit, match="inventory not found"):
        _freq_ids_in_inventory(tmp_path / "missing.jsonl")


def test_parse_freq_id_list_grammar():
    assert _parse_freq_id_list("506-508,521") == [506, 507, 508, 521]
    assert _parse_freq_id_list([844, 829]) == [829, 844]
    assert _parse_freq_id_list("all") is None
    assert _parse_freq_id_list(None) is None


def test_local_source_still_requires_select(tmp_path):
    # bare --source-root keeps the historic local default, and local scans
    # have no inventory to derive the scope from
    with pytest.raises(SystemExit, match="--select is required for --source local"):
        run_chime_scan(output_dir=tmp_path / "o", source_root=tmp_path,
                       analyzer="pilot-proxy-detector", verbose=False)


# -- source inference and conflicts -------------------------------------------

def test_conflicting_source_flags_are_errors(tmp_path):
    with pytest.raises(SystemExit, match="belong to --source cadc-datatrail"):
        run_chime_scan(output_dir=tmp_path / "o", source="local",
                       inventory_name="chime-pilots", select="844",
                       analyzer="pilot-proxy-detector", verbose=False)
    with pytest.raises(SystemExit, match="--input-dir belongs to --source local"):
        run_chime_scan(output_dir=tmp_path / "o", source="cadc-datatrail",
                       input_dir=tmp_path, inventory=tmp_path / "inventory.jsonl",
                       select="844", analyzer="pilot-proxy-detector", verbose=False)


def test_instrument_must_match_inventory_telescope(tmp_path):
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[844])
    _write_meta(inv, telescope="hirax")
    with pytest.raises(SystemExit, match="telescope='hirax'.*expected 'chime'"):
        run_chime_scan(output_dir=tmp_path / "o", inventory=inv, select="844",
                       analyzer="pilot-proxy-detector", verbose=False)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("not json", "invalid inventory metadata"),
        ({"datatrawl_inventory": 2}, "must be exactly 1"),
        (
            {
                "datatrawl_inventory": 1,
                "telescope": "chime",
                "source": "local",
                "reader": "chime-baseband",
            },
            "source='local'.*expected 'cadc-datatrail'",
        ),
        (
            {
                "datatrawl_inventory": 1,
                "telescope": "chime",
                "source": "cadc-datatrail",
                "reader": "other-reader",
            },
            "reader='other-reader'.*expected 'chime-baseband'",
        ),
    ],
)
def test_scan_rejects_incompatible_inventory_metadata(
    tmp_path,
    metadata,
    message,
):
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1", channels=[844])
    text = metadata if isinstance(metadata, str) else json.dumps(metadata)
    inv.with_suffix(".meta.json").write_text(text + "\n")

    with pytest.raises(SystemExit, match=message):
        run_chime_scan(
            output_dir=tmp_path / "o",
            inventory=inv,
            select="844",
            analyzer="pilot-proxy-detector",
            verbose=False,
        )


# -- detect: the quantize sidecar is authoritative ----------------------------

CH14_HZ = physical_channel_to_pilot_hz(14)


def _matrix_with_sidecar(tmp_path, pilot_hz=CH14_HZ):
    d = tmp_path / "detector_input"
    d.mkdir()
    matrix = d / "detector_matrix_i4.npy"
    matrix.write_bytes(b"")  # never loaded by the resolver
    (d / "metadata.json").write_text(json.dumps({"dtv_pilot_hz": float(pilot_hz)}))
    return matrix


def test_detect_pilot_defaults_from_sidecar(tmp_path, capsys):
    matrix = _matrix_with_sidecar(tmp_path)
    ch, mhz = _resolve_pilot_request(None, None, matrix, tolerance_hz=1.0)
    assert ch is None
    assert mhz == pytest.approx(CH14_HZ / 1e6)
    assert "Pilot frequency from sidecar" in capsys.readouterr().out


def test_detect_flag_must_agree_with_sidecar(tmp_path):
    matrix = _matrix_with_sidecar(tmp_path)
    with pytest.raises(SystemExit, match="disagrees with the matrix sidecar"):
        _resolve_pilot_request(15, None, matrix, tolerance_hz=1.0)
    # agreeing flag passes through unchanged
    assert _resolve_pilot_request(14, None, matrix, tolerance_hz=1.0) == (14, None)


def test_detect_refuses_bare_matrix(tmp_path):
    d = tmp_path / "no_sidecar"
    d.mkdir()
    matrix = d / "detector_matrix_i4.npy"
    matrix.write_bytes(b"")
    with pytest.raises(SystemExit, match="Pass the pilot identity explicitly"):
        _resolve_pilot_request(None, None, matrix, tolerance_hz=1.0)


def test_detect_rejects_both_flags(tmp_path):
    matrix = _matrix_with_sidecar(tmp_path)
    with pytest.raises(SystemExit, match="not both"):
        _resolve_pilot_request(14, 470.309441, matrix, tolerance_hz=1.0)


@pytest.mark.parametrize(
    "invalid_tolerance",
    [float("nan"), float("inf"), -float("inf"), -1.0, "bad"],
)
def test_detect_rejects_invalid_pilot_tolerance(
    tmp_path,
    invalid_tolerance: object,
) -> None:
    matrix = _matrix_with_sidecar(tmp_path)

    with pytest.raises(SystemExit, match="must be finite and nonnegative"):
        _resolve_pilot_request(
            14,
            None,
            matrix,
            tolerance_hz=invalid_tolerance,
        )
