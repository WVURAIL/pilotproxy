# coding=utf-8
"""Smoke gate for the coherent-gain Monte Carlo tool.

The critical assertion is the verify stage: the tool's batched
reduction must equal the packaged pipeline (bit-exact integer marginals,
float32-identical fine spectra). The MC smoke then checks the machinery
end to end at a reduced geometry: sane H0 thresholds and Pd -> 1 at
strong injection.
"""
from __future__ import annotations

import importlib.util
import pathlib
import json

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "measure_fine_gain", REPO / "tools" / "measure_fine_gain.py")
mfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfg)


def test_batched_reduction_matches_packaged_pipeline():
    mfg.verify(seed=7, trials=2, streams=32)


def test_mc_smoke_thresholds_and_detection(tmp_path):
    out = str(tmp_path)
    mfg.run_stage(out, "h0", trials=64, seed=1, streams=32, batch=8)
    mfg.run_stage(out, "sweep", trials=32, seed=2, snr_db=-10.0,
                  b0=mfg.ANCHOR, streams=32, batch=8)
    c0, f0, _ = mfg.collect(out, "h0_s*.npz")
    assert c0.size == 64 and f0.size == 64
    # H0 statistics center near 1 (null_power_ratio = 1 by construction)
    assert 0.9 < np.median(c0) < 1.1
    assert 0.9 < np.median(f0) < 1.3
    c1, f1, _ = mfg.collect(out, "h1_*dB_s*.npz")
    # strong injection: fine statistic far above the entire H0 sample
    assert f1.min() > f0.max()
    assert c1.min() > np.quantile(c0, 0.99)


def test_snr_amplitude_roundtrip():
    # amplitude formula inverts the documented SNR definition
    snr_db = -25.0
    amp = float(np.sqrt(2.0 * mfg.SIGMA**2 * 10 ** (snr_db / 10.0)))
    back = 10.0 * np.log10(amp**2 / (2.0 * mfg.SIGMA**2))
    assert abs(back - snr_db) < 1e-9


def _shard(path, *, b0=mfg.ANCHOR, streams=32, snr=-20., seed=1, fine=2., **metadata):
    np.savez(path, coarse=np.ones(10), fine=np.full(10, fine),
             snr_db=snr, b0=b0, streams=streams, sigma=mfg.SIGMA, seed=seed, **metadata)


def test_report_separates_saved_offsets_and_preserves_input_directory(tmp_path, monkeypatch):
    inputs, output = tmp_path / "inputs", tmp_path / "report"
    inputs.mkdir()
    _shard(inputs / "h0_s1.npz", snr=np.nan, fine=1.)
    _shard(inputs / "h1_centered.npz", fine=2.)
    _shard(inputs / "h1_half_offset.npz", b0=mfg.HALF_BIN, fine=.5)
    # A misleading filename must not override recorded experimental identity.
    _shard(inputs / "h1_half_misnamed.npz", seed=2, fine=3.)
    original = {p.name: p.read_bytes() for p in inputs.iterdir()}
    figures = []
    monkeypatch.setattr(mfg, "make_fig", lambda *a, **kw: figures.append(kw))
    assert mfg.report(inputs, report_dir=output) == 0
    report = json.loads((output / "gain_report.json").read_text())
    centered, offset = report["curves"]["centered"][0], report["curves"]["_half"][0]
    assert centered["n"] == 20 and offset["n"] == 10
    assert centered["pd_fine_0.01"] == 1 and offset["pd_fine_0.01"] == 0
    assert all(row["legacy_geometry_inferred"] for row in report["sources"])
    assert figures == [{"streams": 32}]
    assert {p.name: p.read_bytes() for p in inputs.iterdir()} == original


@pytest.mark.parametrize("change,error", [
    ({"streams": 64}, "mixed experimental identities"),
    ({"sigma": 121.}, "mixed experimental identities"),
    ({"b0": 63.}, "unsupported injected offset"),
    ({"seed": -1}, "nonnegative integer"),
    ({"windows": 64}, "incomplete transform/statistic metadata"),
])
def test_report_refuses_incompatible_shards_before_writing(tmp_path, change, error):
    _shard(tmp_path / "h0_s1.npz", snr=np.nan)
    base = dict(coarse=np.ones(10), fine=np.ones(10), snr_db=-20.,
                b0=mfg.ANCHOR, streams=32, sigma=mfg.SIGMA, seed=2)
    np.savez(tmp_path / "h1_candidate.npz", **(base | change))
    with pytest.raises(ValueError, match=error):
        mfg.report(tmp_path, make_figure=False)
    assert not (tmp_path / "gain_report.npz").exists()


def test_report_rejects_duplicate_trials(tmp_path):
    _shard(tmp_path / "h0_s1.npz", snr=np.nan)
    _shard(tmp_path / "h1_a.npz")
    _shard(tmp_path / "h1_duplicate.npz")
    with pytest.raises(ValueError, match="duplicate seed"):
        mfg.report(tmp_path, make_figure=False)


def test_new_shard_records_geometry(tmp_path):
    mfg.run_stage(tmp_path, "h0", trials=2, seed=1, streams=4, batch=2)
    nulls, _, identity, sources = mfg.report_shards(tmp_path)
    assert len(nulls) == 1 and identity["streams"] == 4
    assert identity["windows"] == 128 and identity["bins"] == 256
    assert not sources[0]["legacy_geometry_inferred"]
