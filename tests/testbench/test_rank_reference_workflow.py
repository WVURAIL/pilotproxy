"""Rank-study orchestration, paired uncertainty, and bounded numerical runs."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.stats import ncf

from pilot_proxy.testbench import rank_reference as rank


def test_table_contains_the_independent_five_bin_probability(monkeypatch):
    x = np.asarray([1.03, 1.10, 1.18])
    snr_db = np.asarray([-20.0, -10.0, -5.0])
    monkeypatch.setattr(rank, "XG", x)
    monkeypatch.setattr(rank, "SG", snr_db)
    table = rank.build_table()
    gain = (np.sinc(0.5) / np.sinc(0.5 / 128)) ** 2
    gains = np.asarray([0, gain, 1, gain, 0])
    expected = np.asarray([
        ncf.cdf(x[:, None], 4096, 8192, 4096 * 10 ** (s / 10) * gains).prod(axis=1)
        for s in snr_db
    ])
    np.testing.assert_allclose(table, expected, rtol=2e-12, atol=1e-15)
    assert np.all(np.diff(table, axis=1) >= 0)
    assert np.all(np.diff(table, axis=0) <= 0)


def test_batches_reuse_clean_draws_and_contamination_prefixes(monkeypatch):
    monkeypatch.setattr(rank, "NBATCH", 2)
    monkeypatch.setattr(rank, "NF", 2)
    monkeypatch.setattr(rank, "KS", (1, 2))
    clean = np.asarray([[1, 4, 2], [3, 6, 5]], dtype=float)
    hot = np.asarray([[20, 30], [40, 50]], dtype=float)
    draws = []

    class Draws:
        def f(self, numerator, denominator, size):
            assert (numerator, denominator, size) == (4096, 8192, (2, 3))
            draws.append("clean")
            return clean.copy()

        def noncentral_f(self, numerator, denominator, noncentrality, size):
            assert (numerator, denominator, noncentrality, size) == (4096, 8192, 409600, (2, 2))
            draws.append("hot")
            return hot.copy()

    monkeypatch.setattr(rank.np.random, "default_rng", lambda seed: Draws())
    threshold_refs, detection_refs = [], []

    def threshold(ref):
        threshold_refs.append(ref.copy())
        return float(np.mean(ref))

    def missing_table(eta, ref, table):
        detection_refs.append((eta, ref.copy()))
        return None

    monkeypatch.setattr(rank, "eta_for", threshold)
    monkeypatch.setattr(rank, "snr_table", missing_table)
    monkeypatch.setattr(rank, "snr_direct", lambda eta, ref: float(np.mean(ref) - eta))
    dets, cases, snr, etas, pooled, direct = rank.run_size(3, object(), seed=10)
    assert dets == ["CA", 1, 2, 3] and cases == [0, 1, 2]
    assert draws == ["clean", "hot", "clean", "hot"]
    expected = [
        [[7 / 3, 14 / 3], [1, 3], [2, 5], [4, 6]],
        [[26 / 3, 17], [2, 5], [4, 6], [20, 40]],
        [[52 / 3, 95 / 3], [2, 5], [20, 40], [30, 50]],
    ]
    np.testing.assert_allclose(threshold_refs, expected[0] * 2)
    for batch in range(2):
        for case in cases:
            for index, detector in enumerate(dets):
                eta, reference = detection_refs[batch * 12 + case * 4 + index]
                np.testing.assert_allclose(reference, expected[case][index])
                assert eta == pytest.approx(sum(expected[0][index]) / 2)
                broken = case > 0 and (detector == "CA" or detector > 3 - case)
                assert direct[case][detector] == (0 if broken else 2)
                assert all(value is None for value in snr[case][detector]) == broken
                if broken:
                    np.testing.assert_allclose(pooled[case][detector][batch], reference)
    assert all(len(values) == 2 for values in etas.values())


def _summary_inputs():
    dets, cases = ["CA", 63], [0, 1]
    snr = {
        0: {"CA": [-10.0, -8.0, -6.0], 63: [-9.0, -5.0, -1.0]},
        1: {"CA": [None, None, None], 63: [-8.0, -5.0, -2.0]},
    }
    etas = {"CA": [1.0, 1.1, 1.2], 63: [1.1, 1.2, 1.3]}
    pooled = {
        0: {"CA": [np.ones(2), np.ones(2)], 63: [np.ones(2), np.ones(2)]},
        1: {"CA": [np.asarray([10.0, 20.0])] * 3, 63: []},
    }
    direct = {0: {"CA": 0, 63: 1}, 1: {"CA": 0, 63: 3}}
    return dets, cases, snr, etas, pooled, direct


def test_summary_reports_paired_errors_solver_methods_and_supplied_table(monkeypatch):
    monkeypatch.setattr(rank, "NBATCH", 3)
    monkeypatch.setattr(rank, "NF", 2)
    supplied_table = object()
    seen = []

    def table_solution(eta, ref, table):
        assert table is supplied_table
        seen.append(ref.copy())
        return -8.0

    monkeypatch.setattr(rank, "snr_table", table_solution)
    monkeypatch.setattr(rank, "eta_for", lambda ref, exact=False: 1.1)
    monkeypatch.setattr(rank, "snr_direct", lambda eta, ref, **kwargs: float(np.mean(ref)))
    result = rank.summarise(125, *_summary_inputs(), tab=supplied_table)
    assert result["ca_clean_snr_pd90_db"] == pytest.approx({"mean": -8, "se": 2 / np.sqrt(3)})
    assert result["eta_se"]["CA"] == pytest.approx(0.1 / np.sqrt(3))
    clean = result["cases"]["clean"]
    assert clean["CA"]["loss_db"] == 0 and clean["CA"]["se_db"] == 0
    assert clean["CA"]["method"] == "table, 3 batches"
    assert clean["63"]["loss_db"] == 3
    assert clean["63"]["se_db"] == pytest.approx(2 / np.sqrt(3))
    assert clean["63"]["method"] == "table and direct, 3 batches"
    contaminated = result["cases"]["k1_20dB"]
    assert contaminated["63"]["method"] == "direct, 3 batches"
    assert contaminated["63"]["loss_db"] == 3
    assert contaminated["63"]["se_db"] == pytest.approx(1 / np.sqrt(3))
    assert contaminated["CA"] == {
        "loss_db": 23, "se_db": None, "method": "direct, pooled",
        "batches_table_solved": 0, "broken": True,
    }
    assert len(seen) == 2


def test_summary_refuses_incomplete_pooled_references(monkeypatch):
    monkeypatch.setattr(rank, "NBATCH", 3)
    monkeypatch.setattr(rank, "NF", 2)
    values = _summary_inputs()
    values[4][1]["CA"].pop()
    with pytest.raises(RuntimeError, match="references for the direct solve are incomplete"):
        rank.summarise(125, *values, tab=object())


def test_reduced_rank_study_runs_real_numerics_and_writes_matching_json(monkeypatch, tmp_path):
    monkeypatch.setattr(rank, "F5", None)
    monkeypatch.setattr(rank, "NBATCH", 2)
    monkeypatch.setattr(rank, "NF", 8)
    monkeypatch.setattr(rank, "KS", (1,))
    monkeypatch.setattr(rank, "SG", np.arange(-10.0, -3.0, 0.5))
    # Keep the recorded sizes, seeds, x spacing, distributions and root solves.
    out = tmp_path / "rank"
    result = rank.run(out)
    written = json.loads((out / "rank_bench.json").read_text())
    assert written == json.loads(json.dumps(result))
    assert "not an operational eta_Pfa" in (out / "REFERENCE.md").read_text()
    assert written["batches"] == 2 and written["frames_per_batch"] == 8
    assert written["snr_grid_db"] == [-10.0, -3.5, 0.5]
    assert set(written["sizes"]) == {"125", "126"}
    for size, summary in written["sizes"].items():
        assert len(summary["detectors"]) == int(size) + 1
        clean, contaminated = summary["cases"]["clean"], summary["cases"]["k1_20dB"]
        assert clean["CA"]["loss_db"] == 0
        assert clean["CA"]["se_db"] == 0
        assert clean["63"]["method"] == "table, 2 batches"
        assert abs(clean["63"]["loss_db"]) < 0.1
        for detector in ("CA", size):
            assert contaminated[detector]["method"] == "direct, pooled"
            assert contaminated[detector]["se_db"] is None
            assert contaminated[detector]["loss_db"] > clean[detector]["loss_db"]
        assert contaminated[size]["loss_db"] > contaminated["CA"]["loss_db"]
        for check in summary["table_vs_direct_check"].values():
            assert abs(check["eta_interp"] - check["eta_exact"]) < 2e-6
            # The coarse test SNR grid has interpolation error; this is not
            # the production grid's accuracy claim.
            assert abs(check["table"] - check["direct"]) < 0.05


def test_publication_writes_missing_uncertainty_as_json_null(monkeypatch, tmp_path):
    monkeypatch.setattr(rank, "F5", None)
    monkeypatch.setattr(rank, "build_table", lambda: np.ones((2, 2)))
    monkeypatch.setattr(rank, "run_size", lambda n, tab, seed: ([], [], {}, {}, {}, {}))

    def summary(n, *args, **kwargs):
        return {
            "cases": {"clean": {str(k): {"loss_db": 0.0, "se_db": np.nan}
                                  for k in range(1, n + 1)}},
            "table_vs_direct_check": {"CA": {"table": None, "direct": -8.6}},
        }

    monkeypatch.setattr(rank, "summarise", summary)
    out = tmp_path / "strict-report"
    assert rank.main(["--out", str(out)]) == 0

    def reject_nonfinite(token):
        raise AssertionError(f"non-standard JSON token: {token}")

    written = json.loads((out / "rank_bench.json").read_text(), parse_constant=reject_nonfinite)
    assert written["sizes"]["125"]["cases"]["clean"]["63"]["se_db"] is None
    assert written["sizes"]["125"]["table_vs_direct_check"]["CA"] == {
        "table": None, "direct": -8.6,
    }
