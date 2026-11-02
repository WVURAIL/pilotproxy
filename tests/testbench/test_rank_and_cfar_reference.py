"""Ideal-reference geometry and independent quadrature checks."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from pilot_proxy.testbench import cfar_reference, rank_quadrature, rank_reference


@pytest.mark.parametrize("frames,chunk,requests", [
    (5, 2, [2, 2, 1]), (2, 5, [2]), (6, 2, [2, 2, 2]),
    (np.int64(5), np.int32(2), [2, 2, 1]),
])
def test_cfar_thresholds_use_every_requested_frame(monkeypatch, frames, chunk, requests):
    """Known scores 1..N give an analytical quantile, median, and variance."""
    sizes = []

    class Draws:
        used = 0

        def chisquare(self, degrees, shape):
            rows, columns = shape
            values = np.full(shape, float(degrees))
            if degrees == 4096:
                sizes.append(rows)
                levels = np.arange(self.used + 1, self.used + rows + 1)
                values[:, :5] *= levels[:, None]
                self.used += rows
            else:
                assert degrees == 8192
            assert columns == 130
            return values

    monkeypatch.setattr(cfar_reference.np.random, "default_rng", lambda seed: Draws())
    result = cfar_reference.thresholds(
        cfar_reference.geometry("even"), frames=frames, chunk=chunk)
    assert sizes == requests
    for summary in result.values():
        assert summary["eta"] == pytest.approx(frames - (frames - 1) * 0.001)
        assert summary["median"] == pytest.approx((frames + 1) / 2)
        assert summary["std"] == pytest.approx(np.sqrt((frames ** 2 - 1) / 12))


@pytest.mark.parametrize("field", ["frames", "chunk"])
@pytest.mark.parametrize("value", [0, -1, True, np.bool_(True), 1.5, "2", None])
def test_cfar_thresholds_reject_invalid_sample_counts(field, value):
    settings = {"frames": 2, "chunk": 2, field: value}
    with pytest.raises(ValueError, match=field + " must be a positive integer"):
        cfar_reference.thresholds(cfar_reference.geometry("even"), **settings)


def test_cfar_noise_matches_central_and_noncentral_f_means():
    from scipy.stats import ncf

    count = 40_000
    noise = cfar_reference.Noise(count, 1, np.random.default_rng(20261101))
    for noncentrality in (0, 4096 * 0.25):
        values = noise.T(noncentrality)
        mean, variance = ncf.stats(4096, 8192, noncentrality, moments="mv")
        assert values.shape == (count, 1)
        assert abs(values.mean() - mean) < 6 * np.sqrt(variance / count)
        np.testing.assert_array_equal(noise.T(noncentrality), values)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_cfar_contamination_changes_only_the_requested_bulk_bins(parity):
    geo = cfar_reference.geometry(parity)
    clean = cfar_reference.lam_vector(geo, 0.25, 0, 20)
    hot = cfar_reference.lam_vector(geo, 0.25, 2, 20)
    assert clean[2] == 4096 * 0.25
    expected_change = np.zeros(clean.size)
    expected_change[geo["contam"][:2]] = 4096 * 100
    np.testing.assert_allclose(hot - clean, expected_change, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(hot[:geo["nD"]], clean[:geo["nD"]])


def test_cfar_crossing_uses_the_first_reached_probability():
    probabilities = np.asarray([0.1, 0.7, 0.2, 0.95])
    snr = np.asarray([-20.0, -10.0, 0.0, 10.0])
    assert cfar_reference.snr_at(probabilities, snr, 0.4) == pytest.approx(-15)
    assert cfar_reference.snr_at(probabilities, snr, 0.7) == -10
    assert cfar_reference.snr_at(probabilities, snr, 0.05) == -20
    assert np.isnan(cfar_reference.snr_at(probabilities, snr, 0.99))


def test_rank_null_maximum_matches_the_five_bin_distribution(monkeypatch):
    from scipy.stats import f

    probabilities = np.asarray([0.1, 0.5, 0.9, 0.999])
    quantiles = f.ppf(probabilities, 4096, 8192)
    np.testing.assert_allclose(rank_reference.dprod(quantiles, 0), probabilities ** 5,
                               rtol=2e-10, atol=1e-13)
    expected_eta = f.isf(-np.expm1(np.log1p(-0.001) / 5), 4096, 8192)
    reference = np.ones(3)
    assert rank_reference.eta_for(reference, exact=True) == pytest.approx(
        expected_eta, abs=5e-10)
    monkeypatch.setattr(rank_reference, "F5", rank_reference.dprod(rank_reference.XG, 0))
    assert rank_reference.eta_for(reference) == pytest.approx(expected_eta, abs=2e-7)


def test_rank_table_interpolates_reference_and_snr_coordinates(monkeypatch):
    x = 1 + np.arange(5) * 1e-4
    snr = np.asarray([-3.0, -2.0, -1.0])
    # A linear probability surface has an exact expectation at the mean x.
    table = 1 - (0.5 + 0.2 * (snr[:, None] + 3) + 100 * (x[None, :] - 1))
    monkeypatch.setattr(rank_reference, "XG", x)
    monkeypatch.setattr(rank_reference, "SG", snr)
    reference = np.asarray([1.00005, 1.00025])
    assert rank_reference.snr_table(1, reference, table) == pytest.approx(-1.075)
    assert rank_reference.snr_table(1, np.asarray([x[0]]), table) is None
    assert rank_reference.snr_table(1, np.asarray([x[-1]]), table) is None
    assert rank_reference.snr_table(1, reference, np.ones_like(table)) is None
    assert rank_reference.snr_table(1, reference, np.zeros_like(table)) is None


def test_rank_direct_solve_reproduces_the_recorded_detection_probability():
    reference, weights, _ = rank_quadrature.os_ref(125, 63)
    eta = 1.1006720998335313
    snr = -8.619030735310863
    # This result was recorded independently by the quadrature benchmark.
    assert 1 - weights @ rank_reference.dprod(reference * eta, 10 ** (snr / 10)) == (
        pytest.approx(0.9, abs=2e-10))
    unit_reference = np.ones(3)
    direct = rank_reference.snr_direct(eta, unit_reference)
    # The direct solver stops at 1e-5 dB; check that interval against the CDF.
    unit_weights = np.full(3, 1 / 3)
    assert rank_quadrature.pd(direct - 1e-5, eta, unit_reference, unit_weights) < 0.9
    assert rank_quadrature.pd(direct + 1e-5, eta, unit_reference, unit_weights) > 0.9


@pytest.mark.parametrize("reference", [
    [0.96, 0.99, 1.0, 1.04, 1.08],
    [98.0, 100.0, 101.0, 103.0, 107.0],
    [1.0, 1.01, 99.0, 101.0],
])
def test_rank_direct_solve_matches_the_empirical_cdf_for_dispersed_references(reference):
    from scipy.optimize import brentq

    reference = np.asarray(reference)
    weights = np.full(reference.size, 1 / reference.size)
    eta = 1.1
    # Evaluate the CDF at every empirical reference rather than interpolating
    # the solver's x grid; the mixed case spans clean and contaminated bins.
    expected = brentq(
        lambda snr: rank_quadrature.pd(snr, eta, reference, weights) - 0.9,
        -12, 40, xtol=1e-10,
    )
    actual = rank_reference.snr_direct(eta, reference)
    assert actual == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("parity, bulk", [("even", 125), ("odd", 126)])
def test_reference_geometry_keeps_pilot_and_bulk_separate(parity, bulk):
    geo = cfar_reference.geometry(parity)
    assert geo["nD"] == 5 and geo["nB"] == bulk
    assert not set(geo["D"]) & set(geo["B"])
    assert all(index >= geo["nD"] for index in geo["contam"])
    shoulders = geo["g"][geo["nD"]:]
    assert bool(np.any(shoulders > 0)) == (parity == "odd")


def test_comparison_order_statistics_are_one_based():
    powers = np.ones((1, 130))
    powers[0, 0] = 4
    powers[0, 5:] = np.arange(1, 126)
    scores = cfar_reference.scores(powers, 5, 125)
    assert scores["CA"][0] == 4 / 63
    for rank in (63, 94, 113):
        assert scores[f"OS{rank}"][0] == 4 / rank


@pytest.mark.parametrize("module", [rank_reference, rank_quadrature, cfar_reference])
def test_reference_cli_requires_an_explicit_destination(module):
    with pytest.raises(SystemExit) as exc:
        module.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("module", [rank_reference, rank_quadrature, cfar_reference])
def test_reference_cli_preserves_existing_results(module, tmp_path):
    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        module.main(["--out", str(tmp_path)])
    assert marker.read_text() == "keep"


def test_quadrature_reproduces_the_recorded_median_reference(tmp_path):
    out = tmp_path / "quadrature"
    result = rank_quadrature.run(out, ranks=[63])
    expected = {
        "125": (1.1006720998335313, -8.619030735310863, 0.011527771631191897),
        "126": (1.1009628149528683, -8.619278208272444, 0.011442670568658997),
    }
    for size, values in expected.items():
        row = result["sizes"][size]["63"]
        np.testing.assert_allclose(
            [row["eta"], row["snr_db"], row["loss_db"]], values,
            rtol=0, atol=2e-10)
    assert json.loads((out / "bench_quad.json").read_text()) == result
    assert "not an operational eta_Pfa" in (out / "REFERENCE.md").read_text()


@pytest.mark.parametrize("ranks", [[], [0], [127], [True], [63.0]])
def test_quadrature_refuses_invalid_ranks_before_creating_output(ranks, tmp_path):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="ranks"):
        rank_quadrature.run(out, ranks=ranks)
    assert not out.exists()
