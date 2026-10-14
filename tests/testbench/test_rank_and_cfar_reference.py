"""Ideal-reference geometry and independent quadrature checks."""
import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from pilot_proxy.testbench import cfar_reference, rank_quadrature, rank_reference


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
