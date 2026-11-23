"""The public CHIME plot command publishes checked outputs."""

import csv

import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy import cli
from pilot_proxy.chime import plots
from test_product_workflows import _write_run


def test_public_plot_cli_renders_current_artifacts_and_cleans_owned_figures(tmp_path, monkeypatch, capfd):
    pytest.importorskip("matplotlib")
    run = tmp_path / "run with spaces"
    _write_run(run)
    figures = run / "figures"
    figures.mkdir()
    stale = figures / "mask_spectrogram.svg"
    stale.write_text("old render")
    unrelated = figures / "notes.txt"
    unrelated.write_text("preserve")
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv(plots.FIGURE_FORMATS_ENV, "png")
    assert cli.main(["chime-plot", "--run-dir", str(run), "--clean-figures"]) == 0
    assert not stale.exists() and unrelated.read_text() == "preserve"
    assert {path.name for path in figures.glob("*.png")} == plots.KNOWN_CHIME_FIGURES
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures.glob("*.png"))
    with (run / "tables/data_shelf_snr_histogram_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["physical_channel"]) for row in rows] == [30, 14]
    assert [int(row["num_finite_data_shelf_snr_estimates"]) for row in rows] == [2, 1]
    assert [float(row["mean_estimated_data_shelf_snr_db"]) for row in rows] == [-20, -20]
    assert capfd.readouterr().out.count("Wrote ") == 6
