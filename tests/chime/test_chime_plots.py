# coding=utf-8
from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("matplotlib")

# noinspection PyProtectedMember
from pilot_proxy.chime.plots import (
    _histogram_probability_density,
    _setup_matplotlib,
    _survival_probability_on_grid,
    clean_known_figures,
    generate_chime_plots,
)


def test_setup_matplotlib_keeps_tex_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setattr(
        "pilot_proxy.plot_style._command_available",
        lambda name: True,
    )

    plt = _setup_matplotlib()

    assert plt.rcParams["text.usetex"] is False


def test_histogram_probability_density_uses_finite_values_and_bin_width() -> None:
    bins = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([0.25, 0.75, 1.25, np.nan, 4.0])

    density = _histogram_probability_density(values, bins)

    assert density == pytest.approx([0.5, 0.25, 0.0])
    assert float(np.sum(density * np.diff(bins))) == pytest.approx(0.75)


def test_survival_probability_on_grid_uses_greater_equal_grid_values() -> None:
    values = np.asarray([1.0, 2.0, 3.0, np.nan])
    grid = np.asarray([0.5, 1.0, 2.5, 4.0])

    survival = _survival_probability_on_grid(values, grid)

    assert np.asarray(survival).tolist() == pytest.approx([1.0, 1.0, 1.0 / 3.0, 0.0])


def test_chime_plots_write_expected_files_and_summary_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    physical_channel = np.asarray([14, 30], dtype=np.int32)
    pilot_frequency_hz = np.asarray([470_309_441.0, 566_309_441.0])
    chime_frequency_hz = np.asarray([470_214_843.75, 566_406_250.0])
    frame_index = np.arange(4, dtype=np.int64)
    snr = np.asarray(
        [
            [-30.0, -31.0],
            [-29.0, -32.0],
            [-28.0, -33.0],
            [-27.0, -34.0],
        ]
    )
    mask = np.asarray([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    np.savez_compressed(
        run_dir / "chime_detector_outputs.npz",
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        p_target_u64=np.ones((4, 2), dtype=np.uint64),
        p_ref_sum_u64=np.ones((4, 2), dtype=np.uint64),
        coarse_power_ratio=np.ones((4, 2)),
        normalized_coarse_power_ratio_db=np.zeros((4, 2)),
        pilot_excess_db=np.zeros((4, 2)),
        estimated_data_shelf_snr_db=snr,
        mask=mask,
        valid=np.ones((4, 2), dtype=np.uint8),
    )
    np.savez_compressed(
        run_dir / "chime_spectrogram_cache.npz",
        baseband_power_linear=np.asarray(
            [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
        ),
        baseband_power_db=np.asarray(
            [[10.0, 13.0], [10.4, 13.2], [10.8, 13.4], [11.1, 13.6]]
        ),
        mask=mask,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        relative_time_s=np.asarray([0.0, 1.0, 2.0, 3.0]),
    )

    outputs = generate_chime_plots(run_dir)

    expected = {
        run_dir / "figures" / "data_shelf_snr_histogram_by_channel.png",
        run_dir / "figures" / "coarse_power_ratio_survival_by_channel.png",
        run_dir / "figures" / "normalized_coarse_power_ratio_db_spectrogram.png",
        run_dir / "figures" / "baseband_spectrum_before_after_mask.png",
        run_dir / "figures" / "baseband_spectrogram.png",
        run_dir / "figures" / "mask_spectrogram.png",
    }
    assert expected.issubset(set(outputs))
    for path in expected:
        assert path.exists()

    with (run_dir / "tables" / "data_shelf_snr_histogram_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))
    assert [int(row["physical_channel"]) for row in rows] == [14, 30]
    assert rows[0]["num_detector_valid_frames"] == "4"
    assert rows[0]["num_finite_data_shelf_snr_estimates"] == "4"
    assert "finite_data_shelf_snr_fraction_of_valid_frames" in rows[0]
    assert "mask_fraction_total" in rows[0]
    assert "num_positive_excess_frames" not in rows[0]
    assert "positive_excess_fraction" not in rows[0]
    assert "mean_estimated_data_shelf_snr_db" in rows[0]
    assert "max_estimated_data_shelf_snr_db" in rows[0]
    assert "p95_estimated_data_shelf_snr_db" not in rows[0]
    assert "chime_frequency_hz" in rows[0]
    assert float(rows[0]["chime_frequency_hz"]) == pytest.approx(
        float(chime_frequency_hz[0])
    )

    with (run_dir / "tables" / "normalized_coarse_power_ratio_summary_by_channel.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        fstat_rows = list(csv.DictReader(f))
    assert [int(row["physical_channel"]) for row in fstat_rows] == [14, 30]
    assert fstat_rows[0]["num_detector_valid_frames"] == "4"
    assert "mean_normalized_coarse_power_ratio_db" in fstat_rows[0]
    assert "max_normalized_coarse_power_ratio_db" in fstat_rows[0]
    assert "p99_normalized_coarse_power_ratio_db" not in fstat_rows[0]

    with (run_dir / "tables" / "spectrum_before_after.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        spectrum_rows = list(csv.DictReader(f))
    assert "chime_frequency_hz" in spectrum_rows[0]
    assert float(spectrum_rows[0]["chime_frequency_hz"]) == pytest.approx(
        float(chime_frequency_hz[0])
    )


def test_clean_known_figures_removes_current_figure_names_only(tmp_path) -> None:
    run_dir = tmp_path / "run"
    figures = run_dir / "figures"
    figures.mkdir(parents=True)
    current = figures / "coarse_power_ratio_survival_by_channel.png"
    unknown = figures / "not_a_known_chime_figure.png"
    for path in (current, unknown):
        path.write_text("existing", encoding="utf-8")

    clean_known_figures(run_dir)

    assert not current.exists()
    assert unknown.exists()


def test_figure_formats_env_writes_vector_variants(tmp_path, monkeypatch) -> None:
    from pilot_proxy.chime import plots as chime_plots

    monkeypatch.setenv(chime_plots.FIGURE_FORMATS_ENV, "png,pdf")
    assert chime_plots._figure_formats() == ("png", "pdf")
    plt = chime_plots._setup_matplotlib()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label=r"$F/\mu_0 - 1$")
    ax.legend()
    target = tmp_path / "figures" / "coarse_power_ratio_survival_by_channel.png"
    target.parent.mkdir(parents=True)
    chime_plots._save_figure(fig, target)
    plt.close(fig)
    assert target.exists()
    assert target.with_suffix(".pdf").exists()

    # clean_known_figures removes every format variant of known names
    run_dir = tmp_path
    chime_plots.clean_known_figures(run_dir)
    assert not target.exists()
    assert not target.with_suffix(".pdf").exists()


def test_figure_formats_default_is_png_only(monkeypatch) -> None:
    from pilot_proxy.chime import plots as chime_plots

    monkeypatch.delenv(chime_plots.FIGURE_FORMATS_ENV, raising=False)
    assert chime_plots._figure_formats() == ("png",)
