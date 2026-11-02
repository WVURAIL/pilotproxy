"""Numerical checks for the archive publisher's diagnostic figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from pilot_proxy import archive_health
from test_archive_health import FINE_BINS, K, NFFT, SAMPLE_RATE_HZ, _current_form, _product, _save_product


def _utc(label):
    return float(np.datetime64(label, "s").astype(np.int64))


def _plot_product(*, all_rejected=False):
    pattern = 1 + np.arange(FINE_BINS) / FINE_BINS
    fine = np.asarray([9, 1, 1, 9, 1_000_000, 25])[:, None] * pattern
    product = _product(
        valid=[1, 1, 1, 1, 0, 1],
        reject=[1] * 6 if all_rejected else [0, 0, 1, 1, 0, 0],
        baseband_power=[4, 4, 4, 4, 0, 4],
        p_target=[10, 10, 10, 10, 0, 10],
        p_reference=[20, 20, 20, 20, 0, 20],
        fine=fine,
    )
    product["unit_time0_ctime"] = np.asarray([
        _utc("2026-03-01"), _utc("2025-12-31T23:59:59"),
        _utc("2026-01-01"), _utc("2026-01-31T23:59:59"),
        _utc("2026-04-01"), np.nan,
    ])
    # Both sums satisfy Parseval, while their distinct shapes and frame
    # counts make an omitted time-average normalization observable.
    before_shape = 1 + np.arange(NFFT) % 4
    after_shape = 1 + (np.arange(NFFT) + 1) % 4
    per_frame_total = NFFT**2 * 2 * 4
    product["integrated_spectrum_before_mask"] = (
        before_shape * (5 * per_frame_total / before_shape.sum()))
    product["integrated_spectrum_after_mask"] = (
        after_shape * ((0 if all_rejected else 3) * per_frame_total / after_shape.sum()))
    return product


@pytest.fixture
def saved_figures(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    saved = {}
    original = Figure.savefig

    def save(figure, path, *args, **kwargs):
        saved[Path(path).name] = (figure, kwargs.copy())
        return original(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    with matplotlib.rc_context():
        yield saved


def test_month_codes_keep_calendar_gaps_and_missing_times_without_warnings():
    times = np.asarray([
        _utc("2026-03-01"), np.nan, _utc("2025-12-31T23:59:59"),
        _utc("2026-01-01"), np.inf, -np.inf,
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        codes, months, labels = archive_health._month_codes(times)
    assert labels == ["2025-12", "2026-01", "2026-02", "2026-03"]
    np.testing.assert_array_equal(codes, [3, -1, 0, 1, -1, -1])
    np.testing.assert_array_equal(months.astype(str), labels)


@pytest.mark.parametrize("times", [[], [np.nan, np.inf, -np.inf]])
def test_month_codes_refuse_an_empty_time_axis_without_warnings(times):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(archive_health.ArchiveHealthError, match="no finite UTC times"):
            archive_health._month_codes(np.asarray(times))


def test_default_archive_publication_preserves_numerical_plot_data(tmp_path, saved_figures):
    product = _plot_product()
    product_path = _save_product(tmp_path / "798.npz", product)
    out = tmp_path / "audit"
    result = archive_health.write_archive_audit([product_path], out, fine_chunk_rows=2)
    manifest = json.loads(result["manifest"].read_text())
    assert len(manifest["figures"]) == 4
    assert manifest["figure_style"]["pdf_date_metadata"] == "suppressed"
    for row in manifest["figures"]:
        path = out / row["path"]
        assert not Path(row["path"]).is_absolute()
        assert row["relative_path"] == row["path"]
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.read_bytes().startswith(b"%PDF" if row["format"] == "PDF" else b"\x89PNG")
    for name, (_, settings) in saved_figures.items():
        if name.endswith(".pdf"):
            assert settings["metadata"]["CreationDate"] is None
            assert settings["metadata"]["ModDate"] is None
    spectrum = next(row for row in manifest["figures"] if row["kind"] == "relative_time_averaged_health_corrected_spectra")
    assert spectrum["healthy_before_frame_count"] == 5
    assert spectrum["healthy_after_frame_count"] == 3
    assert spectrum["absolute_psd"] is False
    heatmap = next(row for row in manifest["figures"] if row["kind"] == "fine_f_utc_monthly_mean_heatmap")
    assert heatmap["months_with_frames"] == 3
    assert heatmap["time_bin_reduction"] == "arithmetic_mean_of_health_included_frames"

    histograms, _ = saved_figures["health_filtered_histograms.pdf"]
    assert all(sum(patch.get_height() for patch in axis.patches) == 5 for axis in histograms.axes)
    # Preserve Matplotlib's existing half-unit range for exactly constant
    # data; only ranges too narrow for distinct bins need the new fallback.
    assert histograms.axes[0].patches[0].get_x() == pytest.approx(0.5)
    last_bin = histograms.axes[0].patches[-1]
    assert last_bin.get_x() + last_bin.get_width() == pytest.approx(1.5)
    figure, _ = saved_figures["relative_time_averaged_spectra.pdf"]
    lines = {line.get_label(): line for line in figure.axes[0].lines}
    before = lines["before stored mask, health corrected"]
    after = lines["after stored mask, health corrected"]
    raw_order = np.r_[np.arange(NFFT // 2 - 1, -1, -1), np.arange(NFFT - 1, NFFT // 2 - 1, -1)]
    offset = float(product["chime_frequency_hz"][0] - product["pilot_frequency_hz"][0])
    expected_hz = offset + (np.arange(NFFT) - (NFFT // 2 - 1)) * SAMPLE_RATE_HZ / NFFT
    np.testing.assert_allclose(before.get_xdata(), expected_hz / 1e3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(after.get_xdata(), before.get_xdata(), rtol=0, atol=0)
    np.testing.assert_allclose(before.get_ydata(), 10 * np.log10((1 + raw_order % 4) / 2.5))
    np.testing.assert_allclose(after.get_ydata(), 10 * np.log10((1 + (raw_order + 1) % 4) / 2.5))

    figure, _ = saved_figures["fine_f_utc_monthly_heatmap.png"]
    axis = figure.axes[0]
    image = axis.images[0]
    fine_order = np.r_[np.arange(FINE_BINS // 2, FINE_BINS), np.arange(FINE_BINS // 2)]
    expected = (1 + fine_order / FINE_BINS)[:, None] * np.asarray([1, 5, np.nan, 9, np.nan])
    np.testing.assert_allclose(np.asarray(image.get_array().filled(np.nan)), 10 * np.log10(expected), equal_nan=True)
    # The calendar spans the whole archive, including the unhealthy April
    # frame. Neither that frame nor the untimed healthy frame enters a mean.
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
    # imshow's extent is a set of edges. Its row centers must coincide with
    # the physical fine-bin frequencies used by the anchor overlays.
    bottom, top = image.get_extent()[2:]
    centers = bottom + (np.arange(FINE_BINS) + 0.5) * (top - bottom) / FINE_BINS
    expected_frequency = np.arange(-FINE_BINS // 2, FINE_BINS // 2) * SAMPLE_RATE_HZ / K / FINE_BINS
    np.testing.assert_allclose(centers, expected_frequency, rtol=0, atol=1e-10)
    atlas, _ = saved_figures["channel_17_diagnostic_atlas.pdf"]
    atlas_image = next(axis.images[0] for axis in atlas.axes if axis.images)
    np.testing.assert_array_equal(atlas_image.get_array(), image.get_array())
    np.testing.assert_array_equal(atlas_image.get_extent(), image.get_extent())


def test_all_rejected_frames_omit_the_after_spectrum(tmp_path, saved_figures):
    product = _plot_product(all_rejected=True)
    path = _save_product(tmp_path / "798.npz", product)
    spectra = archive_health.health_correct_integrated_spectra(product)
    outputs = archive_health._plot_product_diagnostics(
        path, spectra, tmp_path / "figures", fine_chunk_rows=2, dissertation_style=False)
    figure, _ = saved_figures["relative_time_averaged_spectra.pdf"]
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert "before stored mask, health corrected" in labels
    assert "after stored mask, health corrected" not in labels
    assert any("No health-included frame survived" in text.get_text() for text in figure.axes[0].texts)
    row = next(row for row in outputs if row["kind"] == "relative_time_averaged_health_corrected_spectra")
    assert row["healthy_before_frame_count"] == 5
    assert row["healthy_after_frame_count"] == 0


@pytest.mark.parametrize("unhealthy_finite_time", [False, True])
def test_missing_healthy_times_fail_before_creating_figure_outputs(
    tmp_path, unhealthy_finite_time, saved_figures
):
    product = _plot_product()
    product["unit_time0_ctime"][:] = np.nan
    if unhealthy_finite_time:
        product["unit_time0_ctime"][4] = _utc("2026-01-15")
    path = _save_product(tmp_path / "798.npz", product)
    spectra = archive_health.health_correct_integrated_spectra(product)
    out = tmp_path / "figures"
    with pytest.raises(archive_health.ArchiveHealthError, match="no health-included frames have finite UTC times"):
        archive_health._plot_product_diagnostics(
            path, spectra, out, fine_chunk_rows=2, dissertation_style=False)
    assert not out.exists()
    assert not saved_figures


@pytest.mark.parametrize("center", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bins", [64, 80])
def test_nearly_constant_histograms_keep_every_finite_sample(center, bins):
    values = np.asarray([np.nextafter(center, -np.inf), center, np.nextafter(center, np.inf)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = archive_health.distribution_summary(values, histogram_bins=bins)
    edges = np.asarray(result["histogram"]["edges"])
    assert edges.size == bins + 1
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)
    assert edges[0] <= values.min() <= values.max() <= edges[-1]
    assert sum(result["histogram"]["counts"]) == 3
    assert result["count_finite"] == 3
    assert result["mean"] == pytest.approx(center, abs=1e-15)


def test_ordinary_histogram_range_and_counts_are_unchanged():
    result = archive_health.distribution_summary(np.asarray([0.0, 0.5, 1.0]), histogram_bins=2)
    assert result["histogram"] == {"edges": [0.0, 0.5, 1.0], "counts": [1, 2]}


def test_plot_title_reports_the_mask_used_for_stored_spectra(tmp_path, saved_figures):
    product = _product(
        valid=[1, 1, 1], reject=[0, 1, 0], baseband_power=[4, 4, 4],
        p_target=[10, 30, 10], p_reference=[20, 20, 20])
    # The fine-sum decision rejects every frame; the stored row-domain mask
    # rejected only one. The displayed stored-mask rate must describe the
    # same selection as the before/after spectrum.
    product = _current_form(product, np.asarray([[1200, 1000, 1000]] * 3))
    for key, count in [("integrated_spectrum_before_mask", 3), ("integrated_spectrum_after_mask", 2)]:
        product[key] = np.full(NFFT, NFFT * 2 * 4 * count, dtype=np.float64)
    path = _save_product(tmp_path / "798.npz", product)
    spectra = archive_health.health_correct_integrated_spectra(product)
    assert spectra.accumulation_reject_disagreements == 2
    archive_health._plot_product_diagnostics(
        path, spectra, tmp_path / "figures", fine_chunk_rows=2, dissertation_style=False)
    figure, _ = saved_figures["health_filtered_histograms.pdf"]
    assert "stored coarse mask rate 33.33%" in figure._suptitle.get_text()
