"""Check product reports and plot summaries against explicit frame populations."""

import csv
import json

import numpy as np
import pytest

from pilot_proxy.chime import plots, validate_products
from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
    WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
    build_detector_contract,
    normalized_positive_excess_policy,
)


def _write_run(run):
    run.mkdir()
    channels = np.asarray([30, 14], dtype=np.int32)
    pilots = np.asarray([566309441.0, 470309441.0])
    centers = np.asarray([566406250.0, 470312500.0])
    valid = np.asarray([[1, 1], [1, 0], [0, 1], [1, 0]], dtype=np.uint8)
    mask = np.asarray([[0, 1], [1, 0], [0, 0], [0, 0]], dtype=np.uint8)
    target = np.asarray([[4, 7], [8, 0], [0, 3], [5, 0]], dtype=np.uint64)
    ratio = np.asarray([[0.8, 1.4], [1.6, np.nan], [np.nan, 0.6], [1, np.nan]])
    common = dict(physical_channel=channels, pilot_frequency_hz=pilots,
                  chime_frequency_hz=centers, frame_index=np.arange(4),
                  valid=valid, mask=mask)
    np.savez(
        run / "chime_detector_outputs.npz", **common,
        p_target_u64=target, p_ref_lower_u64=valid.astype(np.uint64) * 6,
        p_ref_upper_u64=valid.astype(np.uint64) * 4,
        p_ref_sum_u64=valid.astype(np.uint64) * 10,
        target_norm_sq=np.asarray([5, 5]), reference_norm_sum_sq=np.asarray([10, 10]),
        coarse_power_ratio=ratio, normalized_coarse_power_ratio_db=10 * np.log10(ratio),
        pilot_excess_db=np.zeros((4, 2)),
        estimated_data_shelf_snr_db=np.asarray(
            [[-30, -20], [-10, 80], [90, np.nan], [np.nan, 100]]),
    )
    power = np.asarray([[2, 10], [4, 20], [6, 30], [8, 40]])
    np.savez(run / "chime_spectrogram_cache.npz", **common,
             relative_time_s=np.arange(4), baseband_power_linear=power,
             baseband_power_db=10 * np.log10(power))
    np.savez(
        run / "chime_reductions_10s.npz", chunk_index=np.asarray([0]),
        chunk_start_frame=np.asarray([0]), chunk_stop_frame=np.asarray([4]),
        input_power_mean=np.asarray([[14 / 3, 20]]),
        cleaned_power_mean=np.asarray([[5, 30]]),
        valid_count=np.asarray([[3, 2]]), invalid_count=np.asarray([[1, 2]]),
        masked_count_valid=np.asarray([[1, 1]]),
        unmasked_count_valid=np.asarray([[2, 1]]),
        mask_fraction_valid=np.asarray([[1 / 3, 1 / 2]]),
        mask_fraction_total=np.asarray([[1 / 4, 1 / 4]]),
    )
    contract = build_detector_contract(
        detector_window_samples=128, skipped_guard_bins=1, reference_offset_bins=2,
        num_weight_terms=3, weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        time_reverse_detector_windows_before_kernel=True,
    )
    for filename, token in [("run_config.json", CHIME_RUN_CONFIG_SCHEMA_TOKEN),
                            ("stats.json", CHIME_STATS_SCHEMA_TOKEN)]:
        (run / filename).write_text(json.dumps(dict(
            schema_version=token, detector_contract=contract, detector_window_samples=128,
            mask_policy=normalized_positive_excess_policy(), num_frames=4, num_pilots=2,
        )))
    (run / "input_manifest.json").write_text(json.dumps(dict(
        schema_version="pilotproxy_scan_input_manifest_v1", source="chime-scan",
        physical_channels=[30, 14], input_files=["cadc:fixture"],
    )))


def _replace_array(path, name, values):
    with np.load(path) as product:
        arrays = {key: product[key] for key in product.files if key != name}
    if values is not None:
        arrays[name] = np.asarray(values)
    np.savez(path, **arrays)


def test_validation_cli_publishes_a_consistent_product_report(tmp_path, capsys):
    run = tmp_path / "run"
    _write_run(run)
    report_path = tmp_path / "reports" / "validation.json"
    assert validate_products.main([
        "--run-dir", str(run), "--output-json", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert report["valid"] is True and report["errors"] == []
    assert report["num_errors"] == 0 and report["detector_contract_revision"] == 2
    assert f"True, 0, {run}" in capsys.readouterr().out


@pytest.mark.parametrize("product,name,values,check", [
    ("detector_outputs", "mask", np.zeros((3, 2)), "detector.mask.shape"),
    ("detector_outputs", "valid", np.zeros((3, 2)), "detector.valid.shape"),
    ("detector_outputs", "p_ref_sum_u64", np.zeros((4, 3)), "detector.p_ref_sum_u64.shape"),
    ("detector_outputs", "physical_channel", [[30, 14]], "detector.physical_channel.shape"),
    ("detector_outputs", "frame_index", [[0, 1, 2, 3]], "detector.frame_index.shape"),
    ("detector_outputs", "target_norm_sq", [5], "detector.target_norm_sq.shape"),
    ("detector_outputs", "reference_norm_sum_sq", [[10], [10]],
     "detector.reference_norm_sum_sq.shape"),
    ("reductions_10s", "valid_count", [[3, 2, 1]], "reductions.valid_count.shape"),
    ("reductions_10s", "masked_count_valid", 1, "reductions.masked_count_valid.shape"),
    ("reductions_10s", "chunk_index", 0, "reductions.chunk_index.shape"),
    ("reductions_10s", "chunk_index", [[0]], "reductions.chunk_index.shape"),
])
def test_validation_cli_reports_bad_shapes_instead_of_broadcasting(
    tmp_path, capsys, product, name, values, check,
):
    run = tmp_path / "run"
    _write_run(run)
    _replace_array(run / f"chime_{product}.npz", name, values)
    report_path = tmp_path / "validation.json"
    assert validate_products.main([
        "--run-dir", str(run), "--output-json", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    assert report["valid"] is False
    assert report["num_errors"] == len(report["errors"]) > 0
    assert check in {error["check"] for error in report["errors"]}
    assert f"ERROR {check}:" in capsys.readouterr().out


def test_plot_cli_uses_manifest_order_and_the_plotted_valid_population(
    tmp_path, monkeypatch, capsys,
):
    run = tmp_path / "run"
    _write_run(run)
    for filename in ("chime_detector_outputs.npz", "chime_spectrogram_cache.npz"):
        _replace_array(run / filename, "chime_frequency_hz", None)
    (run / "input_manifest.json").write_text(json.dumps({"datasets": [
        {"physical_channel": 14, "coarse_channel_center_hz": 470312500.0},
        {"physical_channel": 30, "coarse_channel_center_hz": 566406250.0},
    ]}))
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv(plots.FIGURE_FORMATS_ENV, "png")
    histogram = []
    save = plots._save_figure

    def capture(fig, path):
        if path.name == "data_shelf_snr_histogram_by_channel.png":
            histogram.extend(patch.get_data() for patch in fig.axes[0].patches)
        save(fig, path)

    monkeypatch.setattr(plots, "_save_figure", capture)
    assert plots.main(["--run-dir", str(run), "--clean-figures"]) == 0
    assert len(list((run / "figures").glob("*.png"))) == 6
    assert len(histogram) == 3
    for row, samples in zip(histogram[:2], [[-30, -10], [-20]], strict=True):
        counts = np.asarray([
            sum(left <= value < right for value in samples)
            for left, right in zip(row.edges[:-1], row.edges[1:], strict=True)
        ])
        np.testing.assert_allclose(row.values, counts / len(samples) / np.diff(row.edges))
    with (run / "tables" / "data_shelf_snr_histogram_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["physical_channel"]) for row in rows] == [30, 14]
    assert [float(row["chime_frequency_hz"]) for row in rows] == [566406250, 470312500]
    assert [int(row["num_detector_valid_frames"]) for row in rows] == [3, 2]
    assert [int(row["num_finite_data_shelf_snr_estimates"]) for row in rows] == [2, 1]
    assert [float(row["finite_data_shelf_snr_fraction_of_valid_frames"]) for row in rows] == pytest.approx([2 / 3, 1 / 2])
    assert [float(row["mean_estimated_data_shelf_snr_db"]) for row in rows] == [-20, -20]
    assert [float(row["max_estimated_data_shelf_snr_db"]) for row in rows] == [-10, -20]
    assert [float(row["mask_fraction_total"]) for row in rows] == [0.25, 0.25]
    assert "Wrote" in capsys.readouterr().out


@pytest.mark.parametrize("manifest", [None, {"datasets": [
    {"physical_channel": 30, "coarse_channel_center_hz": 566406250},
]}])
def test_plot_frequency_requires_complete_receiver_centers(tmp_path, manifest):
    run = tmp_path / "run"
    _write_run(run)
    _replace_array(run / "chime_detector_outputs.npz", "chime_frequency_hz", None)
    if manifest is None:
        (run / "input_manifest.json").unlink()
    else:
        (run / "input_manifest.json").write_text(json.dumps(manifest))
    with np.load(run / "chime_detector_outputs.npz") as product:
        with pytest.raises(KeyError, match="pilot frequency is not a receiver-frequency substitute"):
            plots._run_frequency_hz(run, product)
