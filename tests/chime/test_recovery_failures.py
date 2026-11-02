"""Recovery preserves finite statistics and prior reports on failed output."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.chime import injection_recovery as recovery
from pilot_proxy.chime import plots
from pilot_proxy.detector_contract import DETECTOR_CONTRACT_SCHEMA_TOKEN


def _point(path, amplitude=1.0):
    path.mkdir()
    identity = {
        "frame_index": np.arange(40), "physical_channel": np.array([14]),
        "pilot_frequency_hz": np.array([470309441.0]),
        "chime_frequency_hz": np.array([470312500.0]),
        "valid": np.ones((40, 1), dtype=np.uint8),
    }
    rho = np.tile([1.0, 3.0], 20)[:, None] + 3 * amplitude**2
    np.savez(path / recovery.CHIME_DETECTOR_OUTPUTS_FILENAME,
             normalized_pilot_excess=rho, coarse_power_ratio=rho + 1, **identity)
    np.savez(path / recovery.CHIME_SPECTROGRAM_CACHE_FILENAME,
             baseband_power_linear=rho + 10, **identity)
    (path / "run_config.json").write_text(json.dumps({
        "detector_contract": {"schema_version": DETECTOR_CONTRACT_SCHEMA_TOKEN}}))
    (path / recovery.INJECTION_MANIFEST_FILENAME).write_text(json.dumps({
        "schema_version": "pilotproxy_injection_manifest_v1",
        "files": [{"amplitude_lsb": amplitude, "baseband_frequency_hz": 3059,
                   "clip_count": 0}]}))
    return path


def _replace_array(path, key, value):
    with np.load(path) as product:
        arrays = dict(product)
    arrays[key] = value
    np.savez(path, **arrays)


@pytest.mark.parametrize("value", [-1, 2, 0.5, "yes"])
def test_nonbinary_validity_cannot_change_the_recovery_population(tmp_path, value):
    point = _point(tmp_path / "point")
    for name in (recovery.CHIME_DETECTOR_OUTPUTS_FILENAME,
                 recovery.CHIME_SPECTROGRAM_CACHE_FILENAME):
        _replace_array(point / name, "valid", np.full((40, 1), value))
    with pytest.raises(SystemExit, match="valid.*binary|binary.*valid"):
        recovery._load_point(point)


@pytest.mark.parametrize("product,key", [
    (recovery.CHIME_DETECTOR_OUTPUTS_FILENAME, "normalized_pilot_excess"),
    (recovery.CHIME_DETECTOR_OUTPUTS_FILENAME, "coarse_power_ratio"),
    (recovery.CHIME_SPECTROGRAM_CACHE_FILENAME, "baseband_power_linear"),
])
def test_complex_statistics_are_not_silently_cast_to_real(tmp_path, product, key):
    point = _point(tmp_path / "point")
    _replace_array(point / product, key, np.full((40, 1), 1 + 2j))
    with pytest.raises(SystemExit, match="real-valued"):
        recovery._load_point(point)


@pytest.mark.parametrize("amplitude", [1e200, 1e-200])
def test_finite_amplitude_with_unrepresentable_power_is_refused(tmp_path, amplitude):
    point = _point(tmp_path / "point")
    manifest_path = point / recovery.INJECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["amplitude_lsb"] = amplitude
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="injected.*power.*finite|amplitude.*squared"):
        recovery._load_point(point)


@pytest.mark.parametrize("count", [-1, 1.5, True, "2", float("nan")])
def test_clip_counts_are_nonnegative_integers(tmp_path, count):
    point = _point(tmp_path / "point")
    manifest_path = point / recovery.INJECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["clip_count"] = count
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="clip counts"):
        recovery._load_point(point)


@pytest.mark.parametrize("exponent", [-600, 600, 1023])
def test_finite_excess_mean_and_sem_do_not_square_or_sum_at_raw_scale(tmp_path, exponent):
    point = _point(tmp_path / "point")
    low = math.ldexp(0.5, exponent)
    high = math.ldexp(0.75, exponent)
    rho = np.tile([low, high], 20)[:, None]
    _replace_array(point / recovery.CHIME_DETECTOR_OUTPUTS_FILENAME,
                   "normalized_pilot_excess", rho)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = recovery._load_point(point)
    assert result["rho_hat"] == math.ldexp(0.625, exponent)
    # Twenty observations on either side of the midpoint give SEM=d/sqrt(39).
    expected_sem = math.ldexp(0.125, exponent) / math.sqrt(39)
    assert result["rho_sem"] == pytest.approx(expected_sem, rel=1e-14, abs=0)
    assert result["n_valid"] == 40


@pytest.mark.parametrize("manifest", [[], {}, {"files": None}, {"files": []},
                                      {"files": [None]}, {"files": [{}]}])
def test_malformed_manifest_has_a_named_input_error(tmp_path, manifest):
    point = _point(tmp_path / "point")
    (point / recovery.INJECTION_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="injection_manifest.json"):
        recovery._load_point(point)


@pytest.fixture
def report(tmp_path):
    return recovery.analyze_injection_recovery(
        [_point(tmp_path / f"a{index}", amplitude)
         for index, amplitude in enumerate((0.0, 1.0, 2.0))],
        false_alarm_rates=[0.5],
    )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failed_figure", [1, 2])
def test_failed_render_preserves_report_and_closes_only_owned_figures(
    tmp_path, monkeypatch, report, existing, failed_figure,
):
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv("PILOT_PROXY_FIGURE_FORMATS", "png")
    plt = plots._setup_matplotlib()
    original_figures = set(plt.get_fignums())
    caller = plt.figure()
    output = tmp_path / "report"
    names = (recovery.RECOVERY_CSV_FILENAME, recovery.RECOVERY_SUMMARY_FILENAME,
             recovery.RECOVERY_FIGURE, recovery.BASELINE_FIGURE, "notes.txt")
    expected = {name: f"prior {name}\n".encode() for name in names} if existing else {}
    if existing:
        output.mkdir()
        for name, content in expected.items():
            (output / name).write_bytes(content)
    calls = 0

    def fail_save(fig, path):
        nonlocal calls
        calls += 1
        path.write_bytes(b"partially rendered")
        if calls == failed_figure:
            raise OSError("injected encoder failure")

    monkeypatch.setattr(plots, "_save_figure", fail_save)
    try:
        with pytest.raises(OSError, match="encoder failure"):
            recovery.write_outputs(report, output)
        assert set(plt.get_fignums()) == original_figures | {caller.number}
        actual = {p.name: p.read_bytes() for p in output.iterdir()} if output.exists() else {}
        assert actual == expected
        assert not list(tmp_path.glob(".injection-recovery-*"))
    finally:
        for number in set(plt.get_fignums()) - original_figures:
            plt.close(number)


def test_nonfinite_report_is_refused_before_output_changes(tmp_path, report):
    output = tmp_path / "report"
    report["fit"]["gain_per_lsb2"] = float("nan")
    with pytest.raises(ValueError, match="JSON|range|finite"):
        recovery.write_outputs(report, output)
    assert not output.exists()


@pytest.mark.parametrize("formats", ["svg", "svg,svg"])
def test_report_returns_the_actual_requested_figure_formats(tmp_path, monkeypatch, report, formats):
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv("PILOT_PROXY_FIGURE_FORMATS", formats)
    output = tmp_path / "report"
    written = recovery.write_outputs(report, output)
    expected = {recovery.RECOVERY_CSV_FILENAME, recovery.RECOVERY_SUMMARY_FILENAME,
                "injection_recovery_linearity.svg", "detector_vs_radiometer_pd.svg"}
    assert {path.name for path in written} == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    assert {path.name for path in output.iterdir()} == expected
    parsed = json.loads((output / recovery.RECOVERY_SUMMARY_FILENAME).read_text())
    assert parsed == report
    assert all("<svg" in path.read_text() for path in written if path.suffix == ".svg")


def test_nearby_false_alarm_rates_keep_distinct_report_and_plot_keys(tmp_path, monkeypatch):
    points = [_point(tmp_path / f"a{index}", amplitude)
              for index, amplitude in enumerate((0.0, 1.0, 2.0))]
    report = recovery.analyze_injection_recovery(points, false_alarm_rates=[0.50000001, 0.50000002])
    assert report["false_alarm_rates"] == [0.50000002, 0.50000001]
    assert set(report["thresholds"]["power_ratio"]) == {"0.50000001", "0.50000002"}
    assert report["thresholds"]["power_ratio"]["0.50000001"] != report["thresholds"]["power_ratio"]["0.50000002"]
    for point in report["points"]:
        assert "pd_power_ratio_pfa0.50000001" in point
        assert "pd_power_ratio_pfa0.50000002" in point
    monkeypatch.setenv("PILOT_PROXY_USE_TEX", "0")
    monkeypatch.setenv("PILOT_PROXY_FIGURE_FORMATS", "svg")
    written = recovery.write_outputs(report, tmp_path / "report")
    assert all(path.exists() for path in written)


@pytest.mark.parametrize("failed_name", [recovery.RECOVERY_CSV_FILENAME,
                                        recovery.RECOVERY_FIGURE, recovery.BASELINE_FIGURE,
                                        recovery.RECOVERY_SUMMARY_FILENAME])
def test_publication_failure_leaves_no_success_summary(tmp_path, monkeypatch, report, failed_name):
    output = tmp_path / "report"
    output.mkdir()
    for name in (recovery.RECOVERY_CSV_FILENAME, recovery.RECOVERY_FIGURE,
                 recovery.BASELINE_FIGURE, recovery.RECOVERY_SUMMARY_FILENAME, "notes.txt"):
        (output / name).write_text(f"prior {name}")

    def rendered_figures(_report, directory):
        paths = [directory / recovery.RECOVERY_FIGURE, directory / recovery.BASELINE_FIGURE]
        for path in paths:
            path.write_bytes(b"new figure")
        return paths

    original_replace = Path.replace

    def fail_replace(source, destination):
        assert not (output / recovery.RECOVERY_SUMMARY_FILENAME).exists()
        if source.name == failed_name:
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(recovery, "_write_figures", rendered_figures)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        recovery.write_outputs(report, output)
    assert not (output / recovery.RECOVERY_SUMMARY_FILENAME).exists()
    assert (output / "notes.txt").read_text() == "prior notes.txt"
    assert not list(tmp_path.glob(".injection-recovery-*"))


def test_directory_destination_is_refused_before_invalidating_prior_summary(tmp_path, monkeypatch, report):
    output = tmp_path / "report"
    output.mkdir()
    (output / recovery.RECOVERY_FIGURE).mkdir()
    summary = output / recovery.RECOVERY_SUMMARY_FILENAME
    summary.write_text("previous completed report")

    def rendered_figures(_report, directory):
        path = directory / recovery.RECOVERY_FIGURE
        path.write_bytes(b"new figure")
        return [path]

    monkeypatch.setattr(recovery, "_write_figures", rendered_figures)
    with pytest.raises(IsADirectoryError, match="destination"):
        recovery.write_outputs(report, output)
    assert summary.read_text() == "previous completed report"
    assert not (output / recovery.RECOVERY_CSV_FILENAME).exists()
    assert not list(tmp_path.glob(".injection-recovery-*"))
