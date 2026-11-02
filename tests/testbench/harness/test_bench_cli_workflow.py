"""The bench CLI preserves release inputs and publishes calibrated artifacts."""

import csv
import hashlib
import json
import math

import matplotlib
import pytest

from pilot_proxy import cli
from pilot_proxy.testbench.harness import bench, evaluations
from test_results_estimator_transfer import _write_release
from test_results_evaluations import _write_evaluation


@pytest.mark.parametrize("calibration", [None, "11,3000,6000000", "11,3000,6000000,0.5"])
def test_bench_cli_renders_explicit_calibration_without_modifying_release(tmp_path, capsys, calibration):
    release = _write_release(tmp_path / "release", ota=calibration is None)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob("*") if p.is_file()}
    output = tmp_path / "preview.pdf"
    calibration_args = ["--calibration", calibration] if calibration is not None else []
    with matplotlib.rc_context():
        assert cli.main(["bench", "estimator-transfer", "--release", str(release),
                         "--out", str(output), *calibration_args,
                         "--title", "Controlled transfer", "--y-min", "-50", "--no-tex"]) == 0
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 5000
    assert str(output) in capsys.readouterr().out
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    # The explicit calibration determines the secondary-axis conversion.
    if calibration is not None:
        cal = bench._calibration(calibration)
        efficiency = 0.5 if calibration.endswith(",0.5") else 1.0
        assert cal.offset_db == pytest.approx(-11 + 10 * math.log10(2000 * efficiency))


@pytest.mark.parametrize("calibration", ["", "1,2", "1,2,3,4,5", "invalid,2,3", "nan,2,3", "1,0,3", "1,2,-3", "1,2,3,inf"])
def test_invalid_calibration_refuses_before_release_or_plot_setup(tmp_path, monkeypatch, calibration):
    output = tmp_path / "preview.pdf"

    def unexpected_style(**kwargs):
        raise AssertionError("invalid calibration reached plot setup")

    monkeypatch.setattr(bench.style, "configure", unexpected_style)
    with pytest.raises(SystemExit, match="--calibration"):
        bench.main(["estimator-transfer", "--release", str(tmp_path / "missing"),
                    "--out", str(output), "--calibration", calibration])
    assert not output.exists()


def test_transfer_points_cli_pools_all_declared_shards(tmp_path, capsys):
    sweep = tmp_path / "sweep"
    paths = evaluations.digital_sweep_layout(sweep)
    assert len(paths) == 40
    total_target = total_reference = 0
    for index, path in enumerate(paths):
        _write_evaluation(path, (-6.0,), trials=2, seed=index + 1)
        with (path / evaluations.TRIALS_NAME).open(newline="") as stream:
            for row in csv.DictReader(stream):
                total_target += int(row["p_target_u64"])
                total_reference += int(row["p_ref_sum_u64"])
    output = tmp_path / "points.csv"
    assert cli.main(["bench", "transfer-points", "--sweep", str(sweep), "--out", str(output),
                     "--bootstrap-samples", "32"]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["trials"] == row["positive_excess_trials"] == "80"
    assert float(row["positive_excess_fraction"]) == 1.0
    ratio = total_target * 12650 / (total_reference * 6338)
    report = json.loads((paths[0] / evaluations.REPORT_NAME).read_text())
    pilot_below = report["calibration"]["pilot_below_data_db_used"]
    expected_db = 10 * math.log10(ratio - 1) + pilot_below - 10 * math.log10(6000000 / 3051.7578125)
    assert float(row["gpu_fixed_db"]) == pytest.approx(expected_db)
    assert float(row["gpu_ci95_low_db"]) <= expected_db <= float(row["gpu_ci95_high_db"])
    assert row["waveform_conditioned_expected_db"] == ""
    assert str(output) in capsys.readouterr().out


def test_no_tex_preview_disables_tex_even_when_executables_are_available(tmp_path, monkeypatch):
    release = _write_release(tmp_path / "release", ota=True)
    output = tmp_path / "preview.pdf"
    monkeypatch.setattr(bench.style.shutil, "which", lambda name: f"/available/{name}")
    render = bench.figure_estimator_transfer
    rendering_modes = []

    def render_without_tex(*args, **kwargs):
        rendering_modes.append(matplotlib.rcParams["text.usetex"])
        assert rendering_modes == [False]
        return render(*args, **kwargs)

    monkeypatch.setattr(bench, "figure_estimator_transfer", render_without_tex)
    with matplotlib.rc_context():
        assert bench.main(["estimator-transfer", "--release", str(release),
                           "--out", str(output), "--no-tex"]) == 0
    assert rendering_modes == [False]
    assert output.read_bytes().startswith(b"%PDF-")
