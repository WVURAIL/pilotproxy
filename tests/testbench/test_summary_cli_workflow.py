"""Summary artifacts describe the current input and its finite populations."""

import csv
import json

import pytest

from pilot_proxy.testbench import summarize_results as summary


def _strict_json(path):
    def reject(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def _input(path, *, sweep=False):
    rows = [
        {"coarse_power_ratio": 1, "pilot_excess_db": -2,
         "estimated_data_shelf_snr_db": -30, "requested_data_shelf_snr_db": -30},
        {"coarse_power_ratio": 3, "pilot_excess_db": 2,
         "estimated_data_shelf_snr_db": -10,
         "requested_data_shelf_snr_db": -10 if sweep else -30},
        {"coarse_power_ratio": "nan", "pilot_excess_db": None,
         "estimated_data_shelf_snr_db": "inf"},
        {"coarse_power_ratio": "missing"},
    ]
    path.write_text(json.dumps({"detector_output": {"results": rows}}))
    return path


def test_summary_main_writes_known_csv_and_strict_json_for_a_sweep(tmp_path, capsys):
    source = _input(tmp_path / "input.json", sweep=True)
    output = tmp_path / "output"
    assert summary.main(["--input", str(source), "--output-dir", str(output),
                         "--bins", "3"]) == 0
    report = _strict_json(output / "summary.json")
    expected = [
        {"column": "coarse_power_ratio", "finite_count": 2,
         "mean": 2, "std": 1, "min": 1, "max": 3},
        {"column": "pilot_excess_db", "finite_count": 2,
         "mean": 0, "std": 2, "min": -2, "max": 2},
        {"column": "estimated_data_shelf_snr_db", "finite_count": 2,
         "mean": -20, "std": 10, "min": -30, "max": -10},
    ]
    assert report["columns"] == expected
    assert report["num_rows"] == 4
    assert report["histogram_mode"] == "auto"
    assert "multiple requested_data_shelf_snr_db" in report["histograms_skipped_reason"]
    assert report["power_ratio_histogram_png"] is None
    assert report["snr_shelf_histogram_png"] is None
    with (output / "summary.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    for actual, wanted in zip(records, expected, strict=True):
        assert actual["column"] == wanted["column"]
        assert {key: float(value) for key, value in actual.items() if key != "column"} == {
            key: value for key, value in wanted.items() if key != "column"
        }
    assert set(p.name for p in output.iterdir()) == {"summary.csv", "summary.json"}
    assert "Skipped histograms:" in capsys.readouterr().out


def test_histogram_rerun_removes_only_outputs_without_current_finite_values(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    source = _input(tmp_path / "input.json")
    output = tmp_path / "output"
    with mpl.rc_context():
        assert summary.main(["--input", str(source), "--output-dir", str(output),
                             "--bins", "2", "--histograms", "always"]) == 0
    power = output / "power_ratio_histogram.png"
    snr = output / "snr_shelf_histogram.png"
    assert power.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert snr.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    unrelated = output / "notes.txt"
    unrelated.write_text("retain this file")
    source.write_text(json.dumps({"results": [{"coarse_power_ratio": 2}]}))
    with mpl.rc_context():
        summary.summarize_result_json(input_json=source, output_dir=output, bins=2)
    assert power.is_file()
    assert not snr.exists()
    report = _strict_json(output / "summary.json")
    assert report["power_ratio_histogram_png"] == str(power)
    assert report["snr_shelf_histogram_png"] is None
    assert report["columns"][2] == {
        "column": "estimated_data_shelf_snr_db", "finite_count": 0,
        "mean": None, "std": None, "min": None, "max": None,
    }
    source.write_text(json.dumps({"results": [{"coarse_power_ratio": None}]}))
    summary.summarize_result_json(input_json=source, output_dir=output, bins=2)
    assert not power.exists()
    assert unrelated.read_text() == "retain this file"


@pytest.mark.parametrize("bins", [0, -2, 1.5])
@pytest.mark.parametrize("histograms", ["always", "never"])
def test_summary_api_rejects_invalid_bins_before_io(tmp_path, bins, histograms):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        summary.summarize_result_json(input_json=tmp_path / "missing.json",
                                      output_dir=output, bins=bins,
                                      histograms=histograms)
    assert not output.exists()


def test_summary_cli_rejects_zero_bins_before_io(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="--bins must be positive"):
        summary.main(["--input", str(tmp_path / "missing.json"),
                      "--output-dir", str(output), "--bins", "0"])
    assert not output.exists()
