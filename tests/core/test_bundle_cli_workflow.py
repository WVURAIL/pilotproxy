"""The bundle commands of the CLI on synthetic captures and runs: calibrate, test and qualify a bundle, plan and
process a run, and the spectrograms and score cache of a run."""
import contextlib
import csv
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy import cli, run_scores
from pilot_proxy.bundle_calibration import bundle_digest
from pilot_proxy.characterization import threshold_rule
from pilot_proxy.run_plans import CORE_PROFILE
from pilot_proxy.run_spectrogram import FRAME_SECONDS
from pilot_proxy.runtime_bundle import validate_runtime_weight_bundle
from test_bundle_calibration import ROOT, write_capture

RECEIVER = ROOT / "configs/receiver_profiles/chime_dtv_fengine.json"
CORE = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"
DRAO = ["49.3208", "-119.6236"]


def run_cli(argv):
    """Return the exit code and the standard output of one command."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            code = cli.main([str(a) for a in argv])
        except SystemExit as stop:
            code = stop.code
    return code, out.getvalue()


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    """A knee bundle of channels 35 and 29 with the control, made by calibrate-bundle, and its test plan."""
    tmp = tmp_path_factory.mktemp("cli")
    # strong and clean captures in turn, so the kept fraction has a knee
    mixed = [write_capture(tmp / f"c35_{i}.h5", 35, 2, 10 + i, tone=0.8 * (i % 2)) for i in range(6)]
    control = [write_capture(tmp / "c37.h5", 37, 26, 3)]
    calibrate = dict(receiver_profile=str(RECEIVER), detector_core_profile=str(CORE), capture_format="chime_baseband",
                     epoch="synthetic", null_quantile=0.8, min_knee_captures=4,
                     channels={"35": mixed, "29": [write_capture(tmp / "c29.h5", 29, 6, 2)], "37": control},
                     control=dict(channel=37, captures=control))
    test = dict(capture_format="chime_baseband",
                channels={"35": [write_capture(tmp / f"t35_{i}.h5", 35, 2, 20 + i, tone=0.8 * (i % 2)) for i in range(2)],
                          "29": [write_capture(tmp / "t29.h5", 29, 4, 5)]},
                control=[write_capture(tmp / "t37.h5", 37, 12, 6)])
    (tmp / "calibrate.json").write_text(json.dumps(calibrate))
    (tmp / "test.json").write_text(json.dumps(test))
    code, out = run_cli(["calibrate-bundle", "--plan", tmp / "calibrate.json", "--output-dir", tmp / "bundle"])
    return dict(tmp=tmp, bundle=tmp / "bundle", code=code, out=out, test=test, test_path=tmp / "test.json")


def test_calibrate_bundle_prints_the_word_of_each_channel(calibrated):
    assert calibrated["code"] == 0
    bundle = calibrated["bundle"]
    assert not validate_runtime_weight_bundle(bundle_dir=bundle).get("errors")
    report = json.loads(bundle.with_name("bundle.report.json").read_text())
    assert report["channels"]["35"]["word_source"] == "knee" and report["channels"]["35"]["anchor_source"] == "measured"
    assert report["channels"]["29"]["word_source"] == "floor, no pilot"
    lines = calibrated["out"].splitlines()
    assert lines[:2] == ["threshold knee",
                         f"floor {report['floor_word_q16']} (eta {report['floor_eta']:.4f}) from control channel"]
    for channel in ("29", "35"):
        row = report["channels"][channel]
        assert (f"channel {channel}: anchor {row['anchor_bin']} ({row['anchor_source']}), word {row['word_q16']} "
                f"(eta {row['eta']:.4f}) from {row['word_source']}") in lines
    profiles = json.loads((bundle / "pilot_profiles.json").read_text())["profiles"]
    words = {p["physical_channel"]: p["fine_calibration"]["cfar_multiplier_q16"] for p in profiles}
    assert words == {29: report["floor_word_q16"], 35: report["channels"]["35"]["word_q16"], 37: report["floor_word_q16"]}
    # a bundle is never written over
    with pytest.raises(ValueError, match="never changed"):
        run_cli(["calibrate-bundle", "--plan", calibrated["tmp"] / "calibrate.json", "--output-dir", bundle])


def test_test_bundle_prints_each_check_and_writes_the_report(calibrated, tmp_path):
    report = tmp_path / "report.json"
    code, out = run_cli(["test-bundle", "--bundle-dir", calibrated["bundle"], "--plan", calibrated["test_path"],
                         "--output-json", report])
    assert code == 0
    written = json.loads(report.read_text())
    assert written["passed"] and written["bundle_sha256sums"] == bundle_digest(calibrated["bundle"])
    assert out.splitlines() == [f"PASS  {c['check']}: {c['detail']}" for c in written["checks"]]
    assert "PASS  structure and digests: 0 errors" in out
    assert any(line.startswith("PASS  masked fraction on channel 35: ") for line in out.splitlines())


def test_test_bundle_exits_1_on_a_failed_check(calibrated, tmp_path):
    path = tmp_path / "no_control.json"
    path.write_text(json.dumps(dict(calibrated["test"], control=[])))
    code, out = run_cli(["test-bundle", "--bundle-dir", calibrated["bundle"], "--plan", path,
                         "--output-json", tmp_path / "report.json"])
    assert code == 1
    assert "FAIL  false alarms on the control channel: the test plan has no control" in out
    assert not json.loads((tmp_path / "report.json").read_text())["passed"]


def test_qualify_bundle_prints_its_summary(calibrated, tmp_path):
    report = tmp_path / "report.json"
    assert run_cli(["test-bundle", "--bundle-dir", calibrated["bundle"], "--plan", calibrated["test_path"],
                    "--output-json", report])[0] == 0
    code, out = run_cli(["qualify-bundle", "--bundle-dir", calibrated["bundle"], "--test-report", report,
                         "--reviewer", " A. Reviewer ", "--output-dir", tmp_path / "qualified"])
    assert code == 0
    summary = json.loads(out)
    assert summary["qualified_channels"] == [29, 35] and summary["reviewer"] == "A. Reviewer"
    assert summary["bundle_dir"] == str(tmp_path / "qualified")
    assert summary["source_bundle_sha256sums"] == bundle_digest(calibrated["bundle"])
    assert summary["bundle_sha256sums"] == bundle_digest(tmp_path / "qualified") != summary["source_bundle_sha256sums"]
    assert not validate_runtime_weight_bundle(bundle_dir=tmp_path / "qualified").get("errors")
    profiles = json.loads((tmp_path / "qualified" / "pilot_profiles.json").read_text())["profiles"]
    assert {p["physical_channel"]: p["fine_calibration"]["provenance"]["mask_qualification"]["reviewer"]
            for p in profiles if p["physical_channel"] != 37} == {29: "A. Reviewer", 35: "A. Reviewer"}


def write_run(path, frames, seq, channels=(521, 491)):
    """Write one dtv_fine_powers file of a detect-only run: frames [N, channels, 3, 256]."""
    with h5py.File(path, "w") as handle:
        handle["dtv_fine_powers_n0"] = np.asarray(frames).astype(np.uint64)
        handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = list(channels)
        handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = seq
    return path


@pytest.fixture(scope="module")
def run_bundle(calibrated):
    """A bundle that binds channel 35 and the control 37, as a detect-only run records them."""
    tmp = calibrated["tmp"]
    plan = json.loads((tmp / "calibrate.json").read_text())
    plan["channels"] = {"35": plan["channels"]["35"], "37": plan["control"]["captures"]}
    (tmp / "calibrate_run.json").write_text(json.dumps(plan))
    code, _ = run_cli(["calibrate-bundle", "--plan", tmp / "calibrate_run.json", "--output-dir", tmp / "run_bundle"])
    assert code == 0
    profiles = json.loads((tmp / "run_bundle" / "pilot_profiles.json").read_text())["profiles"]
    anchor = next(p for p in profiles if p["physical_channel"] == 35)["fine_calibration"]["anchor_bin"]
    return dict(bundle=tmp / "run_bundle", anchor=anchor)


def option_arguments(directory):
    """Write copies of the profiles and the power-limit inputs under directory, and return the plan options of
    plan-run and process-run, each set to a value other than its default."""
    receiver, core = directory / "receiver.json", directory / "core.json"
    receiver.write_bytes(RECEIVER.read_bytes())
    core.write_bytes(CORE.read_bytes())
    (directory / "limits.json").write_text(json.dumps({"35": 1e-5}))
    inputs = directory / "inputs.json"
    inputs.write_text(json.dumps(dict(table=str(ROOT / "src/pilot_proxy/characterization/data/tolerance_table.json"),
                                      limits="limits.json", curvature={"35": [0.0, 0.0]}, margin=0.0)))
    return ["--receiver-profile", receiver, "--detector-core-profile", core, "--null-quantile", "0.9",
            "--threshold", "power_limit", "--power-limit", inputs, "--decision-version", "fine_decision_v3",
            "--night", *DRAO]


def assert_plan_options(calibrate, directory):
    """Check that the calibration plan holds each option of option_arguments(directory)."""
    assert calibrate["receiver_profile"] == str((directory / "receiver.json").resolve())
    assert calibrate["detector_core_profile"] == str((directory / "core.json").resolve())
    assert calibrate["null_quantile"] == 0.9 and calibrate["decision_version"] == "fine_decision_v3"
    assert calibrate["threshold"] == "power_limit"
    assert calibrate["power_limit"]["limits"] == str((directory / "limits.json").resolve())
    assert calibrate["default_word"] == dict(night=dict(latitude_deg=49.3208, longitude_deg_east=-119.6236))


def test_plan_run_splits_each_channel_and_passes_its_options(run_bundle, tmp_path):
    frames = np.ones((2, 2, 3, 256))
    for index in range(6):
        write_run(tmp_path / f"run_{index}.h5", frames, index * 2 * 8192)
    common = ["plan-run", "--bundle-dir", run_bundle["bundle"], "--run", tmp_path / "run_*.h5", "--epoch", "synthetic"]
    code, out = run_cli(common + ["--output-dir", tmp_path / "plans"])
    assert code == 0
    assert out.splitlines() == ["channel 35: 4 files to calibrate, 2 to test", "channel 37: 4 files to calibrate, 2 to test"]
    calibrate = json.loads((tmp_path / "plans" / "calibrate.json").read_text())
    test = json.loads((tmp_path / "plans" / "test.json").read_text())
    assert [Path(p).name for p in calibrate["channels"]["35"]] == [f"run_{i}.h5" for i in range(4)]
    assert test["channels"]["35"] == test["control"] == [str(tmp_path / f"run_{i}.h5") for i in (4, 5)]
    assert calibrate["control"]["channel"] == 37 and calibrate["null_quantile"] == 0.99
    # the bundle's receiver profile and the default core profile
    assert calibrate["receiver_profile"] == str(RECEIVER.resolve())
    assert calibrate["detector_core_profile"] == str(Path(CORE_PROFILE).resolve())
    assert not {"threshold", "decision_version", "default_word"} & set(calibrate)
    # every option of the command reaches the plan
    code, out = run_cli(common + ["--output-dir", tmp_path / "options", "--calibration-share", "0.5",
                                  *option_arguments(tmp_path)])
    assert code == 0 and "channel 35: 3 files to calibrate, 3 to test" in out
    assert_plan_options(json.loads((tmp_path / "options" / "calibrate.json").read_text()), tmp_path)


def write_process_run(directory, anchor, pilot):
    """Write 8 files of 32 frames of a detect-only run, with channel 35's pilot in the files that pilot(index) names."""
    rng = np.random.default_rng(3)
    directory.mkdir()
    for index in range(8):
        data = rng.gamma(44, 1e6 / 44, size=(32, 2, 3, 256))
        if pilot(index):
            data[:, 0, 0, anchor] *= 30
        write_run(directory / f"run_{index}.h5", data, index * 32 * 8192)
    return str(directory / "run_*.h5")


def test_process_run_prints_the_summary_of_a_passing_run(run_bundle, tmp_path):
    runs = write_process_run(tmp_path / "run", run_bundle["anchor"], lambda index: True)
    out_dir = tmp_path / "processed"
    code, out = run_cli(["process-run", "--bundle-dir", run_bundle["bundle"], "--run", runs, "--epoch", "synthetic",
                         "--output-dir", out_dir, "--calibration-share", "0.5", "--receiver-profile", RECEIVER,
                         "--chunk-seconds", 16 * FRAME_SECONDS])
    assert code == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["passed"] and summary["calibration_share"] == 0.5
    rows = {row["channel"]: row for row in summary["channels"]}
    assert rows[35]["calibrate_masked_percent"] > 90 and rows[35]["test_masked_percent"] > 90
    assert rows[37]["test_masked_percent"] < 10
    assert out.splitlines()[:4] == ["plans written", f"bundle calibrated, floor eta {summary['floor_eta']:.4f}",
                                    "bundle test passed", "spectrograms of the calibrate part written"]
    assert out.endswith((out_dir / "summary.md").read_text()) and "Test PASSED." in out
    assert json.loads((out_dir / "test.json").read_text())["passed"]
    assert len(json.loads((out_dir / "plans" / "calibrate.json").read_text())["channels"]["35"]) == 4
    # 16 of the 32 frames of each file in a chunk
    with np.load(out_dir / "spectrograms" / "test" / "spectrogram_ch35.npz") as data:
        assert data["frames"].tolist() == [16] * 8
    for part in ("calibrate", "test"):
        assert (out_dir / "spectrograms" / part / "masked_summary.csv").exists()


def test_process_run_exits_1_when_the_test_fails(run_bundle, tmp_path):
    # the pilot is in the calibration half only, so the test half masks far less than calibration predicts
    runs = write_process_run(tmp_path / "run", run_bundle["anchor"], lambda index: index < 4)
    out_dir = tmp_path / "processed"
    code, out = run_cli(["process-run", "--bundle-dir", run_bundle["bundle"], "--run", runs, "--epoch", "synthetic",
                         "--output-dir", out_dir, "--calibration-share", "0.5", "--receiver-profile", RECEIVER])
    assert code == 1
    assert "bundle test FAILED" in out and "Test FAILED." in out
    summary = json.loads((out_dir / "summary.json").read_text())
    rows = {row["channel"]: row for row in summary["channels"]}
    assert not summary["passed"] and rows[35]["test_passed"] is False
    assert "- masked fraction on channel 35: " in (out_dir / "summary.md").read_text()


def test_process_run_passes_its_options_to_the_plans(run_bundle, tmp_path):
    runs = write_process_run(tmp_path / "run", run_bundle["anchor"], lambda index: True)
    out_dir = tmp_path / "processed"
    # the run files record no frame time, so calibration stops at the night option, after the plans are written
    with pytest.raises(ValueError, match="records no fpga_seq_time_nsec"):
        run_cli(["process-run", "--bundle-dir", run_bundle["bundle"], "--run", runs, "--epoch", "synthetic",
                 "--output-dir", out_dir, "--calibration-share", "0.5", *option_arguments(tmp_path)])
    calibrate = json.loads((out_dir / "plans" / "calibrate.json").read_text())
    assert_plan_options(calibrate, tmp_path)
    assert len(calibrate["channels"]["35"]) == 4


def loud_run(path, anchor, seq=0, missing=False):
    """Write 48 frames of channels 35 and 29 in which channel 35's pilot fills the first 16 frames, far above any
    word of the bundle, and the other frames hold noise far below every word."""
    rng = np.random.default_rng(7)
    frames = rng.integers(900, 1100, size=(48, 2, 3, 256)).astype(np.uint64)
    frames[:16, 0, 0, anchor] *= np.uint64(1000)
    if missing:
        # 0xFF fills a block missing from the concat window
        frames[40:] = np.iinfo(np.uint64).max
    return write_run(path, frames, seq, channels=(521, 614))


def test_run_spectrogram_writes_each_channel_and_the_masked_summary(calibrated, tmp_path):
    profiles = json.loads((calibrated["bundle"] / "pilot_profiles.json").read_text())["profiles"]
    anchor = next(p for p in profiles if p["physical_channel"] == 35)["fine_calibration"]["anchor_bin"]
    run = loud_run(tmp_path / "run.h5", anchor)
    out_dir = tmp_path / "out"
    code, out = run_cli(["run-spectrogram", "--bundle-dir", calibrated["bundle"], "--run", run, "--channel", "35", "29",
                         "--chunk-seconds", 16 * FRAME_SECONDS, "--output-dir", out_dir])
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("channel 35: 32 of 48 frames kept, ")
    assert lines[1].startswith("channel 29: 48 of 48 frames kept, ")
    assert lines[2] == str(out_dir / "masked_summary.csv")
    for channel in ("35", "29"):
        assert (out_dir / f"spectrogram_ch{channel}.npz").exists() and (out_dir / f"spectrogram_ch{channel}.pdf").exists()
    with np.load(out_dir / "spectrogram_ch35.npz") as data:
        assert data["frames"].tolist() == [16, 16, 16] and data["kept"].tolist() == [0, 16, 16]
    summary = {row["channel"]: row for row in csv.DictReader((out_dir / "masked_summary.csv").open())}
    assert summary["35"]["masked_frames"] == "16" and summary["35"]["masked_percent"] == "33.33"
    assert summary["29"]["masked_frames"] == "0"


def test_run_scores_write_the_cache_of_each_channel(calibrated, tmp_path, monkeypatch):
    profiles = json.loads((calibrated["bundle"] / "pilot_profiles.json").read_text())["profiles"]
    anchor = next(p for p in profiles if p["physical_channel"] == 35)["fine_calibration"]["anchor_bin"]
    # the second file was recorded first
    runs = [loud_run(tmp_path / f"run_{index}.h5", anchor, seq=(1 - index) * 48 * 8192, missing=True)
            for index in range(2)]
    out_dir = tmp_path / "scores"
    workers = []

    class SerialPool:
        """The process pool of run_scores, which records its workers and maps in this process."""
        def __init__(self, jobs, mp_context=None): workers.append(jobs)
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def map(self, function, *args): return map(function, *args)

    monkeypatch.setattr(run_scores, "ProcessPoolExecutor", SerialPool)
    code, out = run_cli(["run-scores", "--bundle-dir", calibrated["bundle"], "--run", *runs, "--channel", "35", "29",
                         "--event-seconds", 8 * FRAME_SECONDS, "--jobs", "3", "--output-dir", out_dir])
    assert code == 0
    # the files of each channel are scored by --jobs workers
    assert workers == [3, 3]
    lines = out.splitlines()
    # 40 recorded frames of each file, and 16 of them hold the pilot
    assert lines[0].startswith("channel 35: 80 frames, 10 events, 40.0% masked")
    assert lines[1].startswith("channel 29: 80 frames, 10 events, 0.0% masked")
    for channel in (35, 29):
        record = json.loads((out_dir / f"ch{channel}.json").read_text())
        assert (record["frames"], record["events"], record["event_seconds"]) == (80, 10, 8 * FRAME_SECONDS)
        assert (out_dir / f"product_ch{channel}.npz").exists()
    frames = threshold_rule.load_frames(35, out_dir)
    assert np.all(np.diff(frames.time) > 0)
    assert frames.excess[:16].min() > 10 * abs(frames.excess[16:40]).max()
