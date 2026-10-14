# coding=utf-8
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pilot_proxy import cli
from pilot_proxy import detect as detect_module
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
from pilot_proxy.paths import CONFIGS_DIR, DEFAULT_WEIGHTS_PATH, PACKAGE_ROOT

CUSTOM_GNURADIO_PYTHON = "/custom/gnuradio-python"
SMOKE_IQ_SAMPLES = "128"
SMOKE_TS_PACKETS = "16"
CHANNEL_14 = "14"
THRESHOLD_SNR_SHELF_DB = "-26"
PILOT_BELOW_DATA_DB = "11.3"
DTV_BANDWIDTH_HZ_TEXT = "6000000"
NORMALIZED_DTV_BANDWIDTH_HZ_TEXT = "6000000.0"
MAX_DENOMINATOR_TEXT = "1024"
PILOT_FREQUENCY_TOLERANCE_HZ_TEXT = "5"
NORMALIZED_PILOT_FREQUENCY_TOLERANCE_HZ_TEXT = "5.0"
NUM_INPUT_STREAMS_TEXT = "2"
FRAME_SIZE_TEXT = "16384"
CHECK_LAYOUT_ROWS_TEXT = "256, True, ok"
SNR_START_DB_TEXT = "-60"
NORMALIZED_SNR_START_DB_TEXT = "-60.0"
SNR_STOP_DB_TEXT = "0"
NORMALIZED_SNR_STOP_DB_TEXT = "0.0"
SNR_STEP_DB_TEXT = "10"
NORMALIZED_SNR_STEP_DB_TEXT = "10.0"
FREQUENCY_OFFSET_HZ_TEXT = "1000"
NORMALIZED_FREQUENCY_OFFSET_HZ_TEXT = "1000.0"
CHANNEL_GAIN_DB_TEXT = "1.5"
CHANNEL_PHASE_DEG_TEXT = "12"
NORMALIZED_CHANNEL_PHASE_DEG_TEXT = "12.0"
HISTOGRAM_BINS_TEXT = "12"
REFERENCE_PROFILE = str(CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json")


def _package_bridge_snapshot(env: dict[str, str]) -> tuple[Path, list[str], Path]:
    bridge = Path(env["PYTHONPATH"].split(os.pathsep)[0])
    entries = sorted(path.name for path in bridge.iterdir())
    package_target = (bridge / "pilot_proxy").resolve()
    return bridge, entries, package_target


def test_generate_atsc_uses_gnuradio_python_and_disables_user_site(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    bridge_snapshots = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        bridge_snapshots.append(_package_bridge_snapshot(env))
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "existing")
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "generate-atsc",
            "--gnuradio-python",
            CUSTOM_GNURADIO_PYTHON,
            "--output-iq",
            str(tmp_path / "out.cfile"),
            "--output-ts",
            str(tmp_path / "out.ts"),
            "--num-iq-samples",
            SMOKE_IQ_SAMPLES,
            "--num-ts-packets",
            SMOKE_TS_PACKETS,
        ]
    )

    assert result == 0
    assert len(calls) == 1
    cmd, cwd, env = calls[0]
    assert cmd[:3] == [
        CUSTOM_GNURADIO_PYTHON,
        "-m",
        "pilot_proxy.testbench.generate_atsc_signal",
    ]
    assert cwd == tmp_path
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[1:] == ["existing"]
    bridge, entries, package_target = bridge_snapshots[0]
    assert not bridge.exists()  # lifetime is scoped to the child invocation
    assert entries == ["pilot_proxy"]
    assert package_target == PACKAGE_ROOT


def test_installed_wrapper_preserves_cwd_without_fake_source_path(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "SOURCE_CHECKOUT_ROOT", None)
    monkeypatch.setenv("PYTHONPATH", "caller-supplied")
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "audit-atsc",
            "--input-iq",
            "relative-input.cfile",
            "--output-json",
            "relative-output.json",
        ]
    )

    assert result == 0
    _, cwd, env = calls[0]
    assert cwd == tmp_path
    assert env["PYTHONPATH"] == "caller-supplied"


def test_installed_wrapper_exposes_package_to_secondary_python(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    bridge_snapshots = []

    def fake_run(cmd, *, cwd, env):
        bridge_snapshots.append(_package_bridge_snapshot(env))
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "SOURCE_CHECKOUT_ROOT", None)
    monkeypatch.setenv("PYTHONPATH", "caller-supplied")
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "generate-atsc",
            "--gnuradio-python",
            CUSTOM_GNURADIO_PYTHON,
            "--output-iq",
            "relative-output.cfile",
            "--output-ts",
            "relative-output.ts",
            "--num-iq-samples",
            SMOKE_IQ_SAMPLES,
            "--num-ts-packets",
            SMOKE_TS_PACKETS,
        ]
    )

    assert result == 0
    cmd, cwd, env = calls[0]
    assert cmd[0] == CUSTOM_GNURADIO_PYTHON
    assert cwd == tmp_path
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[1:] == ["caller-supplied"]
    assert pythonpath[0] != str(PACKAGE_ROOT.parent)
    bridge, entries, package_target = bridge_snapshots[0]
    assert not bridge.exists()
    assert entries == ["pilot_proxy"]
    assert package_target == PACKAGE_ROOT
    assert cmd[cmd.index("--output-iq") + 1] == str(
        (tmp_path / "relative-output.cfile").resolve()
    )


def test_audit_atsc_forwards_fail_on_quality(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "audit-atsc",
            "--input-iq",
            str(tmp_path / "in.cfile"),
            "--output-json",
            str(tmp_path / "audit.json"),
            "--fail-on-quality",
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [
        cli.sys.executable,
        "-m",
        "pilot_proxy.testbench.audit_atsc_signal",
    ]
    assert "--fail-on-quality" in cmd


def test_detect_wrapper_forwards_public_calibration_controls(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "detect",
            "--input-detector-matrix",
            str(tmp_path / "matrix.npy"),
            "--physical-channel",
            CHANNEL_14,
            "--pilot-below-data-db",
            PILOT_BELOW_DATA_DB,
            "--dtv-bandwidth-hz",
            DTV_BANDWIDTH_HZ_TEXT,
            "--pilot-frequency-tolerance-hz",
            PILOT_FREQUENCY_TOLERANCE_HZ_TEXT,
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [cli.sys.executable, "-m", "pilot_proxy.detect"]
    assert "--pilot-below-data-db" in cmd
    assert cmd[cmd.index("--dtv-bandwidth-hz") + 1] == (
        NORMALIZED_DTV_BANDWIDTH_HZ_TEXT
    )
    assert cmd[cmd.index("--pilot-frequency-tolerance-hz") + 1] == (
        NORMALIZED_PILOT_FREQUENCY_TOLERANCE_HZ_TEXT
    )


@pytest.mark.parametrize(
    "removed_option",
    ("--threshold-data-shelf-snr-db", "--max-denominator"),
)
def test_detect_parsers_reject_removed_noop_options(removed_option: str) -> None:
    for parser, prefix in (
        (cli.build_parser(), ["detect"]),
        (detect_module.build_parser(), []),
    ):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([*prefix, removed_option, "1"])
        assert exc_info.value.code == 2


def test_detect_wrapper_has_default_detector_input(monkeypatch) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "detect",
            "--physical-channel",
            CHANNEL_14,
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert "--input-detector-matrix" in cmd
    assert cmd[cmd.index("--input-detector-matrix") + 1].endswith(
        "generated/detector_input/detector_matrix_i4.npy"
    ) or cmd[cmd.index("--input-detector-matrix") + 1].endswith(
        "generated\\detector_input\\detector_matrix_i4.npy"
    )


def test_evaluate_snr_cli_forwards_parsed_namespace(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(
        "pilot_proxy.testbench.evaluate_snr.run", fake_run
    )

    result = cli.main(
        [
            "evaluate-snr",
            "--input-iq", str(tmp_path / "clean.cfile"),
            "--output-dir", str(tmp_path / "eval"),
            "--snr-start-db", SNR_START_DB_TEXT,
            "--snr-stop-db", SNR_STOP_DB_TEXT,
            "--snr-step-db", SNR_STEP_DB_TEXT,
            "--frequency-offset-hz", FREQUENCY_OFFSET_HZ_TEXT,
            "--standard-frequency-offset-sweep",
            "--channel-gain-db", CHANNEL_GAIN_DB_TEXT,
            "--channel-phase-deg", CHANNEL_PHASE_DEG_TEXT,
            "--num-input-streams", NUM_INPUT_STREAMS_TEXT,
        ]
    )

    assert result == 0
    args = captured["args"]
    # The subparser inherits the testbench parser (parents=), so values arrive
    # typed and normalized by the single authoritative parser.
    assert args.num_input_streams == int(NUM_INPUT_STREAMS_TEXT)
    assert args.snr_start_db == float(SNR_START_DB_TEXT)
    assert args.snr_stop_db == float(SNR_STOP_DB_TEXT)
    assert args.snr_step_db == float(SNR_STEP_DB_TEXT)
    assert args.frequency_offset_hz == [float(FREQUENCY_OFFSET_HZ_TEXT)]
    assert args.standard_frequency_offset_sweep is True
    assert args.channel_gain_db == float(CHANNEL_GAIN_DB_TEXT)
    assert args.channel_phase_deg == float(CHANNEL_PHASE_DEG_TEXT)
    assert args.detector_backend == "cuda"


def test_evaluate_snr_cli_accepts_detector_backend(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        "pilot_proxy.testbench.evaluate_snr.run",
        lambda args: captured.setdefault("args", args) and 0 or 0,
    )
    # The exact publication-sweep command shape from the validation runbook.
    result = cli.main(
        [
            "evaluate-snr",
            "--input-iq", str(tmp_path / "clean.cfile"),
            "--physical-channel", "14",
            "--frame-size-samples", "16384",
            "--num-input-streams", "4",
            "--snr-start-db", "-38", "--snr-stop-db", "-24", "--snr-step-db", "1",
            "--standard-frequency-offset-sweep",
            "--threshold-data-shelf-snr-db", "-32",
            "--noise-trials", "300",
            "--output-dir", str(tmp_path / "pd_curves"),
            "--detector-backend", "cpu-reference",
        ]
    )
    assert result == 0
    assert captured["args"].detector_backend == "cpu-reference"
    assert captured["args"].noise_trials == 300


def test_evaluate_snr_cli_inherits_every_testbench_option() -> None:
    import argparse

    from pilot_proxy.testbench.evaluate_snr import build_parser as tb_build

    def option_strings(parser):
        out = set()
        for action in parser._actions:
            out.update(o for o in action.option_strings if o.startswith("--"))
        out.discard("--help")
        return out

    top = cli.build_parser()
    sub = None
    for action in top._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub = action.choices["evaluate-snr"]
    missing = option_strings(tb_build()) - option_strings(sub)
    assert not missing, (
        f"CLI evaluate-snr is missing testbench options: {sorted(missing)}"
    )

def test_plot_results_wrapper(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "plot-results",
            "--input-csv",
            str(tmp_path / "summary.csv"),
            "--output-png",
            str(tmp_path / "plot.png"),
            "--title",
            "custom",
            "--smooth-window",
            "5",
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [cli.sys.executable, "-m", "pilot_proxy.testbench.plot_results"]
    assert cmd[cmd.index("--input-csv") + 1].endswith("summary.csv")
    assert cmd[cmd.index("--output-png") + 1].endswith("plot.png")
    assert cmd[cmd.index("--smooth-window") + 1] == "5"


def test_summarize_results_wrapper(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "summarize-results",
            "--input",
            str(tmp_path / "result.json"),
            "--output-dir",
            str(tmp_path / "summary"),
            "--bins",
            HISTOGRAM_BINS_TEXT,
            "--histograms",
            "never",
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [
        cli.sys.executable,
        "-m",
        "pilot_proxy.testbench.summarize_results",
    ]
    assert cmd[cmd.index("--input") + 1].endswith("result.json")
    assert cmd[cmd.index("--bins") + 1] == HISTOGRAM_BINS_TEXT
    assert cmd[cmd.index("--histograms") + 1] == "never"


def test_quantize_wrapper_forwards_num_input_streams(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "quantize",
            "--input-iq",
            str(tmp_path / "clean.cfile"),
            "--output-dir",
            str(tmp_path / "detector"),
            "--num-input-streams",
            NUM_INPUT_STREAMS_TEXT,
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [cli.sys.executable, "-m", "pilot_proxy.testbench.quantize"]
    assert cmd[cmd.index("--num-input-streams") + 1] == NUM_INPUT_STREAMS_TEXT


def test_check_layout_reports_detector_rows(capsys) -> None:
    result = cli.main(
        [
            "check-layout",
            "--frame-size-samples",
            FRAME_SIZE_TEXT,
            "--num-input-streams",
            NUM_INPUT_STREAMS_TEXT,
        ]
    )

    assert result == 0
    out = capsys.readouterr().out
    assert "detector_rows_per_frame" in out
    assert CHECK_LAYOUT_ROWS_TEXT in out


def test_check_profile_loads_nested_receiver_profile(capsys) -> None:
    result = cli.main(
        [
            "check-profile",
            "--receiver-profile",
            REFERENCE_PROFILE,
        ]
    )

    assert result == 0
    out = capsys.readouterr().out
    assert "receiver_profile_id" in out
    assert "reference_800mhz_pfb" in out


def test_check_profile_rejects_incompatible_stream_map() -> None:
    with pytest.raises(ValueError, match="receiver_profile_id"):
        cli.main(
            [
                "check-profile",
                "--receiver-profile",
                str(CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"),
                "--stream-map",
                str(CONFIGS_DIR / "stream_maps" / "chord_dish_pol_example.json"),
            ]
        )


def test_make_weights_from_reference_profile(tmp_path, capsys) -> None:
    output = tmp_path / "weights.bin"

    result = cli.main(
        [
            "make-weights",
            "--receiver-profile",
            REFERENCE_PROFILE,
            "--physical-channel-range",
            CHANNEL_14,
            "--weight-coordinate-system",
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.exists()
    assert output.with_suffix(output.suffix + ".manifest.json").exists()
    out = capsys.readouterr().out
    assert "physical_channels" in out
    assert CHANNEL_14 in out


def test_make_weights_reports_adaptive_reference_on_stderr(tmp_path, capsys) -> None:
    output = tmp_path / "weights.bin"

    result = cli.main(
        [
            "make-weights",
            "--receiver-profile",
            REFERENCE_PROFILE,
            "--physical-channel-range",
            "21",
            "--weight-coordinate-system",
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            "--output",
            str(output),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "NOTE adaptive reference placement" in captured.err
    assert "physical_channel=21" in captured.err
    assert "status=edge_wrapped" in captured.err
    assert "lower_reference=edge_wrapped" in captured.err
    assert "physical_channel=21" not in captured.out


def test_make_weights_requires_weight_coordinate_system(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "make-weights",
                "--receiver-profile",
                REFERENCE_PROFILE,
                "--physical-channel-range",
                CHANNEL_14,
                "--output",
                str(tmp_path / "weights.bin"),
            ]
        )

    assert exc_info.value.code == 2


def test_list_channels_reports_adaptive_reference_on_stderr(capsys) -> None:
    result = cli.main(
        [
            "list-channels",
            "--weights-path",
            str(DEFAULT_WEIGHTS_PATH),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "physical_channel,pilot_mhz,coarse_channel_index" in captured.out
    assert "14,470.309441,843,True" in captured.out
    assert "NOTE adaptive reference placement" in captured.err
    assert "physical_channel=14" in captured.err
    assert "reason=edge_reference_wrapped" in captured.err


def test_export_runtime_weight_bundle_wrapper(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        return {
            "detector_contract": tmp_path / "detector_contract.json",
            "pilot_profiles": tmp_path / "pilot_profiles.json",
        }

    monkeypatch.setattr(cli, "export_runtime_weight_bundle", fake_export)

    result = cli.main(
        [
            "export-runtime-weight-bundle",
            "--receiver-profile",
            REFERENCE_PROFILE,
            "--physical-channel-range",
            "14:15",
            "--weight-coordinate-system",
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            "--output-dir",
            str(tmp_path / "bundle"),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["physical_channels"] == [14, 15]
    assert calls[0]["weight_coordinate_system"] == (
        WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    )
    assert calls[0]["output_dir"] == tmp_path / "bundle"
    out = capsys.readouterr().out
    assert "detector_contract" in out
    assert "pilot_profiles" in out


def test_validate_runtime_weight_bundle_wrapper(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    def fake_validate(**kwargs):
        calls.append(kwargs)
        return {
            "valid": True,
            "num_errors": 0,
            "bundle_dir": str(kwargs["bundle_dir"]),
            "errors": [],
        }

    monkeypatch.setattr(cli, "validate_runtime_weight_bundle", fake_validate)

    result = cli.main(
        [
            "validate-runtime-weight-bundle",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-json",
            str(tmp_path / "bundle_validation.json"),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["bundle_dir"] == tmp_path / "bundle"
    assert calls[0]["output_json"] == tmp_path / "bundle_validation.json"
    out = capsys.readouterr().out
    assert "valid, num_errors, bundle_dir" in out


def test_validate_products_wrapper(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "validate-products",
            "--run-dir",
            str(tmp_path / "run"),
            "--output-json",
            str(tmp_path / "validation.json"),
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [cli.sys.executable, "-m", "pilot_proxy.chime.validate_products"]
    assert cmd[cmd.index("--run-dir") + 1].endswith("run")
    assert cmd[cmd.index("--output-json") + 1].endswith("validation.json")


def test_chime_run_wrapper_does_not_forward_detector_window(monkeypatch, tmp_path) -> None:
    calls = []

    # noinspection PyShadowingNames

    def fake_run(cmd, *, cwd, env):
        calls.append((cmd, cwd, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(
        [
            "chime-run",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "run"),
            "--physical-channel",
            CHANNEL_14,
        ]
    )

    assert result == 0
    cmd = calls[0][0]
    assert cmd[:3] == [cli.sys.executable, "-m", "pilot_proxy.chime.runner"]
    assert "--detector-window-samples" not in cmd


def test_chime_scan_forwards_explicit_allow_partial(monkeypatch, tmp_path) -> None:
    calls = []
    fake_scan = ModuleType("pilot_proxy.archive.scan")

    def fake_run_chime_scan(**kwargs):
        calls.append(kwargs)

    fake_scan.run_chime_scan = fake_run_chime_scan  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_scan.__name__, fake_scan)

    result = cli.main(
        [
            "chime-scan",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--select",
            "844",
            "--download-workers",
            "3",
            "--max-staged-files",
            "5",
            "--staging-dir",
            str(tmp_path / "staging"),
            "--source-event-regex",
            r"baseband_(evt\d+)_",
            "--allow-partial",
        ]
    )

    assert result == 0
    assert calls[0]["allow_partial"] is True
    assert calls[0]["download_workers"] == 3
    assert calls[0]["max_staged_files"] == 5
    assert calls[0]["staging_dir"] == tmp_path / "staging"
    assert calls[0]["source_event_regex"] == r"baseband_(evt\d+)_"
