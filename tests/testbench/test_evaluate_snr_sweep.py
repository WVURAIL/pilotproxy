"""Reject invalid sweep requests before loading IQ or detector hardware."""

import argparse

import pytest

from pilot_proxy.testbench import evaluate_snr


def _request(**changes):
    values = dict(requested_data_shelf_snr_db=None, snr_start_db=None,
                  snr_stop_db=None, snr_step_db=None)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("field", ["requested_data_shelf_snr_db", "snr_start_db", "snr_stop_db"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -61.0, 61.0])
def test_sweep_refuses_nonfinite_or_unsupported_snr_before_iteration(field, value):
    changes = dict(snr_start_db=-3, snr_stop_db=3, snr_step_db=1)
    changes[field] = [value] if field == "requested_data_shelf_snr_db" else value
    with pytest.raises(SystemExit, match="finite.*supported range"):
        evaluate_snr._requested_snr_shelf_values(_request(**changes))


@pytest.mark.parametrize("step", [float("nan"), float("inf"), -float("inf"), 0.0])
def test_sweep_refuses_nonfinite_or_zero_steps(step):
    with pytest.raises(SystemExit, match="finite and non-zero"):
        evaluate_snr._requested_snr_shelf_values(
            _request(snr_start_db=-3, snr_stop_db=3, snr_step_db=step))


@pytest.mark.parametrize("changes", [
    {"snr_start_db": 0}, {"snr_stop_db": 0}, {"snr_step_db": 1},
    {"snr_start_db": 0, "snr_stop_db": 3},
    {"snr_start_db": 0, "snr_step_db": 1},
    {"snr_stop_db": 3, "snr_step_db": 1},
])
def test_partial_range_cannot_silently_select_the_default_sweep(changes):
    with pytest.raises(SystemExit, match="range requires"):
        evaluate_snr._requested_snr_shelf_values(_request(**changes))


@pytest.mark.parametrize("start,stop,step", [(0, 3, -1), (3, 0, 1), (1, 2, 1e-300)])
def test_range_step_must_advance_toward_its_endpoint(start, stop, step):
    with pytest.raises(SystemExit, match="advance"):
        evaluate_snr._requested_snr_shelf_values(
            _request(snr_start_db=start, snr_stop_db=stop, snr_step_db=step))


def test_microscopic_step_is_refused_before_allocating_the_range():
    with pytest.raises(SystemExit, match="at most 1000000 points"):
        evaluate_snr._requested_snr_shelf_values(
            _request(snr_start_db=0, snr_stop_db=1, snr_step_db=1e-300))


@pytest.mark.parametrize("sign", [-1, 1])
def test_endpoint_tolerance_cannot_extend_the_supported_snr_range(sign):
    with pytest.raises(SystemExit, match="supported range"):
        evaluate_snr._requested_snr_shelf_values(_request(
            snr_start_db=sign * 60, snr_stop_db=sign * (60 + 5e-13),
            snr_step_db=sign * 2e-13,
        ))


def test_sweep_point_limit_includes_explicit_values_and_range_endpoints(monkeypatch):
    monkeypatch.setattr(evaluate_snr, "MAX_SNR_SWEEP_POINTS", 4)
    assert evaluate_snr._requested_snr_shelf_values(_request(
        requested_data_shelf_snr_db=[-3], snr_start_db=0, snr_stop_db=2, snr_step_db=1,
    )) == [-3, 0, 1, 2]
    assert evaluate_snr._requested_snr_shelf_values(_request(
        requested_data_shelf_snr_db=[0, 1, 2, 3],
    )) == [0, 1, 2, 3]
    for changes in (
        dict(snr_start_db=0, snr_stop_db=4, snr_step_db=1),
        dict(requested_data_shelf_snr_db=[-3], snr_start_db=0, snr_stop_db=3, snr_step_db=1),
        dict(requested_data_shelf_snr_db=[0, 1, 2, 3, 4]),
    ):
        with pytest.raises(SystemExit, match="at most 4 points"):
            evaluate_snr._requested_snr_shelf_values(_request(**changes))


@pytest.mark.parametrize("start,stop,step,expected", [
    (-3, 3, 1.5, [-3, -1.5, 0, 1.5, 3]),
    (3, -3, -1.5, [3, 1.5, 0, -1.5, -3]),
    (-3, 2, 2, [-3, -1, 1]),
    (2, -3, -2, [2, 0, -2]),
    (1, 1, 1e-300, [1]),
])
def test_valid_sweeps_preserve_requested_order_and_stop_boundary(start, stop, step, expected):
    result = evaluate_snr._requested_snr_shelf_values(_request(
        requested_data_shelf_snr_db=[-60, 60],
        snr_start_db=start, snr_stop_db=stop, snr_step_db=step,
    ))
    assert result == [-60, 60, *expected]


def test_invalid_cli_sweep_stops_before_iq_or_hardware_access(monkeypatch, tmp_path):
    def unexpected_access(*args, **kwargs):
        pytest.fail("invalid sweep reached IQ or detector initialization")

    monkeypatch.setattr(evaluate_snr.np, "fromfile", unexpected_access)
    monkeypatch.setattr(evaluate_snr, "PilotProxyKernel", unexpected_access)
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="advance"):
        evaluate_snr.main([
            "--input-iq", str(tmp_path / "absent.cfile"), "--output-dir", str(output),
            "--snr-start-db", "0", "--snr-stop-db", "3", "--snr-step-db", "-1",
        ])
    assert not output.exists()


@pytest.mark.parametrize("kind", ["directory", "nonexecutable_file"])
def test_gnuradio_preflight_rejects_existing_paths_that_cannot_execute(monkeypatch, tmp_path, kind):
    program = tmp_path / "python-helper"
    if kind == "directory":
        program.mkdir()
    else:
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o644)

    def unexpected_access(*args, **kwargs):
        pytest.fail("invalid executable reached IQ or detector initialization")

    monkeypatch.setattr(evaluate_snr.np, "fromfile", unexpected_access)
    monkeypatch.setattr(evaluate_snr, "PilotProxyKernel", unexpected_access)
    with pytest.raises(SystemExit, match="Could not find GNU Radio Python executable"):
        evaluate_snr.main([
            "--input-iq", str(tmp_path / "absent.cfile"),
            "--output-dir", str(tmp_path / "output"),
            "--gnuradio-python", str(program),
        ])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("by_name", [False, True])
def test_gnuradio_preflight_accepts_executable_paths_and_path_commands(monkeypatch, tmp_path, by_name):
    program = tmp_path / "python-helper"
    program.write_text("#!/bin/sh\nexit 0\n")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    def stop_before_loading_iq(*args, **kwargs):
        raise RuntimeError("executable preflight passed")

    monkeypatch.setattr(evaluate_snr.np, "fromfile", stop_before_loading_iq)
    with pytest.raises(RuntimeError, match="executable preflight passed"):
        evaluate_snr.main([
            "--input-iq", str(tmp_path / "absent.cfile"),
            "--output-dir", str(tmp_path / "output"),
            "--gnuradio-python", program.name if by_name else str(program),
            "--detector-backend", "cpu-reference", "--synthesis-backend", "cpu",
        ])
