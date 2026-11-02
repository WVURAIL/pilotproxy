#!/usr/bin/env python3
"""Offline tests for toolkit selection and explicit CuPy setup.

Commands and installation are simulated at the subprocess boundary. Toolkit
metadata determines the wheel before the driver's advertised CUDA capability.
"""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pilot_proxy.archive import accel


def test_cuda_major_parsing():
    # nvidia-smi header form
    assert accel._cuda_major_from_text("...\nCUDA Version: 12.4    |\n...") == 12
    # nvcc --version form
    assert accel._cuda_major_from_text("Cuda compilation tools, release 11.8, V11.8.89") == 11
    # nothing parseable
    assert accel._cuda_major_from_text("no version here") is None


def test_cupy_package_name():
    assert accel.cupy_package(12) == "cupy-cuda12x"
    assert accel.cupy_package(11) == "cupy-cuda11x"


def test_cuda_version_json_is_parsed(tmp_path):
    path = tmp_path / "version.json"
    path.write_text(json.dumps({
        "cuda": {"name": "CUDA SDK", "version": "12.8.0"},
    }))
    assert accel._cuda_major_from_file(str(path)) == 12


@pytest.fixture
def toolkit_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(accel, "_DEFAULT_CUDA_HOME", str(tmp_path))
    return tmp_path


def _commands(monkeypatch, responses):
    calls = []

    def run(command, *, capture_output, text, timeout):
        assert capture_output is True and text is True
        assert timeout == 30
        key = tuple(command)
        calls.append(key)
        response = responses[key]
        if isinstance(response, Exception):
            raise response
        stdout, stderr = response
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(accel.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("variable", ["CUDA_HOME", "CUDA_PATH"])
def test_configured_toolkit_command_precedes_other_version_sources(
        monkeypatch, toolkit_environment, variable):
    home = toolkit_environment
    monkeypatch.setenv(variable, str(home))
    (home / "version.json").write_text('{"cuda":{"version":"11.8.0"}}')
    command = (str(home / "bin/nvcc"), "--version")
    calls = _commands(monkeypatch, {command: ("release 12.8", "")})

    assert accel.detect_cuda_major() == 12
    assert calls == [command]


def test_cuda_home_takes_precedence_over_cuda_path(monkeypatch, toolkit_environment):
    home = toolkit_environment
    monkeypatch.setenv("CUDA_HOME", str(home))
    monkeypatch.setenv("CUDA_PATH", str(home / "other"))
    command = (str(home / "bin/nvcc"), "--version")
    calls = _commands(monkeypatch, {command: ("release 12.8", "")})

    assert accel.detect_cuda_major() == 12
    assert calls == [command]


@pytest.mark.parametrize("filename,contents", [
    ("version.json", '{"cuda":{"version":"12.8.0"}}'),
    ("version.txt", "CUDA Version 12.8.0"),
])
def test_configured_version_file_precedes_path_nvcc_and_driver(
        monkeypatch, toolkit_environment, filename, contents):
    home = toolkit_environment
    monkeypatch.setenv("CUDA_HOME", str(home))
    (home / filename).write_text(contents)
    command = (str(home / "bin/nvcc"), "--version")
    calls = _commands(monkeypatch, {command: FileNotFoundError()})

    assert accel.detect_cuda_major() == 12
    assert calls == [command]


def test_path_nvcc_precedes_default_version_file_and_driver(
        monkeypatch, toolkit_environment):
    (toolkit_environment / "version.json").write_text('{"cuda":"11.8.0"}')
    calls = _commands(monkeypatch, {("nvcc", "--version"): ("release 13.0", "")})

    assert accel.detect_cuda_major() == 13
    assert calls == [("nvcc", "--version")]


def test_unavailable_configured_toolkit_falls_back_to_path(
        monkeypatch, toolkit_environment):
    home = toolkit_environment
    monkeypatch.setenv("CUDA_HOME", str(home))
    configured = (str(home / "bin/nvcc"), "--version")
    calls = _commands(monkeypatch, {
        configured: FileNotFoundError(),
        ("nvcc", "--version"): ("release 13.0", ""),
    })

    assert accel.detect_cuda_major() == 13
    assert calls == [configured, ("nvcc", "--version")]


def test_default_version_file_precedes_driver(monkeypatch, toolkit_environment):
    (toolkit_environment / "version.json").write_text('{"version":"12.8.0"}')
    calls = _commands(monkeypatch, {("nvcc", "--version"): FileNotFoundError()})

    assert accel.detect_cuda_major() == 12
    assert calls == [("nvcc", "--version")]


def test_driver_version_is_the_last_resort(monkeypatch, toolkit_environment):
    calls = _commands(monkeypatch, {
        ("nvcc", "--version"): FileNotFoundError(),
        ("nvidia-smi",): ("CUDA Version: 13.0", ""),
    })

    assert accel.detect_cuda_major() == 13
    assert calls == [("nvcc", "--version"), ("nvidia-smi",)]


def test_no_version_source_returns_none(monkeypatch, toolkit_environment):
    _commands(monkeypatch, {
        ("nvcc", "--version"): FileNotFoundError(),
        ("nvidia-smi",): FileNotFoundError(),
    })
    assert accel.detect_cuda_major() is None


def test_command_version_can_be_reported_on_stderr(monkeypatch):
    _commands(monkeypatch, {("nvcc", "--version"): ("", "release 12.8")})
    assert accel._cuda_major_from_command(["nvcc", "--version"]) == 12


@pytest.mark.parametrize("error", [
    FileNotFoundError(), PermissionError(),
    subprocess.TimeoutExpired(["nvcc", "--version"], 30),
])
def test_unavailable_or_timed_out_version_command_returns_none(monkeypatch, error):
    _commands(monkeypatch, {("nvcc", "--version"): error})
    assert accel._cuda_major_from_command(["nvcc", "--version"]) is None


def test_existing_cupy_is_returned_without_detection_or_installation(monkeypatch):
    cupy = object()
    monkeypatch.setattr(accel, "import_cupy", lambda: cupy)
    monkeypatch.setattr(accel, "detect_cuda_major", lambda: pytest.fail("unexpected detection"))
    monkeypatch.setattr(accel.subprocess, "check_call", lambda *a: pytest.fail("unexpected install"))
    assert accel.ensure_cupy(install=True) is cupy


def test_missing_cupy_requires_explicit_install_opt_in(monkeypatch):
    monkeypatch.setattr(accel, "import_cupy", lambda: None)
    monkeypatch.setattr(accel.subprocess, "check_call", lambda *a: pytest.fail("unexpected install"))
    with pytest.raises(RuntimeError, match="setup_env.sh"):
        accel.ensure_cupy()


def test_explicit_install_uses_detected_toolkit_wheel(monkeypatch):
    cupy = object()
    imports = iter([None, cupy])
    monkeypatch.setattr(accel, "import_cupy", lambda: next(imports))
    monkeypatch.setattr(accel, "detect_cuda_major", lambda: 12)
    commands = []
    monkeypatch.setattr(accel.subprocess, "check_call", commands.append)

    assert accel.ensure_cupy(install=True, quiet=True) is cupy
    assert commands == [[sys.executable, "-m", "pip", "install",
                         "--break-system-packages", "cupy-cuda12x"]]


def test_install_refuses_an_unknown_cuda_version(monkeypatch):
    monkeypatch.setattr(accel, "import_cupy", lambda: None)
    monkeypatch.setattr(accel, "detect_cuda_major", lambda: None)
    monkeypatch.setattr(accel.subprocess, "check_call", lambda *a: pytest.fail("unexpected install"))
    with pytest.raises(RuntimeError, match="CUDA version could not be detected"):
        accel.ensure_cupy(install=True)
