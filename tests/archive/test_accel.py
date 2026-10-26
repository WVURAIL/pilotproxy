#!/usr/bin/env python3
"""
Tests for CuPy resolution in the archive pipeline.

These tests run offline. They cover the CUDA major version parsed from
nvidia-smi, nvcc, and version.json text, the name of the matching CuPy wheel,
the numpy path on the CPU, and the error that points to setup_env.sh when CuPy
is absent. The install itself is not covered, since it needs a CUDA image.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_accel.py
"""
from __future__ import annotations

import json

import numpy as np

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


def test_get_array_module_cpu_is_numpy():
    assert accel.get_array_module(False) is np


def test_gpu_path_matches_cupy_availability():
    cp = accel.import_cupy()
    if cp is None:
        # without cupy the scan path exits with SystemExit and names setup_env.sh
        raised = False
        try:
            accel.get_array_module(True)
        except SystemExit as exc:
            raised = True
            assert "setup_env.sh" in str(exc)
        assert raised, "get_array_module(True) should SystemExit when cupy is absent"
        # ensure_cupy(install=False) raises a RuntimeError that names setup_env.sh
        try:
            accel.ensure_cupy(install=False)
        except RuntimeError as exc:
            assert "setup_env.sh" in str(exc)
        else:
            raise AssertionError("ensure_cupy(install=False) should raise when cupy is absent")
    else:
        # cupy present, as on a CUDA image, so both calls return the cupy module
        assert accel.get_array_module(True) is cp
        assert accel.ensure_cupy(install=False) is cp



if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        globals()[fn]()
        print(f"  ok: {fn}")
    print("GPU MODULE TESTS PASSED")
