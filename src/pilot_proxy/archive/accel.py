"""Resolve the CuPy build used by archive analyzers.

CuPy must match the CUDA environment where the analysis runs. CANFAR images
usually supply that build already, so the package leaves its GPU extra empty
and resolves CuPy at runtime:

* ``import_cupy()`` returns the environment's CuPy, or None when absent.
* ``get_array_module(g)`` selects NumPy or, when g is true, CuPy.
* ``detect_cuda_major()`` returns the CUDA major version of the environment,
  or None when it cannot be found.
* ``ensure_cupy(install)`` can install the matching ``cupy-cuda<major>x`` wheel
  if CuPy is absent and installation was explicitly requested.

Only ``ensure_cupy(install=True)`` installs packages. A scan does not.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Optional

_CUDA_COMMAND_TIMEOUT_SECONDS = 30
_DEFAULT_CUDA_HOME = "/usr/local/cuda"


def import_cupy():
    """Return the environment's CuPy, or None only when it is not installed."""
    try:
        import cupy as cp  # noqa: F401
        return cp
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            return None
        raise RuntimeError(
            "cupy is installed but could not import one of its dependencies "
            f"({exc}). Check that the CuPy wheel matches the installed CUDA "
            "toolkit; do not install a second CuPy package alongside it.") from exc
    except Exception as exc:
        raise RuntimeError(
            "cupy is installed but failed to initialize "
            f"({type(exc).__name__}: {exc}). Check that the CuPy wheel matches "
            "the installed CUDA toolkit.") from exc


def _cuda_major_from_text(text: str) -> Optional[int]:
    m = re.search(r"CUDA Version[:\s]+(\d+)\.", text)          # nvidia-smi header
    if m:
        return int(m.group(1))
    m = re.search(r"release\s+(\d+)\.", text)                  # nvcc --version
    if m:
        return int(m.group(1))
    return None


def _cuda_major_from_file(path: str) -> Optional[int]:
    """Return the CUDA major version from an NVIDIA version file (JSON or text)."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    if path.endswith(".json"):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        cuda = payload.get("cuda") if isinstance(payload, dict) else None
        if isinstance(cuda, dict):
            version = cuda.get("version")
        elif isinstance(cuda, str):
            version = cuda
        else:
            version = payload.get("version") if isinstance(payload, dict) else None
        match = re.match(r"\s*(\d+)(?:\.|\s|$)", str(version or ""))
        return int(match.group(1)) if match else None
    return _cuda_major_from_text(text)


def _cuda_major_from_command(command: list[str]) -> Optional[int]:
    try:
        out = subprocess.run(
            command, capture_output=True, text=True,
            timeout=_CUDA_COMMAND_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return _cuda_major_from_text((out.stdout or "") + (out.stderr or ""))


def detect_cuda_major() -> Optional[int]:
    """Return the CUDA major version of the running image, or None.

    The installed toolkit is read first (the active nvcc, then its version
    file). The ``nvidia-smi`` header gives the maximum CUDA version the driver
    supports, which may differ from the toolkit that a CuPy wheel must match.
    """
    configured_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cuda_home = configured_home or _DEFAULT_CUDA_HOME
    if configured_home:
        major = _cuda_major_from_command(
            [os.path.join(cuda_home, "bin", "nvcc"), "--version"])
        if major:
            return major
        for fname in ("version.json", "version.txt"):
            major = _cuda_major_from_file(os.path.join(cuda_home, fname))
            if major:
                return major

    major = _cuda_major_from_command(["nvcc", "--version"])
    if major:
        return major

    if not configured_home:
        for fname in ("version.json", "version.txt"):
            major = _cuda_major_from_file(os.path.join(cuda_home, fname))
            if major:
                return major

    # driver's maximum CUDA version, used as the wheel guess for setup-cupy
    # only when no toolkit is found
    return _cuda_major_from_command(["nvidia-smi"])


def cupy_package(major: int) -> str:
    """Return the pip wheel name for a CUDA major version (cupy-cuda12x for 12)."""
    return f"cupy-cuda{int(major)}x"


def ensure_cupy(install: bool = False, quiet: bool = False):
    """Return the image's CuPy, optionally installing the matching wheel.

    If CuPy can already be imported (the common CANFAR case), it is returned
    unchanged. Otherwise, when install=True, the CUDA major version of the image
    is detected, and ``cupy-cuda<major>x`` is installed with pip into the active
    environment and imported. Raises RuntimeError with the setup step to take
    if CuPy cannot be resolved.
    """
    cp = import_cupy()
    if cp is not None:
        return cp

    if not install:
        raise RuntimeError(
            "cupy is not importable in this environment. Run setup_env.sh "
            "or install the CuPy wheel matching this image's CUDA version.")

    major = detect_cuda_major()
    if major is None:
        raise RuntimeError(
            "cupy is missing and the CUDA version could not be detected "
            "(no nvidia-smi/nvcc and no CUDA version file). Install the cupy build "
            "matching your session image manually, e.g. `pip install cupy-cuda12x`.")

    pkg = cupy_package(major)
    if not quiet:
        print(f"[gpu] no cupy found; detected CUDA {major}.x -> installing {pkg}",
              flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", pkg])

    cp = import_cupy()
    if cp is None:
        raise RuntimeError(
            f"installed {pkg} but cupy is still not importable; the wheel may not "
            f"match this image's CUDA module. Check `nvidia-smi` and install by hand.")
    return cp


def get_array_module(use_gpu: bool):
    """Return NumPy, or the image's CuPy when use_gpu is true.

    A scan calls this, and a missing or broken CuPy exits with the setup step.
    """
    import numpy as np
    if not use_gpu:
        return np
    try:
        cp = import_cupy()
    except RuntimeError as exc:
        raise SystemExit(f"--gpu was requested but {exc}") from exc
    if cp is None:
        raise SystemExit(
            "--gpu was requested but cupy is not importable. Run setup_env.sh "
            "first, or drop --gpu to run on CPU.")
    return cp
