# coding=utf-8
"""Cross-implementation gate for the size-parameterized C reference.

``cuda/fxfft_ref.c`` is the C companion to :func:`pilot_proxy.fxfft.fxfft`.
This suite compiles it at every supported length and requires bit-identical
output against the Python reference, the same standard the frozen
``fxfft256_ref.c`` is held to, extended across the family.

Two properties matter beyond "it computes an FFT":

* at ``FX_N = 256`` the generalized C path must reproduce the *frozen*
  ``fxfft256`` exactly, so nothing about the deployed geometry moves; and
* ``cuda/fxfft_master_twiddle.h`` must match what the Python master emits,
  so the device tables and the Python reference cannot drift apart.

Skipped when no C compiler is available.
"""

from __future__ import annotations

import pathlib
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from pilot_proxy.fxfft import MASTER_TWIDDLE_SHA256, fxfft, fxfft256

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REF_SRC = REPO_ROOT / "cuda" / "fxfft_ref.c"
TABLE_HDR = REPO_ROOT / "cuda" / "fxfft_master_twiddle.h"
EMITTER = REPO_ROOT / "tools" / "emit_fxfft_tables.py"

FAMILY = (128, 256, 512, 1024, 2048)
CC = shutil.which("cc") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler available")


def _build(tmp_path: pathlib.Path, n: int) -> pathlib.Path:
    exe = tmp_path / f"fxfft_ref_{n}"
    proc = subprocess.run(
        [CC, "-O2", "-Wall", "-Wextra", f"-DFX_N={n}",
         "-I", str(REPO_ROOT / "cuda"), "-o", str(exe), str(REF_SRC)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"compile failed at FX_N={n}:\n{proc.stderr}"
    assert "warning" not in proc.stderr, f"warnings at FX_N={n}:\n{proc.stderr}"
    return exe


def _run(exe: pathlib.Path, tmp_path: pathlib.Path, x: np.ndarray, n: int) -> np.ndarray:
    fin, fout = tmp_path / "in.bin", tmp_path / "out.bin"
    fin.write_bytes(struct.pack("<I", x.shape[0]) + x.astype("<i4").tobytes())
    subprocess.run([str(exe), str(fin), str(fout)], check=True)
    return np.fromfile(fout, dtype="<i4").reshape(x.shape[0], n, 2)


def test_emitted_table_matches_the_python_master():
    """CI gate: the committed header cannot drift from the generator."""
    proc = subprocess.run(
        [sys.executable, str(EMITTER), "--check"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert MASTER_TWIDDLE_SHA256 in TABLE_HDR.read_text()


@pytest.mark.parametrize("n", FAMILY)
def test_c_reference_matches_python(tmp_path, n):
    exe = _build(tmp_path, n)
    rng = np.random.default_rng(n)
    x = rng.integers(-(2 ** 13), 2 ** 13, size=(24, n // 2, 2)).astype(np.int32)
    got = _run(exe, tmp_path, x, n)
    assert np.array_equal(got, fxfft(x.astype(np.int64), n_out=n))


def test_c_reference_at_256_matches_the_frozen_transform(tmp_path):
    """The generalization must not move the deployed geometry."""
    exe = _build(tmp_path, 256)
    rng = np.random.default_rng(20260807)
    x = rng.integers(-(2 ** 13), 2 ** 13, size=(24, 128, 2)).astype(np.int32)
    got = _run(exe, tmp_path, x, 256)
    assert np.array_equal(got, fxfft256(x.astype(np.int64)))


@pytest.mark.parametrize("n", FAMILY)
def test_c_reference_edge_inputs(tmp_path, n):
    exe = _build(tmp_path, n)
    rows = [
        np.zeros((n // 2, 2), dtype=np.int32),
        np.ones((n // 2, 2), dtype=np.int32),
        -np.ones((n // 2, 2), dtype=np.int32),
    ]
    x = np.stack(rows)
    got = _run(exe, tmp_path, x, n)
    assert np.array_equal(got, fxfft(x.astype(np.int64), n_out=n))


def test_unsupported_size_is_a_compile_error(tmp_path):
    proc = subprocess.run(
        [CC, "-O2", "-DFX_N=192", "-I", str(REPO_ROOT / "cuda"),
         "-o", str(tmp_path / "bad"), str(REF_SRC)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "FX_N must be one of" in proc.stderr
