# coding=utf-8
"""Golden-vector and cross-implementation gate for fxfft256 v1.

This suite is the verification contract for the deployed fine-reduction
FFT: the Python reference, the C port template, and (later) the CUDA
implementation must all reproduce ``tests/data/fxfft256_golden_v1.npz``
bit-for-bit. Tolerance tests (float proximity, Parseval drift, fine-
statistic parity) carry frozen bounds set at 4-8x the measured values in
``tools/fxfft_report.py``; a spec change that moves them is a contract
change and must be versioned, not absorbed.
"""
from __future__ import annotations

import math
import pathlib
import shutil
import struct
import subprocess

import numpy as np
import pytest

from pilot_proxy import fine_reduction
from pilot_proxy.fxfft import (
    FXFFT256_SPEC_VERSION,
    INPUT_ABS_MAX,
    TWIDDLE_Q15,
    fine_power_fx,
    fine_power_ratio_fx,
    fxfft256,
    fxfft256_scalar,
    twiddle_table_sha256,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO / "tests" / "data" / "fxfft256_golden_v1.npz"
C_SOURCE = REPO / "cuda" / "fxfft256_ref.c"


@pytest.fixture(scope="module")
def golden():
    z = np.load(GOLDEN_PATH, allow_pickle=False)
    return {
        "inputs": z["inputs"].astype(np.int64),
        "outputs": z["outputs"],
        "family": z["family"],
        "spec_version": str(z["spec_version"]),
        "twiddle_sha256": str(z["twiddle_sha256"]),
    }


def test_twiddle_table_is_frozen_and_tie_free(golden):
    assert golden["spec_version"] == FXFFT256_SPEC_VERSION
    assert golden["twiddle_sha256"] == twiddle_table_sha256()
    for k, (c, s) in enumerate(TWIDDLE_Q15):
        th = 2.0 * math.pi * k / 256.0
        xc, xs = 32768.0 * math.cos(th), -32768.0 * math.sin(th)
        assert c == round(xc) and s == round(xs)
        # no value may sit near a rounding tie (spec unambiguity)
        for v in (xc, xs):
            assert abs(abs(v % 1.0) - 0.5) > 1e-3 or float(v).is_integer()


def test_golden_vectors_bit_exact(golden):
    out = fxfft256(golden["inputs"])
    assert out.dtype == np.int32
    assert (out == golden["outputs"]).all()


def test_scalar_matches_vectorized(golden):
    rng = np.random.default_rng(3)
    idx = rng.choice(golden["inputs"].shape[0], size=6, replace=False)
    for i in idx:
        vec = golden["inputs"][i]
        got = fxfft256_scalar([tuple(p) for p in vec])
        exp = golden["outputs"][i]
        assert all(
            (int(exp[j, 0]), int(exp[j, 1])) == got[j] for j in range(256)
        )


def test_c_reference_bit_exact(golden, tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler available")
    exe = tmp_path / "fxfft256_ref"
    subprocess.run(
        [cc, "-O2", "-o", str(exe), str(C_SOURCE)], check=True, capture_output=True
    )
    x = golden["inputs"].astype("<i4")
    fin = tmp_path / "in.bin"
    fout = tmp_path / "out.bin"
    with open(fin, "wb") as f:
        f.write(struct.pack("<I", x.shape[0]))
        f.write(x.tobytes())
    subprocess.run([str(exe), str(fin), str(fout)], check=True)
    got = np.frombuffer(fout.read_bytes(), dtype="<i4").reshape(-1, 256, 2)
    assert (got == golden["outputs"]).all()


def test_exact_special_cases():
    imp = np.zeros((128, 2), np.int64)
    imp[0, 0] = 12345
    X = fxfft256(imp)
    assert (X[:, 0] == 12345).all() and (X[:, 1] == 0).all()

    dc = np.zeros((128, 2), np.int64)
    dc[:, 0] = -321
    X = fxfft256(dc)
    assert X[0, 0] == -321 * 128 and X[0, 1] == 0

    assert (fxfft256(np.zeros((128, 2), np.int64)) == 0).all()


def test_headroom_at_contract_limit():
    pats = [
        np.full((128, 2), INPUT_ABS_MAX, np.int64),
        np.stack(
            [
                INPUT_ABS_MAX * ((-1) ** np.arange(128)),
                -INPUT_ABS_MAX * ((-1) ** np.arange(128)),
            ],
            axis=-1,
        ),
    ]
    _, maxima = fxfft256(np.stack(pats), return_stage_maxima=True)
    assert max(maxima) < 2**31  # implementation raises otherwise


def test_input_contract_enforced():
    bad = np.zeros((128, 2), np.int64)
    bad[0, 0] = INPUT_ABS_MAX + 1
    with pytest.raises(OverflowError):
        fxfft256(bad)
    with pytest.raises(TypeError):
        fxfft256(np.zeros((128, 2), np.float64))
    with pytest.raises(ValueError):
        fxfft256(np.zeros((64, 2), np.int64))


def test_float64_proximity(golden):
    """Frozen bound: per-vector max error <= 16 + 2e-4 * spectrum peak.

    The constant term covers small-amplitude structured vectors (rounding
    accumulation at ~0.5 LSB per butterfly); the proportional term covers
    Q15 twiddle quantization. Measured maxima sit at roughly half of both
    terms (``tools/fxfft_report.py``).
    """
    x = golden["inputs"]
    X = fxfft256(x).astype(np.float64)
    Xf = np.fft.fft(x[..., 0] + 1j * x[..., 1], n=256, axis=-1)
    err = np.hypot(X[..., 0] - Xf.real, X[..., 1] - Xf.imag).max(axis=-1)
    peak = np.abs(Xf).max(axis=-1)
    assert (err <= 16.0 + 2.0e-4 * peak).all()


def test_parseval_drift_bounded(golden):
    x = golden["inputs"]
    X = fxfft256(x).astype(np.int64)
    par_fx = (X**2).sum(axis=(-1, -2)).astype(np.float64)
    par_in = 256.0 * (x**2).sum(axis=(-1, -2)).astype(np.float64)
    nz = par_in > 0
    drift = np.abs(par_fx[nz] - par_in[nz]) / par_in[nz]
    assert drift.max() <= 5.0e-3
    assert np.median(drift) <= 1.0e-4


def test_fine_statistic_parity_with_float_pipeline():
    """F2 via fx powers tracks the analyzer float pipeline to <= 5e-3."""
    rng = np.random.default_rng(99)
    streams, windows = 32, 128
    rows = streams * windows
    terms = np.zeros((3, rows, 2), np.int64)
    m = np.arange(windows)
    tone = 260.0 * np.exp(1j * 2 * np.pi * 108.0 * m / 256.0)
    for s in range(streams):
        blk = np.round(rng.normal(0.0, 120.0, size=(3, windows, 2)))
        blk[0, :, 0] += np.round(tone.real)
        blk[0, :, 1] += np.round(tone.imag)
        terms[:, s * windows:(s + 1) * windows, :] = blk

    ref = fine_reduction.fine_reduce(
        terms, num_streams=streams, windows_per_stream=windows
    )
    f2_float = ref.fine_power_ratio.astype(np.float64)
    power = fine_power_fx(terms, num_streams=streams, windows_per_stream=windows)
    assert power.dtype == np.uint64
    f2_fx = fine_power_ratio_fx(power)
    rel = np.abs(f2_fx - f2_float) / np.maximum(f2_float, 1e-30)
    assert rel.max() <= 5.0e-3
    # the detection peak lands in the same bin with the same magnitude class
    assert int(np.argmax(f2_fx)) == int(np.argmax(f2_float))
