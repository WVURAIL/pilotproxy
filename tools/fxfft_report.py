#!/usr/bin/env python3
# coding=utf-8
"""Quantization study for fxfft256 v1: fixed-point vs float64 pipeline.

Measures, on the golden families and a realistic multi-stream synthetic:

1. spectrum-domain error of the fixed-point FFT against the complex128
   reference (absolute, and relative to the spectrum peak);
2. Parseval drift |sum_b |X|^2 - 256 sum_m |z|^2| (the float pipeline
   holds this at ULP; the fixed-point path holds it to rounding);
3. headroom: per-stage working maxima against the int32 bound;
4. end-to-end fine-statistic delta: F2[b] via the frozen fixed-point path
   versus F2[b] via the analyzer's float pipeline (``fine_reduction``),
   on a synthetic frame with lines at the survey-measured offsets ---
   the number that shows the quantization is negligible against the
   coherent gain.

These measurements set the frozen tolerances in
``tests/core/test_fxfft256.py`` and are quoted in
``docs/DESIGN_DECISIONS.md``.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pilot_proxy import fine_reduction  # noqa: E402
from pilot_proxy.fxfft import (  # noqa: E402
    fine_power_fx,
    fine_power_ratio_fx,
    fxfft256,
)

GOLDEN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "fxfft256_golden_v1.npz"
)


def spectrum_errors() -> None:
    z = np.load(GOLDEN, allow_pickle=False)
    x = z["inputs"].astype(np.int64)
    fam = z["family"]
    X, maxima = fxfft256(x, return_stage_maxima=True)
    zc = x[..., 0] + 1j * x[..., 1]
    Xf = np.fft.fft(zc, n=256, axis=-1)
    err = np.hypot(X[..., 0] - Xf.real, X[..., 1] - Xf.imag)
    print("== spectrum-domain error vs complex128 (per family) ==")
    for f in sorted(set(fam.tolist())):
        sel = fam == f
        peak = np.abs(Xf[sel]).max()
        e = err[sel].max()
        rel = e / peak if peak > 0 else 0.0
        print(f"  {f:13s} max|dX| = {e:10.3f}   peak|X| = {peak:12.1f}   "
              f"rel-to-peak = {rel:.2e}")
    print(f"  stage maxima (bits): {[int(m).bit_length() for m in maxima]}"
          f"   int32 bound: 31")
    par_fx = (X.astype(np.int64) ** 2).sum(axis=(-1, -2)).astype(np.float64)
    par_in = 256.0 * (x.astype(np.int64) ** 2).sum(axis=(-1, -2))
    nz = par_in > 0
    drift = np.abs(par_fx[nz] - par_in[nz]) / par_in[nz]
    print(f"  Parseval drift: max {drift.max():.3e}, "
          f"median {np.median(drift):.3e}")
    print()


def f2_delta() -> None:
    print("== end-to-end fine-statistic delta (fixed-point vs float pipeline) ==")
    rng = np.random.default_rng(4242)
    streams, windows = 128, 128
    fine_bin_hz = (390625.0 / 128.0) / 256.0
    rows = streams * windows
    terms = np.zeros((3, rows, 2), np.int64)
    m = np.arange(windows)
    for s in range(streams):
        base = rng.normal(0.0, 120.0, size=(3, windows, 2))
        line = 260.0 * np.exp(
            1j * (2 * np.pi * (1287.0 / fine_bin_hz) * m / 256.0
                  + rng.uniform(0, 2 * np.pi))
        )
        base[0, :, 0] += line.real
        base[0, :, 1] += line.imag
        terms[:, s * windows:(s + 1) * windows, :] = np.round(base)

    ref = fine_reduction.fine_reduce(
        terms, num_streams=streams, windows_per_stream=windows
    )
    f2_float = ref.fine_power_ratio.astype(np.float64)
    f2_fx = fine_power_ratio_fx(
        fine_power_fx(terms, num_streams=streams, windows_per_stream=windows)
    )
    d = np.abs(f2_fx - f2_float)
    peak = int(np.argmax(f2_float))
    rel = d / np.maximum(f2_float, 1e-30)
    print(f"  streams = {streams}, line at +1287 Hz, peak bin {peak}")
    print(f"  F2[peak]: float = {f2_float[peak]:.6f}   fx = {f2_fx[peak]:.6f}"
          f"   dB delta = {10*np.log10(f2_fx[peak]/f2_float[peak]):+.5f} dB")
    print(f"  max |dF2| (all bins)      = {d.max():.3e}")
    print(f"  max |dF2|/F2 (all bins)   = {rel.max():.3e}")
    print(f"  null-bulk median F2 float = {np.median(f2_float):.4f}, "
          f"fx = {np.median(f2_fx):.4f}")
    print()


if __name__ == "__main__":
    spectrum_errors()
    f2_delta()
