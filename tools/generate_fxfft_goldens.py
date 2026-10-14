#!/usr/bin/env python3
# coding=utf-8
"""Generate the frozen golden-vector suite for fxfft256 v1.

Deterministic by construction (fixed seeds, frozen spec): rerunning this
script must reproduce ``tests/data/fxfft256_golden_v1.npz`` byte-for-byte
in its arrays. The stored *outputs* are produced by the Python reference;
the test suite bit-compares both the Python and C implementations against
this file, so a regression in either implementation cannot self-confirm.

Families:

* ``zero``, ``impulse``, ``dc`` --- exactness anchors (no-rounding paths).
* ``phasor`` --- coherent lines at the survey-measured pilot offsets of
  the first three scanned channels (+739 Hz / +1287 Hz / +1216 Hz for
  ch36 / ch35 / ch34), a half-bin scalloping case, and a low bin.
* ``phasor_noise`` --- lines embedded in Gaussian row-sum noise.
* ``noise_h0`` --- noise-only frames at three amplitude scales.
* ``int4_dot`` --- exact integer dot products of int4 samples against an
  int4 weight row: the deployed row-sum arithmetic and dynamic range.
* ``stress`` --- deployed-maximum (2**14) and contract-maximum (2**20)
  patterns exercising the no-overflow headroom analysis.
* ``random`` --- seeded uniform vectors at both bounds.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pilot_proxy.fxfft import (  # noqa: E402
    FXFFT256_SPEC_VERSION,
    INPUT_ABS_MAX,
    fxfft256,
    twiddle_table_sha256,
)

OUT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "fxfft256_golden_v1.npz"
)

FINE_BIN_HZ = (390625.0 / 128.0) / 256.0
MEASURED_OFFSETS_HZ = (739.0, 1287.0, 1216.0)  # ch36, ch35, ch34 (survey)


def phasor(amp: float, bin_f: float, phase: float = 0.0) -> np.ndarray:
    m = np.arange(128)
    z = amp * np.exp(1j * (2.0 * np.pi * bin_f * m / 256.0 + phase))
    return np.stack([np.round(z.real), np.round(z.imag)], axis=-1).astype(np.int64)


def main() -> None:
    rng = np.random.default_rng(20260731)
    vecs: list[np.ndarray] = []
    fams: list[str] = []

    def add(fam: str, v: np.ndarray) -> None:
        assert v.shape == (128, 2)
        assert int(np.abs(v).max()) <= INPUT_ABS_MAX
        vecs.append(v.astype(np.int64))
        fams.append(fam)

    add("zero", np.zeros((128, 2), np.int64))
    for pos, amp in ((0, 1), (0, 16384), (1, -16384), (127, 12345)):
        v = np.zeros((128, 2), np.int64)
        v[pos, 0] = amp
        add("impulse", v)
    for amp in (100, -16384):
        v = np.zeros((128, 2), np.int64)
        v[:, 0] = amp
        add("dc", v)

    for off_hz in MEASURED_OFFSETS_HZ:
        for amp in (1000.0, 16000.0):
            add("phasor", phasor(amp, off_hz / FINE_BIN_HZ))
    add("phasor", phasor(16000.0, 108.5))  # half-bin scalloping
    add("phasor", phasor(16000.0, 2.25))
    add("phasor", phasor(3000.0, 62.0, phase=1.234))

    for off_hz, amp, sig in ((1287.0, 900.0, 300.0), (739.0, 120.0, 90.0)):
        for _ in range(3):
            n = np.round(rng.normal(0.0, sig, size=(128, 2)))
            add("phasor_noise", phasor(amp, off_hz / FINE_BIN_HZ) + n.astype(np.int64))

    for sig in (30.0, 300.0, 3000.0):
        for _ in range(3):
            add("noise_h0", np.round(rng.normal(0.0, sig, size=(128, 2))).astype(np.int64))

    w = rng.integers(-7, 8, size=(128, 2))  # int4 weight row (re, im)
    for _ in range(6):
        s = rng.integers(-8, 8, size=(128, 128, 2))  # windows x K samples
        wr, wi = w[:, 0], w[:, 1]
        sr, si = s[..., 0], s[..., 1]
        # z[m] = sum_k s[m,k] * conj(w[k]) — exact integer dot per window
        zr = (sr * wr + si * wi).sum(axis=1)
        zi = (si * wr - sr * wi).sum(axis=1)
        add("int4_dot", np.stack([zr, zi], axis=-1))

    alt = np.empty((128, 2), np.int64)
    alt[:, 0] = 16384 * ((-1) ** np.arange(128))
    alt[:, 1] = -16384 * ((-1) ** np.arange(128))
    add("stress", alt)
    for pat in (
        np.full((128, 2), INPUT_ABS_MAX, np.int64),
        np.stack(
            [
                INPUT_ABS_MAX * ((-1) ** np.arange(128)),
                INPUT_ABS_MAX * ((-1) ** (np.arange(128) // 2)),
            ],
            axis=-1,
        ),
    ):
        add("stress", pat)

    for _ in range(8):
        add("random", rng.integers(-16384, 16385, size=(128, 2)))
    for _ in range(8):
        add("random", rng.integers(-INPUT_ABS_MAX, INPUT_ABS_MAX + 1, size=(128, 2)))

    inputs = np.stack(vecs).astype(np.int32)
    outputs = fxfft256(inputs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        inputs=inputs,
        outputs=outputs,
        family=np.asarray(fams),
        spec_version=np.asarray(FXFFT256_SPEC_VERSION),
        twiddle_sha256=np.asarray(twiddle_table_sha256()),
        rounding=np.asarray("floor((v + 2**14) / 2**15) per component"),
        generator=np.asarray("tools/generate_fxfft_goldens.py seed=20260731"),
    )
    print(f"wrote {OUT_PATH}  ({inputs.shape[0]} vectors)")
    import hashlib

    h = hashlib.sha256(inputs.tobytes() + outputs.tobytes()).hexdigest()
    print(f"vector sha256: {h}")


if __name__ == "__main__":
    main()
