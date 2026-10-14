# coding=utf-8
"""Transform-family gate: one master twiddle table, every size by decimation.

The frozen ``fxfft256 v1`` specification is unchanged and remains the deployed
artifact; this suite gates the generalization that serves other transform
lengths, and in particular the two properties the generalization rests on:

1. the length-n table is an *exact* decimation of the master (identical
   integers rather than a re-derivation), so the tie analysis and exact-rounding
   argument established on the master are inherited by every member; and
2. the general transform reproduces the frozen one bit-for-bit at n = 256.

Together these mean adding a size introduces no new rounding decisions in the
twiddles, and cannot perturb the deployed geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pilot_proxy.fxfft import (
    MASTER_N,
    MASTER_TWIDDLE_Q15,
    MASTER_TWIDDLE_SHA256,
    N,
    TWIDDLE_Q15,
    fxfft,
    fxfft256,
    input_abs_max,
    master_twiddle_sha256,
    twiddle_table,
)

FAMILY = (128, 256, 512, 1024, 2048)


def _direct_half(n: int) -> list[tuple[int, int]]:
    """Generate W_n[k] for k < n/2 from first principles, no decimation."""
    return [
        (
            round(32768.0 * math.cos(2.0 * math.pi * k / n)),
            round(-32768.0 * math.sin(2.0 * math.pi * k / n)),
        )
        for k in range(n // 2)
    ]


def test_master_hash_is_pinned():
    assert master_twiddle_sha256() == MASTER_TWIDDLE_SHA256


def test_master_has_no_rounding_ties():
    """'nearest' must be unambiguous for every master entry."""
    worst = 1.0
    for k in range(MASTER_N // 2):
        ang = 2.0 * math.pi * k / MASTER_N
        for v in (32768.0 * math.cos(ang), -32768.0 * math.sin(ang)):
            worst = min(worst, abs(v - math.floor(v) - 0.5))
    assert worst > 1e-6, f"master entry sits within {worst} of a rounding tie"


def test_master_values_need_int32_storage():
    """C[0] = +32768 does not fit int16; the spec stores twiddles in int32."""
    values = [v for pair in MASTER_TWIDDLE_Q15 for v in pair]
    assert max(values) == 32768
    assert min(values) == -32768


@pytest.mark.parametrize("n", FAMILY)
def test_decimation_is_exact(n):
    """Decimating the master equals generating length n directly."""
    assert list(twiddle_table(n)) == _direct_half(n)


def test_decimation_reproduces_the_frozen_literal():
    """The join between the frozen artifact and the general construction."""
    assert twiddle_table(N) == TWIDDLE_Q15


@pytest.mark.parametrize("n", FAMILY)
def test_table_length(n):
    assert len(twiddle_table(n)) == n // 2


def test_rejects_non_power_of_two_and_oversize():
    with pytest.raises(ValueError):
        twiddle_table(192)
    with pytest.raises(ValueError):
        twiddle_table(2 * MASTER_N)


def test_general_transform_matches_frozen_at_256():
    """The deployed geometry must be untouched by the generalization."""
    rng = np.random.default_rng(20260807)
    for _ in range(64):
        x = rng.integers(-(2 ** 13), 2 ** 13, size=(3, N // 2, 2), dtype=np.int64)
        assert np.array_equal(fxfft(x, n_out=N), fxfft256(x))


@pytest.mark.parametrize("value", [0, 1, -1, 2 ** 20, -(2 ** 20)])
def test_general_transform_matches_frozen_at_contract_edges(value):
    x = np.full((N // 2, 2), value, dtype=np.int64)
    assert np.array_equal(fxfft(x, n_out=N), fxfft256(x))


@pytest.mark.parametrize("n", FAMILY)
def test_agrees_with_exact_dft_within_rounding_budget(n):
    """Sanity that each member computes the transform it claims to."""
    rng = np.random.default_rng(1000 + n)
    x = rng.integers(-(2 ** 13), 2 ** 13, size=(n // 2, 2), dtype=np.int64)
    got = fxfft(x, n_out=n).astype(np.float64)
    ref = np.fft.fft(np.pad(x[:, 0] + 1j * x[:, 1], (0, n - n // 2)))
    err = np.max(np.abs((got[:, 0] + 1j * got[:, 1]) - ref)) / np.max(np.abs(ref))
    assert err < 1e-3


@pytest.mark.parametrize("n", FAMILY)
def test_input_bound_tightens_and_admits_deployed_matched_filter_row_projections(n):
    """The eight-stage contract is not reusable at nine stages."""
    limit = input_abs_max(n)
    assert limit & (limit - 1) == 0
    # deployed row sums are bounded by 128 * K <= 2**14 for K <= 128
    assert limit >= 2 ** 14
    if n > 128:
        assert input_abs_max(n) <= input_abs_max(n // 2)


def test_input_bound_matches_the_frozen_contract_at_256():
    from pilot_proxy.fxfft import INPUT_ABS_MAX

    assert input_abs_max(N) == INPUT_ABS_MAX


@pytest.mark.parametrize("n", FAMILY)
def test_overflow_contract_is_enforced(n):
    over = input_abs_max(n) + 1
    x = np.full((n // 2, 2), over, dtype=np.int64)
    with pytest.raises(OverflowError):
        fxfft(x, n_out=n)
