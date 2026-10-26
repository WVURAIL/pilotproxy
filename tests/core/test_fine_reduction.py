# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

import pilot_proxy.fine_reduction as fine_reduction_module
from pilot_proxy.fine_reduction import (
    CFAR_MODE_MEDIAN_LEFT,
    CFAR_MODE_QUANTILE_FALLBACK,
    FINE_PAD_FACTOR,
    calibrate_cfar,
    fine_bin_count,
    fine_bin_frequencies_hz,
    fine_reduce,
    frame_powers_from_fine_u64,
    independent_bin_mask,
    p_fa_to_threshold_k,
    coarse_power_ratio_from_powers,
)

TERMS = 3
STREAMS = 8
WINDOWS = 32
ROWS = STREAMS * WINDOWS
RNG_SEED = 0xC0FFEE
MAX_COMPONENT = 14336  # kernel bound for the locked 4-bit path


def _random_matched_filter_row_projections(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        -MAX_COMPONENT, MAX_COMPONENT + 1, size=(TERMS, ROWS, 2), dtype=np.int32
    )


def test_frame_powers_are_the_exact_all_bin_sums_of_the_fixed_point_fine_powers() -> None:
    # In kernel core 3.0.0, P[n] = sum_b S[n, b] over the frozen fxfft256
    # powers, in exact integers. fxfft256 is frozen at 128 windows per stream.
    from pilot_proxy.fxfft import fine_power_fx

    rng = np.random.default_rng(RNG_SEED)
    rows = rng.integers(-MAX_COMPONENT, MAX_COMPONENT + 1, size=(TERMS, 4 * 128, 2), dtype=np.int32)
    powers = frame_powers_from_fine_u64(rows, num_streams=4)
    fine = fine_power_fx(rows, num_streams=4)
    assert powers.dtype == np.dtype(np.uint64) and powers.shape == (TERMS,)
    assert [int(p) for p in powers] == [sum(int(v) for v in fine[n]) for n in range(TERMS)]

    # The fixed-point transform does not conserve energy exactly, so the frame
    # power differs from 256 times the row-domain energy by less than 1 part
    # in 10^4.
    energy = (rows.astype(np.int64) ** 2).sum(axis=(1, 2))
    ratio = powers.astype(np.float64) / (256.0 * energy.astype(np.float64))
    assert np.all(np.abs(ratio - 1.0) < 1e-4)


def test_the_row_domain_marginal_identity_is_retired() -> None:
    for name in ("exact_coarse_" + "power_by_term", "check_coarse_power_" + "marginal_identity"):
        assert not hasattr(fine_reduction_module, name)
    fields = fine_reduction_module.FineReductionResult.__dataclass_fields__
    assert ("coarse_power_ratio_from_" + "marginal") not in fields


def test_fine_reduce_parseval_and_identity() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    matched_filter_row_projections = _random_matched_filter_row_projections(rng)

    result = fine_reduce(
        matched_filter_row_projections, num_streams=STREAMS, windows_per_stream=WINDOWS
    )
    p2 = fine_bin_count(WINDOWS)
    assert result.fine_power.shape == (TERMS, p2)
    assert result.fine_power_ratio.shape == (p2,)

    # Only the float prototype has this property. Its FFT conserves energy in
    # exact arithmetic, sum_b S[n, b] == p2 * sum_m |z[n, m]|^2, and
    # coarse_power_by_term is that row energy, computed exactly.
    sums = result.fine_power.astype(np.float64).sum(axis=-1)
    expected = float(p2) * result.coarse_power_by_term.astype(np.float64)
    np.testing.assert_allclose(sums, expected, rtol=1e-6)
    energy = (matched_filter_row_projections.astype(np.int64) ** 2).sum(axis=(1, 2))
    np.testing.assert_array_equal(result.coarse_power_by_term, energy)
    assert coarse_power_ratio_from_powers(result.coarse_power_by_term) == pytest.approx(
        2.0 * energy[0] / (energy[1] + energy[2])
    )


def test_synthetic_tone_lands_in_correct_padded_bin() -> None:
    # A pure envelope tone at unpadded bin q must peak at padded bin
    # q * FINE_PAD_FACTOR in the target spectrum, coherently across streams.
    q = 5
    n = np.arange(WINDOWS)
    tone = np.exp(2j * np.pi * q * n / WINDOWS)
    amplitude = 100.0
    rng = np.random.default_rng(RNG_SEED + 9)
    z = np.zeros((TERMS, STREAMS, WINDOWS), dtype=np.complex128)
    z[0] = amplitude * tone[None, :]
    # broadband references so the denominator is positive in every padded
    # bin (a constant reference has zero power at even padded bins, and the
    # zero-denominator rule correctly maps those to F2 = 0)
    z[1] = rng.standard_normal((STREAMS, WINDOWS)) + 1j * rng.standard_normal(
        (STREAMS, WINDOWS)
    )
    z[2] = rng.standard_normal((STREAMS, WINDOWS)) + 1j * rng.standard_normal(
        (STREAMS, WINDOWS)
    )
    result = fine_reduce(
        z.reshape(TERMS, ROWS),
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
    )
    peak = int(np.argmax(result.fine_power[0]))
    assert peak == q * FINE_PAD_FACTOR
    assert int(np.argmax(result.fine_power_ratio)) == peak

    freqs = fine_bin_frequencies_hz(WINDOWS, envelope_rate_hz=3051.7578125)
    assert freqs[peak] == pytest.approx(q * 3051.7578125 / WINDOWS)


def test_independent_bin_mask_guard_and_census() -> None:
    p2 = fine_bin_count(WINDOWS)
    designated = [10]
    census = [40]
    mask = independent_bin_mask(
        p2,
        designated_bins=designated,
        census_excluded_bins=census,
        guard_fine_bins=1,
    )
    assert mask[0]
    assert not mask[1]  # padded neighbor of bin 0, so not independent
    guard = FINE_PAD_FACTOR
    for b in range(10 - guard, 10 + guard + 1):
        assert not mask[b % p2]
    assert not mask[40]
    assert mask[(10 + guard + FINE_PAD_FACTOR) % p2]


def test_cfar_median_left_mode_and_detection() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    p2 = fine_bin_count(WINDOWS)
    f = 1.0 + 0.05 * rng.standard_normal(p2)
    hot = 7 * FINE_PAD_FACTOR
    f[hot] = 5.0
    calibration = calibrate_cfar(f, designated_bins=[], p_fa=1.0e-3)
    assert calibration.mode == CFAR_MODE_MEDIAN_LEFT
    assert calibration.location == pytest.approx(1.0, abs=0.05)
    assert 0.0 < calibration.scale < 0.2
    assert calibration.null_bulk_exceedance_fraction < 0.05
    assert "nd_flag_rate" not in calibration.as_dict()
    assert "null_bulk_exceedance_fraction" in calibration.as_dict()


def test_cfar_quantile_fallback_triggers_on_contaminated_bulk() -> None:
    rng = np.random.default_rng(RNG_SEED + 3)
    p2 = fine_bin_count(WINDOWS)
    f = 1.0 + 0.02 * rng.standard_normal(p2)
    # contaminate a third of the independent bins with a hot shelf
    idx = np.flatnonzero(independent_bin_mask(p2))
    f[idx[:: 3]] = 3.0
    calibration = calibrate_cfar(f, p_fa=1.0e-3, fallback_flag_fraction=0.05)
    assert calibration.mode == CFAR_MODE_QUANTILE_FALLBACK


def test_threshold_k_matches_gaussian_tail() -> None:
    assert p_fa_to_threshold_k(0.1587) == pytest.approx(1.0, abs=2e-3)
    assert p_fa_to_threshold_k(1.0e-3) == pytest.approx(3.0902, abs=1e-3)
    with pytest.raises(ValueError):
        p_fa_to_threshold_k(0.9)
