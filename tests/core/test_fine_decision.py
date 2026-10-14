# coding=utf-8
"""Gates for the frozen fine decision reference (fine decision v1).

The reference is exact integer arithmetic; the oracle here is
``fractions.Fraction`` (also exact), applied naively: sort the bulk F2
values, take the requested rank, compare each designated bin against
multiplier * rank value. Agreement is required exactly --- including
tie, wrap, and degenerate-denominator cases --- because the CUDA
epilogue is gated bit-for-bit against this module.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    FINE_BINS,
    MAX_MULTIPLIER_Q16,
    MULTIPLIER_ONE,
    designated_bins,
    fine_mask_decision,
    fine_required_multiplier_q16,
    pack_bulk_mask,
    unpack_bulk_mask,
)
from pilot_proxy.fine_reduction import independent_bin_mask

RNG_SEED = 20260805


def _oracle(S, anchor, half_width, mask, rank, mult_q16):
    """Naive exact oracle with Fractions."""
    num = 2 * S[0].astype(object)
    den = S[1].astype(object) + S[2].astype(object)
    bulk = [
        Fraction(int(num[b]), int(den[b]))
        for b in range(FINE_BINS)
        if mask[b] and int(den[b]) > 0
    ]
    if rank >= len(bulk):
        return 0
    thr = (
        Fraction(int(mult_q16), MULTIPLIER_ONE) * sorted(bulk)[rank]
    )
    for b in designated_bins(anchor, half_width):
        if int(den[b]) > 0 and Fraction(int(num[b]), int(den[b])) > thr:
            return 1
    return 0


def _required_multiplier_oracle(S, anchor, half_width, mask, rank):
    """Naive exact keep boundary with Fractions."""
    num = 2 * S[0].astype(object)
    den = S[1].astype(object) + S[2].astype(object)
    bulk = [
        Fraction(int(num[b]), int(den[b]))
        for b in range(FINE_BINS)
        if mask[b] and int(den[b]) > 0
    ]
    if rank >= len(bulk):
        return None
    rank_value = sorted(bulk)[rank]
    required = 1
    for b in designated_bins(anchor, half_width):
        if int(den[b]) <= 0 or int(num[b]) == 0:
            continue
        if rank_value == 0:
            return ALWAYS_MASKED_Q16
        ratio = Fraction(int(num[b]), int(den[b])) / rank_value
        scaled = ratio * MULTIPLIER_ONE
        boundary = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
        required = max(required, boundary)
    return (required if required <= MAX_MULTIPLIER_Q16
            else ALWAYS_MASKED_Q16)


def _random_powers(rng, scale=1 << 40):
    return rng.integers(0, scale, size=(3, FINE_BINS), dtype=np.uint64)


def test_designated_window_wraps_like_survey():
    np.testing.assert_array_equal(
        designated_bins(254, 2), np.array([252, 253, 254, 255, 0])
    )
    np.testing.assert_array_equal(
        designated_bins(1, 2), np.array([255, 0, 1, 2, 3])
    )
    np.testing.assert_array_equal(designated_bins(62, 0), np.array([62]))


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(20):
        m = rng.random(FINE_BINS) < 0.5
        np.testing.assert_array_equal(unpack_bulk_mask(pack_bulk_mask(m)), m)


def test_matches_fraction_oracle_randomized():
    rng = np.random.default_rng(RNG_SEED + 1)
    mask = independent_bin_mask(FINE_BINS, designated_bins=[62])
    for trial in range(200):
        S = _random_powers(rng)
        rank = int(rng.integers(0, 128))
        mult = int(rng.integers(1, 4 * MULTIPLIER_ONE))
        got = fine_mask_decision(
            S,
            anchor_bin=62,
            designated_half_width=2,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=mult,
        )
        assert got.mask == _oracle(S, 62, 2, mask, rank, mult), trial


def test_required_multiplier_reproduces_the_exact_decision():
    rng = np.random.default_rng(RNG_SEED + 3)
    mask = independent_bin_mask(FINE_BINS, designated_bins=[62])
    probes = (1, MULTIPLIER_ONE, 2 * MULTIPLIER_ONE,
              4 * MULTIPLIER_ONE)
    for _ in range(80):
        S = _random_powers(rng)
        rank = int(rng.integers(0, 100))
        boundary = fine_required_multiplier_q16(
            S, anchor_bin=62, designated_half_width=2,
            bulk_mask=mask, cfar_rank=rank)
        assert boundary.valid
        assert boundary.multiplier_q16 == _required_multiplier_oracle(
            S, 62, 2, mask, rank)
        exact_probes = list(probes)
        if boundary.multiplier_q16 <= MAX_MULTIPLIER_Q16:
            exact_probes.append(boundary.multiplier_q16)
            if boundary.multiplier_q16 > 1:
                exact_probes.append(boundary.multiplier_q16 - 1)
        for multiplier in exact_probes:
            decision = fine_mask_decision(
                S, anchor_bin=62, designated_half_width=2,
                bulk_mask=mask, cfar_rank=rank,
                multiplier_q16=multiplier)
            assert decision.mask == int(multiplier < boundary.multiplier_q16)


def test_zero_rank_value_has_an_always_masked_boundary():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    S[1, 2] = S[2, 2] = 1
    mask[2] = True
    S[0, 100] = 1
    S[1, 100] = S[2, 100] = 1
    boundary = fine_required_multiplier_q16(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=0)
    assert boundary.valid
    assert boundary.multiplier_q16 == ALWAYS_MASKED_Q16


def test_required_multiplier_marks_an_invalid_rank():
    S = np.ones((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    mask[2] = True
    boundary = fine_required_multiplier_q16(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=1)
    assert not boundary.valid
    assert boundary.multiplier_q16 is None


def test_required_multiplier_uses_one_when_no_designated_bin_can_fire():
    S = np.ones((3, FINE_BINS), dtype=np.uint64)
    S[0, 100] = 0
    mask = np.zeros(FINE_BINS, dtype=bool)
    mask[2] = True
    boundary = fine_required_multiplier_q16(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=0)
    assert boundary.valid
    assert boundary.multiplier_q16 == 1
    assert fine_mask_decision(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=0, multiplier_q16=1).mask == 0


def test_required_multiplier_marks_a_boundary_above_uint64():
    top = np.iinfo(np.uint64).max
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    S[0, 2] = 1
    S[1, 2] = S[2, 2] = top
    mask[2] = True
    S[0, 100] = top
    S[1, 100] = S[2, 100] = 1
    boundary = fine_required_multiplier_q16(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=0)
    assert boundary.valid
    assert boundary.multiplier_q16 == ALWAYS_MASKED_Q16
    assert fine_mask_decision(
        S, anchor_bin=100, designated_half_width=0,
        bulk_mask=mask, cfar_rank=0,
        multiplier_q16=MAX_MULTIPLIER_Q16).mask == 1


def test_required_multiplier_is_tie_representative_independent():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    for bin_index, values in {0: (1, 2, 2), 2: (2, 4, 4)}.items():
        S[0, bin_index], S[1, bin_index], S[2, bin_index] = values
        mask[bin_index] = True
    S[0, 100], S[1, 100], S[2, 100] = 3, 2, 2
    expected = 3 * MULTIPLIER_ONE
    for rank in (0, 1):
        boundary = fine_required_multiplier_q16(
            S, anchor_bin=100, designated_half_width=0,
            bulk_mask=pack_bulk_mask(mask), cfar_rank=rank)
        assert boundary.valid
        assert boundary.rank_bin == 0
        assert boundary.multiplier_q16 == expected
        assert fine_mask_decision(
            S, anchor_bin=100, designated_half_width=0,
            bulk_mask=mask, cfar_rank=rank,
            multiplier_q16=expected - 1).mask == 1
        assert fine_mask_decision(
            S, anchor_bin=100, designated_half_width=0,
            bulk_mask=mask, cfar_rank=rank,
            multiplier_q16=expected).mask == 0


def test_required_multiplier_matches_decision_input_validation():
    S = np.ones((3, FINE_BINS), dtype=np.uint64)
    mask = np.ones(FINE_BINS, dtype=bool)
    with pytest.raises(TypeError, match="fine_powers.*uint64"):
        fine_required_multiplier_q16(
            S.astype(float), anchor_bin=0, designated_half_width=0,
            bulk_mask=mask, cfar_rank=0)
    with pytest.raises(TypeError, match="bulk mask entries must be booleans"):
        fine_required_multiplier_q16(
            S, anchor_bin=0, designated_half_width=0,
            bulk_mask=mask.astype(np.uint8), cfar_rank=0)
    with pytest.raises(TypeError, match="cfar_rank"):
        fine_required_multiplier_q16(
            S, anchor_bin=0, designated_half_width=0,
            bulk_mask=mask, cfar_rank=0.0)


def test_tied_rank_values_are_representative_independent():
    # Bins 0 and 2 carry the same rational with different representations
    # (2/4 == 1/2); either representative must give the same decision.
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    for b, (t, lo, up) in {0: (1, 2, 2), 2: (2, 4, 4), 4: (5, 1, 1)}.items():
        S[0, b], S[1, b], S[2, b] = t, lo, up
        mask[b] = True
    S[0, 100], S[1, 100], S[2, 100] = 3, 2, 2  # designated bin: F2 = 3/2
    for rank in (0, 1):  # both ranks land on the tied value 1/2
        d = fine_mask_decision(
            S,
            anchor_bin=100,
            designated_half_width=0,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=2 * MULTIPLIER_ONE,  # threshold = 2 * 1/2 = 1
        )
        assert d.valid and d.mask == 1  # 3/2 > 1
        assert d.rank_bin == 0  # lowest qualifying representative
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=2,  # rank value now 5/2; threshold 5 > 3/2
        multiplier_q16=2 * MULTIPLIER_ONE,
    )
    assert d.valid and d.mask == 0


def test_degenerate_rules():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    # Bulk bin with zero denominator is excluded before ranking.
    S[0, 0] = 7  # den == 0
    mask[0] = True
    S[0, 2], S[1, 2], S[2, 2] = 1, 1, 1  # F2 = 1
    mask[2] = True
    S[0, 100], S[1, 100], S[2, 100] = 100, 1, 1
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=0,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.n_bulk == 1 and d.rank_bin == 2 and d.mask == 1
    # Rank beyond the usable bulk: invalid frame, mask forced 0.
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=1,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.mask == 0 and not d.valid
    # Designated bin with zero denominator can never fire.
    S2 = np.zeros((3, FINE_BINS), dtype=np.uint64)
    S2[0, 100] = 1 << 50  # den == 0 at the anchor
    S2[0, 2], S2[1, 2], S2[2, 2] = 1, 1, 1
    m2 = np.zeros(FINE_BINS, dtype=bool)
    m2[2] = True
    d = fine_mask_decision(
        S2,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=m2,
        cfar_rank=0,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.mask == 0 and d.valid


def test_large_power_magnitudes_stay_exact():
    # Near the Parseval-corner scale (~2^55): cross products reach
    # ~2^110 and the threshold triple ~2^127 --- the regime the kernel
    # must cover with its fixed-width helpers.
    rng = np.random.default_rng(RNG_SEED + 2)
    mask = independent_bin_mask(FINE_BINS, designated_bins=[0])
    for _ in range(50):
        S = rng.integers(
            (1 << 54), (1 << 55), size=(3, FINE_BINS), dtype=np.uint64
        )
        rank = int(rng.integers(0, 64))
        mult = int(rng.integers(MULTIPLIER_ONE, 2 * MULTIPLIER_ONE))
        got = fine_mask_decision(
            S,
            anchor_bin=0,
            designated_half_width=2,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=mult,
        )
        assert got.mask == _oracle(S, 0, 2, mask, rank, mult)


def test_injected_line_fires_and_null_does_not():
    # Synthetic spectrum: flat references, flat target except an
    # injected excess at the anchor. The designated set fires; with the
    # line removed the same calibration keeps the mask at 0.
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    S[1, :] = 1000
    S[2, :] = 1000
    S[0, :] = 1000  # F2 = 1 everywhere
    mask = independent_bin_mask(FINE_BINS, designated_bins=[62])
    rank = int(np.count_nonzero(mask) // 2)
    mult = int(1.5 * MULTIPLIER_ONE)
    base = fine_mask_decision(
        S, anchor_bin=62, designated_half_width=2, bulk_mask=mask,
        cfar_rank=rank, multiplier_q16=mult,
    )
    assert base.mask == 0 and base.valid
    S[0, 62] = 10000  # injected pilot: F2 = 10 at the anchor
    fired = fine_mask_decision(
        S, anchor_bin=62, designated_half_width=2, bulk_mask=mask,
        cfar_rank=rank, multiplier_q16=mult,
    )
    assert fired.mask == 1 and fired.fired_bin == 62


def test_input_validation():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.ones(FINE_BINS, dtype=bool)
    with pytest.raises(TypeError):
        fine_mask_decision(
            S.astype(np.float64), anchor_bin=0, designated_half_width=0,
            bulk_mask=mask, cfar_rank=0, multiplier_q16=MULTIPLIER_ONE,
        )
    with pytest.raises(ValueError):
        fine_mask_decision(
            S, anchor_bin=256, designated_half_width=0, bulk_mask=mask,
            cfar_rank=0, multiplier_q16=MULTIPLIER_ONE,
        )
    with pytest.raises(ValueError):
        fine_mask_decision(
            S, anchor_bin=0, designated_half_width=0, bulk_mask=mask,
            cfar_rank=-1, multiplier_q16=MULTIPLIER_ONE,
        )
    with pytest.raises(ValueError, match="cfar_rank"):
        fine_mask_decision(
            S, anchor_bin=0, designated_half_width=0, bulk_mask=mask,
            cfar_rank=FINE_BINS, multiplier_q16=MULTIPLIER_ONE,
        )


@pytest.mark.parametrize(
    "powers",
    [
        np.zeros((3, FINE_BINS), dtype=np.int64),
        -np.ones((3, FINE_BINS), dtype=np.int64),
        np.zeros((3, FINE_BINS), dtype=np.uint32),
        np.zeros((3, FINE_BINS), dtype=np.float64),
    ],
)
def test_fine_powers_require_exact_uint64_dtype(powers: np.ndarray) -> None:
    with pytest.raises(TypeError, match="fine_powers.*uint64"):
        fine_mask_decision(
            powers,
            anchor_bin=0,
            designated_half_width=0,
            bulk_mask=np.ones(FINE_BINS, dtype=bool),
            cfar_rank=0,
            multiplier_q16=MULTIPLIER_ONE,
        )


@pytest.mark.parametrize(
    "mask",
    [
        np.ones(FINE_BINS, dtype=np.uint8),
        np.ones(FINE_BINS, dtype=np.float64),
        np.full(FINE_BINS, np.nan, dtype=np.float64),
        np.full(FINE_BINS, "true", dtype="U4"),
    ],
)
def test_bulk_mask_requires_exact_boolean_dtype(mask: np.ndarray) -> None:
    from pilot_proxy.fine_decision import pack_bulk_mask

    with pytest.raises(TypeError, match="bulk mask entries must be booleans"):
        pack_bulk_mask(mask)

    with pytest.raises(TypeError, match="bulk mask entries must be booleans"):
        fine_mask_decision(
            np.ones((3, FINE_BINS), dtype=np.uint64),
            anchor_bin=0,
            designated_half_width=0,
            bulk_mask=mask,
            cfar_rank=0,
            multiplier_q16=MULTIPLIER_ONE,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("anchor_bin", True),
        ("anchor_bin", np.bool_(True)),
        ("designated_half_width", 1.0),
        ("cfar_rank", "0"),
        ("multiplier_q16", 1.0),
    ],
)
def test_scalar_decision_fields_must_be_exact_integers(
    field: str,
    invalid: object,
) -> None:
    S = np.ones((3, FINE_BINS), dtype=np.uint64)
    arguments = {
        "anchor_bin": 0,
        "designated_half_width": 0,
        "bulk_mask": np.ones(FINE_BINS, dtype=bool),
        "cfar_rank": 0,
        "multiplier_q16": MULTIPLIER_ONE,
    }
    arguments[field] = invalid

    with pytest.raises(TypeError, match=field):
        fine_mask_decision(S, **arguments)


@pytest.mark.parametrize("invalid", [0, -1, 1 << 64])
def test_multiplier_domain_matches_runtime_and_kernel(invalid: int) -> None:
    S = np.ones((3, FINE_BINS), dtype=np.uint64)

    with pytest.raises(ValueError, match="multiplier_q16"):
        fine_mask_decision(
            S,
            anchor_bin=0,
            designated_half_width=0,
            bulk_mask=np.ones(FINE_BINS, dtype=bool),
            cfar_rank=0,
            multiplier_q16=invalid,
        )
