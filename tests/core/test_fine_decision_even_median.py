# coding=utf-8
"""Tests of fine decision v3: the statistic [n_peak, d_rho] and its comparison.

The oracle is a direct computation with ``fractions.Fraction``: the reference
powers of the 128 even bins are sorted, the median of rank 63 is taken, and the
largest designated target is compared with the word times that median. As in
Kotekan's RFI mask, 1 keeps and 0 flags.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    FINE_BINS,
    MULTIPLIER_ONE,
    V3_RANK,
    V3_REFERENCE_BINS,
    designated_bins,
    fine_keep_word_v3,
    fine_mask_decision_v2,
    fine_keep_v3,
    fine_stat_v3,
)

ANCHOR, HALF = 141, 32


def frame(rng, *, scale=10**9):
    return rng.integers(scale - scale // 20, scale + scale // 20, size=(3, FINE_BINS)).astype(np.uint64)


def oracle(powers):
    num = [2 * int(powers[0, b]) for b in range(FINE_BINS)]
    den = [int(powers[1, b]) + int(powers[2, b]) for b in range(FINE_BINS)]
    return max(num[b] for b in designated_bins(ANCHOR, HALF)), sorted(den[b] for b in range(0, FINE_BINS, 2))[63]


def test_rank_is_the_median_of_the_even_bins():
    assert V3_REFERENCE_BINS == tuple(range(0, FINE_BINS, 2))
    assert V3_RANK == 63


@pytest.mark.parametrize("seed", range(8))
def test_stat_matches_oracle(seed):
    powers = frame(np.random.default_rng(seed))
    stat = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF)
    assert (stat.n_peak, stat.d_rho) == oracle(powers)
    assert stat.rank_bin % 2 == 0
    assert stat.peak_bin in designated_bins(ANCHOR, HALF)


def test_odd_bins_and_the_search_do_not_set_the_reference():
    powers = frame(np.random.default_rng(1))
    base = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF).d_rho
    powers[1:, 1::2] = np.uint64(10**15)                       # odd bins only
    assert fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF).d_rho == base


def test_median_tolerates_fewer_than_half_outliers():
    powers = frame(np.random.default_rng(2))
    base = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF).d_rho
    even = np.arange(0, FINE_BINS, 2)
    powers[1:, even[:40]] = np.uint64(10**15)                  # 40 of 128 even bins
    moved = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF).d_rho
    assert moved >= base and moved < 3 * base


def test_half_the_even_bins_zero_means_no_decision():
    powers = frame(np.random.default_rng(3))
    powers[1:, np.arange(0, FINE_BINS, 2)[:64]] = 0
    stat = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF)
    assert stat.d_rho == 0
    assert fine_keep_v3(stat.n_peak, stat.d_rho, MULTIPLIER_ONE) is None
    assert fine_keep_word_v3(stat.n_peak, stat.d_rho) is None


@pytest.mark.parametrize("n,d", [(0, 1), (1, 1), (5, 3), (2**57 - 1, 2**40), (2**64 - 1, 1)])
def test_keep_word_is_the_change_point(n, d):
    word = fine_keep_word_v3(n, d)
    if word == ALWAYS_MASKED_Q16:
        assert fine_keep_v3(n, d, 2**64 - 1) == 0
        return
    assert fine_keep_v3(n, d, word) == 1
    if word > 1:
        assert fine_keep_v3(n, d, word - 1) == 0
    assert Fraction(word, MULTIPLIER_ONE) >= Fraction(n, d)


@pytest.mark.parametrize("seed", range(4))
def test_decision_equals_v2_over_the_even_bins(seed):
    powers = frame(np.random.default_rng(100 + seed))
    stat = fine_stat_v3(powers, anchor_bin=ANCHOR, designated_half_width=HALF)
    word = fine_keep_word_v3(stat.n_peak, stat.d_rho)
    even = np.arange(FINE_BINS) % 2 == 0
    for mult in (word - 1, word, word + 1, 1, 3 * MULTIPLIER_ONE):
        v2 = fine_mask_decision_v2(powers, anchor_bin=ANCHOR, designated_half_width=HALF,
                                   bulk_mask=even, cfar_rank=V3_RANK, multiplier_q16=mult)
        assert fine_keep_v3(stat.n_peak, stat.d_rho, mult) == 1 - v2.mask   # 1 keeps, as in the RFI mask


def test_rejects_bad_word():
    with pytest.raises(ValueError):
        fine_keep_v3(1, 1, 0)
    with pytest.raises(ValueError):
        fine_keep_v3(1, 1, 2**64)
