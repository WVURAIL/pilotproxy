# coding=utf-8
"""Tests of fine decision v2, the largest target bin against the median reference.

The oracle is a direct computation with ``fractions.Fraction``. The reference
powers of the bulk are sorted, the value at the requested rank is taken, and
the largest designated target is compared with the multiplier times that value.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    FINE_BINS,
    MULTIPLIER_ONE,
    designated_bins,
    fine_mask_decision,
    fine_mask_decision_v2,
    fine_required_multiplier_q16_v2,
    pack_bulk_mask,
)
from pilot_proxy.rank_scores import required_multipliers_by_rank_v2

ANCHOR, HALF = 141, 32


def geometry(anchor=ANCHOR, half=HALF):
    designated = designated_bins(anchor, half)
    bulk = np.zeros(FINE_BINS, dtype=bool)
    bulk[np.setdiff1d(np.arange(0, FINE_BINS, 2), designated)] = True
    return designated, bulk


def frame(rng, *, scale=10**9, zeros=0, ties=False, line=None):
    powers = rng.integers(scale - scale // 20, scale + scale // 20, size=(3, FINE_BINS)).astype(np.uint64)
    if ties:
        powers[1:] = np.uint64(scale // 2)
    for b in rng.choice(FINE_BINS, size=zeros, replace=False):
        powers[1:, b] = 0
    if line is not None:
        powers[0, line[0]] = np.uint64(line[1])
    return powers


def oracle(powers, designated, bulk, rank, multiplier):
    den = [int(powers[1, b]) + int(powers[2, b]) for b in range(FINE_BINS)]
    usable = sorted(den[b] for b in range(FINE_BINS) if bulk[b] and den[b] > 0)
    if rank >= len(usable):
        return None
    peak = max(Fraction(2 * int(powers[0, b]), usable[rank]) for b in designated)
    return int(peak > Fraction(multiplier, MULTIPLIER_ONE)), peak


@pytest.mark.parametrize("seed", range(40))
def test_decision_and_boundary_match_the_oracle(seed):
    rng = np.random.default_rng(seed)
    designated, bulk = geometry(anchor=int(rng.integers(0, FINE_BINS)), half=int(rng.integers(0, 40)))
    anchor, half = int(designated[len(designated) // 2]), len(designated) // 2
    line = (int(rng.choice(designated)), int(rng.integers(1, 60) * 10**9)) if seed % 2 else None
    powers = frame(rng, zeros=int(rng.integers(0, 30)), ties=seed % 5 == 0, line=line)
    rank = (int(bulk.sum()) - 1) // 2
    multiplier = int(rng.integers(1, 8 * MULTIPLIER_ONE))
    kw = dict(anchor_bin=anchor, designated_half_width=half, bulk_mask=bulk, cfar_rank=rank)
    decision = fine_mask_decision_v2(powers, multiplier_q16=multiplier, **kw)
    required = fine_required_multiplier_q16_v2(powers, **kw)
    expected = oracle(powers, designated, bulk, rank, multiplier)
    if expected is None:
        assert not decision.valid and decision.mask == 0 and required.multiplier_q16 is None
        return
    assert decision.valid and decision.mask == expected[0]
    m = required.multiplier_q16
    assert fine_mask_decision_v2(powers, multiplier_q16=m, **kw).mask == 0
    if m > 1:
        assert fine_mask_decision_v2(powers, multiplier_q16=m - 1, **kw).mask == 1
    assert m == max(1, -(-(expected[1] * MULTIPLIER_ONE).numerator // (expected[1] * MULTIPLIER_ONE).denominator))
    # the packed mask and the sorted-rank scorer give the same boundary
    assert fine_required_multiplier_q16_v2(powers, **dict(kw, bulk_mask=pack_bulk_mask(bulk))) == required
    by_rank = required_multipliers_by_rank_v2(powers, designated=designated, bulk_mask=bulk)
    assert len(by_rank) == required.n_bulk and by_rank[rank] == m


def test_the_test_is_strict_at_an_exact_multiple():
    designated, bulk = geometry()
    powers = np.full((3, FINE_BINS), 500, dtype=np.uint64)      # den = 1000 in every bin
    powers[0, 139] = 1500                                        # num = 3000, three times the reference
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk, cfar_rank=47)
    assert fine_required_multiplier_q16_v2(powers, **kw).multiplier_q16 == 3 * MULTIPLIER_ONE
    assert fine_required_multiplier_q16_v2(powers, **kw).limiting_bin == 139
    assert fine_mask_decision_v2(powers, multiplier_q16=3 * MULTIPLIER_ONE, **kw).mask == 0
    fired = fine_mask_decision_v2(powers, multiplier_q16=3 * MULTIPLIER_ONE - 1, **kw)
    assert fired.mask == 1 and fired.fired_bin == 139


def test_the_rank_value_is_held_by_the_smallest_tied_bin():
    designated, bulk = geometry()
    powers = np.full((3, FINE_BINS), 500, dtype=np.uint64)
    decision = fine_mask_decision_v2(powers, anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk,
                                     cfar_rank=47, multiplier_q16=MULTIPLIER_ONE)
    assert decision.valid and decision.mask == 0 and decision.rank_bin == int(np.flatnonzero(bulk)[0])


def test_a_line_in_the_references_does_not_hide_the_pilot():
    # v1 divides the pilot bin by its own reference; v2 reads the median reference
    designated, bulk = geometry()
    powers = np.full((3, FINE_BINS), 10**6, dtype=np.uint64)
    powers[0, 139] = 10**7                                       # pilot, ten times the level
    powers[1:, 139] = 10**7                                      # a line in both references at the same bin
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk, cfar_rank=47,
              multiplier_q16=4 * MULTIPLIER_ONE)
    assert fine_mask_decision(powers, **kw).mask == 0
    assert fine_mask_decision_v2(powers, **kw).mask == 1


def test_too_few_background_bins_is_invalid():
    designated, bulk = geometry()
    powers = np.full((3, FINE_BINS), 500, dtype=np.uint64)
    powers[1:, np.flatnonzero(bulk)[40:]] = 0                    # 40 usable bins, rank 47 requested
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk, cfar_rank=47)
    decision = fine_mask_decision_v2(powers, multiplier_q16=MULTIPLIER_ONE, **kw)
    assert not decision.valid and decision.mask == 0 and decision.n_bulk == 40
    assert fine_required_multiplier_q16_v2(powers, **kw).multiplier_q16 is None


def test_a_boundary_past_uint64_is_always_masked():
    designated, bulk = geometry()
    powers = np.ones((3, FINE_BINS), dtype=np.uint64)            # den = 2
    powers[0, 139] = np.uint64(2**63)
    kw = dict(anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk, cfar_rank=47)
    assert fine_required_multiplier_q16_v2(powers, **kw).multiplier_q16 == ALWAYS_MASKED_Q16
    assert required_multipliers_by_rank_v2(powers, designated=designated, bulk_mask=bulk)[47] == ALWAYS_MASKED_Q16


def test_archive_scorer_gives_the_v2_boundary_and_score():
    from pilot_proxy.detectors.narrowband_marker.fine_histogram_frames import final_score_arrays

    rng = np.random.default_rng(3)
    designated, bulk = geometry()
    rank = (int(bulk.sum()) - 1) // 2
    powers = np.stack([frame(rng, zeros=int(rng.integers(0, 20)),
                             line=(int(rng.choice(designated)), int(rng.integers(1, 60) * 10**9)) if i % 2 else None)
                       for i in range(12)])
    out = final_score_arrays(powers, np.ones(len(powers), dtype=bool), anchor_bin=ANCHOR, bulk_mask=bulk,
                             designated_half_width=HALF, rho=rank + 1, decision_version=2)
    old = final_score_arrays(powers, np.ones(len(powers), dtype=bool), anchor_bin=ANCHOR, bulk_mask=bulk,
                             designated_half_width=HALF, rho=rank + 1)
    assert not np.array_equal(out["fine_required_multiplier_q16"], old["fine_required_multiplier_q16"])
    for i, f in enumerate(powers):
        ref = fine_required_multiplier_q16_v2(f, anchor_bin=ANCHOR, designated_half_width=HALF, bulk_mask=bulk,
                                              cfar_rank=rank)
        assert out["fine_rank_valid"][i] and int(out["fine_required_multiplier_q16"][i]) == ref.multiplier_q16
        peak = oracle(f, designated, bulk, rank, 1)[1]
        assert out[f"fine_score_z{rank + 1}"][i] == float(peak)
        assert ref.multiplier_q16 == -(-(peak * MULTIPLIER_ONE).numerator // (peak * MULTIPLIER_ONE).denominator)
