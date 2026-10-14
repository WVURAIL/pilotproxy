"""The exact keep boundaries per rank equal an independent rational oracle."""
from __future__ import annotations

from functools import cmp_to_key

import numpy as np

from pilot_proxy.detectors.narrowband_marker.scores import required_eta_q16_by_rank as required_multipliers_for_frame
from pilot_proxy.fine_decision import ALWAYS_MASKED_Q16, MAX_MULTIPLIER_Q16


def _exact_oracle(powers, *, anchor, width, bulk):
    designated = [(anchor + offset) % 256
                  for offset in range(-width, width + 1)]
    num = [2 * int(value) for value in powers[0]]
    den = [int(left) + int(right)
           for left, right in zip(powers[1], powers[2])]

    def compare(left, right):
        difference = num[left] * den[right] - num[right] * den[left]
        return (-1 if difference < 0 else 1 if difference > 0
                else left - right)

    ranked = [index for index in np.flatnonzero(bulk) if den[index] > 0]
    ranked.sort(key=cmp_to_key(compare))
    out = []
    for rank_bin in ranked:
        required = 1
        for designated_bin in designated:
            if den[designated_bin] <= 0 or num[designated_bin] == 0:
                continue
            if num[rank_bin] == 0:
                required = ALWAYS_MASKED_Q16
                break
            top = num[designated_bin] * (1 << 16) * den[rank_bin]
            bottom = num[rank_bin] * den[designated_bin]
            required = max(required, (top + bottom - 1) // bottom)
            if required > MAX_MULTIPLIER_Q16:
                required = ALWAYS_MASKED_Q16
                break
        out.append(required)
    return tuple(out)


def test_every_rank_matches_independent_exact_oracle():
    generator = np.random.default_rng(41)
    powers = generator.integers(
        0, 1 << 30, size=(3, 256), dtype=np.uint64)
    powers[1, 72] = 0
    powers[2, 72] = 0
    bulk = np.zeros(256, dtype=bool)
    bulk[np.arange(3, 220, 3)] = True
    bulk[[19, 20, 21]] = False
    expected = _exact_oracle(powers, anchor=20, width=1, bulk=bulk)

    actual = required_multipliers_for_frame(
        powers, anchor_bin=20, designated_half_width=1, bulk_mask=bulk)

    assert actual == expected
    assert len(actual) == np.count_nonzero(bulk) - 1


def test_randomized_exact_arithmetic_matches_oracle():
    generator = np.random.default_rng(731)
    cases = [(0, 3), (255, 4), (127, 0), (1, 127)]
    limit = np.iinfo(np.uint64).max
    for case in range(20):
        anchor, width = cases[case % len(cases)]
        powers = generator.integers(
            0, limit, size=(3, 256), dtype=np.uint64)
        designated = {(anchor + offset) % 256
                      for offset in range(-width, width + 1)}
        available = np.asarray(
            [index for index in range(256) if index not in designated])
        bulk = np.zeros(256, dtype=bool)
        bulk[available[generator.random(available.size) < 0.45]] = True
        if not bulk.any():
            bulk[int(available[0])] = True
        bulk_bins = np.flatnonzero(bulk)
        if bulk_bins.size > 1:
            powers[1, bulk_bins[0]] = 0
            powers[2, bulk_bins[0]] = 0
            powers[0, bulk_bins[1]] = 0
        if bulk_bins.size > 3:
            powers[:, bulk_bins[3]] = powers[:, bulk_bins[2]]
        powers[0, anchor] = limit
        powers[1, anchor] = 1
        powers[2, anchor] = 0

        expected = _exact_oracle(
            powers, anchor=anchor, width=width, bulk=bulk)
        actual = required_multipliers_for_frame(
            powers, anchor_bin=anchor, designated_half_width=width,
            bulk_mask=bulk)

        assert actual == expected
