"""The designated marker bins and the static reference bulk around an anchor."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import anchors
from pilot_proxy.products.reader import FINE_BINS


def test_unwrap_bins_is_signed_and_circular():
    assert anchors.unwrap_bins(2, 250) == 8 and anchors.unwrap_bins(250, 2) == -8
    np.testing.assert_array_equal(anchors.unwrap_bins([0, 127, 128, 255], 0), [0, 127, -128, -1])


def test_bulk_mask_parity_guard_and_census():
    even = anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1)
    odd = anchors.bulk_mask(65, pad_factor=2, guard_fine_bins=1)
    assert even.sum() == 125 and odd.sum() == 126
    for anchor_bin, mask in ((64, even), (65, odd)):
        assert not mask[list(anchors.designated_set(anchor_bin))].any()
        assert not mask[1::2].any()                       # alternate (even) bins only
    # a census-excluded independent bin leaves the bulk, and an odd one was never in it
    assert anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, census_excluded_bins=[10]).sum() == 124
    assert anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, census_excluded_bins=[11]).sum() == 125
    # a guard narrower than the lobe still leaves D out (the bundle refuses otherwise)
    narrow = anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=0)
    assert narrow.sum() == 125 and not narrow[list(anchors.designated_set(64))].any()
    # wrap at the axis edge
    assert anchors.designated_set(1) == (255, 0, 1, 2, 3)
    assert anchors.bulk_mask(0, pad_factor=2, guard_fine_bins=1).sum() == 125


def test_independent_bin_mask_matches_pilot_proxy():
    fine_reduction = pytest.importorskip("pilot_proxy.fine_reduction")
    rng = np.random.default_rng(3)
    for _ in range(20):
        designated = rng.integers(0, FINE_BINS, size=rng.integers(0, 4)).tolist()
        census = rng.integers(0, FINE_BINS, size=rng.integers(0, 6)).tolist()
        pad, guard = int(rng.choice([1, 2, 4])), int(rng.integers(0, 3))
        ours = anchors.independent_bin_mask(FINE_BINS, pad_factor=pad, designated_bins=designated,
                                            guard_fine_bins=guard, census_excluded_bins=census)
        theirs = fine_reduction.independent_bin_mask(FINE_BINS, pad_factor=pad, designated_bins=designated,
                                                     guard_fine_bins=guard, census_excluded_bins=census)
        np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("anchor", [64, 65])
def test_marker_guard_band_keeps_a_line_outside_the_markers_out_of_the_bulk(anchor):
    """A line 3 to 5 bins outside the markers (as on channels 22 and 28) enters the
    default bulk. A 5-bin guard band keeps it out, so the rank ignores it."""
    from pilot_proxy.detectors.narrowband_marker.scores import required_eta_q16_by_rank

    width = anchors.DESIGNATED_HALF_WIDTH
    line = [anchor + width + k for k in (3, 4, 5)]
    default = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1)
    guarded = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1, marker_guard_bins=5)
    assert default[line].any() and not guarded[line].any()
    # the guard band removes only the bins within h + g of the anchor
    removed = np.flatnonzero(default & ~guarded)
    assert all(width < abs(int(anchors.unwrap_bins(b, anchor))) <= width + 5 for b in removed)
    assert anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1, marker_guard_bins=0).tolist() == default.tolist()

    rng = np.random.default_rng(anchor)
    quiet = rng.integers(900, 1100, size=(3, 256)).astype(np.uint64)
    loud = quiet.copy()
    loud[0, line] = 40 * quiet[0, line]            # target-term line, references unchanged
    rank = {}
    for name, bulk in (("default", default), ("guarded", guarded)):
        rho = (int(bulk.sum()) + 1) // 2
        rank[name] = [required_eta_q16_by_rank(powers, anchor_bin=anchor, designated_half_width=width,
                                               bulk_mask=bulk)[rho - 1] for powers in (quiet, loud)]
    assert rank["guarded"][0] == rank["guarded"][1]   # the line cannot move the guarded reference
    assert rank["default"][0] != rank["default"][1]   # it moves the default one


def test_marker_guard_bins_must_be_a_nonnegative_integer():
    with pytest.raises(ValueError, match="marker_guard_bins"):
        anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, marker_guard_bins=-1)
