"""The pilot's designated marker bins and the reference bulk around a measured anchor.

ch06 ``sec:detection:rule``: "Let f_a be the measured pilot anchor. The
designated set spans the padded main lobe, D = {(f_a + k) mod L_F : |k| <= 2}.
The usable null bulk B contains alternate bins from the independent subset
..., excludes D and any census-marked persistent features, and omits bins with
zero reference denominator. An even-parity anchor leaves 125 bulk bins and an
odd-parity anchor leaves 126." The zero-denominator omission is made per
frame by the frame scorer (:mod:`.scores`). The bulk mask in this module is
the static part of B.

The bulk is ``B = independent_bin_mask(256, pad_factor, designated_bins=(f_a,),
guard_fine_bins, census_excluded_bins)`` with pilot-proxy's semantics. That
is, B is every ``pad_factor``-th bin, minus the ``f_a +- guard * pad`` padded
bins and the census-excluded bins. D is removed as well, so the bundle's
contract holds for any guard. With the campaign's pad factor of 2 and guard
of 1, the guard removes exactly D, giving 125 bulk bins for an even anchor
and 126 for an odd one.

Two calibration options of detector v2 change the marker geometry. With
``marker_guard_bins = g``, the ``g`` padded bins on each side of D are also
removed from the bulk. These are ``{(f_a + k) mod L_F : h < |k| <= h + g}``
for half-width ``h``, so a line just past the markers cannot enter the
reference rank (Paper I finds such lines on channels 22 and 28).
``designated_half_width`` widens the markers themselves. The defaults
(``g = 0``, ``h = 2``) are the calibration of record. A nonzero guard is
recorded in the bundle as ``fine_calibration.marker_guard_bins``.

``WINDOW_HALF_WIDTH`` (30 bins) is the scan's acquisition window about the
nominal fine bin.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from pilot_proxy.products.reader import FINE_BINS

WINDOW_HALF_WIDTH = 30
DESIGNATED_HALF_WIDTH = 2


def unwrap_bins(bin_index, reference_bin: int, fine_bins: int = FINE_BINS):
    """Signed offset of padded bins from a reference bin, unwrapped to [-L/2, L/2)."""
    b = np.asarray(bin_index)
    return (b - int(reference_bin) + fine_bins // 2) % fine_bins - fine_bins // 2


def designated_set(anchor_bin: int, half_width: int = DESIGNATED_HALF_WIDTH,
                   fine_bins: int = FINE_BINS) -> tuple[int, ...]:
    """``D = {(f_a + k) mod L_F : |k| <= half_width}`` in k order."""
    return tuple(int((int(anchor_bin) + k) % fine_bins) for k in range(-int(half_width), int(half_width) + 1))


def independent_bin_mask(num_bins: int, *, pad_factor: int, designated_bins: Sequence[int] = (),
                         guard_fine_bins: int, census_excluded_bins: Sequence[int] = ()) -> np.ndarray:
    """Return the same mask as ``pilot_proxy.fine_reduction.independent_bin_mask``.

    The mask is every ``pad_factor``-th bin, minus each designated bin with
    ``guard_fine_bins * pad_factor`` padded bins on either side, and minus the
    census-excluded bins.
    """
    n = int(num_bins)
    mask = np.zeros(n, dtype=bool)
    mask[:: int(pad_factor)] = True
    guard = int(guard_fine_bins) * int(pad_factor)
    for b in designated_bins:
        mask[np.arange(int(b) - guard, int(b) + guard + 1) % n] = False
    for b in census_excluded_bins:
        mask[int(b) % n] = False
    return mask


def bulk_mask(anchor_bin: int, *, pad_factor: int, guard_fine_bins: int, census_excluded_bins: Sequence[int] = (),
              designated_half_width: int = DESIGNATED_HALF_WIDTH, marker_guard_bins: int = 0,
              fine_bins: int = FINE_BINS) -> np.ndarray:
    """The static bulk ``B`` for a measured anchor, minus any marker guard band (module docstring)."""
    guard = int(marker_guard_bins)
    if guard != marker_guard_bins or guard < 0 or int(designated_half_width) + guard >= fine_bins // 2:
        raise ValueError("marker_guard_bins must be a nonnegative integer with half width + guard below L_F / 2")
    mask = independent_bin_mask(fine_bins, pad_factor=pad_factor, designated_bins=(int(anchor_bin),),
                                guard_fine_bins=guard_fine_bins, census_excluded_bins=census_excluded_bins)
    mask[list(designated_set(anchor_bin, int(designated_half_width) + guard, fine_bins))] = False
    return mask
