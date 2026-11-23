# coding=utf-8
"""Health-gate totals over the 23 per-pilot products of record.

Skipped unless PILOT_PROXY_ATTACHED_PRODUCTS names the folder that holds the
23 ``<freq_id>.npz`` products (the library's
``pilot_proxy_per_pilot_2026-08-19_complete23/_per_pilot``; the September 2026
rebuild has more frames and fails these totals).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.archive_health import (
    REASON_BASEBAND_CEILING,
    REASON_DETECTOR_INVALID,
    REASON_DETECTOR_POWERS_ALL_ZERO,
    evaluate_frame_health,
)


def test_attached_all23_health_totals() -> None:
    value = os.environ.get("PILOT_PROXY_ATTACHED_PRODUCTS")
    if not value:
        pytest.skip("set PILOT_PROXY_ATTACHED_PRODUCTS to run large-archive invariants")
    paths = sorted(Path(value).glob("*.npz"), key=lambda path: int(path.stem))
    assert len(paths) == 23
    totals: Counter[str] = Counter()
    stored = included = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as product:
            health = evaluate_frame_health(product)
        stored += int(health.include.size)
        included += int(np.count_nonzero(health.include))
        totals.update(health.reason_counts)
    assert stored == 750_461
    assert included == 750_279
    assert stored - included == 182
    assert totals[REASON_DETECTOR_INVALID] == 4
    assert totals[REASON_DETECTOR_POWERS_ALL_ZERO] == 4
    assert totals[REASON_BASEBAND_CEILING] == 178
