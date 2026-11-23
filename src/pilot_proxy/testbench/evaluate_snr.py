# coding=utf-8
"""Ideal detector weights from a weight layout, kept for tools/validate_fine_detector.py."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TWO_PI = 2.0 * math.pi


def _ideal_float_weights_from_layout(
    selected_weight_layout: dict[str, Any],
    *,
    detector_window_samples: int,
) -> np.ndarray:
    """Build unquantized complex DFT weights from manifest target frequencies."""
    keys = [
        "target_normalized_frequency",
        "lower_reference_normalized_frequency",
        "upper_reference_normalized_frequency",
    ]
    if not all(key in selected_weight_layout for key in keys):
        raise ValueError("selected weight layout lacks normalized frequencies.")
    sample_index = np.arange(int(detector_window_samples), dtype=np.float64)
    rows = [
        np.exp(-1j * TWO_PI * float(selected_weight_layout[key]) * sample_index)
        for key in keys
    ]
    return np.ascontiguousarray(np.stack(rows).astype(np.complex128))
