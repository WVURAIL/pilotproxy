"""A CPU stand-in for the detector backend, so archive scans run without a GPU (plumbing only; no parity claim)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

K = 128


def stub_detector_fn(*, packed, weights, kernel):
    """Trivial CPU detector: valid-schema per-block sums (plumbing only; no parity claim)."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    n = int(pk.shape[0])
    return {
        "batch": n,
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "results": [
            {
                "block_index": b,
                "mask": 0,
                "p_target_u64": 10,
                "p_ref_lower_u64": 10,
                "p_ref_upper_u64": 10,
                "p_ref_sum_u64": 20,
            }
            for b in range(n)
        ],
    }


def stub_kernel(k):
    specs = SimpleNamespace(
        K=k, N=3, bits=4, reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": k, "num_weight_terms": 3,
            "sample_bits_per_component": 4, "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def cpu_detector_options():
    """CPU detector injection so scan plumbing runs GPU-free."""
    rng = np.random.default_rng(0)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8)
        for ch in range(10, 41)
    }
    return {
        "detector_fn": stub_detector_fn,
        "kernel": stub_kernel(K),
        "weights_by_channel": weights_by_channel,
    }
