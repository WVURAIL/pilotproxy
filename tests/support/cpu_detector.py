"""A CPU stand-in for the detector backend, so archive scans run without a GPU (plumbing only; no parity claim)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

K = 128


# The stub's fixed fine spectra: all power in bin 0, so the frame powers (their
# all-bin sums, per-pilot product schema v6) are 10, 10 and 10.
STUB_FINE_BINS = 256
STUB_FRAME_POWERS = (10, 10, 10)


def stub_fine_powers(n: int) -> np.ndarray:
    fine = np.zeros((n, 3, STUB_FINE_BINS), dtype=np.uint64)
    fine[:, :, 0] = np.asarray(STUB_FRAME_POWERS, dtype=np.uint64)
    return fine


def stub_detector_fn(*, packed, weights, kernel, emit_row_projections=False):
    """Trivial CPU detector: fixed fine spectra and their sums (plumbing only; no parity claim)."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    n = int(pk.shape[0])
    target, lower, upper = STUB_FRAME_POWERS
    return {
        "batch": n,
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "fine_powers_u64": stub_fine_powers(n),
        "results": [
            {
                "block_index": b,
                "mask": 0,
                "p_target_u64": target,
                "p_ref_lower_u64": lower,
                "p_ref_upper_u64": upper,
                "p_ref_sum_u64": lower + upper,
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


def exact_cpu_detector_fn(*, packed, weights, kernel=None, emit_row_projections=False):
    """The kernel core 3.0.0 chain on the CPU, exactly.

    Packed rows -> exact int32 row projections -> frozen fxfft256 fine powers
    (summed over streams) -> frame powers ``P_r = sum_f S_r[f]``. Returns the
    keys ``detect_packed_detector_input`` returns that the archive analyzer
    reads, so an injected backend produces bit-for-bit the kernel's numbers.
    """
    from pilot_proxy.detector_contract import (
        normalized_positive_excess,
        weight_term_norms_sq,
    )
    from pilot_proxy.detector_reference import (
        matched_filter_row_projections_cpu_reference_packed,
    )
    from pilot_proxy.fxfft import fine_power_fx

    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    w = np.asarray(weights, dtype=np.int8)
    nt, nl, nu = weight_term_norms_sq(w)
    nrs = int(nl + nu)
    batch, rows = int(pk.shape[0]), int(pk.shape[1])
    if rows % K:
        raise ValueError("rows must be whole 128-window streams")
    projections, fine, results = [], [], []
    for b in range(batch):
        rs = matched_filter_row_projections_cpu_reference_packed(pk[b], w, 4)
        spectrum = fine_power_fx(rs.astype(np.int64), num_streams=rows // K)
        powers = [int(sum(int(v) for v in spectrum[t])) for t in range(3)]
        target, lower, upper = powers
        projections.append(rs)
        fine.append(spectrum)
        results.append(
            {
                "block_index": b,
                "mask": normalized_positive_excess(
                    target, lower + upper, target_norm_sq=nt, reference_norm_sum_sq=nrs
                ),
                "p_target_u64": target,
                "p_ref_lower_u64": lower,
                "p_ref_upper_u64": upper,
                "p_ref_sum_u64": lower + upper,
            }
        )
    out = {
        "batch": batch,
        "detector_rows_per_block": rows,
        "rational_overflow_count": 0,
        "fine_powers_u64": np.stack(fine).astype(np.uint64),
        "results": results,
    }
    if emit_row_projections:
        out["matched_filter_row_projections"] = np.stack(projections)
    return out
