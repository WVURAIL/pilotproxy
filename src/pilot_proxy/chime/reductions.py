# coding=utf-8
"""Chunk-level reduced products for CHIME detector runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .products import (
    atomic_savez_compressed,
    mean_where,
    relative_time_seconds,
    valid_mask_counts,
)

CHIME_REDUCTIONS_10S_FILENAME = "chime_reductions_10s.npz"


def _finite_stat(values: np.ndarray, op: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for index in range(arr.shape[1]):
        finite = arr[:, index]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        if op == "median":
            out[index] = float(np.median(finite))
        elif op == "p95":
            out[index] = float(np.percentile(finite, 95))
        elif op == "max":
            out[index] = float(np.max(finite))
        else:
            raise ValueError(f"unknown finite statistic: {op}")
    return out


def aggregate_frame_products(
    *,
    frame_index: np.ndarray,
    frame_size_samples: int,
    sample_rate_hz: float,
    chunk_seconds: float,
    coarse_power_ratio: np.ndarray,
    normalized_coarse_power_ratio_db: np.ndarray,
    estimated_data_shelf_snr_db: np.ndarray,
    baseband_power_linear: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate frame-level products into fixed-duration chunks."""
    frames = np.asarray(frame_index, dtype=np.int64)
    if frames.ndim != 1 or frames.size == 0:
        raise ValueError("frame_index must be a non-empty 1D array")
    if chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be positive")

    arrays = [
        np.asarray(coarse_power_ratio),
        np.asarray(normalized_coarse_power_ratio_db),
        np.asarray(estimated_data_shelf_snr_db),
        np.asarray(baseband_power_linear),
        np.asarray(mask),
        np.asarray(valid),
    ]
    expected_shape = (frames.size, arrays[0].shape[1])
    for arr in arrays:
        if arr.shape != expected_shape:
            raise ValueError(
                "all frame products must have shape "
                f"{expected_shape}; got {arr.shape}"
            )

    relative_time_s = relative_time_seconds(
        frames,
        frame_size_samples=int(frame_size_samples),
        sample_rate_hz=float(sample_rate_hz),
    )
    chunk_ids = np.floor(
        (relative_time_s - float(relative_time_s[0])) / float(chunk_seconds)
    ).astype(np.int64)
    unique_chunks = np.unique(chunk_ids)
    num_chunks = int(unique_chunks.size)
    num_pilots = int(expected_shape[1])

    chunk_index = np.arange(num_chunks, dtype=np.int64)
    chunk_start_frame = np.zeros(num_chunks, dtype=np.int64)
    chunk_stop_frame = np.zeros(num_chunks, dtype=np.int64)
    input_power_mean = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    cleaned_power_mean = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    mask_fraction = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    unmasked_count = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    total_count = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    valid_count = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    invalid_count = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    masked_count_valid = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    unmasked_count_valid = np.zeros((num_chunks, num_pilots), dtype=np.int64)
    mask_fraction_valid = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    mask_fraction_total = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    normalized_coarse_power_ratio_db_median = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    normalized_coarse_power_ratio_db_p95 = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    normalized_coarse_power_ratio_db_max = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    estimated_data_shelf_snr_db_median = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    estimated_data_shelf_snr_db_p95 = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)
    estimated_data_shelf_snr_db_max = np.full((num_chunks, num_pilots), np.nan, dtype=np.float64)

    baseband = np.asarray(baseband_power_linear, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    valid_arr = np.asarray(valid, dtype=np.uint8) != 0
    fstat_level = np.asarray(normalized_coarse_power_ratio_db, dtype=np.float64)
    shelf = np.asarray(estimated_data_shelf_snr_db, dtype=np.float64)

    for out_index, chunk_id in enumerate(unique_chunks):
        rows = np.nonzero(chunk_ids == chunk_id)[0]
        chunk_start_frame[out_index] = int(frames[rows[0]])
        chunk_stop_frame[out_index] = int(frames[rows[-1]] + 1)
        baseband_chunk = baseband[rows, :]
        mask_chunk = mask_arr[rows, :]
        valid_chunk = valid_arr[rows, :]
        input_power_mean[out_index, :] = mean_where(baseband_chunk, valid_chunk)
        cleaned_power_mean[out_index, :] = mean_where(
            baseband_chunk,
            valid_chunk & (mask_chunk == 0),
        )
        counts = valid_mask_counts(mask_chunk, valid_chunk)
        mask_fraction[out_index, :] = counts["mask_fraction_valid"]
        unmasked_count[out_index, :] = counts["unmasked_count_valid"]
        total_count[out_index, :] = counts["total_count"]
        valid_count[out_index, :] = counts["valid_count"]
        invalid_count[out_index, :] = counts["invalid_count"]
        masked_count_valid[out_index, :] = counts["masked_count_valid"]
        unmasked_count_valid[out_index, :] = counts["unmasked_count_valid"]
        mask_fraction_valid[out_index, :] = counts["mask_fraction_valid"]
        mask_fraction_total[out_index, :] = counts["mask_fraction_total"]

        fstat_chunk = np.where(valid_chunk, fstat_level[rows, :], np.nan)
        shelf_chunk = np.where(valid_chunk, shelf[rows, :], np.nan)
        normalized_coarse_power_ratio_db_median[out_index, :] = _finite_stat(fstat_chunk, "median")
        normalized_coarse_power_ratio_db_p95[out_index, :] = _finite_stat(fstat_chunk, "p95")
        normalized_coarse_power_ratio_db_max[out_index, :] = _finite_stat(fstat_chunk, "max")
        estimated_data_shelf_snr_db_median[out_index, :] = _finite_stat(shelf_chunk, "median")
        estimated_data_shelf_snr_db_p95[out_index, :] = _finite_stat(shelf_chunk, "p95")
        estimated_data_shelf_snr_db_max[out_index, :] = _finite_stat(shelf_chunk, "max")

    return {
        "chunk_index": chunk_index,
        "chunk_start_frame": chunk_start_frame,
        "chunk_stop_frame": chunk_stop_frame,
        "input_power_mean": input_power_mean,
        "cleaned_power_mean": cleaned_power_mean,
        "mask_fraction": mask_fraction,
        "unmasked_count": unmasked_count,
        "total_count": total_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "masked_count_valid": masked_count_valid,
        "unmasked_count_valid": unmasked_count_valid,
        "mask_fraction_valid": mask_fraction_valid,
        "mask_fraction_total": mask_fraction_total,
        "normalized_coarse_power_ratio_db_median": normalized_coarse_power_ratio_db_median,
        "normalized_coarse_power_ratio_db_p95": normalized_coarse_power_ratio_db_p95,
        "normalized_coarse_power_ratio_db_max": normalized_coarse_power_ratio_db_max,
        "estimated_data_shelf_snr_db_median": estimated_data_shelf_snr_db_median,
        "estimated_data_shelf_snr_db_p95": estimated_data_shelf_snr_db_p95,
        "estimated_data_shelf_snr_db_max": estimated_data_shelf_snr_db_max,
    }


def write_reductions_npz(
    run_dir: Path,
    *,
    frame_index: np.ndarray,
    frame_size_samples: int,
    sample_rate_hz: float,
    chunk_seconds: float = 10.0,
    coarse_power_ratio: np.ndarray,
    normalized_coarse_power_ratio_db: np.ndarray,
    estimated_data_shelf_snr_db: np.ndarray,
    baseband_power_linear: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> Path:
    path = Path(run_dir) / CHIME_REDUCTIONS_10S_FILENAME
    products = aggregate_frame_products(
        frame_index=frame_index,
        frame_size_samples=int(frame_size_samples),
        sample_rate_hz=float(sample_rate_hz),
        chunk_seconds=float(chunk_seconds),
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    return atomic_savez_compressed(path, **products)


__all__ = [
    "CHIME_REDUCTIONS_10S_FILENAME",
    "aggregate_frame_products",
    "write_reductions_npz",
]
