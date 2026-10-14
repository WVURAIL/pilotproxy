# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.reductions import aggregate_frame_products


def test_aggregate_frame_products_writes_one_chunk_for_10s_local_shape() -> None:
    frame_index = np.arange(4, dtype=np.int64)
    baseband = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
            [40.0, 400.0],
        ],
        dtype=np.float64,
    )
    mask = np.asarray([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    fstat_level = np.asarray(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        dtype=np.float64,
    )
    products = aggregate_frame_products(
        frame_index=frame_index,
        frame_size_samples=16_384,
        sample_rate_hz=390_625.0,
        chunk_seconds=10.0,
        coarse_power_ratio=np.ones((4, 2)),
        normalized_coarse_power_ratio_db=fstat_level,
        estimated_data_shelf_snr_db=np.asarray(fstat_level - 30.0, dtype=np.float64),
        baseband_power_linear=baseband,
        mask=mask,
        valid=np.ones((4, 2), dtype=np.uint8),
    )

    assert products["chunk_index"].tolist() == [0]
    assert products["chunk_start_frame"].tolist() == [0]
    assert products["chunk_stop_frame"].tolist() == [4]
    np.testing.assert_allclose(products["input_power_mean"], [[25.0, 250.0]])
    np.testing.assert_allclose(
        products["cleaned_power_mean"], [[80.0 / 3.0, 700.0 / 3.0]]
    )
    np.testing.assert_allclose(products["mask_fraction"], [[0.25, 0.25]])
    assert products["unmasked_count"].tolist() == [[3, 3]]
    assert products["total_count"].tolist() == [[4, 4]]
    np.testing.assert_allclose(products["normalized_coarse_power_ratio_db_p95"], [[2.85, 3.85]])


def test_aggregate_frame_products_excludes_invalid_frames_from_power_denominators() -> (
    None
):
    frame_index = np.arange(4, dtype=np.int64)
    baseband = np.asarray([[10.0], [1000.0], [30.0], [40.0]], dtype=np.float64)
    mask = np.asarray([[0], [1], [0], [0]], dtype=np.uint8)
    valid = np.asarray([[1], [0], [1], [0]], dtype=np.uint8)

    products = aggregate_frame_products(
        frame_index=frame_index,
        frame_size_samples=16_384,
        sample_rate_hz=390_625.0,
        chunk_seconds=10.0,
        coarse_power_ratio=np.ones((4, 1)),
        normalized_coarse_power_ratio_db=np.asarray([[0.0], [10.0], [20.0], [30.0]]),
        estimated_data_shelf_snr_db=np.asarray([[-30.0], [-20.0], [-10.0], [0.0]]),
        baseband_power_linear=baseband,
        mask=mask,
        valid=valid,
    )

    np.testing.assert_allclose(products["input_power_mean"], [[20.0]])
    np.testing.assert_allclose(products["cleaned_power_mean"], [[20.0]])
    assert products["valid_count"].tolist() == [[2]]
    assert products["invalid_count"].tolist() == [[2]]
    assert products["masked_count_valid"].tolist() == [[0]]
    assert products["unmasked_count_valid"].tolist() == [[2]]
    np.testing.assert_allclose(products["mask_fraction_valid"], [[0.0]])
    np.testing.assert_allclose(products["mask_fraction_total"], [[0.25]])
    np.testing.assert_allclose(products["normalized_coarse_power_ratio_db_median"], [[10.0]])
