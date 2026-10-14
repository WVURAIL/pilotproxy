# coding=utf-8
from __future__ import annotations

import numpy as np

from pilot_proxy.chime.products import (
    write_detector_outputs,
    write_spectrogram_cache,
)


def test_missing_coarse_frequency_is_not_replaced_by_pilot_frequency(tmp_path) -> None:
    pilot = np.asarray([470_309_440.559], dtype=np.float64)
    outputs = write_detector_outputs(
        tmp_path,
        physical_channel=np.asarray([14]),
        pilot_frequency_hz=pilot,
        frame_index=np.asarray([0]),
        p_target_u64=np.asarray([[1]], dtype=np.uint64),
        p_ref_sum_u64=np.asarray([[2]], dtype=np.uint64),
        coarse_power_ratio=np.asarray([[1.0]]),
        normalized_coarse_power_ratio_db=np.asarray([[0.0]]),
        pilot_excess_db=np.asarray([[np.nan]]),
        estimated_data_shelf_snr_db=np.asarray([[np.nan]]),
        mask=np.asarray([[0]], dtype=np.uint8),
        valid=np.asarray([[1]], dtype=np.uint8),
        target_norm_sq=np.asarray([1], dtype=np.int64),
        reference_norm_sum_sq=np.asarray([2], dtype=np.int64),
        null_power_ratio=np.asarray([1.0]),
        normalized_pilot_excess=np.asarray([[0.0]]),
    )
    with np.load(outputs) as product:
        assert np.isnan(product["chime_frequency_hz"][0])
        assert product["pilot_frequency_hz"][0] == pilot[0]
        assert {
            "target_norm_sq",
            "reference_norm_sum_sq",
            "null_power_ratio",
            "normalized_pilot_excess",
        }.issubset(product.files)


def test_spectrogram_preserves_supplied_coarse_frequency(tmp_path) -> None:
    pilot = np.asarray([470_309_440.559], dtype=np.float64)
    centre = np.asarray([470_312_500.0], dtype=np.float64)
    cache = write_spectrogram_cache(
        tmp_path,
        baseband_power_linear=np.asarray([[1.0]]),
        mask=np.asarray([[0]], dtype=np.uint8),
        physical_channel=np.asarray([14]),
        pilot_frequency_hz=pilot,
        chime_frequency_hz=centre,
        frame_index=np.asarray([0]),
        frame_size_samples=16384,
        sample_rate_hz=390_625.0,
        valid=np.asarray([[1]], dtype=np.uint8),
    )
    with np.load(cache) as product:
        np.testing.assert_array_equal(product["chime_frequency_hz"], centre)
        assert product["chime_frequency_hz"][0] != product["pilot_frequency_hz"][0]


def test_frequency_metadata_shape_must_match_pilot_axis(tmp_path) -> None:
    from pilot_proxy.chime.products import _optional_frequency_array

    with np.testing.assert_raises(ValueError):
        _optional_frequency_array([1.0, 2.0], shape_like=[1.0])
