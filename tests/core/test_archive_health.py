# coding=utf-8
"""The v2 frame-health gate (pilot_proxy.archive_health.evaluate_frame_health)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.archived_product_keys import (
    ARCHIVED_REFERENCE_NORM_SUM_SQ,
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_DATA_SHELF_SNR_DB,
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB,
    ARCHIVED_PILOT_EXCESS_DB,
)
from pilot_proxy.archive_health import (
    ENCODING_INTERPRETATION,
    FRAME_HEALTH_GATE_SCHEMA_VERSION,
    REASON_BASEBAND_CEILING,
    REASON_BASEBAND_NONFINITE,
    REASON_BASEBAND_OUT_OF_BOUNDS,
    REASON_DETECTOR_INVALID,
    REASON_DETECTOR_POWERS_ALL_ZERO,
    REASON_FILL_ABOVE_ONE_PERCENT,
    REASON_FINE_NEGATIVE,
    ArchiveHealthError,
    evaluate_frame_health,
)
from pilot_proxy.products.reader import HEALTH_GATE_SCHEMA


NFFT = 16_384
K = 128
FINE_BINS = 256
SAMPLE_RATE_HZ = 390_625.0


def _product(
    *,
    valid: list[int],
    reject: list[int],
    baseband_power: list[float],
    p_target: list[int] | None = None,
    p_reference: list[int] | None = None,
    fine: np.ndarray | None = None,
    num_streams: int = 2,
) -> dict[str, np.ndarray]:
    n = len(valid)
    if not (len(reject) == len(baseband_power) == n):
        raise ValueError("fixture frame columns must align")
    target = np.asarray(p_target or [10] * n, dtype=np.uint64)
    reference = np.asarray(p_reference or [20] * n, dtype=np.uint64)
    if fine is None:
        rng = np.random.default_rng(8)
        fine = rng.normal(1.0, 0.04, size=(n, FINE_BINS)).astype(np.float32)
    frame_unit = np.arange(n, dtype=np.int32)
    power_ratio = 2.0 * target.astype(float) / np.where(reference > 0, reference, 1)
    power_ratio = np.where(reference > 0, power_ratio, 0.0)
    return {
        "schema_version": np.asarray("pilotproxy_detector_datatrawl_v3"),
        "physical_channel": np.asarray([17], dtype=np.int32),
        "freq_id": np.asarray([798], dtype=np.int64),
        "pilot_frequency_hz": np.asarray([488_309_440.559], dtype=np.float64),
        "chime_frequency_hz": np.asarray([488_281_250.0], dtype=np.float64),
        "frame_index": np.arange(n, dtype=np.int64),
        "p_target_u64": target[:, None],
        "p_ref_sum_u64": reference[:, None],
        ARCHIVED_COARSE_POWER_RATIO: power_ratio[:, None],
        ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB: (
            10.0 * np.log10(np.maximum(power_ratio, 1.0e-12))
        )[:, None],
        ARCHIVED_PILOT_EXCESS_DB: np.full((n, 1), -10.0),
        ARCHIVED_DATA_SHELF_SNR_DB: np.full((n, 1), -31.0),
        "reject_mask": np.asarray(reject, dtype=np.uint8)[:, None],
        "valid": np.asarray(valid, dtype=np.uint8)[:, None],
        "baseband_power_linear": np.asarray(baseband_power, dtype=np.float64)[:, None],
        ARCHIVED_FINE_POWER_RATIO: np.asarray(fine, dtype=np.float32),
        "fine_pad_factor": np.asarray(2, dtype=np.int64),
        "fine_num_bins": np.asarray(FINE_BINS, dtype=np.int64),
        "fine_p_fa": np.asarray(1.0e-3),
        "fine_guard_fine_bins": np.asarray(1, dtype=np.int64),
        "fine_designated_bins": np.asarray([0], dtype=np.int64),
        "fine_census_excluded_bins": np.asarray([], dtype=np.int64),
        "frame_unit_index": frame_unit,
        "frame_in_unit": np.zeros(n, dtype=np.int32),
        "source_event_keys": np.asarray([f"baseband_{100+i}.h5" for i in range(n)]),
        "unit_order": np.asarray([f"cadc:test/baseband_{100+i}_798.h5" for i in range(n)]),
        "unit_event_id": np.arange(100, 100 + n, dtype=np.int64),
        "unit_time0_ctime": 1_700_000_000.0 + np.arange(n) * 100.0,
        "unit_delta_time": np.full(n, 1.0 / SAMPLE_RATE_HZ),
        "nfft": np.asarray(NFFT, dtype=np.int64),
        "sample_rate_hz": np.asarray(SAMPLE_RATE_HZ),
        "detector_window_samples": np.asarray(K, dtype=np.int64),
        "num_input_streams": np.asarray(num_streams, dtype=np.int64),
        "sense": np.asarray(-1, dtype=np.int64),
        "mu0": np.asarray([1.0]),
        # mu_0 is computed from these two norms. The mu0 entry above is the
        # archived float, which is read through the migration map.
        "target_norm_sq": np.asarray([1], dtype=np.int64),
        ARCHIVED_REFERENCE_NORM_SUM_SQ: np.asarray([2], dtype=np.int64),
        "weight_bank_sha256": np.asarray("a" * 64),
        "weight_manifest_sha256": np.asarray("b" * 64),
        "weights_hash": np.asarray("c" * 64),
        "detector_version": np.asarray(
            "pilot-proxy/1.0.0 source=" + "d" * 64 + " kernel=2.1.0 "
            "kernel_sha256=" + "e" * 64 + " pilotproxy_detector_datatrawl_v3 K=128"
        ),
        "detector_contract_json": np.asarray(
            json.dumps({"equivalent_mask_rule": "F > mu0"})
        ),
        "mask_rule": np.asarray("F > mu0"),
        "pilot_below_data_db": np.asarray(-16.0),
        "dtv_bandwidth_hz": np.asarray(5.38e6),
        "bin_enbw_hz": np.asarray(390_625.0 / K),
        "integrated_spectrum_before_mask": np.zeros(NFFT, dtype=np.float64),
        "integrated_spectrum_after_mask": np.zeros(NFFT, dtype=np.float64),
    }


def _save_product(path: Path, product: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **product)
    return path


def test_health_gate_is_fail_closed_and_uses_correct_encoding_language() -> None:
    fine = np.ones((6, FINE_BINS), dtype=np.float32)
    fine[5, 7] = -0.1
    product = _product(
        valid=[1, 0, 1, 1, 1, 1],
        reject=[0, 0, 0, 0, 0, 0],
        baseband_power=[4.0, 0.0, 128.0, np.nan, 129.0, 4.0],
        p_target=[10, 0, 10, 10, 10, 10],
        p_reference=[20, 0, 20, 20, 20, 20],
        fine=fine,
    )
    health = evaluate_frame_health(product)
    assert health.include.tolist() == [True, False, False, False, False, False]
    assert health.reason_counts == {
        REASON_DETECTOR_INVALID: 1,
        REASON_DETECTOR_POWERS_ALL_ZERO: 1,
        REASON_BASEBAND_NONFINITE: 1,
        REASON_BASEBAND_OUT_OF_BOUNDS: 1,
        REASON_BASEBAND_CEILING: 1,
        REASON_FINE_NEGATIVE: 1,
    }
    assert "native raw byte is 0x00" in ENCODING_INTERPRETATION
    assert "two's-complement 0x88" in ENCODING_INTERPRETATION


def test_health_gate_refuses_missing_or_misaligned_identity() -> None:
    product = _product(valid=[1], reject=[0], baseband_power=[4.0])
    del product["unit_order"]
    with pytest.raises(ArchiveHealthError, match="unit_order"):
        evaluate_frame_health(product)


def _current_form(product: dict[str, np.ndarray], fine_sums: np.ndarray) -> dict[str, np.ndarray]:
    """Return the fixture with exact fine powers whose all-bin sums are ``fine_sums``.

    The exact fine powers are stored as in the per-pilot v5 and v6 products.
    ``fine_sums`` has shape (frames, 3) for [target, lower, upper]. The stored
    coarse columns are left unchanged, as in a v5 product of record. Only a
    ceiling frame is filled.
    """
    product = dict(product)
    product.pop(ARCHIVED_FINE_POWER_RATIO)
    n = len(product["frame_index"])
    total = 2 * NFFT * int(product["num_input_streams"])
    ceiling = product["baseband_power_linear"] == 128.0
    product["fill_sample_count"] = np.where(ceiling, total, 0).astype(np.uint64)
    product["railed_sample_total"] = np.full((n, 1), total, dtype=np.uint64)
    fine = np.full((n, 3, FINE_BINS), 1, dtype=np.uint64)
    fine[:, :, 0] = np.asarray(fine_sums, dtype=np.uint64) - np.uint64(FINE_BINS - 1)
    product["fine_power_u64"] = fine
    product["pilot_capture_efficiency"] = np.asarray(1.0)
    return product


def test_gate_reads_the_fine_sums(tmp_path) -> None:
    # The stored (row-domain) powers give F = 1 on every frame and mark frame 1
    # invalid. The fine powers sum to F = 0.5, 0.8, 0.5, 0.8 with every frame
    # valid. The gate must follow the fine sums.
    product = _product(
        valid=[1, 0, 1, 1],
        reject=[0, 0, 0, 0],
        baseband_power=[4.0, 4.0, 4.0, 128.0],
        p_target=[10, 0, 10, 10],
        p_reference=[20, 0, 20, 20],
    )
    sums = np.asarray(
        [[500, 1000, 1000], [800, 1000, 1000], [500, 1000, 1000], [800, 1000, 1000]]
    )
    product = _current_form(product, sums)
    path = _save_product(tmp_path / "798.npz", product)

    with np.load(path, allow_pickle=False) as archive:
        health = evaluate_frame_health(archive)
    # only the ceiling frame is excluded, since frame 1 is valid by its fine sums
    assert health.include.tolist() == [True, True, True, False]
    assert "detector_invalid" not in health.reason_counts


def test_gate_excludes_frames_more_than_one_percent_filled() -> None:
    product = _current_form(
        _product(valid=[1] * 5, reject=[0] * 5,
                 baseband_power=[4.0, 4.0, 4.0, 4.0, 128.0]),
        np.asarray([[500, 1000, 1000]] * 5),
    )
    total = int(product["railed_sample_total"][0, 0])  # 65536
    # none, the largest count at or below 1%, the smallest above it, one
    # 1/32 step, and the whole frame (also a ceiling frame)
    fill = [0, total // 100, total // 100 + 1, total // 32, total]
    product["fill_sample_count"] = np.asarray(fill, dtype=np.uint64)[:, None]
    health = evaluate_frame_health(product)
    assert health.include.tolist() == [True, True, False, False, False]
    assert health.reason_counts == {
        REASON_FILL_ABOVE_ONE_PERCENT: 3,
        REASON_BASEBAND_CEILING: 1,
    }

    # an archived product stores no fill count and cannot trigger the rule;
    # a current product without it is refused
    assert REASON_FILL_ABOVE_ONE_PERCENT not in evaluate_frame_health(
        _product(valid=[1], reject=[0], baseband_power=[4.0])
    ).reason_counts
    del product["fill_sample_count"]
    with pytest.raises(ArchiveHealthError, match="fill_sample_count"):
        evaluate_frame_health(product)


def test_reader_names_the_gate_schema_it_applies() -> None:
    assert HEALTH_GATE_SCHEMA == FRAME_HEALTH_GATE_SCHEMA_VERSION
    assert FRAME_HEALTH_GATE_SCHEMA_VERSION.endswith("_v2")
