# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


from pilot_proxy.archive.combine import (
    _align_frames,
    _common_sample_rate_hz,
    _detector_contract_from,
    _load_sorted,
)
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_detector_contract,
)


@pytest.fixture()
def current_detector_contract() -> dict[str, object]:
    """Return the complete current contract emitted by production code."""
    return build_detector_contract(
        detector_window_samples=128,
        skipped_guard_bins=1,
        reference_offset_bins=2,
        num_weight_terms=3,
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        time_reverse_detector_windows_before_kernel=True,
    )


def _product(unit_index, frame_in_unit, *, delta=1.0 / 390_625.0):
    n = len(unit_index)
    return {
        "frame_index": np.arange(n, dtype=np.int64),
        "source_event_keys": np.asarray(["event-a", "event-b"]),
        "frame_unit_index": np.asarray(unit_index, dtype=np.int32),
        "frame_in_unit": np.asarray(frame_in_unit, dtype=np.int32),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "unit_delta_time": np.asarray([delta, delta], dtype=np.float64),
    }


def test_event_keyed_alignment_rejects_duplicate_identity():
    first = _product([0, 0], [0, 1])
    second = _product([0, 0], [0, 0])
    second["physical_channel"] = np.asarray([15], dtype=np.int32)
    second["freq_id"] = np.asarray([829], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        _align_frames([first, second])


def test_event_keyed_alignment_requires_identity_fields():
    first = _product([0, 1], [0, 0])
    second = _product([0, 1], [0, 0])
    del second["source_event_keys"]
    with pytest.raises(ValueError, match="missing frame identity arrays"):
        _align_frames([first, second])


def test_combine_refuses_an_unstamped_product(tmp_path: Path):
    path = tmp_path / "844.npz"
    np.savez_compressed(
        path,
        physical_channel=np.asarray([14], dtype=np.int32),
        frame_index=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="missing required field 'schema_name'"):
        _load_sorted([path])


def test_common_sample_rate_rejects_mixed_delta_time():
    first = _product([0, 1], [0, 0])
    second = _product([0, 1], [0, 0], delta=1.01 / 390_625.0)
    with pytest.raises(ValueError, match="unit_delta_time"):
        _common_sample_rate_hz([first, second])


def test_common_sample_rate_accepts_consistent_timing():
    first = _product([0, 1], [0, 0])
    second = _product([0, 1], [0, 0])
    assert _common_sample_rate_hz([first, second]) == pytest.approx(390_625.0)


def test_common_sample_rate_uses_recorded_rate_when_absolute_timing_is_missing():
    first = _product([0, 1], [0, 0], delta=float("nan"))
    second = _product([0, 1], [0, 0], delta=float("nan"))
    first["sample_rate_hz"] = np.asarray(195_312.5)
    second["sample_rate_hz"] = np.asarray(195_312.5)
    assert _common_sample_rate_hz([first, second]) == pytest.approx(195_312.5)


def test_combine_accepts_current_detector_contract(
    current_detector_contract: dict[str, object],
) -> None:
    product = {
        "detector_contract_json": np.asarray(
            json.dumps(current_detector_contract, sort_keys=True)
        )
    }

    assert _detector_contract_from([product]) == current_detector_contract


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda contract: contract.pop("threshold_mode"), "missing required fields"),
        (
            lambda contract: contract.__setitem__("power_accumulator", "float32"),
            "power_accumulator",
        ),
        (
            lambda contract: contract.__setitem__("historical_alias", True),
            "unknown fields",
        ),
    ],
    ids=("missing-field", "wrong-locked-value", "retired-alias"),
)
def test_combine_rejects_noncurrent_detector_contract(
    current_detector_contract: dict[str, object],
    mutate,
    message: str,
) -> None:
    mutate(current_detector_contract)
    product = {
        "detector_contract_json": np.asarray(
            json.dumps(current_detector_contract, sort_keys=True)
        )
    }

    with pytest.raises(
        ValueError,
        match="detector_contract_json does not satisfy the current detector contract",
    ) as exc_info:
        _detector_contract_from([product])

    assert message in str(exc_info.value)
