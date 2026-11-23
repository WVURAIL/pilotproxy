# coding=utf-8
"""The analyzer checks a weight bank's manifest before it uses the weights."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.archive.detector import (
    _reference_placement_summary,
    _weight_coordinate_metadata,
)
from pilot_proxy.detector_contract import WEIGHT_COORDINATE_POST_NYQUIST_ZONE


def test_weight_coordinate_rejects_missing_coordinate_manifest() -> None:
    weight_bank = SimpleNamespace(
        path="weights/chime_raw.bin",
        manifest={
            "receiver_profile": {
                "channelizer": {
                    "frequency_axis": {
                        "nyquist_zone": 2,
                    },
                },
            },
        },
    )

    with pytest.raises(ValueError, match="requires weight_coordinate_system"):
        _weight_coordinate_metadata(
            weight_bank=weight_bank,
            input_nyquist_zone=2,
        )


def test_weight_coordinate_rejects_raw_inverted_manifest() -> None:
    weight_bank = SimpleNamespace(
        path="weights/chime_raw.bin",
        manifest={
            "weight_coordinate_system": "raw_input_frequency_coordinate",
            "receiver_profile": {
                "channelizer": {
                    "frequency_axis": {
                        "nyquist_zone": 2,
                    },
                },
            },
        },
    )

    with pytest.raises(ValueError, match="detector-coordinate weights"):
        _weight_coordinate_metadata(
            weight_bank=weight_bank,
            input_nyquist_zone=2,
        )


def test_weight_coordinate_accepts_declared_post_normalization_manifest() -> None:
    weight_bank = SimpleNamespace(
        path="weights/chime_detector_coordinate.bin",
        manifest={
            "weight_coordinate_system": WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
            "input_preprocessing": {
                "time_reverse_detector_windows_before_kernel": True,
            },
            "receiver_profile": {
                "channelizer": {
                    "frequency_axis": {
                        "nyquist_zone": 2,
                    },
                },
            },
        },
    )

    metadata = _weight_coordinate_metadata(
        weight_bank=weight_bank,
        input_nyquist_zone=2,
    )

    assert metadata["effective_weight_coordinate_system"] == (
        WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    )
    assert metadata["input_requires_time_reversal"] is True


def test_reference_placement_summary_compacts_manifest() -> None:
    manifest = {
        "kernel_spec": {
            "reference_offset_bins": 2,
            "skipped_guard_bins": 1,
        },
        "forbidden_tone_policy": {
            "forbidden_tone": "coarse_channel_dc",
        },
        "target_reference_layout": [
            {
                "physical_channel": 14,
                "reference_placement_status": "nominal",
                "placement_warnings": "",
                "lower_reference_offset_bins": -2,
                "upper_reference_offset_bins": 2,
                "lower_reference_relative_to_target_hz": -6103.515625,
                "upper_reference_relative_to_target_hz": 6103.515625,
                "lower_reference_edge_wrapped": False,
                "upper_reference_edge_wrapped": False,
                "lower_reference_dc_shifted": False,
                "upper_reference_dc_shifted": False,
                "edge_reference_wrapped": False,
                "dc_reference_shifted": False,
                "forbidden_tone_in_skipped_guard": True,
            },
            {
                "physical_channel": 21,
                "reference_placement_status": "edge_wrapped",
                "placement_warnings": "lower reference wrapped across coarse-channel edge",
                "lower_reference_offset_bins": -2,
                "upper_reference_offset_bins": 2,
                "lower_reference_relative_to_target_hz": -6103.515625,
                "upper_reference_relative_to_target_hz": 6103.515625,
                "lower_reference_edge_wrapped": True,
                "upper_reference_edge_wrapped": False,
                "lower_reference_dc_shifted": False,
                "upper_reference_dc_shifted": False,
                "edge_reference_wrapped": True,
                "dc_reference_shifted": False,
                "forbidden_tone_in_skipped_guard": False,
            },
            {
                "physical_channel": 99,
                "reference_placement_status": "nominal",
            },
        ],
    }

    summary = _reference_placement_summary(
        manifest,
        np.asarray([14, 21], dtype=np.int32),
    )

    assert summary is not None
    assert summary["reference_offset_bins"] == 2
    assert summary["skipped_guard_bins"] == 1
    assert summary["reference_placement_status"] == "mixed:edge_wrapped;nominal"
    assert summary["num_channels_with_adaptive_reference"] == 1
    assert summary["channels_with_adaptive_reference"] == [21]
    assert summary["num_edge_wrapped_references"] == 1
    assert summary["channels_with_edge_wrapped_reference"] == [21]
    assert summary["num_forbidden_tone_in_skipped_guard"] == 1
    assert summary["channels_with_forbidden_tone_in_skipped_guard"] == [14]
    assert summary["forbidden_tone_policy"]["forbidden_tone"] == "coarse_channel_dc"
    assert [row["physical_channel"] for row in summary["by_channel"]] == [14, 21]
