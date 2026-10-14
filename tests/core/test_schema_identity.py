# coding=utf-8
from __future__ import annotations

import pytest

from pilot_proxy.detector_contract import (
    DETECTOR_CONTRACT_SCHEMA_TOKEN,
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
)
from pilot_proxy.detector_geometry import STREAM_LAYOUT_SCHEMA_TOKEN
from pilot_proxy.detector_weights import WEIGHT_MANIFEST_SCHEMA_TOKEN
from pilot_proxy.integration.schemas import (
    DETECTOR_CORE_PROFILE_SCHEMA_TOKEN,
    RECEIVER_PROFILE_SCHEMA_TOKEN,
    STREAM_MAP_SCHEMA_TOKEN,
)
from pilot_proxy.result_schema import MASK_CONVENTION_SCHEMA_TOKEN, RESULT_SCHEMA_TOKEN
from pilot_proxy.runtime_bundle import (
    RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN,
    RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN,
    RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN,
)
from pilot_proxy.schema_identity import schema_token


def test_schema_token_is_strict_and_deterministic() -> None:
    assert schema_token("pilotproxy_example", 1) == "pilotproxy_example_v1"
    with pytest.raises(ValueError):
        schema_token("PilotProxy-example", 1)
    with pytest.raises(ValueError):
        schema_token("pilotproxy_example", 0)
    with pytest.raises(ValueError):
        schema_token("pilotproxy_example", 1.0)  # type: ignore[arg-type]


def test_current_schema_tokens_are_ground_zero_revision_one() -> None:
    assert DETECTOR_CORE_PROFILE_SCHEMA_TOKEN == "pilotproxy_detector_core_profile_v1"
    assert RECEIVER_PROFILE_SCHEMA_TOKEN == "pilotproxy_receiver_profile_v1"
    assert STREAM_MAP_SCHEMA_TOKEN == "pilotproxy_stream_map_v1"
    assert STREAM_LAYOUT_SCHEMA_TOKEN == "pilotproxy_stream_layout_v1"
    assert RESULT_SCHEMA_TOKEN == "pilotproxy_result_schema_v1"
    assert MASK_CONVENTION_SCHEMA_TOKEN == "pilotproxy_mask_convention_v1"
    assert DETECTOR_CONTRACT_SCHEMA_TOKEN == "pilotproxy_detector_contract_v1"
    assert CHIME_RUN_CONFIG_SCHEMA_TOKEN == "pilotproxy_chime_run_config_v1"
    assert CHIME_STATS_SCHEMA_TOKEN == "pilotproxy_chime_stats_v1"
    assert WEIGHT_MANIFEST_SCHEMA_TOKEN == "pilotproxy_weight_manifest_v1"
    assert RUNTIME_WEIGHT_MANIFEST_SCHEMA_TOKEN == "pilotproxy_runtime_weights_manifest_v1"
    assert RUNTIME_PILOT_PROFILES_SCHEMA_TOKEN == "pilotproxy_runtime_pilot_profiles_v1"
    assert RUNTIME_BUNDLE_VALIDATION_SCHEMA_TOKEN == "pilotproxy_runtime_bundle_validation_v1"
