# coding=utf-8
from __future__ import annotations

import json

from pilot_proxy.product_contract import (
    ACTIVE_DECISION_METHOD,
    FINE_CANDIDATE_CALIBRATION_STATUS,
    FINE_CANDIDATE_DECISION_METHOD,
    FINE_TERMS_ROLE,
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    current_decision_contract,
    current_decision_contract_json,
)


def test_current_product_schema_identity_is_structured() -> None:
    assert PER_PILOT_PRODUCT_SCHEMA_NAME == "pilotproxy_per_pilot_product"
    assert PER_PILOT_PRODUCT_SCHEMA_REVISION == 5
    assert PER_PILOT_PRODUCT_SCHEMA_TOKEN == "pilotproxy_per_pilot_product_v5"


def test_decision_contract_distinguishes_active_diagnostic_and_candidate() -> None:
    contract = current_decision_contract()
    assert contract["active_decision"]["method"] == ACTIVE_DECISION_METHOD
    assert contract["active_decision"]["output_field"] == "reject_mask"
    # The scan measures the fine terms and decides nothing on them.
    assert contract["fine_measurement"]["role"] == FINE_TERMS_ROLE
    assert contract["fine_measurement"]["terms_field"] == "fine_power_u64"
    assert "fine_diagnostic" not in contract
    assert contract["fine_candidate_decision"]["method"] == FINE_CANDIDATE_DECISION_METHOD
    assert contract["fine_candidate_decision"]["calibration_status"] == (
        FINE_CANDIDATE_CALIBRATION_STATUS
    )
    assert contract["fine_candidate_decision"]["active"] is False
    assert json.loads(current_decision_contract_json()) == contract
