# coding=utf-8
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pilot_proxy.json_utils import json_dumps_strict, json_safe, load_json_object

FINITE_FLOAT32_VALUE = 1.5
STRICT_JSON_GOOD_VALUE = 2.0


def test_json_safe_maps_nonfinite_floats_to_none() -> None:
    payload = {
        "nan": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": [np.float64(math.nan), np.float32(FINITE_FLOAT32_VALUE)],
    }

    safe = json_safe(payload)

    assert safe == {
        "nan": None,
        "pos_inf": None,
        "neg_inf": None,
        "nested": [None, FINITE_FLOAT32_VALUE],
    }


def test_json_dumps_strict_outputs_parseable_json_with_nulls() -> None:
    text = json_dumps_strict(
        {"bad": float("nan"), "good": STRICT_JSON_GOOD_VALUE},
        indent=2,
    )

    assert "NaN" not in text
    assert "Infinity" not in text
    assert "null" in text
    assert json.loads(text) == {"bad": None, "good": STRICT_JSON_GOOD_VALUE}


@pytest.mark.parametrize("payload", [None, b'{"unfinished":', b'[]', b'\xff'])
def test_json_validation_reports_invalid_files_without_crashing(tmp_path, payload):
    path = tmp_path / "profile.json"
    if payload is not None:
        path.write_bytes(payload)
    errors = []
    assert load_json_object(path, errors) == {}
    assert len(errors) == 1
    assert errors[0]["severity"] == "error"
    prefix = "required_file" if payload is None else "json"
    assert errors[0]["check"] == f"{prefix}.profile.json"


def test_json_validation_preserves_prior_errors_and_valid_payload(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"threshold": 18446744073709551615}')
    errors = [{"severity": "error", "check": "previous", "message": "previous error"}]
    assert load_json_object(path, errors) == {"threshold": (1 << 64) - 1}
    assert len(errors) == 1
