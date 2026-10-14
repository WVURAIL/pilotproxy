# coding=utf-8
from __future__ import annotations

import json
import math

import numpy as np

from pilot_proxy.json_utils import json_dumps_strict, json_safe

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
