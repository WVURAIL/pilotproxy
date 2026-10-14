# coding=utf-8
"""Strict JSON helpers for public reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Return a JSON-safe copy with non-finite floats converted to null values."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_dumps_strict(value: Any, **kwargs: Any) -> str:
    """Serialize strict JSON after converting non-finite floats to null values."""
    return json.dumps(json_safe(value), allow_nan=False, **kwargs)


def write_json_strict(path: Path, value: Any, **kwargs: Any) -> None:
    """Write strict JSON to the requested path."""
    path.write_text(json_dumps_strict(value, **kwargs), encoding="utf-8")
