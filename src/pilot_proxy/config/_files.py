"""Strict readers and field checks shared by the profile loaders."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - reported when a YAML file is read
    yaml = None

MONTH_RE = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")


class ProfileError(ValueError):
    """A profile file is missing, malformed or inconsistent."""


def read_yaml(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise ProfileError(f"{path}: pyyaml is required to read profile files")
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ProfileError(f"{path}: profile file not found") from exc
    if not isinstance(data, Mapping):
        raise ProfileError(f"{path}: the root must be a mapping")
    return data


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"{path}: profile file not found") from exc
    if not isinstance(data, Mapping):
        raise ProfileError(f"{path}: the root must be a JSON object")
    return data


def check_keys(data: Mapping[str, Any], *, required: Iterable[str],
               optional: Iterable[str] = (), where: str) -> None:
    required = set(required)
    allowed = required | set(optional)
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError(f"{where}: missing key(s) {missing}")
    unknown = sorted(str(key) for key in set(data) - allowed)
    if unknown:
        raise ProfileError(f"{where}: unknown key(s) {unknown}")


def number(value: Any, *, field: str, where: str, positive: bool = False) -> float:
    """A YAML/JSON number as a float. Strings are refused, so a float written
    as ``6.0e6`` (which PyYAML reads as text) cannot slip through."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where}: {field} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileError(f"{where}: {field} must be finite")
    if positive and result <= 0.0:
        raise ProfileError(f"{where}: {field} must be > 0, got {value!r}")
    return result


def integer(value: Any, *, field: str, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{where}: {field} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ProfileError(f"{where}: {field} must be >= {minimum}, got {value!r}")
    return int(value)


def text(value: Any, *, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{where}: {field} must be a non-empty string, got {value!r}")
    return value.strip()


def choice(value: Any, options: Iterable[str], *, field: str, where: str) -> str:
    options = tuple(options)
    if value not in options:
        raise ProfileError(f"{where}: {field} must be one of {list(options)}, got {value!r}")
    return str(value)


def month(value: Any, *, field: str, where: str) -> str:
    if not isinstance(value, str) or not MONTH_RE.match(value):
        raise ProfileError(f"{where}: {field} must be a YYYY-MM month, got {value!r}")
    return value
