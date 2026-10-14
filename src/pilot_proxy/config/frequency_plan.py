"""Frequency plan: the emitter bands as an explicit list.

A plan is a YAML file with a ``name`` and a ``bands`` list. Each band has a
``label`` (the join key, a string), integer or decimal ``low_mhz`` and
``high_mhz`` edges, a ``role`` and, optionally, a declared ``target_freq_id``.

Roles:
- ``screened``: a band the characterization screens for its emitter;
- ``control``: a band with no licensed emitter, processed the same way as a
  reference. A control band has no marker to measure, so it declares the
  instrument channel it is read at (``target_freq_id``).

No band count, first label or channel formula lives in code.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ._files import ProfileError, check_keys, choice, integer, number, read_yaml, text

ROLES = ("screened", "control")
HZ_PER_MHZ = 1e6


@dataclass(frozen=True)
class Band:
    label: str
    low_mhz: float
    high_mhz: float
    role: str
    target_freq_id: Optional[int] = None

    @property
    def low_hz(self) -> float:
        return self.low_mhz * HZ_PER_MHZ

    @property
    def high_hz(self) -> float:
        return self.high_mhz * HZ_PER_MHZ

    @property
    def width_hz(self) -> float:
        return self.high_hz - self.low_hz

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.low_hz + self.high_hz)

    def contains_hz(self, f_hz: float) -> bool:
        """True for ``low <= f < high`` (a shared edge belongs to the upper band)."""
        return self.low_hz <= float(f_hz) < self.high_hz


@dataclass(frozen=True)
class FrequencyPlan:
    name: str
    all_bands: tuple[Band, ...]

    def band(self, label: str) -> Band:
        for item in self.all_bands:
            if item.label == str(label):
                return item
        raise KeyError(f"frequency plan {self.name!r} has no band {label!r}")

    def bands(self, role: Optional[str] = None) -> tuple[Band, ...]:
        if role is None:
            return self.all_bands
        if role not in ROLES:
            raise ValueError(f"unknown band role {role!r}; expected one of {list(ROLES)}")
        return tuple(item for item in self.all_bands if item.role == role)

    def labels(self, role: Optional[str] = None) -> tuple[str, ...]:
        return tuple(item.label for item in self.bands(role))


def frequency_plan_from_mapping(data: Mapping[str, Any], *, where: str) -> FrequencyPlan:
    check_keys(data, required=("name", "bands"), where=where)
    name = text(data["name"], field="name", where=where)
    raw = data["bands"]
    if not isinstance(raw, list) or not raw:
        raise ProfileError(f"{where}: bands must be a non-empty list")
    bands: list[Band] = []
    for index, entry in enumerate(raw):
        at = f"{where}: bands[{index}]"
        if not isinstance(entry, Mapping):
            raise ProfileError(f"{at} must be a mapping")
        check_keys(entry, required=("label", "low_mhz", "high_mhz", "role"),
                   optional=("target_freq_id",), where=at)
        if not isinstance(entry["label"], str):
            raise ProfileError(f"{at}: label must be a quoted string, got {entry['label']!r}")
        label = text(entry["label"], field="label", where=at)
        low = number(entry["low_mhz"], field="low_mhz", where=at, positive=True)
        high = number(entry["high_mhz"], field="high_mhz", where=at, positive=True)
        if not high > low:
            raise ProfileError(f"{at}: high_mhz must exceed low_mhz")
        role = choice(entry["role"], ROLES, field="role", where=at)
        target = entry.get("target_freq_id")
        if target is not None:
            target = integer(target, field="target_freq_id", where=at, minimum=0)
        if role == "control" and target is None:
            raise ProfileError(f"{at}: a control band must declare target_freq_id")
        bands.append(Band(label, low, high, role, target))
    labels = [item.label for item in bands]
    if len(set(labels)) != len(labels):
        raise ProfileError(f"{where}: band labels must be unique")
    for before, after in zip(bands, bands[1:]):
        if after.low_mhz < before.high_mhz:
            raise ProfileError(
                f"{where}: bands must be listed in frequency order without overlap "
                f"({before.label} and {after.label})")
    return FrequencyPlan(name=name, all_bands=tuple(bands))


def load_frequency_plan(path: Path | str) -> FrequencyPlan:
    path = Path(path)
    return frequency_plan_from_mapping(read_yaml(path), where=str(path))


__all__ = [
    "Band",
    "FrequencyPlan",
    "ROLES",
    "frequency_plan_from_mapping",
    "load_frequency_plan",
]
