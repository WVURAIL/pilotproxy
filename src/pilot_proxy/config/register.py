"""Decision register: auditable parameter choices, each tagged with its side.

A register file holds ``schema``, ``side``, ``source`` and ``decisions``. Every
decision records its value with units, basis, status, rationale, evidence and
sensitivity values, as the science register does. The loader is side-checked:
a detector loader refuses any entry tagged ``science``, so detection theory
cannot read a science choice by accident.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from ._files import ProfileError, check_keys, choice, read_json, text

REGISTER_SCHEMA = "pilot_proxy_detector_register_v1"
SIDES = ("detector", "science")
DECISION_BASES = ("instrument", "derivation", "literature", "empirical", "policy",
                  "implementation")
DECISION_STATUSES = ("derived", "locked", "provisional", "open", "conditional", "historical")
_DECISION_KEYS = ("id", "side", "value", "units", "basis", "status", "rationale",
                  "evidence", "sensitivity_values")


class RegisterSideError(ProfileError):
    """A register entry belongs to the other side of the handoff."""


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def _thaw(value):
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Decision:
    id: str
    side: str
    value: Any
    units: str
    basis: str
    status: str
    rationale: str
    evidence: tuple[str, ...]
    sensitivity_values: tuple[Any, ...]

    def record(self) -> dict[str, Any]:
        """The JSON form, as written in the register file."""
        return {key: _thaw(getattr(self, key)) for key in _DECISION_KEYS}


@dataclass(frozen=True)
class Register:
    schema: str
    side: str
    source: Mapping[str, Any]
    decisions: Mapping[str, Decision]

    def decision(self, id: str) -> Decision:
        try:
            return self.decisions[id]
        except KeyError as exc:
            raise KeyError(f"unknown {self.side} decision {id!r}") from exc

    def value(self, id: str):
        return self.decision(id).value

    def ids(self, prefix: Optional[str] = None) -> tuple[str, ...]:
        return tuple(sorted(i for i in self.decisions if prefix is None or i.startswith(prefix)))

    def canonical_json(self) -> str:
        payload = {"schema": self.schema, "side": self.side,
                   "decisions": [self.decisions[i].record() for i in self.ids()]}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _decision(entry: Any, *, side: str, where: str) -> Decision:
    if not isinstance(entry, Mapping):
        raise ProfileError(f"{where} must be a mapping")
    check_keys(entry, required=_DECISION_KEYS, where=where)
    ident = text(entry["id"], field="id", where=where)
    if any(part == "" for part in ident.split(".")):
        raise ProfileError(f"{where}: id must use non-empty dotted parts")
    entry_side = choice(entry["side"], SIDES, field="side", where=where)
    if entry_side != side:
        raise RegisterSideError(
            f"{where}: decision {ident!r} is tagged side {entry_side!r}; "
            f"a {side} register refuses it")
    status = choice(entry["status"], DECISION_STATUSES, field="status", where=where)
    evidence = entry["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        raise ProfileError(f"{where}: evidence must be a list of strings")
    if status not in ("open", "historical") and not evidence:
        raise ProfileError(f"{where}: decision {ident!r} needs evidence")
    if not isinstance(entry["sensitivity_values"], list):
        raise ProfileError(f"{where}: sensitivity_values must be a list")
    if not isinstance(entry["units"], str):
        raise ProfileError(f"{where}: units must be a string")
    return Decision(
        id=ident,
        side=entry_side,
        value=_freeze(entry["value"]),
        units=entry["units"],
        basis=choice(entry["basis"], DECISION_BASES, field="basis", where=where),
        status=status,
        rationale=text(entry["rationale"], field="rationale", where=where),
        evidence=tuple(evidence),
        sensitivity_values=_freeze(entry["sensitivity_values"]),
    )


def register_from_mapping(data: Mapping[str, Any], *, side: str = "detector",
                          where: str) -> Register:
    side = choice(side, SIDES, field="requested side", where=where)
    check_keys(data, required=("schema", "side", "decisions"), optional=("source",),
               where=where)
    if data["schema"] != REGISTER_SCHEMA:
        raise ProfileError(f"{where}: schema must be {REGISTER_SCHEMA!r}")
    file_side = choice(data["side"], SIDES, field="side", where=where)
    if file_side != side:
        raise RegisterSideError(f"{where}: a {file_side} register cannot be loaded as {side}")
    raw = data["decisions"]
    if not isinstance(raw, list):
        raise ProfileError(f"{where}: decisions must be a list")
    decisions: dict[str, Decision] = {}
    for index, entry in enumerate(raw):
        item = _decision(entry, side=side, where=f"{where}: decisions[{index}]")
        if item.id in decisions:
            raise ProfileError(f"{where}: duplicate decision {item.id!r}")
        decisions[item.id] = item
    source = data.get("source", {})
    if not isinstance(source, Mapping):
        raise ProfileError(f"{where}: source must be a mapping")
    return Register(REGISTER_SCHEMA, side, MappingProxyType(dict(source)),
                    MappingProxyType(decisions))


def load_register(path: Path | str, *, side: str = "detector") -> Register:
    path = Path(path)
    return register_from_mapping(read_json(path), side=side, where=str(path))


__all__ = [
    "DECISION_BASES",
    "DECISION_STATUSES",
    "Decision",
    "REGISTER_SCHEMA",
    "Register",
    "RegisterSideError",
    "SIDES",
    "load_register",
    "register_from_mapping",
]
