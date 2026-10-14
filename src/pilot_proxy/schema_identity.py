# coding=utf-8
"""Small, strict helpers for current PilotProxy schema identities."""

from __future__ import annotations

import re

_SCHEMA_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def schema_token(name: str, revision: int) -> str:
    """Return the canonical ``<name>_v<revision>`` schema token."""
    if not isinstance(name, str) or not _SCHEMA_NAME.fullmatch(name):
        raise ValueError(
            "schema name must use lowercase ASCII letters, digits, and underscores"
        )
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("schema revision must be an integer")
    if revision <= 0:
        raise ValueError("schema revision must be positive")
    return f"{name}_v{revision}"


__all__ = ["schema_token"]
