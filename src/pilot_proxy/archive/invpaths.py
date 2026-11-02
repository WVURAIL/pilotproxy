# coding=utf-8
"""Resolve named inventories independently of the working directory.

Surveys and scans need to agree on where a named inventory lives. Both use
``$PILOT_PROXY_INVENTORY_ROOT`` when set, or ``~/datatrawl-inventories`` by
default. The root must be absolute.

An explicit ``--root`` or ``--inventory`` takes precedence. This module
supplies only the defaults for resolution by name.
"""
from __future__ import annotations

import os
from pathlib import Path

from .names import validate_identifier

ENV = "PILOT_PROXY_INVENTORY_ROOT"
DEFAULT_ROOT = "~/datatrawl-inventories"


def inventory_root() -> Path:
    """Return the inventory root, from ``$PILOT_PROXY_INVENTORY_ROOT`` or the default."""
    root = Path(os.environ.get(ENV, DEFAULT_ROOT)).expanduser()
    if not root.is_absolute():
        raise ValueError(
            f"{ENV} must be an absolute path, got {str(root)!r}")
    return root


def inventory_dir_for_write(name: str) -> Path:
    """Return the directory where a new inventory named ``name`` is written."""
    return inventory_root() / validate_identifier(name, label="inventory name")


def resolve_inventory(name: str) -> Path:
    """Resolve ``<dir>/inventory.jsonl`` for reading, by canonical name."""
    return inventory_dir_for_write(name) / "inventory.jsonl"
