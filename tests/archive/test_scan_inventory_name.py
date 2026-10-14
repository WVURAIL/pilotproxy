# coding=utf-8
"""Named-inventory path handling for pilot-proxy chime-scan."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pilot_proxy.archive import invpaths
from pilot_proxy.archive.scan import _named_inventory_path


def test_named_inventory_path_uses_source_root(tmp_path: Path) -> None:
    assert _named_inventory_path("chime-pilots", tmp_path) == (
        tmp_path / "data" / "chime-pilots" / "inventory.jsonl"
    )


def test_named_inventory_path_requires_canonical_resolver(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pilot_proxy.archive.invpaths", None)
    with pytest.raises(SystemExit, match="canonical inventory resolver"):
        _named_inventory_path("chime-ch614-706", None)


def test_named_inventory_path_delegates_to_archive_invpaths() -> None:
    # The archive resolver is the source of truth for named inventories.
    expected = Path(invpaths.resolve_inventory("chime-ch614-706"))
    assert _named_inventory_path("chime-ch614-706", None) == expected
