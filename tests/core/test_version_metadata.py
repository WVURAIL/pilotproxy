# coding=utf-8
from __future__ import annotations

import re
from pathlib import Path

import pilot_proxy

ROOT = Path(__file__).resolve().parents[2]


def _match(path: Path, pattern: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None, f"pattern {pattern!r} not found in {path}"
    return match.group(1)


def test_version_metadata_is_consistent():
    package_version = pilot_proxy.__version__
    source_version = _match(
        ROOT / "src" / "pilot_proxy" / "_version.py",
        r'^__version__\s*=\s*"([^"]+)"$',
    )
    citation_version = _match(
        ROOT / "CITATION.cff",
        r'^version:\s*"([^"]+)"$',
    )
    assert package_version == source_version == citation_version


def test_build_metadata_reads_version_from_package():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "pilot_proxy._version.__version__"}' in pyproject
