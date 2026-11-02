"""Check that the detector library and profiles contain no cosmology term.

A forecast, tolerance or survey term must not appear in the library or in the
project profiles.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pilot_proxy"
TOKENS = re.compile(r"redshift|\br_tol\b|fisher|cosmolog|bao|omega", re.IGNORECASE)
# Frozen on-disk values are exempted by name, and no other value is exempt.
FROZEN = (
    # the archive health view's schema name, written into every view
    "pilotproxy_baonoise_health_view_v1",
)
# The name of the sibling repository (in a provenance path, in a command it runs, or the repository itself) is not a
# cosmology term. An import of it is refused by test_no_rfisher_import.
REPOSITORY = re.compile(r"\bRFIsher\b|\brfisher\b")
# Each entry of the detector register cites the release code it was read from, and the name of the sibling
# repository in such a provenance path is not a cosmology term.
PROVENANCE = re.compile(r"(src/)?rfisher/[A-Za-z_/]+\.py|RFIsher|rfisher-threshold-decisions-v\d+")


def _files():
    yield from sorted(SRC.rglob("*.py"))
    yield from sorted((ROOT / "projects").rglob("*.yaml"))
    yield from sorted((ROOT / "projects").rglob("*.json"))


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_cosmology_token(path):
    text = path.read_text(encoding="utf-8")
    for token in FROZEN:
        text = text.replace(token, "")
    if path.suffix != ".py":
        text = PROVENANCE.sub("", text)
    text = REPOSITORY.sub("", text)
    found = sorted({m.group(0) for m in TOKENS.finditer(text)})
    assert not found, found


def test_the_frozen_tokens_are_still_written():
    assert '"pilotproxy_baonoise_health_view_v1"' in (SRC / "archive_health.py").read_text(encoding="utf-8")
