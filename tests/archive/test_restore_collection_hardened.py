#!/usr/bin/env python3
"""Live replica layouts and a hardened candidate for _restore_collection.

The payload fixtures below are verbatim `datatrail ps <scope> <event> --json`
output captured on 2026-08-28, with each command given in the comment above
its fixture. Each fixture is replayed through the fake CLI boundary, so no
test reaches the network.

_restore_collection_v2 is a candidate replacement. The tests of acceptance and
refusal run it through the same derivation by monkeypatching
dt._restore_collection, so they test the behavior of the real adapter.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt

_PREFIX = "cadc:CHIMEFRB/"


# layouts captured live on 2026-08-28 with datatrail-cli 0.12.0
# $ python -m dtcli.cli ps kko.event.baseband.raw 307070707 --json
#   file_replica_locations: {"minoc": [1024 x bare], "kko": [1024 x /-rooted]}
# $ cadcinfo cadc:CHIMEFRB/data/kko/baseband/raw/2023/08/01/astro_307070707/
#            baseband_307070707_604.h5
#   -> size 5602304  md5 a5480f4e97d063b093f578ba63657dad   (EXISTS)
_KKO_MINOC = [
    "data/kko/baseband/raw/2023/08/01/astro_307070707/baseband_307070707_604.h5",
    "data/kko/baseband/raw/2023/08/01/astro_307070707/baseband_307070707_945.h5",
    "data/kko/baseband/raw/2023/08/01/astro_307070707/baseband_307070707_560.h5",
]
_KKO_SITE = ["/" + u for u in _KKO_MINOC]
_KKO_CP = "cadc:CHIMEFRB/data/kko/baseband/raw/2023/08/01/astro_307070707"

# $ python -m dtcli.cli ps chime.event.intensity.raw 1121212121 --json
#   file_replica_locations: {"minoc": [27 x bare]}
# $ cadcinfo cadc:CHIMEFRB/<first uri>  -> size 17432688
#   md5 0a3cfffe6e01344768419c27f7f41ef3   (EXISTS)
_INT_MINOC = [
    "data/chime/intensity/raw/2025/06/24/astro_1121212121/2068/"
    "astro_1121212121_20250624202247914073_beam2068_00336896_01.msgpack",
    "data/chime/intensity/raw/2025/06/24/astro_1121212121/2068/"
    "astro_1121212121_20250624202247914073_beam2068_00336898_01.msgpack",
]


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _install_fake_cli(monkeypatch, handler):
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))

    def fake_run(cmd, **kw):
        assert cmd[:3] == [sys.executable, "-m", "dtcli.cli"], cmd
        assert cmd[-1] == "--json", cmd
        rc, out, err = handler(cmd[3:-1])
        return _Proc(rc, out, err)

    monkeypatch.setattr(dt.subprocess, "run", fake_run)


_ORIGINAL_RESTORE = dt._restore_collection


def _resolve(monkeypatch, locs, restore=None):
    monkeypatch.setattr(dt, "_restore_collection",
                        restore if restore is not None else _ORIGINAL_RESTORE)
    _install_fake_cli(monkeypatch, lambda a: (0, json.dumps(
        {"dataset": "d", "scope": "s",
         "files": {"file_replica_locations": locs},
         "policies": {"p": 1}}), ""))
    try:
        return ("OK",) + dt.DATATRAIL.files("s", "e", retries=0)
    except dt.DatatrailContractError as exc:
        return ("REFUSED", str(exc))


# replica lists as served by Datatrail
def test_kko_outrigger_scope_returns_every_minoc_replica_bare(monkeypatch):
    """Bare minoc URIs are returned by the kko.event.baseband.raw scope, one of
    the four declared in the instrument YAMLs. Restoration is therefore needed
    for the outriggers as well as for CHIME."""
    assert all(not u.startswith("cadc:") for u in _KKO_MINOC)
    r = _resolve(monkeypatch, {"minoc": _KKO_MINOC, "kko": _KKO_SITE})
    assert r[0] == "OK" and r[1] == _KKO_CP, r
    assert sorted(r[2]) == ["baseband_307070707_560.h5",
                            "baseband_307070707_604.h5",
                            "baseband_307070707_945.h5"]
    # without restoration every outrigger event is refused
    pre = _resolve(monkeypatch, {"minoc": _KKO_MINOC}, restore=lambda u: u)
    assert pre[0] == "REFUSED" and "3/3 replicas affected" in pre[1], pre


def test_intensity_scope_also_returns_bare_and_keeps_beam_subdirectories(
        monkeypatch):
    """Bare URIs also occur in scopes other than baseband. This scope nests a
    per-beam subdirectory, so an event that spans two beams has legitimate
    names that contain a '/'. A guard against slashes in names would refuse
    these events."""
    r = _resolve(monkeypatch, {"minoc": _INT_MINOC})
    assert r[0] == "OK", r
    assert r[1] == ("cadc:CHIMEFRB/data/chime/intensity/raw/2025/06/24/"
                    "astro_1121212121/2068")
    assert all(n.endswith(".msgpack") and "/" not in n for n in r[2])

    two_beams = _INT_MINOC[:1] + [_INT_MINOC[0].replace("/2068/", "/2069/")
                                  .replace("beam2068", "beam2069")]
    r2 = _resolve(monkeypatch, {"minoc": two_beams})
    assert r2[0] == "OK", r2
    assert r2[1] == ("cadc:CHIMEFRB/data/chime/intensity/raw/2025/06/24/"
                     "astro_1121212121"), r2[1]
    assert all(n.startswith(("2068/", "2069/")) for n in r2[2]), r2[2]


def test_site_storage_element_spelling_is_the_narrowness_hole(monkeypatch):
    """In the kko site storage element (SE), the same paths are spelled
    '/data/...'. _BARE_REPLICA_ROOTS = ('data/',) does not match that
    spelling, so if minoc adopts it every outrigger event is refused."""
    r = _resolve(monkeypatch, {"minoc": _KKO_SITE})
    assert r[0] == "REFUSED", r
    assert "outside the expected collection" in r[1]


# hardened candidate
_BARE_REPLICA_ROOTS_V2 = ("data/",)


def _restore_collection_v2(uri: str) -> str:
    """Hardened candidate replacement for _restore_collection.

    As in the shipped version, a bare path is restored only when exactly one
    collection is configured, so a wider _MINOC_COLLECTIONS cannot assign a
    bare replica to the wrong collection behind the span check. A single
    leading '/' is also accepted, which is the spelling of the site storage
    elements and what the '//' collapse leaves behind. A non-canonical path,
    with an empty, '.' or '..' segment, is never restored because it cannot
    name a real artifact.
    """
    if uri.startswith(tuple(dt._MINOC_COLLECTIONS)):
        return uri
    if len(dt._MINOC_COLLECTIONS) != 1:
        return uri                      # ambiguous, left for the caller to refuse
    rel = uri[1:] if uri.startswith("/") else uri
    if not rel.startswith(_BARE_REPLICA_ROOTS_V2):
        return uri
    if any(part in ("", ".", "..") for part in rel.split("/")[:-1]):
        return uri
    return dt._MINOC_COLLECTIONS[0] + rel


@pytest.mark.parametrize("uris,expect", [
    (_KKO_MINOC, _KKO_CP),                          # bare, as served
    (_KKO_SITE, _KKO_CP),                           # /-rooted, accepted by v2 only
    ([_PREFIX + u for u in _KKO_MINOC], _KKO_CP),   # prefixed, unchanged
    ([_PREFIX + _KKO_MINOC[0]] + _KKO_MINOC[1:], _KKO_CP),   # mixed
    (["/" + _KKO_MINOC[0]] + _KKO_MINOC[1:], _KKO_CP),       # mixed spelling
])
def test_v2_accepts_every_spelling_of_the_same_replica_set(
        monkeypatch, uris, expect):
    r = _resolve(monkeypatch, {"minoc": uris}, restore=_restore_collection_v2)
    assert r[0] == "OK", r
    assert r[1] == expect, r[1]


def test_v2_refuses_a_non_canonical_bare_path(monkeypatch):
    bad = ["data/../../etc/passwd", "data/../../etc/shadow"]
    assert _resolve(monkeypatch, {"minoc": bad})[0] == "REFUSED"  # shared canonical-path guard
    r = _resolve(monkeypatch, {"minoc": bad}, restore=_restore_collection_v2)
    assert r[0] == "REFUSED", r


def test_v2_refuses_to_guess_when_more_than_one_collection_is_configured(
        monkeypatch):
    """The shipped version and the candidate both refuse a bare path when two
    collections are configured."""
    monkeypatch.setattr(dt, "_MINOC_COLLECTIONS",
                        ("cadc:CHIMEFRB/", "cadc:CHIMEOUTRIGGER/"))
    uris = ["data/kko/baseband/raw/2023/08/01/a/b_0.h5",
            "data/kko/baseband/raw/2023/08/01/a/b_1.h5"]
    shipped = _resolve(monkeypatch, {"minoc": uris})
    assert shipped[0] == "REFUSED"
    v2 = _resolve(monkeypatch, {"minoc": uris}, restore=_restore_collection_v2)
    assert v2[0] == "REFUSED", v2


def test_v2_is_idempotent_and_a_no_op_on_prefixed_uris():
    corpus = (_KKO_MINOC + _KKO_SITE + _INT_MINOC
              + [_PREFIX + u for u in _KKO_MINOC]
              + ["", "/", "//", "data", "data/", "/data/", "datax/y",
                 "cadc:OTHER/data/x", "data/./x", "data/../x", " data/x"])
    for u in corpus:
        once = _restore_collection_v2(u)
        assert _restore_collection_v2(once) == once, u
    for u in [_PREFIX + x for x in _KKO_MINOC]:
        assert _restore_collection_v2(u) == u


def test_v2_matches_the_shipped_version_on_every_form_seen_in_production():
    """For the bare and prefixed spellings seen in production, v2 and the
    shipped version return identical URIs."""
    for u in _KKO_MINOC + _INT_MINOC + [_PREFIX + x for x in _KKO_MINOC]:
        assert _restore_collection_v2(u) == dt._restore_collection(u), u
