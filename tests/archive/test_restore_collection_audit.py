#!/usr/bin/env python3
"""Tests of _restore_collection on hostile minoc replica lists.

Each test drives the real derivation in
pilot_proxy.archive.datatrail_client.Datatrail.files() through the fake CLI
subprocess boundary of tests/archive/test_datatrail_adapter.py, so no test
reaches Datatrail or CADC.

With `unpatched=True`, _restore_collection is replaced by the identity
function, which is the normalization with no collection restored. A test can
then run the same hostile input with and without restoration.
"""
from __future__ import annotations

import itertools
import json
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _install_fake_cli(monkeypatch, handler=None, version=(0, 11, 0)):
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: version)

    def fake_run(cmd, **kw):
        assert handler is not None, f"unexpected datatrail call: {cmd}"
        assert cmd[:3] == [sys.executable, "-m", "dtcli.cli"], cmd
        assert cmd[-1] == "--json", cmd
        rc, out, err = handler(cmd[3:-1])
        return _Proc(rc, out, err)

    monkeypatch.setattr(dt.subprocess, "run", fake_run)


def _ps_payload(files):
    return json.dumps(
        {"dataset": "d", "scope": "s", "files": files, "policies": {"p": 1}})


_ORIGINAL_RESTORE = dt._restore_collection


def _resolve(monkeypatch, uris, extra_ses=None, unpatched=False):
    """Run files() over a minoc replica list and return its verdict.

    The verdict is ("OK", cp, names, ok) or ("REFUSED", message).
    _restore_collection is set on every call, so the identity stub used in
    the first half of a test cannot leak into the second half.
    """
    monkeypatch.setattr(dt, "_restore_collection",
                        (lambda u: u) if unpatched else _ORIGINAL_RESTORE)
    locs = {"minoc": list(uris)}
    if extra_ses:
        locs.update(extra_ses)
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": locs}), ""))
    try:
        cp, names, ok = dt.DATATRAIL.files("chime.event.baseband.raw", "E",
                                           retries=0)
    except dt.DatatrailContractError as exc:
        return ("REFUSED", str(exc))
    return ("OK", cp, names, ok)


# real prefixes taken verbatim from the reference inventories
_D2020 = "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100200200/"
_B2025 = "data/chime/baseband/raw/2025/03/31/astro_1111116060/"


# idempotency over a generated corpus
def test_restore_collection_is_idempotent_over_generated_corpus():
    heads = ["", "/", "//", "data", "data/", "/data/", "cadc:CHIMEFRB",
             "cadc:CHIMEFRB/", "cadc:CHIMEFRB//", "cadc:OTHER/", "datax/",
             "DATA/", " data/", "\tdata/", "cadc:CHIMEFRB/data/"]
    tails = ["", "x", "x.h5", "chime/baseband/raw/2020/07/15/a/b.h5",
             "../../etc/passwd", "data/", "cadc:CHIMEFRB/", "//y"]
    corpus = ["".join(p) for p in itertools.product(heads, tails)]
    corpus += ["\x00", "\n", "data/\x00x", "data/" * 3]
    assert len(corpus) == 15 * 8 + 4
    bad = [u for u in corpus
           if dt._restore_collection(dt._restore_collection(u))
           != dt._restore_collection(u)]
    assert bad == [], bad


def test_restore_collection_never_changes_an_already_prefixed_uri():
    for u in [_D2020 + "b.h5", "cadc:CHIMEFRB/", "cadc:CHIMEFRB/data/",
              "cadc:CHIMEFRB//data/x", "cadc:CHIMEFRB/anything at all"]:
        assert dt._restore_collection(u) == u


# bare paths not spelled exactly "data/..."
@pytest.mark.parametrize("uri", [
    "/data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",   # leading slash
    "//data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",  # collapses to above
    "./data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",
    "DATA/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",
    "chime/baseband/raw/2025/03/31/astro_1/b_0.h5",         # "data/" also dropped
])
def test_narrowness_bare_variants_still_refused(monkeypatch, uri):
    """_BARE_REPLICA_ROOTS = ('data/',) is matched as an exact literal, so any
    other spelling of the same bare path is refused."""
    r = _resolve(monkeypatch, [uri])
    assert r[0] == "REFUSED", r
    assert "outside the expected collection" in r[1]


# bare paths of other CHIME products, which also sit under "data/"
@pytest.mark.parametrize("bare,expect", [
    ("data/kko/baseband/raw/2025/01/01/astro_9/b_0.h5",
     "cadc:CHIMEFRB/data/kko/baseband/raw/2025/01/01/astro_9"),
    ("data/chime/intensity/raw/2025/01/01/x.h5",
     "cadc:CHIMEFRB/data/chime/intensity/raw/2025/01/01"),
    ("data/../../etc/passwd",
     None),
])
def test_data_root_restoration_requires_a_canonical_path(
        monkeypatch, bare, expect):
    """Restoring a collection never authorizes parent traversal."""
    r = _resolve(monkeypatch, [bare, bare.rsplit("/", 1)[0] + "/other.h5"])
    if expect is None:
        assert r[0] == "REFUSED" and "canonical path" in r[1]
    else:
        assert r[0] == "OK", r
        assert r[1] == expect, r[1]


def test_broadness_pre_patch_refused_the_same_input(monkeypatch):
    r = _resolve(monkeypatch, ["data/kko/baseband/raw/2025/01/01/a/b.h5",
                               "data/kko/baseband/raw/2025/01/01/a/c.h5"],
                 unpatched=True)
    assert r[0] == "REFUSED", r


# interaction with .replace("//", "/")
def test_triple_slash_survives_the_collapse(monkeypatch):
    """str.replace makes a single non-overlapping pass, so '///' becomes '//'."""
    assert "cadc:CHIMEFRB///data/x".replace("//", "/") == \
        "cadc:CHIMEFRB//data/x"
    r = _resolve(monkeypatch, ["cadc:CHIMEFRB///data/chime/a/b_0.h5",
                               "cadc:CHIMEFRB///data/chime/a/b_1.h5"])
    assert r[0] == "REFUSED" and "canonical path" in r[1]


def test_doubled_slash_inside_a_bare_path_is_collapsed_then_restored(
        monkeypatch):
    r = _resolve(monkeypatch, ["data//chime/baseband/raw/2025/03/31/a/b_0.h5",
                               "data/chime/baseband/raw/2025/03/31/a/b_1.h5"])
    assert r[0] == "OK", r
    assert r[1] == "cadc:CHIMEFRB/data/chime/baseband/raw/2025/03/31/a"


def test_empty_and_blank_uris_never_reach_restoration(monkeypatch):
    for uris in ([""], [" "], ["\t"], [_B2025 + "b.h5", ""]):
        r = _resolve(monkeypatch, uris)
        assert r[0] == "REFUSED", (uris, r)
        assert "malformed 'minoc' replica locations" in r[1], (uris, r)
    assert dt._restore_collection("") == ""


def test_bare_root_only_uri_is_caught_by_the_split_guard(monkeypatch):
    """'data/' restores to a valid-looking collection URI with an empty
    relative part. The collection check passes it, and the canonical-path
    check refuses it."""
    assert dt._restore_collection("data/") == "cadc:CHIMEFRB/data/"
    r = _resolve(monkeypatch, ["data/"])
    assert r[0] == "REFUSED" and "canonical path" in r[1]


# the "spans multiple collections" check
def test_bare_paths_with_multiple_collections_are_refused(monkeypatch):
    """Both bare and mixed replies remain ambiguous with two collections."""
    monkeypatch.setattr(dt, "_MINOC_COLLECTIONS",
                        ("cadc:CHIMEFRB/", "cadc:CHIMEOUTRIGGER/"))
    # both replicas live in CHIMEOUTRIGGER and both come back bare
    r = _resolve(monkeypatch, ["data/kko/baseband/raw/2025/01/01/a/b_0.h5",
                               "data/kko/baseband/raw/2025/01/01/a/b_1.h5"])
    assert r[0] == "REFUSED" and "outside the expected collection" in r[1]
    # a mixed reply is refused too, since a bare path is not restored when
    # two collections are configured
    r2 = _resolve(monkeypatch, ["cadc:CHIMEOUTRIGGER/data/kko/a/b_0.h5",
                                "data/chime/a/b_1.h5"])
    assert r2[0] == "REFUSED" and "outside the expected collection" in r2[1], r2


# commonprefix and commonpath, where restoration lets more replica sets
# reach the character-wise prefix
def test_restoration_admits_a_mixed_directory_event_that_used_to_refuse(
        monkeypatch):
    """Without restoration this set is refused. With restoration it resolves
    to a common path four levels above either replica's directory, and the
    derived names become multi-segment paths with no warning."""
    uris = [_D2020 + "baseband_100200200_0.h5",
            _B2025 + "baseband_1111116060_832.h5"]
    before = _resolve(monkeypatch, uris, unpatched=True)
    assert before[0] == "REFUSED", before

    after = _resolve(monkeypatch, uris)
    assert after[0] == "OK", after
    assert after[1] == "cadc:CHIMEFRB/data/chime/baseband/raw", after[1]
    assert after[2] == [
        "2020/07/15/astro_100200200/baseband_100200200_0.h5",
        "2025/03/31/astro_1111116060/baseband_1111116060_832.h5"], after[2]
    # the only name guard in files() is emptiness, so slashes pass
    assert all(n for n in after[2])


def test_sibling_directories_sharing_a_character_prefix(monkeypatch):
    """commonprefix is character-wise, so two sibling event directories whose
    names share a prefix are trimmed to their parent. This set reaches
    commonprefix only through restoration, since without it the bare half is
    a stray."""
    uris = ["cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
            "astro_100200200/baseband_100200200_0.h5",
            "data/chime/baseband/raw/2020/07/15/"
            "astro_1002002000/baseband_1002002000_0.h5"]
    assert _resolve(monkeypatch, uris, unpatched=True)[0] == "REFUSED"
    r = _resolve(monkeypatch, uris)
    assert r[0] == "OK", r
    assert r[1] == "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15", r[1]
    assert r[2] == ["astro_100200200/baseband_100200200_0.h5",
                    "astro_1002002000/baseband_1002002000_0.h5"], r[2]


def test_restoration_never_changes_an_all_prefixed_replica_set(monkeypatch):
    """When every URI carries the prefix, restoration has no effect, so no
    common_path recorded by an earlier survey can move."""
    uris = [_D2020 + f"baseband_100200200_{i}.h5" for i in (0, 156, 1023)]
    assert _resolve(monkeypatch, uris) == _resolve(monkeypatch, uris,
                                                   unpatched=True)


# storage elements other than minoc
def test_arc_and_other_storage_elements_are_untouched(monkeypatch):
    """files() reads only locations['minoc'], so no other storage element
    (SE) reaches normalization, with or without restoration."""
    extra = {"arc": ["/arc/projects/chime_frb/data/chime/baseband/raw/x.h5"],
             "chime": ["data/whatever/y.h5"]}
    uris = [_D2020 + "baseband_100200200_0.h5",
            _D2020 + "baseband_100200200_1.h5"]
    with_extra = _resolve(monkeypatch, uris, extra_ses=extra)
    without = _resolve(monkeypatch, uris)
    assert with_extra == without, (with_extra, without)
    assert with_extra[1] == _D2020.rstrip("/")


def test_arc_only_dataset_is_a_no_data_verdict_not_a_refusal(monkeypatch):
    """A reply with no 'minoc' key returns (None, [], True), meaning the query
    succeeded and found no minoc files. Restoration cannot turn this into a
    refusal or a path."""
    _install_fake_cli(monkeypatch, lambda a: (0, _ps_payload(
        {"file_replica_locations":
         {"arc": ["/arc/projects/chime_frb/data/x.h5"]}}), ""))
    assert dt.DATATRAIL.files("s", "e", retries=0) == (None, [], True)
