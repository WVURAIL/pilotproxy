#!/usr/bin/env python3
"""Adversarial audit of the uncommitted _restore_collection patch.

Every test drives the REAL derivation in
pilot_proxy.archive.datatrail_client.Datatrail.files() through the same
fake-CLI subprocess boundary tests/archive/test_datatrail_adapter.py uses, so
nothing here touches Datatrail or CADC.

`unpatched=True` monkeypatches _restore_collection back to the identity
function, which is byte-for-byte the pre-patch normalization line -- that gives
a clean pre/post differential for every hostile input.
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
    """Run files() over a minoc replica list; return (cp, names) or the error.

    Set explicitly on EVERY call so a pre/post differential inside one test
    function cannot leak the identity stub into the second half.
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


# =====================================================================
# 6. IDEMPOTENCY -- brute force over a generated corpus
# =====================================================================
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


# =====================================================================
# 1. TOO NARROW -- a bare path that is not spelled exactly "data/..."
# =====================================================================
@pytest.mark.parametrize("uri", [
    "/data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",   # leading slash
    "//data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",  # collapses to above
    "./data/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",
    "DATA/chime/baseband/raw/2025/03/31/astro_1/b_0.h5",
    "chime/baseband/raw/2025/03/31/astro_1/b_0.h5",         # "data/" also dropped
])
def test_narrowness_bare_variants_still_refused(monkeypatch, uri):
    """_BARE_REPLICA_ROOTS=('data/',) is an exact literal: any other spelling
    of the same bare path is still a hard refusal."""
    r = _resolve(monkeypatch, [uri])
    assert r[0] == "REFUSED", r
    assert "outside the expected collection" in r[1]


# =====================================================================
# 2. TOO BROAD -- "data/" is the root of every CHIME product, not just
#    the CHIMEFRB baseband the collection names
# =====================================================================
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


# =====================================================================
# 3. INTERACTION WITH .replace("//","/")
# =====================================================================
def test_triple_slash_survives_the_collapse(monkeypatch):
    """str.replace is a single non-overlapping pass: '///' -> '//'."""
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
    """'data/' restores to a valid-looking collection URI whose relative part
    is empty; the no-usable-split guard is what stops it, not the predicate."""
    assert dt._restore_collection("data/") == "cadc:CHIMEFRB/data/"
    r = _resolve(monkeypatch, ["data/"])
    assert r[0] == "REFUSED" and "canonical path" in r[1]


# =====================================================================
# 4. THE "spans multiple collections" CHECK
# =====================================================================
def test_bare_paths_with_multiple_collections_are_refused(monkeypatch):
    """Both bare and mixed replies remain ambiguous with two collections."""
    monkeypatch.setattr(dt, "_MINOC_COLLECTIONS",
                        ("cadc:CHIMEFRB/", "cadc:CHIMEOUTRIGGER/"))
    # both replicas really live in CHIMEOUTRIGGER; both come back bare
    r = _resolve(monkeypatch, ["data/kko/baseband/raw/2025/01/01/a/b_0.h5",
                               "data/kko/baseband/raw/2025/01/01/a/b_1.h5"])
    assert r[0] == "REFUSED" and "outside the expected collection" in r[1]
    # ...the genuinely mixed case IS still caught, because the prefixed half
    # still votes for its own collection
    r2 = _resolve(monkeypatch, ["cadc:CHIMEOUTRIGGER/data/kko/a/b_0.h5",
                                "data/chime/a/b_1.h5"])
    assert r2[0] == "REFUSED" and "outside the expected collection" in r2[1], r2


# =====================================================================
# 5. commonprefix vs commonpath -- restoration widens the population that
#    reaches the character-wise prefix
# =====================================================================
def test_restoration_admits_a_mixed_directory_event_that_used_to_refuse(
        monkeypatch):
    """Pre-patch this was a deterministic refusal. Post-patch it resolves to a
    common path two levels ABOVE either replica's directory, and the derived
    'names' silently become multi-segment paths."""
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
    names share a prefix trim to their PARENT. Restoration is what lets this
    set through: pre-patch the bare half was a stray."""
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
    """The July-era guarantee: when every URI carries the prefix, the patch is
    a no-op, so no previously recorded common_path can move."""
    uris = [_D2020 + f"baseband_100200200_{i}.h5" for i in (0, 156, 1023)]
    assert _resolve(monkeypatch, uris) == _resolve(monkeypatch, uris,
                                                   unpatched=True)


# =====================================================================
# 7. NON-MINOC STORAGE ELEMENTS
# =====================================================================
def test_arc_and_other_storage_elements_are_untouched(monkeypatch):
    """files() reads locations['minoc'] only, so no other SE reaches
    normalization -- with or without the patch."""
    extra = {"arc": ["/arc/projects/chime_frb/data/chime/baseband/raw/x.h5"],
             "chime": ["data/whatever/y.h5"]}
    uris = [_D2020 + "baseband_100200200_0.h5",
            _D2020 + "baseband_100200200_1.h5"]
    with_extra = _resolve(monkeypatch, uris, extra_ses=extra)
    without = _resolve(monkeypatch, uris)
    assert with_extra == without, (with_extra, without)
    assert with_extra[1] == _D2020.rstrip("/")


def test_arc_only_dataset_is_a_no_data_verdict_not_a_refusal(monkeypatch):
    """No 'minoc' key at all -> (None, [], True) = queried OK, no minoc files.
    The patch cannot turn this into a refusal or a path."""
    _install_fake_cli(monkeypatch, lambda a: (0, _ps_payload(
        {"file_replica_locations":
         {"arc": ["/arc/projects/chime_frb/data/x.h5"]}}), ""))
    assert dt.DATATRAIL.files("s", "e", retries=0) == (None, [], True)
