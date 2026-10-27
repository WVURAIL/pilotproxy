#!/usr/bin/env python3
"""Real-data invariant for _restore_collection.

Minoc replica lists are rebuilt from the frozen Aug-3 production bundle and
the July datatrawl inventory, whose common_path and file name are the values
the adapter derived before restoration was added. The current derivation is
then rerun over every bare and prefixed masking of each set. Restoration is
safe only if the derived (common_path, names) is the same whichever replicas
arrive bare.

The datatrail CLI is faked at subprocess.run, the same boundary as
tests/archive/test_datatrail_adapter.py, so no test reaches the network. A
test skips itself when its bundle is not mounted.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt

# real-data inputs from the environment, skipped when absent
_BUNDLE = os.path.join(os.environ.get("PILOT_PROXY_ARCHIVE_BUNDLE", "archive_inputs/chime-pilots-v5"),
                       "inventory.source.jsonl")
_JULY = os.path.join(os.environ.get("PILOT_PROXY_JULY_INVENTORIES", "Datatrawl-Inventories"),
                     "chime-pilots", "inventory.jsonl")
_PREFIX = "cadc:CHIMEFRB/"


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


def _resolve(monkeypatch, uris):
    _install_fake_cli(monkeypatch, lambda a: (0, json.dumps(
        {"dataset": "d", "scope": "s",
         "files": {"file_replica_locations": {"minoc": list(uris)}},
         "policies": {"p": 1}}), ""))
    return dt.DATATRAIL.files("chime.event.baseband.raw", "E", retries=0)


def _row_name(row) -> str:
    """Return a row's file name, built from freq_id for July rows, which have no 'name'."""
    if "name" in row:
        return row["name"]
    return "baseband_%s_%s.h5" % (row["event"], row["freq_id"])


def _read(path, limit, per_event_cap=6, want=None):
    """Return ({key: [prefixed uri, ...]}, {key: recorded common_path})."""
    groups: dict = collections.OrderedDict()
    recorded: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["scope"], str(row["event"]))
            wanted = want is not None and key in want
            if key not in groups and len(groups) >= limit and not wanted:
                continue
            recorded.setdefault(key, row["common_path"])
            bucket = groups.setdefault(key, [])
            if len(bucket) < per_event_cap:
                bucket.append(row["common_path"].rstrip("/") + "/"
                              + _row_name(row))
    return groups, recorded


def _trailing_slash_events(path):
    out = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and json.loads(line)["common_path"].endswith("/"):
                row = json.loads(line)
                out.add((row["scope"], str(row["event"])))
    return out


@pytest.mark.skipif(not os.path.exists(_BUNDLE), reason="v5 bundle not mounted")
def test_derivation_is_invariant_under_which_replicas_arrive_bare(monkeypatch):
    groups, _ = _read(_BUNDLE, limit=400)
    assert len(groups) == 400, len(groups)
    checked = 0
    for key, uris in groups.items():
        assert all(u.startswith(_PREFIX) for u in uris), key
        baseline = _resolve(monkeypatch, uris)
        assert baseline[2] is True and baseline[0].startswith(_PREFIX), baseline
        n = len(uris)
        for mask in range(2 ** n):                # every bare/prefixed masking
            mixed = [u[len(_PREFIX):] if (mask >> i) & 1 else u
                     for i, u in enumerate(uris)]
            got = _resolve(monkeypatch, mixed)
            checked += 1
            assert got == baseline, (key, bin(mask), got, baseline)
    expected = sum(2 ** len(u) for u in groups.values())
    assert checked == expected == 25568, (checked, expected)
    assert min(len(u) for u in groups.values()) >= 1


@pytest.mark.skipif(not os.path.exists(_BUNDLE), reason="v5 bundle not mounted")
def test_all_bare_reproduces_the_bundle_common_path_exactly(monkeypatch):
    """With the prefix stripped from every replica, the adapter returns the
    common_path that the Aug-3 survey recorded before restoration was added,
    byte for byte."""
    groups, recorded = _read(_BUNDLE, limit=250)
    assert len(groups) == 250
    for key, uris in groups.items():
        cp, names, ok = _resolve(monkeypatch, [u[len(_PREFIX):] for u in uris])
        assert ok and cp == recorded[key], (key, cp, recorded[key])
        assert sorted(names) == sorted(u.rsplit("/", 1)[1] for u in uris), key


@pytest.mark.skipif(not os.path.exists(_JULY), reason="July inventory not mounted")
def test_july_era_common_paths_reproduce_modulo_a_pre_patch_trailing_slash(
        monkeypatch):
    """The July rows are re-derived in both the prefixed and the all-bare
    form. Any July value that the current adapter does not reproduce byte for
    byte differs only by a trailing slash, which is removed by
    `common.rstrip('/')`. The prefixed form is not changed by restoration and
    mismatches in the same way, so restoration did not cause the mismatch."""
    slashed = _trailing_slash_events(_JULY)
    assert len(slashed) == 43, len(slashed)
    groups, recorded = _read(_JULY, limit=300, want=slashed)
    assert slashed <= set(groups)
    exact = slash_only = 0
    slash_only_keys = set()
    for key, uris in groups.items():
        for form in (uris, [u[len(_PREFIX):] for u in uris]):
            cp, _names, ok = _resolve(monkeypatch, form)
            assert ok, key
            if cp == recorded[key]:
                exact += 1
            else:
                assert cp == recorded[key].rstrip("/"), (key, cp, recorded[key])
                slash_only += 1
                slash_only_keys.add(key)
    assert exact + slash_only == 2 * len(groups)
    # the events that need the rstrip are the 43 that July recorded with a
    # trailing slash, and each mismatches in both forms, so restoration did
    # not move them
    assert slash_only_keys == slashed, slash_only_keys ^ slashed
    assert slash_only == 2 * 43, slash_only
