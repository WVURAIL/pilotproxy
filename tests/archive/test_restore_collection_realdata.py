#!/usr/bin/env python3
"""Replica-restoration invariants from the retained v5 inventory.

Rebuild minoc replica lists from recorded common paths and filenames, then
derive the paths and names for every bare/prefixed masking of each sampled
set. The result must be invariant under which replicas arrive bare.

No network: the datatrail CLI is faked at subprocess.run, same boundary as
tests/archive/test_datatrail_adapter.py.

Skips itself when a bundle is not mounted.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt

# Real-data inputs, read from the environment; the tests skip when they are absent.
_BUNDLE = os.path.join(os.environ.get("PILOT_PROXY_ARCHIVE_BUNDLE", "archive_inputs/chime-pilots-v5"),
                       "inventory.source.jsonl")
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


def _read(path, limit, per_event_cap=6):
    """({key: [prefixed uri, ...]}, {key: recorded common_path})."""
    groups: dict = collections.OrderedDict()
    recorded: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["scope"], str(row["event"]))
            if key not in groups and len(groups) >= limit:
                continue
            recorded.setdefault(key, row["common_path"])
            bucket = groups.setdefault(key, [])
            if len(bucket) < per_event_cap:
                bucket.append(row["common_path"].rstrip("/") + "/"
                              + row["name"])
    return groups, recorded


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
    """Bare replicas must reproduce the retained common paths and names."""
    groups, recorded = _read(_BUNDLE, limit=250)
    assert len(groups) == 250
    for key, uris in groups.items():
        cp, names, ok = _resolve(monkeypatch, [u[len(_PREFIX):] for u in uris])
        assert ok and cp == recorded[key], (key, cp, recorded[key])
        assert sorted(names) == sorted(u.rsplit("/", 1)[1] for u in uris), key
