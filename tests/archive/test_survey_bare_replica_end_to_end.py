"""End-to-end survey() coverage for the bare-minoc-replica patch, plus the
files() contract branches that no test in tests/archive reaches.

Two things measured coverage showed were missing (stdlib sys.monitoring line
trace over the whole tests/archive run, 2026-08-28):

1. Every survey()-level test fakes ``DATATRAIL.common_path`` itself
   (tests/archive/test_survey_contract_refusal.py:106,
   tests/archive/test_survey_empty_events.py, ...), i.e. they stub out the
   exact function the patch changed. Nothing drives survey() through the real
   ``files()`` normalization, so no test can observe what a restored bare
   replica does to the rows survey writes. The tests here fake only the one
   real seam -- ``dt.subprocess.run`` -- so ``_restore_collection`` runs for
   real.

2. datatrail_client.files() lines 388, 390-391, 397-398, 401 and 403-404 --
   the file_replica_locations/minoc shape branches -- were never executed.

Offline: no network, no live datatrail, no CADC.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.datatrail_client import DatatrailContractError
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as cadc_datatrail
from pilot_proxy.chime.baseband_reader import (MINIMUM_ARCHIVE_BYTES,
                                               baseband_filename)

SCOPE = "chime.event.baseband.raw"
EVENT = "100200200"
NEIGHBOUR = "100200201"
DAY = "data/chime/baseband/raw/2020/07/15"
EVENT_DIR = f"{DAY}/astro_{EVENT}"
FREQ_IDS = [0, 1]

# What actually exists in CADC for this event, as cadcinfo would answer.
ARCHIVE = {f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, fid)}"
           for fid in FREQ_IDS}


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_ps(monkeypatch, minoc):
    """Answer `datatrail ps ... --json` with this minoc replica list.

    Patches the adapter's ONE subprocess boundary, so files(),
    _restore_collection() and the common-path split all run for real.
    """
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))
    payload = json.dumps({
        "dataset": EVENT, "scope": SCOPE, "policies": {},
        "files": {"file_replica_locations": {"minoc": list(minoc)}},
    })
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))


def _fake_archive(monkeypatch):
    """cadcinfo: a real size for objects that exist, a definitive absence
    (size None, err None -- an answer, not an outage) for everything else."""
    seen = []

    def fake_size(self, uri, *a, **k):
        seen.append(uri)
        if uri in ARCHIVE:
            return MINIMUM_ARCHIVE_BYTES + 1, None
        return None, None

    monkeypatch.setattr(cadc_datatrail.CadcDatatrailSource,
                        "_cadc_size", fake_size)
    monkeypatch.setattr(cadc_datatrail, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, EVENT): ["ds"]})
    return seen


def _survey(out_dir, *, return_source=False):
    ctx = RunContext(instrument=None, selection=None,
                     options={"scope": SCOPE, "freq_ids": list(FREQ_IDS)})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        source = cadc_datatrail.CadcDatatrailSource()
        source.survey(ctx, str(out_dir))
    return (buf.getvalue(), source) if return_source else buf.getvalue()


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ==========================================================================
# The documented real case: every replica bare, all in the event directory.
# survey() must produce exactly the rows it produced before Datatrail
# dropped the prefix -- same common_path, same URIs.
# ==========================================================================
def test_all_bare_replicas_survey_to_the_same_rows(monkeypatch, tmp_path):
    _fake_ps(monkeypatch, [f"{EVENT_DIR}/{baseband_filename(EVENT, fid)}"
                           for fid in FREQ_IDS])
    probed = _fake_archive(monkeypatch)
    text = _survey(tmp_path)

    rows = _rows(tmp_path / "inventory.jsonl")
    assert [r["name"] for r in rows] == [baseband_filename(EVENT, fid)
                                         for fid in FREQ_IDS]
    assert {r["common_path"] for r in rows} == {f"cadc:CHIMEFRB/{EVENT_DIR}"}
    assert {r["obs_date"] for r in rows} == {"2020-07-15"}
    # every probe went to a URI that exists -- the restored prefix is the one
    # cadcinfo resolves, not merely one that passes the collection check
    assert set(probed) == ARCHIVE
    assert "0 contract-refused" in text
    assert "accepted-empty" in text and "0 accepted-empty" in text


@pytest.mark.parametrize("neighbour_prefix", ["", "cadc:CHIMEFRB/"])
def test_sibling_directory_replicas_are_refused(monkeypatch, tmp_path,
                                               neighbour_prefix):
    _fake_ps(monkeypatch, [
        f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, 0)}",
        f"{neighbour_prefix}{DAY}/astro_{NEIGHBOUR}/{baseband_filename(NEIGHBOUR, 1)}",
    ])
    probed = _fake_archive(monkeypatch)
    with pytest.raises(DatatrailContractError, match="multiple directories"):
        dt.Datatrail().common_path(SCOPE, EVENT)
    text = _survey(tmp_path)
    assert "1 contract-refused" in text
    assert not probed
    assert _rows(tmp_path / "inventory.jsonl") == []
    ledger = _rows(tmp_path / "no_files_events.jsonl")
    assert [row["reason"] for row in ledger] == ["datatrail-contract-refusal"]
    assert "multiple directories" in ledger[0]["detail"]


def test_sibling_directory_refusal_stays_visible_until_resurveyed(
        monkeypatch, tmp_path):
    _fake_ps(monkeypatch, [
        f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, 0)}",
        f"{DAY}/astro_{NEIGHBOUR}/{baseband_filename(NEIGHBOUR, 1)}",
    ])
    probed = _fake_archive(monkeypatch)
    assert "1 contract-refused" in _survey(tmp_path)
    _, source = _survey(tmp_path, return_source=True)
    assert not probed
    assert source.survey_completeness_issues(str(tmp_path))["refused"] == 1
    assert [row["reason"] for row in _rows(tmp_path / "no_files_events.jsonl")] == [
        "datatrail-contract-refusal"]

    _fake_ps(monkeypatch, [f"{EVENT_DIR}/{baseband_filename(EVENT, fid)}"
                           for fid in FREQ_IDS])
    retry = tmp_path / "retry"
    _survey(retry)
    assert len(_rows(retry / "inventory.jsonl")) == len(FREQ_IDS)
    assert set(probed) == ARCHIVE
    assert _rows(retry / "no_files_events.jsonl") == []


# ==========================================================================
# files(): the file_replica_locations / minoc shape branches. Measured as
# never executed by tests/archive before this file
# (datatrail_client.py:388, 390-391, 397-398, 401, 403-404).
# ==========================================================================
def _files(monkeypatch, files_value):
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))
    payload = json.dumps({"dataset": "d", "scope": "s", "policies": {},
                          "files": files_value})
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))
    return dt.DATATRAIL.files("s", "d", retries=0)


def test_absent_replica_locations_is_no_data_not_a_refusal(monkeypatch):
    # line 388: "file_replica_locations": null -> queried OK, no bytes
    assert _files(monkeypatch, {"file_replica_locations": None}) == (
        None, [], True)


def test_empty_minoc_list_is_no_data_not_a_refusal(monkeypatch):
    # line 401: an empty list is an answer, never an outage
    assert _files(monkeypatch, {"file_replica_locations": {"minoc": []}}) == (
        None, [], True)


@pytest.mark.parametrize("locations,fragment", [
    (["cadc:CHIMEFRB/data/x/y.h5"], "non-object 'file_replica_locations'"),
    ("cadc:CHIMEFRB/data/x/y.h5", "non-object 'file_replica_locations'"),
])
def test_non_object_replica_locations_refuses(monkeypatch, locations,
                                              fragment):
    # lines 390-391
    with pytest.raises(DatatrailContractError, match=fragment):
        _files(monkeypatch, {"file_replica_locations": locations})


def test_non_list_minoc_refuses(monkeypatch):
    # lines 397-398: a bare string would otherwise be iterated per-character
    with pytest.raises(DatatrailContractError, match="non-list 'minoc'"):
        _files(monkeypatch, {"file_replica_locations":
                             {"minoc": "cadc:CHIMEFRB/data/x/y.h5"}})


@pytest.mark.parametrize("bad", [None, "", "   ", 7, ["nested"], {}])
def test_malformed_minoc_entry_refuses(monkeypatch, bad):
    # lines 403-404: one bad entry poisons the list; never a partial answer
    with pytest.raises(DatatrailContractError, match="malformed 'minoc'"):
        _files(monkeypatch, {"file_replica_locations": {"minoc": [
            "cadc:CHIMEFRB/data/x/y.h5", bad]}})
