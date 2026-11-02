#!/usr/bin/env python3
"""Survey-level consequences of _restore_collection.

The tests of tests/archive/test_survey_contract_refusal.py fake
DATATRAIL.common_path. These tests fake only `_run_json`, the datatrail
HTTP/CLI boundary, and `_cadc_size`, the CADC boundary, so the real
normalization, the real stray and span checks and the real commonprefix
derivation all run.

Two results are tested:

  A. An all-bare event in a single directory, which is refused without
     restoration, surveys to completion with the correct rows.

  B. A replica set that spans directories is still refused after the
     collection prefixes are restored, and is never treated as missing data.

Run:  TMPDIR=/tmp PYTHONPATH=src python -m pytest \
      tests/archive/test_survey_bare_replica_consequences.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile


from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as cadc_datatrail
from pilot_proxy.chime.baseband_reader import MINIMUM_ARCHIVE_BYTES

SCOPE = "chime.event.baseband.raw"
FREQ_IDS = [614, 706]

BARE_EVENT = "1111116060"                 # every replica bare (the 2025 case)
BARE_DIR = "data/chime/baseband/raw/2025/03/31/astro_1111116060"
BARE_CP = "cadc:CHIMEFRB/" + BARE_DIR

MIXED_EVENT = "100200200"                 # replicas from two directories
MIXED_URIS = [
    "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_100200200/"
    "baseband_100200200_614.h5",
    "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
    "baseband_1111116060_706.h5",
]
MIXED_DERIVED_CP = "cadc:CHIMEFRB/data/chime/baseband/raw"

_ORIGINAL_RESTORE = dt._restore_collection


def _replicas(event):
    if event == BARE_EVENT:
        return [f"{BARE_DIR}/baseband_{BARE_EVENT}_{fid}.h5"
                for fid in FREQ_IDS]
    return list(MIXED_URIS)


@contextlib.contextmanager
def fake_archive(events, unpatched=False):
    """Fake only the two external boundaries, so the adapter itself runs."""
    orig_enum = cadc_datatrail._enumerate_events
    orig_size = cadc_datatrail.CadcDatatrailSource._cadc_size
    orig_restore = dt._restore_collection
    orig_run_json = dt._run_json

    def fake_run_json(args, **kw):
        assert args[0] == "ps", args
        return ({"dataset": args[2], "scope": args[1],
                 "files": {"file_replica_locations":
                           {"minoc": _replicas(args[2])}},
                 "policies": {"p": 1}}, "")

    # The fake CADC holds exactly the artifacts named by the replica lists,
    # with the collection prefix restored. Any other URI is a definitive
    # NotFound, which _cadc_size reports as the answer (None, None) with no
    # error.
    real = set()
    for ev in events:
        for u in _replicas(ev):
            real.add(u if u.startswith("cadc:") else "cadc:CHIMEFRB/" + u)

    def fake_size(self, uri, *a, **k):
        return (MINIMUM_ARCHIVE_BYTES + 1, None) if uri in real else (None, None)

    cadc_datatrail._enumerate_events = lambda *a, **k: {
        (SCOPE, ev): ["ds"] for ev in events}
    cadc_datatrail.CadcDatatrailSource._cadc_size = fake_size
    dt._run_json = fake_run_json
    if unpatched:
        dt._restore_collection = lambda u: u
    try:
        yield
    finally:
        cadc_datatrail._enumerate_events = orig_enum
        cadc_datatrail.CadcDatatrailSource._cadc_size = orig_size
        dt._restore_collection = orig_restore
        dt._run_json = orig_run_json


def _survey(out_dir):
    ctx = RunContext(instrument=None, selection=None,
                     options={"freq_ids": list(FREQ_IDS)})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        src = cadc_datatrail.CadcDatatrailSource()
        src.survey(ctx, out_dir)
    return buf.getvalue(), src


def _rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


# A. an all-bare event surveys to completion
def test_all_bare_event_surveys_to_completion_post_patch():
    with tempfile.TemporaryDirectory() as out, fake_archive([BARE_EVENT]):
        text, src = _survey(out)
        rows = _rows(os.path.join(out, "inventory.jsonl"))
        assert len(rows) == len(FREQ_IDS), rows
        assert {r["common_path"] for r in rows} == {BARE_CP}
        assert {r["obs_date"] for r in rows} == {"2025-03-31"}
        assert sorted(r["name"] for r in rows) == [
            f"baseband_{BARE_EVENT}_{fid}.h5" for fid in sorted(FREQ_IDS)]
        assert "contract refusal" not in text
        assert src.survey_completeness_issues(out) == {
            "incomplete": 0, "refused": 0, "pending": 0}


def test_all_bare_event_was_refused_pre_patch():
    with tempfile.TemporaryDirectory() as out, fake_archive([BARE_EVENT],
                                                            unpatched=True):
        text, src = _survey(out)
        assert _rows(os.path.join(out, "inventory.jsonl")) == []
        assert "contract refusal -- recorded and skipped" in text
        ledger = _rows(os.path.join(out, "no_files_events.jsonl"))
        assert [r["reason"] for r in ledger] == ["datatrail-contract-refusal"]
        assert "2/2 replicas affected" in ledger[0]["detail"]
        assert src.survey_completeness_issues(out)["refused"] == 1


# B. mixed directories remain a contract refusal
def test_mixed_directory_event_was_a_clean_first_pass_refusal_pre_patch():
    with tempfile.TemporaryDirectory() as out, fake_archive([MIXED_EVENT],
                                                            unpatched=True):
        text, src = _survey(out)
        ledger = _rows(os.path.join(out, "no_files_events.jsonl"))
        assert [r["reason"] for r in ledger] == ["datatrail-contract-refusal"]
        assert MIXED_URIS[1] in ledger[0]["detail"]
        assert ledger[0]["common_path"] is None
        assert src.survey_completeness_issues(out)["refused"] == 1


def test_mixed_directory_event_is_refused_after_collection_restoration():
    with tempfile.TemporaryDirectory() as out, fake_archive([MIXED_EVENT]):
        text, src = _survey(out)
        assert "contract refusal -- recorded and skipped" in text
        ledger = _rows(os.path.join(out, "no_files_events.jsonl"))
        assert _rows(os.path.join(out, "inventory.jsonl")) == []
        assert len(ledger) == 1, ledger
        entry = ledger[0]
        assert entry["reason"] == "datatrail-contract-refusal", entry
        assert "multiple directories" in entry["detail"]
        assert entry["common_path"] is None
        assert entry["obs_date"] == "unknown", entry
        assert entry["attempts"] == 1
        assert src.survey_completeness_issues(out)["refused"] == 1


def test_the_derived_path_is_not_where_either_replica_lives():
    """The derived common path, checked here with no call to survey(), is a
    strict prefix of both replica directories, so a reader-supplied name
    joined to it can only miss."""
    for u in MIXED_URIS:
        full = u if u.startswith("cadc:") else "cadc:CHIMEFRB/" + u
        assert full.startswith(MIXED_DERIVED_CP + "/")
        assert os.path.dirname(full) != MIXED_DERIVED_CP
