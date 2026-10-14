#!/usr/bin/env python3
"""
Failure-injection stress tests for survey resume and durable state -- the
paths that only exercise when a real run is interrupted, resumed, or handed a
damaged output directory, which on CANFAR means "after the fact, in
production".

Measured with a stdlib line tracer over a full clean survey run (2026-08-28):
one uninterrupted survey executes 197 of survey_state.py's 292 executable
lines. Every integrity guard in SurveyStore.__init__ (a junk database, table
column drift, an unparseable or unsupported schema row), every branch of
load_attempts(), the manifest corruption/legacy refusals, the Windows lock
branch, and the periodic view flush at VIEW_FLUSH_INTERVAL were all cold.

Offline. The seams faked are exactly the two boundaries a survey has --
`datatrail_client.subprocess.run` (the CLI child) and
`CadcDatatrailSource._cadc_size` (the CADC probe) -- plus
`cadc._enumerate_events` where the test is not about enumeration. Everything
else (SurveyStore's transaction, SurveyOutputLock, render_views,
_commit_decision, the real reader) runs for real.

Interruption is injected on the MAIN thread from inside the fake CLI handler,
which survey() calls at cadc.py:616 before it touches the probe pool. A real
signal is delivered with signal.raise_signal(), so the default SIGINT handler
raises KeyboardInterrupt exactly where a Ctrl-C would -- deterministically,
with no sleeps and no subprocesses.

Run:
  TMPDIR=/tmp PYTHONPATH=src python -m pytest tests/archive/test_survey_stress_resume.py
"""
from __future__ import annotations

import contextlib
import datetime
import errno
import io
import json
import os
import signal
import sqlite3
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive import survey_state as state_mod
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as cadc_src
from pilot_proxy.archive.sources import cadc_inventory as inv
from pilot_proxy.archive.survey_state import (SurveyOutputLock, SurveyStore,
                                              load_attempts)
from pilot_proxy.chime.baseband_reader import (MINIMUM_ARCHIVE_BYTES,
                                               baseband_filename)


SCOPE = "chime.event.baseband.raw"
FREQ_IDS = [0, 1, 2]
BIG = MINIMUM_ARCHIVE_BYTES + 4096
EVENTS = [str(100000000 + i) for i in range(10)]


# ==========================================================================
# fakes -- same shape as tests/archive/test_datatrail_adapter.py so the two
# cannot drift.
# ==========================================================================
class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


class _MisWiredFake(BaseException):
    """Raised as a BaseException on purpose: _run_json catches Exception and
    reports it as "did not answer", which survey escalates to a one-hour
    outage circuit -- so an ordinary AssertionError from a broken fake would
    hang the suite instead of failing it."""


def _old_day() -> str:
    """A day directory old enough that an all-absent event is written off on
    first sighting (>= _EMPTY_TERMINAL_AGE_DAYS)."""
    day = (datetime.datetime.now(datetime.timezone.utc).date()
           - datetime.timedelta(days=400))
    return f"data/chime/baseband/raw/{day:%Y/%m/%d}"


def _cp(event) -> str:
    return f"cadc:CHIMEFRB/{_old_day()}/astro_{event}"


def _install_cli(monkeypatch, *, on_ps=None, freq_ids=FREQ_IDS):
    """Answer `datatrail ps` with a full replica list for every event.

    `on_ps(event, n)` is called on the MAIN thread before the reply is built
    (n is the 1-based ps call count), which is where interruption is injected.
    """
    seen: list = []
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))
    monkeypatch.setattr(dt, "time", types.SimpleNamespace(
        monotonic=dt.time.monotonic, sleep=lambda s: None))

    def fake_run(cmd, **kw):
        if cmd[:3] != [sys.executable, "-m", "dtcli.cli"] or cmd[-1] != "--json":
            raise _MisWiredFake(f"invocation contract broken: {cmd}")
        args = list(cmd[3:-1])
        if args[0] != "ps":
            raise _MisWiredFake(f"unexpected datatrail subcommand: {args}")
        event = args[2]
        seen.append(event)
        if on_ps is not None:
            on_ps(event, len(seen))
        minoc = [f"{_cp(event)}/{baseband_filename(event, f)}"
                 for f in freq_ids]
        return _Proc(0, json.dumps({
            "dataset": event, "scope": args[1], "policies": {},
            "files": {"file_replica_locations": {"minoc": minoc}}}), "")

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    return seen


def _install_archive(monkeypatch, *, absent=(), error_for=()):
    """cadcinfo. Sizes are derived from the URI, so an inventory's bytes are a
    pure function of the event set and never of scheduling or run order.

    `absent` answers a definitive absence (cadc.py:473-474 reports NotFound as
    an ANSWER); `error_for` raises a hard probe error, which is what drives the
    partial / retry / terminal-incomplete ladder.
    """
    probed: list = []
    absent, error_for = set(absent), set(error_for)

    def fake_size(self, uri, *a, **k):
        probed.append(uri)
        if uri in error_for:
            return None, OSError("archive probe failed")
        if uri in absent:
            return None, None
        return BIG + (len(uri) % 97), None
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size", fake_size)
    return probed


def _install_enumeration(monkeypatch, events):
    monkeypatch.setattr(cadc_src, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, ev): ["ds"] for ev in events})


def _survey(out_dir, *, freq_ids=FREQ_IDS, reader=None, **options):
    opts = {"scope": SCOPE, "freq_ids": list(freq_ids)}
    opts.update(options)
    ctx = RunContext(instrument=None, selection=None, options=opts,
                     reader=reader)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cadc_src.CadcDatatrailSource().survey(ctx, str(out_dir))
    return buf.getvalue()


def _lines(path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _db(out):
    """(status by event_key, record count by event_key) straight from SQLite.

    The JSONL/text files are only VIEWS (survey_state.py:385-408), so a test
    that reads only the views cannot tell a missing commit from a missing
    render.
    """
    con = sqlite3.connect(out / "survey_state.sqlite3")
    try:
        status = {k: s for k, s in
                  con.execute("SELECT event_key, status FROM events")}
        counts = {k: n for k, n in con.execute(
            "SELECT event_key, COUNT(*) FROM records GROUP BY event_key")}
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    return status, counts, integrity


def _attempts(out):
    path = out / "attempts.json"
    return json.loads(path.read_text() or "{}") if path.exists() else {}


# ##########################################################################
# 1. SIGINT: a real signal, delivered where a Ctrl-C would land.
# ##########################################################################
def test_sigint_mid_run_commits_whole_events_only(monkeypatch, tmp_path):
    # The interrupt lands on the main thread inside the ps reply for the 6th
    # event, i.e. AFTER five events were committed and BEFORE the sixth is
    # probed. What must hold: no event is half-written (records per event is
    # exactly the candidate count, never a partial), the views were still
    # rendered by the finally block (cadc.py:827-848), and the output lock was
    # released so the directory is not stranded.
    def interrupt(event, n):
        if n == 6:
            signal.raise_signal(signal.SIGINT)

    _install_cli(monkeypatch, on_ps=interrupt)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS)

    with pytest.raises(KeyboardInterrupt):
        _survey(tmp_path)

    status, counts, integrity = _db(tmp_path)
    assert sorted(status) == [f"{SCOPE}|{ev}" for ev in EVENTS[:5]]
    assert set(status.values()) == {"complete"}
    assert set(counts.values()) == {len(FREQ_IDS)}      # no partial event
    assert integrity == "ok"
    assert len(_lines(tmp_path / "inventory.jsonl")) == 5 * len(FREQ_IDS)
    assert len(_lines(tmp_path / "surveyed_events.txt")) == 5
    # the lock is released: a fresh acquisition must not block.
    with SurveyOutputLock(tmp_path):
        pass


def test_resume_after_an_interrupt_re_probes_nothing_and_duplicates_nothing(
        monkeypatch, tmp_path):
    def interrupt(event, n):
        if n == 4:
            signal.raise_signal(signal.SIGINT)

    _install_cli(monkeypatch, on_ps=interrupt)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS)
    with pytest.raises(KeyboardInterrupt):
        _survey(tmp_path)
    prefix = _lines(tmp_path / "inventory.jsonl")

    seen = _install_cli(monkeypatch)                  # no interrupt this time
    probed = _install_archive(monkeypatch)
    text = _survey(tmp_path)

    assert "resume: 3 events already done" in text
    assert seen == EVENTS[3:]                          # committed events untouched
    assert not any(ev in uri for uri in probed for ev in EVENTS[:3])
    rows = _lines(tmp_path / "inventory.jsonl")
    assert rows[:len(prefix)] == prefix                # the prefix is byte-stable
    assert len(rows) == len(EVENTS) * len(FREQ_IDS)
    assert len(set(rows)) == len(rows)                 # no duplicated rows


def test_an_interrupt_leaves_the_attempt_checkpoint_consistent(monkeypatch,
                                                               tmp_path):
    # Interrupting between a commit and its checkpoint rewrite is a real
    # window (cadc.py:601 -> :605). Whatever survives must never claim a
    # pending retry for an event the database already committed.
    def interrupt(event, n):
        if n == 3:
            signal.raise_signal(signal.SIGINT)

    _install_cli(monkeypatch, on_ps=interrupt)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:5])
    with pytest.raises(KeyboardInterrupt):
        _survey(tmp_path)
    status, _counts, _ = _db(tmp_path)
    assert set(_attempts(tmp_path)) & set(status) == set()


# ##########################################################################
# 2. A hard kill: state committed, views not yet rendered.
#
# Constructed rather than signalled, so the window is exact and the test is
# deterministic: a SIGKILL is only a way of reaching this state.
# ##########################################################################
def test_a_killed_run_leaves_a_truthful_looking_but_stale_inventory(
        monkeypatch, tmp_path):
    seen = _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:4])
    _survey(tmp_path)
    full = _lines(tmp_path / "inventory.jsonl")
    assert len(full) == 4 * len(FREQ_IDS)

    # what a kill between the last commit and the next render looks like on
    # disk: SQLite is complete, the view is short -- and NOTHING says so.
    (tmp_path / "inventory.jsonl").write_text("\n".join(full[:3]) + "\n")
    (tmp_path / "surveyed_events.txt").write_text(f"{SCOPE}|{EVENTS[0]}\n")
    stale = _lines(tmp_path / "inventory.jsonl")
    assert len(stale) == 3                     # parses fine, looks plausible
    status, counts, _ = _db(tmp_path)
    assert sum(counts.values()) == 12          # ... and disagrees with the DB

    # the next run repairs both views before doing any new work
    # (cadc.py:585) and re-probes nothing.
    seen.clear()
    probed = _install_archive(monkeypatch)
    _survey(tmp_path)
    assert seen == [] and probed == []
    assert _lines(tmp_path / "inventory.jsonl") == full
    assert len(_lines(tmp_path / "surveyed_events.txt")) == 4


def test_active_commits_mark_views_stale_until_final_render(monkeypatch, tmp_path):
    from pilot_proxy.archive.survey_state import SurveyStore, STALE_VIEWS_MARKER
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    original = SurveyStore.commit
    observed = []
    def commit(store, *args, **kwargs):
        observed.append((tmp_path / STALE_VIEWS_MARKER).exists())
        return original(store, *args, **kwargs)
    monkeypatch.setattr(SurveyStore, "commit", commit)
    _survey(tmp_path)
    assert observed == [True, True]
    assert not (tmp_path / STALE_VIEWS_MARKER).exists()


def test_deleting_the_views_never_loses_a_row(monkeypatch, tmp_path):
    # The four JSONL/text files are pure functions of the database. Deleting
    # them mid-life must be fully recoverable, and a hand-edit must not be.
    _install_cli(monkeypatch)
    _install_archive(monkeypatch,
                     absent={f"{_cp(EVENTS[1])}/{baseband_filename(EVENTS[1], f)}"
                             for f in FREQ_IDS})
    _install_enumeration(monkeypatch, EVENTS[:3])
    _survey(tmp_path)
    views = {name: (tmp_path / name).read_bytes()
             for name in ("inventory.jsonl", "surveyed_events.txt",
                          "incomplete_events.txt", "no_files_events.jsonl")}
    assert views["no_files_events.jsonl"]              # the empty event ledger

    for name in views:
        (tmp_path / name).unlink()
    with open(tmp_path / "no_files_events.jsonl", "w") as handle:
        handle.write('{"hand": "written"}\n')

    probed = _install_archive(monkeypatch)
    _survey(tmp_path)
    assert probed == []                                # nothing re-probed
    for name, expected in views.items():
        assert (tmp_path / name).read_bytes() == expected, name


def test_the_views_are_repaired_before_any_new_work_begins(monkeypatch,
                                                           tmp_path):
    # cadc.py:585 renders the views at STARTUP, before phase 2. A run that
    # merely renders at the end would still leave the directory correct after
    # a clean exit, but wrong for the entire duration of a long resumed run --
    # and an operator watching inventory.jsonl would see rows vanish. Observed
    # from inside the first ps reply of the resumed run, which is the earliest
    # main-thread point after the startup render.
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    recovered = _lines(tmp_path / "inventory.jsonl")
    assert len(recovered) == 2 * len(FREQ_IDS)
    (tmp_path / "inventory.jsonl").unlink()
    (tmp_path / "surveyed_events.txt").unlink()

    seen_at_first_probe: list = []
    _install_cli(monkeypatch, on_ps=lambda ev, n: seen_at_first_probe.append(
        len(_lines(tmp_path / "inventory.jsonl"))))
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:3])      # one NEW event
    _survey(tmp_path)

    assert seen_at_first_probe == [len(recovered)]
    assert _lines(tmp_path / "inventory.jsonl")[:len(recovered)] == recovered
    assert len(_lines(tmp_path / "surveyed_events.txt")) == 3


def test_view_order_is_by_event_key_not_by_commit_order(monkeypatch, tmp_path):
    # survey_state.py:389-390 orders by (event_key, ordinal). Within one run
    # that is indistinguishable from insertion order, because survey() iterates
    # `sorted(events)` -- so only a RESUME can tell the two apart. Make the
    # LOWEST-keyed event commit LAST by failing one of its probes twice, and
    # the rendered inventory must still lead with it.
    low, mid, high = EVENTS[0], EVENTS[1], EVENTS[2]
    flaky = f"{_cp(low)}/{baseband_filename(low, FREQ_IDS[0])}"
    _install_cli(monkeypatch)
    _install_enumeration(monkeypatch, [low, mid, high])
    for _ in range(2):
        _install_archive(monkeypatch, error_for={flaky})
        _survey(tmp_path)
    status, _counts, _ = _db(tmp_path)
    assert sorted(status) == [f"{SCOPE}|{mid}", f"{SCOPE}|{high}"]

    _install_archive(monkeypatch)                     # the flake clears
    _survey(tmp_path)
    rows = [json.loads(line) for line in _lines(tmp_path / "inventory.jsonl")]
    assert len(rows) == 3 * len(FREQ_IDS)
    # insertion order was mid, high, low; key order is low, mid, high.
    assert [r["event"] for r in rows][::len(FREQ_IDS)] == [low, mid, high]
    assert _lines(tmp_path / "surveyed_events.txt") == [
        f"{SCOPE}|{ev}" for ev in (low, mid, high)]


def test_the_periodic_view_flush_happens_at_the_documented_interval(
        monkeypatch, tmp_path):
    # cadc.py:607-608 renders every VIEW_FLUSH_INTERVAL commits. Measured
    # coverage showed this is the ONLY line a 250-event run adds over a
    # 3-event one, so no small test can reach it. Observe it from inside the
    # 101st event's ps reply: exactly 100 events are committed at that moment.
    assert state_mod.VIEW_FLUSH_INTERVAL == 100
    many = [str(100000000 + i) for i in range(101)]
    observed: list = []

    def watch(event, n):
        observed.append((n, len(_lines(tmp_path / "inventory.jsonl"))))

    _install_cli(monkeypatch, on_ps=watch, freq_ids=[0])
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, many)
    _survey(tmp_path, freq_ids=[0])

    assert observed[0] == (1, 0)          # startup render of an empty state
    assert observed[99] == (100, 0)       # 99 committed: below the interval
    assert observed[100] == (101, 100)    # the flush fired at the 100th commit
    assert len(_lines(tmp_path / "inventory.jsonl")) == 101


# ##########################################################################
# 3. The output lock.
# ##########################################################################
def test_a_lock_file_naming_a_dead_process_does_not_strand_the_directory(
        monkeypatch, tmp_path):
    # flock is advisory and kernel-released, so a lock FILE left by a killed
    # run must not block. Refusing here would let one SIGKILL permanently
    # strand an output directory with no documented recovery.
    (tmp_path / ".survey.lock").write_text(json.dumps({"pid": 999999}) + "\n")
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    assert json.loads((tmp_path / ".survey.lock").read_text())["pid"] == os.getpid()
    assert len(_lines(tmp_path / "inventory.jsonl")) == 2 * len(FREQ_IDS)


def test_a_live_holder_refuses_a_second_survey_and_changes_no_state(
        monkeypatch, tmp_path):
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*")
              if p.is_file() and p.name != ".survey.lock"}

    probed = _install_archive(monkeypatch)
    with SurveyOutputLock(tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            _survey(tmp_path)
    message = str(excinfo.value)
    assert "already in use by another active survey" in message
    assert "choose a different --name/output directory" in message
    assert probed == []
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*")
            if p.is_file() and p.name != ".survey.lock"} == before


def test_the_lock_file_is_not_treated_as_legacy_survey_state(tmp_path):
    # SurveyOutputLock.__enter__ creates .survey.lock BEFORE ensure_manifest
    # runs. If that name were ever added to _STATE_NAMES the legacy-state
    # guard (survey_state.py:261-266) would refuse EVERY fresh directory.
    assert ".survey.lock" not in state_mod._STATE_NAMES
    with SurveyOutputLock(tmp_path):
        pass
    state_mod.ensure_manifest(tmp_path, {"schema": 1, "probe": True})
    assert (tmp_path / "survey_manifest.json").exists()


def test_exiting_the_lock_twice_is_a_no_op(tmp_path):
    lock = SurveyOutputLock(tmp_path)
    lock.__enter__()
    assert lock.__exit__(None, None, None) is False
    assert lock.__exit__(None, None, None) is False


def test_the_windows_lock_branch_refuses_a_second_holder(monkeypatch,
                                                         tmp_path):
    # survey_state.py:72-78 and :113-116 are the msvcrt branches. CI is POSIX
    # only, so they have zero coverage; a syntax-level or API-level mistake
    # there would ship unnoticed. Build the Path objects BEFORE os.name is
    # patched -- pathlib picks its flavour from os.name at instantiation.
    first, second = SurveyOutputLock(tmp_path), SurveyOutputLock(tmp_path)
    held: list = []

    class _Msvcrt(types.ModuleType):
        LK_NBLCK, LK_UNLCK = 1, 0

        @staticmethod
        def locking(fileno, mode, nbytes):
            if mode == 1:
                if held:
                    raise OSError(errno.EACCES, "locked")
                held.append(fileno)
            else:
                held.clear()

    monkeypatch.setitem(sys.modules, "msvcrt", _Msvcrt("msvcrt"))
    monkeypatch.setattr(state_mod.os, "name", "nt")
    with first:
        assert held                                   # LK_NBLCK was taken
        with pytest.raises(SystemExit) as excinfo:
            second.__enter__()
        assert "already in use by another active survey" in str(excinfo.value)
    assert held == []                                 # LK_UNLCK on the way out


# ##########################################################################
# 4. attempts.json -- the cross-resume retry checkpoint.
# ##########################################################################
_ATTEMPT_CORRUPTIONS = [
    ("array", "[]", "survey attempt state is corrupt"),
    ("string", '"x"', "survey attempt state is corrupt"),
    ("not-json", "not json at all", "survey attempt state is corrupt"),
    ("no-pipe-key", '{"nokey": 1}', "invalid event keys"),
    ("empty-key", '{"": 1}', "invalid event keys"),
    ("bool-count", '{"a|b": true}', "non-integer counts"),
    ("float-count", '{"a|b": 1.5}', "non-integer counts"),
    ("negative-count", '{"a|b": -1}', "negative counts"),
]


@pytest.mark.parametrize("case_id,body,fragment", _ATTEMPT_CORRUPTIONS,
                         ids=[c[0] for c in _ATTEMPT_CORRUPTIONS])
def test_a_corrupt_attempt_checkpoint_refuses_the_run(tmp_path, case_id, body,
                                                      fragment):
    path = tmp_path / "attempts.json"
    path.write_text(body)
    with pytest.raises(SystemExit) as excinfo:
        load_attempts(path)
    assert fragment in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_a_valid_attempt_checkpoint_loads(tmp_path):
    (tmp_path / "attempts.json").write_text('{"s|e": 2}')
    assert load_attempts(tmp_path / "attempts.json") == {"s|e": 2}
    assert load_attempts(tmp_path / "missing.json") == {}


def test_a_corrupt_attempt_checkpoint_stops_a_real_survey(monkeypatch,
                                                          tmp_path):
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    (tmp_path / "attempts.json").write_text('{"chime|1": -4}')
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path)
    assert "negative counts" in str(excinfo.value)


def test_resume_drops_the_attempt_count_of_an_already_committed_event(
        monkeypatch, tmp_path):
    # cadc.py:580-582. A crash can land after store.commit() and before the
    # checkpoint rewrite, leaving a count for a finished event. The committed
    # event wins; the stale count is discarded, and OTHER counts survive.
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    key = f"{SCOPE}|{EVENTS[0]}"
    (tmp_path / "attempts.json").write_text(
        json.dumps({key: 2, f"{SCOPE}|999999999": 1}))

    _survey(tmp_path)
    assert _attempts(tmp_path) == {f"{SCOPE}|999999999": 1}


def test_the_attempt_checkpoint_never_prunes_a_vanished_event(monkeypatch,
                                                              tmp_path):
    # CHARACTERIZATION of a slow leak. Only keys present in `surveyed` are
    # popped, so counts for events that leave the enumeration are kept
    # forever -- and attempts.json is rewritten and fsynced on EVERY event
    # (cadc.py:605, :612), so a large pending set is quadratic work.
    absent = {f"{_cp(ev)}/{baseband_filename(ev, f)}"
              for ev in EVENTS[:3] for f in FREQ_IDS}
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, absent=absent)
    _install_enumeration(monkeypatch, EVENTS[:3])
    _survey(tmp_path, empty_age_days=99999)      # never write off: keep pending
    pending = _attempts(tmp_path)
    assert set(pending) == {f"{SCOPE}|{ev}" for ev in EVENTS[:3]}

    _install_enumeration(monkeypatch, [])        # the archive dropped them
    _survey(tmp_path, empty_age_days=99999, re_enumerate=True)
    assert _attempts(tmp_path) == pending        # still there, unreferenced


# ##########################################################################
# 5. survey_manifest.json -- the configuration compatibility guard.
# ##########################################################################
def _seed(monkeypatch, out, **options):
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    return _survey(out, **options)


def test_a_corrupt_manifest_refuses_rather_than_re_fingerprinting(monkeypatch,
                                                                  tmp_path):
    _seed(monkeypatch, tmp_path)
    before = (tmp_path / "survey_state.sqlite3").read_bytes()
    (tmp_path / "survey_manifest.json").write_text("{not json")
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path)
    assert "survey manifest is corrupt" in str(excinfo.value)
    assert "existing state was not changed" in str(excinfo.value)
    assert (tmp_path / "survey_state.sqlite3").read_bytes() == before


def test_a_manifest_whose_fingerprint_does_not_match_its_body_is_corrupt(
        monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    manifest = json.loads((tmp_path / "survey_manifest.json").read_text())
    manifest["configuration"]["empty_age_days"] = 7      # body edited in place
    (tmp_path / "survey_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path)
    assert "survey manifest is corrupt" in str(excinfo.value)


def test_state_without_a_manifest_is_refused_as_legacy(monkeypatch, tmp_path):
    _seed(monkeypatch, tmp_path)
    (tmp_path / "survey_manifest.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path)
    message = str(excinfo.value)
    assert "predates configuration manifests" in message
    # the refusal names every state file it found, so the operator can see why
    for name in ("attempts.json", "inventory.jsonl", "survey_state.sqlite3"):
        assert name in message


def test_a_changed_reader_shape_is_refused_on_resume(monkeypatch, tmp_path):
    # The reader owns the archive file shape, so bumping survey_schema (or
    # swapping the reader class) changes what a row MEANS. Mixing both in one
    # inventory is exactly what the fingerprint exists to prevent.
    from pilot_proxy.chime.baseband_reader import ChimeBasebandReader

    class _Bumped(ChimeBasebandReader):
        survey_schema = 3

    _seed(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path, reader=_Bumped())
    assert "survey configuration does not match the state in" in str(excinfo.value)


def test_supplying_the_default_reader_explicitly_is_not_a_configuration_change(
        monkeypatch, tmp_path):
    # _default_shape() (cadc.py:54-63) must be the SAME configuration as
    # passing ChimeBasebandReader() by hand, or every CLI run would refuse to
    # resume a directory made by a direct survey() call.
    from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
    _seed(monkeypatch, tmp_path)
    _survey(tmp_path, reader=ChimeBasebandReader())    # must not raise


# ##########################################################################
# 6. SurveyStore integrity -- a damaged database must refuse, never fail open.
# ##########################################################################
def test_a_junk_file_where_the_database_belongs_is_refused(tmp_path):
    path = tmp_path / "survey_state.sqlite3"
    path.write_bytes(b"this is not a database" * 64)
    with pytest.raises(SystemExit) as excinfo:
        SurveyStore(path)
    assert "invalid survey state database" in str(excinfo.value)


def test_a_drifted_events_table_is_refused(tmp_path):
    path = tmp_path / "survey_state.sqlite3"
    SurveyStore(path).close()
    con = sqlite3.connect(path)
    con.execute("ALTER TABLE events ADD COLUMN sneaky TEXT")
    con.commit()
    con.close()
    with pytest.raises(SystemExit) as excinfo:
        SurveyStore(path)
    assert "invalid survey state table 'events'" in str(excinfo.value)
    assert "expected columns" in str(excinfo.value)


@pytest.mark.parametrize("value,fragment", [
    ("nine", "invalid survey state schema"),
    ("9", "unsupported survey state schema 9"),
])
def test_an_unusable_schema_row_is_refused(tmp_path, value, fragment):
    path = tmp_path / "survey_state.sqlite3"
    SurveyStore(path).close()
    con = sqlite3.connect(path)
    con.execute("UPDATE metadata SET value=? WHERE key='schema'", (value,))
    con.commit()
    con.close()
    with pytest.raises(SystemExit) as excinfo:
        SurveyStore(path)
    assert fragment in str(excinfo.value)


class _FailingExecutemany:
    """A connection proxy that fails the LAST statement of commit()'s
    transaction. __enter__/__exit__ are forwarded to the real connection, so
    the transaction semantics under test are the real ones."""

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def __enter__(self):
        return self._db.__enter__()

    def __exit__(self, *exc):
        return self._db.__exit__(*exc)

    def executemany(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")


def test_commit_is_one_transaction_per_event(tmp_path):
    # The event row, the DELETE of its old records and the INSERT of its new
    # ones are one transaction (survey_state.py:371-383). Fail the last
    # statement and the event must be ABSENT afterwards -- not present with a
    # truncated record set, which is exactly what a resume would then treat as
    # a completed event and never re-probe.
    store = SurveyStore(tmp_path / "survey_state.sqlite3")
    store.commit("s|a", "s", "a", "complete", [{"n": 1}, {"n": 2}])
    store._db = _FailingExecutemany(store._db)
    with pytest.raises(RuntimeError):
        store.commit("s|b", "s", "b", "complete", [{"n": 3}])
    assert store.completed_keys() == {"s|a"}
    store.render_views(tmp_path)
    assert len(_lines(tmp_path / "inventory.jsonl")) == 2
    store.close()


def test_an_unserializable_row_is_rejected_before_any_database_work(tmp_path):
    # Rows are serialized BEFORE the transaction opens (survey_state.py:370),
    # so a reader that produced a non-JSON value fails closed rather than
    # committing a done marker with no rows behind it.
    store = SurveyStore(tmp_path / "survey_state.sqlite3")
    with pytest.raises(TypeError):
        store.commit("s|e", "s", "e", "complete",
                     [{"ok": 1}, {"bad": object()}])
    assert store.completed_keys() == set()
    store.close()


def test_recommitting_an_event_replaces_its_rows_rather_than_appending(
        tmp_path):
    # commit() is documented "safe to repeat after a crash": a re-run of the
    # same event must not double its rows.
    store = SurveyStore(tmp_path / "survey_state.sqlite3")
    three = [{"n": i} for i in range(3)]
    store.commit("s|e", "s", "e", "complete", three)
    store.commit("s|e", "s", "e", "complete", three[:2])
    store.render_views(tmp_path)
    assert len(_lines(tmp_path / "inventory.jsonl")) == 2
    store.close()


# ##########################################################################
# 7. Bounded runs and completeness reporting.
# ##########################################################################
def test_max_events_stops_resumably_and_reports_the_remainder(monkeypatch,
                                                              tmp_path):
    seen = _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:5])
    source = cadc_src.CadcDatatrailSource()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        source.survey(RunContext(instrument=None, options={
            "scope": SCOPE, "freq_ids": list(FREQ_IDS), "max_events": 2}),
            str(tmp_path))
    text = buf.getvalue()

    assert "reached --max-events=2; stopping (resumable)." in text
    assert seen == EVENTS[:2]
    status, _counts, _ = _db(tmp_path)
    assert len(status) == 2
    # this is what commands.py:151-155 turns into
    # "chime-survey: strict completeness failed: 3 pending"
    assert source.survey_completeness_issues(str(tmp_path)) == {
        "incomplete": 0, "refused": 0, "pending": 3}


def test_completeness_counts_terminal_omissions_not_archive_dispositions(
        monkeypatch, tmp_path):
    # An accepted-empty event is an archive FACT, not an omission, so strict
    # mode must pass on it. Anything unresolved must not.
    absent = {f"{_cp(EVENTS[0])}/{baseband_filename(EVENTS[0], f)}"
              for f in FREQ_IDS}
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, absent=absent)
    _install_enumeration(monkeypatch, EVENTS[:2])
    source = cadc_src.CadcDatatrailSource()
    with contextlib.redirect_stdout(io.StringIO()):
        source.survey(RunContext(instrument=None, options={
            "scope": SCOPE, "freq_ids": list(FREQ_IDS)}), str(tmp_path))
    status, _counts, _ = _db(tmp_path)
    assert status[f"{SCOPE}|{EVENTS[0]}"] == "empty"
    assert source.survey_completeness_issues(str(tmp_path)) == {
        "incomplete": 0, "refused": 0, "pending": 0}


def test_completeness_is_none_before_a_survey_has_run():
    assert cadc_src.CadcDatatrailSource().survey_completeness_issues("/x") is None


# ##########################################################################
# 8. Reader-contract and inventory-row guards. A survey must refuse a reader
#    that would poison the inventory, not write the poison out.
# ##########################################################################
class _Info:
    def __init__(self, name):
        self.name = name


class _Reader:
    """Minimal survey-shaped reader with a per-test file shape."""
    minimum_archive_bytes = 1

    def __init__(self, files, *, survey_schema=1, mutate=False):
        self.info = _Info("stub-reader")
        self.survey_schema = survey_schema
        self._files = files
        self._mutate = mutate

    def survey_files(self, event, common_path, selection, ctx):
        return list(self._files)

    def annotate_row(self, row, instrument):
        if self._mutate:
            row["event"] = "TAMPERED"


def _refuse(monkeypatch, tmp_path, reader):
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:1])
    with pytest.raises(SystemExit) as excinfo:
        _survey(tmp_path, reader=reader)
    return str(excinfo.value)


@pytest.mark.parametrize("schema", [0, -1, True, None, "2"])
def test_a_reader_without_a_usable_survey_schema_is_refused(monkeypatch,
                                                            tmp_path, schema):
    reader = _Reader([("a.h5", {})], survey_schema=schema)
    message = _refuse(monkeypatch, tmp_path, reader)
    assert "must declare a positive integer survey_schema" in message


def test_a_reader_with_a_sub_one_byte_floor_is_refused(monkeypatch, tmp_path):
    reader = _Reader([("a.h5", {})])
    reader.minimum_archive_bytes = 0
    message = _refuse(monkeypatch, tmp_path, reader)
    assert "minimum_archive_bytes must be >= 1" in message


_BAD_CANDIDATES = [
    ("duplicate-names", [("a.h5", {}), ("a.h5", {})],
     "yielded duplicate archive names"),
    ("parent-escape", [("../secret.h5", {})], "unsafe archive name"),
    ("absolute", [("/etc/passwd", {})], "unsafe archive name"),
    ("dot-segment", [("a/./b.h5", {})], "unsafe archive name"),
    ("blank", [("   ", {})], "unsafe archive name"),
    ("non-string", [(17, {})], "non-string archive name"),
    ("three-tuple", [("a.h5", {}, "extra")], "must yield (relative_name, fields) pairs"),
    ("non-mapping-fields", [("a.h5", ["not", "a", "mapping"])],
     "non-mapping inventory fields"),
    ("reserved-field", [("a.h5", {"common_path": "/tmp"})],
     "tried to overwrite source-owned inventory field(s)"),
]


@pytest.mark.parametrize("case_id,files,fragment", _BAD_CANDIDATES,
                         ids=[c[0] for c in _BAD_CANDIDATES])
def test_a_reader_yielding_an_unusable_candidate_is_refused(monkeypatch,
                                                            tmp_path, case_id,
                                                            files, fragment):
    message = _refuse(monkeypatch, tmp_path, _Reader(files))
    assert fragment in message


def test_a_reader_that_rewrites_source_owned_identity_is_refused(monkeypatch,
                                                                 tmp_path):
    reader = _Reader([("a.h5", {})], mutate=True)
    message = _refuse(monkeypatch, tmp_path, reader)
    assert "annotate_row() changed source-owned inventory field(s)" in message
    assert "'event'" in message


_BAD_ROWS = [
    ("malformed-json", "{not json", "malformed JSON"),
    ("not-an-object", "[1, 2]", "expected a JSON object"),
    ("missing-name", '{"scope": "s", "event": "e", "size_bytes": 1, '
                     '"common_path": "cadc:CHIMEFRB/d"}',
     "missing required field(s) ['name']"),
    ("padded-scope", '{"scope": " s ", "event": "e", "name": "a.h5", '
                     '"size_bytes": 1, "common_path": "cadc:CHIMEFRB/d"}',
     "'scope' must be a non-empty, unpadded string"),
    ("non-canonical-name", '{"scope": "s", "event": "e", "name": "../a.h5", '
                           '"size_bytes": 1, "common_path": "cadc:CHIMEFRB/d"}',
     "'name' must be a canonical relative path"),
    ("zero-size", '{"scope": "s", "event": "e", "name": "a.h5", '
                  '"size_bytes": 0, "common_path": "cadc:CHIMEFRB/d"}',
     "'size_bytes' must be a positive integer"),
    ("bool-size", '{"scope": "s", "event": "e", "name": "a.h5", '
                  '"size_bytes": true, "common_path": "cadc:CHIMEFRB/d"}',
     "'size_bytes' must be a positive integer"),
    ("negative-freq-id", '{"scope": "s", "event": "e", "name": "a.h5", '
                         '"size_bytes": 1, "common_path": "cadc:CHIMEFRB/d", '
                         '"freq_id": -1}',
     "'freq_id' must be a non-negative integer"),
    ("bad-datasets", '{"scope": "s", "event": "e", "name": "a.h5", '
                     '"size_bytes": 1, "common_path": "cadc:CHIMEFRB/d", '
                     '"datasets": ["ok", "  "]}',
     "'datasets' must be a list of non-empty strings"),
]


@pytest.mark.parametrize("case_id,text,fragment", _BAD_ROWS,
                         ids=[c[0] for c in _BAD_ROWS])
def test_parse_row_rejects_a_legacy_or_partial_row_with_its_location(
        case_id, text, fragment):
    # This is the guard that must reject the July-2026 datatrawl inventories
    # (which carry no "name"), and it is what stands between a drifted survey
    # and a scan that silently processes the wrong files.
    with pytest.raises(SystemExit) as excinfo:
        inv.parse_row(text, "/inv/inventory.jsonl", 42)
    message = str(excinfo.value)
    assert fragment in message
    assert "/inv/inventory.jsonl:42" in message
    assert "legacy or partial rows cannot be scanned safely" in message


def test_a_row_written_by_this_survey_parses_back(monkeypatch, tmp_path):
    # Round-trip: whatever survey() writes must satisfy the strict reader that
    # enumerate() applies to it. A schema change that breaks this would only
    # surface at scan time, on CANFAR, after the survey had finished.
    _install_cli(monkeypatch)
    _install_archive(monkeypatch)
    _install_enumeration(monkeypatch, EVENTS[:2])
    _survey(tmp_path)
    lines = _lines(tmp_path / "inventory.jsonl")
    assert lines
    for number, line in enumerate(lines, 1):
        row = inv.parse_row(line, str(tmp_path / "inventory.jsonl"), number)
        assert row["common_path"].startswith("cadc:CHIMEFRB/")
        assert inv.join_uri(row["common_path"], row["name"]).count("//") == 0
