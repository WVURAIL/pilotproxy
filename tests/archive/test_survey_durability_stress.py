#!/usr/bin/env python3
"""Durability, resume and concurrency stress tests for the cadc-datatrail survey.

Every test here drives the REAL survey() loop -- the real SurveyStore
transaction, the real SurveyOutputLock, the real view renderer -- with only the
three live seams faked: event enumeration, `datatrail ps`, and cadcinfo. No
network, no Datatrail CLI, no CADC certificate.

The claims under test are the ones a third party must be able to trust:

  * a committed event is durable and is never committed partially;
  * an interrupted run resumes with zero duplicate rows and zero re-probes of
    already-committed events;
  * inventory.jsonl and the text ledgers are pure views of SQLite and are
    regenerated after any stop between commit and render;
  * the worker count is a performance knob only -- it cannot change one byte of
    the inventory; and
  * two surveys can never interleave in one directory, and an incompatible
    configuration can never append to an existing one.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as C
from pilot_proxy.archive.survey_state import SurveyOutputLock

SCOPE = "chime.event.baseband.raw"
FLOOR = 1 << 20                       # ChimeBasebandReader.minimum_archive_bytes
OBS_DIR = "raw/2020/07/15"            # matches _DATE_RE (cadc.py:111)
FIDS = [614, 706, 800]

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX signal and flock semantics under test")


# ---------------------------------------------------------------- fixture ---
def events(n, start=100000000):
    return [str(start + i) for i in range(n)]


def size_for(event, name):
    """Deterministic above-floor size, so the inventory bytes are a pure
    function of the event set and never of thread scheduling."""
    digest = hashlib.sha256(f"{event}/{name}".encode()).digest()
    return FLOOR + int.from_bytes(digest[:4], "big") % 1000


class Probes:
    """What the survey actually asked the archive for."""

    def __init__(self):
        self.uris = []
        self.threads = set()
        self.enumerations = 0
        self.lock = threading.Lock()

    @property
    def events(self):
        return sorted({u.rsplit("/", 1)[-1].split("_")[1] for u in self.uris})


@contextlib.contextmanager
def fake_archive(evs, present=None, size_hook=None, cp_hook=None, jitter=0.0):
    """Patch the three live seams survey() leans on; restore them on exit."""
    present = set(evs) if present is None else set(present)
    probes = Probes()
    saved = (C._enumerate_events, C.DATATRAIL.common_path,
             C.CadcDatatrailSource._cadc_size)
    membership = {(SCOPE, ev): [f"dataset{int(ev) % 3}"] for ev in evs}

    def fake_size(self, uri, *args, **kwargs):
        uri = str(uri)
        name = uri.rsplit("/", 1)[-1]
        event = name.split("_")[1]
        with probes.lock:
            probes.uris.append(uri)
            probes.threads.add(threading.get_ident())
        if jitter:
            time.sleep(int(hashlib.sha256(uri.encode()).hexdigest()[:4], 16)
                       % 7 * jitter)
        if size_hook is not None:
            forced = size_hook(uri, event, name)
            if forced is not None:
                return forced
        if event in present:
            return size_for(event, name), None
        return None, None                                  # NotFound: absent

    def fake_cp(scope, event, **kwargs):
        if cp_hook is not None:
            forced = cp_hook(scope, event)
            if forced is not None:
                return forced
        return (f"cadc:CHIMEFRB/data/chime/baseband/{OBS_DIR}/astro_{event}",
                True)

    def fake_enumerate(*args, **kwargs):
        probes.enumerations += 1
        return dict(membership)

    C._enumerate_events = fake_enumerate
    C.DATATRAIL.common_path = fake_cp
    C.CadcDatatrailSource._cadc_size = fake_size
    try:
        yield probes
    finally:
        (C._enumerate_events, C.DATATRAIL.common_path,
         C.CadcDatatrailSource._cadc_size) = saved


OMIT = object()


def survey(out, freq_ids=FIDS, workers=4, **options):
    """One survey pass; returns its stdout. freq_ids=OMIT leaves the option
    unset, which is how a real CLI run reaches _resolve_freq_ids with None."""
    options = {"workers": workers, **options}
    if freq_ids is not OMIT:
        options["freq_ids"] = list(freq_ids)
    source = C.CadcDatatrailSource()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        source.survey(RunContext(instrument=None, selection=None,
                                 options=options), str(out))
    return buffer.getvalue()


# ------------------------------------------------------------- inspection ---
def db_state(out):
    db = sqlite3.connect(Path(out) / "survey_state.sqlite3")
    try:
        return {
            "status": dict(db.execute("SELECT event_key, status FROM events")),
            "records": dict(db.execute(
                "SELECT event_key, COUNT(*) FROM records GROUP BY event_key")),
            "integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
            "orphans": db.execute(
                "SELECT COUNT(*) FROM records LEFT JOIN events USING(event_key)"
                " WHERE events.event_key IS NULL").fetchone()[0],
        }
    finally:
        db.close()


VIEW_NAMES = ("inventory.jsonl", "surveyed_events.txt",
              "incomplete_events.txt", "no_files_events.jsonl")


def views(out):
    out = Path(out)
    return {name: ((out / name).read_text().splitlines()
                   if (out / name).exists() else None)
            for name in VIEW_NAMES}


def digests(out, exclude=(".survey.lock",)):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(out).iterdir())
            if p.is_file() and p.name not in exclude}


# ============================================================= 1. SIGINT ====
def interrupt_before_event(target):
    """Raise a real SIGINT on the MAIN thread just as survey() starts `target`.

    DATATRAIL.common_path is called on the main thread (cadc.py:616), before
    the probe pool is involved, so the default SIGINT handler raises
    KeyboardInterrupt at the very next bytecode boundary. That makes the
    committed prefix exact -- every event before `target` and nothing else --
    with no sleep and no race.
    """
    fired = []

    def cp_hook(scope, event):
        if event == target and not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        return None

    return cp_hook


def test_sigint_mid_run_commits_whole_events_only_and_releases_the_lock(
        tmp_path):
    """A real SIGINT arriving between two events."""
    out = tmp_path / "inv"
    evs = events(10)

    with fake_archive(evs, cp_hook=interrupt_before_event(evs[5])):
        with pytest.raises(KeyboardInterrupt):
            survey(out, workers=2)

    state = db_state(out)
    assert state["integrity"] == "ok"
    assert state["orphans"] == 0
    # Exactly the events BEFORE the interrupted one are committed ...
    assert sorted(state["status"]) == [f"{SCOPE}|{ev}" for ev in evs[:5]]
    # ... and every one of them is whole: no half-written event.
    assert set(state["records"].values()) == {len(FIDS)}
    # The finally block (cadc.py:827-848) still rendered the views.
    assert len(views(out)["inventory.jsonl"]) == 5 * len(FIDS)
    assert len(views(out)["surveyed_events.txt"]) == 5
    # The OS lock is gone -- a second survey can start immediately.
    with SurveyOutputLock(out):
        pass


def test_sigint_while_probes_are_in_flight_commits_nothing_partial(tmp_path):
    """The harder case: the signal arrives while the pool is mid-event.

    The interrupted event has probes that already returned. None of them may
    be persisted, and pool.shutdown(wait=True) in the finally (cadc.py:832)
    must still join every worker before the lock is released.
    """
    out = tmp_path / "inv"
    evs = events(10)
    fired = []

    def size_hook(uri, event, name):
        if event == evs[5] and not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.2)        # keep one probe running across the unwind
        return None

    with fake_archive(evs, size_hook=size_hook) as probes:
        with pytest.raises(KeyboardInterrupt):
            survey(out, workers=4)

    state = db_state(out)
    assert state["integrity"] == "ok"
    assert state["orphans"] == 0
    assert f"{SCOPE}|{evs[5]}" not in state["status"]     # never half-landed
    assert set(state["records"].values()) <= {len(FIDS)}
    assert len(state["status"]) == 5
    assert evs[5] in probes.events                        # it really was probed
    with SurveyOutputLock(out):                           # workers were joined
        pass


def test_resume_after_sigint_has_no_duplicates_and_no_re_probes(tmp_path):
    out = tmp_path / "inv"
    evs = events(10)

    with fake_archive(evs, cp_hook=interrupt_before_event(evs[5])):
        with pytest.raises(KeyboardInterrupt):
            survey(out, workers=2)
    first = list(views(out)["inventory.jsonl"])

    with fake_archive(evs) as probes:
        log = survey(out, workers=2)

    assert "resume: 5 events already done" in log
    assert probes.events == evs[5:]            # committed events never re-probed
    rows = views(out)["inventory.jsonl"]
    assert len(rows) == len(evs) * len(FIDS)
    assert len(set(rows)) == len(rows)         # zero duplicates
    assert rows[:len(first)] == first          # the surviving prefix is stable
    assert db_state(out)["integrity"] == "ok"


def test_an_interrupted_run_reports_the_interruption(tmp_path):
    out = tmp_path / "inv"
    evs = events(4)
    buffer = io.StringIO()
    with fake_archive(evs, cp_hook=interrupt_before_event(evs[2])):
        with contextlib.redirect_stdout(buffer), pytest.raises(KeyboardInterrupt):
            C.CadcDatatrailSource().survey(RunContext(
                instrument=None, selection=None,
                options={"workers": 2, "freq_ids": list(FIDS)}), str(out))
    assert len(db_state(out)["status"]) == 2
    assert "committed state was preserved" in buffer.getvalue()
    assert "Rerun the same survey command" in buffer.getvalue()
    run = json.loads((out / "survey_manifest.json").read_text())["latest_run"]
    assert run["status"] == "interrupted" and run["views_current"]

# ============================================================ 2. SIGKILL ====
KILL_CHILD = '''
"""Run one faked survey pass and park at a chosen point so the parent can
SIGKILL this process there. Configured by a JSON blob on argv[1]."""
import json, os, socket, sys, time
from pathlib import Path


def _no_network(*a, **k):
    raise AssertionError("the kill child attempted a live connection")


socket.socket.connect = _no_network
socket.create_connection = _no_network

sys.path.insert(0, os.environ["STRESS_DIR"])
from test_survey_durability_stress import fake_archive, survey, events
from pilot_proxy.archive import survey_state as S

cfg = json.loads(sys.argv[1])
marker = Path(cfg["marker"])
phase, target = cfg["phase"], cfg["target"]
after_rows = cfg.get("after_rows", 3)


def park():
    marker.write_text("parked\\n")
    os.sync()
    while True:
        time.sleep(0.01)


class Parking:
    """Delegates to the real sqlite3 connection and parks at one of three
    points: inside the commit transaction (survey_state.py:371-383), just
    after that transaction commits, or part-way through the cursor that feeds
    the inventory.jsonl view (survey_state.py:387-390)."""

    def __init__(self, real):
        self._real = real
        self._commits = 0
        self._renders = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        result = self._real.__exit__(*exc)
        if phase == "post-commit" and self._commits == target:
            park()
        return result

    def executemany(self, *a, **k):
        result = self._real.executemany(*a, **k)
        self._commits += 1
        if phase == "in-transaction" and self._commits == target:
            park()
        return result

    def execute(self, sql, *a, **k):
        cursor = self._real.execute(sql, *a, **k)
        if phase == "render-cursor" and sql.startswith("SELECT row_json"):
            self._renders += 1
            if self._renders == target:
                def parking_rows():
                    for index, row in enumerate(cursor):
                        if index == after_rows:
                            park()
                        yield row
                return parking_rows()
        return cursor


original = S.SurveyStore.__init__


def patched(self, path):
    original(self, path)
    self._db = Parking(self._db)


S.SurveyStore.__init__ = patched

with fake_archive(events(cfg["events"]), present=cfg.get("present")):
    survey(Path(cfg["out"]), workers=cfg.get("workers", 2))
'''


def _kill_at(tmp_path, phase, target, out=None, n_events=10, **extra):
    """Run a survey in a child process and SIGKILL it at a chosen point."""
    out = Path(out) if out is not None else tmp_path / f"inv-{phase}"
    marker = tmp_path / f"marker-{phase}-{target}"
    script = tmp_path / "kill_child.py"
    script.write_text(KILL_CHILD)
    config = {"out": str(out), "marker": str(marker), "phase": phase,
              "target": target, "events": n_events, **extra}
    src_root = str(Path(C.__file__).resolve().parents[3])
    env = {**os.environ,
           "STRESS_DIR": str(Path(__file__).resolve().parent),
           "PYTHONPATH": os.pathsep.join(
               [src_root] + [p for p in [os.environ.get("PYTHONPATH")] if p]),
           "TMPDIR": "/tmp", "TEMP": "/tmp", "TMP": "/tmp"}
    child = subprocess.Popen(
        [sys.executable, str(script), json.dumps(config)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 60
        while not marker.exists():
            if child.poll() is not None:
                raise AssertionError(
                    "child exited before parking:\n"
                    + child.stdout.read().decode(errors="replace"))
            assert time.monotonic() < deadline, "child never reached the park"
            time.sleep(0.005)
        child.kill()
        assert child.wait(timeout=30) == -signal.SIGKILL
    finally:
        child.stdout.close()
        if child.poll() is None:                       # never leave one running
            child.kill()
            child.wait(timeout=10)
    return out


@pytest.mark.parametrize("phase,committed", [("in-transaction", 3),
                                             ("post-commit", 4)])
def test_sigkill_inside_the_commit_transaction_is_atomic(
        tmp_path, phase, committed):
    """SIGKILL with the transaction open discards ALL of that event's writes.

    The park point is the last statement inside `with self._db:`
    (survey_state.py:381-383): the events row, the records DELETE and every
    records INSERT are staged and only the COMMIT is missing. Killing one
    statement later ("post-commit") lands exactly one more WHOLE event, and
    that difference of exactly one -- with no partial event in either case --
    is the atomicity proof.
    """
    out = _kill_at(tmp_path, phase, target=4)
    state = db_state(out)
    assert state["integrity"] == "ok"
    assert state["orphans"] == 0
    assert len(state["status"]) == committed
    assert set(state["records"].values()) == {len(FIDS)}     # no partial event
    assert set(state["status"].values()) == {"complete"}


def test_sigkill_leaves_stale_views_that_the_next_run_repairs(tmp_path):
    """The known hazard: a killed run leaves a truthful-LOOKING but stale
    inventory.jsonl on disk. Nothing warns; only the next run repairs it."""
    out = _kill_at(tmp_path, "post-commit", target=4)
    # Checked before db_state(): opening the DB checkpoints the WAL away.
    assert (out / "survey_state.sqlite3-wal").exists()
    assert (out / ".survey.lock").exists()              # OS lock already gone
    assert views(out)["inventory.jsonl"] == []          # <- the hazard
    assert len(db_state(out)["status"]) == 4

    with fake_archive(events(10)) as probes:
        log = survey(out, workers=2)

    assert "resume: 4 events already done" in log
    assert probes.events == events(10)[4:]
    rows = views(out)["inventory.jsonl"]
    assert len(rows) == 10 * len(FIDS)
    assert len(set(rows)) == len(rows)
    assert db_state(out)["integrity"] == "ok"


def test_a_kill_between_commit_and_checkpoint_leaves_no_stale_attempt_count(
        tmp_path):
    """The one-statement window cadc.py:601-605 leaves attempts.json claiming
    a retry for an event SQLite has already committed. The resume
    reconciliation at cadc.py:580-582 must drop it; otherwise the checkpoint
    accumulates entries for finished events forever.
    """
    out = tmp_path / "inv"
    evs = events(3)
    stalled = evs[2]

    def size_hook(uri, event, name):
        if event == stalled and name.endswith("_800.h5"):
            return None, OSError("transient")
        return None

    # Pass 1: the last event is partial, so it is bumped rather than committed.
    with fake_archive(evs, size_hook=size_hook):
        survey(out, workers=1)
    assert json.loads((out / "attempts.json").read_text()) == {
        f"{SCOPE}|{stalled}": 1}

    # Pass 2: it now resolves cleanly. Kill immediately after its transaction
    # commits and before the checkpoint is rewritten.
    _kill_at(tmp_path, "post-commit", target=1, out=out, n_events=3)
    assert len(db_state(out)["status"]) == 3
    assert json.loads((out / "attempts.json").read_text()) == {
        f"{SCOPE}|{stalled}": 1}, "the crash window did not reproduce"

    # Pass 3: the committed event wins and the stale count is discarded.
    with fake_archive(evs) as probes:
        log = survey(out, workers=1)
    assert "resume: 3 events already done" in log
    assert probes.uris == []
    assert json.loads((out / "attempts.json").read_text()) == {}


def test_a_killed_run_marks_views_stale_until_resume(tmp_path):
    out = _kill_at(tmp_path, "post-commit", target=4)
    assert len(db_state(out)["status"]) == 4
    assert views(out)["inventory.jsonl"] == []
    assert (out / ".survey.views-stale").is_file()
    with fake_archive(events(10)):
        survey(out, workers=2)
    assert not (out / ".survey.views-stale").exists()
    assert len(views(out)["inventory.jsonl"]) == 10 * len(FIDS)
    assert db_state(out)["integrity"] == "ok"

# ========================================================= 3. STALE LOCK ====
def test_a_stale_lock_file_does_not_strand_the_directory(tmp_path):
    """flock is kernel-released, so a dead owner's lock FILE is not a lock.

    Refusing here would be the real bug: one SIGKILL would strand the output
    directory permanently with no documented way to clear it.
    """
    out = tmp_path / "inv"
    out.mkdir()
    (out / ".survey.lock").write_text(json.dumps({"pid": 999999}) + "\n")

    with fake_archive(events(3)):
        survey(out)

    assert len(views(out)["inventory.jsonl"]) == 3 * len(FIDS)
    assert json.loads((out / ".survey.lock").read_text())["pid"] == os.getpid()


def test_takeover_after_a_real_kill_is_correct_not_merely_permitted(tmp_path):
    """Take over the lock left by an actually-killed process and prove the
    resulting inventory equals a clean single-pass run."""
    out = _kill_at(tmp_path, "post-commit", target=4)
    stale = json.loads((out / ".survey.lock").read_text())["pid"]
    with fake_archive(events(10)):
        survey(out, workers=2)
    assert json.loads((out / ".survey.lock").read_text())["pid"] == os.getpid()
    assert stale != os.getpid()

    reference = tmp_path / "reference"
    with fake_archive(events(10)):
        survey(reference, workers=2)
    for view in VIEW_NAMES:
        assert (out / view).read_bytes() == (reference / view).read_bytes()


def test_a_lock_file_alone_does_not_trip_the_legacy_state_guard(tmp_path):
    """.survey.lock is deliberately absent from _STATE_NAMES
    (survey_state.py:36-41). If it were listed, __enter__ creating it would
    make ensure_manifest refuse every fresh directory."""
    from pilot_proxy.archive.survey_state import _STATE_NAMES
    assert ".survey.lock" not in _STATE_NAMES
    out = tmp_path / "inv"
    out.mkdir()
    (out / ".survey.lock").write_text("{}\n")
    with fake_archive(events(2)):
        survey(out)                     # a manifest is created, not refused
    assert (out / "survey_manifest.json").exists()


def test_a_live_holder_is_never_taken_over(tmp_path):
    """Takeover must not extend to a lock that is actually held."""
    out = tmp_path / "inv"
    with SurveyOutputLock(out):
        with fake_archive(events(3)):
            with pytest.raises(SystemExit) as excinfo:
                survey(out)
    assert "already in use by another active survey" in str(excinfo.value)


# ======================================================= 4. VIEW DELETION ===
def test_views_deleted_mid_run_are_fully_regenerated(tmp_path):
    """Delete every view while the survey is running; the final render
    (cadc.py:835) restores them from SQLite alone."""
    out = tmp_path / "inv"
    evs = events(8)
    deleted = []

    def size_hook(uri, event, name):
        if event == evs[4] and not deleted:
            for view in VIEW_NAMES:
                with contextlib.suppress(FileNotFoundError):
                    (out / view).unlink()
            deleted.append(True)
        return None

    with fake_archive(evs, size_hook=size_hook):
        survey(out, workers=2)

    assert deleted, "the deletion hook never fired"
    reference = tmp_path / "reference"
    with fake_archive(evs):
        survey(reference, workers=2)
    for view in VIEW_NAMES:
        assert (out / view).read_bytes() == (reference / view).read_bytes()


def test_views_deleted_between_runs_are_restored_before_any_new_work(tmp_path):
    """cadc.py:585 renders the views at startup, before phase 2 probes."""
    out = tmp_path / "inv"
    evs = events(3)
    with fake_archive(evs):
        survey(out)
    complete = {v: (out / v).read_bytes() for v in VIEW_NAMES}
    for view in VIEW_NAMES:
        (out / view).unlink()

    with fake_archive(evs) as probes:
        survey(out)
    assert probes.uris == []                       # nothing left to survey
    for view in VIEW_NAMES:
        assert (out / view).read_bytes() == complete[view]


def test_view_order_is_by_event_key_not_by_commit_history(tmp_path):
    """The lowest-keyed event is committed LAST, three passes later.

    inventory.jsonl must still equal a clean single-pass run, because
    render_views orders by (event_key, ordinal) (survey_state.py:390) and not
    by insertion. A view ordered by rowid would put the retried event at the
    end -- and would still look perfectly sorted on any run that never
    retried, which is why a plain happy-path survey cannot see this.
    """
    out = tmp_path / "inv"
    evs = events(3)
    passes = []

    def size_hook(uri, event, name):
        # The FIRST event fails one probe on the first two passes, so it is
        # committed only on the third, long after its higher-keyed siblings.
        if event == evs[0] and len(passes) < 2 and name.endswith("_800.h5"):
            return None, OSError("transient")
        return None

    for _ in range(3):
        with fake_archive(evs, size_hook=size_hook):
            survey(out, workers=1)
        passes.append(True)

    db = sqlite3.connect(out / "survey_state.sqlite3")
    try:
        by_rowid = [r[0] for r in db.execute(
            "SELECT event_key FROM records GROUP BY event_key "
            "ORDER BY MIN(rowid)")]
    finally:
        db.close()
    assert by_rowid[-1] == f"{SCOPE}|{evs[0]}", "the retry did not land last"

    reference = tmp_path / "reference"
    with fake_archive(evs):
        survey(reference, workers=1)
    assert ((out / "inventory.jsonl").read_bytes()
            == (reference / "inventory.jsonl").read_bytes())
    assert ((out / "surveyed_events.txt").read_bytes()
            == (reference / "surveyed_events.txt").read_bytes())


def test_a_view_write_killed_part_way_leaves_the_previous_view_intact(
        tmp_path):
    """atomic_write_lines writes a temp file and os.replace()s it
    (survey_state.py:134-150), so a kill during a render can never expose a
    truncated inventory: the reader sees the previous complete file."""
    out = tmp_path / "inv"
    with fake_archive(events(3)):
        survey(out, workers=1)
    intact = (out / "inventory.jsonl").read_bytes()
    assert intact.count(b"\n") == 3 * len(FIDS)

    # Kill part-way through the SECOND render of this child (the first is the
    # startup recovery render at cadc.py:585; the second is the final one).
    _kill_at(tmp_path, "render-cursor", target=2, out=out, n_events=6,
             after_rows=4)

    assert (out / "inventory.jsonl").read_bytes() == intact   # never torn
    leftovers = [p.name for p in out.iterdir()
                 if p.name.startswith(".inventory.jsonl.")]
    assert leftovers, "the interrupted write left no temp file to prove it ran"

    with fake_archive(events(6)):
        survey(out, workers=1)                    # a leftover temp is harmless
    assert len(views(out)["inventory.jsonl"]) == 6 * len(FIDS)


def test_a_hand_edited_view_is_silently_overwritten(tmp_path):
    """Characterisation: the views are outputs, never inputs. Anyone editing
    no_files_events.jsonl by hand loses the edit at the next render."""
    out = tmp_path / "inv"
    with fake_archive(events(2)):
        survey(out)
    (out / "inventory.jsonl").write_text('{"hand": "edited"}\n')
    with fake_archive(events(2)):
        survey(out)
    assert '"hand"' not in (out / "inventory.jsonl").read_text()
    assert len(views(out)["inventory.jsonl"]) == 2 * len(FIDS)


def test_deleting_the_database_between_runs_re_surveys_from_scratch(tmp_path):
    """The DB is the source of truth; losing it loses resume, not correctness.
    Note the views are NOT consulted to rebuild it -- they are pure views."""
    out = tmp_path / "inv"
    with fake_archive(events(4)):
        survey(out)
    first = (out / "inventory.jsonl").read_bytes()
    for suffix in ("", "-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            (out / f"survey_state.sqlite3{suffix}").unlink()

    with fake_archive(events(4)) as probes:
        log = survey(out)
    assert "resume: 0 events already done" in log
    assert probes.events == events(4)                       # all re-probed
    assert (out / "inventory.jsonl").read_bytes() == first  # same bytes


# ========================================================= 5. CONCURRENCY ===
WORKER_COUNTS = (1, 4, 12, 32)
WORKER_EVENTS = 8
WORKER_FIDS = list(range(614, 646))               # 32 candidates per event


@pytest.fixture(scope="module")
def worker_reference(tmp_path_factory):
    """One serial reference inventory that every worker count must reproduce."""
    out = tmp_path_factory.mktemp("worker-reference")
    with fake_archive(events(WORKER_EVENTS)):
        survey(out, freq_ids=WORKER_FIDS, workers=1)
    return (out / "inventory.jsonl").read_bytes()


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_inventory_is_byte_identical_at_every_worker_count(
        tmp_path, workers, worker_reference):
    """--workers is a throughput knob; it may not change one inventory byte.

    Jitter derived from the URI digest makes probe completion order genuinely
    differ between worker counts, so a scheduling-dependent write order shows
    up as a byte mismatch rather than passing by luck.
    """
    out = tmp_path / f"w{workers}"
    with fake_archive(events(WORKER_EVENTS), jitter=0.001) as probes:
        survey(out, freq_ids=WORKER_FIDS, workers=workers)
    # Prove the parallelism was real: the pool grows a thread per queued
    # candidate up to its cap, so this equals min(workers, candidates).
    assert len(probes.threads) == min(workers, len(WORKER_FIDS))
    payload = (out / "inventory.jsonl").read_bytes()
    assert payload == worker_reference
    assert payload.count(b"\n") == WORKER_EVENTS * len(WORKER_FIDS)


def test_record_ordinals_follow_candidate_order_not_completion_order(tmp_path):
    """pool.map preserves input order (cadc.py:650), so records.ordinal is
    freq_id order however the probes finish."""
    out = tmp_path / "inv"
    with fake_archive(["100000000"], jitter=0.002):
        survey(out, freq_ids=WORKER_FIDS, workers=32)
    db = sqlite3.connect(out / "survey_state.sqlite3")
    try:
        rows = [json.loads(r[0]) for r in db.execute(
            "SELECT row_json FROM records ORDER BY ordinal")]
    finally:
        db.close()
    assert [row["freq_id"] for row in rows] == WORKER_FIDS


def test_workers_do_not_touch_the_store_or_the_checkpoint(tmp_path):
    """sqlite3.connect defaults to check_same_thread=True, so any future move
    of commit() into a pool worker raises instead of corrupting silently."""
    out = tmp_path / "inv"
    seen = []
    main = threading.get_ident()

    def size_hook(uri, event, name):
        seen.append(threading.get_ident())
        return None

    with fake_archive(events(4), size_hook=size_hook, jitter=0.001):
        survey(out, freq_ids=WORKER_FIDS, workers=8)
    assert main not in seen, "probes ran on the main thread"
    assert len(views(out)["inventory.jsonl"]) == 4 * len(WORKER_FIDS)


# ======================================================== 6. INTERLEAVING ===
def test_two_surveys_in_one_directory_are_refused_not_interleaved(tmp_path):
    """flock excludes per open file description, so a second SurveyOutputLock
    is refused whether it lives in this process or another one."""
    out = tmp_path / "inv"
    with fake_archive(events(4)):
        survey(out)
    before = digests(out)

    with SurveyOutputLock(out):
        with fake_archive(events(8)) as probes:
            with pytest.raises(SystemExit) as excinfo:
                survey(out)
    message = str(excinfo.value)
    assert "already in use by another active survey" in message
    assert str(out.resolve()) in message
    assert probes.uris == []
    # The refused run touched nothing but the diagnostic lock file.
    assert digests(out) == before


def test_the_lock_is_held_for_the_whole_call_including_enumeration(tmp_path):
    """with_survey_output_lock wraps survey() itself (cadc.py:487), so the
    window covers the manifest check, enumeration and every commit."""
    out = tmp_path / "inv"
    contended = []

    def size_hook(uri, event, name):
        if not contended:
            try:
                with SurveyOutputLock(out):
                    contended.append("acquired")
            except SystemExit as exc:
                contended.append(str(exc))
        return None

    with fake_archive(events(3), size_hook=size_hook):
        survey(out, workers=1)
    assert contended and "already in use" in contended[0]


def test_a_refused_second_survey_creates_no_state_in_a_fresh_directory(
        tmp_path):
    out = tmp_path / "fresh"
    with SurveyOutputLock(out):
        with fake_archive(events(2)):
            with pytest.raises(SystemExit):
                survey(out)
    assert sorted(p.name for p in out.iterdir()) == [".survey.lock"]


def test_two_processes_cannot_both_survey_one_directory(tmp_path):
    """The cross-process case, proven with a real second interpreter."""
    out = tmp_path / "inv"
    out.mkdir()
    src_root = str(Path(C.__file__).resolve().parents[3])
    program = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pilot_proxy.archive.survey_state import SurveyOutputLock\n"
        "try:\n"
        "    with SurveyOutputLock(%r):\n"
        "        print('ACQUIRED')\n"
        "except SystemExit as exc:\n"
        "    print('REFUSED', exc)\n" % (src_root, str(out)))
    with SurveyOutputLock(out):
        result = subprocess.run([sys.executable, "-c", program],
                                capture_output=True, text=True, timeout=60)
    assert result.stdout.startswith("REFUSED"), result.stdout + result.stderr
    assert "already in use by another active survey" in result.stdout


# ================================================ 7. CONFIGURATION CHANGE ===
def test_a_different_freq_id_set_is_refused_by_the_manifest_guard(tmp_path):
    out = tmp_path / "inv"
    with fake_archive(events(4)):
        survey(out, freq_ids=[614, 706])
    before = digests(out)

    with fake_archive(events(4)) as probes:
        with pytest.raises(SystemExit) as excinfo:
            survey(out, freq_ids=[614, 707])
    message = str(excinfo.value)
    assert "survey configuration does not match the state in" in message
    assert "use a fresh --name/output directory" in message
    assert probes.uris == []                # refused before any probe ...
    assert digests(out) == before           # ... and before any mutation


def test_the_manifest_guard_runs_before_any_enumeration(tmp_path):
    """ensure_manifest (cadc.py:552) precedes _enumerate_events (cadc.py:566),
    so an incompatible run cannot reuse or overwrite the event cache."""
    out = tmp_path / "inv"
    with fake_archive(events(4)) as first:
        survey(out, freq_ids=[614, 706])
    assert first.enumerations == 1

    with fake_archive(events(9)) as second:
        with pytest.raises(SystemExit):
            survey(out, freq_ids=[614, 707])
    assert second.enumerations == 0
    assert second.uris == []


@pytest.mark.parametrize("changed", [
    pytest.param({"freq_ids": [614]}, id="fewer-freq-ids"),
    pytest.param({"empty_age_days": 7}, id="empty-age-days"),
    pytest.param({"include_outrigger": True}, id="include-outrigger"),
])
def test_every_row_affecting_option_is_refused_on_resume(tmp_path, changed):
    out = tmp_path / "inv"
    base = {"freq_ids": [614, 706]}
    with fake_archive(events(2)):
        survey(out, **base)
    with fake_archive(events(2)):
        with pytest.raises(SystemExit) as excinfo:
            survey(out, **{**base, **changed})
    assert "does not match the state in" in str(excinfo.value)


@pytest.mark.parametrize("operational", [
    pytest.param({"workers": 7}, id="workers"),
    pytest.param({"max_events": 1}, id="max-events"),
    pytest.param({"re_enumerate": True}, id="re-enumerate"),
])
def test_operational_options_may_change_freely_on_resume(
        tmp_path, operational):
    """_OPERATIONAL_OPTIONS (survey_state.py:31-35) is the allow-list; these
    change throughput or scheduling, never a durable row."""
    out = tmp_path / "inv"
    with fake_archive(events(4)):
        survey(out, freq_ids=[614, 706])
    with fake_archive(events(4)):
        survey(out, freq_ids=[614, 706], **operational)       # must not raise
    assert len(views(out)["inventory.jsonl"]) == 8


def test_reordering_freq_ids_is_normalised_and_therefore_allowed(tmp_path):
    """_resolve_freq_ids sorts (cadc.py:149), so [706, 614] and [614, 706]
    yield the same manifest. build_configuration itself does NOT sort freq_ids
    (survey_state.py:210, unlike scopes on :209), so this invariant lives
    entirely in the resolver -- pin it, because a resolver that stopped
    sorting would strand every existing inventory directory.
    """
    out = tmp_path / "inv"
    with fake_archive(events(2)):
        survey(out, freq_ids=[614, 706])
    with fake_archive(events(2)):
        survey(out, freq_ids=[706, 614])                     # must not raise
    manifest = json.loads((out / "survey_manifest.json").read_text())
    assert manifest["configuration"]["freq_ids"] == [614, 706]


def test_a_corrupt_manifest_refuses_rather_than_re_fingerprinting(tmp_path):
    out = tmp_path / "inv"
    with fake_archive(events(2)):
        survey(out)
    manifest = out / "survey_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    before = digests(out, exclude=(".survey.lock", "survey_manifest.json"))
    with fake_archive(events(2)):
        with pytest.raises(SystemExit) as excinfo:
            survey(out)
    assert "survey manifest is corrupt" in str(excinfo.value)
    assert digests(out, exclude=(".survey.lock",
                                 "survey_manifest.json")) == before


# ======================================================== 8. SIZE EXTREMES ==
def test_one_event_with_1024_replicas_commits_as_a_single_transaction(
        tmp_path):
    out = tmp_path / "inv"
    fids = list(range(1024))
    started = time.perf_counter()
    with fake_archive(["100000000"]) as probes:
        survey(out, freq_ids=fids, workers=12)
    elapsed = time.perf_counter() - started

    assert len(probes.uris) == 1024
    db = sqlite3.connect(out / "survey_state.sqlite3")
    try:
        ordinals = [r[0] for r in db.execute(
            "SELECT ordinal FROM records ORDER BY ordinal")]
    finally:
        db.close()
    assert ordinals == list(range(1024))
    assert len(views(out)["inventory.jsonl"]) == 1024
    assert len(views(out)["surveyed_events.txt"]) == 1
    assert elapsed < 20


def test_a_1024_replica_event_is_all_or_nothing_under_sigint(tmp_path, monkeypatch):
    """One event is one transaction, so interrupting it late commits nothing --
    not 1000 of 1024 rows."""
    out = tmp_path / "inv"
    fids = list(range(1024))
    fired = []

    annotate = C.annotate_row

    def interrupt_late_row(shape, row, instrument):
        annotate(shape, row, instrument)
        freq_id = int(row["name"].rsplit("_", 1)[1].split(".")[0])
        if freq_id == 1000 and not fired:
            fired.append(True)
            # Interrupt row assembly independently of probe scheduling.
            assert threading.current_thread() is threading.main_thread()
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(C, "annotate_row", interrupt_late_row)
    with fake_archive(["100000000"]):
        with pytest.raises(KeyboardInterrupt):
            survey(out, freq_ids=fids, workers=12)
    assert fired == [True]
    state = db_state(out)
    assert state["status"] == {}                 # nothing committed at all
    assert state["records"] == {}
    assert views(out)["inventory.jsonl"] == []

    with fake_archive(["100000000"]):
        survey(out, freq_ids=fids, workers=12)
    assert len(views(out)["inventory.jsonl"]) == 1024


def test_an_event_with_zero_selected_freq_ids_has_selection_status(
        tmp_path):
    """No selected files must not imply archive absence."""
    out = tmp_path / "inv"
    with fake_archive(events(3)) as probes:
        log = survey(out, freq_ids=OMIT)
    assert probes.uris == []
    assert views(out)["inventory.jsonl"] == []
    ledger = [json.loads(line) for line in views(out)["no_files_events.jsonl"]]
    assert len(ledger) == 3
    assert {entry["reason"] for entry in ledger} == {"no-selected-candidates"}
    assert {entry["n_expected"] for entry in ledger} == {0}
    assert set(db_state(out)["status"].values()) == {"empty-selection"}
    assert "inventory.jsonl is EMPTY" in log


def test_a_zero_candidate_event_is_distinguished_from_an_absent_one(
        tmp_path):
    """The ledger records selection and archive absence separately."""
    empty_selection = tmp_path / "selection"
    absent_archive = tmp_path / "absent"
    with fake_archive(events(3)):
        survey(empty_selection, freq_ids=OMIT)
    with fake_archive(events(3), present=set()):
        survey(absent_archive, freq_ids=FIDS)
    left = [json.loads(x) for x in views(empty_selection)["no_files_events.jsonl"]]
    right = [json.loads(x) for x in views(absent_archive)["no_files_events.jsonl"]]
    assert {e["reason"] for e in left} == {"no-selected-candidates"}
    assert {e["reason"] for e in right} == {"aged-out"}
    assert [e["n_expected"] for e in left] == [0, 0, 0]
    assert [e["n_expected"] for e in right] == [len(FIDS)] * 3


def test_an_event_whose_selected_freq_ids_are_all_absent_is_empty_not_clean(
        tmp_path):
    out = tmp_path / "inv"
    with fake_archive(events(3), present=set()):
        survey(out, freq_ids=FIDS)
    assert views(out)["inventory.jsonl"] == []
    ledger = [json.loads(line) for line in views(out)["no_files_events.jsonl"]]
    assert {entry["n_expected"] for entry in ledger} == {len(FIDS)}
    assert {entry["reason"] for entry in ledger} == {"aged-out"}
    assert set(db_state(out)["status"].values()) == {"empty"}


# ================================ 9. RESUME OVER NON-COMPLETE DISPOSITIONS ==
# A resume must treat EVERY terminal disposition as done, not only "complete".
# completed_keys() selects every row of `events` (survey_state.py:351-353); a
# narrower query would silently re-probe written-off events on every pass and
# rewrite their ledger timestamps, so the write-off would never settle.
def test_an_accepted_empty_event_is_never_re_probed_on_resume(tmp_path):
    out = tmp_path / "inv"
    evs = events(3)
    with fake_archive(evs, present=set()):
        survey(out)
    ledger = (out / "no_files_events.jsonl").read_bytes()
    assert set(db_state(out)["status"].values()) == {"empty"}

    with fake_archive(evs, present=set()) as probes:
        log = survey(out)
    assert "resume: 3 events already done" in log
    assert probes.uris == []
    assert (out / "no_files_events.jsonl").read_bytes() == ledger


def test_a_contract_refused_event_is_never_re_probed_on_resume(tmp_path):
    """A refusal is committed as done with its reason in the ledger
    (cadc.py:724-748), so a resume must skip it too."""
    from pilot_proxy.archive.datatrail_client import DatatrailContractError
    out = tmp_path / "inv"
    evs = events(3)

    def cp_hook(scope, event):
        if event == evs[1]:
            raise DatatrailContractError("replica outside the collection")
        return None

    with fake_archive(evs, cp_hook=cp_hook):
        survey(out, workers=1)
    status = db_state(out)["status"]
    assert status[f"{SCOPE}|{evs[1]}"] == "refused"
    ledger = [json.loads(x) for x in views(out)["no_files_events.jsonl"]]
    assert [e["reason"] for e in ledger] == ["datatrail-contract-refusal"]
    assert len(views(out)["inventory.jsonl"]) == 2 * len(FIDS)

    with fake_archive(evs, cp_hook=cp_hook) as probes:
        log = survey(out, workers=1)
    assert "resume: 3 events already done" in log
    assert probes.uris == []
    assert [json.loads(x) for x in views(out)["no_files_events.jsonl"]] == ledger


def test_a_terminal_incomplete_event_is_never_re_probed_on_resume(tmp_path):
    """After _MAX_ATTEMPTS a partial event is accepted with its unresolved
    names in incomplete_events.txt (cadc.py:795-802) and must then settle."""
    out = tmp_path / "inv"
    evs = events(2)

    def size_hook(uri, event, name):
        if event == evs[0] and name.endswith("_800.h5"):
            return None, OSError("permanently unresolvable")
        return None

    for _ in range(3):
        with fake_archive(evs, size_hook=size_hook):
            survey(out, workers=1)
    status = db_state(out)["status"]
    assert status[f"{SCOPE}|{evs[0]}"] == "incomplete"
    incomplete = (out / "incomplete_events.txt").read_text()
    assert f"{SCOPE}|{evs[0]}\tunresolved=baseband_{evs[0]}_800.h5" in incomplete
    # The two files that DID verify were kept.
    assert len(views(out)["inventory.jsonl"]) == (len(FIDS) - 1) + len(FIDS)

    with fake_archive(evs, size_hook=size_hook) as probes:
        log = survey(out, workers=1)
    assert "resume: 2 events already done" in log
    assert probes.uris == []
    assert (out / "incomplete_events.txt").read_text() == incomplete


def test_zero_selection_fails_strict_completeness(tmp_path):
    source = C.CadcDatatrailSource()
    with fake_archive(events(3)):
        source.survey(RunContext(instrument=None, selection=None,
                                options={"workers": 2}), str(tmp_path))
    assert source.survey_completeness_issues(str(tmp_path))["incomplete"] == 3


def test_enumeration_refuses_inventory_views_marked_stale(tmp_path):
    path = tmp_path / "inventory.jsonl"
    path.write_text("")
    (tmp_path / ".survey.views-stale").write_text("{}")
    context = RunContext(instrument=None, selection=None, options={"inventory": str(path)})
    with pytest.raises(SystemExit, match="may be stale"):
        list(C.CadcDatatrailSource().enumerate(context))
