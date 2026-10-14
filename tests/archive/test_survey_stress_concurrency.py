#!/usr/bin/env python3
"""
Failure-injection stress tests for survey concurrency and output ownership --
properties a production run depends on but a single happy-path run cannot
demonstrate, because it only ever proves that ONE worker count produced ONE
file.

What is asserted here:
  * the inventory is byte-identical at every --workers value. Row CONTENT is
    made a pure function of the URI, and each probe is deliberately staggered,
    so if ordering ever came from completion order instead of candidate order
    (cadc.py:650 `pool.map`, whose input order is what makes record ordinals
    deterministic) the bytes would move;
  * the pool really is `workers`-wide -- enforced with a Barrier, so a
    regression that serialized it fails instead of passing quietly and slowly;
  * the pool parallelizes WITHIN one event only. Events are strictly
    sequential (the main thread blocks on the map), which is why --workers on
    a one-file-per-event reader buys nothing. Documented nowhere, asserted
    nowhere until now;
  * probes never touch the store or the checkpoint. sqlite3.connect defaults
    to check_same_thread=True, so moving commit() into a worker would raise --
    but nothing pinned that the split exists;
  * two surveys can never interleave in one directory, in-process AND across
    processes, and a refused second survey mutates nothing.

Offline: the datatrail CLI child and the CADC probe are the only faked seams,
except the one cross-process test, which spawns a real second interpreter that
touches nothing but SurveyOutputLock.

Run:
  TMPDIR=/tmp PYTHONPATH=src python -m pytest tests/archive/test_survey_stress_concurrency.py
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as cadc_src
from pilot_proxy.archive.survey_state import SurveyOutputLock
from pilot_proxy.chime.baseband_reader import baseband_filename


SCOPE = "chime.event.baseband.raw"
EVENTS = ["100000000", "100000001", "100000002"]
# 16 divides every worker count under test, so the Barrier below always
# completes whole rounds and can never dead-end on a remainder.
FREQ_IDS = list(range(16))
WORKER_COUNTS = [1, 2, 4, 8, 16]


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


class _MisWiredFake(BaseException):
    """BaseException on purpose -- _run_json catches Exception and would
    reclassify a broken fake as a service outage, hanging the suite in the
    one-hour retry circuit instead of failing it."""


def _cp(event) -> str:
    day = (datetime.datetime.now(datetime.timezone.utc).date()
           - datetime.timedelta(days=400))
    return f"cadc:CHIMEFRB/data/chime/baseband/raw/{day:%Y/%m/%d}/astro_{event}"


def _install_cli(monkeypatch, *, on_ps=None):
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
        if on_ps is not None:
            on_ps(event)
        minoc = [f"{_cp(event)}/{baseband_filename(event, f)}"
                 for f in FREQ_IDS]
        return _Proc(0, json.dumps({
            "dataset": event, "scope": args[1], "policies": {},
            "files": {"file_replica_locations": {"minoc": minoc}}}), "")

    monkeypatch.setattr(dt.subprocess, "run", fake_run)


class _Probes:
    """Every CADC probe, with the thread that ran it and when."""

    def __init__(self, barrier=None):
        self.barrier = barrier
        self.lock = threading.Lock()
        self.records: list = []

    def size(self, uri):
        started = time.monotonic()
        if self.barrier is not None:
            # Forces `parties` probes to be in flight simultaneously. If the
            # pool ever stopped being parties-wide this raises instead of
            # silently passing on a serialized run.
            try:
                self.barrier.wait(timeout=20)
            except threading.BrokenBarrierError:  # pragma: no cover - failure
                raise AssertionError(
                    "the survey probe pool did not run "
                    f"{self.barrier.parties} probes concurrently")
        with self.lock:
            self.records.append((uri, threading.get_ident(), started,
                                 time.monotonic()))
        # size derived from the URI: inventory CONTENT is then a pure function
        # of the event set, never of scheduling or completion order.
        digest = hashlib.sha256(uri.encode()).hexdigest()[:8]
        return (1 << 20) + int(digest, 16) % 4096, None

    @property
    def threads(self):
        return {ident for _uri, ident, _a, _b in self.records}

    def spans_for(self, event):
        return [(a, b) for uri, _i, a, b in self.records if f"_{event}_" in uri]


def _install_archive(monkeypatch, probes):
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size",
                        lambda self, uri, *a, **k: probes.size(uri))


def _install_enumeration(monkeypatch, events=EVENTS):
    monkeypatch.setattr(cadc_src, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, ev): ["ds"] for ev in events})


def _survey(out_dir, **options):
    opts = {"scope": SCOPE, "freq_ids": list(FREQ_IDS)}
    opts.update(options)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cadc_src.CadcDatatrailSource().survey(
            RunContext(instrument=None, selection=None, options=opts),
            str(out_dir))
    return buf.getvalue()


def _run_at(monkeypatch, out_dir, workers, *, barrier=True):
    parties = min(workers, len(FREQ_IDS))
    probes = _Probes(threading.Barrier(parties) if barrier else None)
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    _survey(out_dir, workers=workers)
    return probes


# ##########################################################################
# 1. Worker-count invariance.
# ##########################################################################
@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """One single-threaded inventory, the baseline every worker count must
    reproduce byte for byte."""
    out = tmp_path_factory.mktemp("reference")
    probes = _Probes()
    with pytest.MonkeyPatch.context() as monkeypatch:
        _install_cli(monkeypatch)
        _install_archive(monkeypatch, probes)
        _install_enumeration(monkeypatch)
        _survey(out, workers=1)
    return (out / "inventory.jsonl").read_bytes()


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_the_inventory_is_byte_identical_at_every_worker_count(
        monkeypatch, tmp_path, reference, workers):
    probes = _run_at(monkeypatch, tmp_path, workers)
    assert (tmp_path / "inventory.jsonl").read_bytes() == reference
    # ... and the run really was `workers`-wide, so the equality above is not
    # the trivial equality of five serial runs.
    assert len(probes.threads) == min(workers, len(FREQ_IDS))
    assert len(probes.records) == len(EVENTS) * len(FREQ_IDS)


def test_record_ordinals_follow_candidate_order_not_completion_order(
        monkeypatch, tmp_path):
    # pool.map preserves INPUT order (cadc.py:650), which is what makes the
    # `records` ordinal -- and therefore inventory.jsonl's within-event order
    # -- deterministic. Staggering the probes so completion order differs is
    # the whole point: with as_completed() semantics this list would shuffle.
    _run_at(monkeypatch, tmp_path, 16)
    rows = [json.loads(line) for line
            in (tmp_path / "inventory.jsonl").read_text().splitlines() if line]
    for event in EVENTS:
        ids = [r["freq_id"] for r in rows if r["event"] == event]
        assert ids == FREQ_IDS, event


def test_enumeration_order_does_not_leak_into_the_inventory(monkeypatch,
                                                            tmp_path):
    # survey() iterates `sorted(events)` (cadc.py:568), so the order datatrail
    # happened to list events in cannot reach the file. (The stronger property
    # -- that the VIEW is ordered by event_key rather than by commit order --
    # can only be told apart across a resume, and is pinned by
    # test_view_order_is_by_event_key_not_by_commit_order in
    # tests/archive/test_survey_stress_resume.py.)
    probes = _Probes()
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    monkeypatch.setattr(cadc_src, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, ev): ["ds"]
                                         for ev in reversed(EVENTS)})
    _survey(tmp_path, workers=4)
    rows = [json.loads(line) for line
            in (tmp_path / "inventory.jsonl").read_text().splitlines() if line]
    assert [r["event"] for r in rows][::len(FREQ_IDS)] == sorted(EVENTS)


def test_a_falsy_worker_count_falls_back_to_the_default(monkeypatch, tmp_path,
                                                        reference):
    # cadc.py:518-519 clamps a falsy --workers to _DEFAULT_SURVEY_WORKERS.
    # Unreachable through the CLI (commands.py:67-68 rejects < 1), so a direct
    # survey() call is the only way to cover it.
    probes = _Probes()
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    _survey(tmp_path, workers=0)
    assert (tmp_path / "inventory.jsonl").read_bytes() == reference
    assert 1 < len(probes.threads) <= cadc_src._DEFAULT_SURVEY_WORKERS


# ##########################################################################
# 2. What the pool does and does not parallelize.
# ##########################################################################
def test_events_are_processed_strictly_sequentially(monkeypatch, tmp_path):
    # The pool is created once (cadc.py:591) but iterated from the main loop,
    # so it fans out ACROSS ONE EVENT'S CANDIDATES and never across events.
    # That is why --workers 12 buys nothing on a reader whose survey_files()
    # yields a single file per event. Nothing else states this.
    probes = _run_at(monkeypatch, tmp_path, 8)
    windows = []
    for event in EVENTS:
        spans = probes.spans_for(event)
        assert len(spans) == len(FREQ_IDS)
        windows.append((min(a for a, _b in spans), max(b for _a, b in spans)))
    for (_a0, end), (start, _b1) in zip(windows, windows[1:]):
        assert end <= start, "two events' probes overlapped in time"


def test_probes_run_off_the_main_thread_while_the_store_stays_on_it(
        monkeypatch, tmp_path):
    # The durable state is main-thread-only: SurveyStore holds a
    # sqlite3 connection created with check_same_thread=True (the default),
    # so a future refactor that committed from a worker would raise
    # ProgrammingError rather than corrupt anything -- but only as long as the
    # probes are the only thing on the pool.
    main = threading.get_ident()
    probes = _run_at(monkeypatch, tmp_path, 4)
    assert main not in probes.threads
    assert len(probes.threads) == 4


# ##########################################################################
# 3. Output ownership -- two surveys can never interleave.
# ##########################################################################
def test_a_second_survey_in_one_directory_is_refused_not_interleaved(
        monkeypatch, tmp_path):
    probes = _Probes()
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    _survey(tmp_path, workers=2)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*")
              if p.is_file() and p.name != ".survey.lock"}
    n_probes = len(probes.records)

    with SurveyOutputLock(tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            _survey(tmp_path, workers=2)
    assert "already in use by another active survey" in str(excinfo.value)
    assert str(tmp_path.resolve()) in str(excinfo.value)
    assert len(probes.records) == n_probes          # nothing was probed
    # ... and nothing on disk moved, except the diagnostic lock file itself.
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*")
            if p.is_file() and p.name != ".survey.lock"} == before


def test_the_lock_covers_enumeration_and_every_commit(monkeypatch, tmp_path):
    # with_survey_output_lock wraps the WHOLE call (cadc.py:487), so the
    # window is not "while writing" but "from the manifest check to the final
    # render". Contend from inside a ps reply, i.e. mid-run.
    contended: list = []

    def contend(event):
        if not contended:
            try:
                with SurveyOutputLock(tmp_path):
                    contended.append("acquired")
            except SystemExit as exc:
                contended.append(str(exc))

    probes = _Probes()
    _install_cli(monkeypatch, on_ps=contend)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    _survey(tmp_path, workers=2)
    assert contended and "already in use by another active survey" in contended[0]


def test_a_refused_second_survey_creates_no_state_in_a_fresh_directory(
        monkeypatch, tmp_path):
    out = tmp_path / "fresh"
    probes = _Probes()
    _install_cli(monkeypatch)
    _install_archive(monkeypatch, probes)
    _install_enumeration(monkeypatch)
    with SurveyOutputLock(out):
        with pytest.raises(SystemExit):
            _survey(out, workers=2)
    # no manifest, no database, no views -- only the diagnostic lock file the
    # HOLDER created.
    assert sorted(p.name for p in out.iterdir()) == [".survey.lock"]
    assert probes.records == []


def _src_dir() -> str:
    """The src/ directory this test's pilot_proxy was imported from, so the
    child under test is the same tree (including a mutated scratch copy)."""
    import pilot_proxy
    return os.path.dirname(os.path.dirname(pilot_proxy.__file__))


_LOCK_CHILD = textwrap.dedent(
    """
    import sys
    from pilot_proxy.archive.survey_state import SurveyOutputLock
    try:
        with SurveyOutputLock(sys.argv[1]):
            print("ACQUIRED")
    except SystemExit as exc:
        print("REFUSED", exc)
    """)


def _lock_child(out_dir):
    return subprocess.run(
        [sys.executable, "-c", _LOCK_CHILD, str(out_dir)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": _src_dir()})


def test_two_processes_cannot_both_own_one_output_directory(tmp_path):
    # The in-process tests above share one flock table; this one proves the
    # exclusion is real ACROSS PROCESSES, which is the case that matters when
    # an operator starts a second survey in another terminal.
    with SurveyOutputLock(tmp_path):
        held = _lock_child(tmp_path)
    assert held.returncode == 0, held.stderr
    assert held.stdout.startswith("REFUSED"), (held.stdout, held.stderr)
    assert "already in use by another active survey" in held.stdout

    # ... and that the lock is genuinely released, not merely unavailable.
    free = _lock_child(tmp_path)
    assert free.stdout.strip() == "ACQUIRED", (free.stdout, free.stderr)
