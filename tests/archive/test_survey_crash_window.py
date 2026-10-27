#!/usr/bin/env python3
"""
Offline tests for the two survey() guarantees that matter only when a run
stops. Without these tests they are exercised only on CANFAR against live
Datatrail and CADC, and a green suite cannot see them because the finally
block of survey() cleans up before any test can look.

  * attempts.json is checkpointed per event, in mark_done and in bump, as well
    as once on the way out. The final rewrite in the finally block leaves the
    same end state either way, so the per-event flush is observable only
    during the event loop. Here it is observed from a faked cadcinfo probe of a
    later event, which is the window a killed process falls into.
  * pool.shutdown(wait=True, cancel_futures=True) in survey() joins probes that
    are still in flight before survey() returns and the output lock is
    released. This is observed by timestamping the entry and exit of every
    faked probe and comparing them with the instant survey() returns the
    directory.

No test reaches the network or a certificate. Event enumeration,
`datatrail ps` and cadcinfo are the three faked seams, the same three faked
by tests/archive/test_survey_durability_stress.py.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_survey_crash_window.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as C
from pilot_proxy.archive.survey_state import SurveyOutputLock

SCOPE = "chime.event.baseband.raw"
OBS_DIR = "raw/2020/07/15"            # matches _DATE_RE in cadc.py
FIDS = [614, 706, 800, 900]           # four candidate files per event
FLOOR = 1 << 20                       # ChimeBasebandReader.minimum_archive_bytes
# The faked observation is from 2020. A far larger --empty-age-days keeps an
# event with no files off the path that accepts it as empty on first sighting,
# so it takes bump() instead of mark_done(). bump() is the branch under test.
NEVER_AGED_OUT = 100_000

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX signal and flock semantics under test")


# fixtures
def events(n, start=100000000):
    return [str(start + i) for i in range(n)]


def key_of(event):
    return f"{SCOPE}|{event}"


def read_attempts(out):
    """Return attempts.json as a killed process would leave it."""
    return json.loads((Path(out) / "attempts.json").read_text())


@contextlib.contextmanager
def fake_archive(evs, present=(), on_probe=None):
    """Patch the three live seams of survey() and restore them on exit.

    `on_probe(event, name)` runs in the faked cadcinfo call, on a pool worker
    while survey() is still in the middle of an event. Per-event durable state
    is visible only from there.
    """
    present = set(present)
    saved = (C._enumerate_events, C.DATATRAIL.common_path,
             C.CadcDatatrailSource._cadc_size)
    membership = {(SCOPE, ev): ["dataset0"] for ev in evs}
    # event by common path, so the probe never has to re-derive, and so
    # re-implement, the archive naming of the reader to know its event
    by_path: dict[str, str] = {}

    def fake_cp(scope, event, **kwargs):
        cp = f"cadc:CHIMEFRB/data/chime/baseband/{OBS_DIR}/astro_{event}"
        by_path[cp] = event
        return cp, True

    def fake_size(self, uri, *args, **kwargs):
        path, _, name = str(uri).rpartition("/")
        event = by_path[path]
        if on_probe is not None:
            on_probe(event, name)
        # An absent file is the answer (None, None) with no error. cadc.py uses
        # this to distinguish between "empty" and "incomplete".
        return (FLOOR + 1, None) if event in present else (None, None)

    def fake_enumerate(*args, **kwargs):
        return dict(membership)

    C._enumerate_events = fake_enumerate
    C.DATATRAIL.common_path = fake_cp
    C.CadcDatatrailSource._cadc_size = fake_size
    try:
        yield
    finally:
        (C._enumerate_events, C.DATATRAIL.common_path,
         C.CadcDatatrailSource._cadc_size) = saved


def survey(out, workers=4, **options):
    """Run one survey pass over the faked archive and return its stdout."""
    source = C.CadcDatatrailSource()
    buffer = io.StringIO()
    opts = {"freq_ids": list(FIDS), "workers": workers,
            "empty_age_days": NEVER_AGED_OUT, **options}
    with contextlib.redirect_stdout(buffer):
        source.survey(
            RunContext(instrument=None, selection=None, options=opts), str(out))
    return buffer.getvalue()


class FirstSighting:
    """Record attempts.json once per event, the first time it is probed.

    The probes of one event all run in the same verify() call, so they all
    see the same durable state. The first to arrive is enough, and taking only
    the first keeps the observation independent of probe scheduling.
    """

    def __init__(self, out):
        self.out = out
        self.seen: dict[str, dict] = {}
        self._guard = threading.Lock()

    def __call__(self, event, name):
        with self._guard:
            if event not in self.seen:
                self.seen[event] = read_attempts(self.out)


# 1. per-event attempts.json checkpoint
def test_bump_checkpoints_attempts_before_the_next_event_is_probed(tmp_path):
    """bump() writes the raised counter to disk as well as to memory.

    Event 0 resolves a common path and has no file above the floor, so it is
    bumped and not committed. If the bump lived only in the in-memory dict, a
    process killed later in the run would lose it. The event would restart
    from attempt 0 on every resume, be probed again forever, and never reach
    the accept-as-empty terminus. The flush is seen only by the probe of
    event 1, which runs while survey() is still in the loop.
    """
    out = tmp_path / "inv"
    evs = events(2)
    sighting = FirstSighting(out)

    with fake_archive(evs, present={evs[1]}, on_probe=sighting):
        survey(out)

    seen = sighting.seen
    assert sorted(seen) == sorted(evs)          # both events were probed
    # Nothing is checkpointed when event 0 is probed, and the attempt of event
    # 0 is durable by the time event 1 is probed. The state is unchanged once
    # the run is over, which is why the end state alone cannot show this flush.
    assert [seen[evs[0]], seen[evs[1]], read_attempts(out)] == [
        {}, {key_of(evs[0]): 1}, {key_of(evs[0]): 1}]


def test_mark_done_checkpoints_the_dropped_attempt_immediately(tmp_path):
    """mark_done() writes the removal of the counter to disk as well as to memory.

    The attempt counter of a committed event is stale as soon as the event
    lands. If it stayed on disk until the finally block, a kill in between
    would resume with a phantom count against a completed key. A real first
    pass leaves both events bumped. The second pass commits event 0, and the
    probe of event 1 reports the counter file as it was right after.
    """
    out = tmp_path / "inv"
    evs = events(2)

    with fake_archive(evs, present=()):        # nothing present, so both are bumped
        survey(out)
    assert read_attempts(out) == {key_of(evs[0]): 1, key_of(evs[1]): 1}

    sighting = FirstSighting(out)
    # The files of event 0 have landed, so it commits. Event 1 is still empty
    # and is probed afterward in the same loop.
    with fake_archive(evs, present={evs[0]}, on_probe=sighting):
        survey(out)

    seen = sighting.seen
    assert sorted(seen) == sorted(evs)
    # Before event 0 commits, both counters are on disk.
    assert seen[evs[0]] == {key_of(evs[0]): 1, key_of(evs[1]): 1}
    # Its counter is off disk before anything else is probed.
    assert seen[evs[1]] == {key_of(evs[1]): 1}
    # The end state is the same with or without the flush in mark_done, which
    # is why the assertion above is made during the loop.
    assert read_attempts(out) == {key_of(evs[1]): 2}


# 2. probes joined at teardown
LINGER = 0.5           # time (s) a probe runs on after the interrupt


def test_survey_joins_in_flight_probes_before_it_returns(tmp_path):
    """survey() calls pool.shutdown(wait=True, cancel_futures=True) on the way out.

    A SIGINT arrives while every probe of an event is running. survey() must
    not return, which releases the output lock and hands the directory back
    to the caller, while a probe is still executing against the archive. The
    probes are synchronized by a barrier, so all of them are known to be in
    flight, and their linger starts only after the signal is raised, so the
    window cannot close early under load.
    """
    out = tmp_path / "inv"
    evs = events(1)
    gate = threading.Barrier(len(FIDS))
    interrupted = threading.Event()
    fired: list[float] = []
    entered: list[str] = []
    exited: list[tuple[str, float]] = []
    guard = threading.Lock()

    def probe_lifetime(event, name):
        with guard:
            entered.append(name)
        index = gate.wait(timeout=10)     # all four probes are now running
        if index == 0:
            fired.append(time.monotonic())
            os.kill(os.getpid(), signal.SIGINT)     # lands on the main thread
            interrupted.set()
        else:
            assert interrupted.wait(timeout=10)
            time.sleep(LINGER)            # still working after the interrupt
        with guard:
            exited.append((name, time.monotonic()))

    with fake_archive(evs, present=set(evs), on_probe=probe_lifetime):
        with pytest.raises(KeyboardInterrupt):
            survey(out, workers=len(FIDS))
        returned_at = time.monotonic()
        with guard:
            started, finished = list(entered), dict(exited)

    assert len(started) == len(FIDS)                # every probe ran
    # Nothing was still executing when survey() returned.
    assert sorted(finished) == sorted(started)
    assert max(finished.values()) <= returned_at
    # The other three probes were running when the signal landed and kept
    # running for the whole linger, so the join had real work to do.
    t_sigint = fired[0]
    lingered = [t for t in finished.values() if t >= t_sigint + LINGER]
    assert len(lingered) == len(FIDS) - 1
    with SurveyOutputLock(out):                     # the lock was freed
        pass
