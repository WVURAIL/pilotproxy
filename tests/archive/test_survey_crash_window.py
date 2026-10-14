#!/usr/bin/env python3
"""
Offline tests for the two survey() guarantees that only ever matter when a run
STOPS -- the pieces that otherwise only exercise on CANFAR against live
Datatrail/CADC, and that a green suite cannot see because survey()'s own finally
block tidies up after itself before any test gets to look.

  * attempts.json is checkpointed PER EVENT (cadc.py:605 in mark_done, :612 in
    bump), not merely once on the way out. The final rewrite in the finally
    block (cadc.py:833) leaves the same end state either way, so the per-event
    flush is observable only from INSIDE the event loop -- here from a faked
    cadcinfo probe belonging to a LATER event, which is exactly the window a
    killed process falls into.
  * pool.shutdown(wait=True, cancel_futures=True) (cadc.py:832) joins probes
    that are still in flight before survey() returns and the output lock is
    released. Observed by timestamping every faked probe's entry and exit and
    comparing them against the instant survey() handed the directory back.

Nothing here touches the network or a certificate: event enumeration,
`datatrail ps` and cadcinfo are the three faked seams -- the same three
tests/archive/test_survey_durability_stress.py fakes.

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
OBS_DIR = "raw/2020/07/15"            # matches _DATE_RE (cadc.py:111)
FIDS = [614, 706, 800, 900]           # -> four candidate files per event
FLOOR = 1 << 20                       # ChimeBasebandReader.minimum_archive_bytes
# The faked observation is from 2020. Declaring a far larger --empty-age-days
# keeps a 0-file event OFF the accept-as-empty-on-first-sighting path, so it
# takes bump() instead of mark_done() -- which is the branch under test.
NEVER_AGED_OUT = 100_000

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX signal and flock semantics under test")


# ---------------------------------------------------------------- fixture ---
def events(n, start=100000000):
    return [str(start + i) for i in range(n)]


def key_of(event):
    return f"{SCOPE}|{event}"


def read_attempts(out):
    """attempts.json exactly as a killed process would leave it behind."""
    return json.loads((Path(out) / "attempts.json").read_text())


@contextlib.contextmanager
def fake_archive(evs, present=(), on_probe=None):
    """Patch the three live seams survey() leans on; restore them on exit.

    `on_probe(event, name)` runs INSIDE the faked cadcinfo call, i.e. on a pool
    worker while survey() is still mid-event. That is the only vantage point
    from which per-event durable state is visible at all.
    """
    present = set(present)
    saved = (C._enumerate_events, C.DATATRAIL.common_path,
             C.CadcDatatrailSource._cadc_size)
    membership = {(SCOPE, ev): ["dataset0"] for ev in evs}
    # common path -> event, so the probe never has to re-derive (and thereby
    # re-implement) the reader's archive naming just to know who it is.
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
        # Absent is an ANSWER (None, None), not an error -- cadc.py leans on
        # that distinction to tell "empty" from "incomplete".
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
    """One survey pass over the faked archive; returns its stdout."""
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

    One event's probes all run inside the same verify() call, so they all see
    the same durable state; the first one to arrive is enough, and taking only
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


# ============================ 1. per-event attempts.json checkpoint =========
def test_bump_checkpoints_attempts_before_the_next_event_is_probed(tmp_path):
    """cadc.py:612 -- bump() must FLUSH the counter, not just raise it in RAM.

    Event 0 resolves a common path but has no file above the floor, so it is
    bumped rather than committed. If that bump only lived in the in-memory
    dict, a process killed later in the run would lose it: the event would
    restart from attempt 0 on every resume and re-probe forever instead of ever
    reaching the accept-as-empty terminus. The only witness is event 1's probe,
    which runs while survey() is still inside the loop.
    """
    out = tmp_path / "inv"
    evs = events(2)
    sighting = FirstSighting(out)

    with fake_archive(evs, present={evs[1]}, on_probe=sighting):
        survey(out)

    seen = sighting.seen
    assert sorted(seen) == sorted(evs)          # both events really were probed
    # Nothing checkpointed when event 0 is probed; event 0's attempt already
    # DURABLE by the time event 1 is probed; unchanged once the run is over --
    # that third entry is why the end state alone can never see this flush.
    assert [seen[evs[0]], seen[evs[1]], read_attempts(out)] == [
        {}, {key_of(evs[0]): 1}, {key_of(evs[0]): 1}]


def test_mark_done_checkpoints_the_dropped_attempt_immediately(tmp_path):
    """cadc.py:605 -- mark_done() must FLUSH the pop, not just pop in RAM.

    A committed event's attempt counter is stale the instant the event lands.
    Leaving it on disk until the finally block means a kill in between resumes
    carrying a phantom count against an already-completed key. Setup is a real
    first pass that leaves both events bumped; the second pass commits event 0,
    and event 1's probe reports what the counter file looked like right after.
    """
    out = tmp_path / "inv"
    evs = events(2)

    with fake_archive(evs, present=()):        # nothing present -> both bumped
        survey(out)
    assert read_attempts(out) == {key_of(evs[0]): 1, key_of(evs[1]): 1}

    sighting = FirstSighting(out)
    # Event 0's files have now landed, so it commits; event 1 is still empty
    # and is probed afterwards, from inside the same loop.
    with fake_archive(evs, present={evs[0]}, on_probe=sighting):
        survey(out)

    seen = sighting.seen
    assert sorted(seen) == sorted(evs)
    # Before event 0 commits, both counters are on disk ...
    assert seen[evs[0]] == {key_of(evs[0]): 1, key_of(evs[1]): 1}
    # ... and its counter is off disk before anything else is probed.
    assert seen[evs[1]] == {key_of(evs[1]): 1}
    # The end state is identical whether or not mark_done flushed, which is
    # precisely why the assertion above has to be taken from inside the loop.
    assert read_attempts(out) == {key_of(evs[1]): 2}


# ================================ 2. probes joined at teardown =============
LINGER = 0.5           # how long a probe deliberately outlives the interrupt


def test_survey_joins_in_flight_probes_before_it_returns(tmp_path):
    """cadc.py:832 -- pool.shutdown(wait=True, cancel_futures=True) on the way out.

    A SIGINT arrives with every probe of an event genuinely running. survey()
    must not return -- releasing the output lock, telling the caller the
    directory is theirs again -- while a probe is still executing against the
    archive. The probes are barrier-synchronised so "in flight" is a fact and
    not a timing hope, and their linger starts only after the signal has been
    raised, so the window cannot close early under load.
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
            time.sleep(LINGER)            # ...still working, post-interrupt
        with guard:
            exited.append((name, time.monotonic()))

    with fake_archive(evs, present=set(evs), on_probe=probe_lifetime):
        with pytest.raises(KeyboardInterrupt):
            survey(out, workers=len(FIDS))
        returned_at = time.monotonic()
        with guard:
            started, finished = list(entered), dict(exited)

    assert len(started) == len(FIDS)                # every probe really ran
    # THE property: nothing was still executing when survey() returned.
    assert sorted(finished) == sorted(started)
    assert max(finished.values()) <= returned_at
    # Not vacuous: the other three were provably mid-probe when the signal
    # landed and stayed there for the whole linger, so the join was real work.
    t_sigint = fired[0]
    lingered = [t for t in finished.values() if t >= t_sigint + LINGER]
    assert len(lingered) == len(FIDS) - 1
    with SurveyOutputLock(out):                     # ...and the lock was freed
        pass
