#!/usr/bin/env python3
"""
Selection of the events that a CHIME survey probes.

Two lines of survey() in cadc.py shape every inventory this tool publishes:

  * The outrigger membership filter removes 6154 of the 16523 enumerated
    events on the live archive (37%). A regression would add about 6000
    outrigger events to a CHIME-only inventory with no warning, and every
    downstream gate would read them as archive drift instead of as a bug.
  * `sorted(...)` sets which events a bounded `--max-events` run picks, and
    so makes a resume boundary reproducible. The row order of the inventory
    is set instead by the view query in `survey_state.py` (ORDER BY
    event_key, ordinal), so a test of row order says nothing about this line.

The tests drive the real survey() and check what it did. They do not
reimplement the predicate, since a copied predicate drifts with the code it is
meant to protect.

Enumeration and every probe are faked, so nothing is contacted.
"""
from __future__ import annotations

import contextlib
import io
import re

from pilot_proxy.archive.sources import cadc as cadc_datatrail
from pilot_proxy.archive.interfaces import RunContext

SCOPE = "chime.event.baseband.raw"
FREQ_IDS = [506, 521]

PLAIN_A = (SCOPE, "100000000")
OUTRIGGER = (SCOPE, "100000001")
PLAIN_B = (SCOPE, "100000002")

# labels sampled from the live enum_cache.json
MEMBERSHIP = {
    PLAIN_A: ["backlog.pulsar.B0355+54"],
    OUTRIGGER: ["outrigger.commissioning.B0643"],
    PLAIN_B: ["backlog.pulsar.B0329+54"],
}


@contextlib.contextmanager
def fake_archive(membership):
    """Fake enumeration and every probe, and record the events that were probed."""
    probed = []
    orig_enum = cadc_datatrail._enumerate_events
    orig_cp = cadc_datatrail.DATATRAIL.common_path
    orig_size = cadc_datatrail.CadcDatatrailSource._cadc_size

    def fake_size(self, uri, **kwargs):
        m = re.search(r"astro_(\d+)", uri)
        if m:
            probed.append(m.group(1))
        return 91311880, None

    cadc_datatrail._enumerate_events = lambda *a, **k: dict(membership)
    cadc_datatrail.DATATRAIL.common_path = (
        lambda scope, ev, **kwargs:
        (f"cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_{ev}", True))
    cadc_datatrail.CadcDatatrailSource._cadc_size = fake_size
    try:
        yield probed
    finally:
        cadc_datatrail._enumerate_events = orig_enum
        cadc_datatrail.DATATRAIL.common_path = orig_cp
        cadc_datatrail.CadcDatatrailSource._cadc_size = orig_size


def _survey(out_dir, **options):
    opts = {"freq_ids": list(FREQ_IDS)}
    opts.update(options)
    ctx = RunContext(instrument=None, selection=None, options=opts)
    src = cadc_datatrail.CadcDatatrailSource()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        src.survey(ctx, str(out_dir))
    return buf.getvalue()


def _to_survey(out):
    m = re.search(r"to survey: (\d+) events", out)
    assert m, out
    return int(m.group(1))


def test_outrigger_labelled_events_are_excluded_by_default(tmp_path):
    with fake_archive(MEMBERSHIP) as probed:
        out = _survey(tmp_path / "inv")
    assert _to_survey(out) == 2, out
    assert sorted(set(probed)) == ["100000000", "100000002"], (
        "an outrigger-labelled event must never be probed by a CHIME survey")


def test_include_outrigger_admits_them(tmp_path):
    with fake_archive(MEMBERSHIP) as probed:
        out = _survey(tmp_path / "inv", include_outrigger=True)
    assert _to_survey(out) == 3, out
    assert "100000001" in probed


def test_the_outrigger_pattern_matches_the_labels_the_archive_really_uses():
    for label in ("outrigger.commissioning.B0643",
                  "outrigger.commissioning.B2319",
                  "outrigger.commissioning.J0341+5711"):
        assert cadc_datatrail._OUTRIGGER_RE.search(label), label
    for label in ("backlog.pulsar.B0355+54", "chime.event.baseband.raw"):
        assert not cadc_datatrail._OUTRIGGER_RE.search(label), label


def test_max_events_takes_the_lowest_keys_whatever_order_enumeration_yields(
        tmp_path):
    # the cache returns events in an arbitrary order, reversed here, and a
    # bounded run must still take a deterministic prefix, or two resumes of the
    # same command survey different subsets of the archive
    reversed_membership = dict(reversed(list(MEMBERSHIP.items())))
    with fake_archive(reversed_membership) as probed:
        _survey(tmp_path / "inv", max_events=1)
    assert sorted(set(probed)) == ["100000000"], (
        f"--max-events must start from the lowest key, probed {probed}")
