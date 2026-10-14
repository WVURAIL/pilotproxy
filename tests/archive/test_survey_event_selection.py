#!/usr/bin/env python3
"""
Which events a CHIME survey agrees to look at.

Two things on `cadc.py:568-570` shape every inventory this tool publishes, and
neither had a test:

  * the outrigger membership filter. On the live archive it removes 6,154 of
    16,523 enumerated events -- 37% -- so a regression would quietly add ~6,000
    outrigger events to a CHIME-only inventory, where every downstream gate
    would read them as archive drift rather than as a bug.
  * `sorted(...)`. Inventory row order is independently guaranteed by the view
    query (`survey_state.py` ORDER BY event_key, ordinal), so asserting row
    order proves nothing about this line. What `sorted` actually pins is WHICH
    events a bounded `--max-events` run picks, and therefore whether a resume
    boundary is reproducible.

These drive the REAL survey() and assert on what it did, deliberately not on a
re-implementation of the predicate -- a copied predicate drifts with the code
it is supposed to protect.

Offline: enumeration and every probe are faked; nothing is contacted.
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
    """Fake enumeration and every probe; record which events were probed."""
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
    # the cache hands events back in an arbitrary (here: reversed) order; a
    # bounded run must still take a deterministic prefix, or two resumes of the
    # same command survey different subsets of the archive
    reversed_membership = dict(reversed(list(MEMBERSHIP.items())))
    with fake_archive(reversed_membership) as probed:
        _survey(tmp_path / "inv", max_events=1)
    assert sorted(set(probed)) == ["100000000"], (
        f"--max-events must start from the lowest key, probed {probed}")
