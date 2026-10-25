#!/usr/bin/env python3
"""
Offline tests for the shared `--select` grammar (pilot_proxy.archive.selection)
and for the one place its verdict becomes science: the tail of
CadcDatatrailSource.enumerate(), which turns a finished inventory into the scan
units -- a filter defect there silently drops units from a SCAN.

Nothing here touches CANFAR, CADC or datatrail, which is where the enumerate
path otherwise only exercises: the archive boundary is a hand-written
inventory.jsonl under tmp_path, and enumerate() reads it with no client
constructed and no network call on that path at all.

Every assertion drives the real parser and checks what it returned, or the
exact SystemExit text a user would see. No test re-states the predicate under
test -- a copied predicate passes just as happily when the source is broken.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_selection_parsing.py
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from pilot_proxy.archive import selection as selection_mod
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.selection import (
    Selection,
    _parse_events,
    parse_freq_ids,
    parse_selection,
)
from pilot_proxy.archive.sources.cadc import CadcDatatrailSource


# ==========================================================================
# parse_freq_ids: every accepted form
# ==========================================================================
@pytest.mark.parametrize("sel,expected", [
    (844, {844}),
    ("844", {844}),
    ("+844", {844}),                        # an explicitly signed token is fine
    ("614,706", {614, 706}),
    ("  614 , 706 ,614 ", {614, 706}),      # surrounding space + a duplicate
    ([614, 706], {614, 706}),
    ((614, 706), {614, 706}),
    ({614, 706}, {614, 706}),
    (frozenset({614}), {614}),
    ([614, 614, 706], {614, 706}),          # duplicates collapse in collections too
    ("506-509", {506, 507, 508, 509}),      # a range is inclusive at BOTH ends
    ("5-5", {5}),                           # a degenerate range is one channel
    ("614,506-508", {506, 507, 508, 614}),  # ranges and singles mix in one list
    ("0-2", {0, 1, 2}),                     # ... and a range may start at 0
])
def test_accepted_freq_id_forms(sel, expected):
    assert parse_freq_ids(sel) == frozenset(expected)


@pytest.mark.parametrize("sel", [None, "all", "ALL", "  All  ", "*", " * "])
def test_only_none_or_explicit_all_means_no_freq_id_filter(sel):
    # None (not an empty set) is the sentinel every consumer tests for.
    assert parse_freq_ids(sel) is None


def test_written_order_is_not_a_promise():
    # The grammar answers with an unordered frozenset, so '706,614' cannot and
    # does not reorder anything downstream (scan order comes from the
    # inventory -- see test_unit_order_follows_the_inventory_not_the_select).
    forward, backward = parse_freq_ids("614,706"), parse_freq_ids("706,614")
    assert forward == backward == frozenset({614, 706})
    assert isinstance(forward, frozenset)   # immutable: safe on a frozen Selection


# ==========================================================================
# parse_freq_ids: every rejected form, and the exact reason given
# ==========================================================================
@pytest.mark.parametrize("sel,detail", [
    ("", "an empty value does not mean 'all'"),
    ("   ", "an empty value does not mean 'all'"),
    ([], "an empty collection does not mean 'all'"),
    ((), "an empty collection does not mean 'all'"),
    (set(), "an empty collection does not mean 'all'"),
    (-5, "freq_id -5 is negative"),
    ([-5], "freq_id -5 is negative"),
    ("abc", "bad token 'abc'"),
    ("6a4", "bad token '6a4'"),
    ("844 706", "bad token '844 706'"),     # space is not a separator
    (3.5, "bad token '3.5'"),
    (True, "bad token 'true'"),             # a bare bool is not channel 1
    ([True], "boolean True is not a channel id"),
    ([3.5], "float 3.5 is not an integer"),
    ([None], "NoneType None is not an integer"),
    ("1,,2", "empty comma-separated token"),
    ("1-2-3", "bad range token '1-2-3'"),
    ("-5", "bad token ''"),                 # a lone '-' reads as a low-less range
])
def test_rejected_freq_id_forms(sel, detail):
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids(sel)
    message = str(excinfo.value)
    assert detail in message                # the specific reason, not a generic failure
    assert repr(sel) in message             # the offending value is echoed back
    assert "a freq_id selection is an int" in message   # ... plus the grammar


def test_inverted_range_is_rejected_and_the_fix_is_spelled_out():
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids("844-506")
    message = str(excinfo.value)
    # Accepting this would resolve to nothing (or, if the bound check were
    # dropped, to everything) -- both are silent science loss.
    assert message.startswith("empty freq_id range '844-506' (low > high)")
    assert "write it as '506-844'" in message


def test_range_beyond_the_cardinality_guard_is_refused_not_materialised():
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids("0-1000000")
    assert "range '0-1000000' expands to too many channels" in str(excinfo.value)


def test_comma_list_cardinality_guard_counts_only_distinct_channels(monkeypatch):
    # The real bound is a million channels; patching it drives the same guard
    # without allocating a million channels inside a unit test.
    monkeypatch.setattr(selection_mod, "_MAX_SELECTION_CARDINALITY", 3)
    assert parse_freq_ids("1,2,3") == frozenset({1, 2, 3})    # exactly at the bound
    assert parse_freq_ids("1,2,3,3,1") == frozenset({1, 2, 3})  # repeats are free
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids("1,2,3,4")
    assert "selection expands to too many channels" in str(excinfo.value)


def test_event_string_in_a_freq_id_slot_names_the_mistake():
    for sel in ("events:343434343", "EVENT:343434343"):
        with pytest.raises(SystemExit) as excinfo:
            parse_freq_ids(sel)
        message = str(excinfo.value)
        assert "is an event selection in a freq_id slot" in message
        assert "never inside 'freq_ids'" in message


# ==========================================================================
# instrument geometry: out-of-range vs n_channels, and the 0 / n-1 boundary
# ==========================================================================
@pytest.mark.parametrize("sel", [1024, "1024", [1024], "1020-1024", "700,1024"])
def test_freq_id_at_or_above_n_channels_is_rejected(sel):
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids(sel, n_channels=1024)
    assert ("freq_id 1024 is outside this instrument's channels 0..1023"
            in str(excinfo.value))


@pytest.mark.parametrize("sel,expected", [
    (0, {0}),                               # first channel
    (1023, {1023}),                         # last channel of a 1024-ch instrument
    ("0,1023", {0, 1023}),
    ("1021-1023", {1021, 1022, 1023}),      # a range may END on the last channel
    ("0-1", {0, 1}),
])
def test_boundary_channels_are_inside_the_geometry(sel, expected):
    # An off-by-one here silently drops the first or last channel of a survey.
    assert parse_freq_ids(sel, n_channels=1024) == frozenset(expected)


def test_geometry_is_only_enforced_when_the_instrument_declares_it():
    # No n_channels means no upper bound -- large ids are legal, not clamped.
    assert parse_freq_ids(9999) == frozenset({9999})


@pytest.mark.parametrize("bad", [0, -1, True, 3.0, "1024"])
def test_invalid_n_channels_is_rejected(bad):
    with pytest.raises(SystemExit) as excinfo:
        parse_freq_ids(844, n_channels=bad)
    assert str(excinfo.value) == (
        f"invalid n_channels {bad!r}: expected a positive integer")


def test_n_channels_is_validated_even_for_a_no_filter_selection():
    # parse_selection validates geometry up front, before it looks at `sel`.
    with pytest.raises(SystemExit, match="invalid n_channels 0"):
        parse_selection(None, n_channels=0)


# ==========================================================================
# _parse_events: ids are archive identifiers, compared as strings
# ==========================================================================
@pytest.mark.parametrize("sel,expected", [
    ("343434343", {"343434343"}),
    (343434343, {"343434343"}),             # an int id normalizes to its string
    ("343434343,353535353", {"343434343", "353535353"}),
    (" 343434343 , 353535353 , 343434343 ", {"343434343", "353535353"}),
    ([343434343, "353535353"], {"343434343", "353535353"}),
    (("A", "B"), {"A", "B"}),
    (frozenset({"A"}), {"A"}),
    (["A", "A"], {"A"}),                    # duplicates collapse
    ("E1", {"E1"}),                         # ids need not be numeric ...
    ("AbC", {"AbC"}),                       # ... and keep their case
    ("ALL_EVENTS", {"ALL_EVENTS"}),         # only bare 'all' is the wildcard
])
def test_accepted_event_forms(sel, expected):
    assert _parse_events(sel) == frozenset(expected)


@pytest.mark.parametrize("sel", [None, "all", "ALL", " All ", "*", " * "])
def test_only_none_or_explicit_all_means_no_event_filter(sel):
    assert _parse_events(sel) is None


@pytest.mark.parametrize("sel,detail", [
    ("", "an empty value does not mean 'all'"),
    ("   ", "an empty value does not mean 'all'"),
    ([], "an empty collection does not mean 'all'"),
    ((), "an empty collection does not mean 'all'"),
    ("A,,B", "empty event id"),
    (" , ", "empty event id"),
    (["A", " "], "empty event id"),
    (True, "expected event ids"),
    ([None], "nested NoneType value None is not an event id"),
    ([True], "nested bool value True is not an event id"),
    ([3.5], "nested float value 3.5 is not an event id"),
    (3.5, "expected an event id or a collection of event ids"),
    ({"a": 1}, "expected an event id or a collection of event ids"),
])
def test_rejected_event_forms(sel, detail):
    with pytest.raises(SystemExit) as excinfo:
        _parse_events(sel)
    assert str(excinfo.value) == f"malformed event selection {sel!r}: {detail}"


def test_event_cardinality_guard_counts_only_distinct_events(monkeypatch):
    monkeypatch.setattr(selection_mod, "_MAX_SELECTION_CARDINALITY", 3)
    assert _parse_events("a,b,c") == frozenset({"a", "b", "c"})
    assert _parse_events("a,b,c,a") == frozenset({"a", "b", "c"})
    with pytest.raises(SystemExit) as excinfo:
        _parse_events("a,b,c,d")
    assert str(excinfo.value) == ("malformed event selection 'a,b,c,d': "
                                  "selection expands to too many events")


# ==========================================================================
# parse_selection: the CLI prefix form, the dict form, and the fall-through
# ==========================================================================
@pytest.mark.parametrize("text,expected", [
    ("events:343434343", {"343434343"}),
    ("event:343434343", {"343434343"}),            # singular prefix too
    ("events:343434343,353535353", {"343434343", "353535353"}),
    ("  EVENTS: AbC , dEF ", {"AbC", "dEF"}),      # prefix folds case, ids do not
])
def test_event_prefix_selection(text, expected):
    sel = parse_selection(text)
    assert sel.events == frozenset(expected)
    assert sel.freq_ids is None       # an events: selection filters events ONLY


@pytest.mark.parametrize("text", ["events:all", "event:*", "EVENTS:ALL"])
def test_all_inside_an_event_prefix_is_an_error_not_a_no_op(text):
    # 'all' here would parse to "no event filter" -- i.e. every event -- from a
    # string the user wrote to NARROW the run. Refused rather than inverted.
    with pytest.raises(SystemExit) as excinfo:
        parse_selection(text)
    assert str(excinfo.value) == (
        f"empty event selection: {text!r} (expected e.g. 'events:343434343' "
        "or 'events:E1,E2')")


def test_bare_event_prefix_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        parse_selection("events:")
    assert str(excinfo.value) == (
        "malformed event selection '': an empty value does not mean 'all'")


@pytest.mark.parametrize("text", ["eventx:1", "events", "event"])
def test_a_near_miss_prefix_falls_through_to_the_freq_id_grammar(text):
    with pytest.raises(SystemExit, match=r"bad token"):
        parse_selection(text)


@pytest.mark.parametrize("sel,freq_ids,events", [
    (None, None, None),
    ("all", None, None),
    ("844", frozenset({844}), None),
    (844, frozenset({844}), None),
    ([614, 706], frozenset({614, 706}), None),
    ("506-508", frozenset({506, 507, 508}), None),
])
def test_whole_selection_forms_reaching_the_freq_id_grammar(sel, freq_ids, events):
    parsed = parse_selection(sel)
    assert parsed.freq_ids == freq_ids and parsed.events == events


def test_dict_form_ands_the_two_filters():
    sel = parse_selection({"freq_ids": "614,706", "events": [343434343]},
                          n_channels=1024)
    assert sel.freq_ids == frozenset({614, 706})
    assert sel.events == frozenset({"343434343"})


def test_dict_form_may_carry_one_axis_only():
    channels = parse_selection({"freq_ids": 844})
    assert channels.freq_ids == frozenset({844}) and channels.events is None
    events = parse_selection({"events": "E1"})
    assert events.events == frozenset({"E1"}) and events.freq_ids is None


def test_dict_with_explicit_none_is_no_filter_on_that_axis():
    assert parse_selection({"freq_ids": None, "events": None}) == Selection()


def test_empty_dict_is_rejected():
    with pytest.raises(SystemExit) as excinfo:
        parse_selection({})
    assert str(excinfo.value) == (
        "selection dict is empty; specify 'freq_ids' and/or 'events', or use "
        "explicit 'all'/'*' for no filter")


def test_unknown_dict_keys_are_named_and_sorted():
    with pytest.raises(SystemExit) as excinfo:
        parse_selection({"freq_id": 844, "evens": "E"})
    message = str(excinfo.value)
    # a typo'd key must not be ignored: it would widen the run, not narrow it
    assert "unknown key(s) ['evens', 'freq_id']" in message
    assert "the accepted keys are ['events', 'freq_ids']" in message


def test_event_string_in_the_freq_ids_key_is_rejected():
    with pytest.raises(SystemExit, match="event selection in a freq_id slot"):
        parse_selection({"freq_ids": "events:343434343"})


def test_events_key_takes_bare_ids_the_prefix_is_cli_sugar_only():
    # Inside the dict the value is a list of ids, taken verbatim -- so this is
    # one event literally named 'events:1', not a nested prefix form.
    assert (parse_selection({"events": "events:1"}).events
            == frozenset({"events:1"}))


def test_selection_is_frozen():
    # enumerate() shares one Selection across every row; it must not be
    # mutable state.
    with pytest.raises(dataclasses.FrozenInstanceError):
        Selection().freq_ids = frozenset({1})


# ==========================================================================
# Selection.wants_freq_id / wants_event truth tables
# ==========================================================================
_CHANNELS = Selection(freq_ids=frozenset({0, 614, 1023}))
_EVENTS = Selection(events=frozenset({"343434343", "E1"}))


@pytest.mark.parametrize("value,wanted", [
    (0, True),                  # boundary: first channel
    (1023, True),               # boundary: last channel of a 1024-ch instrument
    (614, True),
    ("614", True),              # a row may carry freq_id as a string
    (" 614 ", True),
    ("+614", True),
    ("0", True),
    (1, False),
    (615, False),
    (1022, False),
    (1024, False),              # boundary: one past the last channel
    (-1, False),
    (None, False),              # a unit with NO freq_id fails an exact filter
    (True, False),              # bool is not channel 1 ...
    (False, False),             # ... and not channel 0, which IS selected here
    (614.0, False),             # a float is not a channel id
    ("614x", False),
    ("", False),
    ("   ", False),
])
def test_wants_freq_id_truth_table(value, wanted):
    assert _CHANNELS.wants_freq_id(value) is wanted


@pytest.mark.parametrize("value", [0, 1023, 1024, None, True, "anything", 614.0])
def test_no_freq_id_filter_wants_every_unit(value):
    assert Selection().wants_freq_id(value) is True


@pytest.mark.parametrize("value,wanted", [
    ("343434343", True),
    (343434343, True),          # int in the row, str in the filter: same event
    (" 343434343 ", True),
    ("E1", True),
    ("e1", False),              # event ids are case-sensitive
    ("349382978", False),
    ("3434343430", False),
    (None, False),              # a unit with NO event fails an exact filter
    (True, False),
    (1, False),
    ("", False),
])
def test_wants_event_truth_table(value, wanted):
    assert _EVENTS.wants_event(value) is wanted


@pytest.mark.parametrize("value", [None, "anything", 7, False])
def test_no_event_filter_wants_every_unit(value):
    assert Selection().wants_event(value) is True


def test_the_two_filters_are_independent():
    both = Selection(freq_ids=frozenset({614}), events=frozenset({"A"}))
    assert both.wants_freq_id(614) is True and both.wants_event("B") is False
    assert both.wants_freq_id(706) is False and both.wants_event("A") is True


# ==========================================================================
# The consequence: parse_selection inside CadcDatatrailSource.enumerate().
# The inventory below is the fake archive boundary -- four rows, three shapes:
# two per-channel baseband files for event A, a per-event calibration product
# for A with NO freq_id column, and one baseband file for event B.
# ==========================================================================
_ROWS = [
    dict(scope="test.scope", event="A", name="a614.h5",
         common_path="cadc:TEST/A", size_bytes=10, freq_id=614),
    dict(scope="test.scope", event="A", name="a706.h5",
         common_path="cadc:TEST/A", size_bytes=10, freq_id=706),
    dict(scope="test.scope", event="A", name="gains_A.h5",
         common_path="cadc:TEST/A", size_bytes=10),
    dict(scope="test.scope", event="B", name="b614.h5",
         common_path="cadc:TEST/B", size_bytes=10, freq_id=614),
]


def _enumerate(tmp_path, selection, *, rows=_ROWS, n_channels=1024):
    """Names of the units the real enumerate() yields for `selection`.

    `--inventory` short-circuits path resolution, so this never reads the
    shared archive, and enumerate() constructs no CADC client.
    """
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ctx = RunContext(
        instrument=SimpleNamespace(name="chime", n_channels=n_channels),
        selection=selection,
        options={"inventory": str(inventory)},
    )
    return [unit.name for unit in CadcDatatrailSource().enumerate(ctx)]


def test_no_selection_scans_the_whole_inventory(tmp_path):
    every = ["a614.h5", "a706.h5", "gains_A.h5", "b614.h5"]
    assert _enumerate(tmp_path, None) == every
    assert _enumerate(tmp_path, "all") == every


def test_selection_naming_an_absent_freq_id_invents_no_unit(tmp_path):
    # 999 is a legal channel this inventory simply does not contain. The
    # answer is the units that exist -- never a fabricated one for 999.
    assert _enumerate(tmp_path, "614,999") == ["a614.h5", "b614.h5"]
    assert _enumerate(tmp_path, 999) == []
    assert _enumerate(tmp_path, "900-999") == []


def test_inventory_freq_id_outside_the_selection_is_dropped(tmp_path):
    # 706 is in the inventory but not in the selection: it must not be scanned.
    assert _enumerate(tmp_path, "614") == ["a614.h5", "b614.h5"]
    assert _enumerate(tmp_path, "706") == ["a706.h5"]


def test_a_unit_with_no_freq_id_fails_a_freq_id_filter(tmp_path):
    # Exact-match semantics: the per-event calibration row has no freq_id at
    # all, so a channel selection excludes it -- but 'all' still yields it.
    assert _enumerate(tmp_path, "614,706") == ["a614.h5", "a706.h5", "b614.h5"]
    assert "gains_A.h5" in _enumerate(tmp_path, "all")


def test_event_selection_keeps_every_shape_of_that_event(tmp_path):
    assert _enumerate(tmp_path, "events:A") == ["a614.h5", "a706.h5",
                                                "gains_A.h5"]
    assert _enumerate(tmp_path, "events:Z") == []


def test_both_filters_are_anded_over_the_inventory(tmp_path):
    assert _enumerate(tmp_path, {"freq_ids": 614, "events": "A"}) == ["a614.h5"]
    assert _enumerate(tmp_path, {"freq_ids": 706, "events": "B"}) == []


def test_unit_order_follows_the_inventory_not_the_select_string(tmp_path):
    # The grammar returns an unordered set, so the written order of --select
    # cannot reorder a run; the inventory's order is the run's order.
    assert (_enumerate(tmp_path, "706,614")
            == _enumerate(tmp_path, "614,706")
            == ["a614.h5", "a706.h5", "b614.h5"])


def test_a_malformed_select_stops_the_scan_before_any_unit(tmp_path):
    with pytest.raises(SystemExit, match="low > high"):
        _enumerate(tmp_path, "706-614")


def test_an_inventory_row_outside_instrument_geometry_is_a_hard_error(tmp_path):
    # The row parser proves freq_id is a non-negative int; the shared grammar
    # supplies the instrument bound, so a 1024-channel scan cannot ingest 5000.
    rows = [dict(scope="test.scope", event="A", name="x.h5",
                 common_path="cadc:TEST/A", size_bytes=10, freq_id=5000)]
    with pytest.raises(SystemExit) as excinfo:
        _enumerate(tmp_path, "all", rows=rows)
    assert ("freq_id 5000 is outside this instrument's channels 0..1023"
            in str(excinfo.value))


def test_a_selection_outside_instrument_geometry_is_refused_not_silently_empty(
        tmp_path):
    # enumerate() must hand the instrument geometry to the SELECTION parser
    # (cadc.py:357). Without it, --select 5000 on a 1024-channel instrument is
    # not an error: it resolves to a legal-looking filter that matches nothing,
    # and the run reports zero units instead of naming the mistake. The sibling
    # test covers the same geometry check on the ROW path; this covers the
    # selection path, which nothing else reaches.
    with pytest.raises(SystemExit) as excinfo:
        _enumerate(tmp_path, 5000)
    assert "freq_id 5000 is outside this instrument" in str(excinfo.value)
    assert "0..1023" in str(excinfo.value)
    with pytest.raises(SystemExit, match=r"0\.\.1023"):
        _enumerate(tmp_path, "1020-1030")
    with pytest.raises(SystemExit, match=r"0\.\.1023"):
        _enumerate(tmp_path, {"freq_ids": 2048})
