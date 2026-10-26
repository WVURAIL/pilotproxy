"""Select archive units by frequency channel, event, or both.

Every source reads the user's selection from ``ctx.selection``. A channel
filter selects that frequency across events, and an event filter can select all
frequencies of one acquisition. When both are supplied, a unit must pass both.

Accepted forms::

    None / "all" / "*"             -> no filter (every unit)
    844 / [614, 706] / "614,706"
      / "506-844"                  -> freq_ids
    "events:343434343,353535353"
      / "event:343434343"          -> events (CLI prefix form)
    {"freq_ids": <any form above>,
     "events": [343434343, ...]}   -> both filters

Event selection requires an explicit dictionary key or prefix. Inferring it
from an integer's size would make the grammar depend on the current range of
archive IDs. A unit with no value for a filtered field does not match.
For example, a frequency filter excludes a calibration product that has no
freq_id.

Empty strings or collections, comma-only values, nested collections, negative
channels, and channels at or above the instrument's channel count are errors.
Only an explicit ``all`` or ``*`` selects every unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, FrozenSet, Mapping, Optional

_ALL = ("all", "*")
_EVENT_PREFIXES = ("events:", "event:")
_DICT_KEYS = {"freq_ids", "events"}
_GRAMMAR = ("a freq_id selection is an int, a collection of ints, '844', "
            "'614,706', a range '506-844', or 'all'")
_MAX_SELECTION_CARDINALITY = 1_000_000


def _selection_error(sel, detail: str = "") -> SystemExit:
    suffix = f" ({detail})" if detail else ""
    return SystemExit(f"malformed freq_id selection {sel!r}{suffix}: {_GRAMMAR}")


def _validated_n_channels(n_channels: Optional[int]) -> Optional[int]:
    """Return the optional instrument channel count after a strict type check."""
    if n_channels is None:
        return None
    if (isinstance(n_channels, bool)
            or not isinstance(n_channels, Integral)
            or n_channels <= 0):
        raise SystemExit(
            f"invalid n_channels {n_channels!r}: expected a positive integer")
    return int(n_channels)


def _freq_id(value, original, n_channels: Optional[int]) -> int:
    """Return one channel id as an int, checked against the instrument geometry."""
    if isinstance(value, bool):
        raise _selection_error(original, f"boolean {value!r} is not a channel id")
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.lstrip("+-").isdigit():
            raise _selection_error(original, f"bad token {value!r}")
        out = int(text)
    else:
        raise _selection_error(
            original, f"{type(value).__name__} {value!r} is not an integer")
    if out < 0:
        raise _selection_error(original, f"freq_id {out} is negative")
    if n_channels is not None and out >= n_channels:
        raise _selection_error(
            original,
            f"freq_id {out} is outside this instrument's channels "
            f"0..{n_channels - 1}",
        )
    return out


def _add_bounded(out: set[int], values, original) -> None:
    """Add each channel to ``out`` once, up to ``_MAX_SELECTION_CARDINALITY``."""
    for channel in values:
        if channel in out:
            continue
        if len(out) >= _MAX_SELECTION_CARDINALITY:
            raise _selection_error(
                original, "selection expands to too many channels")
        out.add(channel)


def parse_freq_ids(sel, *, n_channels: Optional[int] = None) -> Optional[FrozenSet[int]]:
    """Parse a freq_id selection into a set, or None for every freq_id.

    The accepted forms are those an analyzer's plan_runs passes down::

      None / 'all' / '*'       -> None  (no filter, every freq_id)
      int                      -> {int}
      list / tuple / set       -> {ints}
      '844'                    -> {844}
      '614,706'                -> {614, 706}
      '506-844'                -> {506, 507, ..., 844}

    Malformed input raises SystemExit with a message that names the bad token,
    so a typo in plan_runs is reported at its source. A bare ``int()`` call
    would raise ``ValueError: invalid literal`` three frames deep.
    """
    n_channels = _validated_n_channels(n_channels)
    if sel is None:
        return None
    if isinstance(sel, Integral) and not isinstance(sel, bool):
        return frozenset({_freq_id(sel, sel, n_channels)})
    if isinstance(sel, (list, tuple, set, frozenset)):
        if not sel:
            raise _selection_error(sel, "an empty collection does not mean 'all'")
        out: set[int] = set()
        _add_bounded(
            out, (_freq_id(x, sel, n_channels) for x in sel), sel)
        return frozenset(out)
    s = str(sel).strip().lower()
    if s in _ALL:
        return None
    if not s:
        raise _selection_error(sel, "an empty value does not mean 'all'")
    if s.startswith(_EVENT_PREFIXES):
        # an event selection in the freq_ids slot, {"freq_ids": "events:..."}
        raise SystemExit(
            f"{sel!r} is an event selection in a freq_id slot -- events go in "
            f"the 'events' key of a selection dict, or as the whole selection "
            f"('--select events:...'), never inside 'freq_ids'")
    out: set[int] = set()

    for part in s.split(","):
        part = part.strip()
        if not part:
            raise _selection_error(sel, "empty comma-separated token")
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise _selection_error(sel, f"bad range token {part!r}")
            lo = _freq_id(pieces[0], sel, n_channels)
            hi = _freq_id(pieces[1], sel, n_channels)
            if lo > hi:
                raise SystemExit(
                    f"empty freq_id range {part!r} (low > high) -- a "
                    f"reversed range would silently select every freq_id; "
                    f"write it as '{hi}-{lo}'")
            if hi - lo + 1 > _MAX_SELECTION_CARDINALITY:
                raise _selection_error(
                    sel, f"range {part!r} expands to too many channels")
            _add_bounded(out, range(lo, hi + 1), sel)
        else:
            _add_bounded(out, (_freq_id(part, sel, n_channels),), sel)
    if not out:
        raise _selection_error(sel, "selection resolved to no channels")
    return frozenset(out)


def _parse_events(sel) -> Optional[FrozenSet[str]]:
    """Parse an event selection into a frozenset of strings, or None for no filter.

    Event IDs name archive acquisitions and are never used as numbers, so they
    are compared as strings. Inventory rows can hold them as int or str,
    depending on how the JSON was written, and a filename parse always yields
    str.
    """
    if sel is None:
        return None
    if isinstance(sel, bool):
        raise SystemExit(f"malformed event selection {sel!r}: expected event ids")
    if isinstance(sel, (Integral, str)):
        text = str(sel).strip()
        if text.lower() in _ALL:
            return None
        if not text:
            raise SystemExit(
                f"malformed event selection {sel!r}: an empty value does not "
                "mean 'all'")
        items = text.split(",")
    elif isinstance(sel, (list, tuple, set, frozenset)):
        if not sel:
            raise SystemExit(
                f"malformed event selection {sel!r}: an empty collection does "
                "not mean 'all'")
        items = list(sel)
    else:
        raise SystemExit(
            f"malformed event selection {sel!r}: expected an event id or a "
            "collection of event ids")
    out: set[str] = set()
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (Integral, str)):
            raise SystemExit(
                f"malformed event selection {sel!r}: nested "
                f"{type(item).__name__} value {item!r} is not an event id")
        value = str(item).strip()
        if not value:
            raise SystemExit(
                f"malformed event selection {sel!r}: empty event id")
        if value not in out and len(out) >= _MAX_SELECTION_CARDINALITY:
            raise SystemExit(
                f"malformed event selection {sel!r}: selection expands to "
                "too many events")
        out.add(value)
    return frozenset(out)


@dataclass(frozen=True)
class Selection:
    """Parsed selection with two independent filters that a unit must both pass.

    A filter of None selects every unit.
    """
    freq_ids: Optional[FrozenSet[int]] = None
    events: Optional[FrozenSet[str]] = None

    def wants_freq_id(self, freq_id) -> bool:
        """Return True when ``freq_id`` passes the channel filter.

        Matching is exact, so a unit with no freq_id (None) fails any set filter.
        """
        if self.freq_ids is None:
            return True
        if isinstance(freq_id, bool) or freq_id is None:
            return False
        if isinstance(freq_id, Integral):
            channel = int(freq_id)
        elif isinstance(freq_id, str):
            text = freq_id.strip()
            if not text or not text.lstrip("+-").isdigit():
                return False
            channel = int(text)
        else:
            return False
        return channel in self.freq_ids

    def wants_event(self, event) -> bool:
        if self.events is None:
            return True
        if event is None:
            return False
        return str(event).strip() in self.events


def parse_selection(sel: Any, *, n_channels: Optional[int] = None) -> Selection:
    """Parse any accepted selection form into a ``Selection``.

    A malformed dict or an empty ``events:`` prefix raises SystemExit with a
    message, so a typo in a plan_runs sub-selection stops the run instead of
    selecting nothing.
    """
    n_channels = _validated_n_channels(n_channels)
    if sel is None:
        return Selection()
    if isinstance(sel, Mapping):
        if not sel:
            raise SystemExit(
                "selection dict is empty; specify 'freq_ids' and/or 'events', "
                "or use explicit 'all'/'*' for no filter")
        unknown = set(sel) - _DICT_KEYS
        if unknown:
            raise SystemExit(
                f"selection dict has unknown key(s) {sorted(unknown)}; "
                f"the accepted keys are {sorted(_DICT_KEYS)} "
                f"(got: {dict(sel)!r})")
        return Selection(freq_ids=parse_freq_ids(sel.get("freq_ids"),
                                                 n_channels=n_channels),
                         events=_parse_events(sel.get("events")))
    if isinstance(sel, str):
        low = sel.strip().lower()
        for pfx in _EVENT_PREFIXES:
            if low.startswith(pfx):
                events = _parse_events(sel.strip()[len(pfx):])
                if events is None:
                    raise SystemExit(
                        f"empty event selection: {sel!r} (expected e.g. "
                        f"'events:343434343' or 'events:E1,E2')")
                return Selection(events=events)
    return Selection(freq_ids=parse_freq_ids(sel, n_channels=n_channels))
