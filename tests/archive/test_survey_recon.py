#!/usr/bin/env python3
"""
Offline tests for recon() -- the `--scopes-only` discovery walk, which
otherwise only exercises on CANFAR against the live Datatrail service.

No datatrail access. The CLI is faked at the same boundary
tests/archive/test_datatrail_adapter.py fakes it -- subprocess.run, BELOW the
adapter -- so every test here drives the real `datatrail ls --json` parsing and
therefore the real outage-vs-empty contract that recon is written against. A
fake stubbed in above the adapter (patching list_datasets_checked) would have
to re-state that contract itself, and could not tell a scope that did not
answer from a scope that answered "nothing".

Run:  PYTHONPATH=src python -m pytest tests/archive/test_survey_recon.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive import recon as rc
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as src


# ==========================================================================
# the fake datatrail CLI. recon only ever calls `datatrail ls` (three
# arities), so `listing` maps the ls argument tuple -- () for the namespace,
# (scope,) for a scope's datasets, (scope, dataset) for one level of children
# -- onto the list the service returns, or _OUTAGE for datatrail's error
# envelope. An argument tuple the test did NOT stage trips an assertion: a
# walk that recurses further than it should, or opens a dataset without
# --expand, fails loudly here rather than passing quietly.
# ==========================================================================
_OUTAGE = object()                 # {"error": ...} + exit 1 -> checked ok=False

_LS_KEY = {0: "scopes", 1: "larger_datasets", 2: "datasets"}


class _Proc:
    def __init__(self, rc_, out, err=""):
        self.returncode, self.stdout, self.stderr = rc_, out, err


def _install_fake_cli(monkeypatch, listing, calls=None):
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))

    def fake_run(cmd, **kw):
        # the invocation contract, asserted so a drifting boundary is visible
        assert cmd[:3] == [sys.executable, "-m", "dtcli.cli"], cmd
        assert cmd[3] == "ls" and cmd[-1] == "--json", cmd
        args = tuple(cmd[4:-1])
        if calls is not None:
            calls.append(args)
        assert args in listing, f"recon asked for an unstaged listing: {args}"
        answer = listing[args]
        if answer is _OUTAGE:
            return _Proc(1, json.dumps({"error": "Server not responding."}))
        return _Proc(0, json.dumps({_LS_KEY[len(args)]: list(answer)}))

    monkeypatch.setattr(dt.subprocess, "run", fake_run)


def _rows(path):
    """Every row of the map, decoded -- schema included, so an extra or
    missing key (`parent`, say) shows up as an inequality."""
    return [json.loads(line)
            for line in Path(path).read_text().splitlines() if line]


class _Inst:
    """The instrument shim survey() reads a telescope name off of."""

    def __init__(self, name):
        self.name = name


# ==========================================================================
# the happy path: explicit scopes, no filter -- one row per dataset
# ==========================================================================
def test_happy_path_writes_one_row_per_dataset(tmp_path, monkeypatch, capsys):
    listing = {
        ("chime.event.baseband.raw",): ["2023", "2024"],
        ("gbo.acquisition.processed",): ["20230530"],
    }
    calls = []
    _install_fake_cli(monkeypatch, listing, calls)

    out_dir = tmp_path / "made" / "here"          # recon creates its out_dir
    path = rc.recon(["chime.event.baseband.raw", "gbo.acquisition.processed"],
                    [], str(out_dir))

    assert path == str(out_dir / "scopes.jsonl")
    # the whole schema: two keys, scope-major order, no `parent` without
    # --expand, one JSON object per line and a trailing newline
    assert _rows(path) == [
        {"scope": "chime.event.baseband.raw", "dataset": "2023"},
        {"scope": "chime.event.baseband.raw", "dataset": "2024"},
        {"scope": "gbo.acquisition.processed", "dataset": "20230530"},
    ]
    assert Path(path).read_text().endswith("}\n")
    # explicit scopes win: the namespace is never listed, and without --expand
    # no dataset is opened (either call would be unstaged and trip the fake)
    assert calls == [("chime.event.baseband.raw",),
                     ("gbo.acquisition.processed",)]

    printed = capsys.readouterr().out
    assert "[recon] listing datasets across 2 scope(s)" in printed
    assert "(2 dataset(s))" in printed and "(1 dataset(s))" in printed
    assert f"wrote {path}: 3 rows" in printed
    assert "dataset(s) expanded" not in printed        # only with --expand
    assert "--expand" in printed                       # the not-expanded tail
    assert "[warn]" not in printed        # nothing failed -> no INCOMPLETE


# ==========================================================================
# --match semantics, read off the real filter rather than assumed
# ==========================================================================
def test_match_terms_folds_case_and_drops_empty_terms():
    assert rc.match_terms("Casa, GAIN ,, ") == ["casa", "gain"]
    assert rc.match_terms(None) == []
    assert rc.match_terms("") == []
    assert rc.match_terms(" , ") == []


def test_match_terms_are_anded_case_insensitive_substrings(tmp_path,
                                                           monkeypatch):
    listing = {("gbo.acquisition.processed",):
               ["complex_gains", "Gain_A_CasA", "housekeeping"]}
    _install_fake_cli(monkeypatch, listing)

    def kept(spec):
        path = rc.recon(["gbo.acquisition.processed"], rc.match_terms(spec),
                        str(tmp_path))
        return [row["dataset"] for row in _rows(path)]

    # one term: a plain substring, case-folded on BOTH sides (the term by
    # match_terms, the candidate by _keep) -- "Gain_A_CasA" matches "gain"
    assert kept("gain") == ["complex_gains", "Gain_A_CasA"]
    assert kept("GAIN") == ["complex_gains", "Gain_A_CasA"]
    # comma-separated terms are ANDed, not ORed
    assert kept("gain,casa") == ["Gain_A_CasA"]
    # so one unmatched term empties the map even when the other matches all
    assert kept("gain,nope") == []
    # the candidate text is "<scope> <dataset>": a term that matches only the
    # scope keeps every dataset under it, and mixes with a dataset term
    assert kept("acquisition") == ["complex_gains", "Gain_A_CasA",
                                   "housekeeping"]
    assert kept("acquisition,casa") == ["Gain_A_CasA"]
    # substring, NOT glob -- '*' is a literal character here
    assert kept("gain*") == []


# ==========================================================================
# --expand: one level down, and what happens to a container with no children
# ==========================================================================
def test_expand_opens_each_match_exactly_one_level(tmp_path, monkeypatch,
                                                   capsys):
    listing = {
        ("gbo.acquisition.processed",): ["complex_gains", "housekeeping"],
        ("gbo.acquisition.processed", "complex_gains"): ["20230530",
                                                         "20230531"],
    }
    calls = []
    _install_fake_cli(monkeypatch, listing, calls)

    path = rc.recon(["gbo.acquisition.processed"], rc.match_terms("gains"),
                    str(tmp_path), expand=True)

    # the children REPLACE their container as map rows, each carrying the
    # container it came from
    assert _rows(path) == [
        {"scope": "gbo.acquisition.processed", "dataset": "20230530",
         "parent": "complex_gains"},
        {"scope": "gbo.acquisition.processed", "dataset": "20230531",
         "parent": "complex_gains"},
    ]
    # one level only, and only for MATCHES: the filtered-out container is
    # never opened, and the children's own children are never asked for
    # (both would be unstaged listings and trip the fake)
    assert calls == [("gbo.acquisition.processed",),
                     ("gbo.acquisition.processed", "complex_gains")]

    printed = capsys.readouterr().out
    assert "expanding matches one level" in printed
    assert "match=['gains']" in printed
    assert "(2 child(ren))" in printed
    assert "2 rows (1 dataset(s) expanded)" in printed


def test_expand_keeps_a_childless_container_in_the_map(tmp_path, monkeypatch,
                                                       capsys):
    # answered, and the answer is "nothing under here" -- the container is
    # itself the resolvable handle, so it stays a row
    listing = {("s.one",): ["empty_container"],
               ("s.one", "empty_container"): []}
    _install_fake_cli(monkeypatch, listing)

    path = rc.recon(["s.one"], [], str(tmp_path), expand=True)

    assert _rows(path) == [{"scope": "s.one", "dataset": "empty_container"}]
    printed = capsys.readouterr().out
    assert "(no children listed)" in printed
    assert "1 rows (0 dataset(s) expanded)" in printed
    assert "[warn]" not in printed      # an empty answer IS an answer


def test_all_children_are_written_even_when_the_printout_is_capped(
        tmp_path, monkeypatch, capsys):
    kids = [f"2023{n:04d}" for n in range(23)]
    listing = {("s.one",): ["big"], ("s.one", "big"): kids}
    _install_fake_cli(monkeypatch, listing)

    path = rc.recon(["s.one"], [], str(tmp_path), expand=True,
                    map_name="probe.jsonl")

    assert path == str(tmp_path / "probe.jsonl")
    # the console cap is cosmetic: every child reaches the map
    assert [row["dataset"] for row in _rows(path)] == kids
    printed = capsys.readouterr().out
    shown = [kid for kid in kids if f"            {kid}\n" in printed]
    assert shown == kids[:20]
    # ...and the printout says where the rest went, by the real map name
    assert "... and 3 more (all in probe.jsonl)" in printed


# ==========================================================================
# the INCOMPLETE map: what puts a gap in it, and what merely looks like one
# ==========================================================================
def test_unlistable_scope_is_not_an_empty_scope(tmp_path, monkeypatch, capsys):
    listing = {("s.down",): _OUTAGE,      # did not answer
               ("s.empty",): [],          # answered: nothing registered
               ("s.full",): ["d1"]}
    _install_fake_cli(monkeypatch, listing)

    path = rc.recon(["s.down", "s.empty", "s.full"], [], str(tmp_path))

    assert _rows(path) == [{"scope": "s.full", "dataset": "d1"}]
    printed = capsys.readouterr().out
    assert printed.count("NOT LISTED") == 1
    assert "s.down  -- NOT LISTED" in printed
    # only the unanswered scope makes the map incomplete; the empty one is a
    # verdict, and the walk must not confess to a gap it does not have
    warn = printed.rsplit("[warn]", 1)[1]   # the NOT LISTED line cites it too
    assert "the map is INCOMPLETE" in warn
    assert "datasets under scope s.down" in warn
    assert "s.empty" not in warn
    # a scope that contributes no rows is skipped outright, so neither its
    # name nor an empty-count progress line can reach the printout
    assert "s.empty" not in printed
    assert "(0 dataset(s))" not in printed
    assert str(tmp_path / "scopes.jsonl") in warn      # names what to re-run


def test_expand_children_outage_keeps_the_container_and_warns(
        tmp_path, monkeypatch, capsys):
    listing = {("s.one",): ["a", "b"],
               ("s.one", "a"): _OUTAGE,       # container could not be opened
               ("s.one", "b"): ["b1"]}
    _install_fake_cli(monkeypatch, listing)

    path = rc.recon(["s.one"], [], str(tmp_path), expand=True)

    rows = _rows(path)
    # the unopened container is still a row -- and its row is byte-identical
    # in shape to the childless container of the test above, which is exactly
    # why the unexpanded case has to be reported out of band
    assert rows == [{"scope": "s.one", "dataset": "a"},
                    {"scope": "s.one", "dataset": "b1", "parent": "b"}]
    printed = capsys.readouterr().out
    assert "children NOT listed" in printed
    # the retained container still counts toward the map size the operator
    # is shown; a lost increment would under-report it with no other signal
    assert "2 rows (1 dataset(s) expanded)" in printed
    warn = printed.rsplit("[warn]", 1)[1]
    assert "children of s.one a" in warn
    assert "unexpanded" in warn
    assert "children of s.one b" not in warn


def test_a_clean_walk_prints_no_incomplete_warning(tmp_path, monkeypatch,
                                                   capsys):
    listing = {("s.one",): ["d1"], ("s.one", "d1"): ["c1"]}
    _install_fake_cli(monkeypatch, listing)

    rc.recon(["s.one"], [], str(tmp_path), expand=True)

    printed = capsys.readouterr().out
    assert "[warn]" not in printed and "INCOMPLETE" not in printed


# ==========================================================================
# the namespace walk (no explicit --scope): outages, emptiness, --telescope
# ==========================================================================
def test_namespace_outage_aborts_before_writing_a_map(tmp_path, monkeypatch):
    _install_fake_cli(monkeypatch, {(): _OUTAGE})

    with pytest.raises(SystemExit) as excinfo:
        rc.recon(None, [], str(tmp_path))

    message = str(excinfo.value)
    assert "could not list its scopes" in message
    assert "--scope" in message                 # the actionable way forward
    # a truncated/empty map would read as "the archive has nothing"
    assert not (tmp_path / "scopes.jsonl").exists()


def test_zero_scopes_is_reported_as_a_config_problem(tmp_path, monkeypatch):
    _install_fake_cli(monkeypatch, {(): []})

    with pytest.raises(SystemExit) as excinfo:
        rc.recon(None, [], str(tmp_path))

    message = str(excinfo.value)
    assert "zero scopes" in message and "account/config" in message
    assert "could not list its scopes" not in message   # a distinct verdict
    assert not (tmp_path / "scopes.jsonl").exists()


def test_telescope_narrows_by_exact_first_scope_component(tmp_path,
                                                          monkeypatch, capsys):
    listing = {(): ["chime.event.baseband.raw", "gbo.acquisition.processed",
                    "gbo2.acquisition.processed"],
               ("gbo.acquisition.processed",): ["20230530"]}
    _install_fake_cli(monkeypatch, listing)

    path = rc.recon(None, [], str(tmp_path), telescope="GBO")

    # the first dot-component must EQUAL the telescope, case-insensitively:
    # "gbo2.*" is a different instrument, and a prefix test would take it
    # (then its unstaged datasets would trip the fake)
    assert _rows(path) == [{"scope": "gbo.acquisition.processed",
                            "dataset": "20230530"}]
    printed = capsys.readouterr().out
    assert "telescope=GBO (1/3 scope(s)" in printed
    assert "omit --telescope to walk all" in printed


def test_telescope_matching_nothing_aborts_naming_what_was_visible(
        tmp_path, monkeypatch):
    _install_fake_cli(monkeypatch, {(): ["chime.event.baseband.raw", "gbo.x"]})

    with pytest.raises(SystemExit) as excinfo:
        rc.recon(None, [], str(tmp_path), telescope="kko")

    message = str(excinfo.value)
    assert "'kko'" in message and "2 scope(s) visible" in message


# ==========================================================================
# the shipped entry point: options -> recon, at sources/cadc.py's dispatch
# ==========================================================================
def test_scopes_only_dispatch_maps_survey_options_onto_recon(
        tmp_path, monkeypatch, capsys):
    listing = {(): ["gbo.acquisition.processed", "chime.event.baseband.raw"],
               ("gbo.acquisition.processed",): ["complex_gains",
                                                "housekeeping"],
               ("gbo.acquisition.processed", "complex_gains"): ["20230530"]}
    _install_fake_cli(monkeypatch, listing)
    ctx = RunContext(instrument=_Inst("gbo"),
                     options={"scopes_only": True, "match": "GAINS",
                              "expand": True, "name": "probe"})

    path = src.CadcDatatrailSource().survey(ctx, str(tmp_path))

    # --name renames the map; --match is comma-split and case-folded;
    # --expand opens the match; the instrument name becomes --telescope, so
    # the chime scope is never walked (its listing is unstaged)
    assert path == str(tmp_path / "scopes-probe.jsonl")
    assert _rows(path) == [{"scope": "gbo.acquisition.processed",
                            "dataset": "20230530",
                            "parent": "complex_gains"}]
    printed = capsys.readouterr().out
    assert "match=['gains']" in printed
    assert "telescope=gbo (1/2 scope(s)" in printed


def test_scopes_only_dispatch_splits_the_scope_option(tmp_path, monkeypatch):
    listing = {("s.one",): ["d1"], ("s.two",): ["d2"]}
    _install_fake_cli(monkeypatch, listing)
    # instrument name deliberately matches NEITHER scope: explicit scopes win
    # outright, telescope narrowing does not apply to them
    ctx = RunContext(instrument=_Inst("chime"),
                     options={"scopes_only": True, "scope": "s.one, s.two"})

    path = src.CadcDatatrailSource().survey(ctx, str(tmp_path))

    assert path == str(tmp_path / "scopes.jsonl")     # no --name -> default
    assert _rows(path) == [{"scope": "s.one", "dataset": "d1"},
                           {"scope": "s.two", "dataset": "d2"}]


# ==========================================================================
# durability: the map is streamed in place, deliberately un-staged
# ==========================================================================
def test_the_map_is_rewritten_in_place_not_atomically(tmp_path, monkeypatch):
    """An interrupted recon leaves a PARTIAL map at the final path.

    recon opens the map and streams rows into it as the walk proceeds -- no
    tempfile+replace staging -- so the previous map is gone the moment the
    walk starts. That is the reason the INCOMPLETE warning tells you to re-run
    the whole (cheap) walk instead of trusting what is on disk, and a reader
    of a recon map must not assume it was published atomically.
    """
    stale = tmp_path / "scopes.jsonl"
    stale.write_text('{"scope": "from.a.previous.run", "dataset": "old"}\n')

    _install_fake_cli(monkeypatch, {("s.one",): ["d1"], ("s.two",): ["d2"]})
    answering_run = dt.subprocess.run

    def interrupt_on_the_second_scope(cmd, **kw):
        if tuple(cmd[4:-1]) == ("s.two",):
            raise KeyboardInterrupt("operator ^C mid-walk")
        return answering_run(cmd, **kw)

    monkeypatch.setattr(dt.subprocess, "run", interrupt_on_the_second_scope)

    with pytest.raises(KeyboardInterrupt):
        rc.recon(["s.one", "s.two"], [], str(tmp_path))

    # the stale row is already gone (truncated at open), the rows written
    # before the interrupt are at the FINAL path, and nothing was staged
    # beside it
    assert _rows(stale) == [{"scope": "s.one", "dataset": "d1"}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scopes.jsonl"]


# ==========================================================================
# The CLI surface. recon() shipped for a long time with no way to invoke it
# from the command line, so these pin the flags through to the options the
# source actually reads.
# ==========================================================================
def test_scopes_only_flags_reach_the_source(monkeypatch):
    from pilot_proxy import cli
    from pilot_proxy.archive import commands

    seen = {}

    def fake_survey(**kwargs):
        seen.update(kwargs)
        return "/tmp/scopes.jsonl"

    monkeypatch.setattr(commands, "survey_chime", fake_survey)
    args = cli.build_parser().parse_args([
        "chime-survey", "--scopes-only", "--match", "GAINS,corr", "--expand",
        "--name", "probe",
    ])
    args.func(args)
    assert seen["scopes_only"] is True
    assert seen["match"] == "GAINS,corr"
    assert seen["expand"] is True
    assert seen["name"] == "probe"


def test_the_recon_flags_default_off(monkeypatch):
    from pilot_proxy import cli
    from pilot_proxy.archive import commands

    seen = {}
    monkeypatch.setattr(commands, "survey_chime",
                        lambda **kw: seen.update(kw) or "/tmp/x")
    args = cli.build_parser().parse_args(["chime-survey", "--name", "plain"])
    args.func(args)
    assert seen["scopes_only"] is False
    assert seen["match"] is None
    assert seen["expand"] is False


def test_match_and_expand_are_reported_as_ignored_without_scopes_only(
        tmp_path, capsys):
    # they only mean something during discovery; saying so beats dropping them
    from pilot_proxy.archive.commands import survey_chime

    survey_chime(out_dir=tmp_path / "inv", match="gains", expand=True,
                 dry_run=True)
    out = capsys.readouterr().out
    assert "--match applies to --scopes-only; ignoring" in out
    assert "--expand applies to --scopes-only; ignoring" in out

    survey_chime(out_dir=tmp_path / "inv", scopes_only=True, match="gains",
                 dry_run=True)
    assert "ignoring" not in capsys.readouterr().out
