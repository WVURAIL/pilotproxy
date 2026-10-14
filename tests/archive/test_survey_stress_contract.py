#!/usr/bin/env python3
"""
Failure-injection stress tests for the datatrail contract boundary -- the part
of a survey that otherwise only exercises against the live CANFAR/CADC service.

A healthy `chime-survey` run reaches exactly ONE of the three verdicts this
source defines (`progress`), so a single happy-path run is no evidence at all
about the other two. Measured with a stdlib line tracer over a full clean
survey_chime() call (2026-08-28): `service_down`, `refused` and `no_data` were
never executed, nor were any of the eight DatatrailContractError raise sites in
datatrail_client.files(), nor any of the five not-answered branches in
_list_result_checked(). This file drives every one of them.

Everything here is offline. The ONLY seams faked are:
  * `datatrail_client.subprocess.run` -- the adapter's single child-process
    boundary, so files(), _restore_collection(), the collection/span checks and
    the common-path split all run for real (and the fake asserts the invocation
    contract itself: same-interpreter `-m dtcli.cli`, trailing `--json`);
  * `CadcDatatrailSource._cadc_size` -- the single CADC boundary. It returns
    (size, None) present, (None, None) DEFINITIVELY ABSENT (cadc.py:473-474
    reports NotFound as an answer, not an error) and (None, exc) hard error;
  * `cadc._enumerate_events` for the tests that are not about enumeration.
    The `ls`-drift tests deliberately leave it real and drive the walk through
    the fake CLI.
  * the clocks, for the outage-circuit tests -- ONE fake clock shared by
    cadc._monotonic, cadc._sleep and datatrail_client.time, because the outage
    deadline is compared inside BOTH modules. Patching only one of them makes
    _run_json return at datatrail_client.py:110-111 before spawning a child, so
    the test silently measures nothing (measured: 0 child calls).

Run:
  TMPDIR=/tmp PYTHONPATH=src python -m pytest tests/archive/test_survey_stress_contract.py
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import sqlite3
import subprocess
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.datatrail_client import DatatrailContractError
from pilot_proxy.archive.interfaces import RunContext, SurveyUnavailableError
from pilot_proxy.archive.sources import cadc as cadc_src
from pilot_proxy.chime.baseband_reader import (MINIMUM_ARCHIVE_BYTES,
                                               baseband_filename)


SCOPE = "chime.event.baseband.raw"
EVENT = "100200200"
FREQ_IDS = [0, 1]
BIG = MINIMUM_ARCHIVE_BYTES + 4096


# ==========================================================================
# the fake datatrail CLI -- same shape as tests/archive/test_datatrail_adapter.py
# so the two files cannot drift: handler(args) -> (rc, stdout, stderr), where
# args excludes the [sys.executable, -m, dtcli.cli] prefix and the trailing
# --json, both of which the fake asserts.
# ==========================================================================
class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


class _MisWiredFake(BaseException):
    """A fake-CLI contract violation, raised as a BaseException ON PURPOSE.

    _run_json catches `Exception` and reports it as "the CLI did not answer"
    (datatrail_client.py:116-117), which survey escalates to a one-hour outage
    circuit. An AssertionError from a mis-wired fake would therefore be
    swallowed and the test would hang for 3600 fake-free seconds instead of
    failing. Deriving from BaseException makes a harness mistake loud.
    """


def _install_fake_cli(monkeypatch, handler=None, version=(0, 11, 0)):
    """datatrail "installed" at `version`; CLI calls answered by `handler`.

    Returns the list of arg-vectors the adapter actually spawned, so a test can
    assert how many children a code path costs (an outage circuit that spins
    is a service-abuse bug, not just a slow test). The invocation contract
    itself is checked here -- same-interpreter `-m dtcli.cli`, trailing
    `--json` -- so a drifting invocation fails loudly.
    """
    calls: list = []
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: version)

    def fake_run(cmd, **kw):
        if handler is None:
            raise _MisWiredFake(f"unexpected datatrail call: {cmd}")
        if cmd[:3] != [sys.executable, "-m", "dtcli.cli"] or cmd[-1] != "--json":
            raise _MisWiredFake(f"invocation contract broken: {cmd}")
        args = list(cmd[3:-1])
        calls.append(args)
        rc, out, err = handler(args)
        return _Proc(rc, out, err)

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    return calls


# dtcli's group callback prints this to STDOUT ahead of any command output when
# PyPI shows a newer release -- _extract_json must skip it.
_BANNER = "A new release of datatrail-cli is available: 0.11.0 -> 0.12.0\n\n"


def _ps_payload(files):
    return json.dumps(
        {"dataset": "d", "scope": "s", "files": files, "policies": {"p": 1}})


def _minoc(uris):
    """The `ps --json` files half for a dataset with these minoc replicas."""
    return {"file_replica_locations": {"minoc": list(uris)}}


def _ls_handler(*, scopes=None, larger_datasets=None, datasets=None,
                ps=None):
    """Route `ls` by arity exactly as _list_result_checked does.

    len(args)==1 -> {"scopes"}; ==2 -> {"larger_datasets"}; ==3 -> {"datasets"}.
    Any value may be a callable(args) returning (rc, stdout, stderr) so a test
    can fail one dataset part-way through a walk.

    `ps` answers the file-listing arity; it defaults to the no-data shape so an
    enumeration test that happens to find an event resolves it immediately
    instead of falling into the outage circuit.
    """
    def handler(args):
        if not args:
            raise _MisWiredFake("empty datatrail argv")
        if args[0] == "ps":
            if ps is None:
                return 0, _ps_payload(None), ""
            return ps(args) if callable(ps) else (0, _ps_payload(ps), "")
        if args[0] != "ls":
            raise _MisWiredFake(f"unexpected datatrail subcommand: {args}")
        if len(args) == 1:
            key, value = "scopes", scopes
        elif len(args) == 2:
            key, value = "larger_datasets", larger_datasets
        else:
            key, value = "datasets", datasets
        if callable(value):
            return value(args)
        if value is None:
            return 1, json.dumps({"error": f"no {key} configured for {args}"}), ""
        return 0, json.dumps({key: list(value)}), ""
    return handler


# ==========================================================================
# clocks. datatrail_client.files() compares the outage deadline against
# `time.monotonic()` in ITS OWN module (datatrail_client.py:109, :355), while
# survey compares it against cadc._monotonic. One clock must therefore drive
# both, or the test measures a code path that never spawns a child.
# ==========================================================================
class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.t += float(seconds)


def _install_clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(cadc_src, "_monotonic", clock.monotonic)
    monkeypatch.setattr(cadc_src, "_sleep", clock.sleep)
    monkeypatch.setattr(dt, "time", types.SimpleNamespace(
        monotonic=clock.monotonic, sleep=clock.sleep))
    return clock


def _no_backoff(monkeypatch) -> list:
    """Real monotonic, recorded no-op sleep inside the adapter.

    Without this, every not-answered files() call sleeps 4+8+16 = 28 REAL
    seconds (measured), which is the difference between a 6-second suite and a
    stalled one.
    """
    slept: list = []
    monkeypatch.setattr(dt, "time", types.SimpleNamespace(
        monotonic=dt.time.monotonic, sleep=slept.append))
    return slept


# ==========================================================================
# the survey harness. Three seams, nothing else: the CLI child, the CADC
# probe, and (where the test is not about enumeration) the event walk.
# ==========================================================================
def _day_dir(days_ago: int) -> str:
    """An archive day directory `days_ago` days before today, UTC.

    _obs_age_days is UTC (cadc.py:127-136); datetime.date.today() is LOCAL and
    would skew the 30-day boundary by a day in some timezones.
    """
    day = (datetime.datetime.now(datetime.timezone.utc).date()
           - datetime.timedelta(days=days_ago))
    return f"data/chime/baseband/raw/{day:%Y/%m/%d}"


def _event_dir(event=EVENT, days_ago=400) -> str:
    return f"{_day_dir(days_ago)}/astro_{event}"


def _fake_enumeration(monkeypatch, events):
    monkeypatch.setattr(cadc_src, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, ev): ["ds"] for ev in events})


def _fake_archive(monkeypatch, present=(), *, size=BIG, error_for=()):
    """cadcinfo answers. Returns the list of URIs actually probed."""
    probed: list = []
    present, error_for = set(present), set(error_for)

    def fake_size(self, uri, *a, **k):
        probed.append(uri)
        if uri in error_for:
            return None, OSError("archive probe failed")
        if uri in present:
            return size, None
        return None, None                 # definitive absence, NOT an error
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size", fake_size)
    return probed


def _survey(out_dir, **options):
    """Run a real survey() and return its stdout."""
    opts = {"scope": SCOPE, "freq_ids": list(FREQ_IDS)}
    opts.update(options)
    ctx = RunContext(instrument=None, selection=None, options=opts)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cadc_src.CadcDatatrailSource().survey(ctx, str(out_dir))
    return buf.getvalue()


def _lines(path):
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _state(out):
    """Everything a survey leaves behind, read from the files AND the DB.

    The JSONL/text files are only VIEWS of survey_state.sqlite3
    (survey_state.py:385-408), so a test that reads only the views cannot tell
    a missing commit from a missing render.
    """
    rows = [json.loads(ln) for ln in _lines(out / "inventory.jsonl")]
    ledger = [json.loads(ln) for ln in _lines(out / "no_files_events.jsonl")]
    statuses = {}
    db = out / "survey_state.sqlite3"
    if db.exists():
        con = sqlite3.connect(db)
        try:
            statuses = {k: s for k, s in
                        con.execute("SELECT event_key, status FROM events")}
        finally:
            con.close()
    attempts = {}
    if (out / "attempts.json").exists():
        attempts = json.loads((out / "attempts.json").read_text() or "{}")
    return types.SimpleNamespace(
        rows=rows, ledger=ledger, statuses=statuses, attempts=attempts,
        surveyed=_lines(out / "surveyed_events.txt"),
        incomplete=_lines(out / "incomplete_events.txt"),
    )


# ##########################################################################
# TIER C -- `ps --json` schema drift. Every deterministic refusal site in
# datatrail_client.files(), plus the four shapes that are a no-data ANSWER.
# retries=0 keeps these sub-millisecond and proves no retry is attempted for a
# deterministic answer.
# ##########################################################################
_B = f"cadc:CHIMEFRB/{_event_dir(days_ago=2000)}"

# (id, files-half, expectation). Expectation is one of
#   ("refuse", <substring of the message>) | ("nodata",) | ("ok", cp, names)
_PS_SHAPES = [
    # -- deterministic refusals -------------------------------------------
    ("files-non-dict-str", "Server hiccup", ("refuse", "non-dict 'files' value (str)")),
    ("files-non-dict-list", [], ("refuse", "non-dict 'files' value (list)")),
    ("locations-list", {"file_replica_locations": []},
     ("refuse", "non-object 'file_replica_locations' value")),
    ("locations-str", {"file_replica_locations": "x"},
     ("refuse", "non-object 'file_replica_locations' value")),
    ("minoc-str", {"file_replica_locations": {"minoc": "a,b"}},
     ("refuse", "non-list 'minoc' replica locations")),
    ("minoc-dict", {"file_replica_locations": {"minoc": {"a": 1}}},
     ("refuse", "non-list 'minoc' replica locations")),
    ("minoc-blank", _minoc([f"{_B}/a.h5", "   "]),
     ("refuse", "malformed 'minoc' replica locations")),
    ("minoc-empty-str", _minoc([f"{_B}/a.h5", ""]),
     ("refuse", "malformed 'minoc' replica locations")),
    ("minoc-int", _minoc([f"{_B}/a.h5", 17]),
     ("refuse", "malformed 'minoc' replica locations")),
    ("minoc-null-entry", _minoc([f"{_B}/a.h5", None]),
     ("refuse", "malformed 'minoc' replica locations")),
    ("foreign-collection", _minoc(["cadc:SOMEONE_ELSE/data/x/y.h5"]),
     ("refuse", "outside the expected collection(s) ['cadc:CHIMEFRB/'] "
                "(1/1 replicas affected)")),
    ("collection-root-only", _minoc(["cadc:CHIMEFRB/"]),
     ("refuse", "not a canonical path")),
    ("bare-root-only", _minoc(["data/"]),
     ("refuse", "not a canonical path")),
    ("directory-only", _minoc([f"{_B}/"]),
     ("refuse", "not a canonical path")),
    # -- no-data ANSWERS: queried fine, this dataset has no minoc files ----
    ("files-null", None, ("nodata",)),
    ("files-empty-dict", {}, ("nodata",)),
    ("locations-null", {"file_replica_locations": None}, ("nodata",)),
    ("no-minoc-key", {"file_replica_locations": {"arc": ["/arc/p/x.h5"]}},
     ("nodata",)),
    ("minoc-null", {"file_replica_locations": {"minoc": None}}, ("nodata",)),
    ("minoc-empty-list", _minoc([]), ("nodata",)),
    # -- positive shapes ---------------------------------------------------
    ("single-replica", _minoc([f"{_B}/b_0.h5"]), ("ok", _B, ["b_0.h5"])),
    ("two-replicas", _minoc([f"{_B}/b_0.h5", f"{_B}/b_1.h5"]),
     ("ok", _B, ["b_0.h5", "b_1.h5"])),
    # multi-segment names are LEGITIMATE: the intensity scope splits one event
    # across per-beam subdirectories, so a "no slashes in names" guard would be
    # wrong. Pinned here so nobody adds one.
    ("beam-subdirectories", _minoc([f"{_B}/2068/a.msgpack", f"{_B}/2069/b.msgpack"]),
     ("ok", _B, ["2068/a.msgpack", "2069/b.msgpack"])),
]


@pytest.mark.parametrize("case_id,files_half,expect",
                         _PS_SHAPES, ids=[c[0] for c in _PS_SHAPES])
def test_ps_payload_shape_taxonomy(monkeypatch, case_id, files_half, expect):
    _install_fake_cli(monkeypatch,
                      lambda args: (0, _BANNER + _ps_payload(files_half), ""))
    adapter = dt.Datatrail()
    if expect[0] == "refuse":
        with pytest.raises(DatatrailContractError) as excinfo:
            adapter.files(SCOPE, EVENT, retries=0)
        assert expect[1] in str(excinfo.value), str(excinfo.value)
    elif expect[0] == "nodata":
        # a no-data ANSWER: ok=True. Never conflated with an outage, because
        # survey commits ok=False as `service_down` and rides the circuit.
        assert adapter.files(SCOPE, EVENT, retries=0) == (None, [], True)
    else:
        cp, names, ok = adapter.files(SCOPE, EVENT, retries=0)
        assert (cp, names, ok) == (expect[1], expect[2], True)


def test_a_payload_without_a_files_key_is_a_refusal_not_no_data(monkeypatch):
    # 0.11's success payload always carries "files" (it may be null). A
    # non-error dict WITHOUT it is a newer CLI, and must never read as "this
    # dataset has no files" -- that would silently drop the event's rows.
    _install_fake_cli(monkeypatch, lambda args: (
        0, json.dumps({"dataset": "d", "scope": "s", "policies": {}}), ""))
    with pytest.raises(DatatrailContractError) as excinfo:
        dt.Datatrail().files(SCOPE, EVENT, retries=0)
    assert "no 'files' key" in str(excinfo.value)
    assert "newer than this adapter understands" in str(excinfo.value)


def test_a_deterministic_refusal_spawns_exactly_one_child(monkeypatch):
    # Retrying cannot change a deterministic answer. If a refusal ever started
    # being retried it would cost 4x the calls against a shared production
    # service for every affected event.
    calls = _install_fake_cli(
        monkeypatch,
        lambda args: (0, _ps_payload(_minoc(["cadc:SOMEONE_ELSE/x/y.h5"])), ""))
    _no_backoff(monkeypatch)
    with pytest.raises(DatatrailContractError):
        dt.Datatrail().files(SCOPE, EVENT)          # default retries=3
    assert len(calls) == 1, calls


def test_ambiguous_bare_and_mixed_replies_are_refused(monkeypatch):
    # A bare reply cannot identify a collection when two are configured.
    monkeypatch.setattr(dt, "_MINOC_COLLECTIONS",
                        ("cadc:CHIMEFRB/", "cadc:CHIMEOUTRIGGER/"))
    bare = ["data/kko/baseband/raw/2025/01/01/a/b_0.h5",
            "data/kko/baseband/raw/2025/01/01/a/b_1.h5"]
    _install_fake_cli(monkeypatch,
                      lambda args: (0, _ps_payload(_minoc(bare)), ""))
    with pytest.raises(DatatrailContractError, match="outside the expected collection"):
        dt.Datatrail().files(SCOPE, EVENT, retries=0)

    # A prefixed member cannot establish the identity of a bare member.
    mixed = ["cadc:CHIMEOUTRIGGER/data/kko/x/b_0.h5", "data/kko/x/b_1.h5"]
    _install_fake_cli(monkeypatch,
                      lambda args: (0, _ps_payload(_minoc(mixed)), ""))
    with pytest.raises(DatatrailContractError) as excinfo:
        dt.Datatrail().files(SCOPE, EVENT, retries=0)
    assert "outside the expected collection" in str(excinfo.value)


def test_restoration_refuses_to_guess_between_two_collections(monkeypatch):
    monkeypatch.setattr(dt, "_MINOC_COLLECTIONS",
                        ("cadc:CHIMEFRB/", "cadc:CHIMEOUTRIGGER/"))
    _install_fake_cli(monkeypatch, lambda args: (
        0, _ps_payload(_minoc(["data/kko/x/b_0.h5", "data/kko/x/b_1.h5"])), ""))
    with pytest.raises(DatatrailContractError):
        dt.Datatrail().files(SCOPE, EVENT, retries=0)


def test_restoration_stays_narrow(monkeypatch):
    # The patch's blast radius is _BARE_REPLICA_ROOTS. Every spelling that is
    # NOT exactly "data/..." must still be refused, or a drifting replica list
    # gets a fabricated common_path instead of a loud refusal.
    for uri in ("/data/chime/x/f.h5",            # site-SE spelling
                "//data/chime/x/f.h5",           # collapses to the above
                "./data/chime/x/f.h5",
                "DATA/chime/x/f.h5",
                "chime/baseband/x/f.h5",
                "junk/f.h5",
                "cadc:SOMEONE_ELSE/data/x/f.h5"):
        _install_fake_cli(monkeypatch,
                          lambda args, u=uri: (0, _ps_payload(_minoc([u])), ""))
        with pytest.raises(DatatrailContractError) as excinfo:
            dt.Datatrail().files(SCOPE, EVENT, retries=0)
        assert "outside the expected collection(s)" in str(excinfo.value), uri


def test_restoration_is_idempotent_and_does_not_double_prefix(monkeypatch):
    # A restored URI must satisfy the collection guard on a second pass, or a
    # future re-normalization would stack prefixes.
    once = dt._restore_collection("data/chime/baseband/raw/2020/07/15/a/b.h5")
    assert once == "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/a/b.h5"
    assert dt._restore_collection(once) == once
    assert dt._restore_collection("") == ""        # blank never reaches here


def test_a_triple_slash_survives_the_collapse_but_not_the_split(monkeypatch):
    # str.replace does ONE non-overlapping pass: "a///b" -> "a//b". The
    # residual slash is swallowed by lstrip("/") in the split, so the derived
    # path is unharmed -- but the collapse is not a normalizer, and nothing
    # else canonicalizes. Pinned so a "cleanup" that removes lstrip() is caught.
    assert "cadc:CHIMEFRB///d/x".replace("//", "/") == "cadc:CHIMEFRB//d/x"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc([
        f"cadc:CHIMEFRB//{_event_dir(days_ago=2000)}/b_0.h5",
        f"cadc:CHIMEFRB//{_event_dir(days_ago=2000)}/b_1.h5"])), ""))
    cp, names, ok = dt.Datatrail().files(SCOPE, EVENT, retries=0)
    assert ok and cp == _B and names == ["b_0.h5", "b_1.h5"]


def test_duplicate_replicas_are_not_deduplicated(monkeypatch):
    # CHARACTERIZATION. Nothing between datatrail_client.py:424 and :434
    # deduplicates, so a repeated replica yields a repeated name. Harmless for
    # common_path() (which discards names) but wrong for the documented
    # companion-resolution use of files().
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc([
        f"{_B}/b_0.h5", f"{_B}/b_0.h5", f"{_B}/b_1.h5"])), ""))
    cp, names, ok = dt.Datatrail().files(SCOPE, EVENT, retries=0)
    assert names == ["b_0.h5", "b_0.h5", "b_1.h5"]
    assert dt.Datatrail().common_path(SCOPE, EVENT, retries=0) == (cp, True)


# ##########################################################################
# TIER D -- `ls --json` drift. These drive the REAL _enumerate_events walk.
# ##########################################################################
_LS_SHAPES = [
    ("scopes-str", 0, {"scopes": "Host not in allowlist"},
     "non-list 'scopes' value (str)"),
    ("scopes-dict", 0, {"scopes": {"a": 1}}, "non-list 'scopes' value (dict)"),
    ("scopes-null-entry", 0, {"scopes": ["a", None]},
     "contains non-string or empty entries"),
    ("scopes-int-entry", 0, {"scopes": ["a", 7]},
     "contains non-string or empty entries"),
    ("scopes-blank-entry", 0, {"scopes": ["a", "   "]},
     "contains non-string or empty entries"),
    ("error-envelope", 1, {"error": "Server not responding."},
     "Server not responding."),
    # a FALSY error is reported as a schema problem for what is really a
    # service problem -- worth knowing, and pinned.
    ("empty-error", 1, {"error": ""}, "no 'scopes' key"),
    ("null-error", 1, {"error": None}, "no 'scopes' key"),
    ("renamed-key", 0, {"renamed_in_0_12": ["a"]},
     "newer than this adapter understands"),
]


@pytest.mark.parametrize("case_id,rc,payload,expect_stderr",
                         _LS_SHAPES, ids=[c[0] for c in _LS_SHAPES])
def test_ls_payload_shape_taxonomy(monkeypatch, capsys, case_id, rc, payload,
                                   expect_stderr):
    _install_fake_cli(monkeypatch, lambda args: (rc, json.dumps(payload), ""))
    scopes, ok = dt.Datatrail().list_scopes_checked()
    # ok=False is "couldn't determine", never "no scopes" -- a discovery walk
    # that treated these as emptiness would write an empty inventory and exit 0.
    assert (scopes, ok) == ([], False)
    assert expect_stderr in capsys.readouterr().err


def test_an_empty_scope_list_is_a_real_answer(monkeypatch):
    _install_fake_cli(monkeypatch, lambda args: (0, json.dumps({"scopes": []}), ""))
    assert dt.Datatrail().list_scopes_checked() == ([], True)


@pytest.mark.parametrize("stdout", ["", "Killed\n", "  scope  | datasets \n"])
def test_non_json_stdout_is_not_answered(monkeypatch, capsys, stdout):
    _install_fake_cli(monkeypatch, lambda args: (2, stdout, "boom"))
    assert dt.Datatrail().list_scopes_checked() == ([], False)
    assert "no JSON on stdout" in capsys.readouterr().err


def test_a_top_level_json_array_is_not_answered(monkeypatch):
    # _extract_json returns None for a non-dict payload (datatrail_client.py:93).
    _install_fake_cli(monkeypatch, lambda args: (0, "[1, 2]", ""))
    assert dt.Datatrail().list_scopes_checked() == ([], False)


def test_enumeration_aborts_when_a_scope_cannot_be_listed(monkeypatch, tmp_path):
    # SurveyUnavailableError, and -- the load-bearing half -- NO enum_cache.json
    # is written. A half-walked map cached as complete would silently shrink
    # every subsequent run of that directory.
    _install_fake_cli(monkeypatch, _ls_handler())          # every ls errors
    _no_backoff(monkeypatch)
    _fake_archive(monkeypatch)
    with pytest.raises(SurveyUnavailableError) as excinfo:
        _survey(tmp_path)
    assert "could not list datasets under scope" in str(excinfo.value)
    assert not (tmp_path / "enum_cache.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".survey.lock", ".survey.views-stale", "survey_manifest.json", "survey_runs"]
    assert json.loads((tmp_path / "survey_manifest.json").read_text())["latest_run"]["status"] == "failed"


def test_enumeration_aborts_mid_walk_without_a_partial_cache(monkeypatch,
                                                             tmp_path):
    def datasets(args):
        if args[-1] == "2020":
            return 0, json.dumps({"datasets": [f"astro-{EVENT}"]}), ""
        return 1, json.dumps({"error": "backend timeout"}), ""

    calls = _install_fake_cli(monkeypatch, _ls_handler(
        larger_datasets=["2020", "2021"], datasets=datasets))
    _no_backoff(monkeypatch)
    _fake_archive(monkeypatch)
    with pytest.raises(SurveyUnavailableError) as excinfo:
        _survey(tmp_path)
    assert "could not list children of" in str(excinfo.value)
    assert "'2021'" in str(excinfo.value)
    # the walk stopped at the failing dataset; the events it DID find are
    # discarded rather than cached as the whole map.
    assert [a[1:] for a in calls] == [
        [SCOPE], [SCOPE, "2020"], [SCOPE, "2021"]]
    assert not (tmp_path / "enum_cache.json").exists()


# ##########################################################################
# TIER A -- silent wrong data: no exception, no warning, wrong inventory.
# ##########################################################################
@pytest.mark.parametrize("child,expected_events", [
    ("100200200", 1),
    ("astro-100200200", 1),
    ("chime.event.baseband.raw.100200200", 1),
    # \b never matches next to an underscore (it is a word character), so the
    # underscore-separated names datatrail actually uses for astro datasets
    # yield NOTHING. If datatrail renames child datasets to this form, the
    # survey enumerates 0 events, writes an empty inventory, and EXITS 0.
    ("astro_100200200", 0),
    ("baseband_100200200", 0),
    ("2020-07-15_100200200", 0),
])
def test_child_dataset_name_grammar(monkeypatch, tmp_path, child,
                                    expected_events):
    _install_fake_cli(monkeypatch, _ls_handler(
        larger_datasets=["2020"], datasets=[child]))
    _no_backoff(monkeypatch)
    _fake_archive(monkeypatch)
    assert dt._EVENT_RE.findall(child) == ([] if not expected_events
                                           else ["100200200"])
    text = _survey(tmp_path / child.replace("/", "_"))
    assert f"enumerated {expected_events} unique events" in text


def test_a_grammar_drift_reports_success_with_an_empty_inventory(monkeypatch,
                                                                 tmp_path):
    # The whole failure, end to end: the ONLY signal is the generic empty
    # inventory warning -- identical to the signal for a legitimately empty
    # archive -- and --strict-completeness still PASSES, because there is
    # nothing pending, incomplete or refused.
    _install_fake_cli(monkeypatch, _ls_handler(
        larger_datasets=["2020"], datasets=[f"astro_{EVENT}"]))
    _no_backoff(monkeypatch)
    probed = _fake_archive(monkeypatch)
    source = cadc_src.CadcDatatrailSource()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        source.survey(RunContext(instrument=None, options={
            "scope": SCOPE, "freq_ids": list(FREQ_IDS)}), str(tmp_path))
    text = buf.getvalue()
    assert "to survey: 0 events" in text
    assert "0 rows written" in text
    assert "[warn] inventory.jsonl is EMPTY" in text
    assert probed == []
    assert source.survey_completeness_issues(str(tmp_path)) == {
        "incomplete": 0, "refused": 0, "pending": 0}
    assert json.loads((tmp_path / "enum_cache.json").read_text())["events"] == {}


def test_an_empty_enumeration_is_cached_and_silently_reused(monkeypatch,
                                                            tmp_path):
    # Pass 1 hits the grammar drift and caches an EMPTY map. Pass 2 runs after
    # the service is fixed and still reports 0 events, making ZERO ls calls --
    # the cache is only invalidated by --re-enumerate.
    _install_fake_cli(monkeypatch, _ls_handler(
        larger_datasets=["2020"], datasets=[f"astro_{EVENT}"]))
    _no_backoff(monkeypatch)
    _fake_archive(monkeypatch)
    assert "enumerated 0 unique events" in _survey(tmp_path)

    calls = _install_fake_cli(monkeypatch, _ls_handler(
        larger_datasets=["2020"], datasets=[EVENT]))
    assert "to survey: 0 events" in _survey(tmp_path)
    assert calls == []                       # cache hit: no listing at all
    assert "enumerated 1 unique events" in _survey(tmp_path, re_enumerate=True)


def test_a_valid_but_empty_listing_is_not_an_outage(monkeypatch, tmp_path):
    # key present with [] is ok=True: a genuine "nothing registered".
    _install_fake_cli(monkeypatch, _ls_handler(larger_datasets=[]))
    _no_backoff(monkeypatch)
    _fake_archive(monkeypatch)
    text = _survey(tmp_path)
    assert "enumerated 0 unique events" in text
    assert "[warn] inventory.jsonl is EMPTY" in text


@pytest.mark.parametrize("case,size", [
    ("absent", None),
    ("sub-floor", MINIMUM_ARCHIVE_BYTES - 1),
    ("zero-byte", 0),
])
def test_absent_and_present_but_unusable_bytes_have_explicit_reasons(
        monkeypatch, tmp_path, case, size):
    # Byte-presence and reader eligibility are separate observations.
    cp = f"cadc:CHIMEFRB/{_event_dir(days_ago=400)}"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])

    def fake_size(self, uri, *a, **k):
        return size, None
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size", fake_size)

    _survey(tmp_path)
    state = _state(tmp_path)
    obs = str(datetime.datetime.now(datetime.timezone.utc).date()
              - datetime.timedelta(days=400))
    assert state.rows == []
    assert state.statuses == {f"{SCOPE}|{EVENT}": "empty"}
    (row,) = state.ledger
    row.pop("ts")
    assert row == {
        "scope": SCOPE, "event": EVENT, "n_expected": 2, "attempts": 1,
        "obs_date": obs, "age_days": 400, "common_path": cp,
        "reason": "aged-out" if size is None else "below-reader-floor",
        "collection_restored": False, "minimum_archive_bytes": MINIMUM_ARCHIVE_BYTES,
        "n_absent": 2 if size is None else 0,
        "n_sub_floor": 0 if size is None else 2,
        "sub_floor_files": [] if size is None else [
            {"name": baseband_filename(EVENT, f), "size_bytes": size} for f in FREQ_IDS],
    }


def test_a_sub_floor_file_is_ledgered_differently_from_an_absent_one(
        monkeypatch, tmp_path):
    cp = f"cadc:CHIMEFRB/{_event_dir(days_ago=400)}"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])

    def absent(self, uri, *a, **k):
        return None, None
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size", absent)
    _survey(tmp_path / "absent")
    absent_row = _state(tmp_path / "absent").ledger[0]

    def small(self, uri, *a, **k):
        return MINIMUM_ARCHIVE_BYTES - 1, None
    monkeypatch.setattr(cadc_src.CadcDatatrailSource, "_cadc_size", small)
    _survey(tmp_path / "small")
    small_row = _state(tmp_path / "small").ledger[0]

    absent_row.pop("ts"), small_row.pop("ts")
    assert absent_row != small_row


@pytest.mark.parametrize("days_ago,written_off", [
    (28, False), (29, False), (30, True), (31, True)])
def test_the_empty_age_gate_flips_exactly_at_the_configured_day(
        monkeypatch, tmp_path, days_ago, written_off):
    # aged_out = age_days >= empty_age_days (cadc.py:771-772), default 30.
    # Asserting the constant here too, so moving it forces this table to be
    # re-derived rather than silently changing every write-off decision.
    assert cadc_src._EMPTY_TERMINAL_AGE_DAYS == 30
    cp = f"cadc:CHIMEFRB/{_event_dir(days_ago=days_ago)}"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)               # everything absent
    _survey(tmp_path)

    state = _state(tmp_path)
    key = f"{SCOPE}|{EVENT}"
    if written_off:
        assert state.statuses == {key: "empty"}
        assert state.ledger[0]["reason"] == "aged-out"
        assert state.ledger[0]["age_days"] == days_ago
        assert state.attempts == {}          # cleared on commit
    else:
        assert state.statuses == {}          # not committed: retried next run
        assert state.ledger == []
        assert state.attempts == {key: 1}


def test_empty_age_days_zero_writes_off_a_fresh_observation(monkeypatch,
                                                            tmp_path):
    cp = f"cadc:CHIMEFRB/{_event_dir(days_ago=3)}"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    _survey(tmp_path, empty_age_days=0)
    state = _state(tmp_path)
    assert state.statuses == {f"{SCOPE}|{EVENT}": "empty"}
    assert state.ledger[0]["reason"] == "aged-out"


def test_an_unreachable_age_gate_still_terminates_at_max_attempts(monkeypatch,
                                                                  tmp_path):
    # --empty-age-days can never make an empty event immortal: _MAX_ATTEMPTS
    # closes it on the third sighting with reason "max-attempts".
    cp = f"cadc:CHIMEFRB/{_event_dir(days_ago=400)}"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    key = f"{SCOPE}|{EVENT}"
    ladder = []
    for _ in range(3):
        _survey(tmp_path, empty_age_days=99999)
        state = _state(tmp_path)
        ladder.append((dict(state.attempts), dict(state.statuses)))
    assert ladder[0] == ({key: 1}, {})
    assert ladder[1] == ({key: 2}, {})
    assert ladder[2] == ({}, {key: "empty"})
    assert _state(tmp_path).ledger[0]["reason"] == "max-attempts"
    assert _state(tmp_path).ledger[0]["attempts"] == 3


def test_an_undatable_common_path_fails_open_into_the_retry_path(monkeypatch,
                                                                 tmp_path):
    # _DATE_RE needs "/raw/YYYY/MM/DD/". A common path without it is
    # obs_date="unknown", age_days=None, and the age gate CANNOT fire -- the
    # event keeps its cross-resume allowance instead of being written off.
    cp = "cadc:CHIMEFRB/data/chime/misc/astro_100200200"
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(
        [f"{cp}/{baseband_filename(EVENT, f)}" for f in FREQ_IDS])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    text = _survey(tmp_path, empty_age_days=0)
    state = _state(tmp_path)
    assert state.statuses == {}
    assert state.attempts == {f"{SCOPE}|{EVENT}": 1}
    assert "re-checking in case transient (1/3)" in text


def test_unverified_restoration_remains_pending_across_retries(
        monkeypatch, tmp_path):
    # Failed resolution of a reconstructed collection remains resumable.
    bare = [f"{_event_dir(days_ago=400)}/{baseband_filename(EVENT, f)}"
            for f in FREQ_IDS]
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(bare)), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)               # the restored URI resolves to nothing

    _survey(tmp_path / "patched")
    patched = _state(tmp_path / "patched")
    assert patched.statuses == {}
    assert patched.ledger == []
    for _ in range(4):
        _survey(tmp_path / "patched")
    assert _state(tmp_path / "patched").statuses == {}
    pending = json.loads((tmp_path / "patched" / "pending_events.json").read_text())
    assert pending[f"{SCOPE}|{EVENT}"]["reason"] == "unverified-restored-collection"

    # the same reply with restoration reverted: a refusal that names the URI.
    monkeypatch.setattr(dt, "_restore_collection", lambda uri: uri)
    _survey(tmp_path / "reverted")
    reverted = _state(tmp_path / "reverted")
    assert reverted.statuses == {f"{SCOPE}|{EVENT}": "refused"}
    assert reverted.ledger[0]["reason"] == "datatrail-contract-refusal"
    assert bare[0] in reverted.ledger[0]["detail"]


def test_a_restored_uri_that_resolves_to_nothing_is_not_terminal(monkeypatch,
                                                                 tmp_path):
    bare = [f"{_event_dir(days_ago=400)}/{baseband_filename(EVENT, f)}"
            for f in FREQ_IDS]
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(_minoc(bare)), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    _survey(tmp_path)
    assert _state(tmp_path).statuses == {}


def test_restored_and_prefixed_rows_share_identity_but_record_origin(
        monkeypatch, tmp_path):
    # URI identity and verified bytes agree while restoration remains auditable.
    day = _day_dir(400)
    event_dir = f"{day}/astro_{EVENT}"
    names = [baseband_filename(EVENT, f) for f in FREQ_IDS]
    present = {f"cadc:CHIMEFRB/{event_dir}/{n}" for n in names}

    def run(sub, minoc):
        _install_fake_cli(monkeypatch,
                          lambda args: (0, _ps_payload(_minoc(minoc)), ""))
        _no_backoff(monkeypatch)
        probed = _fake_archive(monkeypatch, present=present)
        _fake_enumeration(monkeypatch, [EVENT])
        _survey(tmp_path / sub)
        return _state(tmp_path / sub), probed

    prefixed, probed_a = run("prefixed",
                             [f"cadc:CHIMEFRB/{event_dir}/{n}" for n in names])
    bare, probed_b = run("bare", [f"{event_dir}/{n}" for n in names])
    assert prefixed.rows and bare.rows
    assert all(not r["collection_restored"] for r in prefixed.rows)
    assert all(r["collection_restored"] for r in bare.rows)
    assert [{k: v for k, v in r.items() if k != "collection_restored"} for r in prefixed.rows] == [
        {k: v for k, v in r.items() if k != "collection_restored"} for r in bare.rows]
    assert {r["common_path"] for r in bare.rows} == {f"cadc:CHIMEFRB/{event_dir}"}
    assert sorted(probed_a) == sorted(probed_b) == sorted(present)
    assert bare.statuses == {f"{SCOPE}|{EVENT}": "complete"}


# ##########################################################################
# TIER B -- the three-way verdict taxonomy, driven end to end.
# ##########################################################################
_OLD = _event_dir(days_ago=400)
_YOUNG = _event_dir(days_ago=3)


def _present(event_dir):
    return {f"cadc:CHIMEFRB/{event_dir}/{baseband_filename(EVENT, f)}"
            for f in FREQ_IDS}


_VERDICTS = [
    # (id, files-half, present-set, expected status, rows, ledger reason, tag)
    ("complete", _minoc(sorted(_present(_OLD))), _present(_OLD),
     "complete", 2, None, "2 rows written"),
    ("no-data-files-null", None, set(), "no-data", 0, None, "1 no-data"),
    ("no-data-empty-minoc", _minoc([]), set(), "no-data", 0, None, "1 no-data"),
    ("refused-foreign", _minoc(["cadc:SOMEONE_ELSE/data/x/y.h5"]), set(),
     "refused", 0, "datatrail-contract-refusal", "1 contract-refused"),
    ("aged-out", _minoc(sorted(_present(_OLD))), set(),
     "empty", 0, "aged-out", "1 accepted-empty"),
]


@pytest.mark.parametrize(
    "case_id,files_half,present,status,n_rows,reason,tag",
    _VERDICTS, ids=[c[0] for c in _VERDICTS])
def test_verdict_taxonomy(monkeypatch, tmp_path, case_id, files_half, present,
                          status, n_rows, reason, tag):
    _install_fake_cli(monkeypatch,
                      lambda args: (0, _ps_payload(files_half), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch, present=present)
    text = _survey(tmp_path)

    state = _state(tmp_path)
    assert state.statuses == {f"{SCOPE}|{EVENT}": status}
    assert len(state.rows) == n_rows
    assert tag in text
    if reason is None:
        # NOTE: a `no-data` verdict writes NO ledger row anywhere. It is only
        # recoverable as the residual surveyed - rows - no_files.
        assert state.ledger == []
    else:
        assert [r["reason"] for r in state.ledger] == [reason]


def test_a_young_empty_event_is_retried_rather_than_committed(monkeypatch,
                                                              tmp_path):
    _install_fake_cli(monkeypatch, lambda args: (
        0, _ps_payload(_minoc(sorted(_present(_YOUNG)))), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    text = _survey(tmp_path)
    assert "1 resolved-but-empty (retry next run)" in text
    assert _state(tmp_path).statuses == {}


def test_a_contract_refusal_names_the_offending_uri_and_does_not_stall(
        monkeypatch, tmp_path):
    _install_fake_cli(monkeypatch, lambda args: (0, _ps_payload(
        _minoc(["cadc:SOMEONE_ELSE/data/x/y.h5"])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT, "100200201"])
    _fake_archive(monkeypatch)
    text = _survey(tmp_path)
    state = _state(tmp_path)
    assert set(state.statuses.values()) == {"refused"}
    detail = state.ledger[0]["detail"]
    assert "cadc:SOMEONE_ELSE/data/x/y.h5" in detail
    assert "1/1 replicas affected" in detail
    assert state.ledger[0]["common_path"] is None
    # a refusal is a per-event decision: it never rides the outage circuit.
    assert "service unreachable" not in text


def test_a_mixed_probe_result_becomes_incomplete_on_the_third_sighting(
        monkeypatch, tmp_path):
    # 1 of 2 candidates hard-errors. Passes 1-2 write nothing and bump the
    # attempt counter; pass 3 accepts what verified and flags the event.
    cp = f"cadc:CHIMEFRB/{_OLD}"
    names = [baseband_filename(EVENT, f) for f in FREQ_IDS]
    _install_fake_cli(monkeypatch, lambda args: (
        0, _ps_payload(_minoc([f"{cp}/{n}" for n in names])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch, present={f"{cp}/{names[0]}"},
                  error_for={f"{cp}/{names[1]}"})
    key = f"{SCOPE}|{EVENT}"

    ladder = []
    for _ in range(3):
        text = _survey(tmp_path)
        state = _state(tmp_path)
        ladder.append((dict(state.attempts), dict(state.statuses), len(state.rows)))
    assert ladder == [({key: 1}, {}, 0), ({key: 2}, {}, 0),
                      ({}, {key: "incomplete"}, 1)]
    assert "INCOMPLETE(1)" in text
    assert _state(tmp_path).incomplete == [f"{key}\tunresolved={names[1]}"]
    # and the single written row carries NO marker that its event is partial:
    # incomplete_events.txt is the only signal.
    assert "incomplete" not in json.dumps(_state(tmp_path).rows[0])


def test_every_candidate_erroring_takes_the_outage_circuit_not_incomplete(
        monkeypatch, tmp_path):
    # The true expired-certificate signature: datatrail answers fine, every
    # CADC probe fails. len(errored)==len(cand) diverts to service_down
    # (cadc.py:666), so such an event can never be committed as `incomplete`.
    cp = f"cadc:CHIMEFRB/{_OLD}"
    names = [baseband_filename(EVENT, f) for f in FREQ_IDS]
    _install_fake_cli(monkeypatch, lambda args: (
        0, _ps_payload(_minoc([f"{cp}/{n}" for n in names])), ""))
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch, error_for={f"{cp}/{n}" for n in names})
    clock = _install_clock(monkeypatch)

    with pytest.raises(SurveyUnavailableError) as excinfo:
        _survey(tmp_path)
    assert "remained unreachable for 3600s" in str(excinfo.value)
    state = _state(tmp_path)
    assert state.statuses == {} and state.rows == [] and state.attempts == {}
    # the circuit backs off exponentially and never spins: every wait is at
    # least the initial backoff except the final clamp to the deadline.
    assert clock.sleeps[:4] == [60.0, 120.0, 240.0, 480.0]
    assert sum(clock.sleeps) == pytest.approx(float(cadc_src._MAX_SERVICE_WAIT))


def test_a_mixed_result_can_never_reach_the_outage_circuit(monkeypatch,
                                                           tmp_path):
    # Structural gate: with 1..n-1 candidates erroring the event stays on the
    # `progress` path, so a partial archive outage cannot abort a whole survey.
    cp = f"cadc:CHIMEFRB/{_OLD}"
    names = [baseband_filename(EVENT, f) for f in FREQ_IDS]
    _install_fake_cli(monkeypatch, lambda args: (
        0, _ps_payload(_minoc([f"{cp}/{n}" for n in names])), ""))
    _no_backoff(monkeypatch)
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch, present={f"{cp}/{names[0]}"},
                  error_for={f"{cp}/{names[1]}"})
    clock = _install_clock(monkeypatch)
    text = _survey(tmp_path)
    assert clock.sleeps == []                 # no outage circuit entered
    assert "1 unresolved, retry" in text
    assert _state(tmp_path).attempts == {f"{SCOPE}|{EVENT}": 1}


def test_a_sustained_transient_aborts_with_the_certificate_advice(monkeypatch,
                                                                  tmp_path):
    # The full ladder through the REAL child boundary: datatrail keeps
    # answering with the error envelope, survey rides the circuit for exactly
    # _MAX_SERVICE_WAIT and then aborts, having preserved its state.
    calls = _install_fake_cli(monkeypatch, lambda args: (
        1, json.dumps({"error": "Server not responding."}), ""))
    _fake_enumeration(monkeypatch, [EVENT])
    _fake_archive(monkeypatch)
    clock = _install_clock(monkeypatch)

    with pytest.raises(SurveyUnavailableError) as excinfo:
        _survey(tmp_path)
    message = str(excinfo.value)
    assert "remained unreachable for 3600s" in message
    assert "cadc-get-cert" in message
    # 9 outage rounds x files()'s 4 attempts (1 + 3 retries)
    assert len(calls) == 36, len(calls)
    outage = [s for s in clock.sleeps if s >= 48.0]
    assert outage == [60.0, 120.0, 240.0, 480.0, 600.0, 600.0, 600.0, 600.0,
                      48.0]
    assert sum(clock.sleeps) == pytest.approx(float(cadc_src._MAX_SERVICE_WAIT))
    # the finally block still rendered every view before propagating
    for name in ("inventory.jsonl", "surveyed_events.txt",
                 "incomplete_events.txt", "no_files_events.jsonl",
                 "attempts.json"):
        assert (tmp_path / name).exists(), name


# ##########################################################################
# The child-process boundary itself: how a broken child is classified.
# ##########################################################################
def test_a_valid_payload_with_a_stdout_epilogue_reads_as_an_outage(monkeypatch):
    # _extract_json parses from the FIRST '{' to end-of-string, so anything
    # printed AFTER the payload breaks the parse and the healthy service reads
    # as "did not answer" -- which survey escalates to a 1-hour outage and a
    # misleading "renew the certificate" abort.
    good = json.dumps({"scopes": ["a"]})
    assert dt._extract_json(good) == {"scopes": ["a"]}
    assert dt._extract_json(good + "\nDone.\n") is None
    _install_fake_cli(monkeypatch, lambda args: (0, good + "\nDone.\n", ""))
    assert dt.Datatrail().list_scopes_checked() == ([], False)


def test_a_banner_containing_a_brace_defeats_the_payload_parse(monkeypatch):
    # The documented banner tolerance is narrower than datatrail_client.py:24-26
    # claims: it holds only for a preamble with no '{'.
    good = json.dumps({"scopes": ["a"]})
    assert dt._extract_json(_BANNER + good) == {"scopes": ["a"]}
    assert dt._extract_json("note {see docs}\n" + good) is None


@pytest.mark.parametrize("exc,fragment", [
    (OSError("no exe"), "OSError: no exe"),
    (subprocess.TimeoutExpired(["x"], 300), "TimeoutExpired:"),
    (MemoryError(), "MemoryError"),
])
def test_child_spawn_failures_are_transient_not_refusals(monkeypatch, exc,
                                                         fragment):
    # datatrail_client.py:116 catches Exception, so even MemoryError is
    # classified as a retryable outage rather than a fault. Characterized here
    # because widening or narrowing that except clause changes the verdict.
    def boom(args):
        raise exc
    _install_fake_cli(monkeypatch, boom)
    _no_backoff(monkeypatch)
    payload, diag = dt._run_json(["ps", SCOPE, EVENT])
    assert payload is None and fragment in diag
    assert dt.Datatrail().files(SCOPE, EVENT, retries=0) == (None, [], False)


def test_keyboardinterrupt_propagates_through_the_cli_boundary(monkeypatch):
    # KeyboardInterrupt is a BaseException, so it escapes the `except
    # Exception` above. Widening that clause would silently turn Ctrl-C into a
    # one-hour outage wait.
    def boom(args):
        raise KeyboardInterrupt
    _install_fake_cli(monkeypatch, boom)
    with pytest.raises(KeyboardInterrupt):
        dt.Datatrail().files(SCOPE, EVENT, retries=0)


def test_a_killed_child_is_an_outage(monkeypatch, capsys):
    _install_fake_cli(monkeypatch, lambda args: (-9, "", ""))
    payload, diag = dt._run_json(["ps", SCOPE, EVENT])
    assert payload is None and diag == "exit -9, no JSON on stdout"
    assert dt.Datatrail().list_scopes_checked() == ([], False)
    assert "exit -9, no JSON on stdout" in capsys.readouterr().err


def test_an_exhausted_deadline_returns_before_spawning_a_child(monkeypatch):
    # datatrail_client.py:108-112. This is the trap that silently neuters
    # clock-faking tests: with the deadline already past, NO child runs.
    calls = _install_fake_cli(monkeypatch, lambda args: (0, "{}", ""))
    payload, diag = dt._run_json(["ps", SCOPE, EVENT],
                                 deadline=dt.time.monotonic() - 1)
    assert (payload, diag) == (None, "survey outage deadline exceeded")
    assert calls == []


def test_a_pre_json_cli_aborts_with_the_upgrade_message(monkeypatch):
    # A datatrail-cli older than 0.11 rejects --json. That must abort loudly,
    # not read as an outage -- and it must escape BOTH arities.
    _install_fake_cli(monkeypatch,
                      lambda args: (2, "", "Error: No such option: --json"))
    for call in (lambda: dt.Datatrail().files(SCOPE, EVENT, retries=0),
                 lambda: dt.Datatrail().list_scopes_checked()):
        with pytest.raises(SystemExit) as excinfo:
            call()
        assert "datatrail-cli is too old" in str(excinfo.value)
        assert "datatrail-cli>=0.11" in str(excinfo.value)


@pytest.mark.parametrize("version,fragment", [
    (None, "cannot determine the datatrail-cli version"),
    ((0, 10, 3), "predates the --json machine-readable mode"),
])
def test_api_available_reports_an_unusable_cli(monkeypatch, version, fragment):
    monkeypatch.setattr(dt, "_cli_version", lambda: version)
    ok, detail = dt.Datatrail.api_available()
    assert not ok and fragment in detail
