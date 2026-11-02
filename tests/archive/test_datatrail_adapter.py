#!/usr/bin/env python3
"""
Offline tests for the cadc-datatrail source.

These tests cover the certificate default, the survey prerequisites checked by
preflight(), and the parsing of the datatrail CLI output. These paths otherwise
run only on CANFAR. No CADC or datatrail access is needed, since the CANFAR
mount, the certificate files, and the --json boundary of the datatrail CLI are
faked. The CLI is faked at subprocess.run, so each test also pins the
invocation itself: the same interpreter with `-m dtcli.cli`, a trailing
`--json`, and a parse that skips the update banner.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_datatrail_adapter.py
"""
from __future__ import annotations

import json
import os

import pytest
import sys
import types

from pilot_proxy.archive.sources import cadc as src
from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.interfaces import RunContext


# _default_cert() uses an explicit CADC_CERT, else the personal ~/.ssl default
def test_cert_explicit_env_wins(monkeypatch):
    monkeypatch.setenv("CADC_CERT", "/somewhere/explicit.pem")
    # an explicit path is used even when the file does not exist, since the user set it
    monkeypatch.setattr(src.os.path, "exists", lambda p: False)
    assert src._default_cert() == "/somewhere/explicit.pem"


def test_cert_defaults_to_personal_when_no_env(monkeypatch):
    # without CADC_CERT the personal ~/.ssl path is used, and preflight checks that it exists
    monkeypatch.delenv("CADC_CERT", raising=False)
    assert src._default_cert() == os.path.expanduser("~/.ssl/cadcproxy.pem")


# fake datatrail CLI
# The one subprocess call of the adapter is routed to handler(args), which
# returns (returncode, stdout, stderr). Here, args excludes the
# [sys.executable, -m, dtcli.cli] prefix and the trailing --json. The fake
# asserts both, so a changed invocation fails here.
class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _install_fake_cli(monkeypatch, handler=None, version=(0, 11, 0)):
    """Fake a datatrail install at `version` with CLI calls answered by `handler`.

    With handler=None the test expects no datatrail subprocess, and any call
    fails an assertion. The preflight tests fake list_scopes above this
    boundary, so no subprocess may be spawned.
    """
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: version)

    def fake_run(cmd, **kw):
        assert handler is not None, f"unexpected datatrail call: {cmd}"
        assert cmd[:3] == [sys.executable, "-m", "dtcli.cli"], cmd
        assert cmd[-1] == "--json", cmd
        rc, out, err = handler(cmd[3:-1])
        return _Proc(rc, out, err)

    monkeypatch.setattr(dt.subprocess, "run", fake_run)


# The dtcli group callback prints this banner to stdout ahead of the command
# output when PyPI has a newer release. The parse must skip it (see
# _extract_json).
_BANNER = "A new release of datatrail-cli is available: 0.11.0 \u2192 0.12.0\n\n"


def _ps_payload(files):
    return json.dumps(
        {"dataset": "d", "scope": "s", "files": files, "policies": {"p": 1}})


# preflight() checks the survey prerequisites, which are the CLI --json
# contract and the presence of each scope
class _Inst:
    name = "chime"
    scopes = ("chime.event.baseband.raw", "chime.scheduled.baseband.raw")


def _ctx(scope=None):
    return RunContext(instrument=_Inst(),
                      options=({"scope": scope} if scope is not None else {}))


def _clean_env_source(monkeypatch):
    """Return a source whose cert and cadc checks pass, so only the datatrail checks are tested."""
    s = src.CadcDatatrailSource()
    monkeypatch.setattr(src.os.path, "exists", lambda p: True)   # cert "present"
    monkeypatch.setitem(sys.modules, "cadcdata", types.ModuleType("cadcdata"))
    monkeypatch.setitem(sys.modules, "cadcutils", types.ModuleType("cadcutils"))
    return s


def test_preflight_silent_without_datatrail(monkeypatch):
    s = _clean_env_source(monkeypatch)
    monkeypatch.setitem(sys.modules, "dtcli", None)   # `import dtcli` raises ImportError
    ok, problems, notes = s.preflight(_ctx())
    assert ok and problems == []


def test_preflight_flags_pre_json_datatrail(monkeypatch):
    # datatrail is installed but older than the --json contract (< 0.11), so the
    # check must name this cause before the survey starts its walk
    s = _clean_env_source(monkeypatch)
    _install_fake_cli(monkeypatch, version=(0, 10, 3))
    monkeypatch.setattr(src.DATATRAIL, "list_scopes", lambda: list(_Inst.scopes))
    ok, problems, notes = s.preflight(_ctx())
    assert not ok
    assert any("--json" in p and "0.11" in p for p in problems)


def test_preflight_flags_unknown_scope(monkeypatch):
    s = _clean_env_source(monkeypatch)
    _install_fake_cli(monkeypatch)
    # datatrail knows only one of the two scopes configured for the instrument
    monkeypatch.setattr(src.DATATRAIL, "list_scopes",
                        lambda: ["chime.event.baseband.raw"])
    ok, problems, notes = s.preflight(_ctx())
    assert not ok
    assert any("not found in datatrail" in p
               and "chime.scheduled.baseband.raw" in p for p in problems)


def test_preflight_clean_when_scopes_known(monkeypatch):
    s = _clean_env_source(monkeypatch)
    _install_fake_cli(monkeypatch)
    monkeypatch.setattr(src.DATATRAIL, "list_scopes", lambda: list(_Inst.scopes))
    ok, problems, notes = s.preflight(_ctx())
    assert ok and problems == []
    assert notes == []          # scopes validated, so no skip note


def test_preflight_validates_scope_override(monkeypatch):
    # an explicit --scope override unknown to datatrail must be flagged
    s = _clean_env_source(monkeypatch)
    _install_fake_cli(monkeypatch)
    monkeypatch.setattr(src.DATATRAIL, "list_scopes", lambda: list(_Inst.scopes))
    ok, problems, notes = s.preflight(_ctx(scope="chime.typo.baseband.raw"))
    assert not ok
    assert any("chime.typo.baseband.raw" in p for p in problems)


def test_preflight_skips_scope_check_when_ls_unreachable(monkeypatch):
    s = _clean_env_source(monkeypatch)
    _install_fake_cli(monkeypatch)

    def _boom():
        raise RuntimeError("datatrail server unreachable")   # unexpected failure

    monkeypatch.setattr(src.DATATRAIL, "list_scopes", _boom)
    # a failed listing skips scope validation with a visible, non-fatal note,
    # and no scope is reported as invalid
    ok, problems, notes = s.preflight(_ctx())
    assert ok and problems == []
    assert any("not validated" in n for n in notes)   # the [--] skip note


# api_available() is the doctor-time gate. It checks the datatrail-cli version,
# since the adapter depends only on the public `--json` flag (>= 0.11).
def test_api_available_ok_at_contract_version(monkeypatch):
    _install_fake_cli(monkeypatch, version=(0, 11, 0))
    ok, detail = src.Datatrail.api_available()
    assert ok and detail == ""


def test_api_available_flags_pre_json_cli(monkeypatch):
    _install_fake_cli(monkeypatch, version=(0, 10, 3))
    ok, detail = src.Datatrail.api_available()
    assert not ok and "0.11" in detail and "--json" in detail


def test_api_available_flags_unknown_version(monkeypatch):
    _install_fake_cli(monkeypatch, version=None)
    ok, detail = src.Datatrail.api_available()
    assert not ok and "version" in detail


# Listing goes through `datatrail ls --json`, the public contract of the CLI.
# These tests check that the adapter reads each payload shape, parses past the
# update banner, and returns [] for any case the CLI did not answer.
def test_listing_via_ls_json(monkeypatch):
    def handler(args):
        if args == ["ls"]:                       # list scopes, with the banner first
            return 0, _BANNER + json.dumps(
                {"scopes": ["chime.event.baseband.raw",
                            "chime.scheduled.baseband.raw"]}), ""
        if args == ["ls", "chime.event.baseband.raw"]:
            return 0, json.dumps({"larger_datasets": ["2023", "2024"]}), ""
        if args == ["ls", "chime.event.baseband.raw", "2024"]:
            return 0, json.dumps(
                {"datasets": ["123456789", "987654321"]}), ""
        raise AssertionError(f"unexpected ls args: {args}")

    _install_fake_cli(monkeypatch, handler)

    assert src.DATATRAIL.list_scopes() == ["chime.event.baseband.raw",
                                           "chime.scheduled.baseband.raw"]
    assert src.DATATRAIL.list_datasets_checked("chime.event.baseband.raw") == (["2023", "2024"], True)
    assert src.DATATRAIL.events_in_dataset_checked("chime.event.baseband.raw", "2024")[0] == \
        ["123456789", "987654321"]


def test_listing_error_degrades_to_empty(monkeypatch):
    # an error reported by datatrail (JSON {"error": ...}, exit 1) must return
    # [] for an undetermined listing and must not raise
    _install_fake_cli(monkeypatch, lambda a: (
        1, json.dumps({"error": "Server not responding."}), ""))
    assert src.DATATRAIL.list_scopes() == []
    assert src.DATATRAIL.list_datasets_checked("s")[0] == []
    assert src.DATATRAIL.events_in_dataset_checked("s", "d")[0] == []


def test_listing_nonjson_stdout_is_not_answered(monkeypatch):
    # for an invalid scope the CLI prints a Rich scopes table (exit 0) instead
    # of JSON, and this must be reported as not answered (ok=False)
    table = "\u250f\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2513\n" \
            "\u2503 Scopes \u2503\n"
    _install_fake_cli(monkeypatch, lambda a: (0, table, "Scope does not exist!"))
    datasets, ok = src.DATATRAIL.list_datasets_checked("chime.typo")
    assert (datasets, ok) == ([], False)


def test_listing_shape_drift_is_not_answered(monkeypatch):
    # a parsed non-error dict without the key defined for this arity must be
    # reported as not answered (ok=False). Such a dict comes from a changed
    # contract in a newer datatrail-cli, since the version pin has no ceiling.
    _install_fake_cli(monkeypatch, lambda a: (
        0, json.dumps({"renamed_in_0_12": []}), ""))
    assert src.DATATRAIL.list_scopes_checked() == ([], False)
    assert src.DATATRAIL.list_datasets_checked("s") == ([], False)
    assert src.DATATRAIL.children_checked("s", "d") == ([], False)
    # the key present with an empty list is an answered empty listing, and
    # recon may write that row to the map
    _install_fake_cli(monkeypatch, lambda a: (
        0, json.dumps({"larger_datasets": []}), ""))
    assert src.DATATRAIL.list_datasets_checked("s") == ([], True)


def test_listing_nonlist_value_is_not_answered(monkeypatch):
    # captured live from dtcli 0.11.0 behind a blocking proxy, where the error
    # text of the scopes endpoint came back as the value with exit 0. A plain
    # list() would split it into one "scope" per character.
    _install_fake_cli(monkeypatch, lambda a: (0, json.dumps(
        {"scopes": "Host not in allowlist: frb.chimenet.ca."}), ""))
    assert src.DATATRAIL.list_scopes_checked() == ([], False)
    _install_fake_cli(monkeypatch, lambda a: (
        0, json.dumps({"datasets": "oops"}), ""))
    assert src.DATATRAIL.children_checked("s", "d") == ([], False)


# files() and common_path() go through `datatrail ps --json`. These tests check
# that each payload shape maps to the contract that distinguishes between an
# outage and an empty answer.
def test_files_via_ps_json(monkeypatch):
    """files() reads `ps --json` and normalizes the paths as dtcli does.

    The normalization strips the prefix, collapses //, and trims the common
    prefix to the last /.
    """
    day_uris = [
        "cadc:CHIMEFRB/data/gbo/complex_gains/20230530/gain_A_casa.h5",
        "cadc:CHIMEFRB/data/gbo//complex_gains/20230530/gain_B_cyga.h5",
    ]

    def handler(args):
        assert args == ["ps", "gbo.acquisition.processed", "20230530"], args
        return 0, _ps_payload(
            {"file_replica_locations": {"minoc": list(day_uris)}}), ""

    _install_fake_cli(monkeypatch, handler)
    cp, names, ok = src.DATATRAIL.files("gbo.acquisition.processed", "20230530")
    assert ok
    assert cp == "cadc:CHIMEFRB/data/gbo/complex_gains/20230530"
    assert names == ["gain_A_casa.h5", "gain_B_cyga.h5"]
    # a fetch URI is exactly path/name, the same join used by enumerate
    assert f"{cp}/{names[0]}" == day_uris[0]


def test_files_no_minoc_null_and_outage(monkeypatch):
    # answered with no minoc replica, so there are no bytes (ok=True)
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": {"arc": ["x"]}}), ""))
    assert src.DATATRAIL.files("s", "d") == (None, [], True)
    # "files": null is how the CLI renders functions.ps returning (None, policy).
    # The find half had no answer for this name, which is still a no-data answer.
    _install_fake_cli(monkeypatch, lambda a: (0, _ps_payload(None), ""))
    assert src.DATATRAIL.files("s", "d") == (None, [], True)
    # the error envelope (exit 1) is an outage and is retried (ok=False)
    _install_fake_cli(monkeypatch, lambda a: (1, json.dumps(
        {"error": {"files": "boom", "policies": "boom"}}), ""))
    assert src.DATATRAIL.files("s", "d", retries=0) == (None, [], False)


def test_files_shape_drift_raises_without_retry(monkeypatch):
    # a success payload without the "files" key is schema drift, since 0.11
    # always emits the key, null or not. Drift is deterministic, so it is raised
    # at once as a contract refusal with no retry, and it is reported as neither
    # an empty dataset nor a transient outage.
    calls = []

    def handler(args):
        calls.append(args)
        return 0, json.dumps({"dataset": "d", "scope": "s",
                              "policies": {}}), ""

    _install_fake_cli(monkeypatch, handler)
    with pytest.raises(src.DatatrailContractError):
        src.DATATRAIL.files("s", "d", retries=3)
    assert len(calls) == 1


def test_files_nonnull_nondict_value_raises(monkeypatch):
    # a string "files" value is degradation or drift, since 0.11 wraps string
    # halves in the error envelope and its contract has no such shape. The
    # no-data answer is kept for "files": null alone, and this deterministic
    # case is raised as a contract refusal with no retry.
    calls = []

    def handler(args):
        calls.append(args)
        return 0, json.dumps({"dataset": "d", "scope": "s",
                              "files": "Server hiccup", "policies": {}}), ""

    _install_fake_cli(monkeypatch, handler)
    with pytest.raises(src.DatatrailContractError):
        src.DATATRAIL.files("s", "d", retries=3)
    assert len(calls) == 1


def test_files_spawn_failure_is_outage(monkeypatch):
    def handler(args):
        raise OSError("cannot spawn")
    _install_fake_cli(monkeypatch, handler)
    assert src.DATATRAIL.files("s", "d", retries=0) == (None, [], False)


def test_common_path_shares_ps_contract(monkeypatch):
    uris = ["cadc:CHIMEFRB/data/chime/baseband/raw/2024/01/02/astro_303030303/"
            "baseband_303030303_100.h5"]
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": {"minoc": uris}}), ""))
    cp, ok = src.DATATRAIL.common_path("chime.event.baseband.raw", "303030303")
    assert ok and cp == ("cadc:CHIMEFRB/data/chime/baseband/raw/2024/01/02/"
                         "astro_303030303")
    _install_fake_cli(monkeypatch, lambda a: (0, _ps_payload(None), ""))
    assert src.DATATRAIL.common_path("s", "e") == (None, True)   # no-data
    _install_fake_cli(monkeypatch, lambda a: (1, json.dumps(
        {"error": "down"}), ""))
    assert src.DATATRAIL.common_path("s", "e", retries=0) == (None, False)


# Since 2026-08-03, datatrail omits the collection prefix on some minoc
# replicas. The bare paths name the same artifacts, so the adapter restores the
# prefix and accepts the event.
def test_bare_replica_paths_get_collection_restored(monkeypatch):
    # a 2025 event whose replicas all come back without "cadc:CHIMEFRB/"
    uris = ["data/chime/baseband/raw/2025/03/31/astro_1111116060/"
            "baseband_1111116060_832.h5",
            "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
            "baseband_1111116060_506.h5"]
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": {"minoc": uris}}), ""))
    cp, names, ok = src.DATATRAIL.files("chime.event.baseband.raw",
                                        "1111116060")
    assert ok
    assert cp == ("cadc:CHIMEFRB/data/chime/baseband/raw/2025/03/31/"
                  "astro_1111116060")
    assert sorted(names) == ["baseband_1111116060_506.h5",
                             "baseband_1111116060_832.h5"]


def test_mixed_bare_and_prefixed_replicas_share_one_common_path(monkeypatch):
    # a 2020 event whose directory returns both forms (1 in 900 bare)
    uris = ["cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
            "astro_100200200/baseband_100200200_0.h5",
            "data/chime/baseband/raw/2020/07/15/astro_100200200/"
            "baseband_100200200_156.h5"]
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": {"minoc": uris}}), ""))
    cp, names, ok = src.DATATRAIL.files("chime.event.baseband.raw",
                                        "100200200")
    assert ok
    assert cp == ("cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
                  "astro_100200200")
    assert sorted(names) == ["baseband_100200200_0.h5",
                             "baseband_100200200_156.h5"]


def test_replica_outside_any_known_layout_still_refuses(monkeypatch):
    # the prefix is restored only for the archive layout, so a URI that names
    # neither a known collection nor that layout is still refused
    uris = ["cadc:SOMEONE_ELSE/data/chime/baseband/raw/2025/03/31/"
            "astro_1111116060/baseband_1111116060_832.h5"]
    _install_fake_cli(monkeypatch, lambda a: (
        0, _ps_payload({"file_replica_locations": {"minoc": uris}}), ""))
    with pytest.raises(src.DatatrailContractError):
        src.DATATRAIL.files("chime.event.baseband.raw", "1111116060")
