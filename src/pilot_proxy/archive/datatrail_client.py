"""Discover CHIME/FRB archive holdings through the Datatrail CLI.

A survey needs to know which scopes, datasets, and events exist, and where
an event's files live at CADC. Datatrail supplies that information. Downloads
and size checks use CADC directly, so the scan/fetch path does not depend on
Datatrail.

This adapter uses ``datatrail ls --json`` and ``datatrail ps --json``, the
public interface introduced in datatrail-cli 0.11.0
(CHIMEFRB/datatrail-cli#160). The structured output avoids dependence on
terminal formatting or internal ``dtcli.src.functions`` APIs. ``ls`` returns
a ``scopes``, ``larger_datasets``, ``datasets``, or ``error`` dictionary.
``ps`` returns ``{"dataset", "scope", "files", "policies"}`` or an
``{"error": ...}`` envelope. An error exits with status 1.

An unavailable service and an empty dataset have different meanings. Adapter
methods preserve that distinction so a failed request can be retried without
recording missing data. A partially decoded response, such as a string in
place of the files or policies object, also reports ``ok=False``.

Each call runs ``sys.executable -m dtcli.cli`` in the active environment.
``_extract_json()`` skips any update banner before the JSON object. The child
inherits the environment with ``NO_COLOR=1``, ``TERM=dumb``, and ``PAGER=cat``
to keep output plain and prevent pager waits. A hard timeout bounds each call.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import List, Optional, Sequence, Tuple

# event IDs (six or more digits) are taken from datatrail's child-dataset names
_EVENT_RE = re.compile(r"\b\d{6,}\b")

# `datatrail ls/ps --json` was added in datatrail-cli 0.11.0. api_available()
# checks the version so that doctor reports an old install before a run.
# Without the check, a survey would read the old install as an outage partway
# through the walk.
_MIN_CLI = (0, 11)

# Hard timeout for each datatrail child (s). Running datatrail in a subprocess
# allows for a hung HTTP request to be killed. A timeout is treated as an outage
# and retried.
_TIMEOUT_S = float(os.environ.get(
    "PILOT_PROXY_DATATRAIL_TIMEOUT",
    os.environ.get("DATATRAWL_DATATRAIL_TIMEOUT", "300"),
))


def _cli_version() -> "Optional[Tuple[int, ...]]":
    """Installed datatrail-cli version as an int tuple, or None if unknown."""
    try:
        from importlib.metadata import version
        raw = version("datatrail-cli")
    except Exception:
        return None
    parts = re.findall(r"\d+", raw)
    return tuple(int(p) for p in parts[:3]) if parts else None


def _extract_json(stdout: str) -> "Optional[dict]":
    """Return the JSON object in the stdout of a --json call, or None.

    dtcli prints its update-available banner to stdout ahead of the command
    output, so the object is parsed from the first '{'. Every --json payload is
    a single object. Stdout without one, such as empty output or the Rich
    scopes table printed for an invalid scope, returns None, and callers treat
    None as no answer from the CLI.
    """
    i = stdout.find("{")
    if i < 0:
        return None
    try:
        obj = json.loads(stdout[i:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _run_json(args: "Sequence[str]", *, deadline=None
              ) -> "tuple[Optional[dict], str]":
    """Run `datatrail <args> --json` and return (payload, diag).

    payload=None means the CLI did not answer with JSON because of a spawn
    failure, a timeout (the child is killed), or non-JSON stdout, and diag
    gives the reason in one line. Service-level errors are JSON
    ({"error": ...} with exit 1) and are returned as a payload for the caller
    to classify. The only exception raised is SystemExit, when the installed
    CLI does not accept --json.
    """
    cmd = [sys.executable, "-m", "dtcli.cli", *args, "--json"]
    env = dict(os.environ, NO_COLOR="1", TERM="dumb", PAGER="cat")
    timeout = _TIMEOUT_S
    if deadline is not None:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            return None, "survey outage deadline exceeded"
        timeout = min(timeout, remaining)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except Exception as exc:                      # spawn failure / TimeoutExpired
        return None, f"{type(exc).__name__}: {exc}"
    payload = _extract_json(proc.stdout)
    if payload is None:
        tail = (proc.stderr or "").strip().splitlines()
        last = tail[-1] if tail else ""
        if "No such option" in last and "--json" in last:
            raise SystemExit(
                "datatrail-cli is too old for the --json contract this "
                "adapter requires (>= 0.11). Upgrade with: "
                "pip install 'datatrail-cli>=0.11'")
        return None, (f"exit {proc.returncode}, no JSON on stdout"
                      + (f" ({last})" if last else ""))
    return payload, ""


class DatatrailContractError(RuntimeError):
    """Datatrail answered, but the payload breaks the contract of this adapter.

    It is raised for schema drift in a newer CLI, a malformed replica list, a
    minoc replica that is not under the expected CADC collections, or replica
    paths with no usable common split. A retry returns the same answer, so this
    error is never treated as a transient failure to answer. Callers record the
    refusal and continue, and the survey commits the event as `refused` with
    the reason in its ledger."""


# CADC collections that minoc replicas are expected under. Any other replica
# is refused, and the refusal message names its URI, so any change to this
# tuple is made from a recorded case.
_MINOC_COLLECTIONS = ("cadc:CHIMEFRB/",)

# After 2026-08-03, Datatrail stopped writing the collection prefix on some
# minoc replicas. A 2025 event returns all 736 replicas as bare "data/..."
# paths, and a 2020 event can return both forms from the same directory. The
# bare paths name the same files, and cadcinfo resolves them once the prefix is
# restored. The prefix is therefore restored for paths that match the archive
# layout. Adding a bare root to _MINOC_COLLECTIONS would accept an arbitrary
# URI.
_MINOC_DEFAULT_COLLECTION = "cadc:CHIMEFRB/"
_BARE_REPLICA_ROOTS = ("data/",)


class ArchivePath(str):
    """A resolved path that records if its collection prefix was restored."""
    def __new__(cls, value, *, collection_restored=False):
        result = super().__new__(cls, value)
        result.collection_restored = bool(collection_restored)
        return result


def _restore_collection(uri: str) -> str:
    """Restore the collection prefix on a bare archive-relative replica path.

    A URI that names a known collection, or that matches no bare root, is
    returned unchanged. The caller then refuses a URI of the second kind.
    """
    if uri.startswith(_MINOC_COLLECTIONS):
        return uri
    if uri.startswith(_BARE_REPLICA_ROOTS) and len(_MINOC_COLLECTIONS) == 1:
        return _MINOC_COLLECTIONS[0] + uri
    return uri


class Datatrail:
    """Typed adapter for the machine-readable datatrail CLI (`--json`).

    The adapter holds no state. Every method call spawns one `datatrail` child,
    so one shared instance can be reused and called from the survey verify
    pool. The listing methods return [] on any error that datatrail reports,
    and the error is logged to stderr. Callers treat [] as unknown and never as
    an empty listing.
    """

    # availability, for doctor and preflight
    @staticmethod
    def installed() -> bool:
        """True if datatrail-cli (`dtcli`) can be imported."""
        try:
            import dtcli  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def api_available() -> "tuple[bool, str]":
        """Return (ok, detail) for the JSON contract of the installed datatrail-cli.

        A datatrail-cli older than 0.11 has no --json and would read as a
        service outage an hour into a survey. This check runs in doctor and
        reports the cause before the survey starts. When ok is False, `detail`
        says what to install.
        """
        v = _cli_version()
        if v is None:
            return False, ("cannot determine the datatrail-cli version "
                           "(is datatrail-cli installed in this environment?)")
        if v < _MIN_CLI:
            return False, (f"datatrail-cli {'.'.join(map(str, v))} predates "
                           f"the --json machine-readable mode; install "
                           f"datatrail-cli>=0.11")
        return True, ""

    # discovery listings, from `datatrail ls --json`
    @staticmethod
    def _list_result_checked(scope=None, dataset=None) -> "tuple[dict, bool]":
        """Run `datatrail ls [scope [dataset]] --json` and return (dict, ok).

        ok=False means the service could not answer, either because no JSON
        came back or because datatrail returned its own {"error": ...}. The
        dict is then {}. ok=True with an empty listing means nothing is
        registered there. A discovery walk must never record an outage as an
        empty listing, so the two cases are kept apart here. The unchecked
        helpers below return [] for both, for callers that already treat [] as
        unknown.

        A non-error answer must carry the key defined for the number of
        arguments: `scopes`, `larger_datasets`, or `datasets`. dtcli builds the
        first and third, the server builds the second, and 0.11 writes the key
        even when its list is empty. A parsed dict without the key is a
        contract change in a newer datatrail-cli. Since the version pin is
        open-ended, such a dict is treated as no answer and never as an empty
        scope."""
        args = ["ls"] + [a for a in (scope, dataset) if a]
        key = "datasets" if dataset else ("larger_datasets" if scope
                                          else "scopes")
        payload, diag = _run_json(args)
        if payload is None:
            sys.stderr.write(f"[datatrail ls scope={scope} dataset={dataset}] "
                             f"{diag}\n")
            return {}, False
        if payload.get("error"):
            sys.stderr.write(f"[datatrail ls scope={scope} dataset={dataset}] "
                             f"{payload['error']}\n")
            return {}, False
        if key not in payload:
            sys.stderr.write(
                f"[datatrail ls scope={scope} dataset={dataset}] unexpected "
                f"--json shape (no {key!r} key) -- datatrail-cli newer than "
                f"this adapter understands?\n")
            return {}, False
        if not isinstance(payload[key], list):
            # With the scopes endpoint unreachable behind a proxy, dtcli
            # 0.11.0 was seen to return the proxy error text as
            # {"scopes": "<proxy error prose>"} with exit 0, since
            # decode_response passes non-JSON bodies through as str.
            # list("<str>") would split that text into one "scope" per
            # character, so the answer is treated as no answer.
            sys.stderr.write(
                f"[datatrail ls scope={scope} dataset={dataset}] non-list "
                f"{key!r} value ({type(payload[key]).__name__}) -- treating "
                f"as not answered\n")
            return {}, False
        if any(not isinstance(value, str) or not value.strip()
               for value in payload[key]):
            sys.stderr.write(
                f"[datatrail ls scope={scope} dataset={dataset}] "
                f"{key!r} contains non-string or empty entries -- treating "
                "as not answered\n")
            return {}, False
        return payload, True

    def list_scopes(self) -> List[str]:
        """Every scope datatrail can see (`datatrail ls` with no scope)."""
        return self.list_scopes_checked()[0]

    def list_scopes_checked(self) -> "tuple[List[str], bool]":
        """Return (scopes, ok), where ok=False means datatrail did not answer."""
        res, ok = self._list_result_checked()
        return list(res.get("scopes", [])), ok

    def list_datasets_checked(self, scope: str) -> "tuple[List[str], bool]":
        """Return (datasets, ok), where ok=False means the scope could not be listed.

        The scope map records an outage of this kind apart from a scope with no
        datasets."""
        res, ok = self._list_result_checked(scope)
        return list(res.get("larger_datasets", [])), ok

    def children(self, scope: str, dataset: str) -> List[str]:
        """Return the child dataset names one level under (scope, dataset), unchanged.

        `events_in_dataset_checked` takes event IDs from this list. Recon's
        --expand writes the list unfiltered, since non-event products such as
        timestamped acquisitions and calibration containers have no event ID.
        Their names are the handles passed to `datatrail ps`. Like the other
        listing methods, this returns [] on an error, and [] means unknown.
        """
        return self.children_checked(scope, dataset)[0]

    def children_checked(self, scope: str,
                         dataset: str) -> "tuple[List[str], bool]":
        """Return (children, ok), the checked form of children().

        Recon's --expand uses it so that a container that cannot be listed is
        reported and is never written to the map as an empty container."""
        res, ok = self._list_result_checked(scope, dataset)
        return [str(n) for n in res.get("datasets", [])], ok

    def events_in_dataset_checked(
            self, scope: str, dataset: str) -> "tuple[List[str], bool]":
        """Checked event extraction used by a durable discovery walk."""
        children, ok = self.children_checked(scope, dataset)
        return ([ev for name in children for ev in _EVENT_RE.findall(name)], ok)

    # file listing, from `datatrail ps --json`
    def files(self, scope: str, dataset: str, *, retries: int = 3,
              base: float = 4.0, deadline=None) -> tuple:
        """Return (common_path, names, ok) for the minoc files of one dataset.

        The listing comes from `datatrail ps <scope> <dataset> --json`, reduced
        to the minoc replica list. An analyzer that needs a per-day companion,
        such as a gains day-dataset, calls this method so that it never parses
        the Rich table or imports dtcli internals.

        The return values follow common_path():

          (None, [], False)    the CLI or service did not answer, with the
                               {"error": ...} envelope (exit 1) or with no
                               payload. This is transient and is retried.
          DatatrailContractError
                               the service answered, but the payload breaks
                               the contract of this adapter (schema drift, or
                               malformed or out-of-collection replicas). It is
                               never retried and never treated as an outage.
          (None, [], True)     the query succeeded, and the dataset has no
                               minoc files ("files" is null or has no minoc
                               replicas).
          (path, names, True)  resolved. `path` has the cadc:CHIMEFRB/ prefix,
                               as in common_path(), and `names` are relative
                               to it in server order, so a fetch URI is
                               f"{path}/{name}".

        The paths are normalized as in dtcli's find_dataset_common_path. The
        cadc:CHIMEFRB/ prefix is stripped, '//' is collapsed, and the common
        prefix is trimmed to the last '/'.
        """
        delay = base
        for k in range(retries + 1):
            if deadline is None:
                payload, _diag = _run_json(["ps", scope, dataset])
            else:
                payload, _diag = _run_json(
                    ["ps", scope, dataset], deadline=deadline)
            if payload is None or "error" in payload:
                if k < retries:
                    sleep_for = delay
                    if deadline is not None:
                        remaining = float(deadline) - time.monotonic()
                        if remaining <= 0:
                            return None, [], False
                        sleep_for = min(sleep_for, remaining)
                    time.sleep(sleep_for)
                    delay *= 2
                    continue
                return None, [], False        # did not answer, retried out
            if "files" not in payload:
                # The 0.11 success payload always has the "files" key, which
                # may be null. A non-error dict without it is a contract change
                # in a newer datatrail-cli. A retry returns the same answer, so
                # none is made, and the dict never means a dataset with no
                # files.
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] unexpected --json "
                    "shape (no 'files' key) -- datatrail-cli newer than this "
                    "adapter understands?")
            files_resp = payload["files"]
            # "files": null is how the CLI renders functions.ps returning
            # (None, policy), where the find half had no answer for this name
            # and the policies resolved. For a dataset that the map lists, this
            # means no files, as in the no-minoc case below. Any other non-dict
            # value, such as a string, is a fault or schema drift and never
            # means no data. 0.11 wraps string halves in the error envelope, so
            # no known path returns this shape.
            if files_resp is not None and not isinstance(files_resp, dict):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] non-dict 'files' "
                    f"value ({type(files_resp).__name__})")
            if files_resp is None:
                return None, [], True
            locations = files_resp.get("file_replica_locations")
            if locations is None:
                return None, [], True
            if not isinstance(locations, Mapping):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] non-object "
                    "'file_replica_locations' value")
            uris = locations.get("minoc")
            if uris is None:
                return None, [], True
            if not isinstance(uris, (list, tuple)):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] non-list 'minoc' "
                    "replica locations")
            if not uris:
                return None, [], True
            if any(not isinstance(uri, str) or not uri.strip() for uri in uris):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] malformed 'minoc' "
                    "replica locations")
            canonical = [uri.replace("//", "/") for uri in uris]
            normalized = [_restore_collection(uri) for uri in canonical]
            stray = [u for u in normalized
                     if not u.startswith(_MINOC_COLLECTIONS)]
            if stray:
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] minoc replica "
                    f"{stray[0]!r} outside the expected collection(s) "
                    f"{list(_MINOC_COLLECTIONS)} "
                    f"({len(stray)}/{len(normalized)} replicas affected)")
            prefixes = {p for u in normalized
                        for p in _MINOC_COLLECTIONS if u.startswith(p)}
            if len(prefixes) != 1:
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] minoc replicas span "
                    f"multiple collections {sorted(prefixes)}; one dataset "
                    "must resolve to one common path")
            prefix = prefixes.pop()
            paths = [uri[len(prefix):] for uri in normalized]
            from .sources.cadc_inventory import _safe_archive_name
            if any(not _safe_archive_name(path) for path in paths):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] replica path is not "
                    "a canonical path below the collection root")
            common = os.path.commonprefix(paths)
            if not common.endswith("/"):
                common = "/".join(common.split("/")[:-1])
            common = common.rstrip("/")
            names = [p[len(common):].lstrip("/") for p in paths]
            if not common or any(not name for name in names):
                raise DatatrailContractError(
                    f"[datatrail ps {scope} {dataset}] replica paths have no "
                    "usable common directory/name split")
            return ArchivePath(
                f"{prefix}{common.lstrip('/')}",
                collection_restored=normalized != canonical), names, True
        return None, [], False

    # common-path resolution
    def common_path(self, scope: str, event: str, *, retries: int = 3,
                    base: float = 4.0, deadline=None) -> tuple:
        """Return (path, ok), the CADC common path of one event.

        This is the `ps --json` call of files(), restricted to a single
        directory. Survey readers append basenames to this path, so nested
        relative names cannot be represented. A nested name raises
        DatatrailContractError so that it is never read as absent data.

        (None, False) means the query failed (transient or service down, after
        retries). (None, True) means the query succeeded and found no minoc
        files. (path, True) is resolved, with the cadc:CHIMEFRB collection
        prefix.
        """
        cp, names, ok = self.files(
            scope, event, retries=retries, base=base, deadline=deadline)
        if any("/" in name for name in names):
            raise DatatrailContractError(
                f"[datatrail ps {scope} {event}] minoc replicas span "
                "multiple directories; cannot resolve one event directory")
        return cp, ok


# shared default instance for survey orchestration and preflight
DATATRAIL = Datatrail()
