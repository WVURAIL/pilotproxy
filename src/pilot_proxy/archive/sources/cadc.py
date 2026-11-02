"""Data source for CADC storage and the CHIME/FRB Datatrail archive.

This source is used for CANFAR archive work and has three parts:

  enumerate()  reads an inventory (inventory.jsonl) and yields one Unit per
               file, filtered to the selected freq_ids, events, or both. The
               inventory is a cheap offline listing, so enumerate never uses
               the network.

  survey()     builds inventory.jsonl. It walks the Datatrail scope(s),
               discovers every event, and verifies at CADC each file that the
               reader's archive shape (Reader.survey_files) declares for the
               event. For baseband this is one HDF5 file per freq_id, and the
               reader of another product declares that product's files. The
               survey is resumable and incremental.

  fetch()      stages one file with cadcget through a CADC
               StorageInventoryClient, authenticated by a proxy certificate,
               with bounded retries.

Prerequisites, checked by ``doctor``:

  * a valid CADC proxy cert (CADC_CERT or ~/.ssl/cadcproxy.pem)
  * the ``cadcdata`` and ``cadcutils`` packages, for enumerate, fetch, and verify
  * the ``datatrail-cli`` package, for survey only (the ``[survey]`` extra)
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Optional

from ..datatrail_client import DATATRAIL, Datatrail, DatatrailContractError
from ..interfaces import (
    READY,
    DataSource,
    PluginInfo,
    RunContext,
    SurveyUnavailableError,
    Unit,
)
from ..names import validate_identifier
from ..recon import match_terms, recon
from ..selection import parse_freq_ids, parse_selection
from ..survey_state import (
    STALE_VIEWS_MARKER,
    VIEW_FLUSH_INTERVAL,
    SurveyRun,
    SurveyStore,
    atomic_write_json,
    build_configuration,
    ensure_manifest,
    load_attempts,
    with_survey_output_lock,
)
from . import cadc_transport as _cadc_transport
from .cadc_inventory import (
    annotate_row,
    candidate_file,
    join_uri,
    logical_unit_key,
    parse_row,
)


def _default_shape():
    """Return the reader whose file shape survey uses when the caller supplies none.

    This covers direct src.survey() calls in tests. The chime-baseband shape is
    this source's default naming. The reader is imported lazily, so importing
    this module does not load the reader stack.
    """
    from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
    return ChimeBasebandReader()


def _default_cert() -> str:
    """Return the path of the CADC proxy cert used for the direct fetch path.

    CADC_CERT takes precedence and is used even when the file is absent, since it
    states the user's intent and is the simple way to point a headless or batch
    job at a different cert. Otherwise the standard ~/.ssl/cadcproxy.pem written
    by ``cadc-get-cert`` is used. preflight() checks that the resolved path
    exists, so a missing or unrenewed cert is reported by ``doctor`` before any
    fetch starts.
    """
    return os.environ.get("CADC_CERT") or os.path.expanduser("~/.ssl/cadcproxy.pem")


STORAGE_SERVICE_ENV = "PILOT_PROXY_STORAGE_SERVICE"


def storage_service_override() -> Optional[str]:
    """Return the Storage Inventory service to fetch through, or None for the default.

    cadcdata routes every download through ivo://cadc.nrc.ca/global/raven, which
    locates a replica before the bytes are transferred. When that locator is
    degraded, the replicas themselves stay healthy, and a run can continue with
    the client pointed straight at one, such as ivo://cadc.nrc.ca/uvic/minoc.
    Replicas serve the same objects, and fetch() still checks every staged file
    against the size recorded in the inventory. This setting therefore changes
    only the route of the bytes, and the same bytes arrive. The resolved value
    is recorded in scan_scope.json, so the route a run actually took is kept in
    its provenance.
    """
    value = (os.environ.get(STORAGE_SERVICE_ENV) or "").strip()
    return value or None


# Survey defaults and retry limits
# Final fallback when neither --scope nor the telescope's YAML ``scopes``
# resolve. Each station (chime and the kko, gbo, and hco outriggers) declares
# its own baseband scope(s) in instruments/*.yaml, and survey uses those. This
# constant applies only to a telescope that declares none. CHIME events with an
# outrigger label are still dropped unless --include-outrigger is given.
_DEFAULT_SCOPES = ("chime.event.baseband.raw", "chime.scheduled.baseband.raw")

_DEFAULT_MINIMUM_ARCHIVE_BYTES = 1

_MAX_ATTEMPTS = 3            # per-event verification retries across resumes
_MAX_SERVICE_WAIT = 3600     # max wait (s) for a service or cert outage before aborting
_DEFAULT_SURVEY_WORKERS = 12
_DATASET_PROGRESS_SEGMENTS = 10
_INITIAL_SERVICE_BACKOFF = 60
_MAX_SERVICE_BACKOFF = 600
_EVENT_PROGRESS_INTERVAL = 100
_RETRY_BACKOFF_MULTIPLIER = 2
_ERROR_SUMMARY_LENGTH = 200
_DEFAULT_ARCHIVE_RETRIES = 3
_INITIAL_ARCHIVE_BACKOFF = 4.0

# Minimum age (days) for an all-absent ("empty") event to be accepted on its
# first sighting. By then replication to CADC has long settled, so a definitive
# NotFound on every probe means the bytes aged off or never arrived, and no
# number of re-checks recovers them. Younger or undatable events keep the
# _MAX_ATTEMPTS cross-resume allowance, which exists for the one transient
# case of archive-side replication still in progress for a recently
# registered event. The age is set per run with --empty-age-days.
_EMPTY_TERMINAL_AGE_DAYS = 30

_DATE_RE = re.compile(r"/raw/(\d{4})/(\d{2})/(\d{2})/")
_OUTRIGGER_RE = re.compile(r"outrigger", re.IGNORECASE)

_ENUM_CACHE_SCHEMA = 1


def _monotonic() -> float:
    """Patchable monotonic clock for the survey outage budget."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Patchable sleep boundary for survey outage backoff."""
    time.sleep(seconds)


def _obs_age_days(obs_date) -> Optional[int]:
    """Return the whole days since ``obs_date``, or None when it is absent or unparseable.

    ``obs_date`` is 'YYYY-MM-DD' as _DATE_RE extracts it from the Common Path,
    or 'unknown'. A None result sends the event to the retry path, so an event
    that cannot be dated is never written off on a single sighting.
    """
    try:
        d = datetime.date.fromisoformat(str(obs_date))
    except ValueError:
        return None
    return (datetime.datetime.now(datetime.timezone.utc).date() - d).days


def _resolve_freq_ids(spec, n_channels: Optional[int]) -> List[int]:
    """Parse --freq-ids into a sorted list of freq_id ints.

    Accepts 'all' (every freq_id in the instrument's band, range(n_channels)),
    a list, or the same string forms as --select.
    """
    all_ids = (list(range(n_channels)) if n_channels is not None else [])
    if spec is None:
        return all_ids
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return all_ids
    freq_ids = parse_freq_ids(spec, n_channels=n_channels)
    return sorted(freq_ids) if freq_ids else all_ids


def _load_json_file(path: Path, default):
    """Read and parse JSON from ``path``, or return ``default``.

    ``default`` is returned when the file is missing, empty, or corrupt, such as
    a checkpoint that a previous run was killed while writing.
    """
    try:
        if not path.exists():
            return default
        text = path.read_text().strip()
        return json.loads(text) if text else default
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def _enumerate_events(scopes, include_outrigger, cache_path: Path, re_enumerate):
    """Return {(scope, event): [labels...]} for all larger-datasets, cached to disk.

    This is phase 1 of the survey, a cheap network-only listing. The result is
    cached, so a re-run goes straight to verification unless --re-enumerate is
    given.
    """
    if cache_path.exists() and not re_enumerate:
        raw = _load_json_file(cache_path, None)
        if isinstance(raw, dict):
            if raw.get("schema") == _ENUM_CACHE_SCHEMA:
                cached_scopes = sorted(map(str, raw.get("scopes", ())))
                event_map = raw.get("events")
            else:
                event_map = None
                cached_scopes = []
            valid = (
                isinstance(event_map, dict)
                and all(isinstance(key, str) and "|" in key
                        and isinstance(labels, list)
                        and all(isinstance(label, str) for label in labels)
                        for key, labels in event_map.items())
            )
            if valid:
                if cached_scopes == sorted(map(str, scopes)):
                    return {tuple(key.split("|", 1)): list(labels)
                            for key, labels in event_map.items()}
                # The directory was enumerated for different scope(s). The stale
                # list would survey the wrong scope without notice, so the run
                # stops with an error.
                raise SystemExit(
                    f"inventory directory {cache_path.parent} was already "
                    f"enumerated for scope(s) {cached_scopes}, but this run "
                    f"requests {sorted(scopes)}. An inventory is tied to its "
                    "scope(s); use a fresh --name/output directory.")
        # a corrupt or empty cache is re-enumerated below

    membership: dict = defaultdict(set)
    for scope in scopes:
        datasets, ok = DATATRAIL.list_datasets_checked(scope)
        if not ok:
            raise SurveyUnavailableError(
                f"datatrail could not list datasets under scope {scope!r}; "
                f"the existing event cache at {cache_path} was left untouched. "
                "Re-run when the service responds.")
        # One ``datatrail ls`` per dataset makes a slow walk, so progress is
        # printed without all N names. The --scopes-only recon lists the
        # dataset structure.
        print(f"{scope}: walking {len(datasets)} larger-dataset(s)", flush=True)
        step = max(1, len(datasets) // _DATASET_PROGRESS_SEGMENTS)
        for i, ds in enumerate(datasets, 1):
            discovered, child_ok = DATATRAIL.events_in_dataset_checked(scope, ds)
            if not child_ok:
                raise SurveyUnavailableError(
                    f"datatrail could not list children of {scope!r}/{ds!r}; "
                    f"the existing event cache at {cache_path} was left "
                    "untouched. Re-run when the service responds.")
            for ev in discovered:
                membership[(scope, ev)].add(ds)
            if i % step == 0 or i == len(datasets):
                print(f"  ...{i}/{len(datasets)} datasets", flush=True)

    out_map = {f"{s}|{e}": sorted(lbls) for (s, e), lbls in membership.items()}
    atomic_write_json(cache_path, {
        "schema": _ENUM_CACHE_SCHEMA,
        "scopes": sorted(map(str, scopes)),
        "events": out_map,
    })
    n_out = sum(1 for lbls in membership.values()
                if any(_OUTRIGGER_RE.search(x) for x in lbls))
    print(f"\nenumerated {len(membership)} unique events; {n_out} carry an "
          f"outrigger label ({'KEPT' if include_outrigger else 'BLOCKED'})", flush=True)
    return {k: sorted(v) for k, v in membership.items()}


def _commit_decision(n_errored: int, attempts: int, n_records: int,
                     max_attempts: int = _MAX_ATTEMPTS,
                     empty_max_attempts: Optional[int] = None) -> tuple:
    """Return (write_records, mark_done, incomplete) for one event.

      case                          result
      clean, rows resolved          write rows, mark done
      partial, attempts left        write nothing, do not mark (clean retry)
      partial, out of attempts      write what verified, mark done, flag
      empty, retries left           write nothing, retry on the next resume
      empty, out of attempts        accept as empty, mark done

    ``n_records`` is the number of files verified present. An event is empty
    when its common path resolved but it has zero present files and zero hard
    errors. Such an event produced no usable file, so it is never written out as
    a clean 0-row done.

    ``empty_max_attempts`` caps the empty case only (None uses
    ``max_attempts``). survey passes 1 when the observation is old enough that
    absence cannot be replication lag (_EMPTY_TERMINAL_AGE_DAYS or
    --empty-age-days). A definitively absent old event is then accepted on its
    first sighting, and no cross-resume probes are spent on events that have
    never been seen to recover. The errored and partial path is unaffected,
    since a hard probe error is not an answer, however old the observation.
    """
    if n_errored == 0:
        if n_records == 0:                        # resolved, but nothing present
            cap = (max_attempts if empty_max_attempts is None
                   else empty_max_attempts)
            if attempts + 1 >= cap:
                return False, True, False         # accept-as-empty: mark done
            return False, False, False            # retry on the next resume
        return True, True, False                  # clean, rows to write
    if attempts + 1 >= max_attempts:
        return True, True, True
    return False, False, False


class CadcDatatrailSource(DataSource):
    info = PluginInfo(
        name="cadc-datatrail",
        kind="source",
        summary="CHIME/FRB baseband on CADC; survey to build, enumerate offline.",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("CADC proxy cert", "cadcdata", "cadcutils",
                  "datatrail CLI (survey only)"),
        needs_archive_config=True,
        notes="survey() walks the Datatrail scope(s) to build inventory.jsonl; "
              "enumerate() then reads it offline; fetch() uses cadcget with "
              f"{_cadc_transport.CADC_CONNECT_TIMEOUT_SECONDS:g}s connect and "
              f"{_cadc_transport.CADC_READ_TIMEOUT_SECONDS:g}s "
              "read-inactivity timeouts.",
    )

    def __init__(self) -> None:
        # One client per thread. fetch() and cadcinfo() may run concurrently in
        # the download workers or the survey verify pool, and a shared CADC
        # client is not guaranteed safe for concurrent calls.
        self._local = threading.local()
        # The optional exception families are resolved before any timed network
        # operation, so imports on the first failure do not use up its retry
        # and timeout budget.
        self._expected_errors = _cadc_transport.expected_errors()
        self._last_survey_completeness: Optional[dict[str, int]] = None

    def survey_completeness_issues(
            self, out_dir: str) -> Optional[dict[str, int]]:
        """Return the omission counts in the persisted state of the latest survey.

        Accepted-empty and definitive no-data events are explicit archive
        dispositions and are not counted as omissions. Strict mode fails on
        partial terminal events, contract refusals, and events still pending a
        later resumable pass, as after a ``--max-events`` stop.
        """
        del out_dir
        if self._last_survey_completeness is None:
            return None
        return dict(self._last_survey_completeness)

    def _get_client(self):
        cl = getattr(self._local, "client", None)
        if cl is None:
            cl = self._make_client()
            self._local.socket_timeout_fallback = not (
                _cadc_transport.configure_request_timeout(cl))
            self._local.client = cl
        return cl

    def _make_client(self, cert=None):
        from cadcdata import StorageInventoryClient
        from cadcutils import net
        cert = cert or _default_cert()
        subj = (net.Subject(certificate=cert)
                if cert and os.path.exists(cert) else net.Subject())
        service = storage_service_override()
        if service:
            return StorageInventoryClient(subj, resource_id=service)
        return StorageInventoryClient(subj)

    # Enumerate
    def _inventory_path(self, ctx: RunContext) -> str:
        o = ctx.options or {}
        p = o.get("inventory")
        if p:
            return p
        raw_name = o.get("name") or ctx.instrument.name
        try:
            name = validate_identifier(raw_name, label="inventory name")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        base = o.get("root")
        if base:
            return os.path.join(str(base), "data", name, "inventory.jsonl")
        from ..invpaths import resolve_inventory
        return str(resolve_inventory(name))

    def enumerate(self, ctx: RunContext) -> Iterable[Unit]:
        path = self._inventory_path(ctx)
        if not os.path.exists(path):
            raise SystemExit(
                f"inventory not found: {path}\n"
                "Build one with `pilot-proxy chime-survey` "
                "(or pass --inventory <path>).")
        if (Path(path).parent / STALE_VIEWS_MARKER).exists():
            raise SystemExit(
                f"inventory views in {Path(path).parent} may be stale after an active "
                "or interrupted survey; rerun the same survey command to refresh them")
        n_channels = getattr(ctx.instrument, "n_channels", None)
        sel = parse_selection(ctx.selection, n_channels=n_channels)
        seen, units = set(), []
        with open(path) as fh:
            for line_number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = parse_row(line, path, line_number)
                ch = r.get("freq_id")
                if ch is not None and n_channels is not None:
                    # The row parser has already checked that this is an
                    # integer, and the shared grammar checks the instrument's
                    # upper bound.
                    parse_freq_ids([ch], n_channels=n_channels)
                if not sel.wants_freq_id(ch):
                    continue
                if not sel.wants_event(r.get("event")):
                    continue
                # Each row carries the exact verified relative name written by
                # survey, and enumerate joins it to the common path. The name is
                # never derived again, so survey and read cannot disagree on it.
                name = r["name"]
                uri = join_uri(r["common_path"], name)
                logical_key = logical_unit_key(r["scope"], r["event"], name)
                if logical_key in seen:
                    continue
                seen.add(logical_key)
                # meta is the row without the URI parts, plus the stable
                # quarantine identity. Shape-specific columns, such as freq_id
                # for baseband or the columns a calibration shape wrote, are
                # passed through unchanged. The engine does not interpret these
                # columns, and an analyzer uses them to look up companion files,
                # such as gains by event.
                meta = {k: v for k, v in r.items()
                        if k not in ("common_path", "name")}
                if ch is not None:
                    meta["freq_id"] = int(ch)
                meta["size_bytes"] = int(r.get("size_bytes", 0))
                meta["cadc_uri"] = uri
                meta["quarantine_key"] = logical_key
                units.append(Unit(key=logical_key, name=name, meta=meta))
        return units

    # Fetch
    def fetch(self, unit: Unit, dest: str,
              retries: int = _DEFAULT_ARCHIVE_RETRIES,
              base: float = _INITIAL_ARCHIVE_BACKOFF):
        """Fetch one object with finite connect and read-inactivity timeouts."""
        client = self._get_client()
        delay, last = base, None
        expected = (unit.meta or {}).get("size_bytes")
        expected = (int(expected)
                    if (isinstance(expected, Integral)
                        and not isinstance(expected, bool) and expected > 0)
                    else None)

        def discard_partial():
            try:
                os.unlink(dest)
            except FileNotFoundError:
                pass

        for k in range(retries + 1):
            try:
                uri = unit.meta.get("cadc_uri", unit.key)
                if getattr(self._local, "socket_timeout_fallback", True):
                    # The socket default is process-wide, so this compatibility
                    # path is serialized and the default is always restored.
                    # Current CADC clients use the per-session timeout set in
                    # _get_client and stay fully concurrent.
                    with _cadc_transport.socket_timeout(
                            _cadc_transport.CADC_READ_TIMEOUT_SECONDS):
                        client.cadcget(uri, dest=dest)
                else:
                    client.cadcget(uri, dest=dest)
                actual = os.path.getsize(dest) if os.path.exists(dest) else 0
                if expected is not None and actual != expected:
                    last = (f"size mismatch: expected {expected} bytes, "
                            f"received {actual}")
                elif actual > 0:
                    return True, ""
                else:
                    last = "empty file"
            except self._expected_errors as exc:
                if "NotFound" in type(exc).__name__:
                    discard_partial()
                    return False, "NotFound"
                last = _cadc_transport.describe_exception(exc)
            if k < retries:
                discard_partial()
                time.sleep(delay)
                delay *= _RETRY_BACKOFF_MULTIPLIER
        discard_partial()
        return False, str(last)[:_ERROR_SUMMARY_LENGTH]

    # Survey
    # The per-product file shape, which gives the files one event contributes
    # and their names, is defined on the reader (Reader.survey_files). Survey
    # and read then share one naming definition and cannot disagree. survey()
    # uses ctx.reader and falls back to the chime-baseband reader's shape.
    def _cadc_size(self, uri, retries: int = _DEFAULT_ARCHIVE_RETRIES,
                   base: float = _INITIAL_ARCHIVE_BACKOFF, deadline=None):
        delay, last = base, None
        for k in range(retries + 1):
            try:
                client = self._get_client()
                with _cadc_transport.request_deadline(deadline):
                    if getattr(self._local, "socket_timeout_fallback", True):
                        with _cadc_transport.socket_timeout(
                                _cadc_transport.CADC_READ_TIMEOUT_SECONDS,
                                deadline=deadline):
                            info = client.cadcinfo(uri)
                    else:
                        info = client.cadcinfo(uri)
                return info.size, None
            except self._expected_errors as exc:
                if "NotFound" in type(exc).__name__:
                    return None, None                       # definitive: absent
                last = exc
                if k < retries:
                    sleep_for = delay
                    if deadline is not None:
                        remaining = float(deadline) - time.monotonic()
                        if remaining <= 0:
                            return None, last
                        sleep_for = min(sleep_for, remaining)
                    time.sleep(sleep_for)
                    delay *= _RETRY_BACKOFF_MULTIPLIER
        return None, last

    @with_survey_output_lock
    def survey(self, ctx: RunContext, out_dir: str) -> str:
        """Build inventory.jsonl for the selected scope(s) and freq_ids.

        The survey runs in two resumable, incremental phases. First, the unique
        events under each scope are enumerated and cached. Following this, the
        Common Path of every event not yet done is resolved, each requested
        freq_id is checked with cadcinfo, and the verified rows are written
        atomically per event. A re-run surveys the remaining cached events and
        skips the completed ones. Newly registered events are discovered with
        ``--re-enumerate``, and an already completed event is probed again only
        in a fresh inventory.
        """
        o = ctx.options or {}
        scope_opt = o.get("scope")
        named = (tuple(s.strip() for s in
                       (scope_opt.split(",") if isinstance(scope_opt, str) else scope_opt)
                       if str(s).strip())
                 if scope_opt else None)
        if o.get("scopes_only"):                 # recon: recursive `datatrail ls`
            label = o.get("name")
            return recon(named, match_terms(o.get("match")), out_dir,
                         expand=bool(o.get("expand", False)),
                         telescope=getattr(ctx.instrument, "name", None),
                         map_name=(f"scopes-{label}.jsonl" if label
                                   else "scopes.jsonl"))
        inst_scopes = tuple(getattr(ctx.instrument, "scopes", ()) or ())
        scopes = named or inst_scopes or _DEFAULT_SCOPES
        n_ch = (ctx.instrument.n_channels
                if ctx.instrument is not None else None)
        freq_ids = _resolve_freq_ids(o.get("freq_ids", ctx.selection), n_ch)
        include_outrigger = bool(o.get("include_outrigger", False))
        workers = max(1, int(o.get("workers", _DEFAULT_SURVEY_WORKERS)
                             or _DEFAULT_SURVEY_WORKERS))
        max_events = o.get("max_events")
        re_enumerate = bool(o.get("re_enumerate", False))
        eag = o.get("empty_age_days")
        empty_age_days = (_EMPTY_TERMINAL_AGE_DAYS if eag is None else int(eag))
        if empty_age_days < 0:
            raise SystemExit("--empty-age-days must be zero or greater")

        # The reader defines the archive file shape, the files one event
        # contributes and their names (Reader.survey_files). The CLI sets the
        # run's reader on ctx.reader, and a caller that supplies none gets the
        # chime-baseband shape.
        shape = ctx.reader if ctx.reader is not None else _default_shape()
        reader_name = getattr(getattr(shape, "info", None), "name",
                              type(shape).__name__)
        survey_schema = getattr(shape, "survey_schema", None)
        if (isinstance(survey_schema, bool)
                or not isinstance(survey_schema, int)
                or survey_schema < 1):
            raise SystemExit(
                f"reader {reader_name!r} must declare a positive integer "
                "survey_schema for archive surveys; bump it whenever "
                "survey_files(), annotate_row(), or inventory row semantics "
                "change")
        minimum_spec = getattr(
            shape, "minimum_archive_bytes", _DEFAULT_MINIMUM_ARCHIVE_BYTES)
        minimum_bytes = int(minimum_spec(ctx) if callable(minimum_spec)
                            else minimum_spec)
        if minimum_bytes < 1:
            raise SystemExit(
                f"reader {reader_name!r} minimum_archive_bytes must be >= 1")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ensure_manifest(
            out,
            build_configuration(ctx, scopes, freq_ids, shape,
                                include_outrigger, empty_age_days,
                                minimum_bytes, survey_schema),
        )
        with SurveyRun(out) as execution:
            inv_path = out / "inventory.jsonl"
            fid_note = (f" freq_ids={len(freq_ids)} ({freq_ids[0]}..{freq_ids[-1]})"
                        if freq_ids else "")
            print(f"[survey] scopes={list(scopes)}"
                  f" shape={reader_name} minimum_bytes={minimum_bytes}"
                  f"{fid_note} -> {inv_path}", flush=True)

            # Phase 1: enumeration of the unique events (cached)
            membership = _enumerate_events(scopes, include_outrigger,
                                           out / "enum_cache.json", re_enumerate)
            events = sorted(ev for ev, lbls in membership.items()
                            if include_outrigger
                            or not any(_OUTRIGGER_RE.search(x) for x in lbls))
            print(f"to survey: {len(events)} events", flush=True)

            # Phase 2: verification of each event's freq_ids (resumable)
            attempts_path = out / "attempts.json"
            attempts = load_attempts(attempts_path)
            pending_path = out / "pending_events.json"
            pending = _load_json_file(pending_path, {})
            if not isinstance(pending, dict):
                raise SystemExit(f"invalid pending survey state: {pending_path}")
            store = SurveyStore(out / "survey_state.sqlite3")
            surveyed = store.completed_keys()
            # A crash can occur after the event transaction and before its
            # attempts checkpoint. The committed event takes precedence, and the
            # harmless stale count is discarded.
            for key in surveyed:
                attempts.pop(key, None)
                pending.pop(key, None)
            atomic_write_json(attempts_path, attempts)
            atomic_write_json(pending_path, pending)
            # The JSONL and text views are rebuilt in case a previous process
            # stopped after a SQLite commit and before its periodic or final
            # render.
            store.render_views(out)
            print(f"resume: {len(surveyed)} events already done", flush=True)

            # The accept-as-empty ledger holds one JSON object per written-off
            # event, so each write-off can be audited (when, how many sightings,
            # why) and found with grep.
            no_files_path = out / "no_files_events.jsonl"
            pool = ThreadPoolExecutor(max_workers=workers)
            n_new = 0
            n_committed = 0
            # run-level accounting so a 0-row inventory can never read as success
            n_rows = n_no_data = n_incomplete = n_empty_retry = n_empty_accepted = 0
            n_refused = 0

            def mark_done(key, scope, ev, status, records=(), *, incomplete=None,
                          no_files=None):
                nonlocal n_committed
                store.commit(key, scope, ev, status, records,
                             incomplete=incomplete, no_files=no_files)
                surveyed.add(key)
                attempts.pop(key, None)
                pending.pop(key, None)
                atomic_write_json(attempts_path, attempts)
                atomic_write_json(pending_path, pending)
                n_committed += 1
                if n_committed % VIEW_FLUSH_INTERVAL == 0:
                    store.render_views(out)

            def bump(key):
                attempts[key] = attempts.get(key, 0) + 1
                atomic_write_json(attempts_path, attempts)

            def verify(scope, ev, deadline):
                try:
                    cp, ps_ok = DATATRAIL.common_path(
                        scope, ev, deadline=deadline)
                except DatatrailContractError as exc:
                    # The service answered with a payload this adapter cannot
                    # act on. The refusal is deterministic, so a retry cannot
                    # change it. It must never be treated as an outage, which
                    # would stall the whole survey on one event and then abort
                    # with a misleading certificate message.
                    return "refused", [], [], 0, "unknown", None, str(exc), {}
                if not ps_ok:
                    return "service_down", [], [], 0, "unknown", None, None, {}
                if not cp:
                    return "no_data", [], [], 0, "unknown", None, None, {}
                obs_date = (lambda m: f"{m[1]}-{m[2]}-{m[3]}" if m else "unknown")(
                    _DATE_RE.search(cp))
                labels = membership[(scope, ev)]
                # The reader's shape gives the candidate files of one event as
                # (name, fields) pairs. Baseband yields one per freq_id, and a
                # per-event product may yield a single file with its own fields.
                cand = [candidate_file(item, reader_name)
                        for item in shape.survey_files(ev, cp, freq_ids, ctx)]
                candidate_names = [name for name, _fields in cand]
                if len(candidate_names) != len(set(candidate_names)):
                    raise SystemExit(
                        f"reader {reader_name!r} yielded duplicate archive names "
                        f"for event {ev!r}")

                def probe(item):
                    name, fields = item
                    size, err = self._cadc_size(
                        join_uri(cp, name), deadline=deadline)
                    return name, fields, size, err

                records, errored, sub_floor = [], [], []
                for name, fields, size, err in pool.map(probe, cand):
                    if err is not None:
                        errored.append(name)
                    elif size is not None and size >= minimum_bytes:
                        # The row is self-describing. ``name`` is the file that
                        # enumerate and fetch stage (joined to common_path), and
                        # the shape's per-file fields are written verbatim as
                        # columns.
                        rec = {
                            "scope": str(scope), "event": str(ev), "name": name,
                            "size_bytes": int(size),
                            "common_path": str(cp), "obs_date": obs_date,
                            "datasets": list(labels),
                            "collection_restored": bool(getattr(cp, "collection_restored", False)),
                        }
                        rec.update(fields)
                        annotate_row(shape, rec, ctx.instrument)
                        records.append(rec)
                    elif size is not None:
                        sub_floor.append({"name": name, "size_bytes": int(size)})
                details = {
                    "collection_restored": bool(getattr(cp, "collection_restored", False)),
                    "minimum_archive_bytes": minimum_bytes,
                    "sub_floor_files": sub_floor,
                    "n_sub_floor": len(sub_floor),
                    "n_absent": len(cand) - len(records) - len(errored) - len(sub_floor),
                }
                if errored and len(errored) == len(cand):       # all probes errored, treated as an outage
                    return "service_down", records, errored, len(cand), obs_date, cp, None, details
                return "progress", records, errored, len(cand), obs_date, cp, None, details

            try:
                for i, (scope, ev) in enumerate(events, 1):
                    key = f"{scope}|{ev}"
                    if key in surveyed:
                        continue
                    if max_events is not None and n_new >= int(max_events):
                        print(f"reached --max-events={max_events}; stopping "
                              f"(resumable).", flush=True)
                        break

                    # A transient outage is waited out on the same event. Only a
                    # sustained outage aborts the survey, since it is the
                    # signature of an expired cert, which does not recover.
                    backoff = _INITIAL_SERVICE_BACKOFF
                    outage_started = _monotonic()
                    outage_deadline = outage_started + max(
                        0.0, float(_MAX_SERVICE_WAIT))
                    attempted = False

                    def unavailable(elapsed):
                        return SurveyUnavailableError(
                            "Datatrail/CADC remained unreachable for "
                            f"{elapsed:g}s. "
                            f"Partial survey state was preserved in {out}. Renew "
                            "the certificate with `cadc-get-cert -u <user>` (or "
                            "wait for the service to recover), then rerun the same "
                            "survey command."
                        )

                    while True:
                        if attempted:
                            now = _monotonic()
                            if now >= outage_deadline:
                                raise unavailable(max(0.0, now - outage_started))
                        (status, records, errored, n_cand,
                         obs_date, cp, refusal, details) = verify(scope, ev, outage_deadline)
                        attempted = True
                        if status != "service_down":
                            break
                        now = _monotonic()
                        elapsed = max(0.0, now - outage_started)
                        remaining = max(0.0, outage_deadline - now)
                        if remaining <= 0:
                            raise unavailable(elapsed)
                        sleep_for = min(float(backoff), remaining)
                        print(f"[{i}/{len(events)}] {ev}: service unreachable -- "
                              f"waiting {sleep_for:g}s "
                              f"(elapsed {elapsed:g}s)", flush=True)
                        _sleep(sleep_for)
                        backoff = min(
                            backoff * _RETRY_BACKOFF_MULTIPLIER,
                            _MAX_SERVICE_BACKOFF,
                        )

                    n_new += 1
                    if status == "refused":
                        # The event is committed as done, so resume skips it,
                        # and the reason is written to the accept-as-empty
                        # ledger, where the write-off can be audited and found
                        # with grep. If the named URI is a legitimate new
                        # collection, widen _MINOC_COLLECTIONS in
                        # datatrail_client.py and re-open these rows (DELETE FROM
                        # events WHERE status='refused' in survey_state.sqlite3)
                        # before re-running.
                        n_refused += 1
                        ledger = {
                            "ts": datetime.datetime.now(datetime.timezone.utc)
                                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "scope": scope, "event": ev,
                            "n_expected": 0,
                            "attempts": attempts.get(key, 0) + 1,
                            "obs_date": "unknown", "age_days": None,
                            "common_path": None,
                            "reason": "datatrail-contract-refusal",
                            "detail": refusal,
                        }
                        print(f"[{i}/{len(events)}] {ev}: datatrail contract "
                              f"refusal -- recorded and skipped: {refusal}",
                              flush=True)
                        mark_done(key, scope, ev, "refused", no_files=ledger)
                        continue
                    if status == "no_data":
                        n_no_data += 1
                        mark_done(key, scope, ev, "no-data")
                        continue
                    if n_cand == 0:
                        mark_done(key, scope, ev, "empty-selection", no_files={
                            "scope": scope, "event": ev, "n_expected": 0,
                            "common_path": str(cp), "reason": "no-selected-candidates",
                            "attempts": attempts.get(key, 0) + 1,
                            **details,
                        })
                        print(f"[{i}/{len(events)}] {ev}: reader selection produced "
                              "no candidate files; no archive absence inferred", flush=True)
                        continue

                    # An event is empty when a common path resolved but every
                    # requested freq_id came back absent or below the byte floor,
                    # with 0 rows and 0 hard errors. Such an event is not a clean,
                    # fully resolved event and must never be written out as a
                    # silent 0-row done. Absence here is already definitive for
                    # each probe, since _cadc_size returns NotFound as an answer.
                    # Outages take the service_down path, and hard errors take
                    # the incomplete path. The only transient case it could mask
                    # is archive-side replication lag, which can affect only a
                    # recent observation. Thus, an empty event whose
                    # obs_date is at least empty_age_days old is accepted on its
                    # first sighting. A younger or undatable event stays pending
                    # and is retried across resumes until _MAX_ATTEMPTS. In both
                    # cases the acceptance is recorded in no_files_events.jsonl,
                    # so a 0-row event can neither vanish without notice nor be
                    # probed again forever.
                    empty = not records and not errored
                    sightings = attempts.get(key, 0) + 1    # counts this run's probe
                    if empty and details["collection_restored"] and not details["n_sub_floor"]:
                        # A guessed prefix with no independently found object
                        # gives no evidence of archive absence, at any age or
                        # repeat count. The event stays resumable, and the reason
                        # is recorded.
                        pending[key] = {
                            "scope": scope, "event": ev, "common_path": str(cp),
                            "reason": "unverified-restored-collection",
                            "attempts": sightings, "n_expected": n_cand, **details,
                        }
                        atomic_write_json(pending_path, pending)
                        bump(key)
                        n_empty_retry += 1
                        print(f"[{i}/{len(events)}] {ev}: restored collection "
                              "has no verified object; remains pending", flush=True)
                        continue
                    age_days = _obs_age_days(obs_date) if empty else None
                    aged_out = (empty and age_days is not None
                                and age_days >= empty_age_days)
                    write_recs, done, incomplete = _commit_decision(
                        len(errored), attempts.get(key, 0), len(records),
                        empty_max_attempts=(1 if aged_out else None))
                    if write_recs:
                        n_rows += len(records)
                    if incomplete:
                        n_incomplete += 1
                    if done:
                        no_files_record = None
                        if empty:                  # terminal 0-file event, written to the ledger
                            no_files_record = {
                                "ts": datetime.datetime.now(datetime.timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "scope": scope, "event": ev,
                                "n_expected": n_cand,
                                "attempts": sightings,
                                "obs_date": obs_date,
                                "age_days": age_days,
                                "common_path": cp,
                                "reason": "below-reader-floor" if details["n_sub_floor"]
                                          else "aged-out" if aged_out else "max-attempts",
                                **details,
                            }
                            n_empty_accepted += 1
                        mark_done(
                            key, scope, ev,
                            "empty" if empty else
                            "incomplete" if incomplete else "complete",
                            records if write_recs else (),
                            incomplete=errored if incomplete else None,
                            no_files=no_files_record,
                        )
                    else:
                        bump(key)
                        if empty:
                            n_empty_retry += 1

                    # The empty case is printed too. Without it, a run of empty
                    # events prints nothing and looks like it did nothing at all
                    # (0 events shown, 0 rows on disk).
                    if (records or errored or empty
                            or i % _EVENT_PROGRESS_INTERVAL == 0):
                        if empty and done:
                            why = (f"obs {obs_date}, {age_days}d old" if aged_out
                                   else f"attempt {sightings}/{_MAX_ATTEMPTS}")
                            cause = ("objects below reader byte floor" if details["n_sub_floor"]
                                     else "not in CADC storage")
                            tag = (f" -- {cause}; accepting as empty "
                                   f"({why})")
                        elif empty:
                            cause = ("objects below reader byte floor" if details["n_sub_floor"]
                                     else "not in CADC storage")
                            tag = (f" -- {cause}; re-checking in case "
                                   f"transient ({sightings}/{_MAX_ATTEMPTS})")
                        else:
                            tag = (f" INCOMPLETE({len(errored)})" if incomplete
                                   else f" ({len(errored)} unresolved, retry)"
                                   if errored else "")
                        print(f"[{i}/{len(events)}] {ev}: "
                              f"{len(records)}/{n_cand} files{tag}", flush=True)
            finally:
                # Work that has not started is cancelled, and active probes are
                # joined before the output lock is released or an error is
                # returned. Supported CADC calls carry finite per-request
                # timeouts and the outage deadline, so shutdown is bounded by the
                # remaining in-flight requests.
                pool.shutdown(wait=True, cancel_futures=True)
                atomic_write_json(attempts_path, attempts)
                try:
                    store.render_views(out)
                    execution.views_current()
                finally:
                    try:
                        status_counts = store.status_counts()
                        current_event_keys = {
                            f"{scope}|{event}" for scope, event in events
                        }
                        self._last_survey_completeness = {
                            "incomplete": (status_counts.get("incomplete", 0)
                                           + status_counts.get("empty-selection", 0)),
                            "refused": status_counts.get("refused", 0),
                            "pending": len(current_event_keys - surveyed),
                        }
                        execution.record["completeness"] = dict(self._last_survey_completeness)
                    finally:
                        store.close()

            # The row count is the final report of the run, so an empty
            # inventory is never hidden behind "survey wrote <path>" when 0 rows
            # were written.
            total_rows = (sum(1 for ln in open(inv_path) if ln.strip())
                          if inv_path.exists() else 0)
            print(f"\nsurvey: {n_new} events this run -- {n_rows} rows written, "
                  f"{n_no_data} no-data, {n_empty_accepted} accepted-empty, "
                  f"{n_empty_retry} resolved-but-empty (retry next run), "
                  f"{n_incomplete} incomplete, {n_refused} contract-refused",
                  flush=True)
            if n_empty_accepted or n_refused:
                print(f"accepted-empty ledger: {no_files_path}", flush=True)
            if n_refused:
                print("contract refusals name the offending replica URI in the "
                      "ledger; if it is a legitimate new collection, widen "
                      "_MINOC_COLLECTIONS in pilot_proxy.archive.datatrail_client "
                      "re-open refused rows before re-running.", flush=True)
            if total_rows == 0:
                print(
                    "[warn] inventory.jsonl is EMPTY (0 rows). Every surveyed event "
                    "resolved to zero retrievable files, so nothing was written -- "
                    "usually the environment, not the survey. Sanity-check one event: "
                    "`datatrail ps <scope> <event> -s` (is a 'Common Path:' line "
                    "printed?), then `cadcinfo --cert ~/.ssl/cadcproxy.pem <cadc-uri>` "
                    "for one freq_id (NotFound = the bytes aged off storage, or a size "
                    f"under this reader's {minimum_bytes}-byte floor; pass the cert or "
                    "the CLI runs anonymously and "
                    "reports a misleading 'Unauthorized'). The lowest event IDs are the "
                    "likeliest to have aged out of the archive, so a larger "
                    "--max-events often starts filling the inventory.", flush=True)
            print(f"survey wrote {inv_path}", flush=True)
            return str(inv_path)

    # Doctor
    def fetch_preflight(self, ctx: RunContext) -> tuple[bool, list[str], list[str]]:
        """Check only the dependencies needed to read an inventory and fetch."""
        del ctx
        problems: list = []
        cert = _default_cert()
        if not os.path.exists(cert):
            problems.append(
                f"no CADC proxy cert at {cert} "
                f"(run `cadc-get-cert -u <user>` or set CADC_CERT)")
        try:
            import cadcdata  # noqa: F401
            import cadcutils  # noqa: F401
        except Exception:
            problems.append("cadcdata/cadcutils not importable "
                            "(pip install -e \".[archive]\")")
        return (not problems), problems, []

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str], list[str]]:
        _ok, problems, notes = self.fetch_preflight(ctx)

        # Survey prerequisites (datatrail)
        # scan and fetch never call datatrail. These checks apply to survey only
        # and are skipped when datatrail is absent, since ``requires`` already
        # lists it. All datatrail calls go through
        # pilot_proxy.archive.datatrail_client.
        if Datatrail.installed():
            # (a) The machine-readable CLI contract that survey uses
            # (``datatrail ls/ps --json``, datatrail-cli >= 0.11). A pre-0.11
            # install has no --json flag. The live call would then be read as a
            # service outage and stall the survey for about an hour before a
            # misleading 'expired cert' abort, so the real cause is reported
            # here before the run.
            ok, detail = Datatrail.api_available()
            if not ok:
                problems.append(
                    "datatrail is installed but the machine-readable CLI "
                    f"contract survey drives is unavailable: {detail}. survey "
                    "uses `datatrail ls/ps --json` (datatrail-cli>=0.11); "
                    "upgrade datatrail-cli (scan/fetch are unaffected).")

            # (b) The scope(s) survey will walk are checked against datatrail's
            # live namespace, so a stale or renamed scope fails here, before
            # survey can walk it into an empty inventory. datatrail defines which
            # scopes exist, and the instrument YAML defines which ones to walk.
            o = ctx.options or {}
            scope_opt = o.get("scope")
            effective = (tuple(s.strip() for s in
                              (scope_opt.split(",") if isinstance(scope_opt, str)
                               else scope_opt) if str(s).strip())
                         if scope_opt
                         else tuple(getattr(ctx.instrument, "scopes", ()) or ()))
            if effective:
                try:
                    known = set(DATATRAIL.list_scopes())
                except Exception:
                    known = set()    # datatrail unreachable, so no validation
                if known:            # empty means a transient or auth failure
                    missing = [s for s in effective if s not in known]
                    if missing:
                        problems.append(
                            f"scope(s) not found in datatrail: {missing} -- survey "
                            "would walk nothing for them. Fix the instrument YAML "
                            "`scopes:` or --scope (`datatrail ls` lists valid "
                            "scopes).")
                else:
                    # The scopes could not be listed (datatrail down or not
                    # installed), so they are not reported as invalid. A visible,
                    # non-fatal 'skipped' note is added, so doctor's READY does
                    # not hide a check that never ran.
                    notes.append(
                        "datatrail scope(s) not validated: could not reach "
                        "datatrail to list scopes -- survey will still attempt "
                        f"{list(effective)} (re-run doctor once datatrail responds)")

        return (not problems), problems, notes
