"""Define the source, reader, and analyzer roles in an archive run.

For each discovered unit, the engine stages a file, reads its frames, passes
them to an analyzer, deletes the staged copy, and checkpoints the result.
Three interfaces supply the parts that depend on the telescope or analysis:

* ``DataSource`` locates and stages the data. The reference source uses CADC
  and the CHIME/FRB Datatrail archive.
* ``Reader`` turns a staged file into arrays and per-unit metadata. It also
  defines which files belong to an event and how they are named through
  ``survey_files``, so discovery and reading use the same file layout.
  The reference reader handles CHIME 4+4-bit baseband HDF5.
* ``Analyzer`` consumes those arrays in one streaming pass and accumulates a
  product that can be checkpointed and saved. The averaged power-spectrum
  analyzer is one example.

Instrument geometry is supplied separately as YAML under ``instruments/``.
Plugins describe their capabilities through ``describe()`` so ``list`` and
``doctor`` can report which combinations are available and ready to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

# discovery metadata
READY = "ready"            # implemented and validated for production use
EXPERIMENTAL = "experimental"  # works, but not yet trusted for science
STUB = "stub"             # interface only, with TODOs, and raises if run

_STATUS_ORDER = {READY: 0, EXPERIMENTAL: 1, STUB: 2}


# Semantic types for the items that a Reader yields to an Analyzer. They
# describe the in-memory stream at the boundary between reader and analyzer,
# and are unrelated to file names and stored product schemas. A new kind must
# be specific enough that two plugins that declare the same value can be
# composed without guessing.
STREAM_COMPLEX_BASEBAND = "complex-baseband-frame"
STREAM_VISIBILITY_CHUNK = "visibility-chunk"
STREAM_ANY = "*"          # analyzer only, accepts every stream kind


class SurveyUnavailableError(RuntimeError):
    """A survey stopped cleanly because an external service stayed unavailable.

    Sources raise this after preserving their resumable state. The CLI turns it
    into a nonzero exit without presenting the partial survey as complete.
    """


class UnreadableUnitError(RuntimeError):
    """A reader found that the bytes or schema of this unit are unusable.

    The engine may quarantine a unit only when this error is raised.
    Dependency, environment, and reader-programming exceptions must remain
    run-level failures so good data is never permanently excluded by mistake.
    """


@dataclass(frozen=True)
class PluginInfo:
    """One row in the discovery tables (`list` / `doctor`)."""
    name: str
    kind: str                      # "source" | "reader" | "analyzer"
    summary: str
    status: str = READY            # READY | EXPERIMENTAL | STUB
    instruments: tuple[str, ...] = ()   # telescopes this plugin is known to fit ("*" = any)
    produces: str = ""             # such as "<freq_id>.npz" or "n2_<scope>.npz"
    requires: tuple[str, ...] = () # human-readable prerequisites (env, creds, deps)
    notes: str = ""
    # Sources only. True if this source pulls from the CADC archive. The doctor
    # ready combinations then expect the telescope to declare a default
    # baseband `scopes` list in YAML. A geometry-only telescope still
    # works when --scope is passed. A "local" source leaves this False, so a
    # geometry-only telescope can be used with it before any archive scope is
    # configured.
    needs_archive_config: bool = False
    # Reader/analyzer stream contract. A Reader declares the one semantic kind
    # yielded by iter_arrays(), and an Analyzer declares every kind it accepts.
    # Empty declarations are invalid for an executable reader/analyzer pair.
    stream_kind: str = ""
    accepts_stream_kinds: tuple[str, ...] = ()

    @property
    def status_rank(self) -> int:
        return _STATUS_ORDER.get(self.status, 99)


@dataclass(frozen=True)
class StreamCompatibility:
    """Result of the declared stream contract between one reader and one analyzer."""

    reader_name: str
    analyzer_name: str
    compatible: bool
    reader_kind: str
    accepted_kinds: tuple[str, ...]

    @property
    def detail(self) -> str:
        """Human-readable explanation suitable for CLI diagnostics."""
        if self.compatible is True:
            if STREAM_ANY in self.accepted_kinds:
                return (f"analyzer {self.analyzer_name!r} explicitly accepts "
                        "every stream kind")
            return (f"reader {self.reader_name!r} emits {self.reader_kind!r}; "
                    f"analyzer {self.analyzer_name!r} accepts it")
        if not self.reader_kind or not self.accepted_kinds:
            missing = []
            if not self.reader_kind:
                missing.append(f"reader {self.reader_name!r} stream_kind")
            if not self.accepted_kinds:
                missing.append(
                    f"analyzer {self.analyzer_name!r} accepts_stream_kinds")
            return ("stream contract is incomplete (missing " +
                    " and ".join(missing) + ")")
        if self.compatible is False:
            return (f"reader {self.reader_name!r} emits {self.reader_kind!r}, but "
                    f"analyzer {self.analyzer_name!r} accepts only "
                    f"{list(self.accepted_kinds)!r}")
        raise AssertionError("unreachable stream compatibility state")


def stream_compatibility(reader_info: PluginInfo,
                         analyzer_info: PluginInfo) -> StreamCompatibility:
    """Return the declared compatibility of a Reader and Analyzer.

    Only the declared metadata are compared, and no array shapes are sampled
    or inferred at runtime. Missing and mismatched declarations are both
    rejected before any data is staged.
    """

    raw_reader_kind = getattr(reader_info, "stream_kind", "")
    reader_kind = (raw_reader_kind.strip()
                   if isinstance(raw_reader_kind, str) else "")
    raw_accepted = getattr(analyzer_info, "accepts_stream_kinds", ())
    accepted = (
        tuple(kind.strip() for kind in raw_accepted
              if isinstance(kind, str) and kind.strip())
        if isinstance(raw_accepted, tuple) else ()
    )
    if not reader_kind or not accepted:
        compatible = False
    elif STREAM_ANY in accepted:
        compatible = True
    else:
        compatible = reader_kind in accepted
    return StreamCompatibility(
        reader_name=str(getattr(reader_info, "name", "reader")),
        analyzer_name=str(getattr(analyzer_info, "name", "analyzer")),
        compatible=compatible,
        reader_kind=reader_kind,
        accepted_kinds=accepted,
    )


# unit of work and the streaming context
@dataclass(frozen=True)
class Unit:
    """One item (typically one file) that the engine stages and processes.

    `key`   source-namespaced logical identity used for resume. It must remain
            stable if a physical archive URI or storage location changes. A
            source must not list duplicate logical units in one enumeration.
    `name`  local filename to stage it under.
    `meta`  source-specific fields a reader/analyzer may need (event id, freq_id,
            obs date, size, ...). Opaque to the engine.
    """
    key: str
    name: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Context shared by the reader and analyzer of a run, treated as read-only."""
    instrument: Any                      # archive instrument definition
    selection: Any = None                # what the user asked to process (freq_ids,
                                         # events, scope, ...), parsed by
                                         # archive.selection
    options: Mapping[str, Any] = field(default_factory=dict)
    reader: Any = None                   # the run's Reader instance, when the caller
                                         # has resolved one. survey() consults it for
                                         # the archive file shape (Reader.survey_files)
                                         # so survey and read share one definition of
                                         # what a unit's file is named.


# discovery and staging (DataSource)
class DataSource:
    """Where the data lives and how to enumerate and stage it.

    The two methods are split so that the cheap discovery step can be cached
    and run again without reading bulk data:

      enumerate()  -> iterable of Unit, the inventory for a given selection.
      fetch(unit)  -> stage one unit to a local path and return (ok, error).

    Implementations should make `fetch` retry transient failures, use finite
    network connect and read timeouts, and return exactly
    ``(False, detail: str)`` for an expected retryable or file failure.
    Exceptions and malformed returns are run-level source failures. A fetch
    must never require more than one staged file to exist at a time, since the
    engine deletes each file right after it is analyzed.

    With the default engine settings (download_workers=1), fetch is called
    serially in enumerate() order. With multiple workers and staging slots, the
    engine may call fetch() on one source instance from several threads at
    once. A source that supports parallel fetch must therefore keep no mutable
    per-call state on self. Each thread can be given its own client, such as
    through threading.local, or a shared client can be guarded with a lock.
    """
    info: PluginInfo

    def enumerate(self, ctx: RunContext) -> Iterable[Unit]:
        raise NotImplementedError

    def fetch(self, unit: Unit, dest: str) -> tuple[bool, str]:
        raise NotImplementedError

    # Optional one-shot survey that writes a persistent inventory file. Sources
    # whose enumerate() is expensive (a network listing) override it so that
    # `survey` can cache the listing to disk. Cheap sources keep this default
    # and enumerate on demand.
    def survey(self, ctx: RunContext, out_dir: str) -> str:
        raise NotImplementedError(
            f"{self.info.name}: survey-to-disk not implemented; this source "
            f"enumerates on demand.")

    def survey_completeness_issues(
            self, out_dir: str) -> Optional[Mapping[str, int]]:
        """Return final-state omission counts after ``survey()``, if supported.

        ``None`` means the source has no completeness contract. A mapping with
        nonzero counts lets the CLI's opt-in strict mode preserve the inventory
        and metadata while returning a failure status for automation.
        """
        return None

    # Optional self-check for `doctor`. It returns (ok, problems), where any
    # problem makes doctor report NOT READY, or (ok, problems, notes). Each note
    # is a non-fatal check that could not run, and doctor prints it as a [--]
    # skipped line without failing readiness.
    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []


# file to arrays and metadata (Reader)
class Reader:
    """Turn one staged file into the stream items an analyzer consumes.

    `probe(path)`            -> dict of per-file metadata (such as the channel
                                center frequency, shape, and sample rate), read
                                without the bulk data.
    `iter_arrays(path, ctx)` -> items in streaming order. Most readers yield
                                NumPy-compatible arrays, and structured formats
                                may yield mappings when their declared stream
                                contract requires one.

    The reader holds the knowledge of the on-disk format (dataset names, dtype
    packing, and attribute conventions). A different file format needs a
    different reader, and the engine and analyzer do not change.
    """
    info: PluginInfo
    # Archive sources use this to decide if a remotely reported object is large
    # enough to be a usable file of this format. It is a positive byte count or
    # a callable that takes a RunContext. The streaming engine has no minimum
    # file size.
    minimum_archive_bytes: int = 1
    # Archive-capable readers must set a positive schema revision, and increment
    # it whenever survey_files() or annotate_row() changes the meaning of stored
    # rows. A survey resume refuses state created under a different revision.
    survey_schema: Optional[int] = None

    def probe(self, path: str) -> Mapping[str, Any]:
        raise NotImplementedError

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        raise NotImplementedError

    # archive file shape (optional)
    # An archive survey needs to know which files of one event this reader's
    # product uses, and their names. This is format knowledge, so it is kept on
    # the reader, the same class that later opens the files, and survey and
    # read always agree on naming. A reader that only scans pre-listed local
    # files can leave both methods as they are.
    def survey_files(self, event, common_path, selection,
                     ctx: RunContext) -> Iterable[tuple]:
        """Yield (filename, fields) for every candidate file of one event.

        `filename` is relative to the archive common path of the event and may
        contain a sub-path. `fields` is a mapping of the per-file inventory
        columns defined by this format. The baseband reader yields
        ({"freq_id": ch}) for each channel, and a per-event calibration product
        might yield a single file with no fields. `selection` is the per-survey
        spec resolved by the source (the baseband survey passes the freq_id
        list), and a file shape that is not selected this way ignores it.
        Everything yielded here is written unchanged to the inventory row, and
        enumerate and fetch later stage the row's `name` as written.
        """
        raise NotImplementedError(
            f"reader {getattr(self.info, 'name', type(self).__name__)!r} does "
            f"not declare an archive file shape (survey_files); an archive "
            f"survey needs a reader that does. See docs/ADDING_A_READER.md.")

    def annotate_row(self, row: dict, instrument) -> None:
        """Add fields to one verified inventory row in place (optional).

        Survey calls this after the file size is known, with the run's
        instrument, which may be None. The baseband reader adds ``freq_mhz``
        and an ``n_frames_estimate`` derived from the file size. The default
        adds nothing.
        """
        return None

    def survey_fingerprint(self, ctx: RunContext) -> Any:
        """Return extra JSON-safe inputs that determine survey inventory rows.

        Archive survey state uses this value to reject an incompatible resume.
        Readers whose row shape depends only on their class, selection,
        instrument, and ``minimum_archive_bytes`` can keep the default.
        """
        return None

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []


# streaming accumulation into a saved product (Analyzer)
class Analyzer:
    """Accumulate a science product over a stream of arrays in one pass.

    The analyzer owns its product file (the saved .npz, .h5, ...). It writes
    the file, loads it again on resume, and reports which units are already in
    it. Since the analyzer owns the file, it can keep a specific product schema
    that downstream tools read directly, and the product is never wrapped in an
    opaque engine checkpoint. The self-describing <freq_id>.npz of the spectrum
    analyzer is one such product.

    The engine drives this lifecycle:

        resume(path, ctx)          -> bool : load an existing product if present
                                             and compatible; True if it resumed
        processed_keys()           -> set  : Unit.key values already in the product
        begin(ctx, first_meta)             : once, when the first new file is read
        consume_file(arrays, meta) -> n    : per file; update accumulators
        save(path)                         : persist the product, provenance, and
                                             processed keys for recovery
        summary()                  -> dict : small human-readable status line

    All accumulator writes are made on the main thread of the engine, which
    parallelizes only fetch, so an analyzer needs no locking.

    `requires_in_order` is set to True when the product depends on the
    consumption order of the files, such as a running or trailing statistic, a
    CFAR baseline, or any accumulation that does not commute. The engine
    buffers early fetch completions and gives these analyzers the exact source
    order at any supported worker and staging count. The default, False, is for
    a commutative accumulation such as a summed PSD.
    """
    info: PluginInfo
    requires_in_order: bool = False

    def resume(self, path: str, ctx: RunContext) -> bool:
        """Load an existing product to continue it, and return False only if absent.

        An existing `path` built with incompatible parameters must raise an
        error, so that two runs are never mixed into one product.
        """
        raise NotImplementedError

    def processed_keys(self) -> set:
        """Unit.key values already accumulated (for resume skip)."""
        raise NotImplementedError

    def processed_key_order(self) -> Optional[list[str]]:
        """Accumulated Unit.key values in consumption order, when recorded."""
        return None

    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        """Interpret a user `--select` spec for this analysis (optional).

        The default returns the spec unchanged. The spectrum analyzer overrides
        this to read `--select` as explicit freq_ids (`844`, `614,706`,
        `506-552`), and other analyses give it their own meaning (a scope, a
        date range, a feed list, ...).
        """
        return spec

    def plan_runs(self, ctx: RunContext, spec: Any) -> list:
        """Split a selection into independent runs, each with its own product.

        Returns a list of sub-selections. The engine runs once for each
        sub-selection with a new analyzer instance, so each has its own
        resumable product.

        The default is one run over the whole resolved selection. A per-item
        analysis overrides this. The spectrum analyzer returns one
        sub-selection per freq_id, so `--select 614,706` becomes two
        <freq_id>.npz products that resume and fail independently. A
        single-item list `[freq_id]` makes the engine name that product
        `<freq_id>.npz`.
        """
        return [self.resolve_selection(ctx, spec)]

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        """Consume one file and return a positive integer item count.

        A return of zero, a boolean, or a non-integer is an error, and the
        engine then neither checkpoints the unit nor marks it complete. A reader
        that finds the staged bytes cannot yield a usable item should raise
        ``UnreadableUnitError`` so that the engine can apply its quarantine
        policy.

        Analyzers with ``requires_in_order = True`` receive files in exact
        source order even when downloads finish out of order. Other analyzers
        may receive files in completion order when concurrent fetch is
        enabled.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def summary(self) -> Mapping[str, Any]:
        return {}

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        return True, []
