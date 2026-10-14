# coding=utf-8
"""Stream CHIME archive data and combine per-pilot products.

This is the implementation behind ``pilot-proxy chime-scan``. A multi-channel
pull is storage-safe and resumable, then stacks the per-pilot
products with the event-keyed combine step into PilotProxy's canonical
products; if no event is common to every completed channel, the scan still
succeeds with per-pilot products and defers stacking to ``chime-combine``
(``--report`` / ``--drop``). This is the
recommended archive-scale entry point; ``pilot-proxy chime-run`` (the
``run_chime_analysis`` batch path) remains for pre-staged local directories.

* ``--source local``          : files already on disk (a 10 s chunk, /arc, ...).
* ``--source cadc-datatrail``  : storage-safe streaming from the CADC archive.

The detector analyzer defaults to the real CUDA kernel (GPU). For tests, the
detector / kernel / weights can be injected via ``analyzer_options`` (the same
hooks ``run_chime_analysis`` exposes), which is how the GPU-free parity tests run.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pilot_proxy.atomic_io import atomic_write_json
from pilot_proxy.provenance import file_sha256

_DETECTOR_ANALYZER = "pilot-proxy-detector"
_READER_FOR_ANALYZER = {
    _DETECTOR_ANALYZER: "chime-baseband-packed",  # native int4 -> lossless kernel pack
}
_SCAN_SCOPE_SCHEMA_VERSION = "pilotproxy_chime_scan_scope_v1"
_LOCAL_INPUT_MANIFEST_SCHEMA_VERSION = "pilotproxy_local_input_manifest_v1"
_DEFAULT_LOCAL_FREQ_ID_REGEX = r"_(\d+)\.h5$"
_DEFAULT_LOCAL_EVENT_REGEX = r"baseband_(\d+)_"


def _positive_count(value: Any, *, option: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise SystemExit(f"chime-scan: --{option} must be a positive integer")
    return int(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def _read_existing_scope(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"chime-scan: existing scan scope is unreadable: {path}. "
            "Restore it or use a fresh --output-dir."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SCAN_SCOPE_SCHEMA_VERSION
    ):
        raise SystemExit(
            f"chime-scan: existing scan scope is incompatible: {path}. "
            "Use a fresh --output-dir."
        )
    return payload


def _prior_execution_attempts(
    scope: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if scope is None:
        return []
    attempts = scope.get("execution_attempts")
    if isinstance(attempts, list) and all(isinstance(item, dict) for item in attempts):
        return [dict(item) for item in attempts]
    execution = scope.get("execution")
    return [dict(execution)] if isinstance(execution, dict) else []


def _local_input_manifest(
    prepared_runs: list[tuple[Any, list[Any]]],
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for _selection, units in prepared_runs:
        for unit in units:
            meta = dict(unit.meta or {})
            relative_path = meta.get("relative_path")
            size_bytes = meta.get("size_bytes")
            sha256 = meta.get("sha256")
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise SystemExit(
                    f"chime-scan: local source returned an invalid relative path "
                    f"for {unit.name!r}"
                )
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, Integral)
                or size_bytes < 0
            ):
                raise SystemExit(
                    f"chime-scan: local source returned an invalid size for "
                    f"{relative_path!r}"
                )
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(char not in "0123456789abcdef" for char in sha256)
            ):
                raise SystemExit(
                    f"chime-scan: local source returned an invalid SHA-256 for "
                    f"{relative_path!r}"
                )
            record = {
                "path": Path(relative_path).as_posix(),
                "size_bytes": int(size_bytes),
                "sha256": sha256,
            }
            previous = records.setdefault(record["path"], record)
            if previous != record:
                raise SystemExit(
                    f"chime-scan: local file identity disagrees across selections: "
                    f"{record['path']!r}"
                )
    files = [records[path] for path in sorted(records)]
    canonical = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": _LOCAL_INPUT_MANIFEST_SCHEMA_VERSION,
        "file_count": len(files),
        "total_bytes": sum(record["size_bytes"] for record in files),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _require_preflight(
    component: object,
    ctx: object,
    *,
    label: str,
    method: str = "preflight",
) -> None:
    try:
        result = getattr(component, method)(ctx)
        ok, problems = bool(result[0]), [str(item) for item in result[1]]
    except Exception as exc:
        raise SystemExit(
            f"chime-scan: {label} preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not ok or problems:
        detail = problems or ["prerequisite check did not pass"]
        raise SystemExit(
            f"chime-scan: {label} preflight failed:\n  - "
            + "\n  - ".join(detail)
        )


def _selection_values(selection: Any) -> list[int]:
    if isinstance(selection, (list, tuple)):
        return [int(value) for value in selection]
    return [int(selection)]


def _saved_unit_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with np.load(path, allow_pickle=False) as product:
            if "unit_keys" not in product.files:
                return set()
            return {
                str(value)
                for value in np.asarray(product["unit_keys"]).reshape(-1).tolist()
            }
    except (OSError, TypeError, ValueError):
        return set()


def _framed_unit_keys(path: Path) -> set[str]:
    """Saved unit keys that own at least one persisted detector frame.

    A frame proves that a unit contributed data, not that the whole unit was
    consumed: ``max_chunks_per_file`` deliberately stops the reader early.
    Callers must classify these keys as completed or capped using the product's
    scan parameters.
    """
    if not path.exists():
        return set()
    try:
        with np.load(path, allow_pickle=False) as product:
            unit_order = [
                str(value)
                for value in np.asarray(product["unit_order"]).reshape(-1).tolist()
            ]
            frame_units = np.asarray(
                product["frame_unit_index"], dtype=np.int64
            ).reshape(-1)
        if np.any(frame_units < 0) or np.any(frame_units >= len(unit_order)):
            return set()
        return {unit_order[int(index)] for index in np.unique(frame_units)}
    except (KeyError, OSError, TypeError, ValueError):
        return set()


def _stored_chunk_cap(path: Path) -> tuple[bool, int | None]:
    """Return ``(known, cap)`` for a saved product's per-file cap."""
    if not path.exists():
        return False, None
    try:
        with np.load(path, allow_pickle=False) as product:
            value = np.asarray(product["max_chunks_per_file"])
            if value.shape != () or value.dtype.kind not in "iu":
                return False, None
            cap = int(value.item())
    except (KeyError, OSError, TypeError, ValueError):
        return False, None
    return True, cap if cap >= 0 else None


def _quarantined_unit_keys(
    units: list[Any], quarantine_path: Path
) -> set[str]:
    keys: set[str] = set()
    names: set[str] = set()
    try:
        lines = quarantine_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        key = row.get("quarantine_key", row.get("key"))
        if key is not None:
            keys.add(str(key))
        elif row.get("name") is not None:
            names.add(str(row["name"]))
    quarantined: set[str] = set()
    for unit in units:
        metadata = getattr(unit, "meta", None) or {}
        stable_key = str(metadata.get("quarantine_key", unit.key))
        if stable_key in keys or str(unit.name) in names:
            quarantined.add(str(unit.key))
    return quarantined


def _quarantined_units(units: list[Any], quarantine_path: Path) -> int:
    return len(_quarantined_unit_keys(units, quarantine_path))


def _analyzer_progress_keys(analyzer: Any, product_path: Path) -> set[str]:
    keys = _framed_unit_keys(product_path)
    processed = getattr(analyzer, "processed_keys", None)
    if callable(processed):
        try:
            keys.update(str(value) for value in processed())
        except Exception:
            pass
    return keys


def _failed_current_unit_count(
    exc: BaseException,
    units: list[Any],
    *,
    completed_keys: set[str],
    quarantined_keys: set[str],
    max_files: int | None,
) -> int:
    """Identify only the in-order unit named by a pipeline unit failure."""
    considered = units if not max_files else units[: int(max_files)]
    pending = [
        unit
        for unit in considered
        if str(unit.key) not in completed_keys
        and str(unit.key) not in quarantined_keys
    ]
    if not pending or not isinstance(exc, RuntimeError):
        return 0
    name = str(pending[0].name)
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if name in str(current):
            return 1
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return 0


def _refresh_scope_totals(scope: dict[str, Any]) -> None:
    entries = list(scope["pilots"])
    count_fields = (
        "requested",
        "enumerated",
        "completed",
        "capped",
        "failed",
        "quarantined",
        "unprocessed",
        "extra_completed",
    )
    scope["totals"] = {
        field: int(sum(int(entry.get(field, 0)) for entry in entries))
        for field in count_fields
    }
    scope["totals"]["pilots_requested"] = len(entries)
    scope["complete"] = bool(entries) and all(
        entry.get("status") == "complete" for entry in entries
    )


def _named_inventory_path(name: str, source_root: str | Path | None = None) -> Path:
    """Resolve a named archive inventory to its ``inventory.jsonl`` path.

    ``source_root`` (the directory passed to ``chime-survey --root``)
    always wins when given: ``<root>/data/<name>/inventory.jsonl``. Otherwise
    the archive resolver is the single source of truth for named inventories.
    """
    if not str(name).strip():
        raise ValueError("inventory name may not be empty")
    if source_root is not None:
        return Path(source_root) / "data" / str(name).strip() / "inventory.jsonl"
    try:
        from pilot_proxy.archive.invpaths import resolve_inventory
    except ImportError as exc:
        raise SystemExit(
            "chime-scan: the canonical inventory resolver is unavailable. "
            "Reinstall Pilot Proxy or pass --source-root explicitly."
        ) from exc
    return Path(resolve_inventory(str(name).strip()))


def _read_inventory_meta(inventory_path: Path) -> dict | None:
    """Read a validated survey sidecar when one exists."""
    from .inventory import read_inventory_meta

    try:
        metadata = read_inventory_meta(inventory_path)
    except ValueError as exc:
        raise SystemExit(f"chime-scan: {exc}") from exc
    return dict(metadata) if metadata is not None else None


def _freq_ids_in_inventory(inventory_path: Path) -> list[int]:
    """Sorted distinct freq_ids across the inventory's rows.

    Rows without a freq_id (companion shapes: gains, N2) are skipped, since they are
    exactly the rows the source's freq_id filter never serves to a per-pilot
    run. Malformed lines are skipped too, matching the source's tolerant
    ``enumerate``."""
    if not inventory_path.exists():
        raise SystemExit(
            f"chime-scan: inventory not found: {inventory_path}\n"
            "Build one with `pilot-proxy chime-survey` "
            "(or pass --inventory <path>)."
        )
    ids: set[int] = set()
    with open(inventory_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            ch = row.get("freq_id") if isinstance(row, dict) else None
            if ch is None:
                continue
            try:
                ids.add(int(ch))
            except (TypeError, ValueError):
                continue
    return sorted(ids)


def _parse_freq_id_list(spec: Any) -> list[int] | None:
    """Tolerant parse of a sidecar ``freq_ids`` field ('506,521', '506-521',
    [506, 521], ...). None when absent or not confidently parseable ('all')."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        try:
            return sorted({int(s) for s in spec})
        except (TypeError, ValueError):
            return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-")
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        except ValueError:
            return None
    return sorted(out) if out else None


def _selection_is_empty(select: Any) -> bool:
    return (
        select is None
        or (isinstance(select, str) and not select.strip())
        or (isinstance(select, (list, tuple)) and len(select) == 0)
    )


def _default_selection_from_inventory(
    inventory_path: Path, *, label: str, meta: dict | None, verbose: bool
) -> list[int]:
    """No --select: every freq_id the inventory contains, echoed before any
    staging so the scope of the run is visible up front. When the sidecar's
    requested ``freq_ids`` disagree with the rows (patchy replication, partial
    survey), say so."""
    found = _freq_ids_in_inventory(inventory_path)
    if not found:
        raise SystemExit(
            f"chime-scan: no --select given and inventory {inventory_path} has no "
            f"rows with a freq_id -- pass --select explicitly."
        )
    if verbose:
        print(
            f"[chime-scan] no --select: scanning all {len(found)} freq_id(s) from "
            f"inventory '{label}': {','.join(str(f) for f in found)}",
            flush=True,
        )
        requested = _parse_freq_id_list((meta or {}).get("freq_ids"))
        if requested is not None and set(requested) != set(found):
            missing = sorted(set(requested) - set(found))
            extra = sorted(set(found) - set(requested))
            parts = []
            if missing:
                parts.append("missing from inventory: " + ",".join(map(str, missing)))
            if extra:
                parts.append("not in the survey request: " + ",".join(map(str, extra)))
            print(
                f"[chime-scan] note: survey requested {len(requested)} freq_id(s); "
                f"inventory rows cover {len(found)} ({'; '.join(parts)})",
                flush=True,
            )
    return found


def run_chime_scan(
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path,
    source: str | None = None,
    analyzer: str = _DETECTOR_ANALYZER,
    select: Any = None,
    instrument: str = "chime",
    reader: str | None = None,
    max_files: int | None = None,
    max_chunks_per_file: int | None = None,
    work_dir: str | Path | None = None,
    staging_dir: str | Path | None = None,
    source_glob: str = "*.h5",
    source_freq_id_regex: str | None = None,
    source_event_regex: str | None = None,
    inventory: str | Path | None = None,
    inventory_name: str | None = None,
    source_root: str | Path | None = None,
    download_workers: int = 1,
    max_staged_files: int = 1,
    checkpoint_every: int | None = None,
    stop_after_checkpoint: bool = False,
    allow_partial: bool = False,
    analyzer_options: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Fan out the chosen analyzer over CHIME data and combine into canonical products.

    ``--source`` is inferred from the flags that name it: ``--inventory`` /
    ``--inventory-name`` select ``cadc-datatrail``; everything else keeps the
    local default (``--source-root`` alone serves both layouts and never
    infers). An explicit ``--source`` that conflicts with those flags is an
    error. ``--source local`` reads files under ``--input-dir``.
    ``--source cadc-datatrail`` streams from an inventory, provided as one of:

    * ``--inventory <inventory.jsonl>`` for an explicit inventory path;
    * ``--inventory-name <name>`` for a named inventory, resolved
      through the archive inventory root (default
      ``~/datatrawl-inventories/<name>/``); with ``--source-root <r>`` it
      resolves as ``<r>/data/<name>/inventory.jsonl`` instead.

    ``--select`` scopes the run to specific freq_ids; for the archive source it
    defaults to every freq_id the inventory contains (echoed before any
    staging, with a note when the survey sidecar's requested set disagrees
    with the rows). Local scans have no inventory and still require it.

    The detector analyzer's CUDA kernel is GPU-only, so ``--analyzer pilot-proxy-detector``
    requires a GPU node. A missing ``cupy`` is caught up front (rather than
    surfacing as every file failing to analyze). There is no CPU detector path for
    production (the CPU reference exists only as a test fixture), so there is no
    GPU/CPU toggle here.

    By default every requested pilot must enumerate at least one unit and every
    enumerated unit must complete. ``allow_partial=True`` is the explicit escape
    hatch for capped smoke runs or intentionally incomplete inventories; either
    way, ``scan_scope.json`` durably records the per-pilot outcome, and its
    ``terminal_combine`` entry records whether the optional terminal combine
    ran or was soft-failed (with the refusing error class and message).

    ``stop_after_checkpoint=True`` stops after the next durable checkpoint. It
    is an execution control and is not part of the product resume identity.
    """
    from pilot_proxy.archive import pipeline
    from pilot_proxy.archive.instruments import load_instrument
    from pilot_proxy.archive.interfaces import RunContext
    from pilot_proxy.archive.sources import CadcDatatrailSource, LocalDirectorySource

    from .combine import (
        CombineDuplicateIdentityError,
        CombineEmptyIntersectionError,
        combine_detector_products,
    )
    from .detector import PilotProxyDetectorAnalyzer
    from .packed_reader import ChimeBasebandPackedReader

    download_workers = _positive_count(
        download_workers, option="download-workers")
    max_staged_files = _positive_count(
        max_staged_files, option="max-staged-files")
    if download_workers > max_staged_files:
        raise SystemExit(
            "chime-scan: --max-staged-files must be at least "
            "--download-workers"
        )
    checkpoint_every = _positive_count(
        pipeline.DEFAULT_CHECKPOINT_EVERY
        if checkpoint_every is None
        else checkpoint_every,
        option="checkpoint-every",
    )

    if analyzer != _DETECTOR_ANALYZER:
        raise SystemExit(
            f"chime-scan: unknown analyzer {analyzer!r} "
            f"(expected {_DETECTOR_ANALYZER!r})"
        )
    reader_name = reader or _READER_FOR_ANALYZER[analyzer]

    # -- source: infer from the flags that name it ---------------------------
    # --inventory/--inventory-name only mean anything for the archive source, so
    # their presence names it; everything else keeps the historic local default
    # (--source-root serves both sources -- the local input dir, or the survey
    # root for --inventory-name -- and never infers).
    # Conflicting pairings are errors rather than silent ignores.
    has_inventory_flags = inventory is not None or inventory_name is not None
    if source is None:
        source = "cadc-datatrail" if has_inventory_flags else "local"
        if verbose and has_inventory_flags:
            flag = "--inventory" if inventory is not None else "--inventory-name"
            print(f"[chime-scan] source: cadc-datatrail (inferred from {flag})",
                  flush=True)
    if source == "local" and has_inventory_flags:
        raise SystemExit(
            "chime-scan: --inventory/--inventory-name belong to --source "
            "cadc-datatrail, but --source local was requested. Drop --source (it is "
            "inferred from the inventory flags) or drop the inventory flags."
        )
    if source == "cadc-datatrail" and input_dir is not None:
        raise SystemExit(
            "chime-scan: --input-dir belongs to --source local; the archive source "
            "reads from the inventory. Drop one of the two."
        )

    inst = load_instrument(instrument)
    options: dict[str, Any] = dict(analyzer_options or {})
    if source == "local":
        if input_dir is None and source_root is None:
            raise SystemExit(
                "chime-scan: --source local needs --input-dir <dir> (the directory "
                "of baseband_<event>_<freq_id>.h5 files)."
            )
        options["source_root"] = str(input_dir if input_dir is not None else source_root)
        options["source_glob"] = source_glob
        if source_freq_id_regex:
            # An explicit --set of this key wins over the flag.
            options.setdefault("source_freq_id_regex", source_freq_id_regex)
        if source_event_regex:
            options.setdefault("source_event_regex", source_event_regex)
        if _selection_is_empty(select):
            raise SystemExit(
                "chime-scan: --select is required for --source local (e.g. "
                "--select 844 or --select 829,844): a local directory has no "
                "inventory to derive the freq_id scope from. For archive scans, "
                "--select defaults to every freq_id in the inventory."
            )
    elif source == "cadc-datatrail":
        # Resolve named inventories before calling the source.
        if inventory is not None and inventory_name is not None:
            raise SystemExit(
                "chime-scan: pass either --inventory <inventory.jsonl> or "
                "--inventory-name <name>, not both."
            )
        if inventory is not None:
            options["inventory"] = str(inventory)
        elif inventory_name is not None:
            options["inventory"] = str(_named_inventory_path(inventory_name, source_root))
        else:
            raise SystemExit(
                "chime-scan: --source cadc-datatrail needs --inventory "
                "<inventory.jsonl> or --inventory-name <name> (resolved through "
                "the archive inventory root; with --source-root <r> the name "
                "resolves as <r>/data/<name>/inventory.jsonl). Build the "
                "inventory with `pilot-proxy chime-survey` first."
            )
        inv_path = Path(options["inventory"])
        meta = _read_inventory_meta(inv_path)
        expected_meta = {
            "telescope": str(inst.name),
            "source": "cadc-datatrail",
            "reader": "chime-baseband",
        }
        mismatches = [
            f"{key}={meta.get(key)!r} (expected {value!r})"
            for key, value in expected_meta.items()
            if meta is not None and meta.get(key) != value
        ]
        if mismatches:
            raise SystemExit(
                f"chime-scan: incompatible inventory metadata in "
                f"{inv_path.with_suffix('.meta.json')}: " + ", ".join(mismatches)
            )
        if _selection_is_empty(select):
            select = _default_selection_from_inventory(
                inv_path,
                label=(inventory_name or str(inv_path)),
                meta=meta,
                verbose=verbose,
            )
    else:
        raise SystemExit(
            f"chime-scan: unknown source {source!r} "
            f"(expected 'local' or 'cadc-datatrail')."
        )

    if source == "local":
        input_details = {
            "root": str(Path(options["source_root"]).expanduser().resolve()),
            "glob": str(options["source_glob"]),
            "freq_id_regex": str(
                options.get("source_freq_id_regex")
                or _DEFAULT_LOCAL_FREQ_ID_REGEX
            ),
            "source_event_regex": str(
                options.get("source_event_regex")
                or _DEFAULT_LOCAL_EVENT_REGEX
            ),
        }
    else:
        inventory_path = Path(options["inventory"]).expanduser().resolve()
        inventory_sha256 = file_sha256(inventory_path)
        if inventory_sha256 is None:
            raise SystemExit(f"chime-scan: inventory not found: {inventory_path}")
        input_details = {
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha256,
        }

    if max_chunks_per_file is not None:
        max_chunks_per_file = int(max_chunks_per_file)
        options["max_chunks_per_file"] = max_chunks_per_file

    ctx = RunContext(instrument=inst, options=options)

    source_classes = {
        "local": LocalDirectorySource,
        "cadc-datatrail": CadcDatatrailSource,
    }
    if reader_name != "chime-baseband-packed":
        raise SystemExit(
            f"chime-scan: unknown reader {reader_name!r} "
            "(expected 'chime-baseband-packed')"
        )
    src = source_classes[source]()
    rdr = ChimeBasebandPackedReader()
    analyzer_cls = PilotProxyDetectorAnalyzer
    ctx.reader = rdr

    source_preflight = (
        "fetch_preflight" if isinstance(src, CadcDatatrailSource) else "preflight"
    )
    _require_preflight(src, ctx, label="source", method=source_preflight)
    _require_preflight(rdr, ctx, label="reader")

    # Fail fast on missing runtime artifacts before any file is staged, instead
    # of quarantining every unit or dying with a raw error mid-scan. For
    # pilot-proxy-detector, that means cupy, the CUDA kernel library, and the
    # weight bank.
    analyzer_probe = analyzer_cls()
    _ok, _problems = analyzer_probe.preflight(ctx)
    if not _ok:
        raise SystemExit(
            f"chime-scan: {analyzer} preflight failed:\n  - "
            + "\n  - ".join(_problems)
            + "\n  (a detector run needs a GPU node with a built "
            "cuda/libfstatistic.so and the weight bank; run setup_env.sh on a "
            "GPU node.)")
    fine_retention = analyzer_probe.fine_retention_scope(ctx)

    runs = analyzer_cls().plan_runs(ctx, select)
    if not runs:
        raise SystemExit("chime-scan: --select resolved to an empty set")
    requested_selections = [_selection_values(run) for run in runs]
    enumerated_runs: list[tuple[Any, list[Any]]] = []
    for sub_sel in runs:
        ctx.selection = sub_sel
        enumerated_runs.append((sub_sel, list(src.enumerate(ctx))))
    if source == "local":
        input_details["file_manifest"] = _local_input_manifest(enumerated_runs)

    work = Path(work_dir) if work_dir is not None else Path(output_dir) / "_per_pilot"
    work.mkdir(parents=True, exist_ok=True)
    staging = Path(staging_dir) if staging_dir is not None else work / "_staging"
    staging = staging.expanduser().resolve()
    if source == "local":
        input_root = Path(options["source_root"]).expanduser().resolve()
        try:
            staging.relative_to(input_root)
        except ValueError:
            pass
        else:
            raise SystemExit(
                "chime-scan: --staging-dir must be outside the local input tree"
            )
    staging.mkdir(parents=True, exist_ok=True)
    quarantine_path = str(work / "quarantine.jsonl")
    scope_path = Path(output_dir) / "scan_scope.json"
    previous_scope = _read_existing_scope(scope_path)
    if previous_scope is not None and previous_scope.get("source") != source:
        raise SystemExit(
            "chime-scan: source differs from the existing scan scope; "
            "restore the original source or use a fresh --output-dir"
        )
    if previous_scope is not None:
        previous_selections = previous_scope.get("requested_selections")
        if previous_selections != requested_selections:
            raise SystemExit(
                "chime-scan: selection differs from the existing scan scope; "
                "restore the original --select value or use a fresh --output-dir"
            )
        if previous_scope.get("fine_retention") != fine_retention:
            raise SystemExit(
                "chime-scan: fine-product retention differs from the existing "
                "scan scope; restore the original fine_products setting and "
                "backend capability or use a fresh --output-dir"
            )
    if source == "cadc-datatrail" and previous_scope is not None:
        previous_input = previous_scope.get("input")
        previous_digest = (
            previous_input.get("inventory_sha256")
            if isinstance(previous_input, dict)
            else None
        )
        current_digest = input_details["inventory_sha256"]
        if not isinstance(previous_digest, str) or not previous_digest:
            raise SystemExit(
                "chime-scan: existing scan scope has no inventory SHA-256; "
                "use a fresh --output-dir"
            )
        if previous_digest != current_digest:
            raise SystemExit(
                "chime-scan: inventory SHA-256 differs from the existing "
                "scan scope; restore the original inventory or use a fresh "
                "--output-dir"
            )
    if source == "local" and previous_scope is not None:
        if previous_scope.get("input") != input_details:
            raise SystemExit(
                "chime-scan: local input differs from the existing scan scope; "
                "restore the original files and local source settings or use a "
                "fresh --output-dir"
            )
    from pilot_proxy.archive.sources.cadc import storage_service_override
    execution = {
        "preserve_source_order": True,
        "download_workers": download_workers,
        "max_staged_files": max_staged_files,
        "checkpoint_every": checkpoint_every,
        "staging_dir": str(staging),
        # None when the default locator served the run; set when the fetch was
        # routed straight at a replica.
        "storage_service": storage_service_override(),
    }
    execution_attempts = _prior_execution_attempts(previous_scope)
    execution_attempts.append(dict(execution))
    scope: dict[str, Any] = {
        "schema_version": _SCAN_SCOPE_SCHEMA_VERSION,
        "source": str(source),
        "input": input_details,
        "allow_partial": bool(allow_partial),
        "max_files": max_files,
        "max_chunks_per_file": max_chunks_per_file,
        "execution": execution,
        "execution_attempts": execution_attempts,
        "requested_selections": requested_selections,
        "fine_retention": fine_retention,
        "requested_pilots": len(runs),
        "pilots": [
            {
                "selection": _selection_values(run),
                "requested": 0,
                "enumerated": 0,
                "completed": 0,
                "capped": 0,
                "failed": 0,
                "quarantined": 0,
                "unprocessed": 0,
                "extra_completed": 0,
                "non_durable_completed": 0,
                "extra_unit_keys": [],
                "capped_unit_keys": [],
                "zero_frame_unit_keys": [],
                "status": "pending",
            }
            for run in runs
        ],
    }
    _refresh_scope_totals(scope)
    _atomic_write_json(scope_path, scope)

    prepared_runs: list[tuple[Any, list[Any], Path]] = []
    for run_index, (sub_sel, units) in enumerate(enumerated_runs):
        scope_entry = scope["pilots"][run_index]
        ctx.selection = sub_sel
        unit_keys = {str(unit.key) for unit in units}
        scope_entry["enumerated"] = len(units)
        scope_entry["requested"] = len(units)
        scope_entry["requested_units"] = len(units)
        scope_entry["unprocessed"] = len(units)
        stem = ("_".join(str(s) for s in sub_sel)
                if isinstance(sub_sel, (list, tuple)) else str(sub_sel))
        scope_entry["product"] = str(work / f"{stem}.npz")
        out_path = work / f"{stem}.npz"
        saved_keys = _saved_unit_keys(out_path)
        framed_saved_keys = _framed_unit_keys(out_path)
        _saved_cap_known, saved_chunk_cap = _stored_chunk_cap(out_path)
        extra_keys = sorted(saved_keys - unit_keys)
        zero_frame_keys = sorted(
            (saved_keys & unit_keys) - framed_saved_keys
        )
        scope_entry["extra_completed"] = len(extra_keys)
        scope_entry["extra_unit_keys"] = extra_keys
        scope_entry["zero_frame_unit_keys"] = zero_frame_keys
        prepared_runs.append((sub_sel, units, out_path))
        # Enumeration is durable before staging or analyzing any file, so an
        # interruption still leaves an auditable requested scope.
        _refresh_scope_totals(scope)
        _atomic_write_json(scope_path, scope)
        if extra_keys or zero_frame_keys:
            quarantined = _quarantined_units(units, Path(quarantine_path))
            framed_keys = unit_keys & framed_saved_keys
            capped_keys = framed_keys if saved_chunk_cap is not None else set()
            completed_keys = framed_keys - capped_keys
            scope_entry.update(
                completed=len(completed_keys),
                capped=len(capped_keys),
                capped_unit_keys=sorted(capped_keys),
                quarantined=quarantined,
                failed=0,
                unprocessed=max(
                    0,
                    len(units)
                    - len(completed_keys)
                    - len(capped_keys)
                    - quarantined,
                ),
                status="stale" if extra_keys else "zero_frames",
            )
            _refresh_scope_totals(scope)
            _atomic_write_json(scope_path, scope)
        elif not units:
            scope_entry["status"] = "empty"
            _refresh_scope_totals(scope)
            _atomic_write_json(scope_path, scope)

    product_paths: list[str] = []
    for run_index, (sub_sel, units, out_path) in enumerate(prepared_runs):
        scope_entry = scope["pilots"][run_index]
        unit_keys = {str(unit.key) for unit in units}
        if scope_entry["status"] in {"stale", "zero_frames"}:
            continue
        if not units:
            if verbose:
                print(f"  [chime-scan] select={sub_sel}: no files matched; skipping",
                      flush=True)
            continue
        ctx.selection = sub_sel
        out = str(out_path)
        if verbose:
            print(f"  [chime-scan] select={sub_sel}: {len(units)} file(s) -> {out}",
                      flush=True)
        analyzer_obj = analyzer_cls()  # fresh analyzer per product
        pilot_staging = str(staging / out_path.stem)
        try:
            result = pipeline.run(
                source=src, reader=rdr, analyzer=analyzer_obj, units=units,
                out_path=out, tmp_dir=pilot_staging, ctx=ctx,
                download_workers=download_workers,
                max_staged_files=max_staged_files,
                max_files=max_files, max_frames_per_file=max_chunks_per_file,
                checkpoint_every=checkpoint_every,
                quarantine_path=quarantine_path,
                stop_after_checkpoint=stop_after_checkpoint,
                verbose=False,
            )
        except BaseException as exc:
            progress_keys = _analyzer_progress_keys(analyzer_obj, Path(out))
            durable_framed_keys = unit_keys & _framed_unit_keys(Path(out))
            attempted_completed_keys = unit_keys & progress_keys
            quarantined_keys = _quarantined_unit_keys(
                units, Path(quarantine_path)
            )
            failed = _failed_current_unit_count(
                exc,
                units,
                completed_keys=attempted_completed_keys,
                quarantined_keys=quarantined_keys,
                max_files=max_files,
            )
            saved_cap_known, saved_chunk_cap = _stored_chunk_cap(Path(out))
            cap_is_active = (
                saved_chunk_cap is not None
                if saved_cap_known
                else max_chunks_per_file is not None
            )
            capped_keys = durable_framed_keys if cap_is_active else set()
            completed_keys = durable_framed_keys - capped_keys
            quarantined = len(quarantined_keys)
            scope_entry.update(
                completed=len(completed_keys),
                capped=len(capped_keys),
                capped_unit_keys=sorted(capped_keys),
                non_durable_completed=len(
                    attempted_completed_keys - durable_framed_keys
                ),
                quarantined=quarantined,
                failed=failed,
                unprocessed=max(
                    0,
                    len(units)
                    - len(completed_keys)
                    - len(capped_keys)
                    - quarantined
                    - failed,
                ),
                status="aborted",
            )
            _refresh_scope_totals(scope)
            _atomic_write_json(scope_path, scope)
            raise
        framed_keys = unit_keys & _framed_unit_keys(Path(out))
        capped_keys = framed_keys if max_chunks_per_file is not None else set()
        completed_keys = framed_keys - capped_keys
        completed = len(completed_keys)
        capped = len(capped_keys)
        failed = int(getattr(result, "n_failed", 0))
        quarantined = _quarantined_units(units, Path(quarantine_path))
        unprocessed = max(
            0, len(units) - completed - capped - failed - quarantined
        )
        complete = (
            max_chunks_per_file is None
            and completed == len(units)
            and failed == 0
            and quarantined == 0
            and unprocessed == 0
        )
        scope_entry.update(
            completed=completed,
            capped=capped,
            capped_unit_keys=sorted(capped_keys),
            failed=failed,
            quarantined=quarantined,
            unprocessed=unprocessed,
            status=(
                "planned_stop"
                if result.stopped_after_checkpoint
                else "complete"
                if complete
                else "capped"
                if capped
                else "partial"
            ),
        )
        _refresh_scope_totals(scope)
        _atomic_write_json(scope_path, scope)
        # The engine only writes the product if at least one unit was accumulated;
        # if every unit failed/quarantined there is no product (or it has zero
        # frames). Treat that as an error rather than silently feeding an absent/
        # empty product to combine, since it usually signals a systemic problem
        # (missing GPU, a bad inventory) that would hit every channel rather than
        # bad input for this one. Use n_done (total accumulated, this run plus any
        # resumed) instead of n_new, so a relaunch that finds a channel already
        # complete (n_new == 0) is recognized as produced rather than mistaken
        # for a failure.
        produced = Path(out).exists() and bool(framed_keys)
        if not produced:
            continue
        product_paths.append(out)
        if result.stopped_after_checkpoint:
            scope["terminal_combine"] = {
                "status": "not_run",
                "reason": "planned_stop",
            }
            _atomic_write_json(scope_path, scope)
            if verbose:
                print(
                    f"[chime-scan] planned stop after checkpoint: {out}",
                    flush=True,
                )
            return {"per_pilot_work_dir": work, "scan_scope": scope_path}

    incomplete = [
        entry for entry in scope["pilots"] if entry.get("status") != "complete"
    ]
    stale = [
        entry
        for entry in scope["pilots"]
        if entry.get("extra_unit_keys") or entry.get("zero_frame_unit_keys")
    ]
    if stale:
        detail = "; ".join(
            f"select={entry['selection']}: extra saved units="
            f"{entry['extra_unit_keys']}, zero-frame saved units="
            f"{entry['zero_frame_unit_keys']}"
            for entry in stale
        )
        raise SystemExit(
            "chime-scan: saved per-pilot product does not match the current "
            f"enumeration ({detail}); refusing to publish stale frames. "
            f"Details: {scope_path}. Use a clean output directory or restore "
            "the original inventory."
        )
    if incomplete and not allow_partial:
        detail = "; ".join(
            f"select={entry['selection']}: enumerated={entry['enumerated']}, "
            f"completed={entry['completed']}, capped={entry['capped']}, "
            f"failed={entry['failed']}, "
            f"quarantined={entry['quarantined']}, "
            f"unprocessed={entry['unprocessed']}"
            for entry in incomplete
        )
        no_product = "no usable product; " if not product_paths else ""
        raise SystemExit(
            f"chime-scan: {no_product}incomplete requested scope; refusing to publish a "
            f"partial run ({detail}). Details: {scope_path}. Rerun to resume, "
            "or pass --allow-partial to accept this incomplete scope explicitly."
        )

    if not product_paths:
        raise SystemExit(
            "chime-scan: no products produced; no files matched the selection "
            f"(source={source}, select={select}); scope: {scope_path}"
        )

    try:
        outputs = combine_detector_products(product_paths, output_dir)
    except ValueError as exc:
        # Combine's validation pass (CombineIntegrityError and its plain
        # ValueError siblings) refuses the stack, not the scan: every per-pilot
        # product is already durable, so soft-fail the optional terminal
        # combine instead of losing the run to it. The two typed refusals get
        # tailored guidance; anything else gets the generic escape. Non-
        # ValueError failures are not integrity refusals and still propagate.
        print(f"[chime-scan] terminal combine skipped: {exc}", flush=True)
        if isinstance(exc, CombineEmptyIntersectionError):
            print(
                "[chime-scan] per-pilot products are preserved under "
                f"{work}; the scan itself succeeded. Choose a channel subset "
                "with `pilot-proxy chime-combine --report --work-dir <work>` "
                "and stack it with `chime-combine --work-dir <work> "
                "--drop <freq_ids> --output-dir <run>`.",
                flush=True,
            )
        elif isinstance(exc, CombineDuplicateIdentityError):
            print(
                "[chime-scan] per-pilot products are preserved under "
                f"{work}; the scan itself succeeded. Unlike an empty "
                "intersection this is a data-integrity signal, not a routine "
                "skip: one product carries the same acquisition twice -- "
                "frame identity is (source_event_key, frame_in_unit) -- so "
                "inspect unit_scope and the source keys on the flagged "
                "product, and read the presence histogram with `pilot-proxy "
                "chime-combine --report --work-dir <work>`. Until the "
                "duplicate is explained, `chime-combine --work-dir <work> "
                "--drop <freq_id> --output-dir <run>` stacks without the "
                "affected channel.",
                flush=True,
            )
        else:
            print(
                "[chime-scan] per-pilot products are preserved under "
                f"{work}; the scan itself succeeded. Inspect the products "
                "with `pilot-proxy chime-combine --report --work-dir <work>` "
                "and stack once the refusal is understood.",
                flush=True,
            )
        # stdout scrolls away on a multi-week run; the skip must survive in
        # the run directory alongside the per-pilot outcome it annotates.
        scope["terminal_combine"] = {
            "status": "skipped",
            "error": type(exc).__name__,
            "message": str(exc),
        }
        _atomic_write_json(scope_path, scope)
        return {"per_pilot_work_dir": work, "scan_scope": scope_path}
    scope["terminal_combine"] = {"status": "combined"}
    _atomic_write_json(scope_path, scope)
    outputs["scan_scope"] = scope_path
    if verbose:
        print(f"[chime-scan] combined {len(product_paths)} pilot product(s) -> {output_dir}",
              flush=True)
        for label, path in outputs.items():
            print(f"  {label}: {path}", flush=True)
    return outputs


__all__ = ["run_chime_scan"]
