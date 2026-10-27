# coding=utf-8
"""Combine per-pilot archive products into PilotProxy's canonical products.

Each detector analyzer writes one product per coarse channel. Before the
products are stacked, their frames are aligned by event and by position in the
file. Pilots that processed different event sets contribute only the frames
they have in common, and the dropped frames are recorded in the alignment
report.

The aligned arrays are then passed through the same writer functions as
``run_chime_analysis``. This produces ``chime_detector_outputs``,
``chime_spectrogram_cache``, ``chime_reductions_10s``, and ``mask_summary`` in
the canonical formats used by plotting and ``validate-products``.

A per-pilot input is the product written by ``PilotProxyDetectorAnalyzer.save``.
Its two-dimensional frame arrays have shape ``(frames, 1)``.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pilot_proxy.archive.chime_coarse import CHIME_COARSE_WIDTH_HZ
from pilot_proxy.atomic_io import (
    atomic_write_json,
    create_temporary_sibling,
    fsync_directory,
    fsync_file,
)
from pilot_proxy.chime.products import (
    CHIME_COMBINE_CANONICAL_RELATIVE_PATHS,
    CHIME_COMBINE_GENERATION_MANIFEST_FILENAME,
    CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA,
    CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME,
    CHIME_COMBINE_PUBLISH_LOCK_FILENAME,
    SCAN_INPUT_MANIFEST_SCHEMA_TOKEN,
    atomic_savez_compressed,
    ensure_run_dirs,
    write_detector_outputs,
    write_integrated_spectra,
    write_mask_summary,
    write_spectrogram_cache,
)
from pilot_proxy.chime.reductions import write_reductions_npz
from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
    normalized_positive_excess_policy,
    validate_detector_contract,
)
from pilot_proxy.product_contract import (
    PER_FRAME_PRODUCT_KEYS,
    CurrentProductContractError,
    exact_integer_array,
    exact_integer_scalar,
    null_power_ratio_of,
    validate_current_product_identity,
)
from pilot_proxy.provenance import (
    detector_version_build_id,
    detector_version_geometry,
    file_sha256,
)

from .chime_coarse import source_event_key

_PUBLISH_JOURNAL_SCHEMA = "pilotproxy_combine_publish_journal_v1"
_PUBLISH_LOCK_SCHEMA = "pilotproxy_combine_publish_lock_v1"
_TRANSACTION_DIR_PREFIX = ".pilotproxy-combine-transaction."
_GENERATION_LABEL = "combine_generation_manifest"


@dataclass(frozen=True)
class _PublishOwnership:
    path: Path
    fd: int
    owner_token: str

    def assert_owned(self) -> None:
        """Raise RuntimeError if the lock path no longer names our inode and token."""
        try:
            descriptor_stat = os.fstat(self.fd)
            path_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError("combine: publish lock ownership was lost") from exc
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or path_stat.st_nlink != 1
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            raise RuntimeError("combine: publish lock ownership was replaced")
        try:
            raw = os.pread(self.fd, 65_536, 0)
            payload = json.loads(raw)
        except (OSError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("combine: publish lock metadata is invalid") from exc
        if payload.get("owner_token") != self.owner_token:
            raise RuntimeError("combine: publish lock owner token changed")


def _write_locked_metadata(fd: int, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    os.ftruncate(fd, 0)
    offset = 0
    while offset < len(encoded):
        offset += os.pwrite(fd, encoded[offset:], offset)
    os.fsync(fd)


def _unlock_and_close(fd: int) -> None:
    """Always close a lock descriptor, even if explicit unlock is unsupported."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def _exclusive_publish_ownership(run_dir: Path) -> Iterator[_PublishOwnership]:
    """Acquire the single-writer lock of one run directory, recoverable after a crash.

    A fresh lock path is created with ``O_EXCL``. If a crashed process left the
    path behind, the kernel ``flock`` on it can be acquired only when no live
    publisher holds it. The same inode is then given a new owner token, which
    avoids an unlink and recreate race. When a live process holds the lock,
    acquisition fails and nothing is published.
    """
    run = Path(run_dir).absolute()
    _resolved_directory_root(run, what="run directory")
    lock_path = run / CHIME_COMBINE_PUBLISH_LOCK_FILENAME
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    for _ in range(16):
        created = False
        try:
            fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o666)
            created = True
        except FileExistsError:
            try:
                fd = os.open(lock_path, flags)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"combine: existing publish lock is unsafe: {lock_path}"
                ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"combine: cannot create exclusive publish lock {lock_path}"
            ) from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            # The descriptor is always closed after a failed flock. A path
            # created by this attempt is removed only while it still names our
            # one-link inode. The path of an existing publisher is never
            # unlinked here.
            try:
                descriptor_stat = os.fstat(fd)
                try:
                    path_stat = os.stat(lock_path, follow_symlinks=False)
                except OSError:
                    path_stat = None
                if (
                    created
                    and path_stat is not None
                    and stat.S_ISREG(path_stat.st_mode)
                    and descriptor_stat.st_nlink == 1
                    and path_stat.st_nlink == 1
                    and descriptor_stat.st_dev == path_stat.st_dev
                    and descriptor_stat.st_ino == path_stat.st_ino
                ):
                    lock_path.unlink()
                    fsync_directory(run)
            finally:
                os.close(fd)
            if isinstance(exc, BlockingIOError):
                raise RuntimeError(
                    "combine: another process owns the canonical publish lock; "
                    "retry after that chime-combine process finishes"
                ) from exc
            raise RuntimeError(
                "combine: kernel publish-lock acquisition failed; refusing "
                "to publish without exclusive ownership"
            ) from exc
        try:
            descriptor_stat = os.fstat(fd)
            path_stat = os.stat(lock_path, follow_symlinks=False)
        except OSError:
            _unlock_and_close(fd)
            fd = None
            continue
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or path_stat.st_nlink != 1
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            _unlock_and_close(fd)
            raise RuntimeError(
                "combine: publish lock must be one regular, non-hard-linked file"
            )
        break
    if fd is None:
        raise RuntimeError("combine: could not establish publish-lock ownership")

    token = uuid.uuid4().hex
    ownership = _PublishOwnership(lock_path, fd, token)
    initialized = False
    try:
        _write_locked_metadata(
            fd,
            {
                "schema_version": _PUBLISH_LOCK_SCHEMA,
                "owner_token": token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            },
        )
        fsync_directory(run)
        ownership.assert_owned()
        initialized = True
        yield ownership
    finally:
        try:
            descriptor_stat = os.fstat(fd)
            try:
                path_stat = os.stat(lock_path, follow_symlinks=False)
            except OSError:
                path_stat = None
            same_safe_inode = bool(
                path_stat is not None
                and stat.S_ISREG(path_stat.st_mode)
                and descriptor_stat.st_nlink == 1
                and path_stat.st_nlink == 1
                and descriptor_stat.st_dev == path_stat.st_dev
                and descriptor_stat.st_ino == path_stat.st_ino
            )
            if same_safe_inode:
                if initialized:
                    ownership.assert_owned()
                lock_path.unlink()
                fsync_directory(run)
        finally:
            _unlock_and_close(fd)


def _write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


class CombineIntegrityError(ValueError):
    """The per-pilot products are intact but cannot be stacked as given.

    This is the base class of every error that the combine validation raises
    about the products themselves. The scan layer catches this one class to
    soft-fail the optional final combine step.
    """


class CombineEmptyIntersectionError(CombineIntegrityError):
    """No (event, frame) identity is shared by every product handed to combine."""


class CombineDuplicateIdentityError(CombineIntegrityError):
    """One product carries the same (event, frame) identity more than once.

    The frame identity is ``(source_event_key, frame_in_unit)`` and has no
    unit index, so any two units whose keys reduce to the same event key give
    colliding frames. One event listed under two archive scopes is a different
    case. The archive source keys a unit by ``(scope, event, name)``, and
    ``source_event_key`` strips only a trailing freq_id token, so those two
    units keep distinct event keys and are stacked as two events.

    The duplicate must be explained before these products can be combined.
    The per-pilot products remain intact, and ``unit_scope`` shows which source
    units contributed the repeated identity.
    """


def _label(z: Mapping[str, Any]) -> str:
    ch = exact_integer_scalar(
        z, "physical_channel", dtype=np.int32, minimum=14, maximum=69
    )
    fid = exact_integer_scalar(
        z, "freq_id", dtype=np.int64, minimum=0, maximum=1023
    )
    return f"ch{ch}/freq_id {fid}"


def _unit_metadata_by_event(z: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return available acquisition metadata keyed by the namespaced event key."""
    events = np.asarray(z["source_event_keys"]).reshape(-1).astype(str)
    fields = (
        "unit_event_id",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "unit_delta_time",
        "unit_scope",
        "archive_version",
        "unit_git_version_tag",
        "unit_input_map_sha256",
    )
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        if field not in z:
            continue
        values = np.asarray(z[field])
        if values.ndim != 1:
            raise ValueError(
                f"combine: {_label(z)} field {field!r} must be 1D"
            )
        if field in {
            "archive_version",
            "unit_scope",
            "unit_git_version_tag",
            "unit_input_map_sha256",
        }:
            if values.dtype.kind not in {"U", "S"}:
                raise ValueError(
                    f"combine: {_label(z)} field {field!r} must be a string vector"
                )
        elif field == "unit_event_id":
            values = exact_integer_array(
                values,
                field=field,
                dtype=np.int64,
                ndim=1,
                minimum=-1,
            )
        elif field == "unit_time0_fpga":
            values = exact_integer_array(
                values,
                field=field,
                dtype=np.uint64,
                ndim=1,
                minimum=0,
            )
        elif values.dtype != np.dtype(np.float64) or np.any(np.isinf(values)):
            raise ValueError(
                f"combine: {_label(z)} field {field!r} must be float64 "
                "without infinite values"
            )
        if field == "unit_delta_time" and np.any(
            np.isfinite(values) & (values <= 0.0)
        ):
            raise ValueError(
                f"combine: {_label(z)} finite unit_delta_time must be positive"
            )
        if values.size != events.size:
            raise ValueError(
                f"combine: {_label(z)} field {field!r} has {values.size} "
                f"entries for {events.size} source events"
            )
        arrays[field] = values
    out: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events.tolist()):
        if event in out:
            raise ValueError(
                f"combine: {_label(z)} records source event {event!r} more "
                "than once; event timing would be ambiguous"
            )
        out[event] = {field: values[index] for field, values in arrays.items()}
    return out


def _known_event_metadata(field: str, value: Any) -> bool:
    if field in {
        "archive_version",
        "unit_scope",
        "unit_git_version_tag",
        "unit_input_map_sha256",
    }:
        return bool(str(value))
    if field == "unit_event_id":
        return int(value) >= 0
    if field == "unit_time0_fpga":
        return int(value) > 0
    numeric = float(value)
    return bool(np.isfinite(numeric) and (field != "unit_delta_time" or numeric > 0.0))


# The baseband of one burst is captured for each frequency over that
# frequency's own DM-delayed window, so the per-channel file start times of one
# event differ by the dispersion sweep of the burst. In the 2026-09 archive
# run, over 8906 events seen on ten or more channels, the spread was 1.5 s at
# the median and 12.5 s at most. Lower frequencies started later in every
# monotonic case, and the implied DM was up to about 2200 pc cm^-3. The sweep
# across the full 400-800 MHz band at a DM of 3000 pc cm^-3 is 58 s, and twice
# that is wider than any one capture can be. Path-matched units further apart
# than this are not one acquisition.
MAX_EVENT_START_SPREAD_S = 120.0

# Acquisition identity fields. Each must agree across every pilot that carries
# it, and if one pilot carries it, every pilot must. Path-matched units that
# differ here are not one acquisition. In the 2026-09 run, none of the 8906
# events on ten or more channels disagreed.
_IDENTITY_FIELDS = ("unit_scope", "archive_version")
# Per-node provenance fields. The file of each frequency is written by its own
# baseband node, so a rolling software deploy or an input-map change can leave
# two values across the channels of one burst. The same run had 4 events with
# two writer tags and 1 event with two input maps. Mixed values are recorded
# and accepted.
_RECORDED_FIELDS = ("unit_git_version_tag", "unit_input_map_sha256")
_TIMING_FIELDS = ("unit_time0_fpga", "unit_time0_ctime", "unit_delta_time")


def _validate_common_event_metadata(
    products: Sequence[Mapping[str, Any]], common_events: set[str]
) -> dict[str, Any]:
    """Raise ValueError for path-matched events that are not one acquisition.

    The acquisition ID, scope, and archive version must agree wherever they
    are known, and if any pilot knows one, every pilot must. The writer version
    and input map are per-node provenance and can differ across the channels of
    one burst, so mixed values of these two are counted and listed. Timing
    fields are compared only among the pilots that carry them. A missing start
    time or sample period is a gap in the metadata of the archive file and is
    counted. Start times may differ across pilots by up to the dispersion sweep
    bound. Since the stack is keyed by event and is not aligned in time, this
    offset is recorded in the alignment report and accepted.

    Returns the timing and provenance counts for the alignment report.
    """
    metadata = [_unit_metadata_by_event(z) for z in products]
    spread_max = 0.0
    partial_time0: set[str] = set()
    partial_period: set[str] = set()
    partial_recorded: set[str] = set()
    mixed: dict[str, set[str]] = {field: set() for field in _RECORDED_FIELDS}
    for event in sorted(common_events):
        per_pilot = [rows[event] for rows in metadata]
        periods = [
            float(row["unit_delta_time"])
            for row in per_pilot
            if "unit_delta_time" in row
            and _known_event_metadata("unit_delta_time", row["unit_delta_time"])
        ]
        # FPGA counts advance once per coarse-channel sample. The shortest
        # recorded sample period is used, or the CHIME coarse-channel sample
        # period when no pilot records one.
        period = min(periods) if periods else 1.0 / CHIME_COARSE_WIDTH_HZ
        for field in (
            "unit_event_id", *_TIMING_FIELDS, *_IDENTITY_FIELDS, *_RECORDED_FIELDS
        ):
            known = [
                row[field]
                for row in per_pilot
                if field in row and _known_event_metadata(field, row[field])
            ]
            if not known:
                continue
            if len(known) != len(products):
                if field == "unit_delta_time":
                    partial_period.add(event)
                    continue
                if field in _TIMING_FIELDS:
                    partial_time0.add(event)
                    continue
                if field in _RECORDED_FIELDS:
                    partial_recorded.add(event)
                    continue
                raise ValueError(
                    f"combine: common source event {event!r} has {field} "
                    "metadata for only some pilots; refusing an unverifiable "
                    "cross-channel alignment"
                )
            if field in _RECORDED_FIELDS:
                if len({str(value) for value in known}) > 1:
                    mixed[field].add(event)
                continue
            if field in _IDENTITY_FIELDS:
                consistent = len({str(value) for value in known}) == 1
            elif field == "unit_event_id":
                consistent = len({int(value) for value in known}) == 1
            elif field == "unit_delta_time":
                consistent = bool(
                    np.allclose(
                        np.asarray(known, dtype=np.float64),
                        float(known[0]),
                        rtol=1e-12,
                        atol=0.0,
                    )
                )
            else:
                values = np.asarray(known, dtype=np.float64)
                spread = float(values.max() - values.min())
                if field == "unit_time0_fpga":
                    spread *= period
                spread_max = max(spread_max, spread)
                if spread > MAX_EVENT_START_SPREAD_S:
                    raise ValueError(
                        f"combine: common source event {event!r} start times "
                        f"span {spread:.3f} s across pilots on {field}, beyond "
                        f"the {MAX_EVENT_START_SPREAD_S:.0f} s dispersion sweep "
                        "bound; path-matched units are not one acquisition"
                    )
                consistent = True
            if not consistent:
                raise ValueError(
                    f"combine: common source event {event!r} disagrees on "
                    f"{field} across pilots: {known!r}"
                )
    return {
        "time0_spread_max_s": spread_max,
        "n_events_partial_time0": len(partial_time0),
        "n_events_partial_sample_period": len(partial_period),
        "n_events_partial_provenance": len(partial_recorded),
        "n_events_mixed_git_version_tag": len(mixed["unit_git_version_tag"]),
        "n_events_mixed_input_map": len(mixed["unit_input_map_sha256"]),
        "events_mixed_input_map": sorted(mixed["unit_input_map_sha256"]),
    }

def _align_frames(
    products: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Subset and reorder every product onto the frame identities they all share.

    The identity of a frame is ``(source event, frame position in its file)``,
    and the analyzer records it for every frame. The canonical order is the
    order of the reference (lowest-channel) product, restricted to the common
    identities. A fully aligned set therefore passes through unchanged and is
    byte-identical to the output of ``run_chime_analysis``. For a ragged set,
    only the overlap is stacked, and the frames dropped by each pilot are
    reported. Every product must carry event-keyed identities.
    """
    identities = [_frame_identity(z) for z in products]
    for z, ids in zip(products, identities):
        n = int(np.asarray(z["frame_index"]).reshape(-1).size)
        if ids.size != n:
            raise ValueError(
                "combine: frame identity length does not match frame_index length")
        if len(set(ids.tolist())) != ids.size:
            raise CombineDuplicateIdentityError(
                f"combine: {_label(z)} contains duplicate (event, frame) "
                f"identities; one acquisition appears twice in that product. "
                f"Identity is (source_event_key, frame_in_unit), so two units "
                f"whose keys reduce to the same event key collide here; check "
                f"unit_scope and the source keys on that product before "
                f"stacking")
    sets = [set(ids.tolist()) for ids in identities]
    common = set.intersection(*sets)
    if not common:
        counts = ", ".join(
            f"{_label(z)}: {len(s)} frames/"
            f"{len({i.split(chr(0))[0] for i in s})} events"
            for z, s in zip(products, sets))
        raise CombineEmptyIntersectionError(
            f"combine: the {len(products)} per-pilot products share no common "
            f"(event, frame) identity; there is nothing every pilot saw, so an "
            f"event-keyed stack over all of them is empty. Per-pilot inventory: "
            f"{counts}. Stack a channel subset instead (`pilot-proxy "
            f"chime-combine --report` shows the presence histogram and the "
            f"drop-curve; `--drop <freq_ids>` excludes channels).")
    common_events = {identity.split("\0")[0] for identity in common}
    timing = _validate_common_event_metadata(products, common_events)
    ref_ids = identities[0].tolist()
    canonical = [i for i in ref_ids if i in common]
    aligned: list[dict[str, Any]] = []
    by_pilot: list[dict[str, Any]] = []
    kept_events = {i.split("\0")[0] for i in canonical}
    for z, ids in zip(products, identities):
        pos = {i: r for r, i in enumerate(ids.tolist())}
        rows = np.asarray([pos[i] for i in canonical], dtype=np.int64)
        out = dict(z)
        for key in PER_FRAME_PRODUCT_KEYS:
            if key in out:
                out[key] = np.asarray(out[key])[rows]
        aligned.append(out)
        pilot_events = {i.split("\0")[0] for i in ids.tolist()}
        by_pilot.append({
            "physical_channel": int(np.asarray(z["physical_channel"]).reshape(-1)[0]),
            "freq_id": (int(np.asarray(z["freq_id"]).reshape(-1)[0])
                        if "freq_id" in z else None),
            "n_frames_total": int(ids.size),
            "n_frames_dropped": int(ids.size - len(canonical)),
            "n_events_total": len(pilot_events),
            "n_events_dropped": len(pilot_events - kept_events),
        })
    info = {
        "mode": "event_keyed",
        **timing,
        "n_frames_common": len(canonical),
        "n_events_common": len(kept_events),
        "by_pilot": by_pilot,
        "frame_event_key": [i.split("\0")[0] for i in canonical],
        "frame_in_unit": [int(i.split("\0")[1]) for i in canonical],
    }
    dropped = [p for p in by_pilot if p["n_frames_dropped"]]
    if dropped:
        detail = ", ".join(
            f"ch{p['physical_channel']}"
            + (f"/freq_id {p['freq_id']}" if p["freq_id"] is not None else "")
            + f": -{p['n_frames_dropped']} frames/-{p['n_events_dropped']} events"
            for p in dropped)
        print(
            f"[combine] event-keyed alignment: kept {len(canonical)} frame(s) / "
            f"{len(kept_events)} event(s) common to {len(products)} pilot(s); "
            f"dropped {detail}", flush=True)
    frame_index = np.arange(len(canonical), dtype=np.int64)
    return aligned, frame_index, info


def _validate_source_event_key_derivation(
    product: Mapping[str, Any], *, context: str
) -> None:
    """Check that source_event_keys are derived from unit_order and freq_id."""
    try:
        freq_id = exact_integer_scalar(
            product, "freq_id", dtype=np.int64, minimum=0, maximum=1023
        )
        unit_order = [
            str(value)
            for value in np.asarray(product["unit_order"]).reshape(-1)
        ]
        recorded = [
            str(value)
            for value in np.asarray(product["source_event_keys"]).reshape(-1)
        ]
    except KeyError as exc:
        raise ValueError(
            f"combine: {context}: missing event-identity field {exc.args[0]!r}"
        ) from exc
    if len(unit_order) != len(recorded):
        raise ValueError(
            f"combine: {context}: unit_order has {len(unit_order)} entries but "
            f"source_event_keys has {len(recorded)}"
        )
    expected = [source_event_key(key, freq_id) for key in unit_order]
    if recorded != expected:
        raise ValueError(
            f"combine: {context}: source_event_keys are not the required "
            "namespaced derivation of unit_order and freq_id; regenerate this "
            "per-pilot product"
        )


def report_products(product_paths: Sequence[str | Path]) -> str:
    """Return the event-presence report for a set of per-pilot products.

    The report has the event count of each pilot, the presence histogram, the
    intersection over all pilots, and the greedy drop-curve. The drop-curve is
    the intersection left after the most constraining pilot is removed, which
    is repeated until max(2, n // 2) of the n pilots remain. The most
    constraining pilot is the one whose removal leaves the largest
    intersection. The report is used to choose the channel subset for a
    combine."""
    import collections
    ev: dict[str, set[str]] = {}
    for p in product_paths:
        with np.load(str(p), allow_pickle=False) as z:
            product = {name: z[name] for name in z.files}
        validate_current_product_identity(product)
        _validate_source_event_key_derivation(product, context=str(p))
        label = (
            f"ch{exact_integer_scalar(product, 'physical_channel', dtype=np.int32, minimum=14, maximum=69)}"
            + f"/freq_id {exact_integer_scalar(product, 'freq_id', dtype=np.int64, minimum=0, maximum=1023)}"
        )
        events = set(
            np.asarray(product["source_event_keys"]).reshape(-1).astype(str).tolist()
        )
        ev[label] = events
    lines = [f"per-pilot products: {len(ev)}"]
    for label in sorted(ev):
        lines.append(f"  {label}: {len(ev[label])} events")
    if not ev or not any(ev.values()):
        lines.append("no event metadata present; report unavailable")
        return "\n".join(lines)
    union = set().union(*ev.values())
    presence = collections.Counter()
    for s in ev.values():
        for e in s:
            presence[e] += 1
    hist = collections.Counter(presence.values())
    lines.append(f"union: {len(union)} distinct events")
    lines.append("events by how many pilots hold them: "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())))
    lines.append(f"intersection of all {len(ev)} pilots: "
                 f"{len(set.intersection(*ev.values()))}")
    work = dict(ev)
    lines.append("drop-curve (removing the most-constraining pilot each step):")
    while len(work) > max(2, len(ev) // 2):
        best = None
        for c in work:
            n = len(set.intersection(*(work[x] for x in work if x != c)))
            if best is None or n > best[1]:
                best = (c, n)
        c, n = best
        del work[c]
        lines.append(f"  drop {c}: intersection of remaining {len(work)} = {n}")
    return "\n".join(lines)


def _detector_contract_from(
    products: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the detector contract that the analyzer stored in the first product."""
    try:
        contract = json.loads(
            str(np.asarray(products[0]["detector_contract_json"]).reshape(()).item())
        )
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "combine: current per-pilot product lacks a valid detector_contract_json"
        ) from exc
    if not isinstance(contract, dict) or not contract:
        raise ValueError(
            "combine: detector_contract_json must encode a non-empty object"
        )
    try:
        validate_detector_contract(contract)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "combine: detector_contract_json does not satisfy the current "
            f"detector contract: {exc}"
        ) from exc
    return contract


def _load_sorted(product_paths: Sequence[str | Path]) -> list[Mapping[str, Any]]:
    if not product_paths:
        raise ValueError("combine: no per-pilot product files given")
    products: list[dict[str, Any]] = []
    for path in product_paths:
        with np.load(str(path), allow_pickle=False) as product:
            loaded = {name: product[name] for name in product.files}
        try:
            validate_current_product_identity(loaded)
        except CurrentProductContractError as exc:
            raise ValueError(f"combine: {path}: {exc}") from exc
        _validate_source_event_key_derivation(loaded, context=str(path))
        products.append(loaded)
    products.sort(
        key=lambda z: exact_integer_scalar(
            z, "physical_channel", dtype=np.int32, minimum=14, maximum=69
        )
    )
    chans = [
        exact_integer_scalar(
            z, "physical_channel", dtype=np.int32, minimum=14, maximum=69
        )
        for z in products
    ]
    dupes = sorted({channel for channel in chans if chans.count(channel) > 1})
    if dupes:
        raise ValueError(
            f"combine: ATSC physical channel(s) {dupes} appear in more than one "
            "per-pilot product. The combined schema is one pilot per ATSC "
            "channel; drop the duplicate receiver channel."
        )
    return products


def _frame_identity(z: Mapping[str, Any]) -> np.ndarray:
    required = {"source_event_keys", "frame_unit_index", "frame_in_unit"}
    missing = sorted(required.difference(z))
    if missing:
        raise ValueError(
            "combine: current per-pilot product is missing frame identity arrays: "
            + ", ".join(missing)
        )
    events = np.asarray(z["source_event_keys"]).reshape(-1).astype(str)
    unit_index = exact_integer_array(
        z["frame_unit_index"],
        field="frame_unit_index",
        dtype=np.int32,
        ndim=1,
        minimum=0,
    )
    frame_in_unit = exact_integer_array(
        z["frame_in_unit"],
        field="frame_in_unit",
        dtype=np.int32,
        ndim=1,
        minimum=0,
    )
    if unit_index.shape != frame_in_unit.shape:
        raise ValueError("combine: frame_unit_index and frame_in_unit shapes differ")
    if np.any(unit_index < 0) or np.any(unit_index >= events.size):
        raise ValueError("combine: frame_unit_index contains an out-of-range unit")
    return np.asarray(
        [f"{events[u]}\0{int(f)}" for u, f in zip(unit_index, frame_in_unit)],
        dtype=str,
    )


def _check_invariants(products: Sequence[Mapping[str, Any]],
                      keys, what: str) -> dict[str, Any]:
    """Check that all per-pilot products agree on geometry and configuration scalars.

    The combined product takes the frame arrays of every pilot and the metadata
    of the first product only. The other products must therefore have the same
    geometry and configuration (nfft, K, Nyquist-zone sign, schema, and sample
    rate), since inconsistent products would otherwise be stacked into one
    canonical output without warning.

    `detector_version` is compared token by token. Its `pilot-proxy/<version>`
    and `source=` components are build provenance, so products from different
    builds of one survey, even builds on either side of a release version
    bump, are stacked when every other token (kernel hash, K, and schema)
    matches. When more than one build contributed, the full stamps are
    returned as {"detector_versions": [...]} and kept in the combined product.

    `detector_contract_json` is compared with `fine_reduction.designated_bins`
    removed. The designated window is placed on the predicted carrier line of
    each pilot, so it is per-pilot data. It is used only by the fine diagnostic
    CFAR, and no combined output stacks fine arrays. Every other contract field
    must match exactly. When the windows differ, they are returned as
    {"fine_designated_bins_by_pilot": [...]} and kept in the combined
    provenance.

    Returns these provenance notes, an empty dict when all builds and windows
    agree.
    """
    notes: dict[str, Any] = {}
    ref = products[0]
    for key in keys:
        if key not in ref:
            raise ValueError(
                f"combine: current per-pilot product is missing {key!r}, needed "
                f"to verify {what}"
            )
        if key == "detector_contract_json":
            docs = []
            for z in products:
                if key not in z:
                    raise ValueError(
                        f"combine: a product is missing '{key}', needed to "
                        f"verify {what}.")
                docs.append(json.loads(
                    str(np.asarray(z[key]).reshape(()).item())))

            def _without_designated(doc):
                doc = json.loads(json.dumps(doc))
                fine = doc.get("fine_reduction")
                if isinstance(fine, dict):
                    fine.pop("designated_bins", None)
                return json.dumps(doc, sort_keys=True)

            ref_doc = _without_designated(docs[0])
            for doc in docs[1:]:
                if _without_designated(doc) != ref_doc:
                    raise ValueError(
                        f"combine: per-pilot products disagree on '{key}' "
                        f"beyond fine_reduction.designated_bins; refusing to "
                        f"stack mismatched {what}.")
            windows = [
                (doc.get("fine_reduction") or {}).get("designated_bins")
                for doc in docs
            ]
            if len({json.dumps(w) for w in windows}) > 1:
                notes["fine_designated_bins_by_pilot"] = windows
                print(f"[combine] provenance: {len(products)} pilots carry "
                      f"channel-targeted fine designated windows; stacking "
                      f"with the union in the combined contract.", flush=True)
            continue
        if key == "detector_version":
            versions = []
            for z in products:
                if key not in z:
                    raise ValueError(
                        f"combine: a product is missing '{key}', needed to "
                        f"verify {what}.")
                versions.append(str(np.asarray(z[key]).reshape(-1)[0]))
            geoms = {detector_version_geometry(v) for v in versions}
            if len(geoms) > 1:
                raise ValueError(
                    f"combine: per-pilot products disagree on detector_version "
                    f"geometry tokens ({sorted(' '.join(g) for g in geoms)!r}); "
                    f"refusing to stack mismatched {what}.")
            distinct = sorted(set(versions))
            if len(distinct) > 1:
                notes["detector_versions"] = distinct
                short = ", ".join(detector_version_build_id(v) for v in distinct)
                print(f"[combine] provenance: {len(distinct)} source builds "
                      f"with identical detector geometry contributed "
                      f"(builds={short}); stacking.", flush=True)
            continue
        base = np.asarray(ref[key]).reshape(-1)
        for z in products[1:]:
            if key not in z:
                raise ValueError(
                    f"combine: a product is missing '{key}', needed to verify {what}."
                )
            other = np.asarray(z[key]).reshape(-1)
            if base.shape != other.shape or not np.array_equal(base, other):
                raise ValueError(
                    f"combine: per-pilot products disagree on '{key}' "
                    f"({base.tolist()!r} vs {other.tolist()!r}); refusing to stack "
                    f"mismatched {what}."
                )
    return notes


def _common_sample_rate_hz(products: Sequence[Mapping[str, Any]]) -> float:
    """Return the common sample rate from sample_rate_hz or unit_delta_time."""
    per_product: list[float] = []
    for z in products:
        recorded = np.asarray(z.get("sample_rate_hz", []), dtype=np.float64).reshape(-1)
        values = np.asarray(z.get("unit_delta_time", []), dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values) & (values > 0.0)]
        recorded_rate = (
            float(recorded[0])
            if recorded.size == 1 and np.isfinite(recorded[0]) and recorded[0] > 0.0
            else None
        )
        timing_rate = float(1.0 / finite[0]) if finite.size else None
        if finite.size and not np.allclose(finite, finite[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                "combine: one per-pilot product contains inconsistent "
                "unit_delta_time values"
            )
        if (
            recorded_rate is not None
            and timing_rate is not None
            and not np.isclose(recorded_rate, timing_rate, rtol=1e-12, atol=0.0)
        ):
            raise ValueError(
                "combine: sample_rate_hz disagrees with unit_delta_time in a "
                "per-pilot product"
            )
        resolved = recorded_rate if recorded_rate is not None else timing_rate
        if resolved is None:
            raise ValueError(
                "combine: a current per-pilot product has neither a positive "
                "sample_rate_hz nor usable unit_delta_time metadata"
            )
        per_product.append(float(resolved))
    reference = per_product[0]
    if not np.allclose(per_product, reference, rtol=1e-12, atol=0.0):
        raise ValueError(
            "combine: per-pilot products disagree on sample_rate_hz/"
            "unit_delta_time; refusing to construct shared time and spectral axes"
        )
    return float(reference)


def _json_scalar(z: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = z.get(key)
    if raw is None:
        return {}
    try:
        value = json.loads(str(np.asarray(raw).reshape(()).item()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _combined_reference_placement_summary(
    products: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    parsed = [_json_scalar(z, "reference_placement_json") for z in products]
    present = [bool(summary) for summary in parsed]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "combine: reference-placement metadata is present for only some pilots"
        )
    summaries = parsed
    offsets = {int(summary.get("reference_offset_bins", 0)) for summary in summaries}
    guards = {int(summary.get("skipped_guard_bins", 0)) for summary in summaries}
    if len(offsets) != 1 or len(guards) != 1:
        raise ValueError("combine: reference-placement geometry differs between pilots")
    by_channel = [
        row
        for summary in summaries
        for row in summary.get("by_channel", [])
        if isinstance(row, dict)
    ]
    statuses = sorted({str(row.get("reference_placement_status", "unknown")) for row in by_channel})
    status = statuses[0] if len(statuses) == 1 else "mixed:" + ";".join(statuses)
    adaptive = [
        int(row["physical_channel"])
        for row in by_channel
        if str(row.get("reference_placement_status", "unknown")) != "nominal"
    ]
    dc_shifted = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("dc_reference_shifted", False))
    ]
    edge_wrapped = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("edge_reference_wrapped", False))
    ]
    skipped_guard = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("forbidden_tone_in_skipped_guard", False))
    ]
    return {
        "reference_offset_bins": offsets.pop(),
        "skipped_guard_bins": guards.pop(),
        "reference_placement_status": status,
        "num_channels_with_adaptive_reference": len(adaptive),
        "channels_with_adaptive_reference": adaptive,
        "num_dc_shifted_references": sum(
            int(summary.get("num_dc_shifted_references", 0)) for summary in summaries
        ),
        "channels_with_dc_shifted_reference": dc_shifted,
        "num_edge_wrapped_references": sum(
            int(summary.get("num_edge_wrapped_references", 0)) for summary in summaries
        ),
        "channels_with_edge_wrapped_reference": edge_wrapped,
        "num_forbidden_tone_in_skipped_guard": len(skipped_guard),
        "channels_with_forbidden_tone_in_skipped_guard": skipped_guard,
        "forbidden_tone_policy": summaries[0].get("forbidden_tone_policy"),
        "by_channel": by_channel,
    }


def _stack_cols(products: Sequence[Mapping[str, Any]], key: str, dtype) -> np.ndarray:
    """Stack per-pilot (frames, 1) arrays into (frames, pilots)."""
    cols = [np.asarray(z[key], dtype=dtype).reshape(int(np.asarray(z[key]).shape[0]), 1)
            for z in products]
    return np.concatenate(cols, axis=1)


def _scalars(products: Sequence[Mapping[str, Any]], key: str, dtype) -> np.ndarray:
    return np.asarray(
        [np.asarray(z[key]).reshape(-1)[0] for z in products], dtype=dtype
    )


def _combine_detector_products(
    product_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    chunk_seconds: float = 10.0,
    drop_freq_ids: Sequence[int] | None = None,
) -> dict[str, Path]:
    """Stack per-pilot detector products and write the canonical detector products.

    Frames are aligned by their (event, frame-in-file) identity. Pilots that
    processed different event sets are stacked over their common identities,
    and the frames dropped by each pilot are reported and recorded.
    ``drop_freq_ids`` removes whole pilots before alignment, and its values are
    chosen from the drop-curve of ``report_products``.
    """
    products = _load_sorted(product_paths)
    if drop_freq_ids:
        drop = {int(f) for f in drop_freq_ids}
        kept = [z for z in products
                if int(np.asarray(z.get("freq_id", -1)).reshape(-1)[0]) not in drop]
        excluded = len(products) - len(kept)
        if not kept:
            raise ValueError("combine: --drop excluded every per-pilot product")
        if excluded:
            print(f"[combine] --drop excluded {excluded} pilot(s): "
                  f"{sorted(drop)}", flush=True)
        products = kept
    invariant_notes = _check_invariants(
        products,
        ("schema_version", "nfft", "detector_window_samples", "sense",
         "detector_contract_json", "max_chunks_per_file", "num_input_streams",
         "weight_bank_sha256", "weight_manifest_sha256", "mask_rule",
         "detector_version", "pilot_below_data_db", "bin_enbw_hz",
         "dtv_bandwidth_hz", "pilot_capture_efficiency"),
        "detector geometry",
    )
    # The shared contract is validated once, before any output is written. The
    # invariant check has already shown that every selected product carries the
    # same document apart from the per-pilot designated windows, so one
    # validation covers the stack.
    contract = _detector_contract_from(products)
    if "fine_designated_bins_by_pilot" in invariant_notes:
        # The combined contract is scan-level provenance, so the union of the
        # per-pilot windows is recorded as one sorted list of unique bins. The
        # window of each pilot is kept in detector_provenance_by_pilot and in
        # the stats notes.
        contract = dict(contract)
        fine = dict(contract.get("fine_reduction") or {})
        fine["designated_bins"] = sorted({
            int(b)
            for w in invariant_notes["fine_designated_bins_by_pilot"] if w
            for b in w
        })
        contract["fine_reduction"] = fine
    products_full = products
    products, frame_index, align_info = _align_frames(products_full)
    nfft = int(np.asarray(products[0]["nfft"]))

    # The masked fraction is paired with the integrated spectra. The spectra are
    # accumulated by the analyzer over the full processed frame set of each
    # pilot and cannot be subset here, so the masked fraction is also computed
    # over the full set.
    def _masked_fraction(z: Mapping[str, Any]) -> float:
        rej = np.asarray(z["reject_mask"]).reshape(-1).astype(np.float64)
        n_valid = float(np.asarray(z["valid"]).reshape(-1).sum())
        return float(rej.sum() / n_valid) if n_valid > 0 else float("nan")

    masked_fraction = np.asarray(
        [_masked_fraction(z) for z in products_full], np.float64)

    physical_channel = _scalars(products, "physical_channel", np.int32)
    pilot_frequency_hz = _scalars(products, "pilot_frequency_hz", np.float64)
    chime_frequency_hz = _scalars(products, "chime_frequency_hz", np.float64)
    # freq_id is required by the current schema and gives the receiver coarse
    # channel for the later 6 MHz mask-expansion step.
    freq_id = _scalars(products, "freq_id", np.int64)

    p_target_u64 = _stack_cols(products, "p_target_u64", np.uint64)
    p_ref_lower_u64 = _stack_cols(products, "p_ref_lower_u64", np.uint64)
    p_ref_upper_u64 = _stack_cols(products, "p_ref_upper_u64", np.uint64)
    p_ref_sum_u64 = _stack_cols(products, "p_ref_sum_u64", np.uint64)
    coarse_power_ratio = _stack_cols(products, "coarse_power_ratio", np.float64)
    normalized_coarse_power_ratio_db = _stack_cols(products, "normalized_coarse_power_ratio_db", np.float64)
    pilot_excess_db = _stack_cols(products, "pilot_excess_db", np.float64)
    estimated_data_shelf_snr_db = _stack_cols(products, "estimated_data_shelf_snr_db", np.float64)
    # Per-pilot products name the mask reject_mask (1 = discard, positive
    # excess). The canonical combined outputs keep the field name mask.
    mask = _stack_cols(products, "reject_mask", np.uint8)
    valid = _stack_cols(products, "valid", np.uint8)
    baseband_power_linear = _stack_cols(products, "baseband_power_linear", np.float64)
    # The current schema always carries the exact quantized-weight null point.
    target_norm_sq = _scalars(products, "target_norm_sq", np.int64)
    reference_norm_sum_sq = _scalars(products, "reference_norm_sum_sq", np.int64)
    # The null power ratio is derived from the exact integer pair stored in the
    # per-pilot product, which has no float copy. The combined output publishes
    # the derived value, since the analysis layer reads it.
    null_power_ratio = np.asarray(
        [null_power_ratio_of(p) for p in products], dtype=np.float64
    )
    normalized_pilot_excess = _stack_cols(
        products, "normalized_pilot_excess", np.float64
    )

    # The integrated spectra have one 1-D [nfft] array per channel and are
    # stacked along the pilot axis to [n_pilots, nfft]. The masked fraction of a
    # channel is the number of valid, rejected frames over the number of valid
    # frames, and is NaN for a channel with no valid frames, such as an
    # out-of-band channel.
    spec_before = np.stack([
        np.asarray(z["integrated_spectrum_before_mask"], np.float64).reshape(-1)
        for z in products])
    spec_after = np.stack([
        np.asarray(z["integrated_spectrum_after_mask"], np.float64).reshape(-1)
        for z in products])

    # Sample rate for the spectra frequency axis is shared only when every
    # per-pilot product carries consistent timing metadata.
    sample_rate_hz = _common_sample_rate_hz(products)

    run_dir = Path(run_dir)
    ensure_run_dirs(run_dir)
    outputs: dict[str, Path] = {}
    outputs["detector_outputs"] = write_detector_outputs(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        p_target_u64=p_target_u64,
        p_ref_lower_u64=p_ref_lower_u64,
        p_ref_upper_u64=p_ref_upper_u64,
        p_ref_sum_u64=p_ref_sum_u64,
        # The fine powers stay in the per-pilot products (a combined archive
        # copy would be num_frames x num_pilots x 768 words).
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        pilot_excess_db=pilot_excess_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        mask=mask,
        valid=valid,
        target_norm_sq=target_norm_sq,
        reference_norm_sum_sq=reference_norm_sum_sq,
        null_power_ratio=null_power_ratio,
        normalized_pilot_excess=normalized_pilot_excess,
    )
    outputs["spectrogram_cache"] = write_spectrogram_cache(
        run_dir,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        frame_size_samples=nfft,
        sample_rate_hz=sample_rate_hz,
        valid=valid,
    )
    outputs["integrated_spectra"] = write_integrated_spectra(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        integrated_spectrum_before_mask=spec_before,
        integrated_spectrum_after_mask=spec_after,
        masked_fraction_by_channel=masked_fraction,
        sample_rate_hz=sample_rate_hz,
        nfft=nfft,
        freq_id=freq_id,
    )
    outputs["reductions_10s"] = write_reductions_npz(
        run_dir,
        frame_index=frame_index,
        frame_size_samples=nfft,
        sample_rate_hz=sample_rate_hz,
        chunk_seconds=float(chunk_seconds),
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    outputs["mask_summary"] = write_mask_summary(
        run_dir,
        physical_channel=[int(v) for v in physical_channel],
        pilot_frequency_hz=[float(v) for v in pilot_frequency_hz],
        chime_frequency_hz=[float(v) for v in chime_frequency_hz],
        mask=mask,
        valid=valid,
    )

    # run_config, stats, and input_manifest are written so that validate-products
    # accepts the scan output. They carry the schema-gated fields
    # (detector_contract, mask_policy, and geometry) and are labeled as
    # chime-scan provenance, since they do not come from a single
    # run_chime_analysis run.
    reference_placement = _combined_reference_placement_summary(products)
    if reference_placement is not None:
        contract = dict(contract)
        contract["reference_placement_summary"] = reference_placement
    mask_policy = normalized_positive_excess_policy()
    k = int(contract["detector_window_samples"])
    provenance_by_pilot = []
    for z in products:
        provenance_by_pilot.append({
            "physical_channel": int(np.asarray(z["physical_channel"]).reshape(-1)[0]),
            "freq_id": int(np.asarray(z["freq_id"]).reshape(-1)[0]),
            "weights_hash": str(np.asarray(z.get("weights_hash", "")).reshape(()).item()),
            "weight_bank_sha256": str(
                np.asarray(z.get("weight_bank_sha256", "")).reshape(()).item()
            ),
            "weight_manifest_sha256": str(
                np.asarray(z.get("weight_manifest_sha256", "")).reshape(()).item()
            ),
            "detector_version": str(np.asarray(z.get("detector_version", "")).reshape(()).item()),
            "mask_rule": str(np.asarray(z.get("mask_rule", "")).reshape(()).item()),
            "fine_designated_bins": [
                int(b) for b in
                np.asarray(z.get("fine_designated_bins", []), dtype=np.int64).reshape(-1)
            ],
        })
    common = {
        "source": "chime-scan",
        "physical_channels": [int(v) for v in physical_channel],
        "pilot_frequency_hz": [float(v) for v in pilot_frequency_hz],
        "chime_frequency_hz": [float(v) for v in chime_frequency_hz],
        "frame_size_samples": int(nfft),
        "detector_window_samples": k,
        "num_input_streams": int(np.asarray(products[0].get("num_input_streams", 0))),
        "mask_policy": mask_policy,
        "detector_contract": contract,
        "detector_provenance_by_pilot": provenance_by_pilot,
    }
    if reference_placement is not None:
        common["reference_placement_summary"] = reference_placement
    common["freq_id_by_pilot"] = [int(v) for v in freq_id]
    _write_json(run_dir / "run_config.json",
                {"schema_version": CHIME_RUN_CONFIG_SCHEMA_TOKEN, **common})
    if align_info.get("mode") == "event_keyed":
        identity_path = run_dir / "chime_frame_identity.npz"
        atomic_savez_compressed(
            identity_path,
            frame_event_key=np.asarray(align_info["frame_event_key"], dtype=str),
            frame_in_unit=np.asarray(align_info["frame_in_unit"], dtype=np.int64),
        )
        outputs["frame_identity"] = identity_path
    stats_alignment = {
        k: v for k, v in align_info.items()
        if k not in ("frame_event_key", "frame_in_unit")
    }
    _write_json(run_dir / "stats.json", {
        "schema_version": CHIME_STATS_SCHEMA_TOKEN,
        "num_frames": int(frame_index.size),
        "num_pilots": len(products),
        "combine_alignment": stats_alignment,
        **invariant_notes,
        "windows_per_stream": int(nfft) // k,
        "rational_overflow_count_by_pilot": [
            int(np.asarray(z.get("rational_overflow_count", 0))) for z in products
        ],
        **common,
    })
    _write_json(run_dir / "input_manifest.json", {
        "schema_version": SCAN_INPUT_MANIFEST_SCHEMA_TOKEN,
        "source": "chime-scan",
        "physical_channels": [int(v) for v in physical_channel],
        "input_files": sorted({
            str(x) for z in products
            for x in np.asarray(z.get("unit_keys", np.asarray([], dtype=object)))
            .reshape(-1).tolist()
        }),
    })
    outputs["run_config"] = run_dir / "run_config.json"
    outputs["stats"] = run_dir / "stats.json"
    outputs["input_manifest"] = run_dir / "input_manifest.json"
    return outputs


def _validate_staged_outputs(outputs: Mapping[str, Path]) -> None:
    for path in outputs.values():
        candidate = Path(path)
        if not candidate.is_file():
            raise RuntimeError(f"combine: staged output was not created: {candidate}")
        if candidate.suffix == ".npz":
            with np.load(candidate, allow_pickle=False) as archive:
                for name in archive.files:
                    np.asarray(archive[name])
        elif candidate.suffix == ".json":
            json.loads(candidate.read_text(encoding="utf-8"))


def _safe_relative_path(value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(
            f"combine: invalid path in publish transaction: {value!r}"
        )
    if relative.as_posix() not in CHIME_COMBINE_CANONICAL_RELATIVE_PATHS:
        raise RuntimeError(
            "combine: path is not a canonical output allowed in a publish "
            f"transaction: {relative}"
        )
    return relative


def _resolved_directory_root(path: Path, *, what: str) -> Path:
    root = Path(path).absolute()
    if root.is_symlink():
        raise RuntimeError(f"combine: {what} may not be a symlink: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"combine: cannot resolve {what}: {root}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"combine: {what} is not a directory: {root}")
    return resolved


def _safe_descendant(
    root: Path,
    relative: Path,
    *,
    what: str,
    require_file: bool = False,
) -> Path:
    """Return a lexical child after rejecting symlink traversal and escapes."""
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"combine: invalid {what} path: {relative}")
    root_path = Path(root).absolute()
    resolved_root = _resolved_directory_root(root_path, what=f"{what} root")
    candidate = root_path
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(
                f"combine: refusing symlinked {what} path component: {candidate}"
            )
    try:
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"combine: {what} escapes its resolved root: {candidate}"
        ) from exc
    if require_file and not candidate.is_file():
        raise RuntimeError(f"combine: required {what} is missing: {candidate}")
    return candidate


def _generation_manifest_payload(
    root: Path, relative_paths: Sequence[Path], *, state: str
) -> dict[str, Any]:
    files: dict[str, str] = {}
    for raw_relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        relative = _safe_relative_path(raw_relative)
        if relative.as_posix() == CHIME_COMBINE_GENERATION_MANIFEST_FILENAME:
            continue
        candidate = _safe_descendant(
            root, relative, what="generation output", require_file=True
        )
        digest = file_sha256(candidate)
        if digest is None:
            raise RuntimeError(f"combine: could not hash generation output {candidate}")
        files[relative.as_posix()] = digest
    return {
        "schema_version": CHIME_COMBINE_GENERATION_MANIFEST_SCHEMA,
        "generation_id": uuid.uuid4().hex,
        "state": str(state),
        "files": files,
    }


def _stage_generation_manifest(
    staged_outputs: Mapping[str, Path], staging_dir: Path
) -> dict[str, Path]:
    staging_dir = Path(staging_dir).absolute()
    _resolved_directory_root(staging_dir, what="transaction directory")
    outputs = {
        str(label): Path(path).absolute() for label, path in staged_outputs.items()
    }
    if _GENERATION_LABEL in outputs:
        raise RuntimeError("combine: duplicate generation-manifest output label")
    relative_paths = [
        _safe_relative_path(path.relative_to(staging_dir))
        for path in outputs.values()
    ]
    marker = _safe_descendant(
        staging_dir,
        Path(CHIME_COMBINE_GENERATION_MANIFEST_FILENAME),
        what="staged generation manifest",
    )
    payload = _generation_manifest_payload(
        staging_dir, relative_paths, state="committed"
    )
    atomic_write_json(marker, payload)
    outputs[_GENERATION_LABEL] = marker
    return outputs


def _write_recovery_generation_manifest(run_dir: Path) -> Path:
    relative_paths: list[Path] = []
    for name in CHIME_COMBINE_CANONICAL_RELATIVE_PATHS:
        if name == CHIME_COMBINE_GENERATION_MANIFEST_FILENAME:
            continue
        relative = Path(name)
        candidate = _safe_descendant(
            run_dir, relative, what="recovered canonical output"
        )
        if candidate.exists():
            if not candidate.is_file():
                raise RuntimeError(
                    f"combine: recovered canonical output is not a file: {candidate}"
                )
            relative_paths.append(relative)
    marker_relative = Path(CHIME_COMBINE_GENERATION_MANIFEST_FILENAME)
    marker = _safe_descendant(
        run_dir, marker_relative, what="recovery generation manifest"
    )
    payload = _generation_manifest_payload(
        run_dir, relative_paths, state="recovered"
    )
    atomic_write_json(marker, payload)
    return marker


def _durable_replace(source: Path, destination: Path) -> None:
    source_parent = Path(source).parent
    destination_parent = Path(destination).parent
    os.replace(source, destination)
    fsync_directory(destination_parent)
    if source_parent != destination_parent:
        fsync_directory(source_parent)


def _restore_file_from_backup(
    backup: Path,
    destination: Path,
    *,
    backup_root: Path,
    destination_root: Path,
    relative: Path,
) -> None:
    """Atomically restore a backup without consuming it (recovery is idempotent)."""
    backup = _safe_descendant(
        backup_root, relative, what="publish backup", require_file=True
    )
    destination = _safe_descendant(
        destination_root, relative, what="canonical restore destination"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _safe_descendant(
        destination_root, relative, what="canonical restore destination"
    )
    fd, temporary = create_temporary_sibling(
        destination, suffix=".restore.tmp"
    )
    os.close(fd)
    try:
        shutil.copy2(backup, temporary)
        fsync_file(temporary)
        _durable_replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_publish_journal(
    run_dir: Path,
) -> tuple[Path, list[dict[str, Any]], str]:
    _resolved_directory_root(run_dir, what="run directory")
    journal_path = _safe_descendant(
        run_dir, Path(CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME), what="publish journal"
    )
    if journal_path.is_symlink():
        raise RuntimeError(
            f"combine: publish journal may not be a symlink: {journal_path}"
        )
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"combine: cannot recover invalid publish journal {journal_path}"
        ) from exc
    if payload.get("schema_version") != _PUBLISH_JOURNAL_SCHEMA:
        raise RuntimeError(
            f"combine: unsupported publish journal schema in {journal_path}"
        )
    journal_owner = payload.get("owner_token")
    if not isinstance(journal_owner, str) or len(journal_owner) != 32:
        raise RuntimeError(
            f"combine: publish journal has an invalid owner token: {journal_path}"
        )
    transaction_name = str(payload.get("transaction_directory", ""))
    if (
        Path(transaction_name).name != transaction_name
        or not transaction_name.startswith(_TRANSACTION_DIR_PREFIX)
    ):
        raise RuntimeError(
            f"combine: invalid transaction directory in {journal_path}"
        )
    transaction_dir = _safe_descendant(
        run_dir, Path(transaction_name), what="transaction directory"
    )
    if not transaction_dir.is_dir():
        raise RuntimeError(
            f"combine: transaction directory is missing: {transaction_dir}"
        )
    _resolved_directory_root(transaction_dir, what="transaction directory")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RuntimeError(f"combine: publish journal has no entries: {journal_path}")
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise RuntimeError(f"combine: invalid publish journal entry: {raw!r}")
        relative = _safe_relative_path(raw.get("relative_path", ""))
        if relative in seen:
            raise RuntimeError(
                f"combine: duplicate publish path in journal: {relative}"
            )
        seen.add(relative)
        had_previous = raw.get("had_previous")
        if not isinstance(had_previous, bool):
            raise RuntimeError(
                f"combine: invalid had_previous flag in journal: {raw!r}"
            )
        entries.append(
            {
                "label": str(raw.get("label", "")),
                "relative_path": relative,
                "had_previous": had_previous,
            }
        )
    return transaction_dir, entries, journal_owner


def _unlink_owned_journal(
    journal_path: Path,
    *,
    expected_journal_owner: str,
    ownership: _PublishOwnership,
) -> None:
    ownership.assert_owned()
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "combine: refusing to unlink an unreadable publish journal"
        ) from exc
    if payload.get("owner_token") != expected_journal_owner:
        raise RuntimeError(
            "combine: refusing to unlink a publish journal owned by another token"
        )
    journal_path.unlink()
    fsync_directory(journal_path.parent)


def _recover_interrupted_publish(
    run_dir: Path, ownership: _PublishOwnership
) -> bool:
    """Roll back a journaled partial publication before starting new work."""
    ownership.assert_owned()
    destination = Path(run_dir).absolute()
    _resolved_directory_root(destination, what="run directory")
    journal_path = _safe_descendant(
        destination,
        Path(CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME),
        what="publish journal",
    )
    if not journal_path.exists() and not journal_path.is_symlink():
        return False
    transaction_dir, entries, journal_owner = _load_publish_journal(destination)
    backup_dir = _safe_descendant(
        transaction_dir, Path("_previous_outputs"), what="backup directory"
    )
    if backup_dir.exists():
        if not backup_dir.is_dir():
            raise RuntimeError(
                f"combine: publish backup root is not a directory: {backup_dir}"
            )
        _resolved_directory_root(backup_dir, what="backup directory")

    # Every path named by the journal is validated before any canonical file is
    # changed. The recovery generation manifest lists the full allowlist, so a
    # symlink anywhere in that namespace is also refused here.
    for name in CHIME_COMBINE_CANONICAL_RELATIVE_PATHS:
        _safe_descendant(
            destination,
            Path(name),
            what="canonical generation namespace",
        )
    validated: list[tuple[dict[str, Any], Path, Path | None]] = []
    for entry in entries:
        relative = entry["relative_path"]
        canonical = _safe_descendant(
            destination, relative, what="canonical recovery destination"
        )
        staged = _safe_descendant(
            transaction_dir, relative, what="staged recovery source"
        )
        if staged.exists() and not staged.is_file():
            raise RuntimeError(
                f"combine: staged recovery source is not a file: {staged}"
            )
        backup: Path | None = None
        if entry["had_previous"]:
            if not backup_dir.is_dir():
                raise RuntimeError(
                    "combine: cannot recover interrupted publication because "
                    f"backup directory is missing: {backup_dir}"
                )
            backup = _safe_descendant(
                backup_dir,
                relative,
                what="publish backup",
                require_file=True,
            )
        validated.append((entry, canonical, backup))

    for entry, canonical, backup in validated:
        ownership.assert_owned()
        relative = entry["relative_path"]
        if entry["had_previous"]:
            if backup is None:
                raise RuntimeError(
                    "validated publish journal is missing a required backup"
                )
            _restore_file_from_backup(
                backup,
                canonical,
                backup_root=backup_dir,
                destination_root=destination,
                relative=relative,
            )
        elif canonical.exists():
            canonical.unlink()
            fsync_directory(canonical.parent)
    # A rollback also writes a new generation ID, so a validator that reads
    # during even a fast failed and recovered publication cannot take its reads
    # for one stable generation.
    _write_recovery_generation_manifest(destination)
    _unlink_owned_journal(
        journal_path,
        expected_journal_owner=journal_owner,
        ownership=ownership,
    )
    shutil.rmtree(transaction_dir, ignore_errors=False)
    fsync_directory(destination)
    return True


def _prepare_publish_transaction(
    staged_outputs: Mapping[str, Path],
    staging_dir: Path,
    run_dir: Path,
    ownership: _PublishOwnership,
) -> list[dict[str, Any]]:
    """Durably back up the old generation and record rollback instructions."""
    ownership.assert_owned()
    run_dir = Path(run_dir).absolute()
    staging_dir = Path(staging_dir).absolute()
    run_root = _resolved_directory_root(run_dir, what="run directory")
    staging_root = _resolved_directory_root(
        staging_dir, what="transaction directory"
    )
    try:
        staging_root.relative_to(run_root)
    except ValueError as exc:
        raise RuntimeError(
            "combine: transaction directory must resolve beneath the run directory"
        ) from exc
    backup_dir = _safe_descendant(
        staging_dir, Path("_previous_outputs"), what="backup directory"
    )
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for label, staged_value in staged_outputs.items():
        staged = Path(staged_value).absolute()
        try:
            relative = _safe_relative_path(staged.relative_to(staging_dir))
        except ValueError as exc:
            raise RuntimeError(
                f"combine: staged output is outside transaction root: {staged}"
            ) from exc
        if relative in seen:
            raise RuntimeError(f"combine: duplicate staged output path: {relative}")
        seen.add(relative)
        staged = _safe_descendant(
            staging_dir, relative, what="staged generation output", require_file=True
        )
        destination = _safe_descendant(
            run_dir, relative, what="canonical publish destination"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _safe_descendant(
            run_dir, relative, what="canonical publish destination"
        )
        had_previous = destination.exists()
        if had_previous:
            if not destination.is_file():
                raise RuntimeError(
                    f"combine: output destination is not a file: {destination}"
                )
            mode = stat.S_IMODE(destination.stat().st_mode)
            staged.chmod(mode)
            backup = _safe_descendant(
                staging_dir,
                Path("_previous_outputs") / relative,
                what="publish backup",
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup = _safe_descendant(
                staging_dir,
                Path("_previous_outputs") / relative,
                what="publish backup",
            )
            shutil.copy2(destination, backup)
            fsync_file(backup)
            fsync_directory(backup.parent)
            fsync_directory(backup_dir)
            fsync_directory(staging_dir)
        fsync_file(staged)
        fsync_directory(staged.parent)
        fsync_directory(staging_dir)
        entries.append(
            {
                "label": str(label),
                "relative_path": relative,
                "had_previous": had_previous,
            }
        )
    journal_payload = {
        "schema_version": _PUBLISH_JOURNAL_SCHEMA,
        "owner_token": ownership.owner_token,
        "transaction_directory": staging_dir.name,
        "entries": [
            {
                "label": entry["label"],
                "relative_path": entry["relative_path"].as_posix(),
                "had_previous": entry["had_previous"],
            }
            for entry in entries
        ],
    }
    journal_path = _safe_descendant(
        run_dir, Path(CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME), what="publish journal"
    )
    if journal_path.exists() or journal_path.is_symlink():
        raise RuntimeError(
            "combine: refusing to replace an existing publish journal"
        )
    ownership.assert_owned()
    atomic_write_json(journal_path, journal_payload)
    return entries


def _publish_output_set(
    staged_outputs: Mapping[str, Path], staging_dir: Path, run_dir: Path
) -> dict[str, Path]:
    """Publish one generation with a durable, restart-recoverable rollback log."""
    run_dir = Path(run_dir).absolute()
    staging_dir = Path(staging_dir).absolute()
    _resolved_directory_root(run_dir, what="run directory")
    staged_outputs = _stage_generation_manifest(staged_outputs, staging_dir)
    _validate_staged_outputs(staged_outputs)
    with _exclusive_publish_ownership(run_dir) as ownership:
        journal_path = _safe_descendant(
            run_dir,
            Path(CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME),
            what="publish journal",
        )
        if journal_path.exists() or journal_path.is_symlink():
            _recover_interrupted_publish(run_dir, ownership)
        entries = _prepare_publish_transaction(
            staged_outputs, staging_dir, run_dir, ownership
        )
        try:
            for entry in entries:
                ownership.assert_owned()
                relative = entry["relative_path"]
                staged = _safe_descendant(
                    staging_dir,
                    relative,
                    what="staged generation output",
                    require_file=True,
                )
                destination = _safe_descendant(
                    run_dir, relative, what="canonical publish destination"
                )
                _durable_replace(staged, destination)
        except BaseException:
            _recover_interrupted_publish(run_dir, ownership)
            raise
        _unlink_owned_journal(
            journal_path,
            expected_journal_owner=ownership.owner_token,
            ownership=ownership,
        )
        return {
            entry["label"]: run_dir / entry["relative_path"]
            for entry in entries
        }


def _cleanup_unreferenced_staging(run_dir: Path, staging_dir: Path) -> bool:
    """Remove staging unless a valid durable journal names this transaction."""
    run = Path(run_dir).absolute()
    staging = Path(staging_dir).absolute()
    journal = run / CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    preserve = False
    if journal.exists() or journal.is_symlink():
        try:
            transaction, _entries, _owner = _load_publish_journal(run)
        except RuntimeError:
            # An unreadable journal may describe a partial publication whose
            # only rollback copy is this tree. The staging tree is kept, since
            # without a valid journal that names a different transaction it
            # cannot be shown to belong to a concurrent process that lost the
            # lock.
            preserve = True
        else:
            preserve = transaction == staging
    if preserve:
        return False
    shutil.rmtree(staging, ignore_errors=True)
    fsync_directory(run)
    return True


def combine_detector_products(
    product_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    chunk_seconds: float = 10.0,
    drop_freq_ids: Sequence[int] | None = None,
) -> dict[str, Path]:
    """Combine per-pilot products and publish the canonical set as one generation.

    The full set is first built in a staging directory under ``run_dir``.
    Returns the published paths keyed by output label.
    """
    destination = Path(run_dir)
    destination_existed = destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=_TRANSACTION_DIR_PREFIX,
            dir=destination,
        )
    )
    try:
        staged_outputs = _combine_detector_products(
            product_paths,
            staging,
            chunk_seconds=chunk_seconds,
            drop_freq_ids=drop_freq_ids,
        )
        return _publish_output_set(staged_outputs, staging, destination)
    finally:
        if _cleanup_unreferenced_staging(destination, staging):
            if not destination_existed:
                try:
                    destination.rmdir()
                except OSError:
                    pass


__all__ = [
    "CombineDuplicateIdentityError",
    "CombineEmptyIntersectionError",
    "combine_detector_products",
    "report_products",
]
