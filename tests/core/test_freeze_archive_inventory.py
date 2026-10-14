import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_DATA_SHELF_SNR_DB,
    ARCHIVED_FINE_NULL_BULK_EXCEEDANCE_FRACTION,
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB,
    ARCHIVED_NORMALIZED_PILOT_EXCESS,
    ARCHIVED_PILOT_EXCESS_DB,
)


SCRIPT = Path(__file__).parents[2] / "scripts" / "freeze_archive_inventory.py"
SPEC = importlib.util.spec_from_file_location("freeze_archive_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
freeze_archive_inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_archive_inventory)


def _row(event, freq_id, name, n_frames=1.0):
    return {
        "scope": "scope.raw",
        "event": str(event),
        "name": name,
        "size_bytes": 100,
        "common_path": f"cadc:data/{event}",
        "freq_id": freq_id,
        "n_frames": n_frames,
    }


def _jsonl(rows):
    return b"".join((json.dumps(row) + "\n").encode() for row in rows)


def _npz(unit_order, frame_unit_index, freq_id, *, misalign_frame=False):
    unit_order = list(unit_order)
    frame_unit_index = np.asarray(frame_unit_index, dtype=np.int64)
    frame_count = len(frame_unit_index)
    unit_count = len(unit_order)
    frame_in_unit = []
    next_frame = {}
    for unit_index in frame_unit_index:
        index = int(unit_index)
        frame_in_unit.append(next_frame.get(index, 0))
        next_frame[index] = next_frame.get(index, 0) + 1
    valid_count = frame_count + int(misalign_frame)
    stream = io.BytesIO()
    archived_fields = {
        ARCHIVED_COARSE_POWER_RATIO: np.ones(
            (frame_count, 1), dtype=np.float64
        ),
        ARCHIVED_FINE_POWER_RATIO: np.ones((frame_count, 1), dtype=np.float64),
        ARCHIVED_FINE_NULL_BULK_EXCEEDANCE_FRACTION: np.zeros(
            (frame_count, 1), dtype=np.float64
        ),
        ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB: np.zeros(
            (frame_count, 1), dtype=np.float64
        ),
        ARCHIVED_PILOT_EXCESS_DB: np.zeros(
            (frame_count, 1), dtype=np.float64
        ),
        ARCHIVED_DATA_SHELF_SNR_DB: np.zeros(
            (frame_count, 1), dtype=np.float64
        ),
        ARCHIVED_NORMALIZED_PILOT_EXCESS: np.zeros(
            (frame_count, 1), dtype=np.float64
        ),
    }
    np.savez_compressed(
        stream,
        schema_version=np.asarray("pilotproxy_detector_datatrawl_v3"),
        detector_version=np.asarray(
            "pilot-proxy/1.0.0 kernel=2.1.0 kernel_sha256=" + "a" * 64
        ),
        nfft=np.asarray(16384, dtype=np.int64),
        detector_window_samples=np.asarray(128, dtype=np.int64),
        num_input_streams=np.asarray(2048, dtype=np.int64),
        max_chunks_per_file=np.asarray(-1, dtype=np.int64),
        freq_id=np.asarray(freq_id, dtype=np.int64),
        unit_keys=np.asarray(unit_order),
        unit_order=np.asarray(unit_order),
        source_event_keys=np.asarray(unit_order),
        unit_time0_ctime=np.arange(unit_count, dtype=np.float64),
        unit_time0_fpga=np.arange(unit_count, dtype=np.uint64),
        unit_event_id=np.arange(unit_count, dtype=np.int64),
        unit_delta_time=np.ones(unit_count, dtype=np.float64),
        archive_version=np.full(unit_count, "fixture"),
        frame_index=np.arange(frame_count, dtype=np.int64),
        p_target_u64=np.ones((frame_count, 1), dtype=np.uint64),
        p_ref_sum_u64=np.ones((frame_count, 1), dtype=np.uint64),
        fine_cfar_location=np.zeros((frame_count, 1), dtype=np.float64),
        fine_cfar_scale=np.ones((frame_count, 1), dtype=np.float64),
        fine_cfar_threshold=np.ones((frame_count, 1), dtype=np.float64),
        fine_cfar_mode=np.zeros((frame_count, 1), dtype=np.uint8),
        fine_detected_count=np.zeros((frame_count, 1), dtype=np.int64),
        reject_mask=np.zeros((frame_count, 1), dtype=np.uint8),
        valid=np.ones((valid_count, 1), dtype=np.uint8),
        baseband_power_linear=np.ones((frame_count, 1), dtype=np.float64),
        frame_unit_index=frame_unit_index,
        frame_in_unit=np.asarray(frame_in_unit, dtype=np.int64),
        **archived_fields,
    )
    return stream.getvalue()


def _archives(
    tmp_path,
    *,
    omit_last=False,
    misalign_first=False,
    pending=False,
):
    rows = [
        _row(10, 506, "a.h5"),
        _row(11, 506, "b.h5", 0.5),
        _row(20, 521, "c.h5"),
        _row(21, 521, "d.h5"),
    ]
    uris = [f"{row['common_path']}/{row['name']}" for row in rows]
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("survey/inventory.jsonl", _jsonl(rows))
        archive.writestr(
            "survey/inventory.meta.json",
            json.dumps({
                "datatrawl_inventory": 1,
                "name": "source",
                "telescope": "chime",
                "source": "cadc-datatrail",
            }) + "\n",
        )
        event_keys = [f"scope.raw|{row['event']}" for row in rows]
        enum_keys = event_keys + (["scope.raw|99"] if pending else [])
        archive.writestr(
            "survey/enum_cache.json",
            json.dumps({key: ["chime"] for key in enum_keys}) + "\n",
        )
        archive.writestr(
            "survey/surveyed_events.txt", "\n".join(event_keys) + "\n"
        )
        archive.writestr(
            "survey/attempts.json",
            json.dumps({"scope.raw|99": 2} if pending else {}) + "\n",
        )
        archive.writestr("survey/incomplete_events.txt", "")
        archive.writestr("survey/no_files_events.jsonl", "")
    products = tmp_path / "products.zip"
    with zipfile.ZipFile(products, "w") as archive:
        archive.writestr(
            "products/506.npz",
            _npz(uris[:2], [0], 506, misalign_frame=misalign_first),
        )
        if not omit_last:
            archive.writestr("products/521.npz", _npz([uris[2]], [0], 521))
        archive.writestr(
            "products/quarantine.jsonl",
            json.dumps({
                "quarantine_key": "21:521",
                "key": uris[3],
                "reason": "unreadable",
            }) + "\n",
        )
    return source, products, rows


def _pending_resolution(path):
    path.write_text(json.dumps({
        "schema": "chime_pending_archive_resolution_v1",
        "checked_at": "2026-08-25T03:36:57Z",
        "method": "authenticated_archive_metadata",
        "reader_minimum_bytes": 1048576,
        "selection": [506, 521],
        "events": [{
            "scope": "scope.raw",
            "event": "99",
            "status": "no_selected_files",
            "common_path": "cadc:CHIMEFRB/data/99",
            "absent_freq_ids": [506, 521],
            "usable": [],
            "subfloor": [],
            "errors": [],
        }],
    }) + "\n")
    return path


def test_freeze_inventory_preserves_rows_and_accounts_for_exclusions(tmp_path):
    source, products, rows = _archives(tmp_path)
    output = tmp_path / "frozen"
    expected = {
        "source_units": 4,
        "product_members": 2,
        "zero_frame_units": 1,
        "quarantine_units": 1,
        "excluded_units": 2,
        "frozen_units": 2,
        "frozen_events": 2,
    }

    manifest = freeze_archive_inventory.freeze_inventory(
        source,
        products,
        output,
        selection=[506, 521],
        expected=expected,
    )

    assert (output / "inventory.source.jsonl").read_bytes() == _jsonl(rows)
    assert (output / "inventory.jsonl").read_bytes() == _jsonl([rows[0], rows[2]])
    exclusions = [
        json.loads(line)
        for line in (output / "exclusions.jsonl").read_text().splitlines()
    ]
    assert [row["reasons"] for row in exclusions] == [
        ["prior_product_zero_frames"],
        ["historical_quarantine"],
    ]
    assert manifest["decisions"]["partial_run_acknowledgement"] is False
    assert manifest["decisions"]["terminal_product"] == (
        "per_pilot_v5_products_for_selected_freq_ids"
    )

    repeated = freeze_archive_inventory.freeze_inventory(
        source,
        products,
        output,
        selection=[506, 521],
        expected=expected,
    )
    assert repeated == manifest


def test_freeze_inventory_refuses_unaccounted_unit(tmp_path):
    source, products, _rows = _archives(tmp_path, omit_last=True)

    with pytest.raises(ValueError, match="does not close"):
        freeze_archive_inventory.freeze_inventory(
            source, products, tmp_path / "frozen", selection=[506, 521]
        )


def test_freeze_inventory_refuses_changed_output(tmp_path):
    source, products, _rows = _archives(tmp_path)
    output = tmp_path / "frozen"
    freeze_archive_inventory.freeze_inventory(
        source, products, output, selection=[506, 521]
    )
    (output / "inventory.jsonl").write_text("changed\n")

    with pytest.raises(ValueError, match="existing output differs"):
        freeze_archive_inventory.freeze_inventory(
            source, products, output, selection=[506, 521]
        )


def test_freeze_inventory_refuses_misaligned_product_frames(tmp_path):
    source, products, _rows = _archives(tmp_path, misalign_first=True)

    with pytest.raises(ValueError, match="per-frame arrays are not aligned"):
        freeze_archive_inventory.freeze_inventory(
            source, products, tmp_path / "frozen", selection=[506, 521]
        )


def test_freeze_inventory_requires_pending_event_resolution(tmp_path):
    source, products, _rows = _archives(tmp_path, pending=True)

    with pytest.raises(ValueError, match="unresolved event"):
        freeze_archive_inventory.freeze_inventory(
            source, products, tmp_path / "frozen", selection=[506, 521]
        )


def test_freeze_inventory_accepts_exact_pending_event_resolution(tmp_path):
    source, products, _rows = _archives(tmp_path, pending=True)
    resolution = _pending_resolution(tmp_path / "pending.json")

    manifest = freeze_archive_inventory.freeze_inventory(
        source,
        products,
        tmp_path / "frozen",
        pending_resolution=resolution,
        selection=[506, 521],
    )

    assert manifest["accounting"]["source_pending_events"] == 1
    assert manifest["accounting"]["resolved_pending_events"] == 1
    assert manifest["decisions"]["source_survey_completeness"] == (
        "complete_with_pending_event_resolution"
    )
