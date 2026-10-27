"""Tests that the exclusion and curation pipeline can be re-derived.

``scripts/freeze_archive_inventory.py`` turns a survey inventory and a
detector-product archive into the frozen production bundle
(``inventory.jsonl``, ``exclusions.jsonl``, ``inventory_manifest.json``).
``tests/core/test_freeze_archive_inventory.py`` covers the normal path and a
few refusals. This module covers the parts that a third party must trust to
re-derive the bundle from scratch against a fresh survey:

* The frame-estimate column of the ledger is read by field name, and the
  survey renamed that field from ``n_frames`` to ``n_frames_estimate``.
* ``source_line`` is a positional index into the source inventory, so it is
  not a stable identifier across surveys.
* The inventory-product join is a string join on ``common_path + "/" + name``,
  which ties the freeze directly to the ``cadc:CHIMEFRB/`` collection-prefix
  normalization in ``pilot_proxy.archive.datatrail_client``.
* The closure is asserted in both directions, and the zero-frame and
  quarantine sets are disjoint, so ``reasons`` holds a single value.

The tests of the last section check the published accounting of the Aug-3
bundle when that bundle is present on this machine.
"""
import json
import os
import zipfile
from pathlib import Path

import pytest

from test_freeze_archive_inventory import (  # noqa: F401  (shared fixtures)
    _jsonl,
    _npz,
    freeze_archive_inventory,
)

# One CHIME baseband frame is nfft * n_feeds bytes of packed payload.
BYTES_PER_FRAME = 16384 * 2048

BUNDLE = Path(os.environ.get("PILOT_PROXY_ARCHIVE_BUNDLE", "archive_inputs/chime-pilots-v5"))
requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "exclusions.jsonl").is_file(),
    reason="frozen chime-pilots-v5 bundle is not present",
)


# fixture builders
def _row(event, freq_id, name, size, *, estimate_field="n_frames",
         collection="cadc:CHIMEFRB/"):
    """Return one inventory row.

    ``estimate_field`` selects the survey schema. The Aug-3 survey wrote
    ``"n_frames"`` (a float), and
    ``pilot_proxy.chime.baseband_reader.BasebandReader.annotate_row`` writes
    ``"n_frames_estimate"`` (an int). ``None`` omits the column.
    """
    row = {
        "scope": "scope.raw",
        "event": str(event),
        "name": name,
        "size_bytes": size,
        "common_path": collection + "data/" + str(event),
        "freq_id": freq_id,
    }
    if estimate_field == "n_frames":
        row["n_frames"] = round(size / BYTES_PER_FRAME, 4)
    elif estimate_field == "n_frames_estimate":
        row["n_frames_estimate"] = int(size // BYTES_PER_FRAME)
    return row


def _default_rows(**kwargs):
    return [
        _row(10, 506, "a.h5", 3 * BYTES_PER_FRAME, **kwargs),
        _row(11, 506, "b.h5", BYTES_PER_FRAME // 2, **kwargs),  # < 1 frame
        _row(20, 521, "c.h5", 3 * BYTES_PER_FRAME, **kwargs),
        _row(21, 521, "d.h5", 3 * BYTES_PER_FRAME, **kwargs),
    ]


def _uri(row):
    return row["common_path"].rstrip("/") + "/" + row["name"].lstrip("/")


def _source_zip(path, rows):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("survey/inventory.jsonl", _jsonl(rows))
        archive.writestr("survey/inventory.meta.json", json.dumps({
            "datatrawl_inventory": 1,
            "name": "source",
            "telescope": "chime",
            "source": "cadc-datatrail",
        }) + "\n")
        keys = sorted({"scope.raw|" + row["event"] for row in rows})
        archive.writestr(
            "survey/enum_cache.json",
            json.dumps({key: ["chime"] for key in keys}) + "\n",
        )
        archive.writestr("survey/surveyed_events.txt", "\n".join(keys) + "\n")
        archive.writestr("survey/attempts.json", "{}\n")
        archive.writestr("survey/incomplete_events.txt", "")
        archive.writestr("survey/no_files_events.jsonl", "")
    return path


def _product_zip(path, *, unit_uris_506, unit_uris_521, quarantine=(),
                 frames_506=(0,), frames_521=(0,)):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "products/506.npz",
            _npz(list(unit_uris_506), list(frames_506), 506),
        )
        archive.writestr(
            "products/521.npz",
            _npz(list(unit_uris_521), list(frames_521), 521),
        )
        archive.writestr(
            "products/quarantine.jsonl",
            "".join(json.dumps(row) + "\n" for row in quarantine),
        )
    return path


def _build(tmp_path, rows, *, quarantine=(), product_uris=None):
    """Return source and product archives whose evidence closes over ``rows``."""
    by_freq = {}
    for row in rows:
        by_freq.setdefault(row["freq_id"], []).append(_uri(row))
    quarantined = {row["key"] for row in quarantine}
    if product_uris is None:
        product_uris = {
            freq: [uri for uri in uris if uri not in quarantined]
            for freq, uris in by_freq.items()
        }
    source = _source_zip(tmp_path / "source.zip", rows)
    products = _product_zip(
        tmp_path / "products.zip",
        unit_uris_506=product_uris.get(506, []),
        unit_uris_521=product_uris.get(521, []),
        quarantine=quarantine,
    )
    return source, products


def _freeze(tmp_path, rows, *, name="frozen", **kwargs):
    source, products = _build(tmp_path, rows, **kwargs)
    output = tmp_path / name
    manifest = freeze_archive_inventory.freeze_inventory(
        source, products, output, selection=[506, 521]
    )
    ledger = [
        json.loads(line)
        for line in (output / "exclusions.jsonl").read_text().splitlines()
    ]
    return manifest, ledger, output


# survey schema drift in the frame-estimate column, which is read by field name
def test_ledger_records_the_legacy_frame_estimate_column(tmp_path):
    """The ledger reads the ``n_frames`` column (float) written by the Aug-3 survey."""
    _manifest, ledger, _out = _freeze(
        tmp_path, _default_rows(estimate_field="n_frames")
    )
    excluded = {row["name"]: row for row in ledger}
    assert excluded["b.h5"]["n_frames_estimate"] == pytest.approx(0.5)
    assert excluded["b.h5"]["below_one_frame_estimate"] is True
    assert _manifest["accounting"]["low_estimate_exclusions"] == 1


def test_ledger_records_the_current_survey_frame_estimate_column(tmp_path):
    """The ledger reads the ``n_frames_estimate`` column (int) of a fresh survey."""
    manifest, ledger, _out = _freeze(
        tmp_path, _default_rows(estimate_field="n_frames_estimate")
    )
    excluded = {row["name"]: row for row in ledger}
    assert excluded["b.h5"]["n_frames_estimate"] == 0
    assert excluded["b.h5"]["below_one_frame_estimate"] is True
    assert manifest["accounting"]["low_estimate_exclusions"] == 1


def test_current_frame_estimate_takes_precedence_over_legacy(tmp_path):
    rows = _default_rows(estimate_field="n_frames_estimate")
    for row in rows:
        row["n_frames"] = 999
    manifest, ledger, _out = _freeze(tmp_path, rows)
    excluded = {row["name"]: row for row in ledger}
    assert excluded["b.h5"]["n_frames_estimate"] == 0
    assert excluded["b.h5"]["below_one_frame_estimate"] is True
    assert manifest["accounting"]["low_estimate_exclusions"] == 1
    assert manifest["accounting"]["zero_frame_units"] == 2
    assert manifest["accounting"]["excluded_units"] == 2


def test_missing_frame_estimate_column_is_not_refused(tmp_path):
    """A row without an estimate column still freezes."""
    manifest, ledger, _out = _freeze(
        tmp_path, _default_rows(estimate_field=None)
    )
    assert ledger[0]["n_frames_estimate"] is None
    assert manifest["accounting"]["low_estimate_exclusions"] == 0


# source_line is positional, and the ledger is sorted by identity
def test_source_line_indexes_the_frozen_source_inventory(tmp_path):
    rows = _default_rows()
    _manifest, ledger, output = _freeze(tmp_path, rows)
    lines = (output / "inventory.source.jsonl").read_bytes().splitlines()
    for entry in ledger:
        row = json.loads(lines[entry["source_line"] - 1])
        assert _uri(row) == entry["source_uri"]
        assert row["name"] == entry["name"]


def test_source_line_is_positional_and_shifts_when_the_survey_reorders(tmp_path):
    """A reordered survey gives the same physical unit a different source_line."""
    rows = _default_rows()
    # The product evidence is held fixed, so the same physical units have zero
    # frames in both runs and only the line order of the inventory changes.
    product_uris = {
        506: [_uri(rows[0]), _uri(rows[1])],
        521: [_uri(rows[2]), _uri(rows[3])],
    }
    reordered = tmp_path / "r"
    reordered.mkdir()
    _m1, ledger_a, _o1 = _freeze(
        tmp_path, rows, name="a", product_uris=product_uris)
    _m2, ledger_b, _o2 = _freeze(
        reordered, list(reversed(rows)), name="b", product_uris=product_uris)

    identity = ("scope", "event", "name", "freq_id")
    assert [tuple(r[k] for k in identity) for r in ledger_a] == \
           [tuple(r[k] for k in identity) for r in ledger_b], \
        "ledger order is identity-sorted, so it must survive reordering"
    # b.h5 is source line 2 in survey order and line 3 when reversed.
    assert [r["source_line"] for r in ledger_a] == [2, 4]
    assert [r["source_line"] for r in ledger_b] == [3, 1]
    assert ledger_a[0]["source_line"] != ledger_b[0]["source_line"], \
        "source_line is a positional cursor, not a stable unit id"


def test_ledger_is_sorted_by_identity_tuple(tmp_path):
    rows = _default_rows()
    # Quarantine a unit that sorts before the zero-frame one.
    quarantine = [{
        "quarantine_key": "10:506",
        "key": _uri(rows[0]),
        "reason": "probe/read: truncated",
    }]
    _manifest, ledger, _out = _freeze(tmp_path, rows, quarantine=quarantine)
    keys = [(r["scope"], r["event"], r["name"], r["freq_id"]) for r in ledger]
    assert keys == sorted(keys)
    assert [r["reasons"] for r in ledger] == [
        ["historical_quarantine"],
        ["prior_product_zero_frames"],
    ]


# the inventory-product join is a raw string join on the URI
def test_collection_prefix_drift_breaks_evidence_closure(tmp_path):
    """Bare-replica URIs in the inventory break the join with the product unit_order.

    Datatrail returns some replicas bare, and
    ``datatrail_client._restore_collection`` puts the ``cadc:CHIMEFRB/``
    prefix back. If that normalization regresses, the freeze cannot join a
    fresh inventory to the Aug-3 products, and this test shows the resulting
    failure.
    """
    bare_rows = _default_rows(collection="cadc:")
    prefixed = _default_rows(collection="cadc:CHIMEFRB/")
    product_uris = {
        506: [_uri(prefixed[0]), _uri(prefixed[1])],
        521: [_uri(prefixed[2]), _uri(prefixed[3])],
    }
    with pytest.raises(ValueError, match=r"does not close: missing=4, extra=4"):
        _freeze(tmp_path, bare_rows, product_uris=product_uris)


def test_double_slash_in_common_path_breaks_evidence_closure(tmp_path):
    """``_uri`` strips only a trailing slash, so an interior ``//`` is kept."""
    rows = _default_rows()
    product_uris = {
        506: [_uri(rows[0]), _uri(rows[1])],
        521: [_uri(rows[2]), _uri(rows[3])],
    }
    rows[0]["common_path"] = rows[0]["common_path"].replace(
        "cadc:CHIMEFRB/", "cadc:CHIMEFRB//"
    )
    with pytest.raises(ValueError, match=r"does not close: missing=1, extra=1"):
        _freeze(tmp_path, rows, product_uris=product_uris)


# closure and disjointness assertions
def test_freeze_refuses_product_unit_absent_from_the_inventory(tmp_path):
    rows = _default_rows()
    product_uris = {
        506: [_uri(rows[0]), _uri(rows[1])],
        521: [_uri(rows[2]), _uri(rows[3]), "cadc:CHIMEFRB/data/99/z.h5"],
    }
    with pytest.raises(ValueError, match=r"does not close: missing=0, extra=1"):
        _freeze(tmp_path, rows, product_uris=product_uris)


def test_freeze_refuses_quarantined_unit_that_is_also_a_product_unit(tmp_path):
    rows = _default_rows()
    quarantine = [{
        "quarantine_key": "10:506",
        "key": _uri(rows[0]),
        "reason": "probe/read: truncated",
    }]
    product_uris = {
        506: [_uri(rows[0]), _uri(rows[1])],
        521: [_uri(rows[2]), _uri(rows[3])],
    }
    with pytest.raises(ValueError, match="quarantine unit sets overlap"):
        _freeze(tmp_path, rows, quarantine=quarantine,
                product_uris=product_uris)


def test_freeze_refuses_duplicate_inventory_uri(tmp_path):
    rows = _default_rows()
    product_uris = {
        506: [_uri(rows[0]), _uri(rows[1])],
        521: [_uri(rows[2]), _uri(rows[3])],
    }
    # Same URI, distinct identity tuple (freq_id differs).
    rows.append(_row(10, 521, "a.h5", 3 * BYTES_PER_FRAME))
    with pytest.raises(ValueError, match="duplicate inventory URI"):
        _freeze(tmp_path, rows, product_uris=product_uris)


def test_reasons_are_single_valued_and_from_the_closed_vocabulary(tmp_path):
    rows = _default_rows()
    quarantine = [{
        "quarantine_key": "21:521",
        "key": _uri(rows[3]),
        "reason": "probe/read: bad object header version number",
    }]
    _manifest, ledger, _out = _freeze(tmp_path, rows, quarantine=quarantine)
    vocabulary = {"prior_product_zero_frames", "historical_quarantine"}
    for entry in ledger:
        assert len(entry["reasons"]) == 1
        assert set(entry["reasons"]) <= vocabulary
    assert {"prior_product_zero_frames", "historical_quarantine"} == {
        entry["reasons"][0] for entry in ledger
    }


def test_quarantine_evidence_carries_the_original_reason(tmp_path):
    rows = _default_rows()
    quarantine = [{
        "quarantine_key": "21:521",
        "key": _uri(rows[3]),
        "reason": "probe/read: OSError: truncated file",
    }]
    _manifest, ledger, _out = _freeze(tmp_path, rows, quarantine=quarantine)
    entry = next(r for r in ledger if r["reasons"] == ["historical_quarantine"])
    assert entry["evidence"] == {
        "historical_quarantine_key": "21:521",
        "historical_quarantine_reason": "probe/read: OSError: truncated file",
    }


# published accounting of the Aug-3 bundle
@requires_bundle
def test_published_bundle_exclusion_accounting_is_internally_consistent():
    manifest = json.loads((BUNDLE / "inventory_manifest.json").read_text())
    accounting = manifest["accounting"]
    ledger = [
        json.loads(line)
        for line in (BUNDLE / "exclusions.jsonl").read_text().splitlines()
    ]
    quarantine = {
        json.loads(line)["key"]
        for line in (BUNDLE / "quarantine.source.jsonl").read_text().splitlines()
        if line.strip()
    }

    assert len(ledger) == accounting["excluded_units"] == 4695
    assert len(quarantine) == accounting["quarantine_units"] == 3
    by_reason = {}
    for entry in ledger:
        by_reason.setdefault(tuple(entry["reasons"]), []).append(entry)
    assert {key: len(value) for key, value in by_reason.items()} == {
        ("prior_product_zero_frames",): 4692,
        ("historical_quarantine",): 3,
    }
    assert accounting["zero_frame_units"] == 4692
    assert accounting["low_estimate_exclusions"] == 4693 == sum(
        entry["below_one_frame_estimate"] for entry in ledger
    )
    assert len({entry["event"] for entry in ledger}) == \
        accounting["affected_events"] == 270
    assert accounting["fully_excluded_events"] == 231
    assert accounting["resolved_pending_events"] == 3
    assert accounting["source_units"] - accounting["excluded_units"] == \
        accounting["frozen_units"] == 165682
    assert {entry["source_uri"] for entry in ledger} >= quarantine


@requires_bundle
def test_published_bundle_exclusions_equal_the_sub_frame_size_threshold():
    """The zero-frame verdict of the detector is exactly the size test.

    Every excluded unit is either quarantined or smaller than one baseband
    frame, and no kept unit is smaller than one frame. With this property,
    the exclusion set can be reproduced without the 728 MB product archive.
    The archive is still required to reproduce the ``product_*`` assertions
    of the manifest.
    """
    source_uris = set()
    small_uris = set()
    for line in (BUNDLE / "inventory.source.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        uri = _uri(row)
        source_uris.add(uri)
        if row["size_bytes"] < BYTES_PER_FRAME:
            small_uris.add(uri)
    excluded = {
        json.loads(line)["source_uri"]
        for line in (BUNDLE / "exclusions.jsonl").read_text().splitlines()
    }
    quarantine = {
        json.loads(line)["key"]
        for line in (BUNDLE / "quarantine.source.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert len(source_uris) == 170377
    assert len(small_uris) == 4693
    assert excluded == small_uris | quarantine
    assert excluded - quarantine == small_uris - quarantine
    assert len(excluded - quarantine) == 4692


@requires_bundle
@pytest.mark.skipif(
    os.environ.get("PILOT_PROXY_DEEP_BUNDLE_CHECK") != "1",
    reason="set PILOT_PROXY_DEEP_BUNDLE_CHECK=1 to load the 728 MB product zip",
)
def test_published_bundle_zero_frame_units_match_the_product_archive():
    import io

    import numpy as np

    zero_frame = set()
    units = 0
    frames = 0
    with zipfile.ZipFile(BUNDLE / "products.source.zip") as archive:
        for member in sorted(n for n in archive.namelist() if n.endswith(".npz")):
            with np.load(io.BytesIO(archive.read(member)),
                         allow_pickle=False) as product:
                order = [str(v) for v in
                         np.asarray(product["unit_order"]).reshape(-1).tolist()]
                unit_index = np.asarray(
                    product["frame_unit_index"], dtype=np.int64).reshape(-1)
                frames += int(
                    np.asarray(product["frame_index"]).reshape(-1).size)
            used = {int(v) for v in np.unique(unit_index)}
            units += len(order)
            zero_frame.update(
                uri for index, uri in enumerate(order) if index not in used
            )
    assert units == 170374
    assert frames == 750461
    assert len(zero_frame) == 4692
    ledger = {
        json.loads(line)["source_uri"]
        for line in (BUNDLE / "exclusions.jsonl").read_text().splitlines()
        if json.loads(line)["reasons"] == ["prior_product_zero_frames"]
    }
    assert zero_frame == ledger
