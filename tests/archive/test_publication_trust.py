#!/usr/bin/env python3
"""Publication-trust gates for the cadc-datatrail survey.

The tests exercise view determinism, schema migration, provenance, archive
restoration, and frozen-bundle accounting. Bundle checks skip when their
referenced artifacts are unavailable in the test environment.

No network access is needed, since the datatrail subprocess boundary is faked
as in tests/archive/test_datatrail_adapter.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.sources import cadc as src
from pilot_proxy.archive.sources.cadc_inventory import parse_row
from pilot_proxy.archive.survey_state import SurveyStore
from pilot_proxy.chime.baseband_reader import baseband_filename


# CHIME production geometry, with one packed byte per feed per sample
NFFT, N_FEEDS = 16384, 2048
BYTES_PER_FRAME = NFFT * N_FEEDS

# real-data inputs read from the environment, and the tests that need them
# skip when the paths do not exist
FROZEN_BUNDLE = Path(os.environ.get("PILOT_PROXY_ARCHIVE_BUNDLE", "archive_inputs/chime-pilots-v5"))
JULY_REFERENCES = Path(os.environ.get("PILOT_PROXY_JULY_INVENTORIES", "Datatrawl-Inventories"))

# a schema-1 ("July") pilots row in its original shape, with no "name", a
# float "n_frames", and insertion-ordered keys
SCHEMA1_ROW = {
    "scope": "chime.event.baseband.raw", "event": "100050005",
    "freq_id": 506, "size_bytes": 91311880,
    "common_path": "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
                   "astro_100050005",
    "obs_date": "2020-07-15", "datasets": ["backlog.pulsar.B0355+54"],
    "freq_mhz": 602.3438, "n_frames": 2.7213,
}


def _row(event: str, freq_id: int, size: int = 91311880) -> dict:
    """One current-schema (schema 2) inventory row."""
    return {
        "scope": "chime.event.baseband.raw", "event": event,
        "name": baseband_filename(event, freq_id), "size_bytes": size,
        "common_path": f"cadc:CHIMEFRB/data/chime/baseband/raw/x/astro_{event}",
        "obs_date": "2020-07-15", "datasets": ["ds"], "freq_id": freq_id,
        "freq_mhz": 602.3438,
        "n_frames_estimate": size // BYTES_PER_FRAME,
    }


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_ps(monkeypatch, uris):
    """Answer every `datatrail ps --json` with this minoc replica list."""
    payload = json.dumps({
        "dataset": "d", "scope": "s",
        "files": {"file_replica_locations": {"minoc": list(uris)}},
        "policies": {},
    })
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))


# legacy-schema migration
# On the real references, none of the 161872 rows of the July chime-pilots
# inventory has a "name", and the narrow adapter restores it. The July
# chime-controls inventory carries "name", and all 32614 of its rows parse.
def test_parse_row_restores_only_identified_schema1_baseband_rows():
    parsed = parse_row(json.dumps(SCHEMA1_ROW), "inventory.jsonl", 1)
    assert parsed["name"] == baseband_filename(SCHEMA1_ROW["event"], 506)
    assert parsed["inventory_compatibility"] == "chime_baseband_schema1_filename"
    assert parsed["n_frames_legacy_estimate"] == SCHEMA1_ROW["n_frames"]
    assert "name" not in SCHEMA1_ROW
    for change in ({"scope": "other.product"}, {"event": "../escape"},
                   {"freq_id": True}, {"common_path": "cadc:CHIMEFRB/another_event"}):
        with pytest.raises(SystemExit, match="missing required"):
            parse_row(json.dumps(SCHEMA1_ROW | change), "inventory.jsonl", 1)
    partial = dict(_row("100050005", 506))
    partial.pop("name")
    with pytest.raises(SystemExit, match="missing required"):
        parse_row(json.dumps(partial), "inventory.jsonl", 1)


def test_schema1_row_carries_everything_the_upgrade_needs():
    """The migration is a pure function of the row and needs no archive access.

    The ``name`` is recovered from (event, freq_id) through the one naming
    definition, and the schema-2 ``n_frames_estimate`` is recovered from
    ``size_bytes`` and the instrument geometry. Nothing else is needed, so the
    lack of a shipped migration is a packaging gap and loses no data.
    """
    upgraded = dict(SCHEMA1_ROW)
    upgraded.pop("n_frames")
    upgraded["name"] = baseband_filename(
        upgraded["event"], upgraded["freq_id"])
    upgraded["n_frames_estimate"] = upgraded["size_bytes"] // BYTES_PER_FRAME

    parsed = parse_row(json.dumps(upgraded), "inventory.jsonl", 1)
    assert parsed["name"] == "baseband_100050005_506.h5"
    assert parsed["n_frames_estimate"] == 2
    # and the upgraded row addresses the same archive object as before
    assert (f"{parsed['common_path']}/{parsed['name']}"
            == "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
               "astro_100050005/baseband_100050005_506.h5")


@pytest.mark.parametrize("size", [
    1, BYTES_PER_FRAME - 1, BYTES_PER_FRAME, BYTES_PER_FRAME + 1,
    2 * BYTES_PER_FRAME - 1, 91311880, 32890880, 10 * BYTES_PER_FRAME,
])
def test_low_estimate_rule_survives_the_n_frames_semantics_change(size):
    """Schema 1 stored a float, and schema 2 stores floor() as an int.

    The 4693 low-estimate exclusions of the frozen bundle were selected with
    the float column. For a positive size, ``floor(x) < 1`` if and only if
    ``x < 1``, so the same exclusion set is derived from the int column,
    although the recorded values differ.
    """
    assert (size // BYTES_PER_FRAME < 1) == (size / BYTES_PER_FRAME < 1.0)


def test_completed_legacy_inventory_remains_readable():
    parse_row(json.dumps(SCHEMA1_ROW), "inventory.jsonl", 1)


@pytest.mark.skipif(not JULY_REFERENCES.is_dir(),
                    reason="July reference inventories not on this host")
@pytest.mark.parametrize("name,expect_readable", [
    ("chime-pilots", True),     # narrow schema-1 filename adapter
    ("chime-controls", True),   # already carries "name"
])
def test_july_reference_readability_is_exactly_as_measured(
        name, expect_readable):
    """Pin the measured readability of each real inventory so it cannot drift."""
    path = JULY_REFERENCES / name / "inventory.jsonl"
    accepted = rejected = 0
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                parse_row(line, str(path), number)
                accepted += 1
            except SystemExit:
                rejected += 1
    total = accepted + rejected
    assert total > 0
    assert (rejected == 0) is expect_readable, (
        f"{name}: {accepted} accepted / {rejected} rejected of {total}")


# determinism of the durable views
# The survey_state.py docstring states that inventory.jsonl and the text
# ledgers are "deterministic views of the database", and these tests assert it.
def _render(order, tmp_path, tag):
    out = tmp_path / tag
    out.mkdir()
    store = SurveyStore(out / "survey_state.sqlite3")
    for scope, event, rows in order:
        store.commit(f"{scope}|{event}", scope, event, "complete", rows)
    store.render_views(out)
    store.close()
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir()) if p.suffix in (".jsonl", ".txt")}


def test_render_views_is_independent_of_commit_order(tmp_path):
    """Runs that commit the same events in different orders give identical views.

    The order can differ because the archive answered in a different order or
    because one run resumed. The views must be byte-identical for an inventory
    to be compared between runs.
    """
    events = [("chime.event.baseband.raw", ev,
               [_row(ev, f) for f in (844, 506, 721)])
              for ev in ("100050005", "1111116060", "999", "100200200")]

    baseline = _render(events, tmp_path, "a")
    for seed in (7, 99, 1234):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert _render(shuffled, tmp_path, f"s{seed}") == baseline


def test_view_order_is_lexicographic_on_the_event_key(tmp_path):
    """The views are ordered by SQLite BINARY collation on "scope|event".

    The event id is sorted as text, so "10" comes before "1000" and "999". A
    reader who diffs two inventories must sort the same way.
    """
    out = tmp_path / "lex"
    out.mkdir()
    store = SurveyStore(out / "survey_state.sqlite3")
    scope = "chime.event.baseband.raw"
    for event in ("999", "1000", "10"):
        store.commit(f"{scope}|{event}", scope, event, "complete",
                     [_row(event, 506)])
    store.render_views(out)
    store.close()
    events = [json.loads(line)["event"]
              for line in (out / "inventory.jsonl").read_text().splitlines()]
    assert events == ["10", "1000", "999"]


def test_a_flaky_probe_silently_changes_inventory_content(tmp_path):
    """A flaky probe changes the inventory content with no in-band marker.

    _commit_decision in cadc.py accepts a partial event once the retry
    allowance is exhausted and returns ``True, True, True``. An event whose
    probes were clean writes every row, while the same event under transient
    CADC errors writes a strict subset and is committed as done. The difference
    is recorded in incomplete_events.txt, but inventory.jsonl itself is smaller
    and carries no marker. Two runs against an unchanged archive can therefore
    disagree on the rows themselves as well as on their presentation.
    """
    # clean, with 0 errored and 3 present
    assert src._commit_decision(0, 0, 3) == (True, True, False)
    # 1 of 3 keeps erroring, and the third sighting accepts the partial event
    assert src._commit_decision(1, 0, 2) == (False, False, False)
    assert src._commit_decision(1, 1, 2) == (False, False, False)
    assert src._commit_decision(1, 2, 2) == (True, True, True)

    scope, event = "chime.event.baseband.raw", "100050005"
    clean = _render([(scope, event, [_row(event, f) for f in (506, 721, 844)])],
                    tmp_path, "clean")
    partial = _render([(scope, event, [_row(event, f) for f in (506, 721)])],
                      tmp_path, "partial")
    assert clean["inventory.jsonl"] != partial["inventory.jsonl"]


# provenance a third party needs to reconstruct how the inventory was made
REQUIRED_PROVENANCE = frozenset({
    "source_revision",          # which commit of this tree ran
    "package_source_sha256",    # provenance.package_source_sha256()
    "datatrail_cli_version",    # the CLI whose payloads were parsed
    "cadcdata_version",
    "certificate_not_after",    # identity and expiry of the cert that authorized the run
    "inventory_sha256",         # SHA-256 of the inventory file
    "row_count",
})


def test_inventory_meta_records_enough_to_reproduce_the_survey(tmp_path):
    from pilot_proxy.archive.inventory import write_inventory_meta

    class _Instrument:
        name = "chime"
        scopes = ("chime.event.baseband.raw",)

    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(_row("100050005", 506)) + "\n")
    meta_path = write_inventory_meta(
        inventory, _Instrument(), source="cadc-datatrail",
        reader="chime-baseband", freq_ids="506", name="t")
    meta = json.loads(meta_path.read_text())
    assert REQUIRED_PROVENANCE.issubset(meta), sorted(
        REQUIRED_PROVENANCE - set(meta))


def test_survey_manifest_records_execution_separately_from_configuration(tmp_path):
    from pilot_proxy.archive.survey_state import ensure_manifest, SurveyRun
    configuration = {"schema": 1, "source": "cadc-datatrail"}
    before = ensure_manifest(tmp_path, configuration)
    with SurveyRun(tmp_path) as run:
        (tmp_path / "inventory.jsonl").write_text(json.dumps(_row("100050005", 506)) + "\n")
        run.views_current()
    manifest = json.loads((tmp_path / "survey_manifest.json").read_text())
    assert manifest["configuration"] == configuration
    assert manifest["fingerprint"] == before["fingerprint"]
    latest = manifest["latest_run"]
    assert REQUIRED_PROVENANCE.issubset(latest)
    assert latest["row_count"] == 1 and latest["inventory_sha256"]
    assert latest["started"] <= latest["finished"]
    assert latest["status"] == "finished" and latest["views_current"]
    first_id = latest["run_id"]
    with SurveyRun(tmp_path) as run:
        run.views_current()
    assert len(list((tmp_path / "survey_runs").glob("*.json"))) == 2
    assert json.loads((tmp_path / "survey_manifest.json").read_text())["latest_run"]["run_id"] != first_id


# scope of the collection-restore rule
# _restore_collection() in datatrail_client.py prefixes any replica path that
# begins with "data/" with "cadc:CHIMEFRB/", so a path of this class is
# accepted as a common path. These tests check the documented bare replica,
# the lookalike and escaping paths that must still be refused, and the flag
# that records a restored origin.
def test_restore_collection_recovers_the_documented_bare_replica(monkeypatch):
    """Every replica of the documented case comes back bare and is restored."""
    _fake_ps(monkeypatch, [
        "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
        "baseband_1111116060_506.h5",
        "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
        "baseband_1111116060_844.h5",
    ])
    common, names, ok = dt.DATATRAIL.files(
        "chime.event.baseband.raw", "1111116060", retries=0)
    assert ok and common == ("cadc:CHIMEFRB/data/chime/baseband/raw/2025/03/31/"
                             "astro_1111116060")
    assert sorted(names) == ["baseband_1111116060_506.h5",
                             "baseband_1111116060_844.h5"]


@pytest.mark.parametrize("uri", [
    "database/chime/x/astro_1/f.h5",   # lookalike root, must not be restored
    "/data/chime/x/astro_1/f.h5",      # absolute path, which is not archive-relative
    "cadc:SOMEONE_ELSE/data/chime/x/astro_1/f.h5",
])
def test_restore_collection_stays_narrow(monkeypatch, uri):
    """Restoration applies only to the archive layout."""
    _fake_ps(monkeypatch, [uri, uri.replace("f.h5", "g.h5")])
    with pytest.raises(dt.DatatrailContractError):
        dt.DATATRAIL.files("chime.event.baseband.raw", "1", retries=0)


def test_restore_collection_refuses_paths_that_escape_the_collection_root(
        monkeypatch):
    _fake_ps(monkeypatch, ["data/../../../etc/x/f.h5",
                           "data/../../../etc/x/g.h5"])
    with pytest.raises(dt.DatatrailContractError):
        dt.DATATRAIL.files("chime.event.baseband.raw", "1", retries=0)


def test_a_restored_replica_retains_its_origin(monkeypatch):
    """Reconstructed collection identity remains visible after resolution."""
    bare = ["data/chime/x/astro_1/f.h5", "data/chime/x/astro_1/g.h5"]
    prefixed = ["cadc:CHIMEFRB/" + u for u in bare]

    _fake_ps(monkeypatch, bare)
    from_bare = dt.DATATRAIL.files("s", "1", retries=0)
    _fake_ps(monkeypatch, prefixed)
    from_prefixed = dt.DATATRAIL.files("s", "1", retries=0)
    assert from_bare == from_prefixed
    assert from_bare[0].collection_restored
    assert not from_prefixed[0].collection_restored


# accounting of the published bundle
# The bundle records a one-off script run as its own check, and the tests below
# check its accounting again directly.
requires_bundle = pytest.mark.skipif(
    not (FROZEN_BUNDLE / "inventory_manifest.json").is_file(),
    reason="chime-pilots-v5 frozen bundle not on this host")


def _unit_keys(path: Path) -> set:
    keys = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["scope"], row["event"], row["name"]))
    return keys


@requires_bundle
def test_frozen_inventory_is_an_exact_partition_of_the_source():
    """The frozen inventory and the exclusions partition the source exactly.

    The two sets do not overlap, and their union is the source with no extra
    rows.
    """
    source = _unit_keys(FROZEN_BUNDLE / "inventory.source.jsonl")
    frozen = _unit_keys(FROZEN_BUNDLE / "inventory.jsonl")
    excluded = _unit_keys(FROZEN_BUNDLE / "exclusions.jsonl")
    assert len(source) == 170377
    assert len(frozen) == 165682
    assert len(excluded) == 4695
    assert frozen & excluded == set()
    assert frozen | excluded == source


@requires_bundle
def test_frozen_bundle_files_match_the_digests_it_publishes():
    manifest = json.loads(
        (FROZEN_BUNDLE / "inventory_manifest.json").read_text())
    checked = 0
    for name, entry in manifest["outputs"].items():
        path = FROZEN_BUNDLE / name
        if not path.is_file() or entry["bytes"] > 128 * 1024 * 1024:
            continue                      # skip the 728 MB product archive
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
        assert path.stat().st_size == entry["bytes"], name
        checked += 1
    assert checked >= 10


@requires_bundle
def test_only_three_frozen_exclusions_need_evidence_outside_the_archive():
    """Only 3 frozen exclusions need evidence that the archive does not hold.

    Each of the 4692 exclusions marked ``prior_product_zero_frames`` is also
    caught by the size rule alone, so the prior detector product only confirms
    them, and a third party can re-derive them from the archive. The 3
    ``historical_quarantine`` units cannot be re-derived, since they came from
    direct HDF5 reads that hit truncated files, and 2 of the 3 are invisible to
    any size rule. This test pins that split of 4692 and 3.
    """
    size_derivable, external, invisible = set(), set(), 0
    with (FROZEN_BUNDLE / "exclusions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["scope"], row["event"], row["name"])
            caught = row["size_bytes"] // BYTES_PER_FRAME < 1
            if row["reasons"] == ["prior_product_zero_frames"]:
                assert caught, f"not size-derivable: {key}"
                size_derivable.add(key)
            else:
                assert row["reasons"] == ["historical_quarantine"], row["reasons"]
                external.add(key)
                invisible += not row["below_one_frame_estimate"]
    assert len(size_derivable) == 4692
    assert len(external) == 3
    assert invisible == 2


@requires_bundle
def test_frozen_inventory_cannot_be_regenerated_by_todays_code():
    """The pinned production input is schema 1.5 and cannot be regenerated.

    The input has "name", so it parses, and it carries the float ``n_frames``
    column where schema 2 has the integer ``n_frames_estimate``. The current
    code cannot regenerate it byte for byte, since the JSON key order, the
    column name, and the column type all differ. The locked digest b2cfef75...
    therefore cannot be reproduced by running the survey again.
    """
    with (FROZEN_BUNDLE / "inventory.jsonl").open() as handle:
        first = json.loads(handle.readline())
    parse_row(json.dumps(first), "inventory.jsonl", 1)   # parses, since it has "name"
    assert "n_frames" in first and "n_frames_estimate" not in first
    assert list(first) != sorted(first)                  # insertion-ordered
