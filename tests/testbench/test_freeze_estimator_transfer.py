from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess

import pytest


FREEZER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "freeze_estimator_transfer.py")
)


def test_digital_layout_matches_completed_sweep(tmp_path: Path) -> None:
    shards = FREEZER["digital_shard_dirs"](tmp_path)
    allocation = FREEZER["digital_expected_allocation"]()

    assert len(shards) == 40
    assert shards[0] == tmp_path / "extreme_low" / "snr_m60"
    assert shards[-1] == tmp_path / "high" / "snr_p60"
    assert len(allocation) == 41
    assert sum(allocation.values()) == 9000
    assert allocation[-60.0] == 240
    assert allocation[-45.0] == 1000
    assert allocation[60.0] == 60


def test_source_inventory_is_sorted_and_hashed(tmp_path: Path) -> None:
    second = tmp_path / "b.csv"
    first = tmp_path / "a.csv"
    second.write_text("second\n", encoding="utf-8")
    first.write_text("first\n", encoding="utf-8")

    rows = FREEZER["source_file_inventory"](
        tmp_path,
        [(second, "trial"), (first, "summary")],
    )

    assert [row["relative_path"] for row in rows] == ["a.csv", "b.csv"]
    assert rows[0]["role"] == "summary"
    assert rows[0]["sha256"] == FREEZER["file_sha256"](first)


def test_copied_text_uses_lf(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "release" / "result.csv"
    source.write_bytes(b"a,b\r\n1,2\r\n")

    FREEZER["copy_text_lf"](source, destination)

    assert destination.read_bytes() == b"a,b\n1,2\n"


def test_legacy_conditioning_weights_are_pinned() -> None:
    bank_path = FREEZER["DEFAULT_WEIGHTS_PATH"]
    manifest_path = Path(f"{bank_path}.manifest.json")

    assert FREEZER["file_sha256"](bank_path) == FREEZER[
        "LEGACY_CONDITIONING_WEIGHT_BANK_SHA256"
    ]
    assert FREEZER["file_sha256"](manifest_path) == FREEZER[
        "LEGACY_CONDITIONING_WEIGHT_MANIFEST_SHA256"
    ]


def test_event_ledger_keeps_latest_matching_attempt(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    destination = tmp_path / "release" / "events.jsonl"
    first = {
        "status": "complete",
        "event": {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        "attempt": 1,
    }
    second = {**first, "attempt": 2}
    source.write_text(
        json.dumps(first, sort_keys=True)
        + "\n"
        + json.dumps(second, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    FREEZER["copy_canonical_event_ledger"](source, destination)

    rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [second]


def test_event_ledger_rejects_conflicting_identity(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    rows = [
        {
            "event": {
                "event_index": 1,
                "pass_index": 1,
                "kind": "tx_zero",
            }
        },
        {
            "event": {
                "event_index": 1,
                "pass_index": 1,
                "kind": "signal_only",
            }
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting planned identities"):
        FREEZER["read_canonical_event_rows"](source)


def test_event_ledger_matches_complete_run_plan(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    planned = [
        {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        {"event_index": 2, "pass_index": 1, "kind": "noise_only"},
    ]
    source.write_text(
        "".join(json.dumps({"event": event}) + "\n" for event in planned),
        encoding="utf-8",
    )

    rows = FREEZER["read_canonical_event_rows"](
        source,
        planned_events=planned,
        require_complete=True,
    )

    assert [row["event"] for row in rows] == planned


def test_event_ledger_rejects_missing_planned_index(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    planned = [
        {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        {"event_index": 2, "pass_index": 1, "kind": "noise_only"},
    ]
    source.write_text(
        json.dumps({"event": planned[0]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing planned indices"):
        FREEZER["read_canonical_event_rows"](
            source,
            planned_events=planned,
            require_complete=True,
        )


def test_event_ledger_rejects_extra_index(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    row = {"event": {**planned, "event_index": 2}}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is not in the run plan"):
        FREEZER["read_canonical_event_rows"](
            source,
            planned_events=[planned],
            require_complete=True,
        )


def test_event_ledger_rejects_swapped_index(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    planned = [
        {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        {"event_index": 2, "pass_index": 1, "kind": "noise_only"},
    ]
    row = {"event": {**planned[1], "event_index": 1}}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the run plan"):
        FREEZER["read_canonical_event_rows"](
            source,
            planned_events=planned,
            require_complete=True,
        )


@pytest.mark.parametrize("event_index", [1.0, "1", True])
def test_event_ledger_rejects_non_integer_index(
    tmp_path: Path,
    event_index: object,
) -> None:
    source = tmp_path / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    row = {"event": {**planned, "event_index": event_index}}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid event ledger row"):
        FREEZER["read_canonical_event_rows"](
            source,
            planned_events=[planned],
            require_complete=True,
        )


def test_event_ledger_snapshot_is_stable_after_source_change(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    destination = tmp_path / "release" / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    original = {"event": planned, "attempt": 1}
    changed = {"event": planned, "attempt": 2}
    source.write_text(json.dumps(original) + "\n", encoding="utf-8")
    records = FREEZER["_canonical_event_records"](
        source,
        planned_events=[planned],
        require_complete=True,
    )

    source.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    FREEZER["write_canonical_event_ledger"](records, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == original


def test_freeze_ota_writes_validated_event_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    destination = tmp_path / "release"
    capture = result_root / "captures" / "event_00001.cfile"
    capture.parent.mkdir(parents=True)
    capture.write_bytes(b"capture")
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    original = {
        "event": planned,
        "capture_path": str(capture),
        "attempt": 1,
    }
    changed = {**original, "attempt": 2}
    events_path = result_root / "events.jsonl"
    events_path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    (result_root / "sdr_transfer_summary.csv").write_text("", encoding="utf-8")
    (result_root / "sdr_transfer_trials.csv").write_text("", encoding="utf-8")
    (result_root / "run_plan.json").write_text(
        json.dumps({"events": [planned]}) + "\n",
        encoding="utf-8",
    )
    (result_root / "run_state.json").write_text("{}\n", encoding="utf-8")

    inventory_path = tmp_path / "inventory.csv"
    capture_hash = FREEZER["file_sha256"](capture)
    FREEZER["write_csv"](
        inventory_path,
        ("path", "size_bytes", "sha256"),
        [
            {
                "path": "captures/event_00001.cfile",
                "size_bytes": capture.stat().st_size,
                "sha256": capture_hash,
            }
        ],
    )
    freeze_globals = FREEZER["freeze_ota"].__globals__
    monkeypatch.setitem(
        freeze_globals,
        "CANONICAL_CAPTURE_INVENTORY_BYTES",
        inventory_path.stat().st_size,
    )
    monkeypatch.setitem(
        freeze_globals,
        "CANONICAL_CAPTURE_INVENTORY_SHA256",
        FREEZER["file_sha256"](inventory_path),
    )
    monkeypatch.setattr(
        FREEZER["transfer_plot"],
        "_read_summary_rows",
        lambda _paths: [],
    )
    monkeypatch.setattr(
        FREEZER["transfer_plot"],
        "_read_trial_rows",
        lambda _paths: [],
    )
    monkeypatch.setattr(
        FREEZER["transfer_plot"],
        "plot_summary",
        lambda **_kwargs: None,
    )
    monkeypatch.setitem(
        freeze_globals,
        "_ota_points",
        lambda *_args, **_kwargs: ([], {}, {}),
    )

    def mutate_source(*_args: object, **_kwargs: object) -> dict[str, object]:
        events_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        return {"raw_captures": {}}

    monkeypatch.setitem(freeze_globals, "_ota_qc", mutate_source)
    monkeypatch.setitem(
        freeze_globals,
        "_source_provenance",
        lambda *_args: {},
    )

    FREEZER["freeze_ota"](result_root, destination, bootstrap_samples=1)

    released = json.loads(
        (destination / "run" / "events.jsonl").read_text(encoding="utf-8")
    )
    assert json.loads(events_path.read_text(encoding="utf-8")) == changed
    assert released == original


def test_capture_inventory_prefers_relocated_copy(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    relocated_root = tmp_path / "relocated"
    original_capture = original_root / "captures" / "event_00001.cfile"
    relocated_capture = relocated_root / "captures" / original_capture.name
    original_capture.parent.mkdir(parents=True)
    relocated_capture.parent.mkdir(parents=True)
    original_capture.write_bytes(b"old")
    relocated_capture.write_bytes(b"current")
    rows = [
        {
            "event": {
                "event_index": 1,
                "pass_index": 1,
                "kind": "tx_zero",
                "snr_db": None,
            },
            "capture_path": str(original_capture),
        }
    ]

    inventory = FREEZER["_capture_inventory"](relocated_root, rows)

    assert inventory[0]["relative_path"] == "captures/event_00001.cfile"
    assert inventory[0]["size_bytes"] == len(b"current")
    assert inventory[0]["sha256"] == FREEZER["file_sha256"](relocated_capture)


def test_capture_inventory_rejects_path_outside_root(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    outside = tmp_path / "outside.cfile"
    result_root.mkdir()
    outside.write_bytes(b"outside")
    rows = [
        {
            "event": {
                "event_index": 1,
                "pass_index": 1,
                "kind": "tx_zero",
                "snr_db": None,
            },
            "capture_path": str(outside),
        }
    ]

    with pytest.raises(ValueError, match="outside the result root"):
        FREEZER["_capture_inventory"](result_root, rows)


def test_release_index_has_closed_census(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("release\n", encoding="utf-8")
    data = tmp_path / "data" / "plot_points.csv"
    data.parent.mkdir()
    data.write_text("x,y\n0,0\n", encoding="utf-8")

    manifest_hash, checksum_hash = FREEZER["write_release_index"](
        tmp_path,
        experiment="fixture",
        title="Fixture",
        counts={"plot_points": 1},
    )

    manifest = json.loads((tmp_path / "release_manifest.json").read_text())
    assert manifest["path_semantics"] == "release_root_relative_posix"
    assert manifest["artifact_count"] == 2
    assert [item["path"] for item in manifest["artifacts"]] == [
        "README.md",
        "data/plot_points.csv",
    ]
    checksums = (tmp_path / "SHA256SUMS").read_text()
    assert "  release_manifest.json\n" in checksums
    assert "  SHA256SUMS\n" not in checksums
    assert manifest_hash == FREEZER["file_sha256"](
        tmp_path / "release_manifest.json"
    )
    assert checksum_hash == FREEZER["file_sha256"](tmp_path / "SHA256SUMS")


def test_fit_metrics_distinguish_fitted_line_residuals() -> None:
    result = FREEZER["_fit_line"]([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])

    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept_db"] == pytest.approx(1.0)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["fit_line_rms_residual_db"] < 1.0e-12
    assert result["fit_line_max_abs_residual_db"] < 1.0e-12


def _has_tag(name: str) -> bool:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--verify",
                               "--quiet", f"refs/tags/{name}"], capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _has_tag(FREEZER["PUBLICATION_EXPORT_ARCHIVE_TAG"]),
                    reason="needs the archive tags of the provenance repository (PROVENANCE.md)")
def test_provenance_separates_run_and_export_states(tmp_path: Path) -> None:
    function_globals = FREEZER["_source_provenance"].__globals__
    original_state = function_globals["_repository_export_state"]
    function_globals["_repository_export_state"] = lambda: {
        "repository_revision": "f" * 40,
        "repository_dirty": True,
    }
    try:
        value = FREEZER["_source_provenance"]("digital_synthetic", tmp_path)
    finally:
        function_globals["_repository_export_state"] = original_state
    run = value["run_worktree"]
    export = value["publication_export"]

    assert run["base_revision"] == FREEZER["RUN_BASE_REVISION"]
    assert run["state"] == "uncommitted tracked and untracked source"
    assert "archival_stash_object" in run
    assert (
        run["conditioning_record_plot_source_sha256"]
        == FREEZER["ORIGINAL_DIGITAL_PLOT_SOURCE_SHA256"]
    )
    assert run["conditioning_record_plot_source_archival_tag"] == FREEZER[
        "PLOT_SOURCE_ARCHIVE_TAG"
    ]
    assert run["run_source_archival_tag"] == FREEZER[
        "RUN_SOURCE_ARCHIVE_TAG"
    ]
    assert export["historical_post_update_revision"] == FREEZER[
        "HISTORICAL_POST_UPDATE_REVISION"
    ]
    assert export["repository_revision"] == "f" * 40
    assert export["repository_dirty"] is True
    for source in export["source_files"].values():
        assert source["archival_tag"] == FREEZER[
            "PUBLICATION_EXPORT_ARCHIVE_TAG"
        ]
        assert len(source["archival_tag_object"]) == 40
        assert len(source["archival_commit"]) == 40
        assert len(source["archival_blob"]) == 40
