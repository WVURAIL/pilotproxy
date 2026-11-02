"""Reducer failures must not publish complete or mislabeled experiments."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fine_reducer_failure_fixtures", ROOT / "tests/testbench/test_reduce_fine_validation.py"
)
FIXTURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIXTURES
SPEC.loader.exec_module(FIXTURES)
REDUCER = FIXTURES.REDUCER


def test_invalid_json_does_not_leave_an_output_that_blocks_retry(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        REDUCER.write_json(path, {"partial": "valid prefix", "value": np.nan})
    assert not path.exists()
    REDUCER.write_json(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}


def test_json_publication_preserves_existing_output(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"previous output\n")
    with pytest.raises(FileExistsError):
        REDUCER.write_json(path, {"status": "new"})
    assert path.read_bytes() == b"previous output\n"
    assert list(tmp_path.iterdir()) == [path]


def test_json_publication_refuses_a_competing_writer_without_replacing_it(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    original = REDUCER.os.link

    def competing_writer(source, target):
        Path(target).write_bytes(b"competing output\n")
        original(source, target)

    monkeypatch.setattr(REDUCER.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        REDUCER.write_json(path, {"status": "new"})
    assert path.read_bytes() == b"competing output\n"
    assert list(tmp_path.iterdir()) == [path]


def test_json_publication_error_removes_temporary_file(tmp_path, monkeypatch):
    def fail_link(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(REDUCER.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        REDUCER.write_json(tmp_path / "report.json", {"status": "complete"})
    assert list(tmp_path.iterdir()) == []


def test_score_array_write_failure_leaves_no_npz_or_receipt(tmp_path, monkeypatch):
    path = tmp_path / "score.npz"

    def interrupted(stream, **values):
        stream.write(b"partial archive")
        raise OSError("injected archive failure")

    monkeypatch.setattr(REDUCER.np, "savez", interrupted)
    with pytest.raises(OSError, match="archive failure"):
        REDUCER.save_scores(path, {"values": np.ones(2)}, {})
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    assert list(tmp_path.iterdir()) == []


def test_score_receipt_collision_preserves_prior_receipt_without_new_archive(tmp_path):
    path = tmp_path / "score.npz"
    receipt = path.with_suffix(".json")
    receipt.write_bytes(b"existing receipt\n")
    with pytest.raises(FileExistsError):
        REDUCER.save_scores(path, {"values": np.ones(2)}, {})
    assert not path.exists()
    assert receipt.read_bytes() == b"existing receipt\n"


def test_score_receipt_failure_removes_only_the_new_archive(tmp_path, monkeypatch):
    path = tmp_path / "score.npz"

    def fail_receipt(*args, **kwargs):
        raise OSError("injected receipt failure")

    monkeypatch.setattr(REDUCER, "write_json", fail_receipt)
    with pytest.raises(OSError, match="receipt failure"):
        REDUCER.save_scores(path, {"values": np.ones(2)}, {})
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_score_hash_failure_rolls_back_new_outputs(tmp_path, monkeypatch, suffix):
    path = tmp_path / "score.npz"
    original = REDUCER.sha

    def fail_hash(candidate):
        if Path(candidate).suffix == suffix:
            raise OSError("injected hash failure")
        return original(candidate)

    monkeypatch.setattr(REDUCER, "sha", fail_hash)
    with pytest.raises(OSError, match="hash failure"):
        REDUCER.save_scores(path, {"values": np.ones(2)}, {})
    assert list(tmp_path.iterdir()) == []


def test_failed_reduction_can_retry_the_same_exclusive_stage(tmp_path, monkeypatch):
    out = FIXTURES.fixture(tmp_path)
    original = REDUCER.reduce_powers

    def fail_reduction(*args, **kwargs):
        raise ValueError("injected invalid raw powers")

    monkeypatch.setattr(REDUCER, "reduce_powers", fail_reduction)
    with pytest.raises(ValueError, match="invalid raw powers"):
        REDUCER.score_stage(out, "null_calibration")
    assert not (out / "scores/null_calibration").exists()
    monkeypatch.setattr(REDUCER, "reduce_powers", original)
    result = REDUCER.score_stage(out, "null_calibration")
    assert result["coverage_complete"]
    REDUCER.rehash(out, result["outputs_sha256"])


def test_partial_score_cell_does_not_publish_a_stage(tmp_path, monkeypatch):
    out = FIXTURES.fixture(tmp_path)
    original = REDUCER.save_scores

    def fail_after_cell(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("injected failure after first cell")

    monkeypatch.setattr(REDUCER, "save_scores", fail_after_cell)
    with pytest.raises(OSError, match="first cell"):
        REDUCER.score_stage(out, "discovery")
    assert not (out / "scores/discovery").exists()
    assert not list((out / "scores").glob(".*"))


def test_changed_bound_inputs_do_not_publish_a_derived_stage(tmp_path, monkeypatch):
    out = FIXTURES.fixture(tmp_path)
    original = REDUCER.rehash

    def refuse_changed_inputs(root, mapping):
        if any(name.startswith("raw/") for name in mapping):
            raise ValueError("Bound input changed: injected late failure")
        return original(root, mapping)

    monkeypatch.setattr(REDUCER, "rehash", refuse_changed_inputs)
    with pytest.raises(ValueError, match="late failure"):
        REDUCER.score_stage(out, "null_calibration")
    assert not (out / "scores/null_calibration").exists()
    assert not list((out / "scores").glob(".*"))


def _discovery_campaign(tmp_path):
    out = FIXTURES.fixture(tmp_path)
    REDUCER.score_stage(out, "null_calibration")
    REDUCER.calibrate(out)
    REDUCER.score_stage(out, "discovery")
    return out


def _rewrite_score_metadata(out, **changes):
    path = out / "scores/discovery/ch14_case0.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["meta_json"].item()))
    metadata.update(changes)
    arrays["meta_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    receipt.update(metadata=metadata, sha256=REDUCER.sha(path))
    FIXTURES.write(receipt_path, receipt)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for changed in (path, receipt_path):
        manifest["outputs_sha256"][str(changed.relative_to(out))] = REDUCER.sha(changed)
    FIXTURES.write(manifest_path, manifest)


@pytest.mark.parametrize("snrs", [list(range(-21, -67, -3)), list(range(-65, -19, 3))])
def test_discovery_grid_labels_must_match_the_frozen_plan(tmp_path, snrs):
    out = _discovery_campaign(tmp_path)
    _rewrite_score_metadata(out, snr_db=snrs)
    with pytest.raises(ValueError, match="Discovery score"):
        REDUCER.freeze_grid(out)
    assert not (out / "evaluation-grid.json").exists()
    assert not (out / "evaluation-grid.sha256").exists()


def test_existing_calibration_marker_refuses_before_creating_a_report(tmp_path):
    out = FIXTURES.fixture(tmp_path)
    REDUCER.score_stage(out, "null_calibration")
    marker = out / "calibration.sha256"
    marker.write_text("prior marker\n")
    with pytest.raises(FileExistsError):
        REDUCER.calibrate(out)
    assert marker.read_text() == "prior marker\n"
    assert not (out / "calibration.json").exists()


def test_existing_grid_marker_refuses_before_creating_a_report(tmp_path):
    out = _discovery_campaign(tmp_path)
    marker = out / "evaluation-grid.sha256"
    marker.write_text("prior marker\n")
    with pytest.raises(FileExistsError):
        REDUCER.freeze_grid(out)
    assert marker.read_text() == "prior marker\n"
    assert not (out / "evaluation-grid.json").exists()


@pytest.mark.parametrize("stage,stem", [("calibrate", "calibration"),
                                        ("freeze_grid", "evaluation-grid")])
def test_completion_marker_failure_rolls_back_report_and_allows_retry(
        tmp_path, monkeypatch, stage, stem):
    out = _discovery_campaign(tmp_path) if stage == "freeze_grid" else FIXTURES.fixture(tmp_path)
    if stage == "calibrate":
        REDUCER.score_stage(out, "null_calibration")
    original = REDUCER.os.link

    def fail_marker(source, target):
        if Path(target).name == stem + ".sha256":
            raise OSError("injected completion failure")
        return original(source, target)

    monkeypatch.setattr(REDUCER.os, "link", fail_marker)
    with pytest.raises(OSError, match="completion failure"):
        getattr(REDUCER, stage)(out)
    assert not (out / (stem + ".json")).exists()
    assert not (out / (stem + ".sha256")).exists()
    assert not list(out.glob(".*"))
    monkeypatch.setattr(REDUCER.os, "link", original)
    result = getattr(REDUCER, stage)(out)
    assert result["status"] in ("calibrated", "frozen")
    assert (out / (stem + ".sha256")).read_text().strip() == REDUCER.sha(out / (stem + ".json"))


def test_existing_complete_score_stage_survives_a_repeat_attempt(tmp_path):
    out = FIXTURES.fixture(tmp_path)
    manifest = REDUCER.score_stage(out, "null_calibration")
    with pytest.raises(ValueError, match="overwrite"):
        REDUCER.score_stage(out, "null_calibration")
    REDUCER.rehash(out, manifest["outputs_sha256"])
    assert (out / "scores/null_calibration/manifest.json").is_file()


@pytest.mark.parametrize("failure", ["source_copy", "snapshot_hash", "binding_write"])
def test_failed_initial_source_binding_cleans_only_owned_snapshots_and_retries(
        tmp_path, monkeypatch, failure):
    out = FIXTURES.fixture(tmp_path)
    REDUCER.load_contract(out)
    cache = out / "reduction/cache-plan.json"
    original_cache = cache.read_bytes()
    with monkeypatch.context() as patch:
        if failure == "source_copy":
            original = REDUCER.shutil.copyfile

            def fail_copy(*args, **kwargs):
                original(*args, **kwargs)
                raise OSError("injected snapshot copy failure")

            patch.setattr(REDUCER.shutil, "copyfile", fail_copy)
        elif failure == "snapshot_hash":
            original = REDUCER.sha

            def fail_hash(path):
                return "changed" if "source_snapshots" in Path(path).parts else original(path)

            patch.setattr(REDUCER, "sha", fail_hash)
        else:
            original = REDUCER.os.link

            def fail_binding(source, target):
                if Path(target).name == "source-binding.json":
                    raise OSError("injected binding publication failure")
                return original(source, target)

            patch.setattr(REDUCER.os, "link", fail_binding)
        with pytest.raises((OSError, ValueError)):
            REDUCER.score_stage(out, "null_calibration")
    assert cache.read_bytes() == original_cache
    assert not (out / "reduction/source_snapshots").exists()
    assert not (out / "reduction/source-binding.json").exists()
    assert not (out / "scores/null_calibration").exists()
    result = REDUCER.score_stage(out, "null_calibration")
    assert result["coverage_complete"]
    REDUCER.rehash(out, result["inputs_sha256"])


def test_preexisting_snapshot_tree_without_binding_is_preserved(tmp_path):
    out = FIXTURES.fixture(tmp_path)
    sentinel = out / "reduction/source_snapshots/retain.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("Prior interrupted attempt.\n")
    with pytest.raises(ValueError, match="snapshot"):
        REDUCER.score_stage(out, "null_calibration")
    assert sentinel.read_text() == "Prior interrupted attempt.\n"
    assert list(sentinel.parent.iterdir()) == [sentinel]
    assert not (out / "reduction/source-binding.json").exists()
