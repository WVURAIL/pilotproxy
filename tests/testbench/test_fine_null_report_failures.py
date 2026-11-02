"""A failed independent-null report must preserve inputs and permit a retry."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fine_null_failure_fixtures", ROOT / "tests/testbench/test_report_fine_null_validation.py"
)
FIXTURES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIXTURES
SPEC.loader.exec_module(FIXTURES)
REPORT = FIXTURES.REPORT


def _study(tmp_path):
    out, plan = FIXTURES.study(tmp_path)
    raw = FIXTURES.validation(out, plan)
    return out, raw


def _input_hashes(out):
    return {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.rglob("*")
        if path.is_file() and "reports" not in path.relative_to(out).parts
    }


@pytest.mark.parametrize("failure", [
    "source_copy", "reduction", "array_write", "report_write", "marker_write",
    "final_rehash", "interrupt",
])
def test_failed_report_removes_only_its_output_and_can_be_retried(tmp_path, monkeypatch, failure):
    out, _ = _study(tmp_path)
    before = _input_hashes(out)
    sibling = out / "reports" / "previous.txt"
    sibling.parent.mkdir()
    sibling.write_text("Retain an unrelated report.\n")
    target = out / "reports" / "null_validation"
    with monkeypatch.context() as patch:
        if failure == "source_copy":
            def fail_copy(source, destination, *args, **kwargs):
                Path(destination).write_bytes(b"partial snapshot")
                raise OSError("source copy failure")
            patch.setattr(shutil, "copyfile", fail_copy)
        elif failure in ("reduction", "interrupt"):
            def fail_reduce(*args, **kwargs):
                if failure == "interrupt":
                    raise KeyboardInterrupt("interrupted report")
                raise OSError("reduction failure")
            patch.setattr(REPORT, "reduce_powers", fail_reduce)
        elif failure == "array_write":
            def fail_array(stream, **arrays):
                stream.write(b"partial archive")
                raise OSError("array write failure")
            patch.setattr(np, "savez", fail_array)
        elif failure == "report_write":
            def fail_report(path, payload):
                path.write_text('{"incomplete":')
                raise OSError("report write failure")
            patch.setattr(REPORT, "write_json", fail_report)
        elif failure == "marker_write":
            original = Path.open

            def fail_marker(path, *args, **kwargs):
                if path == target / "report.sha256":
                    raise OSError("marker write failure")
                return original(path, *args, **kwargs)
            patch.setattr(Path, "open", fail_marker)
        else:
            original = REPORT.rehash

            def fail_rehash(directory, mapping):
                if any(str(name).startswith("reports/null_validation/ch") for name in mapping):
                    raise ValueError("artifact changed before publication")
                return original(directory, mapping)
            patch.setattr(REPORT, "rehash", fail_rehash)
        with pytest.raises((OSError, ValueError, KeyboardInterrupt)):
            REPORT.run(out)
    assert not target.exists()
    assert sibling.read_text() == "Retain an unrelated report.\n"
    assert _input_hashes(out) == before
    result = REPORT.run(out)
    assert result["trials_per_policy"] == 2
    assert result["primary_policy_rows"] == 58
    assert (target / "report.sha256").read_text() == REPORT.sha(target / "report.json") + "\n"


def test_authenticated_but_malformed_raw_payload_leaves_no_report(tmp_path):
    out, path = _study(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["fixed_fine"] = arrays["fixed_fine"].astype(float)
    np.savez(path, **arrays)
    receipt = json.loads(path.with_suffix(".json").read_text())
    receipt["sha256"] = REPORT.sha(path)
    path.with_suffix(".json").write_text(json.dumps(receipt))
    before = _input_hashes(out)
    with pytest.raises(ValueError, match="Raw shape/dtype"):
        REPORT.run(out)
    assert not (out / "reports/null_validation").exists()
    assert _input_hashes(out) == before


@pytest.mark.parametrize("target_kind", ["directory", "file", "symlink"])
def test_preexisting_report_target_is_preserved(tmp_path, target_kind):
    out, _ = _study(tmp_path)
    target = out / "reports/null_validation"
    target.parent.mkdir()
    if target_kind == "directory":
        target.mkdir()
        sentinel = target / "keep.txt"
    elif target_kind == "file":
        sentinel = target
    else:
        sentinel = tmp_path / "external.txt"
        target.symlink_to(sentinel)
    sentinel.write_text("Existing content.\n")
    with pytest.raises(ValueError, match="overwrite"):
        REPORT.run(out)
    assert sentinel.read_text() == "Existing content.\n"
    assert target.is_symlink() if target_kind == "symlink" else target.exists()


@pytest.mark.parametrize("threshold", [10**400, np.longdouble(np.finfo(float).max) * 2])
def test_floating_threshold_conversion_cannot_produce_infinity(threshold):
    with pytest.raises(ValueError, match="finite"):
        REPORT.threshold_value(threshold, exact=False)


def test_finite_float_threshold_boundary_is_retained():
    threshold = np.finfo(float).max
    assert REPORT.threshold_value(threshold, exact=False) == threshold
    counts = REPORT.count_outcomes(np.array([0.0, threshold, np.inf, np.nan]),
                                  np.ones(4, dtype=bool), threshold)
    assert counts["exceedances"] == 1
    assert counts["valid_nonexceedances"] == 2
    assert counts["invalid_trials"] == 1
