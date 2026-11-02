"""Refuse malformed paired resampling and preserve incomplete-report boundaries."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import report_fine_sensitivity as R


def _cohort(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "fine_sensitivity_failure_fixtures",
        ROOT / "tests/testbench/test_report_fine_sensitivity.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engineering_cohort(tmp_path)


@pytest.mark.parametrize("counts", [
    [[-1, 3]], [[0.5, 1.5]], [[True, True]], np.empty((0, 2), dtype=int),
    np.ones((1, 1, 2), dtype=int), [[0, 1]], [[2**63, 2**63 + 2]],
])
@pytest.mark.parametrize("workflow", ["threshold", "rate"])
def test_multiplicities_are_nonnegative_integer_resamples(counts, workflow):
    counts = np.asarray(counts)
    with pytest.raises(ValueError, match="multiplicit"):
        if workflow == "threshold":
            R.weighted_thresholds([1.0, 2.0], counts, pfa=0.5)
        else:
            R.bootstrap_rates([[0.0, 2.0]], [1.0], [True], counts)


@pytest.mark.parametrize("available", [[1, 0], [True], [[True, False]]])
def test_availability_is_one_boolean_per_resample(available):
    with pytest.raises(ValueError, match="availability"):
        R.bootstrap_rates([[0.0, 2.0]], [1.0, 1.0], available,
                          np.ones((2, 2), dtype=np.uint16))


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_bootstrap_batch_must_process_every_requested_resample(batch):
    with pytest.raises(ValueError, match="batch"):
        R.bootstrap_rates([[0.0, 2.0]], [1.0], [True],
                          np.ones((1, 2), dtype=np.uint16), batch=batch)


@pytest.mark.parametrize("workflow", ["threshold", "rate"])
def test_complex_responses_cannot_lose_their_imaginary_component(workflow):
    with pytest.raises(ValueError, match="real"):
        if workflow == "threshold":
            R.weighted_thresholds([1 + 1j, 2], np.ones((1, 2), dtype=int), pfa=0.5)
        else:
            R.bootstrap_rates([[1 + 1j, 2]], [1.0], [True],
                              np.ones((1, 2), dtype=int))


@pytest.mark.parametrize("thresholds,exact", [
    ([np.nan], False), ([np.inf], False), ([-1.0], False),
    ([1.5], True), ([True], True), ([0], True), ([R.SENTINEL], True),
    ([[1]], False),
])
def test_available_thresholds_preserve_the_decision_domain(thresholds, exact):
    with pytest.raises(ValueError, match="threshold"):
        R.bootstrap_rates([[1, 2]], thresholds, [True],
                          np.ones((1, 2), dtype=int), exact=exact)


def test_exact_h1_does_not_accept_boolean_requirements():
    with pytest.raises(ValueError, match="requirements"):
        R.bootstrap_rates([[True, 2]], [1], [True],
                          np.ones((1, 2), dtype=int), exact=True)


def test_unavailable_sentinel_threshold_censors_the_entire_resample():
    samples = np.array([[R.SENTINEL, 2], [2, 3]], dtype=object)
    rates = R.bootstrap_rates(samples, [2, R.SENTINEL], [True, False],
                              np.array([[2, 0], [0, 2]], dtype=np.uint16), exact=True)
    np.testing.assert_array_equal(rates[0], [1, 0])
    assert np.isnan(rates[1]).all()


def test_list_multiplicities_match_literal_repeated_observations():
    counts = [[0, 3, 0], [1, 0, 2]]
    thresholds, available = R.weighted_thresholds([1, 4, 9], counts, pfa=0.5)
    np.testing.assert_array_equal(thresholds, [4, 9])
    rates = R.bootstrap_rates([[1, 5, 10], [5, 5, 9]], thresholds, available, counts)
    np.testing.assert_allclose(rates, [[1, 1], [2 / 3, 0]])


@pytest.mark.parametrize("seed", [True, 1.5, -1, 1 << 64])
def test_seed_must_fit_the_recorded_uint64_identity(seed):
    with pytest.raises(ValueError, match="seed"):
        R.resampling([1, 2], [3, 4], replicates=2, seed=seed)


def test_invalid_frozen_bootstrap_count_creates_no_report_directory(tmp_path, monkeypatch):
    out = _cohort(tmp_path)
    load = R.load_contract

    def invalid_count(directory):
        plan, ids, geometry, bindings = load(directory)
        plan["loss_rule"]["bootstrap_replicates"] = 0
        return plan, ids, geometry, bindings

    monkeypatch.setattr(R, "load_contract", invalid_count)
    with pytest.raises(ValueError, match="bootstrap count"):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()


@pytest.mark.parametrize("failure", ["snapshot", "read", "reduce", "cell", "final", "marker", "interrupt"])
def test_report_failure_removes_only_owned_outputs_and_allows_retry(tmp_path, monkeypatch, failure):
    out = _cohort(tmp_path)
    log = out / "execution/interrupted-step.log"
    log.parent.mkdir()
    log.write_text("Retain supervisor diagnostics and raw checkpoints.\n")
    before = {str(p.relative_to(out)): R.sha(p) for p in out.rglob("*") if p.is_file()}
    error = KeyboardInterrupt if failure == "interrupt" else RuntimeError

    def fail(*args, **kwargs):
        raise error("injected report failure")

    with monkeypatch.context() as patch:
        if failure == "snapshot":
            freeze = R.freeze_report_sources

            def freeze_then_fail(*args, **kwargs):
                freeze(*args, **kwargs)
                fail()

            patch.setattr(R, "freeze_report_sources", freeze_then_fail)
        elif failure in {"read", "reduce", "interrupt"}:
            patch.setattr(R, "read_raw" if failure == "read" else "reduce_powers", fail)
        elif failure in {"cell", "final"}:
            write = R.write_json

            def write_then_fail(path, payload):
                write(path, payload)
                if (failure == "cell" and path.parent.name == "cells") or (
                        failure == "final" and path.name == "report.json"):
                    fail()

            patch.setattr(R, "write_json", write_then_fail)
        else:
            open_path = Path.open

            def partial_marker(path, mode="r", *args, **kwargs):
                if path.name == "report.sha256" and mode == "x":
                    with open_path(path, mode, *args, **kwargs) as stream:
                        stream.write("partial marker")
                    fail()
                return open_path(path, mode, *args, **kwargs)

            patch.setattr(Path, "open", partial_marker)
        with pytest.raises(error, match="injected report failure"):
            R.run(out, "evaluation")
    target = out / "reports/evaluation"
    assert not target.exists()
    assert {str(p.relative_to(out)): R.sha(p) for p in out.rglob("*") if p.is_file()} == before
    report = R.run(out, "evaluation")
    assert report["coverage_complete"]
    assert (target / "report.sha256").read_text().strip() == R.sha(target / "report.json")
    R.rehash(out, report["outputs_sha256"])


def test_existing_report_remains_exclusive_and_unchanged(tmp_path):
    out = _cohort(tmp_path)
    target = out / "reports/evaluation"
    target.mkdir(parents=True)
    (target / "report.json").write_bytes(b"preserve existing report bytes")
    (target / "report.sha256").write_bytes(b"preserve existing receipt bytes")
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(ValueError, match="overwrite"):
        R.run(out, "evaluation")
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
