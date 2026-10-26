"""Independent binomial arithmetic and calibration-artifact refusal tests."""
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
from math import comb
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = spec_from_file_location("model_calibration", ROOT / "tools/calibrate_proxy_confidence_band.py")
cal = module_from_spec(spec)
spec.loader.exec_module(cal)


def exact_tail(n, k):
    # exact integer arithmetic with p = 39/40, independent of the SciPy distributions
    return sum(Fraction(comb(n, j) * 39**j, 40**n) for j in range(k, n + 1))


def test_predeclared_order_rank_matches_independent_exact_binomial():
    alpha = Fraction(1, 120)
    assert exact_tail(400, 398) <= alpha < exact_tail(400, 397)
    assert float(exact_tail(400, 398)) == pytest.approx(.002547621772282387, rel=1e-13)
    assert cal.tolerance_rank(400, .975, .05 / 6) == 398
    result = cal.calibrated_limit(np.arange(400, dtype=float))
    assert result["multiplier"] == 397.
    assert result["rank_one_based"] == 398
    assert result["tail_probability"] == pytest.approx(float(exact_tail(400, 398)), rel=1e-13)


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [[1.]]])
def test_no_failed_calibration_score_may_be_dropped(scores):
    with pytest.raises(ValueError):
        cal.calibrated_limit(scores)


def test_unavailable_finite_tolerance_limit_and_multiplier_floor():
    with pytest.raises(ValueError, match="too few"):
        cal.tolerance_rank(100, .975, .05 / 6)
    assert cal.calibrated_limit(np.zeros(400))["multiplier"] == 1.


def test_record_identity_detects_reordering_missing_seed_and_separate_phase_seeds():
    records = [cal.record_identity(20261016031, "null_gaussian", i) for i in range(2)]
    cal.verify_records(records, 400, 20261016031, "null_gaussian")
    with pytest.raises(ValueError, match="identity"):
        cal.verify_records(records[::-1], 400, 20261016031, "null_gaussian")
    with pytest.raises(ValueError, match="identity"):
        cal.verify_records(records, 400, 20261016032, "null_gaussian")
    assert records[0]["dataset_seed"] != records[0]["bootstrap_seed"]


def calibration_fixture(directory):
    frozen = dict(source_sha256={}, bootstrap_replicates=399, critical_floor=1.,
                  calibration_datasets=400, calibration_seed=20261016031)
    limits, hashes = {}, {}
    for name in cal.MATCHED:
        rows = [dict(cal.record_identity(frozen["calibration_seed"], name, i),
                     required_multiplier=i / 200, unavailable=False) for i in range(400)]
        filename = f"calibrate-{name}.json"
        cal.validation.write_json(directory / filename,
            dict(protocol_sha256="protocol", multiplier=None, records=rows))
        hashes[filename] = cal.validation.sha(directory / filename)
        limits[name] = cal.calibrated_limit([row["required_multiplier"] for row in rows])
    artifact = dict(protocol_sha256="protocol", source_sha256={}, bootstrap_replicates=399,
                    minimum_critical=1., calibration_result_sha256=hashes,
                    model_limits=limits, multiplier=max(v["multiplier"] for v in limits.values()))
    cal.validation.write_json(directory / "calibration.json", artifact)
    return frozen, artifact


def test_calibration_loader_recomputes_order_statistics(tmp_path):
    frozen, artifact = calibration_fixture(tmp_path)
    assert cal.verify_calibration(tmp_path, frozen, "protocol") == artifact
    for value in artifact["model_limits"].values():
        value["multiplier"] += 1
    artifact["multiplier"] += 1
    cal.validation.write_json(tmp_path / "calibration.json", artifact)
    with pytest.raises(ValueError, match="reproduce"):
        cal.verify_calibration(tmp_path, frozen, "protocol")


def test_calibration_loader_refuses_changed_records(tmp_path):
    frozen, _ = calibration_fixture(tmp_path)
    (tmp_path / "calibrate-null_gaussian.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        cal.verify_calibration(tmp_path, frozen, "protocol")


def test_nonfinite_target_is_not_silently_scored_as_zero():
    frames, windows, clusters = cal.validation.analytic_dataset("null_gaussian", 5, 320)
    with pytest.raises(ValueError, match="finite truth"):
        cal.validation.evaluate_dataset(frames, windows, clusters,
            lambda words: np.full(len(words), np.nan),
            SimpleNamespace(bootstrap_replicates=19), 9, 1.)


@pytest.mark.parametrize("change", ["unavailable", "missing", "seed"])
def test_calibration_loader_refuses_rehashed_incomplete_or_invalid_records(tmp_path, change):
    import json

    frozen, artifact = calibration_fixture(tmp_path)
    name = "calibrate-null_gaussian.json"
    path = tmp_path / name
    saved = json.loads(path.read_text())
    if change == "unavailable":
        saved["records"][0]["unavailable"] = True
    elif change == "missing":
        saved["records"].pop()
    else:
        saved["records"][0]["dataset_seed"] += 1
    cal.validation.write_json(path, saved)
    artifact["calibration_result_sha256"][name] = cal.validation.sha(path)
    cal.validation.write_json(tmp_path / "calibration.json", artifact)
    with pytest.raises(ValueError):
        cal.verify_calibration(tmp_path, frozen, "protocol")


def test_protocol_rejects_same_calibration_validation_seed():
    with pytest.raises(ValueError, match="seeds must differ"):
        cal.protocol(SimpleNamespace(calibration_seed=1, validation_seed=1), {})
