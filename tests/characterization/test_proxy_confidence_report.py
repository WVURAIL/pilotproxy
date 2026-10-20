"""Frozen-input replay, intact temporal clusters, and policy-margin semantics."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import knee
from pilot_proxy.characterization.threshold_rule import ChannelFrames


SPEC = importlib.util.spec_from_file_location(
    "proxy_confidence_report_under_test",
    Path(__file__).resolve().parents[2] / "tools" / "report_proxy_confidence_band.py",
)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample_frames():
    return ChannelFrames(
        channel=99,
        required=np.array([10, 20, 30, 40], dtype=np.uint64),
        weight=np.full(4, .25),
        supported=np.array([True, False, True, True]),
        event=np.array([3, 3, 9, 9]),
        # dyadic, so the excess formed from integer frame powers is exact
        excess=np.array([-.25, .125, .25, .375]),
        knee_word=20,
        current_word=30,
        time=np.array([86400 - 10, 86400 + 10, 3 * 86400, 3 * 86400 + 1], dtype=float),
    )


def score_fixture(tmp_path):
    scores = tmp_path / "scores"
    scores.mkdir()
    product = tmp_path / "original-product.npz"
    frames = sample_frames()
    # the loaded frames carry the knee of the current rule; the cache records rule v1's
    frames.knee_word = int(knee.propose([int(w) for w in frames.required], [str(int(e)) for e in frames.event],
                                        valid=[bool(v) for v in frames.supported])["knee_word_q16"])
    rows = np.array([3, 0, 2, 1])
    # The excess is formed from the frame powers, the fine powers' all-bin
    # sums (kernel core 3.0.0): with norms 1 : 2 (mu_0 = 1), P_1 = P_2 = 2**40
    # and P_0 = (e + 1) 2**40, exact for these dyadic excesses.
    fine = np.zeros((4, 3, 4), dtype=np.uint64)
    fine[rows, 0, 0] = np.rint((np.asarray(frames.excess) + 1.0) * 2.0**40).astype(np.uint64)
    fine[:, 1, 1] = 1 << 40
    fine[:, 2, 2] = 1 << 40
    np.savez(product, fine_power_u64=fine, target_norm_sq=np.asarray([1], dtype=np.int64),
             reference_norm_sum_sq=np.asarray([2], dtype=np.int64))
    meta = {
        "original_product_identity": {"path": str(product), "bytes": product.stat().st_size},
        "knee_threshold": {"eta_q16": knee.propose([int(w) for w in frames.required], [str(int(e)) for e in frames.event],
                                                   valid=[bool(v) for v in frames.supported],
                                                   rule=knee.RULE_ID_V1)["knee_word_q16"]},
        "current_threshold": {"eta_q16": str(frames.current_word)},
    }
    (scores / "ch99.json").write_text(json.dumps(meta))
    np.savez(
        scores / "ch99.npz", source_row_index=rows, required_q16=frames.required,
        probability_weight=frames.weight, supported=frames.supported,
        frame_event_index=frames.event, current_keep=np.array([True, False, True, False]),
        frame_time=frames.time,
    )
    return scores, product, frames


def test_live_freeze_then_offline_replay_preserves_exact_analysis_arrays(tmp_path, monkeypatch):
    scores, product, expected = score_fixture(tmp_path)
    first = tmp_path / "first"
    frames, provenance = REPORT.freeze_channel(99, scores, first)
    assert provenance["original_archive_sha256_verified"] is False
    assert "size checked only" in provenance["verification_scope"]
    assert provenance["score_sha256"] == REPORT.digest(scores / "ch99.npz")
    assert provenance["score_metadata_sha256"] == REPORT.digest(scores / "ch99.json")
    assert provenance["sha256"] == REPORT.digest(first / "inputs/ch99.npz")
    for key in ("required", "weight", "supported", "event", "excess", "time"):
        np.testing.assert_array_equal(getattr(frames, key), getattr(expected, key))
    # The frozen replay must work with neither an original product nor a score
    # reader. Removing this test-owned temporary product makes that observable.
    product.unlink()
    monkeypatch.setattr(REPORT.rule, "load_frames", lambda *args: pytest.fail("offline replay read live scores"))
    replay, replay_provenance = REPORT.freeze_channel(
        99, tmp_path / "missing-scores", tmp_path / "replay", first / "inputs")
    for key in ("required", "weight", "supported", "event", "excess", "time"):
        np.testing.assert_array_equal(getattr(replay, key), getattr(expected, key))
    assert replay.knee_word == expected.knee_word and replay.current_word == expected.current_word
    assert replay_provenance["sha256"] == REPORT.digest(tmp_path / "replay/inputs/ch99.npz")


def test_frozen_checksum_is_verified_before_loading_or_repacking(tmp_path):
    scores, _, _ = score_fixture(tmp_path)
    first = tmp_path / "first"
    REPORT.freeze_channel(99, scores, first)
    frozen = first / "inputs/ch99.npz"
    # Appended bytes may leave a ZIP readable, so a mere successful NPZ load
    # would not be an adequate identity check.
    frozen.write_bytes(frozen.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        REPORT.freeze_channel(99, None, tmp_path / "replay", first / "inputs")
    assert not (tmp_path / "replay/inputs/ch99.npz").exists()


def test_same_frozen_destination_is_reusable_but_changed_arrays_are_refused(tmp_path):
    scores, _, _ = score_fixture(tmp_path)
    output = tmp_path / "output"
    REPORT.freeze_channel(99, scores, output)
    path = output / "inputs/ch99.npz"
    digest = REPORT.digest(path)
    REPORT.freeze_channel(99, scores, output)
    assert REPORT.digest(path) == digest
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["excess"] = arrays["excess"] + .001
    np.savez_compressed(path, **arrays)
    changed = REPORT.digest(path)
    with pytest.raises(ValueError, match="frozen input changed"):
        REPORT.freeze_channel(99, scores, output)
    assert REPORT.digest(path) == changed  # refusal must not overwrite evidence


def test_day_clusters_keep_midnight_crossing_and_unordered_events_intact():
    f = sample_frames()
    labels = REPORT.event_time_clusters(f, 1)
    np.testing.assert_array_equal(labels, [0, 0, 3, 3])
    order = np.array([2, 1, 3, 0])
    shuffled = replace(f, event=f.event[order], time=f.time[order])
    np.testing.assert_array_equal(REPORT.event_time_clusters(shuffled, 1), labels[order])


def test_week_boundary_uses_first_frame_and_preserves_original_weights():
    f = sample_frames()
    f = replace(f, time=np.array([7 * 86400 - 1, 7 * 86400 + 1, 10 * 86400, 10 * 86400 + 1.]))
    before = f.weight.copy()
    np.testing.assert_array_equal(REPORT.event_time_clusters(f, 7), [0, 0, 1, 1])
    np.testing.assert_array_equal(f.weight, before)


def test_nonfinite_event_time_is_refused():
    f = sample_frames()
    for value in (np.nan, np.inf):
        time = f.time.copy()
        time[:2] = value
        # NumPy may warn while reducing a deliberate NaN; the contractual
        # result being tested is the explicit refusal, not that warning.
        with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="finite event times"):
            REPORT.event_time_clusters(replace(f, time=time), 1)


def test_policy_substitution_encodes_half_margin_not_sampling_error():
    f = sample_frames()
    words = np.array([10, 20, 30], dtype=np.uint64)
    mask = np.array([False, False, True, True])
    window = {
        "signed_mean": np.array([-.2, .1, .2]), "upper": np.array([0., .5, .6]),
        "eligible": np.array([True, True, False]), "se": np.array([99., 99., 99.]),
        "kept_weight": np.array([.1, .2, .3]), "kept_frames": np.array([1, 2, 3]),
        "kept_events": np.array([1, 2, 2]),
    }
    curve = REPORT.curve_for_rule(f, words, mask, window)
    assert REPORT.rule.GUARD_STANDARD_ERRORS == 2
    np.testing.assert_allclose(curve["kappa"], [.2, .4, .6])
    np.testing.assert_allclose(curve["x"], [0, .1, .2])
    np.testing.assert_allclose(curve["se"][:2], [0, .2])
    np.testing.assert_allclose((curve["x"] + 2 * curve["se"])[:2], window["upper"][:2])
    assert np.isposinf(curve["se"][2])
    np.testing.assert_array_equal(window["se"], [99., 99., 99.])


def test_negative_encoded_margin_is_conservatively_clamped():
    f = sample_frames()
    window = {"signed_mean": np.array([.3]), "upper": np.array([.2]),
              "eligible": np.array([True]), "kept_weight": np.array([.5]),
              "kept_frames": np.array([2]), "kept_events": np.array([1])}
    curve = REPORT.curve_for_rule(f, np.array([20], np.uint64), np.ones(4, bool), window)
    assert curve["se"][0] == 0
    assert (curve["x"] + 2 * curve["se"])[0] == .3


def test_serialization_preserves_refusal_status_while_using_json_null():
    value = REPORT.clean({"status": "bootstrap_unavailable", "upper": np.array([np.inf, np.nan]),
                          "eligible": np.array([False, True]), "word": np.uint64(2**63 + 3)})
    assert value == {"status": "bootstrap_unavailable", "upper": [None, None],
                     "eligible": [False, True], "word": 2**63 + 3}
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("mismatch", ["source", "replicates"])
def test_model_artifact_mismatch_refuses_before_output_writes(tmp_path, monkeypatch, mismatch):
    import sys

    sys.path.insert(0, str(Path(REPORT.__file__).parent))
    from calibrate_proxy_confidence_band import source_hashes

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    protocol = dict(source_sha256=source_hashes(), bootstrap_replicates=399)
    if mismatch == "source":
        protocol["source_sha256"] = {}
    else:
        protocol["bootstrap_replicates"] = 1999
    (calibration_dir / "protocol.json").write_text(json.dumps(protocol))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [REPORT.__file__, "--frozen-inputs", str(tmp_path / "inputs"),
        "--output", str(output), "--replicates", "399", "--model-calibration",
        str(calibration_dir / "calibration.json")])
    with pytest.raises(ValueError, match="mismatch"):
        REPORT.main()
    assert not output.exists()
