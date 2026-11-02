"""Tests of the fine score profiles and populations, covering the exact boundaries, the profile guard and the strict reader."""

import numpy as np
import pytest

from pilot_proxy import runtime_calibration as rc
from pilot_proxy.fine_decision import fine_mask_decision, fine_required_multiplier_q16


def profile(name="CH29"):
    mask = np.zeros(256, bool)
    mask[::2] = True
    mask[(100 + np.arange(-4, 5)) % 256] = False
    words = [sum(int(mask[i * 64 + j]) << j for j in range(64)) for i in range(4)]
    return {
        "schema_version": rc.PROFILE_SCHEMA,
        "profile_id": name,
        "decision_version": "fine_decision_v1",
        "score_encoding": rc.SCORE_ENCODING,
        "support_rule": rc.SUPPORT_RULE,
        "receiver_profile_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "producer_sha256": "c" * 64,
        "packing_contract_sha256": "d" * 64,
        "input_state_contract_sha256": "1" * 64,
        "input_population_id": "pathfinder-fixed128",
        "physical_channel": 29,
        "num_input_streams": 128,
        "detector_window_samples": 64,
        "windows_per_stream": 128,
        "frame_size_samples": 8192,
        "fine_bins": 256,
        "anchor_bin": 100,
        "designated_half_width": 2,
        "bulk_mask_words_hex": [hex(w) for w in words],
        "cfar_rank": 61,
    }


def test_required_boundary_matches_actual_integer_runtime():
    p = profile()
    power = np.ones((3, 256), np.uint64)
    power[0, 100] = 7
    kwargs = {k: p[k] for k in ["anchor_bin", "designated_half_width", "cfar_rank"]}
    kwargs["bulk_mask"] = [int(w, 16) for w in p["bulk_mask_words_hex"]]
    required = fine_required_multiplier_q16(power, **kwargs).multiplier_q16
    assert required == 7 * 65536
    assert fine_mask_decision(power, multiplier_q16=required, **kwargs).mask == 0
    assert fine_mask_decision(power, multiplier_q16=required - 1, **kwargs).mask == 1


def test_profile_guard_and_geometry_refusal():
    p = profile()
    assert rc.validate_score_profile(p) == 123
    p["frame_size_samples"] = 16384
    with pytest.raises(ValueError, match="Frame size"):
        rc.validate_score_profile(p)
    p = profile()
    words = [int(w, 16) for w in p["bulk_mask_words_hex"]]
    words[100 // 64] |= 1 << (100 % 64)
    p["bulk_mask_words_hex"] = [hex(w) for w in words]
    with pytest.raises(ValueError, match="designated/guard"):
        rc.validate_score_profile(p)


@pytest.mark.parametrize("anchor,population", [(100, 125), (101, 126)])
def test_profile_accepts_characterization_bulk(anchor, population):
    p = profile()
    mask = np.zeros(256, bool)
    mask[::2] = True
    mask[(anchor + np.arange(-2, 3)) % 256] = False
    words = [sum(int(mask[i * 64 + j]) << j for j in range(64)) for i in range(4)]
    p.update(anchor_bin=anchor, bulk_mask_words_hex=[hex(w) for w in words])
    assert rc.validate_score_profile(p) == population
    p["designated_half_width"] = 0
    words[(anchor + 2) % 256 // 64] |= 1 << ((anchor + 2) % 256 % 64)
    p["bulk_mask_words_hex"] = [hex(w) for w in words]
    with pytest.raises(ValueError, match="designated/guard"):
        rc.validate_score_profile(p)


@pytest.mark.parametrize("value", [None, "", "not-a-digest"])
def test_missing_or_malformed_input_state_contract_refused(value):
    p = profile()
    p["input_state_contract_sha256"] = value
    with pytest.raises(ValueError, match="input_state_contract_sha256"):
        rc.validate_score_profile(p)


def test_json_duplicate_keys_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        rc._read(path)
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        rc._read(path)
