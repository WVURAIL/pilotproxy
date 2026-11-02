"""Rescore CLI retains the population and refuses occupied output paths."""

import hashlib
import json

import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import rescore
from test_rescore import _cache_on_product


def test_rescore_cli_keeps_population_and_threshold_with_new_geometry(tmp_path, capsys):
    cache, original = _cache_on_product(tmp_path, 128)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cache, cache.with_suffix(".npz"))}
    output = tmp_path / "rescored"
    assert rescore.main([str(cache), str(output), "--anchor-bin", "120",
                         "--designated-half-width", "1", "--marker-guard-bins", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    meta = json.loads((output / "ch36.json").read_text())
    geometry = meta["score_geometry"]
    assert result["anchor_bin"] == 120
    assert geometry["designated_bins"] == [119, 120, 121]
    assert not set(geometry["bulk_bins"]) & set(range(117, 124))
    assert result["rho"] == (len(geometry["bulk_bins"]) + 1) // 2
    assert meta["current_threshold"] == {"kind": "finite", "eta_q16": "65536"}
    assert meta["current_assignment_unchanged"] is True
    assert meta["current_exact_mask_unchanged"] is False
    assert meta["rescore"]["current_word"] == 65536
    assert meta["npz_sha256"] == hashlib.sha256((output / "ch36.npz").read_bytes()).hexdigest()
    with np.load(output / "ch36.npz", allow_pickle=False) as arrays:
        for field in ("frame_event_index", "source_row_index", "probability_weight"):
            np.testing.assert_array_equal(arrays[field], original[field])
        assert arrays["score_computed"].tolist() == [True] * len(original["source_row_index"])
        np.testing.assert_array_equal(arrays["current_keep"], arrays["supported"] & (arrays["required_q16"] <= 65536))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


@pytest.mark.parametrize("occupied", ["ch36.json", "ch36.npz"])
def test_rescore_cli_refuses_to_replace_either_existing_output(tmp_path, occupied):
    cache, _ = _cache_on_product(tmp_path, 128)
    output = tmp_path / "rescored"
    output.mkdir()
    existing = output / occupied
    existing.write_bytes(b"existing result must survive")
    with pytest.raises(ValueError, match="already exists"):
        rescore.main([str(cache), str(output)])
    assert existing.read_bytes() == b"existing result must survive"
    assert {p.name for p in output.iterdir()} == {occupied}


def test_rescore_cli_rejects_changed_cache_before_publication(tmp_path):
    cache, _ = _cache_on_product(tmp_path, 128)
    with cache.with_suffix(".npz").open("ab") as stream:
        stream.write(b"changed input")
    output = tmp_path / "rescored"
    with pytest.raises(ValueError, match="Cache npz differs"):
        rescore.main([str(cache), str(output)])
    assert not output.exists()
