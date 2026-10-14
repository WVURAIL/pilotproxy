"""Final fine histograms preserve frame populations and exact mask boundaries."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pilot_proxy.characterization import histogram_frames as coarse
from pilot_proxy.detectors.narrowband_marker import anchors, scores
from pilot_proxy.detectors.narrowband_marker import fine_histogram_frames as export
from pilot_proxy.fine_decision import fine_required_multiplier_q16
from pilot_proxy.testbench.fine_validation_scores import construct_geometry, float_scores
from test_control_band_export import _control_product


@pytest.mark.parametrize("anchor", [0, 1, 255])
def test_scores_preserve_zeros_invalid_support_and_exact_overflow(anchor):
    bulk = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1)
    designated = list(anchors.designated_set(anchor))
    powers = np.ones((9, 3, 256), dtype=np.uint64)
    powers[1, 0, designated] = 0                         # A genuine zero score.
    powers[2, 0, bulk] = 0                              # Positive / zero.
    powers[3, 0] = 0                                    # Zero / zero is non-firing.
    powers[4, 1:, np.flatnonzero(bulk)[62:]] = 0           # Only 62 references remain.
    powers[5, 1:, designated] = 0                        # No designated references.
    powers[6, 0, designated] = np.iinfo(np.uint64).max    # Boundary exceeds uint64.
    powers[7, 0, designated] = 900                       # Health-excluded row.
    powers[8] = np.iinfo(np.uint64).max                  # Reference addition overflows uint64.
    selected = np.ones(9, dtype=bool)
    selected[7] = False
    result = export.final_score_arrays(powers, selected, anchor_bin=anchor, bulk_mask=bulk)
    np.testing.assert_array_equal(result["fine_source_row_index"], np.arange(9))
    np.testing.assert_equal(result["fine_score_z63"], [1, 0, np.inf, 0, np.nan, 0,
                                                     float(2**64 - 1), np.nan, 1])
    np.testing.assert_array_equal(result["fine_always_masked"], [0, 0, 1, 0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(result["fine_rank_zero"], [0, 0, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result["fine_required_multiplier_q16"],
                                  [65536, 1, 0, 1, 0, 1, 0, 0, 65536])
    assert result["fine_usable_bulk_count"][4] == 62
    assert result["fine_usable_designated_count"][5] == 0
    assert result["fine_score_zero"][[1, 3, 5]].all()
    assert not result["fine_score_computed"][7]
    for row in np.flatnonzero(selected):
        reference = fine_required_multiplier_q16(
            powers[row], anchor_bin=anchor, designated_half_width=2, bulk_mask=bulk, cfar_rank=62)
        expected_always = not reference.valid or reference.multiplier_q16 == scores.ALWAYS_MASKED_Q16
        assert result["fine_always_masked"][row] == expected_always
        if not expected_always:
            assert int(result["fine_required_multiplier_q16"][row]) == reference.multiplier_q16
    # Existing bench zero-reference and zero-rank semantics remain the same.
    bench = float_scores(powers, construct_geometry(anchor))
    np.testing.assert_allclose(result["fine_score_z63"][selected], bench["response"][selected, 1], equal_nan=True)


def test_rank_stays_63_when_references_are_missing_and_ties_keep():
    anchor = 1
    bulk = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1)
    bins = np.flatnonzero(bulk)
    powers = np.ones((2, 3, 256), dtype=np.uint64)
    powers[:, 0, bins] = np.arange(1, bins.size + 1)
    powers[:, 0, list(anchors.designated_set(anchor))] = 90
    powers[1, 1:, bins[:10]] = 0
    result = export.final_score_arrays(powers, np.ones(2, bool), anchor_bin=anchor, bulk_mask=bulk)
    np.testing.assert_equal(result["fine_score_z63"], [90 / 63, 90 / 73])
    for row, denominator in enumerate((63, 73)):
        expected = (90 * 65536 + denominator - 1) // denominator
        assert result["fine_required_multiplier_q16"][row] == expected
    tied = np.full((1, 3, 256), 3, dtype=np.uint64)
    assert export.final_score_arrays(tied, np.ones(1, bool), anchor_bin=anchor,
                                    bulk_mask=bulk)["fine_required_multiplier_q16"][0] == 65536


def test_bad_geometry_and_noninteger_powers_are_refused():
    powers = np.ones((1, 3, 256), dtype=np.uint64)
    bulk = anchors.bulk_mask(0, pad_factor=2, guard_fine_bins=1)
    with pytest.raises(ValueError, match="uint64"):
        export.final_score_arrays(powers.astype(float), np.ones(1, bool), anchor_bin=0, bulk_mask=bulk)
    with pytest.raises(ValueError, match="boolean frame"):
        export.final_score_arrays(powers, np.ones(1, np.uint8), anchor_bin=0, bulk_mask=bulk)
    bulk[4] = False
    with pytest.raises(ValueError, match="125- or 126"):
        export.final_score_arrays(powers, np.ones(1, bool), anchor_bin=0, bulk_mask=bulk)


def _source_export(tmp_path, monkeypatch):
    product = _control_product(tmp_path, monkeypatch)
    with np.load(product, allow_pickle=False) as saved:
        fields = {key: saved[key] for key in saved.files}
    # Retain untimed rows and a health failure; neither becomes a histogram sample.
    fields["unit_delta_time"][0] = np.nan
    fields["baseband_power_linear"][12] = 128
    np.savez(product, **fields)
    frames = tmp_path / "coarse"
    coarse.run_control(product, frames)
    return product, frames


def test_export_is_additive_authenticated_and_has_no_per_bin_values(tmp_path, monkeypatch):
    product, frames = _source_export(tmp_path, monkeypatch)
    before = {path.name: coarse.sha(path) for path in frames.iterdir()}
    output = tmp_path / "fine"
    manifest = export.run(frames, output)
    assert before == {path.name: coarse.sha(path) for path in frames.iterdir()}
    assert manifest["passed"] is True
    assert not manifest["per_bin_f_values_saved"]
    with np.load(frames / "ch37.npz", allow_pickle=False) as original, np.load(output / "ch37.npz", allow_pickle=False) as saved:
        for key in original.files:
            assert saved[key].dtype == original[key].dtype
            assert saved[key].shape == original[key].shape
            assert saved[key].tobytes() == original[key].tobytes()
        for key in set(saved.files) - set(original.files):
            assert saved[key].shape == (120,)
        expected = saved["selected"] & saved["finite_time"] & np.isfinite(saved["fine_score_z63"])
        np.testing.assert_array_equal(saved["fine_histogram_eligible"], expected)
        assert np.isfinite(saved["fine_score_z63"][:10]).all()
        assert not saved["fine_histogram_eligible"][:10].any()
        assert np.isnan(saved["fine_score_z63"][12])
    metadata = json.loads((output / "ch37.json").read_text())
    assert metadata["geometry"]["anchor_source"] == "nominal product marker; no fitted control anchor"
    assert metadata["geometry"]["rho_one_based"] == 63
    assert metadata["exact_boundary"]["logical_always_masked_sentinel"] == str(2**64)
    assert metadata["product"]["sha256"] == coarse.sha(product)
    for name, digest in manifest["files"].items():
        assert coarse.sha(output / name) == digest
    with pytest.raises(FileExistsError):
        export.run(frames, output)


def _reseal(frames, filename):
    manifest_path = frames / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][filename] = coarse.sha(frames / filename)
    manifest_path.write_text(json.dumps(manifest))


def test_source_tampering_and_coordinate_mismatch_fail_closed(tmp_path, monkeypatch):
    product, frames = _source_export(tmp_path, monkeypatch)
    metadata_path = frames / "ch37.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["product"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source hash differs"):
        export.run(frames, tmp_path / "bad-metadata")
    _reseal(frames, metadata_path.name)
    with pytest.raises(ValueError, match="source hash differs"):
        export.run(frames, tmp_path / "bad-product")
    metadata["product"]["sha256"] = coarse.sha(product)
    with np.load(frames / "ch37.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["frame_index"][0] += 1
    np.savez(frames / "ch37.npz", **arrays)
    metadata["export_sha256"] = coarse.sha(frames / "ch37.npz")
    metadata_path.write_text(json.dumps(metadata))
    _reseal(frames, metadata_path.name)
    _reseal(frames, "ch37.npz")
    with pytest.raises(ValueError, match="frame_index"):
        export.run(frames, tmp_path / "bad-frame")


def test_saved_selector_anchor_is_used_without_refitting(tmp_path, monkeypatch):
    _, frames = _source_export(tmp_path, monkeypatch)
    metadata_path = frames / "ch37.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("role")
    anchor = 17
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"channel": 37, "freq_id": 491,
        "product_sha256": metadata["product"]["sha256"],
        "sections": {"null": {"anchor_bin": anchor, "anchor_source": "saved test selector", "bulk_size": 126}}}))
    metadata["source_records"]["ledger"] = {"path": str(ledger_path), "sha256": coarse.sha(ledger_path)}
    metadata_path.write_text(json.dumps(metadata))
    _reseal(frames, metadata_path.name)
    output = tmp_path / "fine-saved-anchor"
    export.run(frames, output)
    result = json.loads((output / "ch37.json").read_text())
    assert result["geometry"]["anchor_bin"] == anchor
    assert result["geometry"]["anchor_source"] == "saved test selector"
    assert result["geometry"]["intended_bulk_size"] == 126
    ledger_path.write_text("{}")
    with pytest.raises(ValueError, match="source hash differs"):
        export.run(frames, tmp_path / "stale-anchor")
