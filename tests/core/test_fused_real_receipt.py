"""Tests for tools/fused_real_receipt.py on tiny synthetic fixtures."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.fine_decision import fine_mask_decision

SOURCE = Path(__file__).resolve().parents[2] / "tools" / "fused_real_receipt.py"
spec = importlib.util.spec_from_file_location("fused_real_receipt", SOURCE)
receipt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = receipt
spec.loader.exec_module(receipt)

ANCHOR = 107
CHANNEL = 35
FREQ_ID = 521
CENTER_MHZ = 596.484375


def _thresholds(path: Path, *, word: int = 400_000, rho: int = 63) -> Path:
    bulk = [b for b in range(0, 256, 2) if abs(((b - ANCHOR + 128) % 256) - 128) > 3]
    payload = {
        "channels": [
            {
                "channel": CHANNEL,
                "pilot_frequency_id": FREQ_ID,
                "score_geometry": {
                    "anchor_bin": ANCHOR,
                    "designated_half_width": 2,
                    "bulk_bins": bulk,
                    "cfar_rank_zero_based": 62,
                    "rho_one_based": rho,
                },
                "threshold": {"eta_q16": str(word), "kind": "finite"},
            }
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def _fine(seed: int = 7, *, pilot: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = rng.integers(1_000, 2_000, size=(3, 256)).astype(np.uint64)
    fine[0, ANCHOR] += np.uint64(pilot)
    return fine


def _fused_from_reference(fine, geometry, multiplier, powers=(1, 2, 3)):
    decision = fine_mask_decision(
        fine,
        anchor_bin=geometry.anchor_bin,
        designated_half_width=geometry.designated_half_width,
        bulk_mask=geometry.bulk_mask,
        cfar_rank=geometry.cfar_rank,
        multiplier_q16=multiplier,
    )
    return {
        "mask": decision.mask,
        "rank_valid": int(decision.valid),
        "n_bulk": decision.n_bulk,
        "fine": fine.copy(),
        "powers": np.asarray(powers, dtype=np.uint64),
    }


def test_load_geometry_reads_word_rank_and_bulk(tmp_path):
    geometry = receipt.load_geometry(_thresholds(tmp_path / "t.json.gz"))[FREQ_ID]
    assert (geometry.channel, geometry.anchor_bin, geometry.cfar_rank) == (CHANNEL, ANCHOR, 62)
    assert geometry.multiplier_q16 == 400_000
    assert geometry.bulk_mask.dtype == bool and geometry.bulk_mask.shape == (256,)
    assert not geometry.bulk_mask[ANCHOR] and geometry.bulk_mask[0]


def test_load_geometry_refuses_rank_rho_disagreement(tmp_path):
    with pytest.raises(ValueError, match="rho and rank"):
        receipt.load_geometry(_thresholds(tmp_path / "t.json.gz", rho=60))


def test_compare_frame_passes_on_identical_outputs(tmp_path):
    geometry = receipt.load_geometry(_thresholds(tmp_path / "t.json.gz"))[FREQ_ID]
    fine = _fine(pilot=50_000)
    fused = _fused_from_reference(fine, geometry, geometry.multiplier_q16)
    differences, counted, mask = receipt.compare_frame(
        staged_fine=fine, staged_powers=[1, 2, 3], fused=fused, geometry=geometry,
        multiplier_q16=geometry.multiplier_q16, with_support=True,
    )
    assert differences == []
    assert mask == 1
    assert counted == {"mask": 1, "rank_valid": 1, "n_bulk": 1,
                       "fine_powers": 768, "coarse_powers": 3}


@pytest.mark.parametrize("field", ["mask", "n_bulk", "fine_powers", "coarse_powers"])
def test_compare_frame_reports_each_kind_of_difference(tmp_path, field):
    geometry = receipt.load_geometry(_thresholds(tmp_path / "t.json.gz"))[FREQ_ID]
    fine = _fine()
    fused = _fused_from_reference(fine, geometry, geometry.multiplier_q16)
    if field == "mask":
        fused["mask"] ^= 1
    elif field == "n_bulk":
        fused["n_bulk"] += 1
    elif field == "fine_powers":
        fused["fine"][2, 9] += np.uint64(1)
    else:
        fused["powers"][1] += np.uint64(1)
    differences, _, _ = receipt.compare_frame(
        staged_fine=fine, staged_powers=[1, 2, 3], fused=fused, geometry=geometry,
        multiplier_q16=geometry.multiplier_q16, with_support=True,
    )
    assert [d["field"] for d in differences] == [field]
    if field == "fine_powers":
        assert (differences[0]["first_term"], differences[0]["first_bin"]) == (2, 9)


def test_boundary_multipliers_straddle_the_exact_required_word(tmp_path):
    geometry = receipt.load_geometry(_thresholds(tmp_path / "t.json.gz"))[FREQ_ID]
    fine = _fine(seed=3, pilot=5_000)
    below, at = receipt.boundary_multipliers(fine, geometry)
    assert at == below + 1
    for word, want in ((below, 1), (at, 0)):
        decision = fine_mask_decision(
            fine, anchor_bin=ANCHOR, designated_half_width=2,
            bulk_mask=geometry.bulk_mask, cfar_rank=62, multiplier_q16=word,
        )
        assert decision.mask == want


def test_product_index_maps_event_and_frame_to_row(tmp_path):
    fine = np.stack([_fine(seed=s) for s in range(3)])
    path = tmp_path / "521.npz"
    np.savez(
        path,
        freq_id=np.array([FREQ_ID]),
        unit_event_id=np.array([11, 12]),
        frame_unit_index=np.array([0, 0, 1], dtype=np.int32),
        frame_in_unit=np.array([0, 1, 0], dtype=np.int32),
        fine_power_u64=fine,
        p_target_u64=np.arange(3, dtype=np.uint64).reshape(3, 1),
        p_ref_lower_u64=np.arange(3, dtype=np.uint64).reshape(3, 1),
        p_ref_upper_u64=np.arange(3, dtype=np.uint64).reshape(3, 1),
        weights_hash=np.array("abc"),
    )
    index = receipt.ProductIndex(path)
    assert (index.freq_id, index.row(11, 1), index.row(12, 0), index.row(12, 1)) == (
        FREQ_ID, 1, 2, None)
    assert index.powers.shape == (3, 3)


def _part(path: Path, files, *, frames: int, library: str = "lib") -> Path:
    stats = {key: 0 for key in receipt.COUNT_KEYS}
    stats.update(files=len(files), frames=frames, channel=CHANNEL, freq_id=FREQ_ID,
                 anchor_bin=ANCHOR, designated_half_width=2, bulk_bins=124,
                 cfar_rank_zero_based=62, multiplier_q16="400000")
    record = {
        "schema": receipt.SCHEMA, "tool": {"sha256": "t"}, "library": {"sha256": library},
        "host": {}, "weights": {}, "thresholds": {}, "instrument": {}, "launches": {},
        "command": ["python", "x"], "result": "PASS", "products": {},
        "files": [{"path": f} for f in files], "channels": {str(CHANNEL): stats},
        "first_difference": None,
        "totals": {"frames": frames, "fused_launches": 2 * frames,
                   "words_compared": {"fused_legacy.mask": frames}},
        "timings_s": {"wall": 1.0},
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_merge_sums_disjoint_parts_and_refuses_overlap(tmp_path):
    a = _part(tmp_path / "a.json", ["/x/1.h5"], frames=4)
    b = _part(tmp_path / "b.json", ["/x/2.h5", "/x/3.h5"], frames=6)
    merged = receipt.merge_receipts([a, b])
    assert merged["result"] == "PASS"
    assert merged["totals"]["frames"] == 10
    assert merged["channels"][str(CHANNEL)]["files"] == 3
    assert merged["totals"]["words_compared"] == {"fused_legacy.mask": 10}
    with pytest.raises(SystemExit, match="more than one part"):
        receipt.merge_receipts([a, a])
    c = _part(tmp_path / "c.json", ["/x/4.h5"], frames=1, library="other")
    with pytest.raises(SystemExit, match="library differs"):
        receipt.merge_receipts([a, c])


@pytest.mark.cuda
def test_receipt_end_to_end_on_a_tiny_synthetic_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("cupy")
    from pilot_proxy.kernel import FStatKernel
    from pilot_proxy.paths import DEFAULT_LIB_PATH

    try:
        kernel = FStatKernel(DEFAULT_LIB_PATH)
    except (FileNotFoundError, OSError):
        pytest.skip("libfstatistic.so is not built")
    if not kernel.supports_fused_fine_mask_with_support():
        pytest.skip("library predates the fused support outputs")
    rng = np.random.default_rng(20261007)
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "baseband", data=rng.integers(0, 256, size=(2 * 16384, 8), dtype=np.uint8)
        )
        handle.attrs["freq"] = CENTER_MHZ
        handle.attrs["event_id"] = 1
    short = tmp_path / "short.h5"
    with h5py.File(short, "w") as handle:
        handle.create_dataset("baseband", data=np.zeros((1000, 8), dtype=np.uint8))
        handle.attrs["freq"] = CENTER_MHZ
    output = tmp_path / "receipt.json"
    status = receipt.main([str(path), str(short), "--thresholds", str(_thresholds(tmp_path / "t.json.gz")),
                           "--output", str(output)])
    record = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0 and record["result"] == "PASS"
    assert record["totals"]["frames"] == 2
    assert [Path(entry["path"]).name for entry in record["skipped"]] == ["short.h5"]
    assert record["channels"][str(CHANNEL)]["frames"] == 2
