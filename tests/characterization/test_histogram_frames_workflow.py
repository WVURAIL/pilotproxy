"""Completed frame exports preserve the declared populations and source identities."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from characterization_fixtures import _replace, _write_product
from pilot_proxy.characterization import histogram_frames as export
from pilot_proxy.detectors.narrowband_marker import fine_histogram_frames


COUNTS = {
    "total": 6, "valid": 5, "health_include": 4, "selected": 4,
    "finite_time_selected": 3, "histogram_eligible": 3, "nonpositive_histogram": 1,
    "current_selected": 3, "current_histogram_eligible": 2, "current_untimed_selected": 1,
    "unassigned_selected": 0, "unassigned_histogram_eligible": 0,
    "untimed_selected": 1, "nonfinite_Q_selected": 0,
}
TIMES = [datetime(2020, month, 15, tzinfo=timezone.utc).timestamp() for month in (1, 3, 5)]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _product(path, channel):
    _write_product(path, channel, frames=6, units=3)
    fine = np.zeros((6, 3, 256), dtype=np.uint64)
    fine[:, 0, 7] = np.arange(6, dtype=np.uint64) * 6338
    fine[:, 1, 3] = 6325
    fine[:, 2, 9] = 6325
    fine[3, 1:, :] = 0
    _replace(
        path, fine_power_u64=fine,
        baseband_power_linear=np.asarray([[4], [128], [4], [4], [4], [4]], dtype=float),
        unit_time0_ctime=np.asarray(TIMES),
        unit_delta_time=np.asarray([1 / 390625, np.nan, 1 / 390625]),
        unit_event_id=np.asarray([2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.uint64),
        psd_frame_db_i16=np.full((6, 4), 7, dtype=np.int16),
    )
    return path


def _release(tmp_path):
    products, release = tmp_path / "products", tmp_path / "release"
    products.mkdir()
    preflight = []
    for channel in range(14, 37):
        fid = export.default_project().target_freq_id(str(channel))
        product = _product(products / f"{fid}.npz", channel)
        ledger = {
            "channel": channel, "freq_id": fid, "product": product.name,
            "product_sha256": _digest(product),
            "sections": {
                "product": {
                    "n_frames": 6, "n_valid": 5, "n_selected": 4, "n_without_time": 1,
                    "health_schema": export.HEALTH_GATE_SCHEMA,
                    "health_reasons": "detector_invalid:1;baseband_power_at_negative_full_scale_ceiling:1",
                },
                "era": {
                    "current_era": 2, "config_digest": "controlled-era-record",
                    "current_frames": 3, "current_first_month": "2020-03",
                    "current_last_month": "2020-05",
                },
            },
        }
        _write_json(release / "ledger/channels" / f"ch{channel}_fid{fid}.json", ledger)
        _write_json(release / f"channels/ch{channel}/eras.json", {
            "channel": channel, "freq_id": fid, "current_era": 2,
            "config_digest": "controlled-era-record", "config": {"fixture": True},
            "frames_selected": 4, "frames_without_time": 1,
            "eras": [
                {"era": 1, "first_month": "2020-01", "last_month": "2020-01",
                 "frames": 1, "frames_without_time": 0, "units": 1},
                {"era": 2, "first_month": "2020-03", "last_month": "2020-05",
                 "frames": 3, "frames_without_time": 1, "units": 2},
            ],
        })
        preflight.append({
            "filename": product.name, "sha256": _digest(product),
            "bytes": product.stat().st_size, "frames": 6, "valid": 5,
            "health_selected": 4, "health_schema": export.HEALTH_GATE_SCHEMA,
        })
    _write_json(release / "products-preflight.json", {"products": preflight})
    _write_json(release / "ledger/run.json", {"fixture": "independent population counts"})
    return products, release


def _assert_population(path, *, control=False):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_equal(data["Q"], [0, 1, 2, np.nan, 4, 5])
        np.testing.assert_array_equal(data["selected"], [1, 0, 1, 0, 1, 1])
        np.testing.assert_array_equal(data["histogram_eligible"], [1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(data["positive_Q"], [0, 1, 1, 0, 1, 1])
        np.testing.assert_array_equal(data["frame_unit_index"], [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(data["frame_in_unit"], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(data["frame_index"], np.arange(6))
        np.testing.assert_equal(data["frame_time"], [
            TIMES[0], TIMES[0] + 16384 / 390625, np.nan, np.nan,
            TIMES[2], TIMES[2] + 16384 / 390625,
        ])
        np.testing.assert_array_equal(data["frame_month"], [24240, 24240, -1, -1, 24244, 24244])
        np.testing.assert_array_equal(data["membership_month"], [24240, 24240, 24242, 24242, 24244, 24244])
        np.testing.assert_array_equal(data["era_id"], [1] * 6 if control else [1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(data["current_histogram_eligible"],
                                      [1, 0, 0, 0, 1, 1] if control else [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(data["acquisition_id"],
                                      np.repeat(np.asarray([2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.uint64), 2))
        assert data["acquisition_id"].dtype == np.uint64
        assert not any(name.startswith(("fine_power", "psd_")) for name in data.files)


def _assert_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["passed"] is True
    assert manifest["schema"] == "coarse-histogram-frames-manifest-v1"
    assert manifest["fine_power_payload_decoded"] is True
    assert manifest["psd_payloads_decoded"] is False
    for name, digest in manifest["files"].items():
        assert _digest(output / name) == digest
    assert manifest["extraction_plan_sha256"] == _digest(output / "extraction-plan.json")
    plan = json.loads((output / "extraction-plan.json").read_text())
    for name, digest in plan["inputs"].items():
        assert _digest(Path(name)) == digest
    return manifest


def test_screened_cli_exports_the_complete_authenticated_population(tmp_path):
    products, release = _release(tmp_path)
    before = {path: _digest(path) for root in (products, release) for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "frames"
    assert export.main(["--products", str(products), "--release", str(release), "--output", str(output)]) == 0
    manifest = _assert_manifest(output)
    assert [item["channel"] for item in manifest["channels"]] == list(range(14, 37))
    assert manifest["totals"] == {key: value * 23 for key, value in COUNTS.items()}
    for item in manifest["channels"]:
        assert item["counts"] == COUNTS
        _assert_population(output / item["npz"])
        metadata = json.loads((output / item["metadata"]).read_text())
        assert metadata["export_sha256"] == _digest(output / item["npz"])
        assert metadata["product"]["sha256"] == _digest(products / f"{item['freq_id']}.npz")
        assert "fine_power_u64" in metadata["decoded_npz_members"]
        assert not any(name.startswith("psd_") for name in metadata["decoded_npz_members"])
    assert before == {path: _digest(path) for path in before}
    with pytest.raises(FileExistsError, match="new empty"):
        export.run(products, release, output)


def test_control_cli_manifest_remains_readable_by_the_fine_exporter(tmp_path):
    product = _product(tmp_path / "491.npz", 37)
    before = _digest(product)
    output = tmp_path / "control"
    assert export.main(["--control-product", str(product), "--output", str(output)]) == 0
    manifest = _assert_manifest(output)
    assert manifest["role"] == "control"
    assert manifest["totals"] == {**COUNTS, "current_selected": 4, "current_histogram_eligible": 3}
    assert [(item["channel"], item["freq_id"]) for item in manifest["channels"]] == [(37, 491)]
    _assert_population(output / "ch37.npz", control=True)
    assert _digest(product) == before
    fine_manifest = fine_histogram_frames.run(output, tmp_path / "fine")
    assert fine_manifest["passed"] is True
    with np.load(output / "ch37.npz", allow_pickle=False) as coarse, np.load(tmp_path / "fine/ch37.npz", allow_pickle=False) as fine:
        for name in coarse.files:
            np.testing.assert_equal(fine[name], coarse[name])


@pytest.mark.parametrize("record,expected", [
    ("product", "product differs"),
    ("ledger", "ledger mismatch:selected"),
    ("eras", "configuration differs"),
    ("preflight", "byte count differs"),
])
def test_export_refuses_inconsistent_source_records(tmp_path, record, expected):
    products, release = _release(tmp_path)
    ledger_path = next((release / "ledger/channels").glob("ch14_*.json"))
    ledger = json.loads(ledger_path.read_text())
    if record == "product":
        _replace(products / ledger["product"], baseband_power_linear=np.full((6, 1), 5.0))
    elif record == "ledger":
        ledger["sections"]["product"]["n_selected"] = 5
        _write_json(ledger_path, ledger)
    else:
        path = release / ("channels/ch14/eras.json" if record == "eras" else "products-preflight.json")
        data = json.loads(path.read_text())
        if record == "eras":
            data["config_digest"] = "changed"
        else:
            data["products"][0]["bytes"] += 1
        _write_json(path, data)
    output = tmp_path / "frames"
    with pytest.raises(ValueError, match=expected):
        export.run(products, release, output)
    assert not (output / "manifest.json").exists()


def test_release_mutation_during_export_cannot_publish_success(tmp_path, monkeypatch):
    products, release = _release(tmp_path)
    process = export.process_channel
    changed = False

    def alter_release_after_first_channel(*args, **kwargs):
        nonlocal changed
        result = process(*args, **kwargs)
        if not changed:
            _write_json(release / "ledger/run.json", {"changed_during_export": True})
            changed = True
        return result

    monkeypatch.setattr(export, "process_channel", alter_release_after_first_channel)
    output = tmp_path / "frames"
    with pytest.raises(ValueError, match="source/release record changed"):
        export.run(products, release, output)
    assert changed and not (output / "manifest.json").exists()
