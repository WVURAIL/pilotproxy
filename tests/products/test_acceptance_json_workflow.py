"""The archive CLI publishes complete accounting only after cohort acceptance."""

import hashlib
import json

import pytest

from pilot_proxy import cli
from test_pilotproxy_v5 import _write_product


def _reject_constant(value):
    raise AssertionError(f"nonstandard JSON constant: {value}")


def test_archive_cli_json_preserves_product_accounting_and_identities(tmp_path, capsys):
    products = tmp_path / "products"
    products.mkdir()
    for channel in range(14, 37):
        _write_product(products / f"ch{channel}.npz", channel, frames=6, units=2)
    output = tmp_path / "accepted.json"
    assert cli.main(["products", "check-archive", str(products), "--json", str(output)]) == 0
    report = json.loads(output.read_text(), parse_constant=_reject_constant)
    assert set(report) == {
        "products", "detector_version", "kernel_sha256", "detector_contract_sha256",
        "decision_contract_sha256", "weight_bank_sha256", "weight_manifest_sha256",
        "weight_coefficients_sha256",
    }
    for name, letter in [("kernel_sha256", "a"), ("weight_bank_sha256", "b"),
                         ("weight_manifest_sha256", "c"),
                         ("weight_coefficients_sha256", "d")]:
        assert report[name] == letter * 64
    assert f"kernel_sha256={'a' * 64}" in report["detector_version"]
    assert len(report["products"]) == 23
    for channel, record in zip(range(14, 37), report["products"], strict=True):
        freq_id = round((800e6 - (470.309441e6 + (channel - 14) * 6e6)) / 390625)
        assert record == {
            "path": str(products / f"ch{channel}.npz"),
            "physical_channel": channel, "freq_id": freq_id,
            "frame_count": 6, "valid_frames": 6, "invalid_frames": 0,
            "rejected_frames": 6, "unit_count": 2,
            "frames_by_unit": [3, 3], "valid_by_unit": [3, 3],
            "untimed_units": 0, "untimed_frames": 0,
            "weights_hash": hashlib.sha256(f"weights-{channel}".encode()).hexdigest(),
        }
    assert "frames: 138" in capsys.readouterr().out


def test_rejected_cohort_does_not_publish_acceptance_json(tmp_path, capsys):
    output = tmp_path / "accepted.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["products", "check-archive", str(tmp_path), "--json", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "must contain exactly 23 products" in captured.err
    assert "archive cohort accepted" not in captured.out
    assert not output.exists()
