"""Public archive-health routing preserves evidence and fails before publication."""

import hashlib
import json
import zipfile

import numpy as np
import pytest

from pilot_proxy import cli
from pilot_proxy.archive_health import ArchiveHealthError
from test_archive_health import NFFT, _product, _save_product


@pytest.mark.parametrize("source_mode", ["directory", "explicit"])
def test_public_health_cli_binds_archive_and_publishes_exact_counts(tmp_path, monkeypatch, capsys, source_mode):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    product = _product(valid=[1, 0, 1], reject=[0, 0, 0],
                       baseband_power=[4, 0, 128], p_target=[10, 0, 10],
                       p_reference=[20, 0, 20])
    dc = 2 * 34_359_736_320.0
    for name in ("integrated_spectrum_before_mask", "integrated_spectrum_after_mask"):
        product[name][0] = dc
        product[name][9] = NFFT**2 * 2 * 4.0
    source = _save_product(source_dir / "kept-798.npz", product)
    (source_dir / "ignored.npz").write_bytes(b"outside selected glob")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    attachment = tmp_path / "products.zip"
    with zipfile.ZipFile(attachment, "w") as archive:
        archive.write(source, "nested/kept-798.npz")
    output = tmp_path / "audit"
    selection = (["--products-dir", str(source_dir), "--glob", "kept-*.npz"]
                 if source_mode == "directory" else ["--product", str(source)])
    with pytest.raises(SystemExit) as status:
        cli.main(["audit-archive-health", *selection, "--output-dir", str(output),
                  "--no-plots", "--fine-chunk-rows", "1",
                  "--product-archive", str(attachment), "--expect-products", "1",
                  "--expect-excluded-frames", "2", "--expect-invalid-frames", "1",
                  "--expect-ceiling-frames", "1"])
    assert status.value.code == 0
    report = json.loads((output / "archive_health_summary.json").read_text())
    totals = report["totals"]
    assert totals["products"] == 1 and totals["stored_frames"] == 3
    assert totals["included_frames"] == 1 and totals["excluded_unique_frames"] == 2
    ledger = [json.loads(line) for line in (output / "archive_exclusion_ledger.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in ledger] == [1, 2]
    assert all(row["reason_codes"] for row in ledger)
    with np.load(output / "health_corrected_integrated_spectra.npz") as corrected:
        assert int(corrected["ceiling_frames_subtracted_before"][0]) == 1
        assert float(corrected["integrated_spectrum_before_health_gate"][0, 0]) == 0
    assert report["created_utc"] == "2023-11-14T22:13:20+00:00"
    assert not list(output.rglob("*.png")) and not list(output.rglob("*.pdf"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert "Audited 1 products / 3 frames: included 1, excluded 2." in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["expectation", "archive"])
def test_public_health_cli_refuses_wrong_counts_or_attachment_before_output(tmp_path, failure):
    source = _save_product(tmp_path / "798.npz", _product(valid=[0], reject=[0],
                           baseband_power=[0], p_target=[0], p_reference=[0]))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "refused"
    if failure == "expectation":
        options = ["--expect-products", "2"]
        message = "expected products=2, observed 1"
    else:
        attachment = tmp_path / "wrong.zip"
        with zipfile.ZipFile(attachment, "w") as archive:
            archive.writestr("798.npz", b"wrong payload")
        options = ["--product-archive", str(attachment)]
        message = "mismatched"
    with pytest.raises(ArchiveHealthError, match=message):
        cli.main(["audit-archive-health", "--product", str(source),
                  "--output-dir", str(output), "--no-plots", *options])
    assert not output.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
