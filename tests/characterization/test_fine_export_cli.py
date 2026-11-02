"""Fine export CLI preserves coarse rows and binds the selected geometry."""

import hashlib
import json

import numpy as np
import pytest

from pilot_proxy import cli
from pilot_proxy.characterization import histogram_frames as coarse
from pilot_proxy.detectors.narrowband_marker import fine_histogram_frames as fine
from test_pilotproxy_v5 import _replace, _write_product


def _coarse_frames(tmp_path):
    product = _write_product(tmp_path / "491.npz", 37, frames=6, units=2)
    _replace(product, baseband_power_linear=np.full((6, 1), 4.0))
    frames = tmp_path / "frames"
    coarse.run_control(product, frames)
    return frames


def test_fine_export_cli_records_geometry_and_preserves_coarse_arrays(tmp_path):
    frames = _coarse_frames(tmp_path)
    original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frames.iterdir() if p.is_file()}
    output = tmp_path / "fine"
    assert cli.main(["characterize", "fine-histogram-frames", "--frames", str(frames),
                     "--output", str(output), "--designated-half-width", "1",
                     "--marker-guard-bins", "2"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    metadata = json.loads((output / "ch37.json").read_text())
    assert manifest["passed"] is True
    assert [(x["channel"], x["freq_id"]) for x in manifest["channels"]] == [(37, 491)]
    geometry = metadata["geometry"]
    assert geometry["designated_half_width"] == 1
    assert geometry["marker_guard_bins"] == 2
    assert geometry["rho_one_based"] == (len(geometry["bulk_bins"]) + 1) // 2
    anchor = geometry["anchor_bin"]
    assert geometry["designated_bins"] == [anchor - 1, anchor, anchor + 1]
    assert not set(geometry["bulk_bins"]) & set(range(anchor - 3, anchor + 4))
    with np.load(frames / "ch37.npz", allow_pickle=False) as previous, np.load(output / "ch37.npz", allow_pickle=False) as current:
        for name in previous.files:
            np.testing.assert_equal(current[name], previous[name])
        assert current["fine_score_computed"].tolist() == [True] * 6
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert original == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    with pytest.raises(FileExistsError, match="empty fine frames"):
        fine.main(["--frames", str(frames), "--output", str(output)])


def test_fine_export_cli_rejects_changed_coarse_input_before_publication(tmp_path):
    frames = _coarse_frames(tmp_path)
    with (frames / "ch37.npz").open("ab") as stream:
        stream.write(b"changed input")
    output = tmp_path / "fine"
    with pytest.raises(ValueError, match="differ"):
        fine.main(["--frames", str(frames), "--output", str(output)])
    assert not (output / "manifest.json").exists()
