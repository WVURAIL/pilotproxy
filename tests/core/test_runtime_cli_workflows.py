"""The runtime bundle export command."""

import hashlib
import json
from pathlib import Path

import pytest

from pilot_proxy import runtime_bundle


ROOT = Path(__file__).resolve().parents[2]
RECEIVER = ROOT / "configs/receiver_profiles/chime_dtv_fengine.json"
CORE = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"


def _read(path):
    def refuse_constant(value):
        raise AssertionError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_text(), parse_constant=refuse_constant)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_args(output, coordinate="post_nyquist_zone_normalization"):
    return ["--receiver-profile", str(RECEIVER), "--detector-core-profile", str(CORE),
            "--weight-coordinate-system", coordinate, "--output-dir", str(output)]


@pytest.mark.parametrize("coordinate,reverse", [
    ("post_nyquist_zone_normalization", True),
    ("raw_input_frequency_coordinate", False),
])
def test_bundle_cli_preserves_channel_selection_coordinates_and_binary_norms(
    tmp_path, capsys, coordinate, reverse,
):
    output = tmp_path / "runtime bundle"
    assert runtime_bundle.main(_bundle_args(output, coordinate) + [
        "--physical-channel", "29", "--physical-channel", "14",
        "--physical-channel-range", "14:15",
    ]) == 0
    assert set(p.name for p in output.iterdir()) == set(runtime_bundle.REQUIRED_RUNTIME_BUNDLE_FILES)
    printed = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    assert printed == {
        "detector_contract": str(output / "detector_contract.json"),
        "pilot_profiles": str(output / "pilot_profiles.json"),
        "weights": str(output / "weights.bin"),
        "weights_manifest": str(output / "weights.manifest.json"),
        "sha256sums": str(output / "sha256sums.txt"),
    }
    manifest = _read(output / "weights.manifest.json")
    profiles = _read(output / "pilot_profiles.json")
    contract = _read(output / "detector_contract.json")
    assert manifest["physical_channels"] == [14, 15, 29]
    assert manifest["weight_profile_shape"] == [3, 128]
    assert manifest["weight_profile_nbytes"] == 384
    assert contract["weight_coordinate_system"] == coordinate
    assert contract["input_preprocessing"]["time_reverse_detector_windows_before_kernel"] is reverse
    payload = (output / "weights.bin").read_bytes()
    assert len(payload) == 1152
    assert [p["physical_channel"] for p in profiles["profiles"]] == [14, 15, 29]
    for index, row in enumerate(profiles["profiles"]):
        assert row["weight_bank_offset_bytes"] == 384 * index
        assert row["weight_bank_nbytes"] == 384
        chunk = payload[384 * index:384 * (index + 1)]
        norms = []
        for term in range(3):
            norm = 0
            for byte in chunk[128 * term:128 * (term + 1)]:
                real, imag = byte & 15, byte >> 4
                real -= 16 if real >= 8 else 0
                imag -= 16 if imag >= 8 else 0
                norm += real * real + imag * imag
            norms.append(norm)
        assert row["target_norm_sq"] == norms[0]
        assert row["reference_norm_sum_sq"] == norms[1] + norms[2]
        # F uses the mean reference power, so its null ratio has a factor of two.
        assert row["null_power_ratio"] == 2 * norms[0] / (norms[1] + norms[2])
        assert row["fine_calibration"]["status"] == "pending_campaign"
    sums = dict(line.split("  ")[::-1] for line in (output / "sha256sums.txt").read_text().splitlines())
    assert set(sums) == set(runtime_bundle.REQUIRED_RUNTIME_BUNDLE_FILES) - {"sha256sums.txt"}
    assert all(_digest(output / name) == digest for name, digest in sums.items())
    assert runtime_bundle.validate_runtime_weight_bundle(bundle_dir=output)["valid"]


@pytest.mark.parametrize("selection,message", [
    ([], "at least one physical channel"),
    (["--physical-channel", "13"], "physical channel"),
    (["--physical-channel-range", "14:70"], "physical channel"),
])
def test_bundle_cli_refuses_invalid_channel_selection_before_output(tmp_path, selection, message):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match=message):
        runtime_bundle.main(_bundle_args(output) + selection)
    assert not output.exists()


def test_export_refuses_duplicate_profiles_before_output(tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="multiple physical channels map to the same coarse channel"):
        runtime_bundle.export_runtime_weight_bundle(
            receiver_profile_path=RECEIVER, detector_core_profile_path=CORE,
            physical_channels=[29, 29],
            weight_coordinate_system="post_nyquist_zone_normalization", output_dir=output,
        )
    assert not output.exists()
