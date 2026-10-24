# coding=utf-8
"""Nyquist-zone frequency axis: current form, and the older key it replaced.

Profiles, weight manifests and runtime bundles written before 1.1.0 state the
frequency axis as ``spectral_sense: normal|inverted`` and the detector weight
coordinate as ``post_spectral_sense_normalization``. They must still load and
validate, read as zone 1 (odd, upright) or zone 2 (even, mirrored), while
everything written now uses ``nyquist_zone`` and the new coordinate names.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pilot_proxy.detector_contract import (
    INPUT_COORDINATE_POST_NYQUIST_ZONE_NORMALIZED,
    WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
    detector_contract_sha256,
    normalize_input_coordinate_system,
    normalize_weight_coordinate_system,
)
from pilot_proxy.detector_geometry import (
    frequency_axis_nyquist_zone,
    normalize_nyquist_zone,
    nyquist_zone_is_mirrored,
)
from pilot_proxy.integration.receiver_profile import (
    ReceiverProfile,
    load_receiver_profile,
    receiver_profile_hash,
    receiver_profile_hash_matches,
    validate_weight_manifest_profile_hash,
)
from pilot_proxy.provenance import file_sha256
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "configs" / "receiver_profiles"
CORE_PROFILE = (
    REPO_ROOT / "configs" / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
)
EXPECTED_ZONES = {
    "chime_dtv_fengine": 2,
    "chord_dtv_fengine": 1,
    "chord_pathfinder_dtv_fengine": 1,
    "reference_800mhz_pfb": 1,
}
OLD_WEIGHT_COORDINATE = "post_spectral_sense_normalization"
OLD_INPUT_COORDINATE = "post_spectral_sense_normalized"


def _old_form(profile: ReceiverProfile) -> dict:
    """The profile as a pre-1.1.0 file stated its frequency axis."""
    data = copy.deepcopy(profile.to_dict())
    axis = data["channelizer"]["frequency_axis"]
    data["channelizer"]["frequency_axis"] = {
        "spectral_sense": "inverted" if nyquist_zone_is_mirrored(axis["nyquist_zone"]) else "normal",
        "order": axis["order"],
    }
    return data


@pytest.mark.parametrize("profile_id", sorted(EXPECTED_ZONES))
def test_shipped_profiles_state_their_nyquist_zone(profile_id: str) -> None:
    path = PROFILE_DIR / f"{profile_id}.json"
    raw = json.loads(path.read_text("utf-8"))
    axis = raw["channelizer"]["frequency_axis"]
    assert axis["nyquist_zone"] == EXPECTED_ZONES[profile_id]
    assert "spectral_sense" not in json.dumps(raw)
    profile = load_receiver_profile(path)
    assert profile.nyquist_zone == EXPECTED_ZONES[profile_id]


def test_chime_profile_documents_the_sinc_hamming_window() -> None:
    raw = json.loads((PROFILE_DIR / "chime_dtv_fengine.json").read_text("utf-8"))
    channelizer = raw["channelizer"]
    assert channelizer["response"] == "sinc_hamming"
    assert (channelizer["taps"], channelizer["pfb_fft_size"]) == (4, 2048)


@pytest.mark.parametrize("profile_id", sorted(EXPECTED_ZONES))
def test_old_frequency_axis_key_reads_as_the_same_profile(profile_id: str) -> None:
    current = load_receiver_profile(PROFILE_DIR / f"{profile_id}.json")
    old = ReceiverProfile.from_dict(_old_form(current))
    assert old == current
    assert "spectral_sense" not in json.dumps(old.to_dict())
    assert receiver_profile_hash(old) == receiver_profile_hash(current)


def test_old_manifest_digest_still_binds_the_profile() -> None:
    profile = load_receiver_profile(PROFILE_DIR / "chime_dtv_fengine.json")
    text = json.dumps(_old_form(profile), sort_keys=True, separators=(",", ":"))
    old_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert old_digest != receiver_profile_hash(profile)
    assert receiver_profile_hash_matches(profile, old_digest)
    assert validate_weight_manifest_profile_hash(
        {"receiver_profile_hash": old_digest}, profile
    )
    assert not receiver_profile_hash_matches(profile, "0" * 64)


def test_frequency_axis_keys_must_agree() -> None:
    assert frequency_axis_nyquist_zone({"nyquist_zone": 2}) == 2
    assert frequency_axis_nyquist_zone({"spectral_sense": "inverted"}) == 2
    assert frequency_axis_nyquist_zone({"spectral_sense": "normal"}) == 1
    assert frequency_axis_nyquist_zone({"nyquist_zone": 3, "spectral_sense": "normal"}) == 3
    assert frequency_axis_nyquist_zone({"order": "ascending_rf"}) is None
    with pytest.raises(ValueError, match="disagrees"):
        frequency_axis_nyquist_zone({"nyquist_zone": 2, "spectral_sense": "normal"})
    with pytest.raises(ValueError, match="spectral_sense"):
        frequency_axis_nyquist_zone({"spectral_sense": "sideways"})
    with pytest.raises(ValueError, match="integer"):
        frequency_axis_nyquist_zone({"nyquist_zone": "2"})


def test_profile_requires_a_zone() -> None:
    profile = load_receiver_profile(PROFILE_DIR / "chime_dtv_fengine.json")
    data = profile.to_dict()
    del data["channelizer"]["frequency_axis"]["nyquist_zone"]
    with pytest.raises(ValueError, match="requires nyquist_zone"):
        ReceiverProfile.from_dict(data)
    data["channelizer"]["frequency_axis"]["nyquist_zone"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        ReceiverProfile.from_dict(data)


def test_zone_parity_decides_the_mirror() -> None:
    assert [nyquist_zone_is_mirrored(zone) for zone in (1, 2, 3, 4)] == [
        False,
        True,
        False,
        True,
    ]
    assert normalize_nyquist_zone(None) == 1
    assert normalize_nyquist_zone("inverted") == 2
    with pytest.raises(ValueError):
        normalize_nyquist_zone(True)


def test_old_coordinate_names_read_as_current() -> None:
    assert normalize_weight_coordinate_system(OLD_WEIGHT_COORDINATE) == (
        WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    )
    assert normalize_input_coordinate_system(OLD_INPUT_COORDINATE) == (
        INPUT_COORDINATE_POST_NYQUIST_ZONE_NORMALIZED
    )


def _rewrite_as_old_bundle(bundle: Path) -> None:
    contract_path = bundle / "detector_contract.json"
    contract = json.loads(contract_path.read_text("utf-8"))
    contract["weight_coordinate_system"] = OLD_WEIGHT_COORDINATE
    contract["input_coordinate_system"] = OLD_INPUT_COORDINATE
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    digest = detector_contract_sha256(contract)
    for name in ("pilot_profiles.json", "weights.manifest.json"):
        path = bundle / name
        payload = json.loads(path.read_text("utf-8"))
        payload["weight_coordinate_system"] = OLD_WEIGHT_COORDINATE
        payload["input_coordinate_system"] = OLD_INPUT_COORDINATE
        payload["detector_contract_sha256"] = digest
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    names = (
        "detector_contract.json",
        "pilot_profiles.json",
        "weights.bin",
        "weights.manifest.json",
    )
    (bundle / "sha256sums.txt").write_text(
        "".join(f"{file_sha256(bundle / name)}  {name}\n" for name in names)
    )


def test_bundle_with_old_coordinate_names_still_validates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    export_runtime_weight_bundle(
        receiver_profile_path=PROFILE_DIR / "chime_dtv_fengine.json",
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=[14, 29],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
        output_dir=bundle,
    )
    _rewrite_as_old_bundle(bundle)
    result = validate_runtime_weight_bundle(bundle_dir=bundle)
    assert result["valid"], result["errors"]


def test_cli_accepts_the_old_coordinate_name_and_writes_the_new(tmp_path: Path) -> None:
    """The kotekan CI exports bundles with the old coordinate name."""
    bundle = tmp_path / "bundle"
    env_path = str(REPO_ROOT / "src")
    command = [
        sys.executable,
        "-B",
        "-m",
        "pilot_proxy.cli",
        "export-runtime-weight-bundle",
        "--receiver-profile",
        str(PROFILE_DIR / "chime_dtv_fengine.json"),
        "--detector-core-profile",
        str(CORE_PROFILE),
        "--weight-coordinate-system",
        OLD_WEIGHT_COORDINATE,
        "--physical-channel-range",
        "14:16",
        "--output-dir",
        str(bundle),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": env_path, "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
    )
    texts = {path.name: path.read_text("utf-8") for path in bundle.glob("*.json")}
    assert texts and not any("spectral_sense" in text for text in texts.values())
    profiles = json.loads(texts["pilot_profiles.json"])
    assert profiles["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_NYQUIST_ZONE
    assert profiles["input_preprocessing"] == {
        "time_reverse_detector_windows_before_kernel": True
    }
    assert validate_runtime_weight_bundle(bundle_dir=bundle)["valid"]
