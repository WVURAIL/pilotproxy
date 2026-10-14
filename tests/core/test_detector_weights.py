# coding=utf-8
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.detector_core import (
    DetectorCoreProfile,
    load_detector_core_profile,
)
from pilot_proxy.integration.weight_generation import DetectorCoreLayout, target_layout
from pilot_proxy.paths import CONFIGS_DIR
from pilot_proxy.paths import DEFAULT_CHORD_WEIGHTS_PATH
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

FIRST_SHIPPED_PHYSICAL_CHANNEL = 14
LAST_SHIPPED_PHYSICAL_CHANNEL_EXCLUSIVE = 37
FIRST_SHIPPED_PILOT_MHZ = 470.309441
UNKNOWN_PILOT_MHZ_NEAR_CHANNEL_14 = 470.310000
PILOT_FREQUENCY_TOLERANCE_HZ = 10.0

EDGE_WRAP_PHYSICAL_CHANNEL = 14
EDGE_WRAP_PILOT_MHZ = 470.309441
EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS = -2
EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS = 2
REFERENCE_COARSE_CHANNEL_WIDTH_HZ = 390_625.0
CHORD_WEIGHTS_PATH = DEFAULT_CHORD_WEIGHTS_PATH
SHIPPED_CHANNEL_14_CENTER_MHZ = 470.3125
CHIME_CHANNEL_14_COARSE_INDEX = 843
CHORD_CHANNEL_14_COARSE_INDEX = 2408


def _core(*, k: int = 128, reference_offset_bins: int = 2) -> DetectorCoreLayout:
    return DetectorCoreLayout(
        detector_window_samples=int(k),
        skipped_guard_bins=int(reference_offset_bins) - 1,
        reference_offset_bins=int(reference_offset_bins),
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("detector_window_samples", 128.9),
        ("detector_window_samples", True),
        ("detector_window_samples", np.bool_(True)),
        ("skipped_guard_bins", 1.9),
        ("skipped_guard_bins", np.bool_(True)),
        ("reference_offset_bins", 2.9),
    ],
)
def test_detector_core_layout_requires_exact_geometry(
    field: str,
    invalid: object,
) -> None:
    arguments = {
        "detector_window_samples": 128,
        "skipped_guard_bins": 1,
        "reference_offset_bins": 2,
    }
    arguments[field] = invalid

    with pytest.raises(TypeError, match=field):
        DetectorCoreLayout(**arguments)


def _reference_profile():
    return load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
    )


def test_weight_bank_validates_known_physical_channel() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(
        FIRST_SHIPPED_PHYSICAL_CHANNEL
    )

    assert valid
    assert weights is not None
    assert bank.known_pilot_frequencies_mhz[0] == FIRST_SHIPPED_PILOT_MHZ


@pytest.mark.parametrize(
    ("weight_path", "coarse_index", "first_center_mhz", "last_center_mhz"),
    (
        (
            DEFAULT_WEIGHTS_PATH,
            CHIME_CHANNEL_14_COARSE_INDEX,
            799.609375,
            400.0,
        ),
        (
            CHORD_WEIGHTS_PATH,
            CHORD_CHANNEL_14_COARSE_INDEX,
            0.0,
            1599.8046875,
        ),
    ),
)
def test_expert_lookup_uses_receiver_native_frequency_grid(
    weight_path, coarse_index, first_center_mhz, last_center_mhz
) -> None:
    bank = DetectorWeightBank(explicit_path=weight_path)

    assert bank.reference_freqs[0] == pytest.approx(first_center_mhz)
    assert bank.reference_freqs[-1] == pytest.approx(last_center_mhz)
    assert bank.reference_freqs[coarse_index] == pytest.approx(
        SHIPPED_CHANNEL_14_CENTER_MHZ
    )
    weights, valid = bank.get_weights(FIRST_SHIPPED_PILOT_MHZ)

    assert valid is True
    assert weights is not None
    np.testing.assert_array_equal(weights, bank.rom_table[coarse_index])


@pytest.mark.parametrize(
    ("weight_path", "frequency_mhz"),
    (
        (DEFAULT_WEIGHTS_PATH, 399.999),
        (DEFAULT_WEIGHTS_PATH, 800.0),
        (CHORD_WEIGHTS_PATH, -0.001),
        (CHORD_WEIGHTS_PATH, 1600.0),
    ),
)
def test_expert_lookup_rejects_out_of_receiver_band(
    weight_path, frequency_mhz
) -> None:
    bank = DetectorWeightBank(explicit_path=weight_path)

    with pytest.raises(ValueError, match="outside the receiver profile"):
        bank.get_weights(frequency_mhz)


@pytest.mark.parametrize(
    "frequency_mhz", (float("nan"), float("inf"), -float("inf"))
)
def test_expert_lookup_rejects_nonfinite_frequency(frequency_mhz) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(ValueError, match="must be finite"):
        bank.get_weights(frequency_mhz)


def test_weight_bank_rejects_unknown_pilot_frequency() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(ValueError, match="not in the weight manifest"):
        bank.get_weights_for_pilot_frequency(
            UNKNOWN_PILOT_MHZ_NEAR_CHANNEL_14,
            tolerance_hz=PILOT_FREQUENCY_TOLERANCE_HZ,
        )


@pytest.mark.parametrize(
    "invalid_tolerance",
    [float("nan"), float("inf"), -float("inf"), -1.0, True, np.bool_(True)],
)
def test_weight_bank_rejects_invalid_pilot_tolerance(
    invalid_tolerance: object,
) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(ValueError, match="tolerance_hz.*finite and nonnegative"):
        bank.get_weights_for_pilot_frequency(
            FIRST_SHIPPED_PILOT_MHZ,
            tolerance_hz=invalid_tolerance,
        )


def test_weight_bank_accepts_zero_tolerance_for_exact_pilot() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    weights, valid = bank.get_weights_for_pilot_frequency(
        FIRST_SHIPPED_PILOT_MHZ,
        tolerance_hz=0.0,
    )

    assert valid is True
    assert weights is not None


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("K", {"K": 128.9}),
        ("N", {"N": 3.9}),
        ("reference_offset_bins", {"reference_offset_bins": 2.9}),
    ],
)
def test_weight_bank_constructor_requires_exact_geometry(
    field: str,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match=field):
        DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH, **kwargs)


def test_weight_bank_expected_kernel_requires_exact_geometry() -> None:
    with pytest.raises(TypeError, match="expected_kernel.K"):
        DetectorWeightBank(
            explicit_path=DEFAULT_WEIGHTS_PATH,
            expected_kernel=(128.9, 3, 4, 2),
        )


@pytest.mark.parametrize(
    "invalid_channel",
    [14.9, True, np.bool_(True), "14"],
)
def test_weight_bank_physical_channel_requires_exact_integer(
    invalid_channel: object,
) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(TypeError, match="physical channel"):
        bank.get_weights_for_physical_channel(invalid_channel)


@pytest.mark.parametrize(
    "invalid_index",
    [843.9, True, np.bool_(True), "843"],
)
def test_weight_bank_coarse_index_requires_exact_integer(
    invalid_index: object,
) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(TypeError, match="coarse channel index"):
        bank._weights_for_channel_index(invalid_index)


def test_noncontract_detector_spacing_field_is_rejected() -> None:
    old_key = "_".join(("reference", "guard", "bins"))
    data = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
    ).to_dict()
    data[old_key] = 2

    with pytest.raises(ValueError, match="unknown fields"):
        DetectorCoreProfile.from_dict(data)


def test_weight_bank_reports_channel_14_wrapped_boundary_layout() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    layout = bank.layout_for_physical_channel(EDGE_WRAP_PHYSICAL_CHANNEL)

    assert layout["target_frequency_mhz"] == EDGE_WRAP_PILOT_MHZ
    assert layout["adaptive_reference_placement"] is True
    assert layout["lower_reference_offset_bins"] == EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS
    assert layout["upper_reference_offset_bins"] == EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS
    assert layout["lower_reference_edge_wrapped"] is True
    assert layout["upper_reference_edge_wrapped"] is False
    assert layout["edge_reference_wrapped"] is True
    assert layout["reference_placement_status"] == "edge_wrapped"
    assert layout["strict_reference_offset_pass"] is True
    assert layout["detector_fine_bin_width_hz"] == pytest.approx(
        REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 128.0
    )
    assert layout["lower_reference_relative_to_target_hz"] == pytest.approx(
        EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS
        * REFERENCE_COARSE_CHANNEL_WIDTH_HZ
        / 128.0
    )
    assert layout["upper_reference_relative_to_target_hz"] == pytest.approx(
        EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS
        * REFERENCE_COARSE_CHANNEL_WIDTH_HZ
        / 128.0
    )
    assert "lower_reference_offset_hz" in layout
    assert "upper_reference_offset_hz" in layout


def test_k128_dtv14_dc_in_skipped_guard_not_reference() -> None:
    layout = target_layout(
        physical_channel=14,
        profile=_reference_profile(),
        core=_core(k=128, reference_offset_bins=2),
    )

    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_offset_bins"] == 2
    assert layout["dc_reference_collision"] is False
    assert layout["dc_reference_shifted"] is False
    assert layout["forbidden_tone_in_skipped_guard"] is True
    assert layout["reference_placement_status"] == "nominal"


@pytest.mark.parametrize("invalid_channel", [14.9, True, np.bool_(True), "14"])
def test_target_layout_requires_exact_physical_channel(
    invalid_channel: object,
) -> None:
    with pytest.raises(TypeError, match="physical channel.*integer"):
        target_layout(
            physical_channel=invalid_channel,
            profile=_reference_profile(),
            core=_core(k=128, reference_offset_bins=2),
        )


def test_target_on_forbidden_tone_hard_fails(monkeypatch) -> None:
    profile = _reference_profile()
    center_hz = profile.coarse_channel_center_hz(843)
    monkeypatch.setattr(
        "pilot_proxy.integration.weight_generation.physical_channel_to_pilot_hz",
        lambda channel: center_hz,
    )

    with pytest.raises(ValueError, match="target pilot bin collides"):
        target_layout(
            physical_channel=14,
            profile=profile,
            core=_core(k=128, reference_offset_bins=2),
        )


def test_k256_offset2_dtv14_reference_shifts_away_from_dc() -> None:
    layout = target_layout(
        physical_channel=14,
        profile=_reference_profile(),
        core=_core(k=256, reference_offset_bins=2),
    )

    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_requested_offset_bins"] == 2
    # The configured candidate remains reference_offset_bins=2. The selected
    # upper reference moves to +3 only as the adaptive DC-avoidance correction.
    assert layout["upper_reference_offset_bins"] == 3
    assert layout["upper_reference_requested_dc_collision"] is True
    assert layout["upper_reference_dc_shifted"] is True
    assert layout["dc_reference_collision"] is True
    assert layout["dc_reference_shifted"] is True
    assert layout["reference_placement_status"] == "dc_shifted"
    assert layout["strict_reference_offset_pass"] is True
    assert layout["upper_reference_requested_relative_to_target_hz"] == pytest.approx(
        2 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 256.0
    )
    assert layout["upper_reference_relative_to_target_hz"] == pytest.approx(
        3 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 256.0
    )


def test_dtv21_edge_reference_wraps_without_moving_closer() -> None:
    layout = target_layout(
        physical_channel=21,
        profile=_reference_profile(),
        core=_core(k=128, reference_offset_bins=2),
    )

    assert layout["lower_reference_requested_edge_wrapped"] is True
    assert layout["lower_reference_edge_wrapped"] is True
    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_offset_bins"] == 2
    assert (layout["lower_reference_offset_bins"], layout["upper_reference_offset_bins"]) != (
        -1,
        1,
    )
    assert layout["strict_reference_offset_pass"] is True


def test_weight_bank_lists_shipped_physical_channels() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    assert bank.supported_physical_channels() == list(
        range(
            FIRST_SHIPPED_PHYSICAL_CHANNEL,
            LAST_SHIPPED_PHYSICAL_CHANNEL_EXCLUSIVE,
        )
    )


def test_physical_channel_lookup_requires_manifest(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    bank = DetectorWeightBank(explicit_path=copied)
    with pytest.raises(ValueError, match="require the adjacent weight manifest"):
        bank.get_weights_for_physical_channel(FIRST_SHIPPED_PHYSICAL_CHANNEL)


def test_weight_bank_rejects_manifest_binary_binding_mismatch(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    # The current manifest binds the binary via artifacts.weights_sha256;
    # corrupt that field so the loader's manifest/binary check must reject it.
    manifest["artifacts"]["weights_sha256"] = "0" * 64
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest/binary"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_receiver_profile_hash_mismatch(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["receiver_profile"]["channelizer"]["frequency_axis"][
        "order"
    ] = "ascending_rf"
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receiver.profile.hash|profile hash"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_fractional_manifest_coarse_index(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["target_reference_layout"][0]["coarse_channel_index"] = 843.9
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="coarse_channel_index.*integer"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_wrong_integer_manifest_coarse_index(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["target_reference_layout"][0]["coarse_channel_index"] = 828
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coarse index disagrees"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_requires_sha256_manifest_binding(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["artifacts"].pop("weights_sha256")
    manifest["artifacts"]["weights_git_blob_sha1"] = "0" * 40
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="weights_sha256"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_current_manifest_without_coordinate_system(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("weight_coordinate_system", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires weight_coordinate_system"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_v2_manifest_without_input_coordinate(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("input_coordinate_system", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires input_coordinate_system"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_v2_manifest_without_input_preprocessing(
    tmp_path,
) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("input_preprocessing", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires input_preprocessing"):
        DetectorWeightBank(explicit_path=copied)


def test_frame_origin_and_channel_edge_diagnostics_on_chime_profile() -> None:
    """Under the center-at-DC CHIME profile the channel-14 ``edge_wrapped`` flag
    is a wrap across DC (the forbidden tone sits in the skipped guard), while
    channel 21 is the channel whose reference crosses the physical +-fs/2
    coarse-channel half-width. Neither event changes the placement."""
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    core = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
    )
    half_width_hz = REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 2.0

    ch14 = target_layout(physical_channel=14, profile=profile, core=core)
    assert ch14["lower_reference_edge_wrapped"] is True
    assert ch14["forbidden_tone_in_skipped_guard"] is True
    assert ch14["reference_crosses_channel_edge"] is False
    assert "coarse-channel center / DC" in ch14["frame_origin_description"]
    assert "coarse-channel center / DC" in ch14["placement_warnings"]
    assert "not the channel edge" in ch14["placement_warnings"]
    assert ch14["coarse_channel_half_width_hz"] == pytest.approx(half_width_hz)
    assert abs(ch14["lower_reference_unwrapped_offset_hz"]) < 4_000.0
    assert ch14["target_channel_edge_margin_hz"] > 190_000.0

    ch21 = target_layout(physical_channel=21, profile=profile, core=core)
    assert ch21["reference_placement_status"] == "nominal"
    assert ch21["upper_reference_crosses_channel_edge"] is True
    assert ch21["lower_reference_crosses_channel_edge"] is False
    assert abs(ch21["upper_reference_unwrapped_offset_hz"]) > half_width_hz
    assert ch21["target_channel_edge_margin_hz"] == pytest.approx(
        half_width_hz - abs(ch21["target_offset_hz"])
    )
    assert "beyond the coarse-channel half-width" in ch21["channel_edge_notes"]
    # The aliased (stored) offset is the unwrapped request reduced into +-fs/2.
    assert ch21["upper_reference_offset_hz"] == pytest.approx(
        ch21["upper_reference_unwrapped_offset_hz"] - REFERENCE_COARSE_CHANNEL_WIDTH_HZ
    )

    for channel in (32, 35):
        layout = target_layout(physical_channel=channel, profile=profile, core=core)
        assert layout["reference_crosses_channel_edge"] is False
        assert layout["channel_edge_notes"] == ""
