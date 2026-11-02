"""Receiver quantization defaults retain exact types and valid scales."""
from dataclasses import replace

import numpy as np
import pytest

from pilot_proxy.integration.receiver_profile import (
    ReceiverProfile,
    default_reference_receiver_profile,
    receiver_profile_hash,
)


@pytest.mark.parametrize("bits", [4.5, 4.0, "4", True, np.bool_(True), None, 0, -1, 6, 16])
@pytest.mark.parametrize("construction", ["direct", "serialized"])
def test_receiver_profile_requires_a_supported_exact_bit_depth(bits, construction):
    profile = default_reference_receiver_profile()
    with pytest.raises((TypeError, ValueError), match="bit"):
        if construction == "direct":
            replace(profile, bits_per_component=bits)
        else:
            data = profile.to_dict()
            data["quantization"]["bits_per_component"] = bits
            ReceiverProfile.from_dict(data)


@pytest.mark.parametrize("clip_sigma", [
    0, -1, np.nan, np.inf, -np.inf, True, np.bool_(True), "3.0", None, 1 + 0j,
])
@pytest.mark.parametrize("construction", ["direct", "serialized"])
def test_receiver_profile_requires_a_positive_finite_scale_default(clip_sigma, construction):
    profile = default_reference_receiver_profile()
    with pytest.raises((TypeError, ValueError), match="clip_sigma_default"):
        if construction == "direct":
            replace(profile, clip_sigma_default=clip_sigma)
        else:
            data = profile.to_dict()
            data["quantization"]["clip_sigma_default"] = clip_sigma
            ReceiverProfile.from_dict(data)


@pytest.mark.parametrize("bits,clip_sigma", [
    (np.int8(4), np.float32(2.5)), (np.uint8(4), np.int64(3)),
    (np.int32(8), np.float64(2.5)), (np.uint64(8), 3),
    (4, 3.0), (8, np.float32(3.0)),
])
def test_valid_numeric_defaults_preserve_serialization_and_profile_hash(bits, clip_sigma):
    original = default_reference_receiver_profile()
    profile = replace(original, bits_per_component=bits, clip_sigma_default=clip_sigma)
    canonical = replace(original, bits_per_component=int(bits), clip_sigma_default=float(clip_sigma))
    assert type(profile.bits_per_component) is int
    assert type(profile.clip_sigma_default) is float
    assert profile.to_dict() == canonical.to_dict()
    assert receiver_profile_hash(profile) == receiver_profile_hash(canonical)
    loaded = ReceiverProfile.from_dict(profile.to_dict())
    assert loaded == profile
    assert receiver_profile_hash(loaded) == receiver_profile_hash(profile)
