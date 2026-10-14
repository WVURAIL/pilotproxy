# coding=utf-8
"""Shared default config paths for receiver integration workflows."""

from __future__ import annotations

from pathlib import Path

from pilot_proxy.paths import CONFIGS_DIR

DEFAULT_DETECTOR_CORE_PROFILE: Path = (
    CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
)
DEFAULT_REFERENCE_RECEIVER_PROFILE: Path = (
    CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
)
DEFAULT_CHIME_DTV_RECEIVER_PROFILE: Path = (
    CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
)
DEFAULT_CHIME_STREAM_MAP: Path = (
    CONFIGS_DIR / "stream_maps" / "chime_feed_pol_example.json"
)

__all__ = [
    "DEFAULT_CHIME_DTV_RECEIVER_PROFILE",
    "DEFAULT_CHIME_STREAM_MAP",
    "DEFAULT_DETECTOR_CORE_PROFILE",
    "DEFAULT_REFERENCE_RECEIVER_PROFILE",
]
