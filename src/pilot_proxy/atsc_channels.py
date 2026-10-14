# coding=utf-8
"""ATSC 1.0 physical-channel frequency helpers."""

from __future__ import annotations

import operator

import numpy as np

from pilot_proxy.config.project import default_project

_PROJECT = default_project()

# ATSC 1.0 UHF physical channels are 6 MHz wide, and the 8-VSB pilot sits
# 309.441 kHz above the lower channel edge. Both values are read from the
# project's interference template (projects/chime_atsc/interference_template.yaml).
ATSC_CHANNEL_WIDTH_HZ = _PROJECT.interference.band_width_hz
ATSC_PILOT_OFFSET_HZ = _PROJECT.interference.marker.offset_hz
ATSC_UHF_MIN_PHYSICAL_CHANNEL = 14
ATSC_UHF_MAX_PHYSICAL_CHANNEL = 69
# UHF channel 14 starts at 470 MHz (the plan's lower edge of band "14"); higher
# UHF channels advance in 6 MHz steps.
ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ = _PROJECT.frequency_plan.band(
    str(ATSC_UHF_MIN_PHYSICAL_CHANNEL)
).low_hz
ATSC_CHANNEL_CENTER_OFFSET_HZ = ATSC_CHANNEL_WIDTH_HZ / 2.0


def validate_uhf_physical_channel(channel: int) -> int:
    """Validate and return an ATSC UHF physical channel number."""
    if isinstance(channel, (bool, np.bool_)):
        raise TypeError("physical channel must be an integer, not a boolean.")
    try:
        value = operator.index(channel)
    except TypeError as exc:
        raise TypeError("physical channel must be an integer.") from exc
    if value < ATSC_UHF_MIN_PHYSICAL_CHANNEL or value > ATSC_UHF_MAX_PHYSICAL_CHANNEL:
        raise ValueError(
            "physical channel must be in the UHF range "
            f"{ATSC_UHF_MIN_PHYSICAL_CHANNEL}-{ATSC_UHF_MAX_PHYSICAL_CHANNEL}; "
            f"got {channel!r}."
        )
    return value


def physical_channel_to_lower_edge_hz(channel: int) -> float:
    """Return the lower RF edge for an ATSC UHF physical channel."""
    value = validate_uhf_physical_channel(channel)
    return float(
        ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ
        + (value - ATSC_UHF_MIN_PHYSICAL_CHANNEL) * ATSC_CHANNEL_WIDTH_HZ
    )


def physical_channel_to_center_hz(channel: int) -> float:
    """Return the RF center frequency for an ATSC UHF physical channel."""
    return float(
        physical_channel_to_lower_edge_hz(channel) + ATSC_CHANNEL_CENTER_OFFSET_HZ
    )


def physical_channel_to_pilot_hz(channel: int) -> float:
    """Return the ATSC pilot frequency for an ATSC UHF physical channel."""
    return float(physical_channel_to_lower_edge_hz(channel) + ATSC_PILOT_OFFSET_HZ)


def parse_physical_channel_range(value: str) -> list[int]:
    """Parse ``N`` or ``N:M`` into inclusive physical-channel numbers."""
    text = str(value).strip()
    if ":" not in text:
        return [validate_uhf_physical_channel(int(text))]
    left, right = text.split(":", 1)
    start = validate_uhf_physical_channel(int(left))
    stop = validate_uhf_physical_channel(int(right))
    step = 1 if stop >= start else -1
    return list(range(start, stop + step, step))


def parse_physical_channel_spec(value: str) -> list[int]:
    """Parse comma-separated channel numbers and inclusive ranges.

    Examples are ``"14"``, ``"14:36"``, and ``"14,18:20,36"``. Every
    channel is validated against the supported UHF range.
    """
    channels: list[int] = []
    for part in str(value).split(","):
        item = part.strip()
        if item:
            channels.extend(parse_physical_channel_range(item))
    if not channels:
        raise ValueError("physical-channel selection must not be empty")
    return channels


__all__ = [
    "ATSC_CHANNEL_WIDTH_HZ",
    "ATSC_CHANNEL_CENTER_OFFSET_HZ",
    "ATSC_PILOT_OFFSET_HZ",
    "ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ",
    "ATSC_UHF_MAX_PHYSICAL_CHANNEL",
    "ATSC_UHF_MIN_PHYSICAL_CHANNEL",
    "parse_physical_channel_range",
    "parse_physical_channel_spec",
    "physical_channel_to_center_hz",
    "physical_channel_to_lower_edge_hz",
    "physical_channel_to_pilot_hz",
    "validate_uhf_physical_channel",
]
