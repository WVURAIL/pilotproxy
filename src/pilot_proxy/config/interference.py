"""Interference template: what the emitter looks like, as data.

``kind`` selects the detector adapter and the residual estimator. The only kind
today is ``narrowband_marker``: an emitter that carries a narrowband marker at
a fixed offset from its band edge, with a known power ratio to the band.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ._files import ProfileError, check_keys, choice, number, read_yaml, text

KINDS = ("narrowband_marker",)
RESIDUAL_SHAPES = ("band_thermal",)


@dataclass(frozen=True)
class MarkerTemplate:
    offset_hz: float           # marker frequency above the lower band edge
    marker_to_band_db: float   # marker power below the band's data power
    capture_efficiency: float  # fraction of the marker power the detector captures


@dataclass(frozen=True)
class InterferenceTemplate:
    kind: str
    band_width_hz: float
    residual_shape: str
    reference: str
    marker: Optional[MarkerTemplate] = None


def interference_template_from_mapping(data: Mapping[str, Any], *,
                                       where: str) -> InterferenceTemplate:
    check_keys(data, required=("kind", "band_width_hz", "residual_shape", "reference"),
               optional=("marker",), where=where)
    kind = choice(data["kind"], KINDS, field="kind", where=where)
    width = number(data["band_width_hz"], field="band_width_hz", where=where, positive=True)
    shape = choice(data["residual_shape"], RESIDUAL_SHAPES, field="residual_shape", where=where)
    reference = text(data["reference"], field="reference", where=where)
    marker = None
    if kind == "narrowband_marker":
        block = data.get("marker")
        at = f"{where}: marker"
        if not isinstance(block, Mapping):
            raise ProfileError(f"{at} must be a mapping for kind narrowband_marker")
        check_keys(block, required=("offset_hz", "marker_to_band_db", "capture_efficiency"),
                   where=at)
        offset = number(block["offset_hz"], field="offset_hz", where=at)
        if not 0.0 <= offset < width:
            raise ProfileError(f"{at}: offset_hz must lie inside the band width")
        efficiency = number(block["capture_efficiency"], field="capture_efficiency",
                            where=at, positive=True)
        if efficiency > 1.0:
            raise ProfileError(f"{at}: capture_efficiency must be in (0, 1]")
        marker = MarkerTemplate(
            offset_hz=offset,
            marker_to_band_db=number(block["marker_to_band_db"], field="marker_to_band_db",
                                     where=at),
            capture_efficiency=efficiency,
        )
    return InterferenceTemplate(kind, width, shape, reference, marker)


def load_interference_template(path: Path | str) -> InterferenceTemplate:
    path = Path(path)
    return interference_template_from_mapping(read_yaml(path), where=str(path))


__all__ = [
    "InterferenceTemplate",
    "KINDS",
    "MarkerTemplate",
    "RESIDUAL_SHAPES",
    "interference_template_from_mapping",
    "load_interference_template",
]
