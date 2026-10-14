"""Integration model: ``frame_seconds``, the frame length a frame residual is counted in."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._files import check_keys, number, read_yaml


@dataclass(frozen=True)
class IntegrationModel:
    frame_seconds: float


def integration_model_from_mapping(data: Mapping[str, Any], *, where: str) -> IntegrationModel:
    check_keys(data, required=("frame_seconds",), where=where)
    return IntegrationModel(
        frame_seconds=number(data["frame_seconds"], field="frame_seconds", where=where,
                             positive=True),
    )


def load_integration_model(path: Path | str) -> IntegrationModel:
    path = Path(path)
    return integration_model_from_mapping(read_yaml(path), where=str(path))


__all__ = [
    "IntegrationModel",
    "integration_model_from_mapping",
    "load_integration_model",
]
