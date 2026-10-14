# coding=utf-8
"""Strict detector-core profile for the CUDA local-reference power-ratio contract."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.detector_constants import (
    DEFAULT_DETECTOR_WINDOW_SAMPLES,
    DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET,
    LOCKED_COARSE_POWER_RATIO_DEFINITION,
    LOCKED_HOST_MASKING_POLICY,
    LOCKED_INPUT_FORMAT,
    LOCKED_NUM_WEIGHT_TERMS,
    LOCKED_PACKED_COMPLEX_BITS,
    LOCKED_POWER_ACCUMULATOR,
    LOCKED_RAW_PILOT_EXCESS_DEFINITION,
    LOCKED_REFERENCE_OFFSET_BINS,
    LOCKED_SAMPLE_BITS_PER_COMPONENT,
    LOCKED_SKIPPED_GUARD_BINS,
    MAG_SQUARED_ACCUMULATOR_BITS_TARGET,
    POWER_SUM_ACCUMULATOR_BITS,
    SUPPORTED_DETECTOR_WINDOW_SAMPLES,
)

from .schemas import (
    DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO,
    DETECTOR_CORE_PROFILE_SCHEMA_TOKEN,
)

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "detector_core_id", "kernel_contract", "fixed_point_limits"}
)
_KERNEL_CONTRACT_FIELDS = frozenset(
    {
        "detector_window_samples",
        "supported_detector_window_samples",
        "num_weight_terms",
        "skipped_guard_bins",
        "packed_complex_bits",
        "sample_bits_per_component",
        "input_format",
        "power_accumulator",
        "coarse_power_ratio_definition",
        "raw_pilot_excess_definition",
        "host_masking_policy",
        "per_frequency_threshold",
    }
)
_FIXED_POINT_LIMIT_FIELDS = frozenset(
    {
        "dot_product_component_accumulator_bits_target",
        "mag_squared_accumulator_bits_target",
        "power_sum_accumulator_bits",
    }
)


def _validate_exact_keys(
    data: Mapping[str, Any],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    keys = {str(key) for key in data}
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")


def _require_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    required: frozenset[str],
    context: str,
) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    out = dict(value)
    _validate_exact_keys(out, required=required, context=context)
    return out


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _exact_integer(value: object, *, field: str) -> int:
    """Validate programmatic core geometry without truncation."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


@dataclass(frozen=True)
class DetectorCoreProfile:
    """File-backed description of the exact CUDA detector core contract."""

    schema_version: str
    detector_core_id: str
    detector_window_samples: int
    supported_detector_window_samples: tuple[int, ...]
    num_weight_terms: int
    skipped_guard_bins: int
    packed_complex_bits: int
    sample_bits_per_component: int
    input_format: str
    power_accumulator: str
    coarse_power_ratio_definition: str
    raw_pilot_excess_definition: str
    host_masking_policy: str
    per_frequency_threshold: bool
    fixed_point_limits: dict[str, int]

    def __post_init__(self) -> None:
        if self.schema_version != DETECTOR_CORE_PROFILE_SCHEMA_TOKEN:
            raise ValueError(
                "unsupported detector core profile schema_version: "
                f"{self.schema_version!r}"
            )
        if self.detector_core_id != DETECTOR_CORE_ID_PILOT_PROXY_CUDA_LOCAL_REFERENCE_POWER_RATIO:
            raise ValueError(f"unsupported detector_core_id: {self.detector_core_id!r}")
        supported = tuple(
            _exact_integer(
                value,
                field="supported_detector_window_samples",
            )
            for value in self.supported_detector_window_samples
        )
        expected_supported = tuple(sorted(SUPPORTED_DETECTOR_WINDOW_SAMPLES))
        if supported != expected_supported:
            raise ValueError(
                "supported_detector_window_samples must exactly describe the "
                f"compiled core family {list(expected_supported)}; got "
                f"{list(supported)}."
            )
        object.__setattr__(self, "supported_detector_window_samples", supported)
        detector_window = _exact_integer(
            self.detector_window_samples,
            field="detector_window_samples",
        )
        if detector_window not in supported:
            raise ValueError(
                "detector_window_samples must be one of "
                f"{list(supported)}; got {self.detector_window_samples}."
            )
        object.__setattr__(self, "detector_window_samples", detector_window)
        exact_locked_fields = {
            "num_weight_terms": _exact_integer(
                self.num_weight_terms,
                field="num_weight_terms",
            ),
            "skipped_guard_bins": _exact_integer(
                self.skipped_guard_bins,
                field="skipped_guard_bins",
            ),
            "packed_complex_bits": _exact_integer(
                self.packed_complex_bits,
                field="packed_complex_bits",
            ),
            "sample_bits_per_component": _exact_integer(
                self.sample_bits_per_component,
                field="sample_bits_per_component",
            ),
        }
        for field, value in exact_locked_fields.items():
            object.__setattr__(self, field, value)
        if self.num_weight_terms != LOCKED_NUM_WEIGHT_TERMS:
            raise ValueError("num_weight_terms must be locked to 3.")
        if self.skipped_guard_bins != LOCKED_SKIPPED_GUARD_BINS:
            raise ValueError(
                "skipped_guard_bins must match the compiled reference offset: "
                f"expected {LOCKED_SKIPPED_GUARD_BINS}, got "
                f"{self.skipped_guard_bins}."
            )
        if self.packed_complex_bits != LOCKED_PACKED_COMPLEX_BITS:
            raise ValueError("packed_complex_bits must be locked to 8.")
        if self.sample_bits_per_component != LOCKED_SAMPLE_BITS_PER_COMPONENT:
            raise ValueError("sample_bits_per_component must be locked to 4.")
        if self.input_format != LOCKED_INPUT_FORMAT:
            raise ValueError(f"input_format must be {LOCKED_INPUT_FORMAT!r}.")
        if self.power_accumulator != LOCKED_POWER_ACCUMULATOR:
            raise ValueError(
                f"power_accumulator must be {LOCKED_POWER_ACCUMULATOR!r}."
            )
        if (
            self.coarse_power_ratio_definition
            != LOCKED_COARSE_POWER_RATIO_DEFINITION
        ):
            raise ValueError(
                "coarse_power_ratio_definition must be "
                f"{LOCKED_COARSE_POWER_RATIO_DEFINITION!r}."
            )
        if (
            self.raw_pilot_excess_definition
            != LOCKED_RAW_PILOT_EXCESS_DEFINITION
        ):
            raise ValueError(
                "raw_pilot_excess_definition must be "
                f"{LOCKED_RAW_PILOT_EXCESS_DEFINITION!r}."
            )
        if self.host_masking_policy != LOCKED_HOST_MASKING_POLICY:
            raise ValueError(
                "host_masking_policy must be "
                f"{LOCKED_HOST_MASKING_POLICY!r}."
            )
        if self.per_frequency_threshold is not False:
            raise ValueError("per_frequency_threshold must be false.")
        limits = {
            str(key): _exact_integer(value, field=f"fixed_point_limits.{key}")
            for key, value in self.fixed_point_limits.items()
        }
        expected_limits = {
            "dot_product_component_accumulator_bits_target": (
                DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET
            ),
            "mag_squared_accumulator_bits_target": (
                MAG_SQUARED_ACCUMULATOR_BITS_TARGET
            ),
            "power_sum_accumulator_bits": POWER_SUM_ACCUMULATOR_BITS,
        }
        if limits != expected_limits:
            raise ValueError(
                "fixed_point_limits must exactly match the compiled core: "
                f"expected={expected_limits}, got={limits}."
            )
        object.__setattr__(self, "fixed_point_limits", limits)

    @property
    def reference_offset_bins(self) -> int:
        return int(self.skipped_guard_bins) + 1

    def with_detector_window_samples(
        self, detector_window_samples: int
    ) -> "DetectorCoreProfile":
        """Select one supported compile-time detector window length."""
        return replace(
            self,
            detector_window_samples=detector_window_samples,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorCoreProfile":
        if not isinstance(data, Mapping):
            raise ValueError("detector core profile must be an object.")
        raw = dict(data)
        _validate_exact_keys(
            raw,
            required=_TOP_LEVEL_FIELDS,
            context="detector core profile",
        )
        kernel_contract = _require_mapping(
            raw,
            "kernel_contract",
            required=_KERNEL_CONTRACT_FIELDS,
            context="detector core profile kernel_contract",
        )
        fixed_point_limits = _require_mapping(
            raw,
            "fixed_point_limits",
            required=_FIXED_POINT_LIMIT_FIELDS,
            context="detector core profile fixed_point_limits",
        )
        supported_raw = kernel_contract["supported_detector_window_samples"]
        if not isinstance(supported_raw, list):
            raise ValueError(
                "kernel_contract.supported_detector_window_samples must be a list."
            )
        return cls(
            schema_version=_require_nonempty_str(
                raw["schema_version"], "schema_version"
            ),
            detector_core_id=_require_nonempty_str(
                raw["detector_core_id"], "detector_core_id"
            ),
            detector_window_samples=_require_int(
                kernel_contract["detector_window_samples"],
                "kernel_contract.detector_window_samples",
            ),
            supported_detector_window_samples=tuple(
                _require_int(
                    value,
                    "kernel_contract.supported_detector_window_samples[]",
                )
                for value in supported_raw
            ),
            num_weight_terms=_require_int(
                kernel_contract["num_weight_terms"],
                "kernel_contract.num_weight_terms",
            ),
            skipped_guard_bins=_require_int(
                kernel_contract["skipped_guard_bins"],
                "kernel_contract.skipped_guard_bins",
            ),
            packed_complex_bits=_require_int(
                kernel_contract["packed_complex_bits"],
                "kernel_contract.packed_complex_bits",
            ),
            sample_bits_per_component=_require_int(
                kernel_contract["sample_bits_per_component"],
                "kernel_contract.sample_bits_per_component",
            ),
            input_format=_require_nonempty_str(
                kernel_contract["input_format"],
                "kernel_contract.input_format",
            ),
            power_accumulator=_require_nonempty_str(
                kernel_contract["power_accumulator"],
                "kernel_contract.power_accumulator",
            ),
            coarse_power_ratio_definition=_require_nonempty_str(
                kernel_contract["coarse_power_ratio_definition"],
                "kernel_contract.coarse_power_ratio_definition",
            ),
            raw_pilot_excess_definition=_require_nonempty_str(
                kernel_contract["raw_pilot_excess_definition"],
                "kernel_contract.raw_pilot_excess_definition",
            ),
            host_masking_policy=_require_nonempty_str(
                kernel_contract["host_masking_policy"],
                "kernel_contract.host_masking_policy",
            ),
            per_frequency_threshold=_require_bool(
                kernel_contract["per_frequency_threshold"],
                "kernel_contract.per_frequency_threshold",
            ),
            fixed_point_limits={
                key: _require_int(
                    fixed_point_limits[key],
                    f"fixed_point_limits.{key}",
                )
                for key in sorted(_FIXED_POINT_LIMIT_FIELDS)
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectorCoreProfile":
        return load_detector_core_profile(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_core_id": self.detector_core_id,
            "kernel_contract": {
                "detector_window_samples": int(self.detector_window_samples),
                "supported_detector_window_samples": [
                    int(value)
                    for value in self.supported_detector_window_samples
                ],
                "num_weight_terms": int(self.num_weight_terms),
                "skipped_guard_bins": int(self.skipped_guard_bins),
                "packed_complex_bits": int(self.packed_complex_bits),
                "sample_bits_per_component": int(self.sample_bits_per_component),
                "input_format": self.input_format,
                "power_accumulator": self.power_accumulator,
                "coarse_power_ratio_definition": (
                    self.coarse_power_ratio_definition
                ),
                "raw_pilot_excess_definition": (
                    self.raw_pilot_excess_definition
                ),
                "host_masking_policy": self.host_masking_policy,
                "per_frequency_threshold": bool(self.per_frequency_threshold),
            },
            "fixed_point_limits": dict(self.fixed_point_limits),
        }


def load_detector_core_profile(path: str | Path) -> DetectorCoreProfile:
    """Load one exact detector-core profile JSON document."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("detector core profile JSON must contain an object.")
    return DetectorCoreProfile.from_dict(data)


__all__ = [
    "DEFAULT_DETECTOR_WINDOW_SAMPLES",
    "DetectorCoreProfile",
    "SUPPORTED_DETECTOR_WINDOW_SAMPLES",
    "LOCKED_NUM_WEIGHT_TERMS",
    "LOCKED_REFERENCE_OFFSET_BINS",
    "LOCKED_SKIPPED_GUARD_BINS",
    "load_detector_core_profile",
]
