# coding=utf-8
"""Receiver-neutral detector contracts shared by products and runtime bundles."""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np

from pilot_proxy.detector_constants import (
    LOCKED_INPUT_FORMAT,
    LOCKED_NUM_WEIGHT_TERMS,
    LOCKED_POWER_ACCUMULATOR,
    LOCKED_REFERENCE_OFFSET_BINS,
    LOCKED_SAMPLE_BITS_PER_COMPONENT,
    LOCKED_SKIPPED_GUARD_BINS,
    POWER_SUM_ACCUMULATOR_BITS as LOCKED_POWER_ACCUMULATOR_BITS,
    SUPPORTED_DETECTOR_WINDOW_SAMPLES,
)
from pilot_proxy.json_utils import json_dumps_strict
from pilot_proxy.schema_identity import schema_token

DETECTOR_CONTRACT_SCHEMA_NAME = "pilotproxy_detector_contract"
DETECTOR_CONTRACT_SCHEMA_REVISION = 1
DETECTOR_CONTRACT_SCHEMA_TOKEN = schema_token(
    DETECTOR_CONTRACT_SCHEMA_NAME,
    DETECTOR_CONTRACT_SCHEMA_REVISION,
)
CHIME_RUN_CONFIG_SCHEMA_NAME = "pilotproxy_chime_run_config"
CHIME_RUN_CONFIG_SCHEMA_REVISION = 1
CHIME_RUN_CONFIG_SCHEMA_TOKEN = schema_token(
    CHIME_RUN_CONFIG_SCHEMA_NAME, CHIME_RUN_CONFIG_SCHEMA_REVISION
)
CHIME_STATS_SCHEMA_NAME = "pilotproxy_chime_stats"
CHIME_STATS_SCHEMA_REVISION = 1
CHIME_STATS_SCHEMA_TOKEN = schema_token(
    CHIME_STATS_SCHEMA_NAME, CHIME_STATS_SCHEMA_REVISION
)
NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE = "normalized_positive_excess_decision"
COARSE_POWER_RATIO_VALID_RULE = "p_ref_sum != 0"
# The corrected rule compares against the H0 zero-point of F implied by the
# int4-quantized weight norms, instead of assuming that zero-point is exactly 1.
# For white noise E[P_term] = sigma^2 * ||w_term||^2, so
# E[F] = null_power_ratio = 2*target_norm_sq/reference_norm_sum_sq, and quantization leaves the
# three norms unequal (null_power_ratio spans ~0.985..1.011 across the shipped ATSC 14-36
# bank). "F > 1" therefore pins the H0 mask fraction toward 0 or 1 per channel;
# "F > null_power_ratio" restores a channel-independent zero-point exactly, in integers.
NORMALIZED_POSITIVE_EXCESS_MASK_RULE = (
    "valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)"
)
NORMALIZED_POSITIVE_EXCESS_EQUIVALENT_RULE = (
    "R_coarse > R_null; R_null = "
    "2*target_norm_sq/reference_norm_sum_sq"
)
UINT64_MAX = (1 << 64) - 1
DETECTOR_POWER_RATIO_DEFINITION = "R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)"
ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION = (
    "R_coarse = 2 * sum(P_target) / (sum(P_ref_lower) + sum(P_ref_upper))"
)
COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO = "all_rows_summed_before_ratio"
WEIGHT_COORDINATE_POST_SPECTRAL_SENSE = "post_spectral_sense_normalization"
WEIGHT_COORDINATE_RAW_INPUT = "raw_input_frequency_coordinate"
INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED = (
    "post_spectral_sense_normalized"
)
INPUT_COORDINATE_RAW_INPUT = "raw_input_frequency_coordinate"
VALID_WEIGHT_COORDINATE_SYSTEMS = frozenset(
    {
        WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        WEIGHT_COORDINATE_RAW_INPUT,
    }
)
_DETECTOR_CONTRACT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "detector_window_samples",
        "skipped_guard_bins",
        "reference_offset_bins",
        "num_weight_terms",
        "sample_bits_per_component",
        "input_format",
        "power_accumulator",
        "power_accumulator_bits",
        "coarse_power_ratio_definition",
        "all_rows_coarse_power_ratio_definition",
        "combine_mode",
        "weight_coordinate_system",
        "input_coordinate_system",
        "input_preprocessing",
        "mask_source",
        "valid_rule",
        "mask_rule",
        "equivalent_mask_rule",
        "per_frequency_threshold",
        "threshold_mode",
    }
)
_DETECTOR_CONTRACT_OPTIONAL_FIELDS = frozenset(
    {"fine_reduction", "reference_placement_summary"}
)
_FINE_REDUCTION_FIELDS = frozenset(
    {
        "pad_factor",
        "cfar_policy",
        "p_fa",
        "guard_fine_bins",
        "designated_bins",
        "census_excluded_bins",
        "v1_marginal_identity",
    }
)


def normalize_weight_coordinate_system(value: object) -> str:
    """Return a validated public weight-coordinate-system string."""
    normalized = str(value).strip()
    if normalized not in VALID_WEIGHT_COORDINATE_SYSTEMS:
        raise ValueError(
            "weight_coordinate_system must be one of "
            f"{sorted(VALID_WEIGHT_COORDINATE_SYSTEMS)}; got {value!r}."
        )
    return normalized


def input_coordinate_system_for_weight_coordinate(
    weight_coordinate_system: object,
) -> str:
    """Return the input coordinate needed by the selected weights."""
    normalized = normalize_weight_coordinate_system(weight_coordinate_system)
    if normalized == WEIGHT_COORDINATE_RAW_INPUT:
        return INPUT_COORDINATE_RAW_INPUT
    return INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED


def _contract_int(contract: Mapping[str, Any], field: str) -> int:
    value = contract[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"detector contract {field} must be an integer.")
    return value


def _exact_uint64(value: object, *, field: str, positive: bool = False) -> int:
    """Validate exact detector power/norm integers without coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    minimum = 1 if positive else 0
    if not minimum <= result <= UINT64_MAX:
        interval = "[1, 2**64 - 1]" if positive else "[0, 2**64 - 1]"
        raise ValueError(f"{field} must be in {interval}.")
    return result


def _validate_fine_reduction(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("detector contract fine_reduction must be an object.")
    keys = {str(key) for key in value}
    missing = sorted(_FINE_REDUCTION_FIELDS - keys)
    unknown = sorted(keys - _FINE_REDUCTION_FIELDS)
    if missing:
        raise ValueError(
            f"detector contract fine_reduction is missing fields: {missing}"
        )
    if unknown:
        raise ValueError(
            f"detector contract fine_reduction contains unknown fields: {unknown}"
        )
    for field, minimum in (("pad_factor", 1), ("guard_fine_bins", 0)):
        field_value = value[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < minimum
        ):
            raise ValueError(
                f"detector contract fine_reduction.{field} must be an integer "
                f">= {minimum}."
            )
    p_fa = value["p_fa"]
    if (
        isinstance(p_fa, bool)
        or not isinstance(p_fa, (int, float))
        or not 0.0 < float(p_fa) <= 1.0
    ):
        raise ValueError(
            "detector contract fine_reduction.p_fa must be in (0, 1]."
        )
    if not isinstance(value["cfar_policy"], str) or not value["cfar_policy"]:
        raise ValueError(
            "detector contract fine_reduction.cfar_policy must be non-empty."
        )
    if value["v1_marginal_identity"] != "exact_int64_enforced_per_frame":
        raise ValueError(
            "detector contract fine_reduction.v1_marginal_identity is invalid."
        )
    for field in ("designated_bins", "census_excluded_bins"):
        bins = value[field]
        if (
            not isinstance(bins, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in bins
            )
            or len(set(bins)) != len(bins)
        ):
            raise ValueError(
                f"detector contract fine_reduction.{field} must be a unique "
                "integer list."
            )


def validate_detector_contract(contract: Mapping[str, Any]) -> None:
    """Validate the complete current detector-contract document."""
    keys = {str(key) for key in contract}
    missing = sorted(_DETECTOR_CONTRACT_REQUIRED_FIELDS - keys)
    unknown = sorted(
        keys
        - _DETECTOR_CONTRACT_REQUIRED_FIELDS
        - _DETECTOR_CONTRACT_OPTIONAL_FIELDS
    )
    if missing:
        raise ValueError(f"detector contract is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"detector contract contains unknown fields: {unknown}")
    if contract["schema_version"] != DETECTOR_CONTRACT_SCHEMA_TOKEN:
        raise ValueError(
            "detector contract schema_version does not match the current schema: "
            f"{contract['schema_version']!r} != "
            f"{DETECTOR_CONTRACT_SCHEMA_TOKEN!r}."
        )
    detector_window = _contract_int(contract, "detector_window_samples")
    if detector_window not in SUPPORTED_DETECTOR_WINDOW_SAMPLES:
        raise ValueError(
            "detector contract detector_window_samples must be one of "
            f"{sorted(SUPPORTED_DETECTOR_WINDOW_SAMPLES)}."
        )
    locked_values = {
        "skipped_guard_bins": LOCKED_SKIPPED_GUARD_BINS,
        "reference_offset_bins": LOCKED_REFERENCE_OFFSET_BINS,
        "num_weight_terms": LOCKED_NUM_WEIGHT_TERMS,
        "sample_bits_per_component": LOCKED_SAMPLE_BITS_PER_COMPONENT,
        "power_accumulator_bits": LOCKED_POWER_ACCUMULATOR_BITS,
    }
    for field, expected in locked_values.items():
        actual = _contract_int(contract, field)
        if actual != expected:
            raise ValueError(
                f"detector contract {field} must be {expected}; got {actual}."
            )
    fixed_values = {
        "input_format": LOCKED_INPUT_FORMAT,
        "power_accumulator": LOCKED_POWER_ACCUMULATOR,
        "coarse_power_ratio_definition": DETECTOR_POWER_RATIO_DEFINITION,
        "all_rows_coarse_power_ratio_definition": (
            ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION
        ),
        "combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        "mask_source": NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE,
        "valid_rule": COARSE_POWER_RATIO_VALID_RULE,
        "mask_rule": NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
        "equivalent_mask_rule": NORMALIZED_POSITIVE_EXCESS_EQUIVALENT_RULE,
        "threshold_mode": "none",
    }
    for field, expected in fixed_values.items():
        if contract[field] != expected:
            raise ValueError(
                f"detector contract {field} must be {expected!r}; "
                f"got {contract[field]!r}."
            )
    coordinate = normalize_weight_coordinate_system(
        contract["weight_coordinate_system"]
    )
    expected_input_coordinate = input_coordinate_system_for_weight_coordinate(
        coordinate
    )
    if contract["input_coordinate_system"] != expected_input_coordinate:
        raise ValueError(
            "detector contract input_coordinate_system does not match its "
            f"weight coordinate: {contract['input_coordinate_system']!r} != "
            f"{expected_input_coordinate!r}."
        )
    preprocessing = contract["input_preprocessing"]
    if not isinstance(preprocessing, Mapping) or set(preprocessing) != {
        "time_reverse_detector_windows_before_kernel"
    }:
        raise ValueError(
            "detector contract input_preprocessing must contain exactly "
            "time_reverse_detector_windows_before_kernel."
        )
    time_reverse = preprocessing["time_reverse_detector_windows_before_kernel"]
    if not isinstance(time_reverse, bool):
        raise ValueError(
            "detector contract time_reverse_detector_windows_before_kernel "
            "must be boolean."
        )
    if coordinate == WEIGHT_COORDINATE_RAW_INPUT and time_reverse:
        raise ValueError(
            "detector contract raw input-coordinate weights must not request "
            "detector-window time reversal before the kernel."
        )
    if contract["per_frequency_threshold"] is not False:
        raise ValueError("detector contract per_frequency_threshold must be false.")
    if "fine_reduction" in contract:
        _validate_fine_reduction(contract["fine_reduction"])


def normalized_positive_excess_policy() -> dict[str, Any]:
    """Return the norm-corrected positive-excess masking policy."""
    return {
        "mask_source": NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE,
        "valid_rule": COARSE_POWER_RATIO_VALID_RULE,
        "mask_rule": NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
        "equivalent_rule": NORMALIZED_POSITIVE_EXCESS_EQUIVALENT_RULE,
        "null_normalization": (
            "target_norm_sq and reference_norm_sum_sq are the exact integer squared "
            "norms of the packed target and (lower+upper) reference weight "
            "vectors; they remove the per-channel H0 F zero-point that int4 "
            "weight quantization introduces (E[F] = 2*target_norm_sq/"
            "reference_norm_sum_sq under a flat noise floor)."
        ),
    }


def weight_term_norms_sq(
    packed_weights: "np.ndarray",
    *,
    bits_per_component: int = 4,
) -> tuple[int, int, int]:
    """Return exact integer squared norms (target, ref_lower, ref_upper).

    ``packed_weights`` is the locked ``(3, K)`` packed complex weight array
    (real in the high nibble, imaginary in the low nibble for 4-bit
    components). Components are integers, so the squared norms are exact
    integers; they are returned as Python ints.
    """
    bits = int(bits_per_component)
    if bits not in (4, 8):
        raise ValueError(f"Unsupported component bit depth: {bits}. Expected 4 or 8.")
    w = np.asarray(packed_weights)
    if w.ndim != 2 or w.shape[0] != 3:
        raise ValueError(
            f"packed_weights must have shape (3, K); got {tuple(w.shape)}."
        )
    p = w.astype(np.int32, copy=False)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    real = p >> bits
    imag_raw = p & mask
    imag = np.where(imag_raw & sign_bit, imag_raw - (1 << bits), imag_raw)
    norms = (real.astype(np.int64) ** 2 + imag.astype(np.int64) ** 2).sum(axis=1)
    return int(norms[0]), int(norms[1]), int(norms[2])


def null_power_ratio_from_weight_norms(target_norm_sq: int, reference_norm_sum_sq: int) -> float:
    """Return null_power_ratio = 2*target_norm_sq/reference_norm_sum_sq, the flat-floor E[F]."""
    target_norm = _exact_uint64(
        target_norm_sq,
        field="target_norm_sq",
        positive=True,
    )
    reference_norm = _exact_uint64(
        reference_norm_sum_sq,
        field="reference_norm_sum_sq",
        positive=True,
    )
    return 2.0 * target_norm / reference_norm


def normalized_positive_excess(
    p_target: int,
    p_ref_sum: int,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
) -> int:
    """Exact integer norm-corrected positive-excess mask decision.

    Implements ``valid && (p_target * reference_norm_sum_sq > target_norm_sq *
    p_ref_sum)`` in unbounded Python integers, the exact form of
    ``F > null_power_ratio``. With ``target_norm_sq : reference_norm_sum_sq = 1 : 2`` this
    reduces to the legacy ``2*p_target > p_ref_sum`` rule.
    """
    num = _exact_uint64(p_target, field="p_target")
    den = _exact_uint64(p_ref_sum, field="p_ref_sum")
    target_norm = _exact_uint64(
        target_norm_sq,
        field="target_norm_sq",
        positive=True,
    )
    reference_norm = _exact_uint64(
        reference_norm_sum_sq,
        field="reference_norm_sum_sq",
        positive=True,
    )
    if den == 0:
        return 0
    return int(num * reference_norm > target_norm * den)


def build_detector_contract(
    *,
    detector_window_samples: int,
    skipped_guard_bins: int,
    reference_offset_bins: int,
    num_weight_terms: int,
    sample_bits_per_component: int = 4,
    input_format: str = "complex_int4_packed_int8",
    power_accumulator: str = "uint64",
    power_accumulator_bits: int = 64,
    combine_mode: str = COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    weight_coordinate_system: str,
    input_coordinate_system: str | None = None,
    time_reverse_detector_windows_before_kernel: bool = True,
    reference_placement_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public receiver-neutral contract for detector products."""
    mask_policy = normalized_positive_excess_policy()
    weight_coordinate = normalize_weight_coordinate_system(weight_coordinate_system)
    input_coordinate = (
        input_coordinate_system_for_weight_coordinate(weight_coordinate)
        if input_coordinate_system is None
        else str(input_coordinate_system)
    )
    contract: dict[str, Any] = {
        "schema_version": DETECTOR_CONTRACT_SCHEMA_TOKEN,
        "detector_window_samples": int(detector_window_samples),
        "skipped_guard_bins": int(skipped_guard_bins),
        "reference_offset_bins": int(reference_offset_bins),
        "num_weight_terms": int(num_weight_terms),
        "sample_bits_per_component": int(sample_bits_per_component),
        "input_format": str(input_format),
        "power_accumulator": str(power_accumulator),
        "power_accumulator_bits": int(power_accumulator_bits),
        "coarse_power_ratio_definition": (
            DETECTOR_POWER_RATIO_DEFINITION
        ),
        "all_rows_coarse_power_ratio_definition": (
            ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION
        ),
        "combine_mode": str(combine_mode),
        "weight_coordinate_system": weight_coordinate,
        "input_coordinate_system": input_coordinate,
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": bool(
                time_reverse_detector_windows_before_kernel
            ),
        },
        "mask_source": mask_policy["mask_source"],
        "valid_rule": mask_policy["valid_rule"],
        "mask_rule": mask_policy["mask_rule"],
        "equivalent_mask_rule": mask_policy["equivalent_rule"],
        "per_frequency_threshold": False,
        "threshold_mode": "none",
    }
    if reference_placement_summary is not None:
        contract["reference_placement_summary"] = reference_placement_summary
    validate_detector_contract(contract)
    return contract


def detector_contract_sha256(contract: dict[str, Any]) -> str:
    """Return the stable SHA256 for a detector-contract JSON object."""
    payload = json_dumps_strict(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ALL_ROWS_DETECTOR_POWER_RATIO_DEFINITION",
    "DETECTOR_CONTRACT_SCHEMA_NAME",
    "DETECTOR_CONTRACT_SCHEMA_REVISION",
    "DETECTOR_CONTRACT_SCHEMA_TOKEN",
    "CHIME_RUN_CONFIG_SCHEMA_NAME",
    "CHIME_RUN_CONFIG_SCHEMA_REVISION",
    "CHIME_RUN_CONFIG_SCHEMA_TOKEN",
    "CHIME_STATS_SCHEMA_NAME",
    "CHIME_STATS_SCHEMA_REVISION",
    "CHIME_STATS_SCHEMA_TOKEN",
    "COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO",
    "DETECTOR_POWER_RATIO_DEFINITION",
    "INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED",
    "INPUT_COORDINATE_RAW_INPUT",
    "NORMALIZED_POSITIVE_EXCESS_EQUIVALENT_RULE",
    "NORMALIZED_POSITIVE_EXCESS_MASK_RULE",
    "NORMALIZED_POSITIVE_EXCESS_MASK_SOURCE",
    "COARSE_POWER_RATIO_VALID_RULE",
    "VALID_WEIGHT_COORDINATE_SYSTEMS",
    "WEIGHT_COORDINATE_POST_SPECTRAL_SENSE",
    "WEIGHT_COORDINATE_RAW_INPUT",
    "build_detector_contract",
    "detector_contract_sha256",
    "input_coordinate_system_for_weight_coordinate",
    "normalize_weight_coordinate_system",
    "normalized_positive_excess_policy",
    "validate_detector_contract",
]
