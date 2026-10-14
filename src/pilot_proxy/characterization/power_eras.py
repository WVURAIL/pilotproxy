"""Exploratory power-era proposals from calendar-month, equal-event features.

This module is separate from the archived era rule. Relative power changes
do not establish transmitter on/off states, residual RFI, or science tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

INPUT_VERSION = "monthly-power-features-v1"
OUTPUT_VERSION = "robust-power-era-proposals-v1"
CONFIG_VERSION = "robust-power-eras-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _month_index(month: str) -> int:
    if not isinstance(month, str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month):
        raise ValueError("month must be YYYY-MM")
    year, number = map(int, month.split("-"))
    if year == 0:
        raise ValueError("month year must be >= 0001")
    return year * 12 + number - 1


def _month_name(index: int) -> str:
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}"


def _json_object(value: Any, name: str) -> dict:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    def validate_keys(item):
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} must use string keys at every depth")
            for child in item.values():
                validate_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate_keys(child)
    validate_keys(value)
    try:
        result = json.loads(_canonical(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return result


def _strict_fields(value: Mapping, allowed: set[str], required: set[str], name: str) -> dict:
    value = _json_object(value, name)
    unknown, missing = value.keys() - allowed, required - value.keys()
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {name} fields: {', '.join(sorted(missing))}")
    return value


@dataclass(frozen=True)
class MonthlyPower:
    """One calendar month; level_db is already reduced with equal event weight."""

    month: str
    level_db: float | None
    event_count: int
    frame_count: int
    day_count: int | None = None
    peak_offset_hz: float | None = None
    tracking_usable: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _month_index(self.month)
        if self.level_db is not None:
            object.__setattr__(self, "level_db", _number(self.level_db, "level_db"))
        _integer(self.event_count, "event_count")
        _integer(self.frame_count, "frame_count")
        if self.event_count > self.frame_count:
            raise ValueError("event_count cannot exceed frame_count")
        if self.day_count is not None:
            _integer(self.day_count, "day_count")
            if self.day_count > min(31, self.frame_count):
                raise ValueError("day_count cannot exceed 31 or frame_count")
        if self.peak_offset_hz is not None:
            object.__setattr__(self, "peak_offset_hz", _number(self.peak_offset_hz, "peak_offset_hz"))
        if type(self.tracking_usable) is not bool:
            raise ValueError("tracking_usable must be boolean")
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, value: Mapping) -> MonthlyPower:
        value = _strict_fields(value, {f.name for f in fields(cls)},
                               {"month", "level_db", "event_count", "frame_count"}, "month")
        return cls(**value)

@dataclass(frozen=True)
class PowerEraConfig:
    """Versioned exploratory defaults, not a calibrated false-alarm policy."""

    version: str = CONFIG_VERSION
    min_events: int = 5
    min_frames: int = 30
    min_days: int = 3
    block_months: int = 5
    required_consistent_months: int = 4
    minimum_practical_shift_db: float = 3.0
    robust_scale_multiplier: float = 3.0
    mad_normalization: float = 1.4826
    scale_floor_db: float = 0.25
    within_regime_band_floor_db: float = 1.0
    within_regime_band_max_fraction_of_separation: float = 1.0 / 3.0
    within_regime_band_sigma_multiplier: float = 2.0

    def __post_init__(self):
        if self.version != CONFIG_VERSION:
            raise ValueError(f"version must be {CONFIG_VERSION}")
        for name in ("min_events", "min_frames"):
            _integer(getattr(self, name), name, 1)
        _integer(self.min_days, "min_days")
        # Changing these structural choices requires another algorithm version.
        if type(self.block_months) is not int or self.block_months != 5:
            raise ValueError("this version requires block_months=5")
        if type(self.required_consistent_months) is not int or self.required_consistent_months != 4:
            raise ValueError("this version requires required_consistent_months=4")
        for name in (
            "minimum_practical_shift_db", "robust_scale_multiplier",
            "mad_normalization", "scale_floor_db", "within_regime_band_floor_db",
            "within_regime_band_max_fraction_of_separation",
            "within_regime_band_sigma_multiplier",
        ):
            value = _number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.within_regime_band_max_fraction_of_separation >= 0.5:
            raise ValueError("regime band fraction must be below 0.5")

    def canonical_json(self) -> str:
        return _canonical(asdict(self))

    @property
    def digest(self) -> str:
        return _digest(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping) -> PowerEraConfig:
        return cls(**_strict_fields(value, {f.name for f in fields(cls)}, set(), "config"))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("a median needs at least one value")
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] / 2 + ordered[n // 2] / 2)


def _sigma(values: Sequence[float], config: PowerEraConfig) -> float:
    center = _median(values)
    return config.mad_normalization * _median([abs(value - center) for value in values])


def _support_reasons(month: MonthlyPower, config: PowerEraConfig) -> list[str]:
    reasons = []
    if month.level_db is None:
        reasons.append("missing_level")
    if not month.tracking_usable:
        reasons.append("tracking_unusable")
    if month.event_count < config.min_events:
        reasons.append("insufficient_events")
    if month.frame_count < config.min_frames:
        reasons.append("insufficient_frames")
    if config.min_days > 0:
        if month.day_count is None:
            reasons.append("day_support_unverified")
        elif month.day_count < config.min_days:
            reasons.append("insufficient_days")
    return reasons


def _runs(indices: Sequence[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for index in indices:
        if not result or index != result[-1][-1] + 1:
            result.append([])
        result[-1].append(index)
    return result


def _compare(left: list[MonthlyPower], right: list[MonthlyPower],
             config: PowerEraConfig) -> dict:
    a_values = [row.level_db for row in left]
    b_values = [row.level_db for row in right]
    a, b = _median(a_values), _median(b_values)
    separation = abs(b - a)
    direction = 1 if b > a else -1 if b < a else 0
    a_sigma, b_sigma = _sigma(a_values, config), _sigma(b_values, config)
    sigma = max(config.scale_floor_db, a_sigma, b_sigma)
    required = max(config.minimum_practical_shift_db, config.robust_scale_multiplier * sigma)

    def deletion_medians(values):
        result = [_median(values)]
        if len(values) > 1:
            result.extend(_median(values[:i] + values[i + 1:]) for i in range(len(values)))
        return result

    delete_separation = min(direction * (v - u)
                            for u in deletion_medians(a_values)
                            for v in deletion_medians(b_values))
    a_band = min(separation * config.within_regime_band_max_fraction_of_separation,
                 max(config.within_regime_band_floor_db,
                     config.within_regime_band_sigma_multiplier * a_sigma))
    b_band = min(separation * config.within_regime_band_max_fraction_of_separation,
                 max(config.within_regime_band_floor_db,
                     config.within_regime_band_sigma_multiplier * b_sigma))
    a_consistent = [row.month for row in left if abs(row.level_db - a) <= a_band]
    b_consistent = [row.month for row in right if abs(row.level_db - b) <= b_band]
    all_rows = left + right
    old_observations = [row.month for row in all_rows if abs(row.level_db - a) <= a_band]
    new_observations = [row.month for row in all_rows if abs(row.level_db - b) <= b_band]
    old_last = max(a_consistent) if a_consistent else None
    new_first = min(b_consistent) if b_consistent else None
    # Refine only contiguous edge evidence, never an isolated opposite-regime
    # interior outlier. This separates observation bounds from a tied edge.
    for row in right:
        if abs(row.level_db - a) > a_band:
            break
        old_last = row.month
    for row in reversed(left):
        if abs(row.level_db - b) > b_band:
            break
        new_first = row.month
    ordered = bool(old_last and new_first and old_last < new_first)
    reasons = []
    if len(left) < config.block_months:
        reasons.append("insufficient_left_confirmation")
    if len(right) < config.block_months:
        reasons.append("insufficient_right_confirmation")
    if direction == 0:
        reasons.append("no_level_separation")
    elif delete_separation < required:
        reasons.append("delete_one_separation_below_requirement")
    if len(a_consistent) < config.required_consistent_months:
        reasons.append("left_regime_inconsistent")
    if len(b_consistent) < config.required_consistent_months:
        reasons.append("right_regime_inconsistent")
    return {
        "old_last_month": left[-1].month, "new_first_month": right[0].month,
        "left_months": [row.month for row in left], "right_months": [row.month for row in right],
        "left_center_db": a, "right_center_db": b, "direction": direction,
        "separation_db": separation, "delete_one_min_separation_db": delete_separation,
        "left_sigma_db": a_sigma, "right_sigma_db": b_sigma, "sigma_db": sigma,
        "required_separation_db": required, "strength_ratio": delete_separation / required,
        "left_consistency_band_db": a_band, "right_consistency_band_db": b_band,
        "left_consistent_months": a_consistent, "right_consistent_months": b_consistent,
        "confirmation_event_counts": [row.event_count for row in all_rows],
        "confirmation_frame_counts": [row.frame_count for row in all_rows],
        "confirmation_day_counts": [row.day_count for row in all_rows],
        "observation_bounds": {
            "last_old_consistent_month": old_last,
            "first_new_consistent_month": new_first,
            "old_last_month": old_last, "new_first_month": new_first,
            "ordered": ordered,
        },
        "unclassified_confirmation_months": [
            row.month for row in all_rows
            if row.month not in old_observations and row.month not in new_observations
        ],
        "reasons": reasons,
        "passes_numeric_rule": not reasons,
    }


def _cluster_edges(edges: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for edge in sorted(edges, key=lambda row: _month_index(row["new_first_month"])):
        adjacent = (clusters and edge["kind"] == clusters[-1][-1]["kind"]
                    and _month_index(edge["new_first_month"])
                    == _month_index(clusters[-1][-1]["new_first_month"]) + 1)
        if not adjacent:
            clusters.append([])
        clusters[-1].append(edge)
    return clusters


def _boundary(cluster: list[dict], number: int, eligible: dict[int, MonthlyPower]) -> dict:
    representative = max(cluster, key=lambda edge: (
        edge["strength_ratio"], -_month_index(edge["new_first_month"])))
    bounds = representative["observation_bounds"]
    lower = min(edge["old_last_month"] for edge in cluster)
    upper = max(edge["new_first_month"] for edge in cluster)
    transition_zone = []
    if bounds["ordered"]:
        lower_index = _month_index(bounds["last_old_consistent_month"])
        upper_index = _month_index(bounds["first_new_consistent_month"])
        transition_zone = [_month_name(index) for index in range(lower_index + 1, upper_index)
                           if index in eligible]
    evidence_months = sorted({month for edge in cluster
                             for month in edge["left_months"] + edge["right_months"]})
    return {
        "candidate_id": f"power-change-{number}",
        "kind": representative["kind"],
        "status": "proposed_for_review",
        "representative": representative,
        "passing_edges": cluster,
        "candidate_edge_interval": {"old_last_month": lower, "new_first_month": upper},
        "observation_bounds": bounds,
        "proposed_cohort_first_month": bounds["first_new_consistent_month"] if bounds["ordered"] else None,
        "transition_zone_months": transition_zone,
        "confirmation_months": evidence_months,
        "feature_diagnostics": [
            {"month": month,
             "peak_offset_hz": eligible[_month_index(month)].peak_offset_hz,
             "metadata": eligible[_month_index(month)].metadata}
            for month in evidence_months
        ],
        "overlapping_confirmation_windows": [],
        "warnings": [
            "dominant_feature_identity_unverified",
            "relative_power_not_physical_rfi_or_transmitter_state",
        ] + ([] if bounds["ordered"] else ["regime_ordering_ambiguous"]),
    }


def _excursions(runs: list[list[int]], eligible: dict[int, MonthlyPower],
                config: PowerEraConfig) -> list[dict]:
    marked = []
    for run in runs:
        for position in range(config.block_months, len(run) - config.block_months):
            left = [eligible[index].level_db for index in run[position - config.block_months:position]]
            right = [eligible[index].level_db for index in run[position + 1:position + 1 + config.block_months]]
            if abs(_median(left) - _median(right)) >= config.minimum_practical_shift_db:
                continue
            baseline = _median(left + right)
            scale = max(config.scale_floor_db, _sigma(left, config), _sigma(right, config))
            index = run[position]
            deviation = eligible[index].level_db - baseline
            if abs(deviation) >= max(config.minimum_practical_shift_db, config.robust_scale_multiplier * scale):
                marked.append(index)
    return [{
        "months": [_month_name(index) for index in group],
        "status": "excursion_for_display",
        "reason": "departures_from_compatible_flanking_five_month_medians",
        "creates_boundary": False,
    } for group in _runs(marked)]


def propose_power_eras(months: Sequence[MonthlyPower], *, channel: int,
                       provenance: Mapping, config: PowerEraConfig | None = None) -> dict:
    """Return reviewable candidate clusters; never replace accepted eras.

    Input values must already use the declared equal-event monthly reduction.
    Provenance is preserved and digested, not independently authenticated here.
    Frequency offsets and metadata are diagnostics, never boundary triggers.
    """
    _integer(channel, "channel", 1)
    if config is None:
        config = PowerEraConfig()
    if not isinstance(config, PowerEraConfig):
        raise ValueError("config must be a PowerEraConfig")
    months = tuple(months)
    if any(not isinstance(row, MonthlyPower) for row in months):
        raise ValueError("months must contain MonthlyPower records")
    indices = [_month_index(row.month) for row in months]
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("months must be unique and strictly increasing")
    provenance = _json_object(provenance, "provenance")
    if not provenance:
        raise ValueError("provenance must describe the input source and reduction")
    source = {
        "schema_version": INPUT_VERSION, "channel": channel,
        "provenance": provenance, "months": [asdict(row) for row in months],
    }
    records = []
    eligible = {}
    for index, row in zip(indices, months):
        reasons = _support_reasons(row, config)
        records.append(dict(asdict(row), supported=not reasons, support_reasons=reasons))
        if not reasons:
            eligible[index] = row
    runs = _runs(sorted(eligible))
    edges: list[dict] = []
    passing: list[dict] = []
    pending: list[dict] = []
    gaps: list[dict] = []
    n = config.block_months

    def evaluate(left_indices, right_indices, kind):
        edge = _compare([eligible[i] for i in left_indices],
                        [eligible[i] for i in right_indices], config)
        edge["kind"] = kind
        edge["status"] = "passing" if edge["passes_numeric_rule"] else "rejected"
        first_new = edge["observation_bounds"]["first_new_consistent_month"]
        if edge["passes_numeric_rule"] and first_new:
            first_new_index = _month_index(first_new)
            if not all(index in eligible for index in range(first_new_index, first_new_index + n)):
                edge["status"] = "pending"
                edge["reasons"].append("fewer_than_five_adjacent_supported_months_from_observed_new")
        edges.append(edge)
        if edge["status"] == "passing":
            passing.append(edge)
        elif (edge["status"] == "pending"
              or ((len(left_indices) < n or len(right_indices) < n)
                  and edge["separation_db"] >= edge["required_separation_db"])):
            pending.append({
                "old_last_month": edge["old_last_month"],
                "new_first_month": edge["new_first_month"],
                "kind": kind, "status": "pending_confirmation",
                "reason": "insufficient_consecutive_support_for_confirmation",
                "reasons": edge["reasons"],
                "observation_bounds": edge["observation_bounds"],
                "diagnostics": edge,
            })
        return edge

    for run in runs:
        for position in range(1, len(run)):
            evaluate(run[max(0, position - n):position], run[position:position + n],
                     "observed_power_change")
    for left_run, right_run in zip(runs, runs[1:]):
        last, first = left_run[-1], right_run[0]
        edge = evaluate(left_run[-n:], right_run[:n], "gap_censored_power_change")
        gaps.append({
            "last_supported_month": _month_name(last),
            "first_supported_month": _month_name(first),
            "unsupported_first_month": _month_name(last + 1),
            "unsupported_last_month": _month_name(first - 1),
            "unsupported_month_count": first - last - 1,
            "status": ("censored_change_candidate" if edge["status"] == "passing"
                       else "insufficient_confirmation" if min(len(left_run), len(right_run)) < n
                       else "compatible_observed_levels" if edge["separation_db"] < edge["required_separation_db"]
                       else "difference_unconfirmed"),
            "continuity_established": False,
            "diagnostics": edge,
        })
    boundaries = [_boundary(cluster, i, eligible)
                  for i, cluster in enumerate(_cluster_edges(passing), start=1)]
    for i, boundary in enumerate(boundaries):
        for other in boundaries[i + 1:]:
            if set(boundary["confirmation_months"]) & set(other["confirmation_months"]):
                boundary["overlapping_confirmation_windows"].append(other["candidate_id"])
                other["overlapping_confirmation_windows"].append(boundary["candidate_id"])
    # Do not duplicate a confirmed candidate as a short-window endpoint warning.
    pending = [item for item in pending
               if not any(boundary["candidate_edge_interval"]["old_last_month"]
                          <= item["old_last_month"]
                          < boundary["candidate_edge_interval"]["new_first_month"]
                          for boundary in boundaries)]
    result = {
        "schema_version": OUTPUT_VERSION, "channel": channel,
        "status": "candidate_review_required" if eligible else "no_supported_months",
        "partition_status": "not_constructed_candidate_review_required",
        "accepted_eras_changed": False,
        "science_selection_eligible": False,
        "input_digest": _digest(source), "provenance": provenance,
        "config": asdict(config), "config_digest": config.digest,
        "design_provenance": (
            "Exploratory design informed by previously inspected channel displays and synthetic failure modes; "
            "no channel is held out. Defaults are not a false-alarm calibration."
        ),
        "feature_interpretation": "dominant_relative_spectral_feature",
        "months": records,
        "supported_ranges": [{"first_month": _month_name(run[0]), "last_month": _month_name(run[-1]),
                              "supported_months": len(run)} for run in runs],
        "unsupported_ranges": [
            {"first_month": _month_name(run[0]), "last_month": _month_name(run[-1]),
             "unsupported_months": len(run)}
            for run in _runs([i for i in range(indices[0], indices[-1] + 1) if i not in eligible])
        ] if indices else [],
        "boundaries": boundaries, "pending_changes": pending, "gaps": gaps,
        "excursions": _excursions(runs, eligible, config), "edge_checks": edges,
        "limitations": [
            "No accepted era partition or transmitter on/off claim is produced.",
            "Five-month blocks with four consistent values may still identify four-month or seasonal episodes.",
            "The 3 dB practical separation is not RFIsher tolerance or a statistical significance level.",
            "A broad-window maximum can select noise, secondary tones, or instrument tones; identity is unverified.",
            "Frequency drift alone is not a power-boundary trigger; window exit or feature switching can mimic power changes.",
            "Relative PSD cannot recover absolute power changes. Preserve spectrogram and frequency diagnostics.",
            "Overlapping confirmation windows require explicit review before any production partition.",
        ],
    }
    # Finite input scalars can still overflow derived differences or scales.
    # Refuse such a result rather than emitting nonfinite diagnostic evidence.
    _canonical(result)
    return result


def input_from_dict(value: Mapping) -> tuple[tuple[MonthlyPower, ...], int, dict, PowerEraConfig]:
    """Validate one single-channel document without interpreting extra fields."""
    value = _strict_fields(value, {"schema_version", "channel", "provenance", "months", "config"},
                           {"schema_version", "channel", "provenance", "months"}, "input")
    if value["schema_version"] != INPUT_VERSION:
        raise ValueError(f"schema_version must be {INPUT_VERSION}")
    if not isinstance(value["months"], list):
        raise ValueError("months must be an array")
    rows = tuple(MonthlyPower.from_dict(row) for row in value["months"])
    return rows, value["channel"], value["provenance"], PowerEraConfig.from_dict(value.get("config", {}))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        months, channel, provenance, config = input_from_dict(document)
        result = propose_power_eras(months, channel=channel, provenance=provenance, config=config)
        result["input_file"] = {"path": str(args.input), "sha256": hashlib.sha256(raw).hexdigest()}
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
