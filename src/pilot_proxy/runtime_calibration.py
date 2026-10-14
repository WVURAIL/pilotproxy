"""Fine score profiles and populations for a frozen fine_decision_v1 detector.

A score profile declares the receiver, weights, producer and fine geometry a
population of minimum keeping Q16 words was measured with; a population lists
those words per trial. These validators check declared identities and evidence
contracts, not their physical truth. The capture replay writes and checks
them; the operating threshold itself comes from the threshold rule
(:mod:`pilot_proxy.characterization.threshold_rule`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    FINE_DECISION_VERSION,
    unpack_bulk_mask,
)

PROFILE_SCHEMA = "pilotproxy_fine_score_profile_v1"
POPULATION_SCHEMA = "pilotproxy_fine_score_population_v1"
SCORE_ENCODING = "minimum_keeping_multiplier_q16_v1"
SUPPORT_RULE = "positive_denominator_rank_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, lower=0, upper=None):
    _require(
        type(value) is int and value >= lower and (upper is None or value <= upper),
        f"{name} must be an integer in the declared range",
    )
    return value


def _text(value, name):
    _require(
        isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text"
    )
    return value


def _digest(value, name):
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        f"{name} must be a lowercase SHA256 digest",
    )
    return value


def _probability(value, name):
    _require(
        type(value) in (int, float) and math.isfinite(value) and 0 < value < 1,
        f"{name} must be a finite probability strictly between zero and one",
    )
    return float(value)


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _read(path):
    # Reject duplicate keys and NaN/Infinity rather than quietly changing identity.
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON field {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value {value}")

    with Path(path).open() as stream:
        result = json.load(stream, object_pairs_hook=pairs, parse_constant=nonfinite)
    _require(isinstance(result, dict), "JSON root must be an object")
    return result


def _bound_file(reference, base, bindings):
    _require(isinstance(reference, dict), "File reference must contain path and sha256")
    expected = _digest(reference.get("sha256"), "file sha256")
    path = (base / _text(reference.get("path"), "file path")).resolve()
    _require(
        path.is_file() and sha256(path) == expected, f"File identity mismatch: {path}"
    )
    bindings[str(path)] = expected
    return path


def validate_score_profile(profile):
    """Validate the expressible submitted geometry; choose no target or rank."""
    _require(
        profile.get("schema_version") == PROFILE_SCHEMA, "Unknown score profile schema"
    )
    _text(profile.get("profile_id"), "profile_id")
    _require(
        profile.get("decision_version") == FINE_DECISION_VERSION,
        "Wrong decision version",
    )
    _require(profile.get("score_encoding") == SCORE_ENCODING, "Wrong score encoding")
    _require(profile.get("support_rule") == SUPPORT_RULE, "Wrong runtime support rule")
    for name in (
        "receiver_profile_sha256",
        "weights_sha256",
        "producer_sha256",
        "packing_contract_sha256",
        "input_state_contract_sha256",
    ):
        _digest(profile.get(name), name)
    _text(profile.get("input_population_id"), "input_population_id")
    _integer(profile.get("physical_channel"), "physical_channel", 1)
    _integer(profile.get("num_input_streams"), "num_input_streams", 1)
    window = _integer(
        profile.get("detector_window_samples"), "detector_window_samples", 1
    )
    _require(window in (64, 128), "Unsupported detector window")
    _require(
        type(profile.get("windows_per_stream")) is int
        and profile["windows_per_stream"] == 128,
        "The submitted fine transform requires 128 windows",
    )
    _require(
        type(profile.get("frame_size_samples")) is int
        and profile["frame_size_samples"] == 128 * window,
        "Frame size differs from the submitted fine geometry",
    )
    _require(
        type(profile.get("fine_bins")) is int and profile["fine_bins"] == 256,
        "Fine FFT must have 256 bins",
    )
    anchor = _integer(profile.get("anchor_bin"), "anchor_bin", 0, 255)
    width = _integer(
        profile.get("designated_half_width"), "designated_half_width", 0, 127
    )
    words = profile.get("bulk_mask_words_hex")
    _require(
        isinstance(words, list)
        and len(words) == 4
        and all(
            isinstance(w, str) and re.fullmatch(r"0x[0-9a-fA-F]{1,16}", w)
            for w in words
        ),
        "bulk_mask_words_hex must contain four uint64 hex words",
    )
    mask = unpack_bulk_mask([int(w, 16) for w in words])
    population = int(mask.sum())
    rank = _integer(profile.get("cfar_rank"), "cfar_rank", 0, 255)
    _require(rank < population, "Rank has insufficient frozen reference support")
    guard = max(width, 2)
    _require(
        not any(
            mask[(anchor + offset) % 256] for offset in range(-guard, guard + 1)
        ),
        "Runtime designated/guard bins cannot appear in the reference mask",
    )
    return population


def _load_population(path, profile, profile_sha, role):
    data = _read(path)
    _require(
        data.get("schema_version") == POPULATION_SCHEMA,
        "Unknown score population schema",
    )
    _require(data.get("role") == role, f"Expected {role} population")
    _require(
        data.get("profile_sha256") == profile_sha
        and data.get("profile_id") == profile["profile_id"],
        "Score/profile identity mismatch",
    )
    _require(
        data.get("score_encoding") == SCORE_ENCODING,
        "Population score encoding mismatch",
    )
    _text(data.get("population_id"), "population_id")
    law = data.get("null_law")
    _require(isinstance(law, dict), "Missing null_law identity")
    _text(law.get("id"), "null_law.id")
    _require(
        law.get("kind")
        in ("synthetic_model", "operational_background", "verified_signal_free"),
        "Unknown null_law kind",
    )
    _text(law.get("description"), "null_law.description")
    _digest(law.get("evidence_sha256"), "null_law.evidence_sha256")
    sampling = data.get("sampling")
    _require(
        isinstance(sampling, dict)
        and sampling.get("assumption") == "independent_identically_distributed",
        "This tool requires a justified IID sampling contract; it cannot count correlated frames as IID",
    )
    _text(sampling.get("independence_rationale"), "independence_rationale")
    _digest(sampling.get("design_sha256"), "sampling.design_sha256")
    rows = data.get("scores")
    _require(isinstance(rows, list) and rows, "Scores must be a nonempty list")
    values, trial_ids, independent_ids = [], [], []
    population = validate_score_profile(profile)
    for row in rows:
        _require(isinstance(row, dict), "Score rows must be objects")
        trial_ids.append(_text(row.get("trial_id"), "trial_id"))
        independent_ids.append(
            _text(row.get("independent_unit_id"), "independent_unit_id")
        )
        _require(
            row.get("valid") is True,
            "Invalid score/support: no rows may be dropped or replaced",
        )
        n_bulk = _integer(row.get("n_bulk"), "n_bulk", 0, population)
        _require(n_bulk > profile["cfar_rank"], "Score declares invalid rank support")
        encoded = row.get("required_multiplier_q16")
        _require(
            isinstance(encoded, str) and re.fullmatch(r"[1-9][0-9]*", encoded),
            "Q16 requirements must be exact unsigned decimal strings, not floats or JSON numbers",
        )
        value = int(encoded)
        _require(
            value <= ALWAYS_MASKED_Q16, "Q16 requirement exceeds the logical sentinel"
        )
        values.append(value)
    _require(len(set(trial_ids)) == len(rows), "Repeated trial identity")
    _require(
        len(set(independent_ids)) == len(rows),
        "Repeated independent unit: correlated frames cannot inflate n",
    )
    return data, values, set(trial_ids), set(independent_ids)


def _write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
