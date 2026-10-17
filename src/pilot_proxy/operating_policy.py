"""Compile the threshold rule's decisions into the channel operating policy kotekan reads.

The plan names the receiver profile and its frequency-ID namespace
(``chime_freq_id`` for the CHIME upgrade, ``chord_freq_id`` for CHORD), the
threshold rule's ``decisions.json`` (``tools/threshold_rule.py``), the runtime
bundle, the consumer and the era. Every ATSC channel 14 to 36 of the decisions
becomes one assignment at its receiver frequency ID:

- deployed (outcome ``useful``): a finite threshold at the decided Q16 word.
  The bundle's profile for the channel must carry that word
  (``fine_calibration`` calibrated with ``cfar_multiplier_q16`` equal to it);
  until it does, the channel is pending and the configuration is incomplete;
- not deployed (outcome ``not useful``): ``reject_all``, the permanent mask,
  with the rule's reason; except a channel whose reason is in
  ``UNMASKED_REASONS`` (DTV without a matching pilot: the detector has no
  pilot to act on and would never fire), which is ``pass_all``, left unmasked
  and listed under ``unmasked_freq_ids``.

A decision outside channels 14 to 36 (the channel 37 control band) is not a
DTV channel and is reported, not assigned. On ``chime_freq_id`` the frequency
ID of each channel is the project's (``Project.target_freq_id``), and the
bundle must agree with it.

The reject-all limit is encoded explicitly, not as a zero Q16 multiplier.
Compilation authenticates declared inputs; it never activates a configuration,
verifies consumer capability implementations, or certifies scientific recovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pilot_proxy import runtime_calibration as calibration
from pilot_proxy.atsc_channels import physical_channel_to_lower_edge_hz
from pilot_proxy.config.project import default_project
from pilot_proxy.fine_decision import FINE_DECISION_VERSION, MAX_MULTIPLIER_Q16
from pilot_proxy.integration.receiver_profile import (
    load_receiver_profile,
    receiver_profile_hash,
)
from pilot_proxy.runtime_bundle import (
    CHANNEL_ID_NAMESPACE_FIELD_ALIASES,
    REQUIRED_RUNTIME_BUNDLE_FILES,
    validate_runtime_weight_bundle,
)

PLAN_SCHEMA = "pilotproxy_channel_operating_policy_plan_v2"
POLICY_SCHEMA = "pilotproxy_channel_operating_policy_v2"
MANIFEST_SCHEMA = "pilotproxy_channel_operating_policy_manifest_v1"
CONSUMER_NAME = "Kotekan DtvRfiMask"
REQUIRED_CAPABILITIES = frozenset({
    "dtv_fine_support_v1", "dtv_permanent_mask_freq_ids_v1", "dtv_require_fine_freq_ids_v1",
})
CONSUMER_FREQUENCY_ID_LIMIT = 12288
NAMESPACES = tuple(sorted(CHANNEL_ID_NAMESPACE_FIELD_ALIASES))
FIRST_CHANNEL, LAST_CHANNEL = 14, 36        # the ATSC UHF television channels a DTV policy covers
CHANNEL_WIDTH_HZ = 6_000_000
OUTCOMES = {"useful": "finite", "not useful": "reject_all"}
# not deployed and not masked: the rule's reasons for which the detector has nothing to act on
UNMASKED_REASONS = frozenset({"DTV without a matching pilot", "pilot not seen"})   # the 2026-10-12 wording of the same verdict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, required, name, optional=()):
    _require(isinstance(value, dict), f"{name} must be an object")
    keys = set(value)
    _require(set(required) <= keys and keys <= set(required) | set(optional),
             f"{name} has missing or unknown fields")


def _evidence(value, base, bindings, name):
    _require(isinstance(value, list) and bool(value), f"{name} requires source evidence")
    paths = []
    for ref in value:
        paths.append(_file(ref, base, bindings))
    _require(len(set(paths)) == len(paths), f"Duplicate {name} source")


def _file(reference, base, bindings):
    _fields(reference, {"path", "sha256"}, "file reference")
    new = {}
    path = calibration._bound_file(reference, base, new)
    _remember(bindings, new)
    return path


def _remember(bindings, new):
    for path, digest in new.items():
        _require(path not in bindings or bindings[path] == digest, "Conflicting source identities: " + path)
        bindings[path] = digest


def _q16(value):
    _require(type(value) is int or (isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None),
             "a deployed word must be a positive integer or its canonical decimal string")
    word = int(value)
    _require(1 <= word <= MAX_MULTIPLIER_Q16, "a deployed word must be a positive uint64")
    return word


def _decisions(path):
    """The threshold rule's per-channel decisions, checked: useful carries its word, not useful its reason."""
    # the rule may write a non-finite diagnostic (a time at zero retention); the fields read here are checked below
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(data, dict) and isinstance(data.get("channels"), list) and data["channels"],
             "decisions.json must list its channels")
    rows = {}
    for row in data["channels"]:
        _require(isinstance(row, dict), "Decision row must be an object")
        channel = calibration._integer(row.get("channel"), "decision channel", 1)
        _require(channel not in rows, "Duplicate decision channel")
        _require(row.get("outcome") in OUTCOMES, f"channel {channel}: outcome must be useful or not useful")
        if row["outcome"] == "useful":
            _q16(row.get("deployed_word"))
        else:
            calibration._text(row.get("reason"), f"channel {channel} reason")
        rows[channel] = row
    return rows


def _bundle(plan, base, bindings, receiver):
    refs = plan["bundle_files"]
    _fields(refs, set(REQUIRED_RUNTIME_BUNDLE_FILES), "bundle_files")
    files = {name: _file(refs[name], base, bindings) for name in refs}
    parents = {p.parent for p in files.values()}
    _require(len(parents) == 1 and all(p.name == name for name, p in files.items()),
             "Bundle files must retain names in one directory")
    result = validate_runtime_weight_bundle(bundle_dir=next(iter(parents)))
    _require(result["valid"], f"Invalid runtime bundle: {result['errors']}")
    data = {name: calibration._read(path) for name, path in files.items() if name.endswith(".json")}
    profiles, manifest = data["pilot_profiles.json"], data["weights.manifest.json"]
    _require(profiles["receiver_profile_id"] == receiver.receiver_profile_id
             and manifest["receiver_profile_hash"] == receiver_profile_hash(receiver),
             "Bundle receiver profile differs")
    _require(profiles["receiver_channel_id_namespace"] == plan["receiver_channel_id_namespace"],
             "Bundle receiver frequency namespace differs")
    return data, files


def _binding(channel, freq, eta, bound):
    """The deployed word as the bundle carries it, or None while the bundle's profile is pending."""
    fine = bound["fine_calibration"]
    if fine["status"] != "calibrated":
        return None
    _require(fine["cfar_multiplier_q16"] == eta, f"channel {channel}: the bundle's word differs from the decided word")
    return {
        "physical_channel": channel, "receiver_channel_id": freq, "eta_q16": str(eta),
        "decision_version": FINE_DECISION_VERSION, "cfar_rank": fine["cfar_rank"], "anchor_bin": fine["anchor_bin"],
        "designated_half_width": fine["designated_half_width"], "bulk_mask_words_hex": fine["bulk_mask_words_hex"],
    }


def analyze_policy(plan_path):
    """Authenticate the plan's inputs and return an unactivated review record."""
    path = Path(plan_path).resolve()
    plan = calibration._read(path)
    _fields(plan, {"schema_version", "receiver_profile", "receiver_channel_id_namespace", "decisions",
                   "bundle_files", "era", "consumer"}, "operating policy plan")
    _require(plan["schema_version"] == PLAN_SCHEMA, "Unknown operating policy plan schema")
    bindings = {str(path): calibration.sha256(path)}
    receiver_path = _file(plan["receiver_profile"], path.parent, bindings)
    receiver = load_receiver_profile(receiver_path)
    channel_map = receiver.metadata.get("channel_id_map", {})
    _require(isinstance(channel_map, dict), "Receiver channel map must be an object")
    namespace = plan["receiver_channel_id_namespace"]
    _require(namespace in NAMESPACES and namespace == channel_map.get("namespace"),
             f"Consumer requires an explicit receiver namespace in {list(NAMESPACES)} matching the receiver profile")
    offset = channel_map.get("offset_from_coarse_channel_index")
    _require(type(offset) is int, "Receiver ID offset must be an exact integer")
    decisions_path = _file(plan["decisions"], path.parent, bindings)
    decisions = _decisions(decisions_path)
    bundle, files = _bundle(plan, path.parent, bindings, receiver)
    consumer = plan["consumer"]
    _fields(consumer, {"name", "version", "build_sha256", "capabilities", "evidence"}, "consumer")
    _require(consumer["name"] == CONSUMER_NAME, "Unsupported policy consumer")
    calibration._text(consumer["version"], "consumer version")
    calibration._digest(consumer["build_sha256"], "consumer build_sha256")
    caps = consumer["capabilities"]
    _require(isinstance(caps, list) and all(isinstance(c, str) for c in caps)
             and len(set(caps)) == len(caps) and REQUIRED_CAPABILITIES <= set(caps),
             "Consumer lacks required explicit operating-policy capabilities")
    _evidence(consumer["evidence"], path.parent, bindings, "consumer capability evidence")
    era = plan["era"]
    _fields(era, {"id", "scope", "evidence"}, "era")
    calibration._text(era["id"], "era id")
    calibration._text(era["scope"], "era scope")
    _evidence(era["evidence"], path.parent, bindings, "era evidence")
    profiles = {}
    for profile in bundle["pilot_profiles.json"]["profiles"]:
        profiles.setdefault(profile["physical_channel"], []).append(profile)
    project = default_project() if namespace == "chime_freq_id" else None
    permanent, unmasked, pending, thresholds, rows, outside = [], [], [], [], [], []
    review_candidates = []
    for channel, decision in sorted(decisions.items()):
        if not FIRST_CHANNEL <= channel <= LAST_CHANNEL:
            outside.append({"physical_channel": channel, "outcome": decision["outcome"],
                            "reason": "not an ATSC television channel 14-36; not part of the DTV policy"})
            continue
        bound = profiles.get(channel, [])
        _require(len(bound) == 1, f"channel {channel}: the bundle needs exactly one profile")
        bound = bound[0]
        freq = bound["receiver_channel_id"]
        calibration._integer(freq, f"channel {channel} receiver_channel_id")
        if project is not None:
            _require(freq == project.target_freq_id(str(channel)),
                     f"channel {channel}: bundle freq_id {freq} differs from the project's {project.target_freq_id(str(channel))}")
        _require(freq < CONSUMER_FREQUENCY_ID_LIMIT, "Receiver frequency ID exceeds the consumer limit")
        index = freq - offset
        _require(0 <= index < receiver.num_coarse_channels, "Receiver frequency ID is outside its grid")
        _require(bound["coarse_channel_index"] == index, "Bundle frequency mapping differs from the exact receiver grid")
        center = receiver.coarse_channel_center_hz(index)
        low = physical_channel_to_lower_edge_hz(channel)
        _require(low <= center < low + CHANNEL_WIDTH_HZ,
                 "Receiver frequency center lies outside the assigned physical channel")
        row = {"physical_channel": channel, "frequency_ids": [freq], "outcome": decision["outcome"],
               "frequency_mapping": [{"receiver_channel_id": freq, "coarse_channel_index": index, "rf_center_hz": center}]}
        if OUTCOMES[decision["outcome"]] == "reject_all" and decision["reason"] in UNMASKED_REASONS:
            row.update(threshold={"kind": "pass_all"}, reason=decision["reason"])
            unmasked.append(freq)
        elif OUTCOMES[decision["outcome"]] == "reject_all":
            row.update(threshold={"kind": "reject_all"}, reason=decision["reason"])
            permanent.append(freq)
        else:
            eta = _q16(decision["deployed_word"])
            row.update(threshold={"kind": "finite", "eta_q16": str(eta)}, deployed_at=decision.get("deployed_at"))
            binding = _binding(channel, freq, eta, bound)
            if binding is None:
                row["bundle_status"] = "pending: the bundle's profile does not carry the decided word yet"
                pending.append(freq)
            else:
                selection = bound["fine_calibration"]["provenance"].get("threshold_selection")
                if selection is not None:
                    row["bundle_status"] = "archive review candidate at the decided word"
                    row["selection_granularity"] = selection["selection_granularity"]
                    review_candidates.append(freq)
                else:
                    row["bundle_status"] = "calibrated at the decided word"
                thresholds.append(binding)
        rows.append(row)
    _require(rows, "The decisions name no ATSC channel 14-36")
    permanent.sort()
    unmasked.sort()
    pending.sort()
    thresholds.sort(key=lambda r: r["receiver_channel_id"])
    required = [r["receiver_channel_id"] for r in thresholds]
    complete = not pending and not review_candidates
    report = {
        "schema_version": POLICY_SCHEMA, "status": "review_only",
        "deployment_configuration_complete": complete,
        "activation_authorized": False, "science_certification": False,
        "consumer_capability_implementation_verified": False,
        "consumer": consumer, "era": era,
        "receiver_profile_id": receiver.receiver_profile_id,
        "receiver_profile_sha256": calibration.sha256(receiver_path),
        "receiver_channel_id_namespace": namespace,
        "decisions_sha256": calibration.sha256(decisions_path),
        "bundle_files_sha256": {name: calibration.sha256(p) for name, p in files.items()},
        "channels": rows, "not_in_policy": outside,
        "permanent_mask_freq_ids": permanent, "unmasked_freq_ids": unmasked, "require_fine_freq_ids": required,
        "pending_freq_ids": pending, "required_threshold_bindings": thresholds,
        "kotekan_parameters": {"permanent_mask_freq_ids": permanent, "require_fine_freq_ids": required} if complete else None,
        "input_files_sha256": bindings,
        "interpretation": "reject_all is the permanent-mask limit of the threshold family, explicitly encoded rather than eta=0: the threshold rule's not-deployed channels. A deployed channel's word is the threshold rule's; it is pending until the bundle carries it. A pass_all channel is not deployed and not masked: the rule's reason leaves the detector nothing to act on, so it is absent from both consumer lists. No word certifies physical recovery or minimum observing time.",
        "remaining_runtime_acceptance": [
            "Verify that the named consumer build implements every declared capability; compilation checks declarations and hashes, not implementation.",
            "Match live receiver frequency IDs, input population/state, exact bundle files, frozen threshold bindings, and era to this review record.",
            "Check full-rate fine support and validity; missing or invalid measurements must not silently become accepted finite-threshold frames.",
            "Review the explicit frequency coverage; a pilot-frequency threshold does not imply a whole-six-MHz mask.",
            "A release owner must approve activation after startup and runtime acceptance; no configuration was activated by this compiler.",
        ],
    }
    if review_candidates:
        report["candidate_configuration_complete"] = not pending
        report["review_only_candidate_freq_ids"] = sorted(review_candidates)
        report["remaining_runtime_acceptance"].insert(0,
            "Archive candidate words describe detector-frame selection only. Recompute the policy "
            "for the consumer's actual masking granularity and qualify the physical transfer, "
            "receiver conditions, and release before forming a deployment configuration.")
    for source, expected in bindings.items():
        _require(calibration.sha256(source) == expected, "Source changed during policy compilation: " + source)
    return report


def _identity():
    from pilot_proxy import fine_decision, runtime_bundle
    from pilot_proxy.integration import receiver_profile

    return {module.__name__: calibration.sha256(module.__file__)
            for module in (calibration, fine_decision, runtime_bundle, receiver_profile)}


def write_policy(plan_path, output_dir, *, require_complete=False):
    report = analyze_policy(plan_path)
    _require(not require_complete or report["deployment_configuration_complete"],
             "The configuration is incomplete: a word is pending or an archive review candidate is not qualified for deployment")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    calibration._write_json(directory / "policy.json", report)
    calibration._write_json(directory / "manifest.json", {
        "schema_version": MANIFEST_SCHEMA, "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": calibration.sha256(plan_path), "compiler_source_sha256": calibration.sha256(__file__),
        "dependency_sources_sha256": _identity(), "policy_sha256": calibration.sha256(directory / "policy.json"),
    })
    return report


def validate_policy(output_dir):
    directory = Path(output_dir).resolve()
    manifest = calibration._read(directory / "manifest.json")
    _fields(manifest, {"schema_version", "plan_path", "plan_sha256", "compiler_source_sha256",
                       "dependency_sources_sha256", "policy_sha256"}, "policy manifest")
    _require(manifest["schema_version"] == MANIFEST_SCHEMA, "Unknown policy manifest schema")
    _require(manifest["compiler_source_sha256"] == calibration.sha256(__file__)
             and manifest["dependency_sources_sha256"] == _identity(), "Policy compiler or dependencies changed")
    plan_path = Path(manifest["plan_path"])
    _require(calibration.sha256(plan_path) == manifest["plan_sha256"], "Policy plan changed")
    _require(calibration.sha256(directory / "policy.json") == manifest["policy_sha256"], "Compiled policy changed")
    report = analyze_policy(plan_path)
    _require(calibration._read(directory / "policy.json") == report, "Compiled policy does not reproduce")
    return report
