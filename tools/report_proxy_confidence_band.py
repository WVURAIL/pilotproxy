#!/usr/bin/env python3
"""Reproduce exploratory joint proxy bands on frozen archive score products.

The output is a statistical diagnostic of the event-weighted shelf proxy. The
telescope policy and the accepted detector words are unchanged. This report
does not cover a confidence bound on physical DTV power, since the physical
transfer uncertainty is not modeled. The full arrays are kept in the output
directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pilot_proxy.characterization import threshold_rule as rule
from pilot_proxy.characterization.confidence_band import simultaneous_upper_band, widen_upper_band


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze_channel(channel, scores, output, frozen_inputs=None):
    """Freeze the analysis arrays of one channel in the output, since the score paths can change."""
    if frozen_inputs is None:
        frames = rule.load_frames(channel, scores)
        meta = json.loads((scores / f"ch{channel}.json").read_text())
        provenance = {
            "score_npz": str(scores / f"ch{channel}.npz"),
            "score_sha256": digest(scores / f"ch{channel}.npz"),
            "score_metadata_sha256": digest(scores / f"ch{channel}.json"),
            "original_product_identity_claimed": meta["original_product_identity"],
            "original_archive_sha256_verified": False,
            "verification_scope": "score files and exact frozen analysis arrays hashed; original archive size checked only",
        }
    else:
        frozen_path = frozen_inputs / f"ch{channel}.npz"
        provenance = json.loads((frozen_inputs / f"ch{channel}.json").read_text())
        if digest(frozen_path) != provenance["sha256"]:
            raise ValueError(f"frozen analysis input checksum mismatch: {frozen_path}")
        with np.load(frozen_path, allow_pickle=False) as saved:
            frames = rule.ChannelFrames(channel=channel, **{key: saved[key] for key in
                ("required", "weight", "supported", "event", "excess", "time")},
                knee_word=int(saved["knee_word"]), current_word=int(saved["current_word"]))
    destination = output / "inputs" / f"ch{channel}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: getattr(frames, key) for key in
              ("required", "weight", "supported", "event", "excess", "time")}
    arrays.update(knee_word=np.uint64(frames.knee_word), current_word=np.uint64(frames.current_word))
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[key], value, equal_nan=True) for key, value in arrays.items()):
                raise ValueError(f"frozen input changed: {destination}")
    else:
        np.savez_compressed(destination, **arrays)
    provenance.update({
        "path": str(destination), "sha256": digest(destination),
        "frames": len(frames.required), "events": len(np.unique(frames.event)),
    })
    write_json(destination.with_suffix(".json"), provenance)
    return frames, provenance


def event_time_clusters(frames, days):
    """Return UTC block labels with each observing event assigned whole by its earliest frame."""
    _, event = np.unique(frames.event, return_inverse=True)
    first = np.full(event.max() + 1, np.inf)
    np.minimum.at(first, event, frames.time)
    if not np.isfinite(first).all():
        raise ValueError("temporal block sensitivity requires finite event times")
    return np.floor(first[event] / (86400 * days)).astype(np.int64)


def curve_for_rule(frames, words, mask, window):
    # The band is substituted into the rule's sweep fields for a sensitivity
    # calculation only. The accepted detector words are unchanged, and the
    # band keeps its proxy scale. se holds half the upper margin, so the
    # rule's x + 2 se reproduces the band's upper end.
    upper = np.asarray(window["upper"])
    eligible = np.asarray(window["eligible"], dtype=bool)
    x = np.maximum(np.asarray(window["signed_mean"]), 0)
    return dict(word=words, kappa=np.asarray(window["kept_weight"]) / frames.weight[mask].sum(),
                x=x, se=np.where(eligible, np.maximum(upper - x, 0) / 2, np.inf),
                kept_frames=window["kept_frames"], kept_events=window["kept_events"])


def at_word(words, window, word):
    i = int(np.searchsorted(words, np.uint64(word), side="right")) - 1
    if i < 0:
        return {"word": word, "status": "below_grid"}
    keys = ("signed_mean", "se", "eligible", "status", "signed_upper", "upper",
            "kept_frames", "kept_events", "effective_events")
    return dict(word=word, evaluated_word=int(words[i]), **{
        key: np.asarray(window[key])[i] for key in keys if key in window})


def load_model_calibration(path, replicates):
    """Verify and return a declared-model calibration, which is applied to the archive as a diagnostic only."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate_proxy_confidence_band import source_hashes, verify_calibration

    protocol_path = path.parent / "protocol.json"
    frozen = json.loads(protocol_path.read_text())
    if path.name != "calibration.json":
        raise ValueError("model calibration must be named calibration.json")
    if frozen["source_sha256"] != source_hashes():
        raise ValueError("model calibration source mismatch")
    if frozen["bootstrap_replicates"] != replicates:
        raise ValueError("model calibration bootstrap replicate count mismatch")
    calibration = verify_calibration(path.parent, frozen, digest(protocol_path))
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scores", type=Path)
    inputs.add_argument("--frozen-inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-decisions", type=Path)
    parser.add_argument("--channels", nargs="+", type=int, default=list(range(14, 38)))
    parser.add_argument("--replicates", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=2026101601)
    parser.add_argument("--modes", nargs="+", choices=["event", "day", "week"],
                        default=["event", "day", "week"])
    parser.add_argument("--model-calibration", type=Path,
                        help="Verified declared-model calibration; archive use remains diagnostic")
    args = parser.parse_args()
    calibration = (None if args.model_calibration is None else
                   load_model_calibration(args.model_calibration, args.replicates))
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "schema": "retained-proxy-archive-band-v1", "confidence": .95,
        "bootstrap_replicates": args.replicates, "seed": args.seed,
        "min_kept_clusters": 30, "min_effective_clusters": 20.,
        "grid_rule": "all distinct supported observed Q16 requirements; rebuild in coverage replications",
        "windows": ["full_era", "last_year"], "modes": args.modes,
        "channels": args.channels,
        "model_calibration": None if calibration is None else {
            "sha256": digest(args.model_calibration), "multiplier": calibration["multiplier"],
            "minimum_critical": calibration["minimum_critical"], "scope": calibration["scope"],
            "archive_qualification": "none; declared model assumptions have not been established for this archive"},
        "target": "event-weighted conditional mean signed shelf proxy within the frozen era and profiles",
        "interpretation": "nominal approximate simultaneous empirical-staircase bootstrap diagnostic",
        "family": "per channel, joint over eligible threshold points and both overlapping windows",
        "temporal_blocks": "UTC bins anchored at Unix epoch; whole events assigned by earliest frame; original weights preserved",
        "adoption": "none; existing sampling guard and all accepted detector words retained",
        "physical_transfer_uncertainty_included": False,
        "cross_channel_simultaneous_claim": False,
        "published_decisions_sha256": digest(args.published_decisions) if args.published_decisions else None,
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != settings:
        raise ValueError("protocol differs; use a separate output directory")
    write_json(protocol_path, settings)
    implementation = Path(__import__(simultaneous_upper_band.__module__, fromlist=["x"]).__file__)
    code = {"implementation": str(implementation), "implementation_sha256": digest(implementation),
            "driver_sha256": digest(__file__), "threshold_rule_sha256": digest(rule.__file__)}
    table = rule.load_table()
    measurement_dir = Path(rule.__file__).parent / "data"
    code["policy_inputs"] = {name: digest(measurement_dir / name) for name in
                            ("tolerance_table.json", "coherent_fraction.json", "coherent_gain.json")}
    code["resolved_constants"] = {key: getattr(rule, key) for key in
                                  ("SHELF", "STACK_AMPLITUDE_RATIO", "GAIN_FRAMES", "FRAME_SAMPLES",
                                   "FRAME_SECONDS", "GUARD_STANDARD_ERRORS", "MAX_CALENDAR_YEARS",
                                   "CALENDAR_YEARS_PER_RETAINED_YEAR", "LAST_YEAR_DAYS")}
    published = {} if args.published_decisions is None else {
        row["channel"]: row for row in json.loads(args.published_decisions.read_text())["channels"]}
    results = []
    for ch in args.channels:
        frames, source = freeze_channel(ch, args.scores, args.output, args.frozen_inputs)
        if frames.time is None or not np.isfinite(frames.time).all():
            raise ValueError(f"nonfinite frame time on channel {ch}")
        words = np.unique(frames.required[frames.supported])
        windows = {"full_era": np.ones(len(frames.required), dtype=bool),
                   "last_year": frames.time >= frames.time.max() - rule.LAST_YEAR_DAYS * 86400}
        exclusion = "proxy invalid" if ch in (19, 33) else "radio astronomy control" if ch == 37 else None
        old = rule.decide_guarded(rule.sweep(frames), frames.knee_word, rule.sweep(rule.last_year(frames)),
                                  rule.response(table, ch), gain=rule.temporal_gain(ch), exclude=exclusion)
        if published and ch != 37 and old["deployed_word"] != published[ch]["deployed_word"]:
            raise ValueError(f"original guard word does not reproduce the published decision on channel {ch}")
        for mode in args.modes:
            report_path = args.output / f"ch{ch}_{mode}.json"
            arrays_path = args.output / f"ch{ch}_{mode}.npz"
            identity = dict(code=code, protocol_sha256=digest(protocol_path), input_sha256=source["sha256"],
                            channel=ch, mode=mode)
            if report_path.exists():
                saved = json.loads(report_path.read_text())
                if saved.get("identity") == identity and saved.get("arrays_sha256") == digest(arrays_path):
                    results.append(saved)
                    print(f"channel {ch} {mode}: reused verified result", flush=True)
                    continue
                raise ValueError(f"result identity changed: {report_path}; use a separate output directory")
            clusters = None if mode == "event" else event_time_clusters(frames, 1 if mode == "day" else 7)
            band = simultaneous_upper_band(
                frames, words, windows=windows, cluster_labels=clusters, confidence=.95,
                bootstrap_replicates=args.replicates, seed=args.seed + ch * 10 + args.modes.index(mode),
                min_kept_events=30, min_effective_events=20., batch_size=16)
            if calibration is not None:
                band = widen_upper_band(band, calibration["multiplier"],
                                        minimum_critical=calibration["minimum_critical"])
            # The band means are compared with the accepted sweep in each
            # window. The last-year window has its own denominator and keeps
            # the full-era event weights.
            for name, mask in windows.items():
                selected = rule.ChannelFrames(
                    channel=ch, required=frames.required[mask], weight=frames.weight[mask],
                    supported=frames.supported[mask], event=frames.event[mask], excess=frames.excess[mask],
                    knee_word=frames.knee_word, current_word=frames.current_word)
                reference = rule.sweep(selected)
                idx = np.searchsorted(reference["word"], words, side="right") - 1
                present = idx >= 0
                got = band["windows"][name]
                np.testing.assert_allclose(np.asarray(got["signed_mean"])[present],
                                           rule.SHELF * reference["mean_excess"][idx[present]],
                                           rtol=1e-9, atol=1e-12)
                if mode == "event":
                    # Only eligible points are compared. simultaneous_upper_band
                    # refuses the cancellation-dominated zero-support moments
                    # that rule.sweep rounds to zero.
                    check = present & np.asarray(got["eligible"])
                    np.testing.assert_allclose(np.asarray(got["se"])[check],
                                               reference["se"][idx[check]], rtol=1e-4, atol=1e-10)
            arrays = {"words": words}
            for name, window in band["windows"].items():
                for key, value in window.items():
                    if isinstance(value, np.ndarray):
                        arrays[f"{name}__{key}"] = value
            for key, value in band.items():
                if isinstance(value, np.ndarray):
                    arrays[key] = value
            np.savez_compressed(arrays_path, **arrays)
            modified = {name: curve_for_rule(frames, words, mask, band["windows"][name])
                        for name, mask in windows.items()}
            sensitivity = rule.decide_guarded(modified["full_era"], frames.knee_word, modified["last_year"],
                                              rule.response(table, ch), gain=rule.temporal_gain(ch), exclude=exclusion)
            points = {}
            for name, window in band["windows"].items():
                points[name] = {"knee": at_word(words, window, frames.knee_word)}
                if old["deployed_word"] is not None:
                    points[name]["original_guard_word"] = at_word(words, window, old["deployed_word"])
            report = {"identity": identity, "source": source, "channel": ch, "mode": mode,
                      "band": {key: value for key, value in band.items()
                               if key != "windows" and not isinstance(value, np.ndarray)},
                      "points": points, "guard_decision": old,
                      "published_dtv_word_verified": bool(published and ch != 37),
                      "proxy_band_substitution_diagnostic": sensitivity,
                      "substitution_se_field": "encoded half-margin for the unchanged rule API; not a standard error",
                      "arrays_path": str(arrays_path), "arrays_sha256": digest(arrays_path),
                      "grid_sha256": hashlib.sha256(words.tobytes()).hexdigest(), "grid_points": len(words)}
            write_json(report_path, report)
            results.append(clean(report))
            print(f"channel {ch} {mode}: {band.get('status')}, critical={band.get('critical_value', band.get('critical'))}",
                  flush=True)
    write_json(args.output / "summary.json", dict(protocol=settings, code=code, results=results))


if __name__ == "__main__":
    main()
