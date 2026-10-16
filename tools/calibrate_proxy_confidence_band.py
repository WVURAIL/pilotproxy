#!/usr/bin/env python3
"""Calibrate a declared-model band margin, then evaluate it on fresh datasets.

This does not qualify archive dependence or physical contamination. The old
bootstrap diagnostic and production guard remain available and unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_proxy_confidence_band as validation

SCHEMA = "proxy-band-model-calibration-v1"
MATCHED = tuple(n for n in validation.ANALYTIC if n != "correlated_wrong_unit") + (
    "empirical_frontier", "empirical_campaign")
ALL = tuple(validation.ANALYTIC) + ("empirical_frontier", "empirical_campaign")
SOURCES = ("src/pilot_proxy/characterization/confidence_band.py",
    "src/pilot_proxy/characterization/threshold_rule.py", "src/pilot_proxy/characterization/knee.py",
    "src/pilot_proxy/testbench/fine_validation_scores.py", "tools/validate_proxy_confidence_band.py",
    "tools/calibrate_proxy_confidence_band.py")


def tolerance_rank(n, content, error_probability):
    """Smallest one-based rank k with P[Binomial(n, content) >= k] <= alpha.

    The k-th calibration order statistic then upper-bounds the content
    quantile with confidence at least 1-alpha (ties are conservative).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0 < content < 1 or not 0 < error_probability < 1:
        raise ValueError("content and error_probability must lie between zero and one")
    for k in range(1, n + 1):
        if binom.sf(k - 1, n, content) <= error_probability:
            return k
    raise ValueError("too few calibration datasets for a finite tolerance limit")


def calibrated_limit(scores, *, content=.975, error_probability=.05 / 6):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("all calibration scores must be finite; no failed trial may be discarded")
    rank = tolerance_rank(len(scores), content, error_probability)
    return dict(multiplier=max(1., float(np.sort(scores)[rank - 1])), rank_one_based=rank,
        calibration_datasets=len(scores), target_content=content,
        tail_probability=float(binom.sf(rank - 1, len(scores), content)),
        preceding_rank_tail=float(binom.sf(rank - 2, len(scores), content)),
        allowed_tail_probability=error_probability)


def load_populations(directory):
    receipt_path = directory / "population_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    path = directory / receipt["file"]
    if path.resolve().parent != directory.resolve() or validation.sha(path) != receipt["sha256"]:
        raise ValueError("frozen empirical population identity/checksum mismatch")
    populations = {}
    with np.load(path, allow_pickle=False) as saved:
        for label in ("frontier", "campaign"):
            required, proxy = saved[label + "_required"], saved[label + "_proxy"]
            if (required.dtype != np.dtype(np.uint64) or required.ndim != 1 or not len(required)
                    or np.any(required == 0) or proxy.shape != required.shape
                    or not np.isfinite(proxy).all()):
                raise ValueError("malformed frozen population")
            populations["empirical_" + label] = dict(required=required.copy(), proxy=proxy.copy())
    return populations, dict(population_sha256=validation.sha(path),
        receipt_sha256=validation.sha(receipt_path), receipt=receipt)


def source_hashes():
    return {name: validation.sha(ROOT / name) for name in SOURCES}


def protocol(args, populations):
    if args.calibration_seed == args.validation_seed:
        raise ValueError("calibration and validation seeds must differ")
    rank = tolerance_rank(args.calibration_datasets, .975, .05 / len(MATCHED))
    return dict(schema=SCHEMA, source_sha256=source_hashes(), base_revision=args.base_revision,
        empirical_populations=populations, matched_models=list(MATCHED), validation_models=list(ALL),
        calibration_datasets=args.calibration_datasets, validation_datasets=args.validation_datasets,
        calibration_seed=args.calibration_seed, validation_seed=args.validation_seed,
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_confidence=.95,
        analytic_events=320, empirical_events=160, support_floors=[30, 20],
        resolved_shelf_factor=validation.SHELF,
        critical_floor=1., multiplier_floor=1., calibration_content=.975,
        calibration_family_error=.05, calibration_family_size=len(MATCHED), rank_one_based=rank,
        calibration_score="max(0, max eligible words/windows (truth - signed_mean)/SE) / max(bootstrap_critical, 1)",
        applied_band="signed_mean + common_multiplier * max(bootstrap_critical, 1) * SE",
        calibration_rule="Exact one-sided order-statistic tolerance limit per matched model; common multiplier is their maximum; all datasets must be available, otherwise refuse calibration.",
        resampling="Original event weights and zero-kept events; identical event/block multiplicities across overlapping full/last windows; reconstruct observed staircase and support each dataset.",
        selection="Existing diagnostic knee/loosest-usable rule and complete-population budget from the v1 validator, unchanged.",
        validation_gate="Original family-adjusted gate: zero unavailable and Clopper-Pearson upper miss limit <=.05, tail alpha=.05/7. All six matched models must pass. Wrong-unit negative control is reported separately; its lower miss limit >.05 demonstrates failure, but outcomes never trigger retuning.",
        no_tuning="Freeze before outcomes. Calibration and validation use distinct seeds and purposes. No replacement seeds, population edits, or multiplier changes after held-out outcomes.",
        scope="Only the six declared populations and complete dataset procedure; not physical shelf power, arbitrary event dependence, new populations, or channel-wide simultaneous coverage.")


def sample(name, seed, populations):
    if name in validation.ANALYTIC:
        frames, windows, clusters = validation.analytic_dataset(name, seed, 320)
        target = lambda words: validation.analytic_target(name, words)
        complete = target(np.arange(1, 9, dtype=np.uint64) * 65536)
    else:
        pop = populations[name]
        frames, windows, clusters = validation.empirical_dataset(pop, seed, 160)
        target = lambda words: validation.empirical_target(pop, words)
        complete = target(np.unique(pop["required"]))
    budget = max(float(np.max(np.abs(complete))) * .75, 1e-7)
    return frames, windows, clusters, target, budget


def phase_seed(base, name, replicate, purpose):
    raw = f"{SCHEMA}:{base}:{name}:{replicate}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def record_identity(base_seed, name, index):
    return dict(replicate_index=index,
        dataset_seed=phase_seed(base_seed, name, index, "dataset"),
        bootstrap_seed=phase_seed(base_seed, name, index, "bootstrap"))


def verify_records(records, count, base_seed, name):
    if len(records) > count:
        raise ValueError("checkpoint has too many datasets")
    for index, record in enumerate(records):
        identity = record_identity(base_seed, name, index)
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("checkpoint record order or seed identity mismatch")


def verify_calibration(directory, frozen, protocol_hash):
    """Recompute the factor from complete, hashed calibration records."""
    calibration = json.loads((directory / "calibration.json").read_text())
    if (calibration["protocol_sha256"] != protocol_hash
            or calibration["source_sha256"] != frozen["source_sha256"]
            or calibration["bootstrap_replicates"] != frozen["bootstrap_replicates"]
            or calibration["minimum_critical"] != frozen["critical_floor"]):
        raise ValueError("calibration protocol mismatch")
    expected = {f"calibrate-{name}.json" for name in MATCHED}
    if set(calibration["calibration_result_sha256"]) != expected:
        raise ValueError("calibration model set mismatch")
    limits = {}
    for name in MATCHED:
        filename = f"calibrate-{name}.json"
        path = directory / filename
        if validation.sha(path) != calibration["calibration_result_sha256"][filename]:
            raise ValueError("calibration result checksum mismatch")
        saved = json.loads(path.read_text())
        rows = saved["records"]
        if (saved["protocol_sha256"] != protocol_hash or saved["multiplier"] is not None
                or len(rows) != frozen["calibration_datasets"]
                or any(row["unavailable"] for row in rows)):
            raise ValueError("incomplete, unavailable, or mismatched calibration records")
        verify_records(rows, frozen["calibration_datasets"], frozen["calibration_seed"], name)
        limits[name] = calibrated_limit([row["required_multiplier"] for row in rows])
    if calibration["model_limits"] != limits:
        raise ValueError("calibration limits do not reproduce from records")
    if calibration["multiplier"] != max(row["multiplier"] for row in limits.values()):
        raise ValueError("calibration factor does not match recomputed model limits")
    return calibration


def run(args):
    if args.calibration_datasets < 1 or args.validation_datasets < 1 or args.bootstrap_replicates < 19:
        raise ValueError("positive dataset counts and at least 19 bootstrap replicates required")
    populations, population_receipt = load_populations(args.populations)
    frozen = protocol(args, population_receipt)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("output has a different protocol, source, or population; use a new directory")
    validation.write_json(path, frozen)
    protocol_hash = validation.sha(path)
    runtime = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__)
    runtime_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    validation.write_json(args.output / ("runtime-" + runtime_hash[:12] + ".json"), runtime)
    calibration_path = args.output / "calibration.json"
    factor = None
    if args.phase == "validate":
        calibration = verify_calibration(args.output, frozen, protocol_hash)
        factor = calibration["multiplier"]
    names = MATCHED if args.phase == "calibrate" else ALL
    count = args.calibration_datasets if args.phase == "calibrate" else args.validation_datasets
    base_seed = args.calibration_seed if args.phase == "calibrate" else args.validation_seed
    results = {}
    for name in names:
        checkpoint = args.output / f"{args.phase}-{name}.json"
        records = []
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["protocol_sha256"] != protocol_hash or saved["multiplier"] != factor:
                raise ValueError("checkpoint identity mismatch")
            records = saved["records"]
            verify_records(records, count, base_seed, name)
        for index in range(len(records), count):
            frames, windows, clusters, target, budget = sample(name,
                phase_seed(base_seed, name, index, "dataset"), populations)
            config = SimpleNamespace(bootstrap_replicates=args.bootstrap_replicates,
                critical_multiplier=factor, minimum_critical=1.)
            record = validation.evaluate_dataset(frames, windows, clusters, target, config,
                phase_seed(base_seed, name, index, "bootstrap"), budget)
            record.update(record_identity(base_seed, name, index))
            record["runtime_sha256"] = runtime_hash
            records.append(record)
            if (index + 1) % 10 == 0 or index + 1 == count:
                validation.write_json(checkpoint, dict(protocol_sha256=protocol_hash,
                    multiplier=factor, records=records))
            if (index + 1) % 100 == 0:
                print(f"{args.phase} {name}: {index+1}/{count}", flush=True)
        results[name] = records
    if args.phase == "calibrate":
        if any(r["unavailable"] for rows in results.values() for r in rows):
            raise ValueError("calibration unavailable: no unavailable dataset may be discarded")
        limits = {name: calibrated_limit([r["required_multiplier"] for r in rows])
                  for name, rows in results.items()}
        result = dict(schema=SCHEMA, protocol_sha256=protocol_hash, source_sha256=source_hashes(),
            multiplier=max(v["multiplier"] for v in limits.values()), minimum_critical=1.,
            bootstrap_replicates=args.bootstrap_replicates, model_limits=limits,
            calibration_result_sha256={f"calibrate-{name}.json": validation.sha(args.output / f"calibrate-{name}.json") for name in names},
            scope=frozen["scope"], tolerance_content=.975, family_confidence_at_least=.95)
        validation.write_json(calibration_path, result)
        print(f"frozen model multiplier: {result['multiplier']:.8g}", flush=True)
    else:
        summaries = {}
        for name, rows in results.items():
            interval = validation.miss_interval(sum(r["miss"] for r in rows), len(rows), .05 / 7)
            unavailable = sum(r["unavailable"] for r in rows)
            summaries[name] = dict(joint_signed_miss=interval, unavailable_datasets=unavailable,
                qualified95=unavailable == 0 and interval["one_sided_upper"] <= .05,
                negative_control=name == "correlated_wrong_unit",
                assumption_failure_detected=name == "correlated_wrong_unit" and interval["one_sided_lower"] > .05,
                knee_available=sum(r["knee_miss"] is not None for r in rows),
                knee_misses=sum(r["knee_miss"] is True for r in rows),
                selected_available=sum(r["selected_miss"] is not None for r in rows),
                selected_misses=sum(r["selected_miss"] is True for r in rows))
        validation.write_json(args.output / "summary.json", dict(schema=SCHEMA,
            protocol_sha256=protocol_hash, calibration_sha256=validation.sha(calibration_path),
            multiplier=factor, results=summaries,
            all_matched_models_qualified95=all(summaries[n]["qualified95"] for n in MATCHED),
            negative_control_failure_detected=summaries["correlated_wrong_unit"]["assumption_failure_detected"],
            validation_result_sha256={f"validate-{name}.json": validation.sha(args.output / f"validate-{name}.json") for name in names},
            scope=frozen["scope"], production_guard_changed=False))
        with (args.output / "heldout_scenarios.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", "datasets", "misses", "unavailable",
                "one_sided_upper_miss", "multiplier", "qualified95", "negative_control", "assumption_failure_detected"])
            writer.writeheader()
            for name, row in summaries.items():
                writer.writerow(dict(scenario=name, datasets=row["joint_signed_miss"]["trials"],
                    misses=row["joint_signed_miss"]["misses"], unavailable=row["unavailable_datasets"],
                    one_sided_upper_miss=row["joint_signed_miss"]["one_sided_upper"], multiplier=factor,
                    qualified95=row["qualified95"], negative_control=row["negative_control"],
                    assumption_failure_detected=row["assumption_failure_detected"]))
        print(json.dumps(summaries, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["calibrate", "validate"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--populations", type=Path, required=True)
    p.add_argument("--base-revision", required=True)
    p.add_argument("--calibration-datasets", type=int, default=400)
    p.add_argument("--validation-datasets", type=int, default=400)
    p.add_argument("--bootstrap-replicates", type=int, default=399)
    p.add_argument("--calibration-seed", type=int, default=20261016031)
    p.add_argument("--validation-seed", type=int, default=20261016032)
    return p


if __name__ == "__main__":
    run(parser().parse_args())