#!/usr/bin/env python3
"""Independent-dataset diagnostics for the approximate retained-proxy band.

No detector policy is changed. Analytic populations have known signed proxy
means. Resampling stored power rows instead defines an empirical population;
its independent dataset draws are not new physical waveform realizations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot_proxy.characterization.confidence_band import simultaneous_upper_band, widen_upper_band
from pilot_proxy.characterization.knee import propose
from pilot_proxy.characterization.threshold_rule import ChannelFrames, SHELF
from pilot_proxy.testbench.fine_validation_scores import construct_geometry, fixed_scores

SCHEMA = "proxy-confidence-independent-datasets-v1"
ANALYTIC = ("null_gaussian", "independent_gaussian", "coupled_unequal",
            "correlated_wrong_unit", "correlated_blocks")
CASES = ((14, 0), (14, 4), (35, 0))
REQUESTED_LEVELS = (-55.0, -50.0, -44.0)
TEMPLATE_SIZES = (1, 2, 3, 5, 2, 4, 6, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(base, name, replicate, purpose):
    value = f"{SCHEMA}:{base}:{name}:{replicate}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def template(name, q):
    coupled = name not in ("null_gaussian", "independent_gaussian")
    n = TEMPLATE_SIZES[q] if coupled else 3
    category = (q + 3 * np.arange(n)) % 8
    mean = (.2 + .12 * category + .08 * (q - 3.5) if coupled else
            np.full(n, 0.0 if name == "null_gaussian" else .3))
    return (category + 1).astype(np.uint64) * np.uint64(65536), mean


def analytic_target(name, words):
    """Exact ratio of expected equally weighted event numerator/denominator."""
    a = np.zeros(len(words))
    b = np.zeros(len(words))
    for q in range(8):
        required, means = template(name, q)
        kept = required[:, None] <= words[None, :]
        a += np.mean(means[:, None] * kept, axis=0) / 8
        b += np.mean(kept, axis=0) / 8
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


def empirical_target(population, words):
    order = np.argsort(population["required"], kind="stable")
    required = population["required"][order]
    total = np.cumsum(population["proxy"][order])
    count = np.searchsorted(required, words, side="right")
    return np.divide(total[np.maximum(count - 1, 0)], count,
                     out=np.full(len(words), np.nan), where=count > 0)


def make_frames(required, proxy, event, weight):
    n = len(required)
    return ChannelFrames(channel=14, required=np.asarray(required, np.uint64),
        excess=np.asarray(proxy, float) / SHELF, event=np.asarray(event, np.int64),
        weight=np.asarray(weight, float), supported=np.ones(n, bool),
        knee_word=0, current_word=0)


def analytic_dataset(name, seed, events):
    rng = np.random.default_rng(seed)
    categories = rng.integers(0, 8, events)
    block_noise = rng.normal(0, .6, (events + 3) // 4)
    required, proxy, event, weight = [], [], [], []
    for i, q in enumerate(categories):
        score, mean = template(name, int(q))
        error = (block_noise[i // 4] if name.startswith("correlated") else rng.normal(0, .35))
        values = mean + error + rng.normal(0, .15, len(score))
        required.extend(score)
        proxy.extend(values)
        event.extend([i] * len(score))
        weight.extend([1 / len(score)] * len(score))
    frames = make_frames(required, proxy, event, weight)
    clusters = frames.event // 4 if name == "correlated_blocks" else None
    windows = {"full": np.ones(len(event), bool), "last": frames.event >= events // 2}
    return frames, windows, clusters


def empirical_dataset(population, seed, events):
    rng = np.random.default_rng(seed)
    rows = rng.integers(len(population["required"]), size=events)
    frames = make_frames(population["required"][rows], population["proxy"][rows],
                         np.arange(events), np.ones(events))
    return frames, {"full": np.ones(events, bool), "last": frames.event >= events // 2}, None


def miss_interval(misses, trials, alpha):
    lower = 0.0 if misses == 0 else float(beta.ppf(alpha, misses, trials - misses + 1))
    upper = 1.0 if misses == trials else float(beta.ppf(1 - alpha, misses + 1, trials - misses))
    return dict(misses=int(misses), trials=int(trials), rate=misses / trials,
                one_sided_lower=lower, one_sided_upper=upper, confidence=1-alpha,
                method="Clopper-Pearson on independent dataset replicates")


def evaluate_dataset(frames, windows, clusters, target, args, seed, selection_budget):
    # This observed grid and its support eligibility are reconstructed in EVERY
    # independent dataset. It is not a fixed nonrandom threshold design.
    words = np.unique(frames.required[frames.supported])
    band = simultaneous_upper_band(frames, words, windows=windows, cluster_labels=clusters,
        confidence=.95, bootstrap_replicates=args.bootstrap_replicates, seed=seed,
        min_kept_events=30, min_effective_events=20, shelf=SHELF)
    truth = np.asarray(target(words), dtype=float)
    if truth.shape != words.shape or not np.isfinite(truth).all():
        raise ValueError("target must provide finite truth for every supplied word")
    # A held-out calibration driver may widen the margin. The standalone
    # original protocol keeps multiplier one and its original critical value.
    raw_critical = float(band["critical"])
    multiplier = getattr(args, "critical_multiplier", None)
    critical_floor = getattr(args, "minimum_critical", 1.)
    if multiplier is not None:
        band = widen_upper_band(band, multiplier, minimum_critical=critical_floor)
    required_multiplier = 0.
    for result in band["windows"].values():
        eligible = result["eligible"]
        if eligible.any():
            score = (truth[eligible] - result["signed_mean"][eligible]) / result["se"][eligible]
            required_multiplier = max(required_multiplier, float(np.max(score)) / max(raw_critical, critical_floor))
    any_miss = clipped_miss = False
    eligible_count = 0
    unavailable = band["status"] != "available"
    largest_gap = float("-inf")
    for name in windows:
        result = band["windows"][name]
        eligible = result["eligible"]
        eligible_count += int(eligible.sum())
        finite = eligible & np.isfinite(result["signed_upper"])
        if np.any(eligible & ~finite):
            unavailable = True
        gaps = truth[finite] - result["signed_upper"][finite]
        if len(gaps):
            largest_gap = max(largest_gap, float(np.max(gaps)))
            any_miss |= bool(np.any(gaps > 1e-12 * np.maximum(1e-6, np.abs(truth[finite]))))
            clipped_miss |= bool(np.any(np.maximum(truth[finite], 0) > result["upper"][finite] + 1e-15))
    knee = propose(frames.required, list(map(str, frames.event)))
    knee_word = int(knee["knee_word_q16"])
    knee_index = int(np.searchsorted(words, knee_word))
    full, last = band["windows"]["full"], band["windows"]["last"]
    eligible_both = full["eligible"] & last["eligible"]
    finite_both = eligible_both & np.isfinite(full["signed_upper"]) & np.isfinite(last["signed_upper"])
    # An explicitly diagnostic version of knee-or-loosest-usable selection.
    # The budget is fixed per target scale, not fitted to observed outcomes.
    budget = selection_budget
    candidate = np.flatnonzero(finite_both & (words <= knee_word) &
        (full["upper"] <= budget) & (last["upper"] <= budget) &
        (full["kept_weight"] / frames.weight.sum() >= .2))
    selected = int(candidate[-1]) if len(candidate) else None
    def point_miss(i):
        if i is None or i >= len(words) or not finite_both[i]:
            return None
        return bool(truth[i] > full["signed_upper"][i] + 1e-15 or
                    truth[i] > last["signed_upper"][i] + 1e-15)
    return dict(required_multiplier=required_multiplier, miss=any_miss, clipped_miss=clipped_miss, unavailable=unavailable,
        eligible_points=eligible_count, observed_words=len(words),
        knee_miss=point_miss(knee_index), selected_miss=point_miss(selected),
        selection_budget=budget,
        critical=float(band["critical"]) if np.isfinite(band["critical"]) else None,
        maximum_truth_minus_upper=largest_gap if np.isfinite(largest_gap) else None,
        grid_sha256=hashlib.sha256(words.tobytes()).hexdigest())


def load_frontier():
    relative = Path("docs/evidence/mask_residual_frontier_2026-09-07")
    config_path = ROOT / relative / "study_config.json"
    config = json.loads(config_path.read_text())
    c = config["calibration"]
    population, receipts = [], {str(relative / "study_config.json"): sha(config_path)}
    for label in ("off", "on_m55p0"):
        path = ROOT / relative / "frames" / f"{label}.npz"
        receipts[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            scores = fixed_scores(z["fine_power_u64"], construct_geometry(255))
            if not scores["valid"][:, 1].all() or scores["always_masked"][:, 1].any():
                raise ValueError("Unsupported fixture score")
            # Kernel core 3.0.0: the frame powers are the all-bin sums of the
            # stored exact fine powers. The shard's coarse_marginals_u64 and
            # normalized_excess are the retired row-domain powers and are not
            # read.
            powers = z["fine_power_u64"].sum(axis=2, dtype=np.uint64).astype(float)
            excess = powers[:, 0] * c["reference_norm_sum_sq"] / (
                (powers[:, 1] + powers[:, 2]) * c["target_norm_sq"]) - 1
            population.append(dict(required=scores["required_q16"][:, 1], proxy=SHELF*excess))
    return {k: np.concatenate([v[k] for v in population]) for k in ("required", "proxy")}, receipts


def campaign_diagnostics(campaign):
    """Nine saved cells compared with the requested model data-shelf coordinate.

    That coordinate is not measured total digital input INR. The contrast
    combines waveform, response/offset, quantization, and sampling effects; it
    does not isolate the effect of threshold selection.
    """
    plan_path = campaign / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["coordinate_system"] != "native-inverted-adapted":
        raise ValueError("Unsupported campaign coordinates")
    cache_path = Path(plan["cache_path"]) / "plan.json"
    if sha(cache_path) != plan["cache_sha256"]["plan.json"]:
        raise ValueError("Campaign cache plan checksum mismatch")
    cache = json.loads(cache_path.read_text())
    profiles = {p["physical_channel"]: p for p in cache["profiles"]}
    plan_digest = sha(plan_path)
    receipts = {str(plan_path): plan_digest, str(cache_path): sha(cache_path)}
    rows, empirical = [], None
    for channel, case in CASES:
        profile = profiles[channel]
        geometry = construct_geometry(profile["cases"][case]["calibrated_anchor_bin_inverted"])
        layout = profile["layout"]
        pieces, trial_ids, levels = [], [], None
        for start in range(0, 512, 16):
            path = campaign / "raw/evaluation" / f"ch{channel}_case{case}_M2048_{start:06d}.npz"
            receipt = json.loads(path.with_suffix(".json").read_text())
            digest = sha(path)
            if digest != receipt["sha256"]:
                raise ValueError(f"Campaign shard checksum mismatch: {path}")
            receipts[str(path.relative_to(campaign))] = digest
            meta = receipt["metadata"]
            if (meta["plan_sha256"] != plan_digest or meta["start"] != start or
                    meta["stop"] != start+16 or meta["phase"] != "evaluation" or
                    meta["channel"] != channel or meta["case_index"] != case or meta["streams"] != 2048):
                raise ValueError("Campaign shard identity mismatch")
            grid = np.asarray(meta["snr_db"])
            columns = [int(np.argmin(abs(grid-level))) for level in REQUESTED_LEVELS]
            selected_levels = grid[columns]
            if levels is not None and not np.array_equal(levels, selected_levels):
                raise ValueError("Campaign level grid changed between shards")
            levels = selected_levels
            with np.load(path, allow_pickle=False) as z:
                fine = z["fixed_fine"][:, columns]
                # Frame powers are the all-bin fine sums (kernel core 3.0.0);
                # the shard's row-domain fixed_coarse is not read.
                coarse = fine.sum(axis=-1, dtype=np.uint64).astype(float)
                scored = fixed_scores(fine.reshape(-1, 3, 256), geometry)
                if not scored["valid"][:, 1].all() or scored["always_masked"][:, 1].any():
                    raise ValueError("Unsupported campaign median score")
                required = scored["required_q16"][:, 1].reshape(16, len(columns))
                excess = coarse[..., 0] * layout["reference_norm_sum_sq"] / (
                    (coarse[..., 1] + coarse[..., 2]) * layout["target_norm_sq"]) - 1
                pieces.append((required, SHELF*excess))
            trial_ids.extend(meta["raw_seeds"])
        if len(set(trial_ids)) != 512:
            raise ValueError("Duplicate trial identity within a campaign cell")
        required = np.concatenate([v[0] for v in pieces])
        proxy = np.concatenate([v[1] for v in pieces])
        for column, level in enumerate(levels):
            records = []
            for fraction in (.25, .5, .75, 1.):
                word = int(np.quantile(required[:, column], fraction, method="higher"))
                kept = required[:, column] <= word
                x = proxy[kept, column]
                records.append(dict(requested_kept_fraction=fraction, word=word, kept=int(kept.sum()),
                    signed_proxy=float(x.mean()), standard_error=float(x.std(ddof=1)/np.sqrt(len(x))),
                    requested_model_data_shelf_coordinate=float(10**(level/10)),
                    signed_minus_requested_coordinate=float(x.mean()-10**(level/10))))
            rows.append(dict(channel=channel, case=case, case_name=profile["cases"][case]["name"],
                anchor=geometry.anchor_bin, requested_snr_db=float(level), independent_trial_ids=512,
                selected_points=records, seed_digest=hashlib.sha256(np.asarray(trial_ids,np.uint64).tobytes()).hexdigest()))
            if (channel, case, column) == (14, 0, 1):
                empirical = dict(required=required[:, column], proxy=proxy[:, column])
        print(f"reduced existing campaign channel {channel} case {case}", flush=True)
    return dict(cells=rows, source_sha256=receipts,
        scope="Existing exact powers compared with the requested model data-shelf coordinate; not measured total digital input INR and not isolated selection bias. Reused IDs pair cells and levels, never extra independent trials."), empirical


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(ANALYTIC) + ["empirical_frontier"]
    if args.campaign is not None:
        names.append("empirical_campaign")
    protocol = dict(schema=SCHEMA, dataset_replicates=args.replicates,
        bootstrap_replicates=args.bootstrap_replicates, base_seed=args.seed,
        analytic_events=320, empirical_events=160, confidence=.95,
        scenarios=names, min_kept_events=30, min_effective_events=20,
        grid="Every observed unique Q16 requirement, reconstructed with eligibility in each independent dataset",
        windows="Full and final half of the original events; shared bootstrap draws preserve overlap",
        analytic_population="Eight equally probable event templates; Gaussian zero-mean event/frame errors; exact expected event-weighted conditional proxy ratio",
        correlated_population="One Gaussian component shared by groups of four consecutive events; wrong-event and correct-block resampling are separate diagnostics",
        selection="Diagnostic knee from current equal-event rule; choose loosest eligible word no greater than knee satisfying both-window upper <= 0.75 times maximum absolute complete-population target (floor 1e-7), and full retention >= 0.2; population budget is fixed before dataset draws",
        empirical_population="IID draws with replacement from the fixed saved (score,proxy) row distribution; exact finite-population conditional mean; not new independent physical waveform trials",
        empirical_proxy="SHELF * (P0 * reference_norm_sum_sq / ((P1 + P2) * target_norm_sq) - 1) with P_r the all-bin sums of the stored fine_power_u64 (kernel core 3.0.0); the stored row-domain coarse_marginals_u64 are not read",
        campaign_cells=[list(c) for c in CASES], requested_campaign_levels=list(REQUESTED_LEVELS),
        qualification="Qualified95 only if no unavailable datasets and the one-sided family-adjusted 95% exact binomial miss upper limit is <=0.05; otherwise nominal diagnostic. No tuning after results.",
        family_size=len(names), script_sha256=sha(__file__),
        core_sha256=sha(ROOT/"src/pilot_proxy/characterization/confidence_band.py"))
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output directory contains a different frozen protocol")
    write_json(protocol_path, protocol)  # Frozen before any validation outcomes.
    print(f"frozen protocol: {sha(protocol_path)}", flush=True)
    frontier, fixture_receipts = load_frontier()
    populations = {"empirical_frontier": frontier}
    campaign_report = None
    if args.campaign is not None:
        campaign_report, populations["empirical_campaign"] = campaign_diagnostics(args.campaign)
        write_json(args.output / "campaign_diagnostic.json", campaign_report)
    all_results = {}
    for name in names:
        checkpoint = args.output / f"{name}.checkpoint.json"
        records = json.loads(checkpoint.read_text())["records"] if checkpoint.exists() else []
        if checkpoint.exists() and json.loads(checkpoint.read_text())["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Checkpoint belongs to a different protocol")
        if name in ANALYTIC:
            complete_target = analytic_target(name, np.arange(1, 9, dtype=np.uint64)*65536)
        else:
            complete_target = empirical_target(populations[name], np.unique(populations[name]["required"]))
        selection_budget = max(float(np.max(np.abs(complete_target))) * .75, 1e-7)
        for replicate in range(len(records), args.replicates):
            seed = seed_for(args.seed, name, replicate, "dataset")
            if name in ANALYTIC:
                frames, windows, clusters = analytic_dataset(name, seed, 320)
                target = lambda words, n=name: analytic_target(n, words)
            else:
                population = populations[name]
                frames, windows, clusters = empirical_dataset(population, seed, 160)
                target = lambda words, p=population: empirical_target(p, words)
            record = evaluate_dataset(frames, windows, clusters, target, args,
                seed_for(args.seed, name, replicate, "bootstrap"), selection_budget)
            records.append(record)
            if (replicate+1) % 10 == 0 or replicate+1 == args.replicates:
                write_json(checkpoint, dict(protocol_sha256=sha(protocol_path), records=records))
            if (replicate+1) % 100 == 0:
                print(f"{name}: {replicate+1}/{args.replicates}", flush=True)
        misses = sum(r["miss"] for r in records)
        unavailable = sum(r["unavailable"] for r in records)
        interval = miss_interval(misses, len(records), .05/len(names))
        result = dict(scenario=name, scope="analytic known-target population" if name in ANALYTIC else "conditional empirical-population validation",
            independent_dataset_draws=len(records), unavailable_datasets=unavailable,
            joint_signed_miss=interval, joint_clipped_misses=sum(r["clipped_miss"] for r in records),
            qualified95=unavailable == 0 and interval["one_sided_upper"] <= .05,
            qualification_scope="Conditional declared population only; correlated_wrong_unit intentionally violates the independent-cluster assumption",
            knee_available=sum(r["knee_miss"] is not None for r in records),
            knee_misses=sum(r["knee_miss"] is True for r in records),
            selected_available=sum(r["selected_miss"] is not None for r in records),
            selected_misses=sum(r["selected_miss"] is True for r in records),
            records=records)
        all_results[name] = result
        write_json(args.output / f"{name}.json", result)
        print(f"{name}: misses={misses}, unavailable={unavailable}, family upper={interval['one_sided_upper']:.4f}", flush=True)
    report = dict(schema=SCHEMA, protocol_sha256=sha(protocol_path),
        pilotproxy_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (ROOT/"src/pilot_proxy/characterization/confidence_band.py",
             ROOT/"src/pilot_proxy/characterization/threshold_rule.py",
             ROOT/"src/pilot_proxy/characterization/knee.py",
             ROOT/"src/pilot_proxy/testbench/fine_validation_scores.py", Path(__file__))},
        fixture_sha256=fixture_receipts, numpy=np.__version__,
        results={name:{k:v for k,v in result.items() if k != "records"} for name,result in all_results.items()},
        limitations=["Approximate bootstrap method; tests do not establish universal or physical-shelf coverage.",
            "Empirical populations are finite saved distributions, not new physical waveform observations.",
            "Archive event dependence and pilot-to-shelf transfer remain separate assumptions.",
            "No thresholds, decision rules, or scientific policy were changed."])
    write_json(args.output/"summary.json", report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=400)
    p.add_argument("--bootstrap-replicates", type=int, default=399)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--campaign", type=Path)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.replicates <= 0 or args.bootstrap_replicates < 19:
        raise SystemExit("Positive dataset count and at least 19 bootstrap replicates required")
    started = time.monotonic()
    run(args)
    print(f"elapsed seconds: {time.monotonic()-started:.1f}")
