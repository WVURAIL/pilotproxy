#!/usr/bin/env python3
"""Derive a sealed campaign plan for a supplementary channel from the record campaign plan.

The record campaign covers channels 14-36. A supplementary channel, such as the channel-37 control, has its own cache
built by prepare_fine_validation_cache.py --channels, whose case indices continue the record order, so
validate_fine_detector.engine_for loads them unchanged. Its plan keeps every protocol value of the record (counts,
seeds, gate, shelf coordinate, input powers) and replaces the channel list, the cache binding and the weight bank.

The record grids come from the campaign's discovery stage, which a supplementary channel did not run. Its levels are
the whole lattice that every record grid samples, 0.75 dB steps over the clip range of the grid rule, so a bracketed
crossing is the one a record grid would give.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from validate_fine_detector import load_cache, seal_marker, sha, verify_marker  # noqa: E402
from prepare_fine_validation_cache import CASES_PER_CHANNEL, RECORD_CHANNELS, case_index  # noqa: E402
from pilot_proxy.detector_weights import DetectorWeightBank  # noqa: E402

LEVEL_STEP_DB = 0.75
LEVEL_RANGE_DB = (-66.0, -21.0)


def lattice():
    count = round((LEVEL_RANGE_DB[1] - LEVEL_RANGE_DB[0]) / LEVEL_STEP_DB) + 1
    return [LEVEL_RANGE_DB[0] + LEVEL_STEP_DB * i for i in range(count)]


def derive(record_path, cache, weights):
    record = json.loads(Path(record_path).read_text())
    record_sha = verify_marker(record_path)
    cache_plan = json.loads((cache / "plan.json").read_text())
    channels = [p["physical_channel"] for p in cache_plan["profiles"]]
    if set(channels) & set(RECORD_CHANNELS):
        raise ValueError("a supplementary cache must not repeat a record channel")
    if sha(weights) != cache_plan["weights_sha256"]:
        raise ValueError("the weights differ from the weights of the cache")
    bank = DetectorWeightBank(explicit_path=str(weights))
    files = {"plan.json": sha(cache / "plan.json")}
    cases = []
    for profile in cache_plan["profiles"]:
        channel = profile["physical_channel"]
        packed, valid = bank.get_weights_for_physical_channel(channel)
        if not valid or profile["packed_profile_sha256"] != hashlib.sha256(np.asarray(packed).tobytes()).hexdigest():
            raise ValueError(f"the weights of channel {channel} differ from the cache profile")
        for c, case in enumerate(profile["cases"]):
            for p, payload in enumerate(cache_plan["payloads"]):
                index = case_index(channel, c, p)
                receipt, _, _ = load_cache(cache, index)
                if (receipt["plan_sha256"] != files["plan.json"] or receipt["passed"] is not True
                        or receipt["profile_channel"] != channel or receipt["case"] != case
                        or receipt["payload"] != payload):
                    raise ValueError(f"cache case {index} differs from its plan or failed its gates")
                path = next((cache / "cases").glob(f"{index:04d}_*.json"))
                files[str(path.relative_to(cache))] = sha(path)
                files[str(path.with_suffix(".npz").relative_to(cache))] = receipt["array_sha256"]
                cases.append({"index": index, "channel": channel, "payload": payload["index"], "case": case})
    if len(cases) != CASES_PER_CHANNEL * len(channels):
        raise ValueError("the cache does not cover every case of its channels")
    plan = dict(record)
    plan.update(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        channels=channels,
        cache_path=str(cache.resolve()),
        cache_cases=cases,
        cache_sha256=files,
        weights_path=str(Path(weights).resolve()),
        source_sha256={**record["source_sha256"], str(Path(weights).resolve()): sha(weights)},
        supplement={
            "record_plan": str(Path(record_path).resolve()),
            "record_plan_sha256": record_sha,
            "kept": "counts, seeds and identities, false-alarm gate, shelf coordinate, input powers, coordinate system",
            "replaced": "channels, cache_path, cache_cases, cache_sha256, weights_path and its source digest",
            "evaluation_levels_db": lattice(),
            "evaluation_levels_rule": "the whole 0.75 dB lattice over the clip range of evaluation_grid_rule; every "
                                      "record grid samples this lattice, so a bracketed crossing is unchanged",
        },
    )
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record-plan", type=Path, required=True, help="sealed record campaign plan (plan.json)")
    parser.add_argument("--cache", type=Path, required=True, help="supplementary cache directory")
    parser.add_argument("--weights", type=Path, required=True, help="weight bank of the supplementary cache")
    parser.add_argument("--output", type=Path, required=True, help="new plan.json; its .sha256 marker is written")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("the output plan already exists")
    plan = derive(args.record_plan, args.cache, args.weights)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    seal_marker(args.output)
    print(json.dumps({"plan_sha256": sha(args.output), "channels": plan["channels"], "cases": len(plan["cache_cases"])}))


if __name__ == "__main__":
    main()
