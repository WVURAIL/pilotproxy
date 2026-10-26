#!/usr/bin/env python3
"""Check the threshold code against the frozen sweeps of a companion paper.

The manuscript and its records are only read, and no raw baseband data or GPU
is needed. The code is checked to reproduce the paper's saved calculations and
ideal-year assignments. The check does not cover physical qualification of the
thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from pilot_proxy.characterization import threshold_rule as rule


def check(paper_root: Path) -> dict:
    data = paper_root / "data"
    record_path = data / "decisions.json"
    record = json.loads(record_path.read_text())
    inputs = record["inputs"]
    # The manuscript keeps the duty-based JSON of the earlier rule and builds its
    # primary calendar comparison in calendar_forecast. Both views are checked.
    sys.path.insert(0, str(paper_root.resolve()))
    try:
        manuscript = importlib.import_module("make_numbers")
        calendar = importlib.import_module("calendar_forecast")
        for module in (manuscript, calendar):
            if Path(module.__file__).resolve().parent != paper_root.resolve():
                raise ValueError("paper modules were imported from another checkout")
        decisions, table_record, rows, sweeps, _ = manuscript.load()
        _, primary_rows, _ = calendar.comparison_view(decisions, table_record, rows, sweeps)
    finally:
        sys.path.pop(0)
    table = {row["channel"]: row for row in rule.load_table()["channels"]}
    np.testing.assert_allclose(rule.STACK_AMPLITUDE_RATIO, inputs["coherent_fraction"], rtol=1e-13)
    np.testing.assert_allclose(rule.SHELF, inputs["shelf"], rtol=1e-13)
    results = []
    for saved in record["channels"]:
        channel = saved["channel"]
        with np.load(data / "sweeps" / f"ch{channel}.npz", allow_pickle=False) as archive:
            full = dict(archive)
        with np.load(data / "sweeps_last" / f"ch{channel}.npz", allow_pickle=False) as archive:
            recent = dict(archive)
        response = abs(table[channel]["response_dv"])
        gain = rule.temporal_gain(channel)
        np.testing.assert_allclose(response, saved["response"], rtol=1e-13)
        np.testing.assert_allclose(gain, saved["gain"], rtol=1e-13)
        arguments = dict(
            gain=gain, budget=saved.get("tolerance_budget", 1.0),
            max_years=inputs["max_calendar_years"],
            exclude=inputs["excluded"].get(str(channel)),
        )
        historical = rule.decide_guarded(
            full, saved["knee_word"], recent, response,
            calendar_factor=inputs["calendar_years_per_retained_year"], **arguments,
        )
        ideal = rule.decide_guarded(full, saved["knee_word"], recent, response, **arguments)
        for key in ("outcome", "reason", "deployed_word", "deployed_at", "full_era_word", "guard_acted"):
            if historical[key] != saved[key]:
                raise ValueError(f"channel {channel}: historical {key} differs from the paper")
        if historical["last_year_r_hi"] is None:
            if saved["last_year_r_hi"] is not None:
                raise ValueError(f"channel {channel}: historical recent-year support differs")
        else:
            np.testing.assert_allclose(historical["last_year_r_hi"], saved["last_year_r_hi"], rtol=5e-12)
        point = historical["deployed"] or historical["knee"]
        for key in ("kappa", "time_years", "r", "r_hi"):
            np.testing.assert_allclose(point[key], saved[key], rtol=5e-12, atol=0,
                                       err_msg=f"channel {channel}: {key}")
        for key in ("outcome", "reason", "deployed_word", "deployed_at"):
            if ideal[key] != primary_rows[channel][key]:
                raise ValueError(f"channel {channel}: ideal-year {key} differs from the paper")
        chosen = ideal["deployed"]
        if chosen is not None:
            np.testing.assert_allclose(chosen["time_years"], 1 / chosen["kappa"], rtol=1e-15)
        results.append(dict(channel=channel, word=ideal["deployed_word"],
                            outcome=ideal["outcome"],
                            reason=ideal["reason"],
                            ideal_calendar_years=chosen["time_years"] if chosen else None))
    return dict(status="pass", decisions_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
                channels_checked=len(results),
                scope="Saved calculation and ideal-year assignments; not physical qualification",
                channels=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.paper_root), indent=2))


if __name__ == "__main__":
    main()
