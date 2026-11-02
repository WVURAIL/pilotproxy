"""Select maximum-retention thresholds from a unit-power response table.

The channel table supplies the simple model a = power_scale*x**2.
--power-limits supplies a JSON channel-to-limit mapping in mK^2
from finite-model checks. Results are in-sample candidate thresholds.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pilot_proxy.characterization import threshold_rule as rule

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--channels", help="comma-separated channel numbers")
    ap.add_argument("--power-limits", type=Path, required=True,
                    help="JSON object mapping channel numbers to checked maximum additive powers in mK^2")
    ap.add_argument("--exclude", action="append", default=[], metavar="CH=REASON")
    args = ap.parse_args(argv)
    if (args.out / "decisions.json").exists():
        ap.error("decisions.json already exists; use a new output directory")
    table = rule.load_table(args.table)
    channels = [row["channel"] for row in table["channels"]]
    if args.channels:
        selected = [int(c) for c in args.channels.split(",")]
        if set(selected) - set(channels):
            ap.error("unknown channel")
        channels = selected
    # Channel 37 is a background control, not a DTV operating channel.
    channels = [ch for ch in channels if ch != 37]
    limits = {int(k): float(v) for k, v in json.loads(args.power_limits.read_text()).items()}
    if any(ch not in limits or not np.isfinite(limits[ch]) or limits[ch] <= 0 for ch in channels):
        ap.error("every requested channel needs a finite positive checked power limit")
    excluded = {int(k): v for k, v in (s.split("=", 1) for s in args.exclude)}
    decisions = []
    sweeps = {}
    for channel in channels:
        frames = rule.load_frames(channel, args.scores)
        s = rule.sweep(frames)
        R = rule.response(table, channel)
        d = rule.decide(s, R, power_scale=rule.power_scale(table, channel), power_limit=limits[channel], exclude=excluded.get(channel))
        point = d.pop("deployed")
        decisions.append(dict(channel=channel, **d, **(point or {}), response=R, power_scale=rule.power_scale(table, channel)))
        sweeps[channel] = s
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweeps").mkdir(exist_ok=True)
    for channel, s in sweeps.items():
        np.savez_compressed(args.out / "sweeps" / f"ch{channel}.npz", **s)
    result = dict(schema="pilotproxy-power-decisions-v3",
                  inputs=dict(scores=str(args.scores.resolve()), normalization="flat-temperature-variance-v1",
                              power_units="mK^2", power_limits={str(ch): limits[ch] for ch in channels}, frame_samples=rule.FRAME_SAMPLES,
                              table_provenance=table["provenance"], table_schema=table["schema"],
                              selection="maximum retention subject to upper linear bias <= 1 and upper power <= checked limit",
                              evaluation="in-sample; independent evaluation required",
                              guard_standard_errors=rule.GUARD_STANDARD_ERRORS),
                  channels=decisions)
    (args.out / "decisions.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = ["channel", "outcome", "reason", "deployed_word", "kappa", "a", "a_hi", "power_limit", "r", "r_hi",
               "time_years", "kept_frames", "kept_events"]
    with (args.out / "decisions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(decisions)
    return result

if __name__ == "__main__":
    main()
