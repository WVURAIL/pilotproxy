#!/usr/bin/env python3
"""Find how long a calibration must be, from one long detect-only recording.

    calibration_length.py --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" \
        --hours 3 6 12 24 48 --output-dir length
    calibration_length.py --bundle-dir bundle --run "..." --block-hours 4 --output-dir blocks

With --hours, each length L calibrates on the captures of the first L hours of the recording and tests the
words on the rest. The table gives the floor, each channel's word and its source, the masked fraction on the
calibration part, the failed test checks, and the share of captures to give process-run as
--calibration-share for the same split. The calibration length is the shortest L whose words still pass.
With --block-hours, consecutive blocks of that length are calibrated on their own, which shows how each
word follows the time of day.

Hours are clock hours from the first capture, read from each file's fpga_seq_time_nsec, so a recording
with interruptions keeps its times: a gap leaves its hours without captures, and a block that falls in one
is skipped. Files that record no time (the mock run) count one capture every 43 s. Each capture is one
dtv_fine_powers file of about 43 s, and a knee needs 30 of them. Writes length.json or blocks.json, with a
plan and bundle for each case, and prints the table and the gaps.

With --input-counts (from pathfinder_input_counts.py), each row also gives each channel's good inputs over
its captures (min, median and max), and the blocks end with how each channel's word follows that count.
"""
import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pilot_proxy.bundle_calibration import _rows, calibrate_bundle, test_bundle
from pilot_proxy.fine_decision import FINE_DECISION_VERSIONS
from pilot_proxy.input_counts import load_good_inputs, span_counts
from pilot_proxy.run_plans import ordered_run_files, write_run_plans

CAPTURE_SECONDS = 1024 * 8192 * 5.12e-6     # 1024 detector blocks; the same on CHIME (16384 x 2.56 us)
KNEE_CAPTURES = 30


def ordered_files(args, work):
    """Return the calibration plan template, each channel's files in time order, and the control channel."""
    with contextlib.redirect_stdout(io.StringIO()):
        plans = write_run_plans(bundle_dir=args.bundle_dir, runs=args.run, output_dir=work, epoch=args.epoch,
                                calibration_share=0.5, decision_version=args.decision_version)
    calibrate, test = plans["calibrate"], plans["test"]
    files = {ch: calibrate["channels"][ch] + test["channels"][ch] for ch in calibrate["channels"]}
    return calibrate, files, str(calibrate["control"]["channel"])


def capture_hours(runs, files):
    """Return each channel's capture start times in hours from the first capture, and the first time in ns.

    The first time is None for files that record no time, which are then spaced CAPTURE_SECONDS apart.
    """
    clock = {path: time_ns for time_ns, _, path in ordered_run_files(runs)}
    if all(clock.get(path) is not None for paths in files.values() for path in paths):
        first = min(clock[paths[0]] for paths in files.values())
        return {ch: [(clock[path] - first) / 3.6e12 for path in paths] for ch, paths in files.items()}, first
    return {ch: [i * CAPTURE_SECONDS / 3600 for i in range(len(paths))] for ch, paths in files.items()}, None


def gaps(hours, capture=CAPTURE_SECONDS / 3600):
    """Return (start, length) in hours of each interruption longer than two captures."""
    return [(a + capture, b - a - capture) for a, b in zip(hours, hours[1:]) if b - a > 2 * capture]


def select(files, hours, start, stop):
    """Return each channel's files whose captures start in [start, stop) hours."""
    return {ch: [path for path, h in zip(paths, hours[ch]) if start <= h < stop] for ch, paths in files.items()}


def input_ranges(hours, first_ns, counts, freq_of, start, stop):
    """Return each channel's (min, median, max) good inputs over its captures in [start, stop) hours."""
    out = {}
    for ch, values in hours.items():
        if freq_of.get(ch) not in counts:
            continue
        times, good = counts[freq_of[ch]]
        spans = [span_counts(times, good, int(first_ns + h * 3.6e12), int(first_ns + h * 3.6e12 + CAPTURE_SECONDS * 1e9))
                 for h in values if start <= h < stop]
        spans = [s for s in spans if s]
        if spans:
            out[ch] = [min(s[0] for s in spans), int(np.median([s[1] for s in spans])), max(s[2] for s in spans)]
    return out


def with_inputs(channels, inputs):
    """Add each channel's good-input range to its summary."""
    return {ch: dict(c, inputs=inputs[ch]) if ch in inputs else c for ch, c in channels.items()}


def describe(channels):
    return ", ".join(f"ch{ch} {c['eta']} ({c['source'].split(',')[0]}, {100 * c['masked']:.1f}%"
                     + (f", {c['inputs'][0]}-{c['inputs'][2]} inputs" if "inputs" in c else "") + ")"
                     for ch, c in channels.items())


def follow(rows):
    """Print how each channel's word follows its median good inputs over the blocks."""
    for ch in sorted({ch for r in rows for ch in r.get("channels", {})}, key=int):
        pairs = [(r["channels"][ch]["inputs"][1], r["channels"][ch]["eta"]) for r in rows
                 if ch in r.get("channels", {}) and "inputs" in r["channels"][ch]]
        if len(pairs) < 3:
            continue
        inputs, eta = np.array(pairs, dtype=float).T
        if np.ptp(inputs) == 0 or np.ptp(eta) == 0:
            print(f"ch{ch}: inputs {int(inputs.min())}-{int(inputs.max())}, eta {eta.min():g}-{eta.max():g}; one of them constant")
            continue
        r = float(np.corrcoef(inputs, eta)[0, 1])
        print(f"ch{ch}: inputs {int(inputs.min())}-{int(inputs.max())}, eta {eta.min():g}-{eta.max():g}, "
              f"correlation {r:+.2f} over {len(pairs)} blocks")


def calibrate_on(template, chosen, control, out):
    """Calibrate on the chosen captures of every channel, and return the report."""
    plan = dict(template, channels=chosen, control=dict(channel=int(control), captures=chosen[control]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate.json").write_text(json.dumps(plan, indent=1) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return calibrate_bundle(plan_path=out / "calibrate.json", output_dir=out / "bundle")


def test_on(template, chosen, control, out):
    """Test the bundle in out on the chosen captures, and return the test record."""
    plan = dict(capture_format=template["capture_format"], channels=chosen, control=chosen[control])
    (out / "test.json").write_text(json.dumps(plan, indent=1) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return test_bundle(bundle_dir=out / "bundle", plan_path=out / "test.json", output_json=out / "test.record.json")


def summary(report):
    return {ch: dict(eta=round(c["eta"], 4), source=c["word_source"], anchor=c["anchor_bin"],
                     masked=round(c["masked_fraction"], 4)) for ch, c in report["channels"].items()}


def utc(first_ns, hours):
    return None if first_ns is None else datetime.fromtimestamp(first_ns / 1e9 + hours * 3600, timezone.utc).isoformat(timespec="minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    parser.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hours", type=float, nargs="+", help="Calibration lengths to try, each tested on the rest.")
    mode.add_argument("--block-hours", type=float, help="Calibrate consecutive blocks of this length on their own.")
    parser.add_argument("--decision-version", choices=FINE_DECISION_VERSIONS, default=None)
    parser.add_argument("--epoch", default="calibration length study")
    parser.add_argument("--input-counts", type=Path, help="Good-input counts from pathfinder_input_counts.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")

    template, files, control = ordered_files(args, args.output_dir / "all")
    hours, first_ns = capture_hours(args.run, files)
    counts, freq_of = {}, {}
    if args.input_counts:
        if first_ns is None:
            raise SystemExit("the run files record no times, so they cannot be matched to the input counts")
        counts = load_good_inputs(args.input_counts)
        bundle = json.loads((args.bundle_dir / "pilot_profiles.json").read_text())
        freq_of = {str(ch): int(row["receiver_channel_id"]) for ch, row in _rows(bundle).items()}
    n = min(len(paths) for paths in files.values())
    span = max(h[-1] for h in hours.values()) + CAPTURE_SECONDS / 3600
    print(f"{n} captures per channel ({n * CAPTURE_SECONDS / 3600:.2f} h of data) over {span:.2f} h"
          + ("" if first_ns is None else f" from {utc(first_ns, 0)}"))
    interruptions = gaps(hours[control])
    for start, length in interruptions:
        print(f"gap of {length:.2f} h from {start:.2f} h" + ("" if first_ns is None else f" ({utc(first_ns, start)})"))
    rows = []
    if args.hours:
        for length in sorted(args.hours):
            chosen, rest = select(files, hours, 0, length), select(files, hours, length, float("inf"))
            k, m = min(len(v) for v in chosen.values()), min(len(v) for v in rest.values())
            # the test needs as many captures as a knee, or its masked fractions say little
            if k < KNEE_CAPTURES or m < KNEE_CAPTURES:
                print(f"{length:g} h: {k} captures to calibrate and {m} to test, each needs {KNEE_CAPTURES}; skipped")
                continue
            out = args.output_dir / f"{length:g}h"
            report = calibrate_on(template, chosen, control, out)
            record = test_on(template, rest, control, out)
            failed = [c["check"] + (f": {c['detail']}" if c.get("detail") else "")
                      for c in record["checks"] if not c["passed"]]
            share = round(len(chosen[control]) / len(files[control]), 4)
            inputs = input_ranges(hours, first_ns, counts, freq_of, 0, length)
            rows.append(dict(hours=length, captures=k, test_captures=m, calibration_share=share,
                             floor_eta=round(report["floor_eta"], 4), channels=with_inputs(summary(report), inputs),
                             test_passed=record["passed"], failed=failed))
            line = describe(rows[-1]["channels"])
            print(f"{length:g} h ({k} captures, share {share}): floor {rows[-1]['floor_eta']}; {line}; "
                  f"test {'passed' if record['passed'] else 'FAILED: ' + '; '.join(failed)}")
        name = "length.json"
    else:
        size = args.block_hours
        if round(size * 3600 / CAPTURE_SECONDS) < KNEE_CAPTURES:
            raise SystemExit(f"a block of {size:g} h has fewer than {KNEE_CAPTURES} captures, which a knee needs")
        for b in range(int(span // size + 1e-9)):
            start = b * size
            chosen = select(files, hours, start, start + size)
            k = min(len(v) for v in chosen.values())
            row = dict(block=b, start_hours=round(start, 3), start_utc=utc(first_ns, start), captures=k)
            if k < KNEE_CAPTURES:
                rows.append(dict(row, skipped=f"{k} captures, a knee needs {KNEE_CAPTURES}"))
                print(f"block {b} from {start:g} h: {k} captures; skipped")
                continue
            report = calibrate_on(template, chosen, control, args.output_dir / f"block{b}")
            inputs = input_ranges(hours, first_ns, counts, freq_of, start, start + size)
            rows.append(dict(row, floor_eta=round(report["floor_eta"], 4),
                             channels=with_inputs(summary(report), inputs)))
            print(f"block {b} from {start:g} h" + ("" if first_ns is None else f" ({row['start_utc']})")
                  + f": floor {rows[-1]['floor_eta']}; {describe(rows[-1]['channels'])}")
        if counts:
            follow(rows)
        name = "blocks.json"
    (args.output_dir / name).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
