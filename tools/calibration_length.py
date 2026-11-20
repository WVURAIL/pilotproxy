#!/usr/bin/env python3
"""Find how long a calibration must be, from one long detect-only recording.

    calibration_length.py --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" \
        --hours 3 6 12 24 48 --output-dir length
    calibration_length.py --bundle-dir bundle --run "..." --block-hours 4 --output-dir blocks

With --hours, each length L calibrates on the first L hours of the recording and tests the words on the
rest, as process-run does with the matching --calibration-share. The table gives the floor, each
channel's word and its source, the masked fraction on the calibration part, and the failed test checks.
The calibration length is the shortest L whose words still pass. With --block-hours, consecutive blocks
of that length are calibrated on their own, which shows how each word follows the time of day.

Each capture is one dtv_fine_powers file of about 43 s, and a knee needs 30 of them. Writes length.json
or blocks.json, with a plan and bundle for each case, and prints the table.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

from pilot_proxy.bundle_calibration import calibrate_bundle, test_bundle
from pilot_proxy.fine_decision import FINE_DECISION_VERSIONS
from pilot_proxy.run_plans import write_run_plans

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


def calibrate_on(template, files, control, index, out):
    """Calibrate on the captures at index of every channel, and return the report."""
    plan = dict(template, channels={ch: [paths[i] for i in index] for ch, paths in files.items()},
                control=dict(channel=int(control), captures=[files[control][i] for i in index]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate.json").write_text(json.dumps(plan, indent=1) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return calibrate_bundle(plan_path=out / "calibrate.json", output_dir=out / "bundle")


def test_on(template, files, control, index, out):
    """Test the bundle in out on the captures at index, and return the test record."""
    plan = dict(capture_format=template["capture_format"],
                channels={ch: [paths[i] for i in index] for ch, paths in files.items()},
                control=[files[control][i] for i in index])
    (out / "test.json").write_text(json.dumps(plan, indent=1) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return test_bundle(bundle_dir=out / "bundle", plan_path=out / "test.json", output_json=out / "test.record.json")


def summary(report):
    return {ch: dict(eta=round(c["eta"], 4), source=c["word_source"], anchor=c["anchor_bin"],
                     masked=round(c["masked_fraction"], 4)) for ch, c in report["channels"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    parser.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hours", type=float, nargs="+", help="Calibration lengths to try, each tested on the rest.")
    mode.add_argument("--block-hours", type=float, help="Calibrate consecutive blocks of this length on their own.")
    parser.add_argument("--decision-version", choices=FINE_DECISION_VERSIONS, default=None)
    parser.add_argument("--epoch", default="calibration length study")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")

    template, files, control = ordered_files(args, args.output_dir / "all")
    n = min(len(paths) for paths in files.values())
    print(f"{n} captures per channel, {n * CAPTURE_SECONDS / 3600:.2f} h")
    rows = []
    if args.hours:
        for hours in sorted(args.hours):
            k = round(hours * 3600 / CAPTURE_SECONDS)
            # the test needs as many captures as a knee, or its masked fractions say little
            if k < KNEE_CAPTURES or n - k < KNEE_CAPTURES:
                print(f"{hours:g} h: {k} captures, needs {KNEE_CAPTURES} to {n - KNEE_CAPTURES}; skipped")
                continue
            out = args.output_dir / f"{hours:g}h"
            report = calibrate_on(template, files, control, range(k), out)
            record = test_on(template, files, control, range(k, n), out)
            failed = [c["check"] + (f": {c['detail']}" if c.get("detail") else "")
                      for c in record["checks"] if not c["passed"]]
            rows.append(dict(hours=hours, captures=k, floor_eta=round(report["floor_eta"], 4),
                             channels=summary(report), test_passed=record["passed"], failed=failed))
            line = ", ".join(f"ch{ch} {c['eta']} ({c['source'].split(',')[0]}, {100 * c['masked']:.1f}%)"
                             for ch, c in rows[-1]["channels"].items())
            print(f"{hours:g} h ({k} captures): floor {rows[-1]['floor_eta']}; {line}; "
                  f"test {'passed' if record['passed'] else 'FAILED: ' + '; '.join(failed)}")
        name = "length.json"
    else:
        k = round(args.block_hours * 3600 / CAPTURE_SECONDS)
        if k < KNEE_CAPTURES:
            raise SystemExit(f"a block of {args.block_hours:g} h has {k} captures; a knee needs {KNEE_CAPTURES}")
        for b, start in enumerate(range(0, n - k + 1, k)):
            report = calibrate_on(template, files, control, range(start, start + k), args.output_dir / f"block{b}")
            rows.append(dict(block=b, start_hours=round(start * CAPTURE_SECONDS / 3600, 3), captures=k,
                             floor_eta=round(report["floor_eta"], 4), channels=summary(report)))
            line = ", ".join(f"ch{ch} {c['eta']} ({c['source'].split(',')[0]}, {100 * c['masked']:.1f}%)"
                             for ch, c in rows[-1]["channels"].items())
            print(f"block {b} from {rows[-1]['start_hours']} h: floor {rows[-1]['floor_eta']}; {line}")
        name = "blocks.json"
    (args.output_dir / name).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
