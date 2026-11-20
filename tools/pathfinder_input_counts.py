#!/usr/bin/env python3
"""Count the inputs the Pathfinder detector summed, from the bad feed mask in the visibility files.

    pathfinder_input_counts.py --vis "/mnt/cs00/data/kotekan_vis_files/subset/acq_*/vis_*.h5" \
        "/mnt/cs00/data/kotekan_vis_files/subset/acq_*/.partial/vis_*.h5" --output inputs.npz

The bad-input mask changes as feeds are flagged, so the good-input count of each pilot channel and the
control is a time series. This reads it once from the visibility files (only their small bad_feed_mask
group), writes it to an .npz for calibration_length.py --input-counts, and prints each channel's range
and every change with its UTC time. Default frequencies: the Pathfinder pilots and control (2531, 2715,
2869, 3115).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from pilot_proxy.input_counts import changes, read_good_inputs, save_good_inputs


def utc(time_ns):
    return datetime.fromtimestamp(time_ns / 1e9, timezone.utc).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vis", nargs="+", required=True, help="Visibility files or glob patterns.")
    parser.add_argument("--freq-ids", type=int, nargs="+", default=[2531, 2715, 2869, 3115])
    parser.add_argument("--output", type=Path, required=True, help="The .npz to write.")
    args = parser.parse_args()
    counts = read_good_inputs(args.vis, args.freq_ids)
    if not counts:
        raise SystemExit("no bad_feed_mask rows for these frequencies in the visibility files")
    save_good_inputs(counts, args.output)
    for freq_id, (times, good) in sorted(counts.items()):
        print(f"{freq_id}: {len(good)} frames from {utc(times[0])} to {utc(times[-1])}, "
              f"good inputs {good.min()} to {good.max()}")
        for time_ns, before, after in changes(times, good):
            print(f"  {utc(time_ns)}: {before} -> {after}")
    missing = sorted(set(args.freq_ids) - set(counts))
    if missing:
        print("no mask rows for", missing)


if __name__ == "__main__":
    main()
