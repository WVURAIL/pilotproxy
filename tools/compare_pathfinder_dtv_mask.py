#!/usr/bin/env python3
"""Compare a Pathfinder run's visibilities with the DTV mask on and off.

    compare_pathfinder_dtv_mask.py --dtv-dir /mnt/cs00/data/dtv/<acq> \
        --vis-dir /mnt/cs00/data/kotekan_vis_files/subset/acq_<...> --output-dir compare

For a run in which the detector and recording stay on and /dtv_mask_enabled is switched on and
off (pathfinder_dtv_switch.py). Each visibility bin is labelled by frac_dtv_mask_enabled: on (1),
off (0), or switched inside the bin (left out). For each frequency the detector masks at least once,
and for the pilot frequencies, the two states are compared on:

  detector  the fraction of the bin's blocks that the detector masked, from the dtv_mask files, or for
            a fine_decision_v3 channel of --bundle-dir from its dtv_stat rows at the word N2 applies.
            It should not depend on the state.
  excised   frac_rfi, the fraction of the bin that the correlator excised. With the mask on it
            must be at least the share of the bin that the detector masked, and it rises one for
            one with that share (slope 1). With the mask off it does not follow it (slope 0).
  power     the median autocorrelation over the good inputs, in bins where the detector masked
            under 5% of the bin and in bins where it masked over half, and its slope against the
            masked share. With the mask off, the masked blocks add their power to the bin, so it
            rises with the share. With it on, they are excised, so it stays flat. A bin excised
            whole holds no data and has no power.

Writes compare.json and, with matplotlib, one figure per frequency: the power and the fractions
against time, and the power against the masked share in each state. The exit status is 1 when a
check fails or a frequency could not be compared. Needs numpy, h5py and hdf5plugin.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter of the recv files)
import numpy as np

from pilot_proxy.fine_decision import masks_from_stat
from pilot_proxy.run_files import iter_windows, v3_words

BLOCK_SAMPLES = 8192
FREQ_WIDTH_MHZ = 0.1953125
# freq_id: ATSC channel
PILOTS = {2531: 18, 2715: 24, 2869: 29, 3115: 37}
ON, OFF = 0.999, 0.001
MIN_COVERAGE = 0.9
QUIET, BUSY = 0.05, 0.5


def atsc_channel(freq_id):
    """Return the UHF channel (14 to 37) whose 6 MHz band holds the frequency, or None."""
    mhz = freq_id * FREQ_WIDTH_MHZ
    return 14 + int((mhz - 470.0) // 6.0) if 470.0 <= mhz < 614.0 else None


def read_masks(dtv_dir, words=None):
    """Return {freq_id: (block start sequences, mask)} from the dtv_mask files, per-window or combined, in time order.

    A freq_id in `words` (a fine_decision_v3 channel) is decided from its dtv_stat rows at its word instead.
    """
    words = words or {}
    parts = defaultdict(list)
    for product in ("mask", "stat") if words else ("mask",):
        for path in sorted(glob.glob(os.path.join(dtv_dir, "**", f"dtv_{product}.*.h5"), recursive=True)):
            for attrs, data in iter_windows(path):
                seqs = int(attrs["fpga_seq_num"]) + BLOCK_SAMPLES * np.arange(data.shape[0], dtype=np.int64)
                for column, freq_id in enumerate(int(c) for c in attrs["coarse_freq"]):
                    if (freq_id in words) != (product == "stat"):
                        continue
                    rows = data[:, column]
                    parts[freq_id].append((seqs, masks_from_stat(rows, words[freq_id]) if product == "stat" else rows))
    masks = {}
    for freq_id, chunks in parts.items():
        seqs = np.concatenate([c[0] for c in chunks])
        order = np.argsort(seqs, kind="stable")
        seqs, rows = seqs[order], np.concatenate([c[1] for c in chunks])[order]
        # a block read twice, from a per-window file and a combined file of the same window, counts once
        first = np.r_[True, seqs[1:] != seqs[:-1]]
        masks[freq_id] = (seqs[first], rows[first])
    return masks


def read_vis(vis_dir, freq_ids):
    """Return {freq_id: per-bin arrays} over every visibility file, in time order."""
    rows = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(vis_dir, "vis_*.h5"))):
        with h5py.File(path, "r") as handle:
            ids = np.rint(handle["index_map/freq"]["centre"] / FREQ_WIDTH_MHZ).astype(int)
            n = int(handle.attrs["num_elements"])
            autos = [i * n - i * (i - 1) // 2 for i in range(n)]
            prod = handle["index_map/prod"][autos]
            if not (np.all(prod["input_a"] == np.arange(n)) and np.all(prod["input_b"] == np.arange(n))):
                raise ValueError(f"{path}: products are not in FullUpperTri order")
            start = handle["fpga_start_tick"][()].astype(np.int64)
            length = handle["frame_length_fpga_ticks"][()].astype(np.int64)
            time_ns = handle["time_center_t_inst_ns"][()]
            for freq_id in freq_ids:
                hit = np.flatnonzero(ids == freq_id)
                if not hit.size:
                    continue
                f = int(hit[0])
                good = handle["flags"][f] > 0
                auto = handle["vis"][f, autos, :].real
                power = np.full(auto.shape[1], np.nan)
                for t in range(auto.shape[1]):
                    if good[:, t].any():
                        power[t] = np.median(auto[good[:, t], t])
                row = rows[freq_id]
                row["start"].append(start)
                row["length"].append(length)
                row["time_ns"].append(time_ns)
                row["added"].append(handle["frames_added"][f].astype(bool))
                row["power"].append(power)
                for name in ("frac_dtv_mask_enabled", "frac_rfi"):
                    row[name].append(handle[name][f].astype(np.float64))
    out = {}
    for freq_id, row in rows.items():
        arrays = {k: np.concatenate(v) for k, v in row.items()}
        # frames_added marks a received bin. frac_lost also counts excision, so a bin the mask
        # removed whole has frac_lost 1 and is kept, with no power.
        keep = arrays["added"] & (arrays["length"] > 0)
        # a bin excised whole holds no data
        arrays["power"][arrays["frac_rfi"] >= 0.99] = np.nan
        order = np.argsort(arrays["start"][keep], kind="stable")
        out[freq_id] = {k: v[keep][order] for k, v in arrays.items()}
    return out


def detector_fraction(seqs, mask, start, length):
    """Return, for each bin, the masked fraction of its present blocks, the share of the bin they
    cover, and the share of the bin that the detector masked."""
    lo = np.searchsorted(seqs, start)
    hi = np.searchsorted(seqs, start + length - BLOCK_SAMPLES, side="right")
    masked = np.full(start.size, np.nan)
    coverage = np.zeros(start.size)
    share = np.zeros(start.size)
    for t in range(start.size):
        block = mask[lo[t]:hi[t]]
        present = np.count_nonzero(block != -1)
        coverage[t] = present * BLOCK_SAMPLES / length[t]
        share[t] = np.count_nonzero(block == 1) * BLOCK_SAMPLES / length[t]
        if present:
            masked[t] = np.count_nonzero(block == 1) / present
    return masked, coverage, share


def summarize(bins, detector, share, sel, tol):
    """Return the comparison numbers of one state."""
    def mean(x):
        return round(float(np.mean(x)), 5) if x.size else None

    def median(x):
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else None

    def slope(x, y, use):
        use = use & np.isfinite(y)
        if np.count_nonzero(use) < 3 or np.ptp(x[use]) == 0:
            return None
        return round(float(np.polyfit(x[use], y[use], 1)[0]), 4)

    hit = sel & (detector > 0)
    excised, power = bins["frac_rfi"], bins["power"]
    return dict(
        bins=int(sel.sum()), bins_with_detector_masks=int(hit.sum()),
        detector_masked=mean(detector[sel]), excised=mean(excised[sel]),
        # excised per detector-masked share of the bin: 1 when the mask is applied, 0 when it is not
        excised_slope=slope(share, excised, sel),
        excised_below_detector=int(np.count_nonzero(excised[hit] < share[hit] - tol[hit])),
        power_detector_under_5_percent=median(power[sel & (share < QUIET)]),
        power_detector_over_50_percent=median(power[sel & (share >= BUSY)]),
        power_slope=slope(share, power, sel))


def compare(freq_id, bins, seqs, mask):
    detector, coverage, share = detector_fraction(seqs, mask, bins["start"], bins["length"])
    enabled = bins["frac_dtv_mask_enabled"]
    usable = np.isfinite(detector) & (coverage >= MIN_COVERAGE)
    on = usable & (enabled >= ON)
    off = usable & (enabled <= OFF)
    # a block cut by a bin edge is not counted, so allow two blocks of slack
    tol = 2.0 * BLOCK_SAMPLES / bins["length"]
    report = dict(atsc_channel=atsc_channel(freq_id), pilot=freq_id in PILOTS,
                  bins=int(bins["start"].size), bins_without_detector_cover=int(np.count_nonzero(~usable)),
                  bins_switched=int(np.count_nonzero(usable & ~on & ~off)),
                  mask_on=summarize(bins, detector, share, on, tol),
                  mask_off=summarize(bins, detector, share, off, tol))
    on_report, off_report = report["mask_on"], report["mask_off"]
    report["checks"] = {
        "mask on excises every block the detector masked":
            on_report["excised_below_detector"] == 0 if on_report["bins_with_detector_masks"] else None,
        "excised follows the detector with the mask on (slope above 0.5)":
            on_report["excised_slope"] > 0.5 if on_report["excised_slope"] is not None else None,
        "excised ignores the detector with the mask off (slope below 0.5)":
            off_report["excised_slope"] < 0.5 if off_report["excised_slope"] is not None else None,
    }
    return report, dict(detector=detector, share=share, on=on, off=off, covered=coverage >= MIN_COVERAGE)


def plot(freq_id, bins, flags, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # only the bins the dtv_mask files cover; the visibility acquisition may run much longer
    covered = flags["covered"]
    if not covered.any():
        return
    bins = {k: v[covered] for k, v in bins.items()}
    flags = {k: v[covered] for k, v in flags.items()}
    minutes = (bins["time_ns"] - bins["time_ns"][0]) / 60e9
    fig = plt.figure(figsize=(13, 5.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(2.2, 1))
    top = fig.add_subplot(grid[0, 0])
    bottom = fig.add_subplot(grid[1, 0], sharex=top)
    side = fig.add_subplot(grid[:, 1], sharey=top)
    for sel, label, color in ((flags["off"], "mask off", "#1f77b4"), (flags["on"], "mask on", "#ff7f0e")):
        top.plot(minutes[sel], bins["power"][sel], "o", ms=3, color=color, label=label)
        side.plot(flags["share"][sel], bins["power"][sel], "o", ms=3, color=color, label=label)
    other = ~flags["on"] & ~flags["off"]
    top.plot(minutes[other], bins["power"][other], "o", ms=3, color="0.6", label="left out")
    top.set_ylabel("median autocorrelation")
    top.legend(loc="best", fontsize=8)
    side.set(xlabel="share of the bin the detector masked", title="power against the detector's share")
    side.legend(loc="best", fontsize=8)
    bottom.plot(minutes, flags["share"], color="#2ca02c", label="detector masked")
    bottom.plot(minutes, bins["frac_rfi"], color="#d62728", label="excised (frac_rfi)")
    bottom.plot(minutes, bins["frac_dtv_mask_enabled"], color="0.4", lw=0.8, ls="--", label="mask enabled")
    bottom.set_ylabel("fraction of bin")
    bottom.set_xlabel("minutes from the first bin")
    bottom.legend(loc="best", fontsize=8)
    channel = atsc_channel(freq_id)
    fig.suptitle(f"freq_id {freq_id}" + (f", channel {channel}" if channel else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtv-dir", required=True, help="Directory of dtv_mask.*.h5 files, searched recursively.")
    parser.add_argument("--vis-dir", required=True, help="Visibility acquisition of the run.")
    parser.add_argument("--freq-ids", type=int, nargs="+",
                        help="Default: every freq_id the detector masks at least once, and the pilots.")
    parser.add_argument("--bundle-dir", help="Bundle of the run: its fine_decision_v3 channels are read from dtv_stat.")
    parser.add_argument("--mask-control", action="store_true",
                        help="The run set dtv_mask_control, so the v3 control is decided like a pilot.")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    words = v3_words(args.bundle_dir, mask_control=args.mask_control)
    masks = read_masks(args.dtv_dir, words)
    if not masks:
        raise SystemExit(f"no dtv_mask files in {args.dtv_dir}")
    # the bundle's v3 channels are compared too, so one recorded without dtv_stat is reported, not left out
    freq_ids = args.freq_ids or sorted({f for f, (_, m) in masks.items() if np.any(m == 1)}
                                       | (set(PILOTS) & set(masks)) | set(words))
    vis = read_vis(args.vis_dir, freq_ids)
    reports = {}
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    for freq_id in freq_ids:
        if freq_id not in vis or freq_id not in masks:
            missing = "dtv_stat" if freq_id in words else "dtv_mask"
            reports[str(freq_id)] = dict(error="no visibilities" if freq_id not in vis else f"no {missing} rows")
            continue
        report, flags = compare(freq_id, vis[freq_id], *masks[freq_id])
        reports[str(freq_id)] = report
        if args.output_dir:
            try:
                plot(freq_id, vis[freq_id], flags, os.path.join(args.output_dir, f"compare_{freq_id}.png"))
            except ImportError:
                pass
        line = [f"{freq_id} ch{report['atsc_channel']}:"]
        for state in ("mask_on", "mask_off"):
            s = report[state]
            line.append(f"{state} {s['bins']} bins, detector {s['detector_masked']}, excised {s['excised']} "
                        f"(slope {s['excised_slope']}), power {s['power_detector_under_5_percent']} quiet / "
                        f"{s['power_detector_over_50_percent']} busy (slope {s['power_slope']});")
        print(" ".join(line), report["checks"])
    if args.output_dir:
        with open(os.path.join(args.output_dir, "compare.json"), "w") as out:
            json.dump(dict(dtv_dir=args.dtv_dir, vis_dir=args.vis_dir, frequencies=reports), out, indent=1)
    # a check with no bins to test is None, which is not a failure
    failed = sorted(f for f, r in reports.items() if "error" in r or False in r["checks"].values())
    if failed:
        sys.exit(f"failed or not compared: freq_ids {', '.join(failed)}")


if __name__ == "__main__":
    main()
