#!/usr/bin/env python3
"""Check a Pathfinder DTV run from the recv node's files.

    check_pathfinder_dtv_run.py --dtv-dir /mnt/cs00/data/dtv --bundle-dir <bundle> \
        --vis-dir /mnt/cs00/data/kotekan_vis_files/subset/acq_<...> --output-dir check

For each channel: windows and gaps, missing blocks, valid tests, masked percent, the peak
fine bin against the bundle anchor, and the visibilities' frac_dtv_mask_enabled. With
matplotlib and the fine powers, it also plots the target fine powers in chunks, over all
blocks and over the kept blocks. The masked percent of every channel by UTC hour, from the
window times recv writes (fpga_seq_time_nsec), goes to masked_by_hour.csv and a figure.
Reads per-window or combined files (combine_pathfinder_dtv_run.py). Needs numpy, h5py, hdf5plugin and pilot-proxy.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)
import numpy as np

from pilot_proxy.run_files import iter_windows, read_attrs, windows

SAMPLE_NS = 5120
BLOCK_SAMPLES = 8192
FREQ_WIDTH_MHZ = 0.1953125
# freq_id: ATSC channel
PILOTS = {2531: 18, 2715: 24, 2869: 29, 3115: 37}


def product_files(dtv_dir, product):
    """The per-window and combined files of one product under dtv_dir."""
    return sorted(glob.glob(os.path.join(dtv_dir, "**", f"dtv_{product}.*.h5"), recursive=True))


def collect(dtv_dir, product, channels):
    """Return {channel: {window seq: per-block array}} for one product."""
    found = defaultdict(dict)
    for path in product_files(dtv_dir, product):
        for attrs, data in iter_windows(path):
            coarse = [int(c) for c in attrs.get("coarse_freq", [])]
            for channel in channels:
                if channel in coarse:
                    found[channel][int(attrs["fpga_seq_num"])] = data[:, coarse.index(channel)]
    return found


def window_times(dtv_dir):
    """Return {window seq: UNIX ns of its first block} from the dtv_mask files that record it."""
    times = {}
    for path in product_files(dtv_dir, "mask"):
        for spec in windows(path):
            attrs = read_attrs(spec)
            if "fpga_seq_time_nsec" in attrs:
                times[int(attrs["fpga_seq_num"])] = int(attrs["fpga_seq_time_nsec"])
    return times


def masked_by_hour(masks, times):
    """Return {utc hour: (masked blocks, present blocks)} for one channel."""
    counts = defaultdict(lambda: [0, 0])
    for seq, mask in masks.items():
        if seq not in times:
            continue
        hours = ((times[seq] + np.arange(mask.size) * BLOCK_SAMPLES * SAMPLE_NS) // 3_600_000_000_000) % 24
        present = mask != -1
        for hour in np.unique(hours[present]):
            at = present & (hours == hour)
            counts[int(hour)][0] += int(np.count_nonzero(mask[at] == 1))
            counts[int(hour)][1] += int(np.count_nonzero(at))
    return {hour: tuple(value) for hour, value in sorted(counts.items())}


def plot_by_hour(by_hour, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for channel, counts in by_hour.items():
        if counts:
            hours = list(counts)
            ax.plot(hours, [100.0 * m / p for m, p in counts.values()], "o-", ms=4,
                    label=f"freq_id {channel} (ATSC {PILOTS.get(channel, '?')})")
    ax.set(xlabel="UTC hour", ylabel="masked percent", xlim=(-0.5, 23.5), ylim=(-2, 102), xticks=range(0, 24, 2))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def anchors(bundle_dir):
    if not bundle_dir:
        return {}
    bundle = json.load(open(os.path.join(bundle_dir, "pilot_profiles.json")))
    return {int(p["receiver_channel_id"]): int(p["fine_calibration"]["anchor_bin"]) for p in bundle["profiles"]}


def vis_fractions(vis_dir, channels):
    """Return the mean frac_dtv_mask_enabled of each channel over the visibility files."""
    sums, counts = defaultdict(float), defaultdict(int)
    for path in sorted(glob.glob(os.path.join(vis_dir, "vis_*.h5"))):
        with h5py.File(path, "r") as handle:
            ids = np.rint(handle["index_map/freq"]["centre"] / FREQ_WIDTH_MHZ).astype(int)
            frac = handle["frac_dtv_mask_enabled"][()]
        for channel in channels:
            rows = np.flatnonzero(ids == channel)
            if rows.size:
                sums[channel] += float(frac[rows].sum())
                counts[channel] += frac[rows].size
    return {c: sums[c] / counts[c] for c in counts}


def check_channel(channel, masks, supports, fines, anchor, chunk_blocks):
    seqs = sorted(masks)
    report = dict(channel=channel, atsc=PILOTS.get(channel), windows=len(seqs))
    if not seqs:
        return report, None
    window = len(masks[seqs[0]]) * BLOCK_SAMPLES
    steps = np.diff(seqs)
    report.update(minutes=len(seqs) * window * SAMPLE_NS / 6e10,
                  gaps=int(np.count_nonzero(steps != window)),
                  missing_windows=int(np.sum(np.maximum(steps // window - 1, 0))))
    mask = np.concatenate([masks[s] for s in seqs])
    support = np.concatenate([supports[s][:, 0] for s in seqs if s in supports]) if supports else np.array([])
    present = mask != -1
    report.update(blocks=int(mask.size), missing_blocks=int(np.count_nonzero(~present)),
                  masked_percent=round(100.0 * np.count_nonzero(mask == 1) / max(1, present.sum()), 3))
    if support.size:
        report.update(valid_test_percent=round(100.0 * np.count_nonzero(support == 1) / max(1, present.sum()), 3),
                      invalid_tests=int(np.count_nonzero(support == 0)))

    if not fines:
        return report, None
    target, keep = [], []
    for seq in sorted(fines):
        rows = fines[seq][:, 0].astype(np.float64)
        ok = fines[seq].any(axis=(1, 2)) & ~(fines[seq] == np.iinfo(np.uint64).max).all(axis=(1, 2))
        target.append(rows[ok])
        keep.append((masks[seq] == 0)[ok] if seq in masks else np.zeros(ok.sum(), bool))
    target, keep = np.concatenate(target), np.concatenate(keep)
    if not target.size:
        return report, None
    mean = target.mean(axis=0)
    mean_kept = target[keep].mean(axis=0) if keep.any() else np.full(target.shape[1], np.nan)
    peak = int(np.argmax(mean))
    # near 1 on a channel with no pilot, where the peak bin is noise
    report.update(peak_bin=peak, peak_over_median=round(float(mean[peak] / np.median(mean)), 2), anchor_bin=anchor,
                  peak_minus_anchor=None if anchor is None else peak - anchor,
                  kept_peak_over_median=round(float(mean_kept[peak] / np.median(mean_kept)), 2) if keep.any() else None)
    chunks = range(0, len(target), chunk_blocks)
    every = np.stack([target[i:i + chunk_blocks].mean(axis=0) for i in chunks])
    kept = np.stack([target[i:i + chunk_blocks][keep[i:i + chunk_blocks]].mean(axis=0)
                     if keep[i:i + chunk_blocks].any() else np.full(target.shape[1], np.nan) for i in chunks])
    return report, (every, kept, mean, mean_kept)


def plot(channel, spectra, anchor, chunk_seconds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    every, kept, mean, mean_kept = spectra
    span = [0, every.shape[1], every.shape[0] * chunk_seconds / 60, 0]
    low, high = np.nanpercentile(np.log10(every[every > 0]), [1, 99.5])
    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1))
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0]))
    for ax, data, title in zip(axes, (every, kept), ("all blocks (no mask)", "kept blocks (mask applied)")):
        with np.errstate(divide="ignore"):
            ax.imshow(np.log10(data), aspect="auto", extent=span, vmin=low, vmax=high, interpolation="nearest")
        if anchor is not None:
            ax.plot(anchor + 0.5, 0, "rv", clip_on=False)
        ax.set(title=title, xlabel="fine bin")
    axes[0].set_ylabel("minutes")
    # the time average of each: what the mask removes from the spectrum
    average = fig.add_subplot(grid[1, :])
    bins = np.arange(mean.size) + 0.5
    average.semilogy(bins, mean, color="#1f77b4", label="all blocks (no mask)")
    average.semilogy(bins, mean_kept, color="#ff7f0e", label="kept blocks (mask applied)")
    if anchor is not None:
        average.axvline(anchor + 0.5, color="r", lw=0.8, ls=":", label="bundle anchor")
    average.set(xlim=(0, mean.size), xlabel="fine bin", ylabel="mean target fine power")
    average.legend(loc="best", fontsize=8)
    fig.suptitle(f"freq_id {channel} (ATSC {PILOTS.get(channel, '?')}), log10 target fine power")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtv-dir", required=True, help="Directory of dtv_*.h5 files, searched recursively.")
    parser.add_argument("--channels", type=int, nargs="+", default=list(PILOTS), help="freq_ids to check.")
    parser.add_argument("--bundle-dir", help="Bundle of the run, for the anchor bins.")
    parser.add_argument("--vis-dir", help="Visibility acquisition of the run.")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    masks = collect(args.dtv_dir, "mask", args.channels)
    supports = collect(args.dtv_dir, "fine_support", args.channels)
    fines = collect(args.dtv_dir, "fine_powers", args.channels)
    anchor = anchors(args.bundle_dir)
    fractions = vis_fractions(args.vis_dir, args.channels) if args.vis_dir else {}
    block_seconds = BLOCK_SAMPLES * SAMPLE_NS * 1e-9
    chunk_blocks = max(1, round(args.chunk_seconds / block_seconds))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    reports = []
    for channel in args.channels:
        report, spectra = check_channel(channel, masks.get(channel, {}), supports.get(channel, {}),
                                        fines.get(channel, {}), anchor.get(channel), chunk_blocks)
        if channel in fractions:
            report["frac_dtv_mask_enabled"] = round(fractions[channel], 4)
        reports.append(report)
        print(json.dumps(report))
        if spectra is not None and args.output_dir:
            try:
                plot(channel, spectra, anchor.get(channel), chunk_blocks * block_seconds,
                     os.path.join(args.output_dir, f"spectrogram_{channel}.png"))
            except ImportError:
                pass
    times = window_times(args.dtv_dir)
    by_hour = {channel: masked_by_hour(masks.get(channel, {}), times) for channel in args.channels}
    for report in reports:
        counts = by_hour[report["channel"]]
        if counts:
            report["masked_percent_by_utc_hour"] = {h: round(100.0 * m / p, 3) for h, (m, p) in counts.items()}
    if args.output_dir:
        with open(os.path.join(args.output_dir, "check.json"), "w") as out:
            json.dump(reports, out, indent=1)
        with open(os.path.join(args.output_dir, "masked_by_hour.csv"), "w") as out:
            out.write("freq_id,atsc_channel,utc_hour,blocks,masked_percent\n")
            for channel, counts in by_hour.items():
                for hour, (masked, present) in counts.items():
                    out.write(f"{channel},{PILOTS.get(channel, '')},{hour},{present},{100.0 * masked / present:.3f}\n")
        if any(by_hour.values()):
            try:
                plot_by_hour(by_hour, os.path.join(args.output_dir, "masked_by_hour.png"))
            except ImportError:
                pass


if __name__ == "__main__":
    main()
