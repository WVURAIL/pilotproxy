#!/usr/bin/env python3
"""Check a Pathfinder DTV run from the recv node's files.

    check_pathfinder_dtv_run.py --dtv-dir /mnt/cs00/data/dtv --bundle-dir <bundle> \
        --vis-dir /mnt/cs00/data/kotekan_vis_files/subset/acq_<...> --output-dir check

For each channel: windows and gaps, missing blocks, valid tests, masked percent, the peak
fine bin against the bundle anchor, and the visibilities' frac_dtv_mask_enabled. With
matplotlib and the fine powers, it also plots the target fine powers in chunks, over all
blocks and over the kept blocks. Needs numpy, h5py and hdf5plugin.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import h5py
import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)
import numpy as np

SAMPLE_NS = 5120
BLOCK_SAMPLES = 8192
FREQ_WIDTH_MHZ = 0.1953125
# freq_id: ATSC channel
PILOTS = {2531: 18, 2715: 24, 2869: 29, 3115: 37}


def read(path):
    with h5py.File(path, "r") as handle:
        data = handle[next(iter(handle))]
        coarse = [int(c) for c in data.attrs.get("coarse_freq", [])]
        return int(data.attrs["fpga_seq_num"]), coarse, data[()]


def collect(dtv_dir, product, channels):
    """Return {channel: {window seq: per-block array}} for one product."""
    found = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(dtv_dir, "**", f"dtv_{product}.*.h5"), recursive=True)):
        seq, coarse, data = read(path)
        for channel in channels:
            if channel in coarse:
                found[channel][seq] = data[:, coarse.index(channel)]
    return found


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
    peak = int(np.argmax(target.mean(axis=0)))
    report.update(peak_bin=peak, anchor_bin=anchor, peak_minus_anchor=None if anchor is None else peak - anchor)
    chunks = range(0, len(target), chunk_blocks)
    every = np.stack([target[i:i + chunk_blocks].mean(axis=0) for i in chunks])
    kept = np.stack([target[i:i + chunk_blocks][keep[i:i + chunk_blocks]].mean(axis=0)
                     if keep[i:i + chunk_blocks].any() else np.full(target.shape[1], np.nan) for i in chunks])
    return report, (every, kept)


def plot(channel, spectra, anchor, chunk_seconds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    every, kept = spectra
    span = [0, every.shape[1], every.shape[0] * chunk_seconds / 60, 0]
    low, high = np.nanpercentile(np.log10(every[every > 0]), [1, 99.5])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, data, title in zip(axes, (every, kept), ("all blocks", "kept blocks")):
        with np.errstate(divide="ignore"):
            ax.imshow(np.log10(data), aspect="auto", extent=span, vmin=low, vmax=high, interpolation="nearest")
        if anchor is not None:
            ax.plot(anchor + 0.5, 0, "rv", clip_on=False)
        ax.set(title=title, xlabel="fine bin")
    axes[0].set_ylabel("minutes")
    fig.suptitle(f"freq_id {channel} (ATSC {PILOTS.get(channel, '?')}), log10 target fine power")
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
    if args.output_dir:
        with open(os.path.join(args.output_dir, "check.json"), "w") as out:
            json.dump(reports, out, indent=1)


if __name__ == "__main__":
    main()
