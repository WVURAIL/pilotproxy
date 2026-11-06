#!/usr/bin/env python3
"""Read a detect-only run's raw products (dtv_mask, dtv_powers, dtv_fine_support written by
config/include/dtv_chime_upgrade.j2) and apply the hello-world pass criteria.

    analyze_dtv_run.py --run-dir /scratch/dtv/<run> --bundle-dir <bundle> \
        [--strong 35] [--quiet 29] [--ranges analysis/archive_ranges.json] [--json out.json]

Checks, all from the files alone:
  mechanics  three products with equal frame counts; FPGA sequence advancing by 16384 per frame, no gaps;
             one frequency list for the run; unbound rows read mask 0, powers 0, support (-1,-1);
             bound rows carry valid rank support (1, n_bulk) on every frame (invalid frames are listed)
  yes        the strong channel's masked fraction lies at or above the low end of its archive monthly range
             (5th percentile) at the bundle word
  quiet      the quiet channel's masked fraction lies at or below the high end (95th percentile)
  report     per bound channel: masked fraction, per-minute masked fractions, runs of consecutive masked frames,
             and the frame-power ratio 2 P_target / (P_lower + P_upper) against the profile's null ratio
Needs only numpy. Exit status 0 when every check passes, 1 otherwise.
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

FRAME_SAMPLES = 16384
SAMPLE_SECONDS = 2.56e-6
# chordMetadataFormat prefix: six int32 layout limits, int32 frame counter, ABI padding, int64 fpga_seq_num,
# int64 time_downsampling_fpga ... int32 nfreq at byte 528 and int32 frequencies[12288] after it.
SEQ_OFFSET, PERIOD_OFFSET = 32, 40


def nfreq_offset():
    # limits 24, frame 4, pad 4, sequence 8, period 8, name 24, type 4, dims 4, shape 40, dim_names 240,
    # scaling 80, stride 80, offset 8 -> nfreq
    return 24 + 4 + 4 + 8 + 8 + 24 + 4 + 4 + 40 + 240 + 80 + 80 + 8


def read_product(run_dir, name, frame_bytes):
    paths = sorted(glob.glob(os.path.join(run_dir, f"*dtv_{name}_*.raw")) +
                   glob.glob(os.path.join(run_dir, f"dtv_{name}_*.raw")))
    paths = sorted(set(paths))
    seqs, periods, freqs, payloads, meta_size = [], [], None, [], None
    nf_off = nfreq_offset()
    for path in paths:
        data = open(path, "rb").read()
        off = 0
        while off < len(data):
            (msize,) = struct.unpack_from("<I", data, off)
            off += 4
            meta = data[off:off + msize]
            off += msize
            payload = data[off:off + frame_bytes]
            off += frame_bytes
            if len(meta) != msize or len(payload) != frame_bytes:
                raise ValueError(f"{path}: truncated frame (a reused output directory leaves stale tails)")
            meta_size = msize
            seqs.append(struct.unpack_from("<q", meta, SEQ_OFFSET)[0])
            periods.append(struct.unpack_from("<q", meta, PERIOD_OFFSET)[0])
            nfreq = struct.unpack_from("<i", meta, nf_off)[0]
            f = tuple(struct.unpack_from(f"<{nfreq}i", meta, nf_off + 4))
            if freqs is None:
                freqs = f
            elif f != freqs:
                raise ValueError(f"{path}: frequency list changed during the run")
            payloads.append(payload)
    return dict(paths=paths, seq=np.array(seqs, np.int64), period=np.array(periods), freqs=freqs,
                payload=payloads, metadata_bytes=meta_size)


def runs_of(mask):
    """(start, length) of runs of consecutive True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--strong", type=int, default=35)
    ap.add_argument("--quiet", type=int, default=29)
    ap.add_argument("--ranges", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "analysis", "archive_ranges.json"))
    ap.add_argument("--num-freq", type=int, default=16)
    ap.add_argument("--json")
    a = ap.parse_args()
    F = a.num_freq
    mask = read_product(a.run_dir, "mask", F)
    powers = read_product(a.run_dir, "powers", F * 3 * 8)
    support = read_product(a.run_dir, "fine_support", F * 2 * 4)
    checks, report = {}, {}
    n = len(mask["seq"])
    checks["frames_present"] = n > 0
    checks["equal_frame_counts"] = n == len(powers["seq"]) == len(support["seq"])
    checks["same_sequences"] = checks["equal_frame_counts"] and bool(
        (mask["seq"] == powers["seq"]).all() and (mask["seq"] == support["seq"]).all())
    d = np.diff(mask["seq"])
    gaps = [(int(mask["seq"][i]), int(d[i])) for i in np.nonzero(d != FRAME_SAMPLES)[0]]
    checks["no_sequence_gaps"] = not gaps
    checks["period_16384"] = bool((mask["period"] == FRAME_SAMPLES).all())
    freqs = list(mask["freqs"] or [])
    checks["frequency_list_length"] = len(freqs) == F
    M = np.array([np.frombuffer(p, np.int8) for p in mask["payload"]]).reshape(n, F)
    P = np.array([np.frombuffer(p, "<u8") for p in powers["payload"]]).reshape(n, F, 3)
    S = np.array([np.frombuffer(p, "<i4") for p in support["payload"]]).reshape(n, F, 2)
    bundle = json.load(open(os.path.join(a.bundle_dir, "pilot_profiles.json")))
    prof = {r["receiver_channel_id"]: r for r in bundle["profiles"]}
    bound_cols = {f: i for i, f in enumerate(freqs) if f in prof}
    unbound = [i for i, f in enumerate(freqs) if f not in prof]
    checks["unbound_rows_zero"] = bool(not M[:, unbound].any() and not P[:, unbound].any()
                                       and (S[:, unbound] == -1).all())
    ranges = json.load(open(a.ranges)) if os.path.exists(a.ranges) else {}
    # the archive ranges computed for this bundle: same channels at the same words
    words = {str(r["physical_channel"]): r["fine_calibration"]["cfar_multiplier_q16"] for r in prof.values()}
    ranges = next((rows for rows in ranges.values()
                   if {c: v["word_q16"] for c, v in rows.items()} == words), {})
    checks["archive_ranges_found_for_bundle"] = bool(ranges)
    minute = int(round(60 / (FRAME_SAMPLES * SAMPLE_SECONDS)))
    for f, i in bound_cols.items():
        ch = prof[f]["physical_channel"]
        valid = S[:, i, 0] == 1
        m = M[:, i].astype(bool)
        ratio = 2.0 * P[:, i, 0] / np.maximum(P[:, i, 1] + P[:, i, 2], 1)
        null = prof[f]["null_power_ratio"]
        runs = runs_of(m)
        r = dict(physical_channel=ch, freq_id=f, frames=n, valid=int(valid.sum()),
                 invalid_frames=[int(s) for s in mask["seq"][~valid][:50]],
                 masked=int((m & valid).sum()), masked_fraction=float((m & valid).sum() / max(valid.sum(), 1)),
                 masked_per_minute=[float(m[k:k + minute].mean()) for k in range(0, n, minute)],
                 longest_masked_run=max((l for _, l in runs), default=0), masked_runs=len(runs),
                 power_ratio_median=float(np.median(ratio)), power_ratio_p10=float(np.percentile(ratio, 10)),
                 null_power_ratio=null,
                 # Paper I's pilot excess, 10 log10(R / R_null - 1); None when the median ratio is at or below the null
                 excess_db_median=(float(10 * np.log10(np.median(ratio) / null - 1))
                                   if np.median(ratio) > null else None),
                 archive_range=ranges.get(str(ch)))
        report[ch] = r
        checks[f"ch{ch}_rank_support_valid_every_frame"] = bool(valid.all())
    if a.strong in report and ranges.get(str(a.strong)):
        checks[f"yes_ch{a.strong}_masked_at_or_above_archive_p05"] = \
            report[a.strong]["masked_fraction"] >= ranges[str(a.strong)]["monthly_p05"]
    if a.quiet in report and ranges.get(str(a.quiet)):
        checks[f"quiet_ch{a.quiet}_masked_at_or_below_archive_p95"] = \
            report[a.quiet]["masked_fraction"] <= ranges[str(a.quiet)]["monthly_p95"]
    out = dict(run_dir=a.run_dir, bundle_dir=a.bundle_dir, frames=n,
               seconds=n * FRAME_SAMPLES * SAMPLE_SECONDS, first_fpga_seq=int(mask["seq"][0]) if n else None,
               metadata_bytes_per_frame=mask["metadata_bytes"], frequencies=freqs, sequence_gaps=gaps[:50],
               checks=checks, channels=report, result="PASS" if all(checks.values()) else "FAIL")
    text = json.dumps(out, indent=1)
    if a.json:
        open(a.json, "w").write(text + "\n")
    print(json.dumps(dict(result=out["result"], frames=n, checks=checks,
                          channels={c: {k: r[k] for k in ("freq_id", "valid", "masked_fraction", "longest_masked_run",
                                                         "excess_db_median")} for c, r in report.items()}), indent=1))
    sys.exit(0 if out["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
