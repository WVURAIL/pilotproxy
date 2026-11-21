#!/usr/bin/env python3
"""Combine the per-window DTV files of a Pathfinder run into one file per product, half and acquisition.

    combine_pathfinder_dtv_run.py --acq "/mnt/cs00/data/dtv/acq_2026112*" --out ~/dtv-combined \
        [--products fine_powers mask powers fine_support]

recv writes one file per half and ~43 s window, about 56 000 a day for a calibration-mode run of the pilot nodes,
and each costs about 25 ms to open over NFS. This writes <out>/<acquisition>/dtv_<product>.<half>.combined.h5 for
every half that carries a coarse channel, with the windows in recording order and their fpga_seq_num and
fpga_seq_time_nsec kept, and reads every window back against its source. The PilotProxy run tools and
check_pathfinder_dtv_run.py read the combined files as they read the per-window ones, a window being
"<file>#<index>". The per-window files are not changed; delete them only once the combined ones are checked.
Files of several frames with per-frame metadata (kotekan#1925) need no combining and are left out.
An acquisition with a file changed in the last --settle-minutes (10) is taken as still being written and skipped,
so the run's current acquisition is left for later. Writes combine.json beside each one.
"""
import argparse
import glob
import json
import time
from pathlib import Path

from pilot_proxy.run_files import combine

PRODUCTS = ["fine_powers", "mask", "powers", "fine_support"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acq", nargs="+", required=True, help="Acquisition directories or glob patterns.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the combined acquisitions.")
    parser.add_argument("--products", nargs="+", default=PRODUCTS, choices=PRODUCTS)
    parser.add_argument("--no-verify", action="store_true", help="Skip reading the windows back.")
    parser.add_argument("--settle-minutes", type=float, default=10.0,
                        help="Skip an acquisition with a file changed this recently, as still being written.")
    args = parser.parse_args()
    dirs = sorted({Path(p) for pattern in args.acq for p in glob.glob(pattern)} | {Path(p) for p in args.acq
                   if not glob.has_magic(p)})
    for acq in dirs:
        out = args.out / acq.name
        if (out / "combine.json").exists():
            print(f"{acq.name}: already combined; skipped", flush=True)
            continue
        newest = max((p.stat().st_mtime for p in acq.glob("dtv_*.h5")), default=0.0)
        if time.time() - newest < 60 * args.settle_minutes:
            print(f"{acq.name}: a file changed in the last {args.settle_minutes:g} min, so still being written; skipped",
                  flush=True)
            continue
        start, summary = time.monotonic(), []
        for product in args.products:
            for row in combine(acq, out, product, verify=not args.no_verify):
                summary.append(row)
                if "file" in row:
                    print(f"{acq.name} {Path(row['file']).name}: {row['windows']} windows", flush=True)
        record = dict(acquisition=str(acq.resolve()), products=args.products, files=summary,
                      verified=not args.no_verify, seconds=time.monotonic() - start)
        (out / "combine.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"{acq.name}: done in {record['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
