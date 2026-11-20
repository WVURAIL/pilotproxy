#!/usr/bin/env python3
"""Write synthetic Pathfinder recv files (dtv_fine_powers and dtv_mask) in calibration-mode layout, for
rehearsing the processing chain (process-run, calibration_length.py, the checks) without the telescope.

    make_pathfinder_mock_run.py OUT_DIR [--windows 84] [--burst-blocks 100 800] [--gzip]

Layout as recv1 writes it under /mnt/cs00/data/dtv/acq_<start>/: one file per half and 43 s window,
dataset dtv_<product>, attributes fpga_seq_num and coarse_freq. Bound halves follow the Pathfinder map
of 2026-11-19 (cx47: 2531 in stream 3, 2715 and 3115 in stream 11; cx42: 2869 in stream 5). A fourth
half binds no pilot, so its fine powers carry no coarse_freq, as Kotekan drops an all -1 label.
DTV comes and goes on 2531, 2715 and 2869 at the anchors of pathfinder-commissioning-e2e90219;
3115, the control, has noise only. Synthetic data, for exercising the code only.
"""
import argparse
import os

import h5py
import hdf5plugin
import numpy as np

BLOCK, WINDOW, NF, F_PER_HALF = 8192, 1024, 256, 464
ANCHOR = {2531: 82, 2715: 180, 2869: 51, 3115: 214}
HALVES = {"cx47_s3": (3, [2531, -1]), "cx47_s11": (11, [2715, 3115]),
          "cx42_s5": (5, [2869, -1]), "cx19_s1": (1, [-1, -1])}

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--windows", type=int, default=84)
parser.add_argument("--burst-blocks", type=int, nargs=2, default=(500, 4000),
                    help="Range of DTV on/off stretches in blocks; shorter stretches make the halves alike.")
parser.add_argument("--gzip", action="store_true",
                    help="Compress with gzip where the installed hdf5plugin cannot write bitshuffle.")
args = parser.parse_args()
rng = np.random.default_rng(20261119)
out = os.path.join(args.out, "acq_20261119_000000_000000000")
os.makedirs(out, exist_ok=True)
nblocks = args.windows * WINDOW
seq0 = 85_000_000_000 // BLOCK * BLOCK

# DTV on in bursts of 20 s to 3 min, about half the time, separately on each pilot
level = {}
for channel in (2531, 2715, 2869):
    on = np.zeros(nblocks, bool)
    t = 0
    while t < nblocks:
        run = int(rng.integers(*args.burst_blocks))
        on[t:t + run] = rng.random() < 0.5
        t += run
    level[channel] = np.where(on, rng.uniform(0.5, 2.0, nblocks), 0.0)

SHAPE, FLOOR = 32.0, 1.0e6        # sum of independent powers: gamma, mean FLOOR
# bitshuffle as recv1 writes it
compress = dict(compression="gzip", compression_opts=1) if args.gzip else hdf5plugin.Bitshuffle()
index = 0
for name, (stream, rows) in HALVES.items():
    coarse = [768 + stream + 16 * n for n in range(F_PER_HALF)]
    for w in range(args.windows):
        blocks = slice(w * WINDOW, (w + 1) * WINDOW)
        fine = np.zeros((WINDOW, 2, 3, NF), np.uint64)
        mask = np.zeros((WINDOW, F_PER_HALF), np.int8)
        for r, channel in enumerate(rows):
            if channel < 0:
                continue
            noise = rng.gamma(SHAPE, FLOOR / SHAPE, (WINDOW, 3, NF))
            lv = level.get(channel, np.zeros(nblocks))[blocks][:, None]
            pedestal = np.zeros(NF)
            pedestal[ANCHOR[channel] + 1:] = 2.0          # the DTV data spectrum above the pilot
            line = np.zeros(NF)
            line[ANCHOR[channel]] = 30.0                  # the pilot, in the target row only
            noise[:, 0] *= 1.0 + lv * (pedestal + line)
            noise[:, 1:] *= 1.0 + lv[:, None] * pedestal
            fine[:, r] = noise.astype(np.uint64)
            if channel in level:
                mask[:, coarse.index(channel)] = level[channel][blocks] > 0
        seq = seq0 + w * WINDOW * BLOCK
        for product, data, label in (("fine_powers", fine, rows), ("mask", mask, coarse)):
            path = os.path.join(out, f"dtv_{product}.{index:08d}.h5")
            with h5py.File(path, "w") as f:
                d = f.create_dataset(f"dtv_{product}", data=data, **compress)
                d.attrs["fpga_seq_num"] = seq
                if any(c >= 0 for c in label):              # Kotekan drops an all -1 label
                    d.attrs["coarse_freq"] = np.array(label, np.int32)
        index += 1
print(f"wrote {index} windows per product to {out}; DTV on",
      {c: round(float((lv > 0).mean()), 2) for c, lv in level.items()})
