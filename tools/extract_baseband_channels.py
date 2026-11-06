#!/usr/bin/env python3
"""Copy a few coarse channels of a merged CHORD baseband file into a small file.

Merged files are chunked over every frequency, so one read of a channel costs a read
of all of them. Extract once, then point calibrate-bundle at the small files.
"""
import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coarse-freq", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--block-samples", type=int, default=8192)
    args = parser.parse_args()

    wanted = sorted(args.coarse_freq)
    with h5py.File(args.input, "r") as source, h5py.File(args.output, "w") as target:
        data = source["baseband_subset"]
        coarse = [int(c) for c in data.attrs["coarse_freq"]]
        index = [coarse.index(c) for c in wanted]
        seq = int(data.attrs["fpga_seq_num"])
        start = (-seq) % args.block_samples
        blocks = min(args.blocks, (data.shape[0] - start) // args.block_samples)
        shape = (blocks * args.block_samples, len(wanted)) + data.shape[2:]
        out = target.create_dataset("baseband_subset", shape=shape, dtype=np.uint8)
        for block in range(blocks):
            lo = start + block * args.block_samples
            out[block * args.block_samples:(block + 1) * args.block_samples] = data[lo:lo + args.block_samples, index]
            target.flush()
            print(f"block {block + 1} of {blocks}", flush=True)
        out.attrs["coarse_freq"] = wanted
        out.attrs["fpga_seq_num"] = seq + start
        out.attrs["source_file"] = args.input
        if "dish_label" in data.attrs:
            out.attrs["dish_label"] = data.attrs["dish_label"]


if __name__ == "__main__":
    main()
