"""Per-frame inputs for the rank-by-injection study (DESIGN.md section 5, step 1).

Channels 14 to 36: every current-era frame of the release's era and split (common.channel_setup, counts asserted
against the ledger), the sorted usable bulk T_(1..|B|) (float32), the five designated values T[D], the frame time,
its calendar month, the calibration flag and the screens
    w    = max_D T / median_B T,
    w_lo = max_D T / m_lo,  m_lo = P25 + 0.6745 (P25 - P10) / 0.6071  (RANK.md's lower-quartile location),
with the ideal-law cut w_cut of design/design_counts.json. Rows, max_D T and median_B T are asserted against
cfar-choice-2026-09-29's per-frame bulk arrays (byte copies in data/bulk/).

Channel 37 (argument 37): the control's 40,477 histogram-eligible frames at every channel's geometry (that channel's
anchor, designated set and bulk mask), split by blocks.split_blocks on the control's own frame times, as
decision/references.py builds the geometry.

Usage (analysis interpreter, one thread, no bytecode): python -B extract.py CH [CH ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import common
from common import fine_power_ratio
from rfisher_results.archive import blocks, eras

ROOT = common.HERE.parent
FRAMES = ROOT / "data/frames"
BULK = ROOT / "data/bulk"
W_CUT = json.loads((ROOT / "design/design_counts.json").read_text())["ideal"]["w_cut"]
CH37 = Path("/home/djg/rail/products/ch37_control_491_20260924/_per_pilot/491.npz")
CH37_SHA = "e77d1d4135424510c64be8cb5e796c448cc9d81bd3c77a197128bdaa1233faef"
CH37_EXPORT = Path("/home/djg/rail/output/ch37-control-491-2026-09-24/export/frames/ch37.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def reduce(T, bulk, D):
    bidx = np.flatnonzero(bulk)
    TB = T[:, bidx]
    srt = np.sort(TB, axis=1)
    med = np.median(TB, axis=1)
    p10, p25 = np.quantile(TB, [0.10, 0.25], axis=1)
    m_lo = p25 + 0.6745 * (p25 - p10) / 0.6071
    dv = T[:, D]
    dmax = dv.max(axis=1)
    return dict(sorted_bulk=srt.astype(np.float32), dvals=dv.astype(np.float64), dmax=dmax, med=med, m_lo=m_lo,
                mean=TB.mean(axis=1), w=dmax / med, w_lo=dmax / m_lo), bidx


def run_channel(ch, last):
    t0 = time.time()
    c = common.channel_setup(ch, last)
    p, bulk, D = c["p"], c["bulk"], c["designated"]
    rows_mask = c["era_mask"] & p.selected
    assert int(rows_mask.sum()) == c["split"].calibration_frames + c["split"].evaluation_frames
    rows = np.flatnonzero(rows_mask)
    ref = np.load(BULK / f"ch{ch:02d}.npz")
    assert np.array_equal(rows, ref["rows"]), ch
    fine = np.asarray(p.archive["fine_power_u64"])[rows]
    T = fine_power_ratio(fine)
    del fine
    out, bidx = reduce(T, bulk, D)
    assert np.array_equal(bidx, ref["bulk_bins"]), ch
    assert np.allclose(out["dmax"], ref["dmax"], rtol=1e-12, atol=0) and np.allclose(out["med"], ref["med"], rtol=1e-12, atol=0), ch
    cal = c["split"].calibration[rows]
    assert np.array_equal(cal, ref["calibration"].astype(bool)), ch
    ft = p.frame_time[rows]
    assert np.all(np.isfinite(ft)) and np.array_equal(ft, ref["frame_time"]), ch
    month = blocks.month_index(ft)
    meta = dict(channel=ch, anchor_bin=int(c["anchor_bin"]), designated=[int(x) for x in D], bulk_size=int(bidx.size),
                bulk_bins=[int(x) for x in bidx], census=c["census"], state=c["s"]["era"]["current_state"],
                era=f'{c["s"]["era"]["current_first_month"]}..{c["s"]["era"]["current_last_month"]}',
                frames=int(rows.size), calibration=int(cal.sum()), evaluation=int((~cal).sum()), w_cut=W_CUT,
                near=int((out["w"] <= W_CUT).sum()), near_cal=int(((out["w"] <= W_CUT) & cal).sum()),
                near_eval=int(((out["w"] <= W_CUT) & ~cal).sum()), near_lo=int((out["w_lo"] <= W_CUT).sum()),
                boundary_time=float(c["split"].boundary_time), product=str(c["rec"]["product"]))
    p.close()
    np.savez(FRAMES / f"ch{ch:02d}.npz", rows=rows, frame_time=ft, month=month, calibration=cal, **out)
    (FRAMES / f"ch{ch:02d}.json").write_text(json.dumps(meta, indent=1) + "\n")
    # the design table's counts (design_counts.json) must be reproduced
    dc = json.loads((ROOT / "design/design_counts.json").read_text())["channels"][str(ch)]
    for k in ("near", "near_cal", "near_eval"):
        assert dc[k] == meta[k], (ch, k, dc[k], meta[k])
    print(json.dumps({k: meta[k] for k in ("channel", "anchor_bin", "bulk_size", "frames", "calibration", "near_cal", "near_eval", "near_lo")}),
          f"{time.time() - t0:.1f} s", flush=True)


def run_ch37(last):
    t0 = time.time()
    assert sha256(CH37) == CH37_SHA
    ex = np.load(CH37_EXPORT, allow_pickle=False)
    P = ex["histogram_eligible"].astype(bool)
    assert int(P.sum()) == 40477
    with np.load(CH37, allow_pickle=False) as z:
        assert np.array_equal(np.asarray(z["frame_index"]).reshape(-1), np.asarray(ex["frame_index"]))
        pad, guard = int(z["fine_pad_factor"]), int(z["fine_guard_fine_bins"])
        census37 = np.atleast_1d(z["fine_census_excluded_bins"])
        u = np.asarray(z["fine_power_u64"])[P]
    T = fine_power_ratio(u)
    del u
    ft = np.asarray(ex["frame_time"], dtype=float)[P]
    assert np.all(np.isfinite(ft))
    cfg = eras.DEFAULT_CONFIG
    split = blocks.split_blocks(np.asarray(ex["frame_unit_index"]), np.asarray(ex["unit_time"], dtype=float), P,
                                frame_time=np.asarray(ex["frame_time"], dtype=float), minimum_months=1,
                                month_kwargs=dict(min_frames=cfg.min_frames, min_units=cfg.min_units, min_days=cfg.min_days))
    cal = split.calibration[P]
    month = blocks.month_index(ft)
    geoms = {}
    for ch in common.CHANNELS:
        c = common.channel_setup(ch, last)
        p = c["p"]
        assert (int(p.fine_pad_factor), int(p.fine_guard_bins)) == (pad, guard), ch
        assert census37.size == 0
        bulk, D = c["bulk"], c["designated"]
        p.close()
        out, bidx = reduce(T, bulk, D)
        np.savez(FRAMES / f"ch37_g{ch:02d}.npz", rows=np.flatnonzero(P), frame_time=ft, month=month, calibration=cal, **out)
        geoms[str(ch)] = dict(anchor_bin=int(c["anchor_bin"]), designated=[int(x) for x in D], bulk_size=int(bidx.size),
                              bulk_bins=[int(x) for x in bidx], census=c["census"])
        print(ch, geoms[str(ch)]["anchor_bin"], geoms[str(ch)]["bulk_size"], flush=True)
    meta = dict(channel=37, product=str(CH37), sha256=CH37_SHA, frames=int(P.sum()), calibration=int(cal.sum()),
                evaluation=int((~cal).sum()), boundary_time=float(split.boundary_time),
                boundary_month=blocks.month_label(int(blocks.month_index([split.boundary_time])[0])),
                calibration_months=len(split.calibration_months), evaluation_months=len(split.evaluation_months),
                split_status=split.status, geometries=geoms, seconds=round(time.time() - t0, 1))
    (FRAMES / "ch37.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "geometries"}), flush=True)


if __name__ == "__main__":
    last = json.loads((ROOT / "data/campaign_last.json").read_text())["campaign_last"]
    for a in sys.argv[1:]:
        run_ch37(last) if int(a) == 37 else run_channel(int(a), last)
