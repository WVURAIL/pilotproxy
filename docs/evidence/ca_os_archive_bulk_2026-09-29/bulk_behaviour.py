"""Part 1: bulk behaviour of the fine statistic on the current era of every channel.

For every channel, every current-era frame (calibration plus evaluation block, the release's own era and split):
  r = mean_B T / median_B T   and   u = max_B T / median_B T,
with T[f] = 2 S0[f] / (S1[f] + S2[f]) (the deployed statistic, rfisher_results.archive.products.fine_power_ratio),
B the selector's bulk (anchors.bulk_mask on the selector's anchor) and D the anchor +/- 2 bins. The same frames
give Z(63) = max_D T / T_(63) and Z_CA = max_D T / mean_B T (display floats), stored for the later parts.

Attribution of the mean's excess over the median, per frame:
  * the top 1 and top 3 bulk bins of the frame,
  * the near-anchor bulk bins (within +/- 7 padded bins of the anchor: for an odd anchor these are the pilot's
    Dirichlet shoulders at 1.5, 2.5 and 3.5 natural bins; for an even anchor they sit at 2 and 3 natural bins),
  * persistent bins (bins above the ideal single-bin 0.999 point of T/median in more than half the era's frames).

Usage (archive interpreter, one thread, no bytecode):
  python -B bulk_behaviour.py last            # cache the campaign's last populated month
  python -B bulk_behaviour.py CH [CH ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import common
from common import HERE, fine_power_ratio

OUT = HERE / "data/bulk"
NEAR = 7          # padded bins either side of the anchor
IDEAL = HERE / "data/ideal_bulk.json"


def ideal_reference(sizes=(125, 126), frames=400_000, seed=20260929, chunk=50_000):
    """The i.i.d. F(4096, 8192) law for r, u and the single-bin T/median, by simulation."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        rs, us, singles = [], [], []
        for _ in range(frames // chunk):
            t = (rng.chisquare(4096, (chunk, n)) / 4096.0) / (rng.chisquare(8192, (chunk, n)) / 8192.0)
            med = np.median(t, axis=1)
            rs.append(t.mean(1) / med)
            us.append(t.max(1) / med)
            singles.append((t[:, 0] / med))
        r, u, sgl = np.concatenate(rs), np.concatenate(us), np.concatenate(singles)
        q = lambda a, x: float(np.quantile(a, x))
        out[str(n)] = dict(frames=frames, seed=seed,
                           r_median=q(r, 0.5), r_p99=q(r, 0.99), r_p999=q(r, 0.999), r_max=float(r.max()),
                           r_std=float(r.std()), r_frac_gt_1p05=float((r > 1.05).mean()), r_frac_gt_1p25=float((r > 1.25).mean()),
                           u_median=q(u, 0.5), u_p99=q(u, 0.99), u_p999=q(u, 0.999),
                           single_p999=q(sgl, 0.999), single_p9999=q(sgl, 0.9999))
    return out


def run(ch: int, last: int, ideal: dict) -> None:
    t0 = time.time()
    c = common.channel_setup(ch, last)
    p, bulk, D, a = c["p"], c["bulk"], c["designated"], c["anchor_bin"]
    rows_mask = c["era_mask"] & p.selected
    n_blocks = c["split"].calibration_frames + c["split"].evaluation_frames
    assert int(rows_mask.sum()) == n_blocks, (ch, int(rows_mask.sum()), n_blocks)
    rows = np.flatnonzero(rows_mask)
    fine = np.asarray(p.archive["fine_power_u64"])[rows]
    den_zero_bulk = int(((fine[:, 1, :].astype(np.float64) + fine[:, 2, :]) == 0)[:, bulk].sum())
    T = fine_power_ratio(fine)
    del fine
    bidx = np.flatnonzero(bulk)
    nb = bidx.size
    TB = T[:, bidx]
    srt = np.sort(TB, axis=1)
    med = np.median(TB, axis=1)
    mean = TB.mean(axis=1)
    r = mean / med
    u = srt[:, -1] / med
    dmax = T[:, D].max(axis=1)
    z63 = dmax / srt[:, 62]
    zca = dmax / mean
    arg = bidx[np.argmax(TB, axis=1)]
    off = ((arg - a + 128) % 256) - 128          # signed padded-bin offset of the frame's largest bulk bin
    # excess of the mean over the median, in units of the median, and its parts
    excess = r - 1.0
    top1 = (srt[:, -1] - med) / (nb * med)
    top3 = (srt[:, -3:].sum(1) - 3 * med) / (nb * med)
    boff = ((bidx - a + 128) % 256) - 128
    near = np.abs(boff) <= NEAR
    near_share = (TB[:, near] - med[:, None]).sum(1) / (nb * med)
    single_p999 = ideal[str(nb)]["single_p999"]
    above = TB / med[:, None] > single_p999
    frac_above = above.mean(axis=0)
    persistent = frac_above > 0.5
    frequent = frac_above > 0.1
    pers_share = (TB[:, persistent] - med[:, None]).sum(1) / (nb * med) if persistent.any() else np.zeros(len(r))
    infl = r > 1.05
    # the mean with the candidate explanations removed (the median stays the full bulk's)
    r_no_near = TB[:, ~near].mean(1) / med
    r_no_near_pers = TB[:, ~near & ~persistent].mean(1) / med
    r_trim3 = srt[:, :-3].mean(1) / med
    pilot_on = dmax / med > 10.0
    calib = c["split"].calibration[rows]

    def share(part, mask):
        tot = excess[mask].sum()
        return float(part[mask].sum() / tot) if mask.any() and tot > 0 else None

    def pct(x, q):
        return float(np.quantile(x, q)) if x.size else None

    # per-bin profile: median over frames of T_b / median_B, and the fraction above the ideal 0.999 point
    prof = np.median(TB / med[:, None], axis=0)
    top_bins = np.argsort(-frac_above)[:8]
    summary = dict(
        channel=ch, freq_id=c["rec"]["sections"]["geometry"].get("freq_id") if "freq_id" in c["rec"]["sections"]["geometry"] else None,
        anchor_bin=a, anchor_parity="odd" if a % 2 else "even", anchor_source=c["anchor_source"],
        designated=[int(x) for x in D], bulk_size=int(nb), census_excluded=c["census"], frames=int(len(r)),
        calibration_frames=int(calib.sum()), state=c["s"]["era"]["current_state"],
        era=f'{c["s"]["era"]["current_first_month"]}..{c["s"]["era"]["current_last_month"]}',
        zero_denominator_bulk_entries=den_zero_bulk,
        r=dict(median=pct(r, 0.5), p90=pct(r, 0.9), p99=pct(r, 0.99), max=float(r.max()),
               frac_gt_1p05=float((r > 1.05).mean()), frac_gt_1p25=float((r > 1.25).mean()),
               frac_gt_ideal_p999=float((r > ideal[str(nb)]["r_p999"]).mean()),
               frac_gt_1p05_calibration=float((r[calib] > 1.05).mean()), frac_gt_1p05_evaluation=float((r[~calib] > 1.05).mean())),
        explained=dict(
            frac_gt_1p05_without_near_anchor=float((r_no_near > 1.05).mean()),
            frac_gt_1p05_without_near_and_persistent=float((r_no_near_pers > 1.05).mean()),
            frac_gt_1p05_without_top3=float((r_trim3 > 1.05).mean()),
            frac_gt_1p25_without_near_and_persistent=float((r_no_near_pers > 1.25).mean()),
            median_r_without_near_and_persistent=pct(r_no_near_pers, 0.5), p99_r_without_near_and_persistent=pct(r_no_near_pers, 0.99),
            pilot_on_frames=float(pilot_on.mean()),
            frac_gt_1p05_pilot_on=float((r[pilot_on] > 1.05).mean()) if pilot_on.any() else None,
            frac_gt_1p05_pilot_off=float((r[~pilot_on] > 1.05).mean()) if (~pilot_on).any() else None,
            near_anchor_bins=int(near.sum()), persistent_bins=int(persistent.sum())),
        u=dict(median=pct(u, 0.5), p99=pct(u, 0.99), frac_gt_ideal_p999=float((u > ideal[str(nb)]["u_p999"]).mean())),
        excess_share_over_inflated_frames=dict(frames=int(infl.sum()), top1=share(top1, infl), top3=share(top3, infl),
                                               near_anchor=share(near_share, infl), persistent=share(pers_share, infl)),
        max_bulk_bin_offset=dict(frac_near_anchor=float((np.abs(off) <= NEAR).mean()),
                                 frac_at_pm3=float((np.abs(off) == 3).mean()),
                                 frac_near_anchor_inflated=float((np.abs(off[infl]) <= NEAR).mean()) if infl.any() else None,
                                 most_common_inflated=[[int(k), int(v)] for k, v in zip(*np.unique(off[infl], return_counts=True))]
                                 if infl.any() else []),
        persistent_bins=[dict(bin=int(bidx[i]), offset=int(boff[i]), frac_above=float(frac_above[i]), median_ratio=float(prof[i]))
                         for i in np.flatnonzero(persistent)],
        frequent_bins=[dict(bin=int(bidx[i]), offset=int(boff[i]), frac_above=float(frac_above[i]), median_ratio=float(prof[i]))
                       for i in np.flatnonzero(frequent)],
        top_bins_by_frac_above=[dict(bin=int(bidx[i]), offset=int(boff[i]), frac_above=float(frac_above[i]),
                                     median_ratio=float(prof[i])) for i in top_bins],
        z63_median=pct(z63, 0.5), zca_median=pct(zca, 0.5), zca_over_z63_median=pct(zca / z63, 0.5),
        zca_over_z63_p01=pct(zca / z63, 0.01), dmax_over_med_median=pct(dmax / med, 0.5),
        ideal=ideal[str(nb)], seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = dict(rows=rows, frame_time=np.asarray(p.frame_time)[rows], calibration=calib, r=r, u=u, med=med, mean=mean,
                  dmax=dmax, z63=z63, zca=zca, max_bulk_offset=off.astype(np.int16), top1=top1, top3=top3,
                  near_share=near_share, pers_share=pers_share, r_no_near=r_no_near, r_no_near_pers=r_no_near_pers, r_trim3=r_trim3, bulk_bins=bidx, bulk_offsets=boff,
                  frac_above=frac_above, profile=prof)
    if ch == 35:
        arrays["T"] = T.astype(np.float32)
    np.savez_compressed(OUT / f"ch{ch:02d}.npz", **arrays)
    (OUT / f"ch{ch:02d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("channel", "anchor_bin", "bulk_size", "frames", "seconds")} | {"r": summary["r"]}), flush=True)
    p.close()


if __name__ == "__main__":
    (HERE / "data").mkdir(exist_ok=True)
    if sys.argv[1] == "last":
        last = common.campaign_last()
        common.check_run_identity(last)
        (HERE / "data/campaign_last.json").write_text(json.dumps({"campaign_last": int(last)}) + "\n")
        IDEAL.write_text(json.dumps(ideal_reference(), indent=2) + "\n")
        print("last", last, flush=True)
        sys.exit(0)
    last = json.loads((HERE / "data/campaign_last.json").read_text())["campaign_last"]
    ideal = json.loads(IDEAL.read_text())
    for ch in map(int, sys.argv[1:]):
        run(ch, last, ideal)
