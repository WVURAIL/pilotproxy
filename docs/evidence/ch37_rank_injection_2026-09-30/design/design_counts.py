"""Design numbers for DESIGN.md: the ideal-law pilot-weak cut and the per-channel frame counts.

Reads the per-frame bulk arrays of cfar-choice-2026-09-29 (data/bulk/chNN.npz: dmax = max_D T, med = median_B T,
the release's calibration flag), read-only. No product is opened.

Ideal law: i.i.d. F(4096, 8192) bins, |B| = 125, D gains (0, g1, 1, g1, 0) as rank_bench.py.
  s90  : SNR (dB) where clean CA gives Pd = 0.9 at Pfa = 1e-3 (the bench's -8.631 dB is asserted to 0.01 dB);
  s999 : SNR where clean OS at the median gives Pd = 0.999 at Pfa = 1e-3 (detection saturated);
  w_cut: upper 0.999 point of w = max_D T / median_B T with the pilot at s999.
A frame is near-threshold (pilot-free or pilot-weak) when w <= w_cut; above it, its pilot is stronger than any the
detector could miss. w <= 10 is cfar-choice's pilot-off definition, reported for comparison.
Skirt scaling on proxy-high channels: on pilot-on frames (w > 10), the median over frames of
eps = (med - m0) / (dmax - m0), m0 the channel's median of med over its near-threshold frames (clean level).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import f as fdist, ncf

OUT = Path(__file__).resolve().parent
BULK = Path("/home/djg/rail/output/cfar-choice-2026-09-29/data/bulk")
L = 128
G1 = float((np.sin(np.pi * 0.5) / (L * np.sin(np.pi * 0.5 / L))) ** 2)
GAINS = (0.0, G1, 1.0, G1, 0.0)
STATE = {**{c: "proxy-low" for c in (14, 16, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 32, 33, 34)},
         **{c: "proxy-high" for c in (15, 17, 22, 24, 30, 31, 35)}, 36: "ambiguous"}
CARRIER_OFF = {19: "2024-11", 26: "2023-04", 27: "2020 (CRTC 2020-391)", 32: "2020 (CRTC 2020-391)"}


def dcdf(x, s):
    out = np.ones_like(x)
    for g in GAINS:
        out = out * (fdist.cdf(x, 4096, 8192) if g == 0 or s == 0 else ncf.cdf(x, 4096, 8192, 4096 * s * g))
    return out


def main():
    rng = np.random.default_rng(20260930)
    n, nf = 125, 200_000
    t = rng.f(4096, 8192, size=(nf, n))
    ca, med = t.mean(1), np.median(t, 1)
    grid = np.linspace(0.9, 1.6, 7001)
    res = {}
    for name, ref in (("CA", ca), ("median", med)):
        f0 = dcdf(grid, 0.0)
        eta = brentq(lambda e: 1 - np.interp(e * ref, grid, f0).mean() - 1e-3, 0.95, 1.5, xtol=1e-9)
        tabs = {}

        def pd(sdb):
            s = 10 ** (sdb / 10)
            if sdb not in tabs:
                tabs[sdb] = dcdf(grid, s)
            return 1 - np.interp(eta * ref, grid, tabs[sdb]).mean()
        res[name] = dict(eta_1e3=eta, s90_db=brentq(lambda v: pd(v) - 0.9, -14, 0, xtol=1e-5),
                         s999_db=brentq(lambda v: pd(v) - 0.999, -14, 5, xtol=1e-5))
    assert abs(res["CA"]["s90_db"] + 8.631) < 0.01, res
    s999 = 10 ** (res["median"]["s999_db"] / 10)
    d = np.stack([rng.noncentral_f(4096, 8192, 4096 * s999 * g, nf) if g else rng.f(4096, 8192, nf) for g in GAINS], 1)
    w = d.max(1) / med
    w0 = np.stack([rng.f(4096, 8192, nf) for _ in GAINS], 1).max(1) / med
    ideal = dict(model="i.i.d. F(4096, 8192), |B| = 125, D gains (0, g1, 1, g1, 0)", frames=nf, seed=20260930,
                 detectors=res, w_cut=float(np.quantile(w, 0.999)), w_null_p999=float(np.quantile(w0, 0.999)),
                 w_null_p50=float(np.median(w0)))
    wc = ideal["w_cut"]
    rows = {}
    for ch in list(range(14, 37)):
        z = np.load(BULK / f"ch{ch:02d}.npz")
        dmax, md, cal = z["dmax"], z["med"], z["calibration"].astype(bool)
        w = dmax / md
        near, off10 = w <= wc, w <= 10
        m0 = float(np.median(md[near])) if near.any() else float("nan")
        on = w > 10
        eps = float(np.median((md[on] - m0) / (dmax[on] - m0))) if on.sum() > 30 else None
        rows[ch] = dict(state=STATE[ch], carrier_off_since=CARRIER_OFF.get(ch), bulk_size=int(z["bulk_bins"].size),
                        frames=int(w.size), calibration=int(cal.sum()), evaluation=int((~cal).sum()),
                        near=int(near.sum()), near_cal=int((near & cal).sum()), near_eval=int((near & ~cal).sum()),
                        mid=int((~near & off10).sum()), on=int(on.sum()), near_share=float(near.mean()),
                        clean_level_m0=m0, skirt_eps_median=eps,
                        skirt_excess_at_threshold=(eps * 0.137 if eps is not None else None),
                        r_near_frac_gt_1p05=float((z["r"][near] > 1.05).mean()) if near.any() else None,
                        r_on_frac_gt_1p05=float((z["r"][on] > 1.05).mean()) if on.any() else None)
        print(ch, rows[ch], flush=True)
    (OUT / "design_counts.json").write_text(json.dumps(dict(ideal=ideal, channels=rows), indent=1) + "\n")
    print(json.dumps(ideal, indent=1))


if __name__ == "__main__":
    main()
