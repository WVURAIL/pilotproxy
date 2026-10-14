"""Independent re-run of the two injection modes on real frames (verifier's own code; nothing imported from rank/).

Units: channel 37 at two channel geometries (an even-anchor 125-bin and an odd-anchor 126-bin bulk), read from the
raw product's fine_power_u64; and carrier-off channel 19's near-threshold frames, read from its raw product at the
release rows. T = 2 S0 / (S1 + S2) formed here. Per half and per Pfa in {1e-3, 1e-2}:

Mode I: real sorted bulk, ideal D. eta_I(rho) solves 1 - mean_n F(eta t_n)^5 = Pfa; Pd(s) = 1 - mean_n G(eta t_n; s),
        G the product of the five D CDFs (gains 0, g1, 1, g1, 0), from a (s, x) table; s90 read by interpolation.
Mode R: real bulk, real D. eta_R(rho) = empirical (1 - Pfa) quantile of Z = max_D T / t_rho; the tone is added to each
        real D value as T' = T + a + 2 u sqrt(a T), a = m0 s g, u = 2 Beta(m, m) - 1, m = (4096 - 1)/2, and Pd is the
        exact average over u (Beta CDF), then over frames.
L(rho) = s90(rho) - min over rho. Also the real false-alarm rate of eta_I on the real D, per-frame bulk spread, and a
calendar-month block bootstrap (B = 40) for one unit at a few ranks.
Usage: python -B v_real.py UNIT   (UNIT in 37g21, 37g28, 19)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq

D1, D2 = 4096, 8192
G1 = 0.4053050802342348
GAINS = np.array([0.0, G1, 1.0, G1, 0.0])
M = (D1 - 1) / 2
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CH37 = Path("/home/djg/rail/products/ch37_control_491_20260924/_per_pilot/491.npz")
CH37_EXPORT = Path("/home/djg/rail/output/ch37-control-491-2026-09-24/export/frames/ch37.npz")
PRODUCTS = Path("/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot")
CFAR_BULK = Path("/home/djg/rail/output/cfar-choice-2026-09-29/data/bulk")
W_CUT = 1.2962289629059025
SDB = np.arange(-11.0, 6.0001, 0.02)
XG = np.exp(np.linspace(np.log(0.5), np.log(8.0), 8000))


def ratio(u):
    u = u.astype(np.float64)
    den = u[:, 1] + u[:, 2]
    return np.where(den > 0, 2 * u[:, 0] / np.where(den > 0, den, 1), 0.0)


def geometry(anchor, nbins=256, census=()):
    D = np.array([(anchor + k) % nbins for k in (-2, -1, 0, 1, 2)])
    bulk = np.array([b for b in range(0, nbins, 2) if b not in set(D) and b not in set(census)])
    return D, bulk


def load_unit(unit):
    if unit.startswith("37g"):
        g = unit[3:]
        meta = json.loads((ROOT / "data/frames/ch37.json").read_text())
        geo = meta["geometries"][g]
        ex = np.load(CH37_EXPORT)
        P = ex["histogram_eligible"].astype(bool)
        with np.load(CH37) as z:
            assert np.array_equal(np.asarray(z["frame_index"]).reshape(-1), np.asarray(ex["frame_index"]))
            T = ratio(np.asarray(z["fine_power_u64"])[P])
        ft = np.asarray(ex["frame_time"], float)[P]
        cal = ft < meta["boundary_time"]
        theirs = np.load(ROOT / f"data/frames/ch37_g{g}.npz")
        split_match = bool(np.array_equal(cal, theirs["calibration"]))
        anchor = geo["anchor_bin"]
        census = geo["census"]
        sel = np.ones(T.shape[0], bool)
    else:
        ch = int(unit)
        theirs = np.load(ROOT / f"data/frames/ch{ch:02d}.npz")
        meta = json.loads((ROOT / f"data/frames/ch{ch:02d}.json").read_text())
        cf = np.load(CFAR_BULK / f"ch{ch:02d}.npz")
        rows = np.asarray(cf["rows"])
        assert np.array_equal(rows, theirs["rows"])
        with np.load(PRODUCTS / meta["product"]) as z:
            T = ratio(np.asarray(z["fine_power_u64"])[rows])
        cal = cf["calibration"].astype(bool)
        ft = np.asarray(cf["frame_time"], float)
        split_match = bool(np.array_equal(cal, theirs["calibration"]))
        anchor = meta["anchor_bin"]
        census = meta["census"]
        geo = meta
        sel = None
    D, bulk = geometry(anchor, T.shape[1], census)
    assert list(D) == list(geo["designated"]) and list(bulk) == list(geo["bulk_bins"]), (unit, len(bulk))
    TB = np.sort(T[:, bulk], axis=1)
    dv = T[:, D]
    med = np.median(T[:, bulk], axis=1)
    dmax = dv.max(axis=1)
    w = dmax / med
    if sel is None:
        sel = w <= W_CUT
    data_match = bool(np.allclose(TB, theirs["sorted_bulk"], rtol=1e-6, atol=0) and np.allclose(dv, theirs["dvals"], rtol=1e-12, atol=0))
    month = ft.astype("datetime64[s]").astype("datetime64[M]").astype(int)  # calendar months (UTC) for the bootstrap
    return dict(TB=TB, dv=dv, med=med, dmax=dmax, w=w, cal=cal, sel=sel, month=month, bulk=len(bulk), anchor=anchor,
                split_match=split_match, data_match=data_match, frames=int(T.shape[0]),
                near_cal=int((sel & cal).sum()), near_eval=int((sel & ~cal).sum()))


def g_table():
    Gt = np.empty((SDB.size, XG.size))
    F0 = sp.fdtr(D1, D2, XG)
    for i, sdb in enumerate(SDB):
        s = 10 ** (sdb / 10)
        Gt[i] = F0 ** 2 * sp.ncfdtr(D1, D2, D1 * s * G1, XG) ** 2 * sp.ncfdtr(D1, D2, D1 * s, XG)
    return Gt


def s90_from_curve(pd):
    # pd increasing in s; linear interpolation of the Pd = 0.9 crossing
    i = np.searchsorted(pd, 0.9)
    if i == 0 or i >= pd.size:
        return np.nan
    return SDB[i - 1] + (0.9 - pd[i - 1]) / (pd[i] - pd[i - 1]) * (SDB[i] - SDB[i - 1])


def mode_i(t, pfa, Gt):
    eta = brentq(lambda e: (1 - np.mean(sp.fdtr(D1, D2, e * t) ** 5)) - pfa, 0.5, 20, xtol=1e-12)
    x = np.clip(eta * t, XG[0], XG[-1])
    j = np.clip(np.searchsorted(XG, x) - 1, 0, XG.size - 2)
    f = (x - XG[j]) / (XG[j + 1] - XG[j])
    gv = Gt[:, j] * (1 - f) + Gt[:, j + 1] * f
    pd = 1 - gv.mean(axis=1)
    return eta, s90_from_curve(pd)


@np.errstate(divide="ignore", invalid="ignore")
def pd_r(sdb, t, dv, eta, m0):
    s = 10 ** (sdb / 10)
    x = eta * t
    prod = np.ones_like(t)
    for f, g in enumerate(GAINS):
        T = dv[:, f]
        if g == 0:
            prod *= (T <= x)
            continue
        a = m0 * s * g
        z = (x - T - a) / (2 * np.sqrt(a * T))
        prod *= sp.betainc(M, M, np.clip((z + 1) / 2, 0, 1))
    return 1 - prod.mean()


def mode_r(t, dv, dmax, pfa, m0):
    Z = dmax / t
    eta = float(np.quantile(Z, 1 - pfa))
    s = brentq(lambda sdb: pd_r(sdb, t, dv, eta, m0) - 0.9, -14, 8, xtol=1e-4)
    return eta, s


def run_half(U, mask, pfa, Gt, ranks):
    TB, dv, dmax = U["TB"][mask], U["dv"][mask], U["dmax"][mask]
    m0 = float(np.median(U["med"][mask]))
    out = {}
    for r in ranks:
        t = TB[:, r - 1].astype(np.float64)
        ei, si = mode_i(t, pfa, Gt)
        er, sr = mode_r(t, dv, dmax, pfa, m0)
        fa_real = float(np.mean(dmax / t > ei))
        out[r] = dict(eta_I=ei, s_I=si, eta_R=er, s_R=sr, fa_real_of_eta_I=fa_real)
    sI = np.array([out[r]["s_I"] for r in ranks])
    sR = np.array([out[r]["s_R"] for r in ranks])
    for k, r in enumerate(ranks):
        out[r]["L_I"] = float(sI[k] - np.nanmin(sI))
        out[r]["L_R"] = float(sR[k] - np.nanmin(sR))
    return dict(frames=int(mask.sum()), m0=m0, best_I=int(ranks[int(np.nanargmin(sI))]), best_R=int(ranks[int(np.nanargmin(sR))]),
                s_best_I=float(np.nanmin(sI)), s_best_R=float(np.nanmin(sR)), by_rank=out)


def spread_diag(U, mask):
    TB = U["TB"][mask].astype(np.float64)
    n = TB.shape[1]
    q25, q75 = TB[:, int(round(0.25 * (n - 1)))], TB[:, int(round(0.75 * (n - 1)))]
    sp_ = (q75 - q25) / 1.349
    ideal = float(np.sqrt(2 * D2 ** 2 * (D1 + D2 - 2) / (D1 * (D2 - 2) ** 2 * (D2 - 4))))
    dv = U["dv"][mask]
    med = U["med"][mask]
    dsd = np.sqrt(np.mean((dv - med[:, None]) ** 2, axis=1))
    qs = np.quantile(sp_, [0.5, 0.9, 0.99])
    top = sp_ >= qs[2]
    # D-level check: mean of each D position relative to the bulk mean
    lvl = (dv.mean(axis=0) / TB.mean(axis=1).mean()).tolist()
    return dict(ideal_sd=ideal, bulk_spread_median=float(qs[0]), p90=float(qs[1]), p99=float(qs[2]),
                top1pct_mean_over_ideal=float(sp_[top].mean() / ideal),
                d_rms_dev_top1pct=float(dsd[top].mean()), d_rms_dev_rest=float(dsd[~top].mean()),
                d_rms_over_bulk_spread_top=float(np.mean(dsd[top] / sp_[top])), d_rms_over_bulk_spread_rest=float(np.mean(dsd[~top] / sp_[~top])),
                d_level_over_bulk_mean=lvl)


def bootstrap(U, mask, pfa, Gt, ranks, B=40, seed=7):
    rng = np.random.default_rng(seed)
    months = np.unique(U["month"][mask])
    idx = {m: np.flatnonzero(mask & (U["month"] == m)) for m in months}
    L = {"I": [], "R": []}
    for b in range(B):
        pick = rng.choice(months, size=months.size, replace=True)
        sel = np.concatenate([idx[m] for m in pick])
        TB, dv, dmax = U["TB"][sel], U["dv"][sel], U["dmax"][sel]
        m0 = float(np.median(U["med"][sel]))
        sI, sR = [], []
        for r in ranks:
            t = TB[:, r - 1].astype(np.float64)
            sI.append(mode_i(t, pfa, Gt)[1])
            sR.append(mode_r(t, dv, dmax, pfa, m0)[1])
        L["I"].append(sI)
        L["R"].append(sR)
    sI, sR = np.array(L["I"]), np.array(L["R"])
    out = {}
    for k, r in enumerate(ranks):
        # difference L(r) - L(ref) with ref = the last rank listed (paired)
        out[r] = dict(se_sI_minus_ref=float(np.std(sI[:, k] - sI[:, -1], ddof=1)),
                      se_sR_minus_ref=float(np.std(sR[:, k] - sR[:, -1], ddof=1)))
    return dict(B=B, ref_rank=ranks[-1], se=out)


def main(unit):
    t0 = time.time()
    U = load_unit(unit)
    print(unit, "frames", U["frames"], "bulk", U["bulk"], "split_match", U["split_match"], "data_match", U["data_match"],
          "near", U["near_cal"], U["near_eval"], f"{time.time() - t0:.0f}s", flush=True)
    Gt = g_table()
    print("table", f"{time.time() - t0:.0f}s", flush=True)
    n = U["bulk"]
    ranks = sorted(set(list(range(5, n + 1, 5)) + [1, 2, 41, 62, 63, 64] + list(range(60, 81)) + list(range(110, n + 1))))
    res = dict(unit=unit, frames=U["frames"], bulk=n, anchor=int(U["anchor"]), split_match=U["split_match"], data_match=U["data_match"],
               near_cal=U["near_cal"], near_eval=U["near_eval"], halves={})
    for half, hm in (("cal", U["cal"]), ("eval", ~U["cal"])):
        mask = hm & U["sel"]
        res["halves"][half] = {"spread": spread_diag(U, mask)}
        for pfa in (1e-3, 1e-2):
            r = run_half(U, mask, pfa, Gt, ranks)
            res["halves"][half][str(pfa)] = r
            b = r["by_rank"]
            print(unit, half, pfa, "best I/R", r["best_I"], r["best_R"], "L41 I/R", round(b[41]["L_I"], 3), round(b[41]["L_R"], 3),
                  "L63 I/R", round(b[63]["L_I"], 3), round(b[63]["L_R"], 3), "FA(eta_I) 41/63", round(b[41]["fa_real_of_eta_I"], 4),
                  round(b[63]["fa_real_of_eta_I"], 4), f"{time.time() - t0:.0f}s", flush=True)
    if unit == "37g21":
        res["bootstrap_cal_1e-3"] = bootstrap(U, U["cal"] & U["sel"], 1e-3, Gt, [41, 63, 75, 125])
        print("boot", res["bootstrap_cal_1e-3"], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / f"v_real_{unit}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
