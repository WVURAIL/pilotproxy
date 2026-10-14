"""Independent recompute of part 1 (bulk behaviour) for two channels and part 3 (channel 35 drift).

Own code: era cut from the ledger's current_first_month (UTC month of frame_time), own bulk mask, own T,
own ideal-law simulation, own class and valley logic. Release code used only for Product.selected,
Product.frame_time and the unit times (the health gate), from the release clone.
"""
import sys, json, glob, datetime
import numpy as np
HERE = "/home/djg/rail/output/cfar-choice-2026-09-29"
sys.path[:0] = [HERE + "/clones/RFIsher/src", HERE + "/clones/RadioFisher", HERE + "/clones/pilot-proxy/src"]
from rfisher_results.archive.products import Product

REL = "/home/djg/rail/results/archive_no_split_2026-09-24_r2"
PROD = "/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot"


def ideal(nb, n=300_000, seed=777):
    rng = np.random.default_rng(seed)
    r, u, s = [], [], []
    for _ in range(n // 30_000):
        t = rng.f(4096, 8192, size=(30_000, nb))
        m = np.median(t, 1)
        r.append(t.mean(1) / m); u.append(t.max(1) / m); s.append(t[:, 5] / m)
    r, u, s = map(np.concatenate, (r, u, s))
    return dict(r_med=np.median(r), r_p99=np.quantile(r, .99), r_p999=np.quantile(r, .999), r_max=r.max(),
                u_med=np.median(u), u_p99=np.quantile(u, .99), single_p999=np.quantile(s, .999))


def month(t):
    return np.array([datetime.datetime.fromtimestamp(x, datetime.timezone.utc).strftime("%Y-%m") if np.isfinite(x) else "" for x in t])


def load(ch):
    led = json.load(open(glob.glob(f"{REL}/archive/ledger/channels/ch{ch:02d}_fid*.json")[0]))
    s = led["sections"]
    p = Product(f"{PROD}/{led['product']}", require_health=True)
    ft = np.asarray(p.frame_time)
    mo = month(ft)
    era = p.selected & np.isfinite(ft) & (mo >= s["era"]["current_first_month"]) & (mo <= s["era"]["current_last_month"])
    rows = np.flatnonzero(era)
    ut = np.asarray(p.unit_time)[rows]
    cal = ut < s["blocks"]["boundary_time"]
    assert rows.size == s["blocks"]["calibration_frames"] + s["blocks"]["evaluation_frames"], (rows.size,)
    assert cal.sum() == s["blocks"]["calibration_frames"], (cal.sum(), s["blocks"]["calibration_frames"])
    f = np.asarray(p._z["fine_power_u64"])[rows].astype(np.float64)
    den = f[:, 1] + f[:, 2]
    T = np.where(den > 0, 2 * f[:, 0] / np.where(den > 0, den, 1), 0.0)
    a = int(s["null"]["anchor_bin"])
    bulk = np.zeros(256, bool); bulk[0::2] = True
    for k in range(-2, 3):
        bulk[(a + k) % 256] = False
    D = [(a + k) % 256 for k in range(-2, 3)]
    return dict(s=s, T=T, a=a, bulk=bulk, D=D, cal=cal, ft=ft[rows], zero_den=int((den[:, bulk] == 0).sum()))


def part1(ch, ideals):
    c = load(ch)
    T, a, bulk, D = c["T"], c["a"], c["bulk"], c["D"]
    bi = np.flatnonzero(bulk); nb = bi.size
    TB = T[:, bi]
    med = np.median(TB, 1)
    r = TB.mean(1) / med
    u = TB.max(1) / med
    off = (bi - a + 128) % 256 - 128
    near = np.abs(off) <= 7
    sp = ideals[nb]["single_p999"]
    fa = (TB > sp * med[:, None]).mean(0)
    pers = fa > 0.5
    dmax = T[:, D].max(1)
    on = dmax > 10 * med
    # top 3 removed by partition
    top3sum = -np.partition(-TB, 2, axis=1)[:, :3].sum(1)
    res = dict(ch=ch, anchor=a, parity="odd" if a % 2 else "even", nb=nb, frames=int(len(r)), zero_den_bulk=c["zero_den"],
               state=c["s"]["era"]["current_state"],
               med_r=np.median(r), p99_r=np.quantile(r, .99), f105=(r > 1.05).mean(), f125=(r > 1.25).mean(),
               med_u=np.median(u), p99_u=np.quantile(u, .99),
               f105_no_near=(TB[:, ~near].mean(1) / med > 1.05).mean(),
               f105_no_near_pers=(TB[:, ~near & ~pers].mean(1) / med > 1.05).mean(),
               f105_no_top3=((TB.sum(1) - top3sum) / (nb - 3) / med > 1.05).mean(),
               pers_bins=int(pers.sum()), pilot_on=on.mean(), f105_on=(r[on] > 1.05).mean() if on.any() else None,
               f105_off=(r[~on] > 1.05).mean() if (~on).any() else None,
               pers_offsets=[int(x) for x in off[pers]])
    return res, c, dict(med=med, r=r, dmax=dmax, TB=TB, bi=bi)


def valley(z, lo, hi, nbins=80):
    h, e = np.histogram(np.log10(z), bins=nbins, range=(np.log10(lo), np.log10(hi)))
    hs = np.convolve(h, np.ones(5) / 5, "same")
    cen = 0.5 * (e[1:] + e[:-1])
    mid = nbins // 2
    # two modes: highest bin in each of the two halves split at the global minimum between the two largest peaks
    i1 = int(np.argmax(hs))
    far = np.abs(cen - cen[i1]) >= 0.5
    i2 = int(np.flatnonzero(far)[np.argmax(hs[far])])
    lo_i, hi_i = sorted((i1, i2))
    iv = lo_i + int(np.argmin(hs[lo_i:hi_i + 1]))
    return 10 ** cen[iv], 10 ** cen[lo_i], 10 ** cen[hi_i]


def part3(c, x):
    T, a, bulk = c["T"], c["a"], c["bulk"]
    assert a == 107
    med, dmax, TB = x["med"], x["dmax"], x["TB"]
    srt63 = np.partition(TB, 62, axis=1)[:, 62]
    z63 = dmax / srt63
    zca = dmax / TB.mean(1)
    win = (a + np.arange(-30, 31)) % 256
    pk = win[np.argmax(T[:, win], 1)]
    pr = T[np.arange(len(T)), pk] / med
    car = pr >= 10
    inD = car & np.isin(pk, c["D"])
    inB = car & bulk[pk]
    els = car & ~inD & ~inB
    none = ~car
    gap = (float(pr[none].max()) if none.any() else None, float(pr[car].min()))
    v63 = valley(z63[car], 3, 1000); vca = valley(zca[car], 1, 300)
    out = dict(frames=len(T), gap=gap, valley63=v63, valleyca=vca, classes={})
    cal = c["cal"]
    for name, m in (("inD", inD), ("inB", inB), ("elsewhere", els), ("none", none)):
        row = dict(n=int(m.sum()), cal_share=(m & cal).sum() / cal.sum(), eva_share=(m & ~cal).sum() / (~cal).sum(),
                   med_r=float(np.median(x["r"][m])), med_z63=float(np.median(z63[m])), med_zca=float(np.median(zca[m])),
                   p01_zca=float(np.quantile(zca[m], .01)))
        for lab, e63, eca in (("valley", v63[0], vca[0]), ("eta5", 5, 5), ("eta10", 10, 10), ("ideal", 1.1005, 1.0999)):
            row[lab] = ((z63[m] <= e63).mean(), (zca[m] <= eca).mean())
        out["classes"][name] = row
    from scipy.stats import mannwhitneyu
    outD = inB | els
    for nm, z in (("Z63", z63), ("ZCA", zca)):
        U = mannwhitneyu(z[inD], z[outD]).statistic
        out["auc_" + nm] = U / (inD.sum() * outD.sum())
    return out


if __name__ == "__main__":
    ideals = {n: ideal(n) for n in (125, 126)}
    print("ideal", json.dumps({k: {kk: round(float(vv), 5) for kk, vv in v.items()} for k, v in ideals.items()}), flush=True)
    for ch in map(int, sys.argv[1:]):
        res, c, x = part1(ch, ideals)
        print("part1", json.dumps({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}), flush=True)
        if ch == 35:
            o = part3(c, x)
            print("part3", json.dumps(o, default=lambda v: round(float(v), 4)), flush=True)
