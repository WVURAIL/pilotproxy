"""Independent ideal-law control (verifier's own code, exact quadrature, no shared code with rank/).

Ideal i.i.d. F(4096, 8192) bins; D gains (0, g1, 1, g1, 0), g1 = D_128(1/2)^2; Pfa 1e-3 per detector on the clean
null; Pd 0.9. Rank detector: exact order-statistic density on a fine t grid. CA: Monte Carlo reference mean
(2e6 frames, histogrammed), clean threshold. Contaminated columns: k bulk bins at +20 dB always occupy the top k
ranks, so rank rho <= n - k is the rho-th order statistic of n - k clean bins, under the clean threshold.
Also: the design's screen constants s999 (clean median rank, Pd 0.999) and w_cut (0.999 point of w at s999).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq

D1, D2 = 4096, 8192
PFA, PD = 1e-3, 0.9
N_TONE = 128
g1_mine = float((np.sin(np.pi / 2) / (N_TONE * np.sin(np.pi / (2 * N_TONE)))) ** 2)
g1 = 0.4053050802342348  # the bench value (mine above differs by ~1e-5, inert)
GAINS = np.array([0.0, g1, 1.0, g1, 0.0])
HERE = Path(__file__).resolve().parent
TG = np.linspace(0.80, 1.25, 90001)
logf = np.log(np.maximum(sp.fdtr(D1, D2, TG), 1e-300))
logs = np.log(np.maximum(sp.fdtrc(D1, D2, TG), 1e-300))
pdf = np.gradient(sp.fdtr(D1, D2, TG), TG)


def os_density(n, r):
    lc = sp.gammaln(n + 1) - sp.gammaln(r) - sp.gammaln(n - r + 1)
    ld = lc + (r - 1) * logf + (n - r) * logs
    d = np.exp(ld) * pdf
    keep = d > d.max() * 1e-15
    t, w = TG[keep], d[keep]
    w = w * np.gradient(t)
    return t, w / w.sum(), float((d * np.gradient(TG)).sum())


def G(x, s):
    """P(max_D T <= x) with a tone of per-bin SNR s on the anchor."""
    out = np.ones_like(x)
    for g in GAINS:
        out = out * (sp.fdtr(D1, D2, x) if g == 0 else sp.ncfdtr(D1, D2, D1 * s * g, x))
    return out


def eta_for(t, w, pfa=PFA):
    return brentq(lambda e: (1 - np.sum(w * sp.fdtr(D1, D2, e * t) ** 5)) - pfa, 0.9, 1.6, xtol=1e-13)


def snr_db_for(t, w, eta, pd=PD):
    f = lambda sdb: (1 - np.sum(w * G(eta * t, 10 ** (sdb / 10)))) - pd
    return brentq(f, -14, 4, xtol=1e-9)


def ca_ref(n, frames=2_000_000, seed=20260930):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(frames // 100_000):
        x = rng.standard_gamma(D1 / 2, size=(100_000, n)) / (D1 / 2)
        y = rng.standard_gamma(D2 / 2, size=(100_000, n)) / (D2 / 2)
        means.append((x / y).mean(axis=1))
    m = np.concatenate(means)
    h, e = np.histogram(m, bins=6000)
    c = 0.5 * (e[1:] + e[:-1])
    return c[h > 0], h[h > 0] / h.sum(), m


def main():
    t0 = time.time()
    quad = json.loads((HERE.parent / "data/bench_quad.json").read_text())["sizes"]
    bench = json.loads((HERE.parent / "data/rank_bench.json").read_text())["sizes"]
    res = {"g1": g1}
    for n in (125, 126):
        cr, cw, m = ca_ref(n)
        eca = eta_for(cr, cw)
        sca = snr_db_for(cr, cw, eca)
        # batch spread of the CA MC (10 batches)
        sb = []
        for b in np.array_split(m, 10):
            h, e = np.histogram(b, bins=3000)
            c = 0.5 * (e[1:] + e[:-1])
            ww = h / h.sum()
            sb.append(snr_db_for(c, ww, eta_for(c, ww)))
        row = {"CA": {"eta": eca, "snr_db": sca, "mc_batch_se_db": float(np.std(sb, ddof=1) / np.sqrt(10))}, "rank": {}}
        etas = {}
        for r in range(1, n + 1):
            t, w, mass = os_density(n, r)
            e = eta_for(t, w)
            etas[r] = e
            s = snr_db_for(t, w, e)
            row["rank"][r] = {"eta": e, "snr_db": s, "loss_db": s - sca, "mass": mass}
        L = np.array([row["rank"][r]["loss_db"] for r in range(1, n + 1)])
        row["argmin"] = int(np.argmin(L) + 1)
        row["flat_0.0002"] = [int(r) for r in np.flatnonzero(L - L.min() <= 2e-4) + 1]
        # compare to bench quadrature and bench MC
        cmpq = {}
        for k, v in quad[str(n)].items():
            if k == "CA":
                cmpq["CA"] = {"quad_snr": v["snr_db"], "mine": sca, "d": sca - v["snr_db"]}
            else:
                cmpq[k] = {"quad_loss": v["loss_db"], "mine": row["rank"][int(k)]["loss_db"],
                           "d_loss": row["rank"][int(k)]["loss_db"] - v["loss_db"],
                           "d_eta": etas[int(k)] - v["eta"]}
        row["vs_quad"] = cmpq
        bc = bench[str(n)]["cases"]["clean"]
        dev = []
        for r in range(1, n + 1):
            b = bc[str(r)]
            dev.append((r, row["rank"][r]["loss_db"] - b["loss_db"], b["se_db"]))
        row["vs_bench_clean"] = {"max_abs_d": float(max(abs(d) for _, d, _ in dev)),
                                 "outside_2se": [(r, d, s) for r, d, s in dev if abs(d) > 2 * max(s, 1e-12)]}
        # contaminated columns: clean threshold, order statistic of n - k clean bins
        cont = {}
        for k in (1, 3, 10, 30, 54):
            col = bench[str(n)]["cases"][f"k{k}_20dB"]
            devk = []
            for r in (1, 10, 20, 30, 41, 50, 60, 63, 64, 70, n - k):
                if r > n - k:
                    continue
                t, w, _ = os_density(n - k, r)
                s = snr_db_for(t, w, etas[r])
                b = col.get(str(r))
                if b is None or b.get("loss_db") is None:
                    continue
                devk.append({"rank": r, "mine": s - sca, "bench": b["loss_db"], "se": b["se_db"], "d": s - sca - b["loss_db"]})
            cont[str(k)] = devk
        row["contaminated"] = cont
        res[str(n)] = row
        print(n, "CA", sca, "L41", row["rank"][41]["loss_db"], "L63", row["rank"][63]["loss_db"], "argmin", row["argmin"],
              "flat", row["flat_0.0002"][0], row["flat_0.0002"][-1], f"{time.time() - t0:.0f}s", flush=True)
    # screen constants (|B| = 125, median = rank 63)
    t, w, _ = os_density(125, 63)
    e63 = eta_for(t, w)
    s999 = brentq(lambda sdb: (1 - np.sum(w * G(e63 * t, 10 ** (sdb / 10)))) - 0.999, -12, 0, xtol=1e-10)
    s999l = 10 ** (s999 / 10)
    wcut = brentq(lambda x: np.sum(w * G(x * t, s999l)) - 0.999, 1.05, 2.0, xtol=1e-12)
    null999 = brentq(lambda x: np.sum(w * G(x * t, 0.0)) - 0.999, 1.0, 1.5, xtol=1e-12)
    null50 = brentq(lambda x: np.sum(w * G(x * t, 0.0)) - 0.5, 0.9, 1.5, xtol=1e-12)
    # alternative conventions, to show how the cut moves
    alt = {}
    for pdl in (0.99, 0.999, 0.9999):
        sl = brentq(lambda sdb: (1 - np.sum(w * G(e63 * t, 10 ** (sdb / 10)))) - pdl, -12, 0, xtol=1e-10)
        alt[str(pdl)] = {"s_db": sl, "w_cut": brentq(lambda x: np.sum(w * G(x * t, 10 ** (sl / 10))) - pdl, 1.02, 2.0, xtol=1e-12)}
    res["screen"] = {"eta63": e63, "s999_db": s999, "w_cut": wcut, "null_0.999": null999, "null_median": null50, "alt": alt}
    print("screen", res["screen"], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / "v_ideal.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
