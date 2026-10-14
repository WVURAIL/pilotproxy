"""Mode I machinery: thresholds and Pd = 0.9 SNR per reference column, exact in the designated bins.

Model (DESIGN.md section 2, mode I; the rank bench's model): the five designated bins are independent noncentral
F(4096, 8192, 4096 s g_f) with the tone on the anchor, gains (0, g1, 1, g1, 0), g1 = D_128(1/2)^2; the reference
is a real (or simulated) value per frame, independent of D. For a reference column R (frames) and frame weights w,
    Pfa(eta) = sum_i w_i [1 - prod_D F(eta R_i)] / sum_i w_i,
    Pd(s)    = 1 - sum_i w_i prod_D F_nc(eta R_i; s g_f) / sum_i w_i,
and s* is the SNR (dB per bin) where Pd = 0.9. prod_D F_nc is tabulated on x (step 1e-4, 0.80 to 2.40) by SNR
(step 0.005 dB, -10 to -2 dB); Pd is the table against the linear-interpolation histogram of eta R (the bench's
method). Beyond 2.40 every row of the table is 1 to double precision (the largest SNR's centre bin sits 17 sd
below 2.40), below 0.80 every row is 0 to 1e-19. Crossings above -2 dB are solved by brentq with the CDFs
evaluated at every frame (point estimates only; a bootstrap replicate outside the table is recorded as NaN).

The clean-null tail 1 - prod_D F is tabulated on a finer grid (step 1e-5). Point thresholds are solved by
bisection on it; bootstrap thresholds by interpolating log Pfa on a per-rank eta grid built from month sums.

Offsets: "d0" tone on the anchor (natural-bin offsets -1, -1/2, 0, 1/2, 1); "d05" half a padded bin off grid
(-1.25, -0.75, -0.25, 0.25, 0.75), the straddle worst case.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import f as fdist
from scipy.stats import ncf

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "data/tables"
L = 128
D1, D2 = 4096, 8192
PD = 0.9
S0_DB = -8.631
SEED = 20260907
REPLICATES = 1000
X0, XH, NX = 0.80, 1e-4, 16001
XG = X0 + XH * np.arange(NX)
S0G, SH, NS = -10.0, 0.005, 1601
SG = np.round(S0G + SH * np.arange(NS), 6)
XF0, XFH, NXF = 0.80, 1e-5, 160001
XF = XF0 + XFH * np.arange(NXF)


def dirichlet2(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = (np.sin(np.pi * x[nz]) / (L * np.sin(np.pi * x[nz] / L))) ** 2
    return out


OFFSETS = {"d0": dirichlet2([-1.0, -0.5, 0.0, 0.5, 1.0]), "d05": dirichlet2([-1.25, -0.75, -0.25, 0.25, 0.75])}
G1 = float(OFFSETS["d0"][1])


def dprod(x, s, gains):
    """P(max_D T <= x) at linear SNR s with the given D gains (gains below 1e-12 are pure noise bins)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    vals, counts = np.unique(np.round(np.asarray(gains, dtype=float), 12), return_counts=True)
    for g, k in zip(vals, counts):
        c = fdist.cdf(x, D1, D2) if (g < 1e-12 or s == 0) else ncf.cdf(x, D1, D2, D1 * s * g)
        out = out * c ** k
    return out


def build_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    sf0 = fdist.sf(XF, D1, D2)
    np.save(TABLES / "sf5_fine.npy", -np.expm1(5.0 * np.log1p(-sf0)))
    for name, gains in OFFSETS.items():
        path = TABLES / f"tab_{name}.npy"
        if path.exists():
            continue
        t0 = time.time()
        tab = np.empty((NS, NX))
        for j, sdb in enumerate(SG):
            tab[j] = dprod(XG, 10 ** (sdb / 10), gains)
        assert np.all(np.abs(tab[:, -1] - 1.0) < 1e-12), name
        assert np.all(tab[:, 0] < 1e-12), name
        np.save(path, tab)
        np.save(TABLES / f"row_s0_{name}.npy", dprod(XG, 10 ** (S0_DB / 10), gains))
        print(f"table {name} {tab.shape} {time.time() - t0:.0f} s", flush=True)


class Tables:
    def __init__(self):
        self.sf5 = np.load(TABLES / "sf5_fine.npy")
        self.tab = {k: np.load(TABLES / f"tab_{k}.npy", mmap_mode="r") for k in OFFSETS}
        self.row_s0 = {k: np.load(TABLES / f"row_s0_{k}.npy") for k in OFFSETS}

    def sf5_at(self, x):
        u = (x - XF0) / XFH
        i = np.clip(np.floor(u).astype(np.int64), 0, NXF - 2)
        fr = np.clip(u - i, 0.0, 1.0)
        return self.sf5[i] * (1.0 - fr) + self.sf5[i + 1] * fr


def histogram(x, w, k):
    """Linear-interpolation weights of x (n, k) on XG, per column: (k, NX)."""
    u = (x - X0) / XH
    i = np.clip(np.floor(u).astype(np.int64), 0, NX - 2)
    fr = np.clip(u - i, 0.0, 1.0)
    base = (np.arange(k, dtype=np.int64) * NX)[None, :]
    idx = (i + base).ravel()
    ww = np.broadcast_to(w[:, None], x.shape)
    h = np.bincount(idx, (ww * (1.0 - fr)).ravel(), minlength=k * NX)
    h += np.bincount(idx + 1, (ww * fr).ravel(), minlength=k * NX)
    return h.reshape(k, NX)


def crossing(pd, rows):
    """Linear crossing of pd (len(rows)) through PD; None when it is not bracketed inside rows."""
    above = np.flatnonzero(pd >= PD)
    if not above.size or above[0] == 0:
        return None
    j = above[0]
    s_lo, s_hi = SG[rows[j - 1]], SG[rows[j]]
    return float(s_lo + (PD - pd[j - 1]) * (s_hi - s_lo) / (pd[j] - pd[j - 1]))


def direct_snr(eta, ref, w, gains, lo=-2.5, hi=45.0):
    x = eta * ref
    W = w.sum()

    def f(sdb):
        return 1.0 - np.dot(w, dprod(x, 10 ** (sdb / 10), gains)) / W - PD
    if f(hi) < 0:
        return float("inf")
    return float(brentq(f, lo, hi, xtol=1e-5))


class ModeI:
    """Reference matrix R (n frames x k columns), month labels, weights 1 on every frame by default."""

    def __init__(self, R, month, tables, pfas=(1e-3, 1e-2), offsets=("d0", "d05")):
        order = np.argsort(month, kind="stable")
        self.R = np.ascontiguousarray(R[order], dtype=np.float64)
        self.month = np.asarray(month)[order]
        self.order = order
        self.n, self.k = self.R.shape
        self.T = tables
        self.pfas, self.offsets = tuple(pfas), tuple(offsets)
        self.months, self.minv = np.unique(self.month, return_inverse=True)
        self.starts = np.flatnonzero(np.r_[True, self.month[1:] != self.month[:-1]])
        self.point = None

    # thresholds -----------------------------------------------------------------------------------------
    def pfa_curve(self, eta, w):
        return (w[:, None] * self.T.sf5_at(eta[None, :] * self.R)).sum(0) / w.sum()

    def eta_bisect(self, w, pfa, lo=0.5, hi=20.0, it=70):
        lo = np.full(self.k, lo)
        hi = np.full(self.k, hi)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            p = self.pfa_curve(mid, w)
            big = p > pfa
            lo = np.where(big, mid, lo)
            hi = np.where(big, hi, mid)
        return 0.5 * (lo + hi)

    def build_eta_grid(self, eta_lo, eta_hi, K=241, pad=0.06):
        """Month sums of the clean tail on a per-column eta grid spanning [eta_lo, eta_hi] with a relative pad."""
        g = np.exp(np.linspace(np.log(eta_lo) - pad, np.log(eta_hi) + pad, K))    # (K, k)
        G = np.empty((self.months.size, self.k, K))
        for j in range(K):
            v = self.T.sf5_at(g[j][None, :] * self.R)
            G[:, :, j] = np.add.reduceat(v, self.starts, axis=0)
        self.eta_grid, self.G = g, G

    def eta_from_grid(self, wm, pfa):
        """wm: weight per month. Returns eta per column (NaN where the grid does not bracket the target)."""
        W = float(np.dot(wm, np.bincount(self.minv, minlength=self.months.size)))
        p = np.einsum("m,mkj->kj", wm, self.G) / W                  # (k, K), decreasing in j
        lp = np.log(np.maximum(p, 1e-300))
        t = np.log(pfa)
        below = lp <= t
        j = np.argmax(below, axis=1)
        ok = below.any(axis=1) & (j > 0)
        j = np.where(ok, j, 1)
        cols = np.arange(self.k)
        l0, l1 = lp[cols, j - 1], lp[cols, j]
        e0, e1 = np.log(self.eta_grid[j - 1, cols]), np.log(self.eta_grid[j, cols])
        fr = (t - l0) / (l1 - l0)
        eta = np.exp(e0 + fr * (e1 - e0))
        return np.where(ok, eta, np.nan)

    # Pd -------------------------------------------------------------------------------------------------
    def snr(self, eta, w, offset, windows=None, direct_ok=True):
        """s* (dB) per column; windows: per column row range (lo, hi) to try first."""
        H = histogram(eta[None, :] * self.R, w, self.k)
        W = w.sum()
        tab = self.T.tab[offset]
        out = np.full(self.k, np.nan)
        if windows is None:
            pd = 1.0 - (np.asarray(tab) @ H.T) / W                    # (NS, k)
            rows = np.arange(NS)
            for c in range(self.k):
                v = crossing(pd[:, c], rows)
                out[c] = np.nan if v is None else v
        else:
            for c in range(self.k):
                lo, hi = windows[c]
                v = None
                if hi <= lo:                      # the point estimate lies beyond the table: not solved here
                    out[c] = np.nan
                    continue
                for a, b in ((lo, hi), (max(lo - 200, 0), min(hi + 200, NS)), (0, NS)):
                    v = crossing(1.0 - (tab[a:b] @ H[c]) / W, np.arange(a, b))
                    if v is not None:
                        break
                out[c] = np.nan if v is None else v
        if direct_ok:
            for c in np.flatnonzero(~np.isfinite(out)):
                ww = w[w > 0]
                out[c] = direct_snr(eta[c], self.R[w > 0, c], ww.astype(float), OFFSETS[offset])
        return out, H

    def run_point(self):
        w = np.ones(self.n)
        res = {}
        for pfa in self.pfas:
            eta = self.eta_bisect(w, pfa)
            res[("eta", pfa)] = eta
            for off in self.offsets:
                s, H = self.snr(eta, w, off)
                res[("s", pfa, off)] = s
                res[("pd_s0", pfa, off)] = 1.0 - H @ self.T.row_s0[off] / w.sum()
        self.point = res
        return res

    def replicate_weights(self, seed_key, replicates=REPLICATES):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_key]))
        U = self.months.size
        for _ in range(replicates):
            draw = rng.integers(0, U, size=U)
            yield np.bincount(draw, minlength=U).astype(float)

    def run_boot(self, seed_key, replicates=REPLICATES, halfwin_db=0.12):
        assert self.point is not None
        etas = np.stack([self.point[("eta", p)] for p in self.pfas])
        self.build_eta_grid(np.nanmin(etas, 0), np.nanmax(etas, 0))
        win = {}
        for pfa in self.pfas:
            for off in self.offsets:
                s = self.point[("s", pfa, off)]
                lo = np.clip(np.floor((s - halfwin_db - S0G) / SH).astype(np.int64), 0, NS - 1)
                hi = np.clip(np.ceil((s + halfwin_db - S0G) / SH).astype(np.int64) + 1, 0, NS)
                bad = ~np.isfinite(s) | (s > SG[-1] - 0.25)
                win[(pfa, off)] = [(0, 0) if b else (int(a), int(c)) for a, c, b in zip(lo, hi, bad)]
        out = {(p, o): np.full((replicates, self.k), np.nan, dtype=np.float32) for p in self.pfas for o in self.offsets}
        eta_out = {p: np.full((replicates, self.k), np.nan, dtype=np.float32) for p in self.pfas}
        fallback = 0
        for r, wm in enumerate(self.replicate_weights(seed_key, replicates)):
            w = wm[self.minv]
            for pfa in self.pfas:
                eta = self.eta_from_grid(wm, pfa)
                miss = ~np.isfinite(eta)
                if miss.any():
                    fallback += int(miss.sum())
                    eta[miss] = ModeI._bisect_cols(self, w, pfa, np.flatnonzero(miss))
                eta_out[pfa][r] = eta
                for off in self.offsets:
                    s, _ = self.snr(eta, w, off, windows=win[(pfa, off)], direct_ok=False)
                    out[(pfa, off)][r] = s
        return out, eta_out, fallback

    def _bisect_cols(self, w, pfa, cols, it=60):
        R = self.R[:, cols]
        lo, hi = np.full(cols.size, 0.5), np.full(cols.size, 20.0)
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            p = (w[:, None] * self.T.sf5_at(mid[None, :] * R)).sum(0) / w.sum()
            big = p > pfa
            lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
        return 0.5 * (lo + hi)


if __name__ == "__main__":
    build_tables()
