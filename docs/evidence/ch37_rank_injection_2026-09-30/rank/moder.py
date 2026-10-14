"""Mode R cross-check: real bulk, real designated bins, a tone added exactly given each bin's observed power
(DESIGN.md section 2). Used only where D is pilot-free: channel 37 at every channel geometry (all frames) and the
near-threshold frames of the carrier-off channels 19, 26, 27 and 32.

Given the real value T_f of a designated bin, a coherent tone of per-bin SNR s gives
    T_f' = T_f + c s g_f + 2 u sqrt(c s g_f T_f),   u = 2V - 1,  V ~ Beta((k-1)/2, (k-1)/2),  k = 4096,
c = m0 (median over the selected frames of median_B T), gains (0, g1, 1, g1, 0) with the tone on the anchor.
NDRAW independent u per frame and bin, fixed across ranks and replicates (common random numbers).
Threshold: the empirical upper Pfa point of the real Z(rho) = max_D T / T_(rho) on the half's frames (weighted by
the replicate's month counts). Pd(s) = P(max_D T' > eta T_(rho)); T_f'(s) increases with s = a^2 once a exceeds
|u| sqrt(T/(c g)), so each frame and draw has a critical SNR, the least root over D of
    c g a^2 + 2 u sqrt(c g T) a + (T - x) = 0,  x = eta T_(rho)
(zero when T_f > x already), and s* is the weighted 0.9 quantile of the critical SNRs (histogram, 0.001 dB bins).
Month weights are those of inject.py for the same (channel, half), so the two modes are paired.

Usage (analysis interpreter, one thread, no bytecode): python -B moder.py NAME SCREEN [REPLICATES]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import core
from inject import FRAMES, HALF_CODE, MIN_FRAMES, select

ROOT = core.ROOT
OUT = ROOT / "data/moder"
NDRAW = 4
PFAS = (1e-3, 1e-2)
DB_LO, DB_STEP, DB_N = -30.0, 0.001, 50001


class ModeR:
    def __init__(self, S, dvals, month, c, seed):
        order = np.argsort(month, kind="stable")
        self.S = np.ascontiguousarray(S[order], dtype=np.float64)
        self.dv = np.asarray(dvals, dtype=np.float64)[order]
        self.month = np.asarray(month)[order]
        self.months, self.minv = np.unique(self.month, return_inverse=True)
        self.n, self.k = self.S.shape
        self.c = float(c)
        self.Z = self.dv.max(1)[:, None] / self.S
        self.zord = np.argsort(-self.Z, axis=0, kind="stable")
        self.Zs = np.take_along_axis(self.Z, self.zord, axis=0)
        g = core.OFFSETS["d0"]
        self.act = np.flatnonzero(g > 1e-12)
        rng = np.random.default_rng(seed)
        u = 2.0 * rng.beta((core.D1 - 1) / 2.0, (core.D1 - 1) / 2.0, size=(self.n, NDRAW, self.act.size)) - 1.0
        T = self.dv[:, self.act][:, None, :]                              # (n, 1, a)
        self.T_act = T
        self.uT = u * np.sqrt(T)                                           # u sqrt(T)
        self.u2T = self.uT ** 2
        self.scale = np.sqrt(self.c * g[self.act])[None, None, :]
        self.dmax_all = self.dv.max(1)

    def eta(self, w, pfa):
        ws = w[self.zord]
        cum = np.cumsum(ws, axis=0)
        j = np.argmax(cum > pfa * w.sum(), axis=0)
        return self.Zs[j, np.arange(self.k)]

    def snr_col(self, x, w):
        """Weighted 0.9 quantile (dB) of the critical SNR for thresholds x (n,)."""
        xx = x[:, None, None]
        disc = self.u2T + xx - self.T_act
        a = (-self.uT + np.sqrt(np.maximum(disc, 0.0))) / self.scale
        s = (a * a).min(axis=2)                                            # (n, NDRAW)
        already = (self.dmax_all > x)[:, None]
        s = np.where(already, 0.0, s)
        with np.errstate(divide="ignore"):
            sdb = 10.0 * np.log10(s)
        idx = np.clip(np.floor((sdb - DB_LO) / DB_STEP).astype(np.int64), 0, DB_N - 1)
        h = np.bincount(idx.ravel(), np.repeat(w, NDRAW), minlength=DB_N)
        cum = np.cumsum(h)
        tgt = core.PD * cum[-1]
        j = int(np.searchsorted(cum, tgt, side="left"))
        prev = cum[j - 1] if j > 0 else 0.0
        fr = (tgt - prev) / h[j] if h[j] > 0 else 0.0
        return DB_LO + DB_STEP * (j + fr)

    def run(self, w):
        out = {}
        for pfa in PFAS:
            eta = self.eta(w, pfa)
            out[("eta", pfa)] = eta
            out[("s", pfa)] = np.array([self.snr_col(eta[c] * self.S[:, c], w) for c in range(self.k)])
        return out


def main(name, screen, replicates):
    t0 = time.time()
    ch = 37 if name.startswith("37") else int(name)
    z = np.load(FRAMES / (f"ch37_g{name[3:]}.npz" if ch == 37 else f"ch{ch:02d}.npz"))
    sel = select(z, screen)
    cal = z["calibration"].astype(bool)
    c = float(np.median(z["med"][sel]))
    out, meta = {}, dict(name=name, channel=ch, screen=screen, c=c, ndraw=NDRAW, replicates=replicates, halves={})
    for half, m in (("cal", sel & cal), ("eval", sel & ~cal)):
        n = int(m.sum())
        nm = int(np.unique(z["month"][m]).size) if n else 0
        meta["halves"][half] = dict(frames=n, months=nm)
        if n < MIN_FRAMES or nm < 2:
            continue
        mr = ModeR(z["sorted_bulk"][m], z["dvals"][m], z["month"][m], c, seed=[core.SEED, ch, HALF_CODE[half], 77])
        pt = mr.run(np.ones(mr.n))
        for p in PFAS:
            out[f"{half}_eta_{p:g}"] = pt[("eta", p)]
            out[f"{half}_s_{p:g}"] = pt[("s", p)]
        # the same month draws as inject.py for this (channel, half)
        rng = np.random.default_rng(np.random.SeedSequence([core.SEED, ch, HALF_CODE[half]]))
        U = mr.months.size
        bs = {p: np.full((replicates, mr.k), np.nan, dtype=np.float32) for p in PFAS}
        tb = time.time()
        for r in range(replicates):
            wm = np.bincount(rng.integers(0, U, size=U), minlength=U).astype(float)
            res = mr.run(wm[mr.minv])
            for p in PFAS:
                bs[p][r] = res[("s", p)]
        for p in PFAS:
            out[f"{half}_bs_{p:g}"] = bs[p]
        s = pt[("s", 1e-3)]
        meta["halves"][half]["boot_seconds"] = round(time.time() - tb, 1)
        print(half, n, nm, "best", int(np.argmin(s[:125])) + 1, f"s41 {s[40]:.4f} s63 {s[62]:.4f}", f"{time.time() - tb:.0f} s", flush=True)
    meta["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_{screen}.npz", **out)
    (OUT / f"{name}_{screen}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else core.REPLICATES)
