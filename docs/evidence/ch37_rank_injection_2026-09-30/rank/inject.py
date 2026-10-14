"""Mode I on one channel (or one channel-37 geometry): thresholds, Pd at s0 and the Pd = 0.9 SNR per rank, point
and calendar-month block bootstrap, per half (DESIGN.md sections 2 and 3).

Screens:
  w      near-threshold frames, w = max_D T / median_B T <= w_cut (primary);
  wlo    near-threshold by w_lo = max_D T / m_lo <= w_cut (sensitivity);
  ideal  the w-selected frames with every bulk replaced by i.i.d. F(4096, 8192) draws (same |B|, month and half):
         the ideal control at the channel's own frame counts;
  all    channel 37 at one channel's geometry (name 37gCC): all 40,477 frames.
Halves: cal and eval (the release's split; channel 37 its own), and pool (both) for the diagnostic channels.
A half is run when it holds at least MIN_FRAMES frames in at least 2 months.

Also written for the w screen: the strong-pilot miss count per rank (mid and on frames whose real
Z(rho) = max_D T / T_(rho) falls below the channel's calibrated eta(rho) at Pfa 1e-3) and eps(rho).

Usage (analysis interpreter, one thread, no bytecode): python -B inject.py NAME SCREEN [REPLICATES]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import core

ROOT = core.ROOT
FRAMES, OUT = ROOT / "data/frames", ROOT / "data/inject"
W_CUT = json.loads((ROOT / "design/design_counts.json").read_text())["ideal"]["w_cut"]
MIN_FRAMES = 20
HALF_CODE = {"cal": 1, "eval": 2, "pool": 3}


def select(z, screen):
    if screen in ("w", "ideal"):
        return z["w"] <= W_CUT
    if screen == "wlo":
        return z["w_lo"] <= W_CUT
    if screen == "all":
        return np.ones(z["w"].size, dtype=bool)
    raise ValueError(screen)


def main(name, screen, replicates):
    t0 = time.time()
    ch = 37 if name.startswith("37") else int(name)
    z = np.load(FRAMES / (f"ch37_g{name[3:]}.npz" if ch == 37 else f"ch{ch:02d}.npz"))
    S = z["sorted_bulk"].astype(np.float64)
    n_all, nb = S.shape
    sel = select(z, screen)
    cal = z["calibration"].astype(bool)
    month = z["month"]
    if screen == "ideal":
        rng = np.random.default_rng(np.random.SeedSequence([core.SEED, ch, 99]))
        S = np.sort(rng.f(core.D1, core.D2, size=(n_all, nb)), axis=1)
    tabs = core.Tables()
    out, meta = {}, dict(name=name, channel=ch, screen=screen, bulk_size=nb, frames=int(n_all), selected=int(sel.sum()),
                         replicates=replicates, halves={})
    halves = {"cal": sel & cal, "eval": sel & ~cal}
    diagnostic = ch != 37 and min(int(halves["cal"].sum()), int(halves["eval"].sum())) < 500
    if diagnostic:
        halves["pool"] = sel
    for half, m in halves.items():
        n = int(m.sum())
        nm = int(np.unique(month[m]).size) if n else 0
        meta["halves"][half] = dict(frames=n, months=nm)
        if n < MIN_FRAMES or nm < 2:
            continue
        mi = core.ModeI(S[m], month[m], tabs)
        pt = mi.run_point()
        for p in mi.pfas:
            out[f"{half}_eta_{p:g}"] = pt[("eta", p)]
            for o in mi.offsets:
                out[f"{half}_s_{p:g}_{o}"] = pt[("s", p, o)]
                out[f"{half}_pds0_{p:g}_{o}"] = pt[("pd_s0", p, o)]
        tb = time.time()
        bs, be, fb = mi.run_boot((ch, HALF_CODE[half]), replicates)
        for (p, o), v in bs.items():
            out[f"{half}_bs_{p:g}_{o}"] = v
        for p, v in be.items():
            out[f"{half}_beta_{p:g}"] = v
        meta["halves"][half].update(eta_grid_fallbacks=fb, boot_seconds=round(time.time() - tb, 1),
                                    boot_nan=int(sum(int(np.isnan(v).sum()) for v in bs.values())))
        s = pt[("s", 1e-3, "d0")]
        print(half, n, nm, "best rank", int(np.nanargmin(s[:125])) + 1, f"s41 {s[40]:.4f} s63 {s[62]:.4f} min {np.nanmin(s):.4f}",
              f"boot {time.time() - tb:.0f} s", flush=True)
    if screen == "w" and ch != 37:
        # strong-pilot misses: calibrated eta (cal half, or pooled for the diagnostic channels, else the ideal null)
        src = "cal" if f"cal_eta_0.001" in out and not diagnostic else ("pool" if "pool_eta_0.001" in out else None)
        if src is not None:
            eta = out[f"{src}_eta_0.001"]
        else:
            ideal = core.ModeI(np.sort(np.random.default_rng(7).f(core.D1, core.D2, size=(200_000, nb)), axis=1),
                               np.zeros(200_000, dtype=int), tabs, pfas=(1e-3,), offsets=())
            eta = ideal.eta_bisect(np.ones(200_000), 1e-3)
            src = "ideal"
        Zr = z["dmax"][:, None] / S
        mid = (z["w"] > W_CUT) & (z["w"] <= 10)
        on = z["w"] > 10
        out["miss_mid"] = (Zr[mid] <= eta[None, :]).sum(0)
        out["miss_on"] = (Zr[on] <= eta[None, :]).sum(0)
        near = sel
        m0r = np.median(S[near], axis=0) if near.sum() >= MIN_FRAMES else np.full(nb, np.nan)
        out["m0_rho"] = m0r
        if on.sum() > 30:
            out["eps_rho"] = np.median((S[on] - m0r[None, :]) / (z["dmax"][on][:, None] - m0r[None, :]), axis=0)
        meta.update(miss_eta_source=src, mid_frames=int(mid.sum()), on_frames=int(on.sum()))
    meta["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_{screen}.npz", **out)
    (OUT / f"{name}_{screen}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else core.REPLICATES)
