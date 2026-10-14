"""Controls summary (DESIGN.md section 4): channel 37 at the 23 channel geometries and the carrier-off channels,
mode I against mode R; channel 37's own selection; the real false-alarm rate of mode I's threshold on real D.

Per unit (geometry or channel), half and Pfa:
  L(rho) = s*(rho) - min over rho' <= 125 of s*(rho'), per mode, point and per bootstrap replicate (replicates
  outside the table count as +inf); SE = std over replicates; the modes are paired (same month draws).
  Agreement at rho: |L_I - L_R| <= 2 sqrt(SE_I^2 + SE_R^2).
Channel 37's own procedure (mode I and, for comparison, mode R): J(rho) = median over the 23 geometries of L,
W(rho) = max; rho* = argmin J on the calibration half; tie set J(rho) - J(rho*) <= 2 SE of the paired difference;
least W inside it, then least clean-bench cost; held-out J, W and argmin; budget W_eval(rho*) < 0.9 dB.
Real false-alarm rate: the share of a half's frames whose real Z(rho) = max_D T / T_(rho) exceeds mode I's
eta(rho) at Pfa 1e-3 (mode I assumes ideal D; on a true null this should be 1e-3).
Writes data/controls_summary.json. Usage (analysis interpreter): python -B summarize_controls.py
"""
from __future__ import annotations

import json

import numpy as np

import core

ROOT = core.ROOT
INJ, MR, FR = ROOT / "data/inject", ROOT / "data/moder", ROOT / "data/frames"
W_CUT = json.loads((ROOT / "design/design_counts.json").read_text())["ideal"]["w_cut"]
GEOMS = list(range(14, 37))
CARRIER_OFF = (19, 26, 27, 32)
NR = 125
BUDGET = 0.9
BENCH = json.loads((ROOT / "data/rank_bench.json").read_text())["sizes"]


def loss(s):
    s = np.where(np.isfinite(s), s, np.inf)[..., :NR]
    return s - s.min(axis=-1, keepdims=True)


def unit(name, screen):
    zi, zr = np.load(INJ / f"{name}_{screen}.npz"), np.load(MR / f"{name}_{screen}.npz")
    out = {}
    for half in ("cal", "eval"):
        for p in ("0.001", "0.01"):
            if f"{half}_s_{p}_d0" not in zi.files or f"{half}_s_{p}" not in zr.files:
                continue
            LI, LR = loss(zi[f"{half}_s_{p}_d0"]), loss(zr[f"{half}_s_{p}"])
            bI, bR = loss(zi[f"{half}_bs_{p}_d0"].astype(float)), loss(zr[f"{half}_bs_{p}"].astype(float))
            fin = np.isfinite(bI) & np.isfinite(bR)
            seI = np.array([bI[fin[:, r], r].std(ddof=1) if fin[:, r].sum() > 10 else np.inf for r in range(NR)])
            seR = np.array([bR[fin[:, r], r].std(ddof=1) if fin[:, r].sum() > 10 else np.inf for r in range(NR)])
            d = bI - bR
            seD = np.array([d[fin[:, r], r].std(ddof=1) if fin[:, r].sum() > 10 else np.inf for r in range(NR)])
            comb = np.hypot(seI, seR)
            agree = np.abs(LI - LR) <= 2 * comb
            out[f"{half}_{p}"] = dict(
                argmin_I=int(np.argmin(LI)) + 1, argmin_R=int(np.argmin(LR)) + 1,
                L_I={r: float(LI[r - 1]) for r in (41, 63, 94, 125)}, L_R={r: float(LR[r - 1]) for r in (41, 63, 94, 125)},
                se_I={r: float(seI[r - 1]) for r in (41, 63, 94, 125)}, se_R={r: float(seR[r - 1]) for r in (41, 63, 94, 125)},
                agree={r: bool(agree[r - 1]) for r in (41, 63, 94, 125)}, agree_fraction=float(agree.mean()),
                z_paired={r: float((LI[r - 1] - LR[r - 1]) / seD[r - 1]) for r in (41, 63, 94, 125)},
                L_I_curve=LI.tolist(), L_R_curve=LR.tolist(), se_I_curve=seI.tolist(), se_R_curve=seR.tolist(),
                s_I_min=float(np.nanmin(zi[f"{half}_s_{p}_d0"][:NR])), s_R_min=float(np.nanmin(zr[f"{half}_s_{p}"][:NR])))
    return out


def real_pfa(name, screen):
    fr = np.load(FR / (f"ch37_g{name[3:]}.npz" if name.startswith("37") else f"ch{int(name):02d}.npz"))
    zi = np.load(INJ / f"{name}_{screen}.npz")
    sel = np.ones(fr["w"].size, bool) if screen == "all" else fr["w"] <= W_CUT
    cal = fr["calibration"].astype(bool)
    out = {}
    for half, m in (("cal", sel & cal), ("eval", sel & ~cal)):
        if f"{half}_eta_0.001" not in zi.files:
            continue
        eta = zi[f"{half}_eta_0.001"][:NR]
        Z = fr["dmax"][m][:, None] / fr["sorted_bulk"][m][:, :NR].astype(float)
        f = (Z > eta[None, :]).mean(0)
        out[half] = dict(frames=int(m.sum()), by_rank={r: float(f[r - 1]) for r in (1, 20, 41, 63, 94, 110, 125)},
                         curve=f.tolist())
    return out


def select(Lc, Lb, bench_cost):
    """Lc: (units, NR) point losses; Lb: (units, R, NR) replicate losses. J = median over units."""
    J, W = np.median(Lc, 0), Lc.max(0)
    Jb = np.median(Lb, 0)                                   # (R, NR)
    r0 = int(np.argmin(J))
    diff = Jb - Jb[:, [r0]]
    se = np.array([np.std(diff[np.isfinite(diff[:, r]), r], ddof=1) if np.isfinite(diff[:, r]).sum() > 10 else 0.0
                   for r in range(NR)])
    se[r0] = 0.0
    tie = [r for r in range(NR) if J[r] - J[r0] <= 2 * se[r]]
    wmin = min(W[r] for r in tie)
    cand = [r for r in tie if W[r] == wmin]
    pick = min(cand, key=lambda r: bench_cost[r])
    return dict(argmin_J=r0 + 1, tie_set=[r + 1 for r in tie], rho_star=pick + 1, J=J.tolist(), W=W.tolist(),
                se_diff=se.tolist())


def ch37_selection(mode, p):
    Lc, Lb = {}, {}
    for half in ("cal", "eval"):
        pts, reps = [], []
        for g in GEOMS:
            if mode == "I":
                z = np.load(INJ / f"37g{g}_all.npz")
                pts.append(loss(z[f"{half}_s_{p}_d0"]))
                reps.append(loss(z[f"{half}_bs_{p}_d0"].astype(float)))
            else:
                z = np.load(MR / f"37g{g}_all.npz")
                pts.append(loss(z[f"{half}_s_{p}"]))
                reps.append(loss(z[f"{half}_bs_{p}"].astype(float)))
        Lc[half], Lb[half] = np.array(pts), np.array(reps)
    cost = [BENCH["125"]["cases"]["clean"][str(r + 1)]["loss_db"] for r in range(NR)]
    cal = select(Lc["cal"], Lb["cal"], cost)
    ev = select(Lc["eval"], Lb["eval"], cost)
    rs = cal["rho_star"] - 1
    return dict(calibration=cal, held_out=ev, rho_star=cal["rho_star"], rho_star_in_held_out_tie=cal["rho_star"] in ev["tie_set"],
                J_eval_at_rho_star=ev["J"][rs], W_eval_at_rho_star=ev["W"][rs], budget_db=BUDGET,
                budget_pass=bool(ev["W"][rs] < BUDGET),
                at={r: dict(J_cal=cal["J"][r - 1], W_cal=cal["W"][r - 1], J_eval=ev["J"][r - 1], W_eval=ev["W"][r - 1])
                    for r in (41, 63, cal["rho_star"])})


def main():
    out = dict(units={}, real_pfa={}, ch37={})
    for g in GEOMS:
        out["units"][f"37g{g}"] = unit(f"37g{g}", "all")
        out["real_pfa"][f"37g{g}"] = real_pfa(f"37g{g}", "all")
    for c in CARRIER_OFF:
        out["units"][str(c)] = unit(str(c), "w")
        out["real_pfa"][str(c)] = real_pfa(str(c), "w")
    for mode in ("I", "R"):
        for p in ("0.001", "0.01"):
            out["ch37"][f"{mode}_{p}"] = ch37_selection(mode, p)
    # agreement tallies
    tall = {}
    for key in ("cal_0.001", "eval_0.001", "cal_0.01", "eval_0.01"):
        rows = [u[key] for u in out["units"].values() if key in u]
        tall[key] = dict(units=len(rows), **{f"agree_{r}": int(sum(x["agree"][r] for x in rows)) for r in (41, 63, 94, 125)},
                         median_agree_fraction=float(np.median([x["agree_fraction"] for x in rows])),
                         argmin_I=[x["argmin_I"] for x in rows], argmin_R=[x["argmin_R"] for x in rows])
    out["agreement"] = tall
    (ROOT / "data/controls_summary.json").write_text(json.dumps(out) + "\n")
    for k, v in tall.items():
        print(k, {a: b for a, b in v.items() if not a.startswith("argmin")}, "argmin I", v["argmin_I"], "argmin R", v["argmin_R"])
    for k, v in out["ch37"].items():
        print("ch37", k, "rho*", v["rho_star"], "tie", v["calibration"]["tie_set"][0], "..", v["calibration"]["tie_set"][-1],
              "held-out argmin", v["held_out"]["argmin_J"], "in tie", v["rho_star_in_held_out_tie"],
              "W_eval", round(v["W_eval_at_rho_star"], 3), {r: {a: round(b, 3) for a, b in x.items()} for r, x in v["at"].items()})
    for u in ("37g21", "37g33", "19", "26", "27", "32"):
        for h, v in out["real_pfa"][u].items():
            print("real Pfa at mode I eta(1e-3)", u, h, v["frames"], {r: round(x, 4) for r, x in v["by_rank"].items()})


if __name__ == "__main__":
    main()
