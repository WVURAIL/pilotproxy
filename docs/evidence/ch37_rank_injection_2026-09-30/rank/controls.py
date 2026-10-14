"""Ideal i.i.d. control of the mode I code path against the rank bench (DESIGN.md section 4).

The bench's own layout: |B| = 125 and 126, 20 batches of 20,000 frames of i.i.d. F(4096, 8192) bulk bins (400,000
frames), references = every order statistic and the bulk mean (CA). Per batch the mode I code path (core.ModeI:
threshold at Pfa 1e-3 on the batch, Pd = 0.9 SNR from the table, tone on the anchor) gives s*(rho) and s*(CA);
loss = s*(rho) - s*(CA), mean over batches and its standard error, as rank_bench.py reports it. Checks:
  * every rank within 2 combined SE of rank_bench.json's clean curve;
  * the quadrature values of bench_quad.json (ranks 20, 41, 46, 62, 63, 94) within 2 SE;
  * the argmin, and the flat bottom (ranks within 2 SE of the minimum);
  * k = 1, 3, 10, 30, 54 bulk bins at +20 dB with the clean threshold: every rank the bench solved from its table
    (unbroken, rho <= |B| - k) within 2 combined SE of the bench's contaminated column.
Writes data/controls_bench_N.json. Usage (analysis interpreter, one thread, no bytecode): python -B controls.py [N ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import core

ROOT = core.ROOT
NB, NF = 20, 20_000
KS = (1, 3, 10, 30, 54)
LAM = 4096 * 10 ** (20.0 / 10)


def main(sizes):
    t0 = time.time()
    tabs = core.Tables()
    bench = json.loads((ROOT / "data/rank_bench.json").read_text())
    quad = json.loads((ROOT / "data/bench_quad.json").read_text())
    out = {}
    for n, seed in [(n, sd) for n, sd in ((125, 20260930_125), (126, 20260930_126)) if n in sizes]:
        rng = np.random.default_rng(seed)
        clean = {r: [] for r in range(1, n + 1)}
        cont = {k: {r: [] for r in range(1, n - k + 1)} for k in KS}
        ca = []
        for b in range(NB):
            t = rng.f(4096, 8192, size=(NF, n))
            hot = rng.noncentral_f(4096, 8192, LAM, size=(NF, max(KS)))
            R = np.concatenate([np.sort(t, axis=1), t.mean(1, keepdims=True)], axis=1)
            mi = core.ModeI(R, np.zeros(NF, dtype=int), tabs, pfas=(1e-3,), offsets=("d0",))
            pt = mi.run_point()
            s = pt[("s", 1e-3, "d0")]
            eta = pt[("eta", 1e-3)]
            ca.append(s[-1])
            for r in range(1, n + 1):
                clean[r].append(s[r - 1] - s[-1])
            for k in KS:
                tc = t.copy()
                tc[:, :k] = hot[:, :k]
                Rc = np.sort(tc, axis=1)[:, : n - k]
                mc = core.ModeI(Rc, np.zeros(NF, dtype=int), tabs, pfas=(1e-3,), offsets=("d0",))
                sc, _ = mc.snr(eta[: n - k], np.ones(NF), "d0")
                for r in range(1, n - k + 1):
                    cont[k][r].append(sc[r - 1] - s[-1])
            print(n, "batch", b + 1, f"{time.time() - t0:.0f} s", flush=True)
        B = bench["sizes"][str(n)]["cases"]
        rows, fails = {}, []
        mean = {r: float(np.mean(v)) for r, v in clean.items()}
        se = {r: float(np.std(v, ddof=1) / np.sqrt(NB)) for r, v in clean.items()}
        for r in range(1, n + 1):
            b = B["clean"][str(r)]
            tol = 2 * np.hypot(se[r], b["se_db"])
            ok = abs(mean[r] - b["loss_db"]) <= tol
            rows[r] = dict(loss=mean[r], se=se[r], bench=b["loss_db"], bench_se=b["se_db"], ok=bool(ok))
            if not ok:
                fails.append(("clean", r, mean[r], b["loss_db"], tol))
        qd = {}
        for r in (20, 41, 46, 62, 63, 94):
            q = quad["sizes"][str(n)][str(r)]["loss_db"]
            ok = abs(mean[r] - q) <= 2 * se[r]
            qd[r] = dict(loss=mean[r], se=se[r], quad=q, ok=bool(ok))
            if not ok:
                fails.append(("quad", r, mean[r], q, 2 * se[r]))
        arg = min(mean, key=mean.get)
        flat = [r for r in mean if mean[r] - mean[arg] <= 2 * np.hypot(se[r], se[arg])]
        bargmin = min(range(1, n + 1), key=lambda r: B["clean"][str(r)]["loss_db"])
        cres = {}
        for k in KS:
            col = B[f"k{k}_20dB"]
            cres[k] = {}
            for r in range(1, n - k + 1):
                b = col[str(r)]
                if b["method"] != "table, 20 batches":
                    continue
                m, e = float(np.mean(cont[k][r])), float(np.std(cont[k][r], ddof=1) / np.sqrt(NB))
                ok = abs(m - b["loss_db"]) <= 2 * np.hypot(e, b["se_db"])
                cres[k][r] = dict(loss=m, se=e, bench=b["loss_db"], bench_se=b["se_db"], ok=bool(ok))
                if not ok:
                    fails.append((f"k{k}", r, m, b["loss_db"], 2 * np.hypot(e, b["se_db"])))
        nchk = n + len(qd) + sum(len(v) for v in cres.values())
        out[str(n)] = dict(clean=rows, quad=qd, argmin=int(arg), flat_bottom=[min(flat), max(flat)], bench_argmin=int(bargmin),
                           contaminated=cres, ca_snr_mean=float(np.mean(ca)), checks=nchk, fails=fails,
                           fail_fraction=len(fails) / nchk)
        print(n, "argmin", arg, "flat", min(flat), max(flat), "41", round(mean[41], 5), "63", round(mean[63], 5),
              "fails", len(fails), "of", nchk, flush=True)
        for f in fails:
            print("  fail", f, flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / f"data/controls_bench_{'_'.join(map(str, sizes))}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [125, 126])
