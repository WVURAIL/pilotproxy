"""Rank-choice bench for order-statistic (OS) and cell-averaging (CA) references.

The bench measures the clean-noise cost of OS at every rank rho = 1..|B|
against CA, and the loss when k bulk bins are contaminated at +20 dB. It is the
ideal i.i.d. F(4096, 8192) bench of cfar-choice-2026-09-29 (bench.py, with its
semi-analytic check verify/v_bench.py). The model assumes that:
  * every fine bin is T = (X / 4096) / (Y / 8192), with
    X ~ noncentral chi2(4096, 4096 s g_f) and Y ~ chi2(8192), and the bins are
    independent.
  * the designated set D is the anchor +/- 2 padded bins, with pilot gains
    (0, g1, 1, g1, 0). Here, g1 is the rectangular-window Dirichlet response
    half a natural bin off the tone (L = 128).
  * the bulk carries no pilot on an even anchor. On an odd anchor the
    shoulders reach the bulk at 0.045 x s, which is below 0.01 in T at the
    Pd = 0.9 SNR, so the shoulders are not modeled.
  * the statistic is Z = max_D T / ref, where ref is mean_B T for CA, or
    T_(rho), the rho-th smallest bulk bin, for OS at rank rho.
  * each detector's threshold eta is the upper 1e-3 point of its own Z under
    the clean ideal null.
  * the reference is independent of D, so
    Pd(s) = 1 - E_ref[prod_{f in D} F_nc(eta ref; 4096, 8192, 4096 s g_f)].
    This is exact in the D bins (scipy's noncentral F), and only the
    reference is simulated.

For contamination, the first k bulk columns are replaced by noncentral
F(4096, 8192, 4096 x 100) draws, for k in {1, 3, 10, 30, 54}. At +20 dB these
draws are far above every clean bin, so only their count affects an order
statistic. The clean threshold is applied, since the detector has no previous
knowledge of the contamination. The same clean bins and the same contaminated
columns are used across ranks and cases as common random numbers (k = 3 is a
subset of k = 10).

The loss is the SNR at Pd = 0.9 of the detector and case minus the SNR at
Pd = 0.9 of clean CA, in dB, per batch. The reported value is the mean over 20
batches of 20000 frames, and its error is the standard error across batches.

Pd is read from a table of prod_D F_nc on a grid in x (step 1e-4) and in SNR
(step 0.002 dB, -9.6 to -3 dB). The cases whose crossing the table cannot
reach are the broken references, with an SNR near +20 dB. These are solved by
brentq on the pooled 400000 frames, with the D CDFs evaluated directly, and
carry no batch error.

This is an ideal independent-bin reference only. The simulated or quadrature
threshold is not an operational eta_Pfa calibration.

Run with --out DIR using this module.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import f as fdist, ncf

L = 128
G1 = float((np.sin(np.pi * 0.5) / (L * np.sin(np.pi * 0.5 / L))) ** 2)
PFA, PD = 1e-3, 0.9
SIZES = (125, 126)
KS = (1, 3, 10, 30, 54)
LEVEL_DB = 20.0
NBATCH, NF = 20, 20_000
XG = np.round(np.arange(0.97, 1.45 + 1e-12, 1e-4), 10)
SG = np.round(np.arange(-9.6, -3.0 + 1e-12, 0.002), 6)


def dprod(x, s):
    """P(max_D T <= x) at pilot SNR s (linear), D gains (0, g1, 1, g1, 0)."""
    c0 = fdist.cdf(x, 4096, 8192)
    if s == 0:
        return c0 ** 5
    return c0 ** 2 * ncf.cdf(x, 4096, 8192, 4096 * s * G1) ** 2 * ncf.cdf(x, 4096, 8192, 4096 * s)


def build_table():
    """Tabulate prod_D F_nc on the SNR grid SG (rows) and the x grid XG (columns)."""
    t0 = time.time()
    tab = np.empty((SG.size, XG.size))
    for j, sdb in enumerate(SG):
        tab[j] = dprod(XG, 10 ** (sdb / 10))
    print(f"table {tab.shape} in {time.time() - t0:.0f} s", flush=True)
    return tab


TAB = None
F5 = None      # the clean-null prod_D F on XG, filled in run (interpolation error below 1e-6 in the CDF)


def eta_for(ref, exact=False):
    """Return the threshold eta at the clean Pfa = 1e-3 for a reference sample, from F5 or, if exact, from dprod."""
    if exact:
        return brentq(lambda e: 1.0 - np.mean(dprod(e * ref, 0.0)) - PFA, 0.9, 1.4, xtol=1e-9)
    return brentq(lambda e: 1.0 - np.mean(np.interp(e * ref, XG, F5)) - PFA, 0.9, 1.4, xtol=1e-9)


def snr_table(eta, ref, tab):
    """SNR (dB) at Pd = 0.9 from the table, or None when the table does not cover the crossing or the x values."""
    x = eta * ref
    if x.min() <= XG[0] or x.max() >= XG[-1]:
        return None
    u = (x - XG[0]) / 1e-4
    i = np.floor(u).astype(np.int64)
    w = u - i
    h = np.bincount(i, 1.0 - w, minlength=XG.size) + np.bincount(i + 1, w, minlength=XG.size)
    pd = 1.0 - tab @ h / x.size
    above = np.flatnonzero(pd >= PD)
    if not above.size or above[0] == 0:
        return None
    j = above[0]
    return float(SG[j - 1] + (PD - pd[j - 1]) * (SG[j] - SG[j - 1]) / (pd[j] - pd[j - 1]))


def snr_direct(eta, ref, lo=-12.0, hi=40.0, npts=1501):
    """SNR (dB) at Pd = 0.9 by brentq, with prod_D F_nc evaluated on npts points spanning eta ref."""
    x = eta * ref
    grid = np.linspace(x.min() * 0.999, x.max() * 1.001, npts)

    def pd(sdb):
        return 1.0 - np.interp(x, grid, dprod(grid, 10 ** (sdb / 10))).mean()
    return float(brentq(lambda v: pd(v) - PD, lo, hi, xtol=1e-5))


def run_size(n, tab, seed):
    """Simulate NBATCH batches for bulk size n and return the per-batch SNRs, thresholds and pooled references."""
    rng = np.random.default_rng(seed)
    dets = ["CA"] + list(range(1, n + 1))
    cases = [0] + list(KS)
    snr = {c: {d: [] for d in dets} for c in cases}        # per batch, None where the table cannot solve
    etas = {d: [] for d in dets}
    pooled = {c: {d: [] for d in dets} for c in cases}      # references kept for the direct solves
    direct_batches = {c: {d: 0 for d in dets} for c in cases}
    lam = 4096 * 10 ** (LEVEL_DB / 10)
    for b in range(NBATCH):
        t0 = time.time()
        clean = rng.f(4096, 8192, size=(NF, n))
        hot = rng.noncentral_f(4096, 8192, lam, size=(NF, max(KS)))
        eta_b = {}
        for c in cases:
            t = clean.copy()
            if c:
                t[:, :c] = hot[:, :c]
            srt = np.sort(t, axis=1)
            refs = {"CA": t.mean(axis=1)} | {r: srt[:, r - 1] for r in range(1, n + 1)}
            for d in dets:
                if c == 0:
                    eta_b[d] = eta_for(refs[d])
                    etas[d].append(eta_b[d])
                broken = c > 0 and (d == "CA" or d > n - c)
                v = snr_table(eta_b[d], refs[d], tab)
                if v is None and not broken:     # unbroken reference the table cannot reach, solved directly for this batch
                    v = snr_direct(eta_b[d], refs[d])
                    direct_batches[c][d] += 1
                snr[c][d].append(v)
                if broken:                       # broken references are pooled and solved once
                    pooled[c][d].append(refs[d].astype(np.float32))
                elif c == 0 and b < 2 and d in ("CA", 63):
                    pooled[c][d].append(refs[d].astype(np.float64))
        print(f"|B|={n} batch {b + 1}/{NBATCH} {time.time() - t0:.0f} s", flush=True)
    return dets, cases, snr, etas, pooled, direct_batches


def summarise(n, dets, cases, snr, etas, pooled, direct_batches):
    """Reduce the batches of bulk size n to losses against clean CA, with a table-versus-direct check."""
    base = np.array(snr[0]["CA"], dtype=float)
    assert np.all(np.isfinite(base))
    eta_pooled = {d: float(np.mean(etas[d])) for d in dets}
    out = {"bulk_size": n, "detectors": [str(d) for d in dets], "eta_mean": {str(d): eta_pooled[d] for d in dets},
           "eta_se": {str(d): float(np.std(etas[d], ddof=1) / np.sqrt(NBATCH)) for d in dets},
           "ca_clean_snr_pd90_db": dict(mean=float(base.mean()), se=float(base.std(ddof=1) / np.sqrt(NBATCH))),
           "cases": {}}
    for c in cases:
        rows = {}
        for d in dets:
            v = snr[c][d]
            if all(x is not None for x in v):
                diff = np.array(v, dtype=float) - base
                rows[str(d)] = dict(loss_db=float(diff.mean()), se_db=float(diff.std(ddof=1) / np.sqrt(NBATCH)),
                                    method="table, 20 batches", direct_batches=direct_batches[c][d])
            else:
                ref = np.concatenate(pooled[c][d]).astype(np.float64)
                if ref.size != NBATCH * NF:
                    raise RuntimeError(f"references for the direct solve are incomplete: {c} {d}")
                s = snr_direct(eta_pooled[d], ref)
                rows[str(d)] = dict(loss_db=float(s - base.mean()), se_db=None, method="direct, pooled",
                                    batches_table_solved=int(sum(x is not None for x in v)), broken=True)
        out["cases"]["clean" if c == 0 else f"k{c}_20dB"] = rows
    # the direct solve is compared with the table on the first two pooled batches of clean OS 63 and CA
    chk = {}
    for d in ("CA", 63):
        ref = np.concatenate(pooled[0][d][:2])
        e, ex = eta_for(ref), eta_for(ref, exact=True)
        chk[str(d)] = dict(eta_interp=e, eta_exact=ex, table=snr_table(e, ref, TAB),
                           direct=snr_direct(ex, ref, lo=-10.0, hi=-7.0))
    out["table_vs_direct_check"] = chk
    return out


def run(output_dir):
    """Run the recorded model and its fixed seeds. The output location is the only input."""
    global TAB, F5
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "REFERENCE.md").write_text('Ideal independent-bin reference only. The simulated or quadrature threshold is not an operational eta_Pfa calibration.\n', encoding="utf-8")
    t_all = time.time()
    TAB = build_table()
    F5 = dprod(XG, 0.0)
    result = {"model": "ideal i.i.d. F(4096, 8192); D gains (0, g1, 1, g1, 0); Pfa 1e-3 per detector on the clean null; "
                       "Pd 0.9; contaminated bins noncentral F at +20 dB",
              "g1": G1, "pfa": PFA, "pd": PD, "level_db": LEVEL_DB, "ks": KS, "batches": NBATCH, "frames_per_batch": NF,
              "x_grid": [float(XG[0]), float(XG[-1]), 1e-4], "snr_grid_db": [float(SG[0]), float(SG[-1]), 0.002],
              "sizes": {}}
    for n, seed in zip(SIZES, (20260929_125, 20260929_126)):
        dets, cases, snr, etas, pooled, direct_batches = run_size(n, TAB, seed)
        result["sizes"][str(n)] = summarise(n, dets, cases, snr, etas, pooled, direct_batches)
        del pooled
        (OUT / "rank_bench.json").write_text(json.dumps(result, indent=1) + "\n")
    result["seconds"] = round(time.time() - t_all, 1)
    (OUT / "rank_bench.json").write_text(json.dumps(result, indent=1) + "\n")
    r = result["sizes"]["125"]["cases"]["clean"]
    print({k: round(r[k]["loss_db"], 4) for k in ("1", "2", "3", "8", "34", "63", "94", "113", "125")}, flush=True)
    return result


def main(argv=None):
    """Write the recorded ideal-reference study into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
