"""Quadrature check of the clean-noise loss of order-statistic (OS) references against cell averaging (CA).

The loss of OS at fixed ranks is computed by quadrature, with no Monte Carlo
trials, as an independent check. The model is the one stated in rank/RANK.md.
The fine bins T are i.i.d. F(4096, 8192). The designated set D has 5 bins with
pilot gains (0, g1, 1, g1, 0). Here, g1 is the rectangular-window response half
a natural bin off the tone (L = 128). The statistic is Z = max_D T / ref, and
each detector's threshold is set at its own clean 1e-3 point. The SNR per bin is
found at Pd = 0.9, and the loss is SNR(OS rho) - SNR(CA) in dB.

The OS reference at rank rho uses the exact order-statistic density of
F(4096, 8192). The CA reference is the mean of |B| bins, and its density is
found by FFT convolution of the single-bin density.

This is an ideal independent-bin reference only. The simulated or quadrature
threshold is not an operational eta_Pfa calibration.

Run with --out DIR using this module.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

L = 128
G1 = (1.0 / (L * np.sin(np.pi * 0.5 / L))) ** 2
D1, D2 = 4096, 8192
PFA, PD = 1e-3, 0.9
F = stats.f(D1, D2)


def os_ref(n, rho, lo=0.84, hi=1.20, h=2e-5):
    """Return the grid, normalized weights and grid mass of the rank-rho order statistic of n bins."""
    r = np.arange(lo, hi, h)
    lp = (special.gammaln(n + 1) - special.gammaln(rho) - special.gammaln(n - rho + 1)
          + (rho - 1) * F.logcdf(r) + (n - rho) * F.logsf(r) + F.logpdf(r))
    w = np.exp(lp) * h
    return r, w / w.sum(), w.sum()


def ca_ref(n, h=2e-5, N=1 << 19, bw=4e-6):
    """Return the bin centers, normalized weights and single-bin grid mass of the mean of n bins."""
    # density of Y = X - 1 on a circular grid, n-fold convolution by FFT, then M = 1 + S / n binned to width bw
    k = np.arange(N); k[k >= N // 2] -= N
    y = k * h
    py = np.zeros(N)
    ok = (y > -0.4) & (y < 0.6)
    py[ok] = F.pdf(1.0 + y[ok]) * h
    mass1 = py.sum()
    py /= mass1
    ps = np.real(np.fft.ifft(np.fft.fft(py) ** n))
    ps = np.clip(ps, 0, None)
    m = 1.0 + y / n
    edges = np.arange(m[ps > 1e-300].min() - bw, m[ps > 1e-300].max() + 2 * bw, bw)
    w, _ = np.histogram(m, bins=edges, weights=ps)
    c = 0.5 * (edges[1:] + edges[:-1])
    keep = w > 1e-18
    return c[keep], w[keep] / w[keep].sum(), mass1


def pfa(eta, r, w):
    """Return P(max of the 5 null D bins > eta ref), averaged over the reference density."""
    s = F.sf(eta * r)
    return float(np.dot(w, -np.expm1(5.0 * np.log1p(-s))))


def pd(sdb, eta, r, w):
    """Return Pd at a per-bin pilot SNR of sdb (dB), averaged over the reference density."""
    s = 10 ** (sdb / 10)
    x = eta * r
    c0 = F.cdf(x)
    c1 = stats.ncf.cdf(x, D1, D2, D1 * s * G1)
    c2 = stats.ncf.cdf(x, D1, D2, D1 * s)
    return float(1.0 - np.dot(w, c0 ** 2 * c1 ** 2 * c2))


def solve(r, w):
    """Return the threshold eta at Pfa = 1e-3 and the SNR (dB) at Pd = 0.9 for one reference."""
    eta = brentq(lambda e: pfa(e, r, w) - PFA, 1.0, 1.4, xtol=1e-12)
    snr = brentq(lambda v: pd(v, eta, r, w) - PD, -10.0, -7.0, xtol=1e-8)
    return eta, snr


def run(output_dir, *, ranks=(20, 41, 63)):
    """Evaluate the recorded quadrature model at the requested fixed ranks."""
    ranks = list(ranks)
    if not ranks or any(type(r) is not int or not 1 <= r <= 126 for r in ranks):
        raise ValueError("ranks must be integers from 1 through 126")
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "REFERENCE.md").write_text('Ideal independent-bin reference only. The simulated or quadrature threshold is not an operational eta_Pfa calibration.\n', encoding="utf-8")
    out = {"g1": G1, "method": "quadrature: exact OS density, CA by FFT convolution", "sizes": {}}
    for n in (125, 126):
        t0 = time.time()
        r, w, m1 = ca_ref(n)
        eta_ca, snr_ca = solve(r, w)
        res = {"CA": dict(eta=eta_ca, snr_db=snr_ca, single_bin_grid_mass=m1)}
        for rho in [x for x in ranks if x <= n]:
            r, w, mass = os_ref(n, rho)
            eta, snr = solve(r, w)
            res[str(rho)] = dict(eta=eta, snr_db=snr, loss_db=snr - snr_ca, density_mass=mass)
        out["sizes"][str(n)] = res
        print(n, round(time.time() - t0), {k: (round(v["eta"], 5), round(v.get("loss_db", v["snr_db"]), 5)) for k, v in res.items()}, flush=True)
    (OUT / ("bench_quad_scan.json" if len(ranks) > 20 else "bench_quad.json")).write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None):
    """Write the recorded ideal-reference study into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[20, 41, 63])
    args = parser.parse_args(argv)
    run(args.out, ranks=args.ranks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
