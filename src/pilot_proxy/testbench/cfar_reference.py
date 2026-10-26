"""CA against OS at a fixed ideal false-alarm rate on the synthetic bench.

This is part 4 of the cfar-choice-2026-09-29 study.

The ideal statistic of each fine bin is T = (X / 4096) / (Y / 8192), with X ~ noncentral chi2(4096, lambda),
Y ~ chi2(8192), and independent bins. This is the ideal F(4096, 8192) law of ch05 for the bulk. For D, the
independence is the same assumption the fine-score plates make. The noncentrality is lambda = 4096 (s g_f + c_f),
where s is the pilot SNR (the pilot power over the noise power of one bin, so E[T] = 1 + s at the pilot's bin), g_f
is the pilot's response at padded bin f, and c_f is the SNR of a contaminating bulk bin. X is drawn exactly as
(N + sqrt(lambda))^2 + chi2(4095), so every case shares one noise draw (common random numbers).

The geometry has 256 padded bins with a pad factor of 2 (128 natural bins). D is the anchor +/- 2. The bulk is every
second bin minus the anchor's +/- 2 guard and D (anchors.bulk_mask with guard 1 and no census bins), which gives
|B| = 125 at an even anchor (128) and 126 at an odd anchor (127). The pilot is a tone on the anchor bin with a
rectangular window, whose Dirichlet response is |sin(pi d) / (L sin(pi d / L))|^2 for d in natural bins. On an even
anchor, the tone is on the natural grid and puts nothing in the bulk. On an odd anchor, it is half a natural bin off
the grid, and its shoulders fall on the even bulk bins (0.045 at anchor +/- 3, 0.016 at +/- 5, ...). In the
contaminated cases, 0 to 3 bulk bins far from the anchor are set to +10, +20, or +30 dB.

The detectors are Z = max_D T / ref, with ref = mean_B T (CA), T_(63) (OS 63), T_(94) (OS 94, three quarters), or
T_(113) (OS 113, nine tenths). Each threshold is the upper 1e-3 point of the detector's own Z under the clean ideal
null (no pilot and no contamination) at the same |B|, from 4e6 simulated frames.

This study is limited to the ideal independent-bin reference. The simulated or quadrature threshold is not an
operational eta_Pfa calibration.

Run with --out DIR using this module.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

L, NB, HW = 128, 256, 2
PFA = 1e-3
RANKS = (63, 94, 113)
DETS = ("CA",) + tuple(f"OS{r}" for r in RANKS)
SNR_DB = np.arange(-20.0, 30.01, 0.5)
LEVELS_DB = (10.0, 20.0, 30.0)
CONTAM_OFFSETS = {"even": (40, -60, 86), "odd": (41, -61, 87)}    # padded-bin offsets of the contaminated bulk bins


def geometry(parity: str):
    a = 128 if parity == "even" else 127
    mask = np.zeros(NB, dtype=bool)
    mask[::2] = True
    mask[(a + np.arange(-2, 3)) % NB] = False       # guard 1 natural bin = 2 padded bins, and D
    D = (a + np.arange(-HW, HW + 1)) % NB
    B = np.flatnonzero(mask)
    bins = np.concatenate([D, B])
    d = (((bins - a + 128) % 256) - 128) / 2.0      # natural-bin distance from the tone
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(d == 0, 1.0, (np.sin(np.pi * d) / (L * np.sin(np.pi * d / L))) ** 2)
    g[np.abs(g) < 1e-15] = 0.0
    contam = [int(np.flatnonzero(bins == (a + o) % NB)[0]) for o in CONTAM_OFFSETS[parity]]
    assert all(c >= D.size for c in contam)
    return dict(anchor=a, D=D, B=B, nD=D.size, nB=B.size, g=g, contam=contam)


def scores(T, nD, nB):
    """Z for every detector, T of shape (frames, nD + nB)."""
    dmax = T[:, :nD].max(axis=1)
    bulk = T[:, nD:]
    out = {"CA": dmax / bulk.mean(axis=1)}
    part = np.partition(bulk, [r - 1 for r in RANKS], axis=1)
    for r in RANKS:
        out[f"OS{r}"] = dmax / part[:, r - 1]
    return out


class Noise:
    def __init__(self, n, nbins, rng):
        self.N = rng.standard_normal((n, nbins))
        self.C = rng.chisquare(4095, (n, nbins))
        self.Yn = rng.chisquare(8192, (n, nbins)) / 8192.0

    def T(self, lam):
        return (((self.N + np.sqrt(lam)) ** 2 + self.C) / 4096.0) / self.Yn


def thresholds(geo, frames=4_000_000, chunk=200_000, seed=1):
    rng = np.random.default_rng(seed)
    nbins = geo["nD"] + geo["nB"]
    z = {k: [] for k in DETS}
    for _ in range(frames // chunk):
        T = (rng.chisquare(4096, (chunk, nbins)) / 4096.0) / (rng.chisquare(8192, (chunk, nbins)) / 8192.0)
        for k, v in scores(T, geo["nD"], geo["nB"]).items():
            z[k].append(v)
    out = {}
    for k in DETS:
        v = np.concatenate(z[k])
        out[k] = dict(eta=float(np.quantile(v, 1 - PFA)), median=float(np.median(v)), std=float(v.std()))
    return out


def lam_vector(geo, s, k, level_db):
    c = np.zeros(geo["nD"] + geo["nB"])
    c[geo["contam"][:k]] = 10 ** (level_db / 10.0)
    return 4096.0 * (s * geo["g"] + c)


def snr_at(pd, snr_db, target):
    """Linear interpolation in dB of the first crossing of target (NaN if never reached)."""
    above = np.flatnonzero(pd >= target)
    if not above.size:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(snr_db[0])
    x0, x1, y0, y1 = snr_db[i - 1], snr_db[i], pd[i - 1], pd[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def run(output_dir):
    """Run the recorded clean and contaminated reference cases."""
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "REFERENCE.md").write_text('Ideal independent-bin reference only. The simulated or quadrature threshold is not an operational eta_Pfa calibration.\n', encoding="utf-8")
    t0 = time.time()
    result = {"pfa": PFA, "ranks": RANKS, "snr_db": SNR_DB.tolist(), "levels_db": LEVELS_DB, "parities": {}}
    for parity, seed in (("even", 11), ("odd", 12)):
        geo = geometry(parity)
        thr = thresholds(geo, seed=seed)
        eta = {k: thr[k]["eta"] for k in DETS}
        rng = np.random.default_rng(seed + 100)
        n = 40_000
        noise = Noise(n, geo["nD"] + geo["nB"], rng)
        cases = [(0, 0.0)] + [(k, lv) for lv in LEVELS_DB for k in (1, 2, 3)]
        pr = {"anchor": geo["anchor"], "bulk_size": geo["nB"], "thresholds": thr,
              "bulk_shoulder_gain_sum": float(geo["g"][geo["nD"]:].sum()),
              "bulk_gain_pm3": float(geo["g"][geo["nD"]:][np.abs(((geo["B"] - geo["anchor"] + 128) % 256) - 128) == 3].sum()),
              "frames_per_point": n, "cases": {}}
        for k, lv in cases:
            key = f"k{k}_{int(lv)}dB" if k else "clean"
            pd = {d: [] for d in DETS}
            ratio = []
            for sdb in SNR_DB:
                z = scores(noise.T(lam_vector(geo, 10 ** (sdb / 10.0), k, lv)), geo["nD"], geo["nB"])
                for d in DETS:
                    pd[d].append(float((z[d] > eta[d]).mean()))
                ratio.append(float(np.median(z["CA"] / z["OS63"])))
            z0 = scores(noise.T(lam_vector(geo, 0.0, k, lv)), geo["nD"], geo["nB"])
            pfa = {d: float((z0[d] > eta[d]).mean()) for d in DETS}
            pr["cases"][key] = {"k": k, "level_db": lv, "pd": pd, "pfa_actual": pfa, "median_zca_over_zos63": ratio,
                                "snr_pd50": {d: snr_at(np.array(pd[d]), SNR_DB, 0.5) for d in DETS},
                                "snr_pd90": {d: snr_at(np.array(pd[d]), SNR_DB, 0.9) for d in DETS}}
            print(parity, key, {d: round(pr["cases"][key]["snr_pd90"][d], 2) for d in DETS}, pfa, flush=True)
        result["parities"][parity] = pr
    # clean-noise loss at |B| = 125, fine grid and more frames, common random numbers, 10 batches for its error
    geo = geometry("even")
    eta = {k: result["parities"]["even"]["thresholds"][k]["eta"] for k in DETS}
    fine = np.arange(-14.0, -4.99, 0.05)
    batches = []
    rng = np.random.default_rng(99)
    for b in range(10):
        noise = Noise(40_000, geo["nD"] + geo["nB"], rng)
        pd = {d: [] for d in DETS}
        for sdb in fine:
            z = scores(noise.T(lam_vector(geo, 10 ** (sdb / 10.0), 0, 0.0)), geo["nD"], geo["nB"])
            for d in DETS:
                pd[d].append(float((z[d] > eta[d]).mean()))
        batches.append({d: {"pd50": snr_at(np.array(pd[d]), fine, 0.5), "pd90": snr_at(np.array(pd[d]), fine, 0.9)} for d in DETS})
    loss = {}
    for d in DETS[1:]:
        for q in ("pd50", "pd90"):
            diff = np.array([bt[d][q] - bt["CA"][q] for bt in batches])
            loss[f"{d}_minus_CA_{q}_db"] = dict(mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(diff.size)))
    for q in ("pd50", "pd90"):
        v = np.array([bt["CA"][q] for bt in batches])
        loss[f"CA_{q}_snr_db"] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(v.size)))
    result["clean_loss_125"] = dict(frames=400_000, grid_db=[float(fine[0]), float(fine[-1]), 0.05], **loss)
    result["seconds"] = round(time.time() - t0, 1)
    (OUT / "bench.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["clean_loss_125"], indent=1))

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
