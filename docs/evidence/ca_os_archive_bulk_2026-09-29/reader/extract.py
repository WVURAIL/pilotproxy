"""Extract the numbers for figures B1 (rank choice) and B2 (CA against OS) from the prototype folders.

Reads only, never writes, in the prototype folders. Writes data/b1_rank.json, data/b2_cfar.json and
data/sources.json (sha256 of every input) under this group's folder.

Sources (all under /home/djg/rail/output/):
  fixed-rank-decisions-2026-09-29/check/bench_quad_scan.json   exact quadrature, every rank, |B| = 125 and 126
  fixed-rank-decisions-2026-09-29/check/bench_quad.json        exact quadrature at six ranks (cross-check)
  fixed-rank-decisions-2026-09-29/check/bench_contam.json      exact OS loss, k bins at +20 dB
  fixed-rank-decisions-2026-09-29/check/bench_ca_contam.json   CA loss, k bins at +20 dB (own Monte Carlo)
  fixed-rank-decisions-2026-09-29/check/counts_check.json      independent channel 30 and 31 counts
  fixed-rank-decisions-2026-09-29/rank/data/rank_bench.json    Monte Carlo rank bench (cross-check)
  fixed-rank-decisions-2026-09-29/rank/data/counts/chNN.json   per-channel contamination counts and histograms
  fixed-rank-decisions-2026-09-29/rank/data/ideal_counts.json  ideal-law count constants
  fixed-rank-decisions-2026-09-29/rank/data/archive_ranks.json least R per rank, held-out replay at 41 and 63
  rank-by-injection-2026-09-30/data/controls_summary.json      channel 37 injection, modes I and R
  cfar-choice-2026-09-29/data/bulk/chNN.json, .npz             per-frame bulk mean over median
  cfar-choice-2026-09-29/data/ideal_bulk.json                  ideal-law mean over median
  cfar-choice-2026-09-29/data/bench.json                       CA and OS bench under contamination
  cfar-choice-2026-09-29/verify/v_bench.log                    independent +10 dB check
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

OUT = Path("/home/djg/rail/output")
FRD = OUT / "fixed-rank-decisions-2026-09-29"
RBI = OUT / "rank-by-injection-2026-09-30"
CFC = OUT / "cfar-choice-2026-09-29"
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CHANNELS = list(range(14, 37))
GEOMS = list(range(14, 37))
SOURCES: dict[str, str] = {}


def load(path: Path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def load_npz(path: Path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return np.load(path)


def b1a_clean():
    scan = load(FRD / "check/bench_quad_scan.json")
    quad = load(FRD / "check/bench_quad.json")
    mc = load(FRD / "rank/data/rank_bench.json")
    out = {"method": scan["method"], "source": "fixed-rank-decisions-2026-09-29/check/bench_quad_scan.json"}
    for n in ("125", "126"):
        s = scan["sizes"][n]
        ranks = np.arange(1, int(n) + 1)
        loss = np.array([s[str(r)]["loss_db"] for r in ranks])
        i = int(np.argmin(loss))
        flat = ranks[loss - loss[i] <= 2e-4]
        assert np.all(np.diff(flat) == 1), "flat band is not contiguous"
        # cross-check: the six-rank quadrature file
        q = quad["sizes"][n]
        quad_diff = max(abs(q[str(r)]["loss_db"] - s[str(r)]["loss_db"]) for r in (20, 41, 46, 62, 63, 94))
        # cross-check: the Monte Carlo bench, its argmin and its 2 SE band
        c = mc["sizes"][n]["cases"]["clean"]
        mloss = np.array([c[str(r)]["loss_db"] for r in ranks])
        mse = np.array([c[str(r)]["se_db"] for r in ranks])
        j = int(np.argmin(mloss))
        m2 = ranks[mloss - mloss[j] <= 2 * np.hypot(mse, mse[j])]
        z = (mloss - loss) / np.where(mse > 0, mse, np.nan)
        out[n] = dict(
            ca_snr_pd90_db=s["CA"]["snr_db"],
            loss_db=loss.tolist(),
            argmin=int(ranks[i]), min_loss_db=float(loss[i]),
            flat_2e4=[int(flat[0]), int(flat[-1])],
            at={str(r): float(loss[r - 1]) for r in (41, 62, 63, 64, 1, int(n))},
            price_41_over_63_db=float(loss[40] - loss[62]),
            eta_1e3={str(r): s[str(r)]["eta"] for r in (41, 63)},
            quad_six_rank_max_abs_diff_db=float(quad_diff),
            mc_argmin=int(ranks[j]), mc_within_2se=[int(m2.min()), int(m2.max())],
            mc_minus_quad_max_abs_z=float(np.nanmax(np.abs(z))),
            mc_at={str(r): [float(mloss[r - 1]), float(mse[r - 1])] for r in (41, 63)},
        )
    return out


def b1b_counts():
    ideal = load(FRD / "rank/data/ideal_counts.json")
    chk = load(FRD / "check/counts_check.json")
    rows = []
    for ch in CHANNELS:
        d = load(FRD / f"rank/data/counts/ch{ch}.json")
        nb = d["bulk_size"]
        up, dn, med = d["counts"]["up_lo"], d["counts"]["dn_hi"], d["counts"]["up_med"]
        hu, hd = np.array(up["hist"]), np.array(dn["hist"])
        assert hu.sum() == d["frames"] == hd.sum()
        up99, dn99 = int(math.ceil(up["p99"])), int(math.ceil(dn["p99"]))   # ceil, as decide.py (ch19: 16.01 -> 17)
        mstar = max(d["persistent_bins"], up99)

        def broken_up(rho):  # upward breakdown: more than |B| - rho contaminants
            return int(hu[nb - rho + 1:].sum())

        def broken_dn(rho):  # downward breakdown: rho or more depressed bins
            return int(hd[rho:].sum())

        rows.append(dict(
            channel=ch, state=d["state"], bulk_size=nb, frames=d["frames"], anchor_parity=d["anchor_parity"],
            persistent=d["persistent_bins"], up_p99=up99, up_p99_raw=up["p99"], up_max=int(up["max"]), up_p50=int(up["p50"]),
            up_med_p99=int(med["p99"]), dn_p99=dn99, dn_max=int(dn["max"]),
            m_star=mstar, m_star_from="persistent" if d["persistent_bins"] > up["p99"] else "p99",
            break_rank_up=nb - mstar,
            margin_up={str(r): nb - r - mstar for r in (41, 63)},
            margin_dn_p99={str(r): r - 1 - dn99 for r in (41, 63)},
            margin_dn_max={str(r): r - 1 - int(dn["max"]) for r in (41, 63)},
            broken_up={str(r): broken_up(r) for r in (41, 63)},
            broken_dn={str(r): broken_dn(r) for r in (41, 63)},
        ))
    tot = lambda key, r: sum(row[key][str(r)] for row in rows)
    worst = max(rows, key=lambda row: (row["m_star"] + 63 - row["bulk_size"], row["m_star"]))
    for c in ("30", "31"):  # independent check agrees
        row = next(x for x in rows if x["channel"] == int(c))
        assert chk[c]["m_star"] == row["m_star"] and chk[c]["persistent"] == row["persistent"]
        assert int(chk[c]["up_lo_p99"]) == row["up_p99"] and chk[c]["up_lo_max"] == row["up_max"]
    dec = load(FRD / "rank/data/decision.json")      # the prototype's own m* per channel
    for row in rows:
        assert dec["channels"][str(row["channel"])]["m_star"] == row["m_star"], row["channel"]
    for r in ("41", "63"):
        assert dec["compare"][r]["frames_upward_broken"] == sum(x["broken_up"][r] for x in rows)
        assert dec["compare"][r]["frames_downward_broken"] == sum(x["broken_dn"][r] for x in rows)
    return dict(
        source="fixed-rank-decisions-2026-09-29/rank/data/counts/chNN.json",
        definitions=dict(
            persistent="bulk bins above the ideal single-bin 0.999 point of T/median in more than half the frames",
            up_lo="per frame, bins above m + c s, m and s read from the frame's lower quartile (P10, P25); ideal Pfa 1e-3 per bin",
            dn_hi="mirror of up_lo from the upper quartile (P75, P90): the one-sided downward count",
            up_med="bins with T/median above the ideal 0.999 point; measures spread, saturates near |B|/2",
            m_star="max(persistent, ceil(p99 of up_lo))",
            upward_breakdown="more than |B| - rho contaminants",
            downward_breakdown="rho or more depressed bins"),
        ideal_up_lo={n: ideal[n]["count_law"].get("up_lo") for n in ("125", "126")},
        channels=rows,
        frames_total=sum(r["frames"] for r in rows),
        broken_up_total={str(r): tot("broken_up", r) for r in (41, 63)},
        broken_dn_total={str(r): tot("broken_dn", r) for r in (41, 63)},
        broken_up_by_channel={str(r): {str(x["channel"]): x["broken_up"][str(r)] for x in rows if x["broken_up"][str(r)]}
                              for r in (41, 63)},
        binding_channel=worst["channel"], binding_m_star=worst["m_star"], binding_bulk=worst["bulk_size"],
        binding_margin_63=worst["bulk_size"] - 63 - worst["m_star"],
        m_star_range=[min(r["m_star"] for r in rows), max(r["m_star"] for r in rows)],
        dn_p99_max=max(r["dn_p99"] for r in rows),
        dn_max=max((r["dn_max"], r["channel"]) for r in rows),
        up_med_p99_range=[min(r["up_med_p99"] for r in rows), max(r["up_med_p99"] for r in rows)],
        balance_rank={n: (int(n) + 1) / 2 for n in ("125", "126")},
    )


def b1_heldout():
    d = load(FRD / "rank/data/archive_ranks.json")
    rows = {}
    for ch, c in d["channels"].items():
        r41, r63 = c["ranks"]["41"]["R"], c["ranks"]["63"]["R"]
        h41, h63 = c["replay"]["41"]["R"], c["replay"]["63"]["R"]
        rows[ch] = dict(cal_ratio=r41 / r63, heldout_ratio=h41 / h63)
    hr = [v["heldout_ratio"] for v in rows.values()]
    cr = [v["cal_ratio"] for v in rows.values()]
    return dict(source="fixed-rank-decisions-2026-09-29/rank/data/archive_ranks.json", channels=rows,
                heldout_ratio_range=[min(hr), max(hr)], cal_ratio_range=[min(cr), max(cr)],
                heldout_better_at_41=sum(h < 1 for h in hr), cal_better_at_41=sum(c < 1 for c in cr))


def b1c_injection():
    s = load(RBI / "data/controls_summary.json")
    out = {"source": "rank-by-injection-2026-09-30/data/controls_summary.json", "budget_db": 0.9, "curves": {}}
    for pfa in ("0.001", "0.01"):
        for half, hk in (("cal", "calibration"), ("eval", "held_out")):
            for mode in ("I", "R"):
                L = np.array([s["units"][f"37g{g}"][f"{half}_{pfa}"][f"L_{mode}_curve"] for g in GEOMS])
                am = [s["units"][f"37g{g}"][f"{half}_{pfa}"][f"argmin_{mode}"] for g in GEOMS]
                med, lo, hi = np.median(L, 0), L.min(0), L.max(0)
                J = np.array(s["ch37"][f"{mode}_{pfa}"][hk]["J"])
                W = np.array(s["ch37"][f"{mode}_{pfa}"][hk]["W"])
                assert np.allclose(J, med) and np.allclose(W, hi), "J/W differ from geometry median/max"
                out["curves"][f"{mode}_{half}_{pfa}"] = dict(
                    median=med.tolist(), lo=lo.tolist(), hi=hi.tolist(), argmin_range=[min(am), max(am)],
                    at={str(r): [float(lo[r - 1]), float(med[r - 1]), float(hi[r - 1])] for r in (41, 63, 72, 75, 125)},
                    budget_pass_hi=[int(r) for r in np.arange(1, 126)[hi <= 0.9]],
                )
    # real false-alarm rate of mode I's threshold on channel 37's real designated bins
    fa = {}
    for half in ("cal", "eval"):
        v = {r: [s["real_pfa"][f"37g{g}"][half]["by_rank"][str(r)] for g in GEOMS] for r in (41, 63)}
        fa[half] = {str(r): [min(x), max(x)] for r, x in v.items()}
    out["real_fa_of_mode_I_eta_pfa1e3"] = fa
    out["procedure"] = {k: dict(rho_star=v["rho_star"], budget_pass=v["budget_pass"], at=v["at"])
                        for k, v in s["ch37"].items()}
    # where the median-geometry curve of each mode falls inside the 0.9 dB budget on both halves
    for mode in ("I", "R"):
        a = np.array(out["curves"][f"{mode}_cal_0.001"]["hi"])
        b = np.array(out["curves"][f"{mode}_eval_0.001"]["hi"])
        ok = np.arange(1, 126)[(a <= 0.9) & (b <= 0.9)]
        out[f"budget_both_halves_worst_geometry_{mode}"] = [int(ok.min()), int(ok.max())] if ok.size else None
    return out


def b2a_bulk():
    ideal = load(CFC / "data/ideal_bulk.json")
    rows = []
    for ch in CHANNELS:
        meta = load(CFC / f"data/bulk/ch{ch}.json")
        z = load_npz(CFC / f"data/bulk/ch{ch}.npz")
        r = z["r"]
        assert r.size == meta["frames"]
        q = np.percentile(r, [1, 10, 50, 90, 99])
        f105 = float(np.mean(r > 1.05))
        assert abs(f105 - meta["r"]["frac_gt_1p05"]) < 1e-12 and abs(q[2] - meta["r"]["median"]) < 1e-9
        rows.append(dict(channel=ch, state=meta["state"], parity=meta["anchor_parity"], bulk_size=meta["bulk_size"],
                         frames=int(r.size), p01=q[0], p10=q[1], p50=q[2], p90=q[3], p99=q[4],
                         frac_gt_1p05=f105, frac_gt_1p25=float(np.mean(r > 1.25)),
                         frac_gt_ideal_p999=float(meta["r"]["frac_gt_ideal_p999"])))
    quiet = [x["frac_gt_1p05"] for x in rows if x["state"] == "proxy-low"]
    strong = [x["frac_gt_1p05"] for x in rows if x["state"] == "proxy-high"]
    return dict(source="cfar-choice-2026-09-29/data/bulk/chNN.npz (r = mean_B T / median_B T)",
                ideal={n: {k: ideal[n][k] for k in ("frames", "r_median", "r_p99", "r_p999", "r_max")}
                       for n in ("125", "126")},
                channels=rows, quiet_frac_gt_1p05_range=[min(quiet), max(quiet)],
                strong_frac_gt_1p05_range=[min(strong), max(strong)], frames_total=sum(x["frames"] for x in rows))


def b2b_bench():
    b = load(CFC / "data/bench.json")
    vb = load(CFC / "verify/v_bench.log")
    oc = load(FRD / "check/bench_contam.json")
    cc = load(FRD / "check/bench_ca_contam.json")
    out = {"source": "cfar-choice-2026-09-29/data/bench.json, even anchor, |B| = 125, Pd = 0.9, Pfa = 1e-3"}
    for par in ("even", "odd"):
        cases = b["parities"][par]["cases"]
        ref = cases["clean"]["snr_pd90"]["CA"]
        tab = {}
        for key, c in cases.items():
            tab[key] = dict(k=c["k"], level_db=c["level_db"],
                            loss_vs_clean_CA={d: c["snr_pd90"][d] - ref for d in ("CA", "OS63", "OS94", "OS113")},
                            pfa_actual=c["pfa_actual"])
        out[par] = dict(clean_CA_snr_pd90_db=ref, bulk_size=b["parities"][par]["bulk_size"], cases=tab)
    ev = out["even"]["cases"]
    out["cross_checks"] = dict(
        ca_k1_10dB_exactD=vb["loss_vs_clean_OS63_db"]["k1_10dB"]["CA"] - vb["loss_vs_clean_OS63_db"]["clean"]["CA"],
        os63_k1_10dB_exactD=vb["loss_vs_clean_OS63_db"]["k1_10dB"]["OS63"] - vb["loss_vs_clean_OS63_db"]["clean"]["CA"],
        ca_k1_20dB_mc=cc["1"], ca_k3_20dB_mc=cc["3"], os63_k1_20dB_quad=oc["63_k1"], os63_k3_20dB_quad=oc["63_k3"],
        note="bench.json reads SNR off a 0.5 dB grid; the exact-D values differ by about 0.03 dB on CA",
    )
    os_max = max(abs(v["loss_vs_clean_CA"]["OS63"]) for k, v in ev.items() if v["level_db"] in (10.0, 20.0) or k == "clean")
    out["os63_max_loss_10_20dB_k_le3"] = os_max
    return out


def main():
    DATA.mkdir(exist_ok=True)
    b1 = dict(clean=b1a_clean(), counts=b1b_counts(), heldout=b1_heldout(), injection=b1c_injection())
    b2 = dict(bulk=b2a_bulk(), bench=b2b_bench())
    (DATA / "b1_rank.json").write_text(json.dumps(b1, indent=1))
    (DATA / "b2_cfar.json").write_text(json.dumps(b2, indent=1))
    (DATA / "sources.json").write_text(json.dumps(SOURCES, indent=1, sort_keys=True))
    c, k, h, inj = b1["clean"], b1["counts"], b1["heldout"], b1["injection"]
    for n in ("125", "126"):
        print(f"B1a |B|={n}: argmin {c[n]['argmin']}, flat(2e-4) {c[n]['flat_2e4']}, L41 {c[n]['at']['41']:.5f}, "
              f"L63 {c[n]['at']['63']:.5f}, L64 {c[n]['at']['64']:.5f}, price41 {c[n]['price_41_over_63_db']:.5f}, "
              f"MC argmin {c[n]['mc_argmin']} 2SE {c[n]['mc_within_2se']}, max|z| {c[n]['mc_minus_quad_max_abs_z']:.2f}, "
              f"six-rank diff {c[n]['quad_six_rank_max_abs_diff_db']:.1e}")
    print(f"B1b: binding ch{k['binding_channel']} m*={k['binding_m_star']} |B|={k['binding_bulk']} margin63={k['binding_margin_63']}; "
          f"m* range {k['m_star_range']}; broken up {k['broken_up_total']} {k['broken_up_by_channel']}; "
          f"broken dn {k['broken_dn_total']}; dn p99 max {k['dn_p99_max']}, dn max {k['dn_max']}; "
          f"up_med p99 {k['up_med_p99_range']}; frames {k['frames_total']}")
    for row in k["channels"]:
        print(f"  ch{row['channel']} |B|={row['bulk_size']} pers={row['persistent']} p99={row['up_p99']} max={row['up_max']} "
              f"m*={row['m_star']} break={row['break_rank_up']} margin63={row['margin_up']['63']} dn p99/max={row['dn_p99']}/{row['dn_max']}")
    print(f"held out R(41)/R(63) {h['heldout_ratio_range']} better at 41 on {h['heldout_better_at_41']}/7; "
          f"cal {h['cal_ratio_range']} on {h['cal_better_at_41']}/7")
    for key, v in inj["curves"].items():
        print(f"B1c {key}: argmin {v['argmin_range']}; L41 {['%.2f' % x for x in v['at']['41']]}; "
              f"L63 {['%.2f' % x for x in v['at']['63']]}; L125 {['%.2f' % x for x in v['at']['125']]}")
    print("B1c budget, both halves worst geometry, Pfa 1e-3: I", inj["budget_both_halves_worst_geometry_I"],
          "R", inj["budget_both_halves_worst_geometry_R"])
    print("B1c real FA of mode I eta:", inj["real_fa_of_mode_I_eta_pfa1e3"])
    bb, be = b2["bulk"], b2["bench"]
    print(f"B2a: quiet >1.05 {bb['quiet_frac_gt_1p05_range']}, strong {bb['strong_frac_gt_1p05_range']}, frames {bb['frames_total']}")
    for row in bb["channels"]:
        print(f"  ch{row['channel']} {row['state']} p50 {row['p50']:.4f} p99 {row['p99']:.2f} >1.05 {row['frac_gt_1p05']:.3f}")
    for key, v in be["even"]["cases"].items():
        print(f"B2b {key}: CA {v['loss_vs_clean_CA']['CA']:+.3f}  OS63 {v['loss_vs_clean_CA']['OS63']:+.3f}")
    print("B2b cross:", be["cross_checks"], "OS63 max", be["os63_max_loss_10_20dB_k_le3"])


if __name__ == "__main__":
    main()
