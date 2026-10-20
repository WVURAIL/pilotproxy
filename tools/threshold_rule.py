"""Where each channel is deployed for the science of RFIsher's tolerance table
(its knee, or the loosest usable threshold below it), or why it is not: sweep
every keep word, test the upper end of the kept DTV power, and write the
decisions, the sweeps and one plate per channel. A deployed channel's word is
then checked against the tolerance on its era's last year (one full seasonal
cycle) and lowered only if it fails there (the last-year guard).

The serialized legacy ``coherence_limit_s`` field is the full-era
effective-gain-time diagnostic at the final deployed word, or at the knee
when undeployed. It is not the joint full-era/last-year limit or a physical
phase lifetime. ``last_year_r_hi`` is evaluated at the original
``full_era_word``, before any tightening, so it cannot generally be combined
with the final full-era ``r_hi`` to reconstruct that joint limit.

    python tools/threshold_rule.py --scores <score cache dir> --out <dir>

The score cache is the current-era equal-event cache (``ch<N>.json`` and
``ch<N>.npz``); each names the archive product its pilot excess comes from.
See pilot_proxy.characterization.threshold_rule for the rule itself.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pilot_proxy.characterization import threshold_rule as t  # noqa: E402

INK, MUTED, GRID, SURF = "#1c2126", "#5a6168", "#e4e6e2", "#ffffff"
MEAS, HIGH, KNEE, PICK = "#2a78d6", "#9fc2ef", "#5a6168", "#eb6834"
OUTCOME = {"knee": "useful for BAO: deployed at the knee",
           "below the knee": "useful for BAO: deployed below the knee",
           "above tolerance": "not deployed: above tolerance at every threshold",
           "out of time": f"not deployed: the tolerance needs more than {t.MAX_CALENDAR_YEARS:g} years"}
COLUMNS = ["channel", "outcome", "reason", "deployed_at", "deployed_word", "full_era_word", "last_year_r_hi", "guard_acted", "knee_word", "eta", "kappa", "time_years", "r", "r_hi", "kept_frames", "kept_events",
           "gain", "coherence_limit_s", "below_knee_eta", "below_knee_kappa", "below_knee_years", "below_knee_r_hi"]


def plate(ch, row, s, d, R, gain, out: Path) -> Path:
    eta = s["word"].astype(float) / t.Q16
    r, r_hi = t.bias_over_sigma(s, R, gain)
    k, b = d["knee"], d["below_knee"]
    style = {"font.size": 10.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
             "axes.labelcolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK,
             "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"}
    with plt.rc_context(style):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, constrained_layout=True)
        a1.plot(eta, r_hi, color=HIGH, lw=1.6, label="upper end (power + 2 standard errors): tested")
        a1.plot(eta, r, color=MEAS, lw=1.6, label="measured power")
        a1.axhline(1, color=INK, ls="--", lw=1.1)
        a1.text(eta[0], 1.25, "bias = σ", fontsize=9)
        a2.plot(eta, 1 / np.where(s["kappa"] > 0, s["kappa"], np.nan), color=MEAS, lw=1.8)
        for ax in (a1, a2):
            ax.axvline(k["eta"], color=KNEE, lw=1, ls=":")
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        a1.text(k["eta"], 1e3, " knee", color=KNEE, fontsize=9, va="top")
        a1.plot(k["eta"], k["r_hi"], "o", color=INK, ms=8, zorder=5)
        a2.plot(k["eta"], k["time_years"], "o", color=INK, ms=8, zorder=5)
        a2.annotate(f"knee η = {k['eta']:.3g}: keeps {100 * k['kappa']:.3g}%, {k['time_years']:.3g} yr",
                    (k["eta"], k["time_years"]), xytext=(10, 10), textcoords="offset points", color=INK, fontsize=9.5)
        a2.axhline(t.MAX_CALENDAR_YEARS, color=MUTED, ls="--", lw=1)
        if b is not None:
            a1.plot(b["eta"], b["r_hi"], "o", color=PICK, ms=8, zorder=5)
            a2.plot(b["eta"], b["time_years"], "o", color=PICK, ms=8, zorder=5)
            a2.annotate(f"tolerance first met below the knee: η = {b['eta']:.3g}, {b['time_years']:.3g} yr",
                        (b["eta"], b["time_years"]), xytext=(10, -16), textcoords="offset points", color=PICK, fontsize=9.5)
        p = d["deployed"]
        if p is not None and p["word"] not in (k["word"], b and b["word"]):
            a1.plot(p["eta"], p["r_hi"], "D", color=INK, ms=7, zorder=6)
            a2.plot(p["eta"], p["time_years"], "D", color=INK, ms=7, zorder=6)
            a2.annotate(f"deployed by the last-year guard: η = {p['eta']:.3g}, {p['time_years']:.3g} yr",
                        (p["eta"], p["time_years"]), xytext=(10, 14), textcoords="offset points", color=INK, fontsize=9.5)
        a1.set(xscale="log", yscale="log", ylim=(1e-6, 1e3), ylabel="bias ÷ σ of D_V",
               title="Bias of D_V at each threshold")
        a1.legend(frameon=False, fontsize=9, loc="lower right")
        a2.set(yscale="log", xlabel="threshold η (keep word / 2¹⁶)", ylabel="calendar years for one retained year",
               title="Time cost 1 / κ")
        verdict = OUTCOME.get(d["reason"] or d["deployed_at"], f"not deployed: {d['reason']}")
        fig.suptitle(f"Channel {ch}  ·  {row['low_mhz']:.0f}–{row['high_mhz']:.0f} MHz  ·  {verdict}",
                     x=0.01, ha="left", fontsize=13, fontweight="bold")
        path = out / f"channel_{ch:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return path


def summary(decisions, out: Path) -> Path:
    chans = [d["channel"] for d in decisions]
    with plt.rc_context({"font.size": 10.5, "axes.edgecolor": MUTED, "figure.facecolor": SURF}):
        fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
        use = [d for d in decisions if d["outcome"] == "useful"]
        late = [d for d in decisions if d["reason"] == "out of time"]
        over = [d for d in decisions if d["reason"] == "above tolerance"]
        for at, marker in (("knee", "o"), ("below the knee", "s")):
            pts = [d for d in use if d["deployed_at"] == at]
            if pts:
                ax.plot([d["channel"] for d in pts], [d["time_years"] for d in pts], marker, color=MEAS, ms=9,
                        label=OUTCOME[at])
        ax.plot([d["channel"] for d in late], [d["below_knee_years"] for d in late], "x", color=PICK, ms=10, mew=2.5,
                label=OUTCOME["out of time"] + " (marked at those years)")
        ax.plot([d["channel"] for d in over], [1e3] * len(over), "^", color=INK, ms=9,
                label=OUTCOME["above tolerance"] + " (drawn at the top)")
        for reason in sorted({d["reason"] for d in decisions} - {None, "out of time", "above tolerance"}):
            pts = [d for d in decisions if d["reason"] == reason]
            ax.plot([d["channel"] for d in pts], [1e3] * len(pts), "D", color=MUTED, ms=8,
                    label=f"not deployed: {reason} (drawn at the top)")
        ax.axhline(t.MAX_CALENDAR_YEARS, color=MUTED, ls="--", lw=1)
        ax.set(yscale="log", xticks=chans, xlabel="channel", ylabel="calendar years for one retained year",
               title="Calendar years for one retained year, per channel; the dashed line is the 30-year limit")
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        path = out / "summary.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path, help="RFIsher channel table (default: the packaged copy)")
    ap.add_argument("--channels", help="comma-separated channel numbers (default: all in the table)")
    ap.add_argument("--exclude", action="append", default=[], metavar="CH=REASON",
                    help="do not deploy channel CH, for the stated reason (repeatable)")
    args = ap.parse_args(argv)
    table = t.load_table(args.table)
    rows = {r["channel"]: r for r in table["channels"]}
    pick = sorted(rows) if not args.channels else [int(c) for c in args.channels.split(",")]
    excluded = {int(c): r for c, r in (e.split("=", 1) for e in args.exclude)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweeps").mkdir(exist_ok=True)
    (args.out / "plates").mkdir(exist_ok=True)
    decisions = []
    for ch in pick:
        f = t.load_frames(ch, args.scores)
        s = t.sweep(f)
        y = t.last_year(f)
        R = t.response(table, ch)
        G = t.coherent_gain(ch)
        d = t.decide_guarded(s, f.knee_word, t.sweep(y), R, gain=G, exclude=excluded.get(ch))
        np.savez_compressed(args.out / "sweeps" / f"ch{ch}.npz", **s)
        plate(ch, rows[ch], s, d, R, G, args.out / "plates")
        k, b = d["knee"], d["below_knee"]
        w = d["deployed"] or k
        decisions.append(dict(channel=ch, outcome=d["outcome"], reason=d["reason"], deployed_at=d["deployed_at"],
                              deployed_word=d["deployed_word"], full_era_word=d["full_era_word"],
                              last_year_r_hi=d["last_year_r_hi"], guard_acted=d["guard_acted"],
                              last_year_frames=len(y.required),
                              **{key: w[key] for key in ("eta", "kappa", "time_years", "r", "r_hi", "kept_frames",
                                                         "kept_events", "coherence_limit_s")},
                              knee_word=k["word"], knee_eta=k["eta"], knee_kappa=k["kappa"], knee_r=k["r"],
                              knee_r_hi=k["r_hi"],
                              below_knee_eta=b and b["eta"], below_knee_kappa=b and b["kappa"],
                              below_knee_years=b and b["time_years"], below_knee_r_hi=b and b["r_hi"],
                              response=R, gain=G, swept_words=len(s["word"])))
        print(f"ch{ch}: {d['reason'] or d['deployed_at']:>15} knee eta {k['eta']:.4g} kappa {k['kappa']:.4g} time {k['time_years']:.4g} yr"
              + (f"; tolerance below the knee at eta {b['eta']:.4g}, {b['time_years']:.4g} yr" if b else "")
              + (f"; deployed at eta {d['deployed']['eta']:.4g}" if d["deployed"] else ""))
    summary(decisions, args.out)
    inputs = dict(scores=str(args.scores), knee_rule=t.knee_rule.RULE_ID, table_provenance=table["provenance"], shelf=t.SHELF, excluded=excluded,
                  max_calendar_years=t.MAX_CALENDAR_YEARS,
                  coherent_fraction=t.COHERENT_FRACTION, coherent_fraction_scale=t.COHERENT_FRACTION_SCALE,
                  coherent_fraction_by_band=t.COHERENT_FRACTION_BY_BAND, on_sky_duty=t.ON_SKY_DUTY,
                  calendar_years_per_retained_year=t.CALENDAR_YEARS_PER_RETAINED_YEAR, coherent_gain_rule=t._require(t._GAIN, "coherent_gain.json")["rule"],
                  coherent_gain_by_band={ch: t.coherent_gain(ch) for ch in pick},
                  frame_samples=t.FRAME_SAMPLES, guard_standard_errors=t.GUARD_STANDARD_ERRORS,
                  last_year_guard=dict(window_days=t.LAST_YEAR_DAYS, window="one full seasonal cycle, ending at the era's last frame",
                                       rule="the full-era decision stands if its word has r_hi <= 1 on the last year's curves "
                                            "(same G and R); else the loosest word at or below it with r_hi <= 1 on both curves "
                                            "and full-era years within the limit, else not deployed ('" + t.GUARD_FAILED + "'); "
                                            "kappa, x, r and years are read on the full era; no knee is computed on the last year"))
    (args.out / "decisions.json").write_text(json.dumps(dict(inputs=inputs, channels=decisions), indent=1) + "\n")
    with (args.out / "decisions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(decisions)


if __name__ == "__main__":
    main()
