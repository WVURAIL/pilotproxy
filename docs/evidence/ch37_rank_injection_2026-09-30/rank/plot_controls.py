"""Control figure for the rank-by-injection study (Appendix C plate style: the dissertation's style module, byte
copy in figstyle/).

figs/rank_injection_controls
  (a) ideal i.i.d. control: the mode I code path against the rank bench, clean loss against CA per rank;
  (b) channel 37 at the 23 channel geometries, calibration half, Pfa 1e-3: loss against each geometry's best rank,
      mode I (ideal designated bins) and mode R (real designated bins, tone injected), with the medians over
      geometries and the ch06 straddle budget;
  (c) real false-alarm rate of mode I's threshold on the real designated bins of the same frames (channel 37
      geometries and the carrier-off channels), against the nominal 1e-3.
Reads data/controls_bench_125.json, data/rank_bench.json, data/controls_summary.json.
Usage (dissertation venv): python -B plot_controls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "figstyle"))
import numpy as np  # noqa: E402
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIGS, DATA = ROOT / "figs", ROOT / "data"
NR = 125
MARK = dict(color=style.INK, lw=0.7, ls=(0, (1, 1.6)))


def save(fig, stem, title):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{stem}.png", dpi=150, bbox_inches="tight", pad_inches=0.035)
    with style.stable_pdf_subset_tags():
        style.save(fig, FIGS / f"{stem}.pdf", title=title)


def marks(ax, y=None):
    """Dotted lines at the current rank 41 and the bench optimum region's 63, labelled on the x axis."""
    for r in (41, 63):
        ax.axvline(r, **MARK, zorder=1)
    ax.set_xticks([1, 20, 41, 63, 80, 100, 125])


def main():
    style.configure()
    ctl = json.loads((DATA / "controls_bench_125.json").read_text())["125"]
    bench = json.loads((DATA / "rank_bench.json").read_text())["sizes"]["125"]["cases"]["clean"]
    summ = json.loads((DATA / "controls_summary.json").read_text())
    x = np.arange(1, NR + 1)
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.25, 1.0], hspace=0.55, left=0.11, right=0.97, top=0.975, bottom=0.075)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) ideal control against the bench
    yb = np.array([bench[str(r)]["loss_db"] for r in x])
    sb = np.array([bench[str(r)]["se_db"] for r in x])
    ym = np.array([ctl["clean"][str(r)]["loss"] for r in x])
    sm = np.array([ctl["clean"][str(r)]["se"] for r in x])
    axa.fill_between(x, yb - 2 * sb, yb + 2 * sb, color=style.LIGHT_GRAY, lw=0, zorder=1)
    axa.plot(x, yb, color=style.NEUTRAL, lw=1.6, zorder=2)
    axa.errorbar(x[::4], ym[::4], yerr=2 * sm[::4], fmt="o", ms=2.6, color=style.MEASURED, lw=0.7, zorder=3)
    marks(axa, 0.6)
    axa.set_yscale("log")
    axa.set_ylim(0.008, 0.8)
    axa.set_xlim(0, 126)
    axa.set_ylabel("clean cost\nagainst CA (dB)")
    axa.legend(handles=[Line2D([], [], color=style.NEUTRAL, lw=1.6, label="rank bench, $\\pm 2$ SE band"),
                        Line2D([], [], color=style.MEASURED, marker="o", ms=2.6, lw=0, label="mode I code path, every 4th rank, $\\pm 2$ SE")],
               loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7.0, handlelength=1.8, ncol=2)
    style.clean_axes(axa)
    style.panel_label(axa, "a", x=-0.1, y=1.12)

    # (b) channel 37, mode I against mode R
    key = "cal_0.001"
    LI = np.array([summ["units"][f"37g{g}"][key]["L_I_curve"] for g in range(14, 37)])
    LR = np.array([summ["units"][f"37g{g}"][key]["L_R_curve"] for g in range(14, 37)])
    for L, col in ((LI, style.MEASURED), (LR, style.EXCLUDE)):
        for row in L:
            axb.plot(x, np.clip(row, 0, 6.0), color=col, lw=0.45, alpha=0.35, zorder=2)
        axb.plot(x, np.clip(np.median(L, 0), 0, 6.0), color=col, lw=1.8, zorder=4)
    axb.axhline(0.9, color=style.MODEL, lw=1.0, ls=(0, (4, 2)), zorder=3)
    axb.text(3, 0.98, "ch06 straddle loss, near 0.9 dB", fontsize=7.0, color=style.INK, va="bottom")
    marks(axb, 5.85)
    axb.set_ylim(0, 6.0)
    axb.set_xlim(0, 126)
    axb.set_ylabel("loss against the\nbest rank (dB)")
    axb.legend(handles=[Line2D([], [], color=style.MEASURED, lw=1.8, label="mode I: real bulk, ideal $\\mathcal{D}$"),
                        Line2D([], [], color=style.EXCLUDE, lw=1.8, label="mode R: real bulk, real $\\mathcal{D}$ + tone")],
               loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7.0, handlelength=1.8, ncol=2)
    axb.text(0.70, 0.45, "channel 37, 23 geometries\ncalibration half, $P_{\\rm fa}=10^{-3}$\nthin: geometry, thick: median\nclipped at 6 dB",
             transform=axb.transAxes, ha="right", va="top", fontsize=6.8, color=style.INK)
    style.clean_axes(axb)
    style.panel_label(axb, "b", x=-0.1, y=1.08)

    # (c) real false-alarm rate of mode I's threshold
    for g in range(14, 37):
        f = np.array(summ["real_pfa"][f"37g{g}"]["cal"]["curve"])
        axc.plot(x, np.maximum(f, 1e-5), color=style.MEASURED, lw=0.45, alpha=0.4, zorder=2)
    for c, ls in zip((19, 26, 27, 32), ("-", (0, (4, 1.5)), (0, (1, 1.2)), (0, (6, 1.5, 1, 1.5)))):
        f = np.array(summ["real_pfa"][str(c)]["cal"]["curve"])
        axc.plot(x, np.maximum(f, 1e-5), color=style.EXCLUDE, lw=1.0, ls=ls, zorder=3)
    axc.axhline(1e-3, color=style.INK, lw=0.9, zorder=3)
    axc.text(124, 7e-4, "nominal $10^{-3}$", fontsize=7.0, color=style.INK, va="top", ha="right")
    marks(axc, 0.8)
    axc.set_yscale("log")
    axc.set_ylim(1e-5, 1.0)
    axc.set_xlim(0, 126)
    axc.set_xlabel(r"fixed rank $\rho$ of the bulk reference")
    axc.set_ylabel("real false-alarm rate\nat mode I's $\\eta(\\rho)$")
    axc.legend(handles=[Line2D([], [], color=style.MEASURED, lw=1.0, label="channel 37, each geometry"),
                        *[Line2D([], [], color=style.EXCLUDE, lw=1.0, ls=ls, label=f"channel {c}, carrier off")
                          for c, ls in zip((19, 26, 27, 32), ("-", (0, (4, 1.5)), (0, (1, 1.2)), (0, (6, 1.5, 1, 1.5))))]],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.8, handlelength=2.2, ncol=2)
    style.clean_axes(axc)
    style.panel_label(axc, "c", x=-0.1)
    save(fig, "rank_injection_controls", "Rank by injection: controls")


if __name__ == "__main__":
    main()
