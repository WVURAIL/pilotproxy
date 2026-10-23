"""Dissertation-wide plotting style.

This copy preserves the plotting conventions used by the recorded figures.
The dissertation maintains its current style in figure_src/style.py.

The body text is Latin Modern (LaTeX ``lmodern``).  Submission figures render
all text through the same LaTeX stack, at a fixed physical size, so labels are
not silently substituted with DejaVu or shrunk differently from chapter to
chapter.  This file also fixes the semantic palette and common line/axis rules.
"""
from __future__ import annotations

import contextlib
import hashlib
import shutil
import string
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Dissertation text block is 6.5 in; figures are included at 0.98 linewidth.
TEXT_WIDTH = 6.35
HALF_WIDTH = 3.05

INK = "#202225"
MUTED = "#6F747A"
GRID = "#D9DADD"
PAPER = "#FFFFFF"
PANEL = "#F7F7F5"
# Semantic palette drawn from the WVU visual-identity color system
# (scm.wvu.edu/brand/visual-identity/); role names are unchanged.
MEASURED = "#0062A3"       # archive / instrument measurement (WVU safety blue)
MODEL = "#7F6310"          # model, assumption, or transfer (WVU old gold)
CONDITIONAL = "#6A724F"    # feasible/accepted, but conditional (WVU hemlock)
FAILURE = "#8D4638"        # failure, rejection, or excision (WVU woodburn)
PENDING = "#988E8B"        # unmeasured / contextual (WVU seneca gray)
PURPLE = "#F58672"         # additional series only (WVU sunset)
GOLD = "#EEAA00"           # additional series / warning (WVU gold)
LIGHT_BLUE = "#DEEBF3"     # pale tints of the role colors for fills
LIGHT_ORANGE = "#FCF2D9"
LIGHT_GREEN = "#E1E3DC"
LIGHT_RED = "#EADDDB"
LIGHT_GRAY = "#F0EEEE"
SERIES = [MEASURED, MODEL, CONDITIONAL, PURPLE, GOLD, PENDING]

# Fixed PDF metadata makes repeated figure builds byte-for-byte reproducible.
# The date identifies the frozen dissertation-export interface revision; it is
# not intended to represent the wall-clock time of a local rebuild.
PDF_METADATA_DATE = datetime(2026, 8, 12, tzinfo=timezone.utc)


def configure(*, require_tex: bool = True) -> None:
    """Apply the dissertation figure style.

    The bundled build requires TeX and fails rather than silently substituting
    another font.  ``require_tex=False`` is only for an explicitly requested
    quick preview and is never used by the Makefile or audit.
    """
    have_tex = all(shutil.which(c) for c in ("latex", "dvipng", "kpsewhich"))
    if require_tex and not have_tex:
        raise RuntimeError(
            "Figure generation requires latex, dvipng, and kpsewhich so that "
            "figure text uses Latin Modern exactly like the dissertation."
        )
    matplotlib.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.8,
        "patch.linewidth": 0.8,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.78,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "text.usetex": bool(have_tex),
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
    })
    if have_tex:
        matplotlib.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"
            r"\usepackage{amsmath,amssymb}"
        )


def clean_axes(ax, *, grid: str | None = "both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, which="major")
    return ax


def panel_label(ax, label: str, *, x: float = -0.08, y: float = 1.035) -> None:
    ax.text(x, y, rf"\textbf{{({label})}}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.0, color=INK)


def save(fig, path: Path, *, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata={
        "Title": title,
        "Author": "Dylan Gormley",
        "Subject": "The Pilot-Proxy Method dissertation figure",
        "Creator": "Matplotlib with Latin Modern via LaTeX",
        "CreationDate": PDF_METADATA_DATE,
        "ModDate": PDF_METADATA_DATE,
    })
    plt.close(fig)
    return path



def stable_subset_prefix(charset) -> str:
    """Return a repeatable PDF font-subset tag from glyph content."""
    glyphs = "\0".join(sorted(str(name) for name in charset))
    value = int.from_bytes(
        hashlib.sha256(glyphs.encode("utf-8")).digest()[:8], "big")
    letters = []
    for _ in range(6):
        value, index = divmod(value, 26)
        letters.append(string.ascii_uppercase[index])
    return "".join(letters) + "+"


@contextlib.contextmanager
def stable_pdf_subset_tags():
    """Use content-based tags when Matplotlib exposes its PDF hook."""
    from matplotlib.backends.backend_pdf import PdfFile

    original = vars(PdfFile).get("_get_subset_prefix")
    if original is None:
        yield
        return
    PdfFile._get_subset_prefix = staticmethod(stable_subset_prefix)
    try:
        yield
    finally:
        PdfFile._get_subset_prefix = original
