# coding=utf-8
"""Shared Matplotlib style for PilotProxy figures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _command_available(command: str) -> bool:
    command_text = str(command)
    has_path_separator = any(
        separator and separator in command_text for separator in (os.sep, os.altsep)
    )
    if not command_text or has_path_separator:
        return False
    return any(
        (Path(directory) / command_text).exists()
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
    )


def _latex_package_available(package: str) -> bool:
    """Return whether ``kpsewhich`` resolves a LaTeX package."""

    if not _command_available("kpsewhich"):
        return False
    try:
        return bool(
            subprocess.run(
                ["kpsewhich", package],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError:
        return False


def setup_matplotlib(
    *,
    force_agg: bool = True,
    dissertation_style: bool = False,
):
    """Configure Matplotlib for LaTeX-style PilotProxy plots.

    External TeX rendering is normally opt-in through the
    ``PILOT_PROXY_USE_TEX`` environment variable and is used only when the TeX
    helper commands are available. Otherwise, Matplotlib's Computer Modern
    mathtext renderer gives the same visual language without a TeX runtime
    dependency.

    ``dissertation_style=True`` is stricter. It reproduces the dissertation's
    fail-closed Latin Modern/T1 typography contract and never silently falls
    back to DejaVu. This mode requires ``latex``, ``dvipng``, ``kpsewhich``,
    and the ``lmodern`` package.
    """
    import matplotlib

    if force_agg:
        matplotlib.use("Agg", force=True)
    if dissertation_style:
        missing = [
            command
            for command in ("latex", "dvipng", "kpsewhich")
            if not _command_available(command)
        ]
        if missing:
            raise RuntimeError(
                "dissertation figure generation requires "
                + ", ".join(missing)
                + " so text can be rendered with embedded Latin Modern/T1 fonts"
            )
        if not _latex_package_available("lmodern.sty"):
            raise RuntimeError(
                "dissertation figure generation requires the LaTeX lmodern "
                "package; refusing a fallback font"
            )
        use_tex = True
    else:
        use_tex = (
            os.environ.get("PILOT_PROXY_USE_TEX", "0") == "1"
            and _command_available("latex")
            and _command_available("dvipng")
        )
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.family": "serif",
            "font.serif": (
                ["Latin Modern Roman"]
                if dissertation_style
                else ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
            ),
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": bool(use_tex),
        }
    )
    if use_tex:
        if dissertation_style:
            matplotlib.rcParams["text.latex.preamble"] = (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{lmodern}"
                r"\usepackage{amsmath,amssymb}"
            )
            import matplotlib.pyplot as plt

            return plt

        # Match the journal build (mnras/rasti classes load newtxtext/newtxmath)
        # when the newtx fonts are installed; otherwise fall back to Computer
        # Modern, which matches the article-class draft build. An unconditional
        # newtx preamble crashes savefig on hosts without newtxtext.sty.
        try:
            has_newtx = (
                subprocess.run(
                    ["kpsewhich", "newtxtext.sty"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                != ""
            )
        except OSError:
            has_newtx = False
        preamble = r"\usepackage{amsmath}"
        if has_newtx:
            # newtxmath supplies the AMS symbol set; loading amssymb on top
            # of it clashes (\Bbbk etc. already defined).
            preamble += r"\usepackage{newtxtext,newtxmath}"
        else:
            preamble += r"\usepackage{amssymb}"
        matplotlib.rcParams["text.latex.preamble"] = preamble

    import matplotlib.pyplot as plt

    return plt
