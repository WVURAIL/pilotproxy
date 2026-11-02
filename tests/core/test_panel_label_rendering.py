"""Panel labels remain readable in previews and preserve the TeX contract."""

from io import StringIO
import xml.etree.ElementTree as ET

import matplotlib
import pytest

from pilot_proxy import figure_style as style


@pytest.mark.parametrize("position", [None, (-0.2, 1.12)])
def test_no_tex_preview_renders_bold_label_in_axes_coordinates(monkeypatch, position):
    monkeypatch.setattr(style.shutil, "which", lambda command: None)
    with matplotlib.rc_context():
        style.configure(require_tex=False)
        matplotlib.rcParams["svg.fonttype"] = "none"
        fig, ax = style.plt.subplots()
        try:
            kwargs = {} if position is None else dict(zip(("x", "y"), position))
            assert style.panel_label(ax, "a", **kwargs) is None
            label = ax.texts[-1]
            assert label.get_position() == (position or (-0.08, 1.035))
            assert label.get_transform() is ax.transAxes
            assert label.get_horizontalalignment() == "left"
            assert label.get_verticalalignment() == "bottom"
            assert label.get_fontsize() == 9.0
            assert label.get_color() == style.INK
            assert label.get_usetex() is False
            output = StringIO()
            fig.savefig(output, format="svg")
            rendered = ET.fromstring(output.getvalue())
            text_nodes = list(rendered.iter("{http://www.w3.org/2000/svg}text"))
            labels = ["".join(node.itertext()) for node in text_nodes]
            assert "(a)" in labels
            assert not any("\\textbf" in text for text in labels)
            assert label.get_fontweight() == "bold"
        finally:
            style.plt.close(fig)


def test_tex_panel_label_preserves_latex_text_and_inherited_font_weight():
    # No TeX process is needed to inspect the text artist's rendering contract.
    with matplotlib.rc_context({"text.usetex": True, "font.weight": "light"}):
        fig, ax = style.plt.subplots()
        try:
            assert style.panel_label(ax, "b", x=-0.1, y=1.08) is None
            label = ax.texts[-1]
            assert label.get_text() == r"\textbf{(b)}"
            assert label.get_usetex() is True
            assert label.get_position() == (-0.1, 1.08)
            assert label.get_transform() is ax.transAxes
            assert label.get_fontweight() == "light"
            assert label.get_fontsize() == 9.0
            assert label.get_color() == style.INK
        finally:
            style.plt.close(fig)
