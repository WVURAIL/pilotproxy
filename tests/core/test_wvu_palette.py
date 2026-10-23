"""Keep publication colors legible without changing scientific value ordering."""
import matplotlib
from matplotlib.colors import to_rgb
import numpy as np
import pytest

from pilot_proxy import wvu_palette as wvu


def luminance(rgb):
    rgb = np.asarray(rgb)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def test_sequential_scale_preserves_order_in_grayscale():
    cmap = wvu.sequential_cmap()
    lightness = luminance(cmap(np.linspace(0, 1, cmap.N))[:, :3])
    assert np.all(np.diff(lightness) < 0)
    assert cmap(1.0)[:3] == pytest.approx(to_rgb(wvu.BLUE))
    assert cmap(np.nan)[:3] == pytest.approx(to_rgb(wvu.COOPERS_GRAY))


def test_small_label_colors_have_contrast_on_white():
    for role in (wvu.INK, wvu.MUTED, wvu.MEASURED, wvu.MODEL, wvu.FAILURE, wvu.CONDITIONAL):
        contrast = 1.05 / (luminance(to_rgb(role)) + 0.05)
        assert contrast >= 4.5, (role, contrast)


def test_palette_leaves_journal_typography_and_line_semantics_alone():
    with matplotlib.rc_context({"font.size": 7, "lines.linestyle": "--", "figure.figsize": (3.5, 2)}):
        matplotlib.rcParams.update(wvu.plot_rcparams())
        assert matplotlib.rcParams["font.size"] == 7
        assert matplotlib.rcParams["figure.figsize"] == [3.5, 2]
        assert matplotlib.rcParams["lines.linestyle"] == "--"
        assert list(matplotlib.rcParams["axes.prop_cycle"])[0] == {"color": wvu.BLUE}
