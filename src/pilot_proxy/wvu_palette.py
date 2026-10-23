"""Generated from wvu_beamer_template/styles/beamercolorthemewvu.sty.

Standalone copy; update with tools/export_plot_palette.py in that repository.
"""

BLUE = "#002855"
GOLD = "#EEAA00"
STAR_CITY_BLUE = "#9DDAE6"
SAFETY_BLUE = "#0062A3"
CANARY = "#F7DD63"
SUNSET = "#F58672"
WOODBURN = "#8D4638"
HEMLOCK = "#6A724F"
WILD_FLOUR = "#F2E6C2"
BUCKSKIN = "#B3A169"
OLD_GOLD = "#7F6310"
RATTLER_GRAY = "#554741"
NOT_QUITE_WHITE = "#F7F7F7"
COOPERS_GRAY = "#BEB7B3"
SENECA_GRAY = "#988E8B"
COAL = "#1C2B39"

# White and black remain available for paper backgrounds and binary masks.
WHITE = "#FFFFFF"
BLACK = "#000000"
MEASURED = BLUE
MODEL = OLD_GOLD
CONDITIONAL = HEMLOCK
FAILURE = WOODBURN
PENDING = SENECA_GRAY
INK = COAL
MUTED = RATTLER_GRAY
SERIES = (BLUE, OLD_GOLD, SAFETY_BLUE, WOODBURN, HEMLOCK, SUNSET, COAL, SENECA_GRAY)
LINESTYLES = ("-", "--", "-.", ":", "--", "-.", ":", "-")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def tint(color, fraction=0.12):
    """Mix a palette color with white; fraction is the color's contribution."""
    if not 0 <= fraction <= 1:
        raise ValueError("tint fraction must be between zero and one")
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + fraction * (c - 255)):02X}" for c in rgb)


def sequential_cmap():
    """White-to-blue intensity scale with monotonically decreasing lightness.

    Missing values stay neutral gray. This map is for ordered values, not
    categories or cyclic phase. Importing palette constants needs no Matplotlib.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "wvu_blue", [NOT_QUITE_WHITE, STAR_CITY_BLUE, SAFETY_BLUE, BLUE], N=256
    )
    return cmap.with_extremes(bad=COOPERS_GRAY)


def plot_rcparams():
    """Color defaults only; leave journal fonts, dimensions, and ticks intact."""
    from cycler import cycler

    return {
        "axes.prop_cycle": cycler(color=SERIES),
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": tint(SENECA_GRAY, 0.3),
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
