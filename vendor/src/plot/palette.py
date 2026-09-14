"""Scientific Figure Design System (Nature house conventions) -- unified palette.

Every experiment script imports its colors, line weights, and figure style from
here so the paper's visual language stays consistent. Translated from the
"Scientific Figure Design System" tokens: pure-white canvas, ink-black text,
the Nature-style house palette for data (Wong colourblind-safe set available for
strict CB-safe needs), point-scale type (5-8pt), safe stroke weights (>=0.5pt),
sentence-case labels, lowercase-bold panel labels, no decorative chrome.

The legacy MD3 symbol names (PRIMARY, ERROR, CAT6, ...) are preserved as aliases
onto the new Nature hues so existing call sites keep working unchanged.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ---- Canvas & ink (the only two colours most figures need) ----
CANVAS = "#ffffff"
INK = "#000000"

# ---- Neutral ramp -- gridlines, secondary labels, fills ----
GREY_900 = "#1a1a1a"
GREY_700 = "#4d4d4d"  # secondary text
GREY_500 = "#808080"  # tertiary labels, minor ticks
GREY_300 = "#b3b3b3"  # gridlines
GREY_200 = "#d9d9d9"  # faint fills

# ============================================================
# NATURE-STYLE HOUSE PALETTE -- primary data colours.
# The saturated 400-step of each hue leads.
# ============================================================
NAT_BLUE = "#2e67a6"
NAT_RED = "#b3443a"
NAT_GREEN = "#508837"
NAT_ORANGE = "#d96d1e"
NAT_PURPLE = "#944c87"
NAT_TEAL = "#3f8d98"
NAT_YELLOW = "#bd9737"
NAT_OLIVE = "#919a2b"
NAT_BROWN = "#65483a"  # skin-500
NAT_SLATE = "#434c61"  # grey-500

# Nature blue ramp (light -> dark) for single-hue sequential fills / thumbnails.
NAT_BLUE_RAMP = ["#c8e3f9", "#9dc3e7", "#5f8ec8", "#2e67a6", "#1e4484", "#0e2650"]

# Nature red ramp (light -> dark), the NAT_RED counterpart to NAT_BLUE_RAMP: for
# distinguishing a few members of one warm-hued condition by lightness alone.
NAT_RED_RAMP = ["#f4cfca", "#e39f97", "#cf6d62", "#b3443a", "#8c2f27", "#5c1a15"]

# ============================================================
# WONG / OKABE-ITO colourblind-safe set (Nature Methods 2011).
# Reach for these when a strict CB-safe set is required.
# ============================================================
WONG_BLACK = "#000000"
WONG_ORANGE = "#e69f00"
WONG_SKY = "#56b4e9"
WONG_GREEN = "#009e73"
WONG_YELLOW = "#f0e442"
WONG_BLUE = "#0072b2"
WONG_VERMILLION = "#d55e00"
WONG_PURPLE = "#cc79a7"
WONG = [WONG_BLACK, WONG_ORANGE, WONG_SKY, WONG_GREEN,
        WONG_YELLOW, WONG_BLUE, WONG_VERMILLION, WONG_PURPLE]

# ---- Legacy role aliases (mapped onto Nature hues) ----
PRIMARY = NAT_BLUE          # main data line / highlight
SECONDARY = NAT_TEAL
TERTIARY = NAT_ORANGE
ERROR = NAT_RED             # attention markers, "plain/ordered" series
ON_SURFACE = INK
ON_SURFACE_VARIANT = GREY_700
OUTLINE = GREY_500
OUTLINE_VARIANT = GREY_300

# Six-category palette (bar charts, multi-system comparisons). Six distinct
# Nature hues, ordered for maximum inter-category separation.
CAT6 = [NAT_BLUE, NAT_RED, NAT_GREEN, NAT_ORANGE, NAT_PURPLE, NAT_TEAL]

# Ten-category palette (MNIST digits, etc.). Nature hues + Wong sky for breadth;
# all dark enough to read as markers on a white canvas.
CAT10 = [
    NAT_BLUE,
    NAT_RED,
    NAT_GREEN,
    NAT_ORANGE,
    NAT_PURPLE,
    NAT_TEAL,
    NAT_YELLOW,
    WONG_SKY,
    NAT_BROWN,
    NAT_SLATE,
]

# ---- Colormaps ----
# Diverging blue-white-red for signed data around a midpoint (avoids red/green).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nat_diverging", ["#2166ac", "#f7f7f7", "#b2182b"]
)
# Single-hue sequential blues for ordered/continuous data.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "nat_sequential", ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#084594"]
)
# White -> Nature blue, for spacetime thumbnails on a white canvas.
BLUE_CMAP = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", NAT_BLUE])

# ============================================================
# STROKE / LINE WEIGHTS (at final print size; safe floor 0.5pt).
# ============================================================
LW_PRIMARY = 1.6      # emphasised data line (medium)
LW_SECONDARY = 1.1    # regular data line
LW_BASELINE = 0.8     # thin baseline / reference line
LW_FINE = 0.6         # hairline
LW_CONNECTOR = 0.8    # connectors, error bars, thin frames

GRID_ALPHA = 0.3

# ============================================================
# TYPE SCALE (points, at final reproduction size).
# ============================================================
FS_TICK = 6     # tick numbers, dense keys
FS_BODY = 7     # default in-figure text
FS_LABEL = 8    # axis titles, panel labels
FS_PANEL = 8    # panel labels (bold lowercase)

_FONT_STACK = ["Arimo", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply_style() -> None:
    """Set the design-system rcParams: sans type, sans mathtext, white canvas,
    point-scale fonts, safe strokes, light gridlines, vector-friendly output."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "font.size": FS_BODY,
        "axes.titlesize": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        # Sans mathtext so $S^\\phi$, $\\chi$, $\\tau$ match the sans body
        # instead of rendering in serif italic.
        "mathtext.fontset": "dejavusans",
        # Canvas: pure white everywhere.
        "figure.facecolor": CANVAS,
        "axes.facecolor": CANVAS,
        "savefig.facecolor": CANVAS,
        # Ink: black spines, ticks, text.
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        # Spines stay full by default so imshow/scatter panels keep a clean box;
        # line/bar plots call despine() to drop the top/right spines.
        # Ticks point outward, short.
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Gridlines: light grey, thin, used sparingly.
        "grid.color": GREY_300,
        "grid.linewidth": 0.5,
        "grid.alpha": GRID_ALPHA,
        # No decorative chrome.
        "legend.frameon": False,
        "axes.grid": False,
        # Vector-friendly, editable text in PDFs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(target, x: float, y: float, letter: str, fontsize: float = FS_PANEL):
    """Draw a panel label (bold, lowercase) at figure/axes fraction (x, y).

    ``target`` may be a Figure (``fig.text``) or Axes (``ax.text`` in axes
    fraction). Letters are lowercased to follow the house convention (a, b, c).
    """
    kw = dict(fontsize=fontsize, fontweight="bold", va="bottom", ha="left")
    if hasattr(target, "text") and hasattr(target, "transAxes"):
        target.text(x, y, letter.lower(), transform=target.transAxes, **kw)
    else:
        target.text(x, y, letter.lower(), **kw)


def despine(ax, top: bool = True, right: bool = True) -> None:
    """Hide the chosen spines (top/right by default) for an open-frame look."""
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
