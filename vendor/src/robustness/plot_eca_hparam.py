"""Figure for the ECA hyperparameter robustness ablation.

Reads the summary CSV written by ``eca_hparam_scan.py`` from
``results/robustness/`` and draws the single combined figure (no recompute):

  robustness_combined   two panels sharing the offset-aligned columns
                        (rows = hyperparameter axes, reference column
                        outlined). Panel a: cell color = margin of rule 110
                        over the runner-up, cells print the value scored,
                        bold #rank added wherever rule 110 is not rank 1.
                        Panel b: cell color = rank(30) - rank(54), blue when
                        the class-IV rule 54 ranks above the chaotic rule 30;
                        cells print "#r30 / #r54" on one line.

Run:
    uv run python src/robustness/plot_eca_hparam.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot import palette as pal
from plot.palette import apply_style

from eca_hparam_scan import BASELINE, OUTPUT_DIR

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "eca_hparam_robustness_summary.csv")

# Heatmap row order. The width axis is excluded on the user's call: it varies
# the scored system's size, not the estimator.
HEATMAP_AXES = [
    "lambda_code", "ridge_lambda", "tau_max", "depth",
    "kernel_size", "channels", "num_samples",
]
# Row labels use the paper's notation (method section symbols), not code names.
AXIS_LABELS = {
    "lambda_code": r"resolution $\eta$",
    "ridge_lambda": r"ridge $\lambda$",
    "tau_max": r"window $\tau$",
    "depth": "depth",
    "kernel_size": "kernel size",
    "channels": "channels",
    "num_samples": r"samples $N$",
}


# ---------------------------------------------------------------- data loading

def load_summary() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["axis"], []).append(dict(
                value=float(r["value"]),
                margin=float(r["margin"]),
                rank110=int(r["rank110"]),
                rank30=int(r["rank30"]),
                rank54=int(r["rank54"]),
            ))
    for recs in rows.values():
        recs.sort(key=lambda d: d["value"])
    return rows


def offset_grid(summary: dict[str, list[dict]], field: str):
    """(nrows, ncols) array of ``field`` plus per-cell record lookup, columns
    aligned by step offset from the baseline value (baseline at offset 0)."""
    offsets_by_axis = {}
    for a in HEATMAP_AXES:
        values = [d["value"] for d in summary[a]]
        base_idx = values.index(float(BASELINE[a]))
        offsets_by_axis[a] = [i - base_idx for i in range(len(values))]
    all_offsets = sorted({o for offs in offsets_by_axis.values() for o in offs})
    col_of = {o: j for j, o in enumerate(all_offsets)}
    grid = np.full((len(HEATMAP_AXES), len(all_offsets)), np.nan)
    meta: dict[tuple[int, int], dict] = {}
    for i, a in enumerate(HEATMAP_AXES):
        for d, o in zip(summary[a], offsets_by_axis[a]):
            grid[i, col_of[o]] = d[field]
            meta[(i, col_of[o])] = d
    return grid, meta, all_offsets, col_of


def fmt_value(axis: str, v: float) -> str:
    if axis in ("lambda_code", "ridge_lambda"):
        return f"{v:g}"
    return f"{int(v)}"


def _cell_text_color(rgba) -> str:
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.5 else pal.INK


def _heatmap_frame(ax, all_offsets, col_of, nrows: int, xlabels: bool = True) -> None:
    """Shared cosmetics: baseline-column outline, offset ticks, no spines."""
    j0 = col_of[0]
    ax.add_patch(Rectangle((j0 - 0.5, -0.5), 1.0, nrows, fill=False,
                            edgecolor=pal.INK, linewidth=1.2, zorder=3))
    ax.set_xticks(range(len(all_offsets)))
    if xlabels:
        ax.set_xticklabels([("reference" if o == 0 else f"{o:+d}") for o in all_offsets])
        ax.set_xlabel("step offset from reference value")
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([AXIS_LABELS[a] for a in HEATMAP_AXES])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------- panels

def _draw_margin_panel(ax, summary: dict[str, list[dict]], xlabels: bool = True):
    """Margin heatmap into ``ax``; returns the image for the colorbar."""
    grid, meta, all_offsets, col_of = offset_grid(summary, "margin")
    vmax = np.nanmax(np.abs(grid))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = pal.DIVERGING_CMAP.reversed()  # positive margin -> blue, negative -> red
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for (i, j), d in meta.items():
        color = _cell_text_color(cmap(norm(grid[i, j])))
        label = fmt_value(HEATMAP_AXES[i], d["value"])
        text = label if d["rank110"] == 1 else f"{label}  #{d['rank110']}"
        weight = "normal" if d["rank110"] == 1 else "bold"
        ax.text(j, i, text, ha="center", va="center", fontsize=pal.FS_TICK,
                color=color, fontweight=weight)
    _heatmap_frame(ax, all_offsets, col_of, len(HEATMAP_AXES), xlabels=xlabels)
    return im


RANK_DIFF_CLAMP = 15.0


def _draw_rank_panel(ax, summary: dict[str, list[dict]], xlabels: bool = True):
    """Rank-comparison heatmap into ``ax``; returns the image for the colorbar.

    Cell text is a single line ``#r30 / #r54`` (order fixed: rule 30 first);
    the color already carries who is above whom, so no per-cell rule labels.
    """
    grid30, meta, all_offsets, col_of = offset_grid(summary, "rank30")
    grid54, _, _, _ = offset_grid(summary, "rank54")
    diff = grid30 - grid54  # positive: 54 ranked above 30 (theory-consistent)
    norm = mcolors.TwoSlopeNorm(vmin=-RANK_DIFF_CLAMP, vcenter=0.0, vmax=RANK_DIFF_CLAMP)
    cmap = pal.DIVERGING_CMAP.reversed()  # blue = 54 above 30, red = 30 above 54
    im = ax.imshow(np.clip(diff, -RANK_DIFF_CLAMP, RANK_DIFF_CLAMP), cmap=cmap,
                   norm=norm, aspect="auto", interpolation="nearest")
    for (i, j), d in meta.items():
        color = _cell_text_color(cmap(norm(np.clip(diff[i, j], -RANK_DIFF_CLAMP, RANK_DIFF_CLAMP))))
        ax.text(j, i, f"#{d['rank30']} / #{d['rank54']}", ha="center", va="center",
                fontsize=pal.FS_TICK, color=color)
    _heatmap_frame(ax, all_offsets, col_of, len(HEATMAP_AXES), xlabels=xlabels)
    return im


def _rank_colorbar(fig, im, ax) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"rank(30) $-$ rank(54)", fontsize=pal.FS_LABEL)
    cbar.ax.tick_params(labelsize=pal.FS_TICK)
    # End annotations: which side means which ordering.
    cbar.ax.text(0.5, 1.02, "54 above 30", transform=cbar.ax.transAxes,
                 fontsize=pal.FS_TICK, va="bottom", ha="center")
    cbar.ax.text(0.5, -0.02, "30 above 54", transform=cbar.ax.transAxes,
                 fontsize=pal.FS_TICK, va="top", ha="center")


def _margin_colorbar(fig, im, ax) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"margin: $S^\phi_{110}$ $-$ runner-up (bits)", fontsize=pal.FS_LABEL)
    cbar.ax.tick_params(labelsize=pal.FS_TICK)


# ---------------------------------------------------------------- figure

def plot_combined(summary: dict[str, list[dict]]) -> None:
    """Both heatmaps as one two-panel figure sharing the offset columns:
    (a) margin of rule 110, cells print the hyperparameter values;
    (b) ranks of rules 30/54, cells print ``#r30 / #r54``."""
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(6.5, 5.4), sharex=True,
        gridspec_kw=dict(hspace=0.12),
    )
    im_a = _draw_margin_panel(ax_a, summary, xlabels=False)
    _margin_colorbar(fig, im_a, ax_a)
    im_b = _draw_rank_panel(ax_b, summary)
    _rank_colorbar(fig, im_b, ax_b)
    ax_b.set_title("rank of rule 30 / rank of rule 54", fontsize=pal.FS_TICK,
                   color=pal.GREY_700, pad=3)
    pal.panel_label(ax_a, -0.14, 1.02, "a")
    pal.panel_label(ax_b, -0.14, 1.02, "b")
    _save(fig, "robustness_combined")


# ---------------------------------------------------------------- main

def main() -> None:
    plot_combined(load_summary())


if __name__ == "__main__":
    main()
