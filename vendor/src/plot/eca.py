"""Figures, result records, and CSV io for the ECA scoring experiment.

Run this module directly to regenerate the combined figure from the saved CSV:

    uv run python src/plot/eca.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runnable as a script

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.patches import ConnectionPatch, Patch

from plot import palette as pal
from systems.eca import evolve_eca_np

# Wolfram class of each ECA rule that appears in the combined figure, after
# Wolfram, *A New Kind of Science* (2002). Rule 25 is not classified there; it is
# taken as class II following Vispoel et al., Physica D 432 (2022) 133074.
WOLFRAM_CLASS = {
    1: 2, 2: 2, 3: 2, 9: 2, 15: 2, 25: 2, 73: 2, 142: 2, 154: 2, 170: 2,
    22: 3, 30: 3, 45: 3, 105: 3, 150: 3,
    41: 4, 54: 4, 106: 4, 110: 4,
}

# One Nature hue per class, ordered as a complexity ladder: periodic (cool) ->
# chaotic -> complex (hot). Class I is defined for completeness though absent here.
CLASS_COLOR = {1: pal.CAT6[2], 2: pal.CAT6[0], 3: pal.CAT6[3], 4: pal.CAT6[1]}
CLASS_LABEL = {
    1: "Class I: homogeneous",
    2: "Class II: periodic",
    3: "Class III: chaotic",
    4: "Class IV: complex",
}


def _class_color(rule: int) -> tuple:
    """Bar color: the rule's Wolfram-class hue, at full saturation.

    Fail fast: a rule missing from WOLFRAM_CLASS is a data gap, not a color to
    paint grey. A silent fallback once shipped unclassified rules to the figure.
    """
    if rule not in WOLFRAM_CLASS:
        raise KeyError(
            f"ECA rule {rule} has no Wolfram class in WOLFRAM_CLASS; "
            f"add it before plotting rather than letting it fall through to grey."
        )
    return mcolors.to_rgb(CLASS_COLOR[WOLFRAM_CLASS[rule]])


@dataclass(frozen=True)
class RuleResult:
    rule: int
    reservoir_epiplexity: float
    num_samples: int
    width: int
    feature_dim: int
    residual: float = 0.0
    reservoir_epiplexity_std: float = 0.0
    residual_std: float = 0.0
    n_runs: int = 1


def save_csv(results: list[RuleResult], output_dir: str, extra: dict) -> str:
    path = os.path.join(output_dir, "eca_reservoir_scores.csv")
    fieldnames = [
        "rule",
        "reservoir_epiplexity",
        "reservoir_epiplexity_std",
        "residual",
        "residual_std",
        "n_runs",
        "num_samples",
        "width",
        "feature_dim",
        "reservoir_activation",
        "ridge_lambda",
        "dt",
        "burnin_steps",
        "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.__dict__.copy()
            row.update(extra)
            writer.writerow(row)
    return path


def load_results_from_csv(path: str) -> list[RuleResult]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(
                RuleResult(
                    rule=int(row["rule"]),
                    reservoir_epiplexity=float(row["reservoir_epiplexity"]),
                    num_samples=int(row["num_samples"]),
                    width=int(row["width"]),
                    feature_dim=int(row["feature_dim"]),
                    residual=float(row.get("residual", 0.0) or 0.0),
                    reservoir_epiplexity_std=float(
                        row.get("reservoir_epiplexity_std", 0.0) or 0.0
                    ),
                    residual_std=float(row.get("residual_std", 0.0) or 0.0),
                    n_runs=int(row.get("n_runs", 1) or 1),
                )
            )
    return results


# Space-time thumbnails in plain black-on-white (active cells black), so the only
# color in the figure is the per-class bar coding.
INSET_CMAP = "binary"

# Figure font calibration. The paper body is a 10pt font on a 6.5in text width
# (single-column `article`). We draw the combined ECA figure at exactly the text
# width and include it at width=\textwidth (scale 1:1), so points set here equal
# points on the page. Point-scale type (design system): axis title at the label
# tier, tick numbers a step down, inset titles at the floor.
COMBINED_FIG_WIDTH_IN = 6.5
COMBINED_FIG_HEIGHT_IN = 3.1
AXIS_LABEL_FONTSIZE = pal.FS_LABEL
TICK_FONTSIZE = pal.FS_TICK
INSET_TITLE_FONTSIZE = pal.FS_TICK

def _build_eca_spacetime(rule: int, width: int, steps: int, burnin: int, seed: int) -> np.ndarray:
    """Spacetime image for a single ECA rule, shape (steps, width)."""
    rng = np.random.default_rng(seed + rule)
    state = rng.integers(0, 2, size=width, dtype=np.uint8)
    if burnin > 0:
        state = evolve_eca_np(state, burnin, rule)
    history = np.empty((steps, width), dtype=np.uint8)
    history[0] = state
    for t in range(1, steps):
        state = evolve_eca_np(state, 1, rule)
        history[t] = state
    return history


def _add_spacetime_insets(fig, ax, rules, x_by_rule, score_by_rule,
                          label_fontsize: int, seed: int) -> None:
    """Row of square spacetime thumbnails justified across the top of the axes, in
    the given order, each linked by a dashed line down to the top of its bar."""
    fig.canvas.draw()
    ax_bbox = ax.get_position()
    ax_w_in = ax_bbox.width * fig.get_figwidth()
    ax_h_in = ax_bbox.height * fig.get_figheight()

    n = len(rules)
    size_in = 0.74
    w_frac = size_in / ax_w_in
    h_frac = size_in / ax_h_in
    # Spread the thumbnails evenly across the full axes width (justified), so each
    # sits roughly above its bar and the connectors stay near-vertical.
    side = 0.02
    gap_frac = (1.0 - 2.0 * side - n * w_frac) / (n - 1) if n > 1 else 0.0
    y0 = 0.94 - h_frac  # leave room for the "Rule X" title above

    for i, rule in enumerate(rules):
        x = side + i * (w_frac + gap_frac)
        inset = ax.inset_axes([x, y0, w_frac, h_frac])
        spacetime = _build_eca_spacetime(rule, width=64, steps=64, burnin=16, seed=seed)
        inset.imshow(spacetime, cmap=INSET_CMAP, aspect="auto", interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        # Black border and title; the class color lives on the bars, not the thumbnails.
        inset.set_title(f"Rule {rule}", fontsize=label_fontsize, color=pal.INK, pad=3)
        for spine in inset.spines.values():
            spine.set_edgecolor(pal.INK)
            spine.set_linewidth(pal.LW_CONNECTOR)
        # Dashed leader from the thumbnail's bottom centre to the top of its bar.
        fig.add_artist(
            ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=inset.transAxes,
                xyB=(x_by_rule[rule], score_by_rule[rule]), coordsB=ax.transData,
                color=pal.OUTLINE, linewidth=pal.LW_CONNECTOR,
                linestyle="--", alpha=0.7, zorder=1,
            )
        )


def plot_combined(
    results: list,
    output_dir: str,
    reference_rules: list[int],
    top_n: int,
    seed: int,
) -> str:
    """Top-N rules by epiplexity merged with the reference rules; highlight reference."""
    by_rule = {r.rule: r for r in results}
    ranked = sorted(results, key=lambda r: r.reservoir_epiplexity, reverse=True)
    top_rules = {r.rule for r in ranked[:top_n]}
    pool_rules = sorted(
        top_rules | set(reference_rules),
        key=lambda rule: by_rule[rule].reservoir_epiplexity,
        reverse=True,
    )
    scores = [by_rule[rule].reservoir_epiplexity for rule in pool_rules]
    scores_std = [by_rule[rule].reservoir_epiplexity_std for rule in pool_rules]
    # Color every bar by its Wolfram class, all at the same saturation; the reference
    # rules are instead singled out by the space-time insets drawn above them.
    colors = [_class_color(rule) for rule in pool_rules]
    x = np.arange(len(pool_rules))

    pal.apply_style()
    fig, ax = plt.subplots(figsize=(COMBINED_FIG_WIDTH_IN, COMBINED_FIG_HEIGHT_IN))
    ax.bar(
        x,
        scores,
        width=0.78,
        color=colors,
        yerr=scores_std,
        error_kw={"elinewidth": 0.7, "capsize": 1.8, "ecolor": pal.ON_SURFACE_VARIANT},
        zorder=2,
    )
    ax.set_xlabel("ECA rule", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$S^\phi$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(rule) for rule in pool_rules], rotation=0)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y", alpha=pal.GRID_ALPHA)
    pal.despine(ax)

    # Headroom above the tallest bar so the spacetime insets sit in empty space
    # instead of overlapping the bars.
    ymax = max(s + e for s, e in zip(scores, scores_std))
    top = ymax * 1.95
    ax.set_ylim(top=top)

    # Class key on the right, one swatch per class present in the pool. Centered at
    # S^phi ~ 52 so it sits low in the empty band, clear of the insets above.
    present = sorted({WOLFRAM_CLASS[r] for r in pool_rules})
    handles = [
        Patch(facecolor=CLASS_COLOR[c], edgecolor="none", label=CLASS_LABEL[c])
        for c in present
    ]
    # Built as a figure-level artist with a high zorder so it sits above the dashed
    # inset leaders (which are figure-level artists and would otherwise cross over it).
    legend = Legend(
        ax,
        handles,
        [h.get_label() for h in handles],
        loc="center right",
        bbox_to_anchor=(1.0, 52.0 / top),
        bbox_transform=ax.transAxes,
        fontsize=TICK_FONTSIZE,
        frameon=True,
        handlelength=1.1,
        handleheight=1.1,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.6,
    )
    legend.set_zorder(30)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(pal.OUTLINE_VARIANT)
    legend.get_frame().set_linewidth(0.6)
    fig.add_artist(legend)

    # Insets = the reference rules, ordered by their actual S^phi (descending) so
    # they line up left-to-right with their bars; linked to each bar top by a leader.
    inset_rules = sorted(
        reference_rules, key=lambda r: by_rule[r].reservoir_epiplexity, reverse=True
    )
    x_by_rule = {rule: i for i, rule in enumerate(pool_rules)}
    score_by_rule = {rule: by_rule[rule].reservoir_epiplexity for rule in pool_rules}

    fig.tight_layout()
    _add_spacetime_insets(
        fig, ax, inset_rules, x_by_rule, score_by_rule,
        label_fontsize=INSET_TITLE_FONTSIZE, seed=seed,
    )
    path = os.path.join(output_dir, "eca_combined.pdf")
    # The bars/text/leaders stay vector; only the imshow spacetime insets are
    # raster. Embed them at high dpi so LaTeX scaling stays crisp (with
    # interpolation="nearest" each ECA cell is a sharp square, not a blur).
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def main() -> None:
    import eca as exp  # experiment constants (results dir, reference rules)

    results = load_results_from_csv(os.path.join(exp.RESULTS_ROOT, "eca_reservoir_scores.csv"))
    path = plot_combined(
        results, exp.RESULTS_ROOT,
        reference_rules=exp.REFERENCE_RULES, top_n=exp.COMBINED_TOP_N, seed=exp.SEED,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
