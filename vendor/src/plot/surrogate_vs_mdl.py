"""Figure for the strict-MDL vs ridge-surrogate appendix comparison.

Loads the CSVs written by ``src/appendix/surrogate_vs_mdl.py`` and draws one
two-panel figure: (a) ECA per-rule means, (b) NCA checkpoints colored by
training step. Runnable standalone; also writes ``metrics.json`` with the
rank-agreement numbers quoted in the report.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot import palette as pal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "appendix", "surrogate_vs_mdl")
FIGURE_PATH = os.path.join(RESULTS_DIR, "surrogate_vs_mdl.pdf")

NCA_SEED_MARKERS = {1: "o", 2: "s", 3: "^"}
ECA_LABELED_TOP_N = 5


def _read_csv(path: str) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {k: float(v) for k, v in row.items()} for row in csv.DictReader(f)
        ]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = np.argsort(np.argsort(a)).astype(float)
    ranks_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _identity_line(ax, lo: float, hi: float) -> None:
    pad = 0.04 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        color=pal.GREY_500, lw=pal.LW_BASELINE, ls="--", zorder=1,
    )


def _eca_panel(ax, rows: list[dict[str, float]]) -> dict:
    rules = sorted({int(r["rule"]) for r in rows})
    mean_mdl, mean_ridge, std_mdl, std_ridge = [], [], [], []
    for rule in rules:
        s_mdl = np.array([r["s_mdl"] for r in rows if int(r["rule"]) == rule])
        s_ridge = np.array([r["s_ridge"] for r in rows if int(r["rule"]) == rule])
        mean_mdl.append(s_mdl.mean())
        mean_ridge.append(s_ridge.mean())
        std_mdl.append(s_mdl.std())
        std_ridge.append(s_ridge.std())
    mean_mdl = np.array(mean_mdl)
    mean_ridge = np.array(mean_ridge)

    lo = float(min(mean_mdl.min(), mean_ridge.min()))
    hi = float(max(mean_mdl.max(), mean_ridge.max()))
    _identity_line(ax, lo, hi)
    ax.errorbar(
        mean_mdl, mean_ridge, xerr=std_mdl, yerr=std_ridge,
        fmt="none", ecolor=pal.GREY_300, elinewidth=pal.LW_FINE, zorder=2,
    )
    ax.scatter(
        mean_mdl, mean_ridge, s=9, facecolor=pal.NAT_BLUE,
        edgecolor="none", zorder=3,
    )

    # Highlight the top rules under the strict MDL score; annotate only the
    # points a reader needs to find (labels on the full cluster overprint).
    top_idx = np.argsort(mean_mdl)[::-1][:ECA_LABELED_TOP_N]
    for i in top_idx:
        ax.scatter(
            mean_mdl[i], mean_ridge[i], s=13, facecolor=pal.NAT_RED,
            edgecolor="none", zorder=4,
        )
    annotations = {110: (3, -7, pal.NAT_RED), 22: (-2, 5, pal.NAT_RED)}
    if 41 in rules:  # the one large rank shift: 2nd under ridge, 13th under MDL
        annotations[41] = (-8, 5, pal.GREY_700)
    for rule, (dx, dy, color) in annotations.items():
        if rule not in rules:
            continue
        i = rules.index(rule)
        ax.annotate(
            str(rule), (mean_mdl[i], mean_ridge[i]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=pal.FS_TICK, color=color,
        )

    rho = spearman(mean_mdl, mean_ridge)
    ax.text(
        0.04, 0.96, f"Spearman $\\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=pal.FS_BODY,
    )
    ax.set_xlabel("$S^\\phi_{\\mathrm{MDL}}$ (bits)")
    ax.set_ylabel("$S^\\phi$ (bits)")
    pal.despine(ax)

    order_mdl = [rules[i] for i in np.argsort(mean_mdl)[::-1]]
    order_ridge = [rules[i] for i in np.argsort(mean_ridge)[::-1]]
    return {
        "spearman": rho,
        "pearson": float(np.corrcoef(mean_mdl, mean_ridge)[0, 1]),
        "top5_mdl": order_mdl[:5],
        "top5_ridge": order_ridge[:5],
        "top1_mdl": order_mdl[0],
        "top1_ridge": order_ridge[0],
        "mean_gap_bits": float(np.mean(mean_mdl - mean_ridge)),
        "max_gap_bits": float(np.max(mean_mdl - mean_ridge)),
    }


def _nca_panel(fig, ax, rows: list[dict[str, float]]) -> dict:
    steps = np.array([r["step"] for r in rows])
    norm = matplotlib.colors.Normalize(vmin=steps.min(), vmax=steps.max())
    xs, ys = [], []
    for seed, marker in NCA_SEED_MARKERS.items():
        sub = [r for r in rows if int(r["seed"]) == seed]
        by_step: dict[int, list[dict[str, float]]] = {}
        for r in sub:
            by_step.setdefault(int(r["step"]), []).append(r)
        for step in sorted(by_step):
            s_mdl = np.array([r["s_mdl"] for r in by_step[step]])
            s_ridge = np.array([r["s_ridge"] for r in by_step[step]])
            xs.append(s_mdl.mean())
            ys.append(s_ridge.mean())
            ax.errorbar(
                s_mdl.mean(), s_ridge.mean(), xerr=s_mdl.std(), yerr=s_ridge.std(),
                fmt="none", ecolor=pal.GREY_300, elinewidth=pal.LW_FINE, zorder=2,
            )
            ax.scatter(
                s_mdl.mean(), s_ridge.mean(), s=14, marker=marker,
                facecolor=pal.SEQUENTIAL_CMAP(norm(step)),
                edgecolor=pal.INK, linewidth=0.3, zorder=3,
            )
    xs, ys = np.array(xs), np.array(ys)
    _identity_line(ax, float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max())))

    rho = spearman(xs, ys)
    ax.text(
        0.04, 0.96, f"Spearman $\\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=pal.FS_BODY,
    )
    for seed, marker in NCA_SEED_MARKERS.items():
        ax.scatter(
            [], [], marker=marker, s=14, facecolor="none",
            edgecolor=pal.INK, linewidth=0.5, label=f"Seed {seed}",
        )
    ax.legend(frameon=False, fontsize=pal.FS_TICK, loc="lower right", handletextpad=0.1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=pal.SEQUENTIAL_CMAP)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Training step", fontsize=pal.FS_TICK)
    cbar.ax.tick_params(labelsize=pal.FS_TICK)
    cbar.outline.set_linewidth(pal.LW_FINE)
    ax.set_xlabel("$S^\\phi_{\\mathrm{MDL}}$ (bits)")
    ax.set_ylabel("$S^\\phi$ (bits)")
    pal.despine(ax)
    return {
        "spearman": rho,
        "pearson": float(np.corrcoef(xs, ys)[0, 1]),
        "mean_gap_bits": float(np.mean(xs - ys)),
        "max_gap_bits": float(np.max(xs - ys)),
    }


def render_figure() -> str:
    pal.apply_style()
    eca_path = os.path.join(RESULTS_DIR, "eca.csv")
    nca_path = os.path.join(RESULTS_DIR, "nca.csv")
    have_eca = os.path.exists(eca_path)
    have_nca = os.path.exists(nca_path)
    n_panels = int(have_eca) + int(have_nca)
    if n_panels == 0:
        raise FileNotFoundError(f"no CSVs under {RESULTS_DIR}")

    # Total width = the paper's 6.5in body text width, included at scale 1:1.
    fig, axes = plt.subplots(1, n_panels, figsize=(3.25 * n_panels, 2.6))
    axes = np.atleast_1d(axes)
    metrics: dict[str, dict] = {}
    col = 0
    if have_eca:
        metrics["eca"] = _eca_panel(axes[col], _read_csv(eca_path))
        pal.panel_label(axes[col], -0.16, 1.02, "a")
        col += 1
    if have_nca:
        metrics["nca"] = _nca_panel(fig, axes[col], _read_csv(nca_path))
        if n_panels == 2:
            pal.panel_label(axes[col], -0.16, 1.02, "b")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return FIGURE_PATH


if __name__ == "__main__":
    print(render_figure())
