"""Figure for the continuous-time dynamical-systems experiment.

Takes scored results and pre-integrated phase-portrait trajectories; only draws.
``portraits`` is a list of ``(name, trajectory, (i, j))`` in display order, where
``trajectory`` has shape ``(time, n_trajectories, state_dim)`` and ``(i, j)`` are
the state components to project onto.

Run this module directly to regenerate the figure from the saved CSV (the
phase-portrait trajectories are re-integrated; the scores are not recomputed):

    uv run python src/plot/dynamical_systems_experiment.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runnable as a script

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot import palette as pal

FIG_WIDTH_IN = 6.5
FIG_TITLE = pal.FS_LABEL
FIG_LABEL = pal.FS_LABEL
FIG_TICK = pal.FS_TICK
FIG_ANNOT = pal.FS_TICK

BAR_LABELS = {
    "Lorenz": "Lorenz",
    "Rossler": "Rossler",
    "Thomas": "Thomas",
    "Linear: pure rotation": "Pure rot.",
    "Linear: damped spiral": "Damped sp.",
    "Linear: stable node": "Stable node",
}
PORTRAIT_TITLES = {
    "Lorenz": "Lorenz",
    "Rossler": "Rossler",
    "Thomas": "Thomas",
    "Linear: pure rotation": "Pure rotation",
    "Linear: damped spiral": "Damped spiral",
    "Linear: stable node": "Stable node",
}

# Two-tone grouping (color groups, not decorates): the nonlinear/chaotic systems
# -- the high-epiplexity group -- in the accent blue; the linear systems in
# neutral grey, mirroring the ECA figure's highlight/baseline scheme.
_NONLINEAR = pal.NAT_BLUE
_LINEAR = pal.GREY_500
SYSTEM_COLORS = {
    "Lorenz": _NONLINEAR,
    "Rossler": _NONLINEAR,
    "Thomas": _NONLINEAR,
    "Linear: pure rotation": _LINEAR,
    "Linear: damped spiral": _LINEAR,
    "Linear: stable node": _LINEAR,
}


@dataclass(frozen=True)
class SystemResult:
    name: str
    state_dim: int
    epiplexity_mean: float
    epiplexity_std: float


def save_csv(results: list[SystemResult], path: str, extra: dict) -> None:
    fieldnames = [
        "name",
        "state_dim",
        "epiplexity_mean",
        "epiplexity_std",
        "num_samples",
        "hidden_dim",
        "ridge_lambda",
        "burnin_time",
        "delta_t",
        "rk4_step",
        "reservoir_repeats",
        "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = r.__dict__.copy()
            row.update(extra)
            writer.writerow(row)


def load_csv(path: str) -> list[SystemResult]:
    results: list[SystemResult] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(
                SystemResult(
                    name=row["name"],
                    state_dim=int(row["state_dim"]),
                    epiplexity_mean=float(row["epiplexity_mean"]),
                    epiplexity_std=float(row["epiplexity_std"]),
                )
            )
    return results


def plot_composite_figure(
    results: list,
    portraits: list[tuple[str, np.ndarray, tuple[int, int]]],
    path: str,
) -> None:
    pal.apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.0, 1.0, 1.0])

    # --- (A) bar chart ---
    ax = fig.add_subplot(gs[:, 0])
    names = [r.name for r in results]
    means = [r.epiplexity_mean for r in results]
    stds = [r.epiplexity_std for r in results]
    colors = [SYSTEM_COLORS[name] for name in names]
    x = np.arange(len(results))
    ax.bar(x, means, yerr=stds, capsize=2.0, width=0.74, color=colors,
           error_kw={"lw": pal.LW_CONNECTOR})
    ax.set_ylabel(r"$S^\phi$", fontsize=FIG_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [BAR_LABELS[n] for n in names], rotation=40, ha="right", fontsize=FIG_TICK
    )
    ax.tick_params(axis="y", labelsize=FIG_TICK)
    ax.margins(y=0.12)
    ax.grid(True, axis="y", alpha=pal.GRID_ALPHA)
    pal.despine(ax)
    for xi, m, s in zip(x, means, stds):
        ax.annotate(
            f"{m:.0f}" if m >= 10 else f"{m:.1f}",
            (xi, m + s), textcoords="offset points", xytext=(0, 2.5),
            ha="center", fontsize=FIG_ANNOT,
        )
    ax.annotate(
        "a", xy=(0, 1), xycoords="axes fraction", xytext=(-26, 5),
        textcoords="offset points", fontsize=pal.FS_PANEL, fontweight="bold", va="bottom",
    )

    # --- (B) phase portraits (each system uses its bar-chart color) ---
    for idx, (name, traj, (i, j)) in enumerate(portraits):
        row, col = divmod(idx, 3)
        axp = fig.add_subplot(gs[row, col + 1])
        n_traj = traj.shape[1]
        base_color = SYSTEM_COLORS[name]
        sys_cmap = LinearSegmentedColormap.from_list(
            "sys", ["#FFFFFF", base_color]
        )
        for k in range(n_traj):
            frac = 0.45 + 0.55 * k / max(n_traj - 1, 1)
            color = sys_cmap(frac)
            axp.plot(traj[:, k, i], traj[:, k, j], lw=0.6, color=color, alpha=0.85)
            axp.scatter(traj[0, k, i], traj[0, k, j], s=5, color=color)
        # Equal data scale on both axes so the attractor geometry is not
        # distorted (circles stay circular). Use datalim so every panel box keeps
        # its uniform grid size; only the data limits expand to equalize scale.
        axp.set_aspect("equal", adjustable="datalim")
        axp.set_title(PORTRAIT_TITLES[name], fontsize=FIG_TITLE, pad=2)
        axp.set_xticks([])
        axp.set_yticks([])
        if idx == 0:
            axp.annotate(
                "b", xy=(0, 1), xycoords="axes fraction", xytext=(-10, 5),
                textcoords="offset points", fontsize=pal.FS_PANEL, fontweight="bold", va="bottom",
            )

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    import dynamical_systems_experiment as exp  # experiment constants + portrait integration

    bar_csv = os.path.join(exp.RESULTS_ROOT, "dynamical_systems_scores.csv")
    figure_pdf = os.path.join(exp.RESULTS_ROOT, "dynamical_systems_figure.pdf")
    results = load_csv(bar_csv)
    taus = exp.compute_taus()
    plot_composite_figure(results, exp.portrait_trajectories(taus), figure_pdf)
    print(f"Wrote {figure_pdf}")


if __name__ == "__main__":
    main()
