"""Plot the inverse-MNIST one-at-a-time robustness heatmap."""

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

AXIS_LABELS = {
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "encoder_dim": "code dimension",
    "reservoir_depth": "reservoir depth",
    "ridge_lambda": r"ridge $\lambda$",
    "lambda_code": r"resolution $\eta$",
}


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expanded_rows(rows: list[dict], axis: str, baseline: dict) -> list[dict]:
    selected = [row for row in rows if row["axis"] == axis]
    reference = dict(next(row for row in rows if row["axis"] == "baseline"))
    reference.update(axis=axis, value=str(baseline[axis]))
    return sorted([*selected, reference], key=lambda row: float(row["value"]))


def plot_heatmap(output_dir: str, baseline: dict) -> None:
    rows = load_rows(os.path.join(output_dir, "mnist_hparam_metrics.csv"))
    axes = [axis for axis in AXIS_LABELS if any(row["axis"] == axis for row in rows)]
    offsets_by_axis, rows_by_axis = {}, {}
    for axis in axes:
        axis_rows = expanded_rows(rows, axis, baseline)
        rows_by_axis[axis] = axis_rows
        reference_index = [float(row["value"]) for row in axis_rows].index(float(baseline[axis]))
        offsets_by_axis[axis] = [index - reference_index for index in range(len(axis_rows))]
    offsets = sorted({offset for values in offsets_by_axis.values() for offset in values})
    columns = {offset: index for index, offset in enumerate(offsets)}
    grid = np.full((len(axes), len(offsets)), np.nan)
    labels = np.full(grid.shape, "", dtype=object)
    for row_index, axis in enumerate(axes):
        for row, offset in zip(rows_by_axis[axis], offsets_by_axis[axis]):
            column = columns[offset]
            accuracy = float(row["final_linear"])
            grid[row_index, column] = accuracy
            labels[row_index, column] = f"{float(row['value']):g}\n{accuracy:.2f}"

    pal.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = ax.imshow(grid, cmap=pal.DIVERGING_CMAP.reversed(), norm=norm, aspect="auto")
    for row_index, column in np.ndindex(labels.shape):
        if labels[row_index, column]:
            ax.text(column, row_index, labels[row_index, column], ha="center", va="center",
                    fontsize=pal.FS_TICK)
    ax.set_yticks(range(len(axes)), [AXIS_LABELS[axis] for axis in axes])
    ax.set_xticks(range(len(offsets)),
                  ["reference" if offset == 0 else f"{offset:+d}" for offset in offsets])
    ax.set_xlabel("step offset from reference value")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    reference_column = offsets.index(0)
    ax.add_patch(Rectangle((reference_column - 0.5, -0.5), 1.0, len(axes), fill=False,
                           edgecolor=pal.INK, linewidth=pal.LW_PRIMARY))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("final linear accuracy")
    fig.tight_layout()
    for extension, options in (("pdf", {}), ("png", {"dpi": 150})):
        fig.savefig(os.path.join(output_dir, f"mnist_hparam_robustness.{extension}"), **options)
    plt.close(fig)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot_heatmap(
        os.path.join(root, "results", "robustness", "mnist_hparam"),
        {"learning_rate": 1e-2, "batch_size": 128, "encoder_dim": 64,
         "reservoir_depth": 4, "ridge_lambda": 3.0, "lambda_code": 30.0},
    )
