"""Figures for the inverse MNIST encoder experiment.

Drawing functions take already-computed data (training history arrays and
per-checkpoint 2D projections of the code) and only draw; the assembly
functions load the checkpoints saved by ``inverse_mnist.py``, encode the test
subset, project with t-SNE, and score the supervised probes.

Run this module directly to regenerate the figures from saved outputs:

    uv run python src/plot/inverse_mnist.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runnable as a script

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch
from sklearn.manifold import TSNE

from plot import palette as pal
from systems.checkpoints import checkpoint_paths
from systems.mnist import MNISTEncoder, encode_dataset, mnist_test_set, probe_accuracies, sample_vis_subset

# Paper body text width (single-column `article`, \textwidth = 6.5in). Figures laid out
# wider than this and included at width=\textwidth get downscaled by
# (PAPER_TEXTWIDTH_IN / fig_w); font sizes are divided by that factor to land at the
# intended on-page point sizes (9pt labels, 8pt ticks, 7pt sub-labels, 11pt panel letters).
PAPER_TEXTWIDTH_IN = 6.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HISTORY_FIELDS = ["step", "epiplexity", "z_l2", "residual_std", "grad_norm", "lr"]
DETERMINISTIC_PDF_METADATA = {
    "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
}


def history_path(output_dir: str) -> str:
    return os.path.join(output_dir, "epiplexity_history.csv")


def save_history_csv(history: list[dict[str, float]], path: str) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    return path


def load_history_csv(path: str) -> list[dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


# --- Assembly: load checkpoints, encode, project, probe -------------------------


def tsne_at_checkpoint(
    ckpt_path: str, images: torch.Tensor, perplexity: float, seed: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Encoder codes and their 2D t-SNE projection at one checkpoint; architecture
    comes from the stored config."""
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    encoder = MNISTEncoder(
        encoder_dim=config["encoder_dim"],
        hidden=config["encoder_hidden"],
    ).to(DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    z = encode_dataset(encoder, images, DEVICE)
    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(z)
    return int(checkpoint["step"]), projection, z


def render_combined_figure(
    output_dir: str,
    data_dir: str,
    vis_samples_per_class: int,
    seed: int,
    perplexity: float,
) -> str:
    """Project the code at every checkpoint with t-SNE and draw the combined figure."""
    test_set = mnist_test_set(data_dir)
    images, labels = sample_vis_subset(
        test_set, samples_per_class=vis_samples_per_class, seed=seed
    )
    per_ckpt = [
        tsne_at_checkpoint(ckpt_path, images, perplexity, seed)
        for ckpt_path in checkpoint_paths(output_dir)
    ]
    embeddings = [(step, projection) for step, projection, _ in per_ckpt]
    probe_steps, knn_acc, linear_acc = [], [], []
    for step, _, z in per_ckpt:
        knn, linear = probe_accuracies(z, labels, seed)
        probe_steps.append(step)
        knn_acc.append(knn)
        linear_acc.append(linear)

    history = load_history_csv(history_path(output_dir))
    return plot_combined_mnist_figure(
        embeddings=embeddings,
        labels=labels,
        history_steps=np.array([row["step"] for row in history]),
        history_scores=np.array([row["epiplexity"] for row in history]),
        probe_steps=np.array(probe_steps),
        knn_acc=np.array(knn_acc),
        linear_acc=np.array(linear_acc),
        path=os.path.join(output_dir, "z_projection_combined.pdf"),
    )


# --- Drawing -------------------------------------------------------------------


def plot_history(steps: list[float], scores: list[float], path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps, scores, color=pal.PRIMARY, linewidth=pal.LW_PRIMARY)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$S^\phi$")
    ax.set_title("Inverse MNIST Encoder Training")
    ax.grid(True, alpha=pal.GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(path, metadata=DETERMINISTIC_PDF_METADATA)
    plt.close(fig)
    return path


def plot_combined_mnist_figure(
    embeddings: list[tuple[int, np.ndarray]],
    labels: np.ndarray,
    history_steps: np.ndarray,
    history_scores: np.ndarray,
    probe_steps: np.ndarray,
    knn_acc: np.ndarray,
    linear_acc: np.ndarray,
    path: str,
) -> str:
    """Row of per-checkpoint t-SNE scatters (a) above the epiplexity curve (b); the
    small probe-accuracy panel (c) sits in the freed right third of the curve row."""
    steps = np.asarray([step for step, _ in embeddings])
    train_steps = float(history_steps[-1])

    cmap = pal.CAT10
    n = len(steps)

    ml, mr, mt, mb = 0.80, 0.30, 0.18, 0.90
    title_a = 0.32
    left_w = 10.0
    snap_gap = 0.18
    v_gap = 0.30              # room for the snapshot-to-curve connectors (legend now sits inside panel B)
    curve_h = 2.2

    # The snapshot row (A) keeps the full width; only the curve (B) shrinks to two-thirds,
    # and the freed right third of the bottom row holds the small accuracy panel C.
    b_col_w = left_w * 2.0 / 3.0
    bc_hgap = 0.95           # gap for panel C's y-axis label and panel B's right ticks
    c_w = left_w - b_col_w - bc_hgap

    snap_w = (left_w - (n - 1) * snap_gap) / n
    snap_h = snap_w

    fig_w = ml + left_w + mr
    fig_h = mb + curve_h + v_gap + snap_h + title_a + mt
    pal.apply_style()
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Calibrate fonts to the design point scale, compensating for the downscale to \textwidth.
    page_scale = PAPER_TEXTWIDTH_IN / fig_w
    label_fs = pal.FS_LABEL / page_scale
    tick_fs = pal.FS_TICK / page_scale
    sub_fs = pal.FS_TICK / page_scale
    letter_fs = (pal.FS_PANEL + 1) / page_scale

    def rect(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    snap_bottom = mb + curve_h + v_gap
    snap_top = snap_bottom + snap_h
    c_left = ml + b_col_w + bc_hgap

    x_lim = (0.8, (train_steps + 1) * 1.15)
    tick_steps = [0, 1, 10, 100, 500]
    tick_pos = [s + 1 for s in tick_steps]

    ax_curve = fig.add_axes(rect(ml, mb, b_col_w, curve_h))
    plot_steps = history_steps + 1
    ax_curve.plot(plot_steps, history_scores, color=pal.PRIMARY, linewidth=pal.LW_PRIMARY)
    ax_curve.set_xscale("log")
    ax_curve.set_xlim(*x_lim)
    ax_curve.set_xticks(tick_pos)
    ax_curve.set_xticklabels([str(s) for s in tick_steps])
    ax_curve.minorticks_off()
    ax_curve.set_xlabel("Training step", fontsize=label_fs)
    ax_curve.set_ylabel(r"$S^\phi$", fontsize=label_fs)
    ax_curve.tick_params(labelsize=tick_fs)
    ax_curve.grid(True, alpha=pal.GRID_ALPHA)
    pal.despine(ax_curve)

    # Panel c: supervised read-outs of the unsupervised code, a small panel on the right.
    ax_probe = fig.add_axes(rect(c_left, mb, c_w, curve_h))
    probe_x = probe_steps + 1
    ax_probe.plot(
        probe_x, linear_acc, color=pal.CAT6[1], linewidth=pal.LW_SECONDARY,
        marker="o", markersize=3.5, label="Linear probe",
    )
    ax_probe.plot(
        probe_x, knn_acc, color=pal.CAT6[2], linewidth=pal.LW_SECONDARY,
        marker="s", markersize=3.5, label="5-NN",
    )
    ax_probe.axhline(0.1, color=pal.GREY_500, linewidth=pal.LW_FINE, linestyle=":")
    ax_probe.set_xscale("log")
    ax_probe.set_xlim(*x_lim)
    ax_probe.set_xticks([s + 1 for s in (0, 10, 500)])
    ax_probe.set_xticklabels(["0", "10", "500"])
    ax_probe.minorticks_off()
    ax_probe.set_ylim(0.0, 1.0)
    ax_probe.set_yticks([0.0, 0.5, 1.0])
    ax_probe.set_xlabel("Training step", fontsize=label_fs)
    ax_probe.set_ylabel("Accuracy", fontsize=label_fs)
    ax_probe.tick_params(labelsize=tick_fs)
    ax_probe.grid(True, alpha=pal.GRID_ALPHA)
    pal.despine(ax_probe)
    probe_leg = ax_probe.legend(
        loc="lower right", fontsize=tick_fs, frameon=True,
        handlelength=1.2, handletextpad=0.3, borderaxespad=0.4, labelspacing=0.2,
    )
    probe_leg.get_frame().set_facecolor("white")
    probe_leg.get_frame().set_edgecolor(pal.OUTLINE_VARIANT)
    probe_leg.get_frame().set_linewidth(0.6)

    handles = None
    for i in range(n):
        step = int(steps[i])
        proj = embeddings[i][1]
        x_in = ml + i * (snap_w + snap_gap)
        ax = fig.add_axes(rect(x_in, snap_bottom, snap_w, snap_h))
        for digit in range(10):
            mask = labels == digit
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=5,
                color=cmap[digit],
                label=str(digit),
                alpha=0.7,
                linewidths=0,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {step}", fontsize=sub_fs, pad=3)
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()

        y_at = float(np.interp(step, history_steps, history_scores))
        sx = step + 1
        ax_curve.plot([sx], [y_at], "o", color=pal.ERROR, markersize=4, zorder=5)
        fig.add_artist(
            ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=ax.transAxes,
                xyB=(sx, y_at), coordsB=ax_curve.transData,
                color=pal.OUTLINE, linewidth=pal.LW_CONNECTOR, linestyle="--", alpha=0.7,
            )
        )

    # Digit legend tucked into panel B's empty bottom-right corner (the curve has
    # plateaued near the top there), as a vertical 2-column x 5-row block. A white
    # frame keeps it crisp over the gridlines.
    leg = ax_curve.legend(
        handles,
        [str(d) for d in range(10)],
        title="digit",
        ncol=2,
        loc="lower right",
        fontsize=tick_fs,
        title_fontsize=tick_fs,
        columnspacing=0.9,
        handletextpad=0.3,
        labelspacing=0.25,
        borderaxespad=0.6,
        markerscale=1.8,
        frameon=True,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(pal.OUTLINE_VARIANT)
    leg.get_frame().set_linewidth(0.6)

    letter_kw = dict(fontsize=letter_fs, fontweight="bold", va="bottom", ha="left")
    fig.text((ml - 0.40) / fig_w, snap_top / fig_h, "a", **letter_kw)
    fig.text((ml - 0.40) / fig_w, (mb + curve_h + 0.02) / fig_h, "b", **letter_kw)
    fig.text((c_left - 0.55) / fig_w, (mb + curve_h + 0.02) / fig_h, "c", **letter_kw)

    # Crop flush to the artwork: top/bottom white -> ~0, left/right just clears
    # the panel letters and axis labels, no clipping.
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata=DETERMINISTIC_PDF_METADATA,
    )
    plt.close(fig)
    return path


def main() -> None:
    import inverse_mnist as exp  # experiment constants (dirs, vis subset, t-SNE settings)

    history = load_history_csv(history_path(exp.OUTPUT_DIR))
    print(f"Replotting from existing checkpoints and CSV ({len(history)} steps)")
    plot_history(
        [row["step"] for row in history],
        [row["epiplexity"] for row in history],
        os.path.join(exp.OUTPUT_DIR, "epiplexity_curve.pdf"),
    )
    out = render_combined_figure(
        exp.OUTPUT_DIR, exp.DATA_DIR, exp.VIS_SAMPLES_PER_CLASS, exp.SEED, exp.TSNE_PERPLEXITY
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
