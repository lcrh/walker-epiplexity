"""Figures for the inverse NCA experiment.

Drawing functions take already-computed data (training history arrays and
space-time traces) and only draw; the assembly functions above them load the
checkpoints saved by ``inverse_nca.py``, roll the NCA out, and hand the traces
to the drawing code. A "snapshot" is a pair ``(step, trace)`` where ``trace``
has shape ``(time + 1, width)``.

Run this module directly to regenerate the figures from saved outputs:

    uv run python src/plot/inverse_nca.py             # combined paper figure
    uv run python src/plot/inverse_nca.py --appendix  # appendix variant grid
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # runnable as a script

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot import palette as pal
from systems.checkpoints import checkpoint_paths, load_gzip_checkpoint
from systems.nca import LocalNCA1D, random_unit_state, rollout_trace

# Paper body text width (single-column `article`, \textwidth = 6.5in). Figures that
# are laid out wider than this and included at width=\textwidth get downscaled by
# (PAPER_TEXTWIDTH_IN / fig_w); font sizes are divided by that factor to land at the
# intended on-page point sizes.
PAPER_TEXTWIDTH_IN = 6.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HISTORY_FIELDS = ["step", "epiplexity", "y_l2", "residual_std", "grad_norm"]


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


# --- Assembly: load checkpoints, roll out, gather panel data -------------------


def load_nca_at_step(run_dir: str, step: int, device: str = DEVICE) -> LocalNCA1D:
    """Rebuild the NCA saved at ``step``; architecture comes from the stored config."""
    path = os.path.join(run_dir, "checkpoints", f"nca_step_{step:04d}.pt")
    checkpoint = load_gzip_checkpoint(path, map_location=device)
    config = checkpoint["config"]
    nca = LocalNCA1D(update_mode=config["update_mode"]).to(device)
    nca.load_state_dict(checkpoint["nca_state_dict"])
    return nca


def step_snapshots(
    run_dir: str, seed: int, steps: tuple[int, ...], width: int, time_steps: int
) -> list[tuple[int, np.ndarray]]:
    """Roll the checkpoints at ``steps`` out from one shared random initial state."""
    torch.manual_seed(seed + 1000)
    initial = random_unit_state(1, width, DEVICE)
    return [
        (
            step,
            rollout_trace(load_nca_at_step(run_dir, step), initial.clone(), time_steps),
        )
        for step in steps
    ]


def load_mode_band(
    variants_root: str, mode: str, seeds: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate the epiplexity curves of one update mode across ``seeds``.

    Returns ``(steps, mean, std)`` computed pointwise over the per-seed runs under
    ``variants_root/<mode>/seed_<s>/`` (all seeds share the same step grid)."""
    steps_ref: np.ndarray | None = None
    curves: list[np.ndarray] = []
    for seed in seeds:
        history = load_history_csv(
            history_path(os.path.join(variants_root, mode, f"seed_{seed}"))
        )
        steps = np.array([row["step"] for row in history])
        curves.append(np.array([row["epiplexity"] for row in history]))
        if steps_ref is None:
            steps_ref = steps
    stacked = np.stack(curves)
    return steps_ref, stacked.mean(axis=0), stacked.std(axis=0)


def render_combined_figure(
    output_dir: str,
    seeds: tuple[int, ...],
    train_steps: int,
    variants_root: str,
    modes: tuple[str, ...],
    variant_seeds: tuple[int, ...],
    snapshot_steps: tuple[int, ...] | None = None,
    band_width: int = 128,
    band_time: int = 128,
) -> str:
    """Aggregate the per-seed runs under ``output_dir/seed_<s>/`` into the paper figure.

    Panel (a) rolls out the ``seeds`` runs under ``output_dir`` (the formal residual
    run); panel (b) plots those same seeds' epiplexity curves individually; panel (c)
    shows two mean +/- s.d. bands, one per update mode, aggregated over ``variant_seeds``
    of the runs under ``variants_root/<mode>/``."""
    # Six evenly spaced checkpoints (0 .. train_steps) unless overridden.
    if snapshot_steps is None:
        snapshot_steps = tuple(range(0, train_steps + 1, train_steps // 5))
    seed_histories: list[tuple[np.ndarray, np.ndarray]] = []
    per_seed_snapshots: list[list[tuple[int, np.ndarray]]] = []
    for seed in seeds:
        run_dir = os.path.join(output_dir, f"seed_{seed}")
        history = load_history_csv(history_path(run_dir))
        seed_histories.append(
            (
                np.array([row["step"] for row in history]),
                np.array([row["epiplexity"] for row in history]),
            )
        )
        per_seed_snapshots.append(
            step_snapshots(run_dir, seed, snapshot_steps, band_width, band_time)
        )
    bands = {mode: load_mode_band(variants_root, mode, variant_seeds) for mode in modes}
    # The panel-(a) rollouts and the panel-(b) exemplar curves are the formal run,
    # whose update mode the exemplars are coloured to match.
    first_config = load_gzip_checkpoint(
        checkpoint_paths(os.path.join(output_dir, f"seed_{seeds[0]}"))[-1],
        map_location="cpu",
    )["config"]
    return plot_combined_figure(
        seeds=list(seeds),
        seed_histories=seed_histories,
        train_steps=train_steps,
        per_seed_snapshots=per_seed_snapshots,
        bands=bands,
        exemplar_mode=first_config["update_mode"],
        path=os.path.join(output_dir, "inverse_nca_combined.pdf"),
    )


def render_appendix_variant_grid(
    root: str,
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    scale_width: int = 256,
    scale_time: int = 120,
) -> str:
    """Build the direct/residual x seed grid: each cell is the final-checkpoint
    rollout of the run under ``root/<mode>/seed_<s>/``. The cells are returned in
    mode-major order (all seeds of the first mode, then all of the second)."""
    cells: list[dict[str, object]] = []
    for mode in modes:
        for seed in seeds:
            run_dir = os.path.join(root, mode, f"seed_{seed}")
            history = load_history_csv(history_path(run_dir))
            final_step = int(
                load_gzip_checkpoint(checkpoint_paths(run_dir)[-1], map_location="cpu")[
                    "step"
                ]
            )
            torch.manual_seed(seed + 4000)
            initial = random_unit_state(1, scale_width, DEVICE)
            trace = rollout_trace(
                load_nca_at_step(run_dir, final_step), initial, scale_time
            )
            cells.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "final_score": history[-1]["epiplexity"],
                    "trace": trace,
                }
            )

    return plot_appendix_variant_grid(
        cells,
        len(modes),
        len(seeds),
        os.path.join(root, "inverse_nca_appendix_variants.pdf"),
    )


# --- Drawing -------------------------------------------------------------------


def ema_subsample(
    steps: np.ndarray, *series: np.ndarray, span: int = 20, stride: int = 10
) -> tuple[np.ndarray, ...]:
    """Smooth each of ``series`` with an exponential moving average, then keep every
    ``stride``-th sample. Stride slicing (not binning) preserves the very first sample,
    so the fast initial rise of the epiplexity curve is not flattened away."""
    alpha = 2.0 / (span + 1.0)
    smoothed: list[np.ndarray] = []
    for y in series:
        s = np.empty_like(y, dtype=float)
        s[0] = y[0]
        for i in range(1, len(y)):
            s[i] = alpha * y[i] + (1.0 - alpha) * s[i - 1]
        smoothed.append(s[::stride])
    return (steps[::stride], *smoothed)


def plot_combined_figure(
    seeds: list[int],
    seed_histories: list[tuple[np.ndarray, np.ndarray]],
    train_steps: int,
    per_seed_snapshots: list[list[tuple[int, np.ndarray]]],
    bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    exemplar_mode: str,
    path: str,
) -> str:
    """Combined figure: panel (a) spans the top row; panels (b) and (c) share the
    bottom row.

    (a) A grid of space-time diagrams: one row per seed, one column per training
        step, showing how each seed's rule evolves from step 0 to the final step.
        Rows carry a neutral seed label; the images are not colour-keyed by seed.
    (b) The epiplexity training curve of each formal-run seed individually, one line
        per seed along the ``exemplar_mode`` hue ramp.
    (c) One shaded band per update mode giving the mean +/- s.d. over nine seeds
        (``bands``), the two modes in two Nature hues; ``exemplar_mode`` (the mode
        drawn in panel (a)) is drawn last so it reads on top.

    All curves are EMA-smoothed and stride-subsampled. Sized to the paper body width
    (6.5in) so the point-scale fonts land correctly at width=\\textwidth; every image
    panel keeps its true aspect ratio.
    """
    n_seeds = len(seeds)
    n_cols = len(per_seed_snapshots[0])

    band_time, band_width = per_seed_snapshots[0][0][1].shape
    band_time -= 1

    # Two Nature hues for the two update rules; the formal (exemplar) mode is drawn
    # in the more saturated red, the comparison mode in blue. Series stay distinct by
    # hue and by band-vs-line role, so the panel survives greyscale.
    MODE_COLORS = {"direct": pal.NAT_BLUE, "residual": pal.NAT_RED}
    MODE_RAMPS = {"direct": pal.NAT_BLUE_RAMP, "residual": pal.NAT_RED_RAMP}
    # Three lightness steps along the exemplar-mode ramp: the seeds read as variations
    # of one condition and stay distinct by lightness alone (survives greyscale).
    seed_colors = [MODE_RAMPS[exemplar_mode][i] for i in (1, 3, 5)][:n_seeds]

    # --- Layout, in inches (figure width fixed to the body text width) ---
    ml, mr, mt, mb = 0.55, 0.15, 0.26, 0.55  # outer margins
    seed_gutter = 0.30  # left gutter for row (seed) labels
    header_h = 0.18  # step numbers above panel (a)
    snap_gap, row_gap = 0.06, 0.06  # panel (a) tile gaps
    curve_h, v_gap = 1.45, 0.30  # bottom-row (b, c) height / gap above it
    curve_gap = 0.68  # gap between panels (b) and (c) (room for (c)'s y axis)

    fig_w = PAPER_TEXTWIDTH_IN
    strip_w = fig_w - ml - mr - seed_gutter
    snap_w = (strip_w - (n_cols - 1) * snap_gap) / n_cols
    ratio_a = (band_time + 1) / band_width  # snapshot height / width
    snap_h = snap_w * ratio_a
    a_h = n_seeds * snap_h + (n_seeds - 1) * row_gap

    strip_x0 = ml + seed_gutter
    curve_w = (strip_w - curve_gap) / 2  # each bottom panel is half the strip
    fig_h = mb + curve_h + v_gap + a_h + header_h + mt
    pal.apply_style()
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Font calibration: fig_w == body width, so page_scale == 1 and the point-scale
    # tiers render at their nominal sizes on the page.
    page_scale = PAPER_TEXTWIDTH_IN / fig_w
    label_fs = pal.FS_LABEL / page_scale
    tick_fs = pal.FS_TICK / page_scale
    sub_fs = pal.FS_TICK / page_scale
    letter_fs = (pal.FS_PANEL + 1) / page_scale

    def rect(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    a_bottom = mb + curve_h + v_gap
    a_top = a_bottom + a_h

    def style_curve_axis(ax) -> None:
        ax.set_xlim(0, train_steps)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Training step", fontsize=label_fs)
        ax.set_ylabel(r"$S^\phi$", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=pal.GRID_ALPHA)
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=tick_fs,
            handlelength=1.4,
            borderaxespad=0.4,
        )
        pal.despine(ax)

    # Panel (b) (bottom-left): the formal run's seeds, one EMA-smoothed line each.
    ax_b = fig.add_axes(rect(strip_x0, mb, curve_w, curve_h))
    for color, seed, (steps, scores) in zip(seed_colors, seeds, seed_histories):
        s_steps, s_scores = ema_subsample(steps, scores)
        ax_b.plot(
            s_steps, s_scores, color=color, linewidth=pal.LW_SECONDARY, label=f"Seed {seed}"
        )
    style_curve_axis(ax_b)

    # Panel (c) (bottom-right): mean +/- s.d. band per update mode over nine seeds.
    ax_c = fig.add_axes(rect(strip_x0 + curve_w + curve_gap, mb, curve_w, curve_h))
    # Draw the comparison mode first so the exemplar mode's band sits on top.
    for mode in sorted(bands, key=lambda m: m == exemplar_mode):
        c_steps, c_mean, c_std = ema_subsample(*bands[mode])
        color = MODE_COLORS[mode]
        ax_c.fill_between(
            c_steps, c_mean - c_std, c_mean + c_std, color=color, alpha=0.16, linewidth=0
        )
        ax_c.plot(
            c_steps, c_mean, color=color, linewidth=pal.LW_SECONDARY, label=mode.capitalize()
        )
    style_curve_axis(ax_c)

    # Panel (a): row per seed (top = first seed), column per training step.
    for r in range(n_seeds):
        # Row order runs top-to-bottom, so the top row is the last in y.
        row_bottom = a_bottom + (n_seeds - 1 - r) * (snap_h + row_gap)
        for c, (step, trace) in enumerate(per_seed_snapshots[r]):
            x_in = strip_x0 + c * (snap_w + snap_gap)
            ax = fig.add_axes(rect(x_in, row_bottom, snap_w, snap_h))
            ax.imshow(
                trace,
                aspect="auto",
                interpolation="nearest",
                cmap=pal.DIVERGING_CMAP,
                vmin=-1,
                vmax=1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(str(step), fontsize=sub_fs, pad=2)
        # Neutral seed label in the left gutter (rows are not colour-keyed by seed).
        fig.text(
            (ml + seed_gutter - 0.09) / fig_w,
            (row_bottom + snap_h / 2) / fig_h,
            f"Seed {seeds[r]}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=sub_fs,
            color=pal.INK,
        )
    # "Training step" hint centred over the step-number header row.
    fig.text(
        (strip_x0 + strip_w / 2) / fig_w,
        (a_top + header_h + 0.01) / fig_h,
        "Training step",
        va="bottom",
        ha="center",
        fontsize=sub_fs,
    )

    # Panel letters (bold lowercase), just outside each panel's top-left corner.
    letter_kw = dict(fontsize=letter_fs, fontweight="bold", va="bottom", ha="left")
    curve_letter_y = (mb + curve_h + 0.02) / fig_h
    fig.text((ml - 0.30) / fig_w, a_top / fig_h, "a", **letter_kw)
    fig.text((ml - 0.30) / fig_w, curve_letter_y, "b", **letter_kw)
    fig.text(
        (strip_x0 + curve_w + curve_gap - 0.42) / fig_w, curve_letter_y, "c", **letter_kw
    )

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def plot_appendix_variant_grid(
    cells: list[dict[str, object]], n_modes: int, n_seeds: int, path: str
) -> str:
    """Grid of final NCA rollouts showing the regime is seed-robust under both
    update rules. Each mode's ``n_seeds`` runs fill a 3-column block of rows; the
    blocks stack (direct above residual) into a 6x3 grid. Each cell is that run's
    final-checkpoint space-time rollout, annotated with its score.

    Sized to the paper body width (6.5in) so the point-scale fonts land correctly
    at width=\\textwidth; cells keep their true space-time aspect ratio.
    """
    n_cols = 3
    rows_per_mode = n_seeds // n_cols  # 3
    n_rows = n_modes * rows_per_mode  # 6
    trace0 = np.asarray(cells[0]["trace"])
    t_time, t_width = trace0.shape[0] - 1, trace0.shape[1]
    ratio = (t_time + 1) / t_width  # cell height / width

    # --- Layout, in inches (figure width fixed to the body text width) ---
    ml, mr, mt, mb = 0.04, 0.06, 0.16, 0.14
    gutter = 0.24  # left label gutter (rotated update-mode word, spanning its block)
    col_gap, row_gap = 0.07, 0.07
    mode_gap = 0.16  # extra vertical gap between the two mode blocks

    fig_w = PAPER_TEXTWIDTH_IN
    cell_w = (fig_w - ml - mr - gutter - (n_cols - 1) * col_gap) / n_cols
    cell_h = cell_w * ratio
    grid_h = (
        n_rows * cell_h + (n_rows - 1) * row_gap + (n_modes - 1) * mode_gap
    )
    fig_h = mb + grid_h + mt
    pal.apply_style()
    fig = plt.figure(figsize=(fig_w, fig_h))

    # fig_w == body width, so page_scale == 1 and point tiers render at nominal size.
    page_scale = PAPER_TEXTWIDTH_IN / fig_w
    label_fs = pal.FS_LABEL / page_scale
    sub_fs = pal.FS_TICK / page_scale

    def rect(x, y, w, h):
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    x0 = ml + gutter

    def cell_bottom(visual_row: int) -> float:
        """y (from figure bottom) of the cell in the given visual row (0 = top)."""
        mode_index = visual_row // rows_per_mode
        top_from_top = visual_row * (cell_h + row_gap) + mode_index * mode_gap
        return mb + grid_h - top_from_top - cell_h

    for mode_index in range(n_modes):
        block = cells[mode_index * n_seeds : (mode_index + 1) * n_seeds]
        for i, cell in enumerate(block):
            visual_row = mode_index * rows_per_mode + i // n_cols
            col = i % n_cols
            x = x0 + col * (cell_w + col_gap)
            y = cell_bottom(visual_row)
            ax = fig.add_axes(rect(x, y, cell_w, cell_h))
            ax.imshow(
                np.asarray(cell["trace"]),
                aspect="auto",
                interpolation="nearest",
                cmap=pal.DIVERGING_CMAP,
                vmin=-1,
                vmax=1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.05,
                rf"$S^\phi={float(cell['final_score']):.1f}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=sub_fs,
                color=pal.ON_SURFACE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": pal.OUTLINE_VARIANT,
                    "boxstyle": "square,pad=0.16",
                    "alpha": 0.88,
                    "linewidth": 0.5,
                },
            )
        # Update-mode label centred over the block's three rows.
        top_row = mode_index * rows_per_mode
        bot_row = top_row + rows_per_mode - 1
        y_mid = (cell_bottom(top_row) + cell_h + cell_bottom(bot_row)) / 2
        fig.text(
            (ml + gutter / 2) / fig_w,
            y_mid / fig_h,
            str(block[0]["mode"]),
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fs,
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate inverse-NCA figures from saved runs."
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Render the appendix variant grid instead of the combined figure.",
    )
    args = parser.parse_args()
    import inverse_nca as exp  # experiment constants (dirs, seeds, variants)

    if args.appendix:
        path = render_appendix_variant_grid(
            exp.VARIANTS_ROOT, exp.APPENDIX_MODES, exp.APPENDIX_SEEDS
        )
    else:
        path = render_combined_figure(
            exp.OUTPUT_DIR,
            exp.FIGURE_SEEDS,
            exp.TRAIN_STEPS,
            exp.VARIANTS_ROOT,
            exp.APPENDIX_MODES,
            exp.APPENDIX_SEEDS,
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
