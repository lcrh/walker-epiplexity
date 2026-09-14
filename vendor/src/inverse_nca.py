"""Train a local NCA to maximize fixed Reservoir epiplexity.

The target is a stacked-horizon window: from the warmed-up state ``x`` we roll out
to ``tau_max`` differentiable steps and stack states ``tau_min..tau_max`` as a
multi-output target (one readout column per step and channel), instead of scoring
a single fixed step ahead. Maximizing the singular-value log-volume of that
stacked readout rewards a rule that keeps producing fresh structure across the
whole window, with redundant / repeated horizons charged at ~zero.

The formal run trains one independent NCA per seed in ``FIGURE_SEEDS``; the
appendix compares the direct and residual update rule over ``APPENDIX_SEEDS``.
Figures are assembled by ``plot/inverse_nca.py`` (run it directly to re-plot
from saved checkpoints).
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from rc_epiplexity import RCEpiplexity1D
from systems.checkpoints import clear_previous_checkpoints, save_gzip_checkpoint
from systems.nca import (
    STATE_CHANNELS,
    LocalNCA1D,
    evolve,
    evolve_stacked_window,
    random_unit_state,
)

from plot import inverse_nca as nca_plot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "inverse_nca")
VARIANTS_ROOT = os.path.join(PROJECT_ROOT, "results", "inverse_nca_variants")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIDTH = 64
BATCH_SIZE = 2048
TRAIN_STEPS = 2000
WARMUP_STEPS = 32
# Gaussian perturbation added to the warmed-up state each step so the reservoir
# never sees a spatially constant (fixed-point) input, which collapses the score.
NOISE_STD = 0.1
# Stacked prediction window. Kept short here: the inverse problem backpropagates
# through the SVD log-volume, which is ill-conditioned when the target has many
# columns (= 2 * number of horizons). The scoring experiments tolerate a long
# window; the inverse problem needs a short one.
TAU_MIN = 1
TAU_MAX = 8
SAVE_COUNT = 10
LEARNING_RATE = 1e-4
GRAD_CLIP = 0.5
# Raw epiplexity grows into the tens, which makes its gradient too large for
# the learning rate above; scaling only the loss (not the logged/plotted
# score) keeps the update magnitude reasonable without touching the metric.
EPIPLEXITY_LOSS_SCALE = 100.0
# The reservoir is the SAME minimal local 1D learner the ECA scorer uses
# (depth 4, kernel 3 -> radius 3). Gradient ascent on the very measure that
# crowns rule 110 carries a structureless rule into the soliton (glider)
# regime -- the 1D scoring and 1D inverse problems share one reservoir.
RESERVOIR_DEPTH = 4
RESERVOIR_CHANNELS = 256
RESERVOIR_KERNEL_SIZE = 3
RIDGE_LAMBDA = 0.3
EPS = 1e-8
EVOLUTION_STEPS = (
    64  # recorded in config.json for provenance (rollout length used by diagnostics)
)

# The combined paper figure aggregates these independent seeds: one row of
# space-time snapshots and one training curve per seed. Each seed trains into
# results/inverse_nca/seed_<s>/.
FIGURE_SEEDS = (1, 2, 3)

# Appendix: the same architecture and window as the main run, compared as the
# direct vs the residual update rule, each over nine seeds -- eighteen runs into
# results/inverse_nca_variants/<mode>/seed_<s>/, rendered as a 6x3 grid.
APPENDIX_MODES = ("direct", "residual")
APPENDIX_SEEDS = tuple(range(1, 10))


def config_dict(
    seed: int,
    output_dir: str,
    update_mode: str,
    tau_min: int,
    tau_max: int,
    save_count: int,
) -> dict:
    """Run-settings record written to config.json and into every checkpoint.

    Key order matches the historical dataclass field order so previously saved
    runs compare byte-identical.
    """
    return {
        "output_dir": output_dir,
        "seed": seed,
        "device": DEVICE,
        "update_mode": update_mode,
        "width": WIDTH,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "noise_std": NOISE_STD,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "save_count": save_count,
        "learning_rate": LEARNING_RATE,
        "grad_clip": GRAD_CLIP,
        "epiplexity_loss_scale": EPIPLEXITY_LOSS_SCALE,
        "reservoir_depth": RESERVOIR_DEPTH,
        "reservoir_channels": RESERVOIR_CHANNELS,
        "reservoir_kernel_size": RESERVOIR_KERNEL_SIZE,
        "ridge_lambda": RIDGE_LAMBDA,
        "eps": EPS,
        "evolution_steps": EVOLUTION_STEPS,
    }


def build_models(seed: int, update_mode: str) -> tuple[LocalNCA1D, RCEpiplexity1D]:
    torch.manual_seed(seed)
    # A fixed seed alone does not make CUDA runs reproducible: cuDNN's default
    # convolution algorithms use non-deterministic reduction order, and on this
    # edge-of-chaos dynamics a step-10 rounding difference of ~1e-6 was measured
    # to grow to an epiplexity difference >10 by step 200. Force determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    nca = LocalNCA1D(update_mode=update_mode).to(DEVICE)
    reservoir = RCEpiplexity1D(
        depth=RESERVOIR_DEPTH,
        channels=RESERVOIR_CHANNELS,
        kernel_size=RESERVOIR_KERNEL_SIZE,
        input_channels=STATE_CHANNELS,
        ridge_lambda=RIDGE_LAMBDA,
        eps=EPS,
        device=DEVICE,
    )
    reservoir.init(seed + 1)
    reservoir.eval()
    for param in reservoir.parameters():
        param.requires_grad_(False)
    return nca, reservoir


def save_checkpoint(nca: LocalNCA1D, config: dict, step: int) -> str:
    ckpt_dir = os.path.join(config["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"nca_step_{step:04d}.pt")
    save_gzip_checkpoint(
        {"step": step, "config": config, "nca_state_dict": nca.state_dict()},
        path,
    )
    return path


def train(
    seed: int,
    output_dir: str,
    update_mode: str = "residual",
    tau_min: int = TAU_MIN,
    tau_max: int = TAU_MAX,
    save_count: int = SAVE_COUNT,
) -> list[dict[str, float]]:
    """Train one NCA run and save its checkpoints + history CSV into ``output_dir``."""
    print(
        f"Inverse NCA: steps={TRAIN_STEPS}, warmup={WARMUP_STEPS}, "
        f"tau={tau_min}..{tau_max}, update={update_mode}, "
        f"batch={BATCH_SIZE}, width={WIDTH}, device={DEVICE}, "
        f"seed={seed}",
        flush=True,
    )
    config = config_dict(seed, output_dir, update_mode, tau_min, tau_max, save_count)
    os.makedirs(output_dir, exist_ok=True)
    clear_previous_checkpoints(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    nca, reservoir = build_models(seed, update_mode)
    save_checkpoint(nca, config, 0)
    optimizer = torch.optim.AdamW(nca.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=0
    )
    save_steps = set(
        int(s) for s in np.linspace(0, TRAIN_STEPS, save_count + 1, dtype=int)[1:]
    )
    history = []
    start = time.perf_counter()

    for step in range(1, TRAIN_STEPS + 1):
        nca.train()
        state = random_unit_state(BATCH_SIZE, WIDTH, DEVICE)
        x = evolve(nca, state, WARMUP_STEPS, track_grad=False)
        x = x + torch.randn_like(x) * NOISE_STD
        y = evolve_stacked_window(nca, x, tau_min, tau_max)
        output = reservoir.epiplexity(x, y)
        y_l2 = torch.mean(y * y)  # diagnostic only (~0.5 for unit 2-vectors)
        loss = -output.epiplexity / EPIPLEXITY_LOSS_SCALE

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        row = {
            "step": float(step),
            "epiplexity": float(output.epiplexity.detach().cpu()),
            "y_l2": float(y_l2.detach().cpu()),
            "residual_std": float(output.residual.detach().std().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        history.append(row)

        if step in save_steps:
            save_checkpoint(nca, config, step)
            print(
                f"step={step:04d} epiplexity={row['epiplexity']:.6f} "
                f"y_l2={row['y_l2']:.6f} grad_norm={row['grad_norm']:.6f}",
                flush=True,
            )

    print(
        f"Training finished in {time.perf_counter() - start:.1f}s on {DEVICE}",
        flush=True,
    )
    nca_plot.save_history_csv(history, nca_plot.history_path(output_dir))
    return history


def train_variant(mode: str, seed: int) -> None:
    """Train a single appendix (mode, seed) run; lets the 18 runs shard across processes."""
    train(
        seed=seed,
        output_dir=os.path.join(VARIANTS_ROOT, mode, f"seed_{seed}"),
        update_mode=mode,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        save_count=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appendix-variants",
        action="store_true",
        help="Train the formal appendix NCA variants and render their rollout grid.",
    )
    parser.add_argument(
        "--appendix-train-one",
        nargs=2,
        metavar=("MODE", "SEED"),
        help="Train a single appendix (mode, seed) run, to shard the 18 runs across processes.",
    )
    args = parser.parse_args()

    if args.appendix_train_one is not None:
        mode, seed = args.appendix_train_one
        train_variant(mode, int(seed))
        return
    if args.appendix_variants:
        for mode in APPENDIX_MODES:
            for seed in APPENDIX_SEEDS:
                train_variant(mode, seed)
        path = nca_plot.render_appendix_variant_grid(
            VARIANTS_ROOT, APPENDIX_MODES, APPENDIX_SEEDS
        )
        print(f"Appendix variant figure -> {path}", flush=True)
        return
    for seed in FIGURE_SEEDS:
        train(seed=seed, output_dir=os.path.join(OUTPUT_DIR, f"seed_{seed}"))
    path = nca_plot.render_combined_figure(
        OUTPUT_DIR,
        FIGURE_SEEDS,
        TRAIN_STEPS,
        VARIANTS_ROOT,
        APPENDIX_MODES,
        APPENDIX_SEEDS,
    )
    print(f"Rendering combined figure -> {path}")


if __name__ == "__main__":
    main()
