"""Train an MLP encoder x -> z on MNIST to maximize Reservoir epiplexity.

Two networks are involved:

1. A fixed random MLP reservoir (``RCEpiplexityMLP``) on the flattened image. Its
   parameters are frozen; it only provides the per-sample feature vector read out
   by the normalized ridge readout. Because the code ``z`` is vector-valued, the
   reservoir fits one ridge readout per output coordinate, stacked columnwise,
   and the score is the spectral log-volume of the stacked readout (see
   ``rc_epiplexity.core``).
2. A trainable MLP encoder (``systems.mnist.MNISTEncoder``) mapping an MNIST
   image ``x`` (28x28) to a ``d``-dimensional code ``z``, optimized to maximize
   the reservoir-computed epiplexity of ``x -> z`` with no labels at any stage.

Checkpoints are saved at step 0, at the final step, and whenever the epiplexity
first crosses each of several values, so the figure's panels are spaced by
epiplexity rather than by step. Figures are assembled by ``plot/inverse_mnist.py``
(run it directly to re-plot from saved outputs).
"""

from __future__ import annotations

import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from rc_epiplexity import RCEpiplexityMLP
from systems.checkpoints import clear_previous_checkpoints
from systems.mnist import MNISTEncoder, mnist_train_loader

from plot import inverse_mnist as mnist_plot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "inverse_mnist")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
BATCH_SIZE = 128
TRAIN_STEPS = 500
# Checkpoints for the t-SNE progression are taken when the epiplexity first
# crosses each of these values. With the spectral score the epiplexity rises
# fast and then saturates, so spacing the panels by score value (not by step)
# is what shows the representation at increasing epiplexity; the late plateau
# is deliberately left to the single final panel. Step 0 and the final step
# are always saved, giving six panels.
SAVE_THRESHOLDS = (30.0, 45.0, 55.0, 57.0)

# Encoder (the trainable system).
ENCODER_HIDDEN = 64
ENCODER_DIM = 64
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

# Reservoir (the frozen scorer).
RESERVOIR_HIDDEN = 2048
RIDGE_LAMBDA = 3.0
LAMBDA_CODE = 30.0
EPS = 1e-8

# Visualization subset (used by plot/inverse_mnist.py).
VIS_SAMPLES_PER_CLASS = 200
TSNE_PERPLEXITY = 30.0


def config_dict() -> dict:
    """Run-settings record written to config.json and into every checkpoint.

    Key order matches the historical dataclass field order so previously saved
    runs compare byte-identical.
    """
    return {
        "output_dir": OUTPUT_DIR,
        "data_dir": DATA_DIR,
        "seed": SEED,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "save_thresholds": SAVE_THRESHOLDS,
        "encoder_hidden": ENCODER_HIDDEN,
        "encoder_dim": ENCODER_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "reservoir_hidden": RESERVOIR_HIDDEN,
        "ridge_lambda": RIDGE_LAMBDA,
        "lambda_code": LAMBDA_CODE,
        "eps": EPS,
        "vis_samples_per_class": VIS_SAMPLES_PER_CLASS,
        "tsne_perplexity": TSNE_PERPLEXITY,
    }


def configure_determinism(seed: int) -> None:
    """Fix every RNG used by the formal MNIST run and force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_models() -> tuple[MNISTEncoder, RCEpiplexityMLP]:
    configure_determinism(SEED)
    encoder = MNISTEncoder(
        encoder_dim=ENCODER_DIM,
        hidden=ENCODER_HIDDEN,
    ).to(DEVICE)
    reservoir = RCEpiplexityMLP(
        input_dim=28 * 28,
        hidden_dim=RESERVOIR_HIDDEN,
        target_dim=ENCODER_DIM,
        ridge_lambda=RIDGE_LAMBDA,
        lambda_code=LAMBDA_CODE,
        eps=EPS,
        device=DEVICE,
    )
    reservoir.init(SEED + 1)
    reservoir.eval()
    for param in reservoir.parameters():
        param.requires_grad_(False)
    return encoder, reservoir


def save_checkpoint(encoder: MNISTEncoder, step: int) -> str:
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"encoder_step_{step:05d}.pt")
    torch.save(
        {
            "step": step,
            "config": config_dict(),
            "encoder_state_dict": encoder.state_dict(),
        },
        path,
    )
    return path


def train() -> list[dict[str, float]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clear_previous_checkpoints(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict(), f, indent=2)

    encoder, reservoir = build_models()
    save_checkpoint(encoder, 0)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS
    )
    train_loader = mnist_train_loader(DATA_DIR, BATCH_SIZE)

    remaining_thresholds = sorted(SAVE_THRESHOLDS)

    history: list[dict[str, float]] = []
    start = time.perf_counter()

    # Record step-0 epiplexity so the curve covers the initial state.
    encoder.eval()
    with torch.no_grad():
        init_batch, _ = next(iter(train_loader))
        init_x = init_batch.to(DEVICE)
        init_z = encoder(init_x)
        init_out = reservoir.epiplexity(init_x.flatten(1), init_z)
    history.append({
        "step": 0.0,
        "epiplexity": float(init_out.epiplexity.cpu()),
        "z_l2": float(torch.mean(init_z * init_z).cpu()),
        "residual_std": float(init_out.residual.std().cpu()),
        "grad_norm": 0.0,
        "lr": float(LEARNING_RATE),
    })

    step = 0
    data_iter = iter(train_loader)

    while step < TRAIN_STEPS:
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, _ = next(data_iter)
        step += 1

        encoder.train()
        x = images.to(DEVICE, non_blocking=True)
        z = encoder(x)
        output = reservoir.epiplexity(x.flatten(1), z)
        # Code is unit-norm, so no magnitude penalty is needed.
        z_l2 = torch.mean(z * z)  # diagnostic only (~1/encoder_dim)
        loss = -output.epiplexity

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        epx = float(output.epiplexity.detach().cpu())
        row = {
            "step": float(step),
            "epiplexity": epx,
            "z_l2": float(z_l2.detach().cpu()),
            "residual_std": float(output.residual.detach().std().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)

        should_save = step == TRAIN_STEPS
        while remaining_thresholds and epx >= remaining_thresholds[0]:
            remaining_thresholds.pop(0)
            should_save = True

        if should_save:
            save_checkpoint(encoder, step)
            print(
                f"step={step:05d} epiplexity={row['epiplexity']:.6f} "
                f"grad_norm={row['grad_norm']:.4f}",
                flush=True,
            )

    print(f"Training finished in {time.perf_counter() - start:.1f}s on {DEVICE}")
    return history


def main() -> None:
    print(
        f"Inverse MNIST Encoder: steps={TRAIN_STEPS}, "
        f"batch={BATCH_SIZE}, d={ENCODER_DIM}, "
        f"reservoir_hidden={RESERVOIR_HIDDEN}, device={DEVICE}"
    )
    history = train()
    mnist_plot.save_history_csv(history, mnist_plot.history_path(OUTPUT_DIR))
    mnist_plot.plot_history(
        [row["step"] for row in history],
        [row["epiplexity"] for row in history],
        os.path.join(OUTPUT_DIR, "epiplexity_curve.pdf"),
    )
    out = mnist_plot.render_combined_figure(
        OUTPUT_DIR, DATA_DIR, VIS_SAMPLES_PER_CLASS, SEED, TSNE_PERPLEXITY
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
