"""Measure inverse-MNIST robustness by varying one hyperparameter at a time.

Run:
    uv run python src/robustness/mnist_hparam_scan.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

from rc_epiplexity import RCEpiplexity, RCEpiplexityMLP
from rc_epiplexity.reservoirs import PreActNorm
from robustness.plot_mnist_hparam import plot_heatmap
from systems.mnist import MNISTEncoder, encode_dataset, mnist_test_set, probe_accuracies, sample_vis_subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "robustness", "mnist_hparam")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEVICE = "cuda"
STEPS = 150
SEED = 0

BASELINE = {
    "learning_rate": 1e-2,
    "batch_size": 128,
    "encoder_dim": 64,
    "reservoir_hidden": 2048,
    "reservoir_depth": 4,
    "ridge_lambda": 3.0,
    "lambda_code": 30.0,
    "encoder_hidden": 64,
}

AXES = {
    "learning_rate": (1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    "batch_size": (32, 64, 128, 256, 512),
    "encoder_dim": (16, 32, 64, 128, 256),
    "reservoir_depth": (2, 3, 4, 5, 6),
    "ridge_lambda": (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
    "lambda_code": (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
}

FIELDS = ["axis", "value", "seed", "steps", "final_linear", "runtime_s"]


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class VariableDepthMLP(RCEpiplexity):
    """Robustness-only MLP reservoir; depth counts linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, cfg: dict):
        super().__init__(hidden_dim, ridge_lambda=cfg["ridge_lambda"],
                         lambda_code=cfg["lambda_code"], device=DEVICE)
        layers: list[nn.Module] = []
        for layer_index in range(depth):
            layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim,
                                    hidden_dim, device=self.device))
            if layer_index < depth - 1:
                layers.extend((PreActNorm(), nn.ELU()))
        self.phi = nn.Sequential(*layers)

    def epiplexity(self, x: torch.Tensor, y: torch.Tensor):
        self.eval()
        return self.multioutput_epiplexity(self.phi(x), y)


def build_models(cfg: dict) -> tuple[MNISTEncoder, RCEpiplexity]:
    configure_determinism(SEED)
    encoder = MNISTEncoder(cfg["encoder_dim"], cfg["encoder_hidden"]).to(DEVICE)
    if cfg["reservoir_depth"] == 4:
        reservoir = RCEpiplexityMLP(
            input_dim=28 * 28,
            hidden_dim=cfg["reservoir_hidden"],
            target_dim=cfg["encoder_dim"],
            ridge_lambda=cfg["ridge_lambda"],
            lambda_code=cfg["lambda_code"],
            device=DEVICE,
        )
    else:
        reservoir = VariableDepthMLP(28 * 28, cfg["reservoir_hidden"],
                                     cfg["reservoir_depth"], cfg)
    reservoir.init(SEED + 1)
    reservoir.eval()
    for parameter in reservoir.parameters():
        parameter.requires_grad_(False)
    return encoder, reservoir


def fixed_batches(n: int, batch_size: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + 20_000)
    batches, cursor = [], 0
    order = torch.randperm(n, generator=generator)
    for _ in range(STEPS):
        if cursor + batch_size > n:
            order = torch.randperm(n, generator=generator)
            cursor = 0
        batches.append(order[cursor:cursor + batch_size])
        cursor += batch_size
    return batches


def run_one(cfg: dict, axis: str, value: float, train_images: torch.Tensor,
            probe_images: torch.Tensor, probe_labels: np.ndarray) -> dict:
    encoder, reservoir = build_models(cfg)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg["learning_rate"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for indices in fixed_batches(len(train_images), cfg["batch_size"]):
        x = train_images[indices].unsqueeze(1).to(DEVICE).float().div_(255.0)
        output = reservoir.epiplexity(x.flatten(1), encoder(x))
        optimizer.zero_grad(set_to_none=True)
        (-output.epiplexity).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    z = encode_dataset(encoder, probe_images, DEVICE)
    _, final_linear = probe_accuracies(z, probe_labels, SEED)
    return {"axis": axis, "value": value, "seed": SEED, "steps": STEPS,
            "final_linear": final_linear, "runtime_s": runtime}


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed(rows: list[dict], axis: str, value: float) -> bool:
    return any(row["axis"] == axis and float(row["value"]) == value for row in rows)


def main() -> None:
    assert torch.cuda.is_available(), "The MNIST robustness experiment requires CUDA"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "mnist_hparam_metrics.csv")
    config_path = os.path.join(OUTPUT_DIR, "mnist_hparam_config.json")
    run_config = {"baseline": BASELINE, "axes": AXES, "steps": STEPS, "seed": SEED}
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=False)
    probe_images, probe_labels = sample_vis_subset(
        mnist_test_set(DATA_DIR), samples_per_class=200, seed=SEED
    )
    for axis, values in (("baseline", (0.0,)), *AXES.items()):
        for value in values:
            if axis != "baseline" and value == BASELINE[axis]:
                continue
            rows = load_rows(metrics_path)
            if completed(rows, axis, float(value)):
                continue
            cfg = dict(BASELINE)
            if axis != "baseline":
                cfg[axis] = value
            row = run_one(cfg, axis, float(value), train_set.data, probe_images, probe_labels)
            with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
            print(f"[{axis:>18}={value:g}] final_linear={row['final_linear']:.3f}", flush=True)
            plot_heatmap(OUTPUT_DIR, BASELINE)
    plot_heatmap(OUTPUT_DIR, BASELINE)


if __name__ == "__main__":
    main()
