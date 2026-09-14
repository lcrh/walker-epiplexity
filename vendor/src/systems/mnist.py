"""MNIST encoder and data access for the inverse representation experiment.

Trainable system only: built from plain ``torch.nn`` and never imports from
``rc_epiplexity`` (the frozen scorer must stay a separate observer).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms


class MNISTEncoder(nn.Module):
    """MLP encoder x: (B, 1, 28, 28) -> unit-norm code z: (B, encoder_dim).

    The code is L2-normalized to the unit sphere, so only its direction carries
    information and its scale is fixed. This controls the code scale by hard
    constraint instead of an L2 penalty, mirroring the unit-vector NCA state.
    """

    def __init__(self, encoder_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.input_dim = 28 * 28
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, hidden * 4),
            nn.BatchNorm1d(hidden * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 4, encoder_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-8)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def mnist_train_loader(data_dir: str, batch_size: int) -> torch.utils.data.DataLoader:
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )


def mnist_test_set(data_dir: str) -> datasets.MNIST:
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)


def sample_vis_subset(
    test_set: datasets.MNIST, samples_per_class: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Class-balanced test subset used for the t-SNE panels and probes."""
    targets = np.asarray(test_set.targets)
    rng = np.random.default_rng(seed)
    indices = []
    for digit in range(10):
        digit_indices = np.where(targets == digit)[0]
        chosen = rng.choice(digit_indices, size=samples_per_class, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    images = torch.stack([test_set[i][0] for i in indices], dim=0)
    labels = targets[indices]
    return images, labels


def encode_dataset(
    encoder: MNISTEncoder, images: torch.Tensor, device: str, batch_size: int = 512
) -> np.ndarray:
    encoder.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(device)
            outputs.append(encoder(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def probe_accuracies(z: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    """Supervised read-outs of the unsupervised code: 5-NN and linear-probe test accuracy.

    Labels are used only here, to score how separable the code is — never during the
    encoder's training. A stratified 50/50 split fits on one half and scores the other.
    """
    z_tr, z_te, y_tr, y_te = train_test_split(
        z, labels, test_size=0.5, random_state=seed, stratify=labels
    )
    knn = KNeighborsClassifier(n_neighbors=5).fit(z_tr, y_tr)
    linear = LogisticRegression(max_iter=2000).fit(z_tr, y_tr)
    return float(knn.score(z_te, y_te)), float(linear.score(z_te, y_te))
