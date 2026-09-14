"""Checkpoint IO shared by the training experiments and their plot modules."""

from __future__ import annotations

import gzip
import io
import os

import torch


def clear_previous_checkpoints(output_dir: str) -> None:
    """Empty ``<output_dir>/checkpoints`` so a fresh run cannot mix with a stale one."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name in os.listdir(ckpt_dir):
        if name.endswith(".pt"):
            os.remove(os.path.join(ckpt_dir, name))


def checkpoint_paths(output_dir: str) -> list[str]:
    """All ``<output_dir>/checkpoints/*.pt`` paths, sorted by the trailing step number."""
    root = os.path.join(output_dir, "checkpoints")
    paths = [os.path.join(root, name) for name in os.listdir(root) if name.endswith(".pt")]
    return sorted(paths, key=lambda p: int(os.path.splitext(p)[0].rsplit("_", 1)[1]))


def save_gzip_checkpoint(obj: dict, path: str) -> None:
    """Gzip-compressed torch.save: serialize to memory first so the on-disk
    stream is a plain gzip file, not a zip archive requiring backward seeks."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(buffer.getvalue())


def load_gzip_checkpoint(path: str, map_location) -> dict:
    with gzip.open(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)
