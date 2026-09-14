"""Single-process launcher for one (env, mode, seed) RL run, pinned to one thread.

Forcing one BLAS/torch thread per process is what makes the seed sweep parallelizable: with the
default thread pool many concurrent PPO processes contend and each runs ~50x slower.

Usage: uv run python src/rl/run_one.py <env_name> <mode> <seed>
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runnable as a script

import torch

torch.set_num_threads(1)

from rl.envs import ENV_BY_NAME
from rl.training import train_run

if __name__ == "__main__":
    env_name, mode, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    train_run(ENV_BY_NAME[env_name], mode, seed)
