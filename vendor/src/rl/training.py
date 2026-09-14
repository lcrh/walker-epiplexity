"""PPO training/evaluation for one (env, mode, seed) run, and the comparison table.

All formal settings are module constants; ``train_run`` exposes only the knobs
that legitimately vary between invocations (task, reward mode, seed, and the
timestep/output overrides used for quick checks).
"""

from __future__ import annotations

import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from .calibrate import build_reservoir, calibrate_beta, calibrate_magnitude_beta, estimate_normalization
from .envs import ENVS, EnvSpec
from .reward import EpiplexityRewardWrapper, MagnitudeRewardWrapper

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "rl_classic")

MODES = ("standard", "epiplexity", "magnitude", "mixed")

# Anchors the frozen statistics, probe episodes, and evaluation seeds; per-run
# training seeds vary, these do not.
BASE_SEED = 0

# Reward method (shared across environments; fixed parts of the method).
HIDDEN = 32  # bounded-learner width: small enough that chaos is not learnable
RIDGE_LAMBDA = 0.3
EPS = 1e-8
MIXING_TARGET = 0.1  # bonus whole-episode contribution as a fraction of the task scale

# PPO (shared across environments and modes).
N_ENVS = 8
TOTAL_TIMESTEPS = 600_000
N_STEPS = 1024
BATCH_SIZE = 256
GAMMA = 0.999
GAE_LAMBDA = 0.98
LEARNING_RATE = 3e-4
DEVICE = "cpu"


def run_dir(output_dir, spec, mode, seed):
    sub = mode if seed == BASE_SEED else f"{mode}_s{seed}"
    return os.path.join(output_dir, spec.name, sub)


def make_env(spec, res, norm, beta, mode, rank, log_dir):
    def _init():
        env = gym.make(spec.env_id)
        keywords = ()
        if mode == "magnitude":
            env = MagnitudeRewardWrapper(env, norm, beta)
            keywords = ("task_return",)
        elif mode != "standard":
            env = EpiplexityRewardWrapper(env, res, spec, norm,
                                          beta=(None if mode == "epiplexity" else beta))
            keywords = ("task_return",)
        env = Monitor(env, filename=os.path.join(log_dir, str(rank)), info_keywords=keywords)
        env.action_space.seed(BASE_SEED + rank)
        return env
    return _init


def evaluate(model, spec, episodes=100):
    env = gym.make(spec.env_id)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=BASE_SEED + 7000 + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        rets.append(total)
    env.close()
    return {"task_return": float(np.mean(rets))}


def train_run(
    spec: EnvSpec,
    mode: str,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    output_dir: str = OUTPUT_DIR,
) -> dict:
    """Train one PPO agent on ``spec`` in reward ``mode`` and return the eval metrics."""
    if mode not in ("standard", "epiplexity", "mixed", "magnitude"):
        raise ValueError(mode)
    set_random_seed(seed)
    log_dir = run_dir(output_dir, spec, mode, seed)
    os.makedirs(log_dir, exist_ok=True)

    res = norm = beta = None
    if mode != "standard":
        res = build_reservoir(spec, HIDDEN, RIDGE_LAMBDA, EPS, DEVICE, BASE_SEED)
        norm = estimate_normalization(res, spec, BASE_SEED, EPS)
        if mode == "mixed":
            beta = calibrate_beta(res, spec, norm, BASE_SEED, MIXING_TARGET, EPS)
            print(f"   mixing beta = {beta:.4g}", flush=True)
        elif mode == "magnitude":
            beta = calibrate_magnitude_beta(norm, spec, BASE_SEED, MIXING_TARGET, EPS)
            print(f"   magnitude beta = {beta:.4g}", flush=True)

    venv = DummyVecEnv([make_env(spec, res, norm, beta, mode, i, log_dir)
                        for i in range(N_ENVS)])
    venv.seed(seed)
    model = PPO("MlpPolicy", venv, seed=seed, device=DEVICE, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                learning_rate=LEARNING_RATE, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    venv.close()

    metrics = evaluate(model, spec)
    metrics.update({"env": spec.env_id, "name": spec.name, "mode": mode, "tau": spec.tau_max,
                    "hidden": HIDDEN, "beta": beta, "seed": seed,
                    "timesteps": total_timesteps})
    json.dump(metrics, open(os.path.join(log_dir, "metrics.json"), "w"), indent=2)
    print(f"[{spec.name} {mode} s{seed}] {json.dumps(metrics)}", flush=True)
    return metrics


def aggregate_table(output_dir: str = OUTPUT_DIR) -> str:
    """Read every (env, mode, seed) metrics.json and write the comparison-table CSV.

    Columns: per env, the mean +/- std task return over seeds for each reward mode -- the
    task reward, epiplexity alone, the state-magnitude control, and the task + epiplexity bonus.
    """
    import glob
    rows = []
    for spec in ENVS:
        cells = {}
        for mode in MODES:
            vals = []
            for d in glob.glob(os.path.join(output_dir, spec.name, f"{mode}*")):
                mf = os.path.join(d, "metrics.json")
                if os.path.basename(d) == mode or os.path.basename(d).startswith(mode + "_s"):
                    if os.path.exists(mf):
                        vals.append(json.load(open(mf))["task_return"])
            cells[mode] = (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None
        rows.append((spec.name, cells))
    lines = ["env," + ",".join(f"{m}_mean,{m}_std" for m in MODES) + ",n"]
    for name, cells in rows:
        def fmt(c):
            return f"{c[0]:.2f},{c[1]:.2f}" if c else "nan,nan"
        n = max((c[2] for c in cells.values() if c), default=0)
        lines.append(f"{name}," + ",".join(fmt(cells[m]) for m in MODES) + f",{n}")
    out = os.path.join(output_dir, "rl_comparison_table.csv")
    os.makedirs(output_dir, exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return out
