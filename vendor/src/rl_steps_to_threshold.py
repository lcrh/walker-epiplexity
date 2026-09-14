"""Steps-to-threshold column of the RL comparison table (Table 1).

For every task, set a competence threshold halfway between the random-policy return (the floor)
and the best task-pursuing condition's final smoothed training task return (the ceiling):

    threshold = R_random + 0.5 * (R_best - R_random)

This floor-relative half-way bar is sign-agnostic, so it is defined for every task (negative- and
positive-return alike), unlike a fixed hand-picked value. The steps-to-threshold is then the first
environment step at which the smoothed training task-return reaches that threshold, averaged over
seeds (a seed that never reaches it counts as the full ``total_timesteps`` budget), reported for the
task-reward run (``standard``) and the task+epiplexity run (``mixed``) under the common threshold.
This threshold is computed from the training Monitor curves, not from the 100-episode evaluation
returns reported in the RL comparison table.

Reads the per-run Monitor logs under ``results/rl_classic/<env>/<mode>[_s<seed>]/`` -- training side
only, no model reruns. The training task-return is Monitor ``r`` for ``standard`` and the
``task_return`` column for the bonus modes (where ``r`` includes the bonus). Smoothing matches the
figure code (rolling mean, window ``max(20, n_episodes // 60)``).
"""

from __future__ import annotations

import glob
import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import load_results

from rl.envs import ENVS  # importing rl.envs also registers PointMaze etc.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "rl_classic")
TOTAL_TIMESTEPS = 600_000
FRACTION = 0.5  # competence bar between the random floor and the best training-curve ceiling
# Task-pursuing conditions that define the achievable ceiling and are reported in the column.
CEILING_MODES = ("standard", "mixed", "magnitude")
REPORT_MODES = ("standard", "mixed")


def random_return(spec, episodes=20) -> float:
    """Mean task return of a random policy -- the floor for the threshold."""
    env = gym.make(spec.env_id)
    env.action_space.seed(20000)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=20000 + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += float(r)
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets))


def _seed_dirs(env: str, mode: str) -> list[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(OUTPUT_DIR, env, f"{mode}*"))):
        base = os.path.basename(d)
        if (base == mode or base.startswith(mode + "_s")) and glob.glob(os.path.join(d, "*.monitor.csv")):
            out.append(d)
    return out


def _smoothed(run_dir: str, mode: str):
    """Smoothed training task-return curve (steps, values) for one run."""
    df = load_results(run_dir).sort_values("t")
    steps = np.cumsum(df["l"].to_numpy())
    col = "task_return" if mode != "standard" else "r"
    task = df[col].to_numpy().astype(float)
    win = max(20, len(task) // 60)
    if len(task) >= win:
        smooth = np.convolve(task, np.ones(win) / win, mode="valid")
        return steps[win - 1:], smooth
    return steps, task


def _curves(env: str, mode: str):
    return [_smoothed(d, mode) for d in _seed_dirs(env, mode)]


def steps_to_threshold(env: str, mode: str, threshold: float, curves) -> float:
    """Mean first-crossing step over seeds; the full budget where a seed never crosses."""
    vals = []
    for s_steps, smooth in curves:
        hit = np.where(smooth >= threshold)[0]
        vals.append(float(s_steps[hit[0]]) if len(hit) else float(TOTAL_TIMESTEPS))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    print(f"{'task':24s} {'R_rand':>9s} {'R_best':>9s} {'thresh':>9s} "
          f"{'task':>8s} {'+epi':>8s}")
    rows = [
        "env,random_return,best_final_smoothed_training_return,"
        "threshold,steps_standard,steps_mixed"
    ]
    for spec in ENVS:
        curves = {m: _curves(spec.name, m) for m in CEILING_MODES}
        finals = {m: np.mean([c[1][-1] for c in curves[m]]) for m in CEILING_MODES if curves[m]}
        if not finals:
            continue
        r_rand = random_return(spec)
        r_best = max(finals.values())
        thr = r_rand + FRACTION * (r_best - r_rand)
        steps = {m: steps_to_threshold(spec.name, m, thr, curves[m]) for m in REPORT_MODES}
        print(f"{spec.name:24s} {r_rand:9.1f} {r_best:9.1f} {thr:9.1f} "
              f"{steps['standard']/1e3:7.0f}k {steps['mixed']/1e3:7.0f}k")
        rows.append(f"{spec.name},{r_rand:.1f},{r_best:.1f},{thr:.1f},"
                    f"{steps['standard']:.0f},{steps['mixed']:.0f}")
    out = os.path.join(OUTPUT_DIR, "steps_to_threshold.csv")
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
