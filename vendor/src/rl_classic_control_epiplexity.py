"""Epiplexity as a reward signal for reinforcement learning.

The agent is handed the closed-form epiplexity of its own state trajectory as a reward. A
rollout produces states ``x_0, x_1, ...``; from state ``x_t`` we form the *stacked-horizon*
target ``(x_{t+1}, ..., x_{t+tau_max})`` -- the whole next window jointly -- and score the map
``x_t -> window`` with a frozen random MLP reservoir (the same closed-form estimator used for
the dynamical-systems and MNIST experiments). Stacking the window instead of a single lag
removes the single-``delta`` hyperparameter: the readout's singular-value log-volume prices
redundant or unpredictable horizons at zero, so only horizons whose future is genuinely
determined by the present state contribute, and the score is insensitive to the step size.

Three settings make the reward discriminate *coherent* novelty rather than noise, and all are
fixed parts of the method, not tuned per task:

  * **Input normalization.** Each state coordinate is standardized before the reservoir, so a
    large-magnitude state cannot saturate the fixed nonlinearity ``phi`` and flatten its
    features. This mirrors the feature standardization in the estimator's definition and is
    what makes the reward work on high-dimensional locomotion (large joint velocities).
  * **A small reservoir.** The reservoir width is the "bounded" in bounded learner: a wide
    reservoir can fit even a random/chaotic trajectory, so novelty would reward thrashing; a
    small one (hidden 32) can only fit structured trajectories, so the reward favors coherent,
    committed motion over chaos.
  * **A window matched to the task.** ``tau_max`` is set to twice the lag at which the state's
    RMS displacement under a random policy first reaches one state-standard-deviation -- the
    characteristic time over which the state changes appreciably -- so the window spans real
    motion rather than near-identical consecutive states.

The per-step reward is the increment the latest state adds to the trajectory's epiplexity,
maintained online by a covariance-form recursive-least-squares estimator (one rank-1 update
per step). The increments telescope (``S_0 = 0``), so the undiscounted return equals the
whole-episode epiplexity ``S_T``.

Four reward modes are compared. ``standard`` is the environment's task reward alone (the
reference). ``epiplexity`` drops the task reward and hands the agent only the novelty reward
(the agent never sees the task). ``mixed`` is the task reward plus a small novelty bonus,
calibrated per environment so the bonus defers to the task where it is informative and drives
exploration where it is flat. ``magnitude`` is a control: the task reward plus a
bonus for the squared norm of the *input-normalized* state, calibrated the same way. It rewards
reaching large-magnitude (atypical) state with none of epiplexity's learnable structure, so
comparing it against ``mixed`` isolates whether the gain comes from the novelty's learnability
rather than from raw coverage. PPO maximizes the chosen reward; we report the true task return,
which the reward-free agents never observe.

Components live in ``rl/``: the task battery (``rl.envs``), the reward wrappers
(``rl.reward``), the frozen normalization and mixing weights (``rl.calibrate``), and the
PPO run/table code (``rl.training``).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from rl.envs import selected_envs
from rl.training import MODES, TOTAL_TIMESTEPS, aggregate_table, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=None, help="env name or id (default: all)")
    parser.add_argument("--mode", choices=("standard", "epiplexity", "magnitude", "mixed", "all"),
                        default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--table", action="store_true", help="only (re)build the comparison table")
    args = parser.parse_args()

    if args.table:
        aggregate_table()
        return

    modes = MODES if args.mode == "all" else (args.mode,)
    for spec in selected_envs(args.env):
        for mode in modes:
            for seed in args.seeds:
                train_run(spec, mode, seed, total_timesteps=args.timesteps)
    aggregate_table()


if __name__ == "__main__":
    main()
