"""The RL task battery and env registration."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation

gym.register_envs(gymnasium_robotics)  # PointMaze


def _register_flat(flat_id, base_id, **base_kw):
    """Register ``flat_id`` = ``base_id`` (fixed kwargs) with its dict observation flattened to a
    vector, so the same MLP reservoir applies. PointMaze uses ``continuing_task`` so reaching the
    goal resamples it instead of ending the episode (no goal-avoidance)."""
    from gymnasium.envs.registration import register
    register(id=flat_id,
             entry_point=lambda **kw: FlattenObservation(gym.make(base_id, **base_kw, **kw)),
             max_episode_steps=None)


_register_flat("PointMazeLargeFlat-v0", "PointMaze_Large-v3", continuing_task=True)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    state_dim: int
    tau_max: int  # temporal window = 2 * characteristic state-change time (clipped to [8, 48])
    name: str


# Two task families. Sparse / deceptive classic control, where exploration is the bottleneck
# and the bonus turns unreliable exploration into reliable solving; and MuJoCo / Box2D
# locomotion, where the bonus helps PPO escape mediocre local optima (and input normalization
# is what unlocks it). The last three are reported as the honest neutral / negative cases.
ENVS = [
    EnvSpec("Acrobot-v1", 6, 8, "acrobot"),
    EnvSpec("MountainCarContinuous-v0", 2, 28, "mountaincar_continuous"),
    EnvSpec("Hopper-v5", 11, 10, "hopper"),
    EnvSpec("BipedalWalker-v3", 24, 40, "bipedalwalker"),
    EnvSpec("HalfCheetah-v5", 17, 16, "halfcheetah"),
    EnvSpec("LunarLander-v3", 8, 48, "lunarlander"),
    EnvSpec("Walker2d-v5", 17, 16, "walker2d"),
    EnvSpec("Swimmer-v5", 8, 16, "swimmer"),
    EnvSpec("Pendulum-v1", 3, 16, "pendulum"),
    EnvSpec("PointMazeLargeFlat-v0", 8, 48, "pointmaze"),
]
ENV_BY_NAME = {e.name: e for e in ENVS}
ENV_BY_ID = {e.env_id: e for e in ENVS}


def selected_envs(env_filter):
    if env_filter is None:
        return ENVS
    if env_filter in ENV_BY_NAME:
        return [ENV_BY_NAME[env_filter]]
    if env_filter in ENV_BY_ID:
        return [ENV_BY_ID[env_filter]]
    raise ValueError(f"unknown env '{env_filter}'; choices: {[e.name for e in ENVS]}")
