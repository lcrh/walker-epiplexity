"""Epiplexity as an RL reward signal: components for the classic-control battery.

- ``envs``: the task battery (`EnvSpec`, `ENVS`) and env registration.
- ``reward``: the epiplexity reward wrapper and the magnitude-control wrapper.
- ``calibrate``: frozen reservoir + normalization statistics and the per-task
  mixing weights ``beta``.
- ``training``: PPO training/evaluation for one (env, mode, seed) run and the
  comparison-table aggregation.

The method statement (why input normalization, a small reservoir, and a
task-matched window make the reward discriminate coherent novelty) lives in the
entry script ``rl_classic_control_epiplexity.py``.
"""
