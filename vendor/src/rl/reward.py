"""Reward wrappers: the dense epiplexity increment and the magnitude control."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from rc_epiplexity import CovarianceRLSEpiplexity


class EpiplexityRewardWrapper(gym.Wrapper):
    """Replace the task reward with the dense stacked-horizon epiplexity increment.

    At step ``j`` the window anchored at ``t = j - tau_max`` just completed; folding the pair
    ``(phi(norm(x_t)), [x_{t+1}..x_{t+tau_max}])`` into the RLS recursion is one rank-1 update.
    The increment ``S_t - S_{t-1}`` telescopes to the whole-episode epiplexity ``S_T``. If
    ``beta`` is given the agent gets ``task + beta * increment`` (mixed); otherwise the task
    reward is dropped (pure epiplexity). ``info["task_return"]`` always reports the task return.
    """

    def __init__(self, env, res, spec, norm, beta=None):
        super().__init__(env)
        self.res = res
        self.tau = spec.tau_max
        self.norm = norm
        self.beta = beta

    def _new_estimator(self):
        return CovarianceRLSEpiplexity(
            feature_dim=self.res.hidden_dim,
            target_dim=self.res.target_dim,
            ridge_lambda=self.res.ridge_lambda,
            device=self.res.device,
            dtype=torch.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._states = [np.asarray(obs, np.float32)]
        self._task_return = 0.0
        self._prev = 0.0
        self._est = self._new_estimator()
        return obs, info

    def _increment(self):
        j = len(self._states) - 1
        t = j - self.tau
        if t < 0:
            return 0.0
        x_state = torch.as_tensor(self._states[t], dtype=torch.float32, device=self.res.device)
        window = np.concatenate(self._states[t + 1 : j + 1])
        x_state = (x_state - self.norm["x_mu"]) * self.norm["x_inv"]  # input norm before phi
        with torch.no_grad():
            phi = self.res.phi(x_state.unsqueeze(0)).squeeze(0)
        phi = (phi - self.norm["f_mu"]) * self.norm["f_inv"]
        y = torch.as_tensor(window, dtype=torch.float32, device=self.res.device)
        y = (y - self.norm["y_mu"]) * self.norm["y_inv"]
        self._est.update(phi, y)
        s = float(self._est.epiplexity())
        inc = s - self._prev
        self._prev = s
        return inc

    def step(self, action):
        obs, task_reward, term, trunc, info = self.env.step(action)
        self._states.append(np.asarray(obs, np.float32))
        self._task_return += float(task_reward)
        inc = self._increment()
        reward = inc if self.beta is None else float(task_reward) + self.beta * inc
        if term or trunc:
            info["task_return"] = self._task_return
        return obs, reward, term, trunc, info


class MagnitudeRewardWrapper(gym.Wrapper):
    """Control reward: the task reward plus a bonus for the *size* of the normalized state.

    The bonus at step ``t`` is ``beta * ||(x_t - mu_X) * x_inv||^2`` -- the squared norm of the
    state in the same standardized coordinates the epiplexity reward feeds to the reservoir
    (``mu_X``, ``x_inv`` frozen from random-policy rollouts). It rewards reaching state far from
    the random-policy mean -- raw coverage by magnitude -- with none of epiplexity's learnable
    structure, so comparing it against the epiplexity bonus isolates whether the gain comes from
    the novelty's *learnability* rather than from merely visiting large state.
    """

    def __init__(self, env, norm, beta):
        super().__init__(env)
        self.x_mu = np.asarray(norm["x_mu"].cpu(), dtype=np.float32)
        self.x_inv = np.asarray(norm["x_inv"].cpu(), dtype=np.float32)
        self.beta = beta

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._task_return = 0.0
        return obs, info

    def step(self, action):
        obs, task_reward, term, trunc, info = self.env.step(action)
        self._task_return += float(task_reward)
        z = (np.asarray(obs, np.float32) - self.x_mu) * self.x_inv
        reward = float(task_reward) + self.beta * float(np.dot(z, z))
        if term or trunc:
            info["task_return"] = self._task_return
        return obs, reward, term, trunc, info
