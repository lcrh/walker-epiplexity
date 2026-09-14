"""Frozen reservoir + normalization statistics, and the per-task mixing weights.

Every function takes the *base* seed (the constant that anchors the frozen
statistics and probe episodes), not the per-run training seed: the reservoir,
the normalization, and the calibrated ``beta`` are shared across seeds of a run.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from rc_epiplexity import RCEpiplexityMLP

from .reward import EpiplexityRewardWrapper


def build_reservoir(spec, hidden, ridge_lambda, eps, device, base_seed) -> RCEpiplexityMLP:
    """One frozen random MLP reservoir, target = the stacked next-``tau_max`` states."""
    res = RCEpiplexityMLP(
        input_dim=spec.state_dim,
        hidden_dim=hidden,
        target_dim=spec.state_dim * spec.tau_max,
        ridge_lambda=ridge_lambda,
        eps=eps,
        device=device,
    )
    res.init(base_seed + 1)
    res.eval()
    for p in res.parameters():
        p.requires_grad_(False)
    return res


def estimate_normalization(res, spec, base_seed, eps, target_pairs=8192, max_episodes=8000):
    """Frozen input / feature / target statistics from random-policy rollouts.

    The online RLS recursion needs the normalization fixed in advance. Inputs and features are
    standardized per coordinate; the target is centered and divided by a fixed unit (its scale
    is information, so it is preserved). Input standardization happens *before* the reservoir,
    so a large state does not saturate ``phi``.
    """
    env = gym.make(spec.env_id)
    env.action_space.seed(base_seed + 1000)
    xs, ys, n_pairs, ep = [], [], 0, 0
    tau = spec.tau_max
    while n_pairs < target_pairs and ep < max_episodes:
        obs, _ = env.reset(seed=base_seed + 1000 + ep)
        ep += 1
        traj = [np.asarray(obs, np.float32)]
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            traj.append(np.asarray(obs, np.float32))
            done = term or trunc
        arr = np.stack(traj)
        if arr.shape[0] - tau >= 1:
            for t in range(arr.shape[0] - tau):
                xs.append(arr[t])
                ys.append(arr[t + 1 : t + 1 + tau].reshape(-1))
            n_pairs += arr.shape[0] - tau
    env.close()
    if not xs:
        raise RuntimeError(f"no lagged windows collected for {spec.env_id}")

    x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=res.device)
    y = torch.as_tensor(np.stack(ys), dtype=torch.float32, device=res.device)
    x_mu = x.mean(0)
    x_inv = 1.0 / (x.std(0, correction=0) + eps)  # input norm before phi
    with torch.no_grad():
        h = res.phi((x - x_mu) * x_inv).float()
    c = h.shape[1]
    return {
        "x_mu": x_mu,
        "x_inv": x_inv,
        "f_mu": h.mean(0),
        "f_inv": 1.0 / (h.std(0, correction=0) * (c**0.5) + eps),
        "y_mu": y.mean(0),
        "y_inv": torch.ones(y.shape[1], device=res.device),  # unit target scale = information
    }


def calibrate_beta(res, spec, norm, base_seed, mixing_target, eps, episodes=64):
    """Mixing weight so the bonus's whole-episode contribution is ``mixing_target`` times the
    random-policy task-return scale (anchored at episode level so it survives sparse rewards)."""
    base = gym.make(spec.env_id)
    probe = EpiplexityRewardWrapper(base, res, spec, norm, beta=None)
    probe.action_space.seed(base_seed + 2000)
    tasks, epis = [], []
    for ep in range(episodes):
        probe.reset(seed=base_seed + 2000 + ep)
        done, s_t, info = False, 0.0, {}
        while not done:
            _, r, term, trunc, info = probe.step(probe.action_space.sample())
            s_t += float(r)
            done = term or trunc
        tasks.append(abs(float(info["task_return"])))
        epis.append(s_t)
    probe.close()
    return mixing_target * (float(np.mean(tasks)) + eps) / (float(np.mean(epis)) + eps)


def calibrate_magnitude_beta(norm, spec, base_seed, mixing_target, eps, episodes=64):
    """Weight for the magnitude control, anchored exactly as the epiplexity bonus: its
    whole-episode contribution is ``mixing_target`` times the random-policy task-return scale."""
    env = gym.make(spec.env_id)
    env.action_space.seed(base_seed + 2000)
    x_mu = np.asarray(norm["x_mu"].cpu(), dtype=np.float32)
    x_inv = np.asarray(norm["x_inv"].cpu(), dtype=np.float32)
    tasks, mags = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + 2000 + ep)
        done, tr, m = False, 0.0, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            tr += float(r)
            z = (np.asarray(obs, np.float32) - x_mu) * x_inv
            m += float(np.dot(z, z))
            done = term or trunc
        tasks.append(abs(tr))
        mags.append(m)
    env.close()
    return mixing_target * (float(np.mean(tasks)) + eps) / (float(np.mean(mags)) + eps)
