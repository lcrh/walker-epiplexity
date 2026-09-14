"""Continuous-time dynamical systems: vector fields, RK4 integration, and the
autocorrelation-time estimate used to scale their phase portraits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FieldFn = Callable[[np.ndarray], np.ndarray]

# --- Autocorrelation-time estimation (used only for phase-portrait scaling) ---
TAU_NUM_SAMPLES = 512
TAU_LAG_MAX = 50.0
TAU_SAMPLE_EVERY = 5
TAU_RK4_STEP = 0.01


@dataclass(frozen=True)
class SystemSpec:
    name: str
    state_dim: int
    init_scale: float
    field: FieldFn
    projection: tuple[int, int]
    pre_burnin_time: float = 0.0


def lorenz_field(state: np.ndarray) -> np.ndarray:
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    return np.stack(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1
    )


def rossler_field(state: np.ndarray) -> np.ndarray:
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    a, b, c = 0.2, 0.2, 5.7
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=-1)


def thomas_field(state: np.ndarray) -> np.ndarray:
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    b = 0.18
    return np.stack(
        [np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], axis=-1
    )


def pure_rotation_field(state: np.ndarray) -> np.ndarray:
    x, y = state[..., 0], state[..., 1]
    omega = 1.0
    return np.stack([omega * y, -omega * x], axis=-1)


def damped_spiral_field(state: np.ndarray) -> np.ndarray:
    x, y = state[..., 0], state[..., 1]
    omega, gamma = 1.0, 0.1
    return np.stack([omega * y - gamma * x, -omega * x - gamma * y], axis=-1)


def stable_node_field(state: np.ndarray) -> np.ndarray:
    x, y = state[..., 0], state[..., 1]
    return np.stack([-0.5 * x, -1.5 * y], axis=-1)


SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec("Lorenz", 3, 10.0, lorenz_field, (0, 2)),
    SystemSpec("Rossler", 3, 1.0, rossler_field, (0, 1), pre_burnin_time=200.0),
    SystemSpec("Thomas", 3, 1.0, thomas_field, (0, 1), pre_burnin_time=200.0),
    SystemSpec("Linear: pure rotation", 2, 1.0, pure_rotation_field, (0, 1)),
    SystemSpec("Linear: damped spiral", 2, 1.0, damped_spiral_field, (0, 1)),
    SystemSpec("Linear: stable node", 2, 1.0, stable_node_field, (0, 1)),
)


def rk4_step(field: FieldFn, state: np.ndarray, h: float) -> np.ndarray:
    k1 = field(state)
    k2 = field(state + 0.5 * h * k1)
    k3 = field(state + 0.5 * h * k2)
    k4 = field(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(field: FieldFn, state: np.ndarray, total_time: float, h: float) -> np.ndarray:
    n_steps = int(round(total_time / h))
    for _ in range(n_steps):
        state = rk4_step(field, state, h)
    return state


def trajectory(
    field: FieldFn,
    state: np.ndarray,
    total_time: float,
    h: float,
    sample_every: int,
) -> np.ndarray:
    """Return shape ``(n_samples + 1, ..., state_dim)`` including the initial state."""
    n_steps = int(round(total_time / h))
    out = [state]
    for step in range(n_steps):
        state = rk4_step(field, state, h)
        if (step + 1) % sample_every == 0:
            out.append(state)
    return np.stack(out, axis=0)


def autocorrelation_time(spec: SystemSpec, seed: int) -> float:
    """Lag at which the state autocorrelation first decays below ``1/e``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, spec.init_scale, size=(TAU_NUM_SAMPLES, spec.state_dim))
    if spec.pre_burnin_time > 0.0:
        z = integrate(spec.field, z, spec.pre_burnin_time, TAU_RK4_STEP)

    traj = trajectory(spec.field, z, TAU_LAG_MAX, TAU_RK4_STEP, TAU_SAMPLE_EVERY)
    x0 = traj[0]
    x0_centered = x0 - x0.mean(axis=0, keepdims=True)
    var_x0 = (x0_centered**2).mean(axis=0)
    n_lags = traj.shape[0]
    correlations = np.zeros(n_lags)
    for k in range(n_lags):
        xt_centered = traj[k] - traj[k].mean(axis=0, keepdims=True)
        cov = (x0_centered * xt_centered).mean(axis=0)
        per_dim = np.where(var_x0 > 1e-12, cov / np.maximum(var_x0, 1e-12), 0.0)
        correlations[k] = float(per_dim.mean())

    threshold = float(np.exp(-1.0))
    below = np.where(correlations < threshold)[0]
    if len(below) == 0:
        return TAU_LAG_MAX
    first = int(below[0])
    dt_lag = TAU_SAMPLE_EVERY * TAU_RK4_STEP
    if first == 0:
        return dt_lag
    c_above = correlations[first - 1]
    c_below = correlations[first]
    if abs(c_above - c_below) < 1e-12:
        return first * dt_lag
    frac = (c_above - threshold) / (c_above - c_below)
    return (first - 1 + frac) * dt_lag
