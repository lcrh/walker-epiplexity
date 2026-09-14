"""MLP-reservoir epiplexity for continuous-time dynamical systems.

Each sample draws a random initial condition and integrates the system with RK4
for a burn-in time to land on the attractor, giving the input state ``x``. The
target is a *stacked-horizon* window: the trajectory is snapshotted at
``NUM_SNAPSHOTS`` evenly spaced lead times ``SNAPSHOT_DT, 2*SNAPSHOT_DT, ...``
and the snapshots are stacked into one multi-output target ``y``. The MLP
reservoir maps ``x`` to hidden features and one ridge readout per output
coordinate predicts the whole window; the epiplexity is the singular-value
log-volume of the stacked readout, which charges redundant structure across lead
times at ~zero so no single ``delta_t`` has to be chosen.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from rc_epiplexity import RCEpiplexityMLP
from systems.flows import SYSTEMS, SystemSpec, autocorrelation_time, integrate, trajectory

from plot import dynamical_systems_experiment as ds_plot
from plot.dynamical_systems_experiment import SystemResult


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
EPS = 1e-8

# --- MLP reservoir settings ---
NUM_SAMPLES = 512
HIDDEN_DIM = 64
RIDGE_LAMBDA = 0.1
RESERVOIR_REPEATS = 8

# --- Trajectory sampling ---
BURNIN_TIME = 0.5
# Stacked-horizon target: snapshot the trajectory at lead times
# SNAPSHOT_DT, 2*SNAPSHOT_DT, ..., NUM_SNAPSHOTS*SNAPSHOT_DT (= 0.1 .. 1.0) and
# stack them as a multi-output target (replaces the old single delta_t endpoint).
SNAPSHOT_DT = 0.1
NUM_SNAPSHOTS = 10
RK4_STEP = 0.01

# --- Phase-portrait display trajectories ---
TRAJ_NUM = 8
TRAJ_K_TAU = 30.0
TRAJ_POINTS = 2000
TRAJ_RK4_STEP = 0.002
TRAJ_SEED = 1234
TAU_SEED_OFFSET = 1000

RESULTS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "results", "dynamical_systems"
)


def build_stacked_pair(spec: SystemSpec, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attractor state ``x`` and its NUM_SNAPSHOTS forward snapshots, stacked.

    Returns ``x`` of shape ``(B, state_dim)`` and ``y`` of shape
    ``(B, NUM_SNAPSHOTS, state_dim)`` where ``y[:, k]`` is the state at lead time
    ``(k + 1) * SNAPSHOT_DT``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, spec.init_scale, size=(NUM_SAMPLES, spec.state_dim))
    if spec.pre_burnin_time > 0.0:
        z = integrate(spec.field, z, spec.pre_burnin_time, RK4_STEP)
    x = integrate(spec.field, z, BURNIN_TIME, RK4_STEP)
    futures = []
    state = x
    for _ in range(NUM_SNAPSHOTS):
        state = integrate(spec.field, state, SNAPSHOT_DT, RK4_STEP)
        futures.append(state)
    y = np.stack(futures, axis=1)  # (B, NUM_SNAPSHOTS, state_dim)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError(f"non-finite state from system {spec.name}")
    x_t = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
    y_t = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    return x_t, y_t


def score_system(spec: SystemSpec, sample_seed: int) -> SystemResult:
    x, y = build_stacked_pair(spec, sample_seed)
    # Plan the scale from the data: standardize the state per dimension so the
    # cross-system comparison reflects the map's structure, not raw magnitude.
    # Under the fixed-sigma_y program length, raw target scale would otherwise
    # enter the score as a bonus -- desirable for inverse problems, but not when
    # comparing given systems whose attractors differ in scale by orders of
    # magnitude (Lorenz ~O(40) vs the linear systems ~O(1)). Every snapshot lives
    # in x's coordinate frame, so all share x's per-dim statistics; this preserves
    # the relative spread/decay across lead times (which is itself information).
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, correction=0, keepdim=True) + EPS
    x = (x - mu) / sd
    y = (y - mu.unsqueeze(1)) / sd.unsqueeze(1)  # broadcast over the snapshot axis
    y = y.reshape(y.shape[0], -1)  # (B, NUM_SNAPSHOTS * state_dim)
    scores = np.zeros(RESERVOIR_REPEATS, dtype=np.float64)
    for repeat in range(RESERVOIR_REPEATS):
        torch.manual_seed(SEED + repeat)
        reservoir = RCEpiplexityMLP(
            input_dim=spec.state_dim,
            hidden_dim=HIDDEN_DIM,
            target_dim=y.shape[1],
            ridge_lambda=RIDGE_LAMBDA,
            eps=EPS,
            device=DEVICE,
        )
        for param in reservoir.parameters():
            param.requires_grad_(False)
        output = reservoir.epiplexity(x, y)
        scores[repeat] = output.epiplexity.item()
    return SystemResult(
        name=spec.name,
        state_dim=spec.state_dim,
        epiplexity_mean=float(scores.mean()),
        epiplexity_std=float(scores.std()),
    )


def _portrait_trajectory(spec: SystemSpec, tau: float, rng: np.random.Generator) -> np.ndarray:
    traj_time = TRAJ_K_TAU * tau
    sample_every = max(1, int(round(traj_time / (TRAJ_RK4_STEP * TRAJ_POINTS))))
    z = rng.normal(0.0, spec.init_scale, size=(TRAJ_NUM, spec.state_dim))
    if spec.pre_burnin_time > 0.0:
        z = integrate(spec.field, z, spec.pre_burnin_time, TRAJ_RK4_STEP)
    return trajectory(spec.field, z, traj_time, TRAJ_RK4_STEP, sample_every)


def portrait_trajectories(taus: dict[str, float]) -> list[tuple[str, np.ndarray, tuple[int, int]]]:
    """Integrate the display trajectories for every system, in SYSTEMS order."""
    rng = np.random.default_rng(TRAJ_SEED)
    return [
        (spec.name, _portrait_trajectory(spec, taus[spec.name], rng), spec.projection)
        for spec in SYSTEMS
    ]


def compute_taus() -> dict[str, float]:
    taus: dict[str, float] = {}
    for idx, spec in enumerate(SYSTEMS):
        tau = autocorrelation_time(spec, seed=SEED + TAU_SEED_OFFSET + idx)
        taus[spec.name] = tau
        print(f"{spec.name}: tau={tau:.4f}")
    return taus


def csv_metadata() -> dict:
    return {
        "num_samples": NUM_SAMPLES,
        "hidden_dim": HIDDEN_DIM,
        "ridge_lambda": RIDGE_LAMBDA,
        "burnin_time": BURNIN_TIME,
        "delta_t": f"stack_{SNAPSHOT_DT}..{NUM_SNAPSHOTS * SNAPSHOT_DT:g}",
        "rk4_step": RK4_STEP,
        "reservoir_repeats": RESERVOIR_REPEATS,
        "seed": SEED,
    }


def main() -> None:
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    bar_csv = os.path.join(RESULTS_ROOT, "dynamical_systems_scores.csv")
    figure_pdf = os.path.join(RESULTS_ROOT, "dynamical_systems_figure.pdf")

    print(f"Device: {DEVICE}")
    taus = compute_taus()

    print(
        f"Config: samples={NUM_SAMPLES}, hidden_dim={HIDDEN_DIM}, "
        f"lambda={RIDGE_LAMBDA}, burn-in={BURNIN_TIME}, "
        f"snapshots={NUM_SNAPSHOTS}x{SNAPSHOT_DT}, "
        f"rk4={RK4_STEP}, repeats={RESERVOIR_REPEATS}"
    )
    torch.manual_seed(SEED)
    results = []
    for idx, spec in enumerate(SYSTEMS):
        result = score_system(spec, sample_seed=SEED + idx)
        results.append(result)
        print(
            f"{spec.name}: epiplexity={result.epiplexity_mean:.4f}"
            f"+/-{result.epiplexity_std:.4f}"
        )
    ds_plot.save_csv(results, bar_csv, extra=csv_metadata())

    ds_plot.plot_composite_figure(results, portrait_trajectories(taus), figure_pdf)
    print(f"Saved figure to {figure_pdf}")


if __name__ == "__main__":
    main()
