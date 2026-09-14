"""Strict MDL optimum vs ridge surrogate (Appendix D support).

The estimator locates the readout by the quadratic surrogate (ridge, Eq. 7) and
scores it with the spectral code. This experiment instead minimizes the full
MDL objective (Eq. 6) directly,

    J_MDL(W) = ||Y~ - H~ W||_F^2 / (2 sigma^2 ln 2)
             + alpha log2 det(I + eta W W^T),

with sigma^2 = lambda / (2 alpha eta) (the merge that defines the ridge
parameter), and compares S^phi_MDL = spectral code of W_MDL against the
reported S^phi = spectral code of W_lambda on identical reservoirs and data.

The minimization uses majorize-minimize: log det(I + eta W W^T) is concave in
W W^T, so its linearization at W_k is a global upper bound whose minimizer is
a weighted ridge solve, W_{k+1} = (G + lambda M_k)^{-1} B with
M_k = (I + eta W_k W_k^T)^{-1}. Each sweep monotonically decreases J_MDL and
converges to a stationary point; warm start is the ridge optimum W_lambda.

Tasks:
    --task eca   all 88 rules x N_RUNS reservoir/data draws, the exact seeds of
                 the formal src/eca.py run.
    --task nca   every formal inverse-NCA checkpoint (3 seeds x 11 steps),
                 scoring data regenerated as in training (warmup + noise +
                 stacked window) from fixed evaluation seeds.

Figures are drawn by src/plot/surrogate_vs_mdl.py (runnable standalone).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import matplotlib

matplotlib.use("Agg")

import torch

import eca as eca_experiment
from rc_epiplexity import RCEpiplexity1D
from systems.checkpoints import checkpoint_paths, load_gzip_checkpoint
from systems.eca import eca_stacked_pair
from systems.nca import (
    STATE_CHANNELS,
    LocalNCA1D,
    evolve,
    evolve_stacked_window,
    random_unit_state,
)

from plot import surrogate_vs_mdl as svm_plot

PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "appendix", "surrogate_vs_mdl")
NCA_RUNS_ROOT = os.path.join(PROJECT_ROOT, "results", "inverse_nca")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Code-length constants of the estimator (Table 1: all experiments alpha=1/2,
# eta=1 except MNIST, which is not scored here).
ALPHA = 0.5
ETA = 1.0
_LN2 = math.log(2.0)

# Majorize-minimize settings for the strict MDL solve (float64, warm-started
# at the ridge optimum).
MM_MAX_ITER = 500
MM_REL_TOL = 1e-13

NCA_SEEDS = (1, 2, 3)
NCA_EVAL_DRAWS = 3
NCA_EVAL_SEED = 20260707


def normalized_design(
    reservoir: RCEpiplexity1D, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float64 normalized features/targets, matching ``multioutput_epiplexity``."""
    with torch.no_grad():
        h = reservoir.phi(x)
    features = h.movedim(1, 2).reshape(-1, reservoir.channels).double()
    target = y.movedim(1, 2).reshape(-1, y.shape[1]).double()
    target = (target - target.mean(dim=0, keepdim=True)) / reservoir.sigma_y
    features = (features - features.mean(dim=0, keepdim=True)) / (
        features.std(dim=0, correction=0, keepdim=True)
        * (reservoir.feature_dim**0.5)
        + reservoir.eps
    )
    return features, target


def spectral_code_bits(w: torch.Tensor) -> torch.Tensor:
    """alpha * log2 det(I_D + eta W^T W), via the (D x D) Gram determinant."""
    d = w.shape[1]
    gram = torch.eye(d, dtype=w.dtype, device=w.device) + ETA * (w.mT @ w)
    return ALPHA * torch.logdet(gram) / _LN2


def mdl_readout(
    features: torch.Tensor,
    target: torch.Tensor,
    ridge_lambda: float,
    init_scales: tuple[float, ...] = (1.0,),
) -> dict[str, float]:
    """Minimize the full MDL objective by majorize-minimize from the ridge start.

    Works on the sufficient statistics G = H^T H, B = H^T Y (float64). Each MM
    sweep solves the weighted ridge (G + lambda M_k) W = B with
    M_k = (I + eta W_k W_k^T)^{-1} (Woodbury through the D x D inner matrix),
    which monotonically decreases J_MDL. ``grad_norm`` reports the norm of the
    true gradient of J_MDL at the returned point (stationarity check).
    """
    gram = features.mT @ features
    cross = features.mT @ target
    target_sq = (target * target).sum()
    m, d = features.shape[1], target.shape[1]
    sigma_sq = ridge_lambda / (2.0 * ALPHA * ETA)
    eye_m = torch.eye(m, dtype=features.dtype, device=features.device)
    eye_d = torch.eye(d, dtype=features.dtype, device=features.device)
    w_ridge = torch.linalg.solve(gram + ridge_lambda * eye_m, cross)

    def objective(w: torch.Tensor) -> float:
        residual = target_sq - 2.0 * (w * cross).sum() + (w * (gram @ w)).sum()
        return float(residual / (2.0 * sigma_sq * _LN2) + spectral_code_bits(w))

    def m_inverse(w: torch.Tensor) -> torch.Tensor:
        # (I_m + eta w w^T)^{-1} = I_m - w (eta^{-1} I_D + w^T w)^{-1} w^T
        return eye_m - w @ torch.linalg.solve(eye_d / ETA + w.mT @ w, w.mT)

    best_obj, best_w = None, None
    for scale in init_scales:
        w = w_ridge * scale
        prev = objective(w)
        for _ in range(MM_MAX_ITER):
            w = torch.linalg.solve(gram + ridge_lambda * m_inverse(w), cross)
            obj = objective(w)
            if abs(prev - obj) <= MM_REL_TOL * max(1.0, abs(prev)):
                prev = obj
                break
            prev = obj
        if best_obj is None or prev < best_obj:
            best_obj, best_w = prev, w

    # True gradient of J_MDL at the solution: [G W - B + lambda M W] / (sigma^2 ln2).
    grad = (gram @ best_w - cross + ridge_lambda * (m_inverse(best_w) @ best_w)) / (
        sigma_sq * _LN2
    )
    return {
        "s_mdl": float(spectral_code_bits(best_w)),
        "s_ridge_f64": float(spectral_code_bits(w_ridge)),
        "j_mdl": best_obj,
        "j_ridge": objective(w_ridge),
        "grad_norm": float(grad.norm()),
    }


def run_eca(rules: list[int], n_runs: int, init_scales: tuple[float, ...]) -> str:
    """Score every (rule, run) with both the surrogate path and the strict MDL."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eca.csv")
    fields = ["rule", "run", "s_ridge", "s_mdl", "s_ridge_f64", "j_ridge", "j_mdl", "grad_norm"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rule in rules:
            for run in range(n_runs):
                reservoir = eca_experiment.make_eca_reservoir(
                    eca_experiment.SEED + 1000 * run
                )
                x, y = eca_stacked_pair(
                    rule=rule,
                    num_samples=eca_experiment.NUM_SAMPLES,
                    width=eca_experiment.WIDTH,
                    burnin=eca_experiment.BURNIN_STEPS,
                    tau_max=eca_experiment.TAU_MAX,
                    seed=eca_experiment.SEED + rule + 100_000 * run,
                    device=DEVICE,
                )
                s_ridge = float(reservoir.epiplexity(x, y).epiplexity)
                features, target = normalized_design(reservoir, x, y)
                row = mdl_readout(
                    features, target, eca_experiment.RIDGE_LAMBDA, init_scales
                )
                row.update({"rule": rule, "run": run, "s_ridge": s_ridge})
                writer.writerow(row)
                f.flush()
            print(f"rule {rule:>3}: done ({n_runs} runs)", flush=True)
    return out_path


def build_nca_reservoir(seed: int, config: dict) -> RCEpiplexity1D:
    """The frozen scorer of the formal run: weights depend only on init(seed+1)."""
    reservoir = RCEpiplexity1D(
        depth=config["reservoir_depth"],
        channels=config["reservoir_channels"],
        kernel_size=config["reservoir_kernel_size"],
        input_channels=STATE_CHANNELS,
        ridge_lambda=config["ridge_lambda"],
        eps=config["eps"],
        device=DEVICE,
    )
    reservoir.init(seed + 1)
    reservoir.eval()
    for param in reservoir.parameters():
        param.requires_grad_(False)
    return reservoir


def run_nca(init_scales: tuple[float, ...]) -> str:
    """Score every formal checkpoint with both estimators on fixed eval draws.

    The scoring rollout mirrors training exactly (train-mode BatchNorm, warmup,
    Gaussian perturbation, stacked window tau_min..tau_max); the evaluation
    seeds are fixed and shared across checkpoints so scores are comparable
    along a trajectory.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "nca.csv")
    fields = [
        "seed", "step", "eval_run",
        "s_ridge", "s_mdl", "s_ridge_f64", "j_ridge", "j_mdl", "grad_norm",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in NCA_SEEDS:
            run_dir = os.path.join(NCA_RUNS_ROOT, f"seed_{seed}")
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as cf:
                config = json.load(cf)
            reservoir = build_nca_reservoir(seed, config)
            for path in checkpoint_paths(run_dir):
                ckpt = load_gzip_checkpoint(path, DEVICE)
                nca = LocalNCA1D(update_mode=config["update_mode"]).to(DEVICE)
                nca.load_state_dict(ckpt["nca_state_dict"])
                for eval_run in range(NCA_EVAL_DRAWS):
                    nca.train()  # training scored with batch statistics
                    torch.manual_seed(NCA_EVAL_SEED + eval_run)
                    state = random_unit_state(
                        config["batch_size"], config["width"], DEVICE
                    )
                    x = evolve(nca, state, config["warmup_steps"], track_grad=False)
                    x = x + torch.randn_like(x) * config["noise_std"]
                    with torch.no_grad():
                        y = evolve_stacked_window(
                            nca, x, config["tau_min"], config["tau_max"]
                        )
                    s_ridge = float(reservoir.epiplexity(x, y).epiplexity)
                    features, target = normalized_design(reservoir, x, y)
                    row = mdl_readout(
                        features, target, config["ridge_lambda"], init_scales
                    )
                    row.update(
                        {"seed": seed, "step": ckpt["step"], "eval_run": eval_run,
                         "s_ridge": s_ridge}
                    )
                    writer.writerow(row)
                    f.flush()
                print(
                    f"seed {seed} step {ckpt['step']:>4}: "
                    f"s_ridge={s_ridge:.3f} s_mdl={row['s_mdl']:.3f}",
                    flush=True,
                )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["eca", "nca"], required=True)
    parser.add_argument(
        "--rules", type=int, nargs="*", default=None,
        help="ECA rule subset for a quick pilot (default: the full formal set).",
    )
    parser.add_argument(
        "--runs", type=int, default=eca_experiment.N_RUNS,
        help="Reservoir/data draws per ECA rule (default: the formal N_RUNS).",
    )
    parser.add_argument(
        "--init-scales", type=float, nargs="*", default=[1.0],
        help="Warm-start scales for the MDL solve (local-minimum check).",
    )
    args = parser.parse_args()
    init_scales = tuple(args.init_scales)
    print(f"Device: {DEVICE}; task={args.task}; init_scales={init_scales}", flush=True)

    if args.task == "eca":
        rules = args.rules if args.rules else list(eca_experiment.RULES)
        path = run_eca(rules, args.runs, init_scales)
        print(f"ECA comparison -> {path}", flush=True)
    else:
        path = run_nca(init_scales)
        print(f"NCA comparison -> {path}", flush=True)
    print(f"Figure -> {svm_plot.render_figure()}", flush=True)


if __name__ == "__main__":
    main()
