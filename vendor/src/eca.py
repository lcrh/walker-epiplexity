"""Reservoir-computing diagnostic for Elementary Cellular Automata.

Each rule is scored on a *stacked-horizon* target: from the burned-in state ``x``
we predict the whole window of the next ``TAU_MAX`` one-step states
``(F^1 x, ..., F^{TAU_MAX} x)`` jointly, as a multi-output target. This removes
the single ``dt`` hyperparameter -- the readout's singular-value log-volume
charges redundant / repeated structure across horizons at ~zero, so a rule whose
future window is low-rank (dies, freezes, or merely shifts) scores low while a
rule that keeps producing fresh structure scores high. ``TAU_MAX`` is the
temporal window and is kept *long* (it is not the reservoir's receptive field);
paired with a deliberately local reservoir this surfaces rule 110 as the ECA
maximum.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from rc_epiplexity import RCEpiplexity1D
from systems.eca import AUTHOR_ECA_BURNIN_STEPS, AUTHOR_ECA_WIDTH, eca_stacked_pair

from plot import eca as eca_plot
from plot.eca import RuleResult


REFERENCE_RULES = [1, 2, 3, 30, 54, 110]
COMBINED_TOP_N = 14
RULES = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    18, 19, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 32, 33, 34, 35, 36,
    37, 38, 40, 41, 42, 43, 44, 45,
    46, 50, 51, 54, 56, 57, 58, 60,
    62, 72, 73, 74, 76, 77, 78, 90,
    94, 104, 105, 106, 108, 110, 122, 126,
    128, 130, 132, 134, 136, 138, 140, 142,
    146, 150, 152, 154, 156, 160, 162, 164,
    168, 170, 172, 178, 184, 200, 204, 232,
]
WIDTH = AUTHOR_ECA_WIDTH
BURNIN_STEPS = AUTHOR_ECA_BURNIN_STEPS
NUM_SAMPLES = 512
EPS = 1e-8

RESERVOIR_DEPTH = 3
RESERVOIR_CHANNELS = 256
RESERVOIR_KERNEL_SIZE = 3
RESERVOIR_ACTIVATION = "elu"

# Number of stacked one-step horizons. This is the temporal window, NOT the
# reservoir's receptive field. A long window is what lets a computationally
# structured rule accumulate learnable novelty over time: rule 110's gliders stay
# locally predictable for many steps, so over a long window they fill many
# independent readout directions, while a dense/chaotic rule's far horizons are
# unpredictable from the local view and the SVD log-volume prices them at ~0.
# Paired with a deliberately *local* reservoir (depth 3, kernel 3 -> radius 2),
# this long window surfaces rule 110 as the epiplexity maximum among ECA rules.
TAU_MAX = 32

RIDGE_LAMBDA = 0.03
SEED = 0
N_RUNS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "results", "eca"
)


def make_eca_reservoir(seed: int) -> RCEpiplexity1D:
    """Build a fresh frozen 1D reservoir with weights drawn from ``seed``."""
    torch.manual_seed(seed)
    reservoir = RCEpiplexity1D(
        depth=RESERVOIR_DEPTH,
        channels=RESERVOIR_CHANNELS,
        kernel_size=RESERVOIR_KERNEL_SIZE,
        ridge_lambda=RIDGE_LAMBDA,
        eps=EPS,
        device=DEVICE,
    )
    for param in reservoir.parameters():
        param.requires_grad_(False)
    return reservoir


def run_rule(rule: int, n_runs: int = N_RUNS) -> RuleResult:
    """Average the epiplexity and residual over ``n_runs`` independent draws.

    Each run resamples both the random reservoir weights and the ECA data so the
    reported mean reflects variability over both sources of randomness.
    """
    epiplexities = []
    residuals = []
    for run in range(n_runs):
        reservoir = make_eca_reservoir(SEED + 1000 * run)
        x, y = eca_stacked_pair(
            rule=rule,
            num_samples=NUM_SAMPLES,
            width=WIDTH,
            burnin=BURNIN_STEPS,
            tau_max=TAU_MAX,
            seed=SEED + rule + 100_000 * run,
            device=DEVICE,
        )
        output = reservoir.epiplexity(x, y)
        epiplexities.append(output.epiplexity.item())
        residuals.append(output.residual.pow(2).mean().item())

    epiplexities = np.asarray(epiplexities)
    residuals = np.asarray(residuals)
    return RuleResult(
        rule=rule,
        reservoir_epiplexity=float(epiplexities.mean()),
        num_samples=NUM_SAMPLES,
        width=WIDTH,
        feature_dim=RESERVOIR_CHANNELS,
        residual=float(residuals.mean()),
        reservoir_epiplexity_std=float(epiplexities.std()),
        residual_std=float(residuals.std()),
        n_runs=n_runs,
    )


def eca_csv_metadata() -> dict:
    return {
        "reservoir_activation": RESERVOIR_ACTIVATION,
        "ridge_lambda": RIDGE_LAMBDA,
        "dt": f"stack_1..{TAU_MAX}",
        "burnin_steps": BURNIN_STEPS,
        "seed": SEED,
    }


def run_eca() -> RuleResult:
    print(f"\nTau_max={TAU_MAX}: running full selected rule set ({N_RUNS} runs/rule)...")
    results = []
    for rule in RULES:
        result = run_rule(rule)
        results.append(result)
        print(
            f"rule {rule:>3}: "
            f"epiplexity={result.reservoir_epiplexity:.6f}"
            f"±{result.reservoir_epiplexity_std:.6f} "
            f"residual={result.residual:.6f}±{result.residual_std:.6f}"
        )

    eca_plot.save_csv(results, RESULTS_ROOT, extra=eca_csv_metadata())
    eca_plot.plot_combined(
        results, RESULTS_ROOT, reference_rules=REFERENCE_RULES, top_n=COMBINED_TOP_N, seed=SEED
    )
    best = max(results, key=lambda r: r.reservoir_epiplexity)
    print(f"Top rule {best.rule} with epiplexity={best.reservoir_epiplexity:.6f}")
    return best


def main() -> None:
    print(
        f"Device: {DEVICE}; rules={len(RULES)}; width={WIDTH}; "
        f"tau_max={TAU_MAX}; runs/rule={N_RUNS}"
    )
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    run_eca()


if __name__ == "__main__":
    main()
