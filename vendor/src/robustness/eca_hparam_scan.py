"""One-factor-at-a-time hyperparameter robustness ablation for the ECA ranking.

The formal ECA result (``src/eca.py``) claims rule 110 has the highest
epiplexity among the scored ECA rules under one fixed estimator configuration
(depth=3, channels=256, kernel=3, ridge_lambda=0.03, lambda_code=1.0,
tau_max=32, width=64, burnin=1000, num_samples=512). This scan asks: over what
range of each hyperparameter does rule 110 stay rank 1, how the ranks of the
reference rules 30 and 54 move, and how the overall ranking deforms as each
hyperparameter moves. Seven estimator/sampling axes are scanned one at a time
around the formal defaults (activation is fixed at ELU throughout -- not
varied, since that would require touching the core reservoir; the lattice
width is not scanned either, since it varies the scored system rather than
the estimator).

Three stages:
  A. Timing probe: score the baseline config once (n_runs=1, all rules), time
     it, and extrapolate the total Stage B budget.
  B. Full OFAT scan at n_runs=1 (run index r=0), axis by axis. Every row is
     appended to the raw CSV as it is computed.
  C. Confirmation at n_runs=3 for every config where rule 110 is not rank 1,
     the flanking values of any rank transition along each axis, the flanking
     values of any 30/54 score-order flip, plus the baseline itself. Run
     index r=0 is reused from Stage B via the cache.

This module computes and writes the three CSVs only; all figures are drawn by
``plot_eca_hparam.py`` (called once at the end of the run, and runnable
standalone). Diagnostic only: writes to ``results/robustness/``, never touches
``results/eca/`` (the formal outputs), and never modifies ``src/eca.py``,
``src/rc_epiplexity/``, or ``src/systems/``.

Run:
    uv run python src/robustness/eca_hparam_scan.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from eca import (
    BURNIN_STEPS,
    DEVICE,
    EPS,
    NUM_SAMPLES,
    RESERVOIR_ACTIVATION,
    RESERVOIR_CHANNELS,
    RESERVOIR_DEPTH,
    RESERVOIR_KERNEL_SIZE,
    RIDGE_LAMBDA,
    RULES,
    SEED,
    TAU_MAX,
    WIDTH,
    eca_stacked_pair,
)
from rc_epiplexity import RCEpiplexity1D

try:
    from scipy.stats import spearmanr as _scipy_spearmanr

    def spearman(a, b) -> float:
        rho, _ = _scipy_spearmanr(a, b)
        return float(rho)

except ImportError:  # pragma: no cover - fallback if scipy is unavailable

    def _rankdata(a) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        return ranks

    def spearman(a, b) -> float:
        ra, rb = _rankdata(np.asarray(a, dtype=float)), _rankdata(
            np.asarray(b, dtype=float)
        )
        return float(np.corrcoef(ra, rb)[0, 1])


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "robustness", "ECA"
)

REF_RANK_RULES = [1, 2, 3, 30, 54]
TARGET_RULE = 110

# Baseline hyperparameter configuration: mirrors src/eca.py's formal defaults.
BASELINE = dict(
    tau_max=TAU_MAX,
    ridge_lambda=RIDGE_LAMBDA,
    lambda_code=1.0,
    depth=RESERVOIR_DEPTH,
    kernel_size=RESERVOIR_KERNEL_SIZE,
    channels=RESERVOIR_CHANNELS,
    num_samples=NUM_SAMPLES,
    width=WIDTH,
)

# name -> (values including baseline, log-x when plotted)
AXES = {
    "tau_max": ([4, 8, 16, 32, 64], True),
    "ridge_lambda": ([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], True),
    "lambda_code": ([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], True),
    "depth": ([1, 2, 3, 4, 5], False),
    "kernel_size": ([3, 5, 7], False),
    "channels": ([64, 128, 256, 512], True),
    "num_samples": ([128, 256, 512, 1024], True),
}
AXIS_PRIORITY = [
    "lambda_code",
    "ridge_lambda",
    "tau_max",
    "depth",
    "kernel_size",
    "channels",
    "num_samples",
]

RAW_FIELDS = ["axis", "value", "rule", "run", "epiplexity", "residual"]
SUMMARY_FIELDS = [
    "axis",
    "value",
    "n_runs",
    "rank110",
    "margin",
    "runner_up",
    "top5",
    "rank1",
    "rank2",
    "rank3",
    "rank30",
    "rank54",
    "spearman_rho",
    "activation",
    "burnin_steps",
    "seed",
]
CONFIRM_FIELDS = [
    "axis",
    "value",
    "n_runs",
    "rank110_of_mean",
    "mean_epiplexity_110",
    "std_epiplexity_110",
    "runner_up",
    "mean_epiplexity_runner_up",
    "std_epiplexity_runner_up",
    "margin_of_means",
    "rank30",
    "rank54",
    "s54_minus_s30_mean",
    "s54_minus_s30_std",
    "spearman_rho",
]


# ---------------------------------------------------------------- scoring

_RUN_CACHE: dict[tuple, dict[int, tuple[float, float]]] = {}


def _cfg_key(cfg: dict, run_idx: int) -> tuple:
    return (tuple(sorted(cfg.items())), run_idx)


def run_single(cfg: dict, run_idx: int) -> dict[int, tuple[float, float]]:
    """Score every rule in ``RULES`` for one run index under one hparam config.

    Mirrors ``eca.py``'s seeding exactly: the reservoir is drawn once from
    ``SEED + 1000 * run_idx`` and reused across all rules; each rule's data
    draw uses ``SEED + rule + 100_000 * run_idx``. Cached on (cfg, run_idx) so
    Stage C's confirmation runs can reuse Stage B's run 0 without recomputing.
    """
    key = _cfg_key(cfg, run_idx)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    torch.manual_seed(SEED + 1000 * run_idx)
    reservoir = RCEpiplexity1D(
        depth=cfg["depth"],
        channels=cfg["channels"],
        kernel_size=cfg["kernel_size"],
        ridge_lambda=cfg["ridge_lambda"],
        lambda_code=cfg["lambda_code"],
        eps=EPS,
        device=DEVICE,
    )
    for p in reservoir.parameters():
        p.requires_grad_(False)
    out: dict[int, tuple[float, float]] = {}
    for rule in RULES:
        x, y = eca_stacked_pair(
            rule=rule,
            num_samples=cfg["num_samples"],
            width=cfg["width"],
            burnin=BURNIN_STEPS,
            tau_max=cfg["tau_max"],
            seed=SEED + rule + 100_000 * run_idx,
            device=DEVICE,
        )
        with torch.no_grad():
            output = reservoir.epiplexity(x, y)
        out[rule] = (
            float(output.epiplexity.item()),
            float(output.residual.pow(2).mean().item()),
        )
    _RUN_CACHE[key] = out
    return out


def score_config(cfg: dict, n_runs: int) -> dict:
    """Mean/std epiplexity + residual per rule over run indices ``0..n_runs-1``."""
    per_run = [run_single(cfg, r) for r in range(n_runs)]
    mean, std = {}, {}
    for rule in RULES:
        eps = np.array([pr[rule][0] for pr in per_run])
        res = np.array([pr[rule][1] for pr in per_run])
        mean[rule] = (float(eps.mean()), float(res.mean()))
        std[rule] = (float(eps.std()), float(res.std()))
    return {"per_run": per_run, "mean": mean, "std": std}


def compute_metrics(
    mean_scores: dict[int, float], baseline_rank: dict[int, int] | None
) -> dict:
    order = sorted(RULES, key=lambda r: mean_scores[r], reverse=True)
    rank = {r: i + 1 for i, r in enumerate(order)}
    other_max = max(mean_scores[r] for r in RULES if r != TARGET_RULE)
    margin = mean_scores[TARGET_RULE] - other_max
    runner_up = max(
        (r for r in RULES if r != TARGET_RULE), key=lambda r: mean_scores[r]
    )
    top5 = order[:5]
    ref_ranks = {r: rank[r] for r in REF_RANK_RULES}
    rho = (
        1.0
        if baseline_rank is None
        else spearman([rank[r] for r in RULES], [baseline_rank[r] for r in RULES])
    )
    return dict(
        rank=rank,
        order=order,
        margin=margin,
        runner_up=runner_up,
        top5=top5,
        ref_ranks=ref_ranks,
        rho=rho,
    )


# ---------------------------------------------------------------- CSV / logging


def append_raw_rows(writer, f, axis: str, value, config_result: dict) -> None:
    for run_idx, run_scores in enumerate(config_result["per_run"]):
        for rule in RULES:
            eps, res = run_scores[rule]
            writer.writerow([axis, value, rule, run_idx, eps, res])
    f.flush()


def append_summary_row(writer, f, axis: str, value, n_runs: int, metrics: dict) -> None:
    writer.writerow(
        [
            axis,
            value,
            n_runs,
            metrics["rank"][TARGET_RULE],
            metrics["margin"],
            metrics["runner_up"],
            ";".join(str(r) for r in metrics["top5"]),
            metrics["ref_ranks"][1],
            metrics["ref_ranks"][2],
            metrics["ref_ranks"][3],
            metrics["ref_ranks"][30],
            metrics["ref_ranks"][54],
            metrics["rho"],
            RESERVOIR_ACTIVATION,
            BURNIN_STEPS,
            SEED,
        ]
    )
    f.flush()


def print_config_line(axis: str, value, n_runs: int, metrics: dict) -> None:
    print(
        f"[{axis:>12}] value={value!s:>7} n_runs={n_runs} "
        f"rank110={metrics['rank'][TARGET_RULE]:>2} margin={metrics['margin']:+8.3f} "
        f"runner_up={metrics['runner_up']:>3} rho={metrics['rho']:.3f} top5={metrics['top5']}",
        flush=True,
    )


# ---------------------------------------------------------------- Stage B: OFAT scan


def run_axis(
    axis_name: str,
    values: list,
    n_runs: int,
    raw_writer,
    raw_f,
    summary_writer,
    summary_f,
    baseline_result: dict,
    baseline_metrics: dict,
    baseline_rank: dict[int, int],
) -> list[dict]:
    records = []
    for value in values:
        if value == BASELINE[axis_name]:
            config_result, metrics = baseline_result, baseline_metrics
        else:
            cfg = dict(BASELINE)
            cfg[axis_name] = value
            config_result = score_config(cfg, n_runs=n_runs)
            metrics = compute_metrics(
                {r: config_result["mean"][r][0] for r in RULES}, baseline_rank
            )
        append_raw_rows(raw_writer, raw_f, axis_name, value, config_result)
        append_summary_row(summary_writer, summary_f, axis_name, value, n_runs, metrics)
        print_config_line(axis_name, value, n_runs, metrics)
        mean_scores = {r: config_result["mean"][r][0] for r in RULES}
        records.append(
            dict(
                value=value,
                rank110=metrics["rank"][TARGET_RULE],
                gap_54_30=mean_scores[54] - mean_scores[30],
            )
        )
    return records


# ---------------------------------------------------------------- Stage C: confirmation


def find_confirmation_targets(
    axis_records: dict[str, list[dict]],
) -> list[tuple[str, object]]:
    """Configs where rule 110 lost rank 1, plus the flanking values around every
    rank-1/not-rank-1 transition and every 30/54 score-order flip along each
    axis (values are already in the axis's stated order, so adjacency in the
    list is adjacency in the scan)."""
    targets: set[tuple[str, object]] = set()
    for axis_name, records in axis_records.items():
        for rec in records:
            if rec["rank110"] != 1:
                targets.add((axis_name, rec["value"]))
        for a, b in zip(records, records[1:]):
            rank_flip = (a["rank110"] == 1) != (b["rank110"] == 1)
            order_flip = (a["gap_54_30"] > 0) != (b["gap_54_30"] > 0)
            if rank_flip or order_flip:
                targets.add((axis_name, a["value"]))
                targets.add((axis_name, b["value"]))
    return sorted(targets, key=lambda t: (t[0], str(t[1])))


def run_confirmation(
    targets: list[tuple[str, object]],
    n_runs: int,
    baseline_rank: dict[int, int],
    confirm_writer,
    confirm_f,
) -> None:
    # Always confirm the baseline itself.
    all_targets: list[tuple[str, object]] = [("baseline", "baseline")] + targets
    for axis_name, value in all_targets:
        cfg = dict(BASELINE)
        if axis_name != "baseline":
            cfg[axis_name] = value
        config_result = score_config(cfg, n_runs=n_runs)
        mean_scores = {r: config_result["mean"][r][0] for r in RULES}
        metrics = compute_metrics(
            mean_scores, baseline_rank if axis_name != "baseline" else None
        )
        m110, s110 = (
            config_result["mean"][TARGET_RULE][0],
            config_result["std"][TARGET_RULE][0],
        )
        runner_up = metrics["runner_up"]
        m_ru, s_ru = (
            config_result["mean"][runner_up][0],
            config_result["std"][runner_up][0],
        )
        gaps = np.array([pr[54][0] - pr[30][0] for pr in config_result["per_run"]])
        confirm_writer.writerow(
            [
                axis_name,
                value,
                n_runs,
                metrics["rank"][TARGET_RULE],
                m110,
                s110,
                runner_up,
                m_ru,
                s_ru,
                m110 - m_ru,
                metrics["rank"][30],
                metrics["rank"][54],
                float(gaps.mean()),
                float(gaps.std()),
                metrics["rho"],
            ]
        )
        confirm_f.flush()
        print(
            f"[confirm {axis_name:>10}={value!s:>7}] n_runs={n_runs} "
            f"rank110(mean)={metrics['rank'][TARGET_RULE]:>2} "
            f"S110={m110:.3f}+/-{s110:.3f} runner_up={runner_up} "
            f"S_runner_up={m_ru:.3f}+/-{s_ru:.3f} "
            f"30#{metrics['rank'][30]} 54#{metrics['rank'][54]} "
            f"S54-S30={gaps.mean():+.2f}+/-{gaps.std():.2f}",
            flush=True,
        )


# ---------------------------------------------------------------- main


def main() -> None:
    print(f"Device: {DEVICE}; rules={len(RULES)}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, "eca_hparam_robustness_raw.csv")
    summary_path = os.path.join(OUTPUT_DIR, "eca_hparam_robustness_summary.csv")
    confirm_path = os.path.join(OUTPUT_DIR, "eca_hparam_robustness_confirmation.csv")

    axes = dict(AXES)

    with (
        open(raw_path, "w", newline="") as raw_f,
        open(summary_path, "w", newline="") as summary_f,
        open(confirm_path, "w", newline="") as confirm_f,
    ):
        raw_writer = csv.writer(raw_f)
        raw_writer.writerow(RAW_FIELDS)
        summary_writer = csv.writer(summary_f)
        summary_writer.writerow(SUMMARY_FIELDS)
        confirm_writer = csv.writer(confirm_f)
        confirm_writer.writerow(CONFIRM_FIELDS)

        # ---- Stage A: timing probe ----
        print(
            "Stage A: timing probe (baseline config, n_runs=1, all rules)...",
            flush=True,
        )
        t0 = time.time()
        baseline_result = score_config(BASELINE, n_runs=1)
        t_baseline = time.time() - t0
        n_nonbaseline = sum(len(v[0]) - 1 for v in axes.values())
        extrapolated_min = t_baseline * n_nonbaseline / 60.0
        print(
            f"Stage A: baseline took {t_baseline:.1f}s for {len(RULES)} rules; "
            f"{n_nonbaseline} non-baseline configs remain -> extrapolated "
            f"{extrapolated_min:.1f} min for Stage B",
            flush=True,
        )
        if extrapolated_min > 60.0:
            print(
                "Stage A: over budget -> reducing ridge_lambda/lambda_code to 5 values each",
                flush=True,
            )
            axes["ridge_lambda"] = ([0.003, 0.01, 0.03, 0.1, 0.3], True)
            axes["lambda_code"] = ([0.1, 0.3, 1.0, 3.0, 10.0], True)
            n_nonbaseline = sum(len(v[0]) - 1 for v in axes.values())
            extrapolated_min = t_baseline * n_nonbaseline / 60.0
            print(
                f"Stage A: after reduction, extrapolated {extrapolated_min:.1f} min "
                f"for {n_nonbaseline} configs",
                flush=True,
            )

        baseline_mean_scores = {r: baseline_result["mean"][r][0] for r in RULES}
        baseline_metrics = compute_metrics(baseline_mean_scores, None)
        baseline_rank = baseline_metrics["rank"]
        print(
            f"Baseline: rank110={baseline_metrics['rank'][TARGET_RULE]} "
            f"margin={baseline_metrics['margin']:.3f} top5={baseline_metrics['top5']}",
            flush=True,
        )

        # ---- Stage B: full OFAT scan, n_runs=1 ----
        print("\nStage B: OFAT scan at n_runs=1...", flush=True)
        axis_records: dict[str, list[dict]] = {}
        priority = [a for a in AXIS_PRIORITY if a in axes] + [
            a for a in axes if a not in AXIS_PRIORITY
        ]
        for axis_name in priority:
            values, _ = axes[axis_name]
            print(f"\n-- axis: {axis_name} --", flush=True)
            axis_records[axis_name] = run_axis(
                axis_name,
                values,
                n_runs=1,
                raw_writer=raw_writer,
                raw_f=raw_f,
                summary_writer=summary_writer,
                summary_f=summary_f,
                baseline_result=baseline_result,
                baseline_metrics=baseline_metrics,
                baseline_rank=baseline_rank,
            )

        # ---- Stage C: confirmation at n_runs=3 ----
        targets = find_confirmation_targets(axis_records)
        print(
            f"\nStage C: confirming {len(targets) + 1} configs at n_runs=3 "
            f"(includes baseline)...",
            flush=True,
        )
        run_confirmation(
            targets,
            n_runs=3,
            baseline_rank=baseline_rank,
            confirm_writer=confirm_writer,
            confirm_f=confirm_f,
        )

    print(f"\nWrote {raw_path}\nWrote {summary_path}\nWrote {confirm_path}", flush=True)

    # All figure assembly lives in the plot module (also runnable standalone).
    import plot_eca_hparam

    plot_eca_hparam.main()


if __name__ == "__main__":
    main()
