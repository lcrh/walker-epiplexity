"""Figure for the reservoir-criticality appendix experiment.

Panel A: bulk chi vs depth (plain vs normalized) for MLP / 1D-CNN / 2D-CNN.
Panel B: signal survival through depth (perturbation norm per layer).
Panel C: bulk chi vs the normalization gain for four activations.

Run this module directly to regenerate the figure from the saved npz:

    uv run python src/plot/reservoir_criticality.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # runnable as a script

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot import palette as pal


def plot(data: dict, output_dir: str) -> str:
    pal.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.25))
    archs = [("mlp", "MLP", pal.NAT_BLUE), ("cnn1d", "1D-CNN", pal.NAT_ORANGE),
             ("cnn2d", "2D-CNN", pal.NAT_GREEN)]
    depths = data["depths"]

    ax = axes[0]
    for key, lab, c in archs:
        ax.plot(depths, data[f"chi_{key}_plain"], "o--", color=c,
                lw=pal.LW_SECONDARY, ms=3, alpha=0.65)
        ax.plot(depths, data[f"chi_{key}_ln"], "o-", color=c,
                lw=pal.LW_PRIMARY, ms=3, label=lab)
    ax.axhline(1.0, color=pal.INK, ls=":", lw=pal.LW_BASELINE)
    ax.text(depths[-1], 1.0, " edge of chaos", va="bottom", ha="right",
            fontsize=pal.FS_TICK, color=pal.INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths); ax.set_xticklabels(depths)
    ax.set_xlabel("Depth"); ax.set_ylabel(r"Bulk criticality $\chi$")
    ax.set_ylim(0.3, 1.25)
    ax.plot([], [], "o-", color=pal.GREY_500, label="+LayerNorm")
    ax.plot([], [], "o--", color=pal.GREY_500, alpha=0.65, label="plain")
    ax.legend(loc="lower right", ncol=1, handlelength=1.6)

    ax = axes[1]
    layers = np.arange(len(data["surv_plain"]))
    ax.semilogy(layers, data["surv_plain"], "o--", color=pal.NAT_RED,
                lw=pal.LW_SECONDARY, ms=2.5, label="plain (ordered)")
    ax.semilogy(layers, data["surv_ln"], "o-", color=pal.NAT_BLUE,
                lw=pal.LW_PRIMARY, ms=2.5, label="+LayerNorm")
    ax.set_xlabel("Layer"); ax.set_ylabel("Relative signal norm")
    ax.legend(loc="lower left")

    ax = axes[2]
    gains = data["gains"]
    act_colors = [("elu", pal.NAT_BLUE), ("tanh", pal.NAT_ORANGE),
                  ("relu", pal.NAT_GREEN), ("gelu", pal.NAT_PURPLE)]
    for name, c in act_colors:
        ax.plot(gains, data[f"gain_{name}"], "o-", color=c,
                lw=pal.LW_SECONDARY, ms=3, label=name.upper())
    ax.axhline(1.0, color=pal.INK, ls=":", lw=pal.LW_BASELINE)
    ax.axvline(1.0, color=pal.GREY_500, ls=":", lw=pal.LW_BASELINE)
    ax.set_xlabel("LayerNorm gain"); ax.set_ylabel(r"Bulk criticality $\chi$")
    ax.legend(loc="upper left")

    for ax, lab in zip(axes, "abc"):
        pal.despine(ax)
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-22, 6),
                    textcoords="offset points", fontsize=pal.FS_PANEL,
                    fontweight="bold", va="bottom")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "criticality.pdf")
    fig.savefig(path)
    print(f"saved {path}")
    return path


def main() -> None:
    import reservoir_criticality as exp  # experiment constants (output dir)

    data = dict(np.load(os.path.join(exp.OUT, "data.npz")))
    plot(data, exp.OUT)


if __name__ == "__main__":
    main()
