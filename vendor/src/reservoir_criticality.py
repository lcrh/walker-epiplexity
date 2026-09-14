"""Reservoir criticality figure (paper Appendix, app:criticality).

Self-contained generator for ``results/reservoir_criticality/criticality.pdf``
(copied into the paper as ``reservoir_criticality.pdf``): it illustrates why
every feedforward reservoir in this paper places a normalization before each
nonlinearity. A random feedforward stack, left plain, sits in the ordered phase
(per-layer perturbation multiplier ``chi ~ 0.55``) and computes a nearly linear
map; normalizing the pre-activations over the feature/channel axis pins ``chi``
at the edge of chaos (``~1``) independent of architecture, depth, input scale,
and activation.

To compare plain vs normalized and to vary the gain/activation, this script
builds standalone reservoirs that MIRROR ``rc_epiplexity.reservoirs`` (whose core
maps always carry the normalization); the toggle and the gain live here only, for
the illustration. The criticality order parameter is measured by finite
differences -- the growth of an infinitesimal input perturbation per layer.

Panel A: bulk chi vs depth (plain vs normalized) for MLP / 1D-CNN / 2D-CNN.
Panel B: signal survival through depth (perturbation norm per layer).
Panel C: bulk chi vs the normalization gain for four activations.

The figure is drawn by ``plot/reservoir_criticality.py`` (run it directly to
re-plot from the saved npz).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from plot import reservoir_criticality as crit_plot

SEED = 0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "reservoir_criticality")

ACT_TYPES = (nn.ELU, nn.Tanh, nn.ReLU, nn.GELU)
ACTS = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


class PreActNorm(nn.Module):
    """Normalize over the feature/channel axis, then scale by a global gain.

    Mirrors ``rc_epiplexity.reservoirs.PreActNorm`` (gain fixed to 1 in the core);
    the gain is exposed here only to trace the order--chaos axis in panel C.
    """

    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return self.gain * (x - mean) / torch.sqrt(var + 1e-5)


def mlp_reservoir(input_dim, hidden, depth, use_norm, gain=1.0, act=nn.ELU):
    layers, in_d = [], input_dim
    for i in range(depth):
        is_last = i == depth - 1
        layers.append(nn.Linear(in_d, hidden))
        if not is_last:
            if use_norm:
                layers.append(PreActNorm(gain))
            layers.append(act())
        in_d = hidden
    return nn.Sequential(*layers)


def conv_reservoir(dim, in_ch, hidden, depth, kernel, use_norm, gain=1.0, act=nn.ELU):
    Conv = nn.Conv1d if dim == 1 else nn.Conv2d
    layers, c = [], in_ch
    for i in range(depth):
        is_last = i == depth - 1
        k = 1 if is_last else kernel
        layers.append(Conv(c, hidden, k, padding=k // 2,
                           padding_mode="circular" if k > 1 else "zeros"))
        if not is_last:
            if use_norm:
                layers.append(PreActNorm(gain))
            layers.append(act())
        c = hidden
    return nn.Sequential(*layers)


def init_reservoir(reservoir: nn.Sequential) -> nn.Sequential:
    torch.manual_seed(SEED)
    for m in reservoir.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return reservoir.eval()


def make_input(arch):
    g = torch.Generator().manual_seed(SEED)
    if arch == "mlp":
        x = torch.randn(256, 128, generator=g)
    elif arch == "cnn1d":
        x = torch.randn(64, 4, 64, generator=g)
    else:  # cnn2d
        x = torch.randn(64, 1, 32, 32, generator=g)
    return x


def build(arch, depth, use_norm, gain=1.0):
    if arch == "mlp":
        return mlp_reservoir(128, 128, depth, use_norm, gain)
    if arch == "cnn1d":
        return conv_reservoir(1, 4, 64, depth, 3, use_norm, gain)
    return conv_reservoir(2, 1, 64, depth, 3, use_norm, gain)


@torch.no_grad()
def perturbation_norms(reservoir, x, eps=1e-3, n_dirs=16):
    """Mean perturbation norm at the input and after each activation."""
    g = torch.Generator().manual_seed(SEED + 3)
    flat = lambda t: t.reshape(t.shape[0], -1)
    profiles = []
    for _ in range(n_dirs):
        u = torch.randn(x.shape, generator=g)
        u = u / flat(u).norm(dim=1).reshape(-1, *([1] * (x.dim() - 1)))
        h, hp = x, x + eps * u
        norms = [flat(hp - h).norm(dim=1)]
        for m in reservoir:
            h, hp = m(h), m(hp)
            if isinstance(m, ACT_TYPES):
                norms.append(flat(hp - h).norm(dim=1))
        profiles.append(torch.stack(norms))
    return torch.stack(profiles).mean(0)  # (n_act + 1, batch)


def bulk_chi(reservoir, x):
    # Geometric mean of the interior per-layer multipliers (skip the first step,
    # which also carries the input->hidden dimension change). This bulk value
    # determines whether signal vanishes or explodes through depth.
    P = perturbation_norms(reservoir, x)
    ratios = (P[1:] / (P[:-1] + 1e-30)).mean(1)
    interior = ratios[1:] if ratios.numel() > 1 else ratios
    return float(torch.exp(torch.log(interior + 1e-30).mean()))


def signal_survival(reservoir, x):
    P = perturbation_norms(reservoir, x).mean(1)  # (n_act + 1,)
    return (P / P[1]).numpy()  # normalize to the first activation output


def compute():
    archs = ["mlp", "cnn1d", "cnn2d"]
    depths = [4, 8, 16, 32]
    data = {"depths": np.array(depths)}
    for arch in archs:
        p, l = [], []
        for d in depths:
            x = make_input(arch)
            p.append(bulk_chi(init_reservoir(build(arch, d, False)), x))
            l.append(bulk_chi(init_reservoir(build(arch, d, True)), x))
        data[f"chi_{arch}_plain"] = np.array(p)
        data[f"chi_{arch}_ln"] = np.array(l)

    x = make_input("mlp")
    data["surv_plain"] = signal_survival(init_reservoir(build("mlp", 32, False)), x)
    data["surv_ln"] = signal_survival(init_reservoir(build("mlp", 32, True)), x)

    gains = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    data["gains"] = gains
    for name, act in ACTS.items():
        data[f"gain_{name}"] = np.array(
            [bulk_chi(init_reservoir(mlp_reservoir(128, 128, 16, True, float(g), act)), x)
             for g in gains])

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "data.npz"), **data)
    return data


if __name__ == "__main__":
    crit_plot.plot(compute(), OUT)
