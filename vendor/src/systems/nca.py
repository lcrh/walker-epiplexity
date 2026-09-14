"""Local 1D neural cellular automaton on a unit-vector state.

Trainable system only: built from plain ``torch.nn`` and never imports from
``rc_epiplexity`` (the frozen scorer must stay a separate observer).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

STATE_CHANNELS = 2


class LocalNCA1D(nn.Module):
    """Local 1D NCA on a two-channel unit-vector state.

    Each lattice site holds a 2-vector constrained to unit norm: the state's
    scale is fixed (only the per-site direction carries information), so the
    epiplexity score cannot be inflated by growing magnitude and no L2 penalty is
    needed. The default update is direct, ``x_{t+1} = norm(g_theta(x_t))``; the
    residual form ``x_{t+1} = norm(x_t + g_theta(x_t))`` is the appendix variant.

    ``g_theta`` is a local convolutional rule: a radius-2 perception convolution
    (circular padding keeps the lattice periodic) followed by two BatchNorm-GELU
    1x1 blocks. BatchNorm's statistics are per channel, so once frozen at
    evaluation the rule stays translation invariant and the update remains a
    genuine cellular-automaton rule.
    """

    def __init__(self, update_mode: str = "direct"):
        super().__init__()
        if update_mode not in {"direct", "residual"}:
            raise ValueError(f"unknown update_mode={update_mode!r}")
        self.update_mode = update_mode
        self.net = nn.Sequential(
            nn.CircularPad1d(2),
            nn.Conv1d(STATE_CHANNELS, 128, kernel_size=5, bias=True),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, STATE_CHANNELS, kernel_size=1, bias=True),
        )
        self.initialization()

    def initialization(self):
        """Small random kernels so the initial rule carries little epiplexity."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.03)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.net(x)
        if self.update_mode == "residual":
            out = x + update
        else:
            out = update
        return out / out.norm(dim=-2, keepdim=True).clamp_min(1e-8)


def random_unit_state(
    batch: int, width: int, device: str | torch.device, state_channels: int = 2
) -> torch.Tensor:
    """Random two-channel state with each site's 2-vector on the unit circle."""
    state = torch.empty(batch, state_channels, width, device=device).uniform_(-1.0, 1.0)
    return state / state.norm(dim=1, keepdim=True).clamp_min(1e-8)


def evolve(
    nca: LocalNCA1D, state: torch.Tensor, steps: int, track_grad: bool
) -> torch.Tensor:
    if track_grad:
        for _ in range(steps):
            state = nca(state)
        return state
    with torch.no_grad():
        for _ in range(steps):
            state = nca(state)
    return state


def evolve_stacked_window(
    nca: LocalNCA1D,
    state: torch.Tensor,
    tau_min: int,
    tau_max: int,
) -> torch.Tensor:
    """Roll out a differentiable target window, stacking states tau_min..tau_max.

    Returns shape ``(B, STATE_CHANNELS * (tau_max - tau_min + 1), W)``: the target states
    concatenated along the channel axis, so the reservoir gets one readout column
    per (step, channel) of the whole prediction window.
    """
    states = []
    for tau in range(1, tau_max + 1):
        state = nca(state)
        if tau >= tau_min:
            states.append(state)
    return torch.cat(states, dim=1)


def state_field(state: torch.Tensor) -> np.ndarray:
    """Per-site angle of the unit 2-vector state, in [-1, 1] (= phase / pi)."""
    s = state.detach().cpu().numpy()[0]  # (STATE_CHANNELS, width)
    return np.arctan2(s[1], s[0]) / np.pi


def rollout_trace(nca: LocalNCA1D, state: torch.Tensor, steps: int) -> np.ndarray:
    """Space-time field of a rollout: shape ``(steps + 1, width)``."""
    nca.eval()
    trace = [state_field(state)]
    with torch.no_grad():
        for _ in range(steps):
            state = nca(state)
            trace.append(state_field(state))
    return np.stack(trace, axis=0)
