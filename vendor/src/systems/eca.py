"""Elementary cellular automaton evolution and stacked-horizon sampling.

Data generation follows the original epiplexity reference implementation:
https://github.com/shikaiqiu/epiplexity
(reference commit inspected: 3aa12a1be6a413fe9eaa41374a6a46a4a0d4e100).
"""

from __future__ import annotations

import numpy as np
import torch

AUTHOR_ECA_WIDTH = 64
AUTHOR_ECA_BURNIN_STEPS = 1000


def int2bits(rule: int, bits: int = 8) -> np.ndarray:
    """Convert a Wolfram rule integer to the upstream LSB-first lookup table."""
    return np.array(list(map(int, bin(rule)[2:].zfill(bits)[::-1])), dtype=np.uint8)


def evolve_eca_np(initial_state: np.ndarray, steps: int, rule: int) -> np.ndarray:
    """Evolve an elementary cellular automaton with periodic boundaries."""
    lookup = int2bits(rule, 8)
    state = initial_state
    for _ in range(steps):
        left = np.roll(state, 1, axis=-1)
        right = np.roll(state, -1, axis=-1)
        neighborhood = (left << 2) + (state << 1) + right
        state = lookup[neighborhood]
    return state


def eca_stacked_pair(
    rule: int, num_samples: int, width: int, burnin: int, tau_max: int, seed: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Burned-in state ``x`` and its next ``tau_max`` one-step states, stacked.

    Returns ``x`` of shape ``(B, 1, W)`` and ``y`` of shape ``(B, tau_max, W)``
    where channel ``k`` holds ``F^{k+1}(x)``. ``RCEpiplexity1D.epiplexity`` reads
    the multi-channel target directly, one readout column per horizon.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(num_samples, width), dtype=np.uint8)
    if burnin > 0:
        x = evolve_eca_np(x, burnin, rule)
    futures = []
    state = x
    for _ in range(tau_max):
        state = evolve_eca_np(state, 1, rule)
        futures.append(state)
    y = np.stack(futures, axis=1)  # (B, tau_max, W)
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    return x_t, y_t
