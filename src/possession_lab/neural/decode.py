"""Turn per-frame class probabilities into spell-consistent labels.

``spellF1`` rewards whole possessions, so flicker is expensive. Viterbi decoding with a constant
penalty per label change finds the label sequence maximising ``sum log p(label_t) - penalty *
n_changes``: a principled replacement for majority smoothing. Penalties are selected on external
validation matches only.

With a transition head (N012+) the log-probabilities carry a fourth column, ``log p(change)``.
The team-switch cost then varies per frame: ``max(0, team_switch - transition_gain * logit_t)``
(cheaper where the head sees a possession change, dearer where it is confident there is none).
``transition_gain = 0`` reproduces the constant-penalty decoder exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import X


@dataclass(frozen=True)
class DecodeConfig:
    """Switching penalties in nats: between teams, and into / out of dead ball."""

    team_switch: float = 8.0
    dead_switch: float = 4.0
    transition_gain: float = 0.0

    def matrix(self) -> NDArray[np.float64]:
        """3x3 transition cost matrix over (H, A, D)."""
        t, d = self.team_switch, self.dead_switch
        return np.array([[0.0, t, d], [t, 0.0, d], [d, d, 0.0]])


def team_costs(log_change: NDArray[np.float32], cfg: DecodeConfig) -> NDArray[np.float64]:
    """Per-frame team-switch cost from ``log p(change)`` (see module docstring)."""
    lc = np.clip(log_change.astype(np.float64), -30.0, -1e-7)
    logit = lc - np.log(-np.expm1(lc))
    return cast(NDArray[np.float64], np.maximum(0.0, cfg.team_switch - cfg.transition_gain * logit))


def viterbi(log_probs: NDArray[np.float32], cfg: DecodeConfig) -> NDArray[np.int8]:
    """Most likely path of ``[T, 3]`` (or ``[T, 4]`` with a change column) under switching costs."""
    cost = cfg.matrix()
    n = len(log_probs)
    varying = log_probs.shape[1] > 3 and cfg.transition_gain != 0.0
    team = team_costs(log_probs[:, 3], cfg) if varying else None
    back = np.empty((n, 3), dtype=np.int8)
    score = log_probs[0, :3].astype(np.float64).copy()
    lp = log_probs[:, :3].astype(np.float64)
    for t in range(1, n):
        if team is not None:
            cost[0, 1] = cost[1, 0] = team[t]
        cand = score[:, None] - cost  # from i to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + lp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match(
    log_probs: NDArray[np.float32], period: NDArray[np.int8], cfg: DecodeConfig
) -> NDArray[np.int8]:
    """Decode each period independently; gap frames (period 0) become ``X``."""
    out = np.full(len(period), X, np.int8)
    for p in np.unique(period):
        if p == 0:
            continue
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi(log_probs[idx], cfg)
    return out
