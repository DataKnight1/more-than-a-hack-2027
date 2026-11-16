"""Organiser-equivalent nearest-player baseline on canonical arrays, for external comparison.

Reproduces ``official_documentation/starter/baseline.py`` semantics: the nearest tracked object
to the visible ball gives its team the frame, no ball -> nobody, centred majority vote, ties to
home. This lets C000 be scored
on exactly the same external holdouts and degradations as the neural candidates. The smoothing is
the organiser's own ``smooth`` function, imported unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import A, D, H

ROOT = Path(__file__).resolve().parents[3]
NOBODY = -1


def nearest_player_codes(
    ball: NDArray[np.float32], home: NDArray[np.float32], away: NDArray[np.float32]
) -> NDArray[np.int8]:
    """Per-frame 0 (home) / 1 (away) / -1 (no ball or no player) by nearest object to the ball."""
    players = np.concatenate([home, away], axis=1)
    team = np.concatenate([np.zeros(home.shape[1]), np.ones(away.shape[1])]).astype(np.int8)
    diff = players - ball[:, None, :2]
    dist = np.where(np.isnan(diff[..., 0]), np.inf, np.hypot(diff[..., 0], diff[..., 1]))
    best = np.argmin(dist, axis=1)
    out = team[best].astype(np.int8)
    unseen = np.isnan(ball[:, 0]) | ~np.isfinite(dist[np.arange(len(best)), best])
    out[unseen] = NOBODY
    return cast(NDArray[np.int8], out)


def baseline_codes(
    ball: NDArray[np.float32],
    home: NDArray[np.float32],
    away: NDArray[np.float32],
    window: int = 50,
) -> NDArray[np.int8]:
    """Official-style smoothed baseline mapped to official label codes (H/A/D)."""
    letters = load_official_smoother()(nearest_player_codes(ball, home, away), window)
    codes = np.full(len(letters), D, np.int8)
    codes[letters == "H"] = H
    codes[letters == "A"] = A
    return cast(NDArray[np.int8], codes)


def load_official_smoother() -> Callable[..., NDArray[np.str_]]:
    """Load the unchanged organizer majority smoother from its archived starter script.

    Returns:
        The organizer's ``smooth(codes, window)`` function.

    Raises:
        ImportError: If the archived reference script or its loader is unavailable.
    """
    script = ROOT / "official_documentation" / "starter" / "baseline.py"
    spec = importlib.util.spec_from_file_location("possession_lab._official_baseline", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official baseline module from {script}")
    module = importlib.util.module_from_spec(spec)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return cast(Callable[..., NDArray[np.str_]], module.smooth)
