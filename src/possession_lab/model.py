"""Stable adapter around the untouched organiser baseline inference functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from possession_lab.config import ModelConfig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAME_COUNT = 199_319
LABEL_SET = frozenset({"H", "A", "D", "X"})


def set_ungraded_to_x(labels: NDArray[np.str_]) -> NDArray[np.str_]:
    """Mark only published ungraded intervals as ``X`` without altering graded frames.

    Args:
        labels: Predicted tactical labels for 199,319 frames.

    Returns:
        Copied labels with the three ungraded windows set to ``X``.

    Raises:
        ValueError: If frame count differs from the official contract.
    """
    if len(labels) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames, found {len(labels)}")
    updated = labels.copy()
    updated[:11_185] = "X"
    updated[90_652:110_808] = "X"
    updated[190_198:] = "X"
    return updated


def predict(config: ModelConfig) -> NDArray[np.str_]:
    """Predict the full tactical match using the official nearest-player implementation.

    Args:
        config: Validated experiment and smoothing settings.

    Returns:
        Exactly 199,319 possession labels ``H``, ``A``, ``D`` or ``X``.

    Raises:
        ValueError: If the returned prediction violates contract invariants.
    """
    from official_documentation.starter.baseline import build, smooth  # pylint: disable=import-outside-toplevel

    codes = build()
    labels: NDArray[np.str_] = smooth(codes, config.smoothing_half_window)
    if len(labels) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames, got {len(labels)}")
    if config.apply_ungraded_x:
        labels = set_ungraded_to_x(labels)
    if not set(np.unique(labels)).issubset(LABEL_SET):
        raise ValueError("Invalid labels generated")
    return labels
