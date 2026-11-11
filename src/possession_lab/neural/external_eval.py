"""Score predictions on *external* labelled matches with the unchanged official metric code.

``evaluation/possession_metrics.py`` is imported by path and never modified, so spell matching,
IoU thresholds and the X forward-fill are exactly the organiser's. This module must never be
pointed at the challenge ground truth: it is for other-game holdouts only.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = ROOT / "evaluation" / "possession_metrics.py"


@cache
def official_metrics() -> ModuleType:
    """Load the organiser metric module from its original location (read-only import)."""
    spec = importlib.util.spec_from_file_location("official_possession_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADLINE_KEYS = (
    "spell_f1_mean",
    "macro_f1",
    "n_gt_spells",
    "n_pred_spells",
    "mean_spell_duration_s",
    "gt_mean_spell_duration_s",
    "alive_dead_f1",
    "team_acc_both_alive",
    "fragmentation",
)


def score(gt_codes: NDArray[np.int8], pred_codes: NDArray[np.int8]) -> dict[str, Any]:
    """Run the official ``evaluate`` on one external match and keep the headline numbers.

    Args:
        gt_codes: External ground-truth codes (0..3), ``X`` on gap/ungraded frames.
        pred_codes: Predicted codes of the same length.

    Returns:
        Dict with the headline keys plus transition F1 at both official tolerances.
    """
    result = official_metrics().evaluate(gt_codes.astype(np.int8), pred_codes.astype(np.int8))
    out: dict[str, Any] = {key: result[key] for key in HEADLINE_KEYS}
    for tol, scores in result["transitions"].items():
        out[f"transition_f1_tol{tol}"] = scores["f1"]
    return out


def pooled(per_match: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Unweighted mean over matches of the headline numeric metrics."""
    keys = [
        k
        for k in next(iter(per_match.values()))
        if isinstance(next(iter(per_match.values()))[k], (int, float))
    ]
    return {k: float(np.mean([m[k] for m in per_match.values()])) for k in keys}
