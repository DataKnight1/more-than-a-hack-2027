"""Probability-averaging ensemble used by scripts/neural/evaluate.py (D-022)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_average_log_probs_is_probability_mean() -> None:
    """Verify ensemble log probabilities represent the arithmetic mean of model probabilities."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "scripts" / "neural"))
    from evaluate import average_log_probs

    a = np.log(np.array([[0.7, 0.2, 0.1, 0.9]], np.float32))
    b = np.log(np.array([[0.1, 0.6, 0.3, 0.1]], np.float32))
    out = average_log_probs([a, b])
    np.testing.assert_allclose(np.exp(out), [[0.4, 0.4, 0.2, 0.5]], rtol=1e-5)
    assert average_log_probs([a]) is a
    c = np.log(np.array([[0.5, 0.25, 0.25]], np.float32))  # no transition column
    assert average_log_probs([a, c]).shape == (1, 3)
