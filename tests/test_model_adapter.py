"""Synthetic model-adapter integration checks; no licensed inputs or labels."""

from __future__ import annotations

import sys
import types

import numpy as np

from possession_lab.config import ModelConfig
from possession_lab.model import FRAME_COUNT, predict


def test_official_adapter_shape_without_match(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A stub of the official baseline can drive predictions with no downloaded dataset."""
    synthetic_baseline = types.ModuleType("official_documentation.starter.baseline")
    synthetic_baseline.build = lambda: np.zeros(FRAME_COUNT, dtype=np.int8)  # type: ignore[attr-defined]
    synthetic_baseline.smooth = (  # type: ignore[attr-defined]
        lambda codes, window: np.full(len(codes), "H", dtype="<U1")
    )
    monkeypatch.setitem(sys.modules, "official_documentation.starter.baseline", synthetic_baseline)
    result = predict(ModelConfig(smoothing_half_window=50))
    assert result.shape == (FRAME_COUNT,)
    assert set(np.unique(result)) == {"H"}
