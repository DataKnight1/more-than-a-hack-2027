"""Synthetic configuration checks; never require the supplied match."""

from __future__ import annotations

import pytest

from possession_lab.config import ModelConfig, load_config


def test_default_baseline_config() -> None:
    """Ensure documented configuration reproduces the official window parameter."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs/models/baseline_smooth50.toml"
    config = load_config(path)
    assert config.family == "nearest_player_majority"
    assert config.smoothing_half_window == 50
    assert config.apply_ungraded_x is False


@pytest.mark.parametrize("bad", [-1, 251, 2.5, True, False])
def test_window_validation(bad: float) -> None:
    """Reject invalid smoothing parameters before any data is loaded."""
    with pytest.raises(ValueError):
        ModelConfig(smoothing_half_window=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 2**32, 2.5, True, False])
def test_seed_validation(bad: float) -> None:
    """Reject booleans and values outside the declared integer seed contract."""
    with pytest.raises(ValueError):
        ModelConfig(seed=bad)  # type: ignore[arg-type]


def test_unknown_model_rejected() -> None:
    """Do not silently fall back to nearest-player when model name is wrong."""
    with pytest.raises(ValueError):
        ModelConfig(family="hypothetical_transformer")
