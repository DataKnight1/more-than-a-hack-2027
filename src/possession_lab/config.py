"""Validated TOML configuration for deterministic possession predictions."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """A minimal and deterministic candidate configuration.

    Attributes:
        family: Registered inference family.
        smoothing_half_window: Half-width, in 25-fps tactical frames.
        apply_ungraded_x: Whether known ungraded intervals become ``X``.
        seed: Reproducibility seed, reserved for future stochastic models.
    """

    family: str = "nearest_player_majority"
    smoothing_half_window: int = 50
    apply_ungraded_x: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject unsupported options rather than silently changing the baseline."""
        if self.family != "nearest_player_majority":
            raise ValueError(f"Unsupported model family: {self.family}")
        if (
            not isinstance(self.smoothing_half_window, int)
            or isinstance(self.smoothing_half_window, bool)
            or not 0 <= self.smoothing_half_window <= 250
        ):
            raise ValueError("smoothing_half_window must be an integer in 0..250")
        if not isinstance(self.apply_ungraded_x, bool):
            raise ValueError("apply_ungraded_x must be a boolean")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in 0..2**32-1")


def load_config(path: Path) -> ModelConfig:
    """Read a TOML model configuration from a local path.

    Args:
        path: TOML file with `[model]` and optional `[experiment]` sections.

    Returns:
        Validated immutable model configuration.

    Raises:
        ValueError: When the schema or a setting is invalid.
    """
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("Missing [model] table")
    allowed = {"family", "smoothing_half_window", "apply_ungraded_x"}
    unexpected = set(model) - allowed
    if unexpected:
        raise ValueError(f"Unsupported model keys: {sorted(unexpected)}")
    experiment = raw.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("[experiment] must be a TOML table")
    return ModelConfig(
        family=model.get("family", "nearest_player_majority"),
        smoothing_half_window=model.get("smoothing_half_window", 50),
        apply_ungraded_x=model.get("apply_ungraded_x", False),
        seed=experiment.get("seed", 42),
    )
