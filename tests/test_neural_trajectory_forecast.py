"""Unit tests for N019 confidence-aware ball trajectory forecasting and completion."""

from __future__ import annotations

import numpy as np
import torch

from possession_lab.neural.features import (
    BALL_FEATURES,
    BALL_FEATURES_FORECAST,
    build_tokens,
)
from possession_lab.neural.model import ModelConfig, SetTemporalNet
from possession_lab.neural.trajectory_forecast import (
    benchmark_trajectory_completion,
    forecast_ball_trajectory,
)


def test_forecast_ball_trajectory_clean() -> None:
    """When all frames are observed, forecast returns exact observed coordinates with unit confidence."""
    t_len = 100
    xy = np.zeros((t_len, 2), dtype=np.float32)
    xy[:, 0] = np.linspace(-20.0, 20.0, t_len)
    xy[:, 1] = np.sin(np.linspace(0, np.pi, t_len)) * 10.0
    seen = np.ones(t_len, dtype=bool)

    fc_xy, fc_conf, fc_imputed = forecast_ball_trajectory(xy, seen)

    assert np.allclose(fc_xy, xy)
    assert np.all(fc_conf == 1.0)
    assert np.all(fc_imputed == 0.0)


def test_forecast_ball_trajectory_gap_interpolation() -> None:
    """Interior missing gap is smoothly interpolated with decaying confidence away from endpoints."""
    t_len = 50
    xy = np.zeros((t_len, 2), dtype=np.float32)
    xy[:, 0] = np.linspace(-25.0, 25.0, t_len)
    seen = np.ones(t_len, dtype=bool)

    # Introduce a 10-frame gap between frames 20 and 30
    gap_slice = slice(20, 30)
    seen[gap_slice] = False
    xy[gap_slice] = np.nan

    fc_xy, fc_conf, fc_imputed = forecast_ball_trajectory(xy, seen)

    # Imputed flag should be 1 inside the gap and 0 elsewhere
    assert np.all(fc_imputed[gap_slice] == 1.0)
    assert np.all(fc_imputed[:20] == 0.0)
    assert np.all(fc_imputed[30:] == 0.0)

    # Trajectory must have no NaNs and stay strictly monotonic along x
    assert not np.any(np.isnan(fc_xy))
    assert np.all(np.diff(fc_xy[:, 0]) > 0.0)

    # Confidence must be < 1.0 in the gap and minimum near the center of the gap
    assert np.all(fc_conf[gap_slice] < 1.0)
    assert fc_conf[25, 0] <= fc_conf[20, 0]
    assert fc_conf[25, 0] <= fc_conf[29, 0]


def test_forecast_ball_trajectory_extrapolation() -> None:
    """Leading and trailing gaps extrapolate with drag damping and pitch bounds clamping."""
    t_len = 60
    xy = np.zeros((t_len, 2), dtype=np.float32)
    xy[:, 0] = 10.0
    seen = np.zeros(t_len, dtype=bool)
    # Only observed between frame 10 and 40
    seen[10:40] = True

    fc_xy, fc_conf, fc_imputed = forecast_ball_trajectory(xy, seen)

    assert not np.any(np.isnan(fc_xy))
    assert np.all(fc_imputed[:10] == 1.0)
    assert np.all(fc_imputed[40:] == 1.0)
    # Coordinates must stay within pitch bounds
    assert np.all(np.abs(fc_xy[:, 0]) <= 52.5)
    assert np.all(np.abs(fc_xy[:, 1]) <= 34.0)


def test_benchmark_trajectory_completion() -> None:
    """Benchmark utility computes MAE and RMSE across synthetic gap lengths."""
    t_len = 200
    xy = np.zeros((t_len, 2), dtype=np.float32)
    # Smooth quadratic trajectory
    t = np.linspace(0, 4.0, t_len)
    xy[:, 0] = -30.0 + 15.0 * t - 1.5 * t**2
    xy[:, 1] = 5.0 * np.sin(t)

    results = benchmark_trajectory_completion(xy, gap_lengths=(5, 10), n_trials=5, seed=123)

    assert "5" in results
    assert "10" in results
    assert "forecast_mae" in results["5"]
    assert "fill_mae" in results["5"]
    assert results["5"]["forecast_mae"] >= 0.0


def test_build_tokens_with_forecast() -> None:
    """build_tokens with ball_forecast=True appends 4 channels while keeping original 9 identical."""
    n = 40
    rng = np.random.default_rng(42)
    ball = np.zeros((n, 3), dtype=np.float32)
    ball[:, 0] = np.linspace(-10, 10, n)
    ball[15:20] = np.nan
    home = rng.normal(0, 5, (n, 11, 2)).astype(np.float32)
    away = rng.normal(0, 5, (n, 11, 2)).astype(np.float32)

    base_ball, base_feat, base_mask = build_tokens(ball, home, away, ball_forecast=False)
    fc_ball, fc_feat, fc_mask = build_tokens(ball, home, away, ball_forecast=True)

    assert base_ball.shape == (n, BALL_FEATURES)  # 9
    assert fc_ball.shape == (n, BALL_FEATURES_FORECAST)  # 13

    # Channels 0..8 must be identical
    assert np.allclose(base_ball, fc_ball[:, :BALL_FEATURES])
    assert np.allclose(base_feat, fc_feat)
    assert np.array_equal(base_mask, fc_mask)


def test_set_temporal_net_ball_features_13() -> None:
    """SetTemporalNet forward pass works with ball_features=13."""
    b, t, k = 2, 64, 15
    cfg = ModelConfig(
        d_model=64,
        spatial_layers=2,
        temporal_layers=2,
        heads=4,
        player_features=16,
        ball_features=13,
        context_features=62,
        transition_head=True,
    )
    model = SetTemporalNet(cfg)

    ball = torch.randn(b, t, 13)
    players = torch.randn(b, t, k, 16)
    mask = torch.ones(b, t, k, dtype=torch.bool)
    context = torch.randn(b, t, 62)

    out = model(ball, players, mask, context)
    # 3 class logits + 1 transition logit = 4
    assert out.shape == (b, t, 4)
    assert not torch.isnan(out).any()


def test_model_config_backward_compatibility() -> None:
    """Default ModelConfig retains ball_features=9 for existing checkpoints."""
    default_cfg = ModelConfig()
    assert default_cfg.ball_features == 9
    assert default_cfg.player_features == 8
    assert default_cfg.context_features == 0
