"""Synthetic tests for individual / marking / ball-touch features (feature version 3)."""

from __future__ import annotations

import numpy as np
import pytest

from possession_lab.neural.individual_features import (
    GLOBAL_IND_FEATURES,
    N_IND,
    N_TEAM_IND,
    PLAYER_IND_FEATURES,
    TEAM_IND_FEATURES,
    individual_features,
)


def _scene(n: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ball rolls +x with home player 0 until frame 50, then is struck back -x by away player 0."""
    ball = np.zeros((n, 3), np.float32)
    t = np.arange(n)
    ball[:, 0] = np.where(t < 50, -10 + 0.2 * t, -10 + 0.2 * 50 - 0.6 * (t - 50))
    home = np.full((n, 11, 2), np.nan, np.float32)
    away = np.full((n, 11, 2), np.nan, np.float32)
    for i in range(11):
        home[:, i] = (-40 + 5 * i, -25 + 5 * i)
        away[:, i] = (40 - 5 * i, -25 + 5 * i)
    home[:, 0] = np.stack([ball[:, 0] - 0.5, np.zeros(n)], 1)
    home[45:, 0] = (-1.5, 0.0)  # carrier stops short; the ball runs on to the waiting opponent
    away[:, 0] = (0.4, 0.3)  # waits at the strike point
    return ball, home, away


def test_shapes_and_finite() -> None:
    """Verify individual feature tensors have fixed shapes and finite values."""
    ball, home, away = _scene()
    ctx, per = individual_features(ball, home, away, np.ones(100, np.float32))
    assert ctx.shape == (100, N_IND) and per.shape == (100, 22, PLAYER_IND_FEATURES)
    assert np.isfinite(ctx).all() and np.isfinite(per).all()


def test_touch_team_change_at_strike() -> None:
    """Verify a detected ball touch records the controlling-team change."""
    ball, home, away = _scene()
    ctx, _ = individual_features(ball, home, away, np.ones(100, np.float32), scaled=False)
    g = ctx[:, 2 * N_TEAM_IND :]
    f = GLOBAL_IND_FEATURES.index
    touches = np.flatnonzero(g[:, f("touch")] > 0)
    assert len(touches) and np.any(np.abs(touches - 50) <= 3)
    assert g[60, f("last_touch_team")] == -1.0  # away struck it
    assert np.any(g[45:56, f("touch_team_change")] > 0)


def test_missing_ball_and_players_are_safe() -> None:
    """Verify missing ball and player tracks do not produce invalid feature values."""
    ball, home, away = _scene()
    ball[20:80] = np.nan
    home[30:40] = np.nan
    ctx, per = individual_features(ball, home, away, np.ones(100, np.float32))
    assert np.isfinite(ctx).all() and np.isfinite(per).all()


def test_pressure_and_marking() -> None:
    """Verify pressure and marking features reflect nearby opposing players."""
    ball, home, away = _scene()
    ctx, _ = individual_features(ball, home, away, np.ones(100, np.float32), scaled=False)
    f = TEAM_IND_FEATURES.index
    hb = ctx[:, :N_TEAM_IND]
    # just before the strike the home carrier is < 2.5 m from the waiting away player
    assert hb[46, f("press_on_nearest")] < 2.5
    assert hb[46, f("opp_within5_of_nearest")] >= 1
    assert hb[10, f("press_on_nearest")] > 5.0


def test_v3_inputs_and_model_forward() -> None:
    """Verify version 3 features pass through the model with the expected output shape."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.features import build_inputs, feature_dims, version_of
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    ball, home, away = _scene()
    pdim, cdim = feature_dims(3)
    assert version_of(cdim) == 3 and version_of(feature_dims(2)[1]) == 2 and version_of(0) == 1
    bf, pf, pm, ctx = build_inputs(ball, home, away, 30, np.ones(100, np.float32), version=3)
    assert pf.shape == (100, 30, pdim) and ctx is not None and ctx.shape == (100, cdim)
    model = SetTemporalNet(
        ModelConfig(
            d_model=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=2,
            player_features=pdim,
            context_features=cdim,
        )
    )
    out = model(*(torch.from_numpy(x[None]) for x in (bf, pf, pm, ctx)))
    assert out.shape == (1, 100, 3) and torch.isfinite(out).all()
