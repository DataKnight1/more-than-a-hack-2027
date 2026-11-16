"""Synthetic tests for engineered team features and the Metrica event-to-label algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from possession_lab.neural.canonical import A, D, H, X
from possession_lab.neural.metrica import Event, events_to_labels
from possession_lab.neural.team_features import (
    N_CONTEXT,
    N_TEAM,
    TEAM_FEATURES,
    attack_direction,
    context_features,
    last_known,
)


def _ev(team: str, kind: str, start: int, end: int | None = None, subtype: str = "") -> Event:
    """Construct one synthetic Metrica event with the requested type and interval."""
    return Event(team, kind, subtype, 1, start, end if end is not None else start)


def test_events_to_labels_possession_and_dead_ball() -> None:
    """Verify event possession and stoppage intervals map to H/A/D labels."""
    frames = np.arange(1, 101)
    events = [
        _ev("H", "SET PIECE", 1),
        _ev("H", "PASS", 5, 10),
        _ev("A", "RECOVERY", 30),
        _ev("A", "BALL OUT", 50),
        _ev("H", "CHALLENGE", 55),  # ignored
        _ev("H", "SET PIECE", 70),
    ]
    lab = events_to_labels(events, frames)
    assert lab[0] == H and lab[28] == H  # frames 1..29
    assert lab[29] == A and lab[48] == A  # frames 30..49
    assert np.all(lab[49:69] == D)  # frames 50..69 dead until the restart
    assert np.all(lab[69:] == H)


def test_events_to_labels_shot_out_and_no_events() -> None:
    """Verify out-of-play shots and empty event lists receive safe labels."""
    frames = np.arange(1, 41)
    lab = events_to_labels(
        [_ev("A", "PASS", 1), _ev("A", "SHOT", 10, 15, "OFF TARGET-OUT")], frames
    )
    assert np.all(lab[:14] == A) and np.all(lab[14:] == D)  # ball out from the shot's end frame 15
    assert np.all(events_to_labels([], frames) == X)


def _scene(n: int = 50) -> tuple[np.ndarray, ...]:
    """Home in -x half running forward towards a ball it controls; away retreating, compact."""
    t = np.arange(n, dtype=np.float32) / 25.0
    ball = np.zeros((n, 3), np.float32)
    ball[:, 0] = -10 + 5 * t
    home = np.zeros((n, 11, 2), np.float32)
    away = np.zeros((n, 11, 2), np.float32)
    for i in range(11):
        home[:, i, 0] = -30 + 2 * i + 5 * t
        home[:, i, 1] = -25 + 5 * i
        away[:, i, 0] = 20 + i - 1 * t
        away[:, i, 1] = -10 + 2 * i
    home[:, 0, 0] = ball[:, 0] - 0.5  # carrier next to the ball
    home[:, 0, 1] = 0.0
    return ball, home, away


def test_attack_direction_from_mean_positions() -> None:
    """Verify attack direction is inferred from mean team positions per half."""
    _, home, away = _scene()
    period = np.ones(50, np.int8)
    assert np.all(attack_direction(home, away, period) == 1.0)
    assert np.all(attack_direction(-home, -away, period) == -1.0)


def test_context_features_capture_possession_behaviour() -> None:
    """Verify team context features reflect collective possession movement."""
    ball, home, away = _scene()
    direction = np.ones(50, np.float32)
    ctx = context_features(ball, home, away, direction, scaled=False)
    assert ctx.shape == (50, N_CONTEXT)
    hb, ab = ctx[:, :N_TEAM], ctx[:, N_TEAM : 2 * N_TEAM]
    f = TEAM_FEATURES.index
    assert hb[25, f("nearest_ball_dist")] < ab[25, f("nearest_ball_dist")]
    assert hb[25, f("centroid_vel_fwd")] == pytest.approx(5.0, abs=1e-3)
    assert ab[25, f("centroid_vel_fwd")] == pytest.approx(
        1.0, abs=1e-3
    )  # away attacks -x, retreats +x
    assert hb[25, f("spread_width")] > ab[25, f("spread_width")]
    assert np.isfinite(context_features(ball, home, away, direction)).all()


def test_last_known_fills_missing_ball() -> None:
    """Verify missing ball coordinates use the latest available location."""
    xy = np.array(
        [[np.nan, np.nan], [1, 1], [np.nan, np.nan], [np.nan, np.nan], [3, 3]], np.float32
    )
    filled, age = last_known(xy)
    assert filled[0].tolist() == [1, 1] and filled[3].tolist() == [1, 1]
    assert age[3] == pytest.approx(2 / 25)


def test_symmetry_dir_keeps_direction_consistent() -> None:
    """Verify team symmetry retains consistent direction-aware features."""
    from possession_lab.neural.augment import symmetry_dir
    from possession_lab.neural.team_features import attack_direction

    ball, home, away = _scene()
    labels = np.full(50, H, np.int8)
    period = np.ones(50, np.int8)
    direction = attack_direction(home, away, period)
    for seed in range(16):
        b, h, a, lab, d = symmetry_dir(
            np.random.default_rng(seed), ball, home, away, labels, direction
        )
        # recomputing the direction on the transformed scene must agree with the transformed one
        assert np.all(attack_direction(h, a, period) == d)
        swapped = np.ptp(h[0, :, 1]) < 30  # scene: home spans 50 m of width, away 20 m
        assert (lab[0] == A) == swapped


def test_build_inputs_v2_shapes_and_model_forward() -> None:
    """Verify feature version 2 inputs match the model’s expected tensor layout."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.features import PLAYER_FEATURES_V2, build_inputs
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    ball, home, away = _scene()
    bf, pf, pm, ctx = build_inputs(ball, home, away, 30, np.ones(50, np.float32), version=2)
    assert (
        pf.shape == (50, 30, PLAYER_FEATURES_V2)
        and ctx is not None
        and ctx.shape == (50, N_CONTEXT)
    )
    assert np.isfinite(pf).all() and np.isfinite(ctx).all()
    model = SetTemporalNet(
        ModelConfig(
            d_model=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=2,
            player_features=PLAYER_FEATURES_V2,
            context_features=N_CONTEXT,
        )
    )
    out = model(
        torch.from_numpy(bf[None]),
        torch.from_numpy(pf[None]),
        torch.from_numpy(pm[None]),
        torch.from_numpy(ctx[None]),
    )
    assert out.shape == (1, 50, 3) and torch.isfinite(out).all()


def test_mask_low_detection_only_touches_sparse_frames() -> None:
    """Verify low-detection masking affects only sparse tracking frames."""
    pytest.importorskip("torch")
    from possession_lab.neural.canonical import CanonicalMatch
    from possession_lab.neural.data import mask_low_detection

    ball, home, away = _scene(20)
    home[10:15] = np.nan  # a "replay": nobody tracked
    away[10:15] = np.nan
    labels = np.full(20, D, np.int8)
    match = CanonicalMatch(
        "t:1", "t", ball, home, away, labels, np.ones(20, np.int8), (105.0, 68.0)
    )
    masked = mask_low_detection(match, 4)
    assert np.all(masked.labels[10:15] == X)
    assert np.all(masked.labels[:10] == D) and np.all(masked.labels[15:] == D)
    np.testing.assert_array_equal(masked.home, match.home)
