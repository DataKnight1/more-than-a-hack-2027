"""Synthetic tests for the neural possession package (no challenge data, no external downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from possession_lab.neural.augment import DegradeConfig, degrade, symmetry
from possession_lab.neural.baselines import baseline_codes, nearest_player_codes
from possession_lab.neural.canonical import (
    GAP_FRAMES,
    A,
    CanonicalMatch,
    D,
    H,
    X,
    join_periods,
    label_summary,
)
from possession_lab.neural.decode import DecodeConfig, decode_match, viterbi
from possession_lab.neural.external_eval import score
from possession_lab.neural.features import BALL_FEATURES, PLAYER_FEATURES, build_tokens


def _toy_period(n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Construct a small synthetic period with fixed player and ball positions."""
    ball = np.zeros((n, 3), np.float32)
    ball[:, 0] = np.linspace(-30, 30, n)
    home = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    away = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    labels = np.full(n, H, np.int8)
    labels[n // 2 :] = A
    labels[n // 3 : n // 3 + 20] = D
    return ball, home, away, labels


def _toy_match(n: int = 400, seed: int = 0) -> CanonicalMatch:
    """Assemble a canonical match fixture from synthetic period arrays."""
    rng = np.random.default_rng(seed)
    ball, home, away, labels, period = join_periods([_toy_period(n, rng), _toy_period(n, rng)])
    return CanonicalMatch("toy:1", "toy", ball, home, away, labels, period, (105.0, 68.0))


def test_join_periods_inserts_x_gap() -> None:
    """Verify joined periods receive an X-labeled gap so possession segments cannot cross halftime."""
    match = _toy_match()
    assert match.n_frames == 800 + GAP_FRAMES
    assert np.all(match.labels[400 : 400 + GAP_FRAMES] == X)
    assert np.all(match.period[400 : 400 + GAP_FRAMES] == 0)


def test_canonical_roundtrip(tmp_path: Path) -> None:
    """Verify canonical match arrays and metadata survive serialization and reload."""
    match = _toy_match()
    match.meta["licence"] = "test"
    match.save(tmp_path / "m.npz")
    back = CanonicalMatch.load(tmp_path / "m.npz")
    assert back.match_id == match.match_id and back.meta["licence"] == "test"
    np.testing.assert_array_equal(back.labels, match.labels)
    np.testing.assert_allclose(back.home, match.home)


def test_label_summary_counts_spells() -> None:
    """Verify label summaries count H/A spells and D frames correctly."""
    labels = np.array([H] * 10 + [D] * 5 + [A] * 10 + [X] * 3 + [H] * 5, np.int8)
    summary = label_summary(labels)
    assert summary["n_spells"] == 3


def test_viterbi_removes_flicker_but_keeps_real_switch() -> None:
    """Verify Viterbi removes isolated label flicker while preserving a sustained team switch."""
    n = 600
    probs = np.full((n, 3), 0.05, np.float32)
    probs[:300, H] = 0.9
    probs[300:, A] = 0.9
    probs[100:103] = [0.05, 0.9, 0.05]  # 3-frame flicker to A
    path = viterbi(np.log(probs), DecodeConfig(team_switch=8.0, dead_switch=4.0))
    assert np.all(path[:300] == H)
    assert np.all(path[300:] == A)


def test_decode_match_marks_gaps_x() -> None:
    """Verify decoder output marks inter-period rows as X."""
    match = _toy_match()
    logp = np.log(np.full((match.n_frames, 3), 1 / 3, np.float32))
    out = decode_match(logp, match.period, DecodeConfig())
    assert np.all(out[match.period == 0] == X)


def test_nearest_player_and_missing_ball() -> None:
    """Verify the nearest-player baseline handles absent ball observations safely."""
    ball = np.array([[0, 0, 0], [np.nan, np.nan, np.nan]], np.float32)
    home = np.array([[[1, 0]], [[1, 0]]], np.float32)
    away = np.array([[[5, 0]], [[0.5, 0]]], np.float32)
    codes = nearest_player_codes(ball, home, away)
    assert codes.tolist() == [0, -1]
    assert baseline_codes(ball, home, away, window=0).tolist() == [H, D]


def test_features_have_no_nan_and_fixed_shape() -> None:
    """Verify model features have the expected shape and contain no NaN values."""
    match = _toy_match()
    rng = np.random.default_rng(1)
    sl = slice(0, 256)
    ball, home, away = degrade(rng, match.ball[sl], match.home[sl], match.away[sl])
    bf, pf, pm = build_tokens(ball, home, away, max_players=30)
    assert bf.shape == (256, BALL_FEATURES) and pf.shape == (256, 30, PLAYER_FEATURES)
    assert pm.shape == (256, 30)
    assert np.isfinite(bf).all() and np.isfinite(pf).all()


def test_degrade_is_deterministic_and_strength_zero_is_identity() -> None:
    """Verify seeded degradation is reproducible and zero strength leaves inputs unchanged."""
    match = _toy_match()
    sl = slice(0, 200)
    a = degrade(np.random.default_rng(3), match.ball[sl], match.home[sl], match.away[sl])
    b = degrade(np.random.default_rng(3), match.ball[sl], match.home[sl], match.away[sl])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    clean = degrade(
        np.random.default_rng(3),
        match.ball[sl],
        match.home[sl],
        match.away[sl],
        DegradeConfig(),
        strength=0.0,
    )
    np.testing.assert_array_equal(clean[0], match.ball[sl])


def test_symmetry_swaps_labels_with_teams() -> None:
    """Verify horizontal team symmetry also swaps home and away labels."""
    match = _toy_match()
    sl = slice(0, 400)
    for seed in range(20):
        _, h, _, lab = symmetry(
            np.random.default_rng(seed),
            match.ball[sl],
            match.home[sl],
            match.away[sl],
            match.labels[sl],
        )
        swapped = not np.allclose(np.abs(h), np.abs(match.home[sl]))
        expected_first = A if swapped else H
        assert lab[0] == expected_first


def test_external_score_perfect_prediction() -> None:
    """Verify the external metric returns a perfect score for an exact synthetic prediction."""
    match = _toy_match()
    result = score(match.labels, match.labels)
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_model_forward_shapes() -> None:
    """Verify the neural model emits the configured batch, time, and class dimensions."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    model = SetTemporalNet(ModelConfig(d_model=32, spatial_layers=1, temporal_layers=1, heads=2))
    ball = torch.zeros(2, 16, BALL_FEATURES)
    players = torch.zeros(2, 16, 10, PLAYER_FEATURES)
    mask = torch.zeros(2, 16, 10, dtype=torch.bool)
    mask[..., :3] = True
    logits = model(ball, players, mask)
    assert logits.shape == (2, 16, 3)
    assert torch.isfinite(logits).all()
