"""N012 possession-change head: targets, model output, transition-aware decoder (synthetic only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from possession_lab.neural.canonical import A, CanonicalMatch, D, H, join_periods
from possession_lab.neural.decode import DecodeConfig, decode_match, team_costs, viterbi

ROOT = Path(__file__).resolve().parents[1]


def _train_module():  # noqa: ANN202 - script module
    """Load the training module without running its command-line entrypoint."""
    sys.path.insert(0, str(ROOT / "scripts" / "neural"))
    import train

    return train


def test_transition_targets_mark_direct_team_changes_only() -> None:
    """Verify transition targets mark direct H-to-A or A-to-H label changes."""
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[H] * 10 + [A] * 10 + [D] * 5 + [H] * 5 + [3] * 3 + [A] * 7])
    target, valid = _train_module().transition_targets(labels, radius=2)
    hot = torch.nonzero(target[0]).flatten().tolist()
    assert hot == [8, 9, 10, 11, 12]  # H->A at frame 10 only; D->H and X->A ignored
    assert valid[0, 30:33].sum() == 0 and valid[0, :30].all()


def test_model_transition_channel() -> None:
    """Verify the model exposes the auxiliary transition-probability channel."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.features import BALL_FEATURES, PLAYER_FEATURES
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    kw = {"d_model": 32, "spatial_layers": 1, "temporal_layers": 1, "heads": 2}
    ball = torch.zeros(2, 16, BALL_FEATURES)
    players = torch.zeros(2, 16, 10, PLAYER_FEATURES)
    mask = torch.zeros(2, 16, 10, dtype=torch.bool)
    mask[..., :3] = True
    assert SetTemporalNet(ModelConfig(**kw))(ball, players, mask).shape == (2, 16, 3)
    out = SetTemporalNet(ModelConfig(**kw, transition_head=True))(ball, players, mask)
    assert out.shape == (2, 16, 4) and torch.isfinite(out).all()


def _ambiguous_lp(n: int = 60, change_at: int | None = 30) -> np.ndarray:
    """Weak class evidence for A after frame 30; optional strong change evidence at one frame."""
    p = np.full((n, 3), [0.6, 0.35, 0.05])
    p[30:] = [0.45, 0.5, 0.05]  # 30 x 0.105 nats = 3.2 nats for switching to A
    lp = np.log(p)
    change = np.full(n, 1e-4)
    if change_at is not None:
        change[change_at] = 0.99
    return np.concatenate([lp, np.log(change)[:, None]], 1).astype(np.float32)


def test_zero_gain_reproduces_constant_decoder() -> None:
    """Verify zero transition gain exactly matches the constant-penalty decoder."""
    lp = _ambiguous_lp()
    base = DecodeConfig(team_switch=8.0, dead_switch=4.0)
    np.testing.assert_array_equal(viterbi(lp, base), viterbi(lp[:, :3], base))


def test_change_evidence_enables_switch_and_suppresses_spurious_ones() -> None:
    """Verify change evidence eases true switches and discourages unsupported transitions."""
    lp = _ambiguous_lp()
    assert (viterbi(lp, DecodeConfig(8.0, 4.0)) == H).all()  # too weak to pay 8 nats
    path = viterbi(lp, DecodeConfig(8.0, 4.0, transition_gain=2.0))
    assert (path[:30] == H).all() and (path[30:] == A).all()
    # no change evidence anywhere: the gain makes switching dearer, never cheaper
    quiet = _ambiguous_lp(change_at=None)
    assert (viterbi(quiet, DecodeConfig(1.0, 4.0))[30:] == A).all()  # cheap constant switch
    assert (viterbi(quiet, DecodeConfig(1.0, 4.0, transition_gain=2.0)) == H).all()
    assert team_costs(
        np.log(np.array([1e-4, 0.5, 0.99], np.float32)), DecodeConfig(8.0, 4.0, 2.0)
    ).tolist() == pytest.approx(
        [8.0 + 2.0 * np.log(1e4 - 1), 8.0, max(0.0, 8.0 - 2.0 * np.log(99.0))]
    )


def test_predict_log_probs_returns_change_column() -> None:
    """Verify prediction log probabilities include the transition-head output."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.data import predict_log_probs
    from possession_lab.neural.features import feature_dims
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    rng = np.random.default_rng(0)
    n = 120
    ball = np.zeros((n, 3), np.float32)
    ball[:, 0] = np.linspace(-20, 20, n)
    home = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    away = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    labels = np.full(n, H, np.int8)
    labels[n // 2 :] = A
    b, h, a, lab, per = join_periods([(ball, home, away, labels)])
    match = CanonicalMatch("toy:t", "toy", b, h, a, lab, per, (105.0, 68.0))
    pdim, cdim = feature_dims(3)
    model = SetTemporalNet(
        ModelConfig(
            d_model=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=2,
            player_features=pdim,
            context_features=cdim,
            transition_head=True,
        )
    )
    lp = predict_log_probs(model, match, 64, 32, torch.device("cpu"), 30)
    assert lp.shape == (match.n_frames, 4)
    # window blending averages log-probabilities (geometric mean), so rows sum to <= 1
    assert np.isfinite(lp).all() and (lp <= 1e-6).all()
    assert (np.exp(lp[:, :3]).sum(1) > 0.9).all()
    codes = decode_match(lp, match.period, DecodeConfig(4.0, 4.0, 1.0))
    assert set(np.unique(codes)) <= {H, A, D}
