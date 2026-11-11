"""N014 components: pitch control (v4), graph-relational attention, latent-ball head (synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from possession_lab.neural.canonical import A, CanonicalMatch, H, join_periods
from possession_lab.neural.pitch_control import N_PC_CONTEXT, pitch_control_features

ROOT = Path(__file__).resolve().parents[1]


def _scene(n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a compact synthetic football scene for spatial-feature checks."""
    ball = np.zeros((n, 3), np.float32)
    home = np.full((n, 11, 2), np.nan, np.float32)
    away = np.full((n, 11, 2), np.nan, np.float32)
    home[:, 0] = [0.5, 0.0]  # home player on the ball
    away[:, 0] = [20.0, 0.0]  # away player 20 m away
    return ball, home, away


def test_pitch_control_favours_the_player_who_reaches_the_ball() -> None:
    """Verify pitch control favours the player with the shorter time to the ball."""
    ball, home, away = _scene()
    ctx, per = pitch_control_features(ball, home, away)
    assert ctx.shape == (50, N_PC_CONTEXT) and per.shape == (50, 22, 3)
    assert np.isfinite(ctx).all() and np.isfinite(per).all()
    assert ctx[25, 0] > 0.99  # P(home) now
    ctx_swap, _ = pitch_control_features(ball, away, home)
    assert ctx_swap[25, 0] < 0.01
    # ball moving fast towards the away player: control shifts at the 1 s horizon
    moving = ball.copy()
    moving[:, 0] = np.arange(50) * 0.6  # 15 m/s towards +x
    ctx_m, _ = pitch_control_features(moving, home, away)
    assert ctx_m[25, 6] < ctx_m[25, 0]


def test_pitch_control_with_nobody_tracked_is_neutral() -> None:
    """Verify pitch control remains neutral when no players are observed."""
    ball, home, away = _scene()
    home[:] = np.nan
    away[:] = np.nan
    ball[:10] = np.nan
    ctx, per = pitch_control_features(ball, home, away)
    assert np.isfinite(ctx).all() and np.allclose(ctx[:, 0], 0.5)
    assert np.isfinite(per).all()


def test_v4_inputs() -> None:
    """Verify feature version 4 returns finite context and player tensors."""
    from possession_lab.neural.features import build_inputs, feature_dims, version_of

    ball, home, away = _scene()
    bf, pf, mask, ctx = build_inputs(ball, home, away, 30, np.ones(50, np.float32), 4)
    assert pf.shape[-1] == feature_dims(4)[0] and ctx.shape[-1] == feature_dims(4)[1]
    assert version_of(ctx.shape[-1]) == 4
    assert np.isfinite(pf).all() and np.isfinite(ctx).all()


def _tokens(torch, b: int = 2, t: int = 8, k: int = 6):  # noqa: ANN001, ANN202
    """Build synthetic token tensors with the production model input layout."""
    from possession_lab.neural.features import BALL_FEATURES, feature_dims

    pdim, cdim = feature_dims(4)
    gen = torch.Generator().manual_seed(0)
    ball = torch.randn(b, t, BALL_FEATURES, generator=gen) * 0.3
    players = torch.randn(b, t, k, pdim, generator=gen) * 0.3
    players[..., 0] = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, 0.0])
    mask = torch.ones(b, t, k, dtype=torch.bool)
    mask[..., -1] = False
    players[..., -1, :] = 0.0
    context = torch.randn(b, t, cdim, generator=gen) * 0.3
    return ball, players, mask, context, pdim, cdim


def test_edge_features_geometry() -> None:
    """Verify pairwise edge features encode relative player geometry."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.model import EDGE_FEATURES, edge_features

    ball, players, _, _, _, _ = _tokens(torch)
    feats = edge_features(ball, players, n_fixed=3)  # cls, ball, ctx
    assert feats.shape == (16, 9, 9, EDGE_FEATURES)
    assert torch.allclose(feats[..., 2], feats[..., 2].transpose(1, 2))  # distance symmetric
    assert (feats[:, 0, :, :6] == 0).all() and (feats[:, 2, :, :6] == 0).all()  # cls / ctx
    assert (feats[:, -1, :, :6] == 0).all()  # padded player
    assert feats[0, 3, 4, 6] == 1.0 and feats[0, 3, 6, 6] == -1.0  # mate / opponent
    assert feats[0, 1, 3, 7] == 1.0 and feats[0, 3, 4, 7] == 0.0  # ball edge flag


def test_zero_initialised_edge_bias_matches_plain_set_transformer() -> None:
    """Verify zero-initialized graph bias preserves the plain transformer output."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    ball, players, mask, context, pdim, cdim = _tokens(torch)
    kw = {
        "d_model": 32,
        "spatial_layers": 2,
        "temporal_layers": 1,
        "heads": 2,
        "player_features": pdim,
        "context_features": cdim,
        "dropout": 0.0,
    }
    torch.manual_seed(0)
    plain = SetTemporalNet(ModelConfig(**kw)).eval()
    graph = SetTemporalNet(ModelConfig(**kw, edge_bias=True)).eval()
    missing, unexpected = graph.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(m.startswith("edge_mlp") for m in missing)
    with torch.no_grad():
        torch.testing.assert_close(
            graph(ball, players, mask, context),
            plain(ball, players, mask, context),
            atol=1e-5,
            rtol=1e-5,
        )


def test_heads_layout_and_latent_ball_loss() -> None:
    """Verify optional output heads and latent-ball loss use the expected shapes."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "scripts" / "neural"))
    import train

    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    ball, players, mask, context, pdim, cdim = _tokens(torch)
    model = SetTemporalNet(
        ModelConfig(
            d_model=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=2,
            player_features=pdim,
            context_features=cdim,
            transition_head=True,
            edge_bias=True,
            ball_head=True,
        )
    )
    out = model(ball, players, mask, context)
    assert out.shape == (2, 8, 3 + 1 + 4) and torch.isfinite(out).all()
    target = torch.zeros(2, 8, 4)
    target[..., 2] = 1.0  # valid
    target[:, :4, 3] = 1.0  # input hidden on the first half
    loss = train.latent_ball_loss(out[..., 4:8], target, 0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.ball_out[1].weight.grad is not None
    # perfect, confident prediction beats an off-target one
    good = torch.zeros(2, 8, 4)
    good[..., 2:] = -4.0
    bad = good.clone()
    bad[..., 0] = 1.0
    assert train.latent_ball_loss(good, target, 0.2) < train.latent_ball_loss(bad, target, 0.2)
    target[..., 2] = 0.0  # no true ball anywhere -> no loss
    assert train.latent_ball_loss(bad, target, 0.2) == 0.0


def test_stream_yields_ball_target_and_inference_ignores_ball_channels(tmp_path: Path) -> None:
    """Verify training streams expose ball targets while inference predictions ignore them."""
    torch = pytest.importorskip("torch")
    from possession_lab.neural.data import WindowConfig, WindowStream, predict_log_probs
    from possession_lab.neural.features import feature_dims
    from possession_lab.neural.model import ModelConfig, SetTemporalNet

    rng = np.random.default_rng(0)
    n = 200
    ball = np.zeros((n, 3), np.float32)
    ball[:, 0] = np.linspace(-20, 20, n)
    home = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    away = rng.normal(0, 10, (n, 11, 2)).astype(np.float32)
    labels = np.full(n, H, np.int8)
    labels[n // 2 :] = A
    b, h, a, lab, per = join_periods([(ball, home, away, labels)])
    match = CanonicalMatch("toy:b", "toy", b, h, a, lab, per, (105.0, 68.0))
    match.save(tmp_path / "toy.npz")
    stream = WindowStream(
        [tmp_path / "toy.npz"],
        WindowConfig(
            window=64,
            batch_size=2,
            seed=0,
            max_players=30,
            clean_prob=0.0,
            feature_version=4,
            ball_target=True,
        ),
    )
    batch = next(iter(stream))
    assert len(batch) == 6 and batch[5].shape == (2, 64, 4)
    tgt = batch[5].numpy()
    assert set(np.unique(tgt[..., 2])) <= {0.0, 1.0} and (np.abs(tgt[..., :2]) <= 1.01).all()
    pdim, cdim = feature_dims(4)
    model = SetTemporalNet(
        ModelConfig(
            d_model=32,
            spatial_layers=1,
            temporal_layers=1,
            heads=2,
            player_features=pdim,
            context_features=cdim,
            transition_head=True,
            edge_bias=True,
            ball_head=True,
        )
    )
    lp = predict_log_probs(model, match, 64, 32, torch.device("cpu"), 30)
    assert lp.shape == (match.n_frames, 4) and (lp <= 1e-6).all()


def test_bounded_latent_ball_loss_is_non_negative_and_hidden_only() -> None:
    """Verify the Huber ball loss is nonnegative and applies only to hidden targets."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "scripts" / "neural"))
    import train

    target = torch.zeros(2, 8, 4)
    target[..., 2] = 1.0
    target[:, :4, 3] = 1.0  # hidden on the first half
    perfect = torch.zeros(2, 8, 4)
    perfect[..., 2:] = -8.0  # confident: NLL would go strongly negative, Huber must not
    assert train.latent_ball_loss(perfect, target, 0.2) < 0
    assert train.latent_ball_huber(perfect, target, 0.05) == 0.0
    off_visible = perfect.clone()
    off_visible[:, 4:, 0] = 1.0  # error only where the input ball is visible -> ignored
    assert train.latent_ball_huber(off_visible, target, 0.05) == 0.0
    off_hidden = perfect.clone()
    off_hidden[:, :4, 0] = 1.0
    assert torch.isclose(train.latent_ball_huber(off_hidden, target, 0.05), torch.tensor(0.975))
