"""Set-temporal transformer for per-frame possession (H / A / D).

Spatial stage: a permutation-invariant transformer over {CLS, ball, players} per frame, with a
key-padding mask for absent players. With ``edge_bias`` (N014) it becomes a graph transformer:
pairwise relations between ball and players (relative position and velocity, closing speed,
team-mate / opponent, ball edge) are mapped by a small MLP to an additive attention bias per layer
and head (zero-initialised, so training starts from the plain set transformer). Temporal stage:
a bidirectional transformer over frame embeddings, with sinusoidal positions. Offline inference
uses future context.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from possession_lab.neural.features import BALL_FEATURES, PLAYER_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters (all selected on external validation only)."""

    d_model: int = 96
    spatial_layers: int = 2
    temporal_layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    n_classes: int = 3
    # Feature version 2 (EDA-driven): wider player tokens and a per-frame team-context token.
    # Defaults reproduce the version-1 architecture so older checkpoints load unchanged.
    player_features: int = PLAYER_FEATURES
    context_features: int = 0
    # N012: auxiliary possession-change head (one extra logit per frame: a direct H<->A change
    # within +/- a few frames). Off by default so older checkpoints load and behave unchanged.
    transition_head: bool = False
    # N014: graph-relational spatial attention and a latent-ball auxiliary head (mean xy and
    # log-variance of the true ball position, trained against the clean source tracking).
    edge_bias: bool = False
    edge_hidden: int = 16
    ball_head: bool = False

    def to_dict(self) -> dict[str, int | float]:
        """JSON-friendly representation for checkpoints."""
        return asdict(self)


def _mlp(inp: int, out: int) -> nn.Sequential:
    """Create the shared two-layer GELU projection used to embed model tokens."""
    return nn.Sequential(nn.Linear(inp, out), nn.GELU(), nn.Linear(out, out))


class SinusoidalPositions(nn.Module):  # type: ignore[misc]
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        """Precompute a fixed sinusoidal position table for temporal token sequences.

        Args:
            d_model: Width of every input token embedding.
            max_len: Maximum sequence length represented by the lookup table.
        """
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positions to ``[B, T, D]``."""
        table: Tensor = self.table
        return x + table[: x.shape[1]].unsqueeze(0).to(x.dtype)


EDGE_FEATURES = 9
_HALF = (52.5, 34.0)


def edge_features(ball: Tensor, players: Tensor, n_fixed: int) -> Tensor:
    """Pairwise relations ``[B*T, L, L, EDGE_FEATURES]`` for tokens [cls, ball, (ctx), players].

    Uses the scaled token channels of :func:`features.build_tokens` (player: team, position,
    velocity; ball: last-known position, velocity). Edges touching non-entity tokens (cls,
    context) are zero apart from their indicator channels.
    """
    b, t, k, _ = players.shape
    bt = b * t
    scale = players.new_tensor(_HALF)
    ppos = players[..., 1:3].float() * scale
    pvel = players[..., 3:5].float() * 8.0
    bpos = ball[..., 5:7].float() * scale
    bvel = ball[..., 3:5].float() * 15.0
    pad = n_fixed - 2  # context token(s) between ball and players
    zeros2 = ppos.new_zeros(b, t, 1, 2)
    pos = torch.cat([zeros2, bpos.unsqueeze(2), *([zeros2] * pad), ppos], 2).reshape(bt, -1, 2)
    vel = torch.cat([zeros2, bvel.unsqueeze(2), *([zeros2] * pad), pvel], 2).reshape(bt, -1, 2)
    n = pos.shape[1]
    team = torch.zeros(bt, n, device=players.device)
    team[:, n_fixed:] = players[..., 0].float().reshape(bt, k)
    entity = torch.zeros(bt, n, device=players.device)
    entity[:, 1] = 1.0
    entity[:, n_fixed:] = (players[..., 0] != 0).float().reshape(bt, k)
    is_ball = torch.zeros(bt, n, device=players.device)
    is_ball[:, 1] = 1.0
    ent = entity[:, :, None] * entity[:, None, :]
    d = pos[:, None, :, :] - pos[:, :, None, :]  # j - i
    dist = d.norm(dim=-1).clamp_min(1e-3)
    dv = vel[:, None, :, :] - vel[:, :, None, :]
    closing = -(dv * d).sum(-1) / dist
    feats = torch.cat(
        [
            d / 20.0,
            (dist / 20.0).unsqueeze(-1),
            dv / 8.0,
            (closing / 8.0).unsqueeze(-1),
            (team[:, :, None] * team[:, None, :]).unsqueeze(-1),
            (is_ball[:, :, None] + is_ball[:, None, :]).clamp_max(1).unsqueeze(-1),
            ent.unsqueeze(-1),
        ],
        -1,
    )
    geo = torch.ones(EDGE_FEATURES, device=players.device)
    geo[-3:] = 0  # indicator channels survive for non-entity edges
    feats = feats * (ent.unsqueeze(-1) * geo + (1 - geo))
    return feats.clamp(-5, 5)


class SetTemporalNet(nn.Module):  # type: ignore[misc]
    """Per-frame possession classifier over token sets."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the spatial set encoder and temporal transformer from model settings."""
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.ball_embed = _mlp(BALL_FEATURES, d)
        self.player_embed = _mlp(cfg.player_features, d)
        if cfg.context_features:
            self.context_embed = _mlp(cfg.context_features, d)
            self.context_type = nn.Parameter(torch.zeros(1, d))
            self.context_skip = nn.Linear(cfg.context_features, d)
            nn.init.normal_(self.context_type, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_embed = nn.Parameter(torch.zeros(3, d))  # cls, ball, player
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

        def layer() -> nn.TransformerEncoderLayer:
            """Construct one pre-normalized encoder layer using the experiment config."""
            return nn.TransformerEncoderLayer(
                d,
                cfg.heads,
                d * cfg.ff_mult,
                cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )

        self.spatial = nn.TransformerEncoder(
            layer(), cfg.spatial_layers, enable_nested_tensor=False
        )
        if cfg.edge_bias:
            self.edge_mlp = nn.Sequential(
                nn.Linear(EDGE_FEATURES, cfg.edge_hidden),
                nn.GELU(),
                nn.Linear(cfg.edge_hidden, cfg.spatial_layers * cfg.heads),
            )
            nn.init.zeros_(self.edge_mlp[2].weight)
            nn.init.zeros_(self.edge_mlp[2].bias)
        self.frame_norm = nn.LayerNorm(d)
        self.ball_skip = nn.Linear(BALL_FEATURES, d)
        self.positions = SinusoidalPositions(d)
        self.temporal = nn.TransformerEncoder(
            layer(), cfg.temporal_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.n_classes))
        if cfg.transition_head:
            self.transition = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        if cfg.ball_head:
            self.ball_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))

    def forward(
        self, ball: Tensor, players: Tensor, mask: Tensor, context: Tensor | None = None
    ) -> Tensor:
        """Return per-frame outputs ``[B, T, C (+1) (+4)]``.

        Channels: ``C`` class logits; then the possession-change logit if ``transition_head``;
        then latent-ball mean (x, y, scaled like the ball token) and log-variance if ``ball_head``.

        Args:
            ball: ``[B, T, BALL_FEATURES]``.
            players: ``[B, T, K, PLAYER_FEATURES]``.
            mask: ``[B, T, K]`` boolean, True where a player token is real.
            context: ``[B, T, context_features]`` team-context vector (feature version 2 only).
        """
        b, t, k, _ = players.shape
        d = self.cfg.d_model
        cls = (self.cls + self.type_embed[0]).expand(b * t, 1, d)
        ball_tok = (self.ball_embed(ball) + self.type_embed[1]).reshape(b * t, 1, d)
        player_tok = (self.player_embed(players) + self.type_embed[2]).reshape(b * t, k, d)
        fixed = [cls, ball_tok]
        if self.cfg.context_features:
            if context is None:
                raise ValueError("model expects a context tensor")
            fixed.append((self.context_embed(context) + self.context_type).reshape(b * t, 1, d))
        tokens = torch.cat([*fixed, player_tok], dim=1)
        pad = torch.cat(
            [
                torch.zeros(b * t, len(fixed), dtype=torch.bool, device=mask.device),
                ~mask.reshape(b * t, k),
            ],
            dim=1,
        )
        if self.cfg.edge_bias:
            heads, n = self.cfg.heads, tokens.shape[1]
            bias = self.edge_mlp(edge_features(ball, players, len(fixed)))  # [bt, L, L, l*h]
            bias = bias.permute(0, 3, 1, 2).reshape(b * t, self.cfg.spatial_layers, heads, n, n)
            # padded keys -> -inf (cls / ball are never padded, so no row is fully masked)
            neg = torch.zeros(b * t, 1, 1, 1, n, device=tokens.device)
            bias = bias + neg.masked_fill(pad[:, None, None, None, :], float("-inf"))
            if torch.is_autocast_enabled():
                bias = bias.to(torch.get_autocast_gpu_dtype())
            encoded = tokens
            for i, block in enumerate(self.spatial.layers):
                encoded = block(encoded, src_mask=bias[:, i].reshape(b * t * heads, n, n))
        else:
            encoded = self.spatial(tokens, src_key_padding_mask=pad)
        frame = self.frame_norm(encoded[:, 0].reshape(b, t, d)) + self.ball_skip(ball)
        if self.cfg.context_features and context is not None:
            frame = frame + self.context_skip(context)
        frame = self.temporal(self.positions(frame))
        logits: Tensor = self.head(frame)
        if self.cfg.transition_head:
            logits = torch.cat([logits, self.transition(frame)], dim=-1)
        if self.cfg.ball_head:
            logits = torch.cat([logits, self.ball_out(frame)], dim=-1)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
