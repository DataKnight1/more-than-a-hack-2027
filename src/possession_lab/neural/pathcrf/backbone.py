"""Set Attention backbone for PathCRF with explicit Home/Away and Dead-ball heads.

Adapts SetLSTM (Kim et al., KDD 2026, MPL-2.0) into a pure PyTorch sequence model
predicting:
1. Dead-ball logit (binary Alive vs Dead)
2. Conditional team logits [Home, Away]
3. Pairwise directed edge emissions [B, T, 676] and compressed [B, T, 576]
4. Compressed edge representations [B, T, 576, D_crf] for dynamic CRF transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .edge_states import DENSE_EDGES, build_edge_compression_maps

# ============================================================================
# Set Attention Primitives (MPL-2.0, Kim et al. / Lee et al.)
# ============================================================================


class RFF(nn.Module):  # type: ignore[misc]
    """Row-wise Feedforward layers."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared feedforward network to each set element."""
        return self.layers(x)


class SetMultiheadAttention(nn.Module):  # type: ignore[misc]
    """Multihead Attention equivariant to batch and query permutation."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        assert d % h == 0, f"Dimension {d} must be divisible by heads {h}"
        self.d = d
        self.h = h
        self.p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.temperature = math.sqrt(self.p)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Attend from query elements to the given key and value elements."""
        b, n, d = queries.size()
        _, m, _ = keys.size()
        h, p = self.h, self.p

        q = (
            self.project_queries(queries)
            .view(b, n, h, p)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(h * b, n, p)
        )
        k = (
            self.project_keys(keys)
            .view(b, m, h, p)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(h * b, m, p)
        )
        v = (
            self.project_values(values)
            .view(b, m, h, p)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(h * b, m, p)
        )

        att = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        att = torch.softmax(att, dim=-1)
        out = torch.bmm(att, v).view(h, b, n, p).permute(1, 2, 0, 3).contiguous().view(b, n, d)
        return self.concatenation(out)


class MAB(nn.Module):  # type: ignore[misc]
    """Multihead Attention Block."""

    def __init__(self, d: int, h: int, rff: RFF) -> None:
        super().__init__()
        self.multihead = SetMultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Attend to another set and apply the residual feedforward block."""
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SAB(nn.Module):  # type: ignore[misc]
    """Set Attention Block (self-attention over sets)."""

    def __init__(self, d: int, h: int, rff: RFF) -> None:
        super().__init__()
        self.mab = MAB(d, h, rff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to the input set."""
        return self.mab(x, x)


class ISAB(nn.Module):  # type: ignore[misc]
    """Induced Set Attention Block with M inducing points for O(NM) efficiency."""

    def __init__(self, d: int, m: int, h: int, rff1: RFF, rff2: RFF) -> None:
        super().__init__()
        self.mab1 = MAB(d, h, rff1)
        self.mab2 = MAB(d, h, rff2)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply induced attention through learned inducing points."""
        b = x.size(0)
        i = self.inducing_points.expand(b, -1, -1)
        h = self.mab1(i, x)
        return self.mab2(x, h)


class PositionalEncoding(nn.Module):  # type: ignore[misc]
    """Sinusoidal positional encoding along sequence length."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positions to a time-major input tensor."""
        # x: [T, B, D]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# ============================================================================
# Output Dataclass
# ============================================================================


@dataclass
class N020Outputs:
    """Multi-task outputs from the N020 PathCRF backbone."""

    dead_logit: torch.Tensor  # [B, T]
    team_logits: torch.Tensor  # [B, T, 2] (conditional Home vs Away)
    edge_logits: torch.Tensor  # [B, T, 676] dense directed edge logits
    compressed_edge_logits: torch.Tensor  # [B, T, 576] valid state logits
    edge_embeds: torch.Tensor  # [B, T, 576, D_crf] compressed representations
    node_valid: torch.Tensor  # [B, T, 26] observation validity


# ============================================================================
# PathCRFNet Backbone Architecture
# ============================================================================


class PathCRFNet(nn.Module):  # type: ignore[misc]
    """Complete PathCRF neural backbone adapted for Home / Away / Dead possession recognition."""

    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        feat_dim: int = 8,
        team_size: int = 11,
        macro_dim: int = 32,
        micro_dim: int = 64,
        seq_dim: int = 128,
        crf_edge_dim: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.team_size = team_size
        self.n_nodes = team_size * 2 + 4  # 26
        self.macro_dim = macro_dim
        self.micro_dim = micro_dim
        self.seq_dim = seq_dim
        self.crf_edge_dim = crf_edge_dim
        self.heads = heads
        self.dropout = dropout

        # 1. Spatial Encoders (Per-team and outside pitch node projections)
        self.team1_enc = nn.Sequential(
            nn.Linear(feat_dim, macro_dim),
            nn.ReLU(inplace=True),
            ISAB(macro_dim, m=8, h=heads, rff1=RFF(macro_dim), rff2=RFF(macro_dim)),
        )
        self.team2_enc = nn.Sequential(
            nn.Linear(feat_dim, macro_dim),
            nn.ReLU(inplace=True),
            ISAB(macro_dim, m=8, h=heads, rff1=RFF(macro_dim), rff2=RFF(macro_dim)),
        )
        self.outside_enc = nn.Sequential(
            nn.Linear(feat_dim, macro_dim),
            nn.ReLU(inplace=True),
        )

        # Full player set encoder across all 26 nodes
        self.full_set_enc = nn.Sequential(
            nn.Linear(macro_dim, micro_dim),
            nn.ReLU(inplace=True),
            SAB(micro_dim, h=heads, rff=RFF(micro_dim)),
        )

        # 2. Temporal Sequence Modeling (Bidirectional Temporal SAB)
        self.temporal_proj = nn.Sequential(
            nn.Linear(micro_dim, seq_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_encoder = PositionalEncoding(seq_dim, dropout=dropout)
        self.temporal_sab1 = SAB(seq_dim, h=heads, rff=RFF(seq_dim))
        self.temporal_sab2 = SAB(seq_dim, h=heads, rff=RFF(seq_dim))

        # 3. Micro Edge Heads (Pairwise directed edge emissions)
        self.edge_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, seq_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(seq_dim, 2),
            nn.GLU(dim=-1),  # scalar logit per directed edge
        )

        # CRF Edge Representation
        self.crf_edge_proj = nn.Sequential(
            nn.Linear(seq_dim, crf_edge_dim),
            nn.ReLU(inplace=True),
        )
        self.crf_edge_norm = nn.LayerNorm(crf_edge_dim)

        # 4. Supervised Dead-Ball Head
        self.dead_head = nn.Sequential(
            nn.Linear(seq_dim, seq_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(seq_dim // 2, 1),
        )

        # 5. Supervised Conditional Home/Away Head
        self.team_head = nn.Sequential(
            nn.Linear(seq_dim * 2, seq_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(seq_dim // 2, 2),
        )

        # Precompute edge compression buffers
        valid_ids, orig2comp, comp_src, comp_dst = build_edge_compression_maps(team_size * 2, 4)
        self.register_buffer("valid_orig_edge_ids", valid_ids)  # [576]
        self.register_buffer("orig2comp", orig2comp)  # [676]
        self.register_buffer("comp_src", comp_src)  # [576]
        self.register_buffer("comp_dst", comp_dst)  # [576]

    def _temporal_sab_step(
        self, x: torch.Tensor, block: SAB, pos_enc: PositionalEncoding | None = None
    ) -> torch.Tensor:
        """Apply SAB along the temporal dimension [T, B*N, D]."""
        if pos_enc is not None:
            x = pos_enc(x)
        x = x.transpose(0, 1)  # [B*N, T, D]
        x = block(x)
        x = x.transpose(0, 1)  # [T, B*N, D]
        return x

    def forward(
        self,
        x: torch.Tensor,
        node_valid: torch.Tensor | None = None,
    ) -> N020Outputs:
        """Forward pass.

        Args:
            x: [B, T, 26, F] player-node feature tensor.
            node_valid: [B, T, 26] boolean observation validity mask.

        Returns:
            N020Outputs containing dead, team, edge, compressed-edge and embedding outputs.
        """
        B, T, N, F_dim = x.shape
        team_size = self.team_size
        assert self.n_nodes == N, f"Expected {self.n_nodes} nodes, got {N}"

        # 1. Spatial Encoding per frame
        # Flatten time and batch dimensions for set transformers: [B*T, N, F]
        flat_x = x.view(B * T, N, F_dim)

        h_team1 = self.team1_enc(flat_x[:, :team_size])
        h_team2 = self.team2_enc(flat_x[:, team_size : 2 * team_size])
        h_out = self.outside_enc(flat_x[:, 2 * team_size :])

        spatial_nodes = torch.cat([h_team1, h_team2, h_out], dim=1)  # [B*T, 26, macro_dim]
        full_nodes = self.full_set_enc(spatial_nodes)  # [B*T, 26, micro_dim]

        # 2. Temporal Modeling
        # Reshape to [T, B*N, micro_dim]
        t_nodes = full_nodes.view(B, T, N, -1).permute(1, 0, 2, 3).contiguous().view(T, B * N, -1)
        t_proj = self.temporal_proj(t_nodes)
        t_h = self._temporal_sab_step(t_proj, self.temporal_sab1, self.pos_encoder)
        t_h = self._temporal_sab_step(t_h, self.temporal_sab2, None)  # [T, B*N, seq_dim]

        # Reshape to [B, T, N, seq_dim]
        node_reps = t_h.view(T, B, N, -1).permute(1, 0, 2, 3).contiguous()  # [B, T, 26, seq_dim]

        # 3. Pairwise Edge Projections (Micro directed edge emissions)
        src_reps = node_reps.unsqueeze(3).expand(-1, -1, -1, N, -1)  # [B, T, 26, 26, seq_dim]
        dst_reps = node_reps.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, T, 26, 26, seq_dim]

        edge_pairs = torch.cat([src_reps, dst_reps], dim=-1)  # [B, T, 26, 26, 2*seq_dim]
        edge_hidden = self.edge_proj(edge_pairs)  # [B, T, 26, 26, seq_dim]
        dense_logits = self.edge_head(edge_hidden).squeeze(-1)  # [B, T, 26, 26]
        dense_logits = dense_logits.view(B, T, DENSE_EDGES)  # [B, T, 676]

        # Gather 576 valid compressed edge logits
        compressed_logits = dense_logits.gather(
            2, self.valid_orig_edge_ids.view(1, 1, -1).expand(B, T, -1)
        )

        # Compressed edge representations for DynamicSparseCRF
        flat_edge_hidden = edge_hidden.view(B, T, DENSE_EDGES, -1)
        valid_edge_hidden = flat_edge_hidden.gather(
            2,
            self.valid_orig_edge_ids.view(1, 1, -1, 1).expand(B, T, -1, self.seq_dim),
        )  # [B, T, 576, seq_dim]
        edge_embeds = self.crf_edge_norm(
            self.crf_edge_proj(valid_edge_hidden)
        )  # [B, T, 576, D_crf]

        # 4. Supervised Dead-Ball Head (global pooled player representation)
        player_pool = node_reps[:, :, : 2 * team_size].mean(dim=2)  # [B, T, seq_dim]
        dead_logit = self.dead_head(player_pool).squeeze(-1)  # [B, T]

        # 5. Supervised Conditional Home/Away Head
        h_pool = node_reps[:, :, :team_size].mean(dim=2)  # [B, T, seq_dim]
        a_pool = node_reps[:, :, team_size : 2 * team_size].mean(dim=2)  # [B, T, seq_dim]
        team_features = torch.cat([h_pool, a_pool], dim=-1)  # [B, T, 2*seq_dim]
        team_logits = self.team_head(team_features)  # [B, T, 2]

        if node_valid is None:
            node_valid = torch.ones((B, T, N), dtype=torch.bool, device=x.device)

        return N020Outputs(
            dead_logit=dead_logit,
            team_logits=team_logits,
            edge_logits=dense_logits,
            compressed_edge_logits=compressed_logits,
            edge_embeds=edge_embeds,
            node_valid=node_valid,
        )
