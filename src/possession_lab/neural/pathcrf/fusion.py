"""Probabilistic fusion combining Dead-ball, Home/Away, and PathCRF edge evidence.

Computes joint 3-class posterior [P(H), P(A), P(D)]:
1. P(D) = sigmoid(dead_logit)
2. P(Alive) = 1.0 - P(D)
3. Fuses conditional team logits with path evidence from unambiguous edge subsets.
4. P(H) = P(Alive) * P(H | Alive)
5. P(A) = P(Alive) * P(A | Alive)
"""

from __future__ import annotations

import torch

from .edge_states import get_edge_semantic_masks


def aggregate_path_evidence(
    edge_probs: torch.Tensor,
    comp_src: torch.Tensor,
    comp_dst: torch.Tensor,
    team_size: int = 11,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate edge probability mass into semantic categories.

    Args:
        edge_probs: [..., 576] normalized edge probabilities.
        comp_src: [576] source nodes.
        comp_dst: [576] destination nodes.
        team_size: 11 players per team.

    Returns:
        (p_home, p_away, p_cross, p_outside) of shape [...]
    """
    masks = get_edge_semantic_masks(comp_src, comp_dst, team_size)
    device = edge_probs.device

    m_home = masks["home_same"].to(device)
    m_away = masks["away_same"].to(device)
    m_cross = masks["cross_team"].to(device)
    m_out = (masks["to_outside"] | masks["outside_self"]).to(device)

    p_home = edge_probs[..., m_home].sum(dim=-1)
    p_away = edge_probs[..., m_away].sum(dim=-1)
    p_cross = edge_probs[..., m_cross].sum(dim=-1)
    p_out = edge_probs[..., m_out].sum(dim=-1)

    return p_home, p_away, p_cross, p_out


def fuse_n020_probabilities(
    dead_logit: torch.Tensor,
    team_logits: torch.Tensor,
    compressed_edge_logits: torch.Tensor | None = None,
    comp_src: torch.Tensor | None = None,
    comp_dst: torch.Tensor | None = None,
    path_weight: float = 0.5,
    team_size: int = 11,
) -> torch.Tensor:
    """Fuse multi-task predictions into calibrated [..., 3] class probabilities ordered [H, A, D].

    Args:
        dead_logit: [...] scalar logit for Dead ball.
        team_logits: [..., 2] conditional logits for [Home, Away].
        compressed_edge_logits: [..., 576] compressed edge logits (optional).
        comp_src: [576] source nodes.
        comp_dst: [576] destination nodes.
        path_weight: Scaling factor for path evidence fusion.
        team_size: 11 players per team.

    Returns:
        probs: [..., 3] tensor of float probabilities ordered [P(H), P(A), P(D)].
    """
    p_dead = torch.sigmoid(dead_logit).unsqueeze(-1)  # [..., 1]
    p_alive = 1.0 - p_dead

    fused_ha_logits = team_logits.clone()

    if (
        compressed_edge_logits is not None
        and comp_src is not None
        and comp_dst is not None
        and path_weight > 0.0
    ):
        edge_probs = torch.softmax(compressed_edge_logits, dim=-1)
        p_home_path, p_away_path, _, _ = aggregate_path_evidence(
            edge_probs, comp_src, comp_dst, team_size
        )
        # Shift conditional logits by path bias
        diff = (p_home_path - p_away_path).unsqueeze(-1)  # [..., 1]
        path_bias = torch.cat([diff, -diff], dim=-1)  # [..., 2]
        fused_ha_logits = fused_ha_logits + path_weight * path_bias

    p_ha = torch.softmax(fused_ha_logits, dim=-1)  # [..., 2]

    p_h = p_alive * p_ha[..., 0:1]
    p_a = p_alive * p_ha[..., 1:2]
    p_d = p_dead

    probs = torch.cat([p_h, p_a, p_d], dim=-1)  # [..., 3]
    # Re-normalize to guarantee sum == 1.0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs
