"""Tests for Dead-state classification and probabilistic fusion (N020)."""

import torch

from possession_lab.neural.pathcrf.edge_states import build_edge_compression_maps
from possession_lab.neural.pathcrf.fusion import fuse_n020_probabilities


def test_fuse_probabilities_normalization():
    B, T = 2, 10
    dead_logit = torch.randn(B, T)
    team_logits = torch.randn(B, T, 2)
    edge_logits = torch.randn(B, T, 576)

    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)

    probs = fuse_n020_probabilities(
        dead_logit,
        team_logits,
        compressed_edge_logits=edge_logits,
        comp_src=comp_src,
        comp_dst=comp_dst,
        path_weight=0.5,
    )

    assert probs.shape == (B, T, 3)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()

    # Sum across [H, A, D] must equal 1.0
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)


def test_dead_state_gating():
    # Test extreme dead logit (+100.0) -> p_dead must be ~1.0, p_h and p_a ~0.0
    dead_logit = torch.tensor([[100.0]])
    team_logits = torch.tensor([[[0.0, 0.0]]])

    probs = fuse_n020_probabilities(dead_logit, team_logits, path_weight=0.0)
    assert probs[0, 0, 2].item() > 0.9999
    assert probs[0, 0, 0].item() < 1e-4
    assert probs[0, 0, 1].item() < 1e-4

    # Test extreme alive logit (-100.0) with strong home preference
    dead_logit_alive = torch.tensor([[-100.0]])
    team_logits_home = torch.tensor([[[10.0, -10.0]]])

    probs_alive = fuse_n020_probabilities(dead_logit_alive, team_logits_home, path_weight=0.0)
    assert probs_alive[0, 0, 2].item() < 1e-4
    assert probs_alive[0, 0, 0].item() > 0.9999
    assert probs_alive[0, 0, 1].item() < 1e-4
