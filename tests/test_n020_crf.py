"""Numerical and algorithmic verification of DynamicSparseCRF (N020)."""

import pytest
import torch

from possession_lab.neural.pathcrf.crf import DynamicSparseCRF
from possession_lab.neural.pathcrf.edge_states import build_edge_compression_maps
from possession_lab.neural.pathcrf.transitions import build_allowed_mask, forbid_rate


@pytest.fixture
def crf_fixture():
    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)
    crf = DynamicSparseCRF(comp_src, comp_dst, edge_embed_dim=8, team_size=11)
    allowed_mask = build_allowed_mask(comp_src, comp_dst, team_size=11)
    return crf, allowed_mask, comp_src, comp_dst


def test_crf_logz_ge_gold_score(crf_fixture):
    crf, allowed_mask, comp_src, comp_dst = crf_fixture
    B, T, K = 2, 5, 576
    D = crf.edge_embed_dim

    emissions = torch.randn(B, T, K) * 0.1
    edge_embeds = torch.randn(B, T, K, D) * 0.1

    # Construct a guaranteed legal sequence of states
    # e.g., H0->H0 for all timesteps
    idx_h0 = int(torch.where((comp_src == 0) & (comp_dst == 0))[0][0].item())
    tags = torch.full((B, T), idx_h0, dtype=torch.long)

    logZ = crf._compute_logZ(emissions, edge_embeds)
    gold = crf._compute_gold_score(emissions, edge_embeds, tags)

    assert logZ.shape == (B,)
    assert gold.shape == (B,)

    # Log-partition must strictly be greater than or equal to any individual path score
    diff = logZ - gold
    assert (diff >= -1e-4).all(), f"logZ must be >= gold score, got diff {diff}"


def test_crf_viterbi_is_legal(crf_fixture):
    crf, allowed_mask, comp_src, comp_dst = crf_fixture
    B, T, K = 2, 8, 576
    D = crf.edge_embed_dim

    emissions = torch.randn(B, T, K)
    edge_embeds = torch.randn(B, T, K, D)

    best_paths = crf.decode(emissions, edge_embeds)

    assert best_paths.shape == (B, T)
    rate = forbid_rate(best_paths, allowed_mask)
    assert rate.item() == 0.0, f"Viterbi produced illegal transitions! Forbid rate: {rate.item()}"


def test_crf_gradients_finite(crf_fixture):
    crf, _, comp_src, comp_dst = crf_fixture
    B, T, K = 2, 4, 576
    D = crf.edge_embed_dim

    emissions = torch.randn(B, T, K, requires_grad=True)
    edge_embeds = torch.randn(B, T, K, D, requires_grad=True)

    idx_h0 = int(torch.where((comp_src == 0) & (comp_dst == 0))[0][0].item())
    tags = torch.full((B, T), idx_h0, dtype=torch.long)

    loss = crf.neg_log_likelihood(emissions, edge_embeds, tags)
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-4

    loss.backward()
    assert emissions.grad is not None and torch.isfinite(emissions.grad).all()
    assert edge_embeds.grad is not None and torch.isfinite(edge_embeds.grad).all()


def test_crf_marginals_normalize(crf_fixture):
    crf, _, _, _ = crf_fixture
    B, T, K = 2, 4, 576
    D = crf.edge_embed_dim

    emissions = torch.randn(B, T, K) * 0.1
    edge_embeds = torch.randn(B, T, K, D) * 0.1

    marginals = crf.compute_marginals(emissions, edge_embeds)

    assert marginals.shape == (B, T, K)
    assert (marginals >= 0.0).all()
    # Sum over K states must be 1.0 per frame
    sums = marginals.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
