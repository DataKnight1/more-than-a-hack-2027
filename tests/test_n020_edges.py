"""Tests for 576-state edge compression, semantic masks, and transition graph (N020)."""

import torch

from possession_lab.neural.pathcrf.edge_states import (
    build_edge_compression_maps,
    get_edge_semantic_masks,
)
from possession_lab.neural.pathcrf.transitions import (
    build_allowed_mask,
    build_incoming_structures,
    forbid_rate,
)


def test_edge_compression_bijection():
    valid_orig_ids, orig2comp, comp_src, comp_dst = build_edge_compression_maps(22, 4)

    assert valid_orig_ids.numel() == 576
    assert orig2comp.numel() == 676
    assert comp_src.numel() == 576
    assert comp_dst.numel() == 576

    # Invertibility: orig2comp[valid_orig_ids] == 0..575
    recovered = orig2comp[valid_orig_ids]
    expected = torch.arange(576, dtype=torch.long)
    assert torch.equal(recovered, expected)

    # All non-valid original edges must be -1
    disallowed_mask = torch.ones(676, dtype=torch.bool)
    disallowed_mask[valid_orig_ids] = False
    assert (orig2comp[disallowed_mask] == -1).all()
    assert disallowed_mask.sum().item() == 100  # 4 outside nodes * 25 non-self edges


def test_edge_semantic_masks():
    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)
    masks = get_edge_semantic_masks(comp_src, comp_dst, team_size=11)

    m_home = masks["home_same"]
    m_away = masks["away_same"]
    m_cross = masks["cross_team"]
    m_to_out = masks["to_outside"]
    m_out_self = masks["outside_self"]

    # Check exact counts:
    # Home same: 11 * 11 = 121
    assert m_home.sum().item() == 121
    # Away same: 11 * 11 = 121
    assert m_away.sum().item() == 121
    # Cross team: 11 * 11 * 2 = 242
    assert m_cross.sum().item() == 242
    # Player to outside: 22 * 4 = 88
    assert m_to_out.sum().item() == 88
    # Outside self: 4
    assert m_out_self.sum().item() == 4

    total_accounted = (
        m_home.long() + m_away.long() + m_cross.long() + m_to_out.long() + m_out_self.long()
    )
    assert torch.equal(total_accounted, torch.ones(576, dtype=torch.long))


def test_allowed_mask_structure():
    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)
    allowed = build_allowed_mask(comp_src, comp_dst, team_size=11)

    assert allowed.shape == (576, 576)
    assert allowed.dtype == torch.bool

    # Case 1: Player self-loop (e.g. H0 -> H0, src=0, dst=0)
    # Next edge must originate from H0 (src=0). There are 26 such edges.
    idx_h0_self = int(torch.where((comp_src == 0) & (comp_dst == 0))[0][0].item())
    allowed_from_h0_self = allowed[idx_h0_self]
    assert allowed_from_h0_self.sum().item() == 26
    assert (comp_src[allowed_from_h0_self] == 0).all()

    # Case 4: Outside self loop (e.g. out_left -> out_left, src=22, dst=22)
    # In upstream mode, only out_left -> out_left is allowed next
    idx_out_self = int(torch.where((comp_src == 22) & (comp_dst == 22))[0][0].item())
    assert allowed[idx_out_self].sum().item() == 1
    assert allowed[idx_out_self, idx_out_self]


def test_incoming_structures():
    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)
    inc_idx, inc_type, inc_mask = build_incoming_structures(comp_src, comp_dst, team_size=11)

    assert inc_idx.shape[0] == 576
    assert inc_type.shape[0] == 576
    assert inc_mask.shape[0] == 576

    M = inc_idx.shape[1]
    assert M == 23  # Max incoming degree in PathCRF 576-state graph

    # For each state j, verify that every marked incoming edge i satisfies allowed[i, j]
    allowed = build_allowed_mask(comp_src, comp_dst, team_size=11)
    for j in range(576):
        num_inc = int(inc_mask[j].sum().item())
        prev_indices = inc_idx[j, :num_inc]
        assert allowed[prev_indices, j].all()


def test_forbid_rate():
    _, _, comp_src, comp_dst = build_edge_compression_maps(22, 4)
    allowed = build_allowed_mask(comp_src, comp_dst, team_size=11)

    # Self-loops H0->H0 repeated 10 times is 100% legal
    idx_h0_self = int(torch.where((comp_src == 0) & (comp_dst == 0))[0][0].item())
    legal_seq = torch.full((1, 10), idx_h0_self, dtype=torch.long)
    assert forbid_rate(legal_seq, allowed).item() == 0.0

    # Jumping from H0->H0 to A0->A0 without transfer edge is illegal
    idx_a0_self = int(torch.where((comp_src == 11) & (comp_dst == 11))[0][0].item())
    illegal_seq = torch.tensor([[idx_h0_self, idx_a0_self]], dtype=torch.long)
    assert forbid_rate(illegal_seq, allowed).item() == 1.0
