"""Edge state compression (576 states) and state-to-team semantic mapping.

Original 26 nodes:
- 0..10:  Home players (11)
- 11..21: Away players (11)
- 22..25: Outside pitch boundary nodes (4)

Dense edge space: 26 * 26 = 676 directed edges.
Compressed valid edge space: 22 * 26 + 4 = 576 valid states (K=576).
Disallowed: non-self edges originating from an outside node (4 * 25 = 100 edges).
"""

from __future__ import annotations

import torch

TOTAL_NODES: int = 26
TOTAL_PLAYERS: int = 22
NUM_OUTSIDES: int = 4
COMPRESSED_STATES: int = 576
DENSE_EDGES: int = 676


def build_edge_compression_maps(
    n_players: int = TOTAL_PLAYERS,
    n_outsides: int = NUM_OUTSIDES,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct bijective compression mapping between 676 dense edges and 576 valid states.

    Returns:
        valid_orig_edge_ids: [576] long tensor of original dense edge IDs that are kept.
        orig2comp: [676] long tensor mapping dense edge ID -> compressed state in [0..575], or -1.
        comp_src: [576] long tensor of source node IDs in [0..25].
        comp_dst: [576] long tensor of destination node IDs in [0..25].
    """
    n_nodes = n_players + n_outsides  # 26
    K = n_players * n_nodes + n_outsides  # 22 * 26 + 4 = 576

    valid_orig_edge_ids = []
    comp_src = []
    comp_dst = []

    # All directed edges originating from a player (22 * 26 = 572)
    for s in range(n_players):
        for d in range(n_nodes):
            valid_orig_edge_ids.append(s * n_nodes + d)
            comp_src.append(s)
            comp_dst.append(d)

    # Self-loops of outside nodes (4)
    for o in range(n_players, n_nodes):
        valid_orig_edge_ids.append(o * n_nodes + o)
        comp_src.append(o)
        comp_dst.append(o)

    valid_orig_edge_ids_t = torch.tensor(valid_orig_edge_ids, dtype=torch.long, device=device)
    comp_src_t = torch.tensor(comp_src, dtype=torch.long, device=device)
    comp_dst_t = torch.tensor(comp_dst, dtype=torch.long, device=device)

    orig2comp_t = torch.full((n_nodes * n_nodes,), -1, dtype=torch.long, device=device)
    orig2comp_t[valid_orig_edge_ids_t] = torch.arange(K, dtype=torch.long, device=device)

    return valid_orig_edge_ids_t, orig2comp_t, comp_src_t, comp_dst_t


def get_edge_semantic_masks(
    comp_src: torch.Tensor,
    comp_dst: torch.Tensor,
    team_size: int = 11,
) -> dict[str, torch.Tensor]:
    """Classify all 576 compressed edge states into semantic possession categories.

    Args:
        comp_src: [576] source node IDs.
        comp_dst: [576] destination node IDs.
        team_size: Players per team (11).

    Returns:
        dict of boolean masks [576]:
        - home_same: H_i -> H_j (unambiguous Home)
        - away_same: A_i -> A_j (unambiguous Away)
        - cross_ha:  H_i -> A_j (turnover/tackle/cross-team)
        - cross_ah:  A_i -> H_j (turnover/tackle/cross-team)
        - to_outside: Player -> Outside
        - outside_self: Outside -> Outside
    """
    limit = 2 * team_size

    is_src_home = comp_src < team_size
    is_dst_home = comp_dst < team_size

    is_src_away = (comp_src >= team_size) & (comp_src < limit)
    is_dst_away = (comp_dst >= team_size) & (comp_dst < limit)

    is_src_out = comp_src >= limit
    is_dst_out = comp_dst >= limit

    home_same = is_src_home & is_dst_home
    away_same = is_src_away & is_dst_away
    cross_ha = is_src_home & is_dst_away
    cross_ah = is_src_away & is_dst_home
    to_outside = (comp_src < limit) & is_dst_out
    outside_self = is_src_out & is_dst_out

    return {
        "home_same": home_same,
        "away_same": away_same,
        "cross_ha": cross_ha,
        "cross_ah": cross_ah,
        "cross_team": cross_ha | cross_ah,
        "to_outside": to_outside,
        "outside_self": outside_self,
    }
