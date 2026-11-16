"""Transition graph, legal path constraints, and incoming adjacency structures.

Ported from original PathCRF (Kim et al., KDD 2026, MPL-2.0).
Defines the 12-parameter transition rules between possession edges:
1. Player self-loop A->A: next edge must originate from A (self, teammate, opponent, outside).
2. Player transfer A->B: next edge must be same A->B (in-flight pass) or originate from B.
3. Player to outside A->O: next edge must be same A->O or O->O (ball leaves pitch).
4. Outside self-loop O->O: in upstream mode, only O->O (absorbing until sequence reset).
"""

from __future__ import annotations

import torch

TOTAL_PLAYERS: int = 22
PLAYERS_PER_TEAM: int = 11


def is_same_team(a: int, b: int, team_size: int = PLAYERS_PER_TEAM) -> bool:
    """Check if node indices a and b belong to the same team."""
    limit = 2 * team_size
    if a >= limit or b >= limit:
        return False
    return (a < team_size and b < team_size) or (a >= team_size and b >= team_size)


def edge_relation_type(sender: int, receiver: int, team_size: int = PLAYERS_PER_TEAM) -> str:
    """Classify the receiver as self, teammate, opponent, or out."""
    limit = 2 * team_size
    if receiver == sender:
        return "self"
    if receiver >= limit:
        return "out"
    return "teammate" if is_same_team(sender, receiver, team_size) else "opponent"


def build_allowed_mask(
    comp_src: torch.Tensor,
    comp_dst: torch.Tensor,
    team_size: int = PLAYERS_PER_TEAM,
    allow_outside_restart: bool = False,
) -> torch.Tensor:
    """Construct the [K, K] boolean transition legality matrix.

    Args:
        comp_src: [K] source node index for each state.
        comp_dst: [K] destination node index for each state.
        team_size: 11 players per team.
        allow_outside_restart: Allow outside states to transition to players at a restart.

    Returns:
        allowed: [K, K] boolean matrix where allowed[i, j] is True iff state i -> j is legal.
    """
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-nested-blocks,too-many-branches
    K = comp_src.numel()
    allowed = torch.zeros((K, K), dtype=torch.bool)
    limit = 2 * team_size

    src = comp_src.tolist()
    dst = comp_dst.tolist()

    for i in range(K):
        s_i = src[i]
        d_i = dst[i]

        # Case 4: Outside self-loop O->O (s_i >= 22)
        if s_i >= limit:
            if s_i == d_i:
                allowed[i, i] = True
                if allow_outside_restart:
                    # Can restart to any player self-loop or throw-in/corner/goal-kick
                    for j in range(K):
                        if dst[j] == src[j] < limit:
                            allowed[i, j] = True
            continue

        # Case 1: Player self-loop A->A
        if s_i == d_i:
            for j in range(K):
                if src[j] == s_i:
                    allowed[i, j] = True
            continue

        # Case 2: Player-to-player transfer A->B (B is a player)
        if d_i < limit:
            allowed[i, i] = True  # In-flight continuation
            for j in range(K):
                if src[j] == d_i:
                    allowed[i, j] = True
        else:
            # Case 3: Player to outside node A->O
            allowed[i, i] = True  # In-flight continuation
            for j in range(K):
                if src[j] == d_i and dst[j] == d_i:
                    allowed[i, j] = True
                    break

    return allowed


def build_allowed_next_lists(
    comp_src: torch.Tensor,
    comp_dst: torch.Tensor,
    team_size: int = PLAYERS_PER_TEAM,
) -> list[list[tuple[int, int]]]:
    """Build lists of (next_state_j, theta_type_id) for each state i."""
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-branches
    K = comp_src.numel()
    limit = 2 * team_size
    src = comp_src.tolist()
    dst = comp_dst.tolist()

    allowed_next: list[list[tuple[int, int]]] = [[] for _ in range(K)]

    for i in range(K):
        s_i = src[i]
        d_i = dst[i]

        if s_i >= limit and s_i == d_i:
            # Outside self-loop
            allowed_next[i].append((i, 11))
            continue

        if s_i >= limit and s_i != d_i:
            continue

        if s_i == d_i:
            # Player self-loop: next edge originates from s_i
            for j in range(K):
                if src[j] != s_i:
                    continue
                rel = edge_relation_type(s_i, dst[j], team_size)
                if rel == "self":
                    allowed_next[i].append((j, 0))
                elif rel == "teammate":
                    allowed_next[i].append((j, 1))
                elif rel == "opponent":
                    allowed_next[i].append((j, 2))
                elif rel == "out":
                    allowed_next[i].append((j, 3))
            continue

        if d_i < limit:
            # Player transfer A->B: same edge or originates from B
            allowed_next[i].append((i, 4))
            for j in range(K):
                if src[j] != d_i:
                    continue
                rel = edge_relation_type(d_i, dst[j], team_size)
                if rel == "self":
                    allowed_next[i].append((j, 5))
                elif rel == "teammate":
                    allowed_next[i].append((j, 6))
                elif rel == "opponent":
                    allowed_next[i].append((j, 7))
                elif rel == "out":
                    allowed_next[i].append((j, 8))
        else:
            # Player to outside A->O: same edge or O->O
            allowed_next[i].append((i, 9))
            for j in range(K):
                if src[j] == d_i and dst[j] == d_i:
                    allowed_next[i].append((j, 10))
                    break

    return allowed_next


def build_incoming_structures(
    comp_src: torch.Tensor,
    comp_dst: torch.Tensor,
    team_size: int = PLAYERS_PER_TEAM,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct incoming index, type, and mask tensors for compact CRF dynamic programming.

    Returns:
        inc_idx: [K, M] long tensor of incoming previous state indices (-1 padded).
        inc_type: [K, M] long tensor of transition parameter types.
        inc_mask: [K, M] boolean tensor, True where incoming edge is legal.
    """
    allowed_next = build_allowed_next_lists(comp_src, comp_dst, team_size)
    K = comp_src.numel()

    incoming: list[list[tuple[int, int]]] = [[] for _ in range(K)]
    for i in range(K):
        for j, t in allowed_next[i]:
            incoming[j].append((i, t))

    M = max(len(v) for v in incoming) if K > 0 else 1

    inc_idx = torch.full((K, M), -1, dtype=torch.long, device=device)
    inc_type = torch.full((K, M), -1, dtype=torch.long, device=device)
    inc_mask = torch.zeros((K, M), dtype=torch.bool, device=device)

    for j in range(K):
        pairs = incoming[j]
        if len(pairs) == 0:
            continue
        inc_idx[j, : len(pairs)] = torch.tensor(
            [p[0] for p in pairs], dtype=torch.long, device=device
        )
        inc_type[j, : len(pairs)] = torch.tensor(
            [p[1] for p in pairs], dtype=torch.long, device=device
        )
        inc_mask[j, : len(pairs)] = True

    return inc_idx, inc_type, inc_mask


def forbid_rate(pred_edges: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
    """Compute the fraction of illegal consecutive transitions in a predicted edge sequence."""
    if pred_edges.size(1) < 2:
        return torch.tensor(0.0, device=pred_edges.device)
    prev = pred_edges[:, :-1]
    nxt = pred_edges[:, 1:]
    valid = (prev >= 0) & (nxt >= 0)

    safe_prev = prev.clone()
    safe_nxt = nxt.clone()
    safe_prev[~valid] = 0
    safe_nxt[~valid] = 0

    allowed = allowed_mask[safe_prev, safe_nxt]
    forbid = (~valid) | (~allowed)
    return forbid.float().mean()
