"""Player-node mapping, slot stability, and pitch boundary node definitions.

Canonical 26-node configuration:
- Slots 0..10:  Home team (up to 11 players)
- Slots 11..21: Away team (up to 11 players)
- Slots 22..25: Outside pitch boundary nodes (left, right, bottom, top)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

TOTAL_NODES: int = 26
PLAYERS_PER_TEAM: int = 11
TOTAL_PLAYERS: int = 22
NUM_OUTSIDE: int = 4

OUT_LEFT_IDX: int = 22
OUT_RIGHT_IDX: int = 23
OUT_BOTTOM_IDX: int = 24
OUT_TOP_IDX: int = 25

OUTSIDE_LEFT: int = 22
OUTSIDE_RIGHT: int = 23
OUTSIDE_BOTTOM: int = 24
OUTSIDE_TOP: int = 25

DEFAULT_PITCH_X: float = 105.0
DEFAULT_PITCH_Y: float = 68.0


@dataclass(frozen=True)
class OutsideNodeGeometry:
    """Pitch boundary node coordinates in meters."""

    pitch_x: float = DEFAULT_PITCH_X
    pitch_y: float = DEFAULT_PITCH_Y

    @property
    def centered_coords(self) -> NDArray[Any]:
        """Outside coordinates in pitch-centered meters [-L/2, +L/2]."""
        hx = self.pitch_x / 2.0
        hy = self.pitch_y / 2.0
        return np.array(
            [
                [-hx, 0.0],  # out_left
                [hx, 0.0],  # out_right
                [0.0, -hy],  # out_bottom
                [0.0, hy],  # out_top
            ],
            dtype=np.float32,
        )

    @property
    def standard_coords(self) -> NDArray[Any]:
        """Outside coordinates in standard [0, L] meters matching upstream PathCRF."""
        return np.array(
            [
                [0.0, self.pitch_y / 2.0],  # out_left
                [self.pitch_x, self.pitch_y / 2.0],  # out_right
                [self.pitch_x / 2.0, 0.0],  # out_bottom
                [self.pitch_x / 2.0, self.pitch_y],  # out_top
            ],
            dtype=np.float32,
        )


def select_stable_team_slots(
    team_pos: NDArray[Any],
    target_slots: int = PLAYERS_PER_TEAM,
) -> list[int]:
    """Select up to `target_slots` active slots deterministically by observation frequency.

    Args:
        team_pos: Array of shape [T, N_slots, 2].
        target_slots: Maximum number of slots (11).

    Returns:
        List of selected slot indices, sorted in ascending order.
    """
    n_slots = team_pos.shape[1]
    if n_slots <= target_slots:
        return list(range(n_slots))

    # Compute observation counts for each slot across all frames
    valid_counts = np.isfinite(team_pos[:, :, 0]).sum(axis=0)
    # Rank slots: highest count first, tiebreak with slot index
    ranked = sorted(range(n_slots), key=lambda idx: (-valid_counts[idx], idx))
    selected = sorted(ranked[:target_slots])
    return selected


def map_canonical_players_to_26_nodes(
    home_pos: NDArray[Any],
    away_pos: NDArray[Any],
    pitch: tuple[float, float] = (DEFAULT_PITCH_X, DEFAULT_PITCH_Y),
    standard_coords: bool = True,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Map canonical home and away tracking arrays to a unified 26-node representation.

    Args:
        home_pos: Array of shape [T, Nh, 2] in pitch-centered meters.
        away_pos: Array of shape [T, Na, 2] in pitch-centered meters.
        pitch: (length, width) in meters.
        standard_coords: If True, shift centered coords [-L/2, +L/2] to [0, L].

    Returns:
        nodes_xy: Array of shape [T, 26, 2], float32. NaNs filled with 0.0.
        node_valid: Boolean array of shape [T, 26], True where node is validly observed.
    """
    T = home_pos.shape[0]
    pitch_x, pitch_y = pitch
    geo = OutsideNodeGeometry(pitch_x=pitch_x, pitch_y=pitch_y)

    nodes_xy = np.zeros((T, TOTAL_NODES, 2), dtype=np.float32)
    node_valid = np.zeros((T, TOTAL_NODES), dtype=bool)

    # Shift offset if standard coordinates [0, L] are requested
    x_shift = pitch_x / 2.0 if standard_coords else 0.0
    y_shift = pitch_y / 2.0 if standard_coords else 0.0

    # 1. Map Home team (slots 0..10)
    home_slots = select_stable_team_slots(home_pos, PLAYERS_PER_TEAM)
    for i, s in enumerate(home_slots):
        slot_xy = home_pos[:, s, :2]
        is_val = np.isfinite(slot_xy[:, 0]) & np.isfinite(slot_xy[:, 1])
        node_valid[:, i] = is_val
        if np.any(is_val):
            nodes_xy[is_val, i, 0] = slot_xy[is_val, 0] + x_shift
            nodes_xy[is_val, i, 1] = slot_xy[is_val, 1] + y_shift

    # 2. Map Away team (slots 11..21)
    away_slots = select_stable_team_slots(away_pos, PLAYERS_PER_TEAM)
    for i, s in enumerate(away_slots):
        target_idx = PLAYERS_PER_TEAM + i
        slot_xy = away_pos[:, s, :2]
        is_val = np.isfinite(slot_xy[:, 0]) & np.isfinite(slot_xy[:, 1])
        node_valid[:, target_idx] = is_val
        if np.any(is_val):
            nodes_xy[is_val, target_idx, 0] = slot_xy[is_val, 0] + x_shift
            nodes_xy[is_val, target_idx, 1] = slot_xy[is_val, 1] + y_shift

    # 3. Outside nodes (slots 22..25) - always valid
    out_coords = geo.standard_coords if standard_coords else geo.centered_coords
    for o in range(NUM_OUTSIDE):
        target_idx = TOTAL_PLAYERS + o
        nodes_xy[:, target_idx, :] = out_coords[o]
        node_valid[:, target_idx] = True

    return nodes_xy, node_valid
