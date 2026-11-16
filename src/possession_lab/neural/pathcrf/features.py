"""NaN-safe player kinematics and 8-channel feature representation for PathCRF.

Channels per node:
0: Team indicator (-1.0 Home, +1.0 Away, 0.0 Outside)
1: Goalkeeper indicator (1.0 GK, 0.0 Field player)
2: x coordinate (standard meters [0, 105.0])
3: y coordinate (standard meters [0, 68.0])
4: vx velocity along x (m/s)
5: vy velocity along y (m/s)
6: speed (m/s)
7: acceleration magnitude (m/s^2)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .nodes import (
    PLAYERS_PER_TEAM,
    TOTAL_NODES,
    TOTAL_PLAYERS,
)

FEATURE_DIM: int = 8
MAX_PLAYER_SPEED: float = 12.0  # m/s (~43 km/h physical human sprint ceiling)
MAX_PLAYER_ACCEL: float = 15.0  # m/s^2


def compute_node_kinematics(
    nodes_xy: NDArray[Any],
    node_valid: NDArray[Any],
    period: NDArray[Any] | None = None,
    fps: float = 25.0,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute vx, vy, speed, and acceleration per node respecting periods and observation validity.

    Args:
        nodes_xy: [T, 26, 2] positions in meters.
        node_valid: [T, 26] boolean observation mask.
        period: [T] period indicator (0 for gaps/boundary frames).
        fps: Sampling rate in Hz.

    Returns:
        vx: [T, 26] in m/s.
        vy: [T, 26] in m/s.
        speed: [T, 26] in m/s.
        accel: [T, 26] in m/s^2.
    """
    T, N, _ = nodes_xy.shape
    vx = np.zeros((T, N), dtype=np.float32)
    vy = np.zeros((T, N), dtype=np.float32)
    speed = np.zeros((T, N), dtype=np.float32)
    accel = np.zeros((T, N), dtype=np.float32)

    dt = 1.0 / max(float(fps), 1e-4)

    # Players only (outside nodes remain static with 0 velocity/accel)
    for n in range(TOTAL_PLAYERS):
        is_val = node_valid[:, n]
        if not np.any(is_val):
            continue

        # Valid continuous transitions: consecutive frames both valid and in same period
        valid_trans = is_val[:-1] & is_val[1:]
        if period is not None:
            same_period = (period[:-1] == period[1:]) & (period[:-1] > 0)
            valid_trans = valid_trans & same_period

        if not np.any(valid_trans):
            continue

        trans_idx = np.where(valid_trans)[0]

        dx = nodes_xy[trans_idx + 1, n, 0] - nodes_xy[trans_idx, n, 0]
        dy = nodes_xy[trans_idx + 1, n, 1] - nodes_xy[trans_idx, n, 1]

        v_x = dx / dt
        v_y = dy / dt
        spd = np.sqrt(v_x**2 + v_y**2)

        # Clip unphysical teleports
        spd_clipped = np.clip(spd, 0.0, MAX_PLAYER_SPEED)
        scale = np.where(spd > 1e-5, spd_clipped / (spd + 1e-8), 1.0)
        v_x = v_x * scale
        v_y = v_y * scale

        vx[trans_idx + 1, n] = v_x
        vy[trans_idx + 1, n] = v_y
        speed[trans_idx + 1, n] = spd_clipped

        # Acceleration from consecutive valid velocity steps
        acc_valid = valid_trans[:-1] & valid_trans[1:]
        if np.any(acc_valid):
            acc_idx = np.where(acc_valid)[0] + 1
            dvx = vx[acc_idx + 1, n] - vx[acc_idx, n]
            dvy = vy[acc_idx + 1, n] - vy[acc_idx, n]
            a_mag = np.sqrt(dvx**2 + dvy**2) / dt
            accel[acc_idx + 1, n] = np.clip(a_mag, 0.0, MAX_PLAYER_ACCEL)

    return vx, vy, speed, accel


def detect_goalkeepers_by_depth(
    nodes_xy: NDArray[Any],
    node_valid: NDArray[Any],
    pitch_x: float = 105.0,
) -> tuple[int, int]:
    """Identify goalkeeper slot per team based on mean defensive extremity.

    Args:
        nodes_xy: [T, 26, 2] positions.
        node_valid: [T, 26] validity mask.
        pitch_x: Pitch length in meters.

    Returns:
        (home_gk_slot, away_gk_slot) in 0..10 and 11..21.
    """
    home_means = []
    for i in range(PLAYERS_PER_TEAM):
        val = node_valid[:, i]
        if np.any(val):
            home_means.append(np.mean(nodes_xy[val, i, 0]))
        else:
            home_means.append(pitch_x / 2.0)

    away_means = []
    for i in range(PLAYERS_PER_TEAM):
        idx = PLAYERS_PER_TEAM + i
        val = node_valid[:, idx]
        if np.any(val):
            away_means.append(np.mean(nodes_xy[val, idx, 0]))
        else:
            away_means.append(pitch_x / 2.0)

    home_team_center = np.mean(home_means)
    away_team_center = np.mean(away_means)

    # If home attacks right (center < away), home GK is at minimum x; otherwise maximum x
    if home_team_center <= away_team_center:
        home_gk = int(np.argmin(home_means))
        away_gk = PLAYERS_PER_TEAM + int(np.argmax(away_means))
    else:
        home_gk = int(np.argmax(home_means))
        away_gk = PLAYERS_PER_TEAM + int(np.argmin(away_means))

    return home_gk, away_gk


def build_node_features(
    nodes_xy: NDArray[Any],
    node_valid: NDArray[Any],
    period: NDArray[Any] | None = None,
    fps: float = 25.0,
    pitch_x: float = 105.0,
) -> NDArray[Any]:
    """Build the canonical 8-channel feature tensor [T, 26, 8].

    Args:
        nodes_xy: [T, 26, 2] positions.
        node_valid: [T, 26] boolean validity mask.
        period: [T] period array.
        fps: Frame rate.
        pitch_x: Pitch length.

    Returns:
        features: [T, 26, 8] float32 tensor. All unobserved nodes are zeroed.
    """
    T, _, _ = nodes_xy.shape
    features = np.zeros((T, TOTAL_NODES, FEATURE_DIM), dtype=np.float32)

    # Kinematics
    vx, vy, speed, accel = compute_node_kinematics(nodes_xy, node_valid, period=period, fps=fps)

    # Team indicators (channel 0)
    features[:, :PLAYERS_PER_TEAM, 0] = -1.0  # Home
    features[:, PLAYERS_PER_TEAM:TOTAL_PLAYERS, 0] = 1.0  # Away
    features[:, TOTAL_PLAYERS:, 0] = 0.0  # Outside

    # Goalkeeper indicators (channel 1)
    home_gk, away_gk = detect_goalkeepers_by_depth(nodes_xy, node_valid, pitch_x=pitch_x)
    features[:, home_gk, 1] = 1.0
    features[:, away_gk, 1] = 1.0

    # Coordinates (channels 2, 3)
    features[:, :, 2] = nodes_xy[:, :, 0]
    features[:, :, 3] = nodes_xy[:, :, 1]

    # Kinematics (channels 4, 5, 6, 7)
    features[:, :, 4] = vx
    features[:, :, 5] = vy
    features[:, :, 6] = speed
    features[:, :, 7] = accel

    # Mask out invalid observations
    features[~node_valid] = 0.0

    return features
