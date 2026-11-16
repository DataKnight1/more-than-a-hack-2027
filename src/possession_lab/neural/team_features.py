"""Engineered team-behaviour features per frame (shared by the EDA and the model).

Football hypotheses each feature encodes (tested in the EDA, ``scripts/neural/eda_possession.py``):

- the team in possession has a player *at* the ball (short nearest distance) and opponents
  *close in* on it (positive closing speed = moving towards the ball);
- the team in possession moves *forward* (towards the opponent goal) and *spreads out* (width and
  length grow) to create space; the defending team compacts and drops;
- dead ball shows a collective *slow-down* (low mean speed, few high-intensity runs) and the ball
  near a touch/goal line or static;
- possession changes show a reversal of team forward velocity and a burst of closing speed.

All features are relative to each team's **attacking direction**, estimated per period from the
team's mean position (a team defends its own half on average) — computed from tracking only, so
the same algorithm applies unchanged to any provider and to the challenge ATD at inference.
Missing ball positions use the last-known ball (forward-filled); missing players are excluded.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
FPS = 25.0
HIGH_INTENSITY = 5.5  # m/s (19.8 km/h)
SPRINT = 7.0  # m/s (25.2 km/h)

TEAM_FEATURES = (
    "n_present",
    "centroid_fwd",
    "centroid_y",
    "centroid_vel_fwd",
    "spread_length",
    "spread_width",
    "mean_speed",
    "frac_high_intensity",
    "frac_sprint",
    "nearest_ball_dist",
    "n_within_5m",
    "n_within_10m",
    "closing_speed_near3",
    "closing_speed_max",
    "frac_running_fwd",
    "n_within_3m",
)
GLOBAL_FEATURES = ("ball_speed", "ball_fwd_home", "ball_near_line", "ball_age_s", "all_mean_speed")
N_TEAM = len(TEAM_FEATURES)
N_CONTEXT = 2 * N_TEAM + len(GLOBAL_FEATURES)
# Approximate unit scales so every input is O(1) for the network.
_TEAM_SCALE = np.array([11, 52.5, 34, 5, 20, 20, 5, 1, 1, 20, 5, 11, 5, 8, 1, 3], np.float32)
_GLOBAL_SCALE = np.array([20, 52.5, 1, 2, 5], np.float32)


def velocity(pos: F32, span: int = 2) -> F32:
    """Central-difference velocity (m/s) over ``+/- span`` frames; NaN where an end is missing."""
    pad = np.concatenate([np.repeat(pos[:1], span, 0), pos, np.repeat(pos[-1:], span, 0)])
    return cast(
        F32,
        ((pad[2 * span :] - pad[: -2 * span]) * (FPS / (2 * span))).astype(np.float32),
    )


def attack_direction(home: F32, away: F32, period: NDArray[np.int8]) -> F32:
    """Per-frame +1 if home attacks towards +x, else -1 (estimated per period; 0 on gaps)."""
    out = np.zeros(len(period), np.float32)
    for p in np.unique(period):
        if p == 0:
            continue
        sel = period == p
        hx = np.nanmean(home[sel, :, 0])
        ax = np.nanmean(away[sel, :, 0])
        out[sel] = 1.0 if hx < ax else -1.0  # the team deeper in -x attacks +x
    return out


def last_known(ball_xy: F32) -> tuple[F32, F32]:
    """Forward-filled (then back-filled) ball position and seconds since last observation."""
    n = len(ball_xy)
    seen = ~np.isnan(ball_xy[:, 0])
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    filled = np.zeros_like(ball_xy)
    ok = last >= 0
    filled[ok] = ball_xy[last[ok]]
    if seen.any():
        filled[~ok] = ball_xy[np.flatnonzero(seen)[0]]
    age = np.where(ok, (np.arange(n) - last) / FPS, 10.0).astype(np.float32)
    return cast(F32, np.nan_to_num(filled).astype(np.float32)), age


def _team_block(players: F32, ball: F32, direction: F32) -> F32:
    """Raw (unscaled) ``[T, N_TEAM]`` features for one team."""
    t = len(players)
    present = ~np.isnan(players[..., 0])
    count = present.sum(1).astype(np.float32)
    safe = np.maximum(count, 1)
    pos = np.nan_to_num(players)
    vel = np.nan_to_num(velocity(players))
    speed = np.hypot(vel[..., 0], vel[..., 1])
    fwd_x = pos[..., 0] * direction[:, None]
    fwd_vx = vel[..., 0] * direction[:, None]

    def tmean(v: NDArray[np.generic]) -> F32:
        """Compute each frame's mean over present players, excluding missing tracks."""
        return cast(F32, (v * present).sum(1) / safe)

    cx = tmean(pos[..., 0])
    cy = tmean(pos[..., 1])
    sx = np.sqrt(np.maximum(tmean(pos[..., 0] ** 2) - cx**2, 0))
    sy = np.sqrt(np.maximum(tmean(pos[..., 1] ** 2) - cy**2, 0))
    rel = ball[:, None, :] - pos
    dist = np.hypot(rel[..., 0], rel[..., 1])
    dist_m = np.where(present, dist, np.inf)
    unit = rel / np.maximum(dist, 1e-3)[..., None]
    closing = (vel * unit).sum(-1)
    order = np.argsort(dist_m, axis=1)[:, :3]
    near3 = np.take_along_axis(np.where(present, closing, np.nan), order, axis=1)
    out = np.zeros((t, N_TEAM), np.float32)
    out[:, 0] = count
    out[:, 1] = tmean(fwd_x)
    out[:, 2] = cy
    out[:, 3] = tmean(fwd_vx)
    out[:, 4] = sx
    out[:, 5] = sy
    out[:, 6] = tmean(speed)
    out[:, 7] = tmean(speed > HIGH_INTENSITY)
    out[:, 8] = tmean(speed > SPRINT)
    out[:, 9] = np.where(count > 0, np.min(dist_m, axis=1), 60.0)
    out[:, 10] = (dist_m < 5).sum(1)
    out[:, 11] = (dist_m < 10).sum(1)
    out[:, 12] = np.nan_to_num(np.nanmean(near3, axis=1))
    out[:, 13] = np.where(count > 0, np.max(np.where(present, closing, -np.inf), axis=1), 0.0)
    out[:, 14] = tmean(fwd_vx > 2.0)
    out[:, 15] = (dist_m < 3).sum(1)
    return out


def context_features(ball: F32, home: F32, away: F32, home_dir: F32, scaled: bool = True) -> F32:
    """``[T, N_CONTEXT]`` = home block, away block, global block (scaled to O(1) if ``scaled``).

    ``home_dir`` is +1/-1 per frame (``attack_direction``); away attacks the opposite way.
    """
    lk, age = last_known(ball[:, :2])
    hb = _team_block(home, lk, home_dir)
    ab = _team_block(away, lk, -home_dir)
    bvel = np.nan_to_num(velocity(ball[:, :2]))
    g = np.zeros((len(ball), len(GLOBAL_FEATURES)), np.float32)
    g[:, 0] = np.hypot(bvel[:, 0], bvel[:, 1]) * (~np.isnan(ball[:, 0]))
    g[:, 1] = lk[:, 0] * home_dir
    g[:, 2] = ((np.abs(lk[:, 0]) > 50.5) | (np.abs(lk[:, 1]) > 32.0)).astype(np.float32)
    g[:, 3] = np.minimum(age, 10.0)
    both = np.concatenate([home, away], axis=1)
    bspeed = np.hypot(*np.moveaxis(np.nan_to_num(velocity(both)), -1, 0))
    bpresent = ~np.isnan(both[..., 0])
    g[:, 4] = (bspeed * bpresent).sum(1) / np.maximum(bpresent.sum(1), 1)
    if not scaled:
        return np.concatenate([hb, ab, g], axis=1)
    feats = np.concatenate([hb / _TEAM_SCALE, ab / _TEAM_SCALE, g / _GLOBAL_SCALE], axis=1)
    return cast(F32, np.clip(np.nan_to_num(feats), -5, 5).astype(np.float32))
