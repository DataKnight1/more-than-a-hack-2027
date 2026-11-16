"""Individual-player, marking and ball-touch features (feature version 3).

Each feature encodes a finding of ``experiments/research/neural/EDA_INDIVIDUAL.md`` (external
training matches only), motivated by ``experiments/research/neural/ERROR_ANALYSIS.md`` (missed
short, contested possessions):

- **ball touches**: at a possession change the ball's acceleration peaks (≈ 30–37 m/s² vs ≈ 10 in
  open play) and it turns; the team of the nearest player at each detected touch, and whether it
  differs from the previous touch, locate changes to the frame;
- **carrier pressure**: the chance the opponent wins the ball within 2 s rises from 1.4 % (nearest
  opponent > 10 m from the carrier) to 26 % (< 1 m), and from 4 % (no opponent within 5 m) to 26 %
  (3+);
- **contested balls**: short (< 2 s) turnovers happen with the losing team still close (2.5 m vs
  4.0 m) and more players within 3 m of the ball;
- **marking and roles** (identity-free, per frame): the in-possession striker is marked tightly
  (3.3–3.7 m vs 5.0–5.4 m when defending), the in-possession team is further from its markers
  overall, and its full-backs are wider and higher; the last defender is higher.

All quantities come from tracking positions only and are computed identically on every provider
and on the challenge ATD at inference. Missing values are zero-filled with explicit presence or
age features elsewhere in the input.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.team_features import last_known, velocity

F32 = NDArray[np.float32]
Array = NDArray[Any]
FPS = 25.0

TEAM_IND_FEATURES = (
    "press_on_nearest",  # distance from the team's ball-nearest player to its nearest opponent
    "opp_within5_of_nearest",
    "striker_marker_dist",
    "mean_marker_dist",
    "fullback_width",
    "last_defender_fwd",
    "striker_fwd",
    "nearest_player_accel",
)
GLOBAL_IND_FEATURES = (
    "ball_accel",
    "ball_turn",
    "touch",
    "last_touch_team",  # +1 home, -1 away, 0 none in window
    "time_since_touch",
    "touch_team_change",
    "next_touch_team",
    "time_to_touch",
    "players_within3_ball",
)
N_TEAM_IND = len(TEAM_IND_FEATURES)
N_IND = 2 * N_TEAM_IND + len(GLOBAL_IND_FEATURES)
PLAYER_IND_FEATURES = 4  # nearest-opponent distance, forward rank, is team ball-nearest, accel
_TEAM_SCALE = np.array([10, 3, 10, 10, 40, 52.5, 52.5, 5], np.float32)
_GLOBAL_SCALE = np.array([30, 180, 1, 1, 5, 1, 1, 5, 3], np.float32)
_INF = np.float32(1e6)


def _hyp(v: Array) -> Array:
    """Compute the Euclidean norm of the first two vector components."""
    return np.hypot(v[..., 0], v[..., 1])


def ball_kinematics(ball_xy: F32) -> tuple[F32, F32]:
    """Ball acceleration and turning angle over a three-frame radius.

    Acceleration is in m/s²; the angle is in degrees. Unseen values are NaN.
    """
    v = velocity(ball_xy, 2)
    acc = _hyp(velocity(v, 2))
    k = 3
    vp = np.concatenate([np.full((k, 2), np.nan, np.float32), v[:-k]])
    vn = np.concatenate([v[k:], np.full((k, 2), np.nan, np.float32)])
    norm = _hyp(vp) * _hyp(vn)
    turn = np.degrees(np.arccos(np.clip((vp * vn).sum(1) / np.maximum(norm, 1e-6), -1, 1)))
    turn = np.where((_hyp(vp) < 1.5) & (_hyp(vn) < 1.5), 0.0, turn)
    return acc.astype(np.float32), turn.astype(np.float32)


def touch_candidates(acc: F32, turn: F32) -> NDArray[np.bool_]:
    """Local acceleration maxima (+/-3 frames) that are strong (> 12 m/s^2) or turn the ball."""
    a = np.nan_to_num(acc, nan=0.0)
    n = len(a)
    pad = np.concatenate([np.zeros(3), a, np.zeros(3)])
    local = np.ones(n, bool)
    for s in range(7):
        if s != 3:
            local &= a >= pad[s : s + n]
    strong = (a > 12.0) | ((np.nan_to_num(turn) > 25.0) & (a > 5.0))
    return np.asarray(local & strong & np.isfinite(acc), dtype=np.bool_)


def _dist_matrix(p: F32, q: F32) -> F32:
    """``[T, Np, Nq]`` distances; ``_INF`` where either side is missing."""
    d = np.linalg.norm(p[:, :, None, :] - q[:, None, :, :], axis=-1)
    return np.where(np.isnan(d), _INF, d).astype(np.float32)


def _team_block(own: F32, ball: F32, direction: F32, d_oo: F32) -> tuple[F32, dict[str, F32]]:
    """Raw ``[T, N_TEAM_IND]`` features for one team plus per-player arrays."""
    t, n = own.shape[:2]
    r = np.arange(t)
    present = ~np.isnan(own[..., 0])
    count = present.sum(1)
    db = np.where(present, _hyp(own - ball[:, None, :]), _INF)
    nearest = np.argmin(db, 1)
    marker = np.min(d_oo, axis=2)  # [T, n] nearest opponent distance
    marker = np.where(present & (marker < _INF), marker, np.nan)
    fwd = np.where(present, own[..., 0] * direction[:, None], np.nan)
    acc = _hyp(velocity(velocity(own, 2), 2))
    out = np.zeros((t, N_TEAM_IND), np.float32)
    ok = count > 0
    press = marker[r, nearest]
    out[:, 0] = np.where(ok & np.isfinite(press), press, 30.0)
    out[:, 1] = np.where(ok, (d_oo[r, nearest] < 5).sum(1), 0)
    big = count >= 6
    fwd_hi = np.where(np.isnan(fwd), -np.inf, fwd)
    striker = np.argmax(fwd_hi, 1)
    sm = marker[r, striker]
    out[:, 2] = np.where(big & np.isfinite(sm), sm, 0.0)
    with np.errstate(all="ignore"):
        out[:, 3] = np.where(ok, np.nan_to_num(np.nanmean(marker, 1), nan=0.0), 0.0)
    order = np.argsort(np.where(np.isnan(fwd), np.inf, fwd), 1)  # deepest first (goalkeeper)
    back = order[:, 1:5]
    yb = np.take_along_axis(own[..., 1], back, 1)
    out[:, 4] = (
        np.where(big, np.nan_to_num(np.nanmax(yb, 1) - np.nanmin(yb, 1)), 0.0) if n >= 5 else 0
    )
    out[:, 5] = np.where(big, np.nan_to_num(fwd[r, order[:, 1]]), 0.0) if n >= 2 else 0
    out[:, 6] = np.where(big, np.nan_to_num(fwd[r, striker]), 0.0)
    out[:, 7] = np.where(ok, np.nan_to_num(acc[r, nearest]), 0.0)
    # per-player: nearest-opponent distance, forward rank in team (0 deepest .. 1 most advanced),
    # is the team's ball-nearest player, acceleration magnitude
    rank = np.argsort(order, 1).astype(np.float32) / np.maximum(count[:, None] - 1, 1)
    is_near = np.zeros((t, n), np.float32)
    is_near[r, nearest] = ok
    per = np.stack(
        [np.nan_to_num(marker, nan=30.0), np.where(present, rank, 0), is_near, np.nan_to_num(acc)],
        -1,
    )
    return out, {"per_player": per}


def individual_features(
    ball: F32, home: F32, away: F32, home_dir: F32, scaled: bool = True
) -> tuple[F32, F32]:
    """``(context [T, N_IND], per_player [T, Nh + Na, PLAYER_IND_FEATURES])``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _individual_features(ball, home, away, home_dir, scaled)


def _individual_features(
    ball: F32, home: F32, away: F32, home_dir: F32, scaled: bool
) -> tuple[F32, F32]:
    """Build per-player role, marking, pressure, and touch features for both teams.

    Args:
        ball: Per-frame ball coordinates and optional height.
        home: Home player coordinates shaped ``[T, P, 2]``.
        away: Away player coordinates shaped ``[T, P, 2]``.
        home_dir: Home attack direction by frame.
        scaled: Whether output channels should be normalized for model input.

    Returns:
        Team-context features and concatenated per-player feature tensors.
    """
    t = len(ball)
    xy = ball[:, :2]
    lk, _ = last_known(xy)
    d_ha = _dist_matrix(home, away)
    hb, hp = _team_block(home, lk, home_dir, d_ha)
    ab, ap = _team_block(away, lk, -home_dir, np.transpose(d_ha, (0, 2, 1)))
    acc, turn = ball_kinematics(xy)
    touch = touch_candidates(acc, turn)
    dh = np.where(np.isnan(home[..., 0]), _INF, _hyp(home - lk[:, None, :])).min(1)
    da = np.where(np.isnan(away[..., 0]), _INF, _hyp(away - lk[:, None, :])).min(1)
    team = np.where(dh <= da, 1.0, -1.0).astype(np.float32)
    touch &= np.minimum(dh, da) < 3.0
    idx = np.flatnonzero(touch)
    g = np.zeros((t, len(GLOBAL_IND_FEATURES)), np.float32)
    g[:, 0] = np.nan_to_num(np.minimum(acc, 150.0))
    g[:, 1] = np.nan_to_num(turn)
    g[:, 2] = touch
    g[:, 4] = 5.0
    g[:, 7] = 5.0
    if len(idx):
        frames = np.arange(t)
        last = np.searchsorted(idx, frames, side="right") - 1
        has = last >= 0
        g[has, 3] = team[idx[last[has]]]
        g[has, 4] = np.minimum((frames[has] - idx[last[has]]) / FPS, 5.0)
        change = np.zeros(t, np.float32)
        change[idx[1:]] = team[idx[1:]] != team[idx[:-1]]
        g[:, 5] = change
        nxt = np.searchsorted(idx, frames, side="left")
        hn = nxt < len(idx)
        g[hn, 6] = team[idx[nxt[hn]]]
        g[hn, 7] = np.minimum((idx[nxt[hn]] - frames[hn]) / FPS, 5.0)
    both = np.concatenate([home, away], axis=1)
    g[:, 8] = (np.nan_to_num(_hyp(both - lk[:, None, :]), nan=float(_INF)) < 3).sum(1)
    per = np.concatenate([hp["per_player"], ap["per_player"]], axis=1)
    if not scaled:
        return np.concatenate([hb, ab, g], 1), per
    ctx = np.concatenate([hb / _TEAM_SCALE, ab / _TEAM_SCALE, g / _GLOBAL_SCALE], 1)
    per = cast(F32, per / np.array([10, 1, 1, 5], np.float32))
    return (
        np.clip(np.nan_to_num(ctx), -5, 5).astype(np.float32),
        np.clip(np.nan_to_num(per), -5, 5).astype(np.float32),
    )
