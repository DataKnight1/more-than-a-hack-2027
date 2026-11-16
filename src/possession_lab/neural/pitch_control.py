"""Probabilistic pitch control around the ball (feature version 4).

Who *can reach* the ball matters more for possession than who is nearest: a defender running
onto a pass beats a static attacker closer to it. Following the physics-based pitch-control model
of Spearman (2018), each player's time to intercept a target point is

    tti = reaction + |target - (p + v * reaction)| / v_max

(the player keeps their current velocity during the reaction time, then runs straight at maximum
speed). Team control of a point is a soft-min over all players' interception times,

    P(home) = sum_home exp(-tti / s) / sum_all exp(-tti / s),   s = sigma * sqrt(3) / pi,

i.e. a logistic arrival-time uncertainty with standard deviation ``sigma``. Constants are the
published values (reaction 0.7 s, v_max 5 m/s, sigma 0.45 s); they are **not** fitted on any data.

Targets: the last-known ball position now and extrapolated along its velocity 0.5 s and 1.0 s
ahead (clipped to the pitch), so a moving ball is judged where it is going, not where it was.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.team_features import last_known, velocity

F32 = NDArray[np.float32]
REACTION_S = 0.7
VMAX = 5.0
SIGMA_S = 0.45
SCALE_S = SIGMA_S * np.sqrt(3.0) / np.pi
HORIZONS_S = (0.0, 0.5, 1.0)
HALF = np.array([52.5, 34.0], np.float32)
TTI_CAP_S = 6.0

CONTEXT_FEATURES = tuple(
    f"{name}_{h:g}s" for h in HORIZONS_S for name in ("p_home", "tti_home", "tti_away")
)
N_PC_CONTEXT = len(CONTEXT_FEATURES)
PLAYER_PC_FEATURES = 3  # tti now, tti at +0.5 s, own share of control at +0.5 s


def interception_times(players: F32, player_vel: F32, target: F32) -> F32:
    """``[T, N]`` seconds for each player to reach ``target [T, 2]``; NaN where absent."""
    start = players + np.nan_to_num(player_vel) * REACTION_S
    dist = np.hypot(*(target[:, None, :] - start).transpose(2, 0, 1))
    return cast(F32, (REACTION_S + dist / VMAX).astype(np.float32))


def team_control(tti_home: F32, tti_away: F32) -> tuple[F32, F32]:
    """``P(home controls target) [T]`` and every player's share of control ``[T, Nh + Na]``."""
    tti = np.concatenate([tti_home, tti_away], axis=1)
    logw = np.where(np.isnan(tti), -np.inf, -np.minimum(tti, TTI_CAP_S) / SCALE_S)
    top = np.max(logw, axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    w = np.exp(logw - top)
    total = w.sum(1, keepdims=True)
    share = np.where(total > 0, w / np.maximum(total, 1e-12), 0.0)
    p_home = share[:, : tti_home.shape[1]].sum(1)
    p_home = np.where(total[:, 0] > 0, p_home, 0.5)
    return cast(F32, p_home.astype(np.float32)), cast(F32, share.astype(np.float32))


def pitch_control_features(ball: F32, home: F32, away: F32) -> tuple[F32, F32]:
    """``(context [T, N_PC_CONTEXT], per_player [T, Nh + Na, PLAYER_PC_FEATURES])``, scaled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xy = ball[:, :2]
        lk, _ = last_known(xy)
        bvel = np.nan_to_num(velocity(xy, 2))
        hv, av = velocity(home, 2), velocity(away, 2)
        ctx: list[F32] = []
        tti0: F32 = np.full((len(ball), home.shape[1] + away.shape[1]), np.nan, np.float32)
        tti5 = tti0.copy()
        share5 = np.zeros_like(tti0)
        for h in HORIZONS_S:
            target = np.clip(lk + bvel * h, -HALF, HALF)
            th, ta = interception_times(home, hv, target), interception_times(away, av, target)
            p_home, share = team_control(th, ta)
            min_h = np.nan_to_num(
                np.nanmin(np.where(np.isnan(th), np.inf, th), 1), posinf=TTI_CAP_S
            )
            min_a = np.nan_to_num(
                np.nanmin(np.where(np.isnan(ta), np.inf, ta), 1), posinf=TTI_CAP_S
            )
            ctx += [p_home, np.minimum(min_h, TTI_CAP_S) / 3.0, np.minimum(min_a, TTI_CAP_S) / 3.0]
            both = np.concatenate([th, ta], axis=1)
            if h == 0.0:
                tti0 = both
            elif h == 0.5:
                tti5, share5 = both, share
        per = np.stack(
            [
                np.minimum(np.nan_to_num(tti0, nan=TTI_CAP_S), TTI_CAP_S) / 3.0,
                np.minimum(np.nan_to_num(tti5, nan=TTI_CAP_S), TTI_CAP_S) / 3.0,
                share5,
            ],
            axis=-1,
        )
    return cast(F32, np.stack(ctx, 1).astype(np.float32)), cast(F32, per.astype(np.float32))
