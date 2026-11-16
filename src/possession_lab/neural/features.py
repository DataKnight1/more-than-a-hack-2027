"""Per-frame token sets for the set-temporal network.

Each frame becomes one ball token plus up to ``max_players`` player tokens (home and away pooled,
team as a feature, no identities, so slot swaps and tracklet churn are irrelevant). Positions are
pitch-centred metres scaled to roughly unit range. Missing values never reach the network as NaN:
they are zeroed and described by explicit presence / age features.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
BALL_FEATURES = 9
BALL_FEATURES_FORECAST = (
    13  # N019: +4 channels (forecast_x, forecast_y, forecast_conf, forecast_imputed)
)
PLAYER_FEATURES = 8
HALF_LENGTH = 52.5
HALF_WIDTH = 34.0
AGE_SCALE = 50.0  # frames; ages are clipped at 1.0 (= 2 s)
VEL_SPAN = 2  # central difference over +/- 2 frames


def _central_velocity(pos: F32, fps: float = 25.0) -> F32:
    """``(p[t+k] - p[t-k]) / (2k/fps)`` with edge padding; NaN where either end is missing."""
    k = VEL_SPAN
    padded = np.concatenate([np.repeat(pos[:1], k, axis=0), pos, np.repeat(pos[-1:], k, axis=0)])
    return cast(
        F32,
        ((padded[2 * k :] - padded[: -2 * k]) * (fps / (2 * k))).astype(np.float32),
    )


def _fill_and_age(seen: NDArray[np.bool_], values: F32) -> tuple[F32, F32, F32]:
    """Forward-filled values, frames since last seen and frames until next seen (clipped)."""
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    nxt_idx = np.where(seen, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt_idx[::-1])[::-1]
    filled = np.zeros_like(values)
    has_last = last >= 0
    filled[has_last] = values[last[has_last]]
    # Before the first sighting fall back to the next sighting (backward fill).
    has_next = (~has_last) & (nxt < n)
    filled[has_next] = values[nxt[has_next]]
    age_back = np.where(has_last, np.arange(n) - last, AGE_SCALE).astype(np.float32)
    age_fwd = np.where(nxt < n, nxt - np.arange(n), AGE_SCALE).astype(np.float32)
    return filled, np.minimum(age_back / AGE_SCALE, 1.0), np.minimum(age_fwd / AGE_SCALE, 1.0)


def build_tokens(
    ball: F32,
    home: F32,
    away: F32,
    max_players: int = 30,
    extra: F32 | None = None,
    ball_forecast: bool = False,
) -> tuple[F32, F32, NDArray[np.bool_]]:
    """Build ball features, player features, and a player mask (True means real).

    Shapes are ``ball_feat [T, 9 or 13]``, ``player_feat [T, K, 8]`` and
    ``player_mask [T, K]``.

    Args:
        ball: ``[T, 3]`` metres (z ignored; not available from every provider).
        home: ``[T, Nh, 2]`` metres, NaN where absent.
        away: ``[T, Na, 2]`` metres, NaN where absent.
        max_players: Token budget per frame; the nearest objects to the (last known) ball are kept.
        extra: optional ``[T, Nh + Na, E]`` per-player features (home slots first), appended after
            the base features and sorted with the same nearest-to-ball order.
        ball_forecast: if True (N019), append 4 physical trajectory forecast channels:
            forecast_x, forecast_y (scaled), confidence, and imputed indicator.
    """
    t_len = len(ball)
    xy = ball[:, :2]
    seen = ~np.isnan(xy[:, 0])
    vel = np.nan_to_num(_central_velocity(xy))
    filled, age_back, age_fwd = _fill_and_age(seen, np.nan_to_num(xy))
    scale = np.array([HALF_LENGTH, HALF_WIDTH], np.float32)
    n_ball_feats = BALL_FEATURES_FORECAST if ball_forecast else BALL_FEATURES
    ball_feat = np.zeros((t_len, n_ball_feats), np.float32)
    ball_feat[:, 0] = seen
    ball_feat[:, 1:3] = np.where(seen[:, None], np.nan_to_num(xy) / scale, 0.0)
    ball_feat[:, 3:5] = np.clip(vel / 15.0, -3, 3) * seen[:, None]
    ball_feat[:, 5:7] = filled / scale
    ball_feat[:, 7] = age_back
    ball_feat[:, 8] = age_fwd
    if ball_forecast:
        # Optional or cyclic dependency: load only on this path.
        # pylint: disable-next=import-outside-toplevel
        from possession_lab.neural.trajectory_forecast import forecast_ball_trajectory

        fc_xy, fc_conf, fc_imputed = forecast_ball_trajectory(xy, seen)
        ball_feat[:, 9:11] = fc_xy / scale
        ball_feat[:, 11:12] = fc_conf
        ball_feat[:, 12:13] = fc_imputed

    players = np.concatenate([home, away], axis=1)
    team = np.concatenate([np.ones(home.shape[1]), -np.ones(away.shape[1])]).astype(np.float32)
    present = ~np.isnan(players[..., 0])
    pvel = np.nan_to_num(_central_velocity(players))
    pos = np.nan_to_num(players)
    rel = pos - filled[:, None, :]
    dist = np.hypot(rel[..., 0], rel[..., 1])
    order = np.argsort(np.where(present, dist, np.inf), axis=1, kind="stable")[:, :max_players]
    k = order.shape[1]
    take = np.take_along_axis
    mask = take(present, order, axis=1)
    n_extra = 0 if extra is None else extra.shape[-1]
    feat = np.zeros((t_len, max_players, PLAYER_FEATURES + n_extra), np.float32)
    feat[:, :k, 0] = team[order]
    feat[:, :k, 1:3] = take(pos, order[..., None], axis=1) / scale
    feat[:, :k, 3:5] = np.clip(take(pvel, order[..., None], axis=1) / 8.0, -2, 2)
    feat[:, :k, 5:7] = np.clip(take(rel, order[..., None], axis=1) / 20.0, -3, 3)
    feat[:, :k, 7] = np.clip(take(dist, order, axis=1) / 20.0, 0, 3)
    if extra is not None:
        feat[:, :k, PLAYER_FEATURES:] = take(extra, order[..., None], axis=1)
    feat[:, :k] *= mask[..., None]
    full_mask = np.zeros((t_len, max_players), bool)
    full_mask[:, :k] = mask
    return ball_feat, feat, full_mask


PLAYER_FEATURES_V2 = PLAYER_FEATURES + 4


def build_inputs(
    ball: F32,
    home: F32,
    away: F32,
    max_players: int,
    home_dir: F32 | None = None,
    version: int = 1,
    ball_forecast: bool = False,
) -> tuple[F32, F32, NDArray[np.bool_], F32 | None]:
    """Model inputs for a feature version: ``(ball_feat, player_feat, player_mask, context)``.

    Version 3 adds the individual / marking / ball-touch features of
    :mod:`individual_features` (``experiments/research/neural/EDA_INDIVIDUAL.md``). Version 4 adds
    probabilistic pitch control around the ball (:mod:`pitch_control`). Version 5 (N019) adds
    physics-based confidence-gated ball trajectory forecasting (+4 ball features).

    Version 1 = :func:`build_tokens` (context ``None``). Version 2 (EDA-driven, see
    ``experiments/research/neural/EDA_POSSESSION.md``) adds per player: forward velocity towards
    the goal its team attacks, closing speed towards the (last-known) ball, speed and forward
    position; and a
    per-frame team-context vector (:func:`team_features.context_features`).
    """
    if version == 5:
        version = 3
        ball_forecast = True
    if version == 1:
        ball_feat, feat, mask = build_tokens(
            ball, home, away, max_players, ball_forecast=ball_forecast
        )
        return ball_feat, feat, mask, None

    # Lazy imports avoid loading unrelated feature modules.
    # pylint: disable=import-outside-toplevel
    from possession_lab.neural.team_features import context_features, last_known

    if home_dir is None:
        raise ValueError("feature version 2 needs home_dir")
    players = np.concatenate([home, away], axis=1)
    side = np.concatenate([np.ones(home.shape[1]), -np.ones(away.shape[1])]).astype(np.float32)
    direction = home_dir[:, None] * side[None, :]  # each player's attacking direction (+-1)
    pos = np.nan_to_num(players)
    vel = np.nan_to_num(_central_velocity(players))
    lk, _ = last_known(ball[:, :2])
    rel = lk[:, None, :] - pos
    dist = np.hypot(rel[..., 0], rel[..., 1])
    closing = (vel * rel).sum(-1) / np.maximum(dist, 1e-3)
    extra = np.stack(
        [
            np.clip(vel[..., 0] * direction / 8.0, -2, 2),
            np.clip(closing / 8.0, -2, 2),
            np.clip(np.hypot(vel[..., 0], vel[..., 1]) / 8.0, 0, 2),
            pos[..., 0] * direction / HALF_LENGTH,
        ],
        axis=-1,
    ).astype(np.float32)
    context = context_features(ball, home, away, home_dir.astype(np.float32))
    if version >= 3:
        from possession_lab.neural.individual_features import individual_features

        ind_ctx, ind_player = individual_features(ball, home, away, home_dir.astype(np.float32))
        extra = np.concatenate([extra, ind_player], axis=-1)
        context = np.concatenate([context, ind_ctx], axis=-1)
    if version >= 4:
        from possession_lab.neural.pitch_control import pitch_control_features

        pc_ctx, pc_player = pitch_control_features(ball, home, away)
        extra = np.concatenate([extra, pc_player], axis=-1)
        context = np.concatenate([context, pc_ctx], axis=-1)
    ball_feat, feat, mask = build_tokens(
        ball, home, away, max_players, extra=extra, ball_forecast=ball_forecast
    )
    return ball_feat, feat, mask, context


def feature_dims(version: int) -> tuple[int, int]:
    """``(player_features, context_features)`` of a feature version."""

    # Lazy imports avoid loading unrelated feature modules.
    # pylint: disable=import-outside-toplevel
    from possession_lab.neural.individual_features import N_IND, PLAYER_IND_FEATURES
    from possession_lab.neural.pitch_control import N_PC_CONTEXT, PLAYER_PC_FEATURES
    from possession_lab.neural.team_features import N_CONTEXT

    if version == 1:
        return PLAYER_FEATURES, 0
    if version == 2:
        return PLAYER_FEATURES_V2, N_CONTEXT
    if version in (3, 5):
        return PLAYER_FEATURES_V2 + PLAYER_IND_FEATURES, N_CONTEXT + N_IND
    return (
        PLAYER_FEATURES_V2 + PLAYER_IND_FEATURES + PLAYER_PC_FEATURES,
        N_CONTEXT + N_IND + N_PC_CONTEXT,
    )


def version_of(context_features: int) -> int:
    """Feature version implied by a model's context width (0 -> v1, 37 v2, 62 v3, 71 v4)."""
    if not context_features:
        return 1
    for version in (2, 3, 4):
        if context_features == feature_dims(version)[1]:
            return version
    raise ValueError(f"unknown context width {context_features}")
