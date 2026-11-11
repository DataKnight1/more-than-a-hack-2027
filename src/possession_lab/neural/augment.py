"""Domain randomisation: degrade clean external tracking towards automatic broadcast-style tracking.

Ranges are deliberately broad and physically motivated (aerial and crowded balls are lost more
often, tracklets drop in bursts, trackers duplicate people and occasionally flip a tracklet's
team). They are **not** fitted to statistics of the supplied challenge game: doing so would be
calibration on the supplied game, which the organiser clarification forbids. The only
challenge-related fact used is the organiser's published statement that the ball is missing on a
large share of frames, which motivates covering missing-ball rates up to ~65%.

All functions are pure given a ``numpy.random.Generator`` so augmentation is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import A, H

F32 = NDArray[np.float32]


@dataclass(frozen=True)
class DegradeConfig:
    """Upper bounds of each randomised corruption; each window samples within ``[0, max]``."""

    ball_missing_max: float = 0.65
    ball_burst_frames: tuple[int, int] = (3, 100)
    aerial_drop_prob: float = 0.6
    crowd_drop_prob: float = 0.4
    ball_noise_max: float = 0.6
    player_missing_max: float = 0.25
    player_burst_frames: tuple[int, int] = (5, 250)
    player_noise_max: float = 0.35
    duplicate_rate_max: float = 0.08
    team_flip_rate_max: float = 0.03
    scale_jitter: float = 0.03


def _bursty_mask(
    rng: np.random.Generator, length: int, missing: float, burst: float, shape: tuple[int, ...] = ()
) -> NDArray[np.bool_]:
    """Two-state Markov chain mask (True = missing) with given stationary rate and mean burst."""
    if missing <= 0.0:
        return np.zeros((length, *shape), dtype=bool)
    missing = min(missing, 0.95)
    p_end = 1.0 / max(burst, 1.0)
    visible_mean = burst * (1.0 - missing) / missing
    p_start = 1.0 / max(visible_mean, 1.0)
    out = np.empty((length, *shape), dtype=bool)
    state = rng.random(shape) < missing
    uniforms = rng.random((length, *shape))
    for t in range(length):
        state = np.where(state, uniforms[t] >= p_end, uniforms[t] < p_start)
        out[t] = state
    return out


def degrade(
    rng: np.random.Generator,
    ball: F32,
    home: F32,
    away: F32,
    cfg: DegradeConfig = DegradeConfig(),  # noqa: B008 - frozen dataclass
    strength: float = 1.0,
) -> tuple[F32, F32, F32]:
    """Return corrupted copies of one window's ball ``[T,3]``, home ``[T,Nh,2]``, away ``[T,Na,2]``.

    ``strength`` in ``[0, 1]`` scales every corruption bound (0 = clean copy).
    """
    ball = ball.copy()
    home = home.copy()
    away = away.copy()
    length = len(ball)
    if strength <= 0.0:
        return ball, home, away

    # Ball: bursty dropout + targeted loss when aerial or crowded + positional noise.
    missing = rng.uniform(0.0, cfg.ball_missing_max * strength)
    burst = rng.uniform(*cfg.ball_burst_frames)
    lost = _bursty_mask(rng, length, missing, burst)
    z = ball[:, 2]
    aerial = np.nan_to_num(z, nan=0.0) > 1.8
    lost |= aerial & (rng.random(length) < cfg.aerial_drop_prob * strength)
    players = np.concatenate([home, away], axis=1)
    near = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    crowded = np.nansum(near < 2.0, axis=1) >= 3
    crowd_lost = _bursty_mask(rng, length, cfg.crowd_drop_prob * strength, 8.0)
    lost |= crowded & crowd_lost
    noise = rng.normal(0.0, rng.uniform(0, cfg.ball_noise_max * strength), (length, 2))
    ball[:, :2] = (ball[:, :2] + noise).astype(np.float32)
    ball[lost] = np.nan

    def _players(arr: F32) -> F32:
        """Apply configured burst dropout and coordinate noise to one team's tracks."""
        n = arr.shape[1]
        rate = rng.uniform(0.0, cfg.player_missing_max * strength)
        gone = _bursty_mask(rng, length, rate, rng.uniform(*cfg.player_burst_frames), (n,))
        noise = rng.normal(0.0, rng.uniform(0, cfg.player_noise_max * strength), arr.shape)
        arr = (arr + noise).astype(np.float32)
        arr[gone] = np.nan
        return arr.astype(np.float32)

    home = _players(home)
    away = _players(away)

    # Tracker duplicates: an extra slot shadowing a random present player for a burst.
    def _duplicate(arr: F32) -> F32:
        """Optionally add a short noisy tracklet duplicating one randomly selected player."""
        if rng.random() > cfg.duplicate_rate_max * strength * 10:
            return arr
        src = int(rng.integers(arr.shape[1]))
        start = int(rng.integers(length))
        stop = min(length, start + int(rng.uniform(10, 150)))
        extra = np.full((length, 1, 2), np.nan, np.float32)
        extra[start:stop, 0] = arr[start:stop, src] + rng.normal(0, 0.6, (stop - start, 2))
        return np.concatenate([arr, extra], axis=1)

    home = _duplicate(home)
    away = _duplicate(away)

    # Wrong-team tracklet: move one slot's burst to the other team.
    if rng.random() < cfg.team_flip_rate_max * strength * 10:
        src_home = rng.random() < 0.5
        src, dst = (home, away) if src_home else (away, home)
        slot = int(rng.integers(src.shape[1]))
        start = int(rng.integers(length))
        stop = min(length, start + int(rng.uniform(25, 250)))
        moved = np.full((length, 1, 2), np.nan, np.float32)
        moved[start:stop, 0] = src[start:stop, slot]
        src[start:stop, slot] = np.nan
        dst = np.concatenate([dst, moved], axis=1)
        home, away = (src, dst) if src_home else (dst, src)

    scale = 1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter) * strength
    return (
        (ball * scale).astype(np.float32),
        (home * scale).astype(np.float32),
        (away * scale).astype(np.float32),
    )


def symmetry(
    rng: np.random.Generator, ball: F32, home: F32, away: F32, labels: NDArray[np.int8]
) -> tuple[F32, F32, F32, NDArray[np.int8]]:
    """Random x/y mirror and home/away swap (with H<->A label swap)."""
    sx = -1.0 if rng.random() < 0.5 else 1.0
    sy = -1.0 if rng.random() < 0.5 else 1.0
    flip = np.array([sx, sy], np.float32)
    ball = ball.copy()
    ball[:, :2] *= flip
    home = home * flip
    away = away * flip
    if rng.random() < 0.5:
        home, away = away, home
        labels = labels.copy()
        is_h = labels == H
        labels[labels == A] = H
        labels[is_h] = A
    return ball, home.astype(np.float32), away.astype(np.float32), labels


def symmetry_dir(
    rng: np.random.Generator,
    ball: F32,
    home: F32,
    away: F32,
    labels: NDArray[np.int8],
    home_dir: F32,
    aux_xy: F32 | None = None,
) -> tuple[F32, F32, F32, NDArray[np.int8], F32] | tuple[F32, F32, F32, NDArray[np.int8], F32, F32]:
    """:func:`symmetry` that also transforms the home attacking direction (+-1 per frame).

    An x-mirror reverses every attacking direction; a team swap makes the old away team the home
    team, whose direction is the negation of the old home direction. ``aux_xy [T, 2]`` (e.g. the
    clean ball used as a latent-ball target) is mirrored like the ball and returned last.
    """
    sx = -1.0 if rng.random() < 0.5 else 1.0
    sy = -1.0 if rng.random() < 0.5 else 1.0
    swap = rng.random() < 0.5
    flip = np.array([sx, sy], np.float32)
    ball = ball.copy()
    ball[:, :2] *= flip
    home = home * flip
    away = away * flip
    new_dir = (home_dir * sx).astype(np.float32)
    if swap:
        home, away = away, home
        labels = labels.copy()
        is_h = labels == H
        labels[labels == A] = H
        labels[is_h] = A
        new_dir = -new_dir
    out = (ball, home.astype(np.float32), away.astype(np.float32), labels, new_dir)
    if aux_xy is not None:
        return (*out, (aux_xy * flip).astype(np.float32))
    return out
