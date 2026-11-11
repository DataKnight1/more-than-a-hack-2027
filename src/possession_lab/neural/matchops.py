"""Torch-free match operations: period spans and deterministic evaluation degradation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.augment import degrade
from possession_lab.neural.canonical import CanonicalMatch


def period_spans(period: NDArray[np.int8]) -> list[tuple[int, int]]:
    """``[(start, stop)]`` half-open index spans of each non-gap period."""
    spans = []
    for p in np.unique(period):
        if p == 0:
            continue
        idx = np.flatnonzero(period == p)
        spans.append((int(idx[0]), int(idx[-1]) + 1))
    return spans


def degrade_match(match: CanonicalMatch, seed: int, chunk: int = 1500) -> CanonicalMatch:
    """Deterministically degraded copy of a match, re-sampling corruption every ``chunk`` frames.

    Used to build fixed "automatic-tracking-like" external test conditions. Labels are untouched.
    """
    rng = np.random.default_rng(seed)
    n_home, n_away = match.home.shape[1], match.away.shape[1]
    extra = 2  # room for duplicate / team-flip slots
    ball = np.full_like(match.ball, np.nan)
    home = np.full((match.n_frames, n_home + extra, 2), np.nan, np.float32)
    away = np.full((match.n_frames, n_away + extra, 2), np.nan, np.float32)
    for lo, hi in period_spans(match.period):
        for start in range(lo, hi, chunk):
            sl = slice(start, min(hi, start + chunk))
            b, h, a = degrade(rng, match.ball[sl], match.home[sl], match.away[sl])
            ball[sl] = b
            home[sl, : min(h.shape[1], n_home + extra)] = h[:, : n_home + extra]
            away[sl, : min(a.shape[1], n_away + extra)] = a[:, : n_away + extra]
    return CanonicalMatch(
        match_id=f"{match.match_id}+degraded{seed}",
        provider=match.provider,
        ball=ball,
        home=home,
        away=away,
        labels=match.labels,
        period=match.period,
        pitch=match.pitch,
        meta={**match.meta, "degraded_seed": seed},
    )
