"""Provider-neutral canonical match representation.

All providers are converted to the same arrays so that augmentation, features, models and the
external evaluator never see provider-specific formats:

- coordinates in **metres, origin at the pitch centre**, x along the length;
- one row per 25 fps frame; consecutive periods separated by ``GAP_FRAMES`` rows of ``X`` labels
  and NaN observations so the official metric never joins two halves;
- labels use the official integer codes ``H=0, A=1, D=2, X=3`` from
  ``evaluation/possession_metrics.py``.

Orientation (which goal a team attacks, sign of y) is intentionally *not* normalised: models are
trained with random flips so they must be orientation-invariant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])
FPS = 25
GAP_FRAMES = 25

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int8]


@dataclass
class CanonicalMatch:
    """One match in canonical form.

    Attributes:
        match_id: Provider-scoped identifier, e.g. ``idsse:J03WMX``.
        provider: Source provider key (``idsse``, ``skillcorner``, ``challenge``).
        ball: ``[T, 3]`` ball x, y, z in metres (NaN when unobserved; z NaN if not provided).
        home: ``[T, Nh, 2]`` home object positions (NaN where a slot is absent).
        away: ``[T, Na, 2]`` away object positions.
        labels: ``[T]`` int8 official codes; ``X`` on gap/ungraded frames.
        period: ``[T]`` int8 period number (1, 2, ...); 0 on gap frames.
        pitch: ``(length, width)`` in metres.
        meta: Free-form provenance (licence, source file hashes, label method).
    """

    match_id: str
    provider: str
    ball: FloatArray
    home: FloatArray
    away: FloatArray
    labels: IntArray
    period: IntArray
    pitch: tuple[float, float]
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and label codes."""
        n = len(self.labels)
        if self.ball.shape != (n, 3):
            raise ValueError(f"ball shape {self.ball.shape} != ({n}, 3)")
        for name, arr in (("home", self.home), ("away", self.away)):
            if arr.ndim != 3 or arr.shape[0] != n or arr.shape[2] != 2:
                raise ValueError(f"{name} shape {arr.shape} invalid for {n} frames")
        if self.period.shape != (n,):
            raise ValueError("period length mismatch")
        if self.labels.size and (self.labels.min() < H or self.labels.max() > X):
            raise ValueError("labels must be official codes 0..3")

    @property
    def n_frames(self) -> int:
        """Number of frames including gap rows."""
        return int(len(self.labels))

    def save(self, path: Path) -> None:
        """Write to a compressed ``.npz`` with JSON metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ball=self.ball,
            home=self.home,
            away=self.away,
            labels=self.labels,
            period=self.period,
            pitch=np.asarray(self.pitch, dtype=np.float32),
            header=np.asarray(
                json.dumps(
                    {"match_id": self.match_id, "provider": self.provider, "meta": self.meta}
                )
            ),
        )

    @classmethod
    def load(cls, path: Path) -> CanonicalMatch:
        """Read a match written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            pitch = np.asarray(data["pitch"], dtype=float)
            return cls(
                match_id=header["match_id"],
                provider=header["provider"],
                ball=data["ball"],
                home=data["home"],
                away=data["away"],
                labels=data["labels"],
                period=data["period"],
                pitch=(float(pitch[0]), float(pitch[1])),
                meta=header.get("meta", {}),
            )


def join_periods(
    parts: Sequence[tuple[FloatArray, FloatArray, FloatArray, IntArray]],
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    """Concatenate per-period arrays with ``GAP_FRAMES`` separator rows.

    Args:
        parts: ``(ball, home, away, labels)`` per period, in order; player arrays of a match must
            share their slot dimension.

    Returns:
        ``ball, home, away, labels, period`` for the whole match.
    """
    balls: list[FloatArray] = []
    homes: list[FloatArray] = []
    aways: list[FloatArray] = []
    labels: list[IntArray] = []
    periods: list[IntArray] = []
    n_home = parts[0][1].shape[1]
    n_away = parts[0][2].shape[1]
    for index, (ball, home, away, lab) in enumerate(parts):
        if index:
            balls.append(np.full((GAP_FRAMES, 3), np.nan, np.float32))
            homes.append(np.full((GAP_FRAMES, n_home, 2), np.nan, np.float32))
            aways.append(np.full((GAP_FRAMES, n_away, 2), np.nan, np.float32))
            labels.append(np.full(GAP_FRAMES, X, np.int8))
            periods.append(np.zeros(GAP_FRAMES, np.int8))
        balls.append(ball.astype(np.float32))
        homes.append(home.astype(np.float32))
        aways.append(away.astype(np.float32))
        labels.append(lab.astype(np.int8))
        periods.append(np.full(len(lab), index + 1, np.int8))
    return (
        np.concatenate(balls),
        np.concatenate(homes),
        np.concatenate(aways),
        np.concatenate(labels),
        np.concatenate(periods),
    )


def label_summary(labels: IntArray) -> dict[str, float]:
    """Frame shares and possession-spell statistics, for dataset cards (external data only)."""
    graded = labels != X
    out: dict[str, float] = {"n_graded": float(graded.sum())}
    for code, char in enumerate("HAD"):
        out[f"share_{char}"] = float((labels[graded] == code).mean()) if graded.any() else 0.0
    edges = np.flatnonzero(np.diff(labels.astype(np.int16)) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(labels)]))
    lengths = [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True) if labels[a] in (H, A)]
    out["n_spells"] = float(len(lengths))
    out["mean_spell_s"] = float(np.mean(lengths)) / FPS if lengths else 0.0
    out["median_spell_s"] = float(np.median(lengths)) / FPS if lengths else 0.0
    return out
