"""Parse the raw ATD text into dense arrays, cached in the git-ignored artifacts/ folder.

Line format: ``<frame_id>:<x>,<y>,<speed>;...one group per track slot...:<ball x>,<ball y>,<z?>``.
Coordinates are 0-1, missing values are NaN, and ``video_frame = frame_id - 1``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float32]


class Tracking(NamedTuple):
    """Dense tracking on the tactical timeline."""

    ball: Floats  # (n, 3): x, y and the undocumented third field
    players: Floats  # (n, slots, 2)
    speed: Floats  # (n, slots) in m/s
    slot_team: NDArray[np.int8]  # (slots,) 0 = home, 1 = away


def _fill(target: Floats, groups: list[str]) -> None:
    """Write comma-separated groups into the rows of target, padding short groups with NaN."""
    width = target.shape[-1]
    flat = ",".join(groups).split(",")
    if len(flat) == len(groups) * width:
        target[:] = np.array(flat, dtype=np.float64).reshape(len(groups), width)
        return
    for row, group in zip(target, groups, strict=False):
        values = [value for value in group.split(",") if value][:width]
        row[: len(values)] = np.array(values, dtype=np.float64)


def parse_raw(path: Path, n_frames: int, n_slots: int) -> tuple[Floats, Floats, Floats]:
    """Return ``(ball, players, speed)``; frames without a record stay NaN."""
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    groups = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame, slots = int(parts[0]) - 1, parts[1].split(";")
            if not 0 <= frame < n_frames:
                raise ValueError(f"frame_id {frame + 1} outside 1..{n_frames}")
            if len(slots) != n_slots:
                raise ValueError(f"{len(slots)} player groups, metadata has {n_slots}")
            _fill(groups[frame], slots)
            _fill(ball[frame : frame + 1], [parts[2]])
    return ball, groups[..., :2], groups[..., 2]


def load_tracking(
    raw: Path, slot_team: NDArray[np.int8], n_frames: int, cache_dir: Path | None = None
) -> Tracking:
    """Parse once, then reuse the cached arrays while the raw file is unchanged."""
    stat = raw.stat()
    key = f"v2|{raw.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    name = f"atd_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    cache = cache_dir / name if cache_dir else None
    if cache and cache.exists():
        with np.load(cache) as saved:
            return Tracking(saved["ball"], saved["players"], saved["speed"], slot_team)
    ball, players, speed = parse_raw(raw, n_frames, len(slot_team))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ball=ball, players=players, speed=speed)
    return Tracking(ball, players, speed, slot_team)
