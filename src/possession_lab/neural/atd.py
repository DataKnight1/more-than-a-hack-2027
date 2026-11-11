"""Local-only reader: challenge tactical ATD -> :class:`CanonicalMatch` for fixed inference.

Governance: this is an *inference input* path only. Nothing read here may be used for training,
fitting, calibration, tuning or model selection, and its contents must never be printed, logged
in detail or sent to a hosted service. Labels are all ``X`` (no ground truth is read).

Layout (organiser docs, ``official_documentation/starter/baseline.py``): one line per frame
``<frame_id>:<x,y,s>;<x,y,s>;...:<ball x,y,...>`` with NaN for missing, normalised 0-1 coordinates
with (0, 0) top-left, ``frame_id`` 1-based; track-slot team membership comes from the metadata XML
(first team listed is home). Graded halves come from ``games.half_video_blocks``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import CanonicalMatch, X

ROOT = Path(__file__).resolve().parents[3]


def load_challenge_match(game: str = "BAR-ATH", data_dir: Path | None = None) -> CanonicalMatch:
    """Read the tactical ATD as a canonical match over the full 199,319-frame timeline.

    ``period`` is 1/2 on the officially published graded halves and 0 elsewhere, so decoding and
    inference run per half and ungraded frames stay ``X``.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # pylint: disable=import-outside-toplevel
    from games import GAMES, half_video_blocks, require, source_frames
    from official_documentation.starter.baseline import team_of_each_track

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    home_slots = np.flatnonzero(team == 0)
    away_slots = np.flatnonzero(team == 1)
    n_slots = len(team)

    ball = np.full((n_frames, 3), np.nan, np.float32)
    slots = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")
            for slot, group in enumerate(groups[:n_slots]):
                if group[:1] == "N" or not group:
                    continue
                fields = group.split(",")
                slots[row, slot, 0] = float(fields[0])
                slots[row, slot, 1] = float(fields[1])
            fields = parts[2].split(",")
            if fields and fields[0][:1] != "N" and len(fields) >= 2:
                ball[row, 0] = float(fields[0])
                ball[row, 1] = float(fields[1])

    def to_metres(xy: NDArray[Any]) -> NDArray[np.float32]:
        """Scale normalized pitch coordinates into centered metric coordinates.

        Args:
            xy: Coordinate array whose final dimension stores normalized x and y.

        Returns:
            Coordinates scaled to the configured pitch length and width.
        """
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - 0.5) * length
        out[..., 1] = (0.5 - xy[..., 1]) * width
        return out

    ball[:, :2] = to_metres(ball[:, :2])
    slots = to_metres(slots)
    period = np.zeros(n_frames, np.int8)
    for half, first, last in half_video_blocks(game):
        period[first : last + 1] = half
    return CanonicalMatch(
        match_id=f"challenge:{game}",
        provider="challenge",
        ball=ball,
        home=slots[:, home_slots],
        away=slots[:, away_slots],
        labels=np.full(n_frames, X, np.int8),
        period=period,
        pitch=(float(length), float(width)),
        meta={"use": "inference-only; never training/tuning/selection"},
    )
