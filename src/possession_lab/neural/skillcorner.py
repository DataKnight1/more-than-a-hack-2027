"""Convert SkillCorner Open Data (MIT licence) to :class:`CanonicalMatch` at 25 fps.

Source: https://github.com/SkillCorner/opendata (A-League 2024/25, broadcast tracking at 10 fps).

Label derivation (algorithmic, from provider-produced fields; documented as a different label
definition from DFL and from the challenge GT):

- ball **alive** iff the frame lies inside a SkillCorner *phase of play* (phases are defined only
  while the ball is in play); otherwise ``D``;
- when alive, ``H``/``A`` from the frame's ``possession.group`` (``home team``/``away team``); if
  that is null, the phase's ``team_in_possession_id``.

Observations: objects with ``is_detected == false`` are SkillCorner extrapolations and become NaN
— this keeps real broadcast-camera missingness instead of inventing positions. 10 fps samples are
linearly interpolated to 25 fps where both neighbours are observed; labels use the nearest sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import FPS, A, CanonicalMatch, D, H, X, join_periods

SOURCE_FPS = 10
LICENCE = "MIT (github.com/SkillCorner/opendata)"
FloatArray = NDArray[np.float32]
AnyArray = NDArray[Any]


def _phase_intervals(path: Path) -> list[tuple[int, int, int]]:
    """``[(frame_start, frame_end, team_in_possession_id)]`` in 10 fps frames, inclusive."""
    with path.open(encoding="utf-8") as handle:
        return [
            (int(row["frame_start"]), int(row["frame_end"]), int(row["team_in_possession_id"]))
            for row in csv.DictReader(handle)
        ]


def _interp(values: AnyArray, src_t: NDArray[Any], dst_t: NDArray[Any]) -> FloatArray:
    """Linear interpolation along axis 0 that yields NaN unless both neighbours are finite."""
    idx = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1, 0, len(src_t) - 2)
    t0, t1 = src_t[idx], src_t[idx + 1]
    w = ((dst_t - t0) / np.maximum(t1 - t0, 1e-9)).reshape(-1, *([1] * (values.ndim - 1)))
    return cast(FloatArray, (values[idx] * (1 - w) + values[idx + 1] * w).astype(np.float32))


def _convert_period(
    records: list[dict[str, object]],
    phases: list[tuple[int, int, int]],
    home_id: object,
    away_id: object,
    home_slots: dict[object, int],
    away_slots: dict[object, int],
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int8]]:
    """Convert one provider period to interpolated tracking and labels."""
    recs = sorted(records, key=lambda row: _as_int(row.get("frame"), "frame"))
    frames = np.array([_as_int(row.get("frame"), "frame") for row in recs])
    count = len(recs)
    ball = np.full((count, 3), np.nan, np.float32)
    home = np.full((count, len(home_slots), 2), np.nan, np.float32)
    away = np.full((count, len(away_slots), 2), np.nan, np.float32)
    group = np.full(count, -1, np.int8)
    for frame_index, record in enumerate(recs):
        ball_data = record["ball_data"]
        if (
            isinstance(ball_data, dict)
            and ball_data.get("is_detected")
            and ball_data.get("x") is not None
        ):
            ball[frame_index] = (
                ball_data["x"],
                ball_data["y"],
                ball_data.get("z") if ball_data.get("z") is not None else np.nan,
            )
        possession = record["possession"]
        group_name = possession.get("group") if isinstance(possession, dict) else None
        group[frame_index] = (
            0 if group_name == "home team" else 1 if group_name == "away team" else -1
        )
        player_data = record.get("player_data")
        if not isinstance(player_data, list):
            continue
        for player in player_data:
            if not isinstance(player, dict) or not player.get("is_detected"):
                continue
            player_id = player["player_id"]
            if player_id in home_slots:
                home[frame_index, home_slots[player_id]] = (player["x"], player["y"])
            elif player_id in away_slots:
                away[frame_index, away_slots[player_id]] = (player["x"], player["y"])

    alive = np.zeros(count, bool)
    phase_team = np.full(count, -1, np.int8)
    for start, stop, team in phases:
        selected = (frames >= start) & (frames <= stop)
        alive |= selected
        phase_team[selected] = 0 if team == home_id else 1 if team == away_id else -1
    team_code = np.where(group >= 0, group, phase_team)
    labels = np.full(count, D, np.int8)
    labels[alive & (team_code == 0)] = H
    labels[alive & (team_code == 1)] = A
    labels[alive & (team_code < 0)] = X

    source_time = (frames - frames[0]) / SOURCE_FPS
    target_time = np.arange(0, source_time[-1] + 1e-9, 1.0 / FPS)
    right = np.clip(np.searchsorted(source_time, target_time), 1, count - 1)
    nearest = np.where(
        target_time - source_time[right - 1] <= source_time[right] - target_time,
        right - 1,
        right,
    )
    return (
        _interp(ball, source_time, target_time),
        _interp(home, source_time, target_time),
        _interp(away, source_time, target_time),
        labels[nearest],
    )


def convert(match_dir: Path) -> CanonicalMatch:
    """Convert one ``data/matches/<id>`` folder."""
    match_id = match_dir.name
    meta = json.loads((match_dir / f"{match_id}_match.json").read_text(encoding="utf-8"))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    home_players = sorted(pid for pid, team in team_of.items() if team == home_id)
    away_players = sorted(pid for pid, team in team_of.items() if team == away_id)
    hi = {pid: i for i, pid in enumerate(home_players)}
    ai = {pid: i for i, pid in enumerate(away_players)}
    phases = _phase_intervals(match_dir / f"{match_id}_phases_of_play.csv")

    rows: dict[int, list[dict[str, object]]] = {}
    with (match_dir / f"{match_id}_tracking_extrapolated.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("period") in (1, 2):
                rows.setdefault(int(rec["period"]), []).append(rec)

    parts = [
        _convert_period(rows[period], phases, home_id, away_id, hi, ai) for period in sorted(rows)
    ]

    ball_all, home_all, away_all, labels, period_arr = join_periods(parts)
    return CanonicalMatch(
        match_id=f"skillcorner:{match_id}",
        provider="skillcorner",
        ball=ball_all,
        home=home_all,
        away=away_all,
        labels=labels,
        period=period_arr,
        pitch=(float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68)),
        meta={
            "licence": LICENCE,
            "label_method": "alive = inside phase of play; team = possession.group else phase team",
            "home": meta["home_team"]["short_name"],
            "away": meta["away_team"]["short_name"],
            "source_fps": SOURCE_FPS,
        },
    )


def _as_int(value: object, field: str) -> int:
    """Validate an integer-like JSON field from SkillCorner metadata."""
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"SkillCorner {field} must be an integer-like value")
    return int(value)
