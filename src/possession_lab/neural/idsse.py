"""Convert IDSSE / DFL open positional data (CC BY 4.0) to :class:`CanonicalMatch`.

Label method (provider-supplied, not created by us): every ball frame of the DFL positions feed
carries ``BallStatus`` (1 alive, 0 dead) and ``BallPossession`` (1 home, 2 away). We map
``BallStatus == 0 -> D``, otherwise ``BallPossession 1 -> H``, ``2 -> A``; a missing ball frame
becomes ``X`` (never scored). The DFL possession definition is the provider's and may differ from
the challenge ground truth's definition; this is a documented domain shift, not something we tune.

Coordinates are already metres with the origin at the centre spot. Referees are dropped.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree
from numpy.typing import NDArray

from possession_lab.neural.canonical import A, CanonicalMatch, D, H, X, join_periods

SECTIONS = ("firstHalf", "secondHalf")
BALL_TEAM = "BALL"
LICENCE = "CC BY 4.0 (figshare 10.6084/m9.figshare.28196177.v1)"
_MATCH_RE = re.compile(r"DFL-MAT-([A-Z0-9]+)")
Array = NDArray[Any]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


def _sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest for a downloaded dataset file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_match_information(path: Path) -> dict[str, str | float]:
    """Home/guest team ids and names plus pitch size from the DFL match-information XML."""
    root = etree.parse(str(path)).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    if general is None or environment is None:
        raise ValueError(f"{path}: missing General/Environment")
    return {
        "home_id": general.get("HomeTeamId", ""),
        "away_id": general.get("GuestTeamId", ""),
        "home_name": general.get("HomeTeamName", ""),
        "away_name": general.get("GuestTeamName", ""),
        "pitch_x": float(environment.get("PitchX", "105")),
        "pitch_y": float(environment.get("PitchY", "68")),
    }


def _parse_positions(
    path: Path,
) -> dict[tuple[str, str, str], tuple[IntArray, FloatArray, FloatArray | None]]:
    """Stream the positions XML into ``{(section, team, person): (N, xyz, possession/status)}``."""
    frames: dict[tuple[str, str, str], list[tuple[float, ...]]] = defaultdict(list)
    key: tuple[str, str, str] | None = None
    is_ball = False
    for event, elem in etree.iterparse(str(path), events=("start", "end"), huge_tree=True):
        tag = elem.tag
        if event == "start":
            if tag == "FrameSet":
                key = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
                is_ball = key[1] == BALL_TEAM
            continue
        if tag == "Frame" and key is not None:
            get = elem.get
            if is_ball:
                frames[key].append(
                    (
                        float(get("N")),
                        float(get("X")),
                        float(get("Y")),
                        float(get("Z", "nan")),
                        float(get("BallPossession", "nan")),
                        float(get("BallStatus", "nan")),
                    )
                )
            else:
                frames[key].append((float(get("N")), float(get("X")), float(get("Y"))))
            elem.clear()
        elif tag == "FrameSet":
            key = None
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    out: dict[tuple[str, str, str], tuple[IntArray, FloatArray, FloatArray | None]] = {}
    for frame_key, rows in frames.items():
        arr = np.asarray(rows, dtype=np.float64)
        n = arr[:, 0].astype(np.int64)
        if frame_key[1] == BALL_TEAM:
            out[frame_key] = (n, arr[:, 1:4].astype(np.float32), arr[:, 4:6].astype(np.float32))
        else:
            out[frame_key] = (n, arr[:, 1:3].astype(np.float32), None)
    return out


def convert(positions_path: Path, info_path: Path) -> CanonicalMatch:
    """Convert one DFL match to canonical form.

    Args:
        positions_path: ``DFL_04_03_positions_raw_observed_*.xml``.
        info_path: ``DFL_02_01_matchinformation_*.xml`` of the same match.

    Returns:
        The canonical match; each period spans the ball frameset's frame range.
    """
    info = read_match_information(info_path)
    parsed = _parse_positions(positions_path)
    home_people = sorted({p for (_, team, p) in parsed if team == info["home_id"]})
    away_people = sorted({p for (_, team, p) in parsed if team == info["away_id"]})
    if not home_people or not away_people:
        raise ValueError(f"{positions_path}: team ids not found in positions file")
    home_index = {p: i for i, p in enumerate(home_people)}
    away_index = {p: i for i, p in enumerate(away_people)}

    parts: list[tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int8]]] = []
    for section in SECTIONS:
        ball_keys = [k for k in parsed if k[0] == section and k[1] == BALL_TEAM]
        if not ball_keys:
            continue
        ball_n, ball_xyz, ball_attrs = parsed[ball_keys[0]]
        if ball_attrs is None:
            raise ValueError("ball frameset without attributes")
        first, last = int(ball_n.min()), int(ball_n.max())
        length = last - first + 1
        ball = np.full((length, 3), np.nan, np.float32)
        ball[ball_n - first] = ball_xyz
        possession = np.full(length, np.nan)
        status = np.full(length, np.nan)
        possession[ball_n - first] = ball_attrs[:, 0]
        status[ball_n - first] = ball_attrs[:, 1]

        labels = np.full(length, X, np.int8)
        labels[status == 0] = D
        alive = status == 1
        labels[alive & (possession == 1)] = H
        labels[alive & (possession == 2)] = A

        home = np.full((length, len(home_people), 2), np.nan, np.float32)
        away = np.full((length, len(away_people), 2), np.nan, np.float32)
        for (sec, team, person), (n, xy, _) in parsed.items():
            if sec != section or team == BALL_TEAM:
                continue
            keep = (n >= first) & (n <= last)
            if team == info["home_id"]:
                home[n[keep] - first, home_index[person]] = xy[keep]
            elif team == info["away_id"]:
                away[n[keep] - first, away_index[person]] = xy[keep]
        parts.append((ball, home, away, labels))

    ball_all, home_all, away_all, labels_all, period = join_periods(parts)
    match = _MATCH_RE.search(positions_path.name)
    match_code = match.group(1) if match else positions_path.stem
    return CanonicalMatch(
        match_id=f"idsse:{match_code}",
        provider="idsse",
        ball=ball_all,
        home=home_all,
        away=away_all,
        labels=labels_all,
        period=period,
        pitch=(float(info["pitch_x"]), float(info["pitch_y"])),
        meta={
            "licence": LICENCE,
            "label_method": "DFL positions feed BallStatus/BallPossession (provider-produced)",
            "home": info["home_name"],
            "away": info["away_name"],
            "source_file": positions_path.name,
            "source_sha256": _sha256(positions_path),
        },
    )
