"""Convert Metrica Sports sample data (3 anonymised games, 25 Hz) to :class:`CanonicalMatch`.

Source: github.com/metrica-sports/sample-data @ e706dd5. No licence file; use approved by Tiago
2026-11-06 (decision D-009) with source acknowledgement. Other games, not the challenge fixture.

**Label derivation (algorithmic, from Metrica's human-annotated event data):**

- *possession events* (``PASS``, ``CARRY``, ``SET PIECE``, ``RECOVERY``, ``SHOT``, ``BALL LOST``):
  the acting team holds the ball from the event's start frame until the next possession event
  of any team;
- *dead ball* starts at ``BALL OUT``, ``FAULT RECEIVED``, ``CARD`` or a ``SHOT`` whose subtype
  contains ``OUT``/``GOAL``, and lasts until the start of the next possession event (normally the
  restart ``SET PIECE``);
- ``CHALLENGE`` events are ignored (both teams are involved).

Coordinates are normalised 0–1 with (0, 0) top-left on a 105 x 68 m pitch (README).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from possession_lab.neural.canonical import A, CanonicalMatch, D, H, X, join_periods

PITCH = (105.0, 68.0)
Array = NDArray[Any]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
LICENCE = (
    "no licence file; approved by repository owner 2026-11-06 (D-009); acknowledge Metrica Sports"
)
POSSESSION_TYPES = {"PASS", "CARRY", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}
DEAD_TYPES = {"BALL OUT", "FAULT RECEIVED", "CARD"}


@dataclass(frozen=True)
class Event:
    """Minimal event: team code (``H``/``A``), type, subtype, period, start/end frame (1-based)."""

    team: str
    kind: str
    subtype: str
    period: int
    start: int
    end: int


def _starts_dead(event: Event) -> int | None:
    """Frame at which this event makes the ball dead, or None."""
    if event.kind in DEAD_TYPES:
        return event.start
    if event.kind == "SHOT" and ("OUT" in event.subtype or "GOAL" in event.subtype):
        return max(event.start, event.end)
    return None


def events_to_labels(events: Iterable[Event], frames: IntArray) -> NDArray[np.int8]:
    """Per-frame official codes for one period's tracking frame numbers ``frames`` (sorted).

    Frames before the first possession event take that event's team; frames with no event at all
    in the period are ``X``.
    """
    evs = sorted(events, key=lambda e: (e.start, e.end))
    n = len(frames)
    team_mark = np.full(n, -1, np.int8)
    dead = np.zeros(n, bool)
    pending_dead: int | None = None
    for ev in evs:
        if ev.kind in POSSESSION_TYPES and ev.team in ("H", "A"):
            idx = int(np.searchsorted(frames, ev.start))
            if pending_dead is not None and ev.start > pending_dead:
                lo = int(np.searchsorted(frames, pending_dead))
                dead[lo:idx] = True
            pending_dead = None
            if idx < n:
                team_mark[idx] = H if ev.team == "H" else A
        start_dead = _starts_dead(ev)
        if start_dead is not None:
            pending_dead = start_dead if pending_dead is None else min(pending_dead, start_dead)
    known = np.flatnonzero(team_mark >= 0)
    if known.size == 0:
        return np.full(n, X, np.int8)
    carry = np.maximum.accumulate(np.where(team_mark >= 0, np.arange(n), -1))
    carry[carry < 0] = known[0]
    labels = team_mark[carry].astype(np.int8)
    if pending_dead is not None:  # dead until the end of the period
        dead[int(np.searchsorted(frames, pending_dead)) :] = True
    labels[dead] = D
    return labels


def _to_metres(x: Array, y: Array) -> FloatArray:
    """Convert normalized Metrica coordinates to centered pitch metres."""
    return cast(
        FloatArray,
        np.stack([(x - 0.5) * PITCH[0], (0.5 - y) * PITCH[1]], axis=-1).astype(np.float32),
    )


def _read_csv_team(path: Path) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    """(period, frame, players [T, P, 2] metres, ball [T, 2] metres) from a Metrica CSV."""
    with path.open(encoding="utf-8") as handle:
        header = [next(csv.reader(handle)) for _ in range(3)][2]
    data = np.loadtxt(path, delimiter=",", skiprows=3, dtype=np.float64)
    player_cols = [i for i, h in enumerate(header) if h.startswith("Player")]
    ball_col = header.index("Ball")
    players = _to_metres(data[:, player_cols], data[:, [c + 1 for c in player_cols]])
    ball = _to_metres(data[:, ball_col], data[:, ball_col + 1])
    return (
        cast(IntArray, data[:, 0].astype(np.int64)),
        cast(IntArray, data[:, 1].astype(np.int64)),
        players,
        ball,
    )


def _read_csv_events(path: Path) -> list[Event]:
    """Parse annotated Metrica events into normalized event records.

    Args:
        path: CSV containing team, type, period, and frame interval fields.

    Returns:
        Events in source order with normalized team and event labels.
    """
    with path.open(encoding="utf-8") as handle:
        return [
            Event(
                team="H" if r["Team"] == "Home" else "A" if r["Team"] == "Away" else "",
                kind=r["Type"].strip().upper(),
                subtype=(r["Subtype"] or "").strip().upper(),
                period=int(r["Period"]),
                start=int(r["Start Frame"]),
                end=int(r["End Frame"]),
            )
            for r in csv.DictReader(handle)
        ]


def _assemble(
    name: str,
    period: Array,
    frame: Array,
    home: Array,
    away: Array,
    ball: Array,
    events: Sequence[Event],
) -> CanonicalMatch:
    """Join Metrica tracking periods and event-derived labels into one canonical match."""
    parts: list[tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int8]]] = []
    for p in (1, 2):
        sel = period == p
        if not sel.any():
            continue
        f = frame[sel]
        labels = events_to_labels([e for e in events if e.period == p], f)
        b = np.concatenate([ball[sel], np.full((sel.sum(), 1), np.nan, np.float32)], axis=1)
        parts.append((b, home[sel], away[sel], labels))
    b, h, a, labels, per = join_periods(parts)
    return CanonicalMatch(
        match_id=f"metrica:{name}",
        provider="metrica",
        ball=b,
        home=h,
        away=a,
        labels=labels,
        period=per,
        pitch=PITCH,
        meta={
            "licence": LICENCE,
            "label_method": (
                "algorithmic from Metrica event data: possession events, with dead ball from "
                "out/foul until the next possession event"
            ),
            "source": "github.com/metrica-sports/sample-data",
        },
    )


def convert_csv_game(folder: Path) -> CanonicalMatch:
    """Sample Games 1 and 2 (CSV tracking per team + CSV events)."""
    name = folder.name
    period, frame, home, ball = _read_csv_team(folder / f"{name}_RawTrackingData_Home_Team.csv")
    period_a, frame_a, away, _ = _read_csv_team(folder / f"{name}_RawTrackingData_Away_Team.csv")
    if not (np.array_equal(frame, frame_a) and np.array_equal(period, period_a)):
        raise ValueError(f"{name}: home/away tracking frames differ")
    events = _read_csv_events(folder / f"{name}_RawEventsData.csv")
    return _assemble(name, period, frame, home, away, ball, events)


def _epts_layout(meta: Path) -> tuple[str, dict[str, str], list[tuple[int, int, list[str]]]]:
    """Home team id, ``player_id -> team_id`` and per-frame-range ordered player ids."""
    import xml.etree.ElementTree as ET  # pylint: disable=import-outside-toplevel

    root = ET.parse(meta).getroot()
    teams = list(root.iter("Team"))
    if not teams:
        raise ValueError(f"{meta}: metadata has no Team elements")
    home_id = _required_attr(teams[0], "id")
    team_of = {pl.get("id"): pl.get("teamId") for pl in root.iter("Player")}
    channel_player = {pc.get("id"): pc.get("playerId") for pc in root.iter("PlayerChannel")}
    ranges = []
    for spec in root.iter("DataFormatSpecification"):
        order: list[str] = []
        for ref in spec.iter("PlayerChannelRef"):
            channel_id = ref.get("playerChannelId")
            if not channel_id or not channel_id.endswith("_x"):
                continue
            player_id = channel_player.get(channel_id)
            if player_id is not None:
                order.append(player_id)
        ranges.append(
            (
                int(_required_attr(spec, "startFrame")),
                int(_required_attr(spec, "endFrame")),
                order,
            )
        )
    return str(home_id), {str(k): str(v) for k, v in team_of.items()}, ranges


def _required_attr(element: Any, name: str) -> str:
    """Return a required XML attribute or fail with a useful metadata error."""
    value = element.get(name)
    if value is None:
        raise ValueError(f"metadata element {element.tag!r} is missing {name!r}")
    return str(value)


def convert_epts_game(folder: Path) -> CanonicalMatch:
    """Sample Game 3 (FIFA EPTS tracking — same layout as the challenge ATD — + JSON events).

    Player order changes between data-format frame ranges (substitutions), so each player gets a
    stable slot per team. Periods come from the event data (no Period element in the metadata):
    period 2 starts at its first event's frame.
    """
    name = folder.name
    home_id, team_of, ranges = _epts_layout(folder / f"{name}_metadata.xml")
    home_ids = sorted(
        {pid for _, _, order in ranges for pid in order if team_of.get(pid) == home_id}
    )
    away_ids = sorted(
        {pid for _, _, order in ranges for pid in order if team_of.get(pid) != home_id}
    )
    hslot = {pid: i for i, pid in enumerate(home_ids)}
    aslot = {pid: i for i, pid in enumerate(away_ids)}
    n_frames = max(hi for _, hi, _ in ranges)
    home = np.full((n_frames, len(home_ids), 2), np.nan, np.float32)
    away = np.full((n_frames, len(away_ids), 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    frame = np.arange(1, n_frames + 1)
    with (folder / f"{name}_tracking.txt").open(encoding="utf-8") as handle:
        for line in handle:
            head, body, ball_txt = line.rstrip("\n").split(":")
            f = int(head)
            order = next(o for lo, hi, o in ranges if lo <= f <= hi)
            for pid, group in zip(order, body.split(";"), strict=False):
                if not group or group[0] == "N":
                    continue
                gx, gy = group.split(",")[:2]
                xy = ((float(gx) - 0.5) * PITCH[0], (0.5 - float(gy)) * PITCH[1])
                if pid in hslot:
                    home[f - 1, hslot[pid]] = xy
                else:
                    away[f - 1, aslot[pid]] = xy
            bf = ball_txt.split(",")
            if bf and bf[0][:1] not in ("N", ""):
                ball[f - 1] = ((float(bf[0]) - 0.5) * PITCH[0], (0.5 - float(bf[1])) * PITCH[1])

    raw = json.loads((folder / f"{name}_events.json").read_text(encoding="utf-8"))["data"]

    def subtype(e: dict[str, object]) -> str:
        """Normalize an optional provider subtype field to uppercase text."""
        sub = e.get("subtypes")
        if isinstance(sub, list):
            return " ".join(
                str(x["name"]).upper() for x in sub if isinstance(x, dict) and "name" in x
            )
        if isinstance(sub, dict):
            return str(sub["name"]).upper()
        return ""

    events = [
        Event(
            team="H" if e["team"]["id"] == home_id else "A",
            kind=str(e["type"]["name"]).upper(),
            subtype=subtype(e),
            period=int(e["period"]),
            start=int(e["start"]["frame"]),
            end=int(e["end"]["frame"] or e["start"]["frame"]),
        )
        for e in raw
    ]
    p2_start = min(e.start for e in events if e.period == 2)
    period = np.where(frame >= p2_start, 2, 1)
    return _assemble(name, period, frame, home, away, ball, events)
