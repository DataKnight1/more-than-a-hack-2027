"""Load Smart Tagging intervals and check the tag clock against published kickoff anchors."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

from games import GAME, GAMES, require

KICKOFF_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"
KICKOFF_VIDEO_FRAMES = {"1st Half": 11185, "2nd Half": 113742}


def load_smart_tagging(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    """Read tagging instances as half-open video-frame intervals with their label groups.

    Args:
        game: Competition game key defined by the organizer's ``games.py``.
        data_dir: Optional local root of the licensed data bundle.
        fps: Frame rate override; defaults to the game's configured frame rate.

    Returns:
        Instances in file order, each with code, start/end time, start/end frame, and tags.

    Raises:
        FileNotFoundError: If the Smart Tagging XML is absent.
        ValueError: If an interval start or end is not numeric.
    """
    source_path = require(["smart_tagging"], game, data_dir)[0]
    frame_rate = fps or GAMES[game]["fps"]
    events: list[dict[str, Any]] = []
    for instance in ET.parse(source_path).iter("instance"):
        start_s = float(instance.findtext("start") or "")
        end_s = float(instance.findtext("end") or "")
        tags = {
            label.findtext("group"): label.findtext("text") for label in instance.findall("label")
        }
        events.append(
            {
                "code": instance.findtext("code"),
                "start_s": start_s,
                "end_s": end_s,
                "start_frame": int(round(start_s * frame_rate)),
                "end_frame": int(round(end_s * frame_rate)),
                "tags": tags,
            }
        )
    return events


def check_kickoff_alignment(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that each published real kickoff falls inside its tagged restart interval.

    Args:
        events: Instances returned by :func:`load_smart_tagging`.

    Returns:
        A report containing kickoff-tag count, per-half interval matches, and overall status.
    """
    kickoffs = [
        event
        for event in events
        if (event["code"] or "").strip().lower() == KICKOFF_CODE
        and (event["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE
    ]
    halves: list[dict[str, Any]] = []
    for half, expected_frame in sorted(KICKOFF_VIDEO_FRAMES.items()):
        tagged = [event for event in kickoffs if event["tags"].get("Half") == half]
        match = next(
            (
                event
                for event in tagged
                if event["start_frame"] <= expected_frame <= event["end_frame"]
            ),
            None,
        )
        halves.append(
            {
                "half": half,
                "true_kickoff_frame": expected_frame,
                "tagged": [(event["start_frame"], event["end_frame"]) for event in tagged],
                "contains_kickoff": match is not None,
            }
        )
    return {
        "n_kickoff_tags": len(kickoffs),
        "halves": halves,
        "agrees": all(half["contains_kickoff"] for half in halves),
    }
