"""Access the organizer's Automatic Tracking Data (ATD) with its one-based frame alignment."""

from __future__ import annotations

import os
from typing import Any

from games import DEFAULT_SOURCE, GAME, require, source_frames

FIELD_SEPARATOR = ":"


def atd_paths(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> tuple[str, str]:
    """Return ATD metadata and raw-data paths for the selected recording.

    Args:
        game: Competition game key defined by the organizer's ``games.py``.
        data_dir: Optional local root of the licensed data bundle.
        source: Recording source key, usually tactical or broadcast.

    Returns:
        A pair containing the ATD metadata XML path and raw tracking text path.

    Raises:
        FileNotFoundError: If either required ATD file is absent.
        KeyError: If the game or source is not defined by ``games.py``.
    """
    return tuple(require(["atd_meta", "atd_raw"], game, data_dir, source))


def load_atd(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> Any:
    """Load the selected ATD pair as a Kloppy tracking dataset.

    Coordinates use Kloppy's Metrica convention. The physical pitch dimensions are available in
    ``GAMES[game]["pitch"]`` when a downstream calculation needs metres.

    Args:
        game: Competition game key defined by the organizer's ``games.py``.
        data_dir: Optional local root of the licensed data bundle.
        source: Recording source key, usually tactical or broadcast.

    Returns:
        A Kloppy ``TrackingDataset`` with metadata and tracking frames.

    Raises:
        ImportError: If the optional Kloppy package is not installed.
        FileNotFoundError: If the required ATD files are absent.
    """
    try:
        from kloppy import metrica  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ImportError(
            "ATD loading requires the optional Kloppy dependency; install the starter extras."
        ) from error
    metadata, raw = atd_paths(game, data_dir, source)
    return metrica.load_tracking_epts(meta_data=metadata, raw_data=raw)


def video_frame_of(frame_id: int) -> int:
    """Convert the ATD's one-based record ID to a zero-based video-frame index."""
    return int(frame_id) - 1


def frame_id_of(video_frame: int) -> int:
    """Convert a zero-based video-frame index to the corresponding one-based ATD ID."""
    return int(video_frame) + 1


def check_alignment_raw(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Stream the raw ATD once to verify contiguous IDs and match the expected frame count.

    Args:
        game: Competition game key defined by the organizer's ``games.py``.
        data_dir: Optional local root of the licensed data bundle.
        source: Recording source key, usually tactical or broadcast.

    Returns:
        A diagnostic mapping with row count, ID bounds, contiguity, and alignment status.

    Raises:
        FileNotFoundError: If the raw ATD file is absent.
        ValueError: If a non-empty line does not start with a valid integer frame ID.
    """
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    expected = source_frames(source, game)
    first: int | None = None
    last: int | None = None
    count = 0
    with open(raw_path, encoding="utf-8") as stream:
        for line in stream:
            frame_id = int(line[: line.index(FIELD_SEPARATOR)])
            if first is None:
                first = frame_id
            last = frame_id
            count += 1

    contiguous = count > 0 and last is not None and first is not None and last - first + 1 == count
    return {
        "n_records": count,
        "n_video_frames": expected,
        "frame_id_range": (first, last),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(first) if first is not None else None,
        "agrees": count == expected and contiguous and first == 1,
    }
