"""ATD parsing on a hand-written three-frame file; no licensed data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from possession_lab.atd import load_tracking, parse_raw

LINES = [
    "1:0.10,0.20,1.5;0.30,0.40,2.0;NaN,NaN,NaN:0.11,0.21,0.0",
    "2:0.12,0.22,1.0;NaN,NaN,NaN;0.50,0.60,3.0:NaN,NaN,NaN",
    "3:0.14,0.24,1.1;NaN;0.52,0.62,2.5:0.90,0.10,5.0,1.0",  # short group, long ball group
]


@pytest.fixture(name="raw")
def raw_file(tmp_path: Path) -> Path:
    """The three records above as a raw ATD file."""
    path = tmp_path / "atd.txt"
    path.write_text("\n".join(LINES) + "\n")
    return path


def test_parse(raw: Path) -> None:
    """Values land on frame_id - 1; short groups pad with NaN without shifting slots."""
    ball, players, speed = parse_raw(raw, n_frames=4, n_slots=3)
    np.testing.assert_allclose(ball[0], [0.11, 0.21, 0.0], rtol=1e-6)
    np.testing.assert_allclose(ball[2], [0.90, 0.10, 5.0], rtol=1e-6)
    assert np.isnan(ball[1]).all() and np.isnan(ball[3]).all()
    np.testing.assert_allclose(players[2, 2], [0.52, 0.62], rtol=1e-6)
    assert np.isnan(players[2, 1]).all() and speed[0, 1] == pytest.approx(2.0)


def test_rejects_mismatches(raw: Path) -> None:
    """Wrong slot count or frame range is an error, not a silent shift."""
    with pytest.raises(ValueError, match="player groups"):
        parse_raw(raw, n_frames=4, n_slots=4)
    with pytest.raises(ValueError, match="outside"):
        parse_raw(raw, n_frames=2, n_slots=3)


def test_cache(raw: Path, tmp_path: Path) -> None:
    """The second load comes from the cache and is identical."""
    teams = np.array([0, 0, 1], dtype=np.int8)
    first = load_tracking(raw, teams, 4, tmp_path / "cache")
    second = load_tracking(raw, teams, 4, tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    np.testing.assert_array_equal(first.players, second.players)
