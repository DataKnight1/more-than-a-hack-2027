"""Decoder, evidence and end-to-end checks on synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from possession_lab.config import load_config
from possession_lab.hmm import (
    Features,
    HmmParams,
    dead_evidence,
    slow_log_odds,
    snap_to_contact,
    summary,
    team_evidence,
    viterbi,
)
from possession_lab.model import FRAME_COUNT, predict

ROOT = Path(__file__).resolve().parents[1]
FPS, PITCH = 25, (105.0, 68.0)


def scores(margin: list[float], dead: list[float]) -> np.ndarray:
    """Per-frame scores built the way decode() builds them (alive_share 0.5)."""
    m = np.array(margin)
    return np.column_stack([m / 2, -m / 2, np.array(dead) + np.abs(m) / 4])


@pytest.mark.parametrize(
    ("margin", "dead", "expected"),
    [
        ([1] * 40 + [-1] * 5 + [1] * 40, [0] * 85, "H" * 85),  # a touch is not a hand-over
        ([1] * 30 + [0] * 200 + [1] * 30, [-0.2] * 30 + [-0.05] * 200 + [-0.2] * 30, "H" * 260),
        ([1] * 100 + [0] * 100 + [-1] * 100, [-0.2] * 100 + [2] * 100 + [-0.2] * 100, None),
    ],
    ids=["flicker", "lost-ball-live-play", "stoppage"],
)
def test_viterbi(margin: list[float], dead: list[float], expected: str | None) -> None:
    """Short flickers and live-play ball losses keep possession; strong dead evidence is D."""
    path = "".join("HAD"[c] for c in viterbi(scores(margin, dead), 15, 25))
    if expected:
        assert path == expected
    else:
        assert path[0] == "H" and path[110:190] == "D" * 80 and path[-1] == "A"


def test_hand_over_lands_near_change() -> None:
    """Sustained opposing evidence flips the label within a few frames of the change."""
    path = viterbi(scores([1] * 60 + [-1] * 60, [0] * 120), 15, 25)
    assert int(np.flatnonzero(path == 1)[0]) in range(57, 64)


def make(
    visible: list[bool], d_home: list[float], d_away: list[float], **kw: list[float]
) -> Features:
    """Small Features with sensible defaults."""
    n = len(visible)
    return Features(
        np.array(visible),
        np.array(kw.get("ball", [[50.0, 30.0]] * n), dtype=float),
        np.array(d_home),
        np.array(d_away),
        np.array(kw.get("speed", [2.0] * n)),
        np.full(n, np.nan),
    )


def test_team_evidence() -> None:
    """Close home player -> ~+1; ball far from everyone or in flight -> weak; lost -> 0."""
    f = make(
        [True] * 3 + [False], [0.5, 12, 0.5, np.inf], [6, 20, 6, np.inf], speed=[2, 2, 25, np.nan]
    )
    ev = team_evidence(f, HmmParams())
    assert ev[0] > 0.9 and 0 < ev[1] < 0.05 and ev[2] < 0.2 and ev[3] == 0


def test_out_of_bounds_carries_through_gap() -> None:
    """A ball seen beyond a line keeps counting as out while the tracker has lost it."""
    ball = [[50.0, 30.0], [-3.0, 30.0]] + [[np.nan, np.nan]] * 3
    f = make([True, True] + [False] * 3, [1.0] * 5, [5.0] * 5, ball=ball, speed=[0.0] * 5)
    ev = dead_evidence(f, HmmParams(gap_weight=0, slow_weight=0), PITCH, FPS, np.ones(5, bool))
    assert ev[0] == 0 and np.all(ev[1:] == 2.0)


def test_hand_over_snaps_to_first_touch() -> None:
    """A switch decided mid-pass moves to when the new team actually reaches the ball."""
    away = [6.0] * 10 + [3.0] * 5 + [1.0] * 5
    f = make([True] * 20, [1.0] * 20, away)
    codes = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    assert snap_to_contact(codes, f, HmmParams(), FPS).tolist() == [0] * 15 + [1] * 5


def test_slow_regime_log_odds() -> None:
    """Walking stretches get positive log-odds, running stretches negative."""
    activity = np.r_[np.full(2000, 2.3), np.full(1000, 0.9)] + np.random.default_rng(0).normal(
        0, 0.1, 3000
    )
    odds = slow_log_odds(activity, 25, np.ones(3000, bool))
    assert odds[500] < 0 < odds[2500]


def test_config_and_validation() -> None:
    """The committed config loads; bad values fail before any data is read."""
    config = load_config(ROOT / "configs/models/possession_hmm.toml")
    assert config.family == "possession_hmm" and config.hmm == HmmParams()
    for bad in ({"unknown": 1}, {"gap_s": -1}, {"control_m": True}, {"alive_share": 2}):
        with pytest.raises(ValueError):
            HmmParams.from_table(bad)


def test_summary() -> None:
    """Stoppages split possessions; frames outside the blocks are ignored."""
    assert summary(np.array(list("XHHHDDHHAAX")), [(1, 10)], fps=1) == (
        "3 possessions, mean 2.3 s, dead 22%, home 71% of alive"
    )


def test_end_to_end_on_fake_bundle(bundle) -> None:  # type: ignore[no-untyped-def]
    """Real I/O path: valid shape, X off the halves, hand-over found within a second."""
    labels = predict(load_config(ROOT / "configs/models/possession_hmm.toml"))
    k, h = bundle.kickoff, bundle.half
    assert labels.shape == (FRAME_COUNT,) and set(labels[:k]) == {"X"}
    assert (
        np.mean(labels[k : k + h] == "H") > 0.95
        and np.mean(labels[k + h : k + 2 * h] == "A") > 0.95
    )
    assert abs(int(np.flatnonzero(labels[k:] == "A")[0]) - h) <= 25 + 50  # ball lost 2 s before
