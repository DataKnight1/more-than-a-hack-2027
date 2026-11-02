"""Decoder, evidence and end-to-end checks on synthetic data only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from possession_lab.config import load_config
from possession_lab.hmm import (
    Features,
    HmmParams,
    dead_evidence,
    decode,
    slow_log_odds,
    smooth,
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
    ev = dead_evidence(f, HmmParams(gap_weight=0, slow_weight=0), PITCH, FPS)
    assert ev[0] == 0 and np.all(ev[1:] == 2.0)


def test_slow_regime_log_odds() -> None:
    """Walking stretches get positive log-odds, running stretches negative."""
    activity = np.r_[np.full(2000, 2.3), np.full(1000, 0.9)] + np.random.default_rng(0).normal(
        0, 0.1, 3000
    )
    odds = slow_log_odds(activity, 25)
    assert odds[500] < 0 < odds[2500]


def test_config_and_validation() -> None:
    """The committed config loads; bad values fail before any data is read."""
    config = load_config(ROOT / "configs/models/possession_hmm.toml")
    assert config.family == "possession_hmm" and config.hmm == HmmParams()
    for bad in ({"unknown": 1}, {"gap_s": -1}, {"control_m": True}, {"alive_share": 2}):
        with pytest.raises(ValueError):
            HmmParams.from_table(bad)


def simulate(seed: int, minutes: int = 10) -> tuple[np.ndarray, Features]:
    """Toy match: possessions, stoppages (slow players, ball out then lost), duels and passes."""
    rng = np.random.default_rng(seed)
    n = minutes * 60 * FPS
    truth, t, team = np.empty(n, dtype=np.int8), 0, 0
    while t < n:
        k = int(rng.lognormal(np.log(10), 0.7) * FPS) + FPS
        truth[t : t + k], t = team, t + k
        if rng.random() < 0.35:
            k = int(rng.lognormal(np.log(12), 0.6) * FPS) + 2 * FPS
            truth[t : t + k], t, team = 2, t + k, int(rng.integers(2))
        else:
            team = 1 - team
    alive = truth != 2

    def bursts(rate: float, length: int) -> np.ndarray:
        return np.convolve(rng.random(n) < rate, np.ones(length), "same") > 0

    duel, flight = alive & bursts(0.005, 15), alive & bursts(0.01, 20)
    near_team = np.where(duel, 1 - truth, truth)
    near = np.where(flight, rng.uniform(3, 12, n), np.abs(rng.normal(1, 0.7, n)))
    far = near + np.abs(rng.normal(3, 2, n))
    first_second = np.r_[True, truth[1:] != truth[:-1]]
    stop_start = np.convolve(first_second & ~alive, np.ones(FPS), "full")[:n] > 0
    visible = np.where(alive, ~bursts(0.004, 40), stop_start)
    ball = np.where(stop_start[:, None] & ~alive[:, None], [-3.0, 30.0], [50.0, 30.0])
    speed = np.where(
        flight, rng.uniform(12, 25, n), np.where(alive, np.abs(rng.normal(4, 3, n)), 0.2)
    )
    activity = smooth(np.where(alive, rng.normal(2.2, 0.5, n), rng.normal(0.9, 0.3, n)), FPS // 2)
    f = Features(
        visible,
        np.where(visible[:, None], ball, np.nan),
        np.where(visible, np.where(near_team == 0, near, far), np.inf),
        np.where(visible, np.where(near_team == 0, far, near), np.inf),
        np.where(visible, speed, np.nan),
        activity,
    )
    return truth, f


def majority_vote(f: Features, half: int = 50) -> np.ndarray:
    """The organiser baseline rule on Features: nearest team, centred vote, D with no ball."""
    home = np.r_[0, np.cumsum(f.visible & (f.d_home <= f.d_away))]
    away = np.r_[0, np.cumsum(f.visible & (f.d_home > f.d_away))]
    i = np.arange(len(f.visible))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(i))
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    return np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beats_majority_vote_on_toy_matches(seed: int) -> None:
    """Scored by the unchanged official metric. Shows the machinery works, nothing more."""
    spec = importlib.util.spec_from_file_location("pm", ROOT / "evaluation/possession_metrics.py")
    assert spec and spec.loader
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    truth, f = simulate(seed)
    runs = [(0, len(truth))]
    ours = pm.label_codes(decode(f, HmmParams(), runs, PITCH, FPS))
    vote = majority_vote(f)
    assert (
        pm.spell_scores(truth, ours, runs)["mean_f1"]
        > pm.spell_scores(truth, vote, runs)["mean_f1"] + 0.1
    )


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
