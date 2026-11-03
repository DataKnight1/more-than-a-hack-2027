"""Possession decoder: per-frame H/A/D evidence from the tracking, then Viterbi.

Every label change has a fixed cost, so a touch or a deflection can't flip possession and frames
without a ball just keep the current label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from possession_lab.atd import Tracking, load_tracking

CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "local" / "cache"
LABELS = np.array(["H", "A", "D"])
OUT_MARGIN_M = 0.3  # about the tracker's position noise
Array = NDArray[Any]
Pitch = tuple[float, float]


@dataclass(frozen=True)
class HmmParams:  # pylint: disable=too-many-instance-attributes
    """Model settings. Football/physics priors, not tuned on the public GT."""

    control_m: float = 1.5  # a player this close to the ball controls it
    contrast_m: float = 1.5  # home/away nearest-player gap that counts as decisive
    flight_mps: float = 8.0  # a faster ball is in flight (a pass), so proximity means little
    alive_share: float = 0.9  # how much "someone is on the ball" argues the ball is live
    slow_weight: float = 0.35  # players in their slow regime -> stoppage
    out_weight: float = 2.0  # ball beyond a line, or lost after leaving the pitch
    gap_weight: float = 0.2  # ball lost for longer than gap_s
    gap_s: float = 1.0
    rolling_weight: float = 0.8  # a rolling ball inside the pitch -> live play
    spread_weight: float = 0.006  # per metre the home team is more spread out than the away team
    team_switch_s: float = 0.15  # evidence needed for a hand-over
    dead_switch_s: float = 1.0  # evidence needed to start or end a stoppage

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"hmm.{item.name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"hmm.{item.name} must be finite and >= 0")
        if self.alive_share > 1:
            raise ValueError("hmm.alive_share must be <= 1")

    @classmethod
    def from_table(cls, table: Any) -> HmmParams:
        """Build from the ``[hmm]`` TOML table, rejecting unknown keys."""
        if not isinstance(table, dict):
            raise ValueError("[hmm] must be a TOML table")
        unknown = set(table) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unsupported [hmm] keys: {sorted(unknown)}")
        return cls(**table)


class Features(NamedTuple):
    """Per-frame inputs. Distances are inf and speeds NaN where unknown."""

    visible: Array
    ball: Array  # (n, 2) in metres from a corner
    d_home: Array  # nearest home player to the ball
    d_away: Array
    ball_speed: Array
    activity: Array  # mean player speed
    spread: Array  # home minus away team spread in metres; the team on the ball spreads out


def smooth(x: Array, half: int) -> Array:
    """Centred NaN-aware moving average over +/- half frames."""
    ok = np.isfinite(x)
    total, count = np.r_[0, np.cumsum(np.where(ok, x, 0))], np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(x))
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)


def team_spread(points: Array) -> Array:
    """Std of x plus std of y over a team's tracked players, NaN when nobody is tracked."""
    n = np.isfinite(points[..., 0]).sum(axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(points, axis=1) / n
        var = np.nansum((points - mean[:, None]) ** 2, axis=1) / n
    spread: Array = np.sqrt(var).sum(axis=1)
    return spread


def features(tr: Tracking, pitch: Pitch, fps: int) -> Features:
    """Ball-to-team distances, ball speed and player activity for every frame."""
    scale = np.array(pitch, dtype=np.float32)
    xy = tr.ball[:, :2] * scale
    visible = np.isfinite(xy).all(axis=1)
    players = tr.players * scale
    dist = np.hypot(*np.moveaxis(players - xy[:, None], -1, 0))
    dist = np.where(np.isnan(dist), np.inf, dist)
    step = np.r_[np.nan, np.hypot(*np.diff(xy, axis=0).T) * fps]
    speed = np.clip(tr.speed, 0, 12)
    n = np.isfinite(speed).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        activity = np.where(n >= 8, np.nansum(speed, axis=1) / n, np.nan)
    return Features(
        visible,
        xy,
        dist[:, tr.slot_team == 0].min(axis=1, initial=np.inf),
        dist[:, tr.slot_team == 1].min(axis=1, initial=np.inf),
        np.where(visible, smooth(np.minimum(step, 40), 2), np.nan),
        activity,
        smooth(
            team_spread(players[:, tr.slot_team == 0]) - team_spread(players[:, tr.slot_team == 1]),
            fps,
        ),
    )


def team_evidence(f: Features, p: HmmParams) -> Array:
    """Positive for home, negative for away: who is on the ball, plus a nudge from team spread."""
    near = np.minimum(f.d_home, f.d_away)
    with np.errstate(invalid="ignore", divide="ignore"):
        proximity = np.exp(-np.maximum(near - p.control_m, 0) / 2.0)
        contrast = np.nan_to_num(np.tanh((f.d_away - f.d_home) / p.contrast_m))
        grounded = 1 / (1 + (np.nan_to_num(f.ball_speed) / p.flight_mps) ** 4)
    on_ball = np.where(f.visible & np.isfinite(near), proximity * contrast * grounded, 0.0)
    return on_ball + p.spread_weight * np.nan_to_num(f.spread)


def slow_log_odds(activity: Array, half: int, fit: Array) -> Array:
    """Log-odds that players are in their slow regime; two-Gaussian fit on the fit frames."""
    x = np.log(smooth(activity, half) + 0.1)
    ok = np.isfinite(x)
    out = np.zeros(len(x))
    if (ok & fit).sum() < 100:
        return out
    xs = x[ok & fit, None]
    w, mu, sd = np.full(2, 0.5), np.quantile(xs, [0.25, 0.75]), np.full(2, xs.std() / 2 + 1e-3)
    for _ in range(100):
        logp = np.log(w) - np.log(sd) - 0.5 * ((xs - mu) / sd) ** 2
        r = np.exp(logp - logp.max(axis=1, keepdims=True))
        r /= r.sum(axis=1, keepdims=True)
        w, mu = r.mean(axis=0), (r * xs).sum(axis=0) / r.sum(axis=0)
        sd = np.sqrt((r * (xs - mu) ** 2).sum(axis=0) / r.sum(axis=0)) + 1e-3
    xs = x[ok, None]
    logp = np.log(w) - np.log(sd) - 0.5 * ((xs - mu) / sd) ** 2
    slow = int(mu.argmin())
    out[ok] = np.clip(logp[:, slow] - logp[:, 1 - slow], -3, 3)
    return out


def run_lengths(mask: Array) -> Array:
    """Length of the True run each frame belongs to, 0 where False."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=np.int64)
    out[mask] = np.repeat(stops - starts, stops - starts)
    return out


def dead_evidence(f: Features, p: HmmParams, pitch: Pitch, fps: int, graded: Array) -> Array:
    """Positive where the ball looks out of play, negative where it looks live."""
    x, y = f.ball[:, 0], f.ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = f.visible & (np.maximum.reduce([-x, x - pitch[0], -y, y - pitch[1]]) > OUT_MARGIN_M)
    # The tracker usually loses a ball that has left the pitch, so "out" carries through the gap.
    last_seen = np.maximum.accumulate(np.where(f.visible, np.arange(len(x)), 0))
    out |= ~f.visible & out[last_seen]
    rolling = np.where(f.visible & ~out, np.tanh(np.nan_to_num(f.ball_speed) / 4.0), 0.0)
    evidence: Array = (
        p.slow_weight * slow_log_odds(f.activity, fps, graded)
        + p.out_weight * out
        + p.gap_weight * (run_lengths(~f.visible) >= p.gap_s * fps)
        - p.rolling_weight * rolling
    )
    return evidence


def viterbi(scores: Array, team_cost: float, dead_cost: float) -> NDArray[np.int8]:
    """Best H/A/D path (codes 0/1/2) given per-frame scores and per-change costs."""
    cost = np.array([[0, team_cost, dead_cost], [team_cost, 0, dead_cost], [dead_cost] * 2 + [0]])
    back = np.zeros((len(scores), 3), dtype=np.int8)
    value = np.asarray(scores[0], dtype=np.float64)
    for t in range(1, len(scores)):
        step = value[:, None] - cost  # step[previous, current]
        back[t] = step.argmax(axis=0)
        value = step.max(axis=0) + scores[t]
    path = np.empty(len(scores), dtype=np.int8)
    path[-1] = value.argmax()
    for t in range(len(scores) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_to_contact(codes: Array, f: Features, p: HmmParams, fps: int) -> Array:
    """Move each hand-over to the first frame (within 1 s) where the new team reaches the ball."""
    out = codes.copy()
    near = (f.d_home, f.d_away)
    for t in (np.flatnonzero(codes[1:] != codes[:-1]) + 1).tolist():
        old, new = int(codes[t - 1]), int(codes[t])
        if 2 in (old, new):
            continue
        reached = np.flatnonzero(near[new][t : t + fps] < p.control_m)
        if len(reached) and np.all(codes[t : t + reached[0]] == new):
            out[t : t + reached[0]] = old
    return out


def decode(
    f: Features, p: HmmParams, blocks: list[tuple[int, int]], pitch: Pitch, fps: int
) -> Array:
    """Labels for every frame: Viterbi inside each [start, stop) block, X elsewhere."""
    graded = np.zeros(len(f.visible), dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True
    margin = team_evidence(f, p)
    dead = dead_evidence(f, p, pitch, fps, graded) + (1 - p.alive_share) * np.abs(margin) / 2
    scores = np.column_stack([margin / 2, -margin / 2, dead])
    labels = np.full(len(margin), "X")
    for start, stop in blocks:
        path = viterbi(scores[start:stop], p.team_switch_s * fps, p.dead_switch_s * fps)
        f_block = Features(*(x[start:stop] for x in f))
        labels[start:stop] = LABELS[snap_to_contact(path, f_block, p, fps)]
    return labels


def summary(labels: Array, blocks: list[tuple[int, int]], fps: int) -> str:
    """Quick stats on our own output (no ground truth involved)."""
    graded = np.concatenate([labels[a:b] for a, b in blocks])
    spells = [
        len(run)
        for a, b in blocks
        for run in np.split(labels[a:b], np.flatnonzero(labels[a + 1 : b] != labels[a : b - 1]) + 1)
        if run[0] != "D"
    ]
    alive = graded != "D"
    return (
        f"{len(spells)} possessions, mean {np.mean(spells or [0]) / fps:.1f} s, "
        f"dead {np.mean(~alive):.0%}, home {np.mean(graded[alive] == 'H'):.0%} of alive"
    )


def predict_hmm(p: HmmParams) -> Array:
    """Run on the local data bundle (data/ or MTAH_DATA_DIR)."""
    games, baseline = import_module("games"), import_module("starter.baseline")
    game = games.GAMES[games.GAME]
    fps, pitch = int(game["fps"]), (float(game["pitch"][0]), float(game["pitch"][1]))
    meta, raw = games.require(["atd_meta", "atd_raw"])
    slot_team = np.asarray(baseline.team_of_each_track(meta), dtype=np.int8)
    tracking = load_tracking(Path(raw), slot_team, int(games.source_frames("tactical")), CACHE_DIR)
    blocks = [(int(a), int(b) + 1) for _, a, b in games.half_video_blocks(games.GAME)]
    labels = decode(features(tracking, pitch, fps), p, blocks, pitch, fps)
    print(summary(labels, blocks, fps))
    return labels
