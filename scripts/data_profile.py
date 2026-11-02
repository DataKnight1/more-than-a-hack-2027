"""Aggregate-only profile of the local match bundle: formats, counts, percentiles, shares.

Prints nothing frame-level (no raw lines, no per-frame values), so the output can be discussed
without passing the data on. Run it locally: ``python scripts/data_profile.py``.
"""

from __future__ import annotations

import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from possession_lab.atd import load_tracking  # noqa: E402
from possession_lab.hmm import features, run_lengths  # noqa: E402

CACHE_DIR = ROOT / "artifacts" / "local" / "cache"


def pct(values: np.ndarray, digits: int = 2) -> str:
    """5/25/50/75/95th percentiles of the finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "n=0"
    marks = np.percentile(v, [5, 25, 50, 75, 95])
    return " ".join(f"{m:.{digits}f}" for m in marks) + f"  (p5..p95, n={v.size})"


def main() -> int:
    """Print the profile."""
    games, baseline = import_module("games"), import_module("starter.baseline")
    game = games.GAMES[games.GAME]
    fps, pitch = int(game["fps"]), (float(game["pitch"][0]), float(game["pitch"][1]))
    meta, raw = games.require(["atd_meta", "atd_raw"])
    teams = np.asarray(baseline.team_of_each_track(meta), dtype=np.int8)
    graded = np.zeros(int(games.source_frames("tactical")), dtype=bool)
    for _, a, b in games.half_video_blocks(games.GAME):
        graded[a : b + 1] = True

    widths: Counter[str] = Counter()
    with open(raw, encoding="utf-8") as handle:
        for _, line in zip(range(5000), handle, strict=False):
            _, players, ball = line.rstrip().split(":")
            widths.update(f"player {g.count(',') + 1}" for g in players.split(";"))
            widths[f"ball {ball.count(',') + 1}"] += 1
    tr = load_tracking(Path(raw), teams, graded.size, CACHE_DIR)
    f = features(tr, pitch, fps)
    seen, lost = graded & f.visible, graded & ~f.visible
    gap = run_lengths(~f.visible) / fps  # for each missing frame, the length of its gap
    starts = lost & np.r_[True, f.visible[:-1]]
    x, y = f.ball[:, 0], f.ball[:, 1]
    beyond = np.maximum.reduce([-x, x - pitch[0], -y, y - pitch[1]])
    speed = f.ball_speed[seen]

    rows = {
        "teams (H, A)": tuple(baseline.team_names(meta)),
        "slots home/away": (int(np.sum(teams == 0)), int(np.sum(teams == 1))),
        "fields per group, first 5000 lines": dict(widths),
        "ball third field": pct(tr.ball[graded, 2]),
        "ball visible, graded": f"{f.visible[graded].mean():.1%} of {graded.sum()} frames",
        "gap length s, per gap": f"{pct(gap[starts], 1)}, {starts.sum()} gaps",
        "ball >1 m beyond a line": f"{np.mean(beyond[seen] > 1):.1%} of seen frames",
        "distance beyond a line m": pct(beyond[seen & (beyond > 0)], 1),
        "ball speed m/s": pct(speed, 1),
        "still ball (<0.3 m/s)": f"{np.mean(speed < 0.3):.1%} of seen frames",
        "nearest player m": pct(np.minimum(f.d_home, f.d_away)[seen]),
        "tracked players": pct(np.isfinite(tr.speed[graded]).sum(axis=1), 0),
        "mean player speed, ball seen": pct(f.activity[seen]),
        "mean player speed, ball lost": pct(f.activity[lost]),
        "mean player speed, gaps >= 4 s": pct(f.activity[lost & (gap >= 4)]),
    }
    for low, high in [(0, 1), (1, 4), (4, 10), (10, 30), (30, np.inf)]:
        share = np.mean((gap[lost] >= low) & (gap[lost] < high))
        rows[f"missing frames in gaps {low}-{high:g} s"] = f"{share:.1%}"
    tags = import_module("starter.load_smart_tagging").load()
    for code, n in Counter(t["code"] for t in tags).most_common():
        durations = np.array(
            [(t["end_frame"] - t["start_frame"]) / fps for t in tags if t["code"] == code]
        )
        rows[f"tag {code}"] = f"n={n}, duration s {pct(durations, 0)}"
    rows["tag Team values"] = dict(Counter(str(t["tags"].get("Team")) for t in tags))
    for label, value in rows.items():
        print(f"{label:<38} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
