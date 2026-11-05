"""Score the possession model on open matches with real labels, never on our own match.

Uses the open DFL dataset (Bassek et al., Scientific Data 12, 195, 2025; CC BY 4.0): seven
Bundesliga matches at 25 Hz with per-frame ball status and team in possession. Each match is
turned into the same arrays our ATD parser returns, then degraded to look like our tracking,
using only aggregate statistics of the real ATD (scripts/data_profile.py): the ball is lost
almost only in long gaps, mostly during stoppages, and about 20 players are tracked.

    python scripts/open_data_check.py --download      # once, ~2.6 GB into artifacts/open_data
    python scripts/open_data_check.py                 # prints spellF1 / macroF1 per match

Settings in configs/models/possession_hmm.toml were picked on the first five matches; J03WQQ and
J03WR9 were held out as a final check.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "evaluation")]

from possession_lab.atd import Tracking  # noqa: E402
from possession_lab.config import load_config  # noqa: E402
from possession_lab.hmm import HmmParams, decode, features  # noqa: E402

DATA = ROOT / "artifacts" / "open_data"
ARTICLE = "https://api.figshare.com/v2/articles/28196177"
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
FPS, PITCH = 25, (105.0, 68.0)


def download() -> None:
    """Fetch match information and positions files (events are not needed)."""
    DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE) as response:
        files = json.load(response)["files"]
    for item in files:
        target = DATA / item["name"]
        if "events" in item["name"] or (target.exists() and target.stat().st_size == item["size"]):
            continue
        print("downloading", item["name"])
        urllib.request.urlretrieve(item["download_url"], target)


def load_match(match: str) -> list[dict[str, np.ndarray]]:
    """Per half: ball (n, 3), players (n, k, 2), team per slot, and true labels 0/1/2."""
    info = ET.parse(next(DATA.glob(f"*matchinformation*{match}.xml"))).getroot()
    home = next(t.get("TeamId") for t in info.iter("Team") if t.get("Role") == "home")
    sets: dict[str, list[tuple[str, dict[str, list[float]]]]] = {}
    for _, el in ET.iterparse(next(DATA.glob(f"*positions*{match}.xml"))):
        if el.tag == "FrameSet" and el.get("TeamId") != "referee":
            cols: dict[str, list[float]] = {
                k: [] for k in ("N", "X", "Y", "Z", "BallStatus", "BallPossession")
            }
            for frame in el:
                for key, values in cols.items():
                    values.append(float(frame.get(key, "nan")))
            sets.setdefault(el.get("GameSection", ""), []).append((el.get("TeamId", ""), cols))
            el.clear()
    halves = []
    for section in ("firstHalf", "secondHalf"):
        ball = next(c for team, c in sets[section] if team == "BALL")
        start, n = int(ball["N"][0]), len(ball["N"])
        people = [(team, c) for team, c in sets[section] if team != "BALL"]
        players = np.full((n, len(people), 2), np.nan)
        for slot, (_, c) in enumerate(people):
            idx = np.asarray(c["N"], dtype=int) - start
            ok = (idx >= 0) & (idx < n)
            players[idx[ok], slot] = np.column_stack([c["X"], c["Y"]])[ok]
        status, owner = np.array(ball["BallStatus"]), np.array(ball["BallPossession"])
        halves.append(
            {
                "ball": np.column_stack([ball["X"], ball["Y"], ball["Z"]]),
                "players": players,
                "teams": np.array([0 if team == home else 1 for team, _ in people], dtype=np.int8),
                "truth": np.where(status == 0, 2, np.where(owner == 1, 0, 1)).astype(np.int8),
            }
        )
    return halves


def degrade(
    half: dict[str, np.ndarray], rng: np.random.Generator, lose_dead: float, live_rate: float
) -> Tracking:
    """Our tracking's flaws: ball lost in long gaps (mostly stoppages), noise, missing players."""
    truth, n = half["truth"], len(half["truth"])
    lost = np.zeros(n, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, truth == 2, 0].astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2], strict=True):  # stoppages: lost after the ball leaves,
        if rng.random() < lose_dead:  # seen again briefly when a long stoppage drags on
            gone, back = (
                a + int(rng.uniform(0.5, 2.5) * FPS),
                b - int(rng.uniform(0, 0.2) * (b - a)),
            )
            lost[gone:back] = True
            c = a + int(rng.uniform(8, 40) * FPS)
            while c < b - 5 * FPS:
                lost[c : c + int(rng.uniform(1, 3) * FPS)] = False
                c += int(rng.uniform(8, 40) * FPS)
    for a in np.flatnonzero((truth != 2) & (rng.random(n) < live_rate / FPS)):  # occlusions in play
        lost[a : a + int(max(4.0, rng.lognormal(np.log(8), 0.6)) * FPS)] = True
    ball = half["ball"].copy()
    ball[:, :2] += rng.normal(0, 0.015, (n, 2))  # our tracker's ball is never perfectly still
    ball[lost] = np.nan
    ball[rng.random(n) > 0.15, 2] = np.nan  # height is only filled in now and then
    clean = half["players"]
    speed = np.full(clean.shape[:2], np.nan)
    speed[1:] = np.hypot(*np.moveaxis(np.diff(clean, axis=0), -1, 0)) * FPS
    players = clean + rng.normal(0, 0.15, clean.shape)
    for slot in range(clean.shape[1]):  # tracklets drop out for a few seconds at a time
        for a in np.flatnonzero(rng.random(n) < 0.02 / FPS):
            players[a : a + int(rng.uniform(2, 10) * FPS), slot] = np.nan
    speed[np.isnan(players[..., 0])] = np.nan
    offset = np.array([PITCH[0] / 2, PITCH[1] / 2])
    ball_norm = np.column_stack([(ball[:, :2] + offset) / PITCH, ball[:, 2]])
    return Tracking(
        ball_norm.astype(np.float32),
        ((players + offset) / PITCH).astype(np.float32),
        speed.astype(np.float32),
        half["teams"],
    )


def evaluate(
    params: HmmParams, seed: int = 0, lose_dead: float = 1.0, live_rate: float = 0.003
) -> dict[str, tuple[float, float]]:
    """spellF1 and macroF1 per match for the model and for the organiser's smooth50 rule."""
    pm = import_module("possession_metrics")
    baseline = import_module("starter.baseline")
    scores = {}
    for match in MATCHES:
        rng = np.random.default_rng(seed)
        truth, ours, vote, runs = [], [], [], []
        for half in load_cached(match):
            f = features(degrade(half, rng, lose_dead, live_rate), PITCH, FPS)
            n, start = len(half["truth"]), sum(len(t) for t in truth)
            runs.append((start, start + n))
            truth.append(half["truth"])
            ours.append(pm.label_codes(decode(f, params, [(0, n)], FPS)))
            nearest = np.where(f.visible, np.where(f.d_home <= f.d_away, 0, 1), -1).astype(np.int8)
            vote.append(pm.label_codes(baseline.smooth(nearest, 50)))
        gt = np.concatenate(truth)
        for name, pred in (("model", np.concatenate(ours)), ("smooth50", np.concatenate(vote))):
            cm = pm.scored_confusion(pm.confusion(gt, pred))
            scores[f"{match} {name}"] = (
                pm.spell_scores(gt, pred, runs)["mean_f1"],
                pm.macro_f1(cm),
            )
    return scores


def load_cached(match: str) -> list[dict[str, np.ndarray]]:
    """load_match, cached as npz next to the downloads."""
    cache = DATA / f"{match}.npz"
    if not cache.exists():
        halves = load_match(match)
        np.savez(cache, **{f"{i}_{k}": v for i, h in enumerate(halves) for k, v in h.items()})
    with np.load(cache) as saved:
        return [
            {k: saved[f"{i}_{k}"] for k in ("ball", "players", "teams", "truth")} for i in (0, 1)
        ]


def main() -> int:
    """Download or evaluate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models/possession_hmm.toml")
    args = parser.parse_args()
    if args.download:
        download()
        return 0
    scores = evaluate(load_config(args.config).hmm)
    for name, (spell, macro) in scores.items():
        print(f"{name:<18} spellF1 {spell:.3f}  macroF1 {macro:.3f}")
    for name in ("model", "smooth50"):
        values = np.array([v for k, v in scores.items() if k.endswith(name)])
        print(
            f"{'mean ' + name:<18} spellF1 {values[:, 0].mean():.3f}  macroF1 {values[:, 1].mean():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
