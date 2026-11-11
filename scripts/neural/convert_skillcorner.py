"""Convert SkillCorner open-data matches to canonical ``.npz`` and print a dataset card.

Usage::

    python scripts/neural/convert_skillcorner.py [--src data/external/skillcorner/opendata/data/matches]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from possession_lab.neural.canonical import label_summary  # noqa: E402
from possession_lab.neural.skillcorner import convert  # noqa: E402


def main() -> int:
    """Convert every match folder that has tracking and phases."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src", type=Path, default=ROOT / "data/external/skillcorner/opendata/data/matches"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/canonical")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cards = {}
    for folder in sorted(p for p in args.src.iterdir() if p.is_dir()):
        target = args.out / f"skillcorner_{folder.name}.npz"
        if target.is_file() and not args.force:
            continue
        if not (folder / f"{folder.name}_phases_of_play.csv").is_file():
            print(f"skip {folder.name}: no phases")
            continue
        start = time.perf_counter()
        match = convert(folder)
        match.save(target)
        card = {
            "frames": match.n_frames,
            "ball_observed": float((~np.isnan(match.ball[:, 0]))[match.period > 0].mean()),
            "players_observed_mean": float(
                (~np.isnan(np.concatenate([match.home, match.away], 1)[..., 0]))
                .sum(1)[match.period > 0]
                .mean()
            ),
            **label_summary(match.labels),
            "seconds": round(time.perf_counter() - start, 1),
        }
        cards[folder.name] = card
        print(folder.name, json.dumps(card), flush=True)
    card_path = args.out / "skillcorner_cards.json"
    existing = json.loads(card_path.read_text()) if card_path.is_file() else {}
    existing.update(cards)
    card_path.write_text(json.dumps(existing, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
