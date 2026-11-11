"""Convert the 3 Metrica sample games to canonical ``.npz`` and print dataset cards.

Usage::

    python scripts/neural/convert_metrica.py [--src data/external/metrica/sample-data/data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from possession_lab.neural.canonical import label_summary  # noqa: E402
from possession_lab.neural.metrica import convert_csv_game, convert_epts_game  # noqa: E402


def main() -> int:
    """Convert Sample_Game_1..3."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=ROOT / "data/external/metrica/sample-data/data")
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/canonical")
    args = parser.parse_args()
    cards = {}
    for game in ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"):
        folder = args.src / game
        match = convert_epts_game(folder) if game.endswith("3") else convert_csv_game(folder)
        match.save(args.out / f"metrica_{game.split('_')[-1]}.npz")
        graded = match.period > 0
        cards[game] = {
            "frames": match.n_frames,
            "home_slots": match.home.shape[1],
            "away_slots": match.away.shape[1],
            "ball_observed": float((~np.isnan(match.ball[:, 0]))[graded].mean()),
            **label_summary(match.labels),
        }
        print(game, json.dumps(cards[game]), flush=True)
    (args.out / "metrica_cards.json").write_text(json.dumps(cards, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
