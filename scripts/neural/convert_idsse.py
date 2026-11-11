"""Convert downloaded IDSSE / DFL matches to canonical ``.npz`` and print a dataset card.

Usage::

    .venv-neural/Scripts/python scripts/neural/convert_idsse.py [--raw ...] [--out ...] [--only J03WMX]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from possession_lab.neural.canonical import label_summary  # noqa: E402
from possession_lab.neural.idsse import convert  # noqa: E402


def main() -> int:
    """Convert every positions file that has a matching match-information file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data/external/idsse/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/canonical")
    parser.add_argument("--only", nargs="*", default=None, help="match codes, e.g. J03WMX")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cards = {}
    for positions in sorted(args.raw.glob("DFL_04_03_positions_raw_observed_*.xml")):
        code = positions.stem.rsplit("-", 1)[-1]
        if args.only and code not in args.only:
            continue
        target = args.out / f"idsse_{code}.npz"
        info = next(args.raw.glob(f"DFL_02_01_matchinformation_*{code}.xml"))
        if target.is_file() and not args.force:
            print(f"skip {code} (exists)")
            continue
        start = time.perf_counter()
        match = convert(positions, info)
        match.save(target)
        card = {
            "frames": match.n_frames,
            "home_slots": match.home.shape[1],
            "away_slots": match.away.shape[1],
            "ball_observed": float((~__import__("numpy").isnan(match.ball[:, 0])).mean()),
            **label_summary(match.labels),
            "seconds": round(time.perf_counter() - start, 1),
        }
        cards[code] = card
        print(code, json.dumps(card), flush=True)
    if cards:
        card_path = args.out / "idsse_cards.json"
        existing = json.loads(card_path.read_text()) if card_path.is_file() else {}
        existing.update(cards)
        card_path.write_text(json.dumps(existing, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
