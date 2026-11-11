"""Score organiser-semantics nearest-player baselines on external canonical matches (numpy only).

Reports clean and deterministically degraded conditions, so neural candidates have a reference on
exactly the same inputs. External labelled data only; the challenge GT is never read.

Usage::

    python scripts/neural/baseline_external.py [--matches idsse_J03WN1 ...] [--windows 0 25 50 125]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from possession_lab.neural.baselines import baseline_codes  # noqa: E402
from possession_lab.neural.canonical import CanonicalMatch, X  # noqa: E402
from possession_lab.neural.external_eval import score  # noqa: E402
from possession_lab.neural.matchops import degrade_match  # noqa: E402


def main() -> int:
    """Print a table and write ``artifacts/neural/baseline_external.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canonical", type=Path, default=ROOT / "data/external/canonical")
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=[0, 25, 50, 125])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    names = args.matches or sorted(p.stem for p in args.canonical.glob("*.npz"))
    report: dict[str, dict[str, dict[str, float]]] = {}
    print(f"{'match':<26}{'cond':<10}" + "".join(f"{'w' + str(w):>16}" for w in args.windows))
    for name in names:
        clean = CanonicalMatch.load(args.canonical / f"{name}.npz")
        for cond, match in (("clean", clean), ("degraded", degrade_match(clean, args.seed))):
            cells = []
            for window in args.windows:
                codes = baseline_codes(match.ball, match.home, match.away, window)
                codes[match.period == 0] = X
                result = score(match.labels, codes)
                report.setdefault(name, {})[f"{cond}_w{window}"] = {
                    "spell_f1_mean": result["spell_f1_mean"],
                    "macro_f1": result["macro_f1"],
                    "n_pred_spells": result["n_pred_spells"],
                    "n_gt_spells": result["n_gt_spells"],
                }
                cells.append(f"{result['spell_f1_mean']:.3f}/{result['macro_f1']:.3f}")
            print(f"{name:<26}{cond:<10}" + "".join(f"{c:>16}" for c in cells), flush=True)
    out = ROOT / "artifacts" / "neural" / "baseline_external.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"(cells are spellF1/macroF1)  written {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
