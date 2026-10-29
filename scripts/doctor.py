"""Report local environment readiness without reading licensed asset contents."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "games.py",
    "starter/baseline.py",
    "evaluation/run_evaluation.py",
    "evaluation/validate_submission.py",
    "data/tactical/BAR-ATH_tactical_atd.xml",
    "data/tactical/BAR-ATH_tactical_atd.txt",
    "data/tactical/BAR-ATH_tactical_smart_tagging.xml",
    "data/broadcast/BAR-ATH_broadcast_atd.xml",
)


def main() -> int:
    """Check that the repository and local assets are where expected."""
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Project root: {ROOT}")
    missing = []
    for rel in REQUIRED:
        present = (ROOT / rel).is_file()
        print(f"{'OK' if present else 'MISSING'} {rel}")
        if not present:
            missing.append(rel)
    if missing:
        print("Some prerequisites are missing. Do not download private data automatically.")
        return 1
    print("All checked paths exist; content parsing was not performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
