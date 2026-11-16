"""Delegate validation to the official, unchanged competition validator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Validate supplied CSV path using the official validation program."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_submission.py <candidate.csv>")
        return 2
    check = ROOT / "evaluation" / "validate_submission.py"
    result = subprocess.run([sys.executable, str(check), sys.argv[1]], cwd=ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
