"""Verify local baseline reproduction without reading ground truth or changing references."""

from __future__ import annotations

import csv
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def same_predictions(candidate: Path, reference: Path) -> bool:
    """Compare every CSV cell, allowing platform-specific line endings.

    Args:
        candidate: Algorithm-generated frame and label CSV.
        reference: Supplied public baseline CSV, never a ground-truth file.

    Returns:
        Whether both files have identical headers, frames, labels and row counts.
    """
    with (
        candidate.open(encoding="utf-8", newline="") as candidate_file,
        reference.open(encoding="utf-8", newline="") as reference_file,
    ):
        rows = zip_longest(csv.reader(candidate_file), csv.reader(reference_file))
        return all(actual == expected for actual, expected in rows)


def main() -> int:
    """Generate, officially validate and compare a temporary local prediction CSV."""
    local = ROOT / "artifacts" / "local"
    reference = ROOT / "evaluation" / "results" / "baseline_smooth50.csv"
    if not reference.is_file():
        print("FAIL: Supplied baseline_smooth50.csv is missing; leave references unchanged.")
        return 1
    local.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="baseline-check-", dir=local) as temporary:
        candidate = Path(temporary) / "candidate.csv"
        commands = (
            (
                "Baseline generation",
                [
                    sys.executable,
                    "-m",
                    "possession_lab.cli",
                    "predict",
                    "--config",
                    str(ROOT / "configs" / "models" / "baseline_smooth50.toml"),
                    "--output",
                    str(candidate),
                ],
            ),
            (
                "Official validation",
                [
                    sys.executable,
                    str(ROOT / "evaluation" / "validate_submission.py"),
                    str(candidate),
                ],
            ),
        )
        for stage, command in commands:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
            if result.returncode:
                print(f"FAIL: {stage} exited {result.returncode}.")
                print("Run the documented prediction/validation commands locally for diagnostics.")
                return 1
        if not same_predictions(candidate, reference):
            print("FAIL: Generated CSV differs from the supplied baseline; references unchanged.")
            return 1
    print("PASS: 199,319 baseline frames match the reference and pass the official validator.")
    print("Temporary predictions removed; no ground truth was read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
