"""Snapshot and verify critical official files without reading competition match assets.

Initial `--record` MUST run only after the user reviews the official clone.
Snapshots live under ignored local artifacts and deliberately do not gate updates
from the organisers without a reviewed re-baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "local" / "upstream_manifest.json"
OFFICIAL_FILES = (
    "games.py",
    "official_documentation/RULES.md",
    "official_documentation/DATA.md",
    "official_documentation/EVALUATION.md",
    "official_documentation/SUBMISSION.md",
    "official_documentation/starter/baseline.py",
    "evaluation/possession_metrics.py",
    "evaluation/run_evaluation.py",
    "evaluation/validate_submission.py",
)


def snapshot() -> dict[str, str]:
    """Hash a fixed allowlist of public official files, not the licensed dataset."""
    files = {}
    for relative in OFFICIAL_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Official reference file missing: {relative}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def main() -> int:
    """Record a reviewed baseline or compare current official files against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", action="store_true", help="Approve and record upstream baseline")
    group.add_argument(
        "--check", action="store_true", help="Check upstream references not modified"
    )
    args = parser.parse_args()
    current = snapshot()
    if args.record:
        if MANIFEST.exists():
            print("Manifest exists; refuse overwrite. Review and archive it first.")
            return 2
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Upstream snapshot recorded locally: {MANIFEST}")
        return 0
    if not MANIFEST.is_file():
        print("No baseline recorded. Review files and use --record first.")
        return 2
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = [name for name in OFFICIAL_FILES if current[name] != original.get(name)]
    if mismatches:
        print("FAIL: Official references differ from approved snapshot:")
        for relative in mismatches:
            print(" -", relative)
        return 1
    print("PASS: Official scoring, baseline and rules match locally approved snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
