"""Run a candidate and record minimal LOCAL metadata, never ground truth or raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = ROOT / "artifacts" / "local" / "runs"


def git_info() -> tuple[str, bool]:
    """Return Git revision and working-tree dirty flag; do not alter Git state."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return (commit.stdout.strip() or "unknown", bool(status.stdout.strip()))


def sha256_file(path: Path) -> str:
    """Calculate a prediction file checksum without logging its frame-level content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Predict, officially validate, and persist minimal experiment metadata locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Short experiment identifier")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,80}", args.name):
        parser.error("--name must be 3-80 letters, digits, underscore or dash")
    config = args.config.resolve()
    if not config.is_file():
        parser.error("Config file not found")

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    prediction = ROOT / "evaluation" / "results" / f"candidate_{args.name}_{run_id}.csv"
    command = [
        sys.executable,
        "-m",
        "possession_lab.cli",
        "predict",
        "--config",
        str(config),
        "--output",
        str(prediction),
    ]
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outcome: dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": args.name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else "external",
        "config_sha256": sha256_file(config),
        "status": "started",
        "ground_truth_used": False,
        "evidence_tier": "candidate-generated-and-format-validated-only",
    }
    outcome["git_sha"], outcome["git_dirty"] = git_info()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(
            [sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(prediction)],
            cwd=ROOT,
            check=True,
        )
        outcome["prediction_sha256"] = sha256_file(prediction)
        outcome["validator_pass"] = True
        outcome["status"] = "validated"
    except subprocess.CalledProcessError as error:
        outcome["status"] = "failed"
        outcome["validator_pass"] = False
        outcome["error_returncode"] = error.returncode
    finally:
        outcome["duration_seconds"] = round(time.monotonic() - started, 3)
        manifest = LOCAL_DIR / f"{run_id}.json"
        manifest.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Local experiment metadata: {manifest}")
        print(f"Run status: {outcome['status']}; no ground truth was loaded")
    return 0 if outcome["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
