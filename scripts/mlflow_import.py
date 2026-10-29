"""Opt-in: copy reviewed local experiment metadata into a local-only MLflow database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Import a local metadata JSON file without logging licensed prediction or dataset files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--confirm-local", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local:
        parser.error("Pass --confirm-local after reviewing manifest and environment")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        print("MLflow not installed; use: uv sync --group mlops")
        return 2
    record = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed = {
        "run_id",
        "experiment_name",
        "timestamp_utc",
        "config_sha256",
        "git_sha",
        "git_dirty",
        "validator_pass",
        "status",
        "duration_seconds",
    }
    mlflow.set_experiment("mtah2027-local")
    with mlflow.start_run(run_name=str(record.get("experiment_name", "unnamed"))):
        for key in allowed:
            if key not in record:
                continue
            value = record[key]
            if key == "duration_seconds" and isinstance(value, (float, int)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, str(value))
    print("Local MLflow metadata recorded. No model, dataset or CSV artifact uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
