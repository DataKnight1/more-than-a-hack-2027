"""Run a deterministic candidate prediction and write the official submission format."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path

from possession_lab.config import load_config
from possession_lab.model import predict

ROOT = Path(__file__).resolve().parents[2]


def write_csv(output: Path, labels: Iterable[str]) -> str:
    """Write a compact frame-aligned CSV and return its SHA-256 checksum.

    Args:
        output: File path for the prediction CSV.
        labels: One prediction per frame in tactical video order.

    Returns:
        SHA-256 hex digest of the complete generated CSV.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("wb") as handle:
        header = b"frame,label\n"
        handle.write(header)
        digest.update(header)
        for frame, label in enumerate(labels):
            line = f"{frame},{label}\n".encode()
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def main() -> int:
    """Execute the candidate model from a version-controlled TOML config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predict", choices=["predict"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    labels = predict(cfg)
    digest = write_csv(args.output, labels)
    print(f"Candidate generated: {len(labels)} frames, SHA256 {digest}")
    print("Run official validator on the exact file before comparing or submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
