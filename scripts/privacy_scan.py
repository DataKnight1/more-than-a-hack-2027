"""Inspect the Git index for restricted data and local-only project material."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories that must never be published.
RESTRICTED_PREFIXES = (
    "docs/",
    "research/",
    "setup/",
    "artifacts/",
    "features/",
    "models/",
    "mlruns/",
    ".mlflow/",
    "evaluation/reports/",
    "evaluation/gt/",
)

# Files containing generated or potentially sensitive information.
RESTRICTED_EXACT = {
    "agents.md",
    "claude.md",
    "agents_neural_addendum.md",
    "mlruns.db",
    "evaluation/scoreboard.csv",
}

# Binary and data formats requiring additional protection.
RESTRICTED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".zip",
    ".7z",
    ".parquet",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".pkl",
    ".safetensors",
    ".bak",
    ".pdf",
    ".gzip",
    ".pl3container",
}

# Only explicitly approved files are permitted in evaluation/results/.
ALLOWED_RESULTS = {
    "evaluation/results/baseline_raw.csv",
    "evaluation/results/baseline_smooth25.csv",
    "evaluation/results/baseline_smooth50.csv",
    "evaluation/results/baseline_smooth125.csv",
}

DATA_MARKER = "data/.gitkeep"
# Reviewed externally trained champion; all other model artifacts remain restricted.
ALLOWED_MODEL = "models/n018/best.pt"
ALLOWED_MODEL_SHA256 = "1b42d0f7eab232f6ba4ea2131d3f0a892f41f750504789dfcd5ce5edb5158a81"


def audit_tracked(paths: list[str]) -> list[str]:
    """Identify potentially restricted Git-tracked files.

    Args:
        paths: Paths reported by Git, relative to the repository root.

    Returns:
        Sorted list of suspicious file paths requiring review.
    """
    suspicious: set[str] = set()

    for name in paths:
        normalized = name.replace("\\", "/")
        path = Path(normalized)
        extension = path.suffix.lower()
        lowered = normalized.lower()

        if normalized in {ALLOWED_MODEL, "models/n018/README.md"}:
            continue

        is_restricted_directory = lowered.startswith(RESTRICTED_PREFIXES)

        is_restricted_file = lowered in RESTRICTED_EXACT

        is_restricted_extension = extension in RESTRICTED_EXTENSIONS

        is_unapproved_result = (
            lowered.startswith("evaluation/results/") and normalized not in ALLOWED_RESULTS
        )

        is_hidden_directory = any(part.startswith(".") for part in path.parts[:-1]) and not (
            lowered.startswith(".github/workflows/") and extension in {".yml", ".yaml"}
        )
        is_unapproved_hidden_file = path.name.startswith(".") and normalized not in {
            ".gitignore",
            ".pre-commit-config.yaml",
            DATA_MARKER,
        }
        allowed_markdown = lowered.startswith(
            ("official_documentation/", "experiments/", "assets/brand/")
        )
        is_nested_markdown = extension == ".md" and len(path.parts) > 1 and not allowed_markdown
        is_unapproved_experiment = lowered.startswith("experiments/") and extension not in {
            ".md",
            ".html",
        }
        is_data_content = lowered.startswith("data/") and normalized != DATA_MARKER

        if any(
            (
                is_restricted_directory,
                is_restricted_file,
                is_restricted_extension,
                is_unapproved_result,
                is_hidden_directory,
                is_unapproved_hidden_file,
                is_nested_markdown,
                is_unapproved_experiment,
                is_data_content,
            )
        ):
            suspicious.add(normalized)

    return sorted(suspicious)


def main() -> int:
    """Scan Git-indexed files without reading competition data.

    Returns:
        0 when no suspicious tracked files are detected.
        1 when suspicious files require review.
        2 when Git inspection fails.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        print(f"ERROR: Unable to execute Git: {error}")
        return 2

    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()

        print(f"ERROR: Git inspection failed: {message}")
        return 2

    tracked = [
        item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\x00") if item
    ]

    suspicious = audit_tracked(tracked)

    if ALLOWED_MODEL in tracked:
        try:
            model = subprocess.run(
                ["git", "cat-file", "blob", f":{ALLOWED_MODEL}"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
        except OSError:
            print("ERROR: Unable to inspect the indexed model checkpoint.")
            return 2
        if model.returncode != 0:
            print("ERROR: Unable to inspect the indexed model checkpoint.")
            return 2
        if hashlib.sha256(model.stdout).hexdigest() != ALLOWED_MODEL_SHA256:
            suspicious.append(ALLOWED_MODEL)

    if DATA_MARKER in tracked:
        try:
            marker = subprocess.run(
                ["git", "cat-file", "-s", f":{DATA_MARKER}"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
            if marker.returncode != 0:
                print("ERROR: Unable to inspect the indexed data placeholder.")
                return 2
            if int(marker.stdout.strip()) != 0:
                suspicious.append(DATA_MARKER)
        except (OSError, ValueError):
            print("ERROR: Unable to verify that the indexed data placeholder is empty.")
            return 2

    if suspicious:
        print("FAIL: Suspicious tracked paths detected:")

        for name in suspicious:
            print(f"  - {name}")

        print("\nReview these files before committing or pushing.")
        return 1

    print(f"PASS: Scanned {len(tracked)} Git-tracked paths.")
    print("No restricted paths detected by the configured rules.")
    print("Manual review is still required before publication.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
