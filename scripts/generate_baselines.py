"""Run the archived organizer baseline from the reorganized repository layout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "official_documentation" / "starter" / "baseline.py"


def load_baseline() -> ModuleType:
    """Import the preserved upstream baseline and set its project root for this layout.

    Returns:
        The loaded baseline module with root-relative paths configured.

    Raises:
        ImportError: If the archived baseline cannot be imported.
    """
    spec = importlib.util.spec_from_file_location(
        "possession_lab._official_baseline", BASELINE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the baseline from {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    module.ROOT = str(ROOT)
    module.RESULTS_DIR = str(ROOT / "evaluation" / "results")
    return module


def main() -> int:
    """Execute the upstream generator using its original defaults and output format."""
    return int(load_baseline().main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
