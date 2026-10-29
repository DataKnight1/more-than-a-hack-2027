"""Verify upstream guard has the correct no-data allowlist (synthetic/import only)."""

from __future__ import annotations

import ast
from pathlib import Path


def test_manifest_tool_never_walks_match_data() -> None:
    """The official snapshot file is not permitted to hash restricted match assets."""
    script = Path(__file__).resolve().parents[1] / "scripts/check_upstream.py"
    source = script.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"starter/baseline.py"' in source
    assert '"evaluation/possession_metrics.py"' in source
    assert '"data/tactical/' not in source
