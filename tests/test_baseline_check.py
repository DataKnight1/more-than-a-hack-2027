"""Synthetic comparisons for the local baseline reproduction checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("candidate_text", "expected"),
    [
        ("frame,label\r\n0,H\r\n1,A\r\n", True),
        ("frame,label\n0,H\n1,D\n", False),
        ("frame,label\n0,H\n", False),
        ("frame,label\n0,H\n1,A\n2,X\n", False),
        ("frame,label\n1,H\n2,A\n", False),
    ],
)
def test_reference_comparison(tmp_path: Path, candidate_text: str, expected: bool) -> None:
    """Allow CRLF but reject changed labels, missing/extra rows and shifted frames."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_baseline.py"
    spec = importlib.util.spec_from_file_location("check_baseline", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    candidate.write_bytes(candidate_text.encode("utf-8"))
    reference.write_bytes(b"frame,label\n0,H\n1,A\n")
    assert module.same_predictions(candidate, reference) is expected
