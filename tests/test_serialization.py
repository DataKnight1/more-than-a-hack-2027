"""Full-length synthetic prediction contract tests, no competition data required."""

from __future__ import annotations

import hashlib

import numpy as np

from possession_lab.cli import write_csv
from possession_lab.model import FRAME_COUNT, set_ungraded_to_x


def test_csv_output_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """CSV bytes must match across runs for a declared sequence."""
    labels = ["H", "H", "D", "A", "X"]
    target = tmp_path / "prediction.csv"
    digest1 = write_csv(target, labels)
    bytes1 = target.read_bytes()
    digest2 = write_csv(target, labels)
    assert digest1 == digest2 == hashlib.sha256(bytes1).hexdigest()
    assert bytes1.startswith(b"frame,label\n0,H\n1,H\n2,D\n3,A\n4,X\n")


def test_ungraded_windows_do_not_erase_graded_frames() -> None:
    """Known published ungraded blocks may be X while all graded blocks stay H."""
    base = np.full(FRAME_COUNT, "H", dtype="<U1")
    updated = set_ungraded_to_x(base)
    assert updated[0] == "X"
    assert updated[11_184] == "X"
    assert updated[11_185] == "H"
    assert updated[90_651] == "H"
    assert updated[90_652] == "X"
    assert updated[110_807] == "X"
    assert updated[110_808] == "H"
    assert updated[190_197] == "H"
    assert updated[190_198] == "X"
    assert base[0] == "H"
