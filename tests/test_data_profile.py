"""The data profile runs on the fake bundle and never echoes raw tracking."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profile_prints_aggregates_only(bundle, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """All sections print; no raw line or player/ball group appears in the output."""
    spec = importlib.util.spec_from_file_location("data_profile", ROOT / "scripts/data_profile.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_DIR", bundle.root / "c")

    assert module.main() == 0
    out = capsys.readouterr().out
    for marker in ("fields per group", "ball visible", "gap length", "player speed", "tag Team"):
        assert marker in out
    raw = (bundle.root / "tactical/BAR-ATH_tactical_atd.txt").read_text().splitlines()
    groups = {g for line in raw for g in line.split(":")[1].split(";")} - {"NaN,NaN,NaN"}
    assert not any(line in out for line in raw) and not any(g in out for g in groups)
