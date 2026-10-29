"""Privacy scanner pattern tests, entirely synthetic."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def load_scanner():  # type: ignore[no-untyped-def]
    """Load the checked-out scanner without installing it as a package."""
    script = Path(__file__).resolve().parents[1] / "scripts/privacy_scan.py"
    spec = importlib.util.spec_from_file_location("privacy_scan", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restricted_paths_are_detected() -> None:
    """Flag licensed data and candidate CSVs without flagging original baselines."""
    module = load_scanner()
    found = module.audit_tracked(
        [
            "data/tactical/BAR-ATH_tactical_atd.txt",
            "evaluation/results/candidate_001.csv",
            "evaluation/results/baseline_smooth50.csv",
            "src/possession_lab/model.py",
        ]
    )
    assert found == [
        "data/tactical/BAR-ATH_tactical_atd.txt",
        "evaluation/results/candidate_001.csv",
    ]


@pytest.mark.parametrize(
    "restricted",
    [
        ".github/workflows/quality.yml",
        "src/.cache/results.json",
        ".env",
        ".env.production",
        "docs/plan.txt",
        "research/registry.json",
        "experiments/config.toml",
        "setup/commands.ps1",
        "assets/brand/ATTRIBUTION.md",
        "evaluation/results/README.md",
        "data/random.txt",
        "data/tactical/asset.xml",
        "AGENTS.md",
        "CLAUDE.md",
    ],
)
def test_local_only_material_is_rejected(restricted: str) -> None:
    """Reject excluded directories, hidden material, nested Markdown and data."""
    assert load_scanner().audit_tracked([restricted]) == [restricted]


def test_public_runtime_and_empty_data_marker_paths_are_allowed() -> None:
    """Keep root documentation, public evaluation references and code publishable."""
    assert (
        load_scanner().audit_tracked(
            [
                ".gitignore",
                "README.md",
                "RULES.md",
                "LICENSE",
                "data/.gitkeep",
                "evaluation/gt/BAR-ATH_possession_gt_public.csv",
                "evaluation/gt/BAR-ATH_possession_gt_public_meta.json",
                "evaluation/results/baseline_raw.csv",
                "evaluation/results/baseline_smooth25.csv",
                "evaluation/results/baseline_smooth50.csv",
                "evaluation/results/baseline_smooth125.csv",
                "src/possession_lab/model.py",
            ]
        )
        == []
    )


@pytest.mark.parametrize(("indexed_size", "expected"), [(b"0\n", 0), (b"12\n", 1)])
def test_data_marker_checks_indexed_blob_size(
    monkeypatch, indexed_size: bytes, expected: int
) -> None:  # type: ignore[no-untyped-def]
    """Reject a nonempty staged marker even if the working file was emptied later."""
    module = load_scanner()
    commands = []

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        commands.append(command)
        output = b"data/.gitkeep\0" if command[1] == "ls-files" else indexed_size
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == expected
    assert commands[-1] == ["git", "cat-file", "-s", ":data/.gitkeep"]
