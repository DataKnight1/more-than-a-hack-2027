"""Checkpoint serialization, integrity hashing, and versioned restoration for N020 PathCRF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .backbone import PathCRFNet
from .crf import DynamicSparseCRF
from .provenance import UPSTREAM_COMMIT, UPSTREAM_REPO

MODEL_SCHEMA_VERSION: str = "n020_pathcrf_v1"
CPU_DEVICE = torch.device("cpu")


def compute_file_sha256(path: Path | str) -> str:
    """Compute streaming SHA-256 hex digest of a file."""
    p = Path(path)
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_n020_checkpoint(
    path: Path | str,
    model: PathCRFNet,
    crf: DynamicSparseCRF | None,
    cfg: dict[str, Any],
    step: int,
    val_spell_f1: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save versioned checkpoint with complete provenance and return its SHA-256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "step": step,
        "val_spell_f1": float(val_spell_f1),
        "config": cfg,
        "model_state": model.state_dict(),
        "model_kwargs": {
            "feat_dim": model.feat_dim,
            "team_size": model.team_size,
            "macro_dim": getattr(model, "macro_dim", 32),
            "micro_dim": getattr(model, "micro_dim", 64),
            "seq_dim": model.seq_dim,
            "crf_edge_dim": model.crf_edge_dim,
            "heads": getattr(model, "heads", 4),
            "dropout": getattr(model, "dropout", 0.1),
        },
        "crf_state": crf.state_dict() if crf is not None else None,
        "metadata": metadata or {},
    }

    torch.save(state, p)
    digest = compute_file_sha256(p)

    # Save accompanying JSON metadata
    meta_path = p.with_suffix(".json")
    json_meta = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "step": step,
        "val_spell_f1": float(val_spell_f1),
        "sha256": digest,
        "upstream_commit": UPSTREAM_COMMIT,
        "metadata": metadata or {},
    }
    meta_path.write_text(json.dumps(json_meta, indent=2), encoding="utf-8")
    return digest


def load_n020_checkpoint(
    path: Path | str,
    device: torch.device = CPU_DEVICE,
) -> tuple[PathCRFNet, DynamicSparseCRF | None, dict[str, Any]]:
    """Load model, optional CRF, and configuration from checkpoint.

    Returns:
        (model, crf, checkpoint_dict)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    state = torch.load(p, map_location=device)
    kwargs = state.get("model_kwargs", {})

    model = PathCRFNet(**kwargs)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    crf = None
    if state.get("crf_state") is not None:
        crf = DynamicSparseCRF(
            comp_src=model.comp_src,
            comp_dst=model.comp_dst,
            edge_embed_dim=model.crf_edge_dim,
            team_size=model.team_size,
        )
        crf.load_state_dict(state["crf_state"])
        crf.to(device)
        crf.eval()

    return model, crf, state
