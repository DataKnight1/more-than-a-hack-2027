"""Tests ensuring checkpoint integrity, roundtrip fidelity, and N018 backward compatibility."""

from pathlib import Path

import torch

from possession_lab.neural.pathcrf.backbone import PathCRFNet
from possession_lab.neural.pathcrf.checkpoints import (
    load_n020_checkpoint,
    save_n020_checkpoint,
)
from possession_lab.neural.pathcrf.crf import DynamicSparseCRF


def test_n020_checkpoint_save_and_load(tmp_path):
    model = PathCRFNet(
        feat_dim=8, team_size=11, macro_dim=16, micro_dim=32, seq_dim=64, crf_edge_dim=8
    )
    crf = DynamicSparseCRF(model.comp_src, model.comp_dst, edge_embed_dim=8, team_size=11)

    ckpt_path = tmp_path / "n020_test.pt"
    cfg = {"experiment": "n020_test"}

    digest = save_n020_checkpoint(ckpt_path, model, crf, cfg, step=100, val_spell_f1=0.58)
    assert len(digest) == 64
    assert ckpt_path.is_file()

    # Load back
    loaded_model, loaded_crf, state = load_n020_checkpoint(ckpt_path)
    assert state["step"] == 100
    assert state["val_spell_f1"] == 0.58

    # Verify identical output on random tensor
    model.eval()
    loaded_model.eval()
    x = torch.randn(1, 10, 26, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = loaded_model(x)

    torch.testing.assert_close(out1.dead_logit, out2.dead_logit)
    torch.testing.assert_close(out1.team_logits, out2.team_logits)
    torch.testing.assert_close(out1.compressed_edge_logits, out2.compressed_edge_logits)


def test_n018_checkpoint_compatibility():
    """Verify that existing N018 champion checkpoint remains intact and loadable."""
    n018_path = Path("artifacts/neural/runs/n018_large_long/best.pt")
    if n018_path.is_file():
        ckpt = torch.load(n018_path, map_location="cpu")
        assert "model" in ckpt
        assert "config" in ckpt
