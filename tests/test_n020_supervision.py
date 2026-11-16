"""Tests for multi-task loss computation and gradient propagation (N020)."""

import pytest
import torch

from possession_lab.neural.canonical import A, D, H, X
from possession_lab.neural.pathcrf.backbone import PathCRFNet
from possession_lab.neural.pathcrf.supervision import LossWeights, compute_n020_loss


def test_compute_n020_loss_and_gradients():
    B, T = 2, 8
    model = PathCRFNet(
        feat_dim=8, team_size=11, macro_dim=16, micro_dim=32, seq_dim=64, crf_edge_dim=8
    )

    x = torch.randn(B, T, 26, 8, requires_grad=True)
    node_valid = torch.ones(B, T, 26, dtype=torch.bool)

    outputs = model(x, node_valid)

    labels = torch.tensor([[H, H, A, A, D, D, X, H], [D, D, H, A, A, D, X, H]], dtype=torch.long)
    label_valid = labels != X

    path_labels = torch.tensor(
        [[0, 0, 12, 12, -1, -1, -1, 0], [-1, -1, 0, 12, 12, -1, -1, 0]], dtype=torch.long
    )
    path_valid = path_labels >= 0

    loss, loss_dict = compute_n020_loss(
        outputs,
        labels=labels,
        label_valid=label_valid,
        path_labels=path_labels,
        path_valid=path_valid,
        weights=LossWeights(w_dead=1.0, w_team=1.0, w_edge=1.0, w_crf=0.0),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "loss_dead" in loss_dict
    assert "loss_team" in loss_dict
    assert "loss_edge" in loss_dict

    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Verify backbone parameters received finite gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN in {name} grad"


def test_crf_failure_is_not_silently_discarded():
    """A broken CRF must not silently train the remaining heads without its loss."""
    model = PathCRFNet(
        feat_dim=8, team_size=11, macro_dim=16, micro_dim=32, seq_dim=64, crf_edge_dim=8
    )
    outputs = model(torch.zeros(1, 2, 26, 8), torch.ones(1, 2, 26, dtype=torch.bool))

    class BrokenCRF:
        def neg_log_likelihood(self, *_args, **_kwargs):
            raise RuntimeError("broken CRF")

    with pytest.raises(RuntimeError, match="broken CRF"):
        compute_n020_loss(
            outputs,
            labels=torch.zeros(1, 2, dtype=torch.long),
            label_valid=torch.ones(1, 2, dtype=torch.bool),
            path_labels=torch.zeros(1, 2, dtype=torch.long),
            path_valid=torch.ones(1, 2, dtype=torch.bool),
            crf=BrokenCRF(),
        )
