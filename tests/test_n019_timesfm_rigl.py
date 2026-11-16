"""Offline deterministic unit tests. Live TimesFM weights require a separate preflight."""

import numpy as np
import torch
from torch import nn

from possession_lab.neural.timesfm_rigl import RigLSettings, RigLSparsifier, TimesFMBallFeatures


class FakeTimeFM:
    def __init__(self):
        self.calls = 0

    def forecast(self, *, horizon, inputs):
        self.calls += 1
        # Persistence-only fake for testing shape/causality; NOT production TimesFM.
        return (np.stack([np.full(horizon, x[-1], np.float32) for x in inputs]), None)


def test_timesfm_features_and_strict_call_count():
    fake = FakeTimeFM()
    obj = TimesFMBallFeatures(backend=fake, horizon=4)
    b = np.zeros((12, 9), np.float32)
    b[:, 0] = 1.0
    b[:, 1:3] = 0.25
    b[3:7, 0] = 0.0
    b[3:7, 1:3] = 0.0
    result = obj.from_ball_tokens(b)
    assert result.shape == (12, 4)
    assert fake.calls > 0 and obj.forecast_calls > 0
    assert np.all(result[3:7, 3] == 1)
    assert result[3, 2] > result[6, 2]
    assert np.allclose(result[3:7, :2], 0.25)
    assert np.allclose(result[7, :2], b[7, 1:3])


def test_timesfm_never_uses_future_as_history():
    fake = FakeTimeFM()
    obj = TimesFMBallFeatures(backend=fake)
    b = np.zeros((25, 9), np.float32)
    b[0, 0] = 1
    b[0, 1:3] = 0.2
    b[10, 0] = 1
    b[10, 1:3] = -0.9
    out = obj.from_ball_tokens(b)
    assert np.allclose(out[1:10, :2], 0.2)
    assert np.allclose(out[10, :2], -0.9)
    assert obj.forecast_calls > 0


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Module()
        self.spatial.layers = nn.ModuleList([nn.Module()])
        self.spatial.layers[0].linear1 = nn.Linear(8, 8)

    def forward(self, x):
        return self.spatial.layers[0].linear1(x)


def test_rigl_real_gradient_regrowth_and_mask_count():
    torch.manual_seed(12)
    model = SmallNet()
    rigl = RigLSparsifier(
        model, RigLSettings(sparsity=0.5, update_interval=1, stop_step=2, drop_fraction=0.2)
    )
    lin = model.spatial.layers[0].linear1
    mask0 = lin.rigl_mask.clone()
    nactive = int(mask0.sum())
    optim = torch.optim.AdamW(model.parameters(), lr=0.05)
    for step in range(1, 3):
        optim.zero_grad()
        loss = ((model(torch.randn(16, 8)) - torch.randn(16, 8)) ** 2).mean()
        loss.backward()
        rigl.before_optimizer_step()
        optim.step()
        rigl.after_optimizer_step(optim, step)
    assert rigl.updates == 2
    assert int(lin.rigl_mask.sum()) == nactive
    assert not torch.equal(lin.rigl_mask, mask0)
    assert torch.count_nonzero(lin.weight.detach()[~lin.rigl_mask]) == 0


def test_no_sparse_eligible_modules_fails():
    import pytest

    with pytest.raises(RuntimeError, match="RigL required"):
        RigLSparsifier(nn.Sequential(nn.Linear(2, 2)), RigLSettings())
