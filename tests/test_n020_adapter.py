"""Tests for CanonicalMatch to PathCRF data adapter and ball-invariance contract (N020)."""

import numpy as np
import pytest
import torch

from possession_lab.neural.canonical import A, CanonicalMatch, D, H, X
from possession_lab.neural.pathcrf.adapter import canonical_to_pathcrf


@pytest.fixture
def dummy_match():
    T = 100
    home = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    away = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    ball = np.random.uniform(-40, 40, (T, 3)).astype(np.float32)
    labels = np.random.choice([H, A, D], size=T).astype(np.int8)
    period = np.ones(T, dtype=np.int8)
    period[45:55] = 0  # Gap frames
    labels[45:55] = X

    return CanonicalMatch(
        match_id="test_synthetic",
        provider="synthetic",
        ball=ball,
        home=home,
        away=away,
        labels=labels,
        period=period,
        pitch=(105.0, 68.0),
        meta={"licence": "test"},
    )


def test_adapter_shapes_and_finiteness(dummy_match):
    batch = canonical_to_pathcrf(dummy_match, target_fps=25.0, include_labels=True)

    assert batch.x.shape == (1, 100, 26, 8)
    assert batch.node_valid.shape == (1, 100, 26)
    assert batch.labels.shape == (1, 100)
    assert batch.label_valid.shape == (1, 100)
    assert batch.source_frame_ids.shape == (1, 100)

    assert torch.isfinite(batch.x).all()
    # Gap frames must be invalid for supervision
    assert not batch.label_valid[0, 45:55].any()
    assert batch.label_valid[0, :45].all()


def test_strict_ball_invariance(dummy_match):
    """Perturbing match.ball arbitrarily must yield BIT-IDENTICAL player features."""
    batch1 = canonical_to_pathcrf(dummy_match, target_fps=25.0, include_labels=False)

    # Scramble ball completely with NaNs, infs, and huge numbers
    scrambled_ball = np.full_like(dummy_match.ball, np.nan)
    scrambled_ball[::2] = 9999.0
    match_scrambled = CanonicalMatch(
        match_id=dummy_match.match_id,
        provider=dummy_match.provider,
        ball=scrambled_ball,
        home=dummy_match.home.copy(),
        away=dummy_match.away.copy(),
        labels=dummy_match.labels.copy(),
        period=dummy_match.period.copy(),
        pitch=dummy_match.pitch,
    )

    batch2 = canonical_to_pathcrf(match_scrambled, target_fps=25.0, include_labels=False)

    # Must be bit-identical
    assert torch.equal(batch1.x, batch2.x), "Ball trajectory contaminated PathCRF input features!"


def test_include_labels_false_privacy(dummy_match):
    """With include_labels=False, labels must be zeroed and label_valid must be False."""
    batch = canonical_to_pathcrf(dummy_match, target_fps=25.0, include_labels=False)

    assert (batch.labels == 0).all()
    assert not batch.label_valid.any()


def test_multi_rate_sampling(dummy_match):
    # Test 5Hz sampling (step of 5 from 25Hz)
    batch_5hz = canonical_to_pathcrf(dummy_match, target_fps=5.0, include_labels=True)
    assert batch_5hz.x.shape[1] == 20
    assert torch.equal(batch_5hz.source_frame_ids[0], torch.arange(0, 100, 5))

    # Test 10Hz sampling (step of ~2 or 3)
    batch_10hz = canonical_to_pathcrf(dummy_match, target_fps=10.0, include_labels=True)
    assert batch_10hz.x.shape[1] == 50
