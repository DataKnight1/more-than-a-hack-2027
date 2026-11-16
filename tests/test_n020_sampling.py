"""Tests for multi-rate sampling and temporal upsampling alignment (N020)."""

import numpy as np
import pytest

from possession_lab.neural.canonical import CanonicalMatch
from possession_lab.neural.pathcrf.backbone import PathCRFNet
from possession_lab.neural.pathcrf.inference import predict_n020_probabilities


def test_sampling_and_upsampling_alignment():
    # Construct synthetic match of exact length 250 frames (10 seconds)
    T = 250
    home = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    away = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    ball = np.zeros((T, 3), dtype=np.float32)
    labels = np.zeros(T, dtype=np.int8)
    period = np.ones(T, dtype=np.int8)

    match = CanonicalMatch(
        match_id="test_align",
        provider="synthetic",
        ball=ball,
        home=home,
        away=away,
        labels=labels,
        period=period,
        pitch=(105.0, 68.0),
    )

    model = PathCRFNet(
        feat_dim=8, team_size=11, macro_dim=16, micro_dim=32, seq_dim=64, crf_edge_dim=8
    )

    with pytest.raises(ValueError, match="canonical 25 Hz"):
        predict_n020_probabilities(model, match, output_fps=10.0)

    # 1. Test 25Hz inference (1:1)
    probs_25 = predict_n020_probabilities(
        model, match, sample_fps=25.0, window_seconds=4.0, overlap_seconds=1.0
    )
    assert probs_25.shape == (250, 3)
    np.testing.assert_allclose(probs_25.sum(axis=-1), np.ones(250), atol=1e-5)

    # 2. Test 5Hz subsampled inference with upsampling back to 25Hz
    probs_5 = predict_n020_probabilities(
        model, match, sample_fps=5.0, window_seconds=4.0, overlap_seconds=1.0
    )
    assert probs_5.shape == (250, 3), f"Expected exact 250 rows, got {probs_5.shape}"
    np.testing.assert_allclose(probs_5.sum(axis=-1), np.ones(250), atol=1e-5)
