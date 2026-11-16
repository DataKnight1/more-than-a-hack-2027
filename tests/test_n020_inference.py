"""Tests for full-period inference, period boundary resets, and sequence decoding (N020)."""

import numpy as np

from possession_lab.neural.canonical import CanonicalMatch
from possession_lab.neural.decode import DecodeConfig
from possession_lab.neural.pathcrf.backbone import PathCRFNet
from possession_lab.neural.pathcrf.inference import decode_n020_match, predict_n020_probabilities


def test_full_period_inference_and_gaps():
    T = 200
    home = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    away = np.random.uniform(-40, 40, (T, 11, 2)).astype(np.float32)
    ball = np.zeros((T, 3), dtype=np.float32)
    labels = np.zeros(T, dtype=np.int8)

    # Two periods with a 25-frame gap in between
    period = np.ones(T, dtype=np.int8)
    period[90:115] = 0  # Gap between half 1 and half 2
    period[115:] = 2  # Second half

    match = CanonicalMatch(
        match_id="test_inference",
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

    probs = predict_n020_probabilities(
        model, match, sample_fps=25.0, window_seconds=4.0, overlap_seconds=1.0
    )
    assert probs.shape == (200, 3)

    decoder = DecodeConfig(team_switch=2.0, dead_switch=4.0, transition_gain=0.0)
    pred_chars = decode_n020_match(probs, period, decoder)

    assert len(pred_chars) == 200
    # Gap frames must strictly be 'X'
    assert (pred_chars[90:115] == "X").all()
    # Graded frames must only be in {'H', 'A', 'D'}
    assert set(pred_chars[:90]).issubset({"H", "A", "D"})
    assert set(pred_chars[115:]).issubset({"H", "A", "D"})
