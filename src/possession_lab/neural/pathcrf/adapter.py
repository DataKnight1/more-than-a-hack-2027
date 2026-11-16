"""Data adapter converting CanonicalMatch instances to PathCRF tensors.

Enforces:
1. Strict player-only features (no ball input).
2. Clean separation of inference mode (include_labels=False) where labels are zeroed.
3. Multi-rate sampling (5 Hz, 10 Hz, 25 Hz) with reversible frame mapping.
4. NaN-safe feature generation and explicit observation validity masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from possession_lab.neural.canonical import CanonicalMatch, X

from .features import build_node_features
from .nodes import map_canonical_players_to_26_nodes


@dataclass(frozen=True)
class PathCRFBatch:
    """Standardized batch representation for PathCRF training and inference."""

    x: torch.FloatTensor  # [B, T_s, 26, F]
    node_valid: torch.BoolTensor  # [B, T_s, 26]
    labels: torch.LongTensor  # [B, T_s], H/A/D/X (0..3)
    label_valid: torch.BoolTensor  # [B, T_s], True where frame has supervised H/A/D
    source_frame_ids: torch.LongTensor  # [B, T_s], 25Hz index in source match
    path_labels: torch.LongTensor | None  # [B, T_s], valid compressed edge state (0..575) or None
    path_valid: torch.BoolTensor | None  # [B, T_s], True where strong path is known
    metadata: dict[str, Any]


def canonical_to_pathcrf(
    match: CanonicalMatch,
    lo: int = 0,
    hi: int | None = None,
    target_fps: float = 25.0,
    *,
    include_labels: bool = True,
    path_annotations: NDArray[Any] | None = None,
    path_valid_mask: NDArray[Any] | None = None,
) -> PathCRFBatch:
    """Convert a slice of CanonicalMatch to a single-element PathCRFBatch.

    Args:
        match: CanonicalMatch instance.
        lo: Starting 25Hz frame index (inclusive).
        hi: Ending 25Hz frame index (exclusive). If None, match.n_frames.
        target_fps: Sampling rate in Hz (e.g., 5.0, 10.0, 25.0).
        include_labels: If False, ignores match.labels completely.
        path_annotations: Optional [N_frames] array of known edge IDs (0..575).
        path_valid_mask: Optional [N_frames] boolean array.

    Returns:
        PathCRFBatch with batch dimension B=1.
    """
    total_len = match.n_frames
    if hi is None or hi > total_len:
        hi = total_len
    if lo < 0 or lo >= hi:
        raise ValueError(f"Invalid frame slice [{lo}, {hi}) for match with {total_len} frames")

    # 1. Map player positions to 26 nodes (ignoring match.ball entirely)
    nodes_xy_full, node_valid_full = map_canonical_players_to_26_nodes(
        match.home,
        match.away,
        pitch=match.pitch,
        standard_coords=True,
    )

    # 2. Build 8-channel features at full 25Hz resolution
    features_full = build_node_features(
        nodes_xy_full,
        node_valid_full,
        period=match.period,
        fps=25.0,
        pitch_x=match.pitch[0],
    )

    # 3. Temporal subsampling if target_fps < 25.0
    source_step = max(1, int(round(25.0 / target_fps)))
    sample_indices = np.arange(lo, hi, source_step, dtype=np.int64)
    T_s = len(sample_indices)

    x_sampled = features_full[sample_indices]  # [T_s, 26, F]
    valid_sampled = node_valid_full[sample_indices]  # [T_s, 26]

    # Ensure finite floats
    assert np.isfinite(x_sampled).all(), "Non-finite values encountered in PathCRF features"

    # 4. Supervision labels
    if include_labels and match.labels is not None:
        raw_labels = match.labels[sample_indices].astype(np.int64)
        # Valid labels: graded frames (H, A, D) with positive period and non-X
        lbl_valid = (match.period[sample_indices] > 0) & (raw_labels != X)
        labels_arr = raw_labels
    else:
        labels_arr = np.zeros(T_s, dtype=np.int64)
        lbl_valid = np.zeros(T_s, dtype=bool)

    # 5. Path annotations (if provided and allowed)
    if include_labels and path_annotations is not None:
        p_labels = torch.tensor(path_annotations[sample_indices], dtype=torch.long).unsqueeze(0)
        p_valid = (
            torch.tensor(path_valid_mask[sample_indices], dtype=torch.bool).unsqueeze(0)
            if path_valid_mask is not None
            else torch.ones((1, T_s), dtype=torch.bool)
        )
    else:
        p_labels = None
        p_valid = None

    batch = PathCRFBatch(
        x=torch.tensor(x_sampled, dtype=torch.float32).unsqueeze(0),
        node_valid=torch.tensor(valid_sampled, dtype=torch.bool).unsqueeze(0),
        labels=torch.tensor(labels_arr, dtype=torch.long).unsqueeze(0),
        label_valid=torch.tensor(lbl_valid, dtype=torch.bool).unsqueeze(0),
        source_frame_ids=torch.tensor(sample_indices, dtype=torch.long).unsqueeze(0),
        path_labels=p_labels,
        path_valid=p_valid,
        metadata={
            "match_id": match.match_id,
            "provider": match.provider,
            "target_fps": target_fps,
            "source_fps": 25.0,
            "lo": lo,
            "hi": hi,
            "include_labels": include_labels,
        },
    )
    return batch
