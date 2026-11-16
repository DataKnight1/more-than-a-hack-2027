"""PathCRF-based possession recognition module for Possession Lab (N020).

Adapts the original PathCRF algorithm (Kim et al., KDD 2026, MPL-2.0)
for the More Than A Hack 2027 challenge (H/A/D/X prediction).
"""

from .adapter import PathCRFBatch, canonical_to_pathcrf
from .backbone import N020Outputs, PathCRFNet
from .checkpoints import load_n020_checkpoint, save_n020_checkpoint
from .crf import DynamicSparseCRF
from .edge_states import build_edge_compression_maps, get_edge_semantic_masks
from .features import build_node_features, compute_node_kinematics
from .fusion import aggregate_path_evidence, fuse_n020_probabilities
from .inference import decode_n020_match, predict_n020_probabilities
from .labels import PlayerPathAnnotations, generate_training_path_annotations
from .nodes import (
    NUM_OUTSIDE,
    OUTSIDE_BOTTOM,
    OUTSIDE_LEFT,
    OUTSIDE_RIGHT,
    OUTSIDE_TOP,
    PLAYERS_PER_TEAM,
    TOTAL_NODES,
    TOTAL_PLAYERS,
    map_canonical_players_to_26_nodes,
)
from .provenance import (
    APPROVED_TRAIN_MATCHES,
    DENIED_MATCHES,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
)
from .supervision import LossWeights, compute_n020_loss
from .transitions import build_allowed_mask, build_incoming_structures, forbid_rate

__all__ = [
    "UPSTREAM_REPO",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE",
    "APPROVED_TRAIN_MATCHES",
    "DENIED_MATCHES",
    "TOTAL_NODES",
    "PLAYERS_PER_TEAM",
    "TOTAL_PLAYERS",
    "NUM_OUTSIDE",
    "OUTSIDE_LEFT",
    "OUTSIDE_RIGHT",
    "OUTSIDE_BOTTOM",
    "OUTSIDE_TOP",
    "map_canonical_players_to_26_nodes",
    "build_node_features",
    "compute_node_kinematics",
    "build_edge_compression_maps",
    "get_edge_semantic_masks",
    "build_allowed_mask",
    "build_incoming_structures",
    "forbid_rate",
    "DynamicSparseCRF",
    "N020Outputs",
    "PathCRFNet",
    "PathCRFBatch",
    "canonical_to_pathcrf",
    "PlayerPathAnnotations",
    "generate_training_path_annotations",
    "LossWeights",
    "compute_n020_loss",
    "aggregate_path_evidence",
    "fuse_n020_probabilities",
    "predict_n020_probabilities",
    "decode_n020_match",
    "save_n020_checkpoint",
    "load_n020_checkpoint",
]
