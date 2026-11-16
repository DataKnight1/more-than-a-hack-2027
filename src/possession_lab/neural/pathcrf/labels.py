"""Player path annotation generation and provenance firewall for external training matches.

STRICT GOVERNANCE CONTRACT:
1. ONLY approved training matches (J03WMX, J03WOH, J03WOY, J03WPY) are permitted.
2. Validation (J03WQQ), audit (J03WN1, J03WR9), Metrica 2/3, and challenge matches
   trigger immediate ValueError if accessed.
3. If no trustworthy player path labels exist for a frame/match, edge_id is set to -1
   and confident is False. No artificial player labels are invented from team labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .edge_states import build_edge_compression_maps
from .provenance import APPROVED_TRAIN_MATCHES, DENIED_MATCHES


@dataclass
class PlayerPathAnnotations:
    """Strong player-to-player possession path labels for an eligible external match."""

    edge_id: NDArray[np.int64]  # [T], compressed state in [0..575] or -1
    confident: NDArray[np.bool_]  # [T], True where edge label is trustworthy
    source_match: str  # e.g., 'J03WMX'
    annotation_method: str  # 'synced_events' | 'none_available'
    provenance: dict[str, Any]


def generate_training_path_annotations(
    match_id: str,
    n_frames: int,
    synced_parquet_dir: Path | str | None = None,
) -> PlayerPathAnnotations:
    """Extract player path labels for an eligible external training match.

    Raises:
        ValueError: If match_id belongs to the denied partition (validation, audit, challenge).
    """
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-nested-blocks,too-many-branches,too-many-statements
    clean_id = match_id.replace("idsse_", "")

    # 1. Strict provenance firewall check
    if clean_id in DENIED_MATCHES or match_id in DENIED_MATCHES:
        raise ValueError(
            f"SECURITY / LEAKAGE VIOLATION: Access denied to match {match_id}. "
            f"Validation, audit, and target matches must never be read for path label generation."
        )

    # 2. Check if approved training match
    if clean_id not in APPROVED_TRAIN_MATCHES and match_id not in APPROVED_TRAIN_MATCHES:
        # Not an approved training match: return empty labels
        return PlayerPathAnnotations(
            edge_id=np.full(n_frames, -1, dtype=np.int64),
            confident=np.zeros(n_frames, dtype=bool),
            source_match=match_id,
            annotation_method="none_available",
            provenance={"reason": "Match not in APPROVED_TRAIN_MATCHES"},
        )

    # 3. If parquet directory provided, attempt to load synced training events
    if synced_parquet_dir is not None:
        p_dir = Path(synced_parquet_dir)
        parquet_path = p_dir / f"{clean_id}.parquet"
        if parquet_path.is_file():
            try:
                # Optional or cyclic dependency: load only on this path.
                # pylint: disable-next=import-outside-toplevel
                import pandas as pd

                df = pd.read_parquet(parquet_path)
                _, orig2comp, _, _ = build_edge_compression_maps(22, 4)
                orig2comp_np = orig2comp.numpy()

                edge_ids = np.full(n_frames, -1, dtype=np.int64)
                confident = np.zeros(n_frames, dtype=bool)

                # Canonical period offsets:
                # DFL period 1 raw frame starts at 10000 (canonical 0)
                # DFL period 2 raw frame starts at 100000 (canonical depends on match)
                p1_dfl_start = 10000
                p2_dfl_start = 100000

                # Canonical standard gap is 25 frames between periods
                # Let's inspect canonical match if available, else approximate with standard span
                p2_canon_start = 70733  # fallback standard
                canon_path = Path("data/external/canonical") / f"idsse_{clean_id}.npz"
                if canon_path.is_file():
                    try:
                        # Optional or cyclic dependency: load only on this path.
                        # pylint: disable-next=import-outside-toplevel
                        from possession_lab.neural.canonical import CanonicalMatch

                        # Optional or cyclic dependency: load only on this path.
                        # pylint: disable-next=import-outside-toplevel
                        from possession_lab.neural.matchops import period_spans

                        cm = CanonicalMatch.load(canon_path)
                        spans = period_spans(cm.period)
                        if len(spans) >= 2:
                            p2_canon_start = spans[1][0]
                    except (OSError, ValueError, KeyError, TypeError):
                        pass

                def _to_canonical_frame(p_id: int, raw_f: float) -> int:
                    if pd.isna(raw_f):
                        return -1
                    rf = int(raw_f)
                    if p_id == 1:
                        cf = rf - p1_dfl_start
                    elif p_id == 2:
                        cf = p2_canon_start + (rf - p2_dfl_start)
                    else:
                        return -1
                    return cf if 0 <= cf < n_frames else -1

                def _to_node_idx(pid_str: str) -> int:
                    if not isinstance(pid_str, str):
                        return -1
                    if "home_" in pid_str:
                        try:
                            num = int(pid_str.split("_")[-1])
                            return num % 11
                        except ValueError:
                            return -1
                    elif "away_" in pid_str:
                        try:
                            num = int(pid_str.split("_")[-1])
                            return 11 + (num % 11)
                        except ValueError:
                            return -1
                    return -1

                # Group by episode to track possession chains
                for _, ep_df in df.groupby("episode_id", sort=False):
                    ep_rows = ep_df.sort_values("frame_id")
                    prev_cf = -1
                    prev_node = -1

                    for _, row in ep_rows.iterrows():
                        p_id = int(row.get("period_id", 1))
                        cf = _to_canonical_frame(p_id, row.get("frame_id"))
                        node = _to_node_idx(str(row.get("player_id", "")))

                        if cf < 0:
                            continue

                        # 1. Self possession at event touch
                        if node >= 0:
                            dense_self = node * 26 + node
                            comp_self = orig2comp_np[dense_self]
                            if comp_self >= 0:
                                edge_ids[cf] = comp_self
                                confident[cf] = True

                        # 2. Path transfer between consecutive events in same episode
                        if (
                            prev_node >= 0 and prev_cf >= 0 and 0 < (cf - prev_cf) <= 75
                        ):  # within 3 seconds
                            # Directed transfer edge prev_node -> node
                            target_node = (
                                node if node >= 0 else 22
                            )  # default outside left if out of bounds
                            dense_transfer = prev_node * 26 + target_node
                            comp_transfer = orig2comp_np[dense_transfer]
                            if comp_transfer >= 0:
                                # Mark the transfer frames between touches
                                edge_ids[prev_cf + 1 : cf] = comp_transfer
                                confident[prev_cf + 1 : cf] = True

                        prev_cf = cf
                        prev_node = node

                return PlayerPathAnnotations(
                    edge_id=edge_ids,
                    confident=confident,
                    source_match=clean_id,
                    annotation_method="synced_events",
                    provenance={
                        "parquet_file": str(parquet_path),
                        "total_events": len(df),
                        "labeled_frames": int(confident.sum()),
                    },
                )
            except (OSError, ValueError, KeyError, TypeError, ImportError) as e:
                return PlayerPathAnnotations(
                    edge_id=np.full(n_frames, -1, dtype=np.int64),
                    confident=np.zeros(n_frames, dtype=bool),
                    source_match=clean_id,
                    annotation_method="none_available",
                    provenance={"reason": f"Exception during label extraction: {e}"},
                )

    return PlayerPathAnnotations(
        edge_id=np.full(n_frames, -1, dtype=np.int64),
        confident=np.zeros(n_frames, dtype=bool),
        source_match=clean_id,
        annotation_method="none_available",
        provenance={"reason": "Synced event file unavailable or could not be loaded"},
    )


# Convenient alias matching naming standards
extract_strong_path_labels = generate_training_path_annotations


def is_path_supervision_available(match_id: str) -> bool:
    """Check whether a match is approved for path supervision."""
    clean_id = match_id.replace("idsse_", "")
    return clean_id in APPROVED_TRAIN_MATCHES and clean_id not in DENIED_MATCHES


def audit_path_supervision(
    match_id: str,
    n_frames: int,
    synced_parquet_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Audit path supervision statistics for an external match."""
    if synced_parquet_dir is None:
        synced_parquet_dir = Path("artifacts/local/pathcrf_upstream/data/sportec_synced")

    ann = generate_training_path_annotations(match_id, n_frames, synced_parquet_dir)
    n_valid = int(ann.confident.sum())
    coverage = (n_valid / max(1, n_frames)) * 100.0

    valid_edges = ann.edge_id[ann.confident]
    unique_states = int(len(np.unique(valid_edges))) if len(valid_edges) > 0 else 0

    inside_edges = int(np.sum(valid_edges < 484)) if len(valid_edges) > 0 else 0
    outside_edges = n_valid - inside_edges

    return {
        "match_id": match_id,
        "n_frames": n_frames,
        "available": ann.annotation_method != "none_available",
        "valid_path_frames": n_valid,
        "path_coverage_pct": coverage,
        "inside_pitch_edges": inside_edges,
        "outside_edges": outside_edges,
        "unique_edge_states_observed": unique_states,
        "provenance": ann.provenance,
    }
