"""Inspect and audit strong path labels from permitted external training matches.

Verifies:
1. Hard firewall: denied matches (J03WN1, J03WR9, J03WQQ, Metrica, Challenge) are rejected.
2. Approved matches (J03WMX, J03WOH, J03WOY, J03WPY) are loaded and audited.
3. Statistics on event-derived player-to-player and player-to-outside possession transitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from possession_lab.neural.canonical import CanonicalMatch  # noqa: E402
from possession_lab.neural.pathcrf.labels import (  # noqa: E402
    audit_path_supervision,
    extract_strong_path_labels,
)
from possession_lab.neural.pathcrf.provenance import (  # noqa: E402
    APPROVED_TRAIN_MATCHES,
    DENIED_MATCHES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=ROOT / "data" / "external" / "canonical",
        help="Path to canonical .npz directory",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("N020 PATHCRF: STRONG PATH SUPERVISION AUDIT & DATA PROVENANCE CHECK")
    print("================================================================================")
    print(f"Approved training matches: {APPROVED_TRAIN_MATCHES}")
    print(f"Explicitly denied matches: {sorted(DENIED_MATCHES)}")
    print("--------------------------------------------------------------------------------")

    # 1. Verify firewall rejects denied matches
    print("\n[1] Verifying Data Leakage Firewall...")
    for denied in ["idsse_J03WN1", "idsse_J03WR9", "idsse_J03WQQ", "metrica_3"]:
        try:
            extract_strong_path_labels(denied, 1000)
            print(f"FAIL: Firewall failed to block {denied}!")
            return 1
        except ValueError as e:
            print(f"PASS: Firewall blocked {denied}: {e}")

    # 2. Audit approved matches
    print("\n[2] Auditing Approved External Training Matches...")
    total_frames = 0
    total_valid_path_frames = 0

    approved_unique = sorted({m.replace("idsse_", "") for m in APPROVED_TRAIN_MATCHES})
    for clean_id in approved_unique:
        npz_name = f"idsse_{clean_id}.npz"
        npz_path = args.canonical_dir / npz_name
        if not npz_path.is_file():
            print(f"WARNING: Canonical match file {npz_path} not found; skipping.")
            continue

        match = CanonicalMatch.load(npz_path)
        stats = audit_path_supervision(f"idsse_{clean_id}", match.n_frames)

        print(f"\nMatch: idsse_{clean_id} (Total 25Hz frames: {match.n_frames:,})")
        print(f"  Available for path supervision: {stats['available']}")
        print(
            f"  Total valid path frames:       {stats['valid_path_frames']:,} ({stats['path_coverage_pct']:.2f}%)"
        )
        print(f"  Inside-pitch edge frames:      {stats['inside_pitch_edges']:,}")
        print(f"  Outside boundary edge frames:  {stats['outside_edges']:,}")
        print(f"  Unique 576 edge states active: {stats['unique_edge_states_observed']}")

        total_frames += match.n_frames
        total_valid_path_frames += stats["valid_path_frames"]

    if total_frames > 0:
        overall_pct = (total_valid_path_frames / total_frames) * 100.0
        print("\n--------------------------------------------------------------------------------")
        print(
            f"Overall Approved Matches: {total_valid_path_frames:,} / {total_frames:,} frames ({overall_pct:.2f}% path coverage)"
        )
        print("--------------------------------------------------------------------------------")

    print("\nAudit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
