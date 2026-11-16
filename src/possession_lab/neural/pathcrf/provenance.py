"""Provenance and licensing metadata for the PathCRF adaptation (N020)."""

UPSTREAM_REPO: str = "https://github.com/hyunsungkim-ds/pathcrf"
UPSTREAM_COMMIT: str = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
UPSTREAM_LICENSE: str = "MPL-2.0"
UPSTREAM_PAPER: str = (
    "Kim et al., PathCRF: Ball-Free Soccer Event Detection via Possession Path "
    "Inference from Player Trajectories, KDD 2026 (arXiv:2602.12080)"
)

# Allowlist for training data event derivation: strictly train matches only.
APPROVED_TRAIN_MATCHES: frozenset[str] = frozenset(
    {
        "idsse_J03WMX",
        "idsse_J03WOH",
        "idsse_J03WOY",
        "idsse_J03WPY",
        "J03WMX",
        "J03WOH",
        "J03WOY",
        "J03WPY",
    }
)

# Validation, audit and challenge matches must never provide training labels or statistics.
DENIED_MATCHES: frozenset[str] = frozenset(
    {
        "idsse_J03WQQ",
        "J03WQQ",
        "idsse_J03WN1",
        "J03WN1",
        "idsse_J03WR9",
        "J03WR9",
        "metrica_2",
        "metrica_3",
        "challenge",
    }
)
