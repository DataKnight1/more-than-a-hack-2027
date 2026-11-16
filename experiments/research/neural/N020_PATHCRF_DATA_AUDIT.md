# N020 PathCRF Data Provenance, Governance & Firewall Audit

**Author:** Principal Machine Learning Research Engineer & PyTorch Architect  
**Date:** 2026-11-12  
**Branch:** `experiment/n020-pathcrf`  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

This document certifies that the implementation, training, validation, and evaluation of **N020 — PathCRF-Based Possession Recognition** adhere strictly to:
1. The repository constitution (`docs/governance/RESEARCH_CONSTITUTION.md`).
2. The data policy and competition rules (`docs/governance/DATA_POLICY.md`, `AGENTS.md`, `RULES_CLARIFICATION.md`).
3. The upstream intellectual property rights and license terms (`MPL-2.0`).

All training data originates exclusively from eligible external matches. No ground-truth possession labels or video from the challenge match or the public five-minute GT were used for training, fitting, or model selection.

---

## 2. Upstream Repository & Intellectual Property Provenance

| Property | Detail |
|---|---|
| **Upstream Repository** | `https://github.com/hyunsungkim-ds/pathcrf` |
| **Paper Reference** | Kim et al., *PathCRF: Player Path-Centric Conditional Random Field for Football Possession Recognition*, KDD 2026, arXiv:2602.12080 |
| **Pinned Commit SHA** | `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` |
| **Upstream License** | Mozilla Public License 2.0 (MPL-2.0) |
| **Local Notice** | Documented in `THIRD_PARTY_NOTICES.md` |
| **Algorithmic Adaptations** | Pure PyTorch Set Attention backbone (`ISAB` / `SAB`), 576-state compressed directed edge graph, 12-parameter transition graph, exact log-partition $\log Z$, Viterbi sequence decoding |
| **Pretrained Checkpoint Audit** | Upstream released checkpoint `saved/140` was trained on DFL match `J03WN1`. Because `J03WN1` is part of our established 3-match development audit, `saved/140` is **strictly quarantined and excluded** to prevent test contamination. |

---

## 3. Dataset Partitions and Explicit Match Mapping

### 3.1 Partition Summary

```
                      External Open Datasets
                               │
       ┌───────────────────────┼──────────────────────┐
       ▼                       ▼                      ▼
  DFL Bundesliga       SkillCorner Open Data    Metrica Open Data
  (Tracking + Events)  (Tracking + Dead/Alive)  (Tracking + Ep)
       │                       │                      │
       ├─ Train (4)            ├─ Train (12)          ├─ Train: M1
       ├─ Val (1)              ├─ Val (3)             ├─ Val: M2
       └─ Audit (2)            └─ [None in Audit]     └─ Audit: M3
```

### 3.2 Detailed Split Membership

| Match Identifier | Dataset Origin | Assigned Partition | Available Annotations | Used for Path Supervision? |
|---|---|---|---|---|
| `idsse_J03WMX` | DFL Bundesliga Open Data | Train | Tracking (25Hz) + Events | **YES** (Approved) |
| `idsse_J03WOH` | DFL Bundesliga Open Data | Train | Tracking (25Hz) + Events | **YES** (Approved) |
| `idsse_J03WOY` | DFL Bundesliga Open Data | Train | Tracking (25Hz) + Events | **YES** (Approved) |
| `idsse_J03WPY` | DFL Bundesliga Open Data | Train | Tracking (25Hz) + Events | **YES** (Approved) |
| `skillcorner_1874553` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_1899585` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_1927964` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_1953632` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_1959846` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_1996435` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2006229` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2007448` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2007721` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2010085` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2013725` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `skillcorner_2016236` | SkillCorner Open Data | Train | Tracking (10Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `metrica_1` | Metrica Open Data | Train | Tracking (25Hz) + Team/Dead | NO (Team/Dead CE/BCE only) |
| `idsse_J03WQQ` | DFL Bundesliga Open Data | Validation | Tracking (25Hz) + Team/Dead | **NO (DENIED / FIREWALL)** |
| `skillcorner_1886347` | SkillCorner Open Data | Validation | Tracking (10Hz) + Team/Dead | NO (Validation only) |
| `skillcorner_1996436` | SkillCorner Open Data | Validation | Tracking (10Hz) + Team/Dead | NO (Validation only) |
| `skillcorner_2015213` | SkillCorner Open Data | Validation | Tracking (10Hz) + Team/Dead | NO (Validation only) |
| `metrica_2` | Metrica Open Data | Validation | Tracking (25Hz) + Team/Dead | **NO (DENIED / FIREWALL)** |
| `idsse_J03WN1` | DFL Bundesliga Open Data | Audit | Tracking (25Hz) + Ground Truth | **NO (DENIED / FIREWALL)** |
| `idsse_J03WR9` | DFL Bundesliga Open Data | Audit | Tracking (25Hz) + Ground Truth | **NO (DENIED / FIREWALL)** |
| `metrica_3` | Metrica Open Data | Audit | Tracking (25Hz) + Ground Truth | **NO (DENIED / FIREWALL)** |
| `challenge` | 2027 Competition Fixture | Challenge | 25Hz Tracking Only | **NO (DENIED / FIREWALL)** |

---

## 4. Programmatic Firewall Verification

The file `src/possession_lab/neural/pathcrf/provenance.py` defines:

```python
APPROVED_TRAIN_MATCHES: frozenset[str] = frozenset({
    "idsse_J03WMX", "J03WMX",
    "idsse_J03WOH", "J03WOH",
    "idsse_J03WOY", "J03WOY",
    "idsse_J03WPY", "J03WPY",
})

DENIED_MATCHES: frozenset[str] = frozenset({
    "idsse_J03WN1", "J03WN1",
    "idsse_J03WR9", "J03WR9",
    "idsse_J03WQQ", "J03WQQ",
    "metrica_2", "metrica_3",
    "challenge", "test", "audit", "val",
})
```

`src/possession_lab/neural/pathcrf/labels.py` enforces:
1. `extract_strong_path_labels()` raises a `ValueError` immediately if any denied match or non-approved match is passed.
2. Unit tests in `tests/test_n020_no_leakage.py` verify that all denied match names throw exceptions and cannot generate path annotations under any circumstances.
3. No heuristic pseudo-labeling is performed on team possession annotations to invent synthetic player-level paths.

---

## 5. Input Invariance & Evaluation Protocol

1. **Strict Ball Invariance**: All models run strictly on 26 player/boundary nodes without $(x, y)$ ball coordinates, ball velocity, or ball alive/dead status. Verified by `tests/test_n020_adapter.py::test_strict_ball_invariance`.
2. **Label Privacy**: Inference and evaluation pipelines set `include_labels=False`. Ground-truth labels are never passed to the model during probability computation.
3. **Decoupled Evaluation**: Model hyperparameter selection and decoder calibration are performed exclusively on the 5 degraded external validation matches. The frozen candidate is evaluated once on the 3 development audit matches in both clean and degraded conditions.

