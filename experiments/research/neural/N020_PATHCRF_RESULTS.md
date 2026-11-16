# N020 PathCRF — Evaluation Results & Scientific Report

**Experiment:** N020 — PathCRF-Based Possession Recognition  
**Branch:** `experiment/n020-pathcrf`  
**Date:** 2026-11-12  
**Author:** Principal Machine Learning Research Engineer & PyTorch Architect  
**Champion Target:** N018 (`spellF1 = 0.6919`, clean `0.7673`, degraded `0.6166`, val `0.5568`, alive/dead F1 `0.9679`)  
**Promotion Threshold:** Audit mean `spellF1 >= 0.7019` (+0.0100 absolute over N018), validation `> 0.5568`, alive/dead F1 within `-0.005`  
**Verdict:** **PROMOTION CRITERIA NOT MET. PRESERVE N018 AS UNCHANGED CHAMPION.**  

---

## 1. Executive Summary

We designed, implemented, verified, trained, and audited **N020 — PathCRF-Based Possession Recognition**, faithfully adapting the upstream research from Kim et al. (*PathCRF: Player Path-Centric Conditional Random Field for Football Possession Recognition*, KDD 2026, commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, MPL-2.0).

N020 models continuous possession transitions across player-to-player paths on a 26-node graph (22 player nodes + 4 outside boundary nodes) using a compressed 576-state directed Conditional Random Field, trained on approved external Bundesliga open data (`idsse_J03WMX`, `idsse_J03WOH`, `idsse_J03WOY`, `idsse_J03WPY`) and SkillCorner/Metrica tracking data. The model operates under strict ball-input invariance (no ball coordinates, velocities, or alive/dead flags).

At the pre-registered checkpoint evaluation (Step 1,000 `best.pt`, SHA-256 `5343caff79de8328d5fdd6ddf60106f44db14b2a95b8d965d4521a369dcda3cc`), the model achieves a headline development audit mean `spellF1 = 0.1141` (Clean: `0.1539`, Degraded: `0.0743`). This falls **0.5778 absolute below** the N018 champion (`0.6919`). Alive/dead discrimination degrades to `0.6867` (vs N018 `0.9679`), and boundary transition F1 drops to `0.0571` at $\pm 12$ frames (vs N018 `0.7016`).

Consequently, N020 fails all pre-registered promotion gates. **N018 remains the unchallenged champion.**

---

## 2. Headline Comparative Audit Results

Scored on the frozen 3-match development audit (`idsse_J03WN1`, `idsse_J03WR9`, `metrica_3`) using the unchanged official competition metric (`possession_metrics.py`):

| Metric | N018 Champion | N020 Candidate (PathCRF) | Delta ($\Delta$) | Promotion Gate | Status |
|---|---|---|---|---|---|
| **Headline Audit Mean `spellF1`** | **0.6919** | **0.1141** | **-0.5778** | $\ge 0.7019$ | **FAIL** |
| — Clean Condition `spellF1` | 0.7673 | 0.1539 | -0.6134 | — | — |
| — Degraded Condition `spellF1` | 0.6166 | 0.0743 | -0.5423 | $\ge 0.5731$ | **FAIL** |
| **External Validation `spellF1`** | **0.5568** | **0.1579** | **-0.3989** | $> 0.5568$ | **FAIL** |
| **Alive / Dead F1 (Pooled Mean)** | **0.9679** | **0.6867** | **-0.2812** | $\ge 0.9629$ | **FAIL** |
| — Clean Alive / Dead F1 | 0.9763 | 0.6340 | -0.3423 | — | — |
| — Degraded Alive / Dead F1 | 0.9595 | 0.7394 | -0.2201 | — | — |
| **Macro F1 (Clean / Degraded)** | 0.9311 / 0.8890 | 0.5320 / 0.3135 | -0.3991 / -0.5755 | — | — |
| **Transition F1 @ $\pm 12$ frames (Clean)** | 0.7016 | 0.0571 | -0.6445 | — | — |
| **Transition F1 @ $\pm 25$ frames (Clean)** | 0.7852 | 0.1083 | -0.6769 | — | — |
| **Transition F1 @ $\pm 12$ frames (Degraded)** | 0.5284 | 0.0247 | -0.5037 | — | — |
| **Transition F1 @ $\pm 25$ frames (Degraded)** | 0.6385 | 0.0617 | -0.5768 | — | — |

---

## 3. Per-Match Audit Breakdown

### 3.1 Clean Condition Breakdown

| Match | Model | `spellF1` | `macroF1` | Alive/Dead F1 | Team Acc (Alive) | Pred Spells (GT) | Trans F1 (tol 12/25) |
|---|---|---|---|---|---|---|---|
| `idsse_J03WN1` | N018 | 0.7356 | 0.9351 | 0.9897 | 0.9173 | 382 (500) | 0.7021 / 0.7864 |
| | **N020** | **0.1212** | **0.5420** | **0.6389** | **0.6603** | **137 (500)** | **0.0455 / 0.1102** |
| `idsse_J03WR9` | N018 | 0.7650 | 0.9546 | 0.9933 | 0.9462 | 304 (370) | 0.7283 / 0.8056 |
| | **N020** | **0.1688** | **0.5246** | **0.5697** | **0.7408** | **149 (370)** | **0.0815 / 0.1208** |
| `metrica_3` | N018 | 0.8012 | 0.9037 | 0.9459 | 0.9699 | 309 (365) | 0.6745 / 0.7634 |
| | **N020** | **0.1718** | **0.5295** | **0.6935** | **0.7212** | **196 (365)** | **0.0443 / 0.0940** |

### 3.2 Degraded Condition Breakdown (`degrade_eval_seed = 7`)

| Match | Model | `spellF1` | `macroF1` | Alive/Dead F1 | Team Acc (Alive) | Pred Spells (GT) | Trans F1 (tol 12/25) |
|---|---|---|---|---|---|---|---|
| `idsse_J03WN1` | N018 | 0.6012 | 0.9110 | 0.9822 | 0.8921 | 358 (500) | 0.5412 / 0.6482 |
| | **N020** | **0.0714** | **0.3012** | **0.7412** | **0.3421** | **112 (500)** | **0.0210 / 0.0520** |
| `idsse_J03WR9` | N018 | 0.6241 | 0.9324 | 0.9845 | 0.9214 | 288 (370) | 0.5821 / 0.6841 |
| | **N020** | **0.0812** | **0.3120** | **0.7210** | **0.3842** | **124 (370)** | **0.0312 / 0.0710** |
| `metrica_3` | N018 | 0.6245 | 0.8236 | 0.9118 | 0.8841 | 290 (365) | 0.4619 / 0.5832 |
| | **N020** | **0.0703** | **0.3273** | **0.7560** | **0.3729** | **138 (365)** | **0.0219 / 0.0621** |

---

## 4. Scientific Failure Mode Analysis

Why does PathCRF underperform N018 by over 0.57 `spellF1`?

### 4.1 The Fundamental Flaw: Player-Only Inputs Without Ball Kinematics
In the original PathCRF formulation (Kim et al., KDD 2026), the model was designed to track player paths during continuous *in-play broadcast clips* where the ball was known a priori to be in play.
When applied to full-match 90-minute sequences containing extensive Dead-ball episodes (stoppages, goal kicks, throw-ins, fouls):
1. **Dead vs. Alive Indistinguishability**: In football, players regularly run, press, and jog during throw-in setups and free-kick arrangements. Without ball $(x, y)$ coordinates and ball speed, player trajectory clusters alone cannot disambiguate an active short-passing possession from an impending goal kick or referee stoppage.
2. **Dead-Ball Regression**: N020's alive/dead F1 dropped from N018's **0.9763** down to **0.6340** (Clean) and **0.7394** (Degraded). This severe false-positive dead-ball leakage truncates spells and invalidates IoU matching.

### 4.2 Massive Spell Under-Segmentation & Severe Transition Lag
The spell matching results explain the collapse in `spellF1`:
- On `idsse_J03WN1`, there are **500 true ground-truth spells**.
- N018 predicts **382 spells** (mean duration 7.2 s vs GT 5.5 s), achieving `spellF1 = 0.7356`.
- N020 predicts only **137 spells** (mean duration 10.3 s), collapsing to `spellF1 = 0.1212`.
- Transition F1 at $\pm 12$ frames ($0.48$ s) is only **0.0571** (vs N018 **0.7016**). N020 completely misses exact pass and tackle turnover boundaries, lagging by several seconds because it relies on macroscopic player movement shifts rather than instantaneous ball velocity impulses.

### 4.3 Fragility to Degraded Tracking
Under the degraded tracking protocol (seed 7, tracklet loss, identity jitter, slot swaps):
- Team accuracy on alive frames collapsed from **70.7% (clean)** down to **36.6% (degraded)**.
- When player tracking drops or swaps slots, the directed 26-node graph edges lose continuity. In contrast, N018's Set Attention architecture with trajectory forecasting (`N019-A`) remains resilient because spatial pooling over token sets is permutation-invariant and robust to tracklet dropouts.

---

## 5. Official Submission Artifact Verification

To satisfy competition deliverable constraints, the challenge inference pipeline was run end-to-end to generate the candidate submission file:

| Property | Value |
|---|---|
| **Submission File** | `artifacts/neural/submissions/n020_pathcrf.csv` |
| **Row Count** | Exactly 199,319 rows (frame 0 to 199,318) |
| **Header** | `frame,label` |
| **File SHA-256 Digest** | `64867ecce82ca58a55224e4cd100f9372c82c430577b9be22b8b0c5ce5386671` |
| **Class Distribution** | H: 46,384 (23.3%), A: 22,311 (11.2%), D: 90,162 (45.2%), X: 40,462 (20.3%) |
| **Possession Count** | 198 possessions (mean duration 13.9 s) |
| **Official Validator Check** | **PASSED (VALID)** via `evaluation/validate_submission.py` |

---

## 6. Promotion Decision & Archival Plan

### Verification Against Promotion Gates
- **Gate 1: Audit Mean `spellF1 >= 0.7019`**: **FAILED** (0.1141, -0.5778).
- **Gate 2: External Validation `spellF1 > 0.5568`**: **FAILED** (0.1579, -0.3989).
- **Gate 3: Alive/Dead F1 Noninferior within -0.005**: **FAILED** (0.6867, -0.2812).

### Final Verdict
**N020 is NOT promoted. N018 is preserved as the active champion.**

The `n020_pathcrf` implementation, tests, checkpoints, and documentation are fully committed on branch `experiment/n020-pathcrf` for scientific completeness, ensuring zero adverse impact on `main` or `experiments/possession-research`.

