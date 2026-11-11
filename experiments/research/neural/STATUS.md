# Neural tracking campaign — status log

Append-only. Newest stage at the bottom. Facts only; every number is from a command actually run.

## Stage 0 — audit (2026-11-06)

**Resume authority.** Tiago's message of 2026-11-06 explicitly instructs this agent to download
and normalise eligible public tracking datasets and to train neural models on them. That is the
latest explicit instruction; it is recorded here without rewriting earlier pause records
(`AGENTS.md`, `research/program.md`). SoccerNet downloads remain Tiago's responsibility and are not
touched. Licence verification is still required per dataset (below).

| Check | Result |
|---|---|
| git | `main` @ `3de6b60`; overlay files untracked; no commits made |
| Official files vs HEAD | unchanged (`git diff HEAD` empty for `evaluation/`, `starter/`, `games.py`, rules docs) |
| Upstream | `upstream/main` is 2 commits ahead (`c073f29`, `f8f2629`, 2026-10-22/23): **docs only** (RULES, DATA, EVALUATION, SUBMISSION, FAQ, OVERVIEW, CHALLENGE, README). No scorer/validator/baseline code change. Not pulled — Tiago to review. Key additions: other-game hand labels allowed in any amount; tech doc must list every labelled training dataset (source, who labelled, how much); no human-produced data about the provided game from any source. |
| `scripts/check_upstream.py --check` | "No baseline recorded" — `--record` requires Tiago's review, not run. The local review snapshot in `artifacts/local/setup-upstream-review/` matches the *newer* upstream wording. |
| Baseline C000 | `scripts/check_baseline.py` → **PASS**: regenerated `baseline_smooth50` matches the reference byte-for-byte on 199,319 frames and passes the official validator. No GT read. |
| Tests | 22/22 existing tests pass (`.venv-mlops`, Python 3.11.9) |
| Environments | `.venv` is **broken** (Scripts has only `python.exe`; points at a WindowsApps Python 3.13) — left untouched. `.venv-mlops` works (no torch). New isolated `.venv-neural` (Python 3.11) from `setup/neural/requirements-neural.txt`; root `pyproject.toml`/`uv.lock` untouched. |
| Hardware | NVIDIA RTX 4050 Laptop, 6 GB VRAM, driver CUDA 13.3; 16 logical CPUs; D: 3.1 TB free; C: 96 GB free |
| Challenge data | present locally in `data/tactical`, `data/broadcast` (gitignored). Contents never printed. |
| Instruction pack | `AGENTS_NEURAL_ADDENDUM.md` installed from the supplied text; pointer section appended to `AGENTS.md`. The pack's `docs/agent/00–15_*`, `docs/agent/INDEX.md`, extra skills and `research/neural/*_TEMPLATE.md` were **not on disk** (zip never extracted) — missing, not invented. |

## Stage 1 — external data sources (2026-11-06)

| Source | Licence (verified how) | Labels | Status |
|---|---|---|---|
| **IDSSE / DFL open data** — Bassek et al. 2025, *Sci Data* 12:195, figshare 10.6084/m9.figshare.28196177.v1 | **CC BY 4.0** (figshare API `license` field, checked at download) | Provider per-frame `BallStatus` (alive/dead) + `BallPossession` (home/away) in the 25 Hz positions feed → H/A/D | 7 matches downloaded with MD5 verification (`scripts/neural/fetch_idsse.py`) |
| SkillCorner Open Data (A-League 2024/25, 10 Hz broadcast tracking) | **MIT** (GitHub repo licence) | Per-frame `possession.group`; phases of play only defined while ball in play | eligible; not yet downloaded (Git LFS) — next for cross-provider holdout |
| Metrica Sports sample data (3 anonymised games, 25 Hz) | **No licence file**; README: "be responsible… acknowledge the source" | Events (would need algorithmic possession derivation) | **pending Tiago's decision** — not used |
| StatsBomb 360 | Open data licence (non-continuous snapshots) | — | not continuous tracking; auxiliary only, not planned |
| SoccerNet | — | — | Tiago's responsibility; untouched |

**Frozen IDSSE split** (fixed before any training, recorded in `configs/neural/n001_set_temporal_base.toml`):
train `J03WMX, J03WOH, J03WOY, J03WPY`; validation `J03WQQ`; test `J03WN1, J03WR9`. Split by
match; no windows of one match cross splits.

**Domain gap (documented, not tuned on the challenge):** DFL tracking is clean and near-complete;
the challenge ATD is automatic tracking with ~43 % missing ball, tracklets instead of identities and
duplicates (organiser `DATA.md`). Training uses broad, physically motivated degradation
(`src/possession_lab/neural/augment.py`) whose ranges are *not* fitted to challenge statistics. The
DFL possession definition may differ from the challenge GT definition; this cannot be calibrated
legally and is a stated limitation.

### Stage 1 results (2026-11-06)

- **IDSSE**: 21 files MD5-verified (`data/external/idsse/manifest.json`); 7 matches converted to
  `data/external/canonical/idsse_*.npz` (`scripts/neural/convert_idsse.py`). Per match: 137–147k
  frames at 25 fps, ball observed 99.98 %, D share 41–52 %, 370–511 possession spells, mean spell
  5.5–9.2 s. Cards: `data/external/canonical/idsse_cards.json`.
- **SkillCorner**: repo `SkillCorner/opendata` @ `4340d27` cloned with LFS (1.8 GB); 20 matches
  converted (`scripts/neural/convert_skillcorner.py`). Extrapolated objects dropped to NaN →
  *natural* broadcast missingness: ball observed 52–65 %, 7.5–10.9 detected players/frame. D share
  37–55 %, 295–403 spells, mean spell 7.8–12.6 s. Split frozen (rule in
  `configs/neural/n002_set_temporal_mixed.toml`): 12 train / 3 val / 5 test.
- Label definitions differ by provider (DFL feed flags vs SkillCorner phases+possession group);
  both differ from the challenge GT definition in unknown ways.

**Baseline reference on external data** (`scripts/neural/baseline_external.py`, organiser
`smooth()` unchanged; spellF1 / macroF1):

| match | condition | raw | smooth50 |
|---|---|---|---|
| IDSSE J03WQQ (val) | degraded seed 7 | 0.009 / 0.388 | 0.215 / 0.403 |
| IDSSE J03WN1 (test) | degraded seed 7 | 0.010 / 0.354 | 0.190 / 0.338 |
| IDSSE J03WR9 (test) | degraded seed 7 | 0.008 / 0.408 | 0.193 / 0.409 |
| SkillCorner 5 test matches | natural | 0.069–0.084 / 0.69–0.73 | 0.204–0.257 / 0.665–0.719 |

SkillCorner-natural smooth50 numbers sit close to the organiser's published challenge figure
(0.2433 / 0.6724). This is context only — it is *not* used to tune anything.

### Environment note

`.venv-neural` on D: installed at ~0.3 MB/s (cross-volume copy from the C: uv cache, likely
Defender scanning) and was abandoned mid-install. Working env: `%LOCALAPPDATA%\mtah\venv-neural`
(same volume as the uv cache → hardlinks, 24 s install): torch 2.7.1+cu128, CUDA OK on RTX 4050.
A partial `D:\uv-cache` mirror could not be deleted by the agent (protected path) — safe for
Tiago to remove. 12/12 neural tests pass in this env.

## Stage 2 — first neural model (2026-11-06)

**N001** (`configs/neural/n001_set_temporal_base.toml`): set transformer over {CLS, ball, ≤30
player tokens} per frame (2 layers, d=96) → temporal transformer (4 layers) over 256-frame
(10.2 s) windows → H/A/D logits; 0.69 M params. Trained 6k steps (batch 16, AdamW, cosine, AMP
fp16) on 4 IDSSE matches with per-window domain randomisation + mirror/team-swap symmetry. 7.6
it/s, 1.3 GB peak VRAM on the RTX 4050; ~14 min. Best checkpoint at step 5000 by external-val
spellF1 (0.607 degraded). Decoder chosen on external val (clean+degraded mean): team switch 4,
dead switch 8 nats.

Frozen external test (official metric code, unchanged; spellF1 / macroF1; n_pred vs n_gt spells):

| test set | condition | N001 + Viterbi | N001 argmax | baseline smooth50 |
|---|---|---|---|---|
| IDSSE J03WN1 + J03WR9 (435 spells) | clean | **0.628 / 0.917** (304) | 0.547 / 0.918 | 0.300 / 0.379 (375) |
| IDSSE J03WN1 + J03WR9 | degraded seed 7 | **0.538 / 0.887** (282) | 0.213 / 0.887 | 0.192 / 0.374 (664) |
| SkillCorner 5 test matches (341 spells), **provider unseen in training** | natural | **0.344 / 0.779** (399) | 0.230 / 0.776 | 0.239 / 0.700 (521) |
| SkillCorner 5 test matches | + degraded seed 7 | **0.336 / 0.773** (385) | 0.091 / 0.769 | 0.169 / 0.666 (715) |

Reading: the gain survives a provider/label-definition change (+0.105 spellF1 over smooth50 on
SkillCorner natural, above the official 0.04 noise band), but transfer costs ~0.2 spellF1 versus
in-provider. Spell-aware decoding is essential (argmax fragments 3–6×). Evidence tier A for
*other* games only — it is **not** a claim about the private challenge score.

## Stage 4 (early) — first neural challenge candidate (2026-11-06)

`scripts/neural/predict_challenge.py --run n001_set_temporal_base`: CPU float32 deterministic
inference on the tactical ATD, fixed decoder. Output `artifacts/neural/submissions/n001_set_temporal_base.csv`
(gitignored) — official validator **OK**: 199,319 rows, frames 0..199318, X only on the 40,462
ungraded frames, validator reports 320 possessions (mean 11.0 s). File sha256
`ba69385435db435f4a3f018c4caa4bb412eb2eb0c76c78df2d187d5455f6485c`; checkpoint sha256
`e773a1b49c85eebc45c936256dab18b07bee5acd64d0680e33da2e1195ecb743`. No ground truth read; no
public-GT scoring run. Fallback C000/C001 unchanged.

**Reproducibility check (2026-11-06):** a second independent run of `predict_challenge.py` wrote
`n001_regen.csv` with the **identical sha256** `ba693854…485c`; official validator OK on both.
(A cosmetic path-printing bug with relative `--out` was fixed afterwards; it did not affect bytes.)

## Stage 3 — scaling (in progress, 2026-11-06)

- **N002** (N001 architecture, IDSSE + 12 SkillCorner train matches, 12k steps): crashed at step
  ~6300 with a transient `CUDA error: out of memory` although the run peaks at 1.3 GB — the 6 GB
  laptop GPU is shared with desktop processes. External-val spellF1 before the crash: 0.357 →
  0.421 → 0.441 (steps 2k/4k/6k; mixed IDSSE+SkillCorner val, not comparable to N001's val).
  Resumed from `last.pt` (step 6000) via `scripts/neural/run_chain.sh`, which now auto-resumes up
  to 3 times on crash and runs the frozen external evaluation (+ SkillCorner-only table).
- **N003** large (d=192, 3 spatial + 6 temporal layers, 15.4 s windows, batch 8 × accum 2,
  16k steps; `configs/neural/n003_set_temporal_large.toml`) queued after N002 on the same splits.

### Stage 3 results (2026-11-06, ~15:40)

N002 resumed twice after the same driver-level `CUDA error: out of memory` (raised from
`CUDAEvent::record` in backward, not a tensor allocation; run peaks at 1.3 GB) and completed
12k steps. N003 completed 16k steps with no crash (2.1 it/s, 2.9 GB peak, ~2 h).

Frozen external test, neural + Viterbi (decoder chosen on external val; both chose team=4,
dead=4). spellF1 / macroF1:

| test set | N001 | N002 | N003 large | C000 smooth50 |
|---|---|---|---|---|
| SkillCorner 5, natural | 0.344 / 0.779 (unseen provider) | 0.447 / 0.851 | **0.466 / 0.861** | 0.239 / 0.700 |
| SkillCorner 5, + degraded | 0.336 / 0.773 | 0.423 / 0.837 | **0.440 / 0.848** | 0.169 / 0.666 |
| IDSSE 2, clean | **0.628** / 0.917 | 0.543 / 0.904 | 0.594 / 0.916 | 0.300 / 0.379 |
| IDSSE 2, degraded | **0.538** / 0.887 | 0.443 / 0.860 | 0.521 / 0.887 | 0.192 / 0.374 |
| all 7, clean (unweighted mean) | — | 0.474 / 0.866 | **0.502 / 0.876** | 0.256 / 0.608 |

Reading: scale helps (N003 ≥ N002 on every row, +0.02–0.08, but mostly inside the 0.04 band);
mixed-provider training trades IDSSE accuracy for SkillCorner accuracy. All neural models
under-segment (N003: 216 predicted vs 341 true SkillCorner spells). The challenge is a third
provider, so cross-provider robustness is the relevant property; **N004** (N003 architecture,
SkillCorner-only, tested on unseen IDSSE) is running to measure it for the large model.

**N003 challenge candidate (2026-11-06):** `artifacts/neural/submissions/n003_set_temporal_large.csv`
— official validator OK (199,319 rows), validator reports 265 possessions (mean 12.9 s); file
sha256 `346c887ae89212db365373b93a038a4226a99c7e7c78934d8b5d8bf8e1fe19f3`, checkpoint sha256
`73a90be9744a8d1e615f3977a8fbe86d9c5d3c1237347c31d03cc9da5a65a154`, decoder team=4 dead=4.
Independent regeneration (`n003_regen.csv`) is **byte-identical**. No ground truth read.

### Stage 3 results, continued (2026-11-06, ~17:30)

- **N004** (N003 architecture, SkillCorner-only training) on the **unseen** IDSSE provider (3
  matches, 424 spells): clean **0.482 / 0.874** vs smooth50 0.320 / 0.392; degraded **0.399 /
  0.823** vs 0.200 / 0.384. The large architecture transfers across providers (+0.16–0.20
  spellF1); N001's opposite-direction transfer was +0.105. Still ~2x under-segmented (209–218 vs 424).
- **Decoder study (N003, D-007 rule):** asymmetric dead-ball entry/exit costs (4, 1, 8) vs
  production symmetric (4, 4): val 0.5043 vs 0.5040, 7-match test 0.4830 vs 0.4826 — **no
  material change; production decoder kept.** Under-segmentation is a property of the model's
  probabilities, not of the switch penalties.
- **Ensemble N002 + N003 (D-008 rule):** probability averaging, decoder (3, 4) selected on val:
  val 0.5006 vs N003 0.5040; pooled 7-match test 0.4761 vs N003 0.4826 → **rejected**; N003 stays
  the neural champion candidate. (N002 decoder study for reference: val 0.4816, test 0.4501.)
- **N005** (N003 + transition-weighted loss ×3 within ±12 frames; single change) launched
  2026-11-06 ~17:45 via `run_chain.sh`, followed by its decoder study (~2.5 h total).

## Stage 1b — Metrica sample data (approved by Tiago 2026-11-06, D-009)

`metrica-sports/sample-data` @ `e706dd5` (no licence file; source acknowledged). Converted by
`scripts/neural/convert_metrica.py` (`src/possession_lab/neural/metrica.py`). Labels are
**algorithmic** from Metrica's human-annotated events (possession events → acting team; dead ball
from out/foul/card/shot-out until the next possession event; challenges ignored). Game 3 is FIFA
EPTS — the same file layout as the challenge ATD — with 11 data-format frame ranges
(substitutions) handled per range.

| game | frames | ball observed | H / A / D share | spells | mean / median spell |
|---|---|---|---|---|---|
| 1 | 145,031 | 60.9 % | 0.33 / 0.29 / 0.38 | 348 | 10.3 / 5.3 s |
| 2 | 141,181 | 59.0 % | 0.34 / 0.31 / 0.35 | 330 | 11.2 / 6.0 s |
| 3 | 143,786 | 67.8 % | 0.38 / 0.34 / 0.28 | 365 | 11.4 / 5.9 s |

Memory note: the laptop's commit charge was near its limit during N005 training (trainer 6.4 GB +
6 workers × 1.9 GB each holding all training matches); future runs use fewer workers.
- **Incident (2026-11-06 ~18:30):** N005's data-loader worker failed at step ~3300 with Windows
  error 1455 (paging file / commit limit exhausted) while the agent ran the EDA, tests and a smoke
  test concurrently; the main process hung and was killed; the chain resumed from step 2000 with
  `num_workers` reduced 6 → 3 (each worker holds all training matches, ~1.9 GB commit). The resume
  changes N005's data stream relative to N003 — noted for the comparison. Rule adopted: no heavy
  concurrent CPU jobs during GPU training; ≤ 3 loader workers on this machine.

## N005 result (2026-11-06 ~20:35) — rejected

N005 = N003 + transition-weighted loss (×3 within ±12 frames). Decoder selected on external val:
team 8 / dead 8 (val spellF1 0.5024 vs N003 0.5040).

| test (frozen) | N005 spellF1 / macroF1 | N003 spellF1 / macroF1 |
|---|---|---|
| 7-match clean | 0.506 / 0.878 | 0.502 / 0.876 |
| 7-match degraded | 0.463 / 0.860 | 0.463 / 0.859 |
| SkillCorner natural (5) | 0.470 / 0.861 | 0.466 / 0.861 |
| decoder-study pooled test | 0.4846 | 0.4826 |

No validation gain and a test gain of +0.002 (needs ≥ 0.02) → **rejected** under the D-007
standard. Boundary weighting did not cure under-segmentation (232 predicted vs 368 true spells);
the bottleneck is evidence about *when* possession changes, not the loss weighting. N003 stays
the neural champion candidate. N006 (EDA-driven features v2) started 20:36.

## N006 result (2026-11-06 ~23:25) — not adopted

N006 = N003 + EDA-driven feature version 2 (team-context token with 37 behaviour features,
direction-aware player features). Single change vs N003 apart from loader workers (3 vs 6).

| step | N006 val spellF1 | N003 val spellF1 |
|---|---|---|
| 2,000 | 0.431 | 0.364 |
| 4,000 | 0.458 | 0.422 |
| 8,000 | 0.462 | 0.464 |
| 16,000 | 0.486 | 0.491 |

| frozen test | N006 spellF1 / macroF1 | N003 spellF1 / macroF1 |
|---|---|---|
| decoder-selected val | 0.5069 (team 8, dead 4) | 0.5040 (team 4, dead 4) |
| 7-match clean | 0.497 / 0.874 | 0.502 / 0.876 |
| 7-match degraded | 0.467 / 0.860 | 0.463 / 0.859 |
| SkillCorner natural (5) | 0.454 / 0.857 | 0.466 / 0.861 |

Interpretation: the engineered features carry real signal — the network reaches a given level
in roughly half the steps — but the 4 M-parameter transformer learns equivalent quantities from
raw positions and velocities given 16k steps, so the final external score does not change
(differences ±0.01, inside the ~0.04 noise band). Not adopted under the D-007 standard. N007
(features + low-detection masking + Metrica) started; the pre-registered primary comparison on
IDSSE 2 + Metrica game 3 (N003, N006, N007) runs after it.

## N007 result and pre-registered primary comparison (2026-11-07 ~02:35)

N007 = N006 + low-detection masking (< 4 tracked objects excluded from the loss) + Metrica game 1
in training (game 2 val, game 3 test). Decoder on its own val: team 4 / dead 12 (val 0.5248; val
set includes metrica_2, so not comparable with N003/N006 val).

Primary test (D-011): pooled IDSSE J03WN1 + J03WR9 + Metrica game 3, each model's own decoder.

| model | clean spellF1 / macroF1 | degraded spellF1 / macroF1 | Metrica 3 clean | IDSSE WN1 / WR9 clean |
|---|---|---|---|---|
| N003 | 0.612 / 0.899 | 0.538 / 0.872 | 0.649 | 0.546 / 0.642 |
| N006 | 0.619 / 0.898 | 0.552 / 0.867 | 0.642 | 0.561 / 0.653 |
| N007 | **0.638 / 0.906** | **0.564 / 0.877** | **0.696** | 0.559 / 0.660 |
| baseline smooth50 | 0.328 / 0.515 | 0.200 / 0.488 | | |

Other N007 tests: 8-match clean 0.519 / degraded 0.473; SkillCorner natural 0.447 (N003 0.466;
artefact-biased, D-011).

Reading: N007 vs N006 (its pre-registered comparison) +0.020 clean / +0.012 degraded — at or below
the ≥0.02 bar; the gain sits on Metrica 3 (in-provider for N007). Vs N003 +0.026 / +0.025. Not
adopted yet: D-012 defers the decision to N008 (seed control) and N009 (no engineered features),
queued 02:40 (N009 then N008, ~2.4 h each, then primary re-scores).

## N008 / N009 and champion decision (2026-11-07 ~08:20)

Primary pooled test (IDSSE WN1 + WR9 + Metrica 3), spellF1 / macroF1:

| model | config | clean | degraded | mean spellF1 | Δ vs N003 |
|---|---|---|---|---|---|
| N003 | v1, no Metrica/masking | 0.612 / 0.899 | 0.538 / 0.872 | 0.575 | — |
| N006 | v2 | 0.619 / 0.897 | 0.552 / 0.867 | 0.585 | +0.010 |
| N007 | v2 + masking + Metrica, seed A | 0.638 / 0.906 | 0.564 / 0.877 | 0.601 | +0.026 |
| N008 | same, seed B | 0.640 / 0.904 | 0.571 / 0.879 | 0.605 | +0.030 |
| N009 | v1 + masking + Metrica | 0.645 / 0.905 | 0.545 / 0.874 | 0.595 | +0.019 |

Findings: (1) Metrica + masking is the main gain (N009 vs N003 +0.019; on clean data +0.033).
(2) The EDA-driven features add robustness when the ball is missing: degraded 0.567 (v2 mean of 2
seeds) vs 0.545 (v1), while clean is level (0.639 vs 0.645) — consistent with the EDA that team
width / forward movement separate possession without the ball. (3) Seed spread is small (0.004).
Under D-012 → D-013: champion = N008 checkpoint. SkillCorner natural: N008 0.447, N009 0.452,
N007 0.447 vs N003 0.466 (D-011 caveat).

## Champion challenge candidate (2026-11-07 ~08:35)

`artifacts/neural/submissions/n008_n007_seed2.csv` — CPU deterministic inference, decoder team 4 /
dead 16 (selected on external val), official validator **Valid** (199,319 rows), 257 possessions of
mean 12.9 s, 20.3 % X (ungraded frames). sha256
`82ff440d3245dec15742f0ef4e69f8a115599a019b6f2ece61409f31fe1b09e2`; independent regeneration
byte-identical. Checkpoint `artifacts/neural/runs/n008_n007_seed2/best.pt` sha256
`28bac0cf6ab423eabea9245cf30d17bf4d219bed95f53f2f05494a1c395fa093`. Not scored against the public
window (correctness checks only; no selection use). Fallbacks: N003 candidate (sha256 346c887a…),
organiser baseline C000/C001. Final submission remains human-triggered.

## Error analysis + individual-player EDA → feature version 3 (2026-11-07 ~10:00–11:30)

Requested by Tiago (individual patterns: roles, marking, pressing, recoveries, dribbles).

- `scripts/neural/error_analysis.py` → `research/neural/ERROR_ANALYSIS.md` (N008, external
  validation only). Matched at IoU 0.5: 47 % of true possessions; taken by the other team 24 %
  (median 1.8 s), merged into a longer prediction 22 %, predicted dead 6 %. Recall of < 2 s
  possessions 0.10 (clean) / 0.06 (degraded); ≥ 20 s 0.81. Metric ceilings on perfect labels:
  drop < 1 / < 2 / < 3 s spells → 0.90 / 0.79 / 0.70; all boundaries 0.2 / 0.5 / 1.0 s late →
  0.92 / 0.81 / 0.65.
- `scripts/neural/eda_individual.py` → `research/neural/EDA_INDIVIDUAL.md` (training matches
  only): ball-touch detection, short vs long turnovers, identity-free roles and marking, lead-lag
  (negative result), carrier-pressure hazard, recovery zone, counter-press, dribbles.
- Feature version 3 (`src/possession_lab/neural/individual_features.py`, 25 context + 4 per-player
  features; 12 ms per 384-frame window) with tests (`tests/test_neural_individual.py`, 5 passed).
- N010 (N008 + v3, single change) pre-registered (D-014) and launched ~11:30 with primary
  re-score and validation error analysis afterwards.

## N010 result (2026-11-07 ~12:50) — passes D-014 stage 1

Primary pooled spellF1 (IDSSE J03WN1 + J03WR9 + Metrica 3), Viterbi, decoder selected on val
(N010: team 2 / dead 24):

| run | clean | degraded | mean | macroF1 clean / degraded | val | SC test |
|---|---|---|---|---|---|---|
| N008 (champion) | 0.640 | 0.571 | 0.605 | 0.904 / 0.879 | 0.5305 | 0.447 |
| N010 (v3 features) | **0.705** | 0.569 | **0.637** | 0.925 / 0.880 | 0.5321 | 0.443 |

Per match clean: J03WN1 0.561 → 0.656, J03WR9 0.662 → 0.727, Metrica 3 0.695 → 0.731. Degraded per
match: 0.499 → 0.488, 0.604 → 0.589, 0.609 → 0.631. The gain is concentrated where the ball is
tracked accurately (touch features); under ball degradation and on SkillCorner it is flat.
Validation error analysis (`ERROR_ANALYSIS_N010.md`): clean val 0.560 vs 0.551, degraded 0.504 vs
0.510; recall of < 2 s possessions 0.11 vs 0.10 (clean) — the short contested spells are still
mostly missed. The test gain is boundary timing: transition F1 within ±0.5 s 0.516 → 0.637
(±1 s 0.673 → 0.742), team accuracy when alive 0.912 → 0.935, while the number of predicted
possessions barely changes (285 → 294 per match vs 412 true) — not recovery of the shortest
spells. Caveat for the challenge: its ATD
ball quality determines how much of the clean-condition gain transfers; degraded results are the
conservative estimate. Seed confirmation N011 launched ~13:00 (D-014 stage 2). `error_analysis.py`
exit 1 was a cosmetic `relative_to` print after the report was written; fixed.

## N011 seed confirmation → feature version 3 adopted (D-015, 2026-11-07 ~15:55)

| run | clean | degraded | mean | macroF1 clean / degraded | val | SC test |
|---|---|---|---|---|---|---|
| N008 (former champion) | 0.640 | 0.571 | 0.605 | 0.904 / 0.879 | 0.5305 | 0.447 |
| N010 (v3, seed 1) | 0.705 | 0.569 | 0.637 | 0.925 / 0.880 | 0.5321 | 0.443 |
| N011 (v3, seed 2) | 0.697 | 0.572 | 0.635 | 0.925 / 0.883 | **0.5399** | 0.453 |

Both seeds clear +0.02 → adopted; checkpoint N011 chosen on validation. N011 val error analysis
(`ERROR_ANALYSIS_N011.md`): clean 0.564, degraded 0.516; < 2 s recall 0.09 / 0.06.

**Champion challenge candidate:** `artifacts/neural/submissions/n011_n010_seed2.csv` — CPU
deterministic inference, decoder team 8 / dead 4 (selected on external val), official validator
**Valid** (199,319 rows), 247 possessions of mean 13.9 s, 20.3 % X. sha256
`234efec3c1db7ec5ac36fb7fb725127f257496e063ceedbece213cf3a0adae5a`; independent regeneration
byte-identical. Checkpoint `artifacts/neural/runs/n011_n010_seed2/best.pt` sha256
`0b026841106258c2691ead0d16e8b050339000b25a1a0b9f1593429ccc7177f4`. Not scored against the public
window. Fallbacks: N008 candidate (sha256 82ff440d…), N003 candidate (346c887a…), organiser
baseline C000/C001. Final submission remains human-triggered.

## Review response: decoder-vs-model diagnostic → N012 transition head (2026-11-07 ~16:15–17:30)

- Tiago's review: treat the 3-match primary test as a **development audit** from now on (it
  decided D-013/D-015); no untouched external match remains (30/30 assigned) → D-016.
- `scripts/neural/diagnose_decoder.py` → `DECODER_DIAGNOSTIC.md` (validation only, cached logp):
  N011 clean < 2 s recall raw argmax 0.13 vs Viterbi 0.09; 56 % of true < 2 s possessions have no
  raw frame on the true team (mean p(true) 0.36); team penalty 0.5 → < 2 s recall 0.13, spellF1
  0.562 vs 0.564 (false spells rise: pred/true spells 0.73 vs 0.66). Decoder is not the
  bottleneck. Ball-unseen team accuracy degraded 0.75; clean 0.37–0.46 (different frame set:
  natural SkillCorner gaps).
- N012 = N011 + possession-change head (model `transition_head`, train `transition_targets`, BCE
  ±2 frames on direct H↔A changes) + time-varying team-switch cost in Viterbi (`transition_gain`,
  0 = old decoder; two-stage val grid in `evaluate.py`). Tests `tests/test_neural_transition.py`
  (5); full suite 52 passed; ruff clean. Backward compatibility verified: N011 challenge CSV
  regenerated with the new code is byte-identical (sha256 234efec3…).
- Degradation already models contiguous ball gaps (bursty dropout), aerial and crowd losses,
  noise, duplicate and wrong-team tracklets (`augment.degrade`) — relevant to the proposed
  visibility experiment.
- N012 launched ~17:30 (chain: train → val-selected decoder incl. gain → SC test → audit →
  error analysis → diagnostic `DECODER_DIAGNOSTIC_N012.md`).

## N012 result (2026-11-07 ~19:40) — passes all four D-016 criteria (seed 1)

| run | val (selected) | audit clean | audit degraded | audit mean | alive/dead clean / deg | SC test |
|---|---|---|---|---|---|---|
| N011 (champion) | 0.5399 | 0.697 | 0.572 | 0.635 | 0.978 / 0.959 | 0.453 |
| N012 (transition head) | **0.5453** | **0.715** | **0.593** | **0.654** | 0.977 / 0.958 | 0.450 |

Decoder selected on val: team 2 / dead 16 / gain 1 (constant-penalty best 0.5416 — the head alone
already beats N011 on val). Audit per match, clean: J03WN1 0.645 → 0.667, J03WR9 0.723 → 0.736,
Metrica 3 0.722 → 0.741; degraded: 0.480 → 0.522, 0.594 → 0.609, 0.643 → 0.649 (every match up;
degraded +0.021, the first real degraded gain since N008). Transition F1 ±0.5 s 0.639 → 0.653
(clean), 0.429 → 0.441 (degraded).

Mechanism (`DECODER_DIAGNOSTIC_N012.md`, val): the head mainly **suppresses false changes** —
false transitions / min 3.45 → 3.26 (clean) and 4.02 → 3.85 (degraded) vs the same decoder with
gain 0, spell precision 0.69 → 0.71 — while < 2 s recall stays ~0.06–0.12 and raw evidence for
short possessions is unchanged (56 % have no frame on the true team). Short contested spells
remain the open problem. Seed confirmation N013 launched ~19:45.

## N013 seed confirmation → transition head adopted (D-017, 2026-11-07 ~22:55)

| run | val | audit clean | audit degraded | audit mean | alive/dead clean / deg | SC test |
|---|---|---|---|---|---|---|
| N011 (former champion) | 0.5399 | 0.697 | 0.572 | 0.635 | 0.978 / 0.959 | 0.453 |
| N012 (head, seed 1) | 0.5453 | 0.715 | 0.593 | 0.654 | 0.977 / 0.958 | 0.450 |
| N013 (head, seed 2) | **0.5473** | **0.733** | 0.578 | **0.655** | 0.977 / 0.959 | 0.452 |

N013 decoder (val-selected): team 2 / dead 8 / gain 4; constant-penalty val 0.5453. Audit per
match clean 0.666 / 0.766 / 0.766, degraded 0.495 / 0.598 / 0.642. Transition F1 ±0.5 s clean
0.693 (N011 0.639). Val diagnostic (`DECODER_DIAGNOSTIC_N013.md`): false transitions / min 3.04
clean, 3.67 degraded; < 2 s recall 0.118 / 0.059; raw evidence for < 2 s spells unchanged.

**Champion challenge candidate:** `artifacts/neural/submissions/n013_n012_seed2.csv` — CPU
deterministic inference, official validator **Valid** (199,319 rows), 255 possessions of mean
13.4 s, 20.3 % X. sha256 `a643f4b8875c245bef65b5c177648649f36afb47feb43f4f918da656509b4319`;
independent regeneration byte-identical. Checkpoint `artifacts/neural/runs/n013_n012_seed2/best.pt`
sha256 `cc84b2501ad18220ceed7d4e785873b0320962723c6cea8fd28d2242d39103e4`. Not scored against the
public window. Fallbacks: N011 (234efec3…), N008 (82ff440d…), N003 (346c887a…), organiser
baseline C000/C001. Final submission remains human-triggered.

## Literature review → N014 football-aware extensions (2026-11-07 ~23:30 – 11-08 ~00:45)

Tiago's review proposed an integrated hierarchical multi-agent graph transformer. Implemented as
an extension of the frozen N013 (D-018), three supervised components:

- `src/possession_lab/neural/pitch_control.py` — feature version 4 (+9 context, +3 per player):
  Spearman (2018) time-to-intercept and soft team control at the ball now / +0.5 s / +1.0 s.
- `model.edge_features` + `edge_bias` — graph-relational spatial attention (per-layer, per-head
  bias from 9 pairwise features), zero-initialised; tested equal to the plain model at init.
- `ball_head` + `train.latent_ball_loss` — Gaussian NLL of the clean source ball (stream emits
  the pre-degradation ball, mirrored with the same symmetry); inference ignores these channels.
- Tests `tests/test_neural_fahgt.py` (7); suite 59 passed; ruff clean. Smoke: 1.6 it/s (N013
  2.1), peak 3.5 GB.
- Deferred with reasons (D-018): six-state ownership (no labels), GRU/Mamba memory, multi-scale.
- N014 launched ~00:45 (≈ 3 h + evaluation).

## N014 result → rejected; N015 ablation launched (2026-11-08)

N014 failed D-018 on all criteria (D-019): val 0.4962 (N013 0.5473); audit clean 0.610 /
degraded 0.477 (N013 0.733 / 0.578); SC test 0.430 (0.452). Cause: the latent-ball Gaussian NLL
is unbounded below and dominated training (total loss −2.57 vs CE ≈ 0.4). Reports:
`ERROR_ANALYSIS_N014.md`, `DECODER_DIAGNOSTIC_N014.md`. N013 remains champion; its submission is
unchanged. N015 (N014 minus the ball head) launched to test pitch control + graph attention.

## N015 neutral → N016 / N017 single-change runs (2026-11-08)

N015 (N014 minus the ball head) val 0.5501, audit 0.732 / 0.577 (mean 0.654), SC 0.458: neutral vs
N013, not adopted (D-020). Added `train.latent_ball_huber` (`ball_loss = "huber"`; bounded, hidden
frames only) with a test; suite 60 passed, ruff clean. Launched N016 (pitch control only) then
N017 (bounded ball head), each a single change vs N013.

## N016 result; N017 resumed (2026-11-08 21:00)

N016 (N013 + pitch control only): val 0.5449, audit 0.734 / 0.593 (mean 0.663), SC 0.455 — not
adopted (D-021). The chain was stopped manually during N017 (~step 9 500); N017 resumed at 21:00
from its step-8 000 checkpoint with the same evaluation chain. N013 remains champion.

## N017 result; literature-review programme closed; seed ensemble E001 (2026-11-09 ~00:15)

N017 (N013 + bounded latent-ball head): val 0.5367, audit 0.692 / 0.557, SC 0.451 — not adopted
(D-022). None of pitch control, graph attention or latent ball met the rule (N014–N017); N013 stays
champion. `evaluate.py` now accepts several `--run` values (probability-averaging ensemble;
`average_log_probs`, tested; suite 61 passed, ruff clean). E001 = N012 + N013 under evaluation.

## E001 result; fine-tuning attempt (2026-11-09 01:20)

E001 (N012 + N013 ensemble): val 0.5485, audit 0.728 / 0.589 (mean 0.658) — not adopted (D-022).
`train.py` gained `init_from` (strict, architecture-checked warm start; fresh optimiser). D-023
pre-registered; warm-start fine-tunes of N013 / N012 started (later withdrawn, see below).

## Fine-tunes withdrawn; N018 larger/longer model prepared (2026-11-09 09:15)

The N013 fine-tune stopped at step 1 200 (machine stop); at Tiago's request both fine-tunes were
deleted (configs, run, log) and the id N018 reassigned (D-023 / D-024). N018 = `n018_large_long`:
d 256, 4+8 layers, 8 heads, 9.72M params, 24 000 steps, batch 4 × accum 4. Measured on the RTX 4050
with a synthetic step: peak 2.6 GB, ~0.27 s per micro-batch (≈ 1.1 s/step → ≈ 7.5 h training +
val). Tiago runs it from the terminal.

## N013 public-window correctness checkpoint (2026-11-11)

At Tiago's request, the official scorer was called for the single existing
`n013_n012_seed2.csv` candidate against the public ground truth, strictly as a
format/alignment correctness checkpoint. It validated 199,319 rows and returned
macroF1 0.8466 and spellF1 0.6345 on the public window. No other candidate was
scored in this checkpoint; these scores were not used to compare, rank, tune,
select, promote, or change any model. The five-minute sample is not representative
of the full match, and these numbers must not be treated as performance evidence.

## N013 external-validation confusion counts (2026-11-11)

Computed from N013's cached validation probabilities and the five external validation
match labels, using its validation-selected decoder and the official scored confusion
definition. Rows are actual H/A/D; columns are predicted H/A/D/X. Clean rows:
H `169002 19630 13870 0`; A `19205 169155 10392 0`; D `5880 7362 317008 0`.
Simulated degraded rows: H `164962 22332 15208 0`; A `22187 164499 12066 0`;
D `7622 9938 312690 0`. This is frame-level validation evidence from external
matches, not from the challenge video or public five-minute labels.


## N018 first-seed external audit (2026-11-11)

At Tiago's request, evaluated the trained `n018_large_long/best.pt` on the D-024
three-match external audit (IDSSE J03WN1, J03WR9 and Metrica 3), clean and the
frozen seed-7 degradation. The existing evaluation log records the validation-selected
decoder team 2 / dead 4 / gain 1 and validation spellF1 0.5568 (constant-penalty
0.5541). The training log records its best in-training validation checkpoint at step
22,000 (0.5251 under the training validation procedure; final step 24,000 = 0.5196).
These validation figures have different decoding procedures and are not interchangeable.

| metric | clean | degraded | mean (spellF1 only) |
|---|---:|---:|---:|
| N018 spellF1 | 0.767258 | 0.616607 | 0.691933 |
| N013 spellF1, recorded comparator | 0.732600 | 0.578133 | 0.655367 |
| N018 macroF1 | 0.931133 | 0.888986 | — |
| N018 alive/dead F1 | 0.976311 | 0.959461 | — |
| N018 transition F1 ±12 frames | 0.701639 | 0.469696 | — |

Match-mean audit scores; clean match spellF1 0.7356 / 0.7650 / 0.8012 and
degraded 0.5499 / 0.6488 / 0.6511 in the order above. First seed passes D-024
numerical gates; second-seed confirmation remains required. N013 remains champion.
No SkillCorner test was run in this evaluation. Results are development evidence
and the bundled change does not isolate the cause of improvement.

Provenance: checkpoint SHA-256
`1b42d0f7eab232f6ba4ea2131d3f0a892f41f750504789dfcd5ce5edb5158a81`; local
report `artifacts/neural/runs/n018_large_long/external_eval_primary.json`. The
recorded decoder was applied with the existing inference, degradation and unchanged
official metric code using `artifacts/local/n018_eval_frozen.py` in an isolated
Python 3.11 / torch 2.7.1+cu128 / numpy 2.2.6 / pandas 2.3.3 environment on RTX 4050.
The original evaluator can reproduce decoder selection via `scripts/neural/evaluate.py`
with `--run n018_large_long --test idsse_J03WN1 idsse_J03WR9 metrica_3 --tag _primary`.
An initial attempt stopped before any metric was produced because pandas was absent;
the repeated evaluator was stopped before its decoder search completed and the
previously logged decoder was reused. Missing-player empty-mean warnings are converted
to zero by the existing feature code; the frozen audit suppressed those repeated
warnings. No challenge-game data, public GT or video was used in this audit.
