# Decision log

## D-000 — Protect source of truth

**Decision:** Preserve official baseline, scorer, original data parsing and upstream documents; implement all experiments in namespaced addition with an audit trail.

**Reason:** Avoid inadvertent benchmark drift and preserve reproducible fallback.

## D-001 — Public GT firewall

**Decision:** Five-minute public GT may verify pipeline correctness but cannot drive model training, hyperparameter choice, experiment keep/discard, or champion promotion.

**Reason:** Current organisers' rules expressly forbid those uses, and only 17 possessions are publicly labelled.

## D-002 — Local experiment metadata

**Decision:** Initial runner logs only non-sensitive metadata and output checksums to ignored local directory. Optional MLflow requires local-only config and explicit review of logged fields.

**Reason:** Keep setup functional without external services or leakage.

## D-003 — External labelled data: IDSSE/DFL open data (2026-11-06)

**Decision:** Use the IDSSE / DFL open dataset (Bassek et al. 2025, CC BY 4.0, figshare 10.6084/m9.figshare.28196177.v1) as the primary Tier-A source. Labels are the provider's per-frame `BallStatus`/`BallPossession` (H/A/D) — other games, provider-produced, documented. Frozen split by match: train `J03WMX, J03WOH, J03WOY, J03WPY`; val `J03WQQ`; test `J03WN1, J03WR9`. SkillCorner open data (MIT) is the planned cross-provider holdout. Metrica sample data has no licence file → pending Tiago.

**Reason:** Only eligible source found with continuous 25 Hz tracking *and* frame-level alive/possession labels; licence verified via the figshare API. Satisfies upstream SUBMISSION.md's training-data disclosure requirement (source, labeller, amount).

## D-004 — Domain randomisation is not calibrated on the supplied game (2026-11-06)

**Decision:** Degradation ranges (missing ball up to 65 %, bursty tracklet loss, duplicates, wrong-team tracklets, noise) are broad physical priors. No statistic computed from the supplied ATD/video/GT is used to set them. The organiser's *published* statement that the ball is missing on a large share of frames only motivates the upper range.

**Reason:** The 2026-11-03 clarification forbids fitting/calibrating on the supplied game; repo governance applies this to all supplied data, not only GT and video. Escalation item for Tiago: confirm whether published `DATA.md` aggregate figures may inform augmentation ranges.

## D-005 — Reproducible inference (2026-11-06)

**Decision:** Challenge inference runs on CPU in float32 with deterministic algorithms and log-probabilities quantised to 1e-3 before Viterbi decoding.

**Reason:** RULES.md requires the repository to regenerate *the same file*; GPU fp16 or last-bit differences across CPUs must not flip labels.

## D-006 — Decoder selection rule and primary external benchmark (2026-11-06)

**Decision:** Viterbi switch penalties are grid-searched (team 2–48, dead 1–24 nats) on the mean external-*validation* spellF1 over the clean and degraded conditions, then frozen before the external test split is scored once. SkillCorner's natural broadcast missingness is the primary target-like benchmark; IDSSE-degraded is secondary.

**Reason:** Fixed before any external-test result existed. Selecting on the degraded condition alone would double-corrupt SkillCorner (synthetic corruption on top of natural missingness).

## D-007 — Decoder-study adoption rule (pre-registered 2026-11-06, before results)

**Decision:** `scripts/neural/decoder_study.py` compares the production symmetric decoder with an asymmetric one (separate dead-ball entry/exit penalties), each selected on external validation only. The asymmetric decoder replaces production only if it is better on external val **and** on the frozen external test pooled over all 7 matches, and the test gain is ≥ 0.02 spellF1; otherwise production is kept. A temperature family was dropped as provably redundant (it only rescales the costs).

**Reason:** Avoid adopting decoder complexity on noise; every model under-segments, so the decoder is the cheapest lever, but 7 test matches limit resolution.

## D-008 — Ensemble adoption rule (pre-registered 2026-11-06, before results)

**Decision:** A probability-averaging ensemble of runs sharing the frozen splits (first: N002 + N003) is adopted over N003 alone only if it is better on external val and ≥ 0.02 spellF1 better on the pooled 7-match external test, with its decoder selected on val. Otherwise N003 alone stays the neural champion candidate.

**Reason:** Same evidence standard as D-007; an ensemble doubles inference cost and checkpoint footprint.

## D-009 — Metrica sample data approved; analysis-first feature engineering (2026-11-06)

**Decision:** Tiago approved use of the Metrica Sports sample data (github.com/metrica-sports/sample-data; no licence file — README asks for responsible use and source acknowledgement, which the technical document will give). He also directed that model work be preceded by exploratory data analysis of possession behaviour (player velocity towards the ball, sprints, direction of movement, nearby opponents, team shape) and explicit feature engineering from the patterns found.

**Constraint kept:** all EDA and feature design use external matches only (IDSSE, SkillCorner, Metrica sample). The supplied game remains an inference input; no statistic computed from it informs feature design, thresholds or parameters (2026-11-03 clarification).

## D-010 — Metrica split and EDA scope (frozen 2026-11-06, before any analysis)

**Decision:** Metrica sample games: game 1 train, game 2 validation, game 3 test (game 3 is FIFA EPTS, the challenge's file layout). Exploratory analysis for feature design uses training-split matches only (IDSSE 4, SkillCorner 12, Metrica game 1).

**Reason:** Feature design must not peek at matches used to select or judge models.

## D-011 — SkillCorner test is artefact-biased for dead ball; primary test for cleaned models (2026-11-06)

**Decision:** For candidates trained with low-detection masking (N007+), the primary external test is the pooled IDSSE + Metrica game 3 split; SkillCorner test results are reported with the caveat that 37 % of SkillCorner frames show fewer than 4 tracked objects and 92 % of those are dead-ball labels, which rewards learning "nothing visible ⇒ dead" — a cue that does not exist with the challenge's fixed tactical camera.

**Reason:** EDA finding 7; measured with `mask_low_detection` before any N007 result.

## D-012 — N007 adoption deferred to a seed control and a feature ablation (pre-registered 2026-11-07, before N008/N009 results)

**Context:** N007 (features v2 + low-detection masking + Metrica) leads on the primary pooled
IDSSE 2 + Metrica game 3 test (clean 0.638 / degraded 0.564 vs N003 0.612 / 0.538), but the
pre-registered comparison against N006 clears ≥0.02 only on clean (+0.020) not degraded (+0.012),
the gain is concentrated on one match (Metrica 3, the provider N007 also trained on), and
SkillCorner natural is lower (0.447 vs 0.466, artefact-biased per D-011).

**Decision:** before changing the champion, train N008 (N007, second seed) and N009 (N007 with
feature version 1). Rule, fixed now: the champion becomes the configuration (v1 or v2 with Metrica
+ masking) whose **mean primary pooled test spellF1 over clean and degraded** beats N003 by ≥0.02
in **every** available seed of that configuration; between v1 and v2, prefer v1 (simpler) unless v2
is ≥0.02 better. If no configuration meets this, N003 remains the champion. The challenge
submission candidate is regenerated only for the chosen champion.

## D-013 — Neural champion: features v2 + low-detection masking + Metrica (N008 checkpoint) (2026-11-07)

**Applied rule D-012** (pre-registered before N008/N009). Primary pooled IDSSE 2 + Metrica 3 test,
mean of clean and degraded spellF1: N003 0.5754; v2 config N007 0.6009 (+0.026), N008 0.6053
(+0.030) — both seeds ≥ 0.02 → qualifies; v1 config N009 0.5948 (+0.019) → does not qualify.
Champion configuration = v2 + masking + Metrica; between its two seeds the checkpoint is chosen on
the shared external validation set (N008 0.5305 > N007 0.5248), not on test.

**Caveats recorded:** the gain over N003 is largest on Metrica game 3 (a provider in the
champion's training data); IDSSE gains are +0.01–0.04 per match; SkillCorner natural test is
lower (0.447 vs 0.466), attributed to masking the broadcast artefact (D-011). N003 remains the
documented neural fallback and C000/C001 the organiser-baseline fallback.

## D-014 — Error-analysis-driven individual features; N010 adoption rule (pre-registered 2026-11-07, before N010 results)

**Context:** `research/neural/ERROR_ANALYSIS.md` (N008 on external validation only): 25 % of
true possessions last < 2 s and are recalled at 6–10 %; 22 % are lost by merging across a missed
short opponent possession. Metric sensitivity on perfect labels: dropping < 2 s spells caps
spellF1 at 0.79; boundaries 0.5 s late cap it at 0.81. Tiago asked (2026-11-07) for individual-
player pattern analysis (roles, marking, pressing, recoveries, dribbles); done on training
matches only (`research/neural/EDA_INDIVIDUAL.md`) and encoded as feature version 3.

**Rule (fixed now):** adopt N010 over N008 if its mean (clean, degraded) primary pooled spellF1
is ≥ 0.02 higher **and** its decoder-selected validation spellF1 (same validation set) is higher;
if so, confirm with a second seed (N011) before regenerating the submission — both seeds must
clear +0.02. Also reported, not used for adoption: recall of < 2 s possessions on validation
(error analysis rerun), SkillCorner test.

**Risk noted:** the organiser's public window (17 possessions, mean 14.2 s; not used for tuning)
suggests the challenge labels may contain fewer very short possessions than the external
providers' labels; gains concentrated on < 2 s spells may not transfer.

## D-015 — Neural champion: feature version 3 (N011 checkpoint) (2026-11-07)

**Applied rule D-014** (pre-registered before N010/N011). Primary pooled IDSSE 2 + Metrica 3 test,
mean of clean and degraded spellF1 vs N008 0.6053: N010 0.6371 (+0.032), N011 0.6345 (+0.029) —
both seeds ≥ 0.02, and N010 validation 0.5321 > N008 0.5305 → feature version 3 adopted. Between
the two seeds the checkpoint is chosen on the shared external validation set (N011 0.5399 > N010
0.5321), not on test.

**Caveats recorded:** the gain is clean-condition boundary timing (transition F1 ±0.5 s 0.516 →
0.637/0.639); degraded is flat (0.569 / 0.572 vs 0.571) and SkillCorner test is unchanged (0.443 /
0.453 vs 0.447); recall of < 2 s possessions on validation is unchanged (0.09–0.11). Transfer to
the challenge depends on its ball-tracking quality. N008 becomes the documented neural fallback;
N003 and C000/C001 remain further fallbacks.

## D-016 — Test set becomes a development audit; N012 transition head and its adoption rule (pre-registered 2026-11-07, before N012 results)

**Test-set status (review by Tiago, 2026-11-07).** The three-match primary test (IDSSE J03WN1,
J03WR9, Metrica 3) has decided adoptions D-013 and D-015, so it has influenced development. From
now on it is reported as a **development audit**, not an untouched final test. All 30 external
matches (IDSSE 7, Metrica 3, SkillCorner 20) are already assigned (train 17 / val 5 / test 8);
no untouched external match remains until SoccerNet (or another approved source)
clears rights/provenance. Results on the audit are reported with that caveat and per match.

**Diagnostic behind N012** (`research/neural/DECODER_DIAGNOSTIC.md`, validation only): raw argmax
recall of true < 2 s possessions is 0.13 (N011 clean) vs 0.09 after Viterbi; 56–62 % of those
possessions have no raw frame on the true team; lowering the team penalty to 0.5 lifts < 2 s
recall to 0.13 but spellF1 stays 0.562–0.564 because false spells rise. → the decoder is not the
bottleneck; the network lacks change evidence.

**N012 (single change vs N011, same seed):** auxiliary possession-change head — BCE (weight 1) on
direct H↔A changes taken from the external labels, ±2 frames; X frames excluded; no contact or
touch labels are manufactured. Decoder: team-switch cost per frame `max(0, team − gain·logit)`,
gain ∈ {0.5, 1, 2, 4} searched on validation after the standard grid (gain 0 = old decoder, so
the decoder cannot lose on validation by construction).

**Rule (fixed now):** adopt the transition head only if **both** N012 and a second seed N013 meet
all of: (1) decoder-selected validation spellF1 > N011's 0.5399; (2) audit mean(clean, degraded)
≥ N011 + 0.010 (≥ 0.6445); (3) audit degraded ≥ N011 − 0.005 (≥ 0.5673); (4) audit alive/dead F1
not lower than N011 by > 0.005 in either condition. Reported, not used for adoption: < 2 s / 2–5 s
/ ≥ 5 s recall, transition P/R ±0.5 s, false transitions per minute, D F1, ball seen vs unseen,
per match and per provider (incl. SkillCorner test).

## D-017 — Neural champion: possession-change head (N013 checkpoint) (2026-11-07)

**Applied rule D-016** (pre-registered before N012/N013). Both seeds meet all four criteria vs N011
(val 0.5399; audit mean 0.6345, degraded 0.5723, alive/dead 0.9780 / 0.9585):

| run | val | audit mean | audit degraded | alive/dead clean / deg |
|---|---|---|---|---|
| N012 | 0.5453 | 0.6540 | 0.5934 | 0.9769 / 0.9582 |
| N013 | 0.5473 | 0.6554 | 0.5781 | 0.9772 / 0.9589 |

→ transition head adopted. Checkpoint chosen on validation: N013 (0.5473 > 0.5453), decoder team 2
/ dead 8 / gain 4 (selected on val). **Caveats:** the audit set has influenced development
(D-016), so +0.02 is a development estimate; seed spread is now larger on degraded (N012 +0.021,
N013 +0.006 vs N011); the mechanism is fewer false changes, not recovery of < 2 s possessions
(still ~0.06–0.12 recall on val). N011 becomes the documented neural fallback.

## D-018 — N014 football-aware extensions; adoption rule (pre-registered 2026-11-07, before N014 results)

**Context:** Tiago's literature review (2026-11-07) proposed an integrated hierarchical multi-agent
graph transformer (pitch control, graph interaction, latent ball, match memory, hierarchical
ownership states) as an extension of the frozen N013. Implemented now — the components with direct
supervision or published physics and a measured target (N013 clean 0.733 vs degraded 0.578):

1. **Probabilistic pitch control** (feature version 4, `pitch_control.py`): Spearman (2018)
   time-to-intercept with published constants (reaction 0.7 s, 5 m/s, σ 0.45 s; not fitted) to the
   ball now and extrapolated +0.5 / +1 s; soft team control and per-player control share.
2. **Graph-relational spatial attention** (`edge_bias`): pairwise ball/player relations → MLP →
   additive attention bias per layer and head, zero-initialised (exactly N013's architecture at
   start; equivalence tested).
3. **Latent-ball head** (`ball_head`, Kim et al. 2023 idea): Gaussian NLL of the *clean source*
   ball position (external tracking only), weight 1 where augmentation hid the input ball, 0.2
   where visible, λ 0.5. Auxiliary only — no reconstructed ball is fed back as an observation.

**Deferred, with reasons:** six-state hierarchical ownership (no supervision exists for
"contested / loose"; the possession-change head is the supervised part); persistent GRU/Mamba
memory (windows already give 15.4 s bidirectional context; no evidence that longer memory is
missing; Mamba has no reliable Windows build); multi-scale temporal attention (already full 25 Hz
over the whole window). Weights λ and constants fixed a priori, not tuned on the audit.

**Rule (fixed now):** N014 is a bundled change (three components). Adopt only if **both** N014 and
a second seed N015 meet all of, vs N013: (1) decoder-selected val > 0.5473; (2) audit
mean(clean, degraded) ≥ 0.6654 (N013 + 0.010); (3) audit degraded ≥ 0.5731 (N013 − 0.005);
(4) audit alive/dead F1 not lower by > 0.005 in either condition. Attribution: leave-one-component-
out ablations afterwards as time allows (reported, not used for adoption). If N014 fails, the
ablations identify the harmful component before any retry.

## D-019 — N014 rejected; leave-one-out ablation N015 (2026-11-08, after N014 results)

**Evidence (N014 vs N013):** val 0.4962 vs 0.5473; audit clean 0.610 vs 0.733, degraded 0.477
vs 0.578 (mean 0.543 vs 0.655); alive/dead F1 0.972 / 0.947 vs 0.977 / 0.959; SkillCorner test
0.430 vs 0.452. Fails all four D-018 criteria; the gap (−0.05 val) is far beyond the observed
seed spread (N012 vs N013: 0.002). N015-as-seed-confirmation is moot. N013 stays champion.

**Diagnosis:** the latent-ball Gaussian NLL is unbounded below. Where the ball is visible it is
in the input, so the head copies it and drives log-variance to the clamp; the total training loss
fell to −2.57 while classification CE sits near +0.4, i.e. the auxiliary term dominated the
shared trunk's gradients from the start (val behind N013 from step 4000; raw-argmax val spellF1
0.372 vs 0.489; team accuracy with the ball unseen 0.34 vs 0.40 — the head did not even help the
missing-ball case). Pitch control and graph attention cannot be judged from N014.

**Next (fixed now, before results):** N015 = N014 without the ball head (pitch control + edge
bias + transition head; otherwise identical, same seed). If N015 meets the D-018 criteria vs
N013, a second seed (N016) is required before adoption. If it fails, N016 = N013 + pitch control
only, to separate the two. Any ball-head retry must use a bounded loss (e.g. Huber on the mean,
hidden frames only, with its weight chosen so it stays below CE) and is a separate experiment.

## D-020 — N015 neutral, not adopted; single-change runs N016 / N017 (2026-11-08, before their results)

**Evidence (N015 = N013 + pitch control + graph attention, vs N013):** val 0.5501 vs 0.5473
(passes 1); audit clean 0.732 vs 0.733, degraded 0.577 vs 0.578, mean 0.654 < 0.6654 (fails 2);
degraded ≥ 0.5731 (passes 3); alive/dead −0.002 clean / +0.000 degraded (passes 4); SC test
0.458 vs 0.452; transition F1 ±0.5 s 0.665 vs 0.693 (clean). Not adopted. The two components
are neutral: the val gain is within seed spread, and v3 features (nearest-player distances,
pressure, marking) already carry most of the reachability signal.

**Next (fixed now):** two single changes vs N013, same seed and data:
- **N016** = N013 + pitch control only (feature version 4) — the D-019 follow-up.
- **N017** = N013 + latent-ball head with the bounded loss (`ball_loss = "huber"`: Huber on the
  mean, β 0.05 scaled ≈ 2.6 m, input-hidden frames only, weight 0.5; constants fixed a priori).
  Target: the degraded gap (0.733 vs 0.578).

Each is judged by the D-018 criteria vs N013 and needs a second seed before adoption. If both
pass, the combination is a further run, not assumed.

## D-021 — N016 (pitch control only) not adopted (2026-11-08)

**Evidence vs N013:** val 0.5449 vs 0.5473 (fails 1); audit clean 0.734 vs 0.733, degraded 0.593
vs 0.578, mean 0.663 < 0.6654 (fails 2); alive/dead −0.003 / −0.000 (passes 4); SC test 0.455 vs
0.452 (clean), 0.430 vs 0.437 (degraded). The degraded-audit gain is not used for selection (D-016:
audit is a development check) and is not mirrored on val or on SkillCorner degraded. Conclusion with
N015 (D-020): published-constant pitch control and graph attention add no measurable value over the
v3 features. Both stay available as options; neither enters the champion. N017 (bounded ball head)
continues; it was interrupted by a manual stop at ~step 9 500 and resumed from its step-8 000
checkpoint (`--resume`; identical config).

## D-022 — N017 not adopted; seed ensemble N012 + N013 and its rule (2026-11-09, before ensemble results)

**N017 evidence (N013 + bounded latent-ball head, vs N013):** val 0.5367 vs 0.5473; audit clean
0.692 vs 0.733, degraded 0.557 vs 0.578; SC test 0.451 vs 0.452. Fails 1–3; not adopted. Training
was stopped manually at ~step 9 500 and resumed from the step-8 000 checkpoint (val at 8 000:
0.4805 vs N013 0.4798 — level); it then lagged N013 at every later evaluation, so the stop is not a
plausible cause. Even bounded and restricted to hidden frames, ball reconstruction competes with the
possession objective rather than helping it.

**Summary of the literature-review programme (D-018 – D-022):** pitch control, graph-relational
attention and a latent-ball head were each tested alone or together against N013; none met the
adoption rule. N013 stays champion. The components remain in the code as off-by-default options.

**Next (rule fixed now):** probability-averaging ensemble of the two seeds of the champion
configuration, N012 + N013 (`evaluate.py --run n012_transition_head n013_n012_seed2`; class and
possession-change probabilities averaged; decoder re-selected on val by the same grid). Adopt over
N013 only if it meets the D-018 criteria: val > 0.5473; audit mean ≥ 0.6654; degraded ≥ 0.5731;
alive/dead not lower by > 0.005. Cost: 2× inference (offline, acceptable). The 2026-11-06 ensemble
(N002 + N003, D-008) mixed architectures and failed; this one averages seeds of one configuration.

## D-023 — Fine-tuning both champion-configuration models; rule (pre-registered 2026-11-09, before results)

**Context:** Tiago (2026-11-09) asked to improve both models (N012, N013) further and, if the
football-aware components do not help, to keep tuning in other directions without overfitting.
Evidence for under-training: both N012 and N013 reached their best in-training val at the final
step (16 000; val still rising: N013 0.5138 → 0.5162, N012 0.5137 → 0.5156 over the last 2 000).

**Runs (withdrawn):** two warm-start fine-tunes of the N012 / N013 checkpoints were configured
(6 000 steps, lr 1.5e-4). The first was interrupted by a machine stop at step 1 200 (val@1000 0.5130
vs parent 0.5162, no result used); at Tiago's request both were deleted (configs, run, log) and the
run id N018 reassigned to the larger model of D-024. `train.init_from` remains available.

**Overfitting guards:** checkpoint chosen only by in-training external val; decoder chosen only on
val; the audit is read once per run and never used to choose steps, lr or checkpoints; challenge
data never used. The lr and length are fixed now and will not be re-tuned on the audit.

**Rule:** a fine-tune replaces N013 only if it meets the D-018 criteria vs N013 (val > 0.5473; audit
mean ≥ 0.6654; degraded ≥ 0.5731; alive/dead not lower by > 0.005). If fine-tuning does not help, the next
directions in order are: longer from-scratch training (24 000 steps, same config); then stronger
regularisation (dropout 0.15, label smoothing 0.05) — each a single change, pre-registered in turn.

**E001 result (2026-11-09, D-022 rule):** ensemble N012 + N013, decoder team 2 / dead 8 / gain 4
re-selected on val: val 0.5485 (passes 1); audit clean 0.728, degraded 0.589, mean 0.658 < 0.6654
(fails 2); degraded passes; alive/dead +0.001 / +0.000. SC test 0.455 / 0.436 (N013 0.452 / 0.437).
**Not adopted**; small consistent gain, within the +0.010 bar. Kept as a candidate member type for
D-023 (ensemble of the fine-tunes).

## D-024 — N018 larger and longer model; rule (pre-registered 2026-11-09, before results)

**Context:** Tiago (2026-11-09 09:10) asked for more parameters and longer training, run by him from
the terminal. It replaces the withdrawn fine-tunes (D-023). Both champion-config runs peaked at
their last step and input add-ons did not help (D-018 – D-022).

**Run N018** (`configs/neural/n018_large_long.toml`): N013 configuration with d_model 256, 4 spatial + 8 temporal layers, 8 heads (9.72M
parameters vs 4.15M; measured peak 2.6 GB at batch 4 on the RTX 4050), 24 000 steps (1.5×), batch 4
× accum 4 (same 16 windows per step), lr 3e-4 (from 4e-4, usual for a wider model), warmup 500, val
every 1 000 steps (also limits loss from interruptions; resume with `--resume`). Features v3,
transition head, data, splits and decoder grid unchanged. Bundled change (size + length + lr).
Settings fixed now; not re-tuned on the audit.

**Overfitting guards:** best.pt chosen by in-training external val only; decoder on val only; the
audit is read once; the train-loss vs val curve is reported (rising val loss / falling val spellF1
late in training means the larger model overfits the 17 training matches).

**Rule:** adopt over N013 if it meets the D-018 criteria (val > 0.5473; audit mean ≥ 0.6654;
degraded ≥ 0.5731; alive/dead not lower by > 0.005), then confirm with a second seed.


## D-025 — N018 first-seed result; confirmation pending (2026-11-11)

D-024's first seed meets its numerical gates: external validation spellF1 0.5568
> 0.5473; audit mean 0.6919 ≥ 0.6654; degraded 0.6166 ≥ 0.5731; alive/dead
F1 changes versus N013 are −0.0009 clean and +0.0006 degraded, both within
the −0.005 tolerance. Confirmation with a second seed remains required under
D-024; N013 remains champion. N018 changes capacity, steps and learning rate
together, so the comparison cannot attribute the result to any one change.
See `neural/STATUS.md` for data, decoder, environment, and evaluation provenance.
