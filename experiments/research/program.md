# Autonomous Research Program — Karpathy-inspired, competition-safe

Follow the latest organizer clarification and the competition rules. All training, fitting, calibration, tuning and selection must use eligible external data. Never compare output against the supplied video, use manual supplied-game data or upload supplied data to AI services. Fixed local inference and automatic checks are permitted; supplied-game self-supervised or pseudo-label fitting is not.

**Current campaign status:** SoccerNet preparation has resumed. Downloads remain Tiago's responsibility. Fitting must wait until approved payloads, verified rights and provenance, and a frozen plan are in place. See [the neural campaign status log](neural/STATUS.md).

Read `../../official_documentation/RULES.md`, `../../official_documentation/EVALUATION.md`, and `../README.md` before research. This is an adaptation of Karpathy's research loop, not a claim that this hackathon can repeatedly optimise on its hidden objective.

## Objective

Increase expected private full-match `spellF1`, then `macroF1` tie-break, within the rule-compliant data and experiment budget; output a validated reproducible 199,319-row CSV with a documented best champion. **No public-GT tuning**.

## Edit boundaries

**Allowed after review:** our `src/possession_lab`, `configs/models`, tests, docs, new scripts, synthetic fixtures, research registry and decision logs. **Read-only:** official `evaluation/`, `official_documentation/starter/`, `games.py`, `official_documentation/RULES.md`, `DATA.md`, `SUBMISSION.md`, `EVALUATION.md`, public GT, and licensed `data/` (algorithms may read per terms, never manually correct). Mark score-metric implementation as immutable. Do not edit ignore rules to expose private paths.

## One cycle

1. Read current champion and failed hypotheses from `registry.md` and `decisions.md`.
2. Propose ONE falsifiable improvement; pre-register an `experiments/HYPOTHESIS_TEMPLATE.md` card, including evidence type and budget.
3. Create branch/worktree if available; implement in allowed scope; keep deterministic settings.
4. Run lint, static typing and synthetic tests; inspect privacy scan and relevant diffs.
5. Run configured candidate generator with authorised local data access. Execute **official format validator**. Record minimal run manifest under ignored `artifacts/local`.
6. Evaluate on permitted OTHER-match ground truth, if actually available; otherwise use independent synthetic stress tests. Never select parameters or keep/discard candidates using supplied-game observations, video or GT. Supplied-input automatic checks are engineering checks only; no prediction-versus-video review.
7. Record measurable outcomes, run time, failures, statistical caveats and reproducibility. Make explicit `supported / inconclusive / rejected` decision.
8. If supported by strong independent evidence, propose champion promotion for review; preserve prior champion.
9. Iterate until `max_experiments`, budget or deadline. If no defensible evaluation signal, prefer deeper invariants and data reasoning over speculative brute-force search.

## Stop conditions

- Budget exhausted (initial batch: 3 experiments or 90 minutes; configurable).
- Rule/licence ambiguity, unapproved external data egress or unknown training provenance.
- Official scorer or previously validated champion modified unexpectedly.
- No trustworthy objective signal for further selection; escalate scientifically.
- Local environment failing or research past freeze date; shift to release QA.

## Do not misrepresent evidence

The supplied five-minute public GT is a **single correctness smoke test** only. The model may not use it to pick a smoothing window, choose priors, tune an HSMM transition matrix, or guide automatic experiment acceptance. Synthetic F1 measures performance on synthetic rules; unlabeled real-match plausibility is *not* a measurement of official spellF1.

## Handoff format

For each agent cycle: hypothesis, files changed, Git commit/config/seed, commands and actual results, evidence tier, decision, remaining uncertainty, privacy review, next candidate, submission champion/fallback status.
