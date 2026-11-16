# Experiments — controlled autonomous football research

Recorded experiment scores and decisions are summarized in the [experiment dashboard](dashboard.html).

The latest organizer clarification controls all experiments. Training, fitting, calibration, tuning and selection use eligible external data only. Supplied inputs support fixed local inference and automatic checks only; never compare output with the supplied video, make manual fixture data, or upload supplied data to AI services. See the [research program](research/program.md) for current campaign status and preparation gates.

**Goal:** develop eligible algorithms likely to generalise to high private full-match `spellF1` without tuning on the provided five-minute GT. Read the [program](research/program.md), [official scoring rules](../official_documentation/EVALUATION.md) and [model backlog](MODEL_PORTFOLIO.md).

## Research experiment contract

Each experiment starts with a falsifiable hypothesis, a named baseline, a proposed causal mechanism, a config file, expected diagnostics, computational budget and independent evidence plan. Candidate implementations reside in `src/possession_lab`, not modified scorer files.

1. **Design:** select problem and hypothesis from `research/registry.md`; define permitted evaluation evidence and stopping conditions.
2. **Implement:** branch/commit, implement and test on synthetic fixtures, integrate via a versioned config.
3. **Run:** generate candidate using deterministic seed, log metadata locally, run organiser's **format validator**.
4. **Assess:** use permitted external-match labelled validation where available; otherwise run synthetic robustness/ablation and structural plausibility diagnostics; explicitly mark evidence quality.
5. **Decide:** support/inconclusive/reject; champion promotions require reproducibility plus independent evidence or documented expert review. **Never** select via five-minute public score.
6. **Record:** add entry to `research/decisions.md` and update registry. Preserve failure modes and negative results.

## Why `spellF1` changes the problem

Possessions are contiguous `H`/`A` spells matched one-to-one with temporal IoU. A frame-perfect nearest-player classifier can still fragment one true spell into many predicted spells. Two mechanisms are potentially valuable: preserving possession through occlusion and correctly recognising controlled handovers / stoppages. The missing ball is not a reliable proxy for dead ball. `X` is forward-filled for the main metric and is penalised in macroF1.

## Permitted evidence tiers

| Tier | Evidence | Valid conclusions |
|---|---|---|
| Tier A | Independent other-game labelled possession data, licensed and eligible | Cross-match model selection, generalisation with caveats |
| Tier B | Synthetic event generators and domain perturbations | Invariant correctness and failure robustness only |
| Tier C | Automatic local checks on supplied inputs | Format, alignment, determinism and invariants only; no tuning, output-versus-video review or ground-truth performance claim |
| Restricted | Official supplied five-minute public GT | **One-off correctness/alignment smoke**; NO tuning or ranking |
| Unavailable | Official private full-match GT | No participant access or optimisation |

## Runner usage

```powershell
uv run python scripts/experiment.py --name foundation-smoke --config configs/models/baseline_smooth50.toml
```

The runner records minimal run details under ignored `artifacts/local/runs/` and validates full-length file format. It does **not** automatically score against public GT or promote champions. Future model plugins must satisfy this identical contract.

## Documentation

- [Model portfolio](MODEL_PORTFOLIO.md)
- [Hypothesis template](HYPOTHESIS_TEMPLATE.md)
- [Experiment card template](EXPERIMENT_CARD_TEMPLATE.md)
- [Champion promotion](CHAMPION_POLICY.md)
- [Research program](research/program.md)

Before starting a new research batch, execute `python scripts/check_upstream.py --check`. A mismatch is a hard stop pending documented upstream review.
