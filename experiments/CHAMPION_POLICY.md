# Champion–challenger governance

- **Initial champion:** official `baseline_smooth50` implementation and its validated CSV. The organisers report private full-match `spellF1=0.2433`; participants cannot independently measure that value.
- **Challenge accepted:** new model passes all quality gates and official format validator, has full reproduction recipe, adheres to data rules, and has credible independent evidence (prefer labelled OTHER-game performance) or carefully recorded theory/robustness evidence.
- **Inconclusive:** where only supplied-match unlabeled diagnostics / synthetic tests are available, do not claim superior private performance. Keep as challenger until defensible review; promotion must state its uncertainty.
- **Rejection:** invalid CSV, manual correction, official scorer change, GT training/tuning, data leakage, instability, or failures without resolution.
- **Freeze:** commit selected model config and lockfile, generate final file twice, compare SHA256, run official validator on **exact** team-named CSV, review public repo, write technical report.
- **Rollback:** maintain prior working champion under local ignored files and an immutable Git commit/tag referencing code/config (not predictions or licensed data).

**No auto-submit or auto-publish.** Agentic autonomy stops before external release and the final form upload.
