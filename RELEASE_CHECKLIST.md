# Public release checklist

This checkout is a work in progress; finalize the team-named deliverables before submission.
Complete these checks before changing repository visibility or submitting.

1. Freeze the submitted model and document its exact configuration, checkpoint,
   decoder settings, dependency versions, and commands. The recorded N018
   decoder is in `configs/neural/n018_decoder.json`; the neural inference
   checkpoint is in `models/n018/best.pt`; the team confirmed it may be shared
   for the competition. Verify the packaged file's hash and clean-clone access.
2. Finalize the team-named technical document, including labelled training
   datasets, their provenance and terms, and the exact frozen inference path.
   The public-facing draft is `TECHNICAL_DOCUMENT.md`; finalize and rename it
   for the team. Record the applicable external-data permissions and terms.
3. Review the proposed Git index and all existing history, including files
   removed in later commits. Ignore rules do not remove tracked files or erase
   history. Review official baseline outputs and the public ground truth against
   the applicable licence and publication exclusions before retaining them.
4. Review documentation, configuration, notebooks, metadata and Git history for
   secrets, personal paths, licensed inputs, derived data, and private results.
   Run `python scripts/privacy_scan.py` after preparing the index; it is a
   path-based heuristic, not a content or history scan.
5. On a clean clone with only authorized inputs and the released artifacts, run
   the documented setup, synthetic tests, frozen inference, and official format
   validator. Verify the exact final CSV has 199,319 ordered `frame,label` rows
   and can be regenerated. Never compare output against the supplied video or
   use its labels for model selection. Two local CUDA regenerations on
   2026-11-16 matched the recorded output hash and passed the validator. A
   separate clean checkout also reproduced that hash with authorized inputs.
6. Review the upstream competition documentation for changes, resolve any
   discrepancies with the controlling organizer clarification, and perform the
   final publication and submission only when authorized.

`origin` points to the team's private repository and `upstream` to the organizer.
Verify destination and visibility before any push. A limited history scan found no common private-key or API-key
signatures, but this is not a comprehensive secret or licence audit.

Local-only instructions and older setup/research paths have been removed from
the current Git index while their existing working copies were preserved. Their
earlier commits remain in history, including personal paths in historical setup
documents. Review the intended public history separately; `.gitignore` and staged
deletions do not erase it. The relocated research files, model and other new
deliverables must also be reviewed and staged before the index is complete.
