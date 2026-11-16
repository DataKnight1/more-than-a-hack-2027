# Possession Lab — technical document draft

**Release status:** N018 is the recorded champion. The team has confirmed that
the trained model can be shared for the competition; the exact submission,
reproduction environment and team name still need to be frozen. This document describes the recorded method and
must be checked against the final submission before it is submitted.

## Approach

The model reads the tactical automatic tracking data (ATD). It maps the ball and
players to identity-independent features, applies a set encoder to each frame
and a temporal transformer to overlapping windows, and predicts home possession,
away possession or dead ball. An auxiliary head estimates possession changes. A
Viterbi decoder applies externally selected transition penalties to produce
frame labels; frames outside the designated halves are marked `X`. The N018
configuration uses a 256-dimensional model, four spatial and eight temporal
layers, and 384-frame windows. Its frozen decoder uses team-switch penalty 2,
dead-ball penalty 4 and transition gain 1.

The method learns control and stoppage patterns from the external labels; it
does not apply separately hand-written rules for each loose ball, challenge or
restart. Tracking dropouts, ambiguous control and differing provider label
definitions can cause errors, particularly at short possession boundaries.
Smart Tagging, homographies and video are not inputs to the recorded N018
inference path. The supplied game and its labels were not used for training,
calibration, tuning or model selection.

## Labelled external data in the recorded training configuration

Splits are by match. Approximate quantities and label provenance below come from
the research record; verify them and their usage rights before release.

| Dataset | Recorded use | Labels and attribution |
|---|---|---|
| IDSSE/DFL, [figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177.v1) | Seven matches: four train, one validation, two development audit; approximately one million labelled frames | DFL provider ball status and possession labels; CC BY 4.0 as recorded. |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | 20 matches: 12 train, three validation, five test; approximately 2.9 million frames after resampling to 25 Hz | SkillCorner possession and phase labels; dead ball derived from phases. MIT licence as recorded. |
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | Three games: one train, one validation, one development audit; approximately 430,000 frames | Metrica event annotations converted algorithmically to possession and dead-ball labels. The team confirmed that the model may be shared for the competition; the upstream repository requests acknowledgement but does not identify a standard licence. |

The three-match development audit has been reused during development and is not
an independent final test. For N018 the recorded clean/degraded mean spellF1 was
0.692; this is not a score on the challenge match. A second seed was waived when
N018 was promoted, so seed-to-seed stability remains unconfirmed.

## Reproduction and limitations

See the [README](README.md) for installation and the intended command, and
[`configs/neural/n018_large_long.toml`](configs/neural/n018_large_long.toml) and
[`configs/neural/n018_decoder.json`](configs/neural/n018_decoder.json) for the
recorded configuration. The checkpoint is included at `models/n018/best.pt`.
CUDA inference uses float16 and permits nondeterministic operations. Two local
regenerations and a separate clean-checkout run with the pinned Windows CUDA
environment matched the recorded submission hash and passed the official
validator. Cross-hardware reproduction remains unverified. No video-versus-output inspection or public-ground-truth model
selection is permitted. Final validation must use the official validator on the
exact team-named CSV.
