# Phase 1 notes - possession model (Mats)

Short summary of what I did on branch `mats/possession-hmm`, what I found, and what I think we
should do next. Everything here can be reproduced with the commands at the bottom.

## TL;DR

- I built a possession model that labels every frame H / A / D.
- I couldn't test it on our own match (we only have 5 minutes of answers and we're not allowed to
  tune on them), so I tested it on 7 open Bundesliga matches that do have labels.
- There it gets **spellF1 0.44** vs **0.26** for the organisers' baseline. On our match it gives
  367 possessions of 9.5 s on average, which looks realistic.
- It's not perfect (0.44 out of 1.0), but it works correctly and gives the same file every run.

## The challenge in one minute

There are three rounds: Phase 1 (now, everyone, deadline **17 Nov**), then the top 20 go to the
Innovation Challenge (10 extra matches), and the top 5 pitch in Barcelona in March.

Phase 1 = one CSV with a label for each of the 199,319 frames of Barça - Athletic:
`H` home has the ball, `A` away, `D` dead ball, `X` no answer. We're ranked on **spellF1**, which
counts whole possessions, not frames. So the things that hurt most are:

- a fake stoppage in the middle of a possession (it splits one possession into two),
- flickering between teams,
- being late on a change (1 second late already costs about a third of the score).

## What I found in our data

I wrote `scripts/data_profile.py`, which only prints summary numbers, so we can talk about the
data without sharing it. Main findings:

- The ball is missing on 41% of the graded frames, but almost always in **long gaps** (median
  11 s). Short dropouts basically don't exist.
- While the ball is missing, players move about **half as fast** (1.1 vs 2.0 m/s). So most gaps
  are stoppages, not live play.
- The tracker doesn't stop the ball at the lines: it is seen outside the pitch quite often, which
  is a good dead-ball signal.
- The third ball value is the height (around 0.1 m on the grass, up to ~4 m), but it is only
  filled in for ~15% of frames, so it didn't help.
- The Smart Tagging is weak for this: set pieces are tagged as fixed 35 s windows and the team
  phases often span several possessions. I don't use it.

## What other teams and papers do

About 40 teams already have public repos. The most useful lessons: most good solutions use a
state model with a cost for switching (like mine), the best ones also learn when a touch is a
real change of possession, and one team found an open dataset of 7 Bundesliga matches with
per-frame ball status and possession (Bassek et al. 2025, CC BY 4.0). That dataset is what I used
for testing.

## How my model works

For every frame it gives a score to H, A and D, then picks the best sequence with a small cost
for every change (Viterbi). In football terms:

- **Possession only changes with real control.** The other team needs to be clearly on the ball
  for a short moment. A touch or a deflection isn't enough.
- **A fast ball (over 8 m/s) belongs to nobody.** That's a pass or a shot, so it can't flip
  possession by flying past an opponent.
- **The change happens at the first touch** of the new team, not halfway through the pass.
- **Dead ball** comes from: the ball more than 0.3 m over a line (also while the tracker has lost
  it afterwards), players walking, and long ball gaps. A rolling ball means play is live.
- **Ball lost but players still running** = keep the current possession.
- **The team with the ball spreads out**, so the more spread-out team gets a small extra nudge.
  This also helps while the ball is lost.
- Frames outside the two halves get `X` (they're not scored).

Code: `src/possession_lab/hmm.py` (model) and `src/possession_lab/atd.py` (reads the tracking).
All settings are in `configs/models/possession_hmm.toml`.

## How I tested it (without breaking the rules)

`scripts/open_data_check.py` downloads the 7 open matches and makes them look like our tracking,
using only the summary numbers of our data: ball visible ~61% (ours 59%), same gap lengths, same
ball and player speeds. Good sign: the organisers' baseline scores almost exactly the same there
as on our match (0.257 vs 0.243), so the test setup is realistic.

I picked the settings on 5 matches and kept 2 apart as a final check:

| | spellF1 | macroF1 |
|---|---|---|
| organisers' baseline (smooth50) | 0.257 | 0.677 |
| **my model, all 7 matches** | **0.438** | **0.846** |
| my model, the 2 held-out matches | 0.438 / 0.476 | |

What made the biggest difference: a lower switching cost (the first version was way too sticky),
the 8 m/s pass rule, the smaller 0.3 m out-of-play margin, snapping changes to the first
touch, and (a little) the team-spread rule.

Results on our match (no answers needed, just sanity): first version 212 possessions (too sticky),
then 485 (too jumpy, because of a bug in my test setup that I fixed), now **367 possessions,
9.5 s average, 45% dead ball, Barça 62% of live time**. That fits what other teams estimate and
normal LaLiga numbers.

Checks I did on our match:

- **Same file every run**: running it twice gives the same SHA256 (`2302f46a...e0cd4c`).
- **Pipeline check on the public 5 minutes** (only to see that it works, not to tune): home and
  away are the right way round (87% of live frames get the right team), frames line up, and it
  finds 18 possessions where there are 17. It also confirms dead ball is the weak spot: when it
  says dead ball it's almost always right, but it misses about 40% of them.

Honest caveats: the Bundesliga labels switch possession a bit more often than the organisers'
labels, so the model might be slightly too jumpy for our match. And the real score always moves
around (±0.04 is noise on a full match).

## Rules we need to keep

- No hand labelling of this match, ever, not even to test something.
- Don't tune or choose versions on the public 5 minutes. Only use them to check the pipeline.
- Keep the match data on your own computer: not in git, not in iCloud/Dropbox, never uploaded to
  any cloud service or AI tool.
- The repo must regenerate the exact same CSV. Running the model twice should give the same
  SHA256.

## How to run it

```powershell
git checkout mats/possession-hmm
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location).Path ".venv-mlops"
uv sync --locked --group dev --python 3.11
$env:MTAH_DATA_DIR = "C:\mtah-data"          # folder with tactical\ and broadcast\
uv run --no-sync python scripts/data_profile.py
uv run --no-sync python -m possession_lab.cli predict --config configs/models/possession_hmm.toml --output evaluation/results/team_hmm.csv
uv run --no-sync python evaluation/validate_submission.py evaluation/results/team_hmm.csv
```

Open-data test (needs ~2.6 GB):

```powershell
uv run --no-sync python scripts/open_data_check.py --download
uv run --no-sync python scripts/open_data_check.py
```

## Where the points are lost

I checked on the open matches what we'd score if one part were perfect:

| if this were perfect | spellF1 |
|---|---|
| nothing (current model) | 0.44 |
| ball never lost during play | 0.44 (no gain) |
| dead-ball detection | 0.48 |
| which team has the ball during live play | 0.90 |

So almost everything is in deciding **which team has the ball and when it changes**. Recovering
the ball from the video is not worth our time.

I also tried a small learned model (trained on the open matches) for "which team has the ball".
It picks the right team on many more frames (84% vs 73%), but it did **not** give better
possessions, because its clues change slowly and blur the exact moment of a change. Only the
team-spread part helped, so that is the one I kept, as a simple rule.

One warning: in the Bundesliga labels 43% of possessions last under 3 seconds. The organisers'
5 public minutes average 14 s per possession, so their labels have far fewer tiny possessions.
We shouldn't chase the short ones.

## What I'd do next (can be split)

1. **Timing of changes**: a lot of the wrong-team frames are within a second of a real change.
   Better detection of the actual touch (ball changes direction or speed near a player) should
   help.
2. **Dead-ball detection**: restarts (still ball that gets kicked, players lining up) and
   starting stoppages right when the ball crosses the line.
3. **Long wrong stretches**: half of the wrong-team frames are in stretches over 3 s. Worth
   looking at a few of those in the video to see what goes wrong (just watching, no labelling).
4. **Submission**: technical document, final reproducibility check, clean public repo, file
   named `<team_name>.csv`.

Happy to go through it in the meeting.
