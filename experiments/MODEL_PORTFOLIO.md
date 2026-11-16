# Investigation portfolio

The order is **hypothesis-driven** and may change after robustness evidence. Do not benchmark every new technique against the public five-minute GT; doing so violates organiser rules.

| ID | Family | Why it might work | Potential failure |
|---|---|---|---|
| M00 | Reproduced nearest-player+majority baseline | Deterministic floor | Missing ball, dead-ball misclassification, flicker |
| M01 | Team-relative spatial proximity (metres) | Distinguishes team pressure around ball | Ball aerial/occluded |
| M02 | Speed, velocity, nearest-team confidence | Provides controllability evidence | Track gaps and unstable velocities |
| M03 | Tactical tag priors | Adds attacking context to missing-ball frames | Tags are phases, not possession labels |
| M04 | HMM with constrained state transitions | Suppresses implausible frame flicker | Geometric duration assumptions |
| M05 | HSMM/explicit-duration model | Spell-duration prior | May merge real short possessions |
| M06 | Segment/change-point model | Optimises spell boundaries | Weak change-point evidence |
| M07 | Separate dead-ball / restart detector | Distinguishes occlusion from stop | Hard to observe without referee signal |
| M08 | Event-driven hierarchical state model | Coordinates restart and possession | Model complexity, spurious tag events |
| M09 | Supervised gradient-boosting or sequence model | Learns nonlinear features | No eligible labelled supplied-match training data; needs OTHER match labels |
| M10 | Video feature models | Handles tracking occlusions | Resource demands, permitted processing, reproducibility |

## Research questions

- How does prediction continuity behave under 10%, 30%, 60% additional simulated ball dropouts?
- Can restart evidence reliably identify `D` without confusing an off-camera ball with a stopped ball?
- How sensitive is state segmentation to window sizes, independent of the supplied GT?
- How much do lagged player/ball velocities help detect controlled handover vs deflection?
- What is the relative cost of over-segmentation and under-segmentation under official spell matching on **synthetic labelled** examples?

No prescribed model family guarantees a score gain. Make experimentally defensible choices.
