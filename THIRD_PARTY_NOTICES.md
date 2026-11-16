# Third-Party Software Notices and Attribution

This repository includes adaptations of open-source software under the terms of their respective licenses.

---

## PathCRF

- **Upstream Repository:** https://github.com/hyunsungkim-ds/pathcrf
- **Original Authors:** Hyunsung Kim, Dongwook Chung, Yundung Jo, Jinsung Jeon
- **Original Paper:** *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories*, KDD 2026 (arXiv:2602.12080)
- **Commit SHA:** `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`
- **License:** Mozilla Public License Version 2.0 (MPL-2.0)

### MPL-2.0 License Terms
Covered Software is provided under the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

Portions of `src/possession_lab/neural/pathcrf/` adapt original algorithmic concepts and functions from PathCRF, specifically:
- Set Attention backbone (`SetLSTM`) with spatial and temporal Set Attention Blocks
- 576-state compressed directed possession-edge graph (`build_edge_compression_maps`, `build_allowed_mask`)
- Dynamic Sparse Conditional Random Field (`DynamicSparseCRF`) with incoming-list transitions and Viterbi decoding

Extended in Possession Lab (More Than a Hack 2027) with:
- Canonical 25 Hz match adapter with NaN-safe kinematics and observation masks
- Explicit supervised Home / Away / Dead-ball heads for non-oracle full-match sequence classification
- Training-only provenance filtering to prevent validation and test match leakage

