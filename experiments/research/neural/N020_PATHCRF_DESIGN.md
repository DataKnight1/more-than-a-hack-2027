# N020 — PathCRF Possession Recognition: Architecture & System Design

**Date:** 2026-11-12  
**Branch:** `experiment/n020-pathcrf`  
**Reference Paper:** Kim et al., *PathCRF: Player Path-Centric Conditional Random Field for Football Possession Recognition*, KDD 2026, arXiv:2602.12080.  
**Upstream Repository:** `https://github.com/hyunsungkim-ds/pathcrf` (Commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, MPL-2.0).

---

## 1. Executive Summary & Purpose

The objective of **N020** is to implement, adapt, and rigorously evaluate **PathCRF** for the Possession Lab 2027 Challenge. Unlike existing point-prediction architectures (such as the current N018 champion, a 9.7M-parameter Set-Temporal network with transition head), PathCRF explicitly reasons about player-to-player and player-to-boundary possession transfers:

$$\mathcal{E} = \{ (u, v) \mid u \in \mathcal{V}, v \in \mathcal{V} \}$$

where $\mathcal{V}$ consists of 22 active player slots (11 Home, 11 Away) plus 4 canonical pitch boundary nodes (`Outside_Left`, `Outside_Right`, `Outside_Bottom`, `Outside_Top`).

PathCRF enforces physical continuity through a sparse transition matrix that forbids teleports between non-adjacent edges while learning structured pairwise passing dynamics. We extend this model with an explicit Dead-Ball detection head and conditional Home/Away team heads, enabling zero-leakage, full-period 25 Hz inference for the official competition alphabet:

$$\mathcal{Y} \in \{ \text{H (Home)}, \text{A (Away)}, \text{D (Dead)}, \text{X (Gap/Ungraded)} \}$$

---

## 2. Theoretical Formulation

### 2.1 26-Node Representation
Each frame $t$ is mapped to $N=26$ nodes:
- Nodes `0..10`: Home team player slots $H_0 \dots H_{10}$
- Nodes `11..21`: Away team player slots $A_0 \dots A_{21}$
- Nodes `22..25`: Boundary nodes ($O_{\text{left}}, O_{\text{right}}, O_{\text{bottom}}, O_{\text{top}}$)

Each node carries an 8-channel kinematic feature vector $\mathbf{x}_{t, i} \in \mathbb{R}^8$:
1. `team_indicator`: $-1.0$ (Home), $+1.0$ (Away), $0.0$ (Outside)
2. `goalkeeper_flag`: $1.0$ (GK identified by defensive depth), $0.0$ (Field player)
3. `x`: Standard pitch meters $[0.0, 105.0]$
4. `y`: Standard pitch meters $[0.0, 68.0]$
5. `vx`: Velocity along x (m/s, physical limit clipped at 12.0 m/s)
6. `vy`: Velocity along y (m/s, physical limit clipped at 12.0 m/s)
7. `speed`: Scalar speed $\sqrt{v_x^2 + v_y^2}$
8. `acceleration`: Acceleration magnitude ($m/s^2$, clipped at 15.0 $m/s^2$)

**Player-Only Contract:** Coordinates of `match.ball` are completely excluded from PathCRF inputs. Even when tracking loses the ball completely, PathCRF operates unimpaired on player velocities and spatial formations.

### 2.2 576-State Compressed Edge Graph
A dense graph over 26 nodes contains $26 \times 26 = 676$ directed edges. However, outside boundary nodes cannot pass to players or cross between boundary segments:
- Disallowed outside-origin edges: $4 \times 25 = 100$ edges
- Absorbing/self-transition boundary edges: 4 edges ($O_k \to O_k$)
- Valid compressed edges: $K = 676 - 100 = 576$ states

A bijective mapping is maintained:
- $\text{valid\_orig\_edge\_ids} \in \mathbb{N}^{576}$: indices in dense $[0..675]$
- $\text{orig2comp} \in \mathbb{Z}^{676}$: index in $[0..575]$ or $-1$ if disallowed
- $\text{comp\_src}, \text{comp\_dst} \in \mathbb{N}^{576}$: source and target node indices

### 2.3 Legal Transition Graph & Dynamic Sparse CRF
PathCRF defines a directed transition graph between compressed edge states:
- Self-retention: $(u \to v) \to (u \to v)$
- Next transfer: $(u \to v) \to (v \to w)$
- Turnover: $(u \to v) \to (w \to z)$ where $w$ is an opponent
- Out of play: $(u \to v) \to (v \to O_{\text{boundary}}) \to (O_{\text{boundary}} \to O_{\text{boundary}})$

The maximum incoming degree is $M=23$. The Dynamic Sparse CRF computes sequence log-partition function $\log Z(\mathbf{x})$ and Viterbi paths via custom sparse dynamic programming:

$$\log Z(\mathbf{x}) = \text{LogSumExp}_{\mathbf{y}} \left( \sum_{t=1}^T \psi(y_t, \mathbf{x}_t) + \sum_{t=1}^{T-1} \phi(y_t, y_{t+1}, \mathbf{x}_t) \right)$$

---

## 3. Adaptation for the 2027 Challenge

### 3.1 Multi-Task Supervision
Because the challenge does not evaluate player passing links, but rather match-level possession `H`, `A`, `D`, we implement multi-task supervision:
1. **Dead-Ball Binary Head:** $\mathcal{L}_{\text{dead}} = \text{BCEWithLogits}(\hat{y}_t^{\text{dead}}, \mathbb{I}(y_t = \text{D}))$.
2. **Conditional Team Head:** $\mathcal{L}_{\text{team}} = \text{CrossEntropy}(\hat{\mathbf{y}}_t^{\text{team}}, y_t)$ computed on alive frames ($y_t \in \{\text{H}, \text{A}\}$).
3. **Strong Path Supervision:** Unary edge cross-entropy on approved external training matches with event-derived player touches (`idsse_J03WMX`, `idsse_J03WOH`, `idsse_J03WOY`, `idsse_J03WPY`).
4. **CRF Sequence NLL:** Exact sequence negative log-likelihood on verified legal path segments.

### 3.2 Probabilistic Posterior Fusion
To combine the multi-task predictions into the required 3-class posterior $[P(\text{H}), P(\text{A}), P(\text{D})]$:
1. $P(\text{D}) = \sigma(\hat{y}^{\text{dead}})$
2. $P(\text{Alive}) = 1.0 - P(\text{D})$
3. Team evidence from edge emissions:
   $$P_{\text{path}}(\text{H}) = \sum_{(u, v) \in \mathcal{E}_{\text{Home}}} P(u \to v), \quad P_{\text{path}}(\text{A}) = \sum_{(u, v) \in \mathcal{E}_{\text{Away}}} P(u \to v)$$
4. Conditional team probability:
   $$P(\text{Team} \mid \text{Alive}) = \text{Softmax}\left( (1 - w_{\text{path}}) \cdot \hat{\mathbf{y}}^{\text{team}} + w_{\text{path}} \cdot \log \mathbf{P}_{\text{path}} \right)$$
5. Joint posterior:
   $$P(\text{H}) = P(\text{Alive}) \cdot P(\text{H} \mid \text{Alive})$$
   $$P(\text{A}) = P(\text{Alive}) \cdot P(\text{A} \mid \text{Alive})$$

### 3.3 Full-Period 25 Hz Output Alignment
- Inference is performed in sliding windows with trapezoidal ramp overlap blending.
- Period boundaries are strictly respected (gaps are assigned $P = [1/3, 1/3, 1/3]$ and mapped to official ungraded code `X`).
- Probabilities are upsampled from the model's native sampling rate (5 Hz) to the exact 25 Hz frame count using pure NumPy vectorized linear interpolation.
- The 25 Hz sequence is decoded via Viterbi transition decoding (`DecodeConfig(team_switch, dead_switch)`), selected strictly on external validation.

---

## 4. Hardware and Computational Profile

- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Trainable Parameters:** 553,558 (Backbone: 553,525, CRF: 33).
- **Peak VRAM:** 831.5 MB at batch size 4 (50 frames / 10s window).
- **Training Throughput:** ~2,600 to 3,000 frames/sec with in-memory pre-sampled feature caching.
- **Inference Speed:** ~31 seconds for a full 142,370-frame match (~4,500 frames/sec) with batched sliding windows.

