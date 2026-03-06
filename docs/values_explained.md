# How Constants and Variables Are Determined

**IEEE Trans. Smart Grid -- Learnable Delay-Stability Coupling**
**Plain-language companion explaining where every number comes from**

---

## What This Document Covers

The paper's framework involves dozens of symbols. Some are fixed by the grid hardware, some are computed from physics, some are learned by the neural network, and some change in real time. This document explains **where each value comes from**, **how it is initialized**, and **how it evolves** -- from the moment the IEEE test case file is loaded to the moment the trained model makes a real-time dispatch decision.

Companion documents:
- `theorem_explained.md` -- Assumptions A1--A4, Theorem 1 proof, Corollary 1
- `contributions_explained.md` -- Contributions C1--C5 in plain language

---

## The Four Categories of Values

Every quantity in the paper falls into one of four categories:

| Category | When determined | Changes during training? | Changes at deployment? |
|----------|----------------|--------------------------|------------------------|
| **Given constants** | Before anything runs | No | No |
| **Computed constants** | During data generation | No | No |
| **Learnable parameters** | Initialized, then trained | Yes (gradient descent) | No (frozen after training) |
| **Dynamic variables** | Every time step | N/A (not training) | Yes (real-time inputs) |

Think of it like a car:
- **Given constants** = the car's weight, engine size (stamped at the factory)
- **Computed constants** = the car's center of gravity (calculated from the design)
- **Learnable parameters** = the driver's muscle memory (developed during practice, fixed during the race)
- **Dynamic variables** = current speed, steering angle (changing every moment)

---

## 1. Given Constants (From IEEE Test Case Files)

These come directly from the IEEE standard test case data files distributed with MATPOWER. Every researcher who loads "IEEE 14-bus" gets identical values. Nothing is chosen or computed -- it is a fixed dataset.

### Grid topology

| Symbol | IEEE 14 | IEEE 30 | IEEE 39 | IEEE 57 | IEEE 118 |
|--------|---------|---------|---------|---------|----------|
| $N$ (buses) | 14 | 30 | 39 | 57 | 118 |
| $n_g$ (generators) | 5 | 6 | 10 | 7 | 54 |
| $\|\mathcal{E}_E\|$ (lines) | 20 | 41 | 46 | 80 | 186 |

**Where to find them:** The MATPOWER `.m` file specifies a `bus` matrix (one row per bus), a `gen` matrix (one row per generator, including which bus it connects to), and a `branch` matrix (one row per transmission line, including its impedance).

For example, in IEEE 14-bus:
- Generators sit at buses 1, 2, 3, 6, 8
- That is 5 generators, so $n_g = 5$
- There is no flexibility here -- it is part of the standard

### Per-generator physical parameters

Each generator in the case file specifies:

| Symbol | Meaning | Where from | Example (Gen 1, IEEE 14) |
|--------|---------|------------|--------------------------|
| $H_i$ | Inertia constant (seconds) | `gen` data | ~6.5 s |
| $D_i$ | Damping coefficient (p.u.) | `gen` data | ~2.0 p.u. |
| $P_{m,i}^0$ | Base mechanical power (MW) | Power flow solution | ~232 MW |

**What $H_i$ means physically:** A generator with inertia constant $H = 6.5$s can supply its rated power for 6.5 seconds purely from the kinetic energy stored in its spinning rotor. Higher inertia means slower response to disturbances -- the generator is "heavier" and harder to knock off balance.

### Per-line electrical parameters

Each transmission line specifies:

| Symbol | Meaning | Where from |
|--------|---------|------------|
| $Z_{ij}$ | Impedance (ohms) | `branch` data ($r + jx$) |
| $B_{ij}$ | Susceptance (siemens) | Derived: $B_{ij} = 1/X_{ij}$ |
| Thermal limit | Maximum power flow (MVA) | `branch` data |

**What impedance means:** Low impedance between two buses means a strong electrical connection (like a thick wire). High impedance means a weak connection (like a thin, long wire). The physics mask $M_{\text{phys}}[i,j] = -\gamma \cdot Z_{ij}/Z_{\max}$ uses these impedances directly.

---

## 2. Computed Constants (From Physics, During Data Generation)

These are not in the case file directly but are computed from the given constants using well-known power systems formulas. They are computed once per scenario (or once per grid, for the eigenvalue-related quantities).

### 2a. Power flow solution

**What it is:** Solving the nonlinear power flow equations to find the steady-state operating point.

**Input:** Bus types (slack, PV, PQ), generator setpoints, load demands, line impedances.

**Output:** Voltage magnitudes $V_i$, voltage angles $\theta_i$, and power injections $P_i, Q_i$ at every bus.

**How it works (simplified):**
1. Start with an initial guess for all voltages and angles
2. Iteratively solve the power balance equations $P_i = \sum_j V_i V_j Y_{ij} \cos(\theta_i - \theta_j - \phi_{ij})$ using Newton-Raphson
3. Converge in 3--8 iterations typically

**Analogy:** This is like solving "given the road network and traffic demand, what is the flow on each road?" Power flow is the electrical equivalent of traffic assignment.

### 2b. Synchronizing torque matrix $\mathbf{L}$

**What it is:** The matrix of partial derivatives $L_{ij} = \partial P_{e,i} / \partial \delta_j$ evaluated at the operating point.

**How it is computed:**
```
L_ij = V_i * V_j * B_ij * cos(delta_i - delta_j)    (for i != j)
L_ii = -sum_{j != i} L_ij                             (diagonal)
```

This is a Laplacian-like matrix that captures how strongly each generator's electrical power output depends on other generators' rotor angles. It encodes the **electrical coupling** between generators.

**Analogy:** Think of generators connected by springs. $L_{ij}$ is the spring constant between generator $i$ and generator $j$. Strong springs (low impedance lines) mean strong coupling.

### 2c. Zero-delay Jacobian $\mathbf{J}(0)$

**What it is:** The system matrix that governs how small perturbations evolve over time, assuming zero communication delay.

**How it is assembled:**
```
J(0) = [    0           I        ]     size: 2n_g x 2n_g
       [ -M^{-1} * L   -M^{-1} * D ]
```

For IEEE 14-bus with 5 generators, this is a 10x10 matrix.

**What each block means:**
- Top-left (0): Rotor angles do not directly affect other rotor angles
- Top-right (I): Rotor angle derivatives equal angular velocities (by definition)
- Bottom-left ($-M^{-1}L$): Angular acceleration depends on angle deviations (through electrical coupling), scaled by inertia
- Bottom-right ($-M^{-1}D$): Angular acceleration is damped by the damping coefficient, scaled by inertia

### 2d. Eigenvalues of $\mathbf{J}(0)$ and $\lambda_{\min}(0)$

**What they are:** The eigenvalues of $\mathbf{J}(0)$ determine the oscillatory modes of the power system.

**How they are computed:** Standard numerical eigenvalue decomposition (e.g., `numpy.linalg.eig(J0)` or `scipy.linalg.eig(J0)`).

**What they look like (IEEE 14-bus example):**

The 10x10 Jacobian has 10 eigenvalues. For a stable system, all have negative real parts. They come in conjugate pairs representing oscillatory modes:

```
Mode 1:  -0.73 +/- 8.2j   (inter-area mode: slow, poorly damped)
Mode 2:  -1.05 +/- 6.1j   (local mode)
Mode 3:  -0.91 +/- 4.8j   (local mode)
Mode 4:  -2.30 +/- 3.5j   (well-damped local mode)
Mode 5:  -1.64 +/- 2.1j   (well-damped local mode)
```

**Finding $\lambda_{\min}(0)$:** Sort all eigenvalues by their real part. The one closest to zero (least negative) is the "least stable." In this example:

- $\lambda_{\min}(0) = -0.73$ (real part of Mode 1)
- $|\lambda_{\min}(0)| = 0.73$

This is the "baseline stability margin" -- how much room the system has before becoming unstable.

**Physical meaning of each mode:**
- Real part magnitude: how quickly oscillations damp (larger = faster damping)
- Imaginary part: oscillation frequency in rad/s
- Mode 1 (real part -0.73): inter-area oscillation where groups of generators swing against each other. This is always the weakest mode because distant generators have weak coupling through high-impedance lines.

### 2e. Eigenvalues across IEEE test cases

| IEEE Case | $n_g$ | $\|\lambda_{\min}(0)\|$ | Physical interpretation |
|-----------|-------|-------------------------|------------------------|
| 14-bus | 5 | ~0.73 | Small grid, strong coupling, good damping |
| 30-bus | 6 | ~0.52 | Moderate grid, some weak modes |
| 39-bus | 10 | ~0.59 | New England system, well-studied benchmark |
| 57-bus | 7 | ~0.47 | More complex topology, weaker inter-area mode |
| 118-bus | 54 | ~0.38 | Large system, weak inter-area oscillations |
| 10K synth | ~2000 | ~0.31 | Very large, barely damped inter-area modes |

**Why $|\lambda_{\min}|$ shrinks for larger grids:** Larger grids have generators that are farther apart electrically. The coupling between distant groups is weak (high impedance), so inter-area oscillations damp slowly. This makes larger grids inherently more vulnerable to delay -- which is exactly why the auto-scaled $K_{\text{init}}$ is needed.

### 2f. Eigenvector matrix $\mathbf{V}$ and condition number

**What it is:** $\mathbf{J}(0) = \mathbf{V} \Lambda \mathbf{V}^{-1}$, where $\Lambda$ is the diagonal matrix of eigenvalues and $\mathbf{V}$ is the matrix of eigenvectors.

**How it is computed:** Part of the eigenvalue decomposition (`numpy.linalg.eig` returns both eigenvalues and eigenvectors).

**Condition number:** $\text{cond}(\mathbf{V}) = \|\mathbf{V}\|_2 \cdot \|\mathbf{V}^{-1}\|_2$

- If $\text{cond}(\mathbf{V}) = 1$: the eigenvectors are orthogonal (the system is "normal"). The Bauer-Fike bound is tight.
- If $\text{cond}(\mathbf{V}) \gg 1$: the eigenvectors are nearly parallel (the system is "non-normal"). The bound becomes conservative.

Typical values for IEEE test cases: $\text{cond}(\mathbf{V}) \approx 5$--$50$.

---

## 3. Learnable Parameters (Trained by Gradient Descent)

These are initialized at the start of training and then updated over hundreds of epochs. After training, they are frozen and never change again.

### 3a. Coupling constants $K_i$

**What they represent:** How sensitive generator $i$ is to communication delay. A generator with $K_i = 0.12$ loses 0.12 units of stability margin when its delay reaches $\tau_{\max}$.

**How they are initialized (auto-scaled formula):**
```
K_init = s * |lambda_min(0)| / n_g        where s = 0.9 (safety factor)
```

| IEEE Case | $n_g$ | $\|\lambda_{\min}\|$ | $K_{\text{init}}$ per generator | Total initial budget |
|-----------|-------|---------------------|-------------------------------|---------------------|
| 14-bus | 5 | 0.73 | 0.131 | 0.657 (90% of 0.73) |
| 30-bus | 6 | 0.52 | 0.078 | 0.468 (90% of 0.52) |
| 39-bus | 10 | 0.59 | 0.053 | 0.531 (90% of 0.59) |
| 57-bus | 7 | 0.47 | 0.060 | 0.423 (90% of 0.47) |
| 118-bus | 54 | 0.38 | 0.006 | 0.342 (90% of 0.38) |

**Why auto-scaling matters:** The old fixed $K = 0.1$ gives IEEE 14-bus a total budget of $5 \times 0.1 = 0.5$ (68% of the 0.73 margin -- acceptable) but gives IEEE 118-bus a total of $54 \times 0.1 = 5.4$ (1421% of the 0.38 margin -- the bound starts deeply negative and is useless).

**How they are stored internally:**
```
kappa_i = ln(K_init)          # log-space parameter (unconstrained real number)
K_i = exp(kappa_i)            # actual coupling constant (always positive)
```

For IEEE 14-bus: $\kappa_{\text{init}} = \ln(0.131) = -2.03$

**How they evolve during training:**

At each training step, the gradient of the total loss flows backward through $K_i$:
```
gradient = dL/d(kappa_i) = dL/dK_i * dK_i/d(kappa_i) = dL/dK_i * K_i
```

The coupling loss $\mathcal{L}_{\text{couple}} = \|\rho_{\text{emp}} - \rho_{\text{theo}}\|^2$ drives this:
- If the theoretical bound **overestimates** stability (predicts stable but simulation shows unstable), the gradient pushes $K_i$ **up** (more conservative)
- If the bound **underestimates** stability (predicts unstable but simulation shows stable), the gradient pushes $K_i$ **down** (tighter bound)

**Typical evolution (IEEE 39-bus, 10 generators):**

| Epoch | K_1 | K_2 | K_3 | K_7 | K_10 | Description |
|-------|-----|-----|-----|-----|------|-------------|
| 0 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | All equal (auto-scaled init) |
| 50 | 0.061 | 0.048 | 0.055 | 0.072 | 0.039 | Differentiation begins |
| 200 | 0.068 | 0.041 | 0.052 | 0.085 | 0.032 | Structure emerging |
| 500 | 0.071 | 0.038 | 0.050 | 0.091 | 0.028 | Converged |

Notice: Generator 7 ends up with the largest $K_7 = 0.091$, meaning it is the most delay-sensitive. Generator 10 has the smallest $K_{10} = 0.028$, meaning it can tolerate more delay. The total budget shifts from $10 \times 0.053 = 0.53$ to $\sum K_i \approx 0.56$ -- the network redistributes the budget but keeps the total similar.

**After training:** All $K_i$ are frozen. During deployment, they act as fixed sensitivity ratings.

### 3b. Physics mask strength $\gamma$

**What it is:** Controls how strongly the impedance-based attention bias affects which buses attend to which.

**Initialization:** $\gamma = 0.1$

**How it evolves:** The adaptive gamma mechanism scales $\gamma$ relative to the attention logit magnitude. If attention scores are large, $\gamma$ scales up to remain relevant; if attention scores are small, $\gamma$ scales down to avoid dominating.

### 3c. Neural network weights

**What they include:**
- GNN message-passing weights (energy encoder and communication encoder)
- Attention projection matrices $W_Q, W_K, W_V$ (for cross-domain and causal attention)
- Decoder MLP weights
- Layer normalization parameters

**Total parameter count:**

| IEEE Case | Parameters | Description |
|-----------|-----------|-------------|
| 14-bus | ~469K | Smallest model |
| 118-bus | ~851K | Largest model |

**Initialization:** Standard Xavier/Glorot initialization for linear layers, zeros for biases.

---

## 4. Dynamic Variables (Change Every Time Step)

These are the inputs that change during real-time operation. The trained model receives them and produces a dispatch decision.

### 4a. Communication delays $\tau_i$

**What they are:** The current end-to-end latency for control signals reaching generator $i$.

**How they are generated (during training):**
```
tau_i ~ LogNormal(mu=50 ms, sigma=20 ms), clipped to [5, 500] ms
```

**How they change (during deployment):** They are measured in real time by the SCADA system. They fluctuate based on:
- Network congestion (e.g., peak hours, data storms)
- Equipment failures (e.g., router outage causing rerouting)
- Physical distance (e.g., remote substations have higher base latency)
- Weather (e.g., wireless links degrade in rain)

### 4b. Stability margin $\rho(\tau)$

**What it is:** The "distance from instability" at the current moment.

**How it is computed:**
```
rho(tau) = |lambda_min(0)| - sum_i( K_i * tau_i / tau_max_i )
```

**Key property: It responds instantly to delay changes.** The formula is a simple weighted sum, so as soon as $\tau_i$ changes, $\rho$ updates immediately. But the physical system (generator oscillations) lags behind -- see Section 6.

### 4c. Energy features $\mathbf{x}_E$

| Feature | Symbol | Meaning | How measured |
|---------|--------|---------|--------------|
| Active power | $P_i$ | Real power injection at bus $i$ | PMU or SCADA |
| Reactive power | $Q_i$ | Reactive power at bus $i$ | PMU or SCADA |
| Voltage magnitude | $V_i$ | Voltage at bus $i$ | PMU or SCADA |
| Voltage angle | $\theta_i$ | Angle relative to reference | PMU |
| Frequency deviation | $\omega_i$ | Deviation from 50/60 Hz | PMU |

### 4d. Communication features $\mathbf{x}_I$

| Feature | Symbol | Meaning | How measured |
|---------|--------|---------|--------------|
| Delay | $\tau_i$ | End-to-end latency | Network monitor |
| Data rate | $R_i$ | Available bandwidth | Network monitor |
| Buffer occupancy | $B_i$ | Queue fullness | Network monitor |

---

## 5. The Full Lifecycle of Values

Here is the complete timeline showing when each value is determined:

### Phase 1: Grid Loading (once, before everything)

```
Load IEEE 14-bus case file
  --> N = 14, n_g = 5 (given)
  --> Generator buses: {1, 2, 3, 6, 8} (given)
  --> Inertia: M = diag(6.5, 3.2, 4.1, 2.8, 3.5) seconds (given)
  --> Damping: D = diag(2.0, 1.5, 1.8, 1.2, 1.6) p.u. (given)
  --> Line impedances: Z_ij for all 20 lines (given)
```

### Phase 2: Power Flow + Eigenvalue Computation (once per grid)

```
Solve power flow
  --> V_i, theta_i, P_i, Q_i at every bus (computed)

Build synchronizing torque matrix L (computed)
  --> L_ij = V_i * V_j * B_ij * cos(theta_i - theta_j)

Assemble Jacobian J(0) (computed)
  --> 10x10 matrix for IEEE 14-bus

Eigenvalue decomposition (computed)
  --> lambda_min(0) = -0.73
  --> |lambda_min(0)| = 0.73
  --> V (eigenvector matrix), cond(V) ~ 12.3
```

### Phase 3: Initialization (once, before training starts)

```
Auto-scale K:
  K_init = 0.9 * 0.73 / 5 = 0.131 per generator (computed)
  kappa_init = ln(0.131) = -2.03 (stored as learnable parameter)

Initialize neural network weights (Xavier/Glorot)
Initialize gamma = 0.1
```

### Phase 4: Training (500 epochs, all learnable parameters update)

```
For each epoch:
  For each batch of scenarios:
    1. Sample delays: tau_i ~ LogNormal(50, 20), clipped to [5, 500]
    2. Forward pass through GNN encoders + attention + decoder
    3. Compute losses: L_OPF + L_QoS + L_stab + alpha * L_couple
    4. Backward pass: compute gradients
    5. Update all learnable parameters (K_i, gamma, NN weights)

After 500 epochs:
  K = [0.142, 0.098, 0.121, 0.167, 0.109]  (learned, no longer equal)
  gamma = 0.083 (learned)
  All NN weights converged
```

### Phase 5: Deployment (real-time, learnable parameters frozen)

```
Every control cycle (~50 ms):
  1. Receive current measurements: P, Q, V, theta, omega (dynamic)
  2. Receive current delays: tau_1=45ms, tau_2=120ms, ... (dynamic)
  3. Feed through frozen model
  4. Output: generator dispatch + communication routing
  5. Stability check: rho = 0.73 - sum(K_i * tau_i / 500)
```

---

## 6. How Values Behave During Stability Transitions

This section traces a complete stable-to-danger-to-stable cycle with concrete numbers.

### Setup: IEEE 14-bus, 5 generators

**Fixed after training:**
```
|lambda_min(0)| = 0.73
K = [0.142, 0.098, 0.121, 0.167, 0.109]
tau_max = 500 ms for all generators
```

### t=0: Normal operation

```
tau = [50, 40, 60, 45, 55] ms

rho = 0.73 - (0.142*50/500 + 0.098*40/500 + 0.121*60/500
              + 0.167*45/500 + 0.109*55/500)
    = 0.73 - (0.0142 + 0.00784 + 0.01452 + 0.01503 + 0.01199)
    = 0.73 - 0.06358
    = 0.666

Status: SAFE (91% of baseline margin remaining)
```

### t=1: Network congestion building

```
tau = [200, 180, 220, 190, 210] ms    (4x increase)

rho = 0.73 - (0.142*200/500 + 0.098*180/500 + 0.121*220/500
              + 0.167*190/500 + 0.109*210/500)
    = 0.73 - (0.0568 + 0.03528 + 0.05324 + 0.06346 + 0.04578)
    = 0.73 - 0.25456
    = 0.475

Status: CAUTION (65% of baseline remaining, dropping fast)
```

### t=2: Peak congestion

```
tau = [450, 480, 460, 470, 490] ms    (near maximum)

rho = 0.73 - (0.142*450/500 + 0.098*480/500 + 0.121*460/500
              + 0.167*470/500 + 0.109*490/500)
    = 0.73 - (0.1278 + 0.09408 + 0.11132 + 0.15698 + 0.10682)
    = 0.73 - 0.597
    = 0.133

Status: DANGER (only 18% margin left, approaching instability)
```

### t=2.5: Partial congestion relief (Gen 4 still congested)

```
tau = [200, 150, 180, 490, 160] ms    (most recover, Gen 4 stuck)

rho = 0.73 - (0.142*200/500 + 0.098*150/500 + 0.121*180/500
              + 0.167*490/500 + 0.109*160/500)
    = 0.73 - (0.0568 + 0.0294 + 0.04356 + 0.16366 + 0.03488)
    = 0.73 - 0.3283
    = 0.402

Status: RECOVERING (55% margin, but Gen 4 still draining budget)
```

**Key insight:** Gen 4 alone contributes 0.164 to the delay penalty -- nearly half of the total 0.328. This is because $K_4 = 0.167$ is the largest coupling constant AND its delay is still at 490 ms. The optimizer would prioritize restoring Gen 4's communication first.

### t=3: Full recovery

```
tau = [100, 90, 110, 95, 105] ms     (all delays low)

rho = 0.73 - (0.142*100/500 + 0.098*90/500 + 0.121*110/500
              + 0.167*95/500 + 0.109*105/500)
    = 0.73 - (0.0284 + 0.01764 + 0.02662 + 0.03173 + 0.02289)
    = 0.73 - 0.12728
    = 0.603

Status: SAFE (83% of baseline margin)
```

### The Recovery Gap

While $\rho$ recovers instantly (it is a formula), the physical system does not:

```
t=2.0s  rho = 0.133    Generators oscillating, amplitude growing
t=2.1s  rho = 0.200    Delays dropping, but oscillations still large
t=2.2s  rho = 0.350    Bound is comfortable, but rotors still swinging
t=2.5s  rho = 0.402    Oscillation amplitude halving (damping kicks in)
t=3.0s  rho = 0.603    Oscillations mostly damped, approaching steady state
t=4.0s  rho = 0.603    System fully settled at new operating point
```

**Why the lag?** Three reasons:
1. **Rotor inertia:** A 500 MW turbine rotor weighs hundreds of tonnes. Once swinging, it takes several swing cycles (1--2 seconds each) to damp out.
2. **Stored oscillation energy:** During the danger period, kinetic energy built up in rotor oscillations. Even after stability is restored, that energy must dissipate through damping.
3. **Control correction transient:** When delays drop, generators start receiving fresh control signals. Their first "correct" action is to correct for the mess created during the congested period, causing a transient overshoot.

---

## 7. Special Values and Why They Are Chosen

### Safety factor $s = 0.9$

**Purpose:** Leave a 10% reserve when initializing K values.

**Why 0.9 specifically?** If $s = 1.0$, then at initialization with all generators at maximum delay, $\rho = 0$ exactly. The system starts at the edge of instability, and any noise in the gradient could push it into the unstable region, causing training to diverge. Setting $s = 0.9$ ensures $\rho_{\text{init}} = 0.1 \cdot |\lambda_{\min}(0)| > 0$, providing a positive margin at initialization that stabilizes early training.

**What if $s$ were smaller?** At $s = 0.5$, the initial K values are too small, meaning the bound is very loose and the coupling loss gradients are weak -- training takes longer to converge. The K-init sensitivity experiment confirms that $s \in [0.7, 0.95]$ all converge to similar final K values, but $s = 0.9$ converges fastest.

### Maximum delay $\tau_{\max} = 500$ ms

**Source:** NERC Standard PRC-005 and IEEE C37.118.1 specify that SCADA-based wide-area control must operate within 500 ms end-to-end latency. Beyond this, the control system is considered non-functional.

**Why per-generator:** In the paper, $\tau_{\max,i}$ is allowed to differ per generator, but in all experiments $\tau_{\max,i} = 500$ ms is used uniformly. In practice, generators with dedicated fiber-optic links might have $\tau_{\max} = 100$ ms while generators on shared networks might have $\tau_{\max} = 1000$ ms.

### Coupling loss weight $\alpha = 1.0$

**Source:** Ablation study (Table II) testing $\alpha \in \{0, 0.1, 0.5, 1.0, 2.0, 5.0\}$.

**Why 1.0:** At $\alpha = 0$, the coupling loss is disabled and K values do not learn meaningful physical coupling. At $\alpha \gg 1$, the coupling loss dominates and the network optimizes for theory-data consistency at the expense of actual dispatch quality. At $\alpha = 1.0$, the coupling loss has equal weight to the stability hinge loss, producing the best overall stability margin.

### Embedding dimension $d = 128$

**Chosen by hyperparameter sweep.** $d = 64$ underperforms on larger grids (118-bus). $d = 256$ overfits on smaller grids (14-bus). $d = 128$ is the best compromise across all five IEEE cases.

### GNN layers $L = 3$

**Why 3:** In a GNN, each layer allows information to propagate one hop in the graph. With $L = 3$ layers, each bus can "see" all buses within 3 hops. For typical IEEE test cases, the graph diameter is 5--10 hops, so 3 layers capture most local structure without over-smoothing (a problem where too many layers cause all node embeddings to become identical).

### Attention heads $H = 8$

**Standard in transformer literature.** Each head can specialize in different types of relationships (e.g., one head focuses on electrically close buses, another on generators in the same area, another on buses with similar load profiles). More heads allow more specialization; fewer heads force each head to be more general.

---

## 8. Summary: From Raw Data to Real-Time Decision

```
IEEE case file          -->  N, n_g, Z_ij, H_i, D_i           [GIVEN]
     |
     v
Power flow solver       -->  V_i, theta_i, P_i, Q_i           [COMPUTED]
     |
     v
Jacobian assembly       -->  J(0), lambda_min(0), V, cond(V)  [COMPUTED]
     |
     v
Auto-scaling formula    -->  K_init = 0.9 * |lambda_min| / n_g [COMPUTED]
     |
     v
Training (500 epochs)   -->  K_i (learned), gamma, NN weights  [LEARNED]
     |
     v
Deployment (real-time)  -->  tau_i (measured) --> rho(tau)      [DYNAMIC]
                         -->  x_E, x_I (measured) --> dispatch   [DYNAMIC]
```

Each level builds on the previous one. Nothing can be skipped:
- Without the IEEE case file, there is no grid to study
- Without power flow, there is no operating point to linearize around
- Without eigenvalues, there is no baseline stability to budget
- Without auto-scaling, the K initialization is arbitrary and fragile
- Without training, the K values are rough estimates instead of learned physical constants
- Without real-time measurements, the model has no input to act on

---

## Glossary of Value-Related Terms

| Term | Meaning |
|------|---------|
| Given constant | A value specified in the IEEE test case file, fixed by the benchmark |
| Computed constant | A value derived from physics equations, computed once per grid |
| Learnable parameter | A value initialized and then optimized during training |
| Dynamic variable | A value that changes in real time during deployment |
| Operating point | The steady-state solution around which the system is linearized |
| Eigenvalue decomposition | Factoring a matrix into eigenvalues (how fast modes damp) and eigenvectors (which generators participate in each mode) |
| Auto-scaled initialization | Computing K_init from grid properties so training starts at a sensible point regardless of grid size |
| Frozen parameter | A learnable parameter that is no longer updated (after training ends) |
| Stability budget | The total amount of delay penalty all generators can collectively contribute before the margin reaches zero: $|\lambda_{\min}(0)|$ |
| Recovery gap | The time lag between the mathematical bound recovering (instant) and the physical system settling (seconds) |

---

*Last updated: 2026-02-13*
*Companion to: IEEE Trans. Smart Grid -- Learnable Delay-Stability Coupling for Smart Grid Communication Networks*
*See also: `theorem_explained.md`, `contributions_explained.md`*
