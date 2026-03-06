# Architecture Diagrams for Energy-Information Co-Optimization
# Important Note
``To the best of our knowledge, this is the first work to integrate learnable delay-stability coupling constants within a GNN-based joint energy-communication optimization framework, enabling end-to-end learning of physics-constrained stability margins.``

## 1. Simplified Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    JOINT ENERGY-INFORMATION OPTIMIZER                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   INPUTS    │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                 ▼
           ┌───────────────┐                ┌───────────────┐
           │  Energy Grid  │                │ Communication │
           │    State      │                │    Network    │
           │  (P,Q,V,θ,ω)  │                │   (τ,R,B)     │
           └───────┬───────┘                └───────┬───────┘
                   │                                │
                   ▼                                ▼
           ┌───────────────┐                ┌───────────────┐
           │   GNN-E       │                │   GNN-I       │
           │  (3 layers)   │                │  (3 layers)   │
           └───────┬───────┘                └───────┬───────┘
                   │                                │
                   │         h_energy              │  h_comm
                   │                                │
                   └────────────┬──────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │  HIERARCHICAL FUSION   │
                   │  (Cross-Domain Attn)   │
                   └───────────┬────────────┘
                               │
                               │  h_fused
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────┐ ┌────────────────┐
     │ Control Action │ │ Learnable  │ │   Stability    │
     │    Decoder     │ │     K      │ │    Margin      │
     │      u(t)      │ │ (auto-init)│ │      ρ(τ)      │
     └────────┬───────┘ └─────┬──────┘ └────────┬───────┘
              │               │                 │
              └───────────────┼─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    OUTPUTS      │
                    │  u, ρ, K, h_E   │
                    └─────────────────┘
```

---

## 2. Simplified Block Diagram (Publication-Ready)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────┐      ┌──────────────┐      ┌─────────────────┐    │
│  │ Energy  │      │    Dual      │      │   Hierarchical  │    │
│  │  State  │─────▶│    GNN       │─────▶│     Fusion      │    │
│  │ x_E     │      │  Encoders    │      │    Attention    │    │
│  └─────────┘      └──────────────┘      └────────┬────────┘    │
│                          ▲                       │             │
│  ┌─────────┐             │                       ▼             │
│  │  Comm   │             │              ┌─────────────────┐    │
│  │  State  │─────────────┘              │    Stability    │    │
│  │ x_I     │                            │    Computer     │───▶│ ρ(τ)
│  └─────────┘                            │  (Auto-scaled K)│    │
│                                         └────────┬────────┘    │
│                                                  │             │
│  ┌─────────┐                            ┌────────▼────────┐    │
│  │   τ     │───────────────────────────▶│    Control      │    │
│  │ (delay) │                            │    Decoder      │───▶│ u(t)
│  └─────────┘                            └─────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Architecture (Full Component View)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         JOINT OPTIMIZER ARCHITECTURE                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUTS                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Energy Features (x_E)              Communication Features (x_I)                │
│   ┌─────────────────────┐            ┌─────────────────────┐                    │
│   │ • P (Active Power)  │            │ • τ (Delay)         │                    │
│   │ • Q (Reactive Power)│            │ • R (Rate)          │                    │
│   │ • V (Voltage Mag)   │            │ • B (Buffer Size)   │                    │
│   │ • ω (Frequency)     │            └─────────────────────┘                    │
│   │ • θ (Phase Angle)   │                                                        │
│   └─────────────────────┘                                                        │
│                                                                                  │
│   Graph Structure                    Stability Parameters                        │
│   ┌─────────────────────┐            ┌─────────────────────┐                    │
│   │ • edge_index        │            │ • τ (per generator) │                    │
│   │ • impedance_matrix  │            │ • τ_max             │                    │
│   └─────────────────────┘            │ • λ_min(0)          │                    │
│                                      └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DUAL GNN ENCODER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌────────────────────────────┐    ┌────────────────────────────┐              │
│   │      ENERGY GNN            │    │    COMMUNICATION GNN       │              │
│   │                            │    │                            │              │
│   │   Input: [N, 5]            │    │   Input: [N, 3]            │              │
│   │         ↓                  │    │         ↓                  │              │
│   │   ┌──────────────┐         │    │   ┌──────────────┐         │              │
│   │   │ Linear(5→128)│         │    │   │ Linear(3→128)│         │              │
│   │   └──────┬───────┘         │    │   └──────┬───────┘         │              │
│   │          ↓                 │    │          ↓                 │              │
│   │   ┌──────────────┐         │    │   ┌──────────────┐         │              │
│   │   │  GATConv ×3  │         │    │   │  GATConv ×3  │         │              │
│   │   │  (8 heads)   │         │    │   │  (8 heads)   │         │              │
│   │   └──────┬───────┘         │    │   └──────┬───────┘         │              │
│   │          ↓                 │    │          ↓                 │              │
│   │   Output: [N, 128]         │    │   Output: [N, 128]         │              │
│   │                            │    │                            │              │
│   └────────────┬───────────────┘    └───────────┬────────────────┘              │
│                │                                │                                │
│                │  h_energy [N, 128]             │  h_comm [N, 128]               │
│                │                                │                                │
│                └────────────────┬───────────────┘                                │
│                                 │                                                │
│                                 ▼                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HIERARCHICAL FUSION MODULE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Input: h_energy [B, N, 128], h_comm [B, N, 128]                                │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    CROSS-DOMAIN ATTENTION                        │           │
│   │                                                                  │           │
│   │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │           │
│   │   │   Energy    │     │   Comm      │     │    Comm     │       │           │
│   │   │   Queries   │     │   Keys      │     │   Values    │       │           │
│   │   │  W_Q·h_E    │     │  W_K·h_I    │     │  W_V·h_I    │       │           │
│   │   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │           │
│   │          │                   │                   │               │           │
│   │          └───────────┬───────┘                   │               │           │
│   │                      ▼                           │               │           │
│   │            ┌─────────────────┐                   │               │           │
│   │            │ Attention Scores│                   │               │           │
│   │            │ softmax(QK^T/√d │                   │               │           │
│   │            │  + M_phys       │                   │               │           │
│   │            │  + M_causal)    │                   │               │           │
│   │            └────────┬────────┘                   │               │           │
│   │                     │                            │               │           │
│   │                     └────────────┬───────────────┘               │           │
│   │                                  ▼                               │           │
│   │                        ┌─────────────────┐                       │           │
│   │                        │  Attention × V  │                       │           │
│   │                        │   h_cross_E     │                       │           │
│   │                        └─────────────────┘                       │           │
│   │                                                                  │           │
│   │   (Same process for h_cross_I with reversed Q/K)                 │           │
│   │                                                                  │           │
│   │   Physics mask: M_phys = -γ·Z_ij/Z_max (adaptive γ)             │           │
│   │   Causal mask: M_causal for DAG structure                        │           │
│   │                                                                  │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    ELEMENT-WISE FUSION                           │           │
│   │                                                                  │           │
│   │   h_fused = MLP([h_energy; h_comm; h_energy ⊙ h_comm])          │           │
│   │                                                                  │           │
│   │   ┌────────────┐    ┌────────────┐    ┌────────────────┐        │           │
│   │   │ Concat     │    │  Linear    │    │    Output      │        │           │
│   │   │ [B,N,384]  │───▶│  384→256   │───▶│   [B,N,256]    │        │           │
│   │   └────────────┘    └────────────┘    └────────────────┘        │           │
│   │                                                                  │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
│   Output: h_fused [B, N, 256]                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   ▼                      ▼                      ▼
┌───────────────────────────┐ ┌─────────────────────┐ ┌───────────────────────────┐
│    CONTROL DECODER        │ │   LEARNABLE K       │ │   STABILITY COMPUTER      │
├───────────────────────────┤ ├─────────────────────┤ ├───────────────────────────┤
│                           │ │                     │ │                           │
│ Input: h_fused [B,N,256]  │ │ K = exp(log_K)      │ │ ρ(τ) = λ_min(0) -         │
│                           │ │                     │ │   Σ K_i·τ_i/τ_max         │
│   ┌─────────────────┐     │ │ ┌─────────────────┐ │ │   - Σ K_i^(2)·(τ/τ_max)² │
│   │  Global Pooling │     │ │ │ nn.Parameter    │ │ │   (Padé-2 correction)     │
│   │    mean(dim=1)  │     │ │ │ [n_generators]  │ │ │                           │
│   └────────┬────────┘     │ │ │ init: auto-     │ │ │ ┌─────────────────────┐   │
│            ↓              │ │ │ scaled per grid │ │ │ │ Stability Margin    │   │
│   ┌─────────────────┐     │ │ └────────┬────────┘ │ │ │ Computation         │   │
│   │  Linear 256→256 │     │ │          │          │ │ │                     │   │
│   │  + LayerNorm    │     │ │          ▼          │ │ │  • λ_min(0): base   │   │
│   │  + GELU         │     │ │ ┌────────────────┐  │ │ │  • K: coupling      │   │
│   └────────┬────────┘     │ │ │ Auto-Init:     │  │ │ │  • τ: delays        │   │
│            ↓              │ │ │ k_init =       │  │ │ │  • τ_max: max delay │   │
│   ┌─────────────────┐     │ │ │ safety_factor  │  │ │ └─────────────────────┘   │
│   │  Linear 256→n_g │     │ │ │ * |λ_min|      │  │ │                           │
│   │  + Softplus     │     │ │ │ / n_gen        │  │ │ Output: ρ [B]             │
│   └────────┬────────┘     │ │ │ (sf=0.9)       │  │ │         (stability)       │
│            ↓              │ │ └────────────────┘  │ │                           │
│   Output: u [B, n_gen]    │ │                     │ │ ρ > 0 → STABLE            │
│   (control actions)       │ │ Output: K [n_gen]   │ │ ρ < 0 → UNSTABLE          │
│                           │ │                     │ │                           │
└───────────────────────────┘ └─────────────────────┘ └───────────────────────────┘
```

---

## 4. Training Loss Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              JOINT LOSS FUNCTION                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   L = L_OPF + L_QoS + L_stab + α·L_couple        (α = 1.0, from ablation)      │
│                                                                                  │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐     │
│   │   L_OPF             │  │   L_QoS             │  │   L_stab            │     │
│   │                     │  │                     │  │                     │     │
│   │ Optimal power flow  │  │ Quality of service  │  │ Stability loss      │     │
│   │ Generation cost +   │  │ Latency threshold   │  │ E[max(0, -ρ(τ))]   │     │
│   │ voltage/frequency   │  │ violations +        │  │                     │     │
│   │ constraint penalty  │  │ bandwidth penalty   │  │ Hinge on margin     │     │
│   └─────────────────────┘  └─────────────────────┘  └─────────────────────┘     │
│                                                                                  │
│   ┌─────────────────────┐                                                        │
│   │   L_couple           │                                                       │
│   │                     │                                                        │
│   │ Coupling alignment  │                                                        │
│   │ ||ρ_emp - ρ_theo||² │                                                        │
│   │ Theory-data match   │                                                        │
│   └─────────────────────┘                                                        │
│                                                                                  │
│   Note: Only L_couple is weighted (by α). The other three terms have             │
│   unit weight to balance energy, communication, and stability equally.           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow Diagram

```
                               IEEE Test Cases
                          (14/30/39/57/118-bus)
                            + Synthetic 10K-bus
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Generation   │
                         │  (1000 scenarios)   │
                         └──────────┬──────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐     ┌─────────────────┐
│  Energy State   │      │  Communication      │     │  Stability      │
│  Generation     │      │  Delays             │     │  Labels         │
│                 │      │                     │     │                 │
│  P,Q,V,θ,ω      │      │  τ ~ Weibull,       │     │  λ_min, ρ       │
│  from power     │      │  Lognormal, Pareto, │     │  (ground truth) │
│  flow           │      │  Gamma, Exponential │     │                 │
└────────┬────────┘      └──────────┬──────────┘     └────────┬────────┘
         │                          │                         │
         └──────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    DataLoader       │
                         │  (batch_size=32)    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   JointOptimizer    │
                         │     Forward         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    JointLoss        │
                         │    Backward         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   AdamW Optimizer   │
                         │   (lr=1e-4)         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Trained Model     │
                         │   • Learned K       │
                         │   • Encoder weights │
                         │   • Decoder weights │
                         └─────────────────────┘

Hardware: GPU/CUDA training (shared server, 1-2 GPUs)
```

---

## 6. K Auto-Scaling Initialization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       K_INIT AUTO-SCALING (Per Grid)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   FORMULA:  k_init = safety_factor * |λ_min(0)| / n_gen                          │
│             (safety_factor = 0.9 by default)                                     │
│                                                                                  │
│   PURPOSE:  Ensures initial K values produce a meaningful (positive) stability   │
│             margin at initialization, regardless of grid size. The old fixed     │
│             K=0.1 caused instability for large grids (118+ buses) where          │
│             n_gen * 0.1 > |λ_min(0)|, collapsing the margin below zero.          │
│                                                                                  │
│   EXAMPLE VALUES:                                                                │
│   ┌──────────────┬───────────┬──────────┬──────────────┐                        │
│   │   IEEE Case  │  n_gen    │ |λ_min|  │  k_init      │                        │
│   ├──────────────┼───────────┼──────────┼──────────────┤                        │
│   │   14-bus     │     5     │  ~0.73   │  ~0.131      │                        │
│   │   30-bus     │     6     │  ~0.52   │  ~0.078      │                        │
│   │   39-bus     │    10     │  ~0.59   │  ~0.053      │                        │
│   │   57-bus     │     7     │  ~0.47   │  ~0.060      │                        │
│   │  118-bus     │    54     │  ~0.38   │  ~0.006      │                        │
│   │  10K synth   │   ~2000   │  ~0.31   │  ~0.00014    │                        │
│   └──────────────┴───────────┴──────────┴──────────────┘                        │
│                                                                                  │
│   IMPLEMENTATION (coupling.py):                                                  │
│     def compute_k_init_scale(n_generators, lambda_min_0, safety_factor=0.9):     │
│         abs_lambda = abs(lambda_min_0)                                            │
│         k_init = safety_factor * abs_lambda / n_generators                       │
│         return k_init                                                            │
│                                                                                  │
│   K_init sensitivity experiments confirm convergence to similar final K          │
│   regardless of initialization on IEEE 39/57/118, proving meaningful             │
│   learning of physical coupling constants.                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Comparison: JointOptimizer vs Baselines

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          BASELINE ARCHITECTURES                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

B1: Sequential              B3: GNN Only              B7: No Coupling
─────────────               ──────────               ────────────────

┌─────────┐                 ┌─────────┐              ┌─────────┐
│ Energy  │                 │ Energy  │              │ Energy  │
│ Encoder │                 │   GNN   │              │ Encoder │
└────┬────┘                 └────┬────┘              └────┬────┘
     │                           │                       │
     ▼                           │                       │
┌─────────┐                      │                       ▼
│  Info   │                      │                  ┌─────────┐
│ Encoder │                      │                  │  Info   │
└────┬────┘                      │                  │ Encoder │
     │                           │                  └────┬────┘
     ▼                           │                       │
┌─────────┐                      │                       ▼
│ Concat  │                      │                  ┌─────────┐
└────┬────┘                      │                  │ Concat  │
     │                           │                  │ (no K)  │
     ▼                           ▼                  └────┬────┘
┌─────────┐                 ┌─────────┐                  │
│ Decoder │                 │ Decoder │                  ▼
│(auto K) │                 │(auto K) │             ┌─────────┐
└─────────┘                 └─────────┘             │ Decoder │
                                                    │ K=fixed │
                                                    └─────────┘


B8: HeteroGNN               B9: DeepOPF
──────────────               ──────────

┌─────────┐ ┌─────────┐     ┌───────────────┐
│ Energy  │ │  Comm   │     │   Load Vector │
│ Nodes   │ │  Nodes  │     │   (P_load)    │
└────┬────┘ └────┬────┘     └───────┬───────┘
     │           │                  │
     └─────┬─────┘                  ▼
           ▼                  ┌───────────────┐
    ┌────────────┐            │ MLP [400×3]   │
    │ Type Embed │            │ (Pan et al.)  │
    │ + Unified  │            └───────┬───────┘
    │ GATConv ×3 │                    │
    └─────┬──────┘                    ▼
          │                   ┌───────────────┐
     ┌────┴────┐              │  P, Q Output  │
     ▼         ▼              │ (no delay,    │
  ┌─────┐  ┌─────┐           │  no coupling) │
  │ h_E │  │ h_I │           └───────────────┘
  └──┬──┘  └──┬──┘
     └────┬───┘
          ▼
     ┌─────────┐
     │ Decoder │
     └─────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                     JOINTOPTIMIZER (PROPOSED)                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────┐     ┌─────────┐
│ Energy  │     │  Info   │
│  GNN    │     │  GNN    │
└────┬────┘     └────┬────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌─────────────────┐
    │  Hierarchical   │
    │  Cross-Attn     │
    │  Fusion         │
    │  (Adaptive γ)   │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌───────┐ ┌─────┐ ┌───────┐
│Control│ │Learn│ │Stabil │
│Decoder│ │  K  │ │Margin │
│       │ │auto │ │Padé-2 │
└───────┘ └─────┘ └───────┘
```

---

## 8. Key Equations Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              KEY EQUATIONS                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

THEOREM 1 — STABILITY MARGIN (1st-order Padé):
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ρ(τ) ≥ |λ_min(0)| - Σᵢ Kᵢ · τᵢ / τ_max,i        │
│                                                     │
│   where:                                            │
│   • λ_min(0) : minimum eigenvalue (base stability) │
│   • Kᵢ       : learned coupling constant           │
│   • τᵢ       : communication delay for generator i │
│   • τ_max,i  : maximum delay normalization         │
│                                                     │
│   STABLE if ρ > 0                                   │
│   UNSTABLE if ρ < 0                                 │
│                                                     │
│   Conservative upper bound — never underestimates   │
│   instability risk (safety-critical property)       │
│                                                     │
└─────────────────────────────────────────────────────┘

COROLLARY 1 — HIGHER-ORDER BOUND (2nd-order Padé):
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ρ(τ) ≥ |λ_min(0)|                                 │
│         - Σᵢ Kᵢ · τᵢ/τ_max,i                        │
│         - Σᵢ Kᵢ⁽²⁾ · (τᵢ/τ_max,i)²                 │
│                                                     │
│   Kᵢ⁽²⁾ = exp(κᵢ⁽²⁾) learned alongside Kᵢ          │
│   Reduces bound gap from ~18% to ~3% at τ=500ms    │
│                                                     │
└─────────────────────────────────────────────────────┘

K AUTO-SCALING:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   k_init = safety_factor * |λ_min(0)| / n_gen       │
│                                                     │
│   safety_factor = 0.9 (default)                     │
│   Ensures ρ_init > 0 for all grid sizes             │
│                                                     │
└─────────────────────────────────────────────────────┘

OBSERVATION 1 — DOMAIN SEPARATION (formerly Theorem 2):
┌─────────────────────────────────────────────────────┐
│                                                     │
│   I(h_E; h_I) << H(h_E)                             │
│                                                     │
│   Mutual information between energy and comm        │
│   embeddings is negligible, indicating the dual-    │
│   domain GNN architecture learns approximately      │
│   independent representations.                      │
│                                                     │
│   Measured: I(h_E; h_I) ~ 10⁻⁴ nats across         │
│   all IEEE test cases.                              │
│                                                     │
└─────────────────────────────────────────────────────┘

CROSS-DOMAIN ATTENTION:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Attention(Q,K,V) = softmax(QK^T/√d               │
│                              + M_phys              │
│                              + M_causal) · V       │
│                                                     │
│   A_cross = softmax(W_Q·h_E · (W_K·h_I)^T / √d    │
│                    + M_phys)                        │
│                                                     │
│   M_phys = -γ·Z_ij/Z_max  (γ adaptive, learned)    │
│                                                     │
└─────────────────────────────────────────────────────┘

JOINT LOSS:
┌─────────────────────────────────────────────────────┐
│                                                     │
│   L = α·L_coupling + β·L_control                    │
│     + γ·L_contrastive + δ·L_physics                │
│                                                     │
│   L_coupling = -ρ + ReLU(margin - ρ)               │
│                                                     │
│   L_contrastive = InfoNCE(h_E, h_I)                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 9. Stress Test Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STRESS TEST PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────┐                                                            │
│   │  Trained Model  │ (from normal training on IEEE case)                        │
│   └────────┬────────┘                                                            │
│            │                                                                     │
│            ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐            │
│   │              STRESSED SCENARIO GENERATOR                         │            │
│   │                                                                  │            │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │            │
│   │  │  High Load   │ │  N-1         │ │  Extreme     │             │            │
│   │  │  1.0x→1.2x   │ │ Contingency  │ │  Delay       │             │            │
│   │  │  (5% steps)  │ │ (top-5 lines)│ │  100→1000ms  │             │            │
│   │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘             │            │
│   │         │                │                │                      │            │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │            │
│   │  │  Heavy-Tail  │ │  Combined    │ │  Degradation │             │            │
│   │  │  Delays      │ │  Stress      │ │  Sweep       │             │            │
│   │  │  (Pareto)    │ │ (load+delay+ │ │  (heatmap)   │             │            │
│   │  │              │ │  N-1)        │ │              │             │            │
│   │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘             │            │
│   │         │                │                │                      │            │
│   └─────────┼────────────────┼────────────────┼──────────────────────┘            │
│             └────────────────┼────────────────┘                                  │
│                              ▼                                                   │
│              ┌───────────────────────────────┐                                   │
│              │       EVALUATE (×5 seeds)     │                                   │
│              │                               │                                   │
│              │  For each (scenario, model):  │                                   │
│              │   • stability_rate (%)         │                                   │
│              │   • mean_margin (ρ)            │                                   │
│              │   • margin distribution        │                                   │
│              │   • degradation from baseline  │                                   │
│              └───────────────┬───────────────┘                                   │
│                              │                                                   │
│                              ▼                                                   │
│              ┌───────────────────────────────┐                                   │
│              │   STATISTICAL COMPARISON      │                                   │
│              │                               │                                   │
│              │  • Wilcoxon signed-rank test   │                                   │
│              │  • Holm-Sidak correction       │                                   │
│              │  • Cohen's d effect size       │                                   │
│              │  • Friedman + Nemenyi post-hoc │                                   │
│              └───────────────────────────────┘                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Transfer Learning Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING WORKFLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   PHASE 1: TRAIN ON SOURCE GRID                                                  │
│   ┌──────────────────────────────────────────┐                                  │
│   │  Source: IEEE 39-bus (or 57-bus)         │                                  │
│   │  Full training: 200 epochs              │                                  │
│   │  1000 scenarios, lr=1e-4                │                                  │
│   │  → Pretrained weights (GNN + Attn + K)  │                                  │
│   └────────────────────┬─────────────────────┘                                  │
│                        │                                                         │
│                        ▼                                                         │
│   PHASE 2: TRANSFER TO TARGET GRID                                               │
│   ┌──────────────────────────────────────────┐                                  │
│   │  Target: IEEE 118-bus                    │                                  │
│   │                                          │                                  │
│   │  ┌────────────────────────────────────┐  │                                  │
│   │  │ Mode A: Zero-Shot (0 epochs)      │  │                                  │
│   │  │  Freeze all. Evaluate directly.   │  │                                  │
│   │  │  Expected: ~89% of full perf.     │  │                                  │
│   │  └────────────────────────────────────┘  │                                  │
│   │                                          │                                  │
│   │  ┌────────────────────────────────────┐  │                                  │
│   │  │ Mode B: Few-Shot (20 epochs, 1%)  │  │                                  │
│   │  │  Freeze GNN layers.               │  │                                  │
│   │  │  Fine-tune Attention + Decoder.   │  │                                  │
│   │  │  Only 1% of target data.          │  │                                  │
│   │  └────────────────────────────────────┘  │                                  │
│   │                                          │                                  │
│   │  ┌────────────────────────────────────┐  │                                  │
│   │  │ Mode C: Few-Shot (50 epochs, 10%) │  │                                  │
│   │  │  Freeze GNN layers.               │  │                                  │
│   │  │  Fine-tune Attention + Decoder.   │  │                                  │
│   │  └────────────────────────────────────┘  │                                  │
│   │                                          │                                  │
│   │  ┌────────────────────────────────────┐  │                                  │
│   │  │ Mode D: Full Fine-Tune (100 ep)   │  │                                  │
│   │  │  Unfreeze all layers. 100% data.  │  │                                  │
│   │  └────────────────────────────────────┘  │                                  │
│   │                                          │                                  │
│   │  ┌────────────────────────────────────┐  │                                  │
│   │  │ Baseline: Train from Scratch       │  │                                  │
│   │  │  200 epochs, 100% data, no pretr. │  │                                  │
│   │  └────────────────────────────────────┘  │                                  │
│   └──────────────────────────────────────────┘                                  │
│                                                                                  │
│   TRANSFER EXPERIMENTS:                                                          │
│     39 → 118  (small to large)                                                   │
│     57 → 118  (medium to large)                                                  │
│    118 → 39   (large to small)                                                   │
│                                                                                  │
│   KEY INSIGHT: GNN weights learn general power-system graph structure;            │
│   only Attention + K need adaptation for new grid topology.                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Padé Approximation Order Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PADÉ ORDER ANALYSIS PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   For each IEEE case and delay range (50-500ms):                                 │
│                                                                                  │
│   ┌──────────────────────────┐                                                  │
│   │  DDE System:             │                                                  │
│   │  dx/dt = A·x(t)         │                                                  │
│   │        + B·x(t-τ)       │                                                  │
│   └────────────┬─────────────┘                                                  │
│                │                                                                 │
│      ┌─────────┼──────────┬────────────────┐                                    │
│      ▼         ▼          ▼                ▼                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐                              │
│  │ Exact  │ │ Padé-1 │ │ Padé-2 │ │ Time-Domain  │                              │
│  │ (DDE   │ │ Linear │ │ Quad.  │ │ Simulation   │                              │
│  │  eig)  │ │ bound  │ │ bound  │ │ (Euler+buf)  │                              │
│  └───┬────┘ └───┬────┘ └───┬────┘ └──────┬───────┘                              │
│      │          │          │             │                                       │
│      └──────────┼──────────┼─────────────┘                                      │
│                 ▼                                                                 │
│       ┌─────────────────────────────────────┐                                   │
│       │        COMPARISON FIGURE             │                                   │
│       │                                      │                                   │
│       │  ρ(τ)↑                               │                                   │
│       │  0.4│  ----____                      │                                   │
│       │     │  ===------____     --- Exact   │                                   │
│       │  0.3│  +++--------____   === Padé-1  │                                   │
│       │     │               ---  +++ Padé-2  │                                   │
│       │  0.2│                --  ··· Sim     │                                   │
│       │     │________________________→ τ(ms) │                                   │
│       │     0    100   200   300   400   500 │                                   │
│       │                                      │                                   │
│       │  Error at 200ms: Padé-1 ~4%, Padé-2 │                                   │
│       │  ~0.8%                               │                                   │
│       │  Error at 500ms: Padé-1 ~18%, Padé-2│                                   │
│       │  ~3%                                 │                                   │
│       └─────────────────────────────────────┘                                   │
│                                                                                  │
│   DECISION: Journal uses Padé-2 (Corollary 1). Padé-1 retained for              │
│   conference-version comparison only.                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Convergence and Scalability Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   CONVERGENCE ANALYSIS DIMENSIONS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   DIMENSION 1: Scenarios vs Performance                                          │
│   ┌───────────────────────────────────────────┐                                 │
│   │  scenarios: [100, 200, 500, 1000, 2000]   │                                 │
│   │  Question: How much data is sufficient?    │                                 │
│   └───────────────────────────────────────────┘                                 │
│                                                                                  │
│   DIMENSION 2: Epochs vs Performance                                             │
│   ┌───────────────────────────────────────────┐                                 │
│   │  epochs: [10, 25, 50, 100, 200, 500]      │                                 │
│   │  Question: When does training converge?    │                                 │
│   └───────────────────────────────────────────┘                                 │
│                                                                                  │
│   DIMENSION 3: Grid Size vs Resources                                            │
│   ┌──────────────────────────────────────────────────────────┐                  │
│   │  IEEE Case  │  n_bus  │  Train(s)  │  Infer(ms)  │ Mem  │                  │
│   │─────────────┼─────────┼────────────┼─────────────┼──────│                  │
│   │  14-bus     │    14   │   ~30      │   ~0.5      │ Low  │                  │
│   │  30-bus     │    30   │   ~45      │   ~0.7      │ Low  │                  │
│   │  39-bus     │    39   │   ~60      │   ~0.9      │ Low  │                  │
│   │  57-bus     │    57   │   ~90      │   ~1.2      │ Med  │                  │
│   │  118-bus    │   118   │  ~240      │   ~2.5      │ Med  │                  │
│   │  10K synth  │ 10000   │ ~3600      │  ~25.0      │ High │                  │
│   └──────────────────────────────────────────────────────────┘                  │
│                                                                                  │
│   DIMENSION 4: Model Compression                                                 │
│   ┌───────────────────────────────────────────┐                                 │
│   │  embed_dim: [16, 32, 64, 128, 256]        │                                 │
│   │  Question: Minimum model size for ≥95%    │                                 │
│   │  of full performance?                     │                                 │
│   └───────────────────────────────────────────┘                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
