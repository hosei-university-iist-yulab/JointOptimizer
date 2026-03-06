# Comprehensive Experimental Results & Interpretation
## IEEE Trans. Smart Grid: Energy-Information Co-Optimization

---

# Executive Summary

This document presents the experimental results and interpretive guide for the journal extension of our energy-information co-optimization framework. The **JointOptimizer** is evaluated against 7 baseline architectures across 5 IEEE power system test cases (14, 30, 39, 57, 118 buses), covering a comprehensive suite of experiments: main comparison, ablation study, stress testing, N-1 contingency, delay distribution robustness, convergence analysis, model compression, transfer learning, Theorem 1 validation, and inference benchmarking.

**Key contributions validated experimentally:**
- Learnable coupling constants K that reduce conservative safety margins
- Physics-constrained hierarchical attention that outperforms generic architectures
- Theorem 1 (delay-stability bound) validated across all 5 IEEE cases
- Observation 1 (information-theoretic domain separation) confirmed through cross-domain mutual information analysis

All quantitative results are auto-generated into LaTeX tables in `docs/tables/` and publication-quality figures in `docs/figures/publication/`. This document describes **what each experiment measures, why it matters, and how to interpret the results** -- refer to the referenced tables and figures for the actual numerical values.

---

# Background: Understanding the Problem

## What Problem Are We Solving?

Modern power grids rely on **communication networks** to coordinate generators, renewable energy sources, and loads. However, these communication networks introduce **delays** (measured in milliseconds) that can destabilize the grid.

```
+--------------+     Communication      +--------------+
|   Control    | ---- Network --------+ |   Power      |
|   Center     |     (with delay t)     |   Grid       |
+--------------+                        +--------------+
       ^                                      |
       +------------ Measurements ------------+
                    (also delayed)
```

**The Challenge:** How do we control the power grid safely when our commands and measurements are delayed?

- **Traditional Approach:** Use very conservative safety margins (assume worst-case delays)
- **Our Approach:** Learn the precise relationship between delays and stability, allowing tighter (more efficient) operation

---

## Key Concepts Explained

### Stability Margin (rho) - "How Safe Is the Grid?"

The **stability margin rho** measures how far the grid is from becoming unstable.

| rho Value | Interpretation | Analogy |
|-----------|----------------|---------|
| rho > 0.3 | Very safe | Driving 30 mph in a 60 mph zone |
| rho ~ 0.2 | Moderately safe | Driving 45 mph in a 60 mph zone |
| rho ~ 0.1 | Getting risky | Driving 55 mph in a 60 mph zone |
| rho <= 0 | **UNSTABLE** | Exceeding the speed limit -- loss of control |

**Higher rho = Safer grid operation**

We report two values:
- **Mean rho:** Average safety margin across all test scenarios
- **Min rho:** Worst-case safety margin (most important for safety!)

### Coupling Constant (K) - "How Much Does Delay Hurt Stability?"

The **coupling constant K** quantifies how much communication delay degrades stability. Our core equation (Theorem 1) is:

```
rho(tau) = |lambda_min(0)| - SUM_i( K_i * tau_i / tau_max_i )
```

In plain English: *"Stability margin decreases linearly with communication delay, scaled by coupling constant K."*

**Lower K = We can operate closer to grid limits (more efficient)**

But K must be accurate! If we underestimate K, we could destabilize the grid. Our method **learns** the correct K from data.

### Auto-Scaled K Initialization

In the journal extension, K_init is **auto-scaled per grid size** rather than using a fixed value. The formula is:

```
k_init = safety_factor * |lambda_min| / n_gen
```

This ensures that the initial coupling constant is proportionate to the grid's intrinsic stability properties and the number of generators. For small grids with few generators, K_init will be larger per generator; for large grids (e.g., IEEE 118 with many generators), K_init is distributed across more coupling terms. This auto-scaling eliminates a manual hyperparameter and improves convergence across diverse grid sizes.

### What Does "Learning K" Mean?

Traditional control systems use a **fixed, conservative K** chosen by engineers to guarantee safety in all situations.

Our **JointOptimizer learns K from data**:
1. Start with auto-scaled K_init (proportional to grid characteristics)
2. Train on thousands of grid scenarios
3. Observe actual delay-stability relationship
4. Adjust K to match reality (typically a reduction from the conservative initial value)

**Result:** The learned K is less conservative than the initialization, allowing the grid to operate closer to its limits while remaining stable.

---

# Experimental Setup

## Test Systems

We test on five standard IEEE power system benchmarks, spanning a wide range of grid sizes:

| System | Buses | Generators | Complexity | Real-World Equivalent |
|--------|-------|------------|------------|----------------------|
| **IEEE 14-Bus** | 14 | 5 | Small | Small regional grid |
| **IEEE 30-Bus** | 30 | 6 | Small-Medium | IEEE reliability test system |
| **IEEE 39-Bus** | 39 | 10 | Medium | New England power system |
| **IEEE 57-Bus** | 57 | 7 | Medium-Large | Portion of US Midwest grid |
| **IEEE 118-Bus** | 118 | 54 | Large | Interconnected regional grid |

**Why five systems?** The conference paper used three (14, 39, 57). The journal extension adds IEEE 30 and IEEE 118 to demonstrate scalability and to stress-test the framework at both small and large scales. IEEE 118, with 54 generators, is particularly challenging for the coupling constant learning mechanism.

## Methods Compared

| Method | Description | Key Characteristic |
|--------|-------------|-------------------|
| **JointOptimizer (Ours)** | GNN + Hierarchical Attention + Learnable K | Learns coupling, uses physics constraints |
| B1: Sequential OPF+QoS | Solve power flow, then communication separately | No joint optimization |
| B2: MLP Joint | Simple neural network, joint objective | No graph structure awareness |
| B3: GNN Only | Graph neural network without attention | No hierarchical processing |
| B4: LSTM Joint | Recurrent neural network | Designed for sequences, not graphs |
| B5: CNN Joint | Convolutional neural network | Designed for images, not graphs |
| B6: Vanilla Transformer | Standard attention mechanism | No physics constraints |
| B7: Trans. No Coupling | Same as ours, but K fixed (not learned) | **Ablation: proves K-learning helps** |

**B7 is the most important comparison** -- it is identical to our method except it does not learn K. Any improvement over B7 directly measures the value of coupling constant learning.

Additional baselines used for inference benchmarking only (not in main comparison):
- **B8: HeteroGNN** -- Heterogeneous graph neural network with separate node types
- **B9: DeepOPF** -- Specialized OPF prediction network

## Training Configuration

| Parameter | Value | Why This Choice |
|-----------|-------|-----------------|
| Epochs | 100-500 (varies by experiment) | Convergence analysis determines sufficiency |
| Scenarios | 50-1000 (varies by experiment) | Convergence analysis determines minimum |
| Hardware | NVIDIA GPU (CUDA) | Required for transformer training at scale |
| Batch size | 32 | Balance between throughput and gradient stability |
| Seeds | 5 | All results reported as mean +/- std over 5 seeds |
| K initialization | Auto-scaled per grid | `k_init = safety_factor * |lambda_min| / n_gen` |

---

# Core Results

## Table I: Main Baseline Comparison

> **LaTeX source:** `docs/tables/table01_main_comparison.tex`
> **Figures:** `docs/figures/publication/fig_stability_margin_comparison.pdf`, `fig_k_learning_comparison.pdf`

### What It Measures

This is the central result table. For each of the 5 IEEE test cases, it reports the stability rate (%) and mean stability margin rho(tau) for the JointOptimizer and all 7 baselines, averaged over 5 random seeds.

### How to Read It

- **Stability rate (%):** Fraction of test scenarios where the grid remains stable. 100% means no instability under any tested condition.
- **rho(tau):** The stability margin (higher is better). Reported as mean +/- standard deviation across seeds.
- Compare each baseline's rho against the JointOptimizer row to see the improvement.
- Compare against B7 specifically to isolate the contribution of learned coupling constants.

### What to Look For

1. **Consistent improvement across all 5 cases** proves the method generalizes, not just overfits to a particular grid.
2. **Improvement magnitude may vary by grid size** -- grids with more generators and more complex topology may show larger or smaller gains depending on how much the conservative K_init overshoots.
3. **B7 vs JointOptimizer** directly quantifies the value of K-learning, since B7 is architecturally identical but does not learn K.
4. **Parameter efficiency** -- the JointOptimizer achieves superior margins with moderate parameter count compared to larger baselines (B3-GNN, B4-LSTM, B5-CNN).

### Visualization

- `fig_stability_margin_comparison.pdf`: Bar chart comparing rho across all models and cases
- `fig_k_learning_comparison.pdf`: Shows K_init vs K_final for each case
- `fig_radar_all_baselines.pdf`: Radar/spider plot comparing methods across multiple metrics
- `fig_improvement_heatmap.pdf`: Heatmap of relative improvement over baselines

---

## Table II: Ablation Study

> **LaTeX source:** `docs/tables/table02_ablation.tex`
> **Figure:** `docs/figures/publication/fig_ablation_study.pdf`

### What It Measures

The ablation study systematically disables individual components of the JointOptimizer to measure each one's contribution. Components tested:

1. **Coupling weight alpha** (0.0, 0.1, 0.5, 1.0, 2.0) -- controls how much the model prioritizes learning K
2. **Physics mask** (on/off) -- whether attention is constrained by grid topology
3. **Causal mask** (on/off) -- whether temporal causality is enforced
4. **Cross-attention** (on/off) -- whether energy and communication domains interact via cross-attention
5. **Contrastive loss** (on/off) -- whether domain separation is explicitly encouraged
6. **GNN layers** (1, 2, 3, 4) -- depth of graph neural network backbone

### How to Read It

- Results are averaged across all 5 IEEE cases (14-118).
- Each row shows rho(tau) and stability (%) when one component is modified.
- The baseline (all components enabled, default settings) is the JointOptimizer configuration used in Table I.

### What to Look For

1. **alpha = 0 vs alpha > 0:** The jump from alpha=0 (no K-learning) to alpha=0.1 shows whether coupling loss is necessary.
2. **Diminishing returns in alpha:** Whether alpha=1.0 vs alpha=2.0 shows saturation, indicating robustness to this hyperparameter.
3. **Physics/causal mask impact:** Whether physics-constrained attention provides measurable benefit over unconstrained attention.
4. **GNN depth sensitivity:** Whether deeper GNNs improve results or whether the task is solved with shallow message-passing.

---

## Table VI: Theorem 1 Validation (Delay-Stability Bound)

> **LaTeX source:** `docs/tables/table06_theorem1.tex`
> **Figure:** `docs/figures/publication/fig_theorem1_validation.pdf`
> **Per-case figures:** `fig_theorem1_all_models.pdf`, `fig_pade_analysis_case*.pdf`

### The Theorem

Theorem 1 states:
```
rho(tau) = |lambda_min(0)| - SUM_i( K_i * tau_i / tau_max_i )
```

*"Stability margin decreases linearly with communication delay, scaled by coupling constant K."*

### What It Measures

For each IEEE case, we sweep the communication delay from low to high values and compare:
- **Predicted rho:** Stability margin predicted by Theorem 1 using the learned K values
- **Empirical rho:** Stability margin measured from actual DDE (delay differential equation) simulation

This is reported at multiple delay points (e.g., 50ms, 100ms, 200ms, 300ms, 400ms, 500ms) across all 5 cases.

### How to Read It

- Each row is a delay value. Each case has two columns: predicted and empirical.
- The closer these two values are, the better the theorem fits reality.
- The stability rate column shows what fraction of scenarios remain stable at each delay.

### What to Look For

1. **Theory-practice match:** The gap between predicted and empirical rho should be negligible, validating the linear delay-stability model.
2. **Critical delay threshold:** The delay value at which stability begins to degrade (stability rate drops below 100%).
3. **Cross-case consistency:** Whether Theorem 1 holds equally well for small (14-bus) and large (118-bus) grids.
4. **Practical implication:** The maximum tolerable communication delay for each grid size, which informs network design requirements.

### Pade Approximation Analysis

Beyond the linear Theorem 1 bound, we also evaluate a **Pade-[2/2] rational approximant** for the delay-stability relationship, which can capture nonlinear effects at higher delays. Per-case Pade analysis figures are available for cases where this was computed.

---

## Observation 1: Information-Theoretic Domain Separation

> **Related to:** Table II (ablation of contrastive loss) and cross-domain MI analysis

### What It States

Observation 1 (previously called "Theorem 2" in early drafts) provides an information-theoretic characterization of when domain separation is beneficial. Specifically, the energy embedding h_E and communication embedding h_I should have low mutual information I(h_E; h_I), meaning the two domains learn complementary rather than redundant representations.

### What We Measure

- **Cross-Domain Mutual Information I(h_E; h_I):** Should be low (order of 10^-4 nats or less)
- **Energy Entropy H(h_E):** Complexity of energy embeddings
- **Communication Entropy H(h_I):** Complexity of communication embeddings
- **Attention Entropy H(Attn):** Randomness in attention patterns

### Why It Matters

- Low MI indicates effective domain separation -- each branch learns its own useful features.
- The coupling between domains happens through the learned K, not through entangled representations.
- Physics-constrained attention produces deterministic patterns based on grid topology, which is reflected in the attention entropy.

---

# Journal Extension Experiments

The following experiments are new additions for the IEEE Trans. Smart Grid journal extension, significantly expanding the scope beyond the original conference paper.

## Table III: Stress Test (Stressed Grid Conditions)

> **LaTeX source:** `docs/tables/table03_stress_test.tex`
> **Per-case figures:** `fig_stress_heatmap_case*.pdf`, `fig_stress_degradation_case*.pdf`
> **Aggregate figure:** `docs/figures/publication/fig_stress_heatmap.pdf`

### What It Measures

The stress test evaluates robustness under 11 degraded operating conditions, including:
- **Load scaling** (e.g., 110%, 120% of nominal load)
- **Generator outages** (reduced generation capacity)
- **Combined stressors** (simultaneous load increase + generation reduction, at moderate and severe levels)
- **Normal conditions** (baseline reference)

For each stress scenario, we report stability rate (%) for all 8 models across all 5 IEEE cases.

### How to Read It

- Each column is a stress scenario; each row is a model.
- 100% means all test scenarios remained stable under that stress condition.
- Values below 100% indicate that some scenarios became unstable under stress.

### What to Look For

1. **Differential resilience:** Under which stress conditions does the JointOptimizer maintain 100% stability while baselines begin to degrade?
2. **Severity gradient:** How gracefully does each method degrade as stress increases (moderate vs severe)?
3. **Case dependence:** Larger grids (118-bus) may show more sensitivity to stress due to tighter operating margins.

---

## Table VII: N-1 Contingency Analysis

> **LaTeX source:** `docs/tables/table07_n1_contingency.tex`
> **Per-case figures:** `fig_n1_contingency_case*.pdf`

### What It Measures

N-1 contingency is the standard reliability criterion in power systems: the grid must remain stable following the loss of any single transmission line. For each IEEE case, we:
1. Remove each line one at a time
2. Re-evaluate stability and the margin rho
3. Report the average and worst-case stability rate, plus the average margin degradation

### How to Read It

- **Avg. Stab. (%):** Average stability rate across all single-line outage scenarios
- **Worst Stab. (%):** Stability rate for the most critical line outage
- **Avg. rho Degradation:** Average stability margin under contingency (compared to the no-contingency baseline)

### What to Look For

1. **Vulnerability identification:** Which cases show the most margin degradation under N-1? This reveals structural vulnerabilities in the grid topology.
2. **Critical lines:** Cases where the worst-case stability drops significantly indicate the presence of a few highly critical transmission lines.
3. **Large grid challenges:** IEEE 118 has many more lines to remove, and some outages may cascade more severely -- watch for qualitative differences at this scale.

---

## Table VIII: Delay Distribution Robustness

> **LaTeX source:** `docs/tables/table08_delay_dist.tex`
> **Per-case figures:** `fig_delay_robustness_case*.pdf`

### What It Measures

The JointOptimizer is trained with a default delay model, but real communication networks exhibit diverse delay characteristics. This experiment evaluates stability under 5 different delay distributions:
- **Lognormal:** Heavy-tailed, common in internet latency
- **Exponential:** Memoryless, models queuing delays
- **Gamma:** Flexible shape, generalizes exponential
- **Uniform:** Bounded, models well-provisioned networks
- **Pareto:** Extremely heavy-tailed, models worst-case scenarios

### How to Read It

- For each IEEE case and delay distribution, we report stability rate (%) and margin rho.
- Comparing across distributions for the same case reveals how sensitive the method is to delay modeling assumptions.

### What to Look For

1. **Distribution invariance:** Ideally, the JointOptimizer maintains high stability across all distributions, proving that the learned coupling generalizes beyond the training distribution.
2. **Heavy-tail sensitivity:** Pareto delays have extreme outliers -- does the method remain robust?
3. **Scale effects:** Larger grids may be more sensitive to distribution mismatch, since they have more communication links where tail events can occur.

---

## Table IX: Convergence Analysis

> **LaTeX source:** `docs/tables/table09_convergence.tex`
> **Per-case figures:** `fig_convergence_case*.pdf`

### What It Measures

This experiment answers two practical questions:
1. **How many training scenarios are needed** to achieve adequate stability (>95%)?
2. **How does the stability margin improve with more training epochs** (100, 200, 300, ... epochs)?

### How to Read It

- **Min Scenarios:** The minimum number of training scenarios required to cross the 95% stability threshold.
- **Margin@100, Margin@200, Margin@300:** The stability margin achieved at each epoch checkpoint.
- **Best Margin:** The highest margin achieved across all tested epoch counts.

### What to Look For

1. **Sample efficiency:** Fewer required scenarios = more practical deployment. Does the method achieve good performance with limited data?
2. **Epoch-margin trajectory:** Is there diminishing returns, or does the margin keep improving linearly?
3. **Case scaling:** Do larger grids (118-bus) need proportionally more scenarios/epochs, or does the method transfer its inductive biases?

---

## Table X: Model Compression (Embedding Dimension Sweep)

> **LaTeX source:** `docs/tables/table10_compression.tex`
> **Per-case figures:** `fig_model_compression_case*.pdf`

### What It Measures

For deployment on resource-constrained edge devices (e.g., substation controllers), model size matters. This experiment sweeps the embedding dimension d_embed through values like 32, 64, 96, 128, 192, 256, measuring:
- Parameter count (total trainable parameters)
- Stability margin rho
- Stability rate (%)

The hidden dimension d_hidden and number of attention heads scale proportionally with d_embed.

### How to Read It

- Each row is a model size configuration.
- Compare the parameter count against the achieved stability margin.
- The goal is to find the smallest model that maintains performance comparable to the full-size model.

### What to Look For

1. **Compression tolerance:** How much can d_embed be reduced before performance degrades noticeably?
2. **Pareto frontier:** The parameter-count vs. margin tradeoff -- is there a "sweet spot" that balances size and accuracy?
3. **Deployment implications:** A model with d_embed=32 (tens of thousands of parameters) might fit on a microcontroller, while d_embed=256 (millions of parameters) requires GPU inference.

---

## Table IV: Transfer Learning

> **LaTeX source:** `docs/tables/table04_transfer.tex`

### What It Measures

Can a model trained on one grid be transferred to a different grid? This experiment tests three transfer directions:
- **IEEE 14 -> 39:** Small to medium
- **IEEE 39 -> 118:** Medium to large (significant topology change)
- **IEEE 118 -> 57:** Large to medium (reverse direction)

For each, we report:
- **Zero-shot rho:** Performance on the target grid without any fine-tuning
- **Fine-tuned rho:** Performance after fine-tuning on the target grid
- **Improvement:** Relative gain from fine-tuning

### How to Read It

- A high zero-shot rho means the model generalizes well across grid topologies.
- A large fine-tuning improvement means the model benefits from target-domain adaptation.
- Compare both values against training from scratch on the target grid (Table I) to assess transfer efficiency.

### What to Look For

1. **Zero-shot viability:** Can a pre-trained model provide useful stability predictions on an unseen grid without retraining?
2. **Fine-tuning acceleration:** Does transfer learning converge faster than training from scratch?
3. **Direction asymmetry:** Is transferring from large-to-small easier than small-to-large?

---

## Table V: Inference Benchmarking

> **LaTeX source:** `docs/tables/table05_inference.tex`
> **Figure:** `docs/figures/publication/fig_inference_benchmark.pdf`

### What It Measures

For real-time grid control, inference speed is critical. This experiment benchmarks:
- **Mean latency (ms):** Average time per inference
- **P95 latency (ms):** 95th percentile (worst 5% of inferences)
- **P99 latency (ms):** 99th percentile (worst 1% of inferences)
- **Parameter count:** Model size

Benchmarked across all 5 IEEE cases and including additional baselines (B8: HeteroGNN, B9: DeepOPF) for comprehensive comparison.

### How to Read It

- Each model is benchmarked on each IEEE case separately, since inference time scales with grid size.
- P95/P99 latencies matter for real-time control: the grid control loop must complete within its deadline even in the tail.
- Compare parameter count against latency to understand the efficiency of each architecture.

### What to Look For

1. **Latency budget:** Typical grid control loops run at 10-100ms intervals. Which models fit within these budgets?
2. **Scaling behavior:** How does latency grow with grid size (14 vs 118 vs 10000 buses)?
3. **Architecture efficiency:** The JointOptimizer balances accuracy (from Table I) against moderate inference cost.
4. **Tail latency:** Some models may have acceptable mean latency but unacceptable P99 latency, which is disqualifying for real-time control.

---

## Gamma Sweep (Adaptive Gamma Analysis)

> **Per-case figures:** `fig_gamma_sweep_case*.pdf`

### What It Measures

The gamma parameter controls the adaptive weighting between energy and communication objectives in the joint loss function. This experiment sweeps gamma across a range of values for each IEEE case, measuring how the stability margin and learned K respond.

### What to Look For

1. **Optimal gamma range:** Is there a clear optimal gamma, or is performance robust across a range?
2. **Interaction with grid size:** Whether larger grids prefer different gamma values than smaller ones.
3. **K sensitivity to gamma:** Whether the learned coupling constant changes significantly with gamma.

---

## K_init Sensitivity Analysis

> **Per-case figures:** `fig_k_init_sensitivity_case*.pdf`

### What It Measures

Although K_init is now auto-scaled, this experiment tests robustness to the initial K value by sweeping it across a range. The question: does the learned K_final converge to the same value regardless of initialization?

### What to Look For

1. **Convergence:** Do different K_init values lead to the same K_final? If so, the method is robust to initialization.
2. **Convergence speed:** Do initializations closer to the final K converge faster?
3. **Failure modes:** Are there K_init values that cause training to diverge or get stuck in poor local minima?

---

# Summary of Experimental Coverage

| Experiment | Conference Paper | Journal Extension |
|------------|-----------------|-------------------|
| IEEE test cases | 3 (14, 39, 57) | 5 (14, 30, 39, 57, 118) |
| Baselines | 7 | 7 main + 2 inference-only (B8, B9) |
| Ablation study | alpha + GNN layers | alpha + physics mask + causal mask + cross-attention + contrastive loss + GNN layers |
| Theorem 1 validation | 1 case | 5 cases |
| Stress testing | -- | 11 scenarios per case |
| N-1 contingency | -- | All single-line outages |
| Delay distributions | -- | 5 distributions (lognormal, exp, gamma, uniform, pareto) |
| Convergence analysis | -- | Scenario + epoch sweep |
| Model compression | -- | Embedding dimension sweep (32-256) |
| Transfer learning | -- | 3 transfer directions |
| Inference benchmarking | -- | Latency profiling (mean, P95, P99) |
| Gamma sweep | -- | Per-case gamma sensitivity |
| K_init sensitivity | -- | Per-case K_init robustness |
| Pade approximation | -- | Higher-order delay-stability analysis |
| Hardware | Apple M1 (CPU) | NVIDIA GPU (CUDA) |
| Seeds | 1 | 5 (mean +/- std reported) |

---

# Summary of Key Findings

## For Novices

1. **Our AI is better at controlling power grids** than existing methods, consistently across grids of very different sizes.
2. **It learns to be less conservative** while remaining safe -- this means the grid can handle more renewable energy or defer expensive upgrades.
3. **It works on 5 different grids** from small (14-bus) to large (118-bus), proving generalizability.
4. **Our math predictions match reality** -- Theorem 1 accurately predicts how much communication delay degrades stability.
5. **It survives stress tests** -- even under high load, generator outages, and line failures, the method remains robust.
6. **It can be compressed** for deployment on small devices without significant performance loss.

## For Professionals

1. **Learnable coupling constants K** provide consistent stability margin improvement over fixed-K baselines.
2. **Physics-constrained attention** outperforms generic transformers (B6) and uncoupled transformers (B7).
3. **GNN backbone** effectively captures grid topology; shallow GNNs suffice for moderate grid sizes.
4. **Theorem 1 (delay-stability bound)** is empirically validated across all 5 IEEE cases.
5. **Observation 1 (domain separation)** confirmed via low cross-domain mutual information.
6. **Auto-scaled K_init** eliminates a manual hyperparameter and improves convergence across grid sizes.
7. **Transfer learning** enables cross-grid generalization with fine-tuning.
8. **Model compression** down to d_embed=32 maintains stability for deployment on edge devices.
9. **N-1 contingency** analysis reveals grid-specific vulnerabilities under the co-optimization framework.
10. **Delay distribution robustness** demonstrates generalization beyond the training delay model.

## Practical Recommendations

| Parameter | Recommended Approach | Rationale |
|-----------|---------------------|-----------|
| K initialization | Auto-scale: `k_init = safety_factor * abs(lambda_min) / n_gen` | Adapts to grid size automatically |
| Coupling weight alpha | 1.0 | Optimal or near-optimal in ablation study |
| GNN layers | 3 | Good accuracy-efficiency tradeoff for most grid sizes |
| Embedding dimension | 64-128 for edge deployment; 128-256 for server deployment | Compression analysis shows tradeoff |
| Communication delay target | Determine from Theorem 1 per grid | Grid-specific critical threshold |
| Training scenarios | Refer to convergence analysis per case | Minimum for >95% stability varies by grid |

---

# File Reference Guide

## Tables (`docs/tables/`)

| File | Content |
|------|---------|
| `table01_main_comparison.tex` | Main 8-model comparison across 5 IEEE cases |
| `table02_ablation.tex` | Component ablation study |
| `table03_stress_test.tex` | Stress test (11 scenarios) |
| `table04_transfer.tex` | Transfer learning (3 directions) |
| `table05_inference.tex` | Inference latency benchmarking |
| `table06_theorem1.tex` | Theorem 1 delay-stability validation |
| `table07_n1_contingency.tex` | N-1 contingency analysis |
| `table08_delay_dist.tex` | Delay distribution robustness |
| `table09_convergence.tex` | Convergence (scenario + epoch sweep) |
| `table10_compression.tex` | Model compression (embed_dim sweep) |
| `all_tables.tex` | All tables combined (auto-generated) |

## Figures (`docs/figures/publication/`)

| File | Content |
|------|---------|
| `fig_stability_margin_comparison.pdf` | Bar chart: rho across all models and cases |
| `fig_k_learning_comparison.pdf` | K_init vs K_final comparison |
| `fig_theorem1_validation.pdf` | Predicted vs empirical rho across delays |
| `fig_theorem1_all_models.pdf` | Theorem 1 validation for all models |
| `fig_ablation_study.pdf` | Ablation component impact |
| `fig_efficiency_comparison.pdf` | Parameter count vs accuracy |
| `fig_improvement_heatmap.pdf` | Improvement over baselines (heatmap) |
| `fig_radar_comparison.pdf` | Multi-metric radar plot |
| `fig_radar_all_baselines.pdf` | All-baseline radar plot |
| `fig_stress_heatmap.pdf` | Stress test results (heatmap) |
| `fig_inference_benchmark.pdf` | Latency profiling |
| `fig_stress_heatmap_case*.pdf` | Per-case stress heatmaps |
| `fig_stress_degradation_case*.pdf` | Per-case stress degradation curves |
| `fig_n1_contingency_case*.pdf` | Per-case N-1 contingency results |
| `fig_convergence_case*.pdf` | Per-case convergence curves |
| `fig_gamma_sweep_case*.pdf` | Per-case gamma sensitivity |
| `fig_k_init_sensitivity_case*.pdf` | Per-case K_init robustness |
| `fig_delay_robustness_case*.pdf` | Per-case delay distribution results |
| `fig_model_compression_case*.pdf` | Per-case compression tradeoff |
| `fig_pade_analysis_case*.pdf` | Per-case Pade approximation analysis |

---

# Appendix: Glossary

| Term | Definition |
|------|------------|
| **rho (rho)** | Stability margin -- how far from instability |
| **K** | Coupling constant -- how much delay hurts stability |
| **K_init** | Initial coupling constant (auto-scaled per grid) |
| **tau (tau)** | Communication delay in milliseconds |
| **lambda_min** | Smallest eigenvalue of the system matrix (determines base stability) |
| **GNN** | Graph Neural Network -- AI that understands network structure |
| **OPF** | Optimal Power Flow -- standard power system optimization |
| **QoS** | Quality of Service -- communication network performance |
| **MI** | Mutual Information -- statistical dependence measure |
| **Entropy** | Measure of randomness/complexity |
| **DDE** | Delay Differential Equation -- models systems with time delays |
| **Pade approximant** | Rational function approximation (used for higher-order delay modeling) |
| **N-1 contingency** | Reliability criterion: survive any single component failure |
| **Ablation** | Experiment removing one component to measure its impact |
| **p-value** | Probability that results are due to chance |
| **d_embed** | Embedding dimension (model size parameter) |
| **Transfer learning** | Reusing a model trained on one task/grid for another |
| **Pareto distribution** | Heavy-tailed probability distribution (models extreme events) |

---

**Document Version:** 2.0 (Journal Extension)
**Target Venue:** IEEE Trans. Smart Grid
**Hardware:** NVIDIA GPU (CUDA)
**Contact:** [Research Team]
