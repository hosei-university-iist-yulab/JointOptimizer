# Future Research Directions: Multi-Domain Extensions of Joint Energy-Communication Co-Optimization

**Foundation:** IEEE Transactions on Smart Grid — Learnable Delay-Stability Coupling for Smart Grid Communication Networks

**Scope:** This document identifies six research directions that extend the current two-domain (energy + communication) co-optimization framework to multi-domain settings and enhanced operational capabilities. Directions 1--5 introduce additional infrastructure domains; Direction 6 addresses the integration of large language models (LLMs) for unstructured data processing and operator decision support. Each direction is grounded in a real-world problem, supported by a falsifiable hypothesis, and accompanied by a preliminary theoretical formulation. The emphasis is on identifying what has been accomplished in each area, what remains open, and why the open problems are technically challenging.

**Notation convention:** This document follows the notation established in the parent paper. Specifically, $\mathcal{G}_E$ denotes the power grid graph, $\mathcal{G}_I$ denotes the communication network graph, $\rho(\tau)$ is the delay-dependent stability margin, $K_i$ is the learnable coupling constant for generator $i$, $\lambda_{\min}(0)$ is the least stable eigenvalue of the zero-delay Jacobian, and $\tau_i$ is the communication delay for generator $i$.

---

## Table of Contents

1. [Unifying Vision: From Two-Domain to N-Domain Co-Optimization](#1-unifying-vision)
2. [Direction 1: Energy-Communication-Cybersecurity Tri-Domain Co-Optimization](#2-direction-1)
3. [Direction 2: Multi-Carrier Energy Systems (Gas-Electric-Communication)](#3-direction-2)
4. [Direction 3: Distributed Energy Storage under Communication Constraints](#4-direction-3)
5. [Direction 4: Electric Vehicle Fleet Coordination with Mobile Loads](#5-direction-4)
6. [Direction 5: Scalable Multi-Area Interconnected Grids](#6-direction-5)
7. [Direction 6: Large Language Model Integration for Operational Intelligence](#7-direction-6)
8. [Cross-Cutting Methodological Challenges](#8-cross-cutting-challenges)
9. [Prioritized Research Roadmap](#9-roadmap)

---

## 1. Unifying Vision: From Two-Domain to N-Domain Co-Optimization
<a name="1-unifying-vision"></a>

### 1.1 Current Framework Recap

The parent paper establishes a two-domain co-optimization framework in which the power grid graph $\mathcal{G}_E$ and the communication network graph $\mathcal{G}_I$ are jointly optimized under a formal delay-stability guarantee. The central result (Theorem 1) states:

$$\rho(\tau) \geq |\lambda_{\min}(0)| - \sum_{i=1}^{n_g} K_i \cdot \frac{\tau_i}{\tau_{\max,i}}$$

This bound establishes that the stability margin degrades linearly with per-generator communication delay, governed by learnable coupling constants $K_i$ trained end-to-end via gradient descent. The architecture employs dual-domain GNN encoders with physics-constrained hierarchical attention for cross-domain feature fusion.

### 1.2 The Multi-Domain Extension

A natural question is whether this framework can accommodate additional infrastructure domains beyond energy and communication. Real-world smart grids do not operate in isolation: they depend on gas pipelines for fuel, cybersecurity systems for protection, energy storage for flexibility, electric vehicle fleets for distributed resources, and inter-area coordination for large-scale stability. Each of these introduces a new graph structure $\mathcal{G}_k$, new coupling mechanisms, and new delay-sensitivity parameters.

The generalized multi-domain stability bound takes the form:

$$\rho(\boldsymbol{\tau}, \boldsymbol{\phi}) \geq |\lambda_{\min}(0)| - \sum_{k=1}^{D} \sum_{i=1}^{n_k} K_i^{(k)} \cdot \frac{\phi_i^{(k)}}{\phi_{\max,i}^{(k)}}$$

where $D$ is the number of coupled domains, $n_k$ is the number of coupling agents in domain $k$, $K_i^{(k)}$ is the learnable coupling constant for agent $i$ in domain $k$, and $\phi_i^{(k)}$ is the domain-specific impairment variable (e.g., communication delay for the information domain, security processing latency for the cybersecurity domain, gas supply response time for the gas domain). The key open question is under what conditions this multi-domain generalization preserves the linear structure and the learnability of the coupling constants.

### 1.3 Architectural Implications

Extending from two to $D$ domains requires:
- $D$ domain-specific GNN encoders (one per infrastructure graph)
- $D(D-1)/2$ cross-domain attention mechanisms (one per domain pair), or a hierarchical attention scheme that reduces this to $O(D)$
- A multi-objective loss function that balances $D$ domain-specific objectives with stability enforcement and cross-domain coupling accuracy
- An extended auto-scaled initialization that distributes the stability budget across $D \cdot n_k$ coupling agents

Each of the six directions below either instantiates this general framework for a specific third domain (Directions 1--5) or addresses the integration of complementary AI capabilities for unstructured data processing and operational intelligence (Direction 6), with precise problem statements, hypotheses, and theoretical formulations.

---

## 2. Direction 1: Energy-Communication-Cybersecurity Tri-Domain Co-Optimization
<a name="2-direction-1"></a>

### 2.1 Problem Statement

Smart grid communication networks are targets for cyberattacks, including false data injection (FDI), denial-of-service (DoS), and man-in-the-middle attacks. Deploying cybersecurity countermeasures — such as deep packet inspection, anomaly detection algorithms, and cryptographic authentication protocols — introduces computational processing latency on communication links. This creates a fundamental three-way tradeoff: stronger security increases communication delay, which degrades stability margin; weaker security reduces delay but exposes the system to attacks that can compromise data integrity and cause incorrect dispatch decisions. No existing framework optimizes the security allocation jointly with energy dispatch and communication routing under formal stability guarantees.

### 2.2 Real-World Motivation

The severity of cyber threats to power grids has been demonstrated by documented incidents:

- **Ukraine power grid attacks (2015, 2016):** Coordinated cyberattacks on three Ukrainian distribution companies caused widespread outages affecting approximately 230,000 customers. The attackers exploited compromised SCADA systems to open breakers remotely, demonstrating that control system intrusion can directly cause physical grid failures.
- **TRITON/TRISIS malware (2017):** This attack specifically targeted safety instrumented systems (SIS) at a petrochemical facility in Saudi Arabia, demonstrating that adversaries can target the protection systems themselves — the last line of defense against physical damage.
- **Colonial Pipeline ransomware (2021):** Although targeting a fuel pipeline rather than an electric grid, this incident demonstrated the cascading consequences of cyberattacks on critical energy infrastructure and the economic costs (estimated at over $4.4 billion in disruption).

Regulatory frameworks including NERC CIP (Critical Infrastructure Protection) standards in North America and the EU Network and Information Security (NIS2) Directive mandate cybersecurity measures for grid operators. However, these standards specify minimum security requirements without considering the impact of security processing latency on grid stability. The IEC 62351 standard for power system communication security addresses authentication and encryption but does not quantify the resulting latency overhead or its stability implications.

### 2.3 State of the Art

**What has been accomplished:**

- *Cyber-physical system security for power grids:* A substantial body of work addresses detection and mitigation of specific attack types. False data injection detection has been studied using statistical hypothesis testing, machine learning classifiers, and physics-based residual analysis. Game-theoretic frameworks model attacker-defender interactions for resource allocation in grid security. Moving target defense strategies randomize system configurations to increase attack difficulty.
- *Latency-aware network security:* In the computer networking literature, the tradeoff between security processing overhead and network performance is well-characterized for specific protocols. TLS handshake latency, deep packet inspection throughput, and intrusion detection system (IDS) processing delay have been measured empirically for various hardware platforms and traffic volumes.
- *Resilient control under attack:* Control-theoretic approaches design controllers that maintain stability despite corrupted sensor data or actuator commands, typically using observer-based methods, redundancy, or robust control techniques.

**What has not been accomplished:**

- No framework jointly optimizes (i) which security measures to deploy on which communication links, (ii) how to route grid control traffic given the resulting security-induced latency, and (iii) how to adjust energy dispatch to compensate for the stability margin reduction caused by security processing delays.
- Existing game-theoretic models for grid cybersecurity do not incorporate the delay-stability coupling: the defender's cost function does not include stability margin degradation from security overhead.
- Resilient control methods assume attacks have already occurred and focus on post-attack stability. They do not optimize the pre-attack security allocation to prevent attacks while minimizing stability impact.

### 2.4 Research Hypothesis

**Hypothesis 1:** There exists a non-uniform security allocation across communication links such that the total security-induced latency overhead is minimized subject to a minimum resilience requirement against a defined threat model, while maintaining the delay-stability margin above a specified positive threshold. Furthermore, the optimal allocation concentrates higher security levels on links serving generators with larger coupling constants $K_i$, because these generators are more sensitive to delay and therefore more critical to protect.

### 2.5 Proposed Theoretical Framework

**Definition (Security-augmented delay).** For each communication link serving generator $i$, the effective delay is:

$$\tau_i^{\text{eff}} = \tau_i^{\text{net}} + \Delta_{\text{sec}}(\ell_i)$$

where $\tau_i^{\text{net}}$ is the baseline network delay (propagation, queuing, transmission) and $\Delta_{\text{sec}}(\ell_i)$ is the security processing latency, a monotonically non-decreasing function of the security level $\ell_i \in [0, 1]$. The function $\Delta_{\text{sec}}$ depends on the security mechanism: for AES-256 encryption alone, $\Delta_{\text{sec}} < 0.1$ ms (negligible); for deep packet inspection with anomaly detection, $\Delta_{\text{sec}}$ can range from 1 to 50 ms depending on traffic volume and algorithm complexity.

**Theorem 1 extension.** Substituting the security-augmented delay into the stability bound yields:

$$\rho(\boldsymbol{\tau}^{\text{eff}}) \geq |\lambda_{\min}(0)| - \sum_{i=1}^{n_g} K_i \cdot \frac{\tau_i^{\text{net}} + \Delta_{\text{sec}}(\ell_i)}{\tau_{\max,i}}$$

The tri-domain optimization problem becomes:

$$\min_{\mathbf{P}_g, \, \mathbf{r}, \, \boldsymbol{\ell}} \quad \alpha_E \cdot \mathcal{L}_{\text{OPF}}(\mathbf{P}_g) + \alpha_I \cdot \mathcal{L}_{\text{QoS}}(\mathbf{r}) + \alpha_S \cdot \mathcal{L}_{\text{sec}}(\boldsymbol{\ell})$$

subject to: $\rho(\boldsymbol{\tau}^{\text{eff}}) > 0$, security resilience constraints, and power flow feasibility constraints, where $\mathbf{P}_g$ is the generator dispatch, $\mathbf{r}$ is the communication routing, $\boldsymbol{\ell}$ is the per-link security level vector, and $\mathcal{L}_{\text{sec}}$ penalizes insufficient security coverage.

**Proposition 1.1.** Under the assumption that $\Delta_{\text{sec}}(\ell_i)$ is convex in $\ell_i$ (which holds for common security mechanisms where marginal latency increases with security level), the tri-domain optimization problem is jointly convex in the security allocation $\boldsymbol{\ell}$ when the other decision variables are fixed.

**Proposition 1.2.** At optimality, the security allocation satisfies $\ell_i^* \propto K_i / \tau_{\max,i}$: links with higher coupling sensitivity and tighter delay budgets receive proportionally higher security investment. This follows from the KKT conditions of the stability constraint.

### 2.6 Technical Challenges

1. **Adversarial modeling.** Unlike communication delay, which is a stochastic physical quantity, cyberattack strategies are chosen by an intelligent adversary who adapts to the defender's allocation. A Stackelberg game formulation is required, where the defender (leader) commits to a security allocation and the attacker (follower) responds optimally. The existence and computability of Stackelberg equilibria depend on the compactness of the strategy spaces and the continuity of the payoff functions, which must be verified for the specific threat model.

2. **Dynamic security reallocation.** The optimal security allocation changes with the operating point: during periods of high renewable variability, stability margins are tighter, and the tradeoff shifts toward lower security overhead; during low-variability periods, higher security is affordable. This requires online re-optimization at a timescale faster than the attack response time, which is computationally demanding.

3. **Data integrity versus data latency.** False data injection attacks corrupt measurement data without necessarily increasing delay. Detecting FDI requires statistical tests on measurement residuals, which introduces a different type of computational overhead (detection latency) distinct from encryption overhead. The coupling between data integrity, detection latency, and stability margin requires a more nuanced model than the simple additive delay formulation above.

4. **Quantifying security effectiveness.** The relationship between security level $\ell_i$ and attack probability reduction is difficult to calibrate empirically. Unlike delay, which can be measured directly, security effectiveness depends on the attacker's capabilities, which are inherently uncertain.

### 2.7 Expected Contributions

- A formal three-way tradeoff characterization between dispatch cost, communication quality, and cybersecurity resilience, with the stability margin as a unifying constraint
- An optimal security allocation theorem showing that security investment should be proportional to the coupling sensitivity of each generator
- A trainable architecture with three domain-specific encoders and pairwise cross-domain attention, demonstrating that the physics-constrained attention mechanism generalizes to the cybersecurity domain

---

## 3. Direction 2: Multi-Carrier Energy Systems (Gas-Electric-Communication)
<a name="3-direction-2"></a>

### 3.1 Problem Statement

Natural gas networks supply fuel to gas-fired generators, which accounted for approximately 43% of U.S. electricity generation in 2023. The gas network has its own physical dynamics: gas flows through pipelines according to pressure-driven partial differential equations, compressor stations consume energy to maintain pressure, and the phenomenon of linepack (gas stored within pressurized pipelines) provides limited temporal buffering. A disruption in gas supply directly constrains the ramp rate and maximum output of gas-fired generators, altering the feasible dispatch set and potentially compromising grid stability. Communication networks coordinate both gas and electric operations, but the communication delay affects gas scheduling and electric dispatch on different timescales. No existing framework jointly optimizes electric dispatch, gas network operation, and communication routing under a unified delay-stability guarantee.

### 3.2 Real-World Motivation

- **2021 Texas winter storm (Uri):** The most severe gas-electric interdependency failure in recent history. Freezing temperatures caused gas wellhead freeze-offs and pipeline pressure drops, cutting gas supply to power plants at precisely the moment when heating demand spiked. The Electric Reliability Council of Texas (ERCOT) reported that 28.8 GW of generation capacity was forced offline, of which approximately 44% was gas-fired generation lost due to fuel supply failure. The event caused an estimated $130 billion in economic damage. Post-event analysis by FERC and NERC concluded that improved gas-electric coordination — specifically, better communication and joint scheduling — could have mitigated the severity of the crisis.
- **2014 polar vortex (U.S. Northeast):** Gas pipeline constraints during extreme cold caused gas prices to spike to $120/MMBtu (compared with a typical range of $2-5/MMBtu), and multiple gas-fired generators were unable to secure fuel contracts. PJM Interconnection reported 22% of gas-fired capacity as unavailable due to fuel supply issues.
- **European energy crisis (2022):** Reduced Russian gas supply to Europe demonstrated the geopolitical vulnerability of gas-electric interdependency at continental scale.

### 3.3 State of the Art

**What has been accomplished:**

- *Integrated energy system modeling:* The energy hub concept, introduced by Geidl and Andersson (ETH Zurich, 2007), provides a general framework for modeling multi-carrier energy conversion and coupling. This has been extended to include gas networks with steady-state flow models (Weymouth equation) and, more recently, transient models (isothermal Euler equations for pipeline dynamics).
- *Gas-electric co-optimization:* Several formulations address the joint dispatch of gas and electric systems. Decomposition methods (ADMM, Benders decomposition) solve the coupled problem by iterating between gas and electric subproblems. These methods handle steady-state coupling but typically ignore communication delay and assume instantaneous coordination between gas and electric operators.
- *Transient gas network modeling:* The dynamics of gas flow in pipelines are governed by partial differential equations that describe mass and momentum conservation. Finite-difference and method-of-characteristics discretizations enable numerical simulation. Linepack dynamics have been incorporated into security-constrained unit commitment formulations.

**What has not been accomplished:**

- No framework models the effect of communication delay on gas-electric coordination. In practice, gas nomination cycles (scheduling gas delivery) operate on timescales of hours to days, while electric dispatch operates on timescales of seconds to minutes. Communication delays in the coordination signals between gas and electric system operators introduce temporal mismatches that can lead to suboptimal or infeasible scheduling.
- The delay-stability coupling bound (Theorem 1 of the parent paper) addresses only the electric domain. Extending it to gas-electric systems requires incorporating the gas network dynamics — specifically, the pressure response time of pipelines and the ramp rate constraints imposed by gas supply availability — into the stability model.
- GNN-based approaches for multi-carrier energy systems are in their infancy. While GNNs have been applied to electric grids and to individual gas networks, no GNN architecture processes coupled gas-electric-communication graphs with physics-constrained attention.

### 3.4 Research Hypothesis

**Hypothesis 2:** The delay-stability coupling bound can be extended to multi-carrier energy systems by introducing carrier-specific coupling constants that capture the differential sensitivity of gas-fired and non-gas generators to communication delays. Gas-fired generators exhibit larger effective coupling constants because their power output depends not only on the timeliness of electric dispatch signals but also on the coordination delay with the gas network operator. This compound delay effect can be captured by a multiplicative correction to $K_i$ for gas-fired generators.

### 3.5 Proposed Theoretical Framework

**Gas network coupling.** For a gas-fired generator $i$, the effective delay has two components:

$$\tau_i^{\text{eff}} = \tau_i^{E} + \tau_i^{G}$$

where $\tau_i^{E}$ is the electric dispatch communication delay (as in the parent paper) and $\tau_i^{G}$ is the gas coordination delay — the time between requesting a change in gas supply and the physical arrival of the requested gas flow at the generator's fuel input. The gas coordination delay depends on the pipeline distance from the gas source, the linepack buffer available in the pipeline segment, and the communication delay between the electric operator and the gas operator.

**Extended stability bound.** The generalized bound becomes:

$$\rho(\boldsymbol{\tau}^E, \boldsymbol{\tau}^G) \geq |\lambda_{\min}(0)| - \sum_{i \in \mathcal{G}_{\text{gas}}} K_i^{E} \cdot \frac{\tau_i^{E}}{\tau_{\max}^{E}} - \sum_{i \in \mathcal{G}_{\text{gas}}} K_i^{G} \cdot \frac{\tau_i^{G}}{\tau_{\max}^{G}} - \sum_{j \notin \mathcal{G}_{\text{gas}}} K_j^{E} \cdot \frac{\tau_j^{E}}{\tau_{\max}^{E}}$$

where $\mathcal{G}_{\text{gas}} \subseteq \{1, \ldots, n_g\}$ is the set of gas-fired generators, $K_i^{E}$ and $K_i^{G}$ are the electric and gas coupling constants for generator $i$ (both learnable), and the third sum covers non-gas generators (which experience only electric delay).

**Proposition 2.1.** For gas-fired generators, the total coupling sensitivity satisfies $K_i^{E} + K_i^{G} > K_j^{E}$ for non-gas generators $j$ of comparable size and electrical location. This reflects the compound vulnerability of gas-fired generators to both electric and gas coordination delays.

**Proposition 2.2.** During gas supply constrained periods (e.g., extreme cold weather), the gas coupling constants $K_i^{G}$ increase because the linepack buffer is depleted, reducing the gas network's ability to absorb coordination delays. An adaptive mechanism that increases $K_i^{G}$ in response to declining pipeline pressure would improve the bound accuracy under stress conditions.

### 3.6 Technical Challenges

1. **Timescale separation.** Gas dynamics operate on timescales of minutes to hours (pressure wave propagation in pipelines), while electric dynamics operate on timescales of milliseconds to seconds (electromechanical oscillations). The stability margin $\rho$ as defined in the parent paper is a small-signal stability metric for the electric system. Extending it to incorporate gas dynamics requires either a multi-timescale stability analysis or a quasi-static treatment of gas dynamics within the electric stability framework. Neither approach is straightforward.

2. **Nonlinear gas flow.** The Weymouth equation for steady-state gas flow is inherently quadratic (flow rate squared is proportional to the pressure squared difference). Linearization is valid only for small perturbations around the operating point. Unlike the electric swing equation, where linearization is standard practice (Assumption A1 of Theorem 1), linearization of gas dynamics may introduce larger errors, particularly during contingency events when gas flows deviate significantly from nominal values.

3. **Institutional separation.** In most jurisdictions, gas and electric systems are operated by different entities with separate market structures, regulatory frameworks, and communication protocols. Joint optimization requires either a coordinating entity with access to both systems' state variables or a distributed algorithm that preserves the informational privacy of each operator. This institutional constraint is not merely technical — it reflects regulatory and commercial realities that any practical framework must accommodate.

4. **Data availability.** Gas network operational data (pipeline pressures, flow rates, compressor states) is less publicly available than electric grid data. The IEEE standard test cases that underpin the parent paper's evaluation have no gas network counterpart. Constructing realistic coupled gas-electric test cases with validated parameters is itself a research challenge.

### 3.7 Expected Contributions

- A formal characterization of compound delay in gas-electric systems, distinguishing electric dispatch delay from gas coordination delay
- Carrier-specific learnable coupling constants that quantify the differential vulnerability of gas-fired generators
- A tri-domain GNN architecture with gas, electric, and communication encoders, demonstrating that the physics-constrained attention mechanism can capture cross-carrier coupling (e.g., impedance masks for the electric domain and pressure gradient masks for the gas domain)

---

## 4. Direction 3: Distributed Energy Storage under Communication Constraints
<a name="4-direction-3"></a>

### 4.1 Problem Statement

Battery energy storage systems (BESS) introduce temporal coupling into the power dispatch problem: the state-of-charge (SOC) at any time depends on the entire history of charge and discharge decisions. When communication delays are present between the grid operator and distributed storage assets, the operator observes a stale SOC value and issues dispatch commands based on outdated information. The resulting mismatch between the actual SOC and the operator's estimate can cause commands that violate battery operating limits (overcharging, deep discharging), accelerate degradation, or miss time-sensitive opportunities for frequency regulation and arbitrage. No existing framework quantifies how communication delay affects the controllability and value of distributed storage, or how to jointly optimize storage dispatch and communication quality to maximize both storage utilization and grid stability.

### 4.2 Real-World Motivation

- **Grid-scale battery deployment:** Global grid-scale battery storage capacity exceeded 45 GW by the end of 2023, with projections exceeding 400 GW by 2030 (International Energy Agency). The Hornsdale Power Reserve in South Australia demonstrated that batteries can provide grid stability services (frequency control ancillary services) with response times under 150 ms — far faster than conventional generators. This speed advantage is predicated on receiving accurate, timely dispatch signals.
- **Frequency regulation markets:** In PJM Interconnection, battery storage participates in the RegD fast-response regulation signal, which updates every 2 seconds. A communication delay of even 100 ms can reduce the battery's regulation performance score (mileage), directly impacting revenue. The Federal Energy Regulatory Commission (FERC) Order 755 requires compensation based on accuracy of regulation response, creating a direct financial incentive to minimize communication delay effects.
- **Distributed residential storage:** The proliferation of behind-the-meter storage (e.g., Tesla Powerwall, Enphase Encharge) creates aggregation challenges. Coordinating thousands of distributed batteries through a virtual power plant (VPP) architecture requires communication with each unit, and the aggregate response quality depends on the delay characteristics of the communication links to individual batteries.

### 4.3 State of the Art

**What has been accomplished:**

- *Model predictive control (MPC) for BESS scheduling:* MPC is the dominant control paradigm for battery dispatch, optimizing over a rolling horizon while respecting SOC constraints and degradation models. Commercial battery management systems (BMS) implement variants of MPC for energy arbitrage and ancillary service provision.
- *Degradation-aware optimization:* Battery degradation models, ranging from empirical cycle-counting to electrochemical models, have been integrated into dispatch optimization to balance immediate revenue against long-term capacity fade. These models capture calendar aging, cycle aging, temperature effects, and depth-of-discharge dependence.
- *Distributed storage coordination:* Consensus-based algorithms and ADMM decomposition have been applied to coordinate multiple storage units for voltage regulation, peak shaving, and frequency response. These methods handle communication topology but assume known, fixed communication delays.
- *Delay compensation in control systems:* Smith predictors and related delay compensation techniques have been applied to individual control loops. However, these methods address a single feedback loop with a single delay, not the multi-agent, multi-delay setting of distributed storage coordination.

**What has not been accomplished:**

- No framework characterizes how communication delay translates into SOC estimation error for distributed storage, or how this estimation error propagates to stability margin degradation.
- The interaction between battery degradation and communication delay has not been studied: delayed commands may cause unnecessary cycling or operation outside optimal SOC windows, accelerating degradation through a mechanism distinct from the direct stability impact.
- GNN-based methods for storage dispatch do not account for the temporal coupling between past decisions and current SOC, nor for the communication delay in SOC reporting.

### 4.4 Research Hypothesis

**Hypothesis 3:** Communication delay introduces a state estimation error in the operator's knowledge of battery SOC that grows linearly with the product of the delay magnitude and the maximum charge/discharge rate. This estimation error can be bounded and incorporated into the delay-stability coupling framework as an additional coupling term that captures the reduced controllability of storage assets under delayed communication.

### 4.5 Proposed Theoretical Framework

**SOC dynamics under delay.** The true SOC of battery $j$ at time $t$ evolves according to:

$$\text{SOC}_j(t) = \text{SOC}_j(0) + \int_0^t \frac{\eta_c \cdot P_j^{c}(s) - P_j^{d}(s) / \eta_d}{E_j^{\text{cap}}} \, ds$$

where $P_j^{c}(s)$ and $P_j^{d}(s)$ are the charge and discharge power at time $s$, $\eta_c$ and $\eta_d$ are the charge and discharge efficiencies, and $E_j^{\text{cap}}$ is the energy capacity. The operator observes $\text{SOC}_j(t - \tau_j)$ due to communication delay $\tau_j$. The estimation error is:

$$|e_j(t)| = |\text{SOC}_j(t) - \text{SOC}_j(t - \tau_j)| \leq \frac{P_j^{\max}}{E_j^{\text{cap}}} \cdot \tau_j$$

where $P_j^{\max} = \max(P_j^{c,\max}, P_j^{d,\max})$ is the maximum power rating.

**Extended stability bound with storage.** Storage assets contribute to grid stability through frequency regulation. A delayed SOC estimate causes the operator to issue suboptimal regulation commands. The stability margin extension introduces storage coupling constants:

$$\rho(\boldsymbol{\tau}^E, \boldsymbol{\tau}^S) \geq |\lambda_{\min}(0)| - \sum_{i=1}^{n_g} K_i^{E} \cdot \frac{\tau_i^{E}}{\tau_{\max}^{E}} - \sum_{j=1}^{n_s} K_j^{S} \cdot \frac{P_j^{\max} \cdot \tau_j^{S}}{E_j^{\text{cap}} \cdot \tau_{\max}^{S}}$$

where $n_s$ is the number of storage units, $K_j^{S}$ is the learnable coupling constant for storage unit $j$, and $\tau_j^{S}$ is the communication delay to storage unit $j$. The storage coupling term is weighted by the power-to-energy ratio $P_j^{\max} / E_j^{\text{cap}}$, reflecting that high-power, low-energy batteries (e.g., supercapacitors) are more sensitive to communication delay than low-power, high-energy batteries (e.g., flow batteries).

**Proposition 3.1.** The SOC estimation error bound $P_j^{\max} \cdot \tau_j / E_j^{\text{cap}}$ is tight in the worst case (maximum power output sustained over the delay interval) and conservative in the average case by a factor proportional to the battery utilization factor. Learned coupling constants $K_j^{S}$ can compensate for this conservatism by adapting to the empirical utilization distribution.

**Proposition 3.2.** The optimal communication resource allocation between generators and storage units follows a priority rule based on the product $K_j^{S} \cdot P_j^{\max} / E_j^{\text{cap}}$: high-power, low-energy storage units with high coupling sensitivity should receive lower-latency communication links. This can be derived from the KKT conditions of the extended stability constraint.

### 4.6 Technical Challenges

1. **Temporal coupling.** Unlike generators, whose stability contribution at time $t$ depends only on their current state, batteries have path-dependent behavior: the current SOC depends on all past dispatch decisions. This temporal coupling transforms the optimization from a static (or myopic) problem into a dynamic programming problem, increasing computational complexity.

2. **Multi-timescale dispatch.** Batteries participate in multiple services simultaneously: fast frequency regulation (seconds), energy arbitrage (hours), and peak shaving (daily). Communication delay affects each service differently. A unified framework must handle this multi-timescale structure without requiring separate optimization for each timescale.

3. **Degradation modeling uncertainty.** Battery degradation depends on operating conditions (temperature, SOC range, C-rate) in complex, nonlinear ways that are difficult to model accurately. Calendar aging proceeds regardless of communication delay, but cycle aging may be exacerbated by delay-induced suboptimal cycling patterns. Separating the degradation impact of communication delay from other degradation mechanisms is nontrivial.

4. **Scalability to distributed storage.** While the parent paper considers $n_g \leq 54$ generators (IEEE 118-bus), a distribution grid with behind-the-meter storage may contain thousands of battery units. The quadratic attention complexity in the parent architecture ($O(N^2)$) becomes prohibitive. Sparse or hierarchical attention mechanisms are required, but designing physics-informed sparsity patterns for heterogeneous storage fleets is an open problem.

### 4.7 Expected Contributions

- A formal SOC estimation error bound as a function of communication delay and battery power-to-energy ratio
- Storage-specific learnable coupling constants integrated into the delay-stability framework
- A communication resource allocation priority rule for mixed generator-storage systems
- A multi-timescale architecture that handles fast regulation and slow arbitrage within a unified GNN framework

---

## 5. Direction 4: Electric Vehicle Fleet Coordination with Mobile Loads
<a name="5-direction-4"></a>

### 5.1 Problem Statement

Electric vehicles (EVs) are mobile loads that connect to and disconnect from the power grid at geographically dispersed locations according to stochastic transportation patterns. When participating in vehicle-to-grid (V2G) programs, EVs also act as distributed energy resources, injecting power back into the grid. Coordinating a fleet of EVs — deciding when, where, and how much each EV charges or discharges — requires communication between a fleet aggregator and individual vehicles. The communication occurs over cellular networks (4G/5G) with heterogeneous, time-varying delay characteristics. Unlike stationary generators or batteries, EV connectivity is intermittent: a vehicle may be connected at a charging station (low delay, reliable link) or in transit (high delay, unreliable link, no grid connection). The power network topology itself changes as EVs connect at different buses. No existing framework provides formal stability guarantees for joint energy-communication optimization when the network topology is time-varying due to mobile loads.

### 5.2 Real-World Motivation

- **EV adoption scale:** Global EV stock exceeded 40 million vehicles in 2023, with projections reaching 240 million by 2030 (International Energy Agency Global EV Outlook). Uncoordinated charging of even a fraction of these vehicles can create load surges that strain distribution transformers and violate voltage limits.
- **California "duck curve" exacerbation:** The California Independent System Operator (CAISO) has documented that uncoordinated EV charging deepens the evening ramp (the "neck" of the duck curve) by up to 3-5 GW, requiring faster-ramping generation resources and increasing system stress.
- **V2G pilot programs:** Commercial V2G deployments by Nuvve (San Diego), Enel X (Denmark), and Nissan (Japan) have demonstrated technical feasibility of bidirectional power flow. However, these pilots involve tens to hundreds of vehicles, not the millions required for system-level impact. The communication infrastructure to coordinate millions of EVs with grid-compatible response times does not yet exist at scale.
- **5G and V2X communication:** The 3GPP Release 16 and 17 specifications define Vehicle-to-Everything (V2X) communication standards with target end-to-end latency below 10 ms for safety-critical applications and below 100 ms for non-critical applications. These latency targets overlap with the delay ranges relevant to grid stability, creating a natural interface between the communication and energy domains.

### 5.3 State of the Art

**What has been accomplished:**

- *Smart charging algorithms:* Both centralized (direct load control) and decentralized (price-responsive, game-theoretic) algorithms have been developed for EV charging coordination. These methods optimize charging schedules to minimize cost, flatten load profiles, or maximize renewable energy utilization.
- *Aggregator-based V2G:* Aggregation frameworks model an EV fleet as a single virtual battery, abstracting away individual vehicle behavior. The aggregated resource participates in wholesale markets or ancillary services. These models assume reliable, low-latency communication between the aggregator and each EV.
- *Deep reinforcement learning for EV scheduling:* DRL agents have been trained to make real-time charging decisions under uncertainty in arrival/departure times, electricity prices, and renewable generation. These methods handle stochasticity but do not provide formal stability guarantees.
- *Transportation-power network coupling:* Coupled traffic-power flow models capture the spatial distribution of EV charging demand as a function of traffic patterns. These models are typically used for long-term planning rather than real-time dispatch.

**What has not been accomplished:**

- No framework provides formal delay-stability guarantees for power systems with time-varying topology caused by mobile EV loads. The parent paper's Theorem 1 assumes a fixed network topology (fixed number of generators, fixed connectivity). EVs violate this assumption because they connect and disconnect dynamically.
- The communication delay characteristics of EV coordination over cellular networks are fundamentally different from the wired/fiber communication networks assumed in the parent paper. Cellular delays are higher (10-100 ms for 4G, 1-10 ms for 5G), more variable (jitter), and subject to coverage gaps.
- Dynamic GNN architectures that handle time-varying graph structures while maintaining physics-informed attention have not been applied to the EV-grid coordination problem.

### 5.4 Research Hypothesis

**Hypothesis 4:** The delay-stability coupling framework can be extended to time-varying topologies by introducing a topology change rate constraint: if the rate at which EVs connect and disconnect is bounded relative to the system's natural oscillation frequencies, then a time-averaged stability margin can be defined and bounded by learnable coupling constants that adapt to the connectivity statistics. The key condition is that the topology change timescale (minutes) is slow relative to the electromechanical dynamics timescale (sub-second), enabling a quasi-static stability analysis at each topology snapshot.

### 5.5 Proposed Theoretical Framework

**Time-varying graph formulation.** At time $t$, the power grid graph is $\mathcal{G}_E(t) = (V(t), \mathcal{E}_E(t))$, where $V(t) = V_{\text{fixed}} \cup V_{\text{EV}}(t)$ includes both fixed buses and EV connection points active at time $t$. The number of active nodes $|V(t)|$ varies over time.

**Quasi-static stability bound.** Under the assumption that topology changes occur on a timescale $T_{\text{topo}}$ much larger than the electromechanical oscillation period $T_{\text{osc}}$ (i.e., $T_{\text{topo}} \gg T_{\text{osc}}$), the stability margin can be evaluated at each topology snapshot:

$$\rho(t) \geq |\lambda_{\min}(0; \mathcal{G}_E(t))| - \sum_{i \in \mathcal{A}(t)} K_i(t) \cdot \frac{\tau_i(t)}{\tau_{\max,i}}$$

where $\mathcal{A}(t) \subseteq V(t)$ is the set of active generators and V2G-enabled EVs at time $t$, and $\lambda_{\min}(0; \mathcal{G}_E(t))$ is the least stable eigenvalue evaluated on the current topology.

**Proposition 4.1.** The time-averaged stability margin $\bar{\rho} = \frac{1}{T} \int_0^T \rho(t) \, dt$ is bounded below by a function of the expected number of connected EVs, the average communication delay, and the variance of the topology change rate. Higher topology variance requires larger stability margin reserves.

**Proposition 4.2.** Dynamic GNN encoders with temporal attention can learn topology-adaptive coupling constants $K_i(t)$ that track the evolving connectivity pattern. The key architectural requirement is an attention mechanism that handles variable-size input graphs without retraining, which can be achieved through node-level (rather than graph-level) coupling constant prediction.

### 5.6 Technical Challenges

1. **Non-stationarity.** The graph $\mathcal{G}_E(t)$ changes continuously as EVs move through the transportation network. Standard GNN training assumes a fixed graph. Dynamic GNNs that handle graph evolution exist but have not been combined with physics-constrained attention or delay-stability constraints.

2. **Behavioral uncertainty.** EV owners' decisions — when to plug in, when to depart, whether to allow V2G — are driven by personal preferences, not physical laws. This behavioral uncertainty is fundamentally different from the physical uncertainty in power systems (load variability, renewable intermittency) and is harder to model.

3. **Scale.** A metropolitan area may contain millions of EVs, each requiring a communication link. Even with aggregation (grouping EVs by location or charging station), the number of controllable entities far exceeds the 118-bus systems evaluated in the parent paper.

4. **Eigenvalue recomputation.** The least stable eigenvalue $\lambda_{\min}(0; \mathcal{G}_E(t))$ changes with the topology. Recomputing it at each topology change is computationally expensive for large systems. Efficient eigenvalue tracking algorithms or neural network approximations of the eigenvalue as a function of topology are needed.

### 5.7 Expected Contributions

- A quasi-static stability framework for time-varying power network topologies caused by mobile EV loads
- Topology-adaptive coupling constants that track EV connectivity dynamics
- A dynamic GNN architecture that handles variable-size graphs with physics-constrained attention
- A formal characterization of the relationship between topology change rate and required stability margin reserve

---

## 6. Direction 5: Scalable Multi-Area Interconnected Grids
<a name="6-direction-5"></a>

### 6.1 Problem Statement

Real-world transmission grids span thousands of buses organized into multiple control areas (also called balancing authorities), each operated by an independent system operator (ISO). Inter-area tie lines carry power flows that couple the dynamics of adjacent areas, and inter-area communication links carry coordination signals (dispatch adjustments, frequency regulation, reserve activation) between ISOs. The inter-area communication typically exhibits higher latency (10-200 ms), lower bandwidth, and different security requirements than intra-area communication (1-10 ms). The parent paper's framework has been validated on systems up to 118 buses. Scaling to multi-area grids with thousands of buses requires addressing both computational tractability (the quadratic attention complexity becomes prohibitive) and the hierarchical delay structure (different delay characteristics for intra-area and inter-area communication).

### 6.2 Real-World Motivation

- **Interconnection scale:** The Eastern Interconnection of North America contains over 100,000 buses, approximately 5,000 generators, and spans multiple ISOs (PJM, MISO, NYISO, ISO-NE, SPP, among others). Inter-area oscillations at frequencies of 0.1-1 Hz are a primary stability concern, and their damping depends on wide-area monitoring and control — which in turn depends on inter-area communication quality.
- **Cross-border coordination in Europe:** The European Network of Transmission System Operators for Electricity (ENTSO-E) coordinates 36 countries with heterogeneous communication infrastructure. The 2006 European blackout was triggered by inadequate inter-area coordination when a planned line disconnection in Germany cascaded across borders.
- **Computational barrier:** The attention mechanism in the parent paper has complexity $O(N^2)$ in the number of buses. For $N = 100,000$, this requires $10^{10}$ attention computations per forward pass, which is infeasible without fundamental architectural changes.

### 6.3 State of the Art

**What has been accomplished:**

- *Distributed optimal power flow:* ADMM, dual decomposition, and consensus-based algorithms solve multi-area OPF by iterating between local area subproblems and inter-area coordination. These methods converge to the centralized solution under convexity assumptions but require multiple communication rounds, each subject to inter-area delay.
- *Wide-area monitoring, protection, and control (WAMPAC):* PMU-based wide-area systems provide real-time visibility across control areas. Synchrophasor data is transmitted at 30-60 frames per second with latency requirements of 50-200 ms. The IEEE C37.118 standard specifies synchrophasor measurement and communication requirements.
- *Sparse attention mechanisms:* In the natural language processing and computer vision literature, methods such as Longformer, BigBird, and linear attention reduce the quadratic complexity to $O(N \log N)$ or $O(N)$ by restricting the attention pattern to local windows, global tokens, or random connections.
- *Hierarchical GNNs:* Graph coarsening and pooling methods reduce large graphs to smaller representations, enabling multi-scale analysis. However, these methods have not been combined with physics-constrained attention or delay-stability guarantees.

**What has not been accomplished:**

- No framework jointly optimizes intra-area and inter-area communication quality with energy dispatch under a hierarchical delay-stability guarantee.
- Sparse attention mechanisms have not been adapted for power system graphs. The physics-constrained sparsity pattern (which buses should attend to which) is determined by the grid topology and impedance matrix, not by the proximity-based or random patterns used in NLP.
- The parent paper's auto-scaled initialization $K_{\text{init}} = s \cdot |\lambda_{\min}(0)| / n_g$ was derived for a single area. The multi-area extension must account for the hierarchical coupling structure: intra-area coupling constants govern local stability, while inter-area coupling constants govern inter-area oscillation damping.

### 6.4 Research Hypothesis

**Hypothesis 5:** The delay-stability coupling can be decomposed into intra-area and inter-area components through a block-diagonal approximation of the system Jacobian. The inter-area coupling constants are larger than the intra-area constants because inter-area oscillatory modes are typically less damped and more sensitive to communication delay. A hierarchical attention mechanism with physics-informed sparsity can maintain $O(N \log N)$ complexity while preserving the attention patterns that are physically meaningful for stability.

### 6.5 Proposed Theoretical Framework

**Hierarchical decomposition.** Consider a system with $A$ control areas. The system Jacobian can be decomposed as:

$$J(0) = J_{\text{diag}}(0) + J_{\text{tie}}(0)$$

where $J_{\text{diag}}(0) = \text{blockdiag}(J_1(0), \ldots, J_A(0))$ contains the intra-area dynamics and $J_{\text{tie}}(0)$ contains the inter-area coupling through tie lines.

**Decomposed stability bound.** The stability margin can be bounded by:

$$\rho(\boldsymbol{\tau}) \geq \min_{a=1,\ldots,A} \rho_a^{\text{intra}}(\boldsymbol{\tau}_a) - \sum_{(a,b) \in \mathcal{T}} K_{ab}^{\text{inter}} \cdot \frac{\tau_{ab}^{\text{inter}}}{\tau_{\max}^{\text{inter}}}$$

where $\rho_a^{\text{intra}}$ is the intra-area stability margin for area $a$ (computed using only intra-area delays $\boldsymbol{\tau}_a$ and coupling constants), $\mathcal{T}$ is the set of inter-area tie-line pairs, $K_{ab}^{\text{inter}}$ is the learnable inter-area coupling constant for the tie line between areas $a$ and $b$, and $\tau_{ab}^{\text{inter}}$ is the inter-area communication delay.

**Proposition 5.1.** The inter-area coupling constants satisfy $K_{ab}^{\text{inter}} \geq \max_i K_i^{\text{intra}}$ for any tie line $(a,b)$ adjacent to generator $i$. This reflects the established result in power system stability analysis that inter-area modes are typically less damped than local modes, making them more sensitive to communication delay.

**Proposition 5.2.** A physics-informed sparse attention pattern that attends locally within each area (all bus pairs within the same area) and selectively across areas (only tie-line bus pairs and their topological neighbors up to depth $k$) achieves $O(N_a^2 \cdot A + |\mathcal{T}| \cdot k^2)$ complexity, where $N_a$ is the average area size. For typical grid topologies where $A \approx \sqrt{N}$ and $|\mathcal{T}| \ll N$, this reduces to $O(N^{3/2})$, a significant improvement over the $O(N^2)$ full attention.

### 6.6 Technical Challenges

1. **Jacobian decomposition error.** The block-diagonal approximation introduces error proportional to the spectral norm of $J_{\text{tie}}(0)$. For weakly interconnected areas (typical for transmission grids), this error is small. For strongly coupled areas (e.g., areas connected by HVDC links with fast power electronics), the decomposition may be inaccurate.

2. **Coordination protocol.** Multi-area optimization requires information exchange between ISOs. In practice, ISOs are reluctant to share detailed internal state information due to competitive, regulatory, and security concerns. A distributed algorithm that achieves near-optimal coordination while exchanging only aggregate quantities (tie-line flows, area prices, frequency deviations) is required.

3. **Heterogeneous communication.** Different ISOs use different communication standards, protocols, and infrastructure. Inter-area communication may traverse multiple network hops, each with different delay characteristics. The assumption of a single inter-area delay $\tau_{ab}^{\text{inter}}$ may be overly simplistic.

4. **Validation at scale.** No publicly available test case represents a realistic multi-area grid at the scale of the Eastern Interconnection. Constructing such a test case — with validated topology, impedance values, generator parameters, and communication network overlay — is a prerequisite for experimental validation.

### 6.7 Expected Contributions

- A hierarchical delay-stability bound that decomposes into intra-area and inter-area components
- Inter-area coupling constants that quantify the sensitivity of inter-area oscillatory modes to communication delay
- A physics-informed sparse attention mechanism with sub-quadratic complexity, designed specifically for power grid topology
- A distributed training algorithm that respects the informational privacy constraints of multi-area operation

---

## 7. Direction 6: Large Language Model Integration for Operational Intelligence
<a name="7-direction-6"></a>

### 7.1 Problem Statement

The JointOptimizer framework and its multi-domain extensions (Directions 1--5) operate exclusively on structured numerical data: bus voltages, power injections, communication delays, impedance values, and graph topologies. However, real-world grid operations generate vast quantities of **unstructured data** that carry operationally critical information: SCADA event logs, operator shift reports, equipment maintenance bulletins, weather advisories, regulatory compliance documents, gas nomination schedules, and cybersecurity threat intelligence feeds. This information is currently processed by human operators through manual review, introducing cognitive bottlenecks, inconsistent interpretation, and delayed response. Furthermore, the outputs of the GNN-based optimization framework — stability margins, attention patterns, coupling constants, dispatch recommendations — are opaque to non-specialist operators who must ultimately approve and execute control actions. No existing power system optimization framework integrates unstructured data processing or provides natural language explainability as a formal component of the decision pipeline.

The central question is not whether LLMs can replace the core GNN-based optimization (they cannot, for reasons of latency, numerical precision, and formal guarantees), but rather whether LLMs can serve as a complementary intelligence layer that processes data types inaccessible to GNNs and produces outputs interpretable by human operators, thereby closing the gap between algorithmic capability and operational deployment.

### 7.2 Real-World Motivation

**Regulatory requirement for explainability.** NERC Reliability Standard IRO-001 requires that reliability coordinators "have the authority and tools to take actions to prevent identified events that could lead to Adverse Reliability Impacts." The word "identified" implies that operators must understand the basis for recommended actions. The European Union's AI Act (Regulation 2024/1689), which entered force in August 2024, classifies AI systems used in critical infrastructure management as "high-risk" and mandates that such systems provide "sufficient transparency to enable users to interpret the system's output and use it appropriately" (Article 13). A dispatch recommendation from a GNN that an operator cannot interpret does not satisfy these requirements.

**Information overload in control rooms.** A typical transmission control center processes 10,000-50,000 SCADA alarms per day under normal conditions. During disturbance events, alarm rates can exceed 1,000 per minute, far exceeding human cognitive processing capacity. The 2003 Northeast blackout investigation identified alarm processing failure as a contributing factor: operators at FirstEnergy were unable to identify the critical alarms among the flood of routine telemetry. Automated alarm prioritization and contextual summarization are recognized needs in the industry.

**Unstructured data in gas-electric coordination.** Gas pipeline operations involve contractual instruments — nominations, confirmations, operational flow orders (OFOs), force majeure declarations — that are communicated as semi-structured text documents between gas and electric operators. The time between an OFO issuance and its incorporation into electric dispatch decisions currently ranges from 30 minutes to several hours, depending on operator workload. Automated parsing and integration of these documents into the dispatch framework would reduce this latency to seconds.

**Cybersecurity intelligence.** The National Vulnerability Database (NVD) publishes an average of 60-80 new Common Vulnerabilities and Exposures (CVE) records per day. Assessing which vulnerabilities are relevant to a specific SCADA system configuration requires cross-referencing CVE descriptions, vendor advisories, and the utility's asset inventory — a task that is inherently text-based and well-suited to LLM capabilities.

### 7.3 State of the Art

**What has been accomplished:**

- *LLMs for time series and sensor data:* Recent work has demonstrated that pretrained language models can be reprogrammed for time series forecasting through input embedding alignment, achieving competitive performance with domain-specific architectures on standard benchmarks. The Spec2LLM approach for transformer temperature prediction, cited in the parent paper, exemplifies this paradigm by mapping spectral-domain sensor features into the token embedding space of a frozen language model. These methods demonstrate that LLM representations can encode physical patterns, but they operate on structured numerical inputs, not unstructured text.

- *LLM-based decision support in engineering:* LLMs have been applied as conversational interfaces for engineering design, code generation, and scientific literature synthesis. In the power systems domain, preliminary studies have explored LLMs for load forecasting explanation, outage report summarization, and regulatory compliance checking. These applications demonstrate feasibility but lack integration with formal optimization frameworks.

- *Retrieval-augmented generation (RAG) for domain-specific knowledge:* RAG architectures combine LLM generation with retrieval from curated knowledge bases, reducing hallucination and grounding responses in verified domain knowledge. RAG has been deployed in medical, legal, and financial domains but has not been applied to power system operations, where the knowledge base would comprise grid topology data, operating procedures, historical incident reports, and equipment specifications.

- *Multimodal foundation models:* Vision-language models can process heterogeneous inputs including images (e.g., thermal imagery of equipment, geographic information system maps), tables (e.g., relay settings, bus data), and text simultaneously. This multimodal capability is relevant for grid operations where operators must synthesize information from diverse sources.

**What has not been accomplished:**

- No framework formally integrates LLM-based unstructured data processing into the input pipeline of a physics-constrained grid optimization model. The LLM and the optimizer exist as disconnected systems.
- The question of how to translate LLM-extracted information into the structured feature vectors expected by GNN encoders (e.g., converting a gas OFO document into a gas coordination delay estimate $\tau_i^G$) has not been addressed.
- The fidelity and reliability of LLM-generated explanations for safety-critical power system decisions have not been formally evaluated. In particular, the risk that an LLM explanation could be plausible but physically incorrect — leading an operator to approve an unsafe action — has not been quantified.
- No architecture defines the appropriate boundary between the LLM's advisory role and the GNN's computational role in a way that preserves the formal stability guarantees of Theorem 1 while incorporating LLM-derived intelligence.

### 7.4 Research Hypothesis

**Hypothesis 6:** An LLM operating as an outer-loop intelligence layer can improve the operational effectiveness of the GNN-based optimization framework in three measurable ways: (i) by extracting structured features from unstructured operational data that reduce the information delay between real-world events and optimizer inputs, (ii) by generating physically grounded natural language explanations of dispatch recommendations that increase operator trust and decision accuracy, and (iii) by enabling rapid deployment on new grid topologies through automated parsing of technical documentation. Critically, this improvement is achievable without compromising the formal stability guarantees of Theorem 1, because the LLM operates outside the real-time control loop and its outputs are mediated through structured interfaces that the GNN validates against physics constraints.

### 7.5 Proposed Architecture

The integration follows a strict **separation of concerns** between the LLM (unstructured reasoning, slow, advisory) and the GNN (structured optimization, fast, authoritative). This separation is not merely architectural convenience — it is a safety requirement. The stability guarantee of Theorem 1 holds for the GNN's outputs regardless of the LLM's behavior, ensuring that LLM errors cannot directly cause stability violations.

**Component 1: Unstructured-to-Structured Feature Extraction (Input Side)**

The LLM processes unstructured operational data and produces structured feature vectors that augment the GNN's input:

$$\mathbf{x}_{\text{aug}} = \mathbf{x}_{\text{SCADA}} \oplus f_{\text{LLM}}(\mathcal{D}_{\text{unstructured}})$$

where $\mathbf{x}_{\text{SCADA}}$ is the standard structured input (bus voltages, power injections, delays), $\mathcal{D}_{\text{unstructured}}$ is the corpus of unstructured operational data (logs, reports, advisories), $f_{\text{LLM}}$ is the LLM-based extraction function that outputs a fixed-dimensional feature vector, and $\oplus$ denotes feature concatenation. The extraction function $f_{\text{LLM}}$ is implemented as a RAG pipeline grounded in a utility-specific knowledge base to minimize hallucination.

Specific extraction tasks and their output formats:

| Unstructured Source | Extracted Feature | Type | Target Domain |
|---|---|---|---|
| Gas operational flow orders | Expected gas availability per generator (MW) | Continuous | Direction 2 |
| Cybersecurity threat feeds | Threat level per communication link (0-1) | Continuous | Direction 1 |
| Maintenance bulletins | Equipment outage schedule (binary per bus per hour) | Binary | Direction 5 |
| Weather advisories | Renewable generation forecast adjustment (%) | Continuous | All |
| SCADA alarm logs | Anomaly severity score per bus | Continuous | Direction 1 |

**Component 2: Explainability Layer (Output Side)**

After the GNN produces a dispatch recommendation with associated stability margin, attention weights, and coupling constants, the LLM generates a natural language explanation by conditioning on these structured outputs:

$$\text{explanation} = g_{\text{LLM}}(\rho(\tau), \{K_i\}, \mathbf{A}_{\text{attn}}, \mathcal{K}_{\text{grid}})$$

where $\rho(\tau)$ is the computed stability margin, $\{K_i\}$ are the coupling constants, $\mathbf{A}_{\text{attn}}$ is the attention weight matrix, and $\mathcal{K}_{\text{grid}}$ is the grid-specific knowledge base (bus names, line designations, equipment identifiers, operating limits). The explanation maps abstract quantities to operationally meaningful statements, for example:

*"Stability margin decreased to 0.18 (from 0.39 nominal) due to elevated communication delay on the Bus 15--Bus 30 SCADA link (current: 180 ms, threshold: 200 ms). Generator 7 at Substation North has the highest coupling sensitivity (K_7 = 0.041), indicating that its dispatch is most affected by this delay. Recommended action: reroute SCADA traffic for Generator 7 through the backup fiber path (estimated delay reduction: 120 ms) or reduce Generator 7 output by 15 MW and compensate with Generator 3 (K_3 = 0.012, lower sensitivity)."*

**Component 3: Knowledge-Grounded Deployment Acceleration (Transfer Side)**

When deploying the framework on a new grid topology (Direction 5), the LLM parses the utility's technical documentation to extract:
- Bus parameters (voltage levels, load characteristics, generator ratings)
- Line parameters (impedances, thermal ratings, topology)
- Communication network topology (link types, expected latencies)
- Operating procedures and constraints (N-1 criteria, voltage limits, ramp rates)

These extracted parameters are formatted as the structured inputs required by the GNN, replacing weeks of manual data engineering. The LLM also suggests initial coupling constant estimates by reasoning over similar grid configurations in its training corpus.

**Proposition 6.1 (Safety preservation).** Because the LLM operates outside the stability-critical control loop, any error in LLM-extracted features is bounded in its effect on the stability margin. Formally, if the LLM-extracted feature $f_{\text{LLM}}$ has error $\epsilon$ relative to the true value, and the GNN's stability margin function $\rho$ is Lipschitz continuous with constant $L_\rho$ in the augmented feature dimension, then:

$$|\rho(\mathbf{x}_{\text{aug}}) - \rho(\mathbf{x}_{\text{true}})| \leq L_\rho \cdot \|\epsilon\|$$

This error bound is computable and can be monitored at runtime. If $L_\rho \cdot \|\epsilon\|$ exceeds a safety threshold, the system falls back to SCADA-only inputs ($\mathbf{x}_{\text{SCADA}}$), discarding the LLM-augmented features. This fail-safe mechanism ensures that LLM errors degrade performance gracefully rather than catastrophically.

**Proposition 6.2 (Complementary data modalities).** The LLM and the GNN process fundamentally different data modalities: the GNN operates on graph-structured numerical data with $O(N)$ nodes and $O(E)$ edges, while the LLM operates on sequential text data with $O(T)$ tokens. Neither architecture can efficiently process the other's native data type. The LLM cannot perform graph message-passing over bus impedance matrices, and the GNN cannot parse maintenance bulletins. Their combination is therefore not redundant but complementary, and the information-theoretic value of the augmented input $\mathbf{x}_{\text{aug}}$ strictly exceeds that of either input alone, provided the unstructured data contains operationally relevant information not present in the SCADA telemetry.

### 7.6 Technical Challenges

1. **Hallucination in safety-critical contexts.** LLMs can generate plausible but factually incorrect text. In a grid operations context, a hallucinated claim — for example, stating that a particular transmission line has been de-energized when it has not — could lead an operator to approve an unsafe dispatch. Mitigation strategies include RAG grounding, structured output schemas that constrain the LLM to reference only verified data, and automated cross-checking of LLM outputs against SCADA telemetry. However, no existing mitigation method provides formal guarantees of factual correctness, and quantifying the residual hallucination rate for domain-specific technical content remains an open research problem.

2. **Latency constraints.** Current LLMs require 100--2,000 ms for inference depending on model size and input length. This is incompatible with the sub-25 ms real-time control loop. The proposed architecture explicitly separates the LLM (outer loop, seconds-to-minutes timescale) from the GNN (inner loop, milliseconds timescale). However, for the feature extraction pipeline (Component 1), the LLM must process incoming unstructured data faster than the rate at which operationally relevant events occur. During disturbance events with alarm rates exceeding 1,000 per minute, even a 1-second LLM processing time could introduce information delay. Efficient inference techniques (quantization, speculative decoding, batched processing) and priority-based alarm filtering are required.

3. **Evaluation of explanation quality.** There is no established metric for evaluating whether an LLM-generated explanation of a grid dispatch decision is physically correct, operationally useful, and appropriately calibrated in its confidence. Standard NLP metrics (BLEU, ROUGE) measure textual similarity, not physical accuracy. Developing evaluation protocols — potentially involving grid operator panels and blinded comparison studies — is necessary before deployment.

4. **Knowledge base maintenance.** The RAG knowledge base must be kept current with the utility's evolving equipment inventory, operating procedures, and regulatory requirements. Stale knowledge base entries could cause the LLM to reference decommissioned equipment or superseded operating limits. Automated knowledge base update mechanisms, version control, and staleness detection are required infrastructure that does not currently exist for power system applications.

5. **Adversarial robustness.** In the cybersecurity context (Direction 1), an attacker who can inject crafted text into the LLM's input stream (e.g., by compromising a threat intelligence feed or inserting false entries into SCADA logs) could manipulate the LLM's extracted features, indirectly affecting the optimizer's inputs. The Lipschitz bound in Proposition 6.1 limits the magnitude of this effect, but a sophisticated attacker who understands the extraction function $f_{\text{LLM}}$ could craft inputs that maximize $\|\epsilon\|$ within the detection threshold. Adversarial robustness of the LLM extraction pipeline must be evaluated against realistic attack models.

6. **Regulatory acceptance.** Even with the safety preservation guarantees of Proposition 6.1, regulatory bodies (NERC, ENTSO-E, national regulators) have not established frameworks for certifying AI systems with LLM components in safety-critical grid operations. The approval process for deploying such a system in a production control center is uncertain and may require novel certification approaches that do not yet exist. Early engagement with regulatory stakeholders is advisable.

### 7.7 Scope Boundaries: Where LLMs Do Not Belong

Scientific rigor requires explicitly delineating the boundaries of the proposed LLM integration. The following applications are **excluded** from this direction because they would compromise the framework's core properties:

- **Real-time dispatch computation.** The GNN produces dispatch recommendations in 7--25 ms with formally bounded stability margins. Replacing or augmenting this computation with an LLM would increase latency by 1--2 orders of magnitude and eliminate the formal guarantees of Theorem 1. The LLM is an advisory component, not a computational one.

- **Stability bound verification.** The delay-stability coupling bound (Theorem 1) is a mathematical result derived from Pade approximation and Bauer-Fike perturbation theory. Its validity does not depend on, and cannot be improved by, natural language reasoning. LLMs cannot provide or verify formal stability certificates.

- **Replacing the GNN for graph-structured data.** Power grid topology is inherently graph-structured: buses are nodes, transmission lines are edges, and impedance values are edge weights. GNNs are architecturally matched to this structure through message-passing operations. LLMs process sequential token streams and cannot natively perform the topology-aware aggregation that GNNs provide. Encoding a graph as a token sequence (e.g., as an adjacency list) discards the locality and symmetry properties that make GNNs effective.

- **Replacing human operators.** The LLM provides explanations and recommendations to operators; it does not make autonomous control decisions. Grid operations require situational awareness, regulatory judgment, and accountability that cannot be delegated to a language model. The role of the LLM is to reduce operator cognitive load, not to eliminate operator involvement.

### 7.8 Expected Contributions

- A formal outer-loop/inner-loop architecture that integrates LLM-based unstructured data processing with GNN-based physics-constrained optimization, with provable safety preservation (Proposition 6.1)
- A RAG-based feature extraction pipeline that converts operational text data (gas nominations, cybersecurity feeds, maintenance bulletins) into structured GNN input features, with quantified extraction accuracy
- An explainability layer that generates physically grounded natural language justifications for dispatch recommendations, evaluated through operator studies against regulatory explainability requirements
- A deployment acceleration methodology that reduces the time to configure the framework on a new grid topology from weeks of manual engineering to hours of automated document parsing
- A comprehensive analysis of failure modes, including hallucination rates, adversarial vulnerability, and latency bounds, establishing the safety envelope within which LLM integration is viable

---

## 8. Cross-Cutting Methodological Challenges
<a name="8-cross-cutting-challenges"></a>

Several technical challenges are common across all five directions. Addressing them constitutes foundational work that enables all direction-specific contributions.

### 8.1 Generalized Multi-Domain Attention

Scaling the cross-domain attention mechanism from 2 domains to $D$ domains requires $D(D-1)/2$ pairwise attention matrices, which grows quadratically with $D$. For $D = 3$ (the most immediate extension), this yields 3 cross-domain attention pairs — manageable. For $D > 4$, a hierarchical or factorized attention scheme is needed. One approach is to define a "hub" domain (typically the energy domain) and compute cross-domain attention only between each auxiliary domain and the hub, reducing the count to $D-1$.

### 8.2 Multi-Objective Loss Balancing

Each additional domain introduces at least one new loss term. The current framework uses fixed weights ($\alpha_E, \alpha_I, \alpha_{\text{stab}}, \alpha_{\text{couple}}$) tuned by grid search. With $D$ domains, the loss function has $O(D)$ weights, and the loss landscape becomes increasingly complex. Adaptive loss balancing methods — such as GradNorm (which adjusts weights based on gradient magnitudes) or multi-task uncertainty weighting (which learns task-specific homoscedastic uncertainty as proxy weights) — become necessary.

### 8.3 Extended Auto-Scaled Initialization

The auto-scaled initialization $K_{\text{init}} = s \cdot |\lambda_{\min}(0)| / n_g$ distributes the stability budget equally among generators. In the multi-domain setting, the budget must be distributed across $\sum_k n_k$ coupling agents from $D$ domains. A natural extension is:

$$K_{\text{init}}^{(k)} = s_k \cdot \frac{|\lambda_{\min}(0)|}{\sum_{k'=1}^{D} n_{k'}}$$

where $s_k$ is a domain-specific safety factor reflecting the relative importance of domain $k$ for stability. Determining appropriate values of $s_k$ without grid-specific tuning is an open problem.

### 8.4 Theoretical Soundness of the Linear Bound Extension

The linearity of the stability bound in Theorem 1 derives from the first-order Pade approximation and the triangle inequality applied to the Bauer-Fike perturbation theorem. When multiple impairment variables (communication delay, security latency, gas coordination delay, SOC estimation error) are combined, the question is whether the additive structure is preserved. Formally, this requires that the perturbation $\Delta J$ associated with each domain's impairment can be decomposed into independent, norm-bounded components. This holds when the impairments affect different entries of the Jacobian (e.g., communication delay affects the control input matrix $B$ while SOC estimation error affects the load vector), but may fail when impairments interact (e.g., a cyberattack that simultaneously increases delay and corrupts SOC data).

### 8.5 Experimental Validation Infrastructure

All five directions require new benchmark datasets and simulation environments. Specifically:
- **Cybersecurity:** Attack simulation on communication networks coupled with power system simulators (e.g., PowerWorld, PSS/E)
- **Gas-electric:** Coupled gas-electric simulators with validated pipeline parameters (e.g., SAInt for gas networks, MATPOWER for electric networks)
- **Storage:** Distribution grid models with heterogeneous battery parameters and degradation models
- **EV:** Transportation-power network coupled simulators with realistic mobility traces (e.g., SUMO for traffic simulation, OpenDSS for distribution grids)
- **Multi-area:** Large-scale synthetic grid models (e.g., ACTIVSg 10k, Texas A&M synthetic grids)

Constructing and validating these simulation environments is itself a significant research effort that should precede algorithm development.

---

## 9. Prioritized Research Roadmap
<a name="9-roadmap"></a>

The five directions are ordered by their proximity to the existing framework (lowest barrier to entry first), with estimated timelines for a research team of 2-3 researchers.

### Phase 1 (Year 1): Foundation

**Direction 5 (Multi-Area Scaling)** — Highest priority because it addresses the most immediate limitation of the current work (scalability to real-world grid sizes) and requires only architectural modifications (sparse attention, hierarchical decomposition) rather than fundamentally new domain models. The theoretical framework (Jacobian decomposition) is well-established in power system analysis and can be integrated with the existing delay-stability bound with moderate theoretical effort.

**Deliverables:** Hierarchical attention architecture demonstrated on synthetic 2,000-bus and 10,000-bus grids; decomposed stability bound validated against centralized computation; sub-quadratic complexity confirmed experimentally.

### Phase 2 (Year 1-2): First Tri-Domain Extension and LLM Integration

**Direction 1 (Cybersecurity)** — The natural first extension to three domains because the security processing latency directly augments the existing delay variable, requiring the least modification to the theoretical framework. The tri-domain optimization (cost + QoS + security) is a clean generalization of the current bi-domain problem. The cybersecurity domain is also highly relevant to funding agencies and grid operators, increasing the practical impact and fundability of the research.

**Direction 6 (LLM Integration)** — Should be pursued in parallel with the cybersecurity direction because (i) the cybersecurity domain generates the unstructured data (threat feeds, SCADA logs) that most immediately benefits from LLM processing, creating a natural synergy, (ii) the explainability layer is needed before any multi-domain extension can be deployed in a regulated environment, and (iii) the LLM infrastructure (RAG pipeline, knowledge base, extraction functions) is reusable across all subsequent directions.

**Deliverables:** Security-augmented delay model validated on IEEE test cases with simulated attacks; optimal security allocation theorem; tri-domain GNN architecture with three encoders and pairwise cross-domain attention. RAG-based feature extraction pipeline for cybersecurity threat feeds; explainability layer prototype evaluated with domain experts; safety preservation bounds (Proposition 6.1) validated experimentally.

### Phase 3 (Year 2-3): Temporal Extensions

**Direction 3 (Storage)** — Introduces temporal coupling, which is a fundamental departure from the static optimization in the current framework. This direction builds on the multi-area infrastructure (Phase 1) because distribution grids with storage are a natural extension of transmission grids, and on the tri-domain experience (Phase 2) because the storage encoder follows the same architectural pattern.

**Direction 2 (Gas-Electric)** — Can be pursued in parallel with Direction 3 if resources permit. Requires collaboration with gas network modeling experts and access to gas network operational data, which may introduce logistical delays.

**Deliverables (Direction 3):** SOC estimation error bound; storage-aware stability margin; multi-timescale GNN architecture.

**Deliverables (Direction 2):** Gas-electric coupled delay model; carrier-specific coupling constants; tri-domain GNN for gas-electric-communication systems validated against FERC gas-electric coordination scenarios.

### Phase 4 (Year 3-4): Dynamic Topology

**Direction 4 (EV/Transportation)** — The most technically challenging direction because it requires time-varying graph structures, non-stationary dynamics, and behavioral uncertainty modeling. Should be attempted only after the foundational infrastructure (multi-area scaling, tri-domain architecture, temporal coupling) is mature. Requires collaboration with transportation engineering groups and access to real EV fleet data.

**Deliverables:** Quasi-static stability framework for time-varying topologies; dynamic GNN with topology-adaptive coupling constants; demonstration on coupled traffic-power network simulator.

---

## Summary Table

| Direction | New Domain | New Impairment Variable | Key Difficulty | Proximity to Current Framework |
|-----------|-----------|------------------------|----------------|-------------------------------|
| 1. Cybersecurity | Security overlay | Security processing latency | Adversarial modeling | High (additive delay) |
| 2. Gas-Electric | Gas pipeline network | Gas coordination delay | Timescale separation | Medium (new dynamics) |
| 3. Storage | Battery fleet | SOC estimation error | Temporal coupling | Medium (dynamic optimization) |
| 4. EV/Transport | Transportation network | Topology change rate | Non-stationarity | Low (time-varying graph) |
| 5. Multi-Area | Area interconnection | Inter-area delay | Computational scale | High (architectural) |
| 6. LLM Integration | Unstructured operational data | Information extraction delay | Hallucination safety | High (complementary layer) |

---

*Last updated: 2026-02-13*
*Parent paper: IEEE Trans. Smart Grid — Learnable Delay-Stability Coupling for Smart Grid Communication Networks*
