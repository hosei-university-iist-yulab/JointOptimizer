#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/23/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Reproducible Synthetic Large-Scale Power Grid Generator

Generates a 10,000+ bus power system with realistic structure for
scalability validation in the journal extension. The grid is fully
synthetic and reproducible from a seed — no external data required.

Design principles (following Birchfield et al., IEEE T-PS 2017):
  - Hierarchical zone-based topology (mimics regional interconnections)
  - Small-world properties (high clustering, short path lengths)
  - Realistic impedance, load, and generation distributions
  - Communication network overlay with latency model
  - Fully seeded: identical output for identical seed

Usage:
    from src.data.synthetic_grid import SyntheticGridGenerator
    gen = SyntheticGridGenerator(n_buses=10000, seed=42)
    state = gen.generate()
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SyntheticGridConfig:
    """Configuration for synthetic grid generation.

    All defaults produce a ~10,000-bus grid with realistic properties.
    """

    # --- Size ---
    n_buses: int = 10_000
    n_zones: int = 20          # Regional interconnection zones
    gen_ratio: float = 0.08    # Fraction of buses with generators (~800)
    load_ratio: float = 0.65   # Fraction of buses with loads (~6500)

    # --- Topology ---
    intra_zone_density: float = 0.006   # Edge density within zones
    inter_zone_lines: int = 5           # Tie lines per zone pair (adjacent)
    backbone_buses_per_zone: int = 12   # High-voltage backbone nodes per zone
    radial_branch_prob: float = 0.15    # Probability of radial (non-meshed) branch

    # --- Electrical parameters (per-unit on 100 MVA base) ---
    r_pu_mean: float = 0.02       # Line resistance mean
    r_pu_std: float = 0.015       # Line resistance std
    x_pu_mean: float = 0.08       # Line reactance mean
    x_pu_std: float = 0.04        # Line reactance std
    b_pu_mean: float = 0.04       # Line susceptance mean (charging)

    # --- Generation ---
    p_gen_mean_mw: float = 150.0  # Mean generator output (MW)
    p_gen_std_mw: float = 120.0   # Std of generator output
    p_gen_max_factor: float = 1.5 # P_max = P_gen * factor
    inertia_mean: float = 5.0     # Mean inertia constant H (seconds)
    inertia_std: float = 2.0      # Std of inertia constant

    # --- Loads ---
    p_load_mean_mw: float = 25.0  # Mean load (MW)
    p_load_std_mw: float = 20.0   # Std of load
    power_factor: float = 0.92    # Average power factor

    # --- Communication overlay ---
    comm_backbone_latency_ms: float = 5.0    # Backbone link latency
    comm_access_latency_ms: float = 15.0     # Last-mile access latency
    comm_jitter_std_ms: float = 3.0          # Jitter standard deviation
    comm_bandwidth_gbps: float = 10.0        # Backbone bandwidth
    comm_loss_rate: float = 0.001            # Packet loss probability

    # --- Voltage levels (for realistic multi-level structure) ---
    voltage_levels_kv: List[float] = field(
        default_factory=lambda: [500.0, 230.0, 115.0, 69.0, 34.5, 13.8]
    )

    seed: int = 42


@dataclass
class SyntheticGridState:
    """Output container matching PowerSystemState interface."""

    # Bus quantities
    P: torch.Tensor          # Active power injection [n_bus]
    Q: torch.Tensor          # Reactive power injection [n_bus]
    V: torch.Tensor          # Voltage magnitude [n_bus]
    theta: torch.Tensor      # Voltage angle [n_bus]

    # Generator quantities
    P_gen: torch.Tensor      # Generator active power [n_gen]
    Q_gen: torch.Tensor      # Generator reactive power [n_gen]

    # Topology
    edge_index: torch.Tensor       # Line connections [2, n_lines]
    line_impedance: torch.Tensor   # Line impedance magnitude [n_lines]

    # Indices
    gen_bus_idx: torch.Tensor      # Generator bus indices [n_gen]
    load_bus_idx: torch.Tensor     # Load bus indices [n_load]

    # Dynamics parameters
    inertia: torch.Tensor          # Generator inertia H [n_gen]
    damping: torch.Tensor          # Generator damping D [n_gen]

    # Communication network
    comm_latency_ms: torch.Tensor  # Per-generator communication delay [n_gen]
    comm_bandwidth: torch.Tensor   # Per-link bandwidth [n_gen]
    comm_loss_rate: torch.Tensor   # Per-link packet loss rate [n_gen]

    # Zone assignments
    zone_ids: torch.Tensor         # Zone ID per bus [n_bus]

    # Impedance matrix (sparse representation)
    impedance_matrix: torch.Tensor  # Dense [n_bus, n_bus] or None

    # Metadata
    n_bus: int
    n_gen: int
    n_line: int
    n_zone: int
    config: SyntheticGridConfig


class SyntheticGridGenerator:
    """
    Generates reproducible large-scale synthetic power grids.

    The generator creates a hierarchical zone-based grid with:
      1. Zones connected by inter-zone tie lines (HV backbone)
      2. Intra-zone meshed networks with radial feeders
      3. Realistic impedance/load/generation distributions
      4. Communication network overlay

    Reference topology model: nested small-world (Watts-Strogatz within
    zones, preferential attachment between zones), following structural
    properties observed in real North American grids (Birchfield et al.,
    IEEE Trans. Power Systems, vol. 32, no. 4, 2017).
    """

    def __init__(
        self,
        n_buses: int = 10_000,
        seed: int = 42,
        config: Optional[SyntheticGridConfig] = None,
    ):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = SyntheticGridConfig(n_buses=n_buses, seed=seed)
        self.rng = np.random.RandomState(self.cfg.seed)

    def generate(self) -> SyntheticGridState:
        """Generate a complete synthetic grid.

        Returns:
            SyntheticGridState with all power system and communication data.
        """
        # Reset RNG for reproducibility
        self.rng = np.random.RandomState(self.cfg.seed)

        # Step 1: Assign buses to zones
        zone_ids = self._assign_zones()

        # Step 2: Generate topology (edges)
        edges, is_tie_line = self._generate_topology(zone_ids)

        # Step 3: Assign electrical parameters to lines
        r_pu, x_pu, b_pu = self._generate_line_parameters(len(edges), is_tie_line)

        # Step 4: Place generators and loads
        gen_bus_idx, load_bus_idx = self._place_gen_load(zone_ids)

        # Step 5: Generate bus injections (P, Q, V, theta)
        P, Q, V, theta, P_gen, Q_gen = self._generate_bus_state(
            gen_bus_idx, load_bus_idx
        )

        # Step 6: Generator dynamics parameters
        inertia, damping = self._generate_dynamics(len(gen_bus_idx))

        # Step 7: Communication overlay
        comm_latency, comm_bw, comm_loss = self._generate_comm_network(
            gen_bus_idx, zone_ids, edges
        )

        # Step 8: Build edge_index and impedance tensors
        edge_array = np.array(edges, dtype=np.int64).T  # [2, n_lines]
        z_mag = np.sqrt(r_pu ** 2 + x_pu ** 2)

        # Step 9: Build sparse impedance matrix
        impedance_matrix = self._build_impedance_matrix(
            self.cfg.n_buses, edges, z_mag
        )

        n_line = len(edges)
        n_gen = len(gen_bus_idx)

        return SyntheticGridState(
            P=torch.tensor(P, dtype=torch.float32),
            Q=torch.tensor(Q, dtype=torch.float32),
            V=torch.tensor(V, dtype=torch.float32),
            theta=torch.tensor(theta, dtype=torch.float32),
            P_gen=torch.tensor(P_gen, dtype=torch.float32),
            Q_gen=torch.tensor(Q_gen, dtype=torch.float32),
            edge_index=torch.tensor(edge_array, dtype=torch.long),
            line_impedance=torch.tensor(z_mag, dtype=torch.float32),
            gen_bus_idx=torch.tensor(gen_bus_idx, dtype=torch.long),
            load_bus_idx=torch.tensor(load_bus_idx, dtype=torch.long),
            inertia=torch.tensor(inertia, dtype=torch.float32),
            damping=torch.tensor(damping, dtype=torch.float32),
            comm_latency_ms=torch.tensor(comm_latency, dtype=torch.float32),
            comm_bandwidth=torch.tensor(comm_bw, dtype=torch.float32),
            comm_loss_rate=torch.tensor(comm_loss, dtype=torch.float32),
            zone_ids=torch.tensor(zone_ids, dtype=torch.long),
            impedance_matrix=impedance_matrix,
            n_bus=self.cfg.n_buses,
            n_gen=n_gen,
            n_line=n_line,
            n_zone=self.cfg.n_zones,
            config=self.cfg,
        )

    # ------------------------------------------------------------------
    # Step 1: Zone assignment
    # ------------------------------------------------------------------
    def _assign_zones(self) -> np.ndarray:
        """Assign each bus to a zone with slight size variation."""
        n = self.cfg.n_buses
        nz = self.cfg.n_zones

        # Zone sizes follow a Dirichlet distribution for realistic variation
        # (some zones larger than others, like real interconnections)
        alpha = np.ones(nz) * 5.0  # Concentrated → roughly equal zones
        proportions = self.rng.dirichlet(alpha)
        sizes = np.round(proportions * n).astype(int)

        # Adjust to hit exactly n_buses
        diff = n - sizes.sum()
        for i in range(abs(diff)):
            idx = i % nz
            sizes[idx] += 1 if diff > 0 else -1

        zone_ids = np.concatenate([np.full(sz, z) for z, sz in enumerate(sizes)])
        return zone_ids

    # ------------------------------------------------------------------
    # Step 2: Topology generation
    # ------------------------------------------------------------------
    def _generate_topology(
        self, zone_ids: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Generate grid topology with intra- and inter-zone connections.

        Uses a Watts-Strogatz-inspired approach within each zone (ring lattice
        with random rewiring) and preferential-attachment tie lines between zones.
        """
        n = self.cfg.n_buses
        nz = self.cfg.n_zones
        edges = []
        is_tie = []

        # --- Intra-zone edges ---
        for z in range(nz):
            bus_idx = np.where(zone_ids == z)[0]
            nz_buses = len(bus_idx)
            if nz_buses < 2:
                continue

            # Backbone ring: connect backbone buses in a ring
            n_backbone = min(self.cfg.backbone_buses_per_zone, nz_buses)
            backbone = bus_idx[:n_backbone]

            for i in range(n_backbone):
                j = (i + 1) % n_backbone
                edges.append((backbone[i], backbone[j]))
                is_tie.append(False)

            # Add cross-ring connections for meshing (k-nearest on ring)
            for i in range(n_backbone):
                for step in [2, 3]:
                    j = (i + step) % n_backbone
                    if self.rng.random() < 0.4:
                        edges.append((backbone[i], backbone[j]))
                        is_tie.append(False)

            # Connect non-backbone buses to nearest backbone bus (radial feeders)
            feeder_buses = bus_idx[n_backbone:]
            if len(feeder_buses) > 0:
                # Assign each feeder bus to a random backbone bus
                assignments = self.rng.randint(0, n_backbone, size=len(feeder_buses))
                for fb, bb_idx in zip(feeder_buses, assignments):
                    edges.append((backbone[bb_idx], fb))
                    is_tie.append(False)

            # Add some random intra-zone meshing for non-radial structure
            target_extra = int(
                nz_buses * (nz_buses - 1) / 2 * self.cfg.intra_zone_density
            )
            target_extra = max(0, target_extra - len(backbone))
            added = 0
            attempts = 0
            edge_set = set((min(a, b), max(a, b)) for a, b in edges)

            while added < target_extra and attempts < target_extra * 5:
                a = bus_idx[self.rng.randint(0, nz_buses)]
                b = bus_idx[self.rng.randint(0, nz_buses)]
                if a != b:
                    key = (min(a, b), max(a, b))
                    if key not in edge_set:
                        edges.append((a, b))
                        is_tie.append(False)
                        edge_set.add(key)
                        added += 1
                attempts += 1

        # --- Inter-zone tie lines ---
        # Connect adjacent zones (ring of zones + some random long-distance)
        zone_backbones = {}
        for z in range(nz):
            bus_idx = np.where(zone_ids == z)[0]
            n_bb = min(self.cfg.backbone_buses_per_zone, len(bus_idx))
            zone_backbones[z] = bus_idx[:n_bb]

        # Ring of zones
        for z in range(nz):
            z_next = (z + 1) % nz
            bb_a = zone_backbones[z]
            bb_b = zone_backbones[z_next]
            n_ties = min(self.cfg.inter_zone_lines, len(bb_a), len(bb_b))
            for t in range(n_ties):
                a = bb_a[self.rng.randint(0, len(bb_a))]
                b = bb_b[self.rng.randint(0, len(bb_b))]
                edges.append((a, b))
                is_tie.append(True)

        # Random long-distance ties (5% of zone pairs)
        for z1 in range(nz):
            for z2 in range(z1 + 2, nz):
                if self.rng.random() < 0.05:
                    bb_a = zone_backbones[z1]
                    bb_b = zone_backbones[z2]
                    a = bb_a[self.rng.randint(0, len(bb_a))]
                    b = bb_b[self.rng.randint(0, len(bb_b))]
                    edges.append((a, b))
                    is_tie.append(True)

        return edges, np.array(is_tie, dtype=bool)

    # ------------------------------------------------------------------
    # Step 3: Line parameters
    # ------------------------------------------------------------------
    def _generate_line_parameters(
        self, n_lines: int, is_tie_line: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic line impedance parameters.

        Tie lines (inter-zone) have lower impedance (higher voltage).
        Distribution lines have higher impedance.
        """
        r_pu = np.abs(
            self.rng.normal(self.cfg.r_pu_mean, self.cfg.r_pu_std, n_lines)
        )
        x_pu = np.abs(
            self.rng.normal(self.cfg.x_pu_mean, self.cfg.x_pu_std, n_lines)
        )
        b_pu = np.abs(
            self.rng.normal(self.cfg.b_pu_mean, self.cfg.b_pu_mean * 0.3, n_lines)
        )

        # Tie lines: lower impedance (EHV), scale down by 0.3x
        r_pu[is_tie_line] *= 0.3
        x_pu[is_tie_line] *= 0.3
        b_pu[is_tie_line] *= 2.0  # Higher charging for long lines

        # Enforce X/R ratio > 2 (realistic for transmission)
        low_xr = x_pu < 2.0 * r_pu
        x_pu[low_xr] = 2.0 * r_pu[low_xr] + self.rng.uniform(0, 0.02, low_xr.sum())

        # Clamp to physical range
        r_pu = np.clip(r_pu, 0.0005, 0.5)
        x_pu = np.clip(x_pu, 0.005, 1.0)
        b_pu = np.clip(b_pu, 0.001, 0.2)

        return r_pu, x_pu, b_pu

    # ------------------------------------------------------------------
    # Step 4: Generator and load placement
    # ------------------------------------------------------------------
    def _place_gen_load(
        self, zone_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place generators and loads with realistic spatial distribution.

        Generators are placed preferentially on backbone buses.
        Loads are distributed across most non-generator buses.
        """
        n = self.cfg.n_buses
        nz = self.cfg.n_zones

        # Target counts
        n_gen_target = max(2, int(n * self.cfg.gen_ratio))
        n_load_target = max(2, int(n * self.cfg.load_ratio))

        # Generators: prefer backbone buses, ensure at least 1 per zone
        gen_candidates = []
        for z in range(nz):
            zone_buses = np.where(zone_ids == z)[0]
            n_bb = min(self.cfg.backbone_buses_per_zone, len(zone_buses))
            backbone = zone_buses[:n_bb]
            # Always place at least 1 generator per zone (on backbone)
            gen_candidates.append(backbone[0])

        # Fill remaining generators, preferring backbone
        remaining = n_gen_target - len(gen_candidates)
        gen_set = set(gen_candidates)
        all_buses = np.arange(n)
        self.rng.shuffle(all_buses)

        # Backbone buses first
        for z in range(nz):
            zone_buses = np.where(zone_ids == z)[0]
            n_bb = min(self.cfg.backbone_buses_per_zone, len(zone_buses))
            for b in zone_buses[:n_bb]:
                if remaining <= 0:
                    break
                if b not in gen_set:
                    gen_candidates.append(b)
                    gen_set.add(b)
                    remaining -= 1

        # Then random buses if still needed
        for b in all_buses:
            if remaining <= 0:
                break
            if b not in gen_set:
                gen_candidates.append(b)
                gen_set.add(b)
                remaining -= 1

        gen_bus_idx = np.array(sorted(gen_candidates[:n_gen_target]))

        # Loads: random subset of non-generator buses (plus some gen buses)
        non_gen = np.array([b for b in range(n) if b not in gen_set])
        self.rng.shuffle(non_gen)
        load_bus_idx = non_gen[: min(n_load_target, len(non_gen))]

        # Some generator buses also have co-located load (30% chance)
        for g in gen_bus_idx:
            if self.rng.random() < 0.3 and len(load_bus_idx) < n_load_target:
                load_bus_idx = np.append(load_bus_idx, g)

        load_bus_idx = np.sort(np.unique(load_bus_idx))

        return gen_bus_idx, load_bus_idx

    # ------------------------------------------------------------------
    # Step 5: Bus state (P, Q, V, theta)
    # ------------------------------------------------------------------
    def _generate_bus_state(
        self,
        gen_bus_idx: np.ndarray,
        load_bus_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Generate bus power injections and voltage profile."""
        n = self.cfg.n_buses
        n_gen = len(gen_bus_idx)

        # Generator outputs (lognormal for realistic heavy-tail)
        P_gen = np.abs(
            self.rng.lognormal(
                mean=np.log(self.cfg.p_gen_mean_mw) - 0.5,
                sigma=0.8,
                size=n_gen,
            )
        )
        P_gen = np.clip(P_gen, 10.0, 2000.0)  # MW range

        # Generator reactive power (based on power factor)
        pf_gen = self.rng.uniform(0.85, 0.98, n_gen)
        Q_gen = P_gen * np.tan(np.arccos(pf_gen)) * self.rng.choice(
            [-1, 1], size=n_gen, p=[0.3, 0.7]
        )

        # Load demands (gamma distribution for realistic shape)
        n_load = len(load_bus_idx)
        P_load = self.rng.gamma(
            shape=2.0,
            scale=self.cfg.p_load_mean_mw / 2.0,
            size=n_load,
        )
        P_load = np.clip(P_load, 0.5, 500.0)

        Q_load = P_load * np.tan(np.arccos(self.cfg.power_factor))
        Q_load *= self.rng.uniform(0.8, 1.2, n_load)  # Some variation

        # Build bus injection vectors
        P = np.zeros(n, dtype=np.float64)
        Q = np.zeros(n, dtype=np.float64)

        P[gen_bus_idx] += P_gen
        Q[gen_bus_idx] += Q_gen
        P[load_bus_idx] -= P_load
        Q[load_bus_idx] -= Q_load

        # Balance: scale generation to match load + 5% losses
        total_load = P_load.sum()
        total_gen = P_gen.sum()
        if total_gen > 0:
            scale = (total_load * 1.05) / total_gen
            P_gen *= scale
            P[gen_bus_idx] = 0
            P[gen_bus_idx] += P_gen
            P[load_bus_idx] = 0
            P[load_bus_idx] -= P_load

        # Voltage profile: 1.0 pu nominal with small perturbations
        V = 1.0 + self.rng.normal(0, 0.015, n)
        V = np.clip(V, 0.94, 1.06)
        # Generator buses have tighter voltage (voltage-controlled)
        V[gen_bus_idx] = 1.0 + self.rng.normal(0, 0.005, n_gen)
        V[gen_bus_idx] = np.clip(V[gen_bus_idx], 0.98, 1.05)

        # Voltage angles: small, increasing with electrical distance from slack
        theta = self.rng.normal(0, 0.05, n)  # radians, small
        theta[gen_bus_idx[0]] = 0.0  # Slack bus reference

        return (
            P.astype(np.float32),
            Q.astype(np.float32),
            V.astype(np.float32),
            theta.astype(np.float32),
            P_gen.astype(np.float32),
            Q_gen.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Step 6: Generator dynamics
    # ------------------------------------------------------------------
    def _generate_dynamics(
        self, n_gen: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate inertia and damping constants for swing equation."""
        # Inertia H (seconds): large thermal units ~ 6-8s, small ~ 2-4s
        inertia = np.abs(
            self.rng.normal(self.cfg.inertia_mean, self.cfg.inertia_std, n_gen)
        )
        inertia = np.clip(inertia, 1.0, 15.0)

        # Damping D: typically 1-5% of inertia
        damping = inertia * self.rng.uniform(0.01, 0.05, n_gen)

        return inertia.astype(np.float32), damping.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 7: Communication network overlay
    # ------------------------------------------------------------------
    def _generate_comm_network(
        self,
        gen_bus_idx: np.ndarray,
        zone_ids: np.ndarray,
        edges: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate communication latency for each generator's control link.

        Model: latency = backbone_hops * backbone_latency + access_latency + jitter
        """
        n_gen = len(gen_bus_idx)

        # Base latency: backbone (zone-level) + access (last mile)
        gen_zones = zone_ids[gen_bus_idx]

        # Generators in zones further from the control center have higher latency
        # Simulate control center in zone 0
        zone_distance = np.minimum(gen_zones, self.cfg.n_zones - gen_zones)
        backbone_latency = (
            zone_distance * self.cfg.comm_backbone_latency_ms
            + self.cfg.comm_access_latency_ms
        )

        # Add lognormal jitter (realistic for IP networks)
        jitter = self.rng.lognormal(
            mean=np.log(self.cfg.comm_jitter_std_ms),
            sigma=0.5,
            size=n_gen,
        )
        jitter = np.clip(jitter, 0.5, 50.0)

        latency = backbone_latency + jitter
        latency = np.clip(latency, 5.0, 500.0).astype(np.float32)

        # Bandwidth: backbone is shared, access varies
        bandwidth = self.rng.uniform(
            self.cfg.comm_bandwidth_gbps * 0.1,
            self.cfg.comm_bandwidth_gbps,
            n_gen,
        ).astype(np.float32)

        # Packet loss rate: increases with distance
        loss_rate = (
            self.cfg.comm_loss_rate
            * (1.0 + zone_distance * 0.5)
            * self.rng.uniform(0.5, 2.0, n_gen)
        )
        loss_rate = np.clip(loss_rate, 0.0001, 0.05).astype(np.float32)

        return latency, bandwidth, loss_rate

    # ------------------------------------------------------------------
    # Step 8: Impedance matrix
    # ------------------------------------------------------------------
    def _build_impedance_matrix(
        self,
        n_buses: int,
        edges: List[Tuple[int, int]],
        z_mag: np.ndarray,
    ) -> torch.Tensor:
        """Build symmetric impedance matrix (dense for compatibility).

        For grids > 5000 buses, returns a sparse-compatible representation
        stored as dense zeros with only edge entries filled (the downstream
        code uses masking so this is functionally equivalent).
        """
        # For very large grids, full dense matrix is too large.
        # We store impedance only along edges in a sparse format.
        if n_buses > 5000:
            # Return sparse COO tensor
            row_idx = []
            col_idx = []
            values = []
            for (i, j), z in zip(edges, z_mag):
                row_idx.extend([i, j])
                col_idx.extend([j, i])
                values.extend([z, z])
            indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
            vals = torch.tensor(values, dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices, vals, size=(n_buses, n_buses)
            ).coalesce()
        else:
            Z = torch.zeros(n_buses, n_buses)
            for (i, j), z in zip(edges, z_mag):
                Z[i, j] = z
                Z[j, i] = z
            return Z

    # ------------------------------------------------------------------
    # Validation & statistics
    # ------------------------------------------------------------------
    def validate(self, state: SyntheticGridState) -> Dict[str, float]:
        """Compute structural validation metrics for the synthetic grid.

        Checks that the generated grid exhibits realistic properties:
          - Degree distribution (mean, max, power-law exponent)
          - Clustering coefficient
          - Power balance (gen ≈ load + losses)
          - Impedance statistics
        """
        n = state.n_bus
        ei = state.edge_index.numpy()

        # Degree distribution
        degrees = np.zeros(n, dtype=int)
        for i in range(ei.shape[1]):
            degrees[ei[0, i]] += 1
            degrees[ei[1, i]] += 1
        mean_deg = degrees.mean()
        max_deg = degrees.max()

        # Clustering coefficient (sample for large grids)
        sample_size = min(500, n)
        sample_buses = np.random.choice(n, sample_size, replace=False)
        adj = {}
        for i in range(ei.shape[1]):
            a, b = int(ei[0, i]), int(ei[1, i])
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        cc_values = []
        for bus in sample_buses:
            neighbors = list(adj.get(bus, set()))
            k = len(neighbors)
            if k < 2:
                continue
            triangles = 0
            for i_n in range(k):
                for j_n in range(i_n + 1, k):
                    if neighbors[j_n] in adj.get(neighbors[i_n], set()):
                        triangles += 1
            cc_values.append(2.0 * triangles / (k * (k - 1)))
        clustering = np.mean(cc_values) if cc_values else 0.0

        # Power balance
        total_gen = float(state.P_gen.sum())
        total_load = float((-state.P[state.load_bus_idx]).sum())
        balance_pct = (
            abs(total_gen - total_load) / max(total_load, 1.0) * 100
        )

        # Impedance stats
        z = state.line_impedance.numpy()

        return {
            "n_buses": n,
            "n_generators": state.n_gen,
            "n_lines": state.n_line,
            "n_zones": state.n_zone,
            "mean_degree": float(mean_deg),
            "max_degree": int(max_deg),
            "clustering_coefficient": float(clustering),
            "total_generation_mw": float(total_gen),
            "total_load_mw": float(total_load),
            "gen_load_balance_pct": float(balance_pct),
            "mean_impedance_pu": float(z.mean()),
            "std_impedance_pu": float(z.std()),
            "mean_comm_latency_ms": float(state.comm_latency_ms.mean()),
            "std_comm_latency_ms": float(state.comm_latency_ms.std()),
            "gen_per_zone": float(state.n_gen / state.n_zone),
        }

    @staticmethod
    def get_available_sizes() -> Dict[str, SyntheticGridConfig]:
        """Pre-defined grid sizes for reproducible experiments."""
        return {
            "small": SyntheticGridConfig(n_buses=1_000, n_zones=4, seed=42),
            "medium": SyntheticGridConfig(n_buses=5_000, n_zones=10, seed=42),
            "large": SyntheticGridConfig(n_buses=10_000, n_zones=20, seed=42),
            "xlarge": SyntheticGridConfig(n_buses=20_000, n_zones=40, seed=42),
        }


class SyntheticCaseLoader:
    """Adapter that makes SyntheticGridGenerator match the IEEECaseLoader interface.

    Used by PowerGridDataset when case_id >= 1000 (synthetic grids).
    """

    SUPPORTED_SYNTHETIC = {
        1000: SyntheticGridConfig(n_buses=1_000, n_zones=4, seed=42),
        5000: SyntheticGridConfig(n_buses=5_000, n_zones=10, seed=42),
        10000: SyntheticGridConfig(n_buses=10_000, n_zones=20, seed=42),
        20000: SyntheticGridConfig(n_buses=20_000, n_zones=40, seed=42),
    }

    def __init__(self, case_id: int, seed: int = 42):
        if case_id not in self.SUPPORTED_SYNTHETIC:
            raise ValueError(
                f"Synthetic case {case_id} not supported. "
                f"Available: {list(self.SUPPORTED_SYNTHETIC.keys())}"
            )
        self.case_id = case_id
        cfg = SyntheticGridConfig(
            n_buses=self.SUPPORTED_SYNTHETIC[case_id].n_buses,
            n_zones=self.SUPPORTED_SYNTHETIC[case_id].n_zones,
            seed=seed,
        )
        gen = SyntheticGridGenerator(config=cfg)
        self._state = gen.generate()

    def get_state(self):
        """Return the SyntheticGridState (compatible with PowerSystemState)."""
        return self._state

    def load(self) -> Dict:
        """Return dict matching IEEECaseLoader.load() interface."""
        s = self._state

        # Compute lambda_min from swing equation (same logic as IEEECaseLoader)
        _, lambda_min = self.get_eigenvalues()

        # Build impedance matrix — skip dense for large grids (>5000 buses)
        if s.n_bus > 5000:
            # Dense 10K×10K = 400MB — skip to avoid OOM in attention
            impedance_matrix = None
        elif s.impedance_matrix.is_sparse:
            n = s.n_bus
            impedance_matrix = torch.ones(n, n) * 1e6
            row, col = s.edge_index
            for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
                impedance_matrix[r, c] = s.line_impedance[i]
                impedance_matrix[c, r] = s.line_impedance[i]
            impedance_matrix.fill_diagonal_(0)
        else:
            impedance_matrix = s.impedance_matrix.clone()
            impedance_matrix[impedance_matrix == 0] = 1e6
            impedance_matrix.fill_diagonal_(0)

        return {
            'n_buses': s.n_bus,
            'n_generators': s.n_gen,
            'n_lines': s.n_line,
            'edge_index': s.edge_index,
            'V': s.V,
            'theta': s.theta,
            'P_load': s.P,
            'Q_load': s.Q,
            'P_gen': s.P_gen,
            'Q_gen': s.Q_gen,
            'gen_buses': s.gen_bus_idx,
            'lambda_min': lambda_min.item(),
            'impedance_matrix': impedance_matrix,
            'line_impedance': s.line_impedance,
        }

    def get_system_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build A, B from inertia/damping (matches IEEECaseLoader)."""
        n = self._state.n_gen
        M = self._state.inertia
        D = self._state.damping
        A = torch.zeros(2 * n, 2 * n)
        A[:n, n:] = torch.eye(n)
        A[n:, n:] = -torch.diag(D / M)
        B = torch.zeros(2 * n, n)
        B[n:, :] = torch.diag(1.0 / M)
        return A, B

    def get_eigenvalues(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues of system matrix."""
        A, _ = self.get_system_matrices()
        eigenvalues = torch.linalg.eigvals(A)
        lambda_min = eigenvalues.real.min()
        return eigenvalues, lambda_min


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def generate_synthetic_grid(
    n_buses: int = 10_000, seed: int = 42
) -> SyntheticGridState:
    """One-liner to generate a synthetic grid."""
    gen = SyntheticGridGenerator(n_buses=n_buses, seed=seed)
    return gen.generate()


def validate_synthetic_grid(
    n_buses: int = 10_000, seed: int = 42
) -> Dict[str, float]:
    """Generate and validate a synthetic grid, returning metrics."""
    gen = SyntheticGridGenerator(n_buses=n_buses, seed=seed)
    state = gen.generate()
    return gen.validate(state)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(
        description="Generate and validate a synthetic large-scale power grid"
    )
    parser.add_argument("--n-buses", type=int, default=10_000)
    parser.add_argument("--n-zones", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Save grid state to .pt file")
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    cfg = SyntheticGridConfig(
        n_buses=args.n_buses, n_zones=args.n_zones, seed=args.seed
    )
    gen = SyntheticGridGenerator(config=cfg)

    print(f"Generating synthetic grid: {args.n_buses} buses, "
          f"{args.n_zones} zones, seed={args.seed}")
    t0 = time.time()
    state = gen.generate()
    dt = time.time() - t0
    print(f"Generated in {dt:.2f}s")

    if args.validate:
        metrics = gen.validate(state)
        print("\n=== Grid Validation Metrics ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")

    if args.output:
        torch.save({
            "P": state.P, "Q": state.Q, "V": state.V, "theta": state.theta,
            "P_gen": state.P_gen, "Q_gen": state.Q_gen,
            "edge_index": state.edge_index,
            "line_impedance": state.line_impedance,
            "gen_bus_idx": state.gen_bus_idx,
            "load_bus_idx": state.load_bus_idx,
            "inertia": state.inertia, "damping": state.damping,
            "comm_latency_ms": state.comm_latency_ms,
            "comm_bandwidth": state.comm_bandwidth,
            "comm_loss_rate": state.comm_loss_rate,
            "zone_ids": state.zone_ids,
            "n_bus": state.n_bus, "n_gen": state.n_gen,
            "n_line": state.n_line, "n_zone": state.n_zone,
            "seed": args.seed,
        }, args.output)
        print(f"\nSaved to {args.output}")
