#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/31/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Learnable Coupling Constants and Stability Margin Computation

Implements Theorem 1: Delay-Stability Coupling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LearnableCouplingConstants(nn.Module):
    """
    Learnable coupling constants K_i for each generator.

    From Theorem 1:
        λ_min(τ) ≥ λ_min(0) - Σ_i (K_i · τ_i / τ_max,i)

    K_i represents the sensitivity of stability to delay at generator i.
    We learn K_i from data while ensuring K_i > 0.
    """

    def __init__(
        self,
        n_generators: int,
        init_scale: float = 0.1,
        learnable: bool = True,
    ):
        """
        Args:
            n_generators: Number of generators in the system
            init_scale: Initial scale for K_i values
            learnable: Whether K_i should be learnable
        """
        super().__init__()
        self.n_generators = n_generators

        # Use log parameterization to ensure K_i > 0
        # K_i = exp(log_K_i)
        init_log_K = torch.log(torch.ones(n_generators) * init_scale)

        if learnable:
            self.log_K = nn.Parameter(init_log_K)
        else:
            self.register_buffer("log_K", init_log_K)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            K: Coupling constants [n_generators], all positive
        """
        return torch.exp(self.log_K)

    def get_K_values(self) -> torch.Tensor:
        """Get current K_i values (detached)."""
        return self.forward().detach()


class StabilityMarginComputer(nn.Module):
    """
    Computes the stability margin ρ(τ) from Theorem 1.

    ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i)

    The system is stable when ρ(τ) > 0.
    """

    def __init__(
        self,
        n_generators: int,
        tau_max: Optional[torch.Tensor] = None,
        init_scale: float = 0.1,
    ):
        """
        Args:
            n_generators: Number of generators
            tau_max: Maximum delay margins [n_generators] in seconds
                     If None, defaults to 0.5s (500ms) for all
            init_scale: Initial scale for learnable K_i
        """
        super().__init__()

        self.coupling = LearnableCouplingConstants(n_generators, init_scale)

        # τ_max: delay margin per generator (typically 150-1500 ms)
        if tau_max is None:
            tau_max = torch.ones(n_generators) * 0.5  # 500 ms default
        self.register_buffer("tau_max", tau_max)

    def compute_stability_margin(
        self,
        tau: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stability margin ρ(τ).

        Args:
            tau: Communication delays [batch, n_generators] in seconds
            lambda_min_0: Minimum eigenvalue at zero delay [batch] or scalar

        Returns:
            rho: Stability margin [batch]
        """
        K = self.coupling()  # [n_generators]

        # Normalized delay contribution per generator
        # τ_i / τ_max,i gives unitless ratio
        tau_normalized = tau / self.tau_max.unsqueeze(0)  # [batch, n_gen]

        # Weighted sum: Σ_i K_i · (τ_i / τ_max,i)
        delay_contribution = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)  # [batch]

        # Stability margin
        rho = torch.abs(lambda_min_0) - delay_contribution

        return rho

    def forward(
        self,
        tau: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning stability margin and coupling constants.

        Args:
            tau: Communication delays [batch, n_generators]
            lambda_min_0: Minimum eigenvalue at zero delay

        Returns:
            rho: Stability margin [batch]
            K: Coupling constants [n_generators]
        """
        rho = self.compute_stability_margin(tau, lambda_min_0)
        K = self.coupling()
        return rho, K


class StabilityMarginComputerV2(nn.Module):
    """
    V2 Stability Margin with optional 2nd-order Padé correction (Q1.5).

    Order 1 (original):
        ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i)

    Order 2 (new):
        ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i) - Σ_i (K2_i · (τ_i / τ_max,i)²)

    The quadratic correction improves accuracy at higher delays (>200ms)
    where the 1st-order Padé approximation has ~4% error.
    """

    def __init__(
        self,
        n_generators: int,
        tau_max: Optional[torch.Tensor] = None,
        init_scale: float = 0.1,
        order: int = 1,
    ):
        """
        Args:
            n_generators: Number of generators
            tau_max: Maximum delay margins [n_generators] in seconds
            init_scale: Initial scale for learnable K_i
            order: Padé approximation order (1 or 2)
        """
        super().__init__()
        self.order = order

        self.coupling = LearnableCouplingConstants(n_generators, init_scale)

        # 2nd-order correction constants (Q1.5)
        if order >= 2:
            # K2_i: quadratic coupling, initialized small
            init_log_K2 = torch.log(torch.ones(n_generators) * init_scale * 0.1)
            self.log_K2 = nn.Parameter(init_log_K2)

        if tau_max is None:
            tau_max = torch.ones(n_generators) * 0.5
        self.register_buffer("tau_max", tau_max)

    def compute_stability_margin(
        self,
        tau: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stability margin ρ(τ) with optional 2nd-order correction.

        Args:
            tau: Communication delays [batch, n_generators] in seconds
            lambda_min_0: Minimum eigenvalue at zero delay [batch] or scalar

        Returns:
            rho: Stability margin [batch]
        """
        K = self.coupling()
        tau_normalized = tau / self.tau_max.unsqueeze(0)

        # 1st-order: Σ K_i · (τ_i / τ_max,i)
        delay_contribution = (K.unsqueeze(0) * tau_normalized).sum(dim=-1)
        rho = torch.abs(lambda_min_0) - delay_contribution

        # 2nd-order correction
        if self.order >= 2:
            K2 = torch.exp(self.log_K2)
            tau_norm_sq = tau_normalized ** 2
            quadratic_correction = (K2.unsqueeze(0) * tau_norm_sq).sum(dim=-1)
            rho = rho - quadratic_correction

        return rho

    def forward(
        self,
        tau: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning stability margin and coupling constants.

        Returns:
            rho: Stability margin [batch]
            K: First-order coupling constants [n_generators]
        """
        rho = self.compute_stability_margin(tau, lambda_min_0)
        K = self.coupling()
        return rho, K

    def get_all_K(self) -> dict:
        """Return all coupling constants."""
        result = {'K1': self.coupling().detach()}
        if self.order >= 2:
            result['K2'] = torch.exp(self.log_K2).detach()
        return result


def compute_k_init_scale(
    n_generators: int,
    lambda_min_0: float,
    safety_factor: float = 0.9,
) -> float:
    """
    Compute grid-size-aware K initialization scale.

    Ensures SUM(K_i * tau_i/tau_max_i) ~ safety_factor * |lambda_min(0)|
    so that rho starts positive for any grid size.

    k_init = safety_factor * |lambda_min(0)| / n_generators
    """
    abs_lambda = abs(float(lambda_min_0))
    if abs_lambda < 1e-10 or n_generators <= 0:
        return 0.1
    k_init = safety_factor * abs_lambda / n_generators
    return max(min(k_init, 1.0), 1e-6)


class AdaptiveKInit(nn.Module):
    """Adaptive per-generator K_init via softmax-MLP weighting (R1.10).

    The major-revision response replaces the uniform formula
    K_init = s * |lambda_min(0)| / n_g (which the reviewer flagged as a
    'new bottle of old wine') with a data-driven redistribution

        K_init,i = s * |lambda_min(0)| * w_i,
        w_i = softmax( g_phi(M_i, D_i, deg(i), centrality(i)) ),

    where g_phi is a small two-layer MLP. The softmax guarantees
    Sum_i K_init,i = s * |lambda_min(0)| <= |lambda_min(0)| so the
    budget-preservation invariant of Theorem 1 holds for any phi.

    Per-generator features default to a four-vector
    (M_i, D_i, deg(i), centrality(i)). When the host model lacks topology
    metadata the loader supplies zero-filled features and the softmax
    reduces to a near-uniform distribution, recovering the legacy
    behaviour as a special case.
    """

    def __init__(
        self,
        n_generators: int,
        feature_dim: int = 4,
        hidden_dim: int = 16,
        safety_factor: float = 0.9,
        learnable: bool = True,
    ):
        super().__init__()
        self.n_generators = int(n_generators)
        self.feature_dim = int(feature_dim)
        self.safety_factor = float(safety_factor)

        layers = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if not learnable:
            for param in layers.parameters():
                param.requires_grad_(False)
        self.score_mlp = layers

        # Default features: zeros => uniform softmax (matches legacy K_init).
        self.register_buffer(
            'default_features',
            torch.zeros(self.n_generators, self.feature_dim),
        )

    def set_features(self, features: torch.Tensor) -> None:
        """Override the per-generator feature vector (no grad)."""
        feats = torch.as_tensor(features, dtype=self.default_features.dtype)
        if feats.shape != self.default_features.shape:
            raise ValueError(
                f"AdaptiveKInit features shape {tuple(feats.shape)} != "
                f"({self.n_generators}, {self.feature_dim})"
            )
        self.default_features.copy_(feats.detach())

    def forward(
        self,
        lambda_min_0: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the per-generator K_init vector [n_generators]."""
        feats = features if features is not None else self.default_features
        scores = self.score_mlp(feats).squeeze(-1)            # [n_gen]
        weights = torch.softmax(scores, dim=-1)
        k_total = self.safety_factor * torch.abs(lambda_min_0)
        return k_total * weights


def compute_adaptive_k_init(
    n_generators: int,
    lambda_min_0: float,
    inertia: Optional[torch.Tensor] = None,
    damping: Optional[torch.Tensor] = None,
    degree: Optional[torch.Tensor] = None,
    centrality: Optional[torch.Tensor] = None,
    safety_factor: float = 0.9,
) -> torch.Tensor:
    """Convenience wrapper that produces a per-generator K_init vector.

    Computes the adaptive distribution analytically from physical features
    (no learning; uses the same softmax-over-features structure to give
    the host pipeline a stronger prior than the uniform formula). Falls
    back to the uniform K_init when no features are provided so the
    function can be used as a drop-in replacement for
    :func:`compute_k_init_scale` in scripts that expect a per-generator
    vector instead of a scalar.
    """
    abs_lambda = abs(float(lambda_min_0))
    if abs_lambda < 1e-10 or n_generators <= 0:
        return torch.full((max(n_generators, 1),), 0.1)
    feature_columns = []
    for tensor in (inertia, damping, degree, centrality):
        if tensor is None:
            feature_columns.append(torch.zeros(n_generators))
        else:
            t = torch.as_tensor(tensor, dtype=torch.float32).flatten()
            if t.numel() != n_generators:
                feature_columns.append(torch.zeros(n_generators))
            else:
                feature_columns.append((t - t.mean()) / (t.std(unbiased=False) + 1e-6))
    feats = torch.stack(feature_columns, dim=-1)              # [n_gen, 4]
    score = feats.sum(dim=-1)
    weights = torch.softmax(score, dim=-1)
    return safety_factor * abs_lambda * weights


class DelayMarginEstimator(nn.Module):
    """
    Estimates τ_max for each generator from system dynamics.

    From control theory:
        τ_max ≈ π / (2 · ω_c)

    where ω_c is the crossover frequency of the control loop.
    """

    def __init__(self, method: str = "fixed", fixed_value: float = 0.5):
        """
        Args:
            method: "fixed", "crossover", or "eigenvalue"
            fixed_value: Fixed τ_max in seconds (if method="fixed")
        """
        super().__init__()
        self.method = method
        self.fixed_value = fixed_value

    def estimate_from_crossover(
        self,
        omega_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate τ_max from crossover frequency.

        Args:
            omega_c: Crossover frequencies [n_generators] in rad/s

        Returns:
            tau_max: Delay margins [n_generators] in seconds
        """
        import math
        return math.pi / (2.0 * omega_c)

    def estimate_from_eigenvalue(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate τ_max from closed-loop eigenvalue analysis.

        Uses Padé approximation to find delay margin.

        Args:
            A: System matrix [n, n]
            B: Input matrix [n, m]
            K: Control gain matrix [m, n]

        Returns:
            tau_max: Delay margin (scalar, same for all)
        """
        # Closed-loop matrix: A_cl = A - B @ K
        A_cl = A - B @ K

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(A_cl)

        # Dominant pole (largest real part)
        real_parts = eigenvalues.real
        dominant_idx = torch.argmax(real_parts)
        dominant_pole = eigenvalues[dominant_idx]

        # Approximate τ_max from imaginary part
        omega_n = torch.abs(dominant_pole.imag)
        if omega_n < 1e-6:
            omega_n = torch.abs(dominant_pole.real)

        import math
        tau_max = math.pi / (2.0 * omega_n)
        return tau_max

    def forward(
        self,
        n_generators: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate delay margins.

        Args:
            n_generators: Number of generators
            **kwargs: Additional arguments based on method

        Returns:
            tau_max: Delay margins [n_generators]
        """
        if self.method == "fixed":
            return torch.ones(n_generators) * self.fixed_value

        elif self.method == "crossover":
            omega_c = kwargs.get("omega_c")
            if omega_c is None:
                # Default: 1-10 rad/s typical for power systems
                omega_c = torch.ones(n_generators) * 5.0
            return self.estimate_from_crossover(omega_c)

        elif self.method == "eigenvalue":
            A = kwargs.get("A")
            B = kwargs.get("B")
            K = kwargs.get("K")
            tau_max_scalar = self.estimate_from_eigenvalue(A, B, K)
            return torch.ones(n_generators) * tau_max_scalar

        else:
            raise ValueError(f"Unknown method: {self.method}")
