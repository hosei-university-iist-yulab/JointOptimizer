#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/28/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Coupling Loss: L_coupling

ORIGINAL CONTRIBUTION - Derived from Theorem 1 (Delay-Stability Coupling)

L_coupling(u, τ) = -α · log(ρ(τ)) + β · Σ_i ||∇u_i||² · τ_i²

where:
- ρ(τ) = |λ_min(0)| - Σ_i (K_i · τ_i / τ_max,i)  is the stability margin
- First term: log-barrier for stability constraint
- Second term: control deviation penalty

This loss is DERIVED from control theory, not assumed:
1. Swing equation → DDE under delay
2. Padé approximation → eigenvalue perturbation
3. Theorem 1 → stability margin ρ(τ)
4. Log-barrier → smooth optimization with hard constraint
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LogBarrierStabilityLoss(nn.Module):
    """
    Log-barrier loss for stability constraint.

    L_stability(τ) = -log(ρ(τ) / |λ_min(0)|)

    Properties:
    - L → 0 as ρ → |λ_min(0)| (maximum stability)
    - L → +∞ as ρ → 0 (approaching instability)
    - Smooth gradient for optimization
    """

    def __init__(
        self,
        rho_min: float = 0.01,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            rho_min: Minimum stability margin (to avoid log(0))
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.rho_min = rho_min
        self.epsilon = epsilon

    def forward(
        self,
        rho: torch.Tensor,
        lambda_min_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-barrier loss.

        Args:
            rho: Stability margin [batch]
            lambda_min_0: Reference eigenvalue (denominator)

        Returns:
            loss: Log-barrier loss [batch]
        """
        # Clamp rho to avoid log(0) or log(negative)
        rho_clamped = torch.clamp(rho, min=self.rho_min)

        # Normalize by reference
        rho_normalized = rho_clamped / (torch.abs(lambda_min_0) + self.epsilon)

        # Log-barrier: -log(x) is convex for x > 0
        loss = -torch.log(rho_normalized + self.epsilon)

        return loss


class ControlDeviationLoss(nn.Module):
    """
    Control deviation penalty for delayed control.

    ||u_i(t) - u_i^*(t - τ_i)||² ≈ ||∇u_i||² · τ_i²

    For slowly-varying optimal control, the deviation grows
    quadratically with delay.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        u: torch.Tensor,
        u_prev: torch.Tensor,
        tau: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute control deviation loss.

        Args:
            u: Current control [batch, n_control]
            u_prev: Previous control [batch, n_control]
            tau: Communication delays [batch, n_generators] in seconds
            dt: Time step (for gradient approximation)

        Returns:
            loss: Control deviation loss [batch]
        """
        # Approximate gradient: ∇u ≈ (u - u_prev) / dt
        grad_u = (u - u_prev) / dt  # [batch, n_control]

        # Squared gradient magnitude per control
        grad_u_sq = (grad_u ** 2).sum(dim=-1)  # [batch]

        # Weight by squared delay
        # If n_control != n_generators, need to align
        tau_sq = (tau ** 2).mean(dim=-1)  # [batch]

        loss = grad_u_sq * tau_sq

        return loss


class CouplingLoss(nn.Module):
    """
    Combined Coupling Loss: L_coupling

    L_coupling(u, τ) = α · L_stability + β · L_deviation

    where:
    - L_stability = -log(ρ(τ) / |λ_min(0)|)
    - L_deviation = Σ_i ||∇u_i||² · τ_i²

    This is the CORE NOVEL CONTRIBUTION of the paper.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        rho_min: float = 0.01,
    ):
        """
        Args:
            alpha: Weight for log-barrier stability loss
            beta: Weight for control deviation loss
            rho_min: Minimum stability margin
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.stability_loss = LogBarrierStabilityLoss(rho_min=rho_min)
        self.deviation_loss = ControlDeviationLoss()

    def forward(
        self,
        rho: torch.Tensor,
        lambda_min_0: torch.Tensor,
        u: torch.Tensor,
        u_prev: torch.Tensor,
        tau: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined coupling loss.

        Args:
            rho: Stability margin [batch]
            lambda_min_0: Reference eigenvalue
            u: Current control [batch, n_control]
            u_prev: Previous control [batch, n_control]
            tau: Communication delays [batch, n_generators]
            dt: Time step

        Returns:
            loss: Total coupling loss (scalar)
            components: Dict with individual loss terms
        """
        # Log-barrier stability loss
        L_stability = self.stability_loss(rho, lambda_min_0)

        # Control deviation loss
        L_deviation = self.deviation_loss(u, u_prev, tau, dt)

        # Combined loss
        loss = self.alpha * L_stability.mean() + self.beta * L_deviation.mean()

        components = {
            "L_stability": L_stability.mean().item(),
            "L_deviation": L_deviation.mean().item(),
            "rho_mean": rho.mean().item(),
            "rho_min": rho.min().item(),
        }

        return loss, components


class CouplingLossWithLearning(nn.Module):
    """
    Coupling loss with learnable K_i parameters.

    Integrates:
    1. Learnable coupling constants K_i
    2. Stability margin computation
    3. Log-barrier + deviation loss
    """

    def __init__(
        self,
        n_generators: int,
        tau_max: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.1,
        rho_min: float = 0.01,
        k_init_scale: float = 0.1,
    ):
        """
        Args:
            n_generators: Number of generators
            tau_max: Delay margins [n_generators]
            alpha, beta: Loss weights
            rho_min: Minimum stability margin
            k_init_scale: Initial scale for K_i
        """
        super().__init__()

        from ..models.coupling import StabilityMarginComputer

        self.stability_computer = StabilityMarginComputer(
            n_generators=n_generators,
            tau_max=tau_max,
            init_scale=k_init_scale,
        )

        self.coupling_loss = CouplingLoss(
            alpha=alpha,
            beta=beta,
            rho_min=rho_min,
        )

    def forward(
        self,
        tau: torch.Tensor,
        lambda_min_0: torch.Tensor,
        u: torch.Tensor,
        u_prev: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute coupling loss with learned K_i.

        Args:
            tau: Communication delays [batch, n_generators]
            lambda_min_0: Minimum eigenvalue at zero delay
            u: Current control [batch, n_control]
            u_prev: Previous control [batch, n_control]
            dt: Time step

        Returns:
            loss: Total coupling loss
            components: Dict with loss components and K values
        """
        # Compute stability margin using learned K_i
        rho, K = self.stability_computer(tau, lambda_min_0)

        # Compute coupling loss
        loss, components = self.coupling_loss(
            rho, lambda_min_0, u, u_prev, tau, dt
        )

        # Add K values to components
        components["K_mean"] = K.mean().item()
        components["K_max"] = K.max().item()
        components["K_min"] = K.min().item()

        return loss, components


def simple_stability_loss(
    rho: torch.Tensor,
    K: torch.Tensor,
    lambda_min_0: torch.Tensor,
    rho_min: float = 0.01,
) -> torch.Tensor:
    """
    Simplified but correct stability loss for experiment scripts.

    Uses log-barrier when rho > rho_min, quadratic penalty when rho < rho_min.
    K regularization uses .mean() (not .sum()) so it scales with n_gen.
    """
    # Log-barrier for positive margins
    rho_safe = torch.clamp(rho, min=rho_min)
    log_barrier = -torch.log(rho_safe / (torch.abs(lambda_min_0) + 1e-6) + 1e-6).mean()

    # Quadratic penalty for negative margins (provides gradient when rho << 0)
    neg_penalty = 10.0 * torch.clamp(-rho + rho_min, min=0).pow(2).mean()

    # K regularization — mean not sum
    k_reg = 0.01 * (K ** 2).mean()

    return log_barrier + neg_penalty + k_reg
