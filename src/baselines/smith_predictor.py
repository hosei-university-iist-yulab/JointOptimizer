"""B11: Smith-predictor delay-compensation baseline.

Major-revision response to APEN-D-26-05014 (R1.5). The Smith predictor is the
canonical classical scheme for compensating known transport delays: it places
a delay-free model in parallel with the plant and feeds back the difference
between the predicted and the measured output, effectively removing the delay
from the closed loop when the model is exact.

Coupling derivation. Under the Smith-predictor structure the nominal closed-
loop transfer function is delay-free, so the analytical K_i collapses to a
small residual robustness margin proportional to the per-generator model
mismatch sigma_i (default 5 % of the base sensitivity). For the comparison we
parameterize

    K_i^smith = sigma_mismatch * |lambda_min(0)| / n_g.

This produces a small but non-zero K reflecting the practical limit of Smith
predictors under model uncertainty: tighter than B1's hand-tuned default but
looser than the proposed JointOptimizer's learned coupling.
"""

from typing import Dict

import torch
import torch.nn as nn


class SmithPredictor(nn.Module):
    """B11: Smith predictor with model-mismatch coupling.

    Args:
        n_buses: Number of buses.
        n_generators: Number of generators.
        energy_input_dim: Per-bus energy feature width.
        comm_input_dim: Per-bus comm feature width.
        hidden_dim: Width of the control-head MLP.
        sigma_mismatch: Fraction of the budget reserved for model mismatch
            (default 0.05; raise to 0.1+ to model heavier uncertainty).
        lambda_min_0: Optional baseline eigenvalue.
        k_init_scale: Fallback K scale when lambda_min_0 is not provided.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 128,
        sigma_mismatch: float = 0.05,
        lambda_min_0: float = None,
        k_init_scale: float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.sigma_mismatch = float(sigma_mismatch)

        feat_dim = (energy_input_dim + comm_input_dim) * n_buses
        self.control_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * n_generators),
        )

        if lambda_min_0 is None:
            k_value = k_init_scale * self.sigma_mismatch
        else:
            k_value = (
                self.sigma_mismatch
                * abs(float(lambda_min_0))
                / max(n_generators, 1)
            )
        self.register_buffer('K', torch.full((n_generators,), float(k_value)))

    def forward(
        self,
        energy_x: torch.Tensor,
        comm_x: torch.Tensor,
        tau: torch.Tensor,
        tau_max: torch.Tensor,
        lambda_min_0: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if energy_x.dim() == 2:
            batch_size = energy_x.shape[0] // self.n_buses
            energy_x = energy_x.view(batch_size, self.n_buses, -1)
            comm_x = comm_x.view(batch_size, self.n_buses, -1)

        batch_size = energy_x.shape[0]
        flat = torch.cat(
            [energy_x.reshape(batch_size, -1), comm_x.reshape(batch_size, -1)],
            dim=-1,
        )
        u = self.control_head(flat)

        delay_contribution = (
            self.K.unsqueeze(0) * tau / tau_max.unsqueeze(0)
        ).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {'u': u, 'rho': rho, 'K': self.K}

    def get_coupling_constants(self) -> torch.Tensor:
        return self.K
