"""B12: Neural-MPC delay-compensation baseline.

Major-revision response to APEN-D-26-05014 (R1.5). Neural-MPC trains a neural
network offline to imitate the receding-horizon MPC solution and deploys it as
a fast lookup at inference time. The advantage is sub-millisecond inference
latency; the trade-off is approximation error inherited from the imitator.

Implementation. The architecture is intentionally heavier than B10 (linear
MPC) so the comparison probes whether learning-based delay compensation
recovers the analytic horizon advantage. Coupling is learnable (in contrast
to B10/B11 where it is fixed analytically) so the network can tighten the
bound from data, while the input feature pipeline is the same flattened
(energy + comm) representation used by B10/B11. The K head is parameterized
in log-space (K = exp(log_K)) for strict positivity, mirroring the proposed
JointOptimizer.
"""

from typing import Dict

import torch
import torch.nn as nn


class NeuralMPC(nn.Module):
    """B12: Offline-trained neural-MPC controller.

    Args:
        n_buses: Number of buses.
        n_generators: Number of generators.
        energy_input_dim: Per-bus energy feature width.
        comm_input_dim: Per-bus comm feature width.
        hidden_dim: Width of the control-head MLP (intentionally larger than
            B10/B11 to support imitation of the analytic MPC policy).
        depth: Number of hidden layers in the control head.
        lambda_min_0: Optional baseline eigenvalue used to seed log_K.
        k_init_scale: Fallback K scale when lambda_min_0 is unavailable.
    """

    def __init__(
        self,
        n_buses: int,
        n_generators: int,
        energy_input_dim: int = 5,
        comm_input_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 3,
        lambda_min_0: float = None,
        k_init_scale: float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        feat_dim = (energy_input_dim + comm_input_dim) * n_buses
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, 2 * n_generators))
        self.control_head = nn.Sequential(*layers)

        if lambda_min_0 is None:
            k_value = float(k_init_scale)
        else:
            k_value = abs(float(lambda_min_0)) / max(n_generators, 1) * 0.9

        self.log_K = nn.Parameter(
            torch.full((n_generators,), float(torch.log(torch.tensor(max(k_value, 1e-6)))))
        )

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

        K = torch.exp(self.log_K)
        delay_contribution = (
            K.unsqueeze(0) * tau / tau_max.unsqueeze(0)
        ).sum(dim=-1)
        rho = torch.abs(lambda_min_0.squeeze()) - delay_contribution

        return {'u': u, 'rho': rho, 'K': K}

    def get_coupling_constants(self) -> torch.Tensor:
        return torch.exp(self.log_K)
