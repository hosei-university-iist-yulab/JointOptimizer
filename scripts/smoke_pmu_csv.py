#!/usr/bin/env python3
"""Smoke test the PMU FREQUENCY CSV archive end-to-end against the trained
JointOptimizer. No training, just a forward pass per attack scenario, and
a print of margin / K / dispatch shapes so we can confirm the framework
ingests the archive without crashing.

Run before any full evaluation rerun:
    CUDA_VISIBLE_DEVICES=6 conda run -n llms python scripts/smoke_pmu_csv.py
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.data.pmu_csv_loader import load_pmu_csv, CSV_FILES
from src.models import JointOptimizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def smoke_one_case(case_name: str) -> None:
    print(f"\n=== {case_name} ===")
    data = load_pmu_csv(case_name=case_name, max_timesteps=256, batch_size=8)
    n_pmu = data['n_buses']
    print(f"  n_PMU={n_pmu}, T={data['n_timesteps']}, "
          f"energy_x={tuple(data['energy_x'].shape)}, comm_x={tuple(data['comm_x'].shape)}")

    model = JointOptimizer(
        n_generators=n_pmu,
        energy_input_dim=5,
        comm_input_dim=3,
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        gnn_layers=1,
        lambda_min_0=data['lambda_min'],
        mpc_horizon_ms=30.0,
        use_adaptive_k_init=True,
        use_physics_mask=False,
        use_causal_mask=False,
        use_cross_attention=False,
    ).to(DEVICE)
    model.eval()

    energy_x = data['energy_x'].to(DEVICE)
    comm_x = data['comm_x'].to(DEVICE)
    tau = data['tau'].to(DEVICE)
    tau_max = data['tau_max'].to(DEVICE)
    lambda_min_0 = data['lambda_min_0'].to(DEVICE)
    edge_index = data['edge_index'].to(DEVICE)

    bs, n = energy_x.shape[0], energy_x.shape[1]
    e_flat = energy_x.reshape(-1, energy_x.shape[-1])
    c_flat = comm_x.reshape(-1, comm_x.shape[-1])
    bt = torch.arange(bs, device=DEVICE).repeat_interleave(n)

    with torch.no_grad():
        out = model(
            energy_x=e_flat, energy_edge_index=edge_index,
            comm_x=c_flat, comm_edge_index=edge_index,
            tau=tau, tau_max=tau_max, lambda_min_0=lambda_min_0,
            impedance_matrix=None, batch=bt,
        )

    rho = out['rho']
    K = out['K']
    u = out['u']
    print(f"  rho        = {rho.mean().item():.4f} +/- {rho.std().item():.4f}")
    print(f"  K_mean     = {K.mean().item():.6f}, K_min = {K.min().item():.6f}")
    print(f"  u.shape    = {tuple(u.shape)} (dispatch)")
    print(f"  tau_mult   = {data['tau_multiplier']}, batches = {bs}")


if __name__ == '__main__':
    for case in CSV_FILES:
        try:
            smoke_one_case(case)
        except Exception as e:
            print(f"  ERROR on {case}: {type(e).__name__}: {e}")
            raise
    print("\nSmoke OK -- the JointOptimizer ingests every PMU FREQUENCY case "
          "without crashing. Ready for full evaluation.")
