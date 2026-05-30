#!/usr/bin/env python3
"""Evaluate the trained Strong checkpoint on the IEEE DataPort PMU FREQUENCY
CASE I-IV CSV archive.

Loads each CSV via :func:`src.data.pmu_csv_loader.load_pmu_csv`, runs a
forward pass through ``JointOptimizer`` configured with Lever 3 + lite
architecture + adaptive K_init (matching the canonical Strong variant),
and writes per-case headline metrics ($\bar\rho$, $\tau_{\mathrm{crit}}$,
$\partial\rho/\partial\tau$, K_mean) to JSON for the table generator.

This is a publication-grade evaluation: the model is *not* retrained on
the PMU archive; the loaded checkpoint represents the canonical Strong
training output. The CSV-level inputs are zero-padded for the missing
P, Q, V, theta, R, B channels (the archive only ships per-PMU frequency).
The four scenarios use ascending tau-multipliers (1.0, 1.5, 2.0, 3.0) to
encode the cyber-attack severity (MITM -> Resembling -> Repetition ->
Missing) on the otherwise-identical PMU traces.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.pmu_csv_loader import load_pmu_csv, CSV_FILES
from src.models import JointOptimizer
from src.utils.statistical_tests import set_all_seeds, restrict_gpus

DEFAULT_OUTPUT = REPO_ROOT / "results/full_sweep/real_pmu"


def _estimate_tau_crit_ms(K: torch.Tensor, lambda_min: float,
                          tau_max_seconds: float = 0.5) -> float:
    """Closed-form tau_crit for uniform-K case: rho(tau) = |lambda_min| -
    sum(K_i) * tau / tau_max -> 0 at tau = |lambda_min| * tau_max / sum(K_i).
    """
    K_sum = float(K.detach().sum().item())
    if K_sum <= 1e-12:
        return float("inf")
    return abs(float(lambda_min)) * tau_max_seconds * 1000.0 / K_sum


def _estimate_margin_slope(K: torch.Tensor, tau_max_seconds: float = 0.5) -> float:
    """rho is linear in tau, so dRho/dTau = -sum(K_i) / tau_max."""
    K_sum = float(K.detach().sum().item())
    return -K_sum / tau_max_seconds


def evaluate_case(case_name: str, device: torch.device,
                  default_lambda_min: float = 0.4) -> dict:
    data = load_pmu_csv(case_name=case_name)
    n_pmu = data['n_buses']
    energy_x = data['energy_x'].to(device)
    comm_x = data['comm_x'].to(device)
    tau = data['tau'].to(device)
    tau_max = data['tau_max'].to(device)
    lambda_min_0 = data['lambda_min_0'].to(device)
    edge_index = data['edge_index'].to(device)

    # Strong-equivalent JointOptimizer: lite architecture + adaptive K_init
    # + horizon-aware K (mpc_horizon_ms=30).
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
    ).to(device)
    model.eval()

    bs, n = energy_x.shape[0], energy_x.shape[1]
    e_flat = energy_x.reshape(-1, energy_x.shape[-1])
    c_flat = comm_x.reshape(-1, comm_x.shape[-1])
    bt = torch.arange(bs, device=device).repeat_interleave(n)

    with torch.no_grad():
        out = model(
            energy_x=e_flat, energy_edge_index=edge_index,
            comm_x=c_flat, comm_edge_index=edge_index,
            tau=tau, tau_max=tau_max, lambda_min_0=lambda_min_0,
            impedance_matrix=None, batch=bt,
        )

    rho = out['rho']
    K = out['K']
    tau_crit_ms = _estimate_tau_crit_ms(K, lambda_min=data['lambda_min'])
    slope = _estimate_margin_slope(K)
    return {
        "case": case_name,
        "n_pmu": int(n_pmu),
        "n_timesteps": int(data['n_timesteps']),
        "tau_multiplier": float(data['tau_multiplier']),
        "rho_mean": float(rho.mean().item()),
        "rho_std": float(rho.std().item()),
        "K_mean": float(K.mean().item()),
        "K_min": float(K.min().item()),
        "tau_crit_ms": float(tau_crit_ms),
        "dRho_dTau": float(slope),
        "stability_rate": 100.0 * float((rho > 0).float().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Where to write real_pmu_validation.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    restrict_gpus()
    set_all_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = []
    for case in CSV_FILES:
        print(f"\n=== {case} ===")
        r = evaluate_case(case, device)
        for k, v in r.items():
            print(f"  {k}: {v}")
        results.append(r)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_pmu_validation.json"
    with out_path.open("w") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "results": results,
        }, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
