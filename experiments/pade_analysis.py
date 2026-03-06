#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/10/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Padé Approximation Analysis (V2)

Addresses Q1.5: Compare 1st-order vs 2nd-order Padé approximation accuracy
against exact eigenvalue computation across the delay range.

Expected:
- Padé-1 error ~4% at 200ms, growing at higher delays
- Padé-2 error <1% at 200ms, <3% at 500ms
- Both are conservative upper bounds (rho_pade <= rho_exact)
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.coupling import (
    StabilityMarginComputer,
    StabilityMarginComputerV2,
)
from src.data import create_dataloaders, DelayConfig
from src.data import load_case
from src.utils.time_domain_simulation import (
    DelayedSwingEquationSimulator,
    compute_empirical_margin_independent,
    build_delay_coupling_matrix,
)
from src.utils.statistical_tests import restrict_gpus


DELAY_RANGE_MS = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]


def compute_exact_margin(
    A_sys: np.ndarray,
    B_delay: np.ndarray,
    tau_ms: float,
    trials: int = 3,
) -> Dict:
    """Compute margin from time-domain simulation (ground truth)."""
    tau_s = tau_ms / 1000.0
    rho_mean, rho_std = compute_empirical_margin_independent(
        A_sys, B_delay, tau_vec=np.array([tau_s]),
        dt=0.0005, T=5.0, n_trials=trials,
    )
    return {'rho': rho_mean, 'std': rho_std}


def compute_pade_margins(
    n_generators: int,
    tau_ms: float,
    lambda_min_0: float,
    K_values: torch.Tensor,
    tau_max: torch.Tensor,
    K2_values: torch.Tensor = None,
) -> Dict:
    """Compute margins from Padé-1 and Padé-2 formulas."""
    tau_s = tau_ms / 1000.0
    tau = torch.tensor([[tau_s] * n_generators])
    lambda_min = torch.tensor([lambda_min_0])

    # Padé-1
    computer_v1 = StabilityMarginComputer(
        n_generators=n_generators,
        tau_max=tau_max,
        init_scale=0.1,
    )
    # Set K values
    with torch.no_grad():
        computer_v1.coupling.log_K.copy_(torch.log(K_values))
    rho_v1, _ = computer_v1(tau, lambda_min)

    # Padé-2
    computer_v2 = StabilityMarginComputerV2(
        n_generators=n_generators,
        tau_max=tau_max,
        init_scale=0.1,
        order=2,
    )
    with torch.no_grad():
        computer_v2.coupling.log_K.copy_(torch.log(K_values))
        if K2_values is not None:
            computer_v2.log_K2.copy_(torch.log(K2_values))

    rho_v2, _ = computer_v2(tau, lambda_min)

    return {
        'pade1': rho_v1.item(),
        'pade2': rho_v2.item(),
    }


def run_pade_analysis(
    case_id: int = 39,
    output_dir: str = 'results/pade_analysis',
    figures_dir: str = 'docs/figures/publication',
):
    """Compare Padé-1, Padé-2, and exact margins across delay range."""
    print("=" * 70)
    print("PADÉ APPROXIMATION ANALYSIS")
    print("=" * 70)
    print(f"Case: IEEE {case_id}")
    print(f"Delay range: {DELAY_RANGE_MS} ms")

    # Load case
    loader = load_case(case_id)
    case = loader.load()
    n_gen = case['n_generators']

    # Get system matrices for simulation
    A_sys, B_input = loader.get_system_matrices()
    eigenvalues, lambda_min = loader.get_eigenvalues()
    lambda_min_0 = float(abs(lambda_min))

    B_delay = build_delay_coupling_matrix(A_sys.numpy(), int(n_gen))

    # Use realistic K values (from typical trained model)
    K_values = torch.ones(n_gen) * 0.08
    K2_values = torch.ones(n_gen) * 0.005
    tau_max = torch.ones(n_gen) * 0.5  # 500ms

    print(f"Lambda_min(0) = {lambda_min_0:.4f}")
    print(f"K = {K_values[0].item():.4f}, K2 = {K2_values[0].item():.4f}")

    results = []

    for tau_ms in DELAY_RANGE_MS:
        print(f"\n  tau = {tau_ms} ms:", end=" ", flush=True)

        # Exact (simulation)
        exact = compute_exact_margin(A_sys.numpy(), B_delay, tau_ms, trials=3)

        # Padé approximations
        pade = compute_pade_margins(
            n_gen, tau_ms, lambda_min_0, K_values, tau_max, K2_values
        )

        # Relative errors
        rho_exact = exact['rho']
        err_pade1 = abs(pade['pade1'] - rho_exact) / abs(rho_exact) * 100 if abs(rho_exact) > 1e-6 else 0
        err_pade2 = abs(pade['pade2'] - rho_exact) / abs(rho_exact) * 100 if abs(rho_exact) > 1e-6 else 0

        result = {
            'tau_ms': tau_ms,
            'rho_exact': rho_exact,
            'rho_exact_std': exact['std'],
            'rho_pade1': pade['pade1'],
            'rho_pade2': pade['pade2'],
            'error_pade1_pct': err_pade1,
            'error_pade2_pct': err_pade2,
        }
        results.append(result)

        print(f"exact={rho_exact:.4f}, "
              f"P1={pade['pade1']:.4f} ({err_pade1:.1f}%), "
              f"P2={pade['pade2']:.4f} ({err_pade2:.1f}%)")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'lambda_min_0': lambda_min_0,
        'K_values': K_values.tolist(),
        'K2_values': K2_values.tolist(),
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }

    json_path = f"{output_dir}/pade_analysis_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figure
    plot_pade_comparison(results, case_id,
                         f"{figures_dir}/fig_pade_analysis_case{case_id}.pdf")

    # Summary
    print("\n" + "=" * 70)
    print("PADÉ ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n{'tau (ms)':<10} {'Exact':<10} {'Padé-1':<10} {'Err-1 (%)':<10} {'Padé-2':<10} {'Err-2 (%)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['tau_ms']:<10} {r['rho_exact']:<10.4f} {r['rho_pade1']:<10.4f} "
              f"{r['error_pade1_pct']:<10.1f} {r['rho_pade2']:<10.4f} {r['error_pade2_pct']:<10.1f}")


def plot_pade_comparison(
    results: List[Dict], case_id: int, save_path: str,
):
    """Plot Padé-1 vs Padé-2 vs exact margins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    taus = [r['tau_ms'] for r in results]
    exact = [r['rho_exact'] for r in results]
    pade1 = [r['rho_pade1'] for r in results]
    pade2 = [r['rho_pade2'] for r in results]
    err1 = [r['error_pade1_pct'] for r in results]
    err2 = [r['error_pade2_pct'] for r in results]

    # Left: Stability margin comparison
    ax = axes[0]
    ax.plot(taus, exact, 'ko-', linewidth=2, markersize=8, label='Exact (simulation)')
    ax.plot(taus, pade1, 's--', color='#F18F01', linewidth=2, markersize=7,
            label=r'Padé-1 ($\mathcal{O}(\tau)$)')
    ax.plot(taus, pade2, 'D--', color='#2E86AB', linewidth=2, markersize=7,
            label=r'Padé-2 ($\mathcal{O}(\tau^2)$)')
    ax.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Instability boundary')
    ax.set_xlabel(r'Communication Delay $\tau$ (ms)', fontsize=12)
    ax.set_ylabel(r'Stability Margin $\rho(\tau)$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Padé Approximation Accuracy', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Relative error
    ax = axes[1]
    ax.plot(taus, err1, 's-', color='#F18F01', linewidth=2, markersize=7,
            label=r'Padé-1 error')
    ax.plot(taus, err2, 'D-', color='#2E86AB', linewidth=2, markersize=7,
            label=r'Padé-2 error')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1% threshold')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
    ax.set_xlabel(r'Communication Delay $\tau$ (ms)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Approximation Error', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Padé Approximation Analysis'
    )
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--output', type=str, default='results/pade_analysis')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_pade_analysis(
        case_id=args.case,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
