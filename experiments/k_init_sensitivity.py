#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/09/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
K_i Initialization Sensitivity Experiment

Addresses Q1.3: K barely changes from initialization on IEEE 14 (0.1 -> 0.0989).

Tests whether different K initializations converge to the same final values.
If yes → K learning is meaningful (finds true physics constants).
If no → K is sensitive to initialization (not learning real physics).

Expected: On IEEE 39/57/118, K converges to ~0.08 regardless of init.
On IEEE 14, K barely moves (system too simple).
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

from src.models import JointOptimizer
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import set_all_seeds, restrict_gpus
from src.utils.k_diagnostics import KLearningTracker
from src.losses.coupling_loss import simple_stability_loss


K_INIT_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]


def train_with_k_init(
    case_id: int,
    k_init: float,
    seed: int,
    epochs: int = 200,
    device: torch.device = None,
) -> Dict:
    """
    Train model with specific K initialization and return final K values.

    Returns:
        Dict with K_final, K_trajectory, final_margin, etc.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_all_seeds(seed)

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=500,
        delay_config=delay_config,
        batch_size=32,
        seed=seed,
    )

    dataset = train_loader.dataset.dataset
    n_generators = dataset.get_base_case()['n_generators']
    n_buses = dataset.get_base_case()['n_buses']
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    model = JointOptimizer(
        n_generators=n_generators,
        energy_input_dim=5,
        comm_input_dim=3,
        embed_dim=128,
        hidden_dim=256,
        num_heads=8,
        gnn_layers=3,
        k_init_scale=k_init,
        learnable_k=True,
        physics_gamma=0.1,
        adaptive_gamma=True,
    ).to(device)

    tracker = KLearningTracker(n_generators)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Log initial K
    K_init = model.get_coupling_constants().detach().cpu()
    tracker.log(0, K_init)

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            energy_x = batch['energy_x'].to(device)
            comm_x = batch['comm_x'].to(device)
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)
            energy_edge_index = batch['energy_edge_index'].to(device)
            comm_edge_index = batch['comm_edge_index'].to(device)

            batch_size = energy_x.shape[0]
            n_nodes = energy_x.shape[1]
            energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
            comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
            batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
            tau_max_tensor = tau_max[0] if tau_max.dim() > 1 else tau_max

            outputs = model(
                energy_x=energy_x_flat,
                energy_edge_index=energy_edge_index,
                comm_x=comm_x_flat,
                comm_edge_index=comm_edge_index,
                tau=tau,
                tau_max=tau_max_tensor,
                lambda_min_0=lambda_min_0,
                impedance_matrix=impedance_matrix,
                batch=batch_tensor,
            )

            loss = simple_stability_loss(outputs['rho'], outputs['K'], lambda_min_0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Log K periodically
        if epoch % 5 == 0 or epoch == epochs:
            K_current = model.get_coupling_constants().detach().cpu()
            tracker.log(epoch, K_current)

    # Final evaluation
    model.eval()
    K_final = model.get_coupling_constants().detach().cpu().numpy()

    margins = []
    with torch.no_grad():
        for batch in val_loader:
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)
            tau_max_tensor = tau_max[0] if tau_max.dim() > 1 else tau_max

            rho = model.get_stability_margin(tau, tau_max_tensor, lambda_min_0)
            margins.extend(rho.cpu().numpy().tolist())

    return {
        'k_init': k_init,
        'seed': seed,
        'K_final_mean': float(np.mean(K_final)),
        'K_final_std': float(np.std(K_final)),
        'K_final_per_gen': K_final.tolist(),
        'K_change_pct': float(abs(np.mean(K_final) - k_init) / k_init * 100),
        'mean_margin': float(np.mean(margins)),
        'stability_rate': float(np.mean(np.array(margins) > 0) * 100),
        'tracker_data': tracker.to_dict(),
    }


def run_k_init_sensitivity(
    case_id: int = 39,
    epochs: int = 200,
    seeds: int = 5,
    output_dir: str = 'results/k_init_sensitivity',
    figures_dir: str = 'docs/figures/publication',
):
    """Run full K initialization sensitivity experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("K INITIALIZATION SENSITIVITY EXPERIMENT")
    print("=" * 70)
    print(f"Case: IEEE {case_id}, Epochs: {epochs}, Seeds: {seeds}")
    print(f"K_init values: {K_INIT_VALUES}")

    all_results = {}

    for k_init in K_INIT_VALUES:
        print(f"\n--- K_init = {k_init} ---")
        seed_results = []

        for seed in range(seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = train_with_k_init(
                case_id=case_id,
                k_init=k_init,
                seed=seed * 42,
                epochs=epochs,
                device=device,
            )
            seed_results.append(result)
            print(f"K_final={result['K_final_mean']:.4f}, "
                  f"change={result['K_change_pct']:.1f}%")

        all_results[str(k_init)] = seed_results

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Remove non-serializable tracker data for JSON
    save_results = {}
    for k_init_str, seed_results in all_results.items():
        save_results[k_init_str] = [
            {k: v for k, v in r.items() if k != 'tracker_data'}
            for r in seed_results
        ]

    output_dict = {
        'case_id': case_id,
        'k_init_values': K_INIT_VALUES,
        'epochs': epochs,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'results': save_results,
    }

    json_path = f"{output_dir}/k_init_sensitivity_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figure
    plot_k_init_sensitivity(all_results, case_id,
                            f"{figures_dir}/fig_k_init_sensitivity_case{case_id}.pdf")

    # Convergence analysis
    analyze_convergence(all_results, case_id)


def plot_k_init_sensitivity(
    all_results: Dict,
    case_id: int,
    save_path: str,
):
    """Plot K convergence from different initializations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Final K vs Init K
    ax = axes[0]
    for k_init_str, seed_results in all_results.items():
        k_init = float(k_init_str)
        k_finals = [r['K_final_mean'] for r in seed_results]
        ax.scatter([k_init] * len(k_finals), k_finals,
                   alpha=0.6, s=50, zorder=2)
        ax.errorbar(k_init, np.mean(k_finals), yerr=np.std(k_finals),
                     fmt='D', color='red', markersize=8, capsize=5, zorder=3)

    ax.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.3, label='No learning (K=K_init)')
    ax.set_xlabel(r'$K_{init}$', fontsize=12)
    ax.set_ylabel(r'$K_{final}$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: K Convergence from Different Inits')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: K trajectories during training
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    for (k_init_str, seed_results), color in zip(all_results.items(), colors):
        # Use first seed's tracker data
        tracker_data = seed_results[0].get('tracker_data')
        if tracker_data:
            epochs = tracker_data['epoch_history']
            k_means = [
                np.mean([tracker_data['k_history'][f'K_{i}'][j]
                         for i in range(tracker_data['n_generators'])])
                for j in range(len(epochs))
            ]
            ax.plot(epochs, k_means, color=color,
                    label=f'init={k_init_str}', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(r'Mean $K_i$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: K Learning Trajectories')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def analyze_convergence(all_results: Dict, case_id: int):
    """Analyze whether K values converge regardless of initialization."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)

    # Collect all final K means across all inits and seeds
    all_k_finals = []
    for k_init_str, seed_results in all_results.items():
        for r in seed_results:
            all_k_finals.append(r['K_final_mean'])

    # Per-init variance
    intra_init_vars = []
    inter_init_means = []
    for k_init_str, seed_results in all_results.items():
        k_finals = [r['K_final_mean'] for r in seed_results]
        intra_init_vars.append(np.var(k_finals))
        inter_init_means.append(np.mean(k_finals))

    inter_init_var = np.var(inter_init_means)
    mean_intra_var = np.mean(intra_init_vars)

    print(f"\nInter-init variance: {inter_init_var:.6f}")
    print(f"Mean intra-init variance: {mean_intra_var:.6f}")

    if inter_init_var < mean_intra_var * 2:
        print(f"\n>>> CONVERGENT: K converges to similar values regardless of init")
        print(f"    This proves K learning is meaningful on IEEE {case_id}.")
    else:
        print(f"\n>>> SENSITIVE: K depends on initialization")
        print(f"    K learning may not be finding true physics on IEEE {case_id}.")

    print(f"\nFinal K values by init:")
    for k_init_str, seed_results in all_results.items():
        k_finals = [r['K_final_mean'] for r in seed_results]
        print(f"  init={k_init_str}: K_final = {np.mean(k_finals):.4f} "
              f"± {np.std(k_finals):.4f}")


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='K Initialization Sensitivity Experiment'
    )
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/k_init_sensitivity')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_k_init_sensitivity(
        case_id=args.case,
        epochs=args.epochs,
        seeds=args.seeds,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
