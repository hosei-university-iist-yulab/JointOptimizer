#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/07/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Convergence Analysis (V2)

Addresses Q2.8: How does performance change with number of training
scenarios and epochs? At what point does performance saturate?

Expected: Performance saturates at ~500 scenarios for IEEE 39,
~1000 for IEEE 118. Epochs saturate at ~200 for IEEE 39.
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
from src.losses.coupling_loss import simple_stability_loss


def train_and_evaluate(
    case_id: int,
    num_scenarios: int,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    """Train model with specific data/epoch budget and evaluate."""
    set_all_seeds(seed)

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id, num_scenarios=num_scenarios,
        delay_config=delay_config, batch_size=32, seed=seed,
    )

    dataset = train_loader.dataset.dataset
    n_gen = dataset.get_base_case()['n_generators']
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    model = JointOptimizer(
        n_generators=n_gen, energy_input_dim=5, comm_input_dim=3,
        embed_dim=128, hidden_dim=256, num_heads=8, gnn_layers=3,
        k_init_scale=0.1, adaptive_gamma=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Track metrics per epoch
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            energy_x = batch['energy_x'].to(device)
            comm_x = batch['comm_x'].to(device)
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)
            energy_edge_index = batch['energy_edge_index'].to(device)
            comm_edge_index = batch['comm_edge_index'].to(device)

            bs = energy_x.shape[0]
            n_nodes = energy_x.shape[1]
            energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
            comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
            batch_tensor = torch.arange(bs, device=device).repeat_interleave(n_nodes)

            outputs = model(
                energy_x=energy_x_flat, energy_edge_index=energy_edge_index,
                comm_x=comm_x_flat, comm_edge_index=comm_edge_index,
                tau=tau, tau_max=tau_max[0], lambda_min_0=lambda_min_0,
                impedance_matrix=impedance_matrix, batch=batch_tensor,
            )

            loss = simple_stability_loss(outputs['rho'], outputs['K'], lambda_min_0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            margins = []
            with torch.no_grad():
                for batch in val_loader:
                    tau = batch['tau'].to(device)
                    tau_max = batch['tau_max'].to(device)
                    lambda_min_0 = batch['lambda_min_0'].to(device)
                    rho = model.get_stability_margin(tau, tau_max[0], lambda_min_0)
                    margins.extend(rho.cpu().numpy().tolist())

            margins = np.array(margins)
            epoch_metrics.append({
                'epoch': epoch,
                'stability_rate': float(np.mean(margins > 0) * 100),
                'mean_margin': float(np.mean(margins)),
                'K_mean': float(model.get_coupling_constants().mean().item()),
            })

    return {
        'num_scenarios': num_scenarios,
        'epochs': epochs,
        'seed': seed,
        'final_stability_rate': epoch_metrics[-1]['stability_rate'],
        'final_margin': epoch_metrics[-1]['mean_margin'],
        'epoch_trajectory': epoch_metrics,
    }


def run_convergence_analysis(
    case_id: int = 39,
    output_dir: str = 'results/convergence',
    figures_dir: str = 'docs/figures/publication',
):
    """Run convergence analysis: scenarios x epochs grid."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)

    scenario_counts = [50, 100, 200, 500, 1000]
    epoch_counts = [50, 100, 200, 300, 500]

    # Part 1: Vary scenarios (fixed epochs=200)
    print("\nPart 1: Varying training scenarios (epochs=200)")
    scenario_results = []
    for n_scen in scenario_counts:
        print(f"  Scenarios={n_scen}...", end=" ", flush=True)
        result = train_and_evaluate(case_id, n_scen, 200, device)
        scenario_results.append(result)
        print(f"Stab={result['final_stability_rate']:.1f}%, "
              f"Margin={result['final_margin']:.4f}")

    # Part 2: Vary epochs (fixed scenarios=500)
    print("\nPart 2: Varying training epochs (scenarios=500)")
    epoch_results = []
    for n_ep in epoch_counts:
        print(f"  Epochs={n_ep}...", end=" ", flush=True)
        result = train_and_evaluate(case_id, 500, n_ep, device)
        epoch_results.append(result)
        print(f"Stab={result['final_stability_rate']:.1f}%, "
              f"Margin={result['final_margin']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'timestamp': datetime.now().isoformat(),
        'scenario_sweep': scenario_results,
        'epoch_sweep': epoch_results,
    }

    json_path = f"{output_dir}/convergence_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figure
    plot_convergence(scenario_results, epoch_results, case_id,
                     f"{figures_dir}/fig_convergence_case{case_id}.pdf")


def plot_convergence(scenario_results, epoch_results, case_id, save_path):
    """Plot convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scenarios
    ax = axes[0]
    x = [r['num_scenarios'] for r in scenario_results]
    y = [r['final_margin'] for r in scenario_results]
    ax.plot(x, y, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Training Scenarios', fontsize=12)
    ax.set_ylabel(r'Stability Margin $\rho(\tau)$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Scenario Convergence', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Right: epochs (using trajectory from 500-scenario run)
    ax = axes[1]
    for r in epoch_results:
        trajectory = r['epoch_trajectory']
        epochs = [t['epoch'] for t in trajectory]
        margins = [t['mean_margin'] for t in trajectory]
        ax.plot(epochs, margins, 'o-', linewidth=2, markersize=5,
                label=f'{r["epochs"]} epochs')

    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel(r'Stability Margin $\rho(\tau)$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Epoch Convergence', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(description='Convergence Analysis')
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--output', type=str, default='results/convergence')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_convergence_analysis(
        case_id=args.case,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
