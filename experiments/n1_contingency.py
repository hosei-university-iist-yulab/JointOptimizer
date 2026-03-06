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
N-1 Contingency Robustness Experiment (V2)

Addresses Q3.7: How robust is the trained model when transmission
lines are removed at test time (without retraining)?

For each line in the system, remove it and evaluate stability.
Report per-line and aggregate robustness metrics.

Expected: JointOptimizer maintains stability under most N-1 scenarios,
with GNN encoding providing topological awareness.
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
from src.data.stressed_scenarios import StressedScenarioGenerator, StressConfig
from src.utils.statistical_tests import set_all_seeds, restrict_gpus
from src.losses.coupling_loss import simple_stability_loss


def train_model(
    case_id: int,
    epochs: int,
    num_scenarios: int,
    device: torch.device,
    seed: int = 42,
) -> tuple:
    """Train model on normal conditions."""
    set_all_seeds(seed)

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id, num_scenarios=num_scenarios,
        delay_config=delay_config, batch_size=32, seed=seed,
    )

    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_gen = base_case['n_generators']
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    model = JointOptimizer(
        n_generators=n_gen, energy_input_dim=5, comm_input_dim=3,
        embed_dim=128, hidden_dim=256, num_heads=8, gnn_layers=3,
        k_init_scale=0.1, adaptive_gamma=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model.train()
    for epoch in range(epochs):
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

    return model, base_case


def evaluate_n1(
    model: torch.nn.Module,
    base_case: Dict,
    device: torch.device,
    n_eval: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """Evaluate model under each possible N-1 contingency."""
    model.eval()
    n_gen = base_case['n_generators']
    n_lines = base_case['edge_index'].shape[1]

    generator = StressedScenarioGenerator(base_case)
    results = []

    # Normal (no contingency)
    normal_config = StressConfig(name='normal')
    _, tau_normal = generator.apply_stress(normal_config, seed=seed)
    tau = tau_normal.unsqueeze(0).expand(n_eval, -1).to(device)
    tau_max = torch.ones(n_gen, device=device) * 0.5
    lambda_min_0 = torch.tensor([base_case['lambda_min']], device=device).expand(n_eval)

    with torch.no_grad():
        rho_normal = model.get_stability_margin(tau, tau_max, lambda_min_0)
    margins_normal = rho_normal.cpu().numpy()

    results.append({
        'contingency': 'none',
        'line_removed': -1,
        'stability_rate': float(np.mean(margins_normal > 0) * 100),
        'mean_margin': float(np.mean(margins_normal)),
    })

    # N-1: Remove each line
    for line_idx in range(min(n_lines, 50)):  # Cap at 50 lines for large cases
        config = StressConfig(
            name=f'n1_line{line_idx}',
            remove_line_idx=line_idx,
        )
        try:
            stressed_case, tau_stressed = generator.apply_stress(config, seed=seed)
            tau = tau_stressed.unsqueeze(0).expand(n_eval, -1).to(device)

            with torch.no_grad():
                rho = model.get_stability_margin(tau, tau_max, lambda_min_0)
            margins = rho.cpu().numpy()

            results.append({
                'contingency': f'N-1 (line {line_idx})',
                'line_removed': line_idx,
                'stability_rate': float(np.mean(margins > 0) * 100),
                'mean_margin': float(np.mean(margins)),
            })
        except Exception:
            continue

    return results


def run_n1_experiment(
    case_id: int = 39,
    epochs: int = 100,
    num_scenarios: int = 500,
    seed: int = 42,
    output_dir: str = 'results/n1_contingency',
    figures_dir: str = 'docs/figures/publication',
):
    """Run N-1 contingency experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("N-1 CONTINGENCY ROBUSTNESS")
    print("=" * 70)
    print(f"Case: IEEE {case_id}")

    # Train on normal conditions
    print("\nTraining model on normal conditions...")
    model, base_case = train_model(case_id, epochs, num_scenarios, device, seed)

    # Evaluate N-1
    print("Evaluating N-1 contingencies...")
    results = evaluate_n1(model, base_case, device, seed=seed)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'n_lines': base_case['edge_index'].shape[1],
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }

    json_path = f"{output_dir}/n1_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Plot
    plot_n1_results(results, case_id,
                    f"{figures_dir}/fig_n1_contingency_case{case_id}.pdf")

    # Summary
    n1_results = [r for r in results if r['line_removed'] >= 0]
    stab_rates = [r['stability_rate'] for r in n1_results]

    print("\n" + "=" * 70)
    print("N-1 SUMMARY")
    print("=" * 70)
    print(f"Normal stability:      {results[0]['stability_rate']:.1f}%")
    print(f"N-1 avg stability:     {np.mean(stab_rates):.1f}%")
    print(f"N-1 min stability:     {np.min(stab_rates):.1f}%")
    print(f"N-1 worst-case line:   {n1_results[np.argmin(stab_rates)]['line_removed']}")
    print(f"Lines causing <90%:    {sum(1 for s in stab_rates if s < 90)}/{len(stab_rates)}")


def plot_n1_results(results, case_id, save_path):
    """Plot N-1 stability across all line removals."""
    n1_results = [r for r in results if r['line_removed'] >= 0]
    normal_stab = results[0]['stability_rate']

    fig, ax = plt.subplots(figsize=(12, 5))

    lines = [r['line_removed'] for r in n1_results]
    stabs = [r['stability_rate'] for r in n1_results]

    colors = ['green' if s >= 95 else 'orange' if s >= 80 else 'red' for s in stabs]
    ax.bar(range(len(lines)), stabs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=normal_stab, color='blue', linestyle='--', linewidth=2,
               label=f'Normal ({normal_stab:.0f}%)')
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='90% threshold')

    ax.set_xlabel('Removed Line Index', fontsize=12)
    ax.set_ylabel('Stability Rate (%)', fontsize=12)
    ax.set_title(f'IEEE {case_id}: N-1 Contingency Robustness', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(description='N-1 Contingency Robustness')
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/n1_contingency')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_n1_experiment(
        case_id=args.case,
        epochs=args.epochs,
        num_scenarios=args.scenarios,
        seed=args.seed,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
