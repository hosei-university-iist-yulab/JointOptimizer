#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/08/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Gamma Sweep Experiment — Find optimal physics mask strength.

Addresses Q3.3: Attention entropy H(Attn) = 0 because physics mask gamma=1.0
overwhelms QK attention scores, producing one-hot distributions.

Sweeps gamma values and the adaptive mode to find the setting that balances
physics locality bias with learned attention patterns (entropy > 0).
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
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import restrict_gpus
from src.losses.coupling_loss import simple_stability_loss


GAMMA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def measure_attention_entropy(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    impedance_matrix: torch.Tensor,
) -> Dict:
    """Measure attention entropy and stability margin for a model."""
    model.eval()
    entropies = []
    margins = []

    with torch.no_grad():
        for batch in dataloader:
            energy_x = batch['energy_x'].to(device)
            comm_x = batch['comm_x'].to(device)
            energy_edge_index = batch['energy_edge_index'].to(device)
            comm_edge_index = batch['comm_edge_index'].to(device)
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)

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

            margins.extend(outputs['rho'].cpu().numpy().tolist())

            # Extract attention entropy from outputs
            if 'attn_info' in outputs and outputs['attn_info'] is not None:
                attn_info = outputs['attn_info']
                if 'cross_attn_entropy' in attn_info:
                    entropies.append(attn_info['cross_attn_entropy'])

    return {
        'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
    }


def run_gamma_sweep(
    case_id: int = 39,
    epochs: int = 50,
    num_scenarios: int = 200,
    seeds: int = 3,
    output_dir: str = 'results/gamma_sweep',
    figures_dir: str = 'docs/figures/publication',
):
    """
    Sweep gamma values and adaptive mode. For each setting:
    1. Create model with that gamma
    2. Train for a few epochs
    3. Measure attention entropy and stability margin
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("GAMMA SWEEP: Finding optimal physics mask strength")
    print("=" * 70)
    print(f"Case: IEEE {case_id}, Epochs: {epochs}, Seeds: {seeds}")
    print(f"Gamma values: {GAMMA_VALUES} + adaptive")

    # Load data once
    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        delay_config=delay_config,
        batch_size=32,
    )
    dataset = train_loader.dataset.dataset
    n_generators = dataset.get_base_case()['n_generators']
    n_buses = dataset.get_base_case()['n_buses']
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    results = []

    # Test each fixed gamma + adaptive mode
    settings = [(g, False) for g in GAMMA_VALUES] + [(0.1, True)]

    for gamma, adaptive in settings:
        label = f"adaptive(init={gamma})" if adaptive else f"gamma={gamma}"
        print(f"\n--- Testing {label} ---")

        seed_results = []
        for seed in range(seeds):
            torch.manual_seed(seed * 42)
            np.random.seed(seed * 42)

            model = JointOptimizer(
                n_generators=n_generators,
                energy_input_dim=5,
                comm_input_dim=3,
                embed_dim=128,
                hidden_dim=256,
                num_heads=8,
                gnn_layers=3,
                k_init_scale=0.1,
                learnable_k=True,
                physics_gamma=gamma,
                adaptive_gamma=adaptive,
            ).to(device)

            # Quick training
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
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

            # Measure entropy and margin
            metrics = measure_attention_entropy(model, val_loader, device, impedance_matrix)
            seed_results.append(metrics)

            print(f"  Seed {seed}: entropy={metrics['mean_entropy']:.4f}, "
                  f"margin={metrics['mean_margin']:.4f}")

        # Aggregate across seeds
        result = {
            'gamma': gamma,
            'adaptive': adaptive,
            'label': label,
            'entropy_mean': float(np.mean([r['mean_entropy'] for r in seed_results])),
            'entropy_std': float(np.std([r['mean_entropy'] for r in seed_results])),
            'margin_mean': float(np.mean([r['mean_margin'] for r in seed_results])),
            'margin_std': float(np.std([r['mean_margin'] for r in seed_results])),
        }
        results.append(result)
        print(f"  Avg: entropy={result['entropy_mean']:.4f}±{result['entropy_std']:.4f}, "
              f"margin={result['margin_mean']:.4f}±{result['margin_std']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'epochs': epochs,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }
    json_path = f"{output_dir}/gamma_sweep_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figure
    plot_gamma_sweep(results, case_id, f"{figures_dir}/fig_gamma_sweep_case{case_id}.pdf")

    return results


def plot_gamma_sweep(results: List[Dict], case_id: int, save_path: str):
    """Plot entropy vs gamma with margin overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Separate fixed and adaptive
    fixed = [r for r in results if not r['adaptive']]
    adaptive = [r for r in results if r['adaptive']]

    gammas = [r['gamma'] for r in fixed]
    entropies = [r['entropy_mean'] for r in fixed]
    entropy_stds = [r['entropy_std'] for r in fixed]
    margins = [r['margin_mean'] for r in fixed]
    margin_stds = [r['margin_std'] for r in fixed]

    # Left axis: entropy
    color1 = '#2E86AB'
    ax1.errorbar(gammas, entropies, yerr=entropy_stds, color=color1,
                 marker='o', linewidth=2, markersize=8, capsize=3,
                 label='Attention Entropy')
    ax1.set_xlabel(r'Physics Mask Strength $\gamma$', fontsize=12)
    ax1.set_ylabel('Attention Entropy (nats)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')

    # Adaptive point
    if adaptive:
        ax1.axhline(y=adaptive[0]['entropy_mean'], color=color1,
                     linestyle='--', alpha=0.5, label='Adaptive entropy')

    # Right axis: margin
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.errorbar(gammas, margins, yerr=margin_stds, color=color2,
                 marker='s', linewidth=2, markersize=8, capsize=3,
                 label='Stability Margin')
    ax2.set_ylabel(r'Stability Margin $\rho$', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    if adaptive:
        ax2.axhline(y=adaptive[0]['margin_mean'], color=color2,
                     linestyle='--', alpha=0.5, label='Adaptive margin')

    # Highlight optimal region
    ax1.axvspan(0.05, 0.5, alpha=0.1, color='green', label='Optimal range')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    ax1.set_title(f'IEEE {case_id}-Bus: Gamma Sweep (Entropy vs Margin)', fontsize=13)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(description='Gamma Sweep Experiment')
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scenarios', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=str, default='results/gamma_sweep')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_gamma_sweep(
        case_id=args.case,
        epochs=args.epochs,
        num_scenarios=args.scenarios,
        seeds=args.seeds,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
