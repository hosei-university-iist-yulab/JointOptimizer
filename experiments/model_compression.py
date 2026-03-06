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
Model Size Sweep / Compression Analysis (V2)

Addresses Q3.6: Find the Pareto frontier of model size vs performance.
Sweep embed_dim to find minimum model that achieves target performance.

Expected: embed_dim=64 achieves 90% of full model performance,
embed_dim=128 (default) is near-optimal.
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


EMBED_DIMS = [32, 64, 96, 128, 192, 256]


def train_and_evaluate(
    case_id: int,
    embed_dim: int,
    epochs: int,
    num_scenarios: int,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    """Train model with specific embed_dim and evaluate."""
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

    # Scale hidden_dim proportionally
    hidden_dim = embed_dim * 2
    num_heads = max(1, embed_dim // 16)

    model = JointOptimizer(
        n_generators=n_gen, energy_input_dim=5, comm_input_dim=3,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_heads=num_heads, gnn_layers=3,
        k_init_scale=0.1, adaptive_gamma=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
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

    # Evaluate
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

    return {
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'n_parameters': n_params,
        'stability_rate': float(np.mean(margins > 0) * 100),
        'mean_margin': float(np.mean(margins)),
        'K_mean': float(model.get_coupling_constants().mean().item()),
    }


def run_model_sweep(
    case_id: int = 39,
    epochs: int = 100,
    num_scenarios: int = 500,
    seed: int = 42,
    output_dir: str = 'results/model_compression',
    figures_dir: str = 'docs/figures/publication',
):
    """Sweep model sizes to find Pareto frontier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("MODEL SIZE SWEEP")
    print("=" * 70)
    print(f"Case: IEEE {case_id}")
    print(f"Embed dims: {EMBED_DIMS}")

    results = []
    for embed_dim in EMBED_DIMS:
        print(f"\n  embed_dim={embed_dim}...", end=" ", flush=True)
        result = train_and_evaluate(
            case_id, embed_dim, epochs, num_scenarios, device, seed,
        )
        results.append(result)
        print(f"Margin={result['mean_margin']:.4f}, "
              f"Params={result['n_parameters']:,}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'epochs': epochs,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }

    json_path = f"{output_dir}/model_sweep_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Plot Pareto frontier
    plot_pareto(results, case_id,
                f"{figures_dir}/fig_model_compression_case{case_id}.pdf")

    # Summary
    print("\n" + "=" * 70)
    print("MODEL SIZE SUMMARY")
    print("=" * 70)
    print(f"\n{'Embed':<8} {'Hidden':<8} {'Params':<12} {'Margin':<12} {'Stability':<12}")
    print("-" * 52)
    for r in results:
        print(f"{r['embed_dim']:<8} {r['hidden_dim']:<8} {r['n_parameters']:>10,}   "
              f"{r['mean_margin']:>8.4f}   {r['stability_rate']:>8.1f}%")


def plot_pareto(results, case_id, save_path):
    """Plot Pareto frontier of params vs performance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    params = [r['n_parameters'] / 1000 for r in results]  # K params
    margins = [r['mean_margin'] for r in results]
    dims = [r['embed_dim'] for r in results]

    ax.plot(params, margins, 'o-', color='#2E86AB', linewidth=2, markersize=10)

    for p, m, d in zip(params, margins, dims):
        ax.annotate(f'd={d}', (p, m), textcoords='offset points',
                    xytext=(5, 5), fontsize=8)

    # Highlight default (128)
    default_idx = dims.index(128)
    ax.scatter([params[default_idx]], [margins[default_idx]],
               color='red', s=200, zorder=5, marker='*', label='Default (d=128)')

    ax.set_xlabel('Model Parameters (K)', fontsize=12)
    ax.set_ylabel(r'Stability Margin $\rho(\tau)$', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Model Size vs Performance', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(description='Model Size Sweep')
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/model_compression')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_model_sweep(
        case_id=args.case,
        epochs=args.epochs,
        num_scenarios=args.scenarios,
        seed=args.seed,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
