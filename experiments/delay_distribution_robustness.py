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
Delay Distribution Robustness Experiment (V2)

Addresses Q2.5: Model is trained on lognormal delays only.
How does it perform when evaluated on exponential, gamma, Pareto, uniform?

Expected: <5% degradation on gamma/exponential, 5-15% on Pareto (heavy-tail).
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
from src.data.delay_distributions import (
    DELAY_DISTRIBUTIONS,
    generate_pareto_delays,
)
from src.data.synthetic_delays import SyntheticDelayGenerator
from src.utils.statistical_tests import set_all_seeds, restrict_gpus
from src.losses.coupling_loss import simple_stability_loss


def train_on_lognormal(
    case_id: int,
    epochs: int,
    num_scenarios: int,
    device: torch.device,
    seed: int = 42,
) -> torch.nn.Module:
    """Train model on lognormal delays."""
    set_all_seeds(seed)

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, _, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        delay_config=delay_config,
        batch_size=32,
        seed=seed,
    )

    dataset = train_loader.dataset.dataset
    n_generators = dataset.get_base_case()['n_generators']
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
        k_init_scale=0.1,
        adaptive_gamma=True,
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

            batch_size = energy_x.shape[0]
            n_nodes = energy_x.shape[1]
            energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
            comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
            batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)

            outputs = model(
                energy_x=energy_x_flat,
                energy_edge_index=energy_edge_index,
                comm_x=comm_x_flat,
                comm_edge_index=comm_edge_index,
                tau=tau,
                tau_max=tau_max[0],
                lambda_min_0=lambda_min_0,
                impedance_matrix=impedance_matrix,
                batch=batch_tensor,
            )

            loss = simple_stability_loss(outputs['rho'], outputs['K'], lambda_min_0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def evaluate_on_distribution(
    model: torch.nn.Module,
    distribution_name: str,
    n_generators: int,
    lambda_min_0: float,
    tau_max: torch.Tensor,
    n_eval: int = 500,
    seed: int = 42,
) -> Dict:
    """Evaluate model on a specific delay distribution."""
    model.eval()

    if distribution_name == 'pareto':
        taus = generate_pareto_delays(n_generators, n_eval, seed=seed)
    else:
        config = DELAY_DISTRIBUTIONS[distribution_name]
        gen = SyntheticDelayGenerator(n_generators, config, seed=seed)
        taus = gen.generate(n_eval)

    # Convert to seconds for model
    taus_s = taus / 1000.0

    margins = []
    with torch.no_grad():
        device = next(model.parameters()).device
        taus_s = taus_s.to(device)
        tau_max_dev = tau_max.to(device)
        lm0 = torch.tensor([lambda_min_0], device=device).expand(n_eval)

        rho = model.get_stability_margin(taus_s, tau_max_dev, lm0)
        margins = rho.cpu().numpy()

    return {
        'distribution': distribution_name,
        'stability_rate': float(np.mean(margins > 0) * 100),
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
        'min_margin': float(np.min(margins)),
        'p10_margin': float(np.percentile(margins, 10)),
        'delay_mean_ms': float(taus.mean()),
        'delay_p99_ms': float(torch.quantile(taus.flatten(), 0.99)),
    }


def run_robustness_experiment(
    case_id: int = 39,
    epochs: int = 100,
    num_scenarios: int = 500,
    n_eval: int = 500,
    seed: int = 42,
    output_dir: str = 'results/delay_robustness',
    figures_dir: str = 'docs/figures/publication',
):
    """Run delay distribution robustness experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("DELAY DISTRIBUTION ROBUSTNESS EXPERIMENT")
    print("=" * 70)
    print(f"Case: IEEE {case_id}, Train: lognormal, Epochs: {epochs}")
    print(f"Evaluate on: {list(DELAY_DISTRIBUTIONS.keys())}")

    # Train on lognormal
    print("\nTraining on lognormal delays...")
    model = train_on_lognormal(case_id, epochs, num_scenarios, device, seed)

    # Get case info
    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, _, _ = create_dataloaders(
        case_id=case_id, num_scenarios=100, delay_config=delay_config, seed=seed,
    )
    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']
    lambda_min_0 = base_case['lambda_min']
    tau_max = torch.ones(n_generators) * 0.5

    # Evaluate on all distributions
    results = []
    for dist_name in DELAY_DISTRIBUTIONS:
        print(f"\n  Evaluating on {dist_name}...", end=" ")
        result = evaluate_on_distribution(
            model, dist_name, n_generators, lambda_min_0,
            tau_max, n_eval=n_eval, seed=seed,
        )
        results.append(result)
        print(f"Stab={result['stability_rate']:.1f}%, "
              f"Margin={result['mean_margin']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'train_distribution': 'lognormal',
        'epochs': epochs,
        'n_eval': n_eval,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }

    json_path = f"{output_dir}/delay_robustness_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figure
    plot_robustness(results, case_id,
                    f"{figures_dir}/fig_delay_robustness_case{case_id}.pdf")

    # Summary
    lognormal_stab = [r for r in results if r['distribution'] == 'lognormal'][0]['stability_rate']
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    for r in results:
        degradation = lognormal_stab - r['stability_rate']
        print(f"  {r['distribution']:<15} Stab={r['stability_rate']:>6.1f}%, "
              f"Degradation={degradation:>+5.1f}%")


def plot_robustness(results: List[Dict], case_id: int, save_path: str):
    """Bar chart of stability rate across distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [r['distribution'] for r in results]
    stab_rates = [r['stability_rate'] for r in results]
    margins = [r['mean_margin'] for r in results]

    colors = ['#2E86AB' if n == 'lognormal' else '#F18F01' for n in names]
    bars = ax.bar(names, stab_rates, color=colors, alpha=0.8, edgecolor='black')

    # Annotate bars
    for bar, stab, margin in zip(bars, stab_rates, margins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{stab:.1f}%\n(ρ={margin:.3f})',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Stability Rate (%)', fontsize=12)
    ax.set_xlabel('Evaluation Delay Distribution', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Robustness to Delay Distribution '
                 '(Trained on Lognormal)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)

    # Highlight training distribution
    ax.annotate('Training\ndistribution', xy=(0, stab_rates[0]),
                xytext=(0, stab_rates[0] - 15),
                arrowprops=dict(arrowstyle='->', color='blue'),
                ha='center', fontsize=9, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Delay Distribution Robustness'
    )
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--n-eval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/delay_robustness')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_robustness_experiment(
        case_id=args.case,
        epochs=args.epochs,
        num_scenarios=args.scenarios,
        n_eval=args.n_eval,
        seed=args.seed,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
