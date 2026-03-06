#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/11/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Stressed Scenario Stability Test (V2)

Addresses Q1.6 and Q2.4: All models achieve 100% stability on normal conditions.
This experiment tests models under stressed conditions that push the system
toward instability, differentiating model robustness.

Stress types tested:
- High load (105%, 110%, 115%, 120%)
- N-1 / N-2 contingency
- Extreme delays (300ms, 500ms, 800ms, 1000ms)
- Heavy-tail (Pareto) delays
- Combined stressors

Expected: JointOptimizer maintains 5-20% stability advantage over baselines
under stress, with advantage growing under more severe conditions.
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
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.baselines import (
    SequentialOPFQoS,
    MLPJoint,
    GNNOnly,
    LSTMJoint,
    CNNJoint,
    VanillaTransformer,
    TransformerNoCoupling,
)
from src.data import create_dataloaders, DelayConfig
from src.data.stressed_scenarios import (
    StressedScenarioGenerator,
    StressConfig,
    STRESS_SCENARIOS,
)
from src.losses import JointLoss
from src.losses.coupling_loss import simple_stability_loss
from src.utils.statistical_tests import set_all_seeds, restrict_gpus


# Models to evaluate
MODEL_NAMES = [
    "JointOptimizer",
    "B1_SequentialOPFQoS",
    "B2_MLPJoint",
    "B3_GNNOnly",
    "B4_LSTMJoint",
    "B5_CNNJoint",
    "B6_VanillaTransformer",
    "B7_TransformerNoCoupling",
]

# Stress scenarios to test
STRESS_NAMES = [
    'normal',
    'load_110',
    'load_120',
    'n1',
    'n2',
    'delay_300',
    'delay_500',
    'delay_1000',
    'pareto_delays',
    'combined_moderate',
    'combined_severe',
]


def create_model(
    model_name: str, n_buses: int, n_generators: int, config: Dict, lambda_min_0: float = None
) -> torch.nn.Module:
    """Create model by name."""
    if model_name == "JointOptimizer":
        return JointOptimizer(
            n_generators=n_generators,
            energy_input_dim=5,
            comm_input_dim=3,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            gnn_layers=config['gnn_layers'],
            k_init_scale=config['k_init_scale'],
            learnable_k=True,
            adaptive_gamma=True,
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B1_SequentialOPFQoS":
        return SequentialOPFQoS(
            n_buses=n_buses, n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B2_MLPJoint":
        return MLPJoint(
            n_buses=n_buses, n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B3_GNNOnly":
        return GNNOnly(
            n_buses=n_buses, n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            num_layers=config['gnn_layers'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B4_LSTMJoint":
        return LSTMJoint(
            n_buses=n_buses, n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B5_CNNJoint":
        return CNNJoint(
            n_buses=n_buses, n_generators=n_generators,
            hidden_channels=config['hidden_dim'] // 2,
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B6_VanillaTransformer":
        return VanillaTransformer(
            n_buses=n_buses, n_generators=n_generators,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B7_TransformerNoCoupling":
        return TransformerNoCoupling(
            n_generators=n_generators,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            gnn_layers=config['gnn_layers'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model_quick(
    model: torch.nn.Module,
    model_name: str,
    train_loader,
    device: torch.device,
    impedance_matrix: torch.Tensor,
    epochs: int = 100,
) -> torch.nn.Module:
    """Train model for evaluation under stress."""
    model = model.to(device)
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

            optimizer.zero_grad()

            if model_name in ["JointOptimizer", "B3_GNNOnly", "B7_TransformerNoCoupling"]:
                energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
                comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
                batch_tensor = torch.arange(
                    batch_size, device=device
                ).repeat_interleave(n_nodes)

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
            else:
                outputs = model(
                    energy_x=energy_x,
                    comm_x=comm_x,
                    tau=tau,
                    tau_max=tau_max[0],
                    lambda_min_0=lambda_min_0,
                )

            loss = simple_stability_loss(outputs['rho'], outputs['K'], lambda_min_0)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    return model


def evaluate_under_stress(
    model: torch.nn.Module,
    model_name: str,
    stress_config: StressConfig,
    base_case: Dict,
    device: torch.device,
    impedance_matrix: torch.Tensor,
    n_eval: int = 100,
    seed: int = 42,
) -> Dict:
    """Evaluate a trained model under a stress scenario."""
    model.eval()

    generator = StressedScenarioGenerator(base_case)
    n_gen = base_case['n_generators']

    margins = []
    with torch.no_grad():
        for i in range(n_eval):
            stressed_case, tau = generator.apply_stress(
                stress_config, seed=seed + i
            )

            tau = tau.unsqueeze(0).to(device)
            tau_max_val = stressed_case.get('tau_max', None)
            if tau_max_val is None:
                tau_max = torch.ones(1, n_gen, device=device) * 0.5
            elif isinstance(tau_max_val, torch.Tensor):
                tau_max = tau_max_val.unsqueeze(0).to(device) if tau_max_val.dim() == 1 else tau_max_val.to(device)
            else:
                tau_max = torch.full((1, n_gen), float(tau_max_val), device=device)

            lm = stressed_case.get('lambda_min', base_case['lambda_min'])
            lm_val = float(lm) if not isinstance(lm, (int, float)) else lm
            lambda_min_0 = torch.tensor([lm_val], device=device)

            if hasattr(model, 'get_stability_margin'):
                rho = model.get_stability_margin(tau, tau_max[0], lambda_min_0)
            else:
                K = model.get_coupling_constants()
                tau_normalized = tau / tau_max[0].unsqueeze(0)
                rho = torch.abs(lambda_min_0) - (K.unsqueeze(0) * tau_normalized).sum(dim=-1)
            margins.extend(rho.cpu().numpy().tolist())

    margins = np.array(margins)
    return {
        'stress': stress_config.name,
        'model': model_name,
        'stability_rate': float(np.mean(margins > 0) * 100),
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
        'min_margin': float(np.min(margins)),
        'p10_margin': float(np.percentile(margins, 10)),
    }


def run_stress_test(
    case_id: int = 39,
    epochs: int = 100,
    num_scenarios: int = 500,
    n_eval: int = 100,
    seed: int = 42,
    output_dir: str = 'results/stress_test',
    figures_dir: str = 'docs/figures/publication',
):
    """Run full stress test across all models and scenarios."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(seed)

    print("=" * 70)
    print("STRESSED SCENARIO STABILITY TEST")
    print("=" * 70)
    print(f"Case: IEEE {case_id}, Training epochs: {epochs}")
    print(f"Stress scenarios: {STRESS_NAMES}")

    config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
        'k_init_scale': 0.1,
    }

    # Create training data (normal conditions)
    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        delay_config=delay_config,
        batch_size=32,
        seed=seed,
    )

    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']
    n_buses = base_case['n_buses']
    lambda_min_0 = base_case.get('lambda_min', None)
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    print(f"Buses: {n_buses}, Generators: {n_generators}")

    # Train all models on normal conditions
    trained_models = {}
    for model_name in MODEL_NAMES:
        print(f"\nTraining {model_name}...")
        set_all_seeds(seed)
        model = create_model(model_name, n_buses, n_generators, config, lambda_min_0=lambda_min_0)
        model = train_model_quick(
            model, model_name, train_loader, device,
            impedance_matrix, epochs=epochs,
        )
        trained_models[model_name] = model
        print(f"  Done. K={model.get_coupling_constants().mean().item():.4f}")

    # Evaluate all models under all stress scenarios
    all_results = []

    for stress_name in STRESS_NAMES:
        stress_config = STRESS_SCENARIOS[stress_name]
        print(f"\n--- Stress: {stress_name} ---")

        for model_name, model in trained_models.items():
            result = evaluate_under_stress(
                model, model_name, stress_config, base_case,
                device, impedance_matrix, n_eval=n_eval, seed=seed,
            )
            all_results.append(result)
            print(f"  {model_name:<30} Stab={result['stability_rate']:>6.1f}%, "
                  f"Margin={result['mean_margin']:>8.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_dict = {
        'case_id': case_id,
        'epochs': epochs,
        'n_eval': n_eval,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'stress_scenarios': STRESS_NAMES,
        'models': MODEL_NAMES,
        'results': all_results,
    }

    json_path = f"{output_dir}/stress_test_case{case_id}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figures
    plot_stress_heatmap(all_results, case_id,
                        f"{figures_dir}/fig_stress_heatmap_case{case_id}.pdf")
    plot_stress_degradation(all_results, case_id,
                            f"{figures_dir}/fig_stress_degradation_case{case_id}.pdf")

    # Print summary table
    print_stress_summary(all_results)


def plot_stress_heatmap(
    results: List[Dict], case_id: int, save_path: str,
):
    """Heatmap of stability rate across models x stress scenarios."""
    models = sorted(set(r['model'] for r in results))
    stresses = sorted(set(r['stress'] for r in results),
                      key=lambda s: STRESS_NAMES.index(s) if s in STRESS_NAMES else 999)

    matrix = np.zeros((len(models), len(stresses)))
    for r in results:
        i = models.index(r['model'])
        j = stresses.index(r['stress'])
        matrix[i, j] = r['stability_rate']

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(stresses)))
    ax.set_xticklabels([s.replace('_', '\n') for s in stresses],
                       fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(stresses)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    plt.colorbar(im, label='Stability Rate (%)')
    ax.set_title(f'IEEE {case_id}: Stability Under Stress', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_stress_degradation(
    results: List[Dict], case_id: int, save_path: str,
):
    """Line plot showing how stability degrades with increasing stress."""
    # Focus on load stress progression
    load_stresses = ['normal', 'load_105', 'load_110', 'load_115', 'load_120']
    available = [s for s in load_stresses if any(r['stress'] == s for r in results)]

    if len(available) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    models = sorted(set(r['model'] for r in results))
    for model, color in zip(models, colors):
        x_vals = []
        y_vals = []
        for stress_name in available:
            matching = [r for r in results
                        if r['model'] == model and r['stress'] == stress_name]
            if matching:
                load_factor = float(stress_name.split('_')[1]) / 100 if '_' in stress_name else 1.0
                x_vals.append(load_factor)
                y_vals.append(matching[0]['stability_rate'])

        if x_vals:
            label = model.replace('_', ' ')
            linewidth = 3 if model == 'JointOptimizer' else 1.5
            ax.plot(x_vals, y_vals, 'o-', color=color, label=label,
                    linewidth=linewidth, markersize=6)

    ax.set_xlabel('Load Factor', fontsize=12)
    ax.set_ylabel('Stability Rate (%)', fontsize=12)
    ax.set_title(f'IEEE {case_id}: Stability vs Load Stress', fontsize=13)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_stress_summary(results: List[Dict]):
    """Print a summary table of stress test results."""
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    # Find JointOptimizer advantages
    jo_results = {r['stress']: r for r in results if r['model'] == 'JointOptimizer'}

    for stress_name in STRESS_NAMES:
        stress_results = [r for r in results if r['stress'] == stress_name]
        if not stress_results:
            continue

        jo = jo_results.get(stress_name)
        if jo is None:
            continue

        best_baseline_stab = max(
            r['stability_rate'] for r in stress_results
            if r['model'] != 'JointOptimizer'
        )
        advantage = jo['stability_rate'] - best_baseline_stab

        print(f"\n  {stress_name}:")
        print(f"    JO stability:     {jo['stability_rate']:.1f}%")
        print(f"    Best baseline:    {best_baseline_stab:.1f}%")
        print(f"    JO advantage:     {advantage:+.1f}%")


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Stressed Scenario Stability Test'
    )
    parser.add_argument('--case', type=int, default=39)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--n-eval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/stress_test')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    run_stress_test(
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
