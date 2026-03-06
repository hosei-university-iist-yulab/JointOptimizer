#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/12/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Theorem 1 Validation: Delay-Stability Coupling

Validates the theoretical bound:
    ρ(τ) ≥ |λ_min(0)| - Σ K_i · τ_i / τ_max

Sweeps delay ranges and compares ALL models (Ours + 7 baselines) to show
how different architectures degrade under increasing communication delay.
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
from dataclasses import dataclass, asdict
from typing import List, Dict

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
from src.utils.statistical_tests import restrict_gpus


# Model configurations
MODEL_CONFIGS = {
    "JointOptimizer": {"type": "graph", "label": "Ours", "color": "#2E86AB", "marker": "o"},
    "B1_SequentialOPFQoS": {"type": "simple", "label": "B1", "color": "#F18F01", "marker": "s"},
    "B2_MLPJoint": {"type": "simple", "label": "B2", "color": "#A23B72", "marker": "^"},
    "B3_GNNOnly": {"type": "graph", "label": "B3", "color": "#C73E1D", "marker": "v"},
    "B4_LSTMJoint": {"type": "simple", "label": "B4", "color": "#3D5A80", "marker": "D"},
    "B5_CNNJoint": {"type": "simple", "label": "B5", "color": "#98C1D9", "marker": "p"},
    "B6_VanillaTransformer": {"type": "simple", "label": "B6", "color": "#6A994E", "marker": "h"},
    "B7_TransformerNoCoupling": {"type": "graph", "label": "B7", "color": "#BC6C25", "marker": "*"},
}


@dataclass
class DelayExperimentResult:
    """Results for a single delay setting."""
    model_name: str
    mean_delay_ms: float
    empirical_margin: float
    stability_rate: float
    K_mean: float


def create_model(model_name: str, n_buses: int, n_generators: int, config: Dict, lambda_min_0: float = None) -> torch.nn.Module:
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
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B1_SequentialOPFQoS":
        return SequentialOPFQoS(
            n_buses=n_buses,
            n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B2_MLPJoint":
        return MLPJoint(
            n_buses=n_buses,
            n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B3_GNNOnly":
        return GNNOnly(
            n_buses=n_buses,
            n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            num_layers=config['gnn_layers'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B4_LSTMJoint":
        return LSTMJoint(
            n_buses=n_buses,
            n_generators=n_generators,
            hidden_dim=config['hidden_dim'],
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B5_CNNJoint":
        return CNNJoint(
            n_buses=n_buses,
            n_generators=n_generators,
            hidden_channels=config['hidden_dim'] // 2,
            k_init_scale=config['k_init_scale'],
            lambda_min_0=lambda_min_0,
        )
    elif model_name == "B6_VanillaTransformer":
        return VanillaTransformer(
            n_buses=n_buses,
            n_generators=n_generators,
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


def run_delay_experiment_for_model(
    model: torch.nn.Module,
    model_name: str,
    case_id: int,
    mean_delay_ms: float,
    num_scenarios: int,
    device: torch.device,
    impedance_matrix: torch.Tensor,
) -> DelayExperimentResult:
    """Run experiment with specific delay configuration for a specific model."""

    delay_config = DelayConfig(
        distribution='lognormal',
        mean_ms=mean_delay_ms,
        std_ms=mean_delay_ms * 0.4,
        min_ms=1.0,
        max_ms=1000.0,
    )

    train_loader, _, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        train_split=1.0,
        val_split=0.0,
        batch_size=32,
        num_workers=0,
        seed=42,
        delay_config=delay_config,
    )

    model.eval()
    all_margins = []

    model_type = MODEL_CONFIGS[model_name]["type"]

    with torch.no_grad():
        for batch in train_loader:
            energy_x = batch['energy_x'].to(device)
            comm_x = batch['comm_x'].to(device)
            energy_edge_index = batch['energy_edge_index'].to(device)
            comm_edge_index = batch['comm_edge_index'].to(device)
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)

            batch_size = energy_x.shape[0]
            n_nodes = energy_x.shape[1]

            tau_max_tensor = tau_max[0] if tau_max.dim() > 1 else tau_max

            if model_type == "graph":
                # Graph-based models need flattened input
                energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
                comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
                batch_tensor = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)

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
            else:
                # Simple models take batch format
                outputs = model(
                    energy_x=energy_x,
                    comm_x=comm_x,
                    tau=tau,
                    tau_max=tau_max_tensor,
                    lambda_min_0=lambda_min_0,
                )

            all_margins.extend(outputs['rho'].cpu().numpy().tolist())

    margins = np.array(all_margins)
    K = model.get_coupling_constants().detach().cpu().numpy()

    return DelayExperimentResult(
        model_name=model_name,
        mean_delay_ms=mean_delay_ms,
        empirical_margin=float(np.mean(margins)),
        stability_rate=float(np.mean(margins > 0) * 100),
        K_mean=float(np.mean(K)),
    )


def plot_all_models_comparison(
    all_results: Dict[str, List[DelayExperimentResult]],
    save_path: str,
):
    """Generate comparison plot showing all models under delay sweep."""

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'font.weight': 'bold',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.figsize': (8, 5),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots()

    for model_name, results in all_results.items():
        config = MODEL_CONFIGS[model_name]
        delays = [r.mean_delay_ms for r in results]
        margins = [r.empirical_margin for r in results]

        linewidth = 3 if model_name == "JointOptimizer" else 1.5
        markersize = 10 if model_name == "JointOptimizer" else 6

        ax.plot(delays, margins,
                marker=config['marker'],
                color=config['color'],
                label=config['label'],
                linewidth=linewidth,
                markersize=markersize,
                alpha=0.9 if model_name == "JointOptimizer" else 0.7)

    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Instability')
    ax.set_xlabel(r'Communication Delay $\tau$ (ms)')
    ax.set_ylabel(r'Stability Margin $\rho(\tau)$')
    ax.set_title('Delay Robustness: All Models Comparison')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 520])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def validate_theorem1_all_models(
    case_id: int = 14,
    delay_values: List[float] = [10, 50, 100, 200, 300, 400, 500],
    num_scenarios: int = 100,
    output_dir: str = 'results/theorem1',
    figures_dir: str = 'docs/figures/publication',
):
    """Run Theorem 1 validation for ALL models."""

    print("=" * 70)
    print("THEOREM 1 VALIDATION: All Models Delay Comparison")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"IEEE Case: {case_id}")
    print(f"Delay values (ms): {delay_values}")
    print(f"Models: {len(MODEL_CONFIGS)}")

    # Config
    config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
        'k_init_scale': 0.1,
    }

    # Get dataset info
    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, _, _ = create_dataloaders(
        case_id=case_id, num_scenarios=50, delay_config=delay_config,
    )
    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']
    n_buses = base_case['n_buses']
    lambda_min_0 = base_case.get('lambda_min', None)
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    print(f"\nDataset: IEEE {case_id}-Bus")
    print(f"  Buses: {n_buses}, Generators: {n_generators}")

    all_results = {}

    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'=' * 50}")
        print(f"Testing: {model_name} ({MODEL_CONFIGS[model_name]['label']})")
        print(f"{'=' * 50}")

        # Create fresh model
        model = create_model(model_name, n_buses, n_generators, config, lambda_min_0=lambda_min_0).to(device)

        # Try to load checkpoint if exists
        checkpoint_path = f"checkpoints/baselines/{model_name}/model.pt"
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded checkpoint: {checkpoint_path}")
            except RuntimeError as e:
                print(f"  Checkpoint mismatch, using fresh model: {str(e)[:50]}...")
        else:
            print(f"  Using initialized model (no checkpoint)")

        results = []
        for delay_ms in delay_values:
            result = run_delay_experiment_for_model(
                model=model,
                model_name=model_name,
                case_id=case_id,
                mean_delay_ms=delay_ms,
                num_scenarios=num_scenarios,
                device=device,
                impedance_matrix=impedance_matrix,
            )
            results.append(result)
            print(f"  τ={delay_ms:3.0f}ms: ρ={result.empirical_margin:.4f}, "
                  f"Stable={result.stability_rate:.1f}%")

        all_results[model_name] = results

    # Save results (JSON to results/, figures to figures/)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results_dict = {
        'case_id': case_id,
        'delay_values': delay_values,
        'num_scenarios': num_scenarios,
        'timestamp': datetime.now().isoformat(),
        'models': {
            name: [asdict(r) for r in results]
            for name, results in all_results.items()
        },
    }

    json_path = f"{output_dir}/theorem1_all_models.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate comparison plot (save to figures/)
    plot_all_models_comparison(all_results, f"{figures_dir}/fig_theorem1_all_models.pdf")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Stability Margin at τ=500ms (highest delay)")
    print("=" * 70)
    print(f"{'Model':<25} {'ρ(500ms)':<12} {'Stable %':<10}")
    print("-" * 47)

    for model_name, results in all_results.items():
        final_result = results[-1]  # Last delay value
        print(f"{MODEL_CONFIGS[model_name]['label']:<25} "
              f"{final_result.empirical_margin:>8.4f}    "
              f"{final_result.stability_rate:>6.1f}%")

    return all_results


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=14)
    parser.add_argument('--delays', nargs='+', type=float,
                        default=[10, 50, 100, 200, 300, 400, 500])
    parser.add_argument('--scenarios', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/theorem1')
    parser.add_argument('--figures', type=str, default='docs/figures/publication')
    args = parser.parse_args()

    validate_theorem1_all_models(
        case_id=args.case,
        delay_values=args.delays,
        num_scenarios=args.scenarios,
        output_dir=args.output,
        figures_dir=args.figures,
    )


if __name__ == '__main__':
    main()
