#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/11/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Systematic Ablation Study (V2 — Journal Extension)

Tests the impact of individual components (Q3.2):
- L_coupling weight (α)
- Physics mask (M_physics) on/off
- Causal mask (M_causal) on/off
- Cross-domain attention on/off
- Contrastive loss on/off
- GNN layers

V2 additions:
- Multi-case: IEEE 39, 57, 118
- Multi-seed with statistical aggregation
- New ablation dimensions: physics_mask, causal_mask, cross_attention, contrastive
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.losses import JointLoss
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import (
    set_all_seeds,
    compute_statistics,
    restrict_gpus,
    pairwise_wilcoxon,
)


@dataclass
class AblationResult:
    """Results for a single ablation configuration."""
    config_name: str
    config_value: Any
    stability_rate: float
    mean_stability_margin: float
    final_val_loss: float
    K_mean: float
    training_time: float
    seed: int = 0
    case_id: int = 39


def _ablation_reference(axis_name: str):
    """Reference (full-model) value for each ablation axis.

    The reference is the value that corresponds to the unablated proposed
    configuration; the Wilcoxon test then asks whether the reference's
    margin is statistically greater than each ablated alternative.
    """
    return {
        'alpha': 1.0,
        'physics_mask': True,
        'causal_mask': True,
        'cross_attention': True,
        'contrastive_loss': True,
        'gnn_layers': 3,
        'gamma': 0.1,
    }.get(axis_name)


def train_ablation_config(
    config_name: str,
    config_value: Any,
    case_id: int,
    epochs: int,
    num_scenarios: int,
    device: torch.device,
    base_config: Dict,
    seed: int = 42,
) -> AblationResult:
    """Train model with specific ablation configuration."""

    set_all_seeds(seed)

    # Modify config based on ablation
    config = base_config.copy()
    loss_config = {'alpha': 1.0, 'contrastive_weight': 0.1}
    model_overrides = {}

    # gamma_override carries the user's ablation value when the gamma axis is
    # being swept; None falls back to the model default (adaptive_gamma=True).
    gamma_override: Any = None

    if config_name == 'alpha':
        loss_config['alpha'] = config_value
    elif config_name == 'gnn_layers':
        config['gnn_layers'] = config_value
    elif config_name == 'embed_dim':
        config['embed_dim'] = config_value
    elif config_name == 'num_heads':
        config['num_heads'] = config_value
    elif config_name == 'physics_mask':
        model_overrides['use_physics_mask'] = config_value
    elif config_name == 'causal_mask':
        model_overrides['use_causal_mask'] = config_value
    elif config_name == 'cross_attention':
        model_overrides['use_cross_attention'] = config_value
    elif config_name == 'contrastive_loss':
        loss_config['contrastive_weight'] = 0.1 if config_value else 0.0
    elif config_name == 'gamma':
        # Major-revision ablation (APEN-D-26-05014, R3.1): sweep the physics-mask
        # strength gamma instead of leaving it adaptive. Setting an explicit
        # gamma value disables adaptive_gamma so the swept value is honored.
        gamma_override = float(config_value)

    # Create data
    delay_config = DelayConfig(
        distribution='lognormal',
        mean_ms=50.0,
        std_ms=20.0,
    )

    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
        delay_config=delay_config,
        seed=seed,
    )

    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']
    n_buses = base_case['n_buses']
    lambda_min_0_val = base_case.get('lambda_min', None)
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    # Create model with ablation overrides.
    # When gamma is explicitly swept (revision ablation R3.1), pin it via
    # physics_gamma and turn off adaptive scaling so the swept value applies.
    if gamma_override is not None:
        model_overrides.setdefault('physics_gamma', gamma_override)
        model_overrides.setdefault('adaptive_gamma', False)
    else:
        model_overrides.setdefault('adaptive_gamma', True)

    model = JointOptimizer(
        n_generators=n_generators,
        energy_input_dim=5,
        comm_input_dim=3,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        gnn_layers=config['gnn_layers'],
        k_init_scale=0.1,
        lambda_min_0=lambda_min_0_val,
        **model_overrides,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = JointLoss(
        alpha=loss_config['alpha'],
        beta=0.1,
        rho_min=0.01,
        tau_max=500.0,
        contrastive_weight=loss_config.get('contrastive_weight', 0.1),
    )

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            energy_x = batch['energy_x'].to(device)
            comm_x = batch['comm_x'].to(device)
            energy_edge_index = batch['energy_edge_index'].to(device)
            comm_edge_index = batch['comm_edge_index'].to(device)
            tau = batch['tau'].to(device)
            tau_max = batch['tau_max'].to(device)
            lambda_min_0 = batch['lambda_min_0'].to(device)
            u_prev = batch['u_prev'].to(device)
            P_load = batch['P_load'].to(device)

            batch_size = energy_x.shape[0]
            n_nodes = energy_x.shape[1]

            optimizer.zero_grad()

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

            n_gen = tau.shape[1]
            P_gen = outputs['u'][:, :n_gen]

            loss, _ = criterion(
                u=outputs['u'],
                rho=outputs['rho'],
                h_E=outputs.get('h_E'),
                h_I=outputs.get('h_I'),
                P_gen=P_gen,
                tau=tau,
                lambda_min_0=lambda_min_0,
                u_prev=u_prev,
                P_load=P_load,
                impedance_matrix=impedance_matrix,
                use_coupling_loss=(loss_config['alpha'] > 0),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_margins = []

        with torch.no_grad():
            for batch in val_loader:
                energy_x = batch['energy_x'].to(device)
                comm_x = batch['comm_x'].to(device)
                energy_edge_index = batch['energy_edge_index'].to(device)
                comm_edge_index = batch['comm_edge_index'].to(device)
                tau = batch['tau'].to(device)
                tau_max = batch['tau_max'].to(device)
                lambda_min_0 = batch['lambda_min_0'].to(device)
                u_prev = batch['u_prev'].to(device)
                P_load = batch['P_load'].to(device)

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

                n_gen = tau.shape[1]
                P_gen = outputs['u'][:, :n_gen]

                loss, _ = criterion(
                    u=outputs['u'],
                    rho=outputs['rho'],
                    h_E=outputs.get('h_E'),
                    h_I=outputs.get('h_I'),
                    P_gen=P_gen,
                    tau=tau,
                    lambda_min_0=lambda_min_0,
                    u_prev=u_prev,
                    P_load=P_load,
                    impedance_matrix=impedance_matrix,
                    use_coupling_loss=False,
                )

                val_loss += loss.item()
                val_batches += 1
                all_margins.extend(outputs['rho'].cpu().numpy().tolist())

        avg_val_loss = val_loss / max(val_batches, 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    training_time = time.time() - start_time
    margins = np.array(all_margins)
    K = model.get_coupling_constants().detach().cpu().numpy()

    return AblationResult(
        config_name=config_name,
        config_value=config_value,
        stability_rate=float(np.mean(margins > 0) * 100),
        mean_stability_margin=float(np.mean(margins)),
        final_val_loss=best_val_loss,
        K_mean=float(np.mean(K)),
        training_time=training_time,
        seed=seed,
        case_id=case_id,
    )


def plot_ablation_results(
    results: Dict[str, Dict[str, List]],
    save_dir: str,
):
    """Generate ablation study plots with multi-case + multi-seed stats."""

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'figure.dpi': 300,
    })

    for ablation_name, case_results in results.items():
        if not case_results:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(case_results)))

        for (case_id, agg_results), color in zip(sorted(case_results.items()), colors):
            # Skip meta-keys (e.g. "_wilcoxon_vs_reference") that are not real
            # ablation values and don't carry a 'margin' substructure.
            x_values = [v for v in agg_results.keys()
                        if not (isinstance(v, str) and v.startswith('_'))
                        and isinstance(agg_results[v], dict)
                        and 'margin' in agg_results[v]]
            means = [agg_results[v]['margin']['mean'] for v in x_values]
            stds = [agg_results[v]['margin']['std'] for v in x_values]

            # Convert boolean values to string labels for display
            x_display = []
            for v in x_values:
                if isinstance(v, bool):
                    x_display.append('On' if v else 'Off')
                else:
                    x_display.append(str(v))

            x_pos = range(len(x_values))
            ax.errorbar(x_pos, means, yerr=stds, fmt='o-', color=color,
                        linewidth=2, markersize=8, capsize=3,
                        label=f'IEEE {case_id}')

        ax.set_xticks(range(len(x_display)))
        ax.set_xticklabels(x_display)
        ax.set_xlabel(ablation_name.replace('_', ' ').title())
        ax.set_ylabel(r'Stability Margin $\rho(\tau)$ (mean ± std)')
        ax.set_title(f'Ablation: {ablation_name.replace("_", " ").title()}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{save_dir}/ablation_{ablation_name}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def run_ablation_study(
    case_ids: List[int] = None,
    epochs: int = 100,
    num_scenarios: int = 500,
    num_seeds: int = 20,
    seed_multiplier: int = 42,
    output_dir: str = 'results/ablation',
):
    """Run systematic ablation study (V2 + APEN-D-26-05014 revision).

    The major-revision response bumps the default seed count from 3 to 20 so
    Wilcoxon-signed-rank tests with Holm-Sidak correction reach the
    p < 0.01 regime (R1.9), and adds an explicit physics-mask-strength
    (gamma) sweep (R3.1).
    """

    if case_ids is None:
        case_ids = [39, 57]

    print("=" * 60)
    print("SYSTEMATIC ABLATION STUDY (V2 + revision response)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Cases: {case_ids}")
    print(f"Epochs: {epochs}, Seeds: {num_seeds}")

    base_config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
    }

    # V2 + revision ablation dimensions. The 'gamma' axis is added in the
    # major-revision response to APEN-D-26-05014 (R3.1) and sweeps the
    # physics-mask strength across half a decade in either direction.
    ablations = {
        'alpha': [0.0, 0.1, 0.5, 1.0, 2.0],
        'physics_mask': [False, True],
        'causal_mask': [False, True],
        'cross_attention': [False, True],
        'contrastive_loss': [False, True],
        'gnn_layers': [1, 2, 3, 4],
        'gamma': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    }

    # Results: ablation_name -> {case_id -> {value -> {margin: stats, ...}}}
    all_results = {}

    for ablation_name, values in ablations.items():
        print(f"\n{'=' * 50}")
        print(f"Ablating: {ablation_name}")
        print(f"{'=' * 50}")

        all_results[ablation_name] = {}

        for case_id in case_ids:
            print(f"\n  IEEE {case_id}:")
            case_agg = {}

            for value in values:
                print(f"    [{ablation_name}={value}]", end=" ", flush=True)

                seed_results = []
                for seed_idx in range(num_seeds):
                    seed = seed_idx * seed_multiplier
                    result = train_ablation_config(
                        config_name=ablation_name,
                        config_value=value,
                        case_id=case_id,
                        epochs=epochs,
                        num_scenarios=num_scenarios,
                        device=device,
                        base_config=base_config,
                        seed=seed,
                    )
                    seed_results.append(result)

                margins = [r.mean_stability_margin for r in seed_results]
                stab_rates = [r.stability_rate for r in seed_results]

                case_agg[value] = {
                    'margin': compute_statistics(margins),
                    'stability_rate': compute_statistics(stab_rates),
                    'K_mean': compute_statistics([r.K_mean for r in seed_results]),
                    'per_seed': [asdict(r) for r in seed_results],
                }

                m = case_agg[value]['margin']
                print(f"Margin={m['mean']:.4f}±{m['std']:.4f}")

            # Major-revision response (APEN-D-26-05014, R1.9): for each
            # ablation axis, run paired Wilcoxon-signed-rank tests against
            # the reference (full-model) value with Holm-Sidak correction
            # across the values on that axis, so each ablated component
            # reports a calibrated p-value rather than mean +/- std alone.
            ref_value = _ablation_reference(ablation_name)
            if ref_value is not None and ref_value in case_agg and len(values) > 1:
                ref_margins = case_agg[ref_value]['margin']['values']
                comparisons = {
                    str(v): case_agg[v]['margin']['values']
                    for v in values if v != ref_value and 'values' in case_agg[v]['margin']
                }
                if comparisons:
                    case_agg['_wilcoxon_vs_reference'] = {
                        'reference_value': ref_value,
                        'tests': pairwise_wilcoxon(
                            ref_margins, comparisons, alternative='greater'
                        ),
                    }

            all_results[ablation_name][case_id] = case_agg

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Convert keys to strings for JSON serialization
    def serialize_results(obj):
        if isinstance(obj, dict):
            return {str(k): serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_results(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    results_dict = {
        'case_ids': case_ids,
        'epochs': epochs,
        'num_scenarios': num_scenarios,
        'num_seeds': num_seeds,
        'timestamp': datetime.now().isoformat(),
        'ablations': serialize_results(all_results),
    }

    json_path = f"{output_dir}/ablation_results_v2.json"

    def _json_safe(o):
        # numpy.bool_ -> bool, numpy.float32/int64 etc. -> Python scalar
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, 'item'):
            return o.item()
        return str(o)

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=_json_safe)
    print(f"\nResults saved to: {json_path}")

    # Generate plots
    plot_ablation_results(all_results, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (V2)")
    print("=" * 60)

    for ablation_name, case_data in all_results.items():
        print(f"\n{ablation_name}:")
        for case_id, values in sorted(case_data.items()):
            print(f"  IEEE {case_id}:")
            for value, stats in values.items():
                m = stats['margin']
                s = stats['stability_rate']
                print(f"    {value}: Margin={m['mean']:.4f}±{m['std']:.4f}, "
                      f"Stab={s['mean']:.1f}±{s['std']:.1f}%")

    return all_results


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Systematic Ablation Study (V2)'
    )
    parser.add_argument('--cases', type=str, default='39,57',
                        help='Comma-separated case IDs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed-multiplier', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/ablation')
    args = parser.parse_args()

    case_ids = [int(c.strip()) for c in args.cases.split(',')]

    run_ablation_study(
        case_ids=case_ids,
        epochs=args.epochs,
        num_scenarios=args.scenarios,
        num_seeds=args.seeds,
        seed_multiplier=args.seed_multiplier,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
