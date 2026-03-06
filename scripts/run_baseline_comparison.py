#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/16/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Baseline Comparison Experiments (V2 — Journal Extension)

Compares all baselines against the proposed JointOptimizer model.
V2 additions (Q2.1, Q2.2):
- Multi-seed training with statistical aggregation
- Wilcoxon signed-rank tests + Holm-Sidak correction
- YAML config support for per-case epochs
- IEEE 118 support
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import yaml
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

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
from src.losses import JointLoss
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import (
    set_all_seeds,
    compute_statistics,
    pairwise_wilcoxon,
    friedman_nemenyi,
    restrict_gpus,
)


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    stability_rate: float
    mean_stability_margin: float
    min_stability_margin: float
    final_val_loss: float
    K_mean: float
    training_time: float
    num_parameters: int
    seed: int = 0


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


def train_model(
    model: torch.nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    criterion: JointLoss,
    device: torch.device,
    epochs: int,
    lr: float,
    impedance_matrix: torch.Tensor,
    n_buses: int,
    use_coupling_loss: bool = True,
) -> ModelResult:
    """Train a single model and return results."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Training {model_name} ({num_params:,} params)...")

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
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

            # Forward pass depends on model type
            if model_name in ["JointOptimizer", "B3_GNNOnly", "B7_TransformerNoCoupling"]:
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
                    impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
                    batch=batch_tensor,
                )
            else:
                # Simple models that take batch format
                outputs = model(
                    energy_x=energy_x,
                    comm_x=comm_x,
                    tau=tau,
                    tau_max=tau_max[0],
                    lambda_min_0=lambda_min_0,
                )

            n_gen = tau.shape[1]
            P_gen = outputs['u'][:, :n_gen]

            # Compute loss (skip contrastive for incompatible models)
            # Only use contrastive for JointOptimizer and B7
            h_E = outputs.get('h_E') if model_name in ["JointOptimizer", "B7_TransformerNoCoupling"] else None
            h_I = outputs.get('h_I') if model_name in ["JointOptimizer", "B7_TransformerNoCoupling"] else None

            loss, _ = criterion(
                u=outputs['u'],
                rho=outputs['rho'],
                h_E=h_E,
                h_I=h_I,
                P_gen=P_gen,
                tau=tau,
                lambda_min_0=lambda_min_0,
                u_prev=u_prev,
                P_load=P_load,
                impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
                use_coupling_loss=use_coupling_loss and model_name != "B7_TransformerNoCoupling",
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

                if model_name in ["JointOptimizer", "B3_GNNOnly", "B7_TransformerNoCoupling"]:
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
                        impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
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

                n_gen = tau.shape[1]
                P_gen = outputs['u'][:, :n_gen]

                # Skip contrastive for incompatible models
                h_E = outputs.get('h_E') if model_name in ["JointOptimizer", "B7_TransformerNoCoupling"] else None
                h_I = outputs.get('h_I') if model_name in ["JointOptimizer", "B7_TransformerNoCoupling"] else None

                loss, _ = criterion(
                    u=outputs['u'],
                    rho=outputs['rho'],
                    h_E=h_E,
                    h_I=h_I,
                    P_gen=P_gen,
                    tau=tau,
                    lambda_min_0=lambda_min_0,
                    u_prev=u_prev,
                    P_load=P_load,
                    impedance_matrix=impedance_matrix.to(device) if impedance_matrix is not None else None,
                    use_coupling_loss=False,  # Don't include in val loss
                )

                val_loss += loss.item()
                val_batches += 1
                all_margins.extend(outputs['rho'].cpu().numpy().tolist())

        avg_val_loss = val_loss / max(val_batches, 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if (epoch + 1) % 10 == 0:
            K = model.get_coupling_constants()
            print(f"    Epoch {epoch+1}: Val={avg_val_loss:.2f}, K={K.mean().item():.4f}")

    training_time = time.time() - start_time
    margins = np.array(all_margins)

    K = model.get_coupling_constants().detach().cpu().numpy()

    result = ModelResult(
        model_name=model_name,
        stability_rate=float(np.mean(margins > 0) * 100),
        mean_stability_margin=float(np.mean(margins)),
        min_stability_margin=float(np.min(margins)),
        final_val_loss=best_val_loss,
        K_mean=float(np.mean(K)),
        training_time=training_time,
        num_parameters=num_params,
    )

    # Save checkpoint
    checkpoint_dir = f"checkpoints/baselines/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'result': asdict(result),
    }, f"{checkpoint_dir}/model.pt")

    print(f"    Done: Stability={result.stability_rate:.1f}%, Margin={result.mean_stability_margin:.4f}")

    return result


def run_comparison(
    case_id: int = 14,
    epochs: int = 50,
    num_scenarios: int = 500,
    device_str: str = "auto",
    output_dir: str = "results/baselines",
    seed: int = 42,
):
    """Run baseline comparison for a single seed."""
    device = torch.device('cuda' if device_str == 'auto' and torch.cuda.is_available() else 'cpu')

    set_all_seeds(seed)

    # Config
    config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
        'k_init_scale': 0.1,
    }

    # Create data
    delay_config = DelayConfig(
        distribution='lognormal',
        mean_ms=50.0,
        std_ms=20.0,
        min_ms=5.0,
        max_ms=500.0,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
        num_workers=0,
        seed=seed,
        delay_config=delay_config,
    )

    dataset = train_loader.dataset.dataset
    base_case = dataset.get_base_case()
    n_generators = base_case['n_generators']
    n_buses = base_case['n_buses']
    lambda_min_0 = base_case.get('lambda_min', None)
    impedance_matrix = dataset.get_impedance_matrix()

    criterion = JointLoss(
        alpha=1.0,
        beta=0.1,
        rho_min=0.01,
        tau_max=500.0,
    )

    # Models to compare
    model_names = [
        "JointOptimizer",      # Proposed method
        "B1_SequentialOPFQoS", # Decoupled baseline
        "B2_MLPJoint",         # Simple MLP
        "B3_GNNOnly",          # GNN without attention
        "B4_LSTMJoint",        # Recurrent
        "B5_CNNJoint",         # Convolutional
        "B6_VanillaTransformer", # Standard transformer
        "B7_TransformerNoCoupling", # Ablation: no L_coupling
    ]

    results = []

    for model_name in model_names:
        set_all_seeds(seed)
        model = create_model(model_name, n_buses, n_generators, config, lambda_min_0=lambda_min_0)
        result = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=epochs,
            lr=1e-4,
            impedance_matrix=impedance_matrix,
            n_buses=n_buses,
        )
        result.seed = seed
        results.append(result)

    return results, n_buses, n_generators


def run_multi_seed_comparison(
    case_id: int = 14,
    epochs: int = 50,
    num_scenarios: int = 500,
    num_seeds: int = 5,
    seed_multiplier: int = 42,
    device_str: str = "auto",
    output_dir: str = "results/baselines",
):
    """Run multi-seed baseline comparison with statistical tests (V2, Q2.2)."""
    print("=" * 70)
    print("BASELINE COMPARISON (V2 — Multi-Seed)")
    print("=" * 70)
    print(f"Case: IEEE {case_id}, Epochs: {epochs}, Seeds: {num_seeds}")

    all_seed_results = {}  # model_name -> [ModelResult per seed]

    for seed_idx in range(num_seeds):
        seed = seed_idx * seed_multiplier
        print(f"\n--- Seed {seed_idx} (seed={seed}) ---")
        results, n_buses, n_generators = run_comparison(
            case_id=case_id,
            epochs=epochs,
            num_scenarios=num_scenarios,
            device_str=device_str,
            output_dir=output_dir,
            seed=seed,
        )
        for r in results:
            if r.model_name not in all_seed_results:
                all_seed_results[r.model_name] = []
            all_seed_results[r.model_name].append(r)

    # Aggregate statistics
    os.makedirs(output_dir, exist_ok=True)

    aggregated = {}
    for model_name, seed_results in all_seed_results.items():
        margins = [r.mean_stability_margin for r in seed_results]
        stab_rates = [r.stability_rate for r in seed_results]
        val_losses = [r.final_val_loss for r in seed_results]
        k_means = [r.K_mean for r in seed_results]

        margin_stats = compute_statistics(margins)
        stab_stats = compute_statistics(stab_rates)

        aggregated[model_name] = {
            'margin': margin_stats,
            'stability_rate': stab_stats,
            'val_loss': compute_statistics(val_losses),
            'K_mean': compute_statistics(k_means),
            'training_time_mean': float(np.mean([r.training_time for r in seed_results])),
            'num_parameters': seed_results[0].num_parameters,
            'per_seed': [asdict(r) for r in seed_results],
        }

    # Statistical tests: proposed vs each baseline
    proposed_margins = [r.mean_stability_margin for r in all_seed_results.get('JointOptimizer', [])]
    baseline_margins = {}
    for name, results in all_seed_results.items():
        if name != 'JointOptimizer':
            baseline_margins[name] = [r.mean_stability_margin for r in results]

    stat_tests = {}
    if proposed_margins and baseline_margins:
        stat_tests = pairwise_wilcoxon(proposed_margins, baseline_margins)

    # Friedman test across all models
    friedman_result = None
    if len(all_seed_results) >= 3 and num_seeds >= 3:
        all_model_margins = {
            name: [r.mean_stability_margin for r in results]
            for name, results in all_seed_results.items()
        }
        friedman_result = friedman_nemenyi(all_model_margins)

    # Save results
    output_dict = {
        'case_id': case_id,
        'epochs': epochs,
        'num_scenarios': num_scenarios,
        'num_seeds': num_seeds,
        'seed_multiplier': seed_multiplier,
        'timestamp': datetime.now().isoformat(),
        'aggregated_results': aggregated,
        'statistical_tests': stat_tests,
        'friedman_test': friedman_result,
    }

    json_path = f"{output_dir}/comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2, cls=NumpyEncoder)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Stability':<16} {'Margin':<20} {'p-value':<10} {'Params':<10}")
    print("-" * 86)

    for model_name in all_seed_results:
        agg = aggregated[model_name]
        m = agg['margin']
        s = agg['stability_rate']
        p_str = ""
        if model_name in stat_tests:
            p = stat_tests[model_name].get('p_value_corrected', stat_tests[model_name].get('p_value', ''))
            if isinstance(p, float):
                p_str = f"{p:.4f}"

        print(f"{model_name:<30} {s['mean']:>6.1f}±{s['std']:>4.1f}%   "
              f"{m['mean']:>7.4f}±{m['std']:>6.4f}   {p_str:<10} {agg['num_parameters']:>8,}")

    if friedman_result:
        print(f"\nFriedman test: statistic={friedman_result['statistic']:.2f}, "
              f"p={friedman_result['p_value']:.4f}")

    print("-" * 86)
    print(f"\nResults saved to: {json_path}")


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds for multi-seed comparison')
    parser.add_argument('--seed-multiplier', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (overrides other args)')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)
        case_id = args.case  # Still allow CLI override
        epochs = cfg.get('data', {}).get('epochs_per_case', {}).get(case_id, args.epochs)
        num_scenarios = cfg.get('data', {}).get('num_scenarios', args.scenarios)
        num_seeds = cfg.get('statistics', {}).get('num_seeds', args.seeds)
        seed_multiplier = cfg.get('statistics', {}).get('seed_multiplier', args.seed_multiplier)
    else:
        case_id = args.case
        epochs = args.epochs
        num_scenarios = args.scenarios
        num_seeds = args.seeds
        seed_multiplier = args.seed_multiplier

    output_dir = args.output or f'results/baselines_case{case_id}'

    if num_seeds > 1:
        run_multi_seed_comparison(
            case_id=case_id,
            epochs=epochs,
            num_scenarios=num_scenarios,
            num_seeds=num_seeds,
            seed_multiplier=seed_multiplier,
            device_str=args.device,
            output_dir=output_dir,
        )
    else:
        # Legacy single-seed mode
        print("=" * 70)
        print("BASELINE COMPARISON EXPERIMENT")
        print("=" * 70)

        results, n_buses, n_generators = run_comparison(
            case_id=case_id,
            epochs=epochs,
            num_scenarios=num_scenarios,
            device_str=args.device,
            output_dir=output_dir,
        )

        os.makedirs(output_dir, exist_ok=True)
        results_dict = {
            'case_id': case_id,
            'epochs': epochs,
            'num_scenarios': num_scenarios,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in results],
        }
        with open(f"{output_dir}/comparison_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n{'Model':<30} {'Stability':<12} {'Margin':<12} {'Val Loss':<12} {'Params':<10}")
        print("-" * 76)
        for r in results:
            print(f"{r.model_name:<30} {r.stability_rate:>8.1f}%    {r.mean_stability_margin:>8.4f}    "
                  f"{r.final_val_loss:>10.0f}   {r.num_parameters:>8,}")
        print("-" * 76)
        print(f"\nResults saved to: {output_dir}/comparison_results.json")


if __name__ == '__main__':
    main()
