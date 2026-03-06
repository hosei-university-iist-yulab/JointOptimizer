#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/17/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Transfer Learning Experiment (V2)

Addresses Q4.3: Can a model trained on one IEEE case transfer to another?

Scenarios:
- Zero-shot: Apply model directly to target case (no training)
- Few-shot 1%: Fine-tune on 1% of target case data
- Few-shot 10%: Fine-tune on 10% of target case data
- Full fine-tune: Fine-tune on 100% of target case data
- Train from scratch: Baseline (train on target case only)

Expected: Transfer from IEEE 39 → 118 should work with 10% data,
as both share similar generator-level physics.
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import set_all_seeds, compute_statistics, restrict_gpus
from src.losses.coupling_loss import simple_stability_loss


def create_model(n_generators: int, config: Dict) -> JointOptimizer:
    """Create a fresh JointOptimizer."""
    return JointOptimizer(
        n_generators=n_generators,
        energy_input_dim=5,
        comm_input_dim=3,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        gnn_layers=config['gnn_layers'],
        k_init_scale=0.1,
        adaptive_gamma=True,
    )


def train_model(
    model: JointOptimizer,
    train_loader,
    device: torch.device,
    impedance_matrix: torch.Tensor,
    epochs: int,
    lr: float = 1e-4,
    freeze_gnn: bool = False,
) -> Dict:
    """Train model and return metrics."""
    model = model.to(device)

    if freeze_gnn:
        # Freeze GNN layers, only fine-tune attention + decoder + coupling
        for name, param in model.named_parameters():
            if 'gnn' in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5,
    )

    start_time = time.time()
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

    training_time = time.time() - start_time

    return {
        'training_time': training_time,
        'trainable_params': trainable_params,
    }


def evaluate_model(
    model: JointOptimizer,
    val_loader,
    device: torch.device,
    impedance_matrix: torch.Tensor,
) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    margins = []

    with torch.no_grad():
        for batch in val_loader:
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

            margins.extend(outputs['rho'].cpu().numpy().tolist())

    margins = np.array(margins)
    return {
        'stability_rate': float(np.mean(margins > 0) * 100),
        'mean_margin': float(np.mean(margins)),
        'std_margin': float(np.std(margins)),
        'K_mean': float(model.get_coupling_constants().mean().item()),
    }


def adapt_model_to_case(
    model: JointOptimizer,
    new_n_generators: int,
    device: torch.device,
) -> JointOptimizer:
    """
    Adapt a pretrained model to a new case with different n_generators.

    Reinitializes coupling constants and decoder, keeps GNN + attention.
    """
    from src.models.coupling import LearnableCouplingConstants
    from src.models.joint_optimizer import ControlDecoder, DelayPredictor

    old_config = model.config.copy()
    old_config['n_generators'] = new_n_generators

    new_model = JointOptimizer(**old_config).to(device)

    # Copy GNN weights (architecture-agnostic)
    new_model.gnn.load_state_dict(model.gnn.state_dict())

    # Copy attention weights
    new_model.attention.load_state_dict(model.attention.state_dict())

    return new_model


def run_transfer_learning(
    source_case: int = 39,
    target_case: int = 118,
    source_epochs: int = 200,
    target_epochs: int = 100,
    num_scenarios: int = 500,
    seed: int = 42,
    output_dir: str = 'results/transfer_learning',
):
    """Run full transfer learning experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(seed)

    print("=" * 70)
    print("TRANSFER LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Source: IEEE {source_case} → Target: IEEE {target_case}")

    config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
    }

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)

    # Load source case data
    print(f"\n1. Training on source case IEEE {source_case}...")
    src_train, src_val, _ = create_dataloaders(
        case_id=source_case, num_scenarios=num_scenarios,
        delay_config=delay_config, batch_size=32, seed=seed,
    )
    src_dataset = src_train.dataset.dataset
    src_n_gen = src_dataset.get_base_case()['n_generators']
    src_impedance = src_dataset.get_impedance_matrix()
    if src_impedance is not None:
        src_impedance = src_impedance.to(device)

    source_model = create_model(src_n_gen, config)
    train_model(source_model, src_train, device, src_impedance, source_epochs)
    src_metrics = evaluate_model(source_model, src_val, device, src_impedance)
    print(f"   Source performance: Stab={src_metrics['stability_rate']:.1f}%, "
          f"Margin={src_metrics['mean_margin']:.4f}")

    # Load target case data
    tgt_train, tgt_val, _ = create_dataloaders(
        case_id=target_case, num_scenarios=num_scenarios,
        delay_config=delay_config, batch_size=32, seed=seed,
    )
    tgt_dataset = tgt_train.dataset.dataset
    tgt_n_gen = tgt_dataset.get_base_case()['n_generators']
    tgt_impedance = tgt_dataset.get_impedance_matrix()
    if tgt_impedance is not None:
        tgt_impedance = tgt_impedance.to(device)

    # Scenarios to test
    data_fractions = [0.0, 0.01, 0.1, 1.0]
    results = []

    # 2. Zero-shot transfer
    print(f"\n2. Evaluating transfer scenarios on IEEE {target_case}...")

    for frac in data_fractions:
        label = {0.0: 'zero-shot', 0.01: 'few-shot-1%',
                 0.1: 'few-shot-10%', 1.0: 'full-finetune'}[frac]
        print(f"\n   {label}:", end=" ")

        set_all_seeds(seed)

        # Adapt model to target case
        adapted = adapt_model_to_case(source_model, tgt_n_gen, device)

        if frac > 0:
            # Create subset for fine-tuning
            n_total = len(tgt_train.dataset)
            n_use = max(1, int(n_total * frac))
            subset = torch.utils.data.Subset(
                tgt_train.dataset,
                list(range(n_use)),
            )
            from src.data.dataset import collate_fn
            subset_loader = torch.utils.data.DataLoader(
                subset, batch_size=32, shuffle=True, collate_fn=collate_fn,
            )

            ft_epochs = min(target_epochs, int(target_epochs / max(frac, 0.01)))
            train_info = train_model(
                adapted, subset_loader, device, tgt_impedance,
                epochs=ft_epochs, freeze_gnn=(frac < 0.1),
            )
        else:
            train_info = {'training_time': 0, 'trainable_params': 0}

        metrics = evaluate_model(adapted, tgt_val, device, tgt_impedance)
        metrics.update(train_info)
        metrics['data_fraction'] = frac
        metrics['label'] = label
        results.append(metrics)

        print(f"Stab={metrics['stability_rate']:.1f}%, "
              f"Margin={metrics['mean_margin']:.4f}")

    # 3. Train from scratch baseline
    print(f"\n   from-scratch:", end=" ")
    set_all_seeds(seed)
    scratch_model = create_model(tgt_n_gen, config)
    train_info = train_model(scratch_model, tgt_train, device, tgt_impedance, target_epochs)
    metrics = evaluate_model(scratch_model, tgt_val, device, tgt_impedance)
    metrics.update(train_info)
    metrics['data_fraction'] = 1.0
    metrics['label'] = 'from-scratch'
    results.append(metrics)
    print(f"Stab={metrics['stability_rate']:.1f}%, "
          f"Margin={metrics['mean_margin']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_dict = {
        'source_case': source_case,
        'target_case': target_case,
        'source_epochs': source_epochs,
        'target_epochs': target_epochs,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'source_metrics': src_metrics,
        'transfer_results': results,
    }

    json_path = f"{output_dir}/transfer_{source_case}_to_{target_case}.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING SUMMARY")
    print("=" * 70)
    print(f"\n{'Scenario':<20} {'Data':<10} {'Stability':<12} {'Margin':<12}")
    print("-" * 54)
    for r in results:
        print(f"{r['label']:<20} {r['data_fraction']*100:>6.0f}%   "
              f"{r['stability_rate']:>8.1f}%   {r['mean_margin']:>8.4f}")


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Transfer Learning Experiment'
    )
    parser.add_argument('--source', type=int, default=39)
    parser.add_argument('--target', type=int, default=118)
    parser.add_argument('--source-epochs', type=int, default=200)
    parser.add_argument('--target-epochs', type=int, default=100)
    parser.add_argument('--scenarios', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/transfer_learning')
    args = parser.parse_args()

    run_transfer_learning(
        source_case=args.source,
        target_case=args.target,
        source_epochs=args.source_epochs,
        target_epochs=args.target_epochs,
        num_scenarios=args.scenarios,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
