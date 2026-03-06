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
Observation 1 Validation: Cross-Domain Information Flow Analysis

(Formerly "Theorem 2" — downgraded to Observation per V2 Q1.4.
The information-theoretic rate bound lacks a formal proof chain and
is better presented as an empirical observation.)

Estimates mutual information and entropy from attention distributions
to validate that cross-domain information exchange is meaningful.
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
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import restrict_gpus


@dataclass
class InformationMetrics:
    """Information-theoretic metrics."""
    attention_entropy: float
    cross_domain_mi: float  # Mutual information estimate
    energy_entropy: float
    comm_entropy: float


def estimate_entropy(attention_weights: torch.Tensor) -> float:
    """
    Estimate entropy from attention distribution.
    H(A) = -Σ p_i log(p_i)
    """
    probs = attention_weights.flatten()
    probs = probs / probs.sum()
    probs = probs[probs > 1e-10]

    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return entropy.item()


def estimate_mutual_information(
    h_E: torch.Tensor,
    h_I: torch.Tensor,
) -> float:
    """
    Estimate mutual information between energy and communication embeddings
    using correlation-based proxy.

    I(h_E; h_I) ≈ -0.5 * log(1 - ρ²)
    where ρ is the correlation coefficient.
    """
    h_E_flat = h_E.flatten()
    h_I_flat = h_I.flatten()

    min_len = min(len(h_E_flat), len(h_I_flat))
    h_E_flat = h_E_flat[:min_len]
    h_I_flat = h_I_flat[:min_len]

    h_E_centered = h_E_flat - h_E_flat.mean()
    h_I_centered = h_I_flat - h_I_flat.mean()

    correlation = torch.dot(h_E_centered, h_I_centered) / (
        torch.norm(h_E_centered) * torch.norm(h_I_centered) + 1e-10
    )

    rho_sq = torch.clamp(correlation ** 2, 0, 0.999)
    mi = -0.5 * torch.log(1 - rho_sq)

    return mi.item()


def extract_attention_weights(model: torch.nn.Module) -> Dict:
    """Extract attention weights from model for analysis."""
    attention_weights = {}

    if hasattr(model, 'cross_attention'):
        if hasattr(model.cross_attention, 'last_attention_weights'):
            attention_weights['cross_attention'] = model.cross_attention.last_attention_weights

    return attention_weights


def run_information_analysis(
    model: torch.nn.Module,
    case_id: int,
    num_scenarios: int,
    device: torch.device,
) -> InformationMetrics:
    """Run information-theoretic analysis on model."""

    delay_config = DelayConfig(
        distribution='lognormal',
        mean_ms=50.0,
        std_ms=20.0,
    )

    train_loader, val_loader, _ = create_dataloaders(
        case_id=case_id,
        num_scenarios=num_scenarios,
        train_split=0.7,
        val_split=0.3,
        batch_size=32,
        delay_config=delay_config,
    )
    val_loader = train_loader

    dataset = val_loader.dataset.dataset
    impedance_matrix = dataset.get_impedance_matrix()
    if impedance_matrix is not None:
        impedance_matrix = impedance_matrix.to(device)

    model.eval()
    all_mi = []
    all_entropy_E = []
    all_entropy_I = []

    with torch.no_grad():
        for batch in val_loader:
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

            h_E = outputs.get('h_E')
            h_I = outputs.get('h_I')

            if h_E is not None and h_I is not None:
                for i in range(min(batch_size, h_E.shape[0] // n_nodes)):
                    start_idx = i * n_nodes
                    end_idx = (i + 1) * n_nodes

                    if end_idx <= h_E.shape[0] and end_idx <= h_I.shape[0]:
                        h_E_sample = h_E[start_idx:end_idx]
                        h_I_sample = h_I[start_idx:end_idx]

                        mi = estimate_mutual_information(h_E_sample, h_I_sample)
                        all_mi.append(mi)

                        entropy_E = torch.var(h_E_sample).item()
                        entropy_I = torch.var(h_I_sample).item()
                        all_entropy_E.append(entropy_E)
                        all_entropy_I.append(entropy_I)

    attention_weights = extract_attention_weights(model)
    if attention_weights:
        attention_entropy = np.mean([
            estimate_entropy(w) for w in attention_weights.values()
        ])
    else:
        attention_entropy = 0.0

    return InformationMetrics(
        attention_entropy=attention_entropy,
        cross_domain_mi=float(np.mean(all_mi)) if all_mi else 0.0,
        energy_entropy=float(np.mean(all_entropy_E)) if all_entropy_E else 0.0,
        comm_entropy=float(np.mean(all_entropy_I)) if all_entropy_I else 0.0,
    )


def plot_information_analysis(
    metrics: InformationMetrics,
    save_path: str,
):
    """Generate information analysis visualization."""

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'figure.figsize': (5, 3.5),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots()

    labels = ['Cross-Domain MI', 'Energy H(h_E)', 'Comm H(h_I)']
    values = [metrics.cross_domain_mi, metrics.energy_entropy, metrics.comm_entropy]
    colors = ['#2E86AB', '#4CAF50', '#F18F01']

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Information (nats)')
    ax.set_title('Observation 1: Cross-Domain Information Flow')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def validate_domain_separation(
    case_id: int = 14,
    num_scenarios: int = 200,
    checkpoint_path: str = None,
    output_dir: str = 'results/domain_separation',
):
    """Run Observation 1 validation (formerly Theorem 2)."""

    print("=" * 60)
    print("OBSERVATION 1: Cross-Domain Information Flow Analysis")
    print("(formerly Theorem 2 — downgraded per V2)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"IEEE Case: {case_id}")

    delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
    train_loader, _, _ = create_dataloaders(
        case_id=case_id, num_scenarios=100, delay_config=delay_config,
    )
    dataset = train_loader.dataset.dataset
    n_generators = dataset.get_base_case()['n_generators']

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

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")

    print("\nRunning information analysis...")
    metrics = run_information_analysis(
        model=model,
        case_id=case_id,
        num_scenarios=num_scenarios,
        device=device,
    )

    os.makedirs(output_dir, exist_ok=True)

    results_dict = {
        'case_id': case_id,
        'num_scenarios': num_scenarios,
        'timestamp': datetime.now().isoformat(),
        'metrics': asdict(metrics),
        'note': 'Observation 1 (formerly Theorem 2, downgraded per V2 Q1.4)',
    }

    json_path = f"{output_dir}/domain_separation_validation.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    plot_information_analysis(metrics, f"{output_dir}/observation1_validation.pdf")

    print("\n" + "=" * 60)
    print("INFORMATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Cross-Domain Mutual Information: {metrics.cross_domain_mi:.4f} nats")
    print(f"Energy Embedding Entropy: {metrics.energy_entropy:.4f}")
    print(f"Communication Embedding Entropy: {metrics.comm_entropy:.4f}")

    return metrics


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=14)
    parser.add_argument('--scenarios', type=int, default=200)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/domain_separation')
    args = parser.parse_args()

    validate_domain_separation(
        case_id=args.case,
        num_scenarios=args.scenarios,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
