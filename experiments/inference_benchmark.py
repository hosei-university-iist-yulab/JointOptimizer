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
Inference Latency & Scalability Benchmark (V2)

Addresses Q4.1, Q2.9: Measure inference latency across model sizes
and IEEE case sizes. Reports mean, p50, p95, p99 latency.

Expected: JointOptimizer inference <10ms on IEEE 118,
acceptable for real-time control (50ms cycle).
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import JointOptimizer
from src.baselines import (
    SequentialOPFQoS,
    MLPJoint,
    GNNOnly,
    VanillaTransformer,
    TransformerNoCoupling,
    HeterogeneousGNN,
    DeepOPF,
)
from src.data import create_dataloaders, DelayConfig
from src.utils.statistical_tests import restrict_gpus


MODEL_CONFIGS = {
    'JointOptimizer': lambda nb, ng, cfg, lm=None: JointOptimizer(
        n_generators=ng, energy_input_dim=5, comm_input_dim=3,
        embed_dim=cfg['embed_dim'], hidden_dim=cfg['hidden_dim'],
        num_heads=cfg['num_heads'], gnn_layers=cfg['gnn_layers'],
        k_init_scale=0.1, adaptive_gamma=True, lambda_min_0=lm,
    ),
    'B2_MLPJoint': lambda nb, ng, cfg, lm=None: MLPJoint(
        n_buses=nb, n_generators=ng, hidden_dim=cfg['hidden_dim'],
        k_init_scale=0.1, lambda_min_0=lm,
    ),
    'B3_GNNOnly': lambda nb, ng, cfg, lm=None: GNNOnly(
        n_buses=nb, n_generators=ng, hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['gnn_layers'], k_init_scale=0.1, lambda_min_0=lm,
    ),
    'B6_VanillaTransformer': lambda nb, ng, cfg, lm=None: VanillaTransformer(
        n_buses=nb, n_generators=ng, embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'], k_init_scale=0.1, lambda_min_0=lm,
    ),
    'B8_HeterogeneousGNN': lambda nb, ng, cfg, lm=None: HeterogeneousGNN(
        n_buses=nb, n_generators=ng, hidden_dim=cfg['hidden_dim'],
        num_layers=4, k_init_scale=0.1, lambda_min_0=lm,
    ),
    'B9_DeepOPF': lambda nb, ng, cfg, lm=None: DeepOPF(
        n_buses=nb, n_generators=ng, hidden_dim=400,
        num_layers=3, k_init_scale=0.1, lambda_min_0=lm,
    ),
}

GRAPH_MODELS = {'JointOptimizer', 'B3_GNNOnly'}


def measure_latency(
    model: torch.nn.Module,
    model_name: str,
    batch,
    device: torch.device,
    impedance_matrix: torch.Tensor,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> Dict:
    """Measure inference latency for a single model."""
    model.eval()

    energy_x = batch['energy_x'].to(device)
    comm_x = batch['comm_x'].to(device)
    tau = batch['tau'].to(device)
    tau_max = batch['tau_max'].to(device)
    lambda_min_0 = batch['lambda_min_0'].to(device)
    energy_edge_index = batch['energy_edge_index'].to(device)
    comm_edge_index = batch['comm_edge_index'].to(device)

    batch_size = energy_x.shape[0]
    n_nodes = energy_x.shape[1]

    is_graph_model = model_name in GRAPH_MODELS

    def run_forward():
        with torch.no_grad():
            if is_graph_model:
                energy_x_flat = energy_x.reshape(-1, energy_x.shape[-1])
                comm_x_flat = comm_x.reshape(-1, comm_x.shape[-1])
                batch_tensor = torch.arange(
                    batch_size, device=device
                ).repeat_interleave(n_nodes)

                return model(
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
                return model(
                    energy_x=energy_x,
                    comm_x=comm_x,
                    tau=tau,
                    tau_max=tau_max[0],
                    lambda_min_0=lambda_min_0,
                )

    # Warmup
    for _ in range(n_warmup):
        run_forward()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(n_measure):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        run_forward()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    n_params = sum(p.numel() for p in model.parameters())

    # Memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        run_forward()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0.0

    return {
        'model_name': model_name,
        'n_parameters': n_params,
        'latency_mean_ms': float(latencies.mean()),
        'latency_std_ms': float(latencies.std()),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'memory_mb': memory_mb,
        'throughput_samples_per_sec': float(batch_size * 1000 / latencies.mean()),
    }


def run_benchmark(
    case_ids: List[int] = None,
    batch_size: int = 32,
    output_dir: str = 'results/inference_benchmark',
):
    """Run inference benchmark across models and case sizes."""
    if case_ids is None:
        case_ids = [14, 39, 57, 118]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("INFERENCE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Cases: {case_ids}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")

    config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'gnn_layers': 3,
    }

    all_results = []

    for case_id in case_ids:
        print(f"\n--- IEEE {case_id} ---")

        delay_config = DelayConfig(distribution='lognormal', mean_ms=50.0, std_ms=20.0)
        train_loader, _, _ = create_dataloaders(
            case_id=case_id, num_scenarios=100,
            delay_config=delay_config, batch_size=batch_size,
        )

        dataset = train_loader.dataset.dataset
        base_case = dataset.get_base_case()
        n_buses = base_case['n_buses']
        n_gen = base_case['n_generators']
        lambda_min_0 = base_case.get('lambda_min', None)
        impedance_matrix = dataset.get_impedance_matrix()
        if impedance_matrix is not None:
            impedance_matrix = impedance_matrix.to(device)

        # Get a single batch
        batch = next(iter(train_loader))

        for model_name, model_fn in MODEL_CONFIGS.items():
            print(f"  {model_name}...", end=" ", flush=True)

            model = model_fn(n_buses, n_gen, config, lambda_min_0).to(device)
            result = measure_latency(
                model, model_name, batch, device,
                impedance_matrix, n_warmup=10, n_measure=100,
            )
            result['case_id'] = case_id
            result['n_buses'] = n_buses
            result['n_generators'] = n_gen
            all_results.append(result)

            print(f"mean={result['latency_mean_ms']:.2f}ms, "
                  f"p95={result['latency_p95_ms']:.2f}ms, "
                  f"params={result['n_parameters']:,}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    output_dict = {
        'case_ids': case_ids,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }

    json_path = f"{output_dir}/inference_benchmark.json"
    with open(json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Case':<8} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Params':<12}")
    print("-" * 69)
    for r in all_results:
        print(f"{r['model_name']:<25} {r['case_id']:<8} "
              f"{r['latency_mean_ms']:>8.2f}    {r['latency_p95_ms']:>8.2f}    "
              f"{r['n_parameters']:>10,}")


def main():
    restrict_gpus()
    parser = argparse.ArgumentParser(
        description='Inference Latency Benchmark'
    )
    parser.add_argument('--cases', type=str, default='14,39,57,118',
                        help='Comma-separated case IDs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=str, default='results/inference_benchmark')
    args = parser.parse_args()

    case_ids = [int(c.strip()) for c in args.cases.split(',')]

    run_benchmark(
        case_ids=case_ids,
        batch_size=args.batch_size,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
