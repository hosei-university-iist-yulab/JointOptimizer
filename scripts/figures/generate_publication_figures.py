#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/15/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Generate All Publication-Ready Figures for IEEE Trans. Smart Grid

Loads results from results/full_sweep/ subdirectories:
- baselines/case{X}/comparison_results.json
- theorem1/case{X}/theorem1_all_models.json
- ablation/case{X}/ablation_results_v2.json
- stress_test/case{X}/stress_test_case{X}.json
- inference_benchmark/inference_benchmark.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# IEEE publication style — bold, high-contrast
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'font.weight': 'bold',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'patch.linewidth': 1.0,
})

# Force bold tick labels on every figure save (font.weight doesn't always
# cascade to tick labels with serif fonts)
_orig_savefig = plt.savefig

def _savefig_bold_ticks(*args, **kwargs):
    fig = plt.gcf()
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    return _orig_savefig(*args, **kwargs)

plt.savefig = _savefig_bold_ticks

COLORS = {
    'proposed': '#2E86AB',
    'sequential': '#F18F01',
    'mlp': '#A23B72',
    'gnn': '#4CAF50',
    'lstm': '#9C27B0',
    'cnn': '#FF5722',
    'vanilla': '#607D8B',
    'ablation': '#E91E63',
}

MODEL_COLORS = {
    'JointOptimizer': COLORS['proposed'],
    'B1_SequentialOPFQoS': COLORS['sequential'],
    'B2_MLPJoint': COLORS['mlp'],
    'B3_GNNOnly': COLORS['gnn'],
    'B4_LSTMJoint': COLORS['lstm'],
    'B5_CNNJoint': COLORS['cnn'],
    'B6_VanillaTransformer': COLORS['vanilla'],
    'B7_TransformerNoCoupling': COLORS['ablation'],
}

MODEL_SHORT = {
    'JointOptimizer': 'Ours',
    'B1_SequentialOPFQoS': 'B1',
    'B2_MLPJoint': 'B2',
    'B3_GNNOnly': 'B3',
    'B4_LSTMJoint': 'B4',
    'B5_CNNJoint': 'B5',
    'B6_VanillaTransformer': 'B6',
    'B7_TransformerNoCoupling': 'B7',
}

MODEL_ORDER = [
    'JointOptimizer', 'B1_SequentialOPFQoS', 'B2_MLPJoint', 'B3_GNNOnly',
    'B4_LSTMJoint', 'B5_CNNJoint', 'B6_VanillaTransformer', 'B7_TransformerNoCoupling',
]

IEEE_CASES = [14, 30, 39, 57, 118]


def load_json(path):
    """Load JSON file if exists."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_baselines(results_dir, case):
    """Load baselines comparison results for a case."""
    return load_json(f'{results_dir}/baselines/case{case}/comparison_results.json')



def generate_k_learning_comparison(results_dir, output_dir, cases):
    """Fig 2: K learning dynamics across methods."""
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(cases))
    width = 0.1

    # Collect K values for each model across cases
    model_k_values = {}
    for case in cases:
        data = get_baselines(results_dir, case)
        if not data:
            continue
        agg = data['aggregated_results']
        for model in MODEL_ORDER:
            if model not in agg:
                continue
            if model not in model_k_values:
                model_k_values[model] = []
            model_k_values[model].append(agg[model]['K_mean']['mean'])

    for i, (model, k_vals) in enumerate(model_k_values.items()):
        offset = (i - len(model_k_values)/2) * width
        color = MODEL_COLORS.get(model, '#888888')
        ax.bar(x[:len(k_vals)] + offset, k_vals, width,
               label=MODEL_SHORT.get(model, model),
               color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('IEEE Test Case')
    ax.set_ylabel('Coupling Constant K')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}-Bus' for c in cases])
    ax.legend(loc='upper right', ncol=3, fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_k_learning_comparison.pdf')
    plt.close()
    print(f"  Saved: fig_k_learning_comparison.pdf")



def generate_radar_chart_all_baselines(results_dir, output_dir, case=39):
    """Fig 8: Radar chart comparing JointOptimizer vs ALL 7 baselines.

    Uses percentile-rank normalization to spread models across the [0,1] range
    on each axis, avoiding the compression that min-max causes when all models
    share the same stability rate.  Axes are chosen to be *meaningful*:
      - Stability Margin (higher = better)
      - Margin Consistency (lower std = better → inverted)
      - Training Speed (lower time = better → inverted)
      - Parameter Efficiency (higher margin-per-param = better)
      - K Compactness (lower K = better → inverted)
    """
    from math import pi

    # Aggregate across MULTIPLE cases for a fairer comparison
    multi_cases = [14, 39, 57, 118]
    method_metrics = {}  # {method: {metric: [values across cases]}}

    for c in multi_cases:
        data = get_baselines(results_dir, c)
        if not data:
            continue
        agg = data['aggregated_results']
        for m in MODEL_ORDER:
            if m not in agg:
                continue
            if m not in method_metrics:
                method_metrics[m] = {'margin': [], 'margin_std': [], 'time': [],
                                     'rpp': [], 'k_mean': []}
            d = agg[m]
            method_metrics[m]['margin'].append(d['margin']['mean'])
            method_metrics[m]['margin_std'].append(d['margin'].get('std', 0))
            method_metrics[m]['time'].append(d['training_time_mean'])
            method_metrics[m]['rpp'].append(
                d['margin']['mean'] / max(d['num_parameters'], 1) * 1e6)
            method_metrics[m]['k_mean'].append(d['K_mean']['mean'])

    if not method_metrics:
        # Fallback: single case
        data = get_baselines(results_dir, case)
        if not data:
            print(f"  [SKIP] IEEE {case} results not found")
            return
        agg = data['aggregated_results']
        for m in MODEL_ORDER:
            if m not in agg:
                continue
            d = agg[m]
            method_metrics[m] = {
                'margin': [d['margin']['mean']],
                'margin_std': [d['margin'].get('std', 0)],
                'time': [d['training_time_mean']],
                'rpp': [d['margin']['mean'] / max(d['num_parameters'], 1) * 1e6],
                'k_mean': [d['K_mean']['mean']],
            }

    methods = [m for m in MODEL_ORDER if m in method_metrics]
    if len(methods) < 2:
        print("  [SKIP] Not enough methods for radar chart")
        return

    # Compute averages across cases
    avg = {}
    for m in methods:
        avg[m] = {k: np.mean(v) for k, v in method_metrics[m].items()}

    categories = [
        'Stability\nMargin',
        'Consistency',
        'Training\nSpeed',
        'Param\nEfficiency',
        'K Compactness',
    ]
    N = len(categories)

    # Percentile-rank normalization: rank models and spread evenly in [0.30, 1.0]
    # Floor at 0.30 so even the worst-ranked model is clearly visible on the radar
    def rank_normalize(values_dict, invert=False):
        """Return dict {method: normalized_score} using rank-based scaling."""
        items = sorted(values_dict.items(), key=lambda x: x[1], reverse=(not invert))
        n = len(items)
        result = {}
        for rank, (method, _) in enumerate(items):
            result[method] = 0.30 + 0.70 * (1.0 - rank / max(n - 1, 1))
        return result

    margin_scores = rank_normalize({m: avg[m]['margin'] for m in methods})
    consistency_scores = rank_normalize(
        {m: 1.0 / (avg[m]['margin_std'] + 1e-8) for m in methods})
    speed_scores = rank_normalize({m: avg[m]['time'] for m in methods}, invert=True)
    efficiency_scores = rank_normalize({m: avg[m]['rpp'] for m in methods})
    k_scores = rank_normalize({m: avg[m]['k_mean'] for m in methods}, invert=True)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*']

    for i, method in enumerate(methods):
        values = [
            margin_scores[method],
            consistency_scores[method],
            speed_scores[method],
            efficiency_scores[method],
            k_scores[method],
        ]
        values += values[:1]

        is_ours = method == 'JointOptimizer'
        lw = 3.5 if is_ours else 1.8
        ms = 11 if is_ours else 6
        alpha_fill = 0.30 if is_ours else 0.03
        zorder = 10 if is_ours else 2

        ax.plot(angles, values, marker=markers[i % len(markers)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=lw, markersize=ms, zorder=zorder,
                label=MODEL_SHORT[method], color=MODEL_COLORS[method])
        ax.fill(angles, values, alpha=alpha_fill, color=MODEL_COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_rticks([0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(['0.3', '0.5', '0.7', '0.9'],
                        fontsize=8, fontweight='bold')
    ax.spines['polar'].set_linewidth(1.5)

    legend = ax.legend(loc='upper left', bbox_to_anchor=(-0.15, -0.08),
                       fontsize=9, framealpha=0.95,
                       prop={'weight': 'bold', 'size': 9}, ncol=4,
                       edgecolor='black', columnspacing=0.8)
    ax.set_title(f'Multi-Case Baseline Comparison (avg. IEEE {", ".join(str(c) for c in multi_cases)}-Bus)',
                 y=1.08, fontsize=12, fontweight='bold')

    plt.savefig(f'{output_dir}/fig_radar_all_baselines.pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: fig_radar_all_baselines.pdf")


def _load_model_and_data(case=39, results_dir='results/full_sweep', num_scenarios=10):
    """
    Load a trained JointOptimizer from per-case checkpoint and create dataset.

    Searches for real trained checkpoints at:
      1. {results_dir}/train/case{case}/best.pt  (primary — has config + trained weights)
      2. {results_dir}/train/case{case}/latest.pt (fallback)

    Returns (model, dataset, sample) or (None, None, None) if loading fails.
    """
    import sys
    import os

    # Try to import from project root
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import torch
        from src.models.joint_optimizer import JointOptimizer
        from src.data.dataset import PowerGridDataset
    except ImportError as e:
        print(f"  [SKIP] Cannot import model/data modules: {e}")
        return None, None, None

    # Search for per-case trained checkpoints (real data, attention enabled)
    ckpt_paths = [
        f'{results_dir}/train/case{case}/best.pt',
        f'{results_dir}/train/case{case}/latest.pt',
    ]
    ckpt = None
    ckpt_path = None
    for p in ckpt_paths:
        if Path(p).exists():
            try:
                ckpt = torch.load(p, map_location='cpu', weights_only=False)
                ckpt_path = p
                print(f"  Loaded checkpoint: {p}")
                break
            except Exception:
                continue

    if ckpt is None:
        print(f"  [SKIP] No trained checkpoint found for case {case}")
        print(f"  Searched: {ckpt_paths}")
        return None, None, None

    # Extract config from checkpoint
    config = ckpt.get('config', {})
    if not config:
        print(f"  [SKIP] Checkpoint has no config: {ckpt_path}")
        return None, None, None

    # Create dataset to get graph structure and impedance
    try:
        dataset = PowerGridDataset(case_id=case, num_scenarios=num_scenarios, seed=42)
    except Exception as e:
        print(f"  [SKIP] Cannot create dataset for case {case}: {e}")
        return None, None, None

    # Build model from checkpoint config
    n_gen = config.get('n_generators', dataset.get_base_case()['n_generators'])
    try:
        model = JointOptimizer(
            n_generators=n_gen,
            energy_input_dim=config.get('energy_input_dim', 5),
            comm_input_dim=config.get('comm_input_dim', 3),
            embed_dim=config.get('embed_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_heads=config.get('num_heads', 8),
            gnn_layers=config.get('gnn_layers', 3),
            decoder_layers=config.get('decoder_layers', 2),
            dropout=0.0,
            physics_gamma=config.get('physics_gamma', 1.0),
            k_init_scale=config.get('k_init_scale', 0.1),
            learnable_k=config.get('learnable_k', True),
            adaptive_gamma=config.get('adaptive_gamma', False),
            use_physics_mask=config.get('use_physics_mask', True),
            use_causal_mask=config.get('use_causal_mask', True),
            use_cross_attention=config.get('use_cross_attention', True),
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        epoch = ckpt.get('epoch', '?')
        print(f"  Model: n_gen={n_gen}, embed={config.get('embed_dim', 128)}, "
              f"cross_attn={config.get('use_cross_attention')}, epoch={epoch}")
    except Exception as e:
        print(f"  [SKIP] Cannot create/load model: {e}")
        return None, None, None

    sample = dataset[0]
    return model, dataset, sample


def generate_attention_maps(output_dir, case=39, results_dir='results/full_sweep'):
    """Fig 11: 4-panel attention visualization — causal, cross-domain, physics mask, combined."""
    import torch

    model, dataset, sample = _load_model_and_data(case, results_dir)
    if model is None:
        return

    base = dataset.get_base_case()
    impedance = dataset.get_impedance_matrix()
    n_buses = base['n_buses']

    # GNN expects [N, features] for node inputs (no batch dim);
    # tau and lambda_min_0 need batch dim [1, n_gen] and [1]
    with torch.no_grad():
        out = model(
            energy_x=sample['energy_x'],
            energy_edge_index=sample['energy_edge_index'],
            comm_x=sample['comm_x'],
            comm_edge_index=sample['comm_edge_index'],
            tau=sample['tau'].unsqueeze(0),
            tau_max=sample['tau_max'],
            lambda_min_0=sample['lambda_min_0'].unsqueeze(0),
            impedance_matrix=impedance,
        )

    attn_info = out.get('attn_info')
    if attn_info is None:
        print("  [SKIP] No attention info (model may have skipped attention)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Causal self-attention
    causal = attn_info.get('causal_attn')
    if causal is not None:
        if causal.dim() == 4:
            causal_np = causal[0].mean(dim=0).cpu().numpy()  # avg over heads
        elif causal.dim() == 3:
            causal_np = causal[0].cpu().numpy()
        else:
            causal_np = causal.cpu().numpy()
        im1 = axes[0, 0].imshow(causal_np, cmap='Blues', aspect='auto')
        axes[0, 0].set_title('(a) Causal Self-Attention (Energy)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Key (Source Bus)')
        axes[0, 0].set_ylabel('Query (Target Bus)')
        plt.colorbar(im1, ax=axes[0, 0], label='Attention Weight')

    # Panel 2: Cross-domain attention
    cross = attn_info.get('cross_attn')
    if cross is not None:
        if cross.dim() == 4:
            cross_np = cross[0].mean(dim=0).cpu().numpy()
        elif cross.dim() == 3:
            cross_np = cross[0].cpu().numpy()
        else:
            cross_np = cross.cpu().numpy()
        im2 = axes[0, 1].imshow(cross_np, cmap='Oranges', aspect='auto')
        axes[0, 1].set_title(r'(b) Cross-Domain Attention (Energy $\rightarrow$ Comm)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Communication Key')
        axes[0, 1].set_ylabel('Energy Query')
        plt.colorbar(im2, ax=axes[0, 1], label='Attention Weight')

    # Panel 3: Physics mask (from impedance)
    if impedance is not None:
        Z = impedance.cpu().numpy()
        Z_max = Z.max() + 1e-8
        M_phys = -1.0 * Z / Z_max
        im3 = axes[1, 0].imshow(M_phys, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title(r'(c) Physics Mask $M_{\mathrm{phys}} = -\gamma \cdot Z/Z_{\max}$',
                             fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Bus j')
        axes[1, 0].set_ylabel('Bus i')
        plt.colorbar(im3, ax=axes[1, 0], label='Mask Value')
    else:
        axes[1, 0].text(0.5, 0.5, 'No impedance data', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('(c) Physics Mask', fontsize=11, fontweight='bold')

    # Panel 4: Combined effective attention (cross * softmax(physics))
    if cross is not None and impedance is not None:
        # Effective attention = cross-domain weights (already include physics mask)
        # Show the per-head variance to highlight specialization
        if attn_info['cross_attn'].dim() == 4:
            heads = attn_info['cross_attn'][0].cpu().numpy()  # [H, N, N]
            head_variance = np.var(heads, axis=0)  # [N, N]
            im4 = axes[1, 1].imshow(head_variance, cmap='magma', aspect='auto')
            axes[1, 1].set_title('(d) Head Specialization (Variance Across Heads)',
                                 fontsize=11, fontweight='bold')
            plt.colorbar(im4, ax=axes[1, 1], label='Variance')
        else:
            axes[1, 1].set_visible(False)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('(d) Head Specialization', fontsize=11, fontweight='bold')

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=8)

    fig.suptitle(f'Attention Mechanism Visualization (IEEE {case}-Bus)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_attention_maps.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_attention_maps.pdf")


def generate_per_head_attention(output_dir, case=39, results_dir='results/full_sweep'):
    """Fig 12: Individual attention head patterns for cross-domain attention."""
    import torch

    model, dataset, sample = _load_model_and_data(case, results_dir)
    if model is None:
        return

    impedance = dataset.get_impedance_matrix()

    with torch.no_grad():
        out = model(
            energy_x=sample['energy_x'],
            energy_edge_index=sample['energy_edge_index'],
            comm_x=sample['comm_x'],
            comm_edge_index=sample['comm_edge_index'],
            tau=sample['tau'].unsqueeze(0),
            tau_max=sample['tau_max'],
            lambda_min_0=sample['lambda_min_0'].unsqueeze(0),
            impedance_matrix=impedance,
        )

    attn_info = out.get('attn_info')
    if attn_info is None:
        print("  [SKIP] No attention info")
        return

    cross = attn_info.get('cross_attn')
    if cross is None or cross.dim() < 4:
        print("  [SKIP] No multi-head cross-attention data")
        return

    heads = cross[0].cpu().numpy()  # [H, N, N]
    n_heads = heads.shape[0]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else np.array([[axes]])

    for h in range(n_heads):
        r, c = divmod(h, cols)
        ax = axes[r][c]
        im = ax.imshow(heads[h], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h+1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Comm Key' if r == rows - 1 else '')
        ax.set_ylabel('Energy Query' if c == 0 else '')
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide empty
    for h in range(n_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'Cross-Domain Attention Per Head (IEEE {case}-Bus)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_attention_per_head.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_attention_per_head.pdf")


def generate_embedding_space(output_dir, results_dir='results/full_sweep'):
    """Fig 13: PCA + cosine similarity for all IEEE cases (2x4 grid)."""
    import torch
    from numpy.linalg import norm

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [SKIP] sklearn not available for PCA")
        return

    vis_cases = [14, 39, 57, 118]
    n_cases = len(vis_cases)
    case_colors = ['#2E86C1', '#27AE60', '#E67E22', '#8E44AD']

    fig, axes = plt.subplots(2, n_cases, figsize=(22, 9),
                             gridspec_kw={'height_ratios': [1.0, 1.0],
                                          'hspace': 0.22, 'wspace': 0.22,
                                          'left': 0.05, 'right': 0.97,
                                          'top': 0.92, 'bottom': 0.06})

    for col, case in enumerate(vis_cases):
        model, dataset, sample = _load_model_and_data(case, results_dir)
        if model is None:
            for r in range(2):
                axes[r, col].text(0.5, 0.5, f'Case {case}\nNo checkpoint',
                                  ha='center', va='center',
                                  transform=axes[r, col].transAxes,
                                  fontsize=12, fontweight='bold')
                axes[r, col].set_axis_off()
            continue

        impedance = dataset.get_impedance_matrix()
        out = _run_model_inference(model, sample, impedance)
        h_E = out['h_E'][0].cpu().numpy()
        h_I = out['h_I'][0].cpu().numpy()
        n = len(h_E)

        # --- Row 0: PCA scatter ---
        h_all = np.vstack([h_E, h_I])
        pca = PCA(n_components=2)
        h_2d = pca.fit_transform(h_all)

        ax = axes[0, col]
        # Scatter with subtle edge for readability
        ax.scatter(h_2d[:n, 0], h_2d[:n, 1], c='#1565C0', s=90, alpha=0.85,
                   label='Energy Domain', marker='o', edgecolors='white',
                   linewidths=0.4, zorder=3)
        ax.scatter(h_2d[n:, 0], h_2d[n:, 1], c='#E65100', s=90, alpha=0.85,
                   label='Communication Domain', marker='^', edgecolors='white',
                   linewidths=0.4, zorder=3)

        # Node-pair connections
        for i in range(n):
            ax.plot([h_2d[i, 0], h_2d[n+i, 0]], [h_2d[i, 1], h_2d[n+i, 1]],
                    color='#BBBBBB', alpha=0.20, linewidth=0.5, zorder=1)

        # Convex hulls
        try:
            from scipy.spatial import ConvexHull
            for pts, color in [(h_2d[:n], '#1565C0'), (h_2d[n:], '#E65100')]:
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            alpha=0.08, color=color, zorder=1)
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, linewidth=1.2, alpha=0.35, zorder=2)
        except (ImportError, Exception):
            pass

        # Centroids + separation arrow
        e_ctr = h_2d[:n].mean(axis=0)
        c_ctr = h_2d[n:].mean(axis=0)
        sep = np.linalg.norm(e_ctr - c_ctr)
        ax.scatter(*e_ctr, c='#1565C0', s=250, marker='*', edgecolors='none', zorder=5)
        ax.scatter(*c_ctr, c='#E65100', s=250, marker='*', edgecolors='none', zorder=5)
        ax.annotate('', xy=c_ctr, xytext=e_ctr,
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
        mid = (e_ctr + c_ctr) / 2
        ax.text(mid[0], mid[1] + 0.4, f'd={sep:.1f}', ha='center', fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='black', alpha=0.9))

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)',
                      fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('(a)  PC2', fontsize=11, fontweight='bold')
        leg = ax.legend(fontsize=9, framealpha=0.95, edgecolor='black',
                        markerscale=1.2, loc='best')
        for txt in leg.get_texts():
            txt.set_fontweight('bold')
            txt.set_color('black')
        ax.set_title(f'IEEE {case}-Bus', fontsize=12, fontweight='bold',
                     color='black',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=case_colors[col], alpha=0.15,
                               edgecolor=case_colors[col], linewidth=1.5))

        # --- Row 1: Cosine similarity heatmap ---
        h_E_n = h_E / (norm(h_E, axis=1, keepdims=True) + 1e-8)
        h_I_n = h_I / (norm(h_I, axis=1, keepdims=True) + 1e-8)
        cos_sim = h_E_n @ h_I_n.T

        ax2 = axes[1, col]
        vabs = max(abs(cos_sim.min()), abs(cos_sim.max()), 0.01)
        im = ax2.imshow(cos_sim, cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                        aspect='auto', interpolation='nearest')
        ax2.set_xlabel('Comm. Node', fontsize=10, fontweight='bold')
        if col == 0:
            ax2.set_ylabel('(b)  Energy Node', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Stats annotation
        diag_sim = np.diag(cos_sim)
        off_diag = (cos_sim.sum() - diag_sim.sum()) / max(n*n - n, 1)
        ax2.text(0.03, 0.97,
                 f'diag: {diag_sim.mean():.3f}\noff: {off_diag:.3f}',
                 transform=ax2.transAxes, fontsize=8, fontweight='bold',
                 va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', alpha=0.92))

    fig.suptitle('Domain Separation Analysis (Observation 1)',
                 fontsize=15, fontweight='bold')
    plt.savefig(f'{output_dir}/fig_embedding_space.pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: fig_embedding_space.pdf")


def _run_model_inference(model, sample, impedance):
    """Run model forward pass with correct tensor shapes. Returns output dict."""
    import torch
    with torch.no_grad():
        out = model(
            energy_x=sample['energy_x'],
            energy_edge_index=sample['energy_edge_index'],
            comm_x=sample['comm_x'],
            comm_edge_index=sample['comm_edge_index'],
            tau=sample['tau'].unsqueeze(0),
            tau_max=sample['tau_max'],
            lambda_min_0=sample['lambda_min_0'].unsqueeze(0),
            impedance_matrix=impedance,
        )
    return out


def generate_multi_case_attention(output_dir, results_dir='results/full_sweep'):
    """Fig 14: Cross-domain attention maps compared across IEEE cases (14, 39, 57, 118)."""
    import torch

    vis_cases = [14, 39, 57, 118]
    fig, axes = plt.subplots(2, len(vis_cases), figsize=(5 * len(vis_cases), 9))

    for col, case in enumerate(vis_cases):
        model, dataset, sample = _load_model_and_data(case, results_dir)
        if model is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f'Case {case}\nNo checkpoint',
                                    ha='center', va='center', transform=axes[row, col].transAxes)
            continue

        impedance = dataset.get_impedance_matrix()
        out = _run_model_inference(model, sample, impedance)
        attn_info = out.get('attn_info')

        if attn_info is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f'Case {case}\nNo attention', ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        # Row 0: Cross-domain attention (avg over heads)
        cross = attn_info.get('cross_attn')
        if cross is not None:
            if cross.dim() == 4:
                cross_np = cross[0].mean(dim=0).cpu().numpy()
            else:
                cross_np = cross[0].cpu().numpy() if cross.dim() == 3 else cross.cpu().numpy()
            im = axes[0, col].imshow(cross_np, cmap='Oranges', aspect='auto')
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        axes[0, col].set_title(f'IEEE {case}-Bus', fontsize=12, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel('Cross-Domain Attention\nEnergy Query', fontsize=10, fontweight='bold')

        # Row 1: Causal self-attention (avg over heads)
        causal = attn_info.get('causal_attn')
        if causal is not None:
            if causal.dim() == 4:
                causal_np = causal[0].mean(dim=0).cpu().numpy()
            else:
                causal_np = causal[0].cpu().numpy() if causal.dim() == 3 else causal.cpu().numpy()
            im = axes[1, col].imshow(causal_np, cmap='Blues', aspect='auto')
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        if col == 0:
            axes[1, col].set_ylabel('Causal Self-Attention\nEnergy Query', fontsize=10, fontweight='bold')

    fig.suptitle('Attention Patterns Across Grid Sizes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_multi_case_attention.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_multi_case_attention.pdf")


def generate_attention_distribution(output_dir, results_dir='results/full_sweep'):
    """Fig 15: Cross-domain attention weight distribution per head — 2x2, filled KDE curves."""
    import torch
    from scipy.stats import gaussian_kde

    vis_cases = [14, 39, 57, 118]
    case_colors = ['#2E86C1', '#27AE60', '#E67E22', '#8E44AD']
    head_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6',
                   '#F39C12', '#1ABC9C', '#E67E22', '#34495E']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'hspace': 0.25, 'wspace': 0.20,
                                          'left': 0.07, 'right': 0.97,
                                          'top': 0.92, 'bottom': 0.06})
    axes = axes.flatten()

    for idx, case in enumerate(vis_cases):
        ax = axes[idx]
        model, dataset, sample = _load_model_and_data(case, results_dir)
        if model is None:
            ax.text(0.5, 0.5, f'Case {case}\nNo checkpoint',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, fontweight='bold')
            continue

        impedance = dataset.get_impedance_matrix()
        out = _run_model_inference(model, sample, impedance)
        attn_info = out.get('attn_info')
        if attn_info is None:
            ax.text(0.5, 0.5, 'No attention', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
            continue

        cross = attn_info.get('cross_attn')
        if cross is None or cross.dim() < 4:
            ax.text(0.5, 0.5, 'No multi-head data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
            continue

        cross_heads = cross[0].cpu().numpy()  # [H, N, N]
        n_heads = cross_heads.shape[0]
        colors = head_colors[:n_heads]

        for h in range(n_heads):
            weights = cross_heads[h].flatten()
            weights = weights[weights > 1e-10]  # skip near-zero for cleaner KDE
            if len(weights) < 10:
                continue
            entropy = -np.sum(cross_heads[h].flatten() *
                              np.log(cross_heads[h].flatten() + 1e-12)) / cross_heads[h].size

            # KDE for smooth filled curve
            kde = gaussian_kde(weights, bw_method=0.3)
            x_range = np.linspace(weights.min(), weights.max(), 300)
            density = kde(x_range)

            ax.fill_between(x_range, density, alpha=0.15, color=colors[h])
            ax.plot(x_range, density, color=colors[h], linewidth=2.2, alpha=0.95,
                    label=f'H{h+1} (H={entropy:.3f})')

        ax.set_xlabel('Attention Weight', fontsize=10, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax.set_xlim(0, None)
        ax.set_title(f'IEEE {case}-Bus', fontsize=12, fontweight='bold',
                     color='black',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=case_colors[idx], alpha=0.15,
                               edgecolor=case_colors[idx], linewidth=1.5))
        leg = ax.legend(fontsize=8, ncol=2, loc='upper right',
                        framealpha=0.95, edgecolor='black')
        for txt in leg.get_texts():
            txt.set_fontweight('bold')
            txt.set_color('black')

    fig.suptitle('Cross-Domain Attention Weight Distribution per Head',
                 fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/fig_attention_distribution.pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: fig_attention_distribution.pdf")


def generate_physics_mask_overlay(output_dir, results_dir='results/full_sweep'):
    """Fig 16: Physics mask vs learned attention — 4x4 grid: 4 IEEE cases x 4 panels."""
    import torch

    vis_cases = [14, 39, 57, 118]
    fig, axes = plt.subplots(len(vis_cases), 4, figsize=(22, 5 * len(vis_cases)))
    col_titles = [
        r'Physics Mask $M_{\mathrm{phys}}$',
        'Physics-Only Attention\n(softmax of mask)',
        'Learned Attention\n(physics + data)',
        'Learned - Physics\n(data-driven correction)',
    ]

    for row, c in enumerate(vis_cases):
        model, dataset, sample = _load_model_and_data(c, results_dir)
        if model is None:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f'Case {c}\nNo checkpoint',
                                    ha='center', va='center', transform=axes[row, col].transAxes,
                                    fontsize=12, fontweight='bold')
            continue

        impedance = dataset.get_impedance_matrix()
        out = _run_model_inference(model, sample, impedance)
        attn_info = out.get('attn_info')

        if attn_info is None or impedance is None:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=12)
            continue

        cross = attn_info.get('cross_attn')
        if cross is None:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, 'No cross-attn', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=12)
            continue

        if cross.dim() == 4:
            learned_attn = cross[0].mean(dim=0).cpu().numpy()
        else:
            learned_attn = cross[0].cpu().numpy() if cross.dim() == 3 else cross.cpu().numpy()

        Z = impedance.cpu().numpy()
        Z_max = Z.max() + 1e-8
        physics_mask = -1.0 * Z / Z_max
        physics_soft = np.exp(physics_mask) / np.exp(physics_mask).sum(axis=1, keepdims=True)

        # Panel 1: Raw physics mask
        im1 = axes[row, 0].imshow(physics_mask, cmap='RdBu_r', aspect='auto',
                                   interpolation='nearest')
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.046)

        # Panel 2: Softmax of physics mask
        im2 = axes[row, 1].imshow(physics_soft, cmap='Oranges', aspect='auto',
                                   interpolation='nearest')
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046)

        # Panel 3: Learned attention
        im3 = axes[row, 2].imshow(learned_attn, cmap='Oranges', aspect='auto',
                                   interpolation='nearest')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046)

        # Panel 4: Difference
        n = min(learned_attn.shape[0], physics_soft.shape[0])
        diff = learned_attn[:n, :n] - physics_soft[:n, :n]
        vmax = max(abs(diff.min()), abs(diff.max()), 1e-8)
        im4 = axes[row, 3].imshow(diff, cmap='RdBu_r', aspect='auto',
                                   vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im4, ax=axes[row, 3], fraction=0.046)

        # Correlation annotation on last panel
        corr = np.corrcoef(physics_soft[:n, :n].flatten(),
                           learned_attn[:n, :n].flatten())[0, 1]
        axes[row, 3].text(0.02, 0.98, f'r = {corr:.3f}', transform=axes[row, 3].transAxes,
                          fontsize=11, fontweight='bold', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='black', alpha=0.95))

        # Row label
        axes[row, 0].set_ylabel(f'IEEE {c}-Bus', fontsize=13, fontweight='bold')

        # Axis labels
        for col in range(4):
            axes[row, col].set_xlabel('Bus j', fontsize=10, fontweight='bold')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12, fontweight='bold')

    fig.suptitle('Physics Mask vs Learned Attention Across IEEE Cases',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_physics_mask_overlay.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_physics_mask_overlay.pdf")


def generate_tsne_embeddings(output_dir, results_dir='results/full_sweep'):
    """Fig 17: t-SNE of dual-domain embeddings across multiple IEEE cases."""
    import torch
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [SKIP] sklearn not available for t-SNE")
        return

    vis_cases = [14, 39, 57, 118]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    # Clean, vivid, well-separated colors — no black edges
    energy_color = '#1565C0'   # strong blue
    comm_color = '#E65100'     # strong orange

    for idx, case in enumerate(vis_cases):
        model, dataset, sample = _load_model_and_data(case, results_dir, num_scenarios=20)
        ax = axes[idx]
        if model is None:
            ax.text(0.5, 0.5, f'Case {case}\nNo checkpoint',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        impedance = dataset.get_impedance_matrix()

        # Collect embeddings from multiple samples
        all_hE, all_hI = [], []
        n_samples = min(10, len(dataset))
        for i in range(n_samples):
            s = dataset[i]
            out = _run_model_inference(model, s, impedance)
            all_hE.append(out['h_E'][0].cpu().numpy())
            all_hI.append(out['h_I'][0].cpu().numpy())

        h_E = np.concatenate(all_hE, axis=0)  # [n_samples*N, d]
        h_I = np.concatenate(all_hI, axis=0)

        h_all = np.vstack([h_E, h_I])
        perplexity = min(30, len(h_all) // 4)
        if perplexity < 5:
            perplexity = 5

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        h_2d = tsne.fit_transform(h_all)
        n = len(h_E)

        # Clean scatter: no black edges, larger markers, filled only
        ax.scatter(h_2d[:n, 0], h_2d[:n, 1], c=energy_color, s=55, alpha=0.85,
                   label='Energy', marker='o', edgecolors='none', zorder=3)
        ax.scatter(h_2d[n:, 0], h_2d[n:, 1], c=comm_color, s=55, alpha=0.85,
                   label='Comm', marker='^', edgecolors='none', zorder=3)

        # Draw convex hulls around each cluster for visual clarity
        try:
            from scipy.spatial import ConvexHull
            for pts, color in [(h_2d[:n], energy_color), (h_2d[n:], comm_color)]:
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            alpha=0.12, color=color, zorder=1)
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=color, linewidth=1.5, alpha=0.5, zorder=2)
        except (ImportError, Exception):
            pass  # scipy not available or degenerate hull

        ax.set_title(f'IEEE {case}-Bus ({dataset.get_base_case()["n_buses"]} nodes)',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel('t-SNE 2', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, markerscale=1.5, framealpha=0.95, edgecolor='black',
                  loc='upper right')

        # Annotate inter-domain distance
        e_center = h_2d[:n].mean(axis=0)
        c_center = h_2d[n:].mean(axis=0)
        sep = np.linalg.norm(e_center - c_center)
        ax.text(0.02, 0.02, f'Centroid sep: {sep:.1f}', transform=ax.transAxes,
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', alpha=0.95))

    fig.suptitle('t-SNE of Dual-Domain Embeddings (Observation 1)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_tsne_embeddings.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_tsne_embeddings.pdf")


def generate_graph_attention_topology(output_dir, results_dir='results/full_sweep'):
    """Fig 18: Grid topology with attention weights as edge colors/widths — one PDF per case."""
    import torch
    try:
        import networkx as nx
    except ImportError:
        print("  [SKIP] networkx not available for graph topology")
        return
    from matplotlib.lines import Line2D

    vis_cases = [14, 39, 57, 118]

    for case in vis_cases:
        model, dataset, sample = _load_model_and_data(case, results_dir)
        if model is None:
            continue

        impedance = dataset.get_impedance_matrix()
        base = dataset.get_base_case()
        out = _run_model_inference(model, sample, impedance)
        attn_info = out.get('attn_info')

        edge_index = sample['energy_edge_index'].numpy()
        n_buses = base['n_buses']
        n_gen = base['n_generators']
        gen_buses = base.get('gen_buses', torch.arange(n_gen)).numpy()

        # Build graph
        G = nx.Graph()
        for i in range(n_buses):
            G.add_node(i)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:
                G.add_edge(int(src), int(dst))

        pos = nx.spring_layout(G, seed=42, k=2.0/np.sqrt(n_buses))

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Node styling — scale sizes for larger grids
        size_scale = max(1.0, 39 / n_buses)  # shrink nodes for large grids
        gen_size = int(350 * size_scale)
        load_size = int(120 * size_scale)
        label_size = max(4, int(7 * size_scale))
        node_colors = ['#C0392B' if i in gen_buses else '#1A5276' for i in range(n_buses)]
        node_sizes = [gen_size if i in gen_buses else load_size for i in range(n_buses)]

        # Panel 1: Grid topology with node types
        nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=node_colors,
                               node_size=node_sizes, edgecolors='black', linewidths=1.2)
        nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color='#555555', alpha=0.7, width=1.2)
        nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=label_size, font_color='white',
                                font_weight='bold')
        axes[0].set_title(f'(a) IEEE {case}-Bus Topology\n({n_gen} generators, {n_buses} buses)',
                          fontsize=13, fontweight='bold')
        legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
                               markersize=12, markeredgecolor='black', markeredgewidth=1,
                               label='Generator'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A5276',
                               markersize=10, markeredgecolor='black', markeredgewidth=1,
                               label='Load Bus')]
        axes[0].legend(handles=legend_elems, loc='lower left', fontsize=10,
                       framealpha=0.95, edgecolor='black')

        # Panel 2: Topology with cross-domain attention as edge heat
        if attn_info is not None:
            cross = attn_info.get('cross_attn')
            if cross is not None:
                if cross.dim() == 4:
                    attn_matrix = cross[0].mean(dim=0).cpu().numpy()
                else:
                    attn_matrix = cross[0].cpu().numpy() if cross.dim() == 3 else cross.cpu().numpy()

                edge_weights = []
                for u, v in G.edges():
                    if u < attn_matrix.shape[0] and v < attn_matrix.shape[1]:
                        w = (attn_matrix[u, v] + attn_matrix[v, u]) / 2
                    else:
                        w = 0
                    edge_weights.append(w)

                edge_weights = np.array(edge_weights)
                if edge_weights.max() > 0:
                    edge_weights_norm = edge_weights / edge_weights.max()
                else:
                    edge_weights_norm = edge_weights

                nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=node_colors,
                                       node_size=node_sizes, edgecolors='black', linewidths=1.2)
                nx.draw_networkx_edges(G, pos, ax=axes[1],
                                       edge_color=edge_weights_norm, edge_cmap=plt.cm.YlOrRd,
                                       width=1.5 + 4 * edge_weights_norm, alpha=0.95)
                nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=label_size, font_color='white',
                                        font_weight='bold')
                axes[1].set_title('(b) Cross-Domain Attention\n(edge color = attention strength)',
                                  fontsize=13, fontweight='bold')
                sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                           norm=plt.Normalize(vmin=0, vmax=edge_weights.max()))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046)
                cbar.set_label('Attention Weight', fontsize=11, fontweight='bold')
            else:
                axes[1].text(0.5, 0.5, 'No cross-attention', ha='center', va='center',
                            transform=axes[1].transAxes, fontsize=12, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No attention info', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12, fontweight='bold')

        # Panel 3: Impedance-based physics mask on topology
        if impedance is not None:
            Z = impedance.cpu().numpy()
            edge_impedances = []
            for u, v in G.edges():
                if u < Z.shape[0] and v < Z.shape[1]:
                    edge_impedances.append(Z[u, v])
                else:
                    edge_impedances.append(0)
            edge_impedances = np.array(edge_impedances)
            if edge_impedances.max() > 0:
                edge_imp_norm = edge_impedances / edge_impedances.max()
            else:
                edge_imp_norm = edge_impedances

            nx.draw_networkx_nodes(G, pos, ax=axes[2], node_color=node_colors,
                                   node_size=node_sizes, edgecolors='black', linewidths=1.2)
            nx.draw_networkx_edges(G, pos, ax=axes[2],
                                   edge_color=edge_imp_norm, edge_cmap=plt.cm.viridis,
                                   width=1.5 + 4 * edge_imp_norm, alpha=0.95)
            nx.draw_networkx_labels(G, pos, ax=axes[2], font_size=label_size, font_color='white',
                                    font_weight='bold')
            axes[2].set_title('(c) Physical Coupling\n(edge color = impedance)',
                              fontsize=13, fontweight='bold')
            sm2 = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                         norm=plt.Normalize(vmin=0, vmax=edge_impedances.max()))
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=axes[2], fraction=0.046)
            cbar2.set_label('Impedance |Z|', fontsize=11, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'No impedance data', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12, fontweight='bold')

        for ax in axes:
            ax.set_axis_off()

        fig.suptitle(f'Grid Topology and Attention Structure (IEEE {case}-Bus)',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig_graph_attention_topology_case{case}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig_graph_attention_topology_case{case}.pdf")


def generate_graph_attention_topology_combined(output_dir, results_dir='results/full_sweep'):
    """Fig 18b: All 4 IEEE cases in a single 4x3 grid (rows=cases, cols=panels)."""
    import torch
    try:
        import networkx as nx
    except ImportError:
        print("  [SKIP] networkx not available for graph topology")
        return
    from matplotlib.lines import Line2D

    vis_cases = [14, 39, 57, 118]
    fig, axes = plt.subplots(len(vis_cases), 3, figsize=(20, 6 * len(vis_cases)))
    col_titles = [
        'Grid Topology',
        'Cross-Domain Attention',
        'Physical Coupling (Impedance)',
    ]

    for row, case in enumerate(vis_cases):
        model, dataset, sample = _load_model_and_data(case, results_dir)
        if model is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, f'Case {case}\nNo checkpoint',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes,
                                    fontsize=12, fontweight='bold')
                axes[row, col].set_axis_off()
            continue

        impedance = dataset.get_impedance_matrix()
        base = dataset.get_base_case()
        out = _run_model_inference(model, sample, impedance)
        attn_info = out.get('attn_info')

        edge_index = sample['energy_edge_index'].numpy()
        n_buses = base['n_buses']
        n_gen = base['n_generators']
        gen_buses = base.get('gen_buses', torch.arange(n_gen)).numpy()

        G = nx.Graph()
        for i in range(n_buses):
            G.add_node(i)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:
                G.add_edge(int(src), int(dst))

        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n_buses))

        # Scale node/label sizes for grid size
        size_scale = max(1.0, 39 / n_buses)
        gen_size = int(350 * size_scale)
        load_size = int(120 * size_scale)
        label_size = max(4, int(7 * size_scale))
        node_colors = ['#C0392B' if i in gen_buses else '#1A5276' for i in range(n_buses)]
        node_sizes = [gen_size if i in gen_buses else load_size for i in range(n_buses)]

        # Col 0: Topology
        ax = axes[row, 0]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, edgecolors='black', linewidths=1.0)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#555555', alpha=0.7, width=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_size,
                                font_color='white', font_weight='bold')
        # Row label will be added via fig.text() after tight_layout
        if row == 0:
            legend_elems = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
                       markersize=10, markeredgecolor='black', label='Generator'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A5276',
                       markersize=8, markeredgecolor='black', label='Load Bus')]
            ax.legend(handles=legend_elems, loc='lower left', fontsize=8,
                      framealpha=0.95, edgecolor='black')

        # Col 1: Cross-domain attention
        ax = axes[row, 1]
        if attn_info is not None and attn_info.get('cross_attn') is not None:
            cross = attn_info['cross_attn']
            if cross.dim() == 4:
                attn_matrix = cross[0].mean(dim=0).cpu().numpy()
            else:
                attn_matrix = cross[0].cpu().numpy() if cross.dim() == 3 else cross.cpu().numpy()

            edge_weights = []
            for u, v in G.edges():
                if u < attn_matrix.shape[0] and v < attn_matrix.shape[1]:
                    w = (attn_matrix[u, v] + attn_matrix[v, u]) / 2
                else:
                    w = 0
                edge_weights.append(w)
            edge_weights = np.array(edge_weights)
            ew_norm = edge_weights / max(edge_weights.max(), 1e-8)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, edgecolors='black', linewidths=1.0)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ew_norm,
                                   edge_cmap=plt.cm.YlOrRd,
                                   width=1.0 + 3.5 * ew_norm, alpha=0.95)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_size,
                                    font_color='white', font_weight='bold')
            sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                       norm=plt.Normalize(vmin=0, vmax=edge_weights.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No attention', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')

        # Col 2: Physical coupling (impedance)
        ax = axes[row, 2]
        if impedance is not None:
            Z = impedance.cpu().numpy()
            edge_imp = []
            for u, v in G.edges():
                if u < Z.shape[0] and v < Z.shape[1]:
                    edge_imp.append(Z[u, v])
                else:
                    edge_imp.append(0)
            edge_imp = np.array(edge_imp)
            ei_norm = edge_imp / max(edge_imp.max(), 1e-8)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, edgecolors='black', linewidths=1.0)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ei_norm,
                                   edge_cmap=plt.cm.viridis,
                                   width=1.0 + 3.5 * ei_norm, alpha=0.95)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_size,
                                    font_color='white', font_weight='bold')
            sm2 = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                         norm=plt.Normalize(vmin=0, vmax=edge_imp.max()))
            sm2.set_array([])
            plt.colorbar(sm2, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No impedance', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')

        for col in range(3):
            axes[row, col].set_axis_off()

    # Column titles on top row
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold')

    fig.suptitle('Grid Topology and Attention Structure Across IEEE Cases',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    # Add prominent row labels on the left margin (survives set_axis_off)
    row_colors = ['#2E86C1', '#27AE60', '#E67E22', '#8E44AD']
    for row, case in enumerate(vis_cases):
        # Get vertical center of this row's axes in figure coords
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        # Draw colored banner text on the far left
        fig.text(0.01, y_center, f'IEEE\n{case}-Bus',
                 fontsize=15, fontweight='bold', color='white',
                 ha='center', va='center', rotation=90,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=row_colors[row],
                           edgecolor='black', linewidth=1.5))

    plt.savefig(f'{output_dir}/fig_graph_attention_topology_all.pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  Saved: fig_graph_attention_topology_all.pdf")



def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results', default='results/full_sweep',
                        help='Results directory (default: results/full_sweep)')
    parser.add_argument('--output', default='docs/figures/publication',
                        help='Output directory for figures')
    parser.add_argument('--cases', nargs='+', type=int, default=IEEE_CASES,
                        help='IEEE cases to include')
    args = parser.parse_args()

    results_dir = args.results
    output_dir = args.output
    cases = args.cases
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING ALL PUBLICATION FIGURES")
    print(f"  Results:    {results_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Cases:      {cases}")
    print("=" * 60)

    total = 11

    print(f"\n[1/{total}] K learning comparison...")
    generate_k_learning_comparison(results_dir, output_dir, cases)

    print(f"\n[2/{total}] Radar chart (all 8 methods)...")
    generate_radar_chart_all_baselines(results_dir, output_dir, case=39)

    print(f"\n[3/{total}] Attention maps (4-panel)...")
    generate_attention_maps(output_dir, case=39, results_dir=results_dir)

    print(f"\n[4/{total}] Per-head attention patterns...")
    generate_per_head_attention(output_dir, case=39, results_dir=results_dir)

    print(f"\n[5/{total}] Embedding space (Observation 1)...")
    generate_embedding_space(output_dir, results_dir=results_dir)

    print(f"\n[6/{total}] Multi-case attention comparison...")
    generate_multi_case_attention(output_dir, results_dir=results_dir)

    print(f"\n[7/{total}] Attention weight distribution...")
    generate_attention_distribution(output_dir, results_dir=results_dir)

    print(f"\n[8/{total}] Physics mask vs learned attention (multi-case 4x4)...")
    generate_physics_mask_overlay(output_dir, results_dir=results_dir)

    print(f"\n[9/{total}] t-SNE embeddings (multi-case)...")
    generate_tsne_embeddings(output_dir, results_dir=results_dir)

    print(f"\n[10/{total}] Graph topology with attention (per-case)...")
    generate_graph_attention_topology(output_dir, results_dir=results_dir)

    print(f"\n[11/{total}] Graph topology combined (all cases)...")
    generate_graph_attention_topology_combined(output_dir, results_dir=results_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
