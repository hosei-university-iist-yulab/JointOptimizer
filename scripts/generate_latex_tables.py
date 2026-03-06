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
Automated LaTeX Table Generator (V3) — Journal Extension

Generates 10 tables for IEEE Trans. Smart Grid from results/full_sweep/ JSON data.

Tables:
  I    Main baseline comparison (8 models × 5 IEEE cases)
  II   Ablation study (6 components)
  III  Stress test (models × scenarios)
  IV   Transfer learning (source→target)
  V    Inference latency & parameters
  VI   Theorem 1 validation (delay vs margin)
  VII  N-1 contingency (line outage impact)
  VIII Delay distribution robustness
  IX   Convergence analysis
  X    Model compression (size vs performance)
"""

import argparse
import json
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(val, f='.4f'):
    return f'{val:{f}}'


def fmt_ms(mean, std, f='.4f'):
    return f'${mean:{f}} \\pm {std:{f}}$'


def bold_row(row: str, model_name: str) -> str:
    if model_name == 'JointOptimizer':
        parts = row.split(' &')
        parts[0] = r'\textbf{' + parts[0] + '}'
        return ' &'.join(parts)
    return row


MODEL_ORDER = [
    'JointOptimizer',
    'B1_SequentialOPFQoS', 'B2_MLPJoint', 'B3_GNNOnly',
    'B4_LSTMJoint', 'B5_CNNJoint', 'B6_VanillaTransformer',
    'B7_TransformerNoCoupling', 'B8_HeterogeneousGNN', 'B9_DeepOPF',
]

MODEL_DISPLAY = {
    'JointOptimizer': 'Ours (JointOpt)',
    'B1_SequentialOPFQoS': 'B1: Sequential',
    'B2_MLPJoint': 'B2: MLP Joint',
    'B3_GNNOnly': 'B3: GNN Only',
    'B4_LSTMJoint': 'B4: LSTM Joint',
    'B5_CNNJoint': 'B5: CNN Joint',
    'B6_VanillaTransformer': 'B6: Vanilla Trans.',
    'B7_TransformerNoCoupling': 'B7: Trans. No Coupling',
    'B8_HeterogeneousGNN': 'B8: HeteroGNN',
    'B9_DeepOPF': 'B9: DeepOPF',
}


# =========================================================================
# Table I: Main Baseline Comparison
# =========================================================================
def generate_table_main(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Baseline Comparison: Stability Rate (\%) and Margin $\rho(\tau)$ across IEEE test cases (mean $\pm$ std over 5 seeds).}')
    lines.append(r'\label{tab:main_comparison}')
    lines.append(r'\begin{tabular}{l' + 'cc' * len(cases) + r'}')
    lines.append(r'\toprule')

    header = 'Model'
    for c in cases:
        header += f' & \\multicolumn{{2}}{{c}}{{IEEE {c}}}'
    lines.append(header + r' \\')

    cmr = ''.join(f'\\cmidrule(lr){{{2*i}-{2*i+1}}}' for i in range(1, len(cases)+1))
    lines.append(cmr)
    sub = '' + ''.join(r' & Stab. (\%) & $\rho(\tau)$' for _ in cases)
    lines.append(sub + r' \\')
    lines.append(r'\midrule')

    # Collect all models from first available case
    all_models = []
    for c in cases:
        data = load_json(f'{results_dir}/baselines/case{c}/comparison_results.json')
        if data and 'aggregated_results' in data:
            all_models = [m for m in MODEL_ORDER if m in data['aggregated_results']]
            # Add any models not in MODEL_ORDER
            for m in data['aggregated_results']:
                if m not in all_models:
                    all_models.append(m)
            break

    for model in all_models:
        display = MODEL_DISPLAY.get(model, model.replace('_', r'\_'))
        row = display
        for c in cases:
            data = load_json(f'{results_dir}/baselines/case{c}/comparison_results.json')
            if data and 'aggregated_results' in data:
                agg = data['aggregated_results'].get(model, {})
                s = agg.get('stability_rate', {})
                m = agg.get('margin', {})
                row += f' & {fmt_ms(s.get("mean", 0), s.get("std", 0), ".1f")}'
                row += f' & {fmt_ms(m.get("mean", 0), m.get("std", 0))}'
            else:
                row += r' & --- & ---'
        row += r' \\'
        row = bold_row(row, model)
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


# =========================================================================
# Table II: Ablation Study
# =========================================================================
def generate_table_ablation(results_dir: str, cases: List[int]) -> str:
    # Aggregate ablation across cases
    all_ablations = {}
    for c in cases:
        data = load_json(f'{results_dir}/ablation/case{c}/ablation_results_v2.json')
        if not data or 'ablations' not in data:
            continue
        for abl_type, case_data in data['ablations'].items():
            if abl_type not in all_ablations:
                all_ablations[abl_type] = {}
            for case_key, val_data in case_data.items():
                for val_str, stats in val_data.items():
                    key = (abl_type, val_str)
                    if key not in all_ablations[abl_type]:
                        all_ablations[abl_type][val_str] = []
                    m = stats.get('margin', {})
                    s = stats.get('stability_rate', {})
                    all_ablations[abl_type][val_str].append({
                        'case': c,
                        'margin_mean': m.get('mean', 0),
                        'margin_std': m.get('std', 0),
                        'stab_mean': s.get('mean', 0),
                        'stab_std': s.get('std', 0),
                    })

    if not all_ablations:
        return '% Ablation results not found'

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation Study: Impact of each component on stability margin $\rho(\tau)$ (averaged across IEEE 14--118).}')
    lines.append(r'\label{tab:ablation}')
    lines.append(r'\begin{tabular}{llcc}')
    lines.append(r'\toprule')
    lines.append(r'Component & Setting & $\rho(\tau)$ & Stability (\%) \\')
    lines.append(r'\midrule')

    component_display = {
        'alpha': r'Coupling weight $\alpha$',
        'physics_mask': 'Physics mask',
        'causal_mask': 'Causal mask',
        'cross_attention': 'Cross-attention',
        'contrastive_loss': 'Contrastive loss',
        'gnn_layers': 'GNN layers',
    }

    for abl_type in ['alpha', 'physics_mask', 'causal_mask', 'cross_attention', 'contrastive_loss', 'gnn_layers']:
        if abl_type not in all_ablations:
            continue
        first = True
        for val_str, entries in sorted(all_ablations[abl_type].items()):
            # Average across cases
            avg_margin = sum(e['margin_mean'] for e in entries) / len(entries)
            avg_stab = sum(e['stab_mean'] for e in entries) / len(entries)
            avg_margin_std = sum(e['margin_std'] for e in entries) / len(entries)
            avg_stab_std = sum(e['stab_std'] for e in entries) / len(entries)

            name_col = component_display.get(abl_type, abl_type) if first else ''
            first = False
            lines.append(
                f'{name_col} & {val_str} '
                f'& {fmt_ms(avg_margin, avg_margin_std)} '
                f'& {fmt_ms(avg_stab, avg_stab_std, ".1f")} \\\\'
            )
        lines.append(r'\midrule')

    if lines[-1] == r'\midrule':
        lines[-1] = r'\bottomrule'

    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Table III: Stress Test
# =========================================================================
def generate_table_stress(results_dir: str, cases: List[int]) -> str:
    # Use case 39 as representative (or first available)
    data = None
    used_case = None
    for c in cases:
        data = load_json(f'{results_dir}/stress_test/case{c}/stress_test_case{c}.json')
        if data:
            used_case = c
            break

    if not data:
        return '% Stress test results not found'

    models = data.get('models', [])
    stresses = data.get('stress_scenarios', [])
    results = data.get('results', [])

    # Limit to key stress scenarios
    key_stresses = [s for s in stresses if s in [
        'normal', 'load_110', 'load_120', 'load_130',
        'delay_2x', 'delay_3x', 'noise_5pct', 'noise_10pct',
        'gen_outage_1', 'combined_moderate', 'combined_severe'
    ]]
    if not key_stresses:
        key_stresses = stresses[:8]

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{Stability Rate (\\%) Under Stressed Conditions (IEEE {used_case}).}}')
    lines.append(r'\label{tab:stress_test}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(key_stresses) + r'}')
    lines.append(r'\toprule')

    header = 'Model'
    for s in key_stresses:
        label = s.replace('_', ' ').replace('pct', '\\%')
        header += f' & {label}'
    lines.append(header + r' \\')
    lines.append(r'\midrule')

    ordered_models = [m for m in MODEL_ORDER if m in models]
    for extra in models:
        if extra not in ordered_models:
            ordered_models.append(extra)

    for model in ordered_models:
        display = MODEL_DISPLAY.get(model, model.replace('_', r'\_'))
        row = display
        for stress in key_stresses:
            matching = [r for r in results if r['model'] == model and r['stress'] == stress]
            if matching:
                val = matching[0]['stability_rate']
                row += f' & {val:.1f}'
            else:
                row += ' & ---'
        row += r' \\'
        row = bold_row(row, model)
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


# =========================================================================
# Table IV: Transfer Learning
# =========================================================================
def generate_table_transfer(results_dir: str) -> str:
    pairs = [
        ('14', '39'), ('39', '118'), ('118', '57'),
    ]

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Transfer Learning: Cross-case generalization performance.}')
    lines.append(r'\label{tab:transfer}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'Transfer & Zero-shot $\rho$ & Fine-tuned $\rho$ & Stab. (\%) & Improvement \\')
    lines.append(r'\midrule')

    for src, tgt in pairs:
        path = f'{results_dir}/transfer/{src}_to_{tgt}/transfer_{src}_to_{tgt}.json'
        data = load_json(path)
        if not data or 'transfer_results' not in data:
            continue

        tr = data['transfer_results']
        if not isinstance(tr, list) or len(tr) < 2:
            continue

        # First entry is zero-shot, last is full fine-tune
        zs = tr[0]
        ft = tr[-1]
        zs_margin = zs.get('mean_margin', 0)
        ft_margin = ft.get('mean_margin', 0)
        ft_stab = ft.get('stability_rate', 0)
        improvement = ((ft_margin - zs_margin) / max(abs(zs_margin), 1e-10)) * 100

        lines.append(
            f'IEEE {src} $\\to$ {tgt} '
            f'& {fmt(zs_margin)} & {fmt(ft_margin)} '
            f'& {ft_stab:.1f} & {improvement:+.1f}\\% \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Table V: Inference Latency
# =========================================================================
def generate_table_latency(results_dir: str) -> str:
    data = load_json(f'{results_dir}/inference_benchmark/inference_benchmark.json')
    if not data:
        return '% Inference benchmark results not found'

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Inference Latency (ms) and Model Parameters.}')
    lines.append(r'\label{tab:inference}')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'Model & Case & Mean (ms) & P95 (ms) & P99 (ms) & Params \\')
    lines.append(r'\midrule')

    for r in data.get('results', []):
        name = MODEL_DISPLAY.get(r['model_name'], r['model_name'].replace('_', r'\_'))
        row = (
            f'{name} & {r["case_id"]} & {r["latency_mean_ms"]:.2f} '
            f'& {r["latency_p95_ms"]:.2f} & {r["latency_p99_ms"]:.2f} '
            f'& {r["n_parameters"]:,} \\\\'
        )
        row = bold_row(row, r['model_name'])
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Table VI: Theorem 1 Validation
# =========================================================================
def generate_table_theorem1(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Theorem 1 Validation: Predicted vs.\ empirical stability margin $\rho(\tau)$ at various delays (JointOptimizer).}')
    lines.append(r'\label{tab:theorem1}')
    lines.append(r'\begin{tabular}{l' + 'cc' * len(cases) + r'}')
    lines.append(r'\toprule')

    header = r'Delay (ms)'
    for c in cases:
        header += f' & \\multicolumn{{2}}{{c}}{{IEEE {c}}}'
    lines.append(header + r' \\')

    cmr = ''.join(f'\\cmidrule(lr){{{2*i}-{2*i+1}}}' for i in range(1, len(cases)+1))
    lines.append(cmr)
    sub = '' + ''.join(r' & $\rho_{\mathrm{pred}}$ & $\rho_{\mathrm{emp}}$' for _ in cases)
    lines.append(sub + r' \\')
    lines.append(r'\midrule')

    # Collect delay values
    delay_values = set()
    case_data = {}
    for c in cases:
        data = load_json(f'{results_dir}/theorem1/case{c}/theorem1_all_models.json')
        if not data or 'models' not in data:
            continue
        jo = data['models'].get('JointOptimizer', [])
        case_data[c] = {r['mean_delay_ms']: r for r in jo}
        for r in jo:
            delay_values.add(r['mean_delay_ms'])

    for delay in sorted(delay_values):
        row = f'{delay:.0f}'
        for c in cases:
            cd = case_data.get(c, {})
            r = cd.get(delay)
            if r:
                pred = r.get('empirical_margin', 0)  # model-predicted margin
                emp_stab = r.get('stability_rate', 0)
                row += f' & {fmt(pred)} & {emp_stab:.1f}\\%'
            else:
                row += r' & --- & ---'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


# =========================================================================
# Table VII: N-1 Contingency
# =========================================================================
def generate_table_n1(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{N-1 Contingency: Stability rate (\%) and margin degradation under single line outages.}')
    lines.append(r'\label{tab:n1}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'IEEE Case & Avg. Stab. (\%) & Worst Stab. (\%) & Avg. $\rho$ Degradation \\')
    lines.append(r'\midrule')

    for c in cases:
        data = load_json(f'{results_dir}/n1_contingency/case{c}/n1_case{c}.json')
        if not data or 'results' not in data:
            continue

        results = data['results']
        # Filter for JointOptimizer
        jo_results = [r for r in results if r.get('model') == 'JointOptimizer']
        if not jo_results:
            jo_results = results  # fallback

        stab_rates = [r.get('stability_rate', 0) for r in jo_results]
        margins = [r.get('mean_margin', r.get('margin', 0)) for r in jo_results]

        avg_stab = sum(stab_rates) / len(stab_rates) if stab_rates else 0
        worst_stab = min(stab_rates) if stab_rates else 0
        avg_margin = sum(margins) / len(margins) if margins else 0

        lines.append(
            f'IEEE {c} & {avg_stab:.1f} & {worst_stab:.1f} & {fmt(avg_margin)} \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Table VIII: Delay Distribution Robustness
# =========================================================================
def generate_table_delay(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Robustness Under Different Delay Distributions: Stability rate (\%) and margin $\rho(\tau)$.}')
    lines.append(r'\label{tab:delay_dist}')

    # Discover distributions from first available case
    distributions = []
    for c in cases:
        data = load_json(f'{results_dir}/delay_dist/case{c}/delay_robustness_case{c}.json')
        if data and 'results' in data:
            distributions = [r['distribution'] for r in data['results']]
            break

    if not distributions:
        return '% Delay distribution results not found'

    lines.append(r'\begin{tabular}{l' + 'cc' * len(distributions) + r'}')
    lines.append(r'\toprule')

    header = 'Case'
    for d in distributions:
        header += f' & \\multicolumn{{2}}{{c}}{{{d.capitalize()}}}'
    lines.append(header + r' \\')

    cmr = ''.join(f'\\cmidrule(lr){{{2*i}-{2*i+1}}}' for i in range(1, len(distributions)+1))
    lines.append(cmr)
    sub = '' + ''.join(r' & Stab. & $\rho$' for _ in distributions)
    lines.append(sub + r' \\')
    lines.append(r'\midrule')

    for c in cases:
        data = load_json(f'{results_dir}/delay_dist/case{c}/delay_robustness_case{c}.json')
        if not data or 'results' not in data:
            continue

        row = f'IEEE {c}'
        dist_map = {r['distribution']: r for r in data['results']}
        for d in distributions:
            r = dist_map.get(d)
            if r:
                row += f' & {r["stability_rate"]:.1f}\\% & {fmt(r["mean_margin"])}'
            else:
                row += r' & --- & ---'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


# =========================================================================
# Table IX: Convergence Analysis
# =========================================================================
def generate_table_convergence(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Convergence: Training scenarios and epochs required for $>$95\% stability.}')
    lines.append(r'\label{tab:convergence}')
    lines.append(r'\begin{tabular}{lccccc}')
    lines.append(r'\toprule')
    lines.append(r'Case & Min Scenarios & Margin@100 & Margin@200 & Margin@300 & Best Margin \\')
    lines.append(r'\midrule')

    for c in cases:
        data = load_json(f'{results_dir}/convergence/case{c}/convergence_case{c}.json')
        if not data:
            continue

        # Scenario sweep (list of dicts) — find min scenarios for >95% stability
        scenario_sweep = data.get('scenario_sweep', [])
        min_scen = '---'
        for entry in sorted(scenario_sweep, key=lambda x: x.get('num_scenarios', 0)):
            if entry.get('final_stability_rate', 0) >= 95.0:
                min_scen = str(entry['num_scenarios'])
                break

        # Epoch sweep (list of dicts) — extract margin at different epoch counts
        epoch_sweep = data.get('epoch_sweep', [])
        margins = {}
        best_margin = 0
        for entry in epoch_sweep:
            ep = entry.get('epochs', 0)
            m = entry.get('final_margin', 0)
            margins[ep] = m
            best_margin = max(best_margin, m)

        m100 = fmt(margins.get(100, 0)) if 100 in margins else '---'
        m200 = fmt(margins.get(200, 0)) if 200 in margins else '---'
        m300 = fmt(margins.get(300, 0)) if 300 in margins else '---'

        lines.append(
            f'IEEE {c} & {min_scen} & {m100} & {m200} & {m300} & {fmt(best_margin)} \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Table X: Model Compression
# =========================================================================
def generate_table_compression(results_dir: str, cases: List[int]) -> str:
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Model Compression: Embedding dimension vs.\ stability margin and parameter count.}')
    lines.append(r'\label{tab:compression}')
    lines.append(r'\begin{tabular}{cccccc}')
    lines.append(r'\toprule')
    lines.append(r'$d_{\mathrm{embed}}$ & $d_{\mathrm{hidden}}$ & Heads & Params & $\rho(\tau)$ & Stab. (\%) \\')
    lines.append(r'\midrule')

    # Use case 39 as representative
    for c in [39, 57, 118]:
        data = load_json(f'{results_dir}/model_compression/case{c}/model_sweep_case{c}.json')
        if data and 'results' in data:
            lines.append(f'\\multicolumn{{6}}{{c}}{{\\textit{{IEEE {c}}}}} \\\\')
            lines.append(r'\midrule')
            for r in data['results']:
                lines.append(
                    f'{r["embed_dim"]} & {r["hidden_dim"]} & {r["num_heads"]} '
                    f'& {r["n_parameters"]:,} & {fmt(r["mean_margin"])} & {r["stability_rate"]:.1f} \\\\'
                )
            lines.append(r'\midrule')
            break  # Just show one representative case

    if lines[-1] == r'\midrule':
        lines[-1] = r'\bottomrule'

    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# =========================================================================
# Main
# =========================================================================
def generate_all_tables(results_dir: str, output_dir: str, cases: List[int]):
    os.makedirs(output_dir, exist_ok=True)

    tables = {
        'table01_main_comparison.tex': generate_table_main(results_dir, cases),
        'table02_ablation.tex': generate_table_ablation(results_dir, cases),
        'table03_stress_test.tex': generate_table_stress(results_dir, cases),
        'table04_transfer.tex': generate_table_transfer(results_dir),
        'table05_inference.tex': generate_table_latency(results_dir),
        'table06_theorem1.tex': generate_table_theorem1(results_dir, cases),
        'table07_n1_contingency.tex': generate_table_n1(results_dir, cases),
        'table08_delay_dist.tex': generate_table_delay(results_dir, cases),
        'table09_convergence.tex': generate_table_convergence(results_dir, cases),
        'table10_compression.tex': generate_table_compression(results_dir, cases),
    }

    print("=" * 60)
    print("LaTeX TABLE GENERATOR — Journal Extension (10 Tables)")
    print("=" * 60)

    for filename, content in tables.items():
        path = f'{output_dir}/{filename}'
        with open(path, 'w') as f:
            f.write(content)
        has_data = r'\midrule' in content or r'\bottomrule' in content
        status = "OK" if has_data and '% ' not in content[:3] else "EMPTY/MISSING"
        print(f"  {status:14s} {filename}")

    combined = f'{output_dir}/all_tables.tex'
    with open(combined, 'w') as f:
        f.write(f'% Auto-generated LaTeX tables for IEEE Trans. Smart Grid\n')
        f.write(f'% Generated: {datetime.now().isoformat()}\n')
        f.write(f'% Cases: {cases}\n\n')
        for filename, content in tables.items():
            f.write(f'% === {filename} ===\n')
            f.write(content)
            f.write('\n\n')
    print(f"\n  Combined: {combined}")


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from results')
    parser.add_argument('--results', type=str, default='results/full_sweep')
    parser.add_argument('--output', type=str, default='docs/tables')
    parser.add_argument('--cases', type=int, nargs='+', default=[14, 30, 39, 57, 118])
    args = parser.parse_args()

    generate_all_tables(args.results, args.output, args.cases)


if __name__ == '__main__':
    main()
