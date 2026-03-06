#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/14/2025🚀

Author: Franck Aboya
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Data Audit Script — Verify reported values match raw JSON data.

Addresses Q8.1: K=0.082 reported for IEEE 14 in paper, but raw JSON shows K=0.099.
Flags any discrepancy > 1% between paper-reported values and raw experimental data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Paper-reported values (from conference paper and results documentation)
# Format: {table_name: [{source_file, key_path, reported_value, description}]}
PAPER_VALUES = {
    "Table_II_Baseline_Comparison_Case14": [
        {"source": "results/baselines_case14/comparison_results.json",
         "key": ("results", 0, "mean_stability_margin"),
         "reported": 0.3687, "description": "JointOptimizer mean margin (case14)"},
        {"source": "results/baselines_case14/comparison_results.json",
         "key": ("results", 0, "K_mean"),
         "reported": 0.099, "description": "JointOptimizer K_mean (case14)"},
        {"source": "results/baselines_case14/comparison_results.json",
         "key": ("results", 0, "stability_rate"),
         "reported": 100.0, "description": "JointOptimizer stability rate (case14)"},
        {"source": "results/baselines_case14/comparison_results.json",
         "key": ("results", 0, "num_parameters"),
         "reported": 469140, "description": "JointOptimizer param count"},
    ],
    "Theorem1_Validation": [
        {"source": "results/theorem1/theorem1_validation.json",
         "key": ("results", 0, "empirical_margin"),
         "reported": 0.3931, "description": "rho at tau=10ms"},
        {"source": "results/theorem1/theorem1_validation.json",
         "key": ("results", 0, "theoretical_margin"),
         "reported": 0.3931, "description": "rho_theo at tau=10ms"},
        {"source": "results/theorem1/theorem1_validation.json",
         "key": ("results", -1, "stability_rate"),
         "reported": 83.57, "description": "Stability rate at tau=500ms"},
    ],
    "Known_Discrepancies": [
        # This documents values that were reportedly wrong in earlier versions
        {"source": "results/baselines_case14/comparison_results.json",
         "key": ("results", 0, "K_mean"),
         "reported": 0.082, "description": "KNOWN BUG: K=0.082 was reported for case14, "
                                            "but this value comes from case57"},
    ],
}


def load_json_value(file_path: str, key_path: tuple):
    """
    Load a value from a JSON file using a key path.

    Args:
        file_path: Path to JSON file
        key_path: Tuple of keys/indices to traverse (e.g., ("results", 0, "K_mean"))

    Returns:
        The value at the specified path
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    current = data
    for key in key_path:
        current = current[key]
    return current


def audit_reported_vs_raw(
    base_dir: str,
    paper_values: Dict = None,
) -> List[Dict]:
    """
    Verify every reported number matches raw JSON data.
    Flag any discrepancy > 1%.

    Args:
        base_dir: Base directory for resolving relative paths
        paper_values: Dict of paper values to check. Uses PAPER_VALUES if None.

    Returns:
        List of discrepancy dicts
    """
    if paper_values is None:
        paper_values = PAPER_VALUES

    discrepancies = []
    checks_passed = 0
    checks_total = 0

    for table_name, entries in paper_values.items():
        print(f"\n--- {table_name} ---")

        for entry in entries:
            file_path = os.path.join(base_dir, entry['source'])
            checks_total += 1

            if not os.path.exists(file_path):
                discrepancies.append({
                    'table': table_name,
                    'description': entry['description'],
                    'issue': f"FILE NOT FOUND: {file_path}",
                    'reported': entry['reported'],
                    'actual': None,
                })
                print(f"  [MISSING] {entry['description']}: {file_path}")
                continue

            try:
                actual = load_json_value(file_path, entry['key'])
            except (KeyError, IndexError, TypeError) as e:
                discrepancies.append({
                    'table': table_name,
                    'description': entry['description'],
                    'issue': f"KEY NOT FOUND: {entry['key']} -> {e}",
                    'reported': entry['reported'],
                    'actual': None,
                })
                print(f"  [KEY ERR] {entry['description']}: {e}")
                continue

            reported = entry['reported']

            # Compute discrepancy
            if isinstance(actual, (int, float)) and isinstance(reported, (int, float)):
                if abs(actual) < 1e-10 and abs(reported) < 1e-10:
                    disc_pct = 0.0
                else:
                    disc_pct = abs(actual - reported) / max(abs(actual), 1e-10) * 100

                if disc_pct > 1.0:
                    discrepancies.append({
                        'table': table_name,
                        'description': entry['description'],
                        'reported': reported,
                        'actual': actual,
                        'discrepancy_pct': disc_pct,
                    })
                    print(f"  [MISMATCH] {entry['description']}: "
                          f"reported={reported}, actual={actual} "
                          f"({disc_pct:.1f}% off)")
                else:
                    checks_passed += 1
                    print(f"  [OK]       {entry['description']}: "
                          f"reported={reported}, actual={actual} "
                          f"({disc_pct:.2f}% off)")
            else:
                # Non-numeric comparison
                if actual != reported:
                    discrepancies.append({
                        'table': table_name,
                        'description': entry['description'],
                        'reported': reported,
                        'actual': actual,
                        'discrepancy_pct': None,
                    })
                    print(f"  [MISMATCH] {entry['description']}: "
                          f"reported={reported}, actual={actual}")
                else:
                    checks_passed += 1
                    print(f"  [OK]       {entry['description']}")

    return discrepancies, checks_passed, checks_total


def check_cross_case_consistency(base_dir: str) -> List[Dict]:
    """
    Check that the same model reports consistent parameter counts
    across different IEEE cases.
    """
    issues = []
    param_counts = {}

    for case_dir in sorted(Path(base_dir).glob("results/baselines_case*/comparison_results.json")):
        with open(case_dir, 'r') as f:
            data = json.load(f)

        case_id = data.get('case_id', str(case_dir.parent.name))

        for result in data.get('results', []):
            model_name = result['model_name']
            n_params = result.get('num_parameters', 0)

            if model_name not in param_counts:
                param_counts[model_name] = {}
            param_counts[model_name][case_id] = n_params

    print("\n--- Cross-Case Parameter Count Consistency ---")
    for model_name, counts in param_counts.items():
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            # Different param counts is expected for models that scale with bus count
            # But some baselines should have fixed architecture
            print(f"  [INFO] {model_name}: params vary across cases: {counts}")
        else:
            print(f"  [OK]   {model_name}: consistent {list(unique_counts)[0]} params")

    return issues


def check_circular_validation(base_dir: str):
    """
    Check if theorem1 validation has the circular issue.
    Flag if empirical_margin == theoretical_margin to machine precision.
    """
    theorem1_path = os.path.join(base_dir, "results/theorem1/theorem1_validation.json")
    if not os.path.exists(theorem1_path):
        print("\n--- Circular Validation Check ---")
        print("  [SKIP] No theorem1 results found")
        return

    with open(theorem1_path, 'r') as f:
        data = json.load(f)

    print("\n--- Circular Validation Check (CRITICAL) ---")
    n_circular = 0
    for result in data.get('results', []):
        emp = result.get('empirical_margin', 0)
        theo = result.get('theoretical_margin', 0)
        gap = abs(emp - theo)

        if gap < 1e-6:
            n_circular += 1
            print(f"  [CIRCULAR] tau={result['mean_delay_ms']}ms: "
                  f"emp={emp:.10f}, theo={theo:.10f}, gap={gap:.2e}")

    if n_circular > 0:
        print(f"\n  *** FATAL: {n_circular}/{len(data['results'])} delay points "
              f"have circular validation (gap < 1e-6) ***")
        print("  *** This means rho_emp and rho_theo use the SAME formula ***")
        print("  *** Run validate_theorem1_independent.py for honest validation ***")
    else:
        print("  [OK] No circular validation detected")


def main():
    base_dir = str(Path(__file__).parent.parent)

    print("=" * 70)
    print("DATA AUDIT: Verify reported values match raw JSON data")
    print("=" * 70)
    print(f"Base directory: {base_dir}")

    # Run audit
    discrepancies, passed, total = audit_reported_vs_raw(base_dir)

    # Cross-case consistency
    check_cross_case_consistency(base_dir)

    # Circular validation check
    check_circular_validation(base_dir)

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Checks passed: {passed}/{total}")
    print(f"Discrepancies found: {len(discrepancies)}")

    if discrepancies:
        print("\nDISCREPANCIES:")
        for i, d in enumerate(discrepancies, 1):
            print(f"  {i}. [{d['table']}] {d['description']}")
            print(f"     Reported: {d['reported']}")
            print(f"     Actual:   {d.get('actual', 'N/A')}")
            if d.get('discrepancy_pct') is not None:
                print(f"     Gap:      {d['discrepancy_pct']:.1f}%")
            if d.get('issue'):
                print(f"     Issue:    {d['issue']}")
    else:
        print("\nAll checks passed!")


if __name__ == '__main__':
    main()
