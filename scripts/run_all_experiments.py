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
Master Experiment Orchestrator

Runs ALL experiments on ALL datasets (IEEE 14, 30, 39, 57, 118 + Synthetic 10K)
in parallel across 4 GPUs.

Usage:
    python scripts/run_all_experiments.py --dry-run          # Preview jobs
    python scripts/run_all_experiments.py                    # Full run
    python scripts/run_all_experiments.py --phase 1          # Only Phase 1
    python scripts/run_all_experiments.py --resume           # Skip completed
"""

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
GPUS = [5, 6]  # Conservative: 2 GPUs on shared server
IEEE_CASES = [14, 30, 39, 57, 118]
SYNTHETIC_CASES = [10000]
ALL_CASES = IEEE_CASES + SYNTHETIC_CASES
RESULTS_ROOT = PROJECT_ROOT / "results" / "full_sweep"


@dataclass
class Job:
    name: str
    script: str
    args: List[str]
    priority: int = 1  # 0=light, 1=medium, 2=heavy
    phase: int = 1


def get_epochs(case_id: int, cap: Optional[int] = None) -> int:
    """Get epoch count for a case, optionally capped."""
    table = {14: 200, 30: 200, 39: 300, 57: 300, 118: 500, 10000: 800}
    e = table.get(case_id, 300)
    return min(e, cap) if cap else e


def get_scenarios(case_id: int) -> int:
    return 200 if case_id >= 1000 else 500


def get_batch_size(case_id: int) -> int:
    return 2 if case_id >= 1000 else 32


def build_jobs() -> List[Job]:
    """Build the full job matrix."""
    jobs = []

    # =====================================================================
    # PHASE 1: Independent experiments (self-training)
    # =====================================================================
    for case in ALL_CASES:
        epochs = get_epochs(case)
        scenarios = get_scenarios(case)
        batch = get_batch_size(case)
        out = f"results/full_sweep"

        # --- Training ---
        jobs.append(Job(
            name=f"train_case{case}",
            script="scripts/train.py",
            args=["--config", "configs/journal_experiments.yaml",
                  "--case", str(case), "--epochs", str(epochs),
                  "--batch_size", str(batch),
                  "--save_dir", f"{out}/train/case{case}",
                  "--no_wandb"],
            priority=2, phase=1,
        ))

        # --- Baseline comparison ---
        jobs.append(Job(
            name=f"baselines_case{case}",
            script="scripts/run_baseline_comparison.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 200)),
                  "--scenarios", str(scenarios), "--seeds", "5",
                  "--output", f"{out}/baselines/case{case}"],
            priority=2, phase=1,
        ))

        # --- Ablation ---
        jobs.append(Job(
            name=f"ablation_case{case}",
            script="experiments/run_ablation.py",
            args=["--cases", str(case), "--epochs", str(get_epochs(case, 100)),
                  "--scenarios", str(scenarios), "--seeds", "3",
                  "--output", f"{out}/ablation/case{case}"],
            priority=2, phase=1,
        ))

        # --- Gamma sweep ---
        jobs.append(Job(
            name=f"gamma_case{case}",
            script="experiments/gamma_sweep.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 100)),
                  "--scenarios", str(scenarios), "--seeds", "3",
                  "--output", f"{out}/gamma_sweep/case{case}"],
            priority=1, phase=1,
        ))

        # --- K init sensitivity ---
        jobs.append(Job(
            name=f"k_init_case{case}",
            script="experiments/k_init_sensitivity.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 200)),
                  "--seeds", "3",
                  "--output", f"{out}/k_init/case{case}"],
            priority=1, phase=1,
        ))

        # --- Delay distribution ---
        jobs.append(Job(
            name=f"delay_dist_case{case}",
            script="experiments/delay_distribution_robustness.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 150)),
                  "--scenarios", str(scenarios),
                  "--output", f"{out}/delay_dist/case{case}"],
            priority=1, phase=1,
        ))

        # --- Convergence ---
        jobs.append(Job(
            name=f"convergence_case{case}",
            script="experiments/convergence_analysis.py",
            args=["--case", str(case),
                  "--output", f"{out}/convergence/case{case}"],
            priority=1, phase=1,
        ))

        # --- Stress test ---
        jobs.append(Job(
            name=f"stress_case{case}",
            script="experiments/stress_test_stability.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 100)),
                  "--scenarios", str(scenarios),
                  "--output", f"{out}/stress_test/case{case}"],
            priority=2, phase=1,
        ))

        # --- N-1 contingency ---
        jobs.append(Job(
            name=f"n1_case{case}",
            script="experiments/n1_contingency.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 100)),
                  "--scenarios", str(scenarios),
                  "--output", f"{out}/n1_contingency/case{case}"],
            priority=2, phase=1,
        ))

        # --- Model compression ---
        jobs.append(Job(
            name=f"compression_case{case}",
            script="experiments/model_compression.py",
            args=["--case", str(case), "--epochs", str(get_epochs(case, 100)),
                  "--scenarios", str(scenarios),
                  "--output", f"{out}/model_compression/case{case}"],
            priority=1, phase=1,
        ))

    # =====================================================================
    # Theory / validation (lightweight, IEEE cases only for eigenvalue-heavy)
    # =====================================================================
    for case in ALL_CASES:
        out = f"results/full_sweep"

        # Theorem 1 independent (DDE)
        jobs.append(Job(
            name=f"theorem1_indep_case{case}",
            script="experiments/validate_theorem1_independent.py",
            args=["--case", str(case),
                  "--delays", "50", "100", "200", "300", "400", "500",
                  "--trials", "5",
                  "--output", f"{out}/theorem1_indep/case{case}"],
            priority=0, phase=1,
        ))

        # Padé analysis
        jobs.append(Job(
            name=f"pade_case{case}",
            script="experiments/pade_analysis.py",
            args=["--case", str(case),
                  "--output", f"{out}/pade/case{case}"],
            priority=0, phase=1,
        ))

        # Domain separation
        jobs.append(Job(
            name=f"domain_sep_case{case}",
            script="experiments/validate_domain_separation.py",
            args=["--case", str(case), "--scenarios", str(get_scenarios(case)),
                  "--output", f"{out}/domain_sep/case{case}"],
            priority=0, phase=1,
        ))

    # --- Inference benchmark (all cases) ---
    cases_str = ",".join(str(c) for c in ALL_CASES)
    jobs.append(Job(
        name="inference_benchmark",
        script="experiments/inference_benchmark.py",
        args=["--cases", cases_str, "--batch-size", "32",
              "--output", f"results/full_sweep/inference_benchmark"],
        priority=0, phase=1,
    ))

    # =====================================================================
    # PHASE 2: Post-training experiments
    # =====================================================================
    for case in ALL_CASES:
        out = f"results/full_sweep"

        # Theorem 1 (model-based)
        jobs.append(Job(
            name=f"theorem1_case{case}",
            script="experiments/validate_theorem1.py",
            args=["--case", str(case),
                  "--delays", "50", "100", "200", "300", "400", "500",
                  "--scenarios", str(get_scenarios(case)),
                  "--output", f"{out}/theorem1/case{case}"],
            priority=1, phase=2,
        ))

    # --- Transfer learning ---
    for source, target in [(39, 118), (118, 57), (14, 39), (39, 10000)]:
        jobs.append(Job(
            name=f"transfer_{source}_to_{target}",
            script="scripts/run_transfer_learning.py",
            args=["--source", str(source), "--target", str(target),
                  "--source-epochs", "200", "--target-epochs", "100",
                  "--scenarios", "500",
                  "--output", f"results/full_sweep/transfer/{source}_to_{target}"],
            priority=1, phase=2,
        ))

    # =====================================================================
    # PHASE 3: Aggregation
    # =====================================================================
    jobs.append(Job(
        name="generate_tables",
        script="scripts/generate_latex_tables.py",
        args=["--results", "results/full_sweep",
              "--output", "docs/tables"],
        priority=0, phase=3,
    ))

    jobs.append(Job(
        name="audit_results",
        script="scripts/audit_results.py",
        args=[],
        priority=0, phase=3,
    ))

    return jobs


# =========================================================================
# GPU Queue: ensures one job per GPU at a time
# =========================================================================
class GPUQueue:
    def __init__(self, gpus: List[int], slots_per_gpu: int = 3):
        self._queue = queue.Queue()
        for g in gpus:
            for _ in range(slots_per_gpu):
                self._queue.put(g)

    def acquire(self) -> int:
        return self._queue.get()

    def release(self, gpu_id: int):
        self._queue.put(gpu_id)


def run_job(job: Job, gpu_id: int, log_dir: Path, dry_run: bool = False) -> dict:
    """Run a single experiment on a specific GPU."""
    cmd = [sys.executable, str(PROJECT_ROOT / job.script)] + job.args

    log_file = log_dir / f"{job.name}.log"
    result = {
        "name": job.name,
        "gpu": gpu_id,
        "cmd": " ".join(cmd),
        "phase": job.phase,
    }

    if dry_run:
        result["status"] = "dry-run"
        return result

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    try:
        with open(log_file, "w") as f:
            f.write(f"# Job: {job.name}\n")
            f.write(f"# GPU: {gpu_id}\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Cmd: {' '.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT), env=env,
                stdout=f, stderr=subprocess.STDOUT,
                timeout=3600 * 12,
            )
        elapsed = time.time() - start
        result["returncode"] = proc.returncode
        result["elapsed_s"] = round(elapsed, 1)
        result["status"] = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        result["returncode"] = -1
        result["elapsed_s"] = time.time() - start
        result["status"] = "timeout"
    except Exception as e:
        result["returncode"] = -2
        result["elapsed_s"] = time.time() - start
        result["status"] = f"error: {e}"

    result["log"] = str(log_file)
    return result


def run_phase(jobs: List[Job], gpu_queue: GPUQueue, log_dir: Path,
              dry_run: bool = False, resume_completed: set = None):
    """Run a list of jobs in parallel using the GPU queue."""
    if resume_completed is None:
        resume_completed = set()

    # Filter out already-completed jobs
    to_run = [j for j in jobs if j.name not in resume_completed]
    skipped = len(jobs) - len(to_run)
    if skipped:
        print(f"  Skipping {skipped} already-completed jobs")

    if not to_run:
        return []

    # Sort by priority (light first)
    to_run.sort(key=lambda j: j.priority)

    results = []

    def _run_with_gpu(job):
        gpu_id = gpu_queue.acquire()
        tag = f"[GPU {gpu_id}]"
        print(f"  {tag} START  {job.name}")
        try:
            r = run_job(job, gpu_id, log_dir, dry_run)
            status = r["status"]
            elapsed = r.get("elapsed_s", 0)
            print(f"  {tag} {status.upper():7s} {job.name} ({elapsed:.0f}s)")
            return r
        finally:
            gpu_queue.release(gpu_id)

    with ThreadPoolExecutor(max_workers=len(GPUS) * 3) as executor:
        futures = {executor.submit(_run_with_gpu, j): j for j in to_run}
        for future in as_completed(futures):
            results.append(future.result())

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview jobs without running")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only this phase (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed jobs")
    args = parser.parse_args()

    all_jobs = build_jobs()
    log_dir = RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load completed jobs for resume
    manifest_path = RESULTS_ROOT / "manifest.json"
    completed = set()
    if args.resume and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        completed = {r["name"] for r in manifest.get("completed", [])
                     if r.get("status") == "ok"}
        print(f"Resume mode: {len(completed)} previously completed jobs")

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN — Job Matrix Preview")
        print("=" * 70)
        for phase in [1, 2, 3]:
            phase_jobs = [j for j in all_jobs if j.phase == phase]
            if args.phase and phase != args.phase:
                continue
            print(f"\n--- Phase {phase} ({len(phase_jobs)} jobs) ---")
            for j in sorted(phase_jobs, key=lambda x: (x.priority, x.name)):
                skip = "(SKIP)" if j.name in completed else ""
                print(f"  P{j.priority} {j.name:40s} {j.script} {skip}")
        print(f"\nTotal: {len(all_jobs)} jobs across 3 phases")
        print(f"GPUs: {GPUS}")
        return

    gpu_queue = GPUQueue(GPUS)
    all_results = []

    phases = [1, 2, 3] if args.phase == 0 else [args.phase]

    for phase in phases:
        phase_jobs = [j for j in all_jobs if j.phase == phase]
        if not phase_jobs:
            continue
        print(f"\n{'=' * 70}")
        print(f"PHASE {phase} — {len(phase_jobs)} jobs")
        print(f"{'=' * 70}")

        results = run_phase(phase_jobs, gpu_queue, log_dir,
                            resume_completed=completed)
        all_results.extend(results)

        # Save manifest after each phase
        ok = [r for r in all_results if r.get("status") == "ok"]
        fail = [r for r in all_results if r.get("status") != "ok"]
        with open(manifest_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "completed": ok,
                "failed": fail,
            }, f, indent=2)

        n_ok = sum(1 for r in results if r.get("status") == "ok")
        n_fail = len(results) - n_ok
        print(f"\nPhase {phase}: {n_ok} ok, {n_fail} failed")

        if n_fail > 0 and phase < 3:
            print("WARNING: Some jobs failed. Continuing to next phase anyway.")

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    ok_total = sum(1 for r in all_results if r.get("status") == "ok")
    fail_total = len(all_results) - ok_total
    print(f"Completed: {ok_total}, Failed: {fail_total}")
    if fail_total > 0:
        print("\nFailed jobs:")
        for r in all_results:
            if r.get("status") != "ok":
                print(f"  {r['name']}: {r['status']} (log: {r.get('log', 'N/A')})")


if __name__ == "__main__":
    main()
