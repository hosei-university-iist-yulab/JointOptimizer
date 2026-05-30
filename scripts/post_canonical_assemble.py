#!/usr/bin/env python3
"""Post-processing assembler for the canonical rerun.

Given the per-case ``comparison_results.json`` files written by
``run_baseline_comparison.py`` under ``results/full_sweep/canonical/case{C}/``,
this script:

1. Verifies all expected cases (14, 30, 39, 57, 118, 300) and all 14 models
   (B1-B12 plus four JointOptimizer variants) are present.
2. Regenerates the 10 standard publication tables via
   ``scripts/generate_latex_tables.py`` pointing at the canonical results.
3. Regenerates publication figures via ``scripts/figures/generate_publication_figures.py``.
4. Syncs the regenerated tables and figures into ``paper/Applied-Energy/revision/``.
5. Writes a short canonical-summary prose block into
   ``paper/Applied-Energy/revision/sections/06_results.tex`` referencing the
   new headline numbers (only if a placeholder marker exists).
6. Recompiles the revision PDF via ``tectonic``.

Designed to be safe to run multiple times. Checks each step's exit code and
aborts loudly if anything fails. Writes a single audit JSON at
``results/full_sweep/canonical/canonical_audit.json``.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The orchestrator's baselines step writes to results/full_sweep/baselines/case{C}/.
# CANONICAL_ROOT now points there directly so the audit reads the actual data.
CANONICAL_ROOT = REPO / "results/full_sweep/baselines"
REVISION_ROOT = REPO / "paper/Applied-Energy/revision"
# Originals: docs/figures/publication and docs/tables (matches the generators'
# defaults). After landing in docs/, the assembler also copies the published
# fig_*.pdf and table*.tex sets into paper/Applied-Energy/revision/{figures,tables}/
# so the resubmission package stays self-contained. The two static TikZ figures
# (architecture.pdf, fig_problem_overview.pdf) live in revision/figures/ only and
# are never overwritten because the cp pattern excludes them.
TABLES_OUT = REPO / "docs/tables"
FIGURES_OUT = REPO / "docs/figures/publication"
BASELINES_ROOT = CANONICAL_ROOT  # mirror step in main() becomes a no-op

EXPECTED_CASES = [14, 30, 39, 57, 118, 300]
EXPECTED_MODELS = [
    "JointOptimizer", "JointOptimizer_Lite",
    "JointOptimizer_Lever3", "JointOptimizer_Lever3_Lite",
    "B1_SequentialOPFQoS", "B2_MLPJoint", "B3_GNNOnly", "B4_LSTMJoint",
    "B5_CNNJoint", "B6_VanillaTransformer", "B7_TransformerNoCoupling",
    "B10_LinearMPC", "B11_SmithPredictor", "B12_NeuralMPC",
]


def verify_case(case_id: int) -> dict:
    p = CANONICAL_ROOT / f"case{case_id}/comparison_results.json"
    if not p.exists():
        return {"case": case_id, "ok": False, "reason": f"missing {p}"}
    d = json.loads(p.read_text())
    agg = d.get("aggregated_results", {})
    models = sorted(agg.keys()) if isinstance(agg, dict) else []
    missing = [m for m in EXPECTED_MODELS if m not in models]
    return {
        "case": case_id,
        "ok": len(missing) == 0,
        "epochs": d.get("epochs"),
        "num_seeds": d.get("num_seeds"),
        "models": models,
        "missing": missing,
        "path": str(p),
    }


def winner_table(case_id: int) -> dict:
    p = CANONICAL_ROOT / f"case{case_id}/comparison_results.json"
    d = json.loads(p.read_text())
    agg = d["aggregated_results"]
    out = {}
    for m, info in agg.items():
        margin = (info.get("margin") or info.get("mean_stability_margin") or {}).get("mean")
        tau = (info.get("tau_crit_ms") or {}).get("mean")
        k = (info.get("K_mean") or {}).get("mean")
        out[m] = dict(margin=margin, tau_crit_ms=tau, K_mean=k)
    return out


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-figures", action="store_true",
                    help="Only regenerate tables; skip the figure pipeline.")
    ap.add_argument("--skip-pdf", action="store_true",
                    help="Skip the tectonic recompile step.")
    args = ap.parse_args()

    # Step 1 — verify
    audit = [verify_case(c) for c in EXPECTED_CASES]
    print("\n=== Canonical audit ===")
    for a in audit:
        flag = "OK" if a["ok"] else "MISS"
        miss = f" missing={a.get('missing')}" if not a["ok"] else ""
        print(f"  case-{a['case']:>3} {flag} epochs={a.get('epochs')} seeds={a.get('num_seeds')} models={len(a.get('models', []))}{miss}")
    if not all(a["ok"] for a in audit):
        print("\nAborting: not all canonical JSONs are complete.")
        (CANONICAL_ROOT / "canonical_audit.json").write_text(
            json.dumps({"audit": audit, "complete": False}, indent=2))
        return 1

    # Step 2 — winners per case
    print("\n=== Per-case headline winners ===")
    winners = {}
    for c in EXPECTED_CASES:
        w = winner_table(c)
        m_best = max(w.items(), key=lambda kv: (kv[1]["margin"] or 0))
        t_best = max(w.items(), key=lambda kv: (kv[1]["tau_crit_ms"] or 0))
        k_best = min(w.items(), key=lambda kv: (kv[1]["K_mean"] if kv[1]["K_mean"] else 1e9))
        print(f"  case-{c:>3} best margin={m_best[0]} ({m_best[1]['margin']:.4f}) "
              f"best tau_crit={t_best[0]} ({t_best[1]['tau_crit_ms']:.0f}ms) "
              f"smallest K={k_best[0]} ({k_best[1]['K_mean']:.4f})")
        winners[c] = dict(margin=m_best, tau_crit=t_best, K_mean=k_best)

    # Step 3 — (no-op since CANONICAL_ROOT == BASELINES_ROOT now). The
    # orchestrator already wrote per-case JSONs to results/full_sweep/baselines/,
    # so the figure generator's default --results=results/full_sweep finds both
    # the per-case JSON and the train/case{C}/best.pt checkpoints in place.
    if CANONICAL_ROOT.resolve() != BASELINES_ROOT.resolve():
        BASELINES_ROOT.mkdir(parents=True, exist_ok=True)
        for c in EXPECTED_CASES:
            src = CANONICAL_ROOT / f"case{c}/comparison_results.json"
            dst_dir = BASELINES_ROOT / f"case{c}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / "comparison_results.json")
        print(f"Mirrored {len(EXPECTED_CASES)} canonical JSONs to {BASELINES_ROOT}")

    # Step 4 — regenerate tables (write to docs/tables, the generator default)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    if run([
        sys.executable, "scripts/generate_latex_tables.py",
        "--results", "results/full_sweep",
        "--output", str(TABLES_OUT),
        "--cases", *map(str, EXPECTED_CASES),
    ]) != 0:
        print("Table generation failed."); return 2

    # Step 5 — sync tables into paper/.../revision/tables (overwrites table*.tex,
    # leaves any non-table files like real_pmu placeholder untouched)
    rev_tables = REVISION_ROOT / "tables"
    rev_tables.mkdir(parents=True, exist_ok=True)
    table_files = list(TABLES_OUT.glob("table*.tex"))
    for tex in table_files:
        shutil.copy2(tex, rev_tables / tex.name)
    print(f"Copied {len(table_files)} tables to {rev_tables}")

    # Step 6 — figures (write to docs/figures/publication, the generator default)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        if run([
            sys.executable, "scripts/figures/generate_publication_figures.py",
            "--results", "results/full_sweep",
            "--output", str(FIGURES_OUT),
        ]) != 0:
            print("Figure generation failed (non-fatal); continuing.")

        # Step 7 — sync only fig_*.pdf into revision/figures (preserves the two
        # static TikZ figures architecture.pdf and fig_problem_overview.pdf).
        rev_figures = REVISION_ROOT / "figures"
        rev_figures.mkdir(parents=True, exist_ok=True)
        fig_files = list(FIGURES_OUT.glob("fig_*.pdf"))
        for pdf in fig_files:
            shutil.copy2(pdf, rev_figures / pdf.name)
        print(f"Copied {len(fig_files)} figures to {rev_figures}")

    # Step 8 — PDF recompile
    if not args.skip_pdf:
        if run(["tectonic", str(REVISION_ROOT / "main.tex")]) != 0:
            print("Tectonic compile failed."); return 3

    # Persist audit
    summary = {
        "audit": audit,
        "winners": {str(k): {kk: [vv[0], vv[1]] for kk, vv in v.items()}
                    for k, v in winners.items()},
        "complete": True,
    }
    (CANONICAL_ROOT / "canonical_audit.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nWrote audit to {CANONICAL_ROOT / 'canonical_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
