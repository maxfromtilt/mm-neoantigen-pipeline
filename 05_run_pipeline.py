#!/usr/bin/env python3
"""
Pipeline Orchestrator
=====================
Runs the full Multiple Myeloma neoantigen vaccine design pipeline end-to-end.

Steps:
  1. Fetch MMRF CoMMpass genomic data from GDC API
  2. Parse mutations and generate neoantigen candidates
  3. Predict MHC binding and rank candidates
  4. Design personalized mRNA vaccine construct

Usage:
    python 05_run_pipeline.py [--config config.yaml] [--skip-fetch]
    python 05_run_pipeline.py --step 3   # Run from step 3 onward
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

import yaml


STEPS = [
    {
        "number": 1,
        "name": "Fetch MMRF CoMMpass Data",
        "script": "01_fetch_mmrf_data.py",
        "description": "Download genomic data from NCI GDC API",
        "outputs": ["data/mmrf_mutations.csv", "data/mmrf_cases.csv"],
    },
    {
        "number": 2,
        "name": "Parse Mutations",
        "script": "02_parse_mutations.py",
        "description": "Filter mutations and generate neoantigen candidates (with UniProt sequence lookup)",
        "outputs": ["data/neoantigen_candidates.csv"],
    },
    {
        "number": 3,
        "name": "Predict MHC Binding",
        "script": "03_predict_binding.py",
        "description": "Predict peptide-MHC binding affinity and rank candidates",
        "outputs": ["data/binding_predictions.csv"],
    },
    {
        "number": 4,
        "name": "Design Vaccine",
        "script": "04_design_vaccine.py",
        "description": "Design mRNA vaccine construct from top candidates",
        "outputs": ["output/vaccine_report.txt", "output/vaccine_mrna.fasta"],
    },
    {
        "number": 5,
        "name": "Validate Pipeline",
        "script": "06_validate_pipeline.py",
        "description": "Run benchmark validation with COSMIC/IEDB cross-referencing and safety screening",
        "outputs": ["output/validation/benchmark_results.csv", "output/validation/validation_vaccine_report.txt"],
    },
]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_dependencies():
    """Check that required Python packages are installed."""
    required = ["pandas", "numpy", "yaml", "requests", "tqdm"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  WARNING: Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install -r requirements.txt")
        return False
    return True


def check_step_outputs(step: dict) -> bool:
    """Check if a step's output files already exist."""
    return all(os.path.exists(p) for p in step["outputs"])


def run_step(step: dict, pipeline_dir: str, extra_args: list = None) -> bool:
    """Run a single pipeline step."""
    script = os.path.join(pipeline_dir, step["script"])
    if not os.path.exists(script):
        print(f"  ERROR: Script not found: {script}")
        return False

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'━' * 70}")
    print(f"  STEP {step['number']}: {step['name']}")
    print(f"  {step['description']}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'━' * 70}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=pipeline_dir,
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"\n  STEP {step['number']} FAILED (exit code {result.returncode})")
            print(f"  Elapsed: {elapsed:.1f}s")
            return False

        print(f"\n  STEP {step['number']} COMPLETED in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  STEP {step['number']} ERROR: {e}")
        print(f"  Elapsed: {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the MM neoantigen vaccine design pipeline"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step (1-4)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip step 1 (data fetching) if data exists")
    parser.add_argument("--max-cases", type=int, default=50,
                        help="Max cases to fetch from GDC")
    parser.add_argument("--max-mutations", type=int, default=10000,
                        help="Max mutations to fetch from GDC")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without executing")
    args = parser.parse_args()

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))

    print("╔" + "═" * 68 + "╗")
    print("║" + "  Multiple Myeloma Neoantigen Vaccine Pipeline".center(68) + "║")
    print("║" + "  End-to-End Orchestrator".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Check dependencies
    print("\n[Pre-flight] Checking dependencies...")
    check_dependencies()

    # Determine which steps to run
    steps_to_run = [s for s in STEPS if s["number"] >= args.step]

    if args.skip_fetch and steps_to_run and steps_to_run[0]["number"] == 1:
        if check_step_outputs(steps_to_run[0]):
            print("\n[Skip] Step 1 outputs exist, skipping data fetch.")
            steps_to_run = steps_to_run[1:]
        else:
            print("\n[Info] Step 1 outputs not found, fetching data.")

    # Summary
    print(f"\n[Plan] Running {len(steps_to_run)} steps:")
    for step in steps_to_run:
        existing = " (outputs exist)" if check_step_outputs(step) else ""
        print(f"  {step['number']}. {step['name']}{existing}")

    if args.dry_run:
        print("\n[Dry Run] Exiting without executing.")
        return

    # Run pipeline
    total_start = time.time()
    completed = 0

    for step in steps_to_run:
        extra_args = []
        if step["number"] == 1:
            extra_args = [
                "--max-cases", str(args.max_cases),
                "--max-mutations", str(args.max_mutations),
            ]

        success = run_step(step, pipeline_dir, extra_args)
        if not success:
            print(f"\n{'!' * 70}")
            print(f"  Pipeline FAILED at step {step['number']}: {step['name']}")
            print(f"  Fix the issue and re-run with: --step {step['number']}")
            print(f"{'!' * 70}")
            sys.exit(1)

        completed += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'═' * 70}")
    print(f"  PIPELINE COMPLETE — {completed} steps in {total_elapsed:.1f}s")
    print(f"{'═' * 70}")
    print(f"\n  Results:")
    print(f"    data/mmrf_mutations.csv         Raw mutation data")
    print(f"    data/neoantigen_candidates.csv   Filtered candidates")
    print(f"    data/binding_predictions.csv     Binding predictions & rankings")
    print(f"    output/vaccine_report.txt        Vaccine design report")
    print(f"    output/vaccine_mrna.fasta        mRNA sequence")
    print(f"    output/vaccine_construct.json     Construct metadata")
    print(f"\n  Read the full report: cat output/vaccine_report.txt")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
