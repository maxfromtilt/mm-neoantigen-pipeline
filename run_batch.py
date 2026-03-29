#!/usr/bin/env python3
"""Run binding predictions + vaccine design + enhanced analysis for multiple patients in one process."""
import os
import sys
import subprocess

patients = ["mmrf_1796", "mmrf_1290", "mmrf_1965"]

for pid in patients:
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {pid.upper()}")
    print(f"{'='*60}")

    cand_path = f"output/{pid}_mhcflurry/neoantigen_candidates.csv"
    if not os.path.exists(cand_path):
        print(f"  Skipping {pid}: no candidates file")
        continue

    # Binding predictions
    print(f"\n  [1/3] Binding predictions...")
    r = subprocess.run([
        sys.executable, "-u", "03_predict_binding.py",
        "--input", cand_path,
        "--output-dir", f"output/{pid}_mhcflurry",
    ], timeout=600)
    if r.returncode != 0:
        print(f"  FAILED binding for {pid}")
        continue

    # Vaccine design
    print(f"\n  [2/3] Vaccine design...")
    binding_path = f"output/{pid}_mhcflurry/binding_predictions.csv"
    if os.path.exists(binding_path):
        subprocess.run([
            sys.executable, "-u", "04_design_vaccine.py",
            "--input", binding_path,
            "--output-dir", f"output/{pid}_mhcflurry",
        ], timeout=120)

    # Enhanced analysis
    print(f"\n  [3/3] Enhanced analysis...")
    subprocess.run([
        sys.executable, "-u", "07_enhanced_analysis.py",
        "--patient", pid,
    ], timeout=300)

    print(f"\n  DONE: {pid.upper()}")

print("\n\nALL PATIENTS COMPLETE")
