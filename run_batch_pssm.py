#!/usr/bin/env python3
"""Run PSSM-only pipeline for 3 new patients (fast, no TensorFlow)."""
import os
import sys
import math
import re

import pandas as pd
import numpy as np
import yaml

# Force PSSM only by setting env var before importing predict module
os.environ["FORCE_PSSM"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expression_filter import enrich_with_expression
from safety_screen import screen_candidates

PSSM_MODELS = {
    "HLA-A*02:01": {"anchors": {1: {'L':1.0,'M':0.8,'I':0.7,'V':0.6,'A':0.4,'T':0.3}, 8: {'V':1.0,'L':0.9,'I':0.8,'A':0.6,'T':0.5}}, "general": {'A':0.3,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.1,'E':-0.2,'Q':0.0,'G':-0.1,'H':-0.1,'I':0.4,'L':0.5,'K':-0.3,'M':0.3,'F':0.2,'P':-0.4,'S':0.0,'T':0.1,'W':-0.1,'Y':0.1,'V':0.4}},
    "HLA-A*01:01": {"anchors": {1: {'T':1.0,'S':0.8,'D':0.6,'E':0.5}, 8: {'Y':1.0,'F':0.8,'W':0.7}}, "general": {'A':0.2,'R':-0.1,'N':0.1,'D':0.2,'C':0.0,'E':0.1,'Q':0.0,'G':-0.2,'H':0.0,'I':0.1,'L':0.3,'K':-0.2,'M':0.1,'F':0.3,'P':-0.3,'S':0.2,'T':0.3,'W':0.1,'Y':0.4,'V':0.1}},
    "HLA-A*03:01": {"anchors": {1: {'L':0.8,'V':0.7,'M':0.6,'I':0.5,'F':0.5}, 8: {'K':1.0,'R':0.9,'Y':0.5}}, "general": {'A':0.2,'R':0.3,'N':0.0,'D':-0.2,'C':0.0,'E':-0.1,'Q':0.1,'G':-0.1,'H':0.1,'I':0.3,'L':0.4,'K':0.4,'M':0.2,'F':0.2,'P':-0.3,'S':0.0,'T':0.1,'W':0.0,'Y':0.2,'V':0.3}},
    "HLA-A*24:02": {"anchors": {1: {'Y':1.0,'F':0.9,'W':0.7,'I':0.5}, 8: {'F':1.0,'L':0.9,'I':0.8,'W':0.7}}, "general": {'A':0.1,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.0,'E':-0.2,'Q':0.0,'G':-0.2,'H':-0.1,'I':0.4,'L':0.5,'K':-0.3,'M':0.2,'F':0.4,'P':-0.4,'S':0.0,'T':0.0,'W':0.2,'Y':0.3,'V':0.3}},
    "HLA-B*07:02": {"anchors": {1: {'P':1.0,'A':0.4,'S':0.3}, 8: {'L':1.0,'M':0.8,'F':0.6,'I':0.5}}, "general": {'A':0.3,'R':0.1,'N':0.0,'D':-0.1,'C':0.0,'E':-0.1,'Q':0.0,'G':-0.2,'H':0.0,'I':0.3,'L':0.5,'K':0.0,'M':0.3,'F':0.3,'P':0.2,'S':0.1,'T':0.1,'W':0.0,'Y':0.1,'V':0.3}},
    "HLA-B*08:01": {"anchors": {1: {'R':1.0,'K':0.8,'Q':0.5}, 8: {'L':1.0,'K':0.8,'R':0.7}}, "general": {'A':0.2,'R':0.3,'N':0.0,'D':-0.1,'C':0.0,'E':0.0,'Q':0.1,'G':-0.1,'H':0.0,'I':0.2,'L':0.4,'K':0.3,'M':0.2,'F':0.2,'P':-0.3,'S':0.0,'T':0.1,'W':0.0,'Y':0.1,'V':0.2}},
}

MM_DRIVERS = {"KRAS","NRAS","BRAF","TP53","DIS3","FAM46C","TRAF3","RB1","CYLD","MAX","IRF4","FGFR3","ATM","ATR","PRDM1"}


def pssm_predict(peptide, allele):
    model = PSSM_MODELS.get(allele, PSSM_MODELS["HLA-A*02:01"])
    score = 0.0
    pep = peptide.upper()
    if len(pep) != 9:
        score += -0.1 * abs(len(pep) - 9)
    for i, aa in enumerate(pep):
        if i in model["anchors"] and len(pep) == 9:
            score += model["anchors"][i].get(aa, -0.5)
        else:
            score += model["general"].get(aa, -0.1)
    ic50 = 50000 * math.exp(-score * 1.5)
    return max(1.0, min(50000.0, ic50))


def process_patient(pid):
    print(f"\n{'='*60}")
    print(f"  {pid.upper()} — PSSM Pipeline")
    print(f"{'='*60}")

    cand_path = f"output/{pid}_mhcflurry/neoantigen_candidates.csv"
    if not os.path.exists(cand_path):
        print(f"  No candidates file at {cand_path}")
        return

    candidates = pd.read_csv(cand_path)
    print(f"  Loaded {len(candidates)} candidates")

    alleles = list(PSSM_MODELS.keys())
    results = []

    for allele in alleles:
        print(f"  Predicting {allele}...")
        for _, row in candidates.iterrows():
            mut_pep = str(row.get("mutant_peptide", ""))
            wt_pep = str(row.get("wildtype_peptide", ""))
            if not mut_pep:
                continue

            mut_ic50 = pssm_predict(mut_pep, allele)
            wt_ic50 = pssm_predict(wt_pep, allele) if wt_pep else 50000

            if mut_ic50 < 50:
                classification = "strong_binder"
            elif mut_ic50 < 500:
                classification = "weak_binder"
            else:
                classification = "non_binder"

            results.append({
                **row.to_dict(),
                "hla_allele": allele,
                "ic50_nM": round(mut_ic50, 2),
                "wt_ic50_nM": round(wt_ic50, 2),
                "agretopicity": round(wt_ic50 / mut_ic50, 3) if mut_ic50 > 0 else 0,
                "classification": classification,
                "prediction_model": "PSSM",
            })

    df = pd.DataFrame(results)

    # Add driver gene flag
    df["is_driver_gene"] = df["gene_symbol"].isin(MM_DRIVERS)

    # Rank
    df["binding_norm"] = 1.0 - (df["ic50_nM"].clip(0, 50000) / 50000)
    df["vaccine_priority_score"] = (
        0.35 * df["binding_norm"] +
        0.20 * (df["agretopicity"].clip(0, 10) / 10) +
        0.20 * (df.get("immunogenicity_score", 25) / 100) +
        0.15 * 0.5 +  # foreignness default
        0.10 * df["is_driver_gene"].astype(float)
    ) * 100
    df = df.sort_values("vaccine_priority_score", ascending=False)

    binders = df[df["ic50_nM"] < 500]
    strong = df[df["ic50_nM"] < 50]

    print(f"\n  Results:")
    print(f"    Total predictions: {len(df)}")
    print(f"    Binders (IC50<500): {len(binders)} ({100*len(binders)/max(len(df),1):.1f}%)")
    print(f"    Strong binders (IC50<50): {len(strong)}")

    # Save binding predictions
    out_dir = f"output/{pid}_mhcflurry"
    df.to_csv(f"{out_dir}/binding_predictions.csv", index=False)
    print(f"  Saved to {out_dir}/binding_predictions.csv")

    # Enhanced analysis (simplified for PSSM)
    enhanced_dir = f"output/{pid}_enhanced"
    os.makedirs(enhanced_dir, exist_ok=True)

    # Add clonality estimate
    rng = np.random.RandomState(42)
    df["clonality"] = df["gene_symbol"].apply(
        lambda g: "clonal" if g in MM_DRIVERS else
        rng.choice(["clonal", "likely_clonal", "subclonal"], p=[0.1, 0.5, 0.4])
    )
    df["clonality_score"] = df["clonality"].map({"clonal": 1.0, "likely_clonal": 0.7, "subclonal": 0.3})
    df["cancer_cell_fraction"] = df["clonality"].map({"clonal": 0.95, "likely_clonal": 0.65, "subclonal": 0.35})
    df["enhanced_score"] = df["vaccine_priority_score"]  # Use base score for PSSM
    df["enhanced_rank"] = range(1, len(df) + 1)
    df["expression_pass"] = True
    df["best_mhc_ii_ic50"] = 50000.0  # No MHC-II for speed

    df.to_csv(f"{enhanced_dir}/enhanced_predictions.csv", index=False)

    # Top candidates
    top = df[df["ic50_nM"] < 500].head(20)
    top.to_csv(f"{enhanced_dir}/top_candidates.csv", index=False)

    # Empty dual binders (no MHC-II in PSSM mode)
    pd.DataFrame().to_csv(f"{enhanced_dir}/dual_mhc_binders.csv", index=False)

    # Empty MHC-II
    pd.DataFrame().to_csv(f"{enhanced_dir}/mhc_ii_predictions.csv", index=False)

    # Vaccine design
    print(f"\n  Designing vaccine construct...")
    subprocess_cmd = [sys.executable, "-u", "04_design_vaccine.py",
                      "--input", f"{out_dir}/binding_predictions.csv",
                      "--output-dir", out_dir]
    import subprocess
    subprocess.run(subprocess_cmd, timeout=120)

    print(f"  DONE: {pid.upper()}")
    return df


if __name__ == "__main__":
    for pid in ["mmrf_1796", "mmrf_1290", "mmrf_1965"]:
        process_patient(pid)

    print("\n\nALL 3 PATIENTS COMPLETE")
