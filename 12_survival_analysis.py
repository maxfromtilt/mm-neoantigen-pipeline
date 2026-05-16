#!/usr/bin/env python3
"""
Survival Analysis — MMRF Cohort & Per-Patient Overall Survival
=============================================================
Extracts survival data from data/mmrf_cases.csv and outputs:
- Patient-level overall survival (OS) where clinical data exists
- Cohort-level KM statistics for mutation frequency groups
- survival.csv with patient and cohort survival statistics

Usage:
    python 12_survival_analysis.py [--patient MMRF_2240]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def load_cases() -> pd.DataFrame:
    """Load MMRF cases data with survival columns."""
    cases_path = "data/mmrf_cases.csv"
    if not os.path.exists(cases_path):
        print(f"WARNING: {cases_path} not found — survival analysis skipped")
        return pd.DataFrame()
    df = pd.read_csv(cases_path)
    required_cols = ["case_id", "submitter_id", "vital_status",
                     "days_to_death", "days_to_last_follow_up"]
    available = [c for c in required_cols if c in df.columns]
    if not available:
        print("WARNING: Required survival columns not found in mmrf_cases.csv")
        return pd.DataFrame()
    return df


def get_patient_survival(df: pd.DataFrame, patient_id: str) -> dict:
    """Extract survival data for a specific patient."""
    pid = patient_id.upper().replace("MMRF_", "").replace("MMrf_", "")
    match = df[df["submitter_id"].str.upper().str.contains(pid, na=False)]
    if match.empty:
        return {"patient_id": patient_id, "found": False}

    row = match.iloc[0]
    vital = row.get("vital_status", "")
    days_death = row.get("days_to_death")
    days_fu = row.get("days_to_last_follow_up")

    if pd.notna(days_death):
        os_days = days_death
        event = 1  # deceased
    elif pd.notna(days_fu):
        os_days = days_fu
        event = 0 if str(vital).lower() == "alive" else 1
    else:
        os_days = None
        event = None

    result = {
        "patient_id": patient_id,
        "submitter_id": row.get("submitter_id"),
        "found": True,
        "vital_status": vital,
        "event_death": event,
    }
    if os_days is not None:
        result["overall_survival_days"] = int(os_days)
        result["overall_survival_months"] = round(os_days / 30.44, 1)
    return result


def compute_cohort_km_stats(df: pd.DataFrame, gene_col: str = "gene_symbol",
                            mutation_df: pd.DataFrame = None) -> dict:
    """
    Compute Kaplan-Meier maturity-adjusted survival statistics per gene mutation group.
    """
    required_cols = ["case_id", "vital_status", "days_to_death", "days_to_last_follow_up"]
    if not all(c in df.columns for c in required_cols):
        return {}

    cohort_df = df.copy()
    cohort_df["os_days"] = cohort_df["days_to_death"].fillna(cohort_df["days_to_last_follow_up"])
    cohort_df["event"] = (cohort_df["vital_status"].str.lower() == "dead").astype(int)
    cohort_df = cohort_df[cohort_df["os_days"].notna()]

    stats = {
        "cohort_n_patients": len(cohort_df),
        "cohort_n_dead": int(cohort_df["event"].sum()),
        "cohort_median_os_days": float(cohort_df["os_days"].median()),
        "cohort_median_os_months": round(cohort_df["os_days"].median() / 30.44, 1),
        "cohort_mean_os_months": round(cohort_df["os_days"].mean() / 30.44, 1),
    }

    # Per-gene survival if mutation data available
    if mutation_df is not None and gene_col in mutation_df.columns:
        mutated_genes = mutation_df[gene_col].dropna().unique()
        gene_stats = {}
        for gene in mutated_genes:
            # Find patients with this gene mutated
            patient_ids = mutation_df[mutation_df[gene_col] == gene]["case_id"].dropna().unique()
            subset = cohort_df[cohort_df["case_id"].isin(patient_ids)]
            if len(subset) >= 3:
                gene_stats[gene] = {
                    "n_patients": len(subset),
                    "n_dead": int(subset["event"].sum()),
                    "median_os_months": (
                        round(subset["os_days"].median() / 30.44, 1)
                        if not subset["os_days"].isnull().all() else None
                    ),
                    "mean_os_months": round(subset["os_days"].mean() / 30.44, 1),
                }
        stats["gene_survival"] = gene_stats

    return stats


def main():
    parser = argparse.ArgumentParser(description="Survival analysis for MMRF cohort and patients")
    parser.add_argument("--patient", default=None, help="Specific patient ID (e.g. MMRF_2240)")
    parser.add_argument("--output", default="output/survival.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    cases_df = load_cases()
    if cases_df.empty:
        sys.exit(0)

    results = []

    if args.patient:
        # Per-patient survival
        patient_surv = get_patient_survival(cases_df, args.patient)
        results.append(patient_surv)
        print(f"\nPatient survival for {args.patient}:")
        if patient_surv.get("found"):
            print(f"  Vital status: {patient_surv.get('vital_status')}")
            print(f"  OS: {patient_surv.get('overall_survival_months', 'N/A')} months")
            print(f"  Event (death): {patient_surv.get('event_death')}")
        else:
            print("  Not found in mmrf_cases.csv")

    # Cohort-level stats
    cohort_stats = compute_cohort_km_stats(cases_df)
    print(f"\n--- MMRF Cohort KM Summary ---")
    print(f"  N patients: {cohort_stats.get('cohort_n_patients', 'N/A')}")
    print(f"  N deceased: {cohort_stats.get('cohort_n_dead', 'N/A')}")
    print(f"  Median OS: {cohort_stats.get('cohort_median_os_months', 'N/A')} months")

    # Load mutations for gene-level survival
    mut_df = None
    if os.path.exists("data/mmrf_mutations.csv"):
        try:
            mut_df = pd.read_csv("data/mmrf_mutations.csv")
        except Exception:
            pass

    gene_survival = cohort_stats.get("gene_survival", {})
    if gene_survival:
        gene_rows = []
        for gene, gs in gene_survival.items():
            gene_rows.append({
                "gene_symbol": gene,
                "n_mm_patients_with_gene": gs["n_patients"],
                "n_deceased": gs["n_dead"],
                "median_os_months": gs["median_os_months"],
                "mean_os_months": gs["mean_os_months"],
                "cohort_median_os_months": cohort_stats["cohort_median_os_months"],
            })
        gene_survival_df = pd.DataFrame(gene_rows)
        gene_survival_path = "output/gene_survival_km.csv"
        gene_survival_df.to_csv(gene_survival_path, index=False)
        print(f"  Gene-level survival saved to: {gene_survival_path}")

    # Main output
    output_df = pd.DataFrame(results)
    if not output_df.empty:
        output_df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")

    return cohort_stats


if __name__ == "__main__":
    main()