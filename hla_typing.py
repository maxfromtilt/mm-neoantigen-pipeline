#!/usr/bin/env python3
"""
arCasHLA-Style HLA Typing Enhancement
=====================================
Per-patient HLA allele extraction from MMRF clinical data with
population frequency prior fallback.

Fetches HLA alleles from:
1. MMRF patient clinical data (WES-based HLA calls if available)
2. Population frequency priors (European ancestry defaults from config.yaml)

Outputs: data/hla_typing.json

Usage:
    python hla_typing.py [--patient MMRF_2240] [--population european]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import yaml


# Population frequency priors (European ancestry — most common alleles)
POPULATION_PRIORS = {
    "european": {
        "class_i": [
            "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02",
            "HLA-A*29:02", "HLA-A*26:01", "HLA-A*68:01", "HLA-A*30:01",
            "HLA-B*07:02", "HLA-B*08:01", "HLA-B*44:03", "HLA-B*44:02",
            "HLA-B*35:01", "HLA-B*51:01", "HLA-B*15:01", "HLA-B*40:01",
            "HLA-C*07:01", "HLA-C*05:01", "HLA-C*04:01", "HLA-C*03:04",
            "HLA-C*06:02", "HLA-C*02:02",
        ],
        "class_ii": [
            "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01",
            "HLA-DRB1*07:01", "HLA-DRB1*11:01", "HLA-DRB1*13:01",
            "HLA-DRB1*15:01", "HLA-DQB1*02:01", "HLA-DQB1*03:01",
            "HLA-DQB1*05:01", "HLA-DQB1*06:02",
        ],
    },
    "global": {
        "class_i": [
            "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*11:01",
            "HLA-A*24:02", "HLA-A*26:01", "HLA-A*29:02", "HLA-A*30:01",
            "HLA-A*33:01", "HLA-A*68:01", "HLA-B*07:02", "HLA-B*08:01",
            "HLA-B*13:02", "HLA-B*14:02", "HLA-B*15:01", "HLA-B*18:01",
            "HLA-B*27:05", "HLA-B*35:01", "HLA-B*38:01", "HLA-B*40:01",
            "HLA-B*44:02", "HLA-B*44:03", "HLA-B*51:01", "HLA-B*52:01",
            "HLA-B*57:01", "HLA-B*58:01", "HLA-C*01:02", "HLA-C*02:02",
            "HLA-C*03:03", "HLA-C*03:04", "HLA-C*04:01", "HLA-C*05:01",
            "HLA-C*06:02", "HLA-C*07:01", "HLA-C*12:03", "HLA-C*14:02",
            "HLA-C*15:02",
        ],
        "class_ii": [
            "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01",
            "HLA-DRB1*04:05", "HLA-DRB1*07:01", "HLA-DRB1*08:01",
            "HLA-DRB1*09:01", "HLA-DRB1*10:01", "HLA-DRB1*11:01",
            "HLA-DRB1*12:01", "HLA-DRB1*13:01", "HLA-DRB1*14:01",
            "HLA-DRB1*15:01", "HLA-DRB1*16:01", "HLA-DQB1*02:01",
            "HLA-DQB1*03:01", "HLA-DQB1*04:01", "HLA-DQB1*05:01",
            "HLA-DQB1*06:02", "HLA-DQB1*06:03", "HLA-DQB1*06:04",
        ],
    },
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline config."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def extract_hla_from_clinical(patient_id: str, cases_df: pd.DataFrame = None) -> dict:
    """
    Extract HLA alleles from MMRF patient clinical data.

    The MMRF dataset may include HLA typing results from WES in the
    associated metadata files. This checks for:
    - HLA-A, HLA-B, HLA-C class I alleles
    - HLA-DRB1, HLA-DQB1 class II alleles
    """
    result = {
        "class_i": [],
        "class_ii": [],
        "confidence": "Low",
        "source": "not_found",
    }

    if cases_df is None:
        cases_path = "data/mmrf_cases.csv"
        if os.path.exists(cases_path):
            cases_df = pd.read_csv(cases_path)

    if cases_df is None or cases_df.empty:
        return result

    pid = patient_id.upper().replace("MMRF_", "").replace("MMrf_", "")
    match = cases_df[cases_df["submitter_id"].str.upper().str.contains(pid, na=False)]

    if match.empty:
        return result

    # Check for HLA columns
    hla_cols = [c for c in match.columns if "hla" in c.lower()]
    if hla_cols:
        row = match.iloc[0]
        for col in hla_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                if "a" in col.lower() or "b" in col.lower() or "c" in col.lower():
                    result["class_i"].append(str(val).strip())
                else:
                    result["class_ii"].append(str(val).strip())
        if result["class_i"] or result["class_ii"]:
            result["confidence"] = "High (WES-based)"
            result["source"] = "mmrf_clinical"
            return result

    return result


def get_population_priors(population: str = "european") -> dict:
    """Return the population prior HLA alleles."""
    priors = POPULATION_PRIORS.get(population.lower(), POPULATION_PRIORS["european"])
    return {
        "class_i": priors["class_i"],
        "class_ii": priors["class_ii"],
        "confidence": "Medium (population prior)",
        "source": f"population_frequency_{population}",
    }


def build_patient_hla(patient_id: str, population: str = "european") -> dict:
    """
    Build the complete HLA type for a patient.

    Priority:
    1. WES-based HLA calls from MMRF clinical data (High confidence)
    2. Population frequency priors from config or builtin (Medium confidence)
    """
    # Try to extract from MMRF clinical data
    clinical_hla = extract_hla_from_clinical(patient_id)

    if clinical_hla.get("class_i") or clinical_hla.get("class_ii"):
        # Deduplicate
        clinical_hla["class_i"] = list(dict.fromkeys(clinical_hla["class_i"]))
        clinical_hla["class_ii"] = list(dict.fromkeys(clinical_hla["class_ii"]))
        return clinical_hla

    # Fall back to population priors
    result = get_population_priors(population)

    # Try to get from config as well
    config = load_config()
    config_priors = config.get("mutations", {}).get("hla_typing", {}).get(
        "population_prior_alleles", {}
    )
    if config_priors.get("class_i"):
        result["class_i"] = config_priors["class_i"][:6]  # Limit to top 6
    if config_priors.get("class_ii"):
        result["class_ii"] = config_priors["class_ii"][:4]  # Limit to top 4

    return result


def build_hla_typing_json(patient_ids: list = None, population: str = "european") -> dict:
    """Build HLA typing dict for all known patients."""
    if patient_ids is None:
        # Find all patients from mutation files
        data_dir = Path("data")
        patient_ids = []
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.name.startswith("mmrf_") and f.name.endswith("_mutations.csv"):
                    pid = f.stem.replace("mmrf_", "")
                    patient_ids.append(pid)

    hla_dict = {}
    for pid in patient_ids:
        hla_dict[pid] = build_patient_hla(pid, population)

    return hla_dict


def main():
    parser = argparse.ArgumentParser(description="HLA typing for MMRF patients")
    parser.add_argument("--patient", default=None, help="Specific patient ID")
    parser.add_argument("--population", default="european",
                        choices=["european", "global"],
                        help="Population priors to use as fallback")
    parser.add_argument("--all-patients", action="store_true",
                        help="Build HLA for all patients in data/")
    parser.add_argument("--output", default="data/hla_typing.json",
                        help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.all_patients:
        hla_dict = build_hla_typing_json(population=args.population)
    elif args.patient:
        hla_dict = {args.patient: build_patient_hla(args.patient, args.population)}
        print(f"\nHLA typing for {args.patient}:")
        hla = hla_dict[args.patient]
        print(f"  Confidence: {hla.get('confidence')}")
        print(f"  Source: {hla.get('source')}")
        print(f"  Class I: {', '.join(hla.get('class_i', [])[:6])}")
        print(f"  Class II: {', '.join(hla.get('class_ii', [])[:4])}")
    else:
        print("Specify --patient or --all-patients")
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(hla_dict, f, indent=2)
    print(f"\nSaved HLA typing to: {args.output}")


if __name__ == "__main__":
    main()