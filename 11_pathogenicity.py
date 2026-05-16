#!/usr/bin/env python3
"""
ClinVar / InterVar Pathogenicity Classification
==================================================
Classifies variant pathogenicity using:
1. ClinVar public API (E-utilities) — variant-level evidence
2. InterVar-style rules for variants without ClinVar data

Adds `pathogenicity_class` to neoantigen candidates:
  Pathogenic | Likely Pathogenic | VUS | Likely Benign | Benign

Output: output/pathogenicity.csv

Usage:
    python 11_pathogenicity.py [--input data/neoantigen_candidates.csv]
"""

import os
import sys
import time
import json
import argparse
import requests
import re
from pathlib import Path

import pandas as pd
import numpy as np


CLINVAR_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CLINVAR_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Known oncogenes and tumour suppressors
ONCOGENES = {"KRAS", "NRAS", "BRAF", "FGFR3", "ATM", "ATR", "DIS3", "MAX", "IRF4", "PRDM1"}
TUMOUR_SUPPRESSORS = {"TP53", "RB1", "CYLD", "TRAF3", "FAM46C", "ATM", "ATR", "DIS3"}

# Truncating consequence types
TRUNCATING = {"frameshift_variant", "stop_gained", "frameshift_del", "frameshift_ins",
              "Frame_Shift_Del", "Frame_Shift_Ins", "Nonstop_Mutation"}

# Cache file
CACHE_FILE = "output/pathogenicity_cache.json"
os.makedirs("output", exist_ok=True)


def load_cache():
    """Load cached ClinVar results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    """Save ClinVar cache."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def query_clinvar(gene: str, aa_change: str) -> dict:
    """
    Query ClinVar API for a specific gene+variant combination.

    Returns dict with clinvar_id, clinical_significance, explanation.
    """
    cache = load_cache()
    cache_key = f"{gene}:{aa_change}"
    if cache_key in cache:
        return cache[cache_key]

    # Parse to get clean search terms
    # Try to extract simple AA change from formats like p.V600E, V600E
    clean_aa = re.sub(r'^p\.', '', str(aa_change))

    term = f"{gene}[Gene] AND {clean_aa}[AAChange]"
    params = {
        "db": "clinvar",
        "term": term,
        "retmode": "json",
        "retmax": 5,
    }

    result = {
        "clinvar_id": None,
        "clinical_significance": None,
        "explanation": None,
        "search_term": term,
    }

    try:
        resp = requests.get(CLINVAR_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            result["explanation"] = "No ClinVar record found for this variant"
            cache[cache_key] = result
            save_cache(cache)
            return result

        # Fetch summary for first ID
        result["clinvar_id"] = ids[0]
        summary_params = {
            "db": "clinvar",
            "id": ids[0],
            "retmode": "json",
        }
        sum_resp = requests.get(CLINVAR_SUMMARY, params=summary_params, timeout=10)
        sum_resp.raise_for_status()
        sum_data = sum_resp.json()

        result_set = sum_data.get("result", {})
        uid = str(ids[0])
        if uid in result_set:
            r = result_set[uid]
            sig = r.get("clinical_significance", "")
            desc = r.get("description", "")
            result["clinical_significance"] = sig
            result["explanation"] = f"ClinVar {uid}: {sig}. {desc}".strip()

        cache[cache_key] = result
        save_cache(cache)

    except Exception as e:
        result["explanation"] = f"ClinVar query failed: {e}"
        cache[cache_key] = result
        save_cache(cache)

    return result


def intervar_rules(gene: str, consequence_type: str, aa_change: str) -> dict:
    """
    Apply InterVar-style rules when ClinVar data is unavailable.

    Returns dict with class and explanation.
    """
    consequence_type = str(consequence_type).lower()

    is_truncating = any(t in consequence_type for t in ["frameshift", "stop_gained", "nonstop"])
    is_missense = "missense" in consequence_type
    is_silent = any(s in consequence_type for s in ["synonym", "intron", "utr", "silent", "noncoding"])

    # Rule 1: Truncating in tumour suppressors = Pathogenic
    if is_truncating and gene in TUMOUR_SUPPRESSORS:
        return {
            "class": "Pathogenic",
            "explanation": (
                f"Truncating mutation ({consequence_type}) in tumour suppressor {gene}. "
                "Loss-of-function in tumour suppressors is Pathogenic by InterVar rules."
            ),
        }

    # Rule 2: Truncating in oncogenes = Likely Pathogenic
    if is_truncating and gene in ONCOGENES:
        return {
            "class": "Likely Pathogenic",
            "explanation": (
                f"Truncating mutation ({consequence_type}) in oncogene {gene}. "
                "Loss-of-function in oncogenes is Likely Pathogenic."
            ),
        }

    # Rule 3: Known oncogene missense = Likely Pathogenic
    if is_missense and gene in ONCOGENES:
        return {
            "class": "Likely Pathogenic",
            "explanation": (
                f"Missense mutation in known oncogene {gene}. "
                f"Oncogenic missense variants are Likely Pathogenic by InterVar rules."
            ),
        }

    # Rule 4: Tumour suppressor missense (check known hotspots)
    if is_missense and gene in TUMOUR_SUPPRESSORS:
        return {
            "class": "VUS",
            "explanation": (
                f"Missense mutation in tumour suppressor {gene}. "
                "Requires functional validation — classified as VUS."
            ),
        }

    # Rule 5: Silent/non-coding = Benign/Likely Benign
    if is_silent:
        return {
            "class": "Benign/Likely Benign",
            "explanation": (
                f"Non-coding/synonymous change ({consequence_type}). "
                "Not expected to affect protein function."
            ),
        }

    # Default: VUS
    return {
        "class": "VUS",
        "explanation": (
            f"Insufficient evidence for {gene} {aa_change} ({consequence_type}). "
            "Defaulting to VUS."
        ),
    }


def classify_variant(gene: str, consequence_type: str, aa_change: str,
                     use_clinvar: bool = True) -> dict:
    """
    Classify a variant's pathogenicity.

    Tries ClinVar first, falls back to InterVar rules.
    """
    result = {
        "pathogenicity_class": "VUS",
        "clinvar_id": None,
        "explanation": "Not evaluated",
    }

    if use_clinvar:
        clinvar_result = query_clinvar(gene, aa_change)
        sig = clinvar_result.get("clinical_significance", "")

        if sig:
            # Normalise ClinVar significance terms
            sig_lower = sig.lower()
            if any(k in sig_lower for k in ["pathogenic", "likely pathogenic"]):
                path_class = "Pathogenic" if "pathogenic" in sig_lower and "likely" not in sig_lower else "Likely Pathogenic"
                result["pathogenicity_class"] = path_class
                result["clinvar_id"] = clinvar_result.get("clinvar_id")
                result["explanation"] = clinvar_result.get("explanation", sig)
                return result
            elif "benign" in sig_lower:
                result["pathogenicity_class"] = "Benign/Likely Benign"
                result["clinvar_id"] = clinvar_result.get("clinvar_id")
                result["explanation"] = clinvar_result.get("explanation", sig)
                return result
            elif "uncertain" in sig_lower or sig == "other" or not sig:
                result["pathogenicity_class"] = "VUS"
                result["clinvar_id"] = clinvar_result.get("clinvar_id")
                result["explanation"] = clinvar_result.get("explanation", f"ClinVar: {sig or 'no significance assigned'}")
                return result

        # ClinVar found but no clear classification — still use it as evidence
        if clinvar_result.get("clinvar_id"):
            result["clinvar_id"] = clinvar_result.get("clinvar_id")
            result["explanation"] = clinvar_result.get("explanation", "ClinVar record found but significance unclear")
            # Fall through to InterVar as secondary evidence note

    # Apply InterVar rules
    intervar_result = intervar_rules(gene, consequence_type, aa_change)
    result["pathogenicity_class"] = intervar_result["class"]

    if result["clinvar_id"]:
        result["explanation"] = (
            f"{intervar_result['explanation']} "
            f"[ClinVar:{result['clinvar_id']} found but significance unclear — InterVar rules applied as primary]"
        )
    else:
        result["explanation"] = intervar_result["explanation"]

    return result


def process_file(input_path: str, max_calls: int = 100, use_clinvar: bool = True):
    """
    Process a neoantigen candidates CSV and add pathogenicity classification.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} candidates from {input_path}")

    # Unique gene+aa_change combinations to query
    unique_variants = df[["gene_symbol", "aa_change", "consequence_type"]].drop_duplicates()
    unique_variants = unique_variants[
        unique_variants["gene_symbol"].notna() & unique_variants["aa_change"].notna()
    ]

    print(f"Unique variants to classify: {len(unique_variants)}")
    print(f"ClinVar queries: {'enabled' if use_clinvar else 'disabled'} (max {max_calls})")

    # Load existing cache
    cache = load_cache()

    results = []
    call_count = 0

    for _, row in unique_variants.iterrows():
        gene = str(row["gene_symbol"])
        aa = str(row["aa_change"])
        cons = str(row["consequence_type"]) if pd.notna(row.get("consequence_type")) else ""

        cache_key = f"{gene}:{aa}"
        if cache_key in cache:
            results.append(cache[cache_key])
            continue

        if use_clinvar and call_count < max_calls:
            result = query_clinvar(gene, aa)
            call_count += 1
            time.sleep(0.35)  # Respect NCBI rate limits
        else:
            result = {"clinvar_id": None, "clinical_significance": None, "explanation": "ClinVar skipped (max calls)"}

        # Apply InterVar rules
        intervar = intervar_rules(gene, cons, aa)
        if result.get("clinical_significance"):
            sig_lower = result["clinical_significance"].lower()
            if "pathogenic" in sig_lower and "uncertain" not in sig_lower:
                result["class"] = "Pathogenic" if "pathogenic" in sig_lower and "likely" not in sig_lower else "Likely Pathogenic"
            elif "benign" in sig_lower:
                result["class"] = "Benign/Likely Benign"
            elif "uncertain" in sig_lower:
                result["class"] = "VUS"
            else:
                result["class"] = intervar["class"]
        else:
            result["class"] = intervar["class"]
            result["explanation"] = intervar["explanation"]

        results.append(result)

    # Build results dict keyed by gene:aa_change
    results_dict = {}
    for (_, row), res in zip(unique_variants.iterrows(), results):
        key = f"{row['gene_symbol']}:{row['aa_change']}"
        results_dict[key] = res

    # Apply to full dataframe
    df["pathogenicity_class"] = "VUS"
    df["clinvar_id"] = None
    df["pathogenicity_explanation"] = "Not evaluated"

    for idx, row in df.iterrows():
        key = f"{row['gene_symbol']}:{row['aa_change']}"
        if key in results_dict:
            r = results_dict[key]
            df.at[idx, "pathogenicity_class"] = r.get("class", "VUS")
            df.at[idx, "clinvar_id"] = r.get("clinvar_id")
            df.at[idx, "pathogenicity_explanation"] = r.get("explanation", "")

    # Summary
    print("\n--- Pathogenicity Summary ---")
    print(df["pathogenicity_class"].value_counts().to_string())

    # Save
    output_path = "output/pathogenicity.csv"
    out_cols = ["gene_symbol", "aa_change", "consequence_type",
                "pathogenicity_class", "clinvar_id", "pathogenicity_explanation"]
    available = [c for c in out_cols if c in df.columns]
    df[available].drop_duplicates().to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="ClinVar / InterVar pathogenicity classification")
    parser.add_argument("--input", default="data/neoantigen_candidates.csv",
                        help="Input neoantigen candidates CSV")
    parser.add_argument("--max-calls", type=int, default=100,
                        help="Max ClinVar API calls (NCBI rate limit: 3/sec)")
    parser.add_argument("--no-clinvar", action="store_true",
                        help="Skip ClinVar API, use InterVar rules only")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run 02_parse_mutations.py first.")
        sys.exit(1)

    process_file(args.input, max_calls=args.max_calls, use_clinvar=not args.no_clinvar)


if __name__ == "__main__":
    main()