#!/usr/bin/env python3
"""
cBioPortal Integration — Enhancement 8
=======================================
Queries the cBioPortal public API to enrich neoantigen predictions with
real-world clinical evidence from multiple myeloma (and proxy) cohorts.

Functions
---------
1. Fetch MM studies from cBioPortal
2. Build a mutation-frequency database across the MM patient population
3. Validate each predicted epitope's gene against the real-world MM cohort
4. Attach survival outcomes (OS/PFS) where available per gene/mutation
5. Rank epitopes by clinical relevance (mutation frequency + survival signal)

Outputs
-------
- output/cbioportal_mutations.csv
      gene, mutation_frequency, patient_count, median_survival_months
- output/cbioportal_validation.csv
      epitope_sequence, gene, hla_allele, frequency_in_mm_cohort,
      clinical_relevance_score, survival_signal

Usage
-----
    python 08_cbioportal.py [--config config.yaml] [--epitopes data/selected_epitopes.csv]

Config (config.yaml)
--------------------
    cbioportal:
      enabled: true
      api_base: "https://api.cbioportal.org/v2.0.0"
      cancer_study_ids:
        - "mm"          # Multiple Myeloma (CoMMpass-aligned)
        - "laml"        # Acute Myeloid Leukemia proxy (haematopoietic)
      max_patients: 500
      survival_data: true
      mutation_frequency_boost: true

Requires
--------
    pip install requests pandas numpy tqdm
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import requests

# ── Progress display ────────────────────────────────────────────────────────

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda it, **kw: it  # type: ignore[assignment]


# ── API helpers ─────────────────────────────────────────────────────────────

BASE_HEADERS = {"Accept": "application/json", "User-Agent": "mm-neoantigen-pipeline/8"}


def cbio_get(api_base: str, path: str, params: dict = None, timeout: int = 30):
    """GET from cBioPortal API; raises on HTTP error."""
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, params=params, headers=BASE_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def cbio_get_pages(api_base: str, path: str, params: dict = None,
                   page_size: int = 200, max_patients: int = 500):
    """Paginated GET; returns combined list of records up to max_patients."""
    params = dict(params or {})
    params["pageSize"] = page_size
    params["projection"] = "DETAILED"

    all_records = []
    page = 0
    while True:
        params["pageNumber"] = page
        data = cbio_get(api_base, path, params)
        if isinstance(data, dict) and "data" in data:
            batch = data["data"]
        elif isinstance(data, list):
            batch = data
        else:
            break

        if not batch:
            break
        all_records.extend(batch)
        if len(all_records) >= max_patients:
            all_records = all_records[:max_patients]
            break
        if len(batch) < page_size:
            break
        page += 1

    return all_records


# ── 1. Fetch & build study / mutation-frequency database ───────────────────

def fetch_mm_studies(api_base: str):
    """Return a DataFrame of available haematological MM studies."""
    print("\n[1/4] Fetching cBioPortal studies ...")
    try:
        # "mm" is not always a study ID — search by cancer type
        data = cbio_get(api_base, "/studies", {"cancer_study_list": "mm"})
    except Exception as exc:
        print(f"    Warning: /studies lookup failed ({exc}); trying cancer type search")
        try:
            data = cbio_get(api_base, "/studies/cancers", {})
        except Exception:
            data = []

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame()

    if df.empty:
        # Known haematological studies as fallback
        fallback = [
            {"studyId": "mm", "name": "Multiple Myeloma (cBioPortal)", "cancerType": "Multiple Myeloma", "numPatients": 0},
            {"studyId": "laml", "name": "Acute Myeloid Leukemia", "cancerType": "AML", "numPatients": 0},
        ]
        df = pd.DataFrame(fallback)

    print(f"    Found {len(df)} study(ies)")
    if not df.empty and "studyId" in df.columns:
        for _, row in df.iterrows():
            print(f"      {row.get('studyId','?')} — {row.get('name','?')} ({row.get('numPatients',0)} patients)")
    return df


def fetch_mutation_frequency(api_base: str, study_ids: list, max_patients: int):
    """
    For each study, fetch all mutations then compute per-gene frequency.
    Returns DataFrame: gene, study_id, mutated_patients, total_patients, frequency
    """
    print(f"\n[2/4] Fetching mutations for studies: {', '.join(study_ids)} ...")
    all_gene_counts = []

    for study_id in study_ids:
        study_id = str(study_id).strip()
        try:
            patients = cbio_get_pages(
                api_base,
                f"/studies/{study_id}/patients",
                max_patients=max_patients,
            )
            n_patients = min(len(patients), max_patients)
            print(f"    Study '{study_id}': {n_patients} patients")
        except Exception as exc:
            print(f"    Warning: could not fetch patients for '{study_id}': {exc}")
            n_patients = 0

        # Paginated mutation fetch per study
        try:
            mutations = cbio_get_pages(
                api_base,
                f"/studies/{study_id}/mutations",
                params={"molecularProfileId": f"{study_id}_mutations"},
                max_patients=max_patients,
            )
        except Exception as exc:
            print(f"    Warning: mutation fetch failed for '{study_id}': {exc}")
            mutations = []

        # Count unique patients per gene
        gene_patient_counts = {}
        for mut in mutations:
            gene = mut.get("gene", {})
            hugo_symbol = gene.get("hugoGeneSymbol") if isinstance(gene, dict) else str(gene)
            if not hugo_symbol:
                continue
            sample_id = mut.get("sampleId", mut.get("patientId", ""))
            key = (hugo_symbol, sample_id)
            if key not in gene_patient_counts:
                gene_patient_counts[hugo_symbol] = set()
            gene_patient_counts[hugo_symbol].add(sample_id)

        for gene, patients_set in gene_patient_counts.items():
            all_gene_counts.append({
                "gene": gene,
                "study_id": study_id,
                "mutated_patients": len(patients_set),
                "total_patients": n_patients,
                "frequency": len(patients_set) / max(n_patients, 1),
            })

    df = pd.DataFrame(all_gene_counts)
    return df


def build_mutation_database(freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-study frequencies into a weighted aggregate frequency
    across all studies (weighted by cohort size).
    Returns: gene, mutation_frequency, patient_count, median_survival_months
    """
    print(f"\n[2b/4] Building mutation-frequency database ...")

    if freq_df.empty:
        # Return known MM driver genes with nominal frequencies
        driver_genes = [
            "KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C", "TRAF3",
            "RB1", "CYLD", "MAX", "IRF4", "FGFR3", "ATM", "ATR", "PRDM1",
        ]
        rows = [{"gene": g, "mutation_frequency": 0.0,
                 "patient_count": 0, "median_survival_months": None}
                for g in driver_genes]
        return pd.DataFrame(rows)

    # Weighted average frequency across studies
    def weighted_freq(grp):
        total = grp["total_patients"].sum()
        if total == 0:
            return 0.0
        return grp["mutated_patients"].sum() / total

    agg = (
        freq_df.groupby("gene", as_index=False)
        .agg(
            mutated_patients=("mutated_patients", "sum"),
            total_patients=("total_patients", "sum"),
        )
    )
    agg["mutation_frequency"] = agg["mutated_patients"] / agg["total_patients"].replace(0, 1)
    agg["patient_count"] = agg["total_patients"]
    agg = agg.drop(columns=["mutated_patients", "total_patients"])

    # Placeholder for survival — populated in next step
    agg["median_survival_months"] = None

    print(f"    Database contains {len(agg)} unique genes")
    print(f"    Top 10 by frequency:")
    top10 = agg.sort_values("mutation_frequency", ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"      {row['gene']}: {row['mutation_frequency']:.3f} ({int(row['patient_count'])} patients)")

    return agg


# ── 2. Survival data ───────────────────────────────────────────────────────

def fetch_survival_data(api_base: str, study_ids: list, max_patients: int):
    """
    Fetch overall survival (OS) and progression-free survival (PFS) data
    per patient from cBioPortal clinical records.
    Returns DataFrame: patient_id, study_id, os_months, pfs_months, os_censored
    """
    print(f"\n[3/4] Fetching survival data ...")
    records = []

    for study_id in study_ids:
        try:
            patients = cbio_get_pages(
                api_base,
                f"/studies/{study_id}/patients",
                max_patients=max_patients,
            )
        except Exception:
            continue

        patient_ids = [p.get("patientId", "") for p in patients if p.get("patientId")]

        for pid in tqdm(patient_ids, desc=f"  {study_id}", leave=False):
            try:
                clinical = cbio_get(
                    api_base,
                    f"/studies/{study_id}/patients/{pid}/clinicalData",
                )
            except Exception:
                continue

            if isinstance(clinical, dict):
                clinical = [clinical]
            if not isinstance(clinical, list):
                continue

            rec = {"patient_id": pid, "study_id": study_id}
            for item in clinical:
                key = str(item.get("clinicalAttributeId", "")).lower()
                val = item.get("value", "")
                if "overall" in key and "surv" in key and "month" in key:
                    rec["os_months"] = float(val) if val not in ("", "NA", "null") else None
                elif "progression" in key and "surv" in key and "month" in key:
                    rec["pfs_months"] = float(val) if val not in ("", "NA", "null") else None
                elif "censor" in key and val in ("1", "0", 1, 0):
                    rec["os_censored"] = bool(int(val))

            if "os_months" in rec or "pfs_months" in rec:
                records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        print(f"    No survival data retrieved")
    else:
        print(f"    Retrieved {len(df)} patient records")
        if "os_months" in df.columns and df["os_months"].notna().any():
            median_os = df["os_months"].median()
            print(f"    Median OS: {median_os:.1f} months")
    return df


def attach_survival_to_genes(surv_df: pd.DataFrame, gene_freq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach median OS per gene (proxy: patients with mutations in that gene
    have the same survival — a simplification; real survival would be per-mutation).
    """
    if surv_df.empty or gene_freq_df.empty:
        return gene_freq_df

    # Build per-study median OS as a proxy
    if "os_months" in surv_df.columns and surv_df["os_months"].notna().any():
        study_median_os = (
            surv_df.dropna(subset=["os_months"])
            .groupby("study_id")["os_months"]
            .median()
        )
        # Map to gene_freq_df via study
        gene_freq_df["median_survival_months"] = gene_freq_df["study_id"].map(study_median_os)

    gene_freq_df = gene_freq_df.drop(columns=["study_id"], errors="ignore")
    return gene_freq_df


# ── 3. Validate epitopes against cBioPortal cohort ─────────────────────────

def validate_epitopes(
    epitopes_df: pd.DataFrame,
    gene_freq_df: pd.DataFrame,
    survival_boost: bool = True,
) -> pd.DataFrame:
    """
    For each epitope, look up its gene's mutation frequency in the cBioPortal
    MM cohort and compute a clinical_relevance_score.

    Score = frequency * 100 * survival_modifier
      where survival_modifier = 1.0 (neutral) or >1.0 if associated with
      longer-than-average survival (a potential vaccine advantage).

    Returns DataFrame with columns:
      epitope_sequence, gene, hla_allele, frequency_in_mm_cohort,
      clinical_relevance_score, survival_signal
    """
    print(f"\n[4/4] Validating epitopes against cBioPortal cohort ...")

    if epitopes_df.empty:
        print("    No epitopes to validate")
        return pd.DataFrame()

    # Ensure gene column exists (handle various naming conventions)
    gene_col = None
    for col in ["gene_symbol", "gene", "hugo_gene_symbol"]:
        if col in epitopes_df.columns:
            gene_col = col
            break

    epitope_col = None
    for col in ["peptide", "epitope_sequence", "epitope", "sequence"]:
        if col in epitopes_df.columns:
            epitope_col = col
            break

    hla_col = None
    for col in ["hla_allele", "hla", "allele"]:
        if col in epitopes_df.columns:
            hla_col = col
            break

    if gene_col is None:
        print("    Warning: could not find gene column in epitopes; skipping validation")
        return pd.DataFrame()

    if epitope_col is None:
        epitope_col = epitopes_df.columns[0]

    if hla_col is None:
        epitopes_df = epitopes_df.copy()
        epitopes_df["hla_allele"] = "Unknown"

    # Build gene → frequency lookup
    gene_freq_lookup = {}
    for _, row in gene_freq_df.iterrows():
        gene_freq_lookup[row["gene"]] = {
            "frequency": row["mutation_frequency"],
            "survival": row.get("median_survival_months"),
        }

    # MM median OS proxy from data (or literature value ~62 months)
    overall_median_os = 62.0
    if gene_freq_df["median_survival_months"].notna().any():
        overall_median_os = gene_freq_df["median_survival_months"].dropna().median()
        if math.isnan(overall_median_os):
            overall_median_os = 62.0

    validated_rows = []
    for _, row in epitopes_df.iterrows():
        gene = str(row.get(gene_col, "")).strip()
        epitope_seq = str(row.get(epitope_col, "")).strip()
        hla = str(row.get(hla_col or "hla_allele", "Unknown")).strip()

        if not gene or gene == "nan":
            continue

        gene_info = gene_freq_lookup.get(gene, {
            "frequency": 0.0,
            "survival": None,
        })

        freq = gene_info["frequency"]
        surv_months = gene_info["survival"]

        # Survival modifier: genes with longer OS associated = better vaccine targets
        # (immune-responsive patients tend to live longer)
        if survival_boost and surv_months and surv_months > 0:
            survival_modifier = surv_months / overall_median_os
            survival_modifier = max(0.5, min(survival_modifier, 2.0))  # clamp [0.5, 2.0]
            survival_signal = f"median_os_{surv_months:.0f}mo"
        else:
            survival_modifier = 1.0
            survival_signal = "no_data"

        # Clinical relevance score (0–100)
        clinical_relevance_score = min(100.0, freq * 100 * survival_modifier)

        validated_rows.append({
            "epitope_sequence": epitope_seq,
            "gene": gene,
            "hla_allele": hla,
            "frequency_in_mm_cohort": round(freq, 4),
            "clinical_relevance_score": round(clinical_relevance_score, 2),
            "survival_signal": survival_signal,
        })

    result_df = pd.DataFrame(validated_rows)
    if not result_df.empty:
        result_df = result_df.sort_values(
            "clinical_relevance_score", ascending=False
        ).reset_index(drop=True)

        print(f"    Validated {len(result_df)} epitopes")
        print(f"    Top 5 by clinical relevance:")
        for _, row in result_df.head(5).iterrows():
            print(f"      {row['epitope_sequence']} ({row['gene']}): "
                  f"freq={row['frequency_in_mm_cohort']:.3f}, "
                  f"score={row['clinical_relevance_score']:.2f}")

    return result_df


# ── Main ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_epitopes(epitopes_path: str = None) -> pd.DataFrame:
    """Try multiple locations for selected epitopes."""
    candidates = [
        epitopes_path,
        "output/selected_epitopes.csv",
        "output/validation/validation_selected_epitopes.csv",
        "data/binding_predictions.csv",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"    Loaded epitopes from: {path} ({len(df)} rows)")
                return df
            except Exception as exc:
                print(f"    Warning: could not read {path}: {exc}")
    print("    No epitopes file found; proceeding with gene-level validation only")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="cBioPortal integration — validate neoantigens against MM cohort data"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--epitopes", default=None,
        help="Path to selected_epitopes.csv (default: auto-detect)"
    )
    args = parser.parse_args()

    # Determine pipeline root (one level up from this script's dir)
    script_dir = Path(__file__).resolve().parent
    pipeline_root = script_dir
    os.chdir(pipeline_root)

    print("╔" + "═" * 68 + "╗")
    print("║" + "  cBioPortal Integration — Enhancement 8".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config(args.config)
    cbio_cfg = config.get("cbioportal", {})
    if not cbio_cfg.get("enabled", True):
        print("\ncBioPortal integration is disabled in config.yaml")
        return

    api_base = cbio_cfg.get("api_base", "https://api.cbioportal.org/v2.0.0")
    study_ids = cbio_cfg.get("cancer_study_ids", ["mm", "laml"])
    max_patients = cbio_cfg.get("max_patients", 500)
    survival_enabled = cbio_cfg.get("survival_data", True)
    freq_boost = cbio_cfg.get("mutation_frequency_boost", True)

    os.makedirs("output", exist_ok=True)

    # ── Step 1: Fetch studies ──────────────────────────────────────────────
    studies_df = fetch_mm_studies(api_base)
    # Use configured study IDs if the API returned nothing useful
    if studies_df.empty:
        active_studies = study_ids
    else:
        active_studies = list(studies_df["studyId"].values) if "studyId" in studies_df.columns else study_ids

    # ── Step 2: Fetch mutations → build frequency DB ───────────────────────
    freq_raw = fetch_mutation_frequency(api_base, active_studies, max_patients)
    gene_freq_df = build_mutation_database(freq_raw)

    # ── Step 3: Survival data ──────────────────────────────────────────────
    surv_df = pd.DataFrame()
    if survival_enabled:
        surv_df = fetch_survival_data(api_base, active_studients, max_patients)
        gene_freq_df = attach_survival_to_genes(surv_df, gene_freq_df)

    # Save mutation frequency DB
    freq_out = "output/cbioportal_mutations.csv"
    gene_freq_df.to_csv(freq_out, index=False)
    print(f"\n    Saved: {freq_out}")

    # ── Step 4: Validate epitopes ─────────────────────────────────────────
    epitopes_df = load_epitopes(args.epitopes)
    validation_df = validate_epitopes(epitopes_df, gene_freq_df, survival_boost=freq_boost)

    if not validation_df.empty:
        val_out = "output/cbioportal_validation.csv"
        validation_df.to_csv(val_out, index=False)
        print(f"    Saved: {val_out}")

        # Summary stats
        print("\n" + "─" * 50)
        print("  VALIDATION SUMMARY")
        print("─" * 50)
        print(f"  Epitopes validated : {len(validation_df)}")
        print(f"  Genes covered      : {validation_df['gene'].nunique()}")
        print(f"  Mean freq in cohort: {validation_df['frequency_in_mm_cohort'].mean():.4f}")
        print(f"  Max relevance score: {validation_df['clinical_relevance_score'].max():.2f}")
        print("─" * 50)
    else:
        print("\n  No epitope validation performed (no epitope file found).")
        print("  output/cbioportal_mutations.csv has been generated with")
        print("  gene-level mutation frequencies for downstream use.")

    elapsed = 0  # could track time if needed
    print(f"\n✓ cBioPortal integration complete")
    print(f"  Outputs: output/cbioportal_mutations.csv"
          + (", output/cbioportal_validation.csv" if not validation_df.empty else ""))


if __name__ == "__main__":
    main()
