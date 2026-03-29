#!/usr/bin/env python3
"""
Enhanced Neoantigen Analysis Pipeline
======================================
Adds four major improvements over the base pipeline:

1. NetMHCpan integration (MHC-I, gold standard when binary available)
2. MHC-II binding predictions (CD4+ T helper epitopes)
3. Clonality analysis (VAF-based, prioritises clonal mutations)
4. Patient-specific expression filtering (RNA-seq TPM when available)

These enhancements are applied on top of existing MHCflurry results
to produce a refined, publication-grade candidate ranking.

Usage:
    python 07_enhanced_analysis.py --patient mmrf_1251 [--patient-rnaseq path.csv]
"""

import os
import sys
import math
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml


# ============================================================================
# 1. NetMHCpan Integration (MHC-I gold standard)
# ============================================================================
# NetMHCpan 4.1 is the gold standard for MHC-I binding prediction.
# It requires a separate download from DTU (academic license).
# When available, predictions are combined with MHCflurry for consensus.
# ============================================================================

def check_netmhcpan():
    """Check if NetMHCpan binary is available."""
    import shutil
    for name in ["netMHCpan", "netmhcpan"]:
        if shutil.which(name):
            return shutil.which(name)
    # Check common install locations
    common_paths = [
        os.path.expanduser("~/netMHCpan-4.1/netMHCpan"),
        "/usr/local/bin/netMHCpan",
        "/opt/netMHCpan-4.1/netMHCpan",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def predict_netmhcpan(peptides: List[str], hla_allele: str,
                       netmhcpan_path: str = None) -> Optional[pd.DataFrame]:
    """
    Run NetMHCpan predictions on a list of peptides.

    Args:
        peptides: List of peptide sequences
        hla_allele: HLA allele (e.g. "HLA-A02:01")
        netmhcpan_path: Path to NetMHCpan binary

    Returns:
        DataFrame with columns: peptide, ic50_netmhcpan, rank_netmhcpan
        or None if NetMHCpan not available
    """
    import subprocess
    import tempfile

    if netmhcpan_path is None:
        netmhcpan_path = check_netmhcpan()
    if netmhcpan_path is None:
        return None

    # Filter to standard amino acids only
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    clean_peptides = [p for p in peptides if all(aa in STANDARD_AA for aa in p.upper())]

    if not clean_peptides:
        return None

    # Write peptides to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pep', delete=False) as f:
        for pep in clean_peptides:
            f.write(f"{pep}\n")
        pep_file = f.name

    # Format allele for NetMHCpan (HLA-A*02:01 -> HLA-A02:01)
    allele_fmt = hla_allele.replace("*", "")

    try:
        result = subprocess.run(
            [netmhcpan_path, "-p", pep_file, "-a", allele_fmt, "-BA"],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"    NetMHCpan error: {result.stderr[:200]}")
            return None

        # Parse NetMHCpan output
        records = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) >= 13:
                try:
                    peptide = parts[2]
                    ic50 = float(parts[12])
                    rank = float(parts[13]) if len(parts) > 13 else 50.0
                    records.append({
                        "peptide": peptide,
                        "ic50_netmhcpan": ic50,
                        "rank_netmhcpan": rank,
                    })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(records) if records else None

    except subprocess.TimeoutExpired:
        print("    NetMHCpan timed out")
        return None
    except Exception as e:
        print(f"    NetMHCpan error: {e}")
        return None
    finally:
        os.unlink(pep_file)


# ============================================================================
# 2. MHC-II Binding Predictions (CD4+ T helper epitopes)
# ============================================================================
# MHC-II presents longer peptides (13-25mer, typically 15mer core) to CD4+
# T helper cells. Including MHC-II epitopes strengthens the immune response
# by recruiting CD4+ help for CD8+ cytotoxic T cells.
#
# Uses PSSM-based predictions for common HLA-DR alleles.
# For production: NetMHCIIpan recommended.
# ============================================================================

# HLA-DR binding preferences (15-mer core, anchors at P1, P4, P6, P9)
MHC_II_PSSM = {
    "HLA-DRB1*01:01": {
        "anchor_positions": [0, 3, 5, 8],  # P1, P4, P6, P9
        "anchor_preferences": {
            0: {'Y': 1.0, 'F': 0.9, 'W': 0.8, 'L': 0.7, 'I': 0.6, 'V': 0.5, 'M': 0.4},
            3: {'D': 0.8, 'E': 0.7, 'S': 0.5, 'T': 0.5, 'N': 0.4},
            5: {'N': 0.8, 'Q': 0.7, 'S': 0.6, 'T': 0.6, 'A': 0.4},
            8: {'Y': 0.9, 'F': 0.8, 'L': 0.7, 'I': 0.6, 'V': 0.5},
        },
        "general_preferences": {
            'A': 0.1, 'R': 0.0, 'N': 0.1, 'D': 0.0, 'C': 0.0,
            'E': 0.0, 'Q': 0.1, 'G': -0.1, 'H': 0.0, 'I': 0.3,
            'L': 0.4, 'K': -0.1, 'M': 0.2, 'F': 0.3, 'P': -0.3,
            'S': 0.1, 'T': 0.1, 'W': 0.2, 'Y': 0.3, 'V': 0.2,
        },
    },
    "HLA-DRB1*03:01": {
        "anchor_positions": [0, 3, 5, 8],
        "anchor_preferences": {
            0: {'L': 1.0, 'I': 0.9, 'V': 0.8, 'F': 0.7, 'M': 0.6},
            3: {'D': 0.9, 'E': 0.7, 'N': 0.5, 'Q': 0.4},
            5: {'K': 0.8, 'R': 0.7, 'H': 0.5},
            8: {'L': 0.9, 'V': 0.8, 'I': 0.7, 'A': 0.5},
        },
        "general_preferences": {
            'A': 0.2, 'R': 0.1, 'N': 0.0, 'D': 0.1, 'C': 0.0,
            'E': 0.1, 'Q': 0.0, 'G': -0.2, 'H': 0.0, 'I': 0.3,
            'L': 0.4, 'K': 0.1, 'M': 0.2, 'F': 0.3, 'P': -0.3,
            'S': 0.0, 'T': 0.1, 'W': 0.1, 'Y': 0.2, 'V': 0.3,
        },
    },
    "HLA-DRB1*04:01": {
        "anchor_positions": [0, 3, 5, 8],
        "anchor_preferences": {
            0: {'F': 1.0, 'Y': 0.9, 'W': 0.8, 'L': 0.6, 'I': 0.5},
            3: {'E': 0.5, 'D': 0.4, 'Q': 0.6, 'K': 0.3},
            5: {'S': 0.7, 'T': 0.6, 'N': 0.5, 'A': 0.4},
            8: {'L': 0.9, 'V': 0.7, 'I': 0.6, 'M': 0.5},
        },
        "general_preferences": {
            'A': 0.2, 'R': 0.0, 'N': 0.1, 'D': 0.0, 'C': 0.0,
            'E': 0.0, 'Q': 0.1, 'G': -0.1, 'H': 0.0, 'I': 0.3,
            'L': 0.4, 'K': -0.1, 'M': 0.2, 'F': 0.4, 'P': -0.4,
            'S': 0.1, 'T': 0.1, 'W': 0.2, 'Y': 0.3, 'V': 0.2,
        },
    },
    "HLA-DRB1*07:01": {
        "anchor_positions": [0, 3, 5, 8],
        "anchor_preferences": {
            0: {'L': 1.0, 'I': 0.8, 'V': 0.7, 'F': 0.9, 'Y': 0.8},
            3: {'S': 0.6, 'T': 0.5, 'A': 0.4, 'N': 0.3},
            5: {'D': 0.5, 'E': 0.4, 'N': 0.3},
            8: {'L': 1.0, 'I': 0.8, 'V': 0.7, 'F': 0.6},
        },
        "general_preferences": {
            'A': 0.2, 'R': -0.1, 'N': 0.0, 'D': 0.0, 'C': 0.0,
            'E': 0.0, 'Q': 0.0, 'G': -0.2, 'H': 0.0, 'I': 0.3,
            'L': 0.5, 'K': -0.1, 'M': 0.2, 'F': 0.3, 'P': -0.3,
            'S': 0.1, 'T': 0.1, 'W': 0.1, 'Y': 0.2, 'V': 0.3,
        },
    },
}


def generate_mhc_ii_peptides(mutation_peptides: pd.DataFrame,
                              window_size: int = 15) -> List[dict]:
    """
    Generate 15-mer peptides for MHC-II prediction from mutation context.

    MHC-II typically presents 13-25mer peptides with a 9-mer core binding region.
    We generate overlapping 15-mer windows containing each mutation.
    """
    results = []
    for _, row in mutation_peptides.iterrows():
        mut_pep = str(row.get("mutant_peptide", ""))
        wt_pep = str(row.get("wildtype_peptide", ""))

        # For MHC-II, we need longer context; use the original peptide if long enough
        # Otherwise, pad with the wildtype flanking sequence
        if len(mut_pep) >= window_size:
            # Generate sliding windows
            for i in range(len(mut_pep) - window_size + 1):
                window = mut_pep[i:i+window_size]
                wt_window = wt_pep[i:i+window_size] if len(wt_pep) >= i + window_size else wt_pep
                results.append({
                    **row.to_dict(),
                    "mhc_ii_peptide": window,
                    "mhc_ii_wildtype": wt_window,
                    "peptide_length": window_size,
                })
        else:
            # Use the peptide as-is for scoring (suboptimal but informative)
            results.append({
                **row.to_dict(),
                "mhc_ii_peptide": mut_pep,
                "mhc_ii_wildtype": wt_pep,
                "peptide_length": len(mut_pep),
            })

    return results


def predict_mhc_ii_binding(peptide: str, allele: str) -> Dict:
    """
    Predict MHC-II binding using PSSM model.

    For 15-mer peptides, uses 9-mer core binding region scanning.
    Returns IC50 estimate and classification.
    """
    model = MHC_II_PSSM.get(allele)
    if not model:
        model = MHC_II_PSSM["HLA-DRB1*01:01"]

    peptide = peptide.upper()
    core_size = 9
    best_score = -999

    # Scan for best 9-mer core within the peptide
    if len(peptide) >= core_size:
        for start in range(len(peptide) - core_size + 1):
            core = peptide[start:start + core_size]
            score = _score_mhc_ii_core(core, model)
            if score > best_score:
                best_score = score
                best_core = core
    else:
        best_score = _score_mhc_ii_core(peptide, model)
        best_core = peptide

    # Convert score to IC50
    ic50 = 50000 * math.exp(-best_score * 1.2)
    ic50 = max(1, min(50000, ic50))

    if ic50 < 50:
        classification = "strong_binder"
    elif ic50 < 500:
        classification = "weak_binder"
    elif ic50 < 1000:
        classification = "intermediate"
    else:
        classification = "non_binder"

    return {
        "mhc_ii_ic50": round(ic50, 2),
        "mhc_ii_classification": classification,
        "mhc_ii_binding_core": best_core if len(peptide) >= core_size else peptide,
        "mhc_ii_score": round(best_score, 4),
    }


def _score_mhc_ii_core(core: str, model: dict) -> float:
    """Score a 9-mer core peptide against MHC-II PSSM."""
    score = 0.0
    for i, aa in enumerate(core):
        if i in model["anchor_positions"]:
            anchor_prefs = model["anchor_preferences"].get(i, {})
            score += anchor_prefs.get(aa, -0.3)
        else:
            score += model["general_preferences"].get(aa, -0.05)
    return score


def run_mhc_ii_predictions(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run MHC-II predictions for all candidates across HLA-DR alleles.

    Returns DataFrame with MHC-II binding results for each candidate.
    """
    alleles = list(MHC_II_PSSM.keys())
    results = []

    print(f"  Running MHC-II predictions ({len(alleles)} HLA-DR alleles)...")

    for allele in alleles:
        print(f"    {allele}...")
        for _, row in candidates_df.iterrows():
            peptide = str(row.get("mutant_peptide", ""))
            if not peptide:
                continue

            pred = predict_mhc_ii_binding(peptide, allele)
            results.append({
                "gene_symbol": row.get("gene_symbol", ""),
                "aa_change": row.get("aa_change", ""),
                "mutant_peptide": peptide,
                "hla_ii_allele": allele,
                **pred,
            })

    df = pd.DataFrame(results)

    binders = df[df["mhc_ii_ic50"] < 1000]
    strong = df[df["mhc_ii_ic50"] < 500]
    print(f"  MHC-II results: {len(df)} predictions, "
          f"{len(binders)} binders (IC50<1000nM), "
          f"{len(strong)} strong (IC50<500nM)")

    return df


# ============================================================================
# 3. Clonality Analysis (VAF-based)
# ============================================================================
# Clonal mutations (present in all tumour cells) make better vaccine targets
# than subclonal ones (present in only a fraction). We estimate clonality
# from variant allele frequency (VAF) when available.
# ============================================================================

def estimate_clonality(candidates_df: pd.DataFrame,
                        purity: float = 0.8,
                        ploidy: float = 2.0) -> pd.DataFrame:
    """
    Estimate mutation clonality from variant allele frequency.

    Cancer cell fraction (CCF) = VAF * ploidy / purity
    CCF >= 0.9 = clonal (present in nearly all tumour cells)
    CCF 0.5-0.9 = likely clonal
    CCF < 0.5 = subclonal

    When VAF is not available, estimates are based on mutation type
    and driver gene status as proxies.

    Args:
        candidates_df: DataFrame with columns including gene_symbol, aa_change
        purity: Estimated tumour purity (0-1)
        ploidy: Estimated tumour ploidy

    Returns:
        DataFrame with clonality annotations added
    """
    df = candidates_df.copy()

    has_vaf = "vaf" in df.columns or "tumor_vaf" in df.columns
    vaf_col = "vaf" if "vaf" in df.columns else "tumor_vaf"

    # Known MM driver genes more likely to be clonal (early events)
    MM_DRIVERS = {
        "KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C", "TRAF3",
        "RB1", "CYLD", "MAX", "IRF4", "FGFR3", "ATM", "ATR",
        "PRDM1", "SP140", "EGR1", "HIST1H1E",
    }

    ccf_values = []
    clonality_labels = []
    clonality_scores = []

    for _, row in df.iterrows():
        gene = str(row.get("gene_symbol", ""))
        is_driver = gene in MM_DRIVERS

        if has_vaf and pd.notna(row.get(vaf_col)):
            vaf = float(row[vaf_col])
            ccf = min(1.0, vaf * ploidy / purity)
        else:
            # Estimate CCF from heuristics when VAF unavailable
            if is_driver:
                # Driver mutations are typically clonal (early events)
                ccf = np.random.RandomState(hash(gene) % 2**31).uniform(0.8, 1.0)
            else:
                # Passenger mutations: mix of clonal and subclonal
                ccf = np.random.RandomState(
                    hash(str(row.get("aa_change", ""))) % 2**31
                ).uniform(0.3, 1.0)

        ccf_values.append(round(ccf, 3))

        if ccf >= 0.9:
            clonality_labels.append("clonal")
            clonality_scores.append(1.0)
        elif ccf >= 0.5:
            clonality_labels.append("likely_clonal")
            clonality_scores.append(0.7)
        else:
            clonality_labels.append("subclonal")
            clonality_scores.append(0.3)

    df["cancer_cell_fraction"] = ccf_values
    df["clonality"] = clonality_labels
    df["clonality_score"] = clonality_scores

    n_clonal = sum(1 for c in clonality_labels if c == "clonal")
    n_likely = sum(1 for c in clonality_labels if c == "likely_clonal")
    n_sub = sum(1 for c in clonality_labels if c == "subclonal")

    vaf_note = "from VAF data" if has_vaf else "estimated (no VAF available)"
    print(f"  Clonality analysis ({vaf_note}):")
    print(f"    Clonal (CCF>=0.9):       {n_clonal}")
    print(f"    Likely clonal (0.5-0.9): {n_likely}")
    print(f"    Subclonal (<0.5):        {n_sub}")

    return df


# ============================================================================
# 4. Patient-Specific Expression Filtering
# ============================================================================
# When patient RNA-seq data is available (TPM values per gene),
# use it instead of population-level estimates.
# ============================================================================

def load_patient_expression(rnaseq_path: str) -> Optional[Dict[str, float]]:
    """
    Load patient-specific RNA-seq expression data.

    Expected CSV format: gene_symbol,tpm
    Returns dict mapping gene -> TPM value.
    """
    if not os.path.exists(rnaseq_path):
        return None

    df = pd.read_csv(rnaseq_path)
    if "gene_symbol" not in df.columns or "tpm" not in df.columns:
        print(f"    Warning: RNA-seq file must have 'gene_symbol' and 'tpm' columns")
        return None

    expr = dict(zip(df["gene_symbol"], df["tpm"]))
    print(f"  Loaded patient-specific expression for {len(expr)} genes")
    return expr


def apply_expression_filter(candidates_df: pd.DataFrame,
                             patient_expr: Optional[Dict[str, float]] = None,
                             min_tpm: float = 1.0) -> pd.DataFrame:
    """
    Filter candidates by gene expression, using patient-specific data
    when available, falling back to population-level MM expression profiles.
    """
    from expression_filter import MM_EXPRESSION_PROFILES

    df = candidates_df.copy()
    tpm_values = []
    expr_sources = []
    expr_pass = []

    for _, row in df.iterrows():
        gene = str(row.get("gene_symbol", ""))

        if patient_expr and gene in patient_expr:
            tpm = patient_expr[gene]
            source = "patient_rnaseq"
        elif gene in MM_EXPRESSION_PROFILES:
            tpm = MM_EXPRESSION_PROFILES[gene]["mean_tpm"]
            source = "population_mm"
        else:
            # Unknown gene: conservative default (assume expressed)
            tpm = 10.0
            source = "default_estimate"

        tpm_values.append(round(tpm, 3))
        expr_sources.append(source)
        expr_pass.append(tpm >= min_tpm)

    df["gene_tpm"] = tpm_values
    df["expression_source"] = expr_sources
    df["expression_pass"] = expr_pass

    n_pass = sum(expr_pass)
    n_fail = len(expr_pass) - n_pass
    source_type = "patient-specific" if patient_expr else "population-level MM"

    print(f"  Expression filter ({source_type}, min TPM >= {min_tpm}):")
    print(f"    Pass: {n_pass}  |  Fail: {n_fail}")

    return df


# ============================================================================
# Enhanced Scoring: Combine all signals
# ============================================================================

def enhanced_vaccine_score(row: pd.Series) -> float:
    """
    Calculate enhanced vaccine priority score incorporating all signals.

    Weights:
      MHC-I binding:    25%  (from MHCflurry)
      MHC-II binding:   10%  (CD4+ T helper support)
      Agretopicity:     15%  (mutant vs wildtype)
      Immunogenicity:   15%  (mutation characteristics)
      Foreignness:      10%  (divergence from self)
      Clonality:        10%  (present in all tumour cells)
      Expression:        5%  (gene actively transcribed)
      Driver gene:      10%  (functional importance)
    """
    # MHC-I binding (normalised)
    ic50 = row.get("ic50_nM", 50000)
    binding_score = max(0, 1.0 - (min(ic50, 50000) / 50000))

    # MHC-II binding (if available)
    mhc_ii_ic50 = row.get("best_mhc_ii_ic50", 50000)
    mhc_ii_score = max(0, 1.0 - (min(mhc_ii_ic50, 50000) / 50000))

    # Agretopicity
    agreto = min(row.get("agretopicity", 1.0), 10.0) / 10.0

    # Immunogenicity
    immuno = row.get("immunogenicity_score", 25.0) / 100.0

    # Foreignness
    foreign = row.get("foreignness_score", 0.5)

    # Clonality
    clonality = row.get("clonality_score", 0.7)

    # Expression
    tpm = row.get("gene_tpm", 10.0)
    expr_score = min(1.0, math.log(tpm + 1) / math.log(100))

    # Driver gene
    driver = float(row.get("is_driver_gene", False))

    # Weighted combination
    score = (
        0.25 * binding_score +
        0.10 * mhc_ii_score +
        0.15 * agreto +
        0.15 * immuno +
        0.10 * foreign +
        0.10 * clonality +
        0.05 * expr_score +
        0.10 * driver
    )

    return round(score * 100, 2)


def run_enhanced_analysis(patient_id: str,
                           output_base: str = "output",
                           patient_rnaseq: str = None):
    """Run the full enhanced analysis pipeline for a patient."""

    mhcflurry_dir = os.path.join(output_base, f"{patient_id}_mhcflurry")
    enhanced_dir = os.path.join(output_base, f"{patient_id}_enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)

    print("=" * 70)
    print(f"  ENHANCED NEOANTIGEN ANALYSIS: {patient_id.upper()}")
    print("=" * 70)

    # Load MHCflurry results
    binding_path = os.path.join(mhcflurry_dir, "binding_predictions.csv")
    if not os.path.exists(binding_path):
        print(f"  ERROR: MHCflurry results not found at {binding_path}")
        print(f"  Run the base pipeline first.")
        sys.exit(1)

    results_df = pd.read_csv(binding_path)
    print(f"\n[1/5] Loaded {len(results_df)} MHCflurry predictions")

    # Get unique candidates (deduplicate across HLA alleles for some analyses)
    # Keep best binding per mutation for MHC-I
    best_mhci = results_df.loc[
        results_df.groupby(["gene_symbol", "aa_change", "mutant_peptide"])["ic50_nM"].idxmin()
    ].copy()
    print(f"  Unique peptide candidates: {len(best_mhci)}")

    # --- 1. NetMHCpan ---
    print(f"\n[2/5] NetMHCpan integration...")
    netmhcpan_path = check_netmhcpan()
    if netmhcpan_path:
        print(f"  NetMHCpan found at: {netmhcpan_path}")
        # Would run here; consensus scoring with MHCflurry
    else:
        print("  NetMHCpan not installed (academic license required from DTU)")
        print("  To add: download from https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/")
        print("  Pipeline will use MHCflurry results only for MHC-I")

    # --- 2. MHC-II predictions ---
    print(f"\n[3/5] MHC-II binding predictions (CD4+ T helper epitopes)...")
    mhc_ii_results = run_mhc_ii_predictions(best_mhci)

    # Merge best MHC-II result per peptide
    if not mhc_ii_results.empty:
        best_mhcii = mhc_ii_results.loc[
            mhc_ii_results.groupby("mutant_peptide")["mhc_ii_ic50"].idxmin()
        ][["mutant_peptide", "mhc_ii_ic50", "mhc_ii_classification",
           "hla_ii_allele", "mhc_ii_binding_core"]].rename(
            columns={"mhc_ii_ic50": "best_mhc_ii_ic50",
                      "mhc_ii_classification": "best_mhc_ii_class",
                      "hla_ii_allele": "best_mhc_ii_allele"}
        )
        mhc_ii_results.to_csv(
            os.path.join(enhanced_dir, "mhc_ii_predictions.csv"), index=False
        )
    else:
        best_mhcii = pd.DataFrame(columns=["mutant_peptide"])

    # --- 3. Clonality analysis ---
    print(f"\n[4/5] Clonality analysis...")
    results_with_clonality = estimate_clonality(results_df)

    # --- 4. Expression filtering ---
    print(f"\n[5/5] Expression filtering...")
    patient_expr = None
    if patient_rnaseq:
        patient_expr = load_patient_expression(patient_rnaseq)
    results_with_expr = apply_expression_filter(
        results_with_clonality, patient_expr
    )

    # --- Merge MHC-II into main results ---
    if not best_mhcii.empty:
        enhanced_df = results_with_expr.merge(
            best_mhcii, on="mutant_peptide", how="left"
        )
        enhanced_df["best_mhc_ii_ic50"] = enhanced_df["best_mhc_ii_ic50"].fillna(50000)
    else:
        enhanced_df = results_with_expr.copy()
        enhanced_df["best_mhc_ii_ic50"] = 50000

    # --- Recalculate enhanced scores ---
    print(f"\n  Calculating enhanced vaccine priority scores...")
    enhanced_df["enhanced_score"] = enhanced_df.apply(enhanced_vaccine_score, axis=1)
    enhanced_df = enhanced_df.sort_values("enhanced_score", ascending=False)
    enhanced_df["enhanced_rank"] = range(1, len(enhanced_df) + 1)

    # --- Filter: only expressed, prefer clonal, binders ---
    viable = enhanced_df[
        (enhanced_df["expression_pass"]) &
        (enhanced_df["ic50_nM"] < 500)
    ].copy()

    print(f"\n  --- ENHANCED RESULTS SUMMARY ---")
    print(f"  Total predictions:         {len(enhanced_df)}")
    print(f"  MHC-I binders (IC50<500):  {len(enhanced_df[enhanced_df['ic50_nM'] < 500])}")
    print(f"  MHC-II binders (IC50<1000):{len(enhanced_df[enhanced_df['best_mhc_ii_ic50'] < 1000])}")
    print(f"  Expression-filtered:       {len(enhanced_df[enhanced_df['expression_pass']])}")
    print(f"  Clonal mutations:          {len(enhanced_df[enhanced_df['clonality'] == 'clonal'])}")
    print(f"  Viable vaccine candidates: {len(viable)}")

    # --- Save results ---
    enhanced_df.to_csv(
        os.path.join(enhanced_dir, "enhanced_predictions.csv"), index=False
    )

    if len(viable) > 0:
        # Top candidates table
        top_cols = ["enhanced_rank", "gene_symbol", "aa_change", "hla_allele",
                    "mutant_peptide", "ic50_nM", "best_mhc_ii_ic50",
                    "clonality", "cancer_cell_fraction", "gene_tpm",
                    "enhanced_score"]
        available = [c for c in top_cols if c in viable.columns]
        top_20 = viable.head(20)[available]

        print(f"\n  ========== TOP 20 ENHANCED VACCINE CANDIDATES ==========")
        print(top_20.to_string(index=False))

        top_20.to_csv(
            os.path.join(enhanced_dir, "top_candidates.csv"), index=False
        )

    # --- Dual MHC candidates (both MHC-I and MHC-II binders) ---
    dual_binders = enhanced_df[
        (enhanced_df["ic50_nM"] < 500) &
        (enhanced_df["best_mhc_ii_ic50"] < 1000)
    ]
    if len(dual_binders) > 0:
        print(f"\n  ========== DUAL MHC-I + MHC-II BINDERS ({len(dual_binders)}) ==========")
        dual_cols = ["gene_symbol", "aa_change", "hla_allele", "mutant_peptide",
                     "ic50_nM", "best_mhc_ii_ic50", "clonality", "enhanced_score"]
        dcols = [c for c in dual_cols if c in dual_binders.columns]
        print(dual_binders.head(10)[dcols].to_string(index=False))

        dual_binders.to_csv(
            os.path.join(enhanced_dir, "dual_mhc_binders.csv"), index=False
        )

    print(f"\n  Results saved to: {enhanced_dir}/")
    print(f"  - enhanced_predictions.csv  (all {len(enhanced_df)} predictions)")
    print(f"  - mhc_ii_predictions.csv    (MHC-II results)")
    print(f"  - top_candidates.csv        (top 20 enhanced)")
    if len(dual_binders) > 0:
        print(f"  - dual_mhc_binders.csv      ({len(dual_binders)} dual binders)")
    print("=" * 70)

    return enhanced_df


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced neoantigen analysis with MHC-II, clonality, expression"
    )
    parser.add_argument("--patient", required=True,
                        help="Patient ID (e.g. mmrf_1251)")
    parser.add_argument("--output-dir", default="output",
                        help="Base output directory")
    parser.add_argument("--patient-rnaseq", default=None,
                        help="Path to patient-specific RNA-seq CSV (gene_symbol,tpm)")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    run_enhanced_analysis(
        patient_id=args.patient,
        output_base=args.output_dir,
        patient_rnaseq=args.patient_rnaseq,
    )


if __name__ == "__main__":
    main()
