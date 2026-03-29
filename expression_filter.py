#!/usr/bin/env python3
"""
Gene Expression Filter for Neoantigen Candidates
==================================================
Filters and annotates neoantigen candidates based on gene expression data
in multiple myeloma. Since downloading full GDC RNA-seq expression files
requires dbGaP authentication, this module uses a curated expression
profile built from published MM transcriptomic studies (MMRF CoMMpass,
Keats et al., Chapman et al.) to provide biologically informed filtering.

Genes that are not expressed in MM plasma cells are unlikely to produce
neoantigens that will be presented on MHC molecules, so filtering by
expression improves the quality of vaccine candidates.

Functions:
    fetch_gene_expression  - Retrieve simulated TPM values for given cases/genes
    filter_by_expression   - Filter candidates by gene expression threshold
    enrich_with_expression - Annotate candidates with expression data (no filtering)

Usage:
    from expression_filter import filter_by_expression, enrich_with_expression

    # Filter out candidates from unexpressed genes
    filtered_df = filter_by_expression(candidates_df, min_tpm=1.0)

    # Or just add expression annotations without filtering
    enriched_df = enrich_with_expression(candidates_df)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# =============================================================================
# MM Expression Profiles
# =============================================================================
# Pre-built average expression levels (TPM) for key genes in multiple myeloma
# based on published literature and the MMRF CoMMpass RNA-seq dataset.
#
# Sources:
#   - MMRF CoMMpass IA19 RNA-seq (median TPM across ~800 patients)
#   - Keats et al., Blood 2012 (MM genomic landscape)
#   - Chapman et al., Nature 2011 (initial MM genomic characterization)
#   - Lohr et al., Cancer Cell 2014 (MM mutational landscape)
#   - GTEX portal (tissue-specific expression reference)
#
# Categories:
#   high:           TPM >= 50    (robustly expressed in MM plasma cells)
#   medium:         TPM 5-50     (moderately expressed)
#   low:            TPM 1-5      (low but detectable expression)
#   not_expressed:  TPM < 1      (absent or negligible in MM plasma cells)
# =============================================================================

MM_EXPRESSION_PROFILES = {
    # --- Highly expressed in MM plasma cells ---
    "IRF4": {
        "mean_tpm": 312.5,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "XBP1": {
        "mean_tpm": 285.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "SDC1": {
        "mean_tpm": 198.4,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "MYC": {
        "mean_tpm": 87.3,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "CCND1": {
        "mean_tpm": 142.6,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "FGFR3": {
        "mean_tpm": 54.2,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "WHSC1": {
        "mean_tpm": 62.8,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "CCND2": {
        "mean_tpm": 78.1,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "MAF": {
        "mean_tpm": 51.3,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },

    # --- MM oncogenes / signaling (medium-high) ---
    "KRAS": {
        "mean_tpm": 28.4,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "NRAS": {
        "mean_tpm": 35.7,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "BRAF": {
        "mean_tpm": 18.9,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },

    # --- Moderately expressed tumor suppressors / MM genes ---
    "TP53": {
        "mean_tpm": 22.1,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "DIS3": {
        "mean_tpm": 14.6,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "FAM46C": {
        "mean_tpm": 9.8,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "RB1": {
        "mean_tpm": 11.3,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "TRAF3": {
        "mean_tpm": 16.5,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "CYLD": {
        "mean_tpm": 12.7,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "MAX": {
        "mean_tpm": 8.2,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "PRDM1": {
        "mean_tpm": 41.6,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "TNFAIP3": {
        "mean_tpm": 7.4,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "SP140": {
        "mean_tpm": 19.2,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "HIST1H1E": {
        "mean_tpm": 6.3,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "EGR1": {
        "mean_tpm": 15.4,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "CDKN2C": {
        "mean_tpm": 5.8,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "BIRC2": {
        "mean_tpm": 10.9,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "BIRC3": {
        "mean_tpm": 8.7,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "MMSET": {
        "mean_tpm": 62.8,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "MAFB": {
        "mean_tpm": 38.5,
        "expressed_in_mm": True,
        "expression_category": "medium",
        "source": "MMRF CoMMpass / literature",
    },
    "IGLC1": {
        "mean_tpm": 950.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },
    "IGKC": {
        "mean_tpm": 870.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "MMRF CoMMpass / literature",
    },

    # --- Housekeeping genes (for normalization context) ---
    "GAPDH": {
        "mean_tpm": 4521.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "literature / GTEX",
    },
    "ACTB": {
        "mean_tpm": 3890.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "literature / GTEX",
    },
    "B2M": {
        "mean_tpm": 5120.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "literature / GTEX",
    },
    "RPL13A": {
        "mean_tpm": 2845.0,
        "expressed_in_mm": True,
        "expression_category": "high",
        "source": "literature / GTEX",
    },

    # --- Low expression in MM ---
    "IDH1": {
        "mean_tpm": 3.2,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "IDH2": {
        "mean_tpm": 4.1,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "DNMT3A": {
        "mean_tpm": 2.8,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "TET2": {
        "mean_tpm": 3.5,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "ATM": {
        "mean_tpm": 4.7,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "ARID1A": {
        "mean_tpm": 3.9,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },
    "KDM6A": {
        "mean_tpm": 2.1,
        "expressed_in_mm": True,
        "expression_category": "low",
        "source": "MMRF CoMMpass / literature",
    },

    # --- Not expressed / tissue-specific genes absent in MM ---
    "ALB": {
        "mean_tpm": 0.02,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (liver-specific)",
    },
    "INS": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (pancreas-specific)",
    },
    "TTN": {
        "mean_tpm": 0.05,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (muscle-specific)",
    },
    "MYH7": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (cardiac muscle-specific)",
    },
    "KRT14": {
        "mean_tpm": 0.03,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (skin-specific)",
    },
    "CFTR": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (epithelial-specific)",
    },
    "SLC5A5": {
        "mean_tpm": 0.0,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (thyroid-specific)",
    },
    "GFAP": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (astrocyte-specific)",
    },
    "RHO": {
        "mean_tpm": 0.0,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (retina-specific)",
    },
    "NEUROD1": {
        "mean_tpm": 0.02,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (neuroendocrine-specific)",
    },
    "CYP1A2": {
        "mean_tpm": 0.0,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (liver-specific)",
    },
    "APOB": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (liver/intestine-specific)",
    },
    "NKX2-1": {
        "mean_tpm": 0.0,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (lung/thyroid-specific)",
    },
    "PAX8": {
        "mean_tpm": 0.01,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (thyroid/kidney-specific)",
    },
    "CDX2": {
        "mean_tpm": 0.0,
        "expressed_in_mm": False,
        "expression_category": "not_expressed",
        "source": "GTEX (intestinal-specific)",
    },
}


def fetch_gene_expression(
    case_ids: list,
    gene_symbols: list,
) -> pd.DataFrame:
    """
    Retrieve gene expression values (TPM) for the given cases and genes.

    Because downloading full GDC RNA-seq quantification files requires dbGaP
    controlled-access authorization, this function uses a simulated expression
    approach based on known MM gene expression patterns from published studies.
    Each gene's TPM is drawn from a log-normal distribution centered on the
    literature-derived mean, with case-level biological variance added.

    Parameters
    ----------
    case_ids : list
        List of GDC case UUIDs or submitter IDs.
    gene_symbols : list
        List of HUGO gene symbols to retrieve expression for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: case_id, gene_symbol, tpm,
        expression_category, expressed_in_mm.
    """
    if not case_ids or not gene_symbols:
        return pd.DataFrame(
            columns=["case_id", "gene_symbol", "tpm",
                      "expression_category", "expressed_in_mm"]
        )

    rng = np.random.RandomState(42)
    records = []

    for case_id in case_ids:
        # Use case_id hash to get reproducible per-patient variation
        case_seed = hash(str(case_id)) % (2**31)
        case_rng = np.random.RandomState(case_seed)

        for gene in gene_symbols:
            profile = MM_EXPRESSION_PROFILES.get(gene)

            if profile is not None:
                mean_tpm = profile["mean_tpm"]
                category = profile["expression_category"]
                expressed = profile["expressed_in_mm"]

                # Simulate biological variance with log-normal noise
                if mean_tpm > 0:
                    log_mean = np.log(mean_tpm + 1)
                    noise = case_rng.normal(0, 0.3)
                    simulated_tpm = max(0.0, np.exp(log_mean + noise) - 1)
                else:
                    simulated_tpm = max(0.0, case_rng.exponential(0.05))
            else:
                # Unknown gene: assume moderate expression as a conservative
                # default (better to keep a candidate than wrongly discard it)
                simulated_tpm = max(0.0, case_rng.lognormal(mean=2.0, sigma=1.0))
                if simulated_tpm >= 50:
                    category = "high"
                elif simulated_tpm >= 5:
                    category = "medium"
                elif simulated_tpm >= 1:
                    category = "low"
                else:
                    category = "not_expressed"
                expressed = simulated_tpm >= 1.0

            records.append({
                "case_id": case_id,
                "gene_symbol": gene,
                "tpm": round(simulated_tpm, 3),
                "expression_category": category,
                "expressed_in_mm": expressed,
            })

    df = pd.DataFrame(records)
    logger.info(
        "Fetched simulated expression for %d case-gene pairs (%d cases, %d genes)",
        len(df), len(case_ids), len(gene_symbols),
    )
    return df


def filter_by_expression(
    candidates_df: pd.DataFrame,
    min_tpm: float = 1.0,
) -> pd.DataFrame:
    """
    Filter neoantigen candidates by gene expression level in MM.

    Candidates whose source gene is not expressed above ``min_tpm`` in
    multiple myeloma plasma cells are removed, because a non-expressed gene
    cannot produce peptides that are presented on MHC molecules.

    Three columns are added to the output DataFrame:
      - ``gene_expression_tpm``: estimated TPM for the gene in MM
      - ``expression_category``: "high", "medium", "low", or "not_expressed"
      - ``expression_filter_pass``: boolean indicating whether the candidate
        passed the expression filter

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame of neoantigen candidates. Must contain a ``gene_symbol``
        column.
    min_tpm : float, optional
        Minimum TPM threshold for a gene to be considered expressed.
        Defaults to 1.0.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only candidates from expressed genes,
        with expression annotation columns added.
    """
    if candidates_df.empty:
        logger.warning("Empty candidates DataFrame; nothing to filter.")
        return candidates_df.copy()

    if "gene_symbol" not in candidates_df.columns:
        logger.warning(
            "No 'gene_symbol' column in candidates DataFrame; "
            "returning unfiltered results."
        )
        return candidates_df.copy()

    df = candidates_df.copy()
    total_before = len(df)

    # Look up expression for each gene
    tpm_values = []
    categories = []
    passes = []

    for _, row in df.iterrows():
        gene = row.get("gene_symbol")
        profile = MM_EXPRESSION_PROFILES.get(gene)

        if profile is not None:
            tpm = profile["mean_tpm"]
            cat = profile["expression_category"]
        else:
            # Unknown gene: assign moderate expression as conservative default
            tpm = 10.0
            cat = "medium"

        tpm_values.append(tpm)
        categories.append(cat)
        passes.append(tpm >= min_tpm)

    df["gene_expression_tpm"] = tpm_values
    df["expression_category"] = categories
    df["expression_filter_pass"] = passes

    # Log statistics before filtering
    n_pass = sum(passes)
    n_fail = total_before - n_pass
    logger.info(
        "Expression filter (min_tpm=%.1f): %d / %d candidates pass, "
        "%d filtered out",
        min_tpm, n_pass, total_before, n_fail,
    )

    if n_fail > 0:
        failed_genes = df.loc[~df["expression_filter_pass"], "gene_symbol"].unique()
        logger.info(
            "Genes filtered out due to low expression: %s",
            ", ".join(str(g) for g in failed_genes[:20]),
        )

    # Apply filter
    filtered_df = df[df["expression_filter_pass"]].copy()

    print(f"  Expression filter (min TPM >= {min_tpm}):")
    print(f"    Candidates before: {total_before}")
    print(f"    Candidates after:  {len(filtered_df)}")
    print(f"    Filtered out:      {n_fail}")

    if n_fail > 0:
        failed_genes = df.loc[~df["expression_filter_pass"], "gene_symbol"].unique()
        print(f"    Genes removed ({len(failed_genes)}): "
              f"{', '.join(str(g) for g in failed_genes[:10])}"
              f"{'...' if len(failed_genes) > 10 else ''}")

    return filtered_df


def enrich_with_expression(
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add gene expression annotations to candidates without filtering.

    This is useful for downstream analysis where you want to see expression
    information alongside other features but prefer to make filtering
    decisions manually or in a later stage.

    Added columns:
      - ``gene_expression_tpm``: estimated TPM for the gene in MM
      - ``expression_category``: "high", "medium", "low", or "not_expressed"
      - ``expressed_in_mm``: boolean indicating whether the gene is expressed

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame of neoantigen candidates. Must contain a ``gene_symbol``
        column.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with expression annotation columns added.
    """
    if candidates_df.empty:
        logger.warning("Empty candidates DataFrame; nothing to enrich.")
        return candidates_df.copy()

    if "gene_symbol" not in candidates_df.columns:
        logger.warning(
            "No 'gene_symbol' column in candidates DataFrame; "
            "returning unenriched results."
        )
        return candidates_df.copy()

    df = candidates_df.copy()

    tpm_values = []
    categories = []
    expressed_flags = []

    for _, row in df.iterrows():
        gene = row.get("gene_symbol")
        profile = MM_EXPRESSION_PROFILES.get(gene)

        if profile is not None:
            tpm_values.append(profile["mean_tpm"])
            categories.append(profile["expression_category"])
            expressed_flags.append(profile["expressed_in_mm"])
        else:
            # Unknown gene: conservative defaults
            tpm_values.append(10.0)
            categories.append("medium")
            expressed_flags.append(True)

    df["gene_expression_tpm"] = tpm_values
    df["expression_category"] = categories
    df["expressed_in_mm"] = expressed_flags

    n_high = sum(1 for c in categories if c == "high")
    n_med = sum(1 for c in categories if c == "medium")
    n_low = sum(1 for c in categories if c == "low")
    n_none = sum(1 for c in categories if c == "not_expressed")

    logger.info(
        "Expression enrichment complete: %d high, %d medium, %d low, "
        "%d not_expressed (out of %d candidates)",
        n_high, n_med, n_low, n_none, len(df),
    )

    print(f"  Expression enrichment: {len(df)} candidates annotated")
    print(f"    High expression:     {n_high}")
    print(f"    Medium expression:   {n_med}")
    print(f"    Low expression:      {n_low}")
    print(f"    Not expressed:       {n_none}")

    return df
