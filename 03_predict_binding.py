#!/usr/bin/env python3
"""
MHC Binding Prediction & Neoantigen Ranking
============================================
Predicts MHC-peptide binding affinity for neoantigen candidates and
ranks them by their potential to elicit an immune response.

This module implements:
1. A simplified MHC-I binding prediction model (PSSM-based)
2. Integration hooks for MHCflurry (production-grade predictions)
3. Neoantigen ranking combining binding affinity + immunogenicity
4. Population-level HLA coverage analysis

In production, you would use:
- MHCflurry (open source, neural network-based)
- NetMHCpan (state of the art, requires license)
- pVACseq (full pipeline with multiple algorithms)

Usage:
    python 03_predict_binding.py [--input data/neoantigen_candidates.csv]
"""

import os
import sys
import json
import argparse
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# ============================================================================
# Simplified Position-Specific Scoring Matrix (PSSM) for MHC-I Binding
# ============================================================================
# These are simplified binding preference matrices for common HLA alleles.
# In production, use MHCflurry or NetMHCpan for accurate predictions.
#
# Each matrix represents amino acid preferences at each position of a 9-mer
# peptide for a given HLA allele. Values are log-odds scores.
# ============================================================================

# HLA-A*02:01 binding preferences (most common allele, ~25-30% of population)
# Anchor residues: P2 (L, M, I, V, A) and P9 (V, L, I, A, T)
HLA_A0201_PSSM = {
    'anchor_positions': [1, 8],  # 0-indexed positions 2 and 9
    'anchor_preferences': {
        1: {'L': 1.0, 'M': 0.8, 'I': 0.7, 'V': 0.6, 'A': 0.4, 'T': 0.3},
        8: {'V': 1.0, 'L': 0.9, 'I': 0.8, 'A': 0.6, 'T': 0.5},
    },
    'general_preferences': {
        'A': 0.3, 'R': -0.2, 'N': -0.1, 'D': -0.3, 'C': 0.1,
        'E': -0.2, 'Q': 0.0, 'G': -0.1, 'H': -0.1, 'I': 0.4,
        'L': 0.5, 'K': -0.3, 'M': 0.3, 'F': 0.2, 'P': -0.4,
        'S': 0.0, 'T': 0.1, 'W': -0.1, 'Y': 0.1, 'V': 0.4,
    }
}

# HLA-A*01:01 preferences
HLA_A0101_PSSM = {
    'anchor_positions': [1, 8],
    'anchor_preferences': {
        1: {'T': 1.0, 'S': 0.8, 'D': 0.6, 'E': 0.5},
        8: {'Y': 1.0, 'F': 0.8, 'W': 0.7},
    },
    'general_preferences': {
        'A': 0.2, 'R': -0.1, 'N': 0.1, 'D': 0.2, 'C': 0.0,
        'E': 0.1, 'Q': 0.0, 'G': -0.2, 'H': 0.0, 'I': 0.1,
        'L': 0.3, 'K': -0.2, 'M': 0.1, 'F': 0.3, 'P': -0.3,
        'S': 0.2, 'T': 0.3, 'W': 0.1, 'Y': 0.4, 'V': 0.1,
    }
}

# HLA-A*03:01 preferences
HLA_A0301_PSSM = {
    'anchor_positions': [1, 8],
    'anchor_preferences': {
        1: {'L': 0.8, 'V': 0.7, 'M': 0.6, 'I': 0.5, 'F': 0.5},
        8: {'K': 1.0, 'R': 0.9, 'Y': 0.5},
    },
    'general_preferences': {
        'A': 0.2, 'R': 0.3, 'N': 0.0, 'D': -0.2, 'C': 0.0,
        'E': -0.1, 'Q': 0.1, 'G': -0.1, 'H': 0.1, 'I': 0.3,
        'L': 0.4, 'K': 0.4, 'M': 0.2, 'F': 0.2, 'P': -0.3,
        'S': 0.0, 'T': 0.1, 'W': 0.0, 'Y': 0.2, 'V': 0.3,
    }
}

# HLA-B*07:02 preferences
HLA_B0702_PSSM = {
    'anchor_positions': [1, 8],
    'anchor_preferences': {
        1: {'P': 1.0, 'A': 0.4, 'S': 0.3},
        8: {'L': 1.0, 'M': 0.8, 'F': 0.6, 'I': 0.5},
    },
    'general_preferences': {
        'A': 0.3, 'R': 0.1, 'N': 0.0, 'D': -0.1, 'C': 0.0,
        'E': -0.1, 'Q': 0.0, 'G': -0.2, 'H': 0.0, 'I': 0.3,
        'L': 0.5, 'K': 0.0, 'M': 0.3, 'F': 0.3, 'P': 0.2,
        'S': 0.1, 'T': 0.1, 'W': 0.0, 'Y': 0.1, 'V': 0.3,
    }
}

PSSM_MODELS = {
    "HLA-A*02:01": HLA_A0201_PSSM,
    "HLA-A*01:01": HLA_A0101_PSSM,
    "HLA-A*03:01": HLA_A0301_PSSM,
    "HLA-B*07:02": HLA_B0702_PSSM,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def predict_binding_pssm(peptide: str, hla_allele: str) -> Dict:
    """
    Predict MHC-I binding using a simplified PSSM model.

    Returns a dict with:
    - score: raw binding score (higher = better binding)
    - ic50_estimate: estimated IC50 in nM (lower = stronger binding)
    - percentile_rank: estimated percentile rank
    - classification: 'strong_binder', 'weak_binder', or 'non_binder'
    """
    model = PSSM_MODELS.get(hla_allele)
    if not model:
        # Use HLA-A*02:01 as default model
        model = HLA_A0201_PSSM

    score = 0.0
    peptide = peptide.upper()

    # For non-9mer peptides, use a simplified scoring
    # (In production, NetMHCpan handles variable lengths natively)
    if len(peptide) != 9:
        # Scale score based on length deviation from optimal (9)
        length_penalty = -0.1 * abs(len(peptide) - 9)
        score += length_penalty

    for i, aa in enumerate(peptide):
        if i in model['anchor_positions'] and len(peptide) == 9:
            # Score anchor position
            anchor_prefs = model['anchor_preferences'].get(i, {})
            score += anchor_prefs.get(aa, -0.5)
        else:
            # Score non-anchor position
            score += model['general_preferences'].get(aa, -0.1)

    # Convert score to estimated IC50 (nM)
    # This is a rough mapping; real tools use neural networks
    ic50_estimate = 50000 * math.exp(-score * 1.5)
    ic50_estimate = max(1, min(50000, ic50_estimate))

    # Classify binding strength
    if ic50_estimate < 50:
        classification = "strong_binder"
        percentile_rank = max(0.1, ic50_estimate / 50 * 0.5)
    elif ic50_estimate < 500:
        classification = "weak_binder"
        percentile_rank = 0.5 + (ic50_estimate - 50) / 450 * 1.5
    else:
        classification = "non_binder"
        percentile_rank = 2.0 + (min(ic50_estimate, 50000) - 500) / 49500 * 98

    return {
        "binding_score": round(score, 4),
        "ic50_nM": round(ic50_estimate, 2),
        "percentile_rank": round(percentile_rank, 2),
        "classification": classification,
    }


def predict_binding_mhcflurry(peptides: List[str], hla_allele: str,
                              predictor=None) -> List[Dict]:
    """
    Predict binding using MHCflurry (if installed).

    MHCflurry is a production-grade MHC-I binding prediction tool that uses
    neural networks trained on experimental binding affinity data.

    BATCH MODE: Passes all peptides in a single predictor.predict() call
    for ~100x speedup vs one-at-a-time prediction.

    Args:
        peptides: List of peptide sequences to predict
        hla_allele: HLA allele string (e.g. "HLA-A*02:01")
        predictor: Pre-loaded predictor instance (avoids reloading per call)

    Install: pip install mhcflurry && mhcflurry-downloads fetch
    """
    try:
        from mhcflurry import Class1PresentationPredictor
        if predictor is None:
            predictor = Class1PresentationPredictor.load()

        # Single batch call with all peptides and matching allele list
        alleles = [hla_allele] * len(peptides)
        pred_df = predictor.predict(
            peptides=peptides,
            alleles=alleles,
        )

        results = []
        for i, row in pred_df.iterrows():
            ic50 = row.get("mhcflurry_affinity", 50000)

            if ic50 < 50:
                classification = "strong_binder"
            elif ic50 < 500:
                classification = "weak_binder"
            else:
                classification = "non_binder"

            results.append({
                "binding_score": round(-math.log(max(ic50, 0.01) / 50000), 4),
                "ic50_nM": round(ic50, 2),
                "percentile_rank": round(row.get("mhcflurry_affinity_percentile", 50), 2),
                "classification": classification,
                "presentation_score": round(row.get("mhcflurry_presentation_score", 0), 4),
            })

        return results

    except ImportError:
        # Fall back to PSSM
        return [predict_binding_pssm(p, hla_allele) for p in peptides]
    except Exception as e:
        print(f"    MHCflurry batch prediction failed ({e}), falling back to PSSM")
        return [predict_binding_pssm(p, hla_allele) for p in peptides]


def calculate_agretopicity(mutant_ic50: float, wildtype_ic50: float) -> float:
    """
    Calculate agretopicity index = wildtype_IC50 / mutant_IC50.

    Agretopicity > 1 means the mutant binds better than wildtype.
    Higher agretopicity = mutation creates a better MHC binder = more likely
    to be recognized as foreign by T cells.
    """
    if mutant_ic50 <= 0:
        return 0.0
    return wildtype_ic50 / mutant_ic50


def calculate_foreignness_score(mutant_peptide: str, wildtype_peptide: str) -> float:
    """
    Calculate how 'foreign' a mutant peptide looks compared to self.

    Higher score = more different from self = more likely to trigger immune response.
    """
    if len(mutant_peptide) != len(wildtype_peptide):
        return 1.0  # Frameshifts are maximally foreign

    differences = 0
    total_weight = 0

    for i, (m, w) in enumerate(zip(mutant_peptide, wildtype_peptide)):
        weight = 1.0
        # Anchor positions matter less for T cell recognition
        if i in [1, len(mutant_peptide) - 1]:
            weight = 0.5
        # Central positions matter more
        if len(mutant_peptide) // 3 <= i <= 2 * len(mutant_peptide) // 3:
            weight = 1.5

        if m != w:
            differences += weight
        total_weight += weight

    return differences / total_weight if total_weight > 0 else 0.0


def rank_neoantigens(candidates_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Rank neoantigen candidates by composite vaccine priority score.

    The ranking considers:
    1. MHC binding affinity (IC50)
    2. Agretopicity (mutant vs wildtype binding)
    3. Immunogenicity score (from mutation parser)
    4. Foreignness score
    5. Driver gene status
    6. Population HLA coverage
    """
    df = candidates_df.copy()

    # Normalize scores to 0-1 range for combining
    if "ic50_nM" in df.columns:
        # Lower IC50 = better binding = higher score
        df["binding_norm"] = 1.0 - (df["ic50_nM"].clip(0, 50000) / 50000)
    else:
        df["binding_norm"] = 0.5

    if "agretopicity" in df.columns:
        df["agreto_norm"] = df["agretopicity"].clip(0, 10) / 10
    else:
        df["agreto_norm"] = 0.5

    if "immunogenicity_score" in df.columns:
        df["immuno_norm"] = df["immunogenicity_score"] / 100
    else:
        df["immuno_norm"] = 0.5

    if "foreignness_score" in df.columns:
        df["foreign_norm"] = df["foreignness_score"]
    else:
        df["foreign_norm"] = 0.5

    # Composite vaccine priority score (weighted combination)
    df["vaccine_priority_score"] = (
        0.35 * df["binding_norm"] +        # MHC binding is crucial
        0.20 * df["agreto_norm"] +          # Agretopicity matters
        0.20 * df["immuno_norm"] +          # Immunogenicity
        0.15 * df["foreign_norm"] +         # Foreignness
        0.10 * df["is_driver_gene"].astype(float)  # Driver gene bonus
    )

    # Scale to 0-100
    df["vaccine_priority_score"] = (df["vaccine_priority_score"] * 100).round(2)

    # Rank
    df = df.sort_values("vaccine_priority_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df


def analyze_hla_coverage(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which HLA alleles are covered by the top neoantigen candidates.

    Good vaccine candidates should cover multiple common HLA alleles to
    maximize population coverage.
    """
    # HLA allele frequencies in general population (approximate)
    hla_frequencies = {
        "HLA-A*02:01": 0.28,
        "HLA-A*01:01": 0.15,
        "HLA-A*03:01": 0.13,
        "HLA-A*24:02": 0.10,
        "HLA-B*07:02": 0.12,
        "HLA-B*08:01": 0.09,
    }

    if "hla_allele" not in results_df.columns:
        return pd.DataFrame()

    coverage = []
    for allele, freq in hla_frequencies.items():
        allele_data = results_df[results_df["hla_allele"] == allele]
        binders = allele_data[allele_data["classification"].isin(
            ["strong_binder", "weak_binder"]
        )]
        coverage.append({
            "hla_allele": allele,
            "population_frequency": freq,
            "total_candidates": len(allele_data),
            "binders": len(binders),
            "strong_binders": len(allele_data[allele_data["classification"] == "strong_binder"]),
        })

    return pd.DataFrame(coverage)


def run_binding_predictions(candidates_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run MHC binding predictions for all candidates across HLA alleles.

    Uses batch prediction for MHCflurry (all peptides in one call per allele)
    for ~100x speedup over the previous row-by-row approach.
    """
    neo_config = config.get("neoantigen", {})
    hla_alleles = neo_config.get("default_hla_alleles", {}).get(
        "class_i", ["HLA-A*02:01"]
    )

    # Pre-extract peptide lists once (avoid repeated .tolist() calls)
    mut_peptides = candidates_df["mutant_peptide"].tolist()
    wt_peptides = candidates_df["wildtype_peptide"].tolist()
    n_candidates = len(candidates_df)

    # Check for MHCflurry once, load predictor once
    use_mhcflurry = False
    mhcflurry_predictor = None
    try:
        from mhcflurry import Class1PresentationPredictor
        mhcflurry_predictor = Class1PresentationPredictor.load()
        use_mhcflurry = True
        print(f"  MHCflurry loaded (production-grade batch predictions)")
    except ImportError:
        print(f"  Using PSSM model (install mhcflurry for better accuracy)")

    all_results = []

    for allele in hla_alleles:
        print(f"  Predicting binding for {allele} ({n_candidates} peptides)...")

        # --- Batch predict mutant peptides ---
        if use_mhcflurry:
            mut_preds = predict_binding_mhcflurry(
                mut_peptides, allele, predictor=mhcflurry_predictor
            )
        else:
            mut_preds = [predict_binding_pssm(p, allele) for p in mut_peptides]

        # --- Batch predict wildtype peptides (for agretopicity) ---
        if use_mhcflurry:
            wt_preds = predict_binding_mhcflurry(
                wt_peptides, allele, predictor=mhcflurry_predictor
            )
        else:
            wt_preds = [predict_binding_pssm(p, allele) for p in wt_peptides]

        print(f"    Completed {n_candidates} mutant + {n_candidates} wildtype predictions")

        # --- Assemble results vectorised ---
        for i, (idx, row) in enumerate(candidates_df.iterrows()):
            mut_pred = mut_preds[i]
            wt_pred = wt_preds[i]

            agretopicity = calculate_agretopicity(
                mut_pred["ic50_nM"], wt_pred["ic50_nM"]
            )
            foreignness = calculate_foreignness_score(
                mut_peptides[i], wt_peptides[i]
            )

            result = {
                **row.to_dict(),
                "hla_allele": allele,
                "binding_score": mut_pred["binding_score"],
                "ic50_nM": mut_pred["ic50_nM"],
                "percentile_rank": mut_pred["percentile_rank"],
                "classification": mut_pred["classification"],
                "wt_ic50_nM": wt_pred["ic50_nM"],
                "agretopicity": agretopicity,
                "foreignness_score": foreignness,
            }

            # Include presentation score if available (MHCflurry only)
            if "presentation_score" in mut_pred:
                result["presentation_score"] = mut_pred["presentation_score"]

            all_results.append(result)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Predict MHC binding and rank neoantigen candidates"
    )
    parser.add_argument("--input", default="data/neoantigen_candidates.csv",
                        help="Input candidates CSV")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--max-candidates", type=int, default=5000,
                        help="Max candidates to evaluate (for speed)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("MHC Binding Prediction & Neoantigen Ranking")
    print("Multiple Myeloma Neoantigen Vaccine Pipeline - Step 3")
    print("=" * 70)

    # Load candidates
    print(f"\n[1/4] Loading neoantigen candidates...")
    if not os.path.exists(args.input):
        print(f"  ERROR: File not found: {args.input}")
        print(f"  Run 02_parse_mutations.py first.")
        sys.exit(1)

    candidates_df = pd.read_csv(args.input)
    print(f"  Loaded {len(candidates_df)} candidates")

    # Limit for speed if needed
    if len(candidates_df) > args.max_candidates:
        print(f"  Limiting to top {args.max_candidates} by immunogenicity score...")
        candidates_df = candidates_df.nlargest(args.max_candidates, "immunogenicity_score")

    # Run binding predictions
    print(f"\n[2/4] Running MHC binding predictions...")
    results_df = run_binding_predictions(candidates_df, config)
    print(f"  Completed predictions for {len(results_df)} candidate-allele pairs")

    # Summarize binding results
    binding_threshold = config.get("neoantigen", {}).get(
        "mhc_class_i", {}
    ).get("binding_threshold", 500)

    binders = results_df[results_df["ic50_nM"] < binding_threshold]
    strong_binders = results_df[results_df["ic50_nM"] < 50]
    print(f"\n  --- Binding Prediction Summary ---")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Binders (IC50 < {binding_threshold} nM): {len(binders)} "
          f"({100*len(binders)/max(len(results_df),1):.1f}%)")
    print(f"  Strong binders (IC50 < 50 nM): {len(strong_binders)} "
          f"({100*len(strong_binders)/max(len(results_df),1):.1f}%)")

    # Rank neoantigens
    print(f"\n[3/4] Ranking neoantigens by vaccine priority...")
    ranked_df = rank_neoantigens(results_df, config)

    # Save results
    results_path = output_dir / "binding_predictions.csv"
    ranked_df.to_csv(results_path, index=False)
    print(f"  Saved to: {results_path}")

    # HLA coverage analysis
    print(f"\n[4/4] Analyzing HLA population coverage...")
    coverage_df = analyze_hla_coverage(ranked_df)
    if len(coverage_df) > 0:
        coverage_path = output_dir / "hla_coverage.csv"
        coverage_df.to_csv(coverage_path, index=False)
        print(f"  HLA Coverage Analysis:")
        for _, row in coverage_df.iterrows():
            print(f"    {row['hla_allele']} (freq: {row['population_frequency']:.0%}): "
                  f"{row['binders']} binders ({row['strong_binders']} strong)")

    # Show top candidates
    print(f"\n  ========== TOP 20 VACCINE CANDIDATES ==========")
    top_cols = ["rank", "gene_symbol", "aa_change", "hla_allele",
                "mutant_peptide", "ic50_nM", "classification",
                "agretopicity", "vaccine_priority_score"]
    available_cols = [c for c in top_cols if c in ranked_df.columns]
    top_20 = ranked_df.head(20)[available_cols]
    print(top_20.to_string(index=False))

    print("\n" + "=" * 70)
    print("Binding prediction complete!")
    print("\nNext step: Run 04_design_vaccine.py to design the mRNA vaccine")
    print("=" * 70)


if __name__ == "__main__":
    main()
