#!/usr/bin/env python3
"""
Mutation Parser & Neoantigen Candidate Identifier
==================================================
Processes somatic mutations from MMRF CoMMpass data and identifies
candidate neoantigens by:

1. Filtering mutations by type (missense, frameshift, etc.)
2. Extracting mutant peptide sequences
3. Identifying mutations in known MM driver genes
4. Preparing data for MHC binding prediction

Usage:
    python 02_parse_mutations.py [--input data/mmrf_mutations.csv] [--config config.yaml]
"""

import os
import sys
import json
import argparse
import re
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# Standard genetic code
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Amino acid properties for immunogenicity scoring
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 'small'},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 'large'},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium'},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium'},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 'small'},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 'medium'},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 'medium'},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 'tiny'},
    'H': {'hydrophobicity': -3.2, 'charge': 0, 'size': 'medium'},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 'large'},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 'large'},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 'large'},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 'large'},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 'large'},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 'small'},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 'small'},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 'medium'},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 'large'},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 'large'},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 'medium'},
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_aa_change(aa_change: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse amino acid change notation like 'p.V600E' or 'p.Arg132His'.

    Returns: (wildtype_aa, position, mutant_aa) or None if unparseable.
    """
    if not aa_change or pd.isna(aa_change):
        return None

    # Three-letter to one-letter mapping
    aa3to1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
        'Ter': '*',
    }

    # Try single-letter format: p.V600E or V600E (with or without p. prefix)
    match = re.match(r'(?:p\.)?([A-Z\*])(\d+)([A-Z\*])$', aa_change)
    if match:
        return (match.group(1), int(match.group(2)), match.group(3))

    # Try three-letter format: p.Val600Glu or Val600Glu
    match = re.match(r'(?:p\.)?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', aa_change)
    if match:
        wt = aa3to1.get(match.group(1))
        mt = aa3to1.get(match.group(3))
        if wt and mt:
            return (wt, int(match.group(2)), mt)

    # Frameshift format: H698Sfs*8 or p.H698Sfs*8
    match = re.match(r'(?:p\.)?([A-Z])(\d+)[A-Z]fs\*', aa_change)
    if match:
        return (match.group(1), int(match.group(2)), 'X')  # X = frameshift neosequence

    # Stop gained: Q88* or p.Q88*
    match = re.match(r'(?:p\.)?([A-Z])(\d+)\*$', aa_change)
    if match:
        return (match.group(1), int(match.group(2)), '*')

    return None


def generate_mutant_peptides(wt_aa: str, position: int, mut_aa: str,
                              peptide_lengths: list = [8, 9, 10, 11],
                              gene_symbol: str = None) -> list:
    """
    Generate all possible mutant peptide windows containing the mutation.

    For MHC-I binding prediction, we need peptides of 8-11 amino acids
    that contain the mutant residue.

    This function first attempts to use real protein sequences from UniProt
    (via the uniprot_lookup module). If that fails, it falls back to
    synthetic peptide generation with placeholder flanking residues.
    """
    # Try to use real protein sequences from UniProt
    if gene_symbol:
        try:
            from uniprot_lookup import generate_real_peptides
            real_peptides = generate_real_peptides(
                gene_symbol, wt_aa, position, mut_aa, peptide_lengths
            )
            if real_peptides:
                return real_peptides
        except ImportError:
            pass  # uniprot_lookup not available, use synthetic

    # Fallback: synthetic peptide generation
    peptides = []
    common_residues = "AEKGDLSTPV"

    for length in peptide_lengths:
        for mut_pos in range(length):
            prefix_len = mut_pos
            suffix_len = length - mut_pos - 1

            prefix = "".join(common_residues[i % len(common_residues)]
                           for i in range(prefix_len))
            suffix = "".join(common_residues[(i + 3) % len(common_residues)]
                           for i in range(suffix_len))

            mutant_peptide = prefix + mut_aa + suffix
            wildtype_peptide = prefix + wt_aa + suffix

            peptides.append({
                "mutant_peptide": mutant_peptide,
                "wildtype_peptide": wildtype_peptide,
                "peptide_length": length,
                "mutation_position_in_peptide": mut_pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "protein_position": position,
                "sequence_source": "synthetic",
            })

    return peptides


def calculate_immunogenicity_score(wt_aa: str, mut_aa: str, gene_symbol: str,
                                    driver_genes: list, consequence_type: str) -> float:
    """
    Calculate a heuristic immunogenicity score for a mutation.

    This score considers:
    - How different the mutant AA is from wildtype (physicochemical distance)
    - Whether the gene is a known MM driver
    - The type of mutation (frameshift > missense)

    In production, you would use ML models like PRIME, DeepImmuno, or
    NetMHCpan for proper immunogenicity prediction.
    """
    score = 0.0

    # 1. Physicochemical distance between wildtype and mutant
    if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
        wt_props = AA_PROPERTIES[wt_aa]
        mut_props = AA_PROPERTIES[mut_aa]

        # Hydrophobicity change (bigger change = more immunogenic)
        hydro_diff = abs(wt_props['hydrophobicity'] - mut_props['hydrophobicity'])
        score += min(hydro_diff / 9.0, 1.0) * 30  # Max 30 points

        # Charge change (charge-altering mutations are more immunogenic)
        charge_diff = abs(wt_props['charge'] - mut_props['charge'])
        score += charge_diff * 15  # 0 or 15 or 30 points

        # Size change
        size_order = {'tiny': 0, 'small': 1, 'medium': 2, 'large': 3}
        size_diff = abs(size_order[wt_props['size']] - size_order[mut_props['size']])
        score += size_diff * 5  # Max 15 points

    # 2. Driver gene bonus
    if gene_symbol and gene_symbol in driver_genes:
        score += 15

    # 3. Consequence type bonus
    consequence_scores = {
        "missense_variant": 10,
        "frameshift_variant": 25,
        "inframe_insertion": 15,
        "inframe_deletion": 15,
        "stop_gained": 20,
    }
    if consequence_type:
        score += consequence_scores.get(consequence_type, 5)

    return min(score, 100.0)  # Cap at 100


def filter_mutations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter mutations to those suitable for neoantigen prediction."""
    mutation_config = config.get("mutations", {})

    # Keep only protein-altering mutations
    coding_consequences = [
        "missense_variant",
        "frameshift_variant",
        "inframe_insertion",
        "inframe_deletion",
        "stop_gained",
        "start_lost",
        "protein_altering_variant",
    ]

    if "consequence_type" in df.columns:
        mask = df["consequence_type"].isin(coding_consequences)
        df_filtered = df[mask].copy()
        print(f"  After consequence filter: {len(df_filtered)} mutations "
              f"(from {len(df)})")
    else:
        df_filtered = df.copy()

    # Remove mutations without amino acid change info
    if "aa_change" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["aa_change"].notna()]
        print(f"  After AA change filter: {len(df_filtered)} mutations")

    # Remove duplicate mutations per patient
    dedup_cols = ["case_id", "gene_symbol", "aa_change"]
    available_cols = [c for c in dedup_cols if c in df_filtered.columns]
    if available_cols:
        df_filtered = df_filtered.drop_duplicates(subset=available_cols)
        print(f"  After deduplication: {len(df_filtered)} mutations")

    return df_filtered


def process_mutations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Process filtered mutations to generate neoantigen candidates.

    For each mutation:
    1. Parse the amino acid change
    2. Generate candidate mutant peptides
    3. Score immunogenicity
    """
    neoantigen_config = config.get("neoantigen", {})
    mutation_config = config.get("mutations", {})
    driver_genes = mutation_config.get("driver_genes", [])
    peptide_lengths = neoantigen_config.get("mhc_class_i", {}).get(
        "peptide_lengths", [8, 9, 10, 11]
    )

    candidates = []
    parse_success = 0
    parse_fail = 0

    for _, row in df.iterrows():
        parsed = parse_aa_change(row.get("aa_change", ""))
        if not parsed:
            parse_fail += 1
            continue

        parse_success += 1
        wt_aa, position, mut_aa = parsed

        # Calculate immunogenicity score
        immuno_score = calculate_immunogenicity_score(
            wt_aa, mut_aa,
            row.get("gene_symbol"),
            driver_genes,
            row.get("consequence_type")
        )

        # Generate peptides for MHC binding prediction
        peptides = generate_mutant_peptides(
            wt_aa, position, mut_aa, peptide_lengths,
            gene_symbol=row.get("gene_symbol")
        )

        for pep in peptides:
            candidates.append({
                "case_id": row.get("case_id"),
                "submitter_id": row.get("submitter_id"),
                "gene_symbol": row.get("gene_symbol"),
                "gene_id": row.get("gene_id"),
                "chromosome": row.get("chromosome"),
                "start_position": row.get("start_position"),
                "aa_change": row.get("aa_change"),
                "consequence_type": row.get("consequence_type"),
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "protein_position": position,
                "is_driver_gene": row.get("gene_symbol") in driver_genes,
                "immunogenicity_score": immuno_score,
                **pep,
            })

    print(f"  Parsed {parse_success} mutations successfully ({parse_fail} failed)")
    print(f"  Generated {len(candidates)} peptide candidates")

    return pd.DataFrame(candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Parse MMRF mutations and generate neoantigen candidates"
    )
    parser.add_argument("--input", default="data/mmrf_mutations.csv",
                        help="Input mutations CSV")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("Mutation Parser & Neoantigen Candidate Generator")
    print("Multiple Myeloma Neoantigen Vaccine Pipeline - Step 2")
    print("=" * 70)

    # Load mutations
    print(f"\n[1/3] Loading mutations from {args.input}...")
    if not os.path.exists(args.input):
        print(f"  ERROR: File not found: {args.input}")
        print(f"  Run 01_fetch_mmrf_data.py first to download the data.")
        sys.exit(1)

    mutations_df = pd.read_csv(args.input)
    print(f"  Loaded {len(mutations_df)} mutation records")
    print(f"  Unique patients: {mutations_df['case_id'].nunique()}")

    # Filter mutations
    print(f"\n[2/3] Filtering mutations for neoantigen potential...")
    filtered_df = filter_mutations(mutations_df, config)

    if len(filtered_df) == 0:
        print("  WARNING: No mutations passed filtering.")
        print("  This may happen with limited API data. Using all mutations instead.")
        filtered_df = mutations_df[mutations_df["aa_change"].notna()].copy()

    # Process mutations into neoantigen candidates
    print(f"\n[3/3] Generating neoantigen candidates...")
    candidates_df = process_mutations(filtered_df, config)

    if len(candidates_df) > 0:
        # Save candidates
        candidates_path = output_dir / "neoantigen_candidates.csv"
        candidates_df.to_csv(candidates_path, index=False)
        print(f"\n  Saved {len(candidates_df)} candidates to: {candidates_path}")

        # Summary statistics
        print(f"\n  --- Neoantigen Candidate Summary ---")
        print(f"  Total candidates: {len(candidates_df)}")
        print(f"  Unique mutations: {candidates_df['aa_change'].nunique()}")
        print(f"  Unique genes: {candidates_df['gene_symbol'].nunique()}")

        driver_mask = candidates_df["is_driver_gene"] == True
        print(f"  Driver gene mutations: {candidates_df[driver_mask]['aa_change'].nunique()}")

        print(f"\n  Top genes by candidate count:")
        gene_counts = candidates_df.groupby("gene_symbol").size().sort_values(ascending=False)
        for gene, count in gene_counts.head(10).items():
            is_driver = "** DRIVER" if gene in config.get("mutations", {}).get("driver_genes", []) else ""
            print(f"    {gene}: {count} peptide candidates {is_driver}")

        print(f"\n  Immunogenicity score distribution:")
        scores = candidates_df["immunogenicity_score"]
        print(f"    Mean: {scores.mean():.1f}")
        print(f"    Median: {scores.median():.1f}")
        print(f"    Top quartile (>75th percentile): {scores.quantile(0.75):.1f}")

        # Save high-priority candidates
        high_priority = candidates_df[
            candidates_df["immunogenicity_score"] >= candidates_df["immunogenicity_score"].quantile(0.75)
        ]
        hp_path = output_dir / "high_priority_candidates.csv"
        high_priority.to_csv(hp_path, index=False)
        print(f"\n  High-priority candidates (top 25%): {len(high_priority)}")
        print(f"  Saved to: {hp_path}")
    else:
        print("\n  WARNING: No neoantigen candidates generated.")
        print("  Check your input data and filters.")

    print("\n" + "=" * 70)
    print("Mutation parsing complete!")
    print("\nNext step: Run 03_predict_binding.py for MHC binding prediction")
    print("=" * 70)


if __name__ == "__main__":
    main()
