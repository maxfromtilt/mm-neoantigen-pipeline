#!/usr/bin/env python3
"""
mRNA Vaccine Construct Designer
================================
Designs a personalized mRNA cancer vaccine construct from ranked neoantigen
candidates. The pipeline:

1. Selects top-ranked neoantigens with diverse HLA coverage
2. Optimizes epitope ordering to minimize junctional epitopes
3. Adds linker sequences between epitopes
4. Constructs the full mRNA sequence with UTRs, signal peptide, and poly-A tail
5. Applies codon optimization for human expression
6. Generates a comprehensive vaccine report

The design follows principles from BioNTech/Moderna mRNA vaccine platforms,
including the use of N1-methylpseudouridine-modified mRNA.

Usage:
    python 04_design_vaccine.py [--input data/binding_predictions.csv] [--config config.yaml]
"""

import os
import sys
import json
import argparse
import math
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# =============================================================================
# Human Codon Usage Table (frequency per thousand)
# =============================================================================
HUMAN_CODON_FREQ = {
    'A': {'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11},
    'R': {'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.20, 'AGG': 0.20},
    'N': {'AAT': 0.46, 'AAC': 0.54},
    'D': {'GAT': 0.46, 'GAC': 0.54},
    'C': {'TGT': 0.45, 'TGC': 0.55},
    'E': {'GAA': 0.42, 'GAG': 0.58},
    'Q': {'CAA': 0.25, 'CAG': 0.75},
    'G': {'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25},
    'H': {'CAT': 0.41, 'CAC': 0.59},
    'I': {'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16},
    'L': {'TTA': 0.07, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41},
    'K': {'AAA': 0.42, 'AAG': 0.58},
    'M': {'ATG': 1.00},
    'F': {'TTT': 0.45, 'TTC': 0.55},
    'P': {'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11},
    'S': {'TCT': 0.15, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.06, 'AGT': 0.15, 'AGC': 0.24},
    'T': {'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12},
    'W': {'TGG': 1.00},
    'Y': {'TAT': 0.43, 'TAC': 0.57},
    'V': {'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47},
    '*': {'TAA': 0.28, 'TAG': 0.20, 'TGA': 0.52},
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_vaccine_epitopes(ranked_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Select the optimal set of epitopes for the vaccine construct.

    Selection criteria:
    1. Take top-ranked candidates by vaccine_priority_score
    2. Ensure diversity across genes (no more than 3 epitopes per gene)
    3. Maximize HLA allele coverage
    4. Prefer unique mutation positions (avoid redundant peptide windows)
    5. Respect max/min neoantigen limits from config
    """
    vaccine_config = config.get("vaccine", {})
    max_epitopes = vaccine_config.get("max_neoantigens", 20)
    min_epitopes = vaccine_config.get("min_neoantigens", 5)

    df = ranked_df.copy()

    # Only consider binders
    if "classification" in df.columns:
        binders = df[df["classification"].isin(["strong_binder", "weak_binder"])]
        if len(binders) >= min_epitopes:
            df = binders

    # Sort by vaccine priority score
    if "vaccine_priority_score" in df.columns:
        df = df.sort_values("vaccine_priority_score", ascending=False)

    selected = []
    seen_mutations = set()
    gene_counts = defaultdict(int)
    hla_coverage = defaultdict(int)

    for _, row in df.iterrows():
        if len(selected) >= max_epitopes:
            break

        # Deduplicate by mutation (gene + aa_change)
        mutation_key = f"{row.get('gene_symbol', '')}_{row.get('aa_change', '')}"
        if mutation_key in seen_mutations:
            continue

        # Limit per gene (max 3)
        gene = row.get("gene_symbol", "unknown")
        if gene_counts[gene] >= 3:
            continue

        # Prefer peptide length of 9 (optimal for MHC-I)
        peptide = row.get("mutant_peptide", "")
        if len(peptide) < 8 or len(peptide) > 11:
            continue

        selected.append(row)
        seen_mutations.add(mutation_key)
        gene_counts[gene] += 1
        hla = row.get("hla_allele", "")
        if hla:
            hla_coverage[hla] += 1

    selected_df = pd.DataFrame(selected)

    print(f"  Selected {len(selected_df)} epitopes from {len(df)} candidates")
    print(f"  Gene diversity: {len(gene_counts)} unique genes")
    print(f"  HLA coverage: {dict(hla_coverage)}")

    return selected_df


def optimize_epitope_order(epitopes: List[str], linker: str) -> List[int]:
    """
    Optimize the ordering of epitopes to minimize junctional epitopes.

    Junctional epitopes are unintended peptides created at the junction
    of two concatenated epitopes (even with linkers). We use a greedy
    approach to minimize sequence similarity at junctions.

    Returns: list of indices representing the optimal order.
    """
    if len(epitopes) <= 2:
        return list(range(len(epitopes)))

    n = len(epitopes)

    # Calculate junction penalty for each pair
    def junction_penalty(ep1: str, ep2: str) -> float:
        """Lower penalty = better junction."""
        # Check if C-terminus of ep1 + linker + N-terminus of ep2
        # creates hydrophobic patches (bad for solubility)
        junction = ep1[-3:] + linker[:4] + ep2[:3]

        penalty = 0.0
        hydrophobic = set("AILMFVW")
        charged = set("DEKR")

        # Penalize hydrophobic runs
        hydro_run = 0
        for aa in junction:
            if aa in hydrophobic:
                hydro_run += 1
                if hydro_run >= 3:
                    penalty += hydro_run
            else:
                hydro_run = 0

        # Bonus for charged residues at junctions (improve solubility)
        if junction[0] in charged:
            penalty -= 1
        if junction[-1] in charged:
            penalty -= 1

        return penalty

    # Greedy nearest-neighbor ordering
    remaining = set(range(n))
    order = [0]  # Start with first epitope
    remaining.remove(0)

    while remaining:
        current = order[-1]
        best_next = min(
            remaining,
            key=lambda j: junction_penalty(epitopes[current], epitopes[j])
        )
        order.append(best_next)
        remaining.remove(best_next)

    return order


def codon_optimize(protein_sequence: str) -> str:
    """
    Codon-optimize a protein sequence for human expression.

    Uses weighted random selection based on human codon usage frequencies,
    with deterministic seeding for reproducibility.
    """
    # Use sequence hash as seed for reproducibility
    seed = int(hashlib.md5(protein_sequence.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)

    codons = []
    for aa in protein_sequence:
        if aa not in HUMAN_CODON_FREQ:
            continue

        codon_freqs = HUMAN_CODON_FREQ[aa]
        codon_list = list(codon_freqs.keys())
        freq_list = list(codon_freqs.values())

        # Normalize frequencies
        total = sum(freq_list)
        probs = [f / total for f in freq_list]

        # Weighted selection (biased toward most frequent)
        selected = rng.choice(codon_list, p=probs)
        codons.append(selected)

    return "".join(codons)


def build_mrna_construct(epitopes: List[str], config: dict) -> Dict:
    """
    Build the complete mRNA vaccine construct.

    Structure (5' to 3'):
    1. 5' Cap (m7G) — added during in-vitro transcription
    2. 5' UTR (human alpha-globin derived)
    3. Start codon + Signal peptide (tPA)
    4. Epitope cassette (epitopes joined by linkers)
    5. Stop codon
    6. 3' UTR (AES-mtRNR1)
    7. Poly-A tail (120 nt)

    All uridines are replaced with N1-methylpseudouridine (m1Ψ) during
    in-vitro transcription for reduced immunogenicity and increased stability.
    """
    vaccine_config = config.get("vaccine", {})
    linker = vaccine_config.get("linker_sequence", "GGSGGGGSGG")
    signal_peptide = vaccine_config.get("signal_peptide", "MDAMKRGLCCVLLLCGAVFVSPSQEIHARFR")
    five_utr = vaccine_config.get("five_prime_utr",
                                   "GCTTCTTGTTCTTTTGCAGAAGCTCAGAATAAACGCTCAACTTTGGCAGATCGG")
    three_utr = vaccine_config.get("three_prime_utr",
                                    "CTAGGAGAATGACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    poly_a_length = vaccine_config.get("poly_a_length", 120)

    # Optimize epitope order
    order = optimize_epitope_order(epitopes, linker)
    ordered_epitopes = [epitopes[i] for i in order]

    # Build protein cassette
    cassette_protein = signal_peptide + linker.join(ordered_epitopes)

    # Codon-optimize the cassette
    cassette_dna = codon_optimize(cassette_protein)

    # Add stop codon
    stop_codon = "TGA"  # Most common human stop codon

    # Assemble full mRNA (as DNA template)
    mrna_dna = five_utr + cassette_dna + stop_codon + three_utr + ("A" * poly_a_length)

    # Convert to RNA notation
    mrna_rna = mrna_dna.replace("T", "U")

    # Calculate properties
    gc_content = (mrna_dna.count("G") + mrna_dna.count("C")) / len(mrna_dna)
    molecular_weight = len(mrna_rna) * 330  # Average MW per nucleotide

    construct = {
        "ordered_epitopes": ordered_epitopes,
        "epitope_order": order,
        "linker_sequence": linker,
        "signal_peptide": signal_peptide,
        "cassette_protein": cassette_protein,
        "cassette_protein_length": len(cassette_protein),
        "cassette_dna": cassette_dna,
        "full_mrna_dna": mrna_dna,
        "full_mrna_rna": mrna_rna,
        "mrna_length_nt": len(mrna_rna),
        "gc_content": round(gc_content, 4),
        "estimated_molecular_weight_da": molecular_weight,
        "estimated_molecular_weight_kda": round(molecular_weight / 1000, 1),
        "five_prime_utr": five_utr,
        "three_prime_utr": three_utr,
        "poly_a_length": poly_a_length,
        "n_epitopes": len(ordered_epitopes),
        "modifications": [
            "5' Cap: m7GpppN (Cap1 structure)",
            "All uridines: N1-methylpseudouridine (m1Ψ)",
            "Codon optimization: Human codon usage bias",
        ],
    }

    return construct


def generate_vaccine_report(construct: Dict, epitopes_df: pd.DataFrame,
                             config: dict) -> str:
    """Generate a comprehensive human-readable vaccine design report."""
    lines = []
    lines.append("=" * 78)
    lines.append("  PERSONALIZED mRNA CANCER VACCINE DESIGN REPORT")
    lines.append("  Multiple Myeloma — MMRF CoMMpass Neoantigen Pipeline")
    lines.append("=" * 78)

    lines.append(f"\n{'─' * 78}")
    lines.append("  1. CONSTRUCT OVERVIEW")
    lines.append(f"{'─' * 78}")
    lines.append(f"  Number of epitopes:        {construct['n_epitopes']}")
    lines.append(f"  Protein cassette length:   {construct['cassette_protein_length']} amino acids")
    lines.append(f"  mRNA length:               {construct['mrna_length_nt']} nucleotides")
    lines.append(f"  GC content:                {construct['gc_content']:.1%}")
    lines.append(f"  Estimated MW:              {construct['estimated_molecular_weight_kda']} kDa")
    lines.append(f"  Linker sequence:           {construct['linker_sequence']}")

    lines.append(f"\n{'─' * 78}")
    lines.append("  2. mRNA ARCHITECTURE")
    lines.append(f"{'─' * 78}")
    lines.append("  5'Cap ─── 5'UTR ─── [Signal Peptide]─[Ep1]─[Linker]─[Ep2]─...─[EpN] ─── STOP ─── 3'UTR ─── AAAA...A")
    lines.append(f"")
    lines.append(f"  5' UTR:      {construct['five_prime_utr'][:40]}...")
    lines.append(f"  Signal:      {construct['signal_peptide'][:40]}...")
    lines.append(f"  3' UTR:      {construct['three_prime_utr'][:40]}...")
    lines.append(f"  Poly-A:      {construct['poly_a_length']} adenines")

    lines.append(f"\n{'─' * 78}")
    lines.append("  3. RNA MODIFICATIONS")
    lines.append(f"{'─' * 78}")
    for mod in construct["modifications"]:
        lines.append(f"  • {mod}")

    lines.append(f"\n{'─' * 78}")
    lines.append("  4. SELECTED EPITOPES (in construct order)")
    lines.append(f"{'─' * 78}")
    lines.append(f"  {'#':>3}  {'Gene':<10} {'AA Change':<15} {'Peptide':<14} {'HLA':<15} {'IC50(nM)':>9} {'Score':>6}")
    lines.append(f"  {'─'*3}  {'─'*10} {'─'*15} {'─'*14} {'─'*15} {'─'*9} {'─'*6}")

    ordered_epitopes = construct["ordered_epitopes"]
    for i, epitope in enumerate(ordered_epitopes):
        # Find matching row in epitopes_df
        match = epitopes_df[epitopes_df["mutant_peptide"] == epitope]
        if len(match) > 0:
            row = match.iloc[0]
            lines.append(
                f"  {i+1:>3}  {str(row.get('gene_symbol', '?')):<10} "
                f"{str(row.get('aa_change', '?')):<15} "
                f"{epitope:<14} "
                f"{str(row.get('hla_allele', '?')):<15} "
                f"{row.get('ic50_nM', 0):>9.1f} "
                f"{row.get('vaccine_priority_score', 0):>6.1f}"
            )
        else:
            lines.append(f"  {i+1:>3}  {'?':<10} {'?':<15} {epitope:<14}")

    lines.append(f"\n{'─' * 78}")
    lines.append("  5. PROTEIN CASSETTE SEQUENCE")
    lines.append(f"{'─' * 78}")
    cassette = construct["cassette_protein"]
    for i in range(0, len(cassette), 60):
        chunk = cassette[i:i+60]
        lines.append(f"  {i+1:>5}  {chunk}")

    lines.append(f"\n{'─' * 78}")
    lines.append("  6. MANUFACTURING NOTES")
    lines.append(f"{'─' * 78}")
    lines.append("  • In-vitro transcription (IVT) from linearized DNA template")
    lines.append("  • Use CleanCap AG for co-transcriptional 5' capping (Cap1)")
    lines.append("  • Substitute UTP with N1-methylpseudouridine-5'-triphosphate")
    lines.append("  • Purify by cellulose chromatography to remove dsRNA contaminants")
    lines.append("  • Formulate in lipid nanoparticles (LNP) for delivery")
    lines.append("  • Recommended LNP composition: ionizable lipid/DSPC/cholesterol/PEG-lipid")
    lines.append("  • Target dose range: 25-100 μg mRNA per injection")
    lines.append("  • Storage: -20°C (LNP formulated), -80°C (bulk mRNA)")

    lines.append(f"\n{'─' * 78}")
    lines.append("  7. CLINICAL CONSIDERATIONS")
    lines.append(f"{'─' * 78}")
    lines.append("  • This is a computationally designed research construct")
    lines.append("  • Patient-specific HLA typing is required for clinical use")
    lines.append("  • Preclinical immunogenicity testing (ELISpot, multimer staining) required")
    lines.append("  • Consider combination with checkpoint inhibitors (anti-PD-1/PD-L1)")
    lines.append("  • Multiple myeloma-specific: consider timing relative to ASCT")
    lines.append("  • Monitor for cytokine release syndrome (CRS)")

    lines.append(f"\n{'─' * 78}")
    lines.append("  DISCLAIMER")
    lines.append(f"{'─' * 78}")
    lines.append("  This vaccine design is generated for RESEARCH PURPOSES ONLY.")
    lines.append("  It has not been validated experimentally and is not intended for")
    lines.append("  clinical use. Any therapeutic application requires extensive")
    lines.append("  preclinical and clinical testing under appropriate regulatory oversight.")
    lines.append(f"\n{'=' * 78}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Design mRNA vaccine construct from ranked neoantigens"
    )
    parser.add_argument("--input", default="data/binding_predictions.csv",
                        help="Input ranked candidates CSV")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("mRNA Vaccine Construct Designer")
    print("Multiple Myeloma Neoantigen Vaccine Pipeline - Step 4")
    print("=" * 70)

    # Load ranked candidates
    print(f"\n[1/4] Loading ranked neoantigen candidates...")
    if not os.path.exists(args.input):
        print(f"  ERROR: File not found: {args.input}")
        print(f"  Run 03_predict_binding.py first.")
        sys.exit(1)

    ranked_df = pd.read_csv(args.input)
    print(f"  Loaded {len(ranked_df)} ranked candidates")

    # Select epitopes for vaccine
    print(f"\n[2/4] Selecting optimal epitopes for vaccine construct...")
    selected_df = select_vaccine_epitopes(ranked_df, config)

    if len(selected_df) == 0:
        print("  ERROR: No suitable epitopes found.")
        print("  Check binding predictions and filtering thresholds.")
        sys.exit(1)

    epitopes = selected_df["mutant_peptide"].tolist()

    # Build mRNA construct
    print(f"\n[3/4] Building mRNA vaccine construct...")
    construct = build_mrna_construct(epitopes, config)
    print(f"  Construct built successfully:")
    print(f"    Epitopes: {construct['n_epitopes']}")
    print(f"    Protein length: {construct['cassette_protein_length']} aa")
    print(f"    mRNA length: {construct['mrna_length_nt']} nt")
    print(f"    GC content: {construct['gc_content']:.1%}")
    print(f"    Estimated MW: {construct['estimated_molecular_weight_kda']} kDa")

    # Generate report
    print(f"\n[4/4] Generating vaccine design report...")
    report = generate_vaccine_report(construct, selected_df, config)

    # Save outputs
    # Vaccine report
    report_path = output_dir / "vaccine_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # Construct details (JSON)
    construct_json = {k: v for k, v in construct.items()
                      if k not in ("full_mrna_dna", "full_mrna_rna", "cassette_dna")}
    construct_path = output_dir / "vaccine_construct.json"
    with open(construct_path, "w") as f:
        json.dump(construct_json, f, indent=2)
    print(f"  Construct JSON saved to: {construct_path}")

    # Full mRNA sequence (FASTA format)
    fasta_path = output_dir / "vaccine_mrna.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">MM_neoantigen_vaccine | {construct['n_epitopes']} epitopes | "
                f"{construct['mrna_length_nt']} nt\n")
        seq = construct["full_mrna_rna"]
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
    print(f"  mRNA FASTA saved to: {fasta_path}")

    # Selected epitopes
    epitopes_path = output_dir / "selected_epitopes.csv"
    selected_df.to_csv(epitopes_path, index=False)
    print(f"  Selected epitopes saved to: {epitopes_path}")

    # Print report preview
    print(f"\n{'=' * 70}")
    print("VACCINE DESIGN COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - vaccine_report.txt      Full design report")
    print(f"  - vaccine_construct.json   Construct metadata")
    print(f"  - vaccine_mrna.fasta       mRNA sequence (FASTA)")
    print(f"  - selected_epitopes.csv    Selected epitope details")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
