#!/usr/bin/env python3
"""
Safety Screening Module
========================
Screens vaccine peptide candidates for potential safety concerns including
off-target binding to self-antigens and autoimmune risk.

All reference data is pre-built from published literature.
No external API calls required.

Usage:
    from safety_screen import screen_candidates
    screened_df = screen_candidates(candidates_df)
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# Human Self-Peptides Database
# =============================================================================
# Common MHC-I presented self-peptides from highly expressed human proteins.
# These are peptides the immune system is tolerant to — vaccine candidates
# that look too similar risk triggering autoimmunity.
#
# Sources: Human Immune Peptidome Project, HLA Ligand Atlas, IEDB self-peptides
# =============================================================================

HUMAN_SELF_PEPTIDES = [
    # Housekeeping proteins (very highly expressed)
    {"peptide": "GILGFVFTL", "source_protein": "Influenza M1 (common memory)", "category": "viral_memory"},
    {"peptide": "NLVPMVATV", "source_protein": "CMV pp65 (common memory)", "category": "viral_memory"},
    {"peptide": "GLCTLVAML", "source_protein": "EBV BMLF1", "category": "viral_memory"},
    # GAPDH-derived self peptides
    {"peptide": "VVDLMAAHM", "source_protein": "GAPDH", "category": "housekeeping"},
    {"peptide": "LISWYDNEF", "source_protein": "GAPDH", "category": "housekeeping"},
    {"peptide": "GKVKVGVNG", "source_protein": "GAPDH", "category": "housekeeping"},
    # Beta-actin self peptides
    {"peptide": "DLYANTVLS", "source_protein": "ACTB (beta-actin)", "category": "housekeeping"},
    {"peptide": "AVFPSIVGR", "source_protein": "ACTB", "category": "housekeeping"},
    {"peptide": "YPIEHGIVT", "source_protein": "ACTB", "category": "housekeeping"},
    # HSP70 self peptides
    {"peptide": "TTPSVVAFT", "source_protein": "HSP70 (HSPA1A)", "category": "stress_response"},
    {"peptide": "NQVAMNPTN", "source_protein": "HSP70", "category": "stress_response"},
    # Ubiquitin self peptides
    {"peptide": "MQIFVKTLT", "source_protein": "Ubiquitin B", "category": "housekeeping"},
    {"peptide": "LIFAGKQLE", "source_protein": "Ubiquitin B", "category": "housekeeping"},
    # Histone self peptides
    {"peptide": "AAIRGDEELA", "source_protein": "Histone H2B", "category": "housekeeping"},
    {"peptide": "STELLIRK", "source_protein": "Histone H3", "category": "housekeeping"},
    # Ribosomal protein self peptides
    {"peptide": "SLVSKGTLV", "source_protein": "RPL7A", "category": "housekeeping"},
    {"peptide": "VLKQVHPDT", "source_protein": "RPS3", "category": "housekeeping"},
    # HLA/MHC self peptides (immune system)
    {"peptide": "YPKTHVTHHP", "source_protein": "HLA-A", "category": "immune"},
    {"peptide": "DTQFVRFDSD", "source_protein": "HLA-B", "category": "immune"},
    # Keratin self peptides
    {"peptide": "SLNNQFASFI", "source_protein": "KRT1 (keratin)", "category": "structural"},
    {"peptide": "KLALDIEIAT", "source_protein": "KRT10", "category": "structural"},
    # Albumin self peptides
    {"peptide": "FKDLGEENFK", "source_protein": "ALB (albumin)", "category": "plasma"},
    {"peptide": "KVPQVSTPTL", "source_protein": "ALB", "category": "plasma"},
    # Immunoglobulin self peptides (important for MM context)
    {"peptide": "VTVSSASTK", "source_protein": "IGHG1 (IgG1)", "category": "immunoglobulin"},
    {"peptide": "GPSVFPLAP", "source_protein": "IGHG1", "category": "immunoglobulin"},
    {"peptide": "TPEVTCVVV", "source_protein": "IGHG1", "category": "immunoglobulin"},
    {"peptide": "WYVDGVEVH", "source_protein": "IGHG1", "category": "immunoglobulin"},
    # Lambda/kappa light chain self peptides (relevant to MM)
    {"peptide": "QPKAAPSVT", "source_protein": "IGLC1 (lambda)", "category": "immunoglobulin"},
    {"peptide": "RTVAAPSVF", "source_protein": "IGKC (kappa)", "category": "immunoglobulin"},
    # Collagen self peptides
    {"peptide": "GPAGPPGPK", "source_protein": "COL1A1", "category": "structural"},
    # Transferrin self peptides
    {"peptide": "KPVDEYKDC", "source_protein": "TF (transferrin)", "category": "plasma"},
    # Myosin self peptides
    {"peptide": "ALQEAHQQT", "source_protein": "MYH9", "category": "structural"},
    # Tubulin self peptides
    {"peptide": "VVPGGDLAK", "source_protein": "TUBA1A (tubulin)", "category": "housekeeping"},
    {"peptide": "AVLVDLEPG", "source_protein": "TUBB (tubulin)", "category": "housekeeping"},
    # CD138/SDC1 self peptides (important — MM marker)
    {"peptide": "SLVIPACPL", "source_protein": "SDC1 (CD138)", "category": "mm_marker"},
    # Additional common self peptides
    {"peptide": "ALDFEQEMT", "source_protein": "ENO1 (enolase)", "category": "housekeeping"},
    {"peptide": "YQLDPTASI", "source_protein": "LDHA", "category": "housekeeping"},
    {"peptide": "GVVDSEDLP", "source_protein": "VIM (vimentin)", "category": "structural"},
    {"peptide": "YLAEVAAGD", "source_protein": "EEF1A1", "category": "housekeeping"},
    {"peptide": "LLQLVERIS", "source_protein": "XBP1 (important in MM)", "category": "mm_marker"},
]


# =============================================================================
# Autoimmune-Associated Genes
# =============================================================================
# Genes where mutations might trigger autoimmune responses.
# Targeting these genes carries additional safety considerations.
# =============================================================================

AUTOIMMUNE_RISK_GENES = {
    # Immune checkpoint / tolerance genes
    "CTLA4": {"risk": "high", "conditions": ["autoimmune lymphoproliferative syndrome"],
              "notes": "Immune checkpoint; blocking can cause autoimmunity"},
    "PDCD1": {"risk": "high", "conditions": ["autoimmune disorders"],
              "notes": "PD-1; tolerance maintenance"},
    "FOXP3": {"risk": "high", "conditions": ["IPEX syndrome"],
              "notes": "Master regulator of regulatory T cells"},
    "AIRE": {"risk": "high", "conditions": ["autoimmune polyendocrinopathy"],
             "notes": "Central tolerance in thymus"},
    # Inflammatory genes
    "TNF": {"risk": "medium", "conditions": ["rheumatoid arthritis", "IBD"],
            "notes": "Pro-inflammatory cytokine"},
    "IL6": {"risk": "medium", "conditions": ["Castleman disease"],
            "notes": "Pro-inflammatory; also MM growth factor"},
    # HLA genes
    "HLA-A": {"risk": "medium", "conditions": ["various autoimmune"],
              "notes": "Self-presentation molecules"},
    "HLA-B": {"risk": "medium", "conditions": ["various autoimmune"],
              "notes": "Self-presentation molecules"},
    # Complement system
    "C3": {"risk": "medium", "conditions": ["lupus-like syndrome"],
           "notes": "Complement pathway"},
    # Generally safe genes (MM drivers)
    "KRAS": {"risk": "low", "conditions": [], "notes": "Oncogene; not associated with autoimmunity"},
    "NRAS": {"risk": "low", "conditions": [], "notes": "Oncogene; not associated with autoimmunity"},
    "BRAF": {"risk": "low", "conditions": [], "notes": "Oncogene; not associated with autoimmunity"},
    "TP53": {"risk": "low", "conditions": [], "notes": "Tumor suppressor; mutant-specific targeting is safe"},
    "DIS3": {"risk": "low", "conditions": [], "notes": "MM-specific; no autoimmune association"},
    "FAM46C": {"risk": "low", "conditions": [], "notes": "MM-specific; no autoimmune association"},
}


def calculate_sequence_similarity(peptide1: str, peptide2: str) -> float:
    """
    Calculate sequence similarity between two peptides.

    Uses a simple identity score with position-weighted matching.
    Central positions weighted higher (more important for TCR recognition).

    Returns: similarity score 0.0 to 1.0
    """
    p1 = peptide1.upper()
    p2 = peptide2.upper()

    # Handle different length peptides via sliding window
    if len(p1) != len(p2):
        shorter, longer = (p1, p2) if len(p1) <= len(p2) else (p2, p1)
        best_sim = 0.0
        for i in range(len(longer) - len(shorter) + 1):
            window = longer[i:i+len(shorter)]
            sim = _identity_score(shorter, window)
            best_sim = max(best_sim, sim)
        return best_sim

    return _identity_score(p1, p2)


def _identity_score(seq1: str, seq2: str) -> float:
    """Position-weighted identity score for equal-length sequences."""
    if len(seq1) != len(seq2) or len(seq1) == 0:
        return 0.0

    n = len(seq1)
    total_weight = 0.0
    match_weight = 0.0

    for i in range(n):
        # Central positions weighted higher (TCR contact residues)
        if n // 3 <= i <= 2 * n // 3:
            weight = 1.5
        elif i == 0 or i == n - 1:
            weight = 0.7  # Terminal positions less important
        else:
            weight = 1.0

        total_weight += weight
        if seq1[i] == seq2[i]:
            match_weight += weight

    return match_weight / total_weight if total_weight > 0 else 0.0


def screen_self_similarity(peptide: str, threshold: float = 0.8) -> list:
    """
    Compare a candidate peptide against the self-peptide database.

    Args:
        peptide: Candidate neoantigen peptide sequence
        threshold: Minimum similarity to flag (0.0 to 1.0)

    Returns: List of concerning self-peptide matches with details.
    """
    hits = []
    peptide_upper = peptide.upper()

    for self_pep in HUMAN_SELF_PEPTIDES:
        similarity = calculate_sequence_similarity(
            peptide_upper, self_pep["peptide"]
        )

        if similarity >= threshold:
            risk = "high" if similarity >= 0.9 else "medium"
            hits.append({
                "self_peptide": self_pep["peptide"],
                "source_protein": self_pep["source_protein"],
                "category": self_pep["category"],
                "similarity": round(similarity, 3),
                "risk_level": risk,
            })

    # Sort by similarity descending
    hits.sort(key=lambda x: x["similarity"], reverse=True)
    return hits


def check_known_autoimmune_targets(gene_symbol: str) -> dict:
    """
    Check if a gene is associated with known autoimmune conditions.

    Returns risk assessment dict.
    """
    info = AUTOIMMUNE_RISK_GENES.get(gene_symbol)
    if info:
        return {
            "autoimmune_risk": info["risk"],
            "conditions": info["conditions"],
            "notes": info["notes"],
        }

    # Default: unknown gene → low risk
    return {
        "autoimmune_risk": "low",
        "conditions": [],
        "notes": "No known autoimmune association",
    }


def screen_candidates(candidates_df: pd.DataFrame,
                       similarity_threshold: float = 0.75) -> pd.DataFrame:
    """
    Screen all neoantigen candidates for safety concerns.

    Adds columns:
    - self_similarity_max: highest similarity to any self-peptide
    - self_similarity_hits: number of self-peptide matches above threshold
    - autoimmune_risk: gene-level autoimmune risk assessment
    - safety_flag: "pass" / "review" / "concern"
    """
    df = candidates_df.copy()

    sim_max = []
    sim_hits = []
    auto_risk = []
    safety_flags = []

    for _, row in df.iterrows():
        peptide = row.get("mutant_peptide", "")
        gene = row.get("gene_symbol", "")

        # Self-similarity screening
        hits = screen_self_similarity(peptide, threshold=similarity_threshold)
        max_sim = max((h["similarity"] for h in hits), default=0.0)
        sim_max.append(round(max_sim, 3))
        sim_hits.append(len(hits))

        # Autoimmune risk
        auto = check_known_autoimmune_targets(gene)
        auto_risk.append(auto["autoimmune_risk"])

        # Overall safety flag
        if max_sim >= 0.9 or auto["autoimmune_risk"] == "high":
            safety_flags.append("concern")
        elif max_sim >= 0.8 or auto["autoimmune_risk"] == "medium":
            safety_flags.append("review")
        else:
            safety_flags.append("pass")

    df["self_similarity_max"] = sim_max
    df["self_similarity_hits"] = sim_hits
    df["autoimmune_risk"] = auto_risk
    df["safety_flag"] = safety_flags

    return df
