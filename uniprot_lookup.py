#!/usr/bin/env python3
"""
UniProt Protein Sequence Lookup Module
=======================================
Provides real protein sequence lookups from UniProt to replace synthetic
peptide generation in the neoantigen prediction pipeline.

This module:
- Queries the UniProt REST API for human protein sequences (Swiss-Prot reviewed)
- Generates sliding-window peptides using real flanking residues around mutations
- Falls back to synthetic peptide generation when UniProt lookup fails
- Pre-caches key multiple myeloma driver protein sequences
- Validates mutation annotations against known protein sequences

Usage:
    from uniprot_lookup import generate_real_peptides, validate_mutation

    peptides = generate_real_peptides("KRAS", "G", 12, "D")
    is_valid = validate_mutation("KRAS", 12, "G")
"""

import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-cached sequences for key MM-related proteins
# These avoid API calls during testing and for the most commonly seen genes.
# ---------------------------------------------------------------------------

KNOWN_MM_PROTEINS = {
    # KRAS (UniProt P01116) - 189 aa, full canonical sequence
    "KRAS": (
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"
        "RKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCV"
        "FAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPS"
        "RTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREI"
        "RQYRLKKISKEEKTPGCVKIKKCIIM"
    ),
    # NRAS (UniProt P01111) - 189 aa, full canonical sequence
    "NRAS": (
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"
        "RKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCV"
        "FAINNSKSFADINLYREQIKRVKDSDDVPMVLVGNKCDLPT"
        "RTVDTKQAHELAKSYGIPFIETSAKTRQGVEDAFYTLVREI"
        "RQYRMKKLNSSDDGTQGCMGLPCVVM"
    ),
    # BRAF (UniProt P15056) - partial around V600 region (residues 551-650)
    # The full protein is 766 aa; we store the region relevant to V600 mutations.
    "BRAF": (
        "IKLIDIARQTAQGMDYLHAKSIIHRDLKSNNIFLHEDLTV"
        "KIGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKN"
        "PYSFQSDVYAFGIVLYELM"
    ),
    # TP53 (UniProt P04637) - partial around R175 region (residues 91-250)
    # The full protein is 393 aa; we store the region relevant to R175 hotspot.
    "TP53": (
        "WPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPAL"
        "NKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMT"
        "EVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNT"
        "FRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRP"
    ),
}

# Offset maps for partial sequences: maps protein residue 1 to the
# index into the stored partial string. For full-length sequences the
# offset is 0 (residue 1 == index 0).
_SEQUENCE_OFFSETS = {
    "KRAS": 0,
    "NRAS": 0,
    "BRAF": 550,   # stored string index 0 == protein residue 551
    "TP53": 90,    # stored string index 0 == protein residue 91
}

# ---------------------------------------------------------------------------
# Module-level cache and rate-limit state
# ---------------------------------------------------------------------------

_sequence_cache: dict = {}
_last_request_time: float = 0.0

# Initialise cache with known proteins
for _gene, _seq in KNOWN_MM_PROTEINS.items():
    _sequence_cache[_gene] = _seq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_protein_sequence(gene_symbol: str) -> Optional[str]:
    """
    Look up a human protein sequence from UniProt by gene symbol.

    Queries the UniProt REST API for reviewed (Swiss-Prot) human entries
    matching the exact gene symbol. Results are cached so that repeated
    calls for the same gene do not trigger additional HTTP requests.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol (e.g. "KRAS", "TP53").

    Returns
    -------
    Optional[str]
        The amino acid sequence as a plain string, or None if the lookup
        fails or no matching entry is found.
    """
    global _last_request_time

    # Return cached result if available
    if gene_symbol in _sequence_cache:
        logger.debug("Cache hit for %s", gene_symbol)
        return _sequence_cache[gene_symbol]

    # Rate limiting: wait at least 0.5 s between requests
    elapsed = time.time() - _last_request_time
    if elapsed < 0.5:
        time.sleep(0.5 - elapsed)

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "format": "fasta",
        "size": "1",
    }

    try:
        logger.info("Querying UniProt for %s ...", gene_symbol)
        response = requests.get(url, params=params, timeout=15)
        _last_request_time = time.time()
        response.raise_for_status()

        fasta_text = response.text.strip()
        if not fasta_text:
            logger.warning("UniProt returned empty response for %s", gene_symbol)
            _sequence_cache[gene_symbol] = None
            return None

        # Parse FASTA: skip header lines starting with '>'
        lines = fasta_text.split("\n")
        sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
        sequence = "".join(sequence_lines)

        if not sequence:
            logger.warning("No sequence extracted from UniProt response for %s",
                           gene_symbol)
            _sequence_cache[gene_symbol] = None
            return None

        _sequence_cache[gene_symbol] = sequence
        logger.info("Retrieved %d-aa sequence for %s", len(sequence), gene_symbol)
        return sequence

    except requests.RequestException as exc:
        logger.error("UniProt request failed for %s: %s", gene_symbol, exc)
        _sequence_cache[gene_symbol] = None
        return None


def generate_real_peptides(
    gene_symbol: str,
    wt_aa: str,
    position: int,
    mut_aa: str,
    peptide_lengths: list = None,
) -> list:
    """
    Generate sliding-window peptides around a mutation using real protein context.

    First attempts to retrieve the real protein sequence from UniProt. If the
    lookup succeeds and the sequence covers the mutation site, peptides are
    built with authentic flanking residues. Otherwise the function falls back
    to synthetic context (identical to ``generate_mutant_peptides`` in
    ``02_parse_mutations.py``).

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol.
    wt_aa : str
        Single-letter wildtype amino acid at *position*.
    position : int
        1-based protein residue number of the mutation.
    mut_aa : str
        Single-letter mutant amino acid.
    peptide_lengths : list, optional
        Peptide lengths to generate (default ``[8, 9, 10, 11]``).

    Returns
    -------
    list[dict]
        Each dict contains:
        - mutant_peptide (str)
        - wildtype_peptide (str)
        - peptide_length (int)
        - mutation_position_in_peptide (int) -- 0-based
        - wt_aa (str)
        - mut_aa (str)
        - protein_position (int)
        - sequence_source (str) -- "uniprot" or "synthetic"
    """
    if peptide_lengths is None:
        peptide_lengths = [8, 9, 10, 11]

    sequence = lookup_protein_sequence(gene_symbol)

    # Determine usable sequence and its offset
    offset = _SEQUENCE_OFFSETS.get(gene_symbol, 0)
    idx = position - 1 - offset  # 0-based index into stored string

    use_real = False
    if sequence is not None and 0 <= idx < len(sequence):
        use_real = True

    if use_real:
        return _peptides_from_real_sequence(
            sequence, offset, position, idx, wt_aa, mut_aa, peptide_lengths
        )

    # Fallback: synthetic peptide generation
    return _peptides_synthetic(wt_aa, position, mut_aa, peptide_lengths)


def validate_mutation(
    gene_symbol: str, position: int, expected_wt_aa: str
) -> bool:
    """
    Validate that the expected wildtype amino acid matches the protein sequence.

    Looks up the protein sequence from UniProt (or cache) and checks that the
    residue at *position* equals *expected_wt_aa*.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol.
    position : int
        1-based residue position.
    expected_wt_aa : str
        Expected single-letter amino acid.

    Returns
    -------
    bool
        True if the residue matches, False if it does not match or the
        sequence could not be retrieved / does not cover the position.
    """
    sequence = lookup_protein_sequence(gene_symbol)
    if sequence is None:
        logger.warning(
            "Cannot validate %s %s%d: sequence unavailable",
            gene_symbol, expected_wt_aa, position,
        )
        return False

    offset = _SEQUENCE_OFFSETS.get(gene_symbol, 0)
    idx = position - 1 - offset

    if idx < 0 or idx >= len(sequence):
        logger.warning(
            "Position %d is outside cached sequence range for %s (offset=%d, len=%d)",
            position, gene_symbol, offset, len(sequence),
        )
        return False

    actual_aa = sequence[idx]
    if actual_aa == expected_wt_aa:
        return True

    logger.warning(
        "Mutation validation failed for %s: expected %s at position %d but found %s",
        gene_symbol, expected_wt_aa, position, actual_aa,
    )
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _peptides_from_real_sequence(
    sequence: str,
    offset: int,
    position: int,
    idx: int,
    wt_aa: str,
    mut_aa: str,
    peptide_lengths: list,
) -> list:
    """Build sliding-window peptides from a real protein sequence."""
    peptides = []

    # Build the mutant version of the sequence
    mut_sequence = sequence[:idx] + mut_aa + sequence[idx + 1:]

    for length in peptide_lengths:
        for mut_pos in range(length):
            # start index within the stored sequence string
            start = idx - mut_pos
            end = start + length

            if start < 0 or end > len(sequence):
                continue

            mutant_peptide = mut_sequence[start:end]
            wildtype_peptide = sequence[start:end]

            peptides.append({
                "mutant_peptide": mutant_peptide,
                "wildtype_peptide": wildtype_peptide,
                "peptide_length": length,
                "mutation_position_in_peptide": mut_pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "protein_position": position,
                "sequence_source": "uniprot",
            })

    return peptides


def _peptides_synthetic(
    wt_aa: str, position: int, mut_aa: str, peptide_lengths: list
) -> list:
    """
    Generate peptides using synthetic flanking context.

    This mirrors the logic in ``generate_mutant_peptides`` from
    ``02_parse_mutations.py`` but adds the ``sequence_source`` field.
    """
    peptides = []
    common_residues = "AEKGDLSTPV"

    for length in peptide_lengths:
        for mut_pos in range(length):
            prefix_len = mut_pos
            suffix_len = length - mut_pos - 1

            prefix = "".join(
                common_residues[i % len(common_residues)]
                for i in range(prefix_len)
            )
            suffix = "".join(
                common_residues[(i + 3) % len(common_residues)]
                for i in range(suffix_len)
            )

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
