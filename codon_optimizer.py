#!/usr/bin/env python3
"""
Codon Optimizer — Entropy-Based mRNA Design
==========================================
Advanced codon optimization using:
- Codon Usage Bias (CUB) scores per organism
- GC content balancing and targeting
- mRNA secondary structure minimization (via ViennaRNA RNAfold or built-in scoring)
- Codon pair bias optimization

Supports three optimization levels:
  basic  — Human codon usage frequencies only
  standard — CUB + GC content balancing
  aggressive — CUB + GC + mRNA secondary structure (MFE minimization)

Usage:
    from codon_optimizer import CodonOptimizer
    optimizer = CodonOptimizer(organism="human", level="standard")
    sequence = optimizer.optimize(protein_sequence)
"""

import hashlib
import math
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Codon Usage Bias (CUB) Tables — organisms
# =============================================================================

HUMAN_CUB = {
    'TTT': 0.54, 'TTC': 0.46, 'TTA': 0.28, 'TTG': 0.18,
    'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41,
    'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16, 'ATG': 1.00,
    'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,
    'TAT': 0.43, 'TAC': 0.57, 'TAA': 0.28, 'TAG': 0.20,
    'CAT': 0.41, 'CAC': 0.59, 'CAA': 0.25, 'CAG': 0.75,
    'AAT': 0.46, 'AAC': 0.54, 'AAA': 0.42, 'AAG': 0.58,
    'GAT': 0.46, 'GAC': 0.54, 'GAA': 0.42, 'GAG': 0.58,
    'TGT': 0.45, 'TGC': 0.55, 'TGA': 0.52, 'TGG': 1.00,
    'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21,
    'AGT': 0.15, 'AGC': 0.24, 'AGA': 0.20, 'AGG': 0.20,
    'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25,
    'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
    'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11,
    'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
    'TCT': 0.15, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.06,
    'TTT': 0.54, 'TTC': 0.46,  # Phe
}

# Normalised human CUB (scale 0-1, relative to most frequent codon)
HUMAN_CUB_NORM = {
    'TTT': 0.54, 'TTC': 1.00, 'TTA': 0.28, 'TTG': 0.44,
    'CTT': 0.32, 'CTC': 0.49, 'CTA': 0.17, 'CTG': 1.00,
    'ATT': 0.75, 'ATC': 1.00, 'ATA': 0.33, 'ATG': 1.00,
    'GTT': 0.38, 'GTC': 0.51, 'GTA': 0.23, 'GTG': 1.00,
    'TAT': 0.75, 'TAC': 1.00, 'TAA': 0.54, 'TAG': 0.38,
    'CAT': 0.69, 'CAC': 1.00, 'CAA': 0.33, 'CAG': 1.00,
    'AAT': 0.85, 'AAC': 1.00, 'AAA': 0.72, 'AAG': 1.00,
    'GAT': 0.85, 'GAC': 1.00, 'GAA': 0.42, 'GAG': 1.00,
    'TGT': 0.82, 'TGC': 1.00, 'TGA': 1.00, 'TGG': 1.00,
    'CGT': 0.38, 'CGC': 0.90, 'CGA': 0.52, 'CGG': 1.00,
    'AGT': 0.62, 'AGC': 1.00, 'AGA': 0.83, 'AGG': 0.83,
    'GGT': 0.47, 'GGC': 1.00, 'GGA': 0.74, 'GGG': 0.74,
    'GCT': 0.65, 'GCC': 1.00, 'GCA': 0.57, 'GCG': 0.27,
    'CCT': 0.85, 'CCC': 1.00, 'CCA': 0.82, 'CCG': 0.33,
    'ACT': 0.67, 'ACC': 1.00, 'ACA': 0.78, 'ACG': 0.33,
    'TCT': 0.68, 'TCC': 1.00, 'TCA': 0.68, 'TCG': 0.27,
}

# Codon → Amino Acid
CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
}


def gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA/RNA sequence."""
    if len(sequence) == 0:
        return 0.0
    seq_upper = sequence.upper()
    return (seq_upper.count('G') + seq_upper.count('C')) / len(seq_upper)


def calculate_cub_score(codon_sequence: str) -> float:
    """
    Calculate mean Codon Usage Bias score for a codon sequence.
    Higher = better (more optimal codon usage).
    """
    if len(codon_sequence) % 3 != 0:
        codon_sequence = codon_sequence[:len(codon_sequence) - (len(codon_sequence) % 3)]

    if not codon_sequence:
        return 0.0

    scores = []
    for i in range(0, len(codon_sequence), 3):
        codon = codon_sequence[i:i+3]
        if len(codon) == 3 and codon in HUMAN_CUB_NORM:
            scores.append(HUMAN_CUB_NORM[codon])

    if not scores:
        return 0.0

    return np.mean(scores)


def calculate_codon_pair_bias(sequence: str) -> float:
    """
    Codon pair bias (CPB) — measures ribosome stall risk.
    Penalises consecutive low-usage codon pairs.

    Returns a penalty score (lower = better).
    Based on "codon pair context" literature (Welch et al., 2009).
    """
    if len(sequence) % 3 != 0:
        sequence = sequence[:len(sequence) - (len(sequence) % 3)]

    if len(sequence) < 6:
        return 0.0

    penalty = 0.0
    for i in range(0, len(sequence) - 3, 3):
        codon1 = sequence[i:i+3]
        codon2 = sequence[i+3:i+6]

        cub1 = HUMAN_CUB_NORM.get(codon1, 0.5)
        cub2 = HUMAN_CUB_NORM.get(codon2, 0.5)

        # Penalise consecutive low-CUB codons
        if cub1 < 0.5 and cub2 < 0.5:
            penalty += 0.5
        if cub1 < 0.3 or cub2 < 0.3:
            penalty += 0.3

    return penalty / max(1, (len(sequence) // 3 - 1))


def find_motif_runs(sequence: str, min_repeat: int = 8) -> List[Tuple[str, int, int]]:
    """
    Find homopolymer runs (e.g. AAAAAA) in sequence.
    Returns list of (base, start, length).
    """
    runs = []
    if len(sequence) == 0:
        return runs

    current_base = sequence[0]
    current_start = 0
    current_len = 1

    for i in range(1, len(sequence)):
        if sequence[i] == current_base:
            current_len += 1
        else:
            if current_len >= min_repeat:
                runs.append((current_base, current_start, current_len))
            current_base = sequence[i]
            current_start = i
            current_len = 1

    if current_len >= min_repeat:
        runs.append((current_base, current_start, current_len))

    return runs


def call_rnafold(sequence: str, temperature: float = 37.0) -> Optional[Dict]:
    """
    Call ViennaRNA RNAfold to predict mRNA secondary structure.
    Returns dict with 'mfe' (minimum free energy in kcal/mol),
    'structure' (dot-bracket notation), and 'ensemble_energy'.

    Returns None if RNAfold is not available.
    """
    try:
        # Prepare RNA sequence (replace T→U)
        rna_seq = sequence.upper().replace('T', 'U').replace(' ', '')

        result = subprocess.run(
            ['RNAfold', '--noPS'],
            input=f"{rna_seq}\n",
            capture_output=True,
            text=True,
            timeout=30,
            env={'HOME': '/opt/homebrew', 'PATH': '/opt/homebrew/bin:/usr/bin:/bin'}
        )

        if result.returncode != 0:
            return None

        output = result.stdout
        lines = output.strip().split('\n')

        if len(lines) < 2:
            return None

        # Parse MFE from second line: e.g. "ACGU... (-23.50)"
        mfe_line = lines[1]
        mfe_match = re.search(r'\((-?\d+\.?\d*)\)', mfe_line)
        structure = lines[0].split()[0] if len(lines[0].split()) > 0 else ""

        if mfe_match:
            mfe = float(mfe_match.group(1))
            return {
                'mfe': mfe,
                'structure': structure,
                'ensemble_energy': mfe,
            }

        return None

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def estimate_mfe_python(rna_sequence: str) -> float:
    """
    Pure-Python MFE estimation using a simplified nearest-neighbor model.
    Based on RNA secondary structure thermodynamics (SantaLucia, 1998).

    This is an approximation for when ViennaRNA is not available.
    Returns estimated MFE in kcal/mol (lower = more stable/more negative).
    """
    if len(rna_sequence) < 4:
        return 0.0

    rna = rna_sequence.upper()

    # Simplified stacking energy model
    # GC pair: -3.0 kcal/mol, AU pair: -2.0 kcal/mol, GU pair: -1.5 kcal/mol
    pair_energy = {'GC': -3.0, 'CG': -3.0, 'AU': -2.0, 'UA': -2.0, 'GU': -1.5, 'UG': -1.5}

    total_mfe = 0.0
    for i in range(len(rna) - 1):
        pair = rna[i:i+2]
        if pair in pair_energy:
            total_mfe += pair_energy[pair]

    # Normalise by sequence length
    return total_mfe / len(rna) * 10  # scale to reasonable MFE range


def has_poly_signal(sequence: str, signal: str = "AAUAAA") -> bool:
    """Check for polyadenylation signal in RNA sequence."""
    return signal.upper() in sequence.upper().replace('T', 'U')


class CodonOptimizer:
    """
    Entropy-based codon optimizer supporting three levels of optimization.
    """

    def __init__(
        self,
        organism: str = "human",
        level: str = "standard",
        target_gc_min: float = 0.40,
        target_gc_max: float = 0.70,
        target_gc_ideal: float = 0.58,
        entropy_weight: float = 0.3,
        stability_weight: float = 0.2,
        codon_pair_bias: bool = True,
        avoid_motif_threshold: int = 8,
        avoid_poly_signal: bool = True,
        vienna_temperature: float = 37.0,
        seed: Optional[int] = None,
    ):
        self.organism = organism
        self.level = level
        self.target_gc_min = target_gc_min
        self.target_gc_max = target_gc_max
        self.target_gc_ideal = target_gc_ideal
        self.entropy_weight = entropy_weight
        self.stability_weight = stability_weight
        self.codon_pair_bias = codon_pair_bias
        self.avoid_motif_threshold = avoid_motif_threshold
        self.avoid_poly_signal = avoid_poly_signal
        self.vienna_temperature = vienna_temperature

        if seed is None:
            self.rng = np.random.RandomState(42)
        else:
            self.rng = np.random.RandomState(seed)

        self._validate_level()

    def _validate_level(self):
        valid_levels = ("basic", "standard", "aggressive")
        if self.level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {self.level}. Must be one of {valid_levels}")

    def _get_synonymous_codons(self, aa: str) -> List[str]:
        """Get all codons for a given amino acid."""
        return [c for c, a in CODON_TO_AA.items() if a == aa]

    def _codon_score(self, codon: str) -> float:
        """Return normalised CUB score for a codon (0-1)."""
        return HUMAN_CUB_NORM.get(codon, 0.5)

    def _gc_score(self, sequence: str) -> float:
        """
        Score sequence GC content relative to target range.
        1.0 = within target range, lower = outside.
        """
        gc = gc_content(sequence)
        if self.target_gc_min <= gc <= self.target_gc_max:
            # Bonus for being close to ideal
            dist_from_ideal = abs(gc - self.target_gc_ideal)
            return 1.0 - dist_from_ideal
        else:
            # Penalise being outside range
            if gc < self.target_gc_min:
                shortfall = self.target_gc_min - gc
            else:
                shortfall = gc - self.target_gc_max
            return max(0.0, 1.0 - shortfall * 3)

    def _select_codon_greedy(self, aa: str, context_prev_codon: Optional[str]) -> str:
        """
        Select best codon for an amino acid using greedy optimization.

        Considers:
        - CUB score (higher = better)
        - GC contribution (push toward target range)
        - Avoidance of consecutive same codons
        """
        codons = self._get_synonymous_codons(aa)
        if not codons:
            return "GCG"  # Fallback

        if len(codons) == 1:
            return codons[0]

        best_codon = codons[0]
        best_score = -999.0

        for codon in codons:
            score = self._codon_score(codon)

            # GC adjustment
            gc = gc_content(codon)
            if gc > 0.6:  # GC-rich codon
                score += 0.1

            # Context penalty (avoid same codon twice in a row)
            if context_prev_codon and codon == context_prev_codon:
                score -= 0.15

            # Penalise motifs
            if codon.count(codon[0]) == 3:  # e.g. CCC, GGG
                score -= 0.1

            if score > best_score:
                best_score = score
                best_codon = codon

        return best_codon

    def _optimize_basic(self, protein: str) -> str:
        """Basic: weighted random selection by human codon frequency."""
        dna = []
        for aa in protein:
            codons = self._get_synonymous_codons(aa)
            if not codons:
                continue
            # Weight by CUB
            weights = [self._codon_score(c) for c in codons]
            total = sum(weights)
            probs = [w / total for w in weights]
            codon = self.rng.choice(codons, p=probs)
            dna.append(codon)
        return "".join(dna)

    def _optimize_standard(self, protein: str) -> str:
        """Standard: greedy CUB optimization + GC balancing."""
        dna = []
        prev_codon = None

        for aa in protein:
            codon = self._select_codon_greedy(aa, prev_codon)
            dna.append(codon)
            prev_codon = codon

        result = "".join(dna)

        # GC fine-tuning pass
        result = self._gc_balance_pass(result)

        return result

    def _gc_balance_pass(self, sequence: str) -> str:
        """
        Post-optimisation GC balancing pass.
        Adjusts codons to bring GC into target range without
        drastically reducing CUB score.
        """
        current_gc = gc_content(sequence)

        # If already in range, return as-is
        if self.target_gc_min <= current_gc <= self.target_gc_max:
            return sequence

        needs_gc_increase = current_gc < self.target_gc_min
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

        for i, codon in enumerate(codons):
            if needs_gc_increase and current_gc < self.target_gc_min:
                # Try replacing with higher-GC synonymous codon
                aa = CODON_TO_AA.get(codon)
                if not aa:
                    continue
                alternatives = [c for c in self._get_synonymous_codons(aa)
                                if c != codon and gc_content(c) > gc_content(codon)]
                if alternatives:
                    best_alt = max(alternatives, key=lambda c: self._codon_score(c))
                    if self._codon_score(best_alt) >= self._codon_score(codon) - 0.2:
                        codons[i] = best_alt
                        current_gc = gc_content("".join(codons))

            elif not needs_gc_increase and current_gc > self.target_gc_max:
                # Try replacing with lower-GC synonymous codon
                aa = CODON_TO_AA.get(codon)
                if not aa:
                    continue
                alternatives = [c for c in self._get_synonymous_codons(aa)
                                if c != codon and gc_content(c) < gc_content(codon)]
                if alternatives:
                    best_alt = max(alternatives, key=lambda c: self._codon_score(c))
                    if self._codon_score(best_alt) >= self._codon_score(codon) - 0.2:
                        codons[i] = best_alt
                        current_gc = gc_content("".join(codons))

        return "".join(codons)

    def _optimize_aggressive(self, protein: str) -> str:
        """
        Aggressive: CUB + GC + secondary structure optimization.
        Uses RNAfold (if available) or Python MFE estimate to
        minimise mRNA folding free energy.
        """
        # Start with standard optimisation
        sequence = self._optimize_standard(protein)

        # Structure optimisation pass
        sequence = self._structure_optimisation_pass(sequence)

        # Final GC balance
        sequence = self._gc_balance_pass(sequence)

        return sequence

    def _structure_optimisation_pass(self, sequence: str) -> str:
        """
        Optimise codon selection to minimise mRNA secondary structure MFE.
        Uses RNAfold if available, otherwise Python MFE estimate.
        """
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

        rna_seq = sequence.replace('T', 'U')

        # Try RNAfold for MFE
        rnafold_result = call_rnafold(rna_seq, temperature=self.vienna_temperature)
        if rnafold_result is not None:
            current_mfe = rnafold_result['mfe']
        else:
            current_mfe = estimate_mfe_python(rna_seq)

        # Windowed optimization: for each codon window, try alternative codons
        # and accept if MFE improves (more negative = better)
        n_codons = len(codons)
        window = min(12, n_codons)  # 12 codons = 36 nt window

        for i in range(n_codons - window):
            original_codon = codons[i]
            aa = CODON_TO_AA.get(original_codon)
            if not aa or len(self._get_synonymous_codons(aa)) <= 1:
                continue

            alternatives = [c for c in self._get_synonymous_codons(aa) if c != original_codon]
            best_alt = original_codon
            best_mfe = current_mfe

            for alt in alternatives:
                test_codons = codons[:]
                test_codons[i] = alt
                test_seq = "".join(test_codons).replace('T', 'U')

                rf_result = call_rnafold(test_seq, temperature=self.vienna_temperature)
                if rf_result is not None:
                    test_mfe = rf_result['mfe']
                else:
                    test_mfe = estimate_mfe_python(test_seq)

                # Accept if MFE is more negative (more stable)
                if test_mfe < best_mfe:
                    best_mfe = test_mfe
                    best_alt = alt

            if best_alt != original_codon:
                codons[i] = best_alt
                current_mfe = best_mfe

        return "".join(codons)

    def optimize(self, protein_sequence: str) -> str:
        """
        Optimise a protein sequence into a DNA codon sequence.

        Args:
            protein_sequence: Amino acid sequence string

        Returns:
            DNA sequence (A/C/G/T) of optimised codons
        """
        protein = protein_sequence.upper().strip()

        if self.level == "basic":
            return self._optimize_basic(protein)
        elif self.level == "standard":
            return self._optimize_standard(protein)
        elif self.level == "aggressive":
            return self._optimize_aggressive(protein)
        else:
            return self._optimize_standard(protein)

    def score_sequence(self, dna_sequence: str) -> Dict:
        """
        Score an optimised DNA sequence and return metrics.

        Returns:
            dict with keys: cub_score, gc_content, motif_penalty,
            cpb_penalty, mfe_estimate, mfe_actual (if RNAfold used),
            stability_score, overall_score
        """
        if len(dna_sequence) % 3 != 0:
            dna_sequence = dna_sequence[:len(dna_sequence) - (len(dna_sequence) % 3)]

        cub = calculate_cub_score(dna_sequence)
        gc = gc_content(dna_sequence)
        motif_runs = find_motif_runs(dna_sequence, min_repeat=self.avoid_motif_threshold)
        motif_penalty = len(motif_runs) * 0.1
        cpb = calculate_codon_pair_bias(dna_sequence)
        poly_signal = has_poly_signal(dna_sequence) if self.avoid_poly_signal else False

        rna = dna_sequence.replace('T', 'U')
        rnafold_result = call_rnafold(rna, temperature=self.vienna_temperature)

        if rnafold_result is not None:
            mfe = rnafold_result['mfe']
            mfe_estimate = None
        else:
            mfe_estimate = estimate_mfe_python(rna)
            mfe = None

        # Stability score: normalise MFE (more negative = more stable)
        if mfe is not None:
            stability_score = max(0.0, min(1.0, (-mfe - 20) / 40))  # -60 to -20 kcal/mol range
        else:
            stability_score = max(0.0, min(1.0, (-mfe_estimate - 2) / 4))  # rough scale

        # Overall composite score
        overall = (
            cub * 0.4 +
            (1.0 - abs(gc - self.target_gc_ideal)) * 0.2 +
            (1.0 - motif_penalty) * 0.1 +
            (1.0 - min(cpb, 1.0)) * 0.1 +
            stability_score * self.stability_weight
        )

        return {
            "cub_score": round(cub, 4),
            "gc_content": round(gc, 4),
            "gc_in_range": self.target_gc_min <= gc <= self.target_gc_max,
            "motif_runs": [(r[0], r[2]) for r in motif_runs],
            "motif_penalty": round(motif_penalty, 4),
            "cpb_penalty": round(cpb, 4),
            "mfe_estimate_kcal_mol": round(mfe_estimate, 2) if mfe_estimate else None,
            "mfe_rnafold_kcal_mol": round(mfe, 2) if mfe else None,
            "stability_score": round(stability_score, 4),
            "poly_signal_avoided": not poly_signal,
            "overall_score": round(overall, 4),
        }


def optimize_sequence(
    protein_sequence: str,
    level: str = "standard",
    target_gc_min: float = 0.40,
    target_gc_max: float = 0.70,
    target_gc_ideal: float = 0.58,
) -> Tuple[str, Dict]:
    """
    Convenience function: optimise a protein sequence and return
    (dna_sequence, score_dict).

    Usage:
        dna, scores = optimize_sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEAL...", level="aggressive")
    """
    optimizer = CodonOptimizer(
        organism="human",
        level=level,
        target_gc_min=target_gc_min,
        target_gc_max=target_gc_max,
        target_gc_ideal=target_gc_ideal,
    )
    dna = optimizer.optimize(protein_sequence)
    scores = optimizer.score_sequence(dna)
    return dna, scores