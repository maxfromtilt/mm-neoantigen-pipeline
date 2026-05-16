# Changelog

All notable changes to the MMRF CoMMpass Neoantigen Vaccine Pipeline are documented here.

Entries are grouped by enhancement number and sorted by date (newest first).

---

## [Unreleased] — Pipeline Enhancements 6 & 7

### Enhancement 6 — Clinical Trial Matching (`06_trial_matching.py`)

**Date:** 2026-05-16
**File created:** `06_trial_matching.py`

- New standalone step added to the pipeline
- Fetches actively recruiting multiple myeloma trials via ClinicalTrials.gov API v2
  (`https://clinicaltrials.gov/api/v2/studies`)
- Matches each epitope from `output/selected_epitopes.csv` against trials using
  immunotherapy-keyword proxy signals (no structured epitope-to-trial linkage exists
  in the public API)
- Scores trial relevance 0–100 based on neoantigen/vaccine/personalized keywords
- Outputs:
  - `output/trial_matches.csv` — epitope × trial matrix with NCT, phase, location, score
  - `output/trial_matches.json` — full structured data with summary statistics
  - `output/trial_matching_report.txt` — human-readable ranked report
- Falls back to 8 real-world mock MM trials when API is unreachable
- Includes eligibility summaries and full location lists for each trial

**Columns in `trial_matches.csv`:**
`epitope_sequence | nct_number | trial_title | phase | location | epitopes_matched_count | trial_relevance_score`

---

### Enhancement 7 — Codon Optimisation Upgrade (Entropy-Based)

**Date:** 2026-05-16
**Files created:** `codon_optimizer.py`
**Files modified:** `04_design_vaccine.py`, `config.yaml`

#### New module: `codon_optimizer.py`

- `CodonOptimizer` class implementing three optimisation levels:
  - `basic` — weighted random selection by human codon usage frequency
  - `standard` — greedy CUB maximisation + GC content balancing pass
  - `aggressive` — CUB + GC + mRNA secondary structure minimisation
    (via ViennaRNA `RNAfold` subprocess; falls back to Python nearest-neighbor MFE
    estimator when ViennaRNA is unavailable)
- Implements Codon Usage Bias (CUB) normalised scores (0–1 relative to most frequent codon)
- GC content target range: configurable min/max (default 0.40–0.70, ideal 0.58)
- Codon pair bias (CPB) penalty — reduces ribosome stall risk from consecutive low-usage pairs
- Homopolymer run detection (configurable threshold, default ≥8 consecutive identical nucleotides)
- Polyadenylation signal avoidance (AAUAAA)
- MFE scoring via `RNAfold` subprocess with pure-Python fallback
- Exports `optimize_sequence()` convenience function

#### `04_design_vaccine.py` changes

- Replaced simple frequency-weighted `codon_optimize()` with new entropy-based version
  wrapping `CodonOptimizer`
- `build_mrna_construct()` now reads optimisation level and GC targets from
  `config.yaml → vaccine.codon_optimisation`
- Construct dict extended with: `codon_optimisation_level`, `cassette_cub_score`,
  `cassette_gc_content`, `cassette_mfe_kcal_mol`, `cassette_stability_score`,
  `cassette_motif_runs`
- Vaccine report (section 1) updated with new codon optimisation metrics
- Console output updated with CUB score and stability score

#### `config.yaml` — new entries under `vaccine`

```yaml
vaccine:
  codon_optimisation:
    level: "standard"           # basic | standard | aggressive
    target_gc_content: 0.58    # Optimal human mRNA GC
    min_gc: 0.40
    max_gc: 0.70
    codon_pair_bias: true
    avoid_motif_threshold: 8
    entropy_weight: 0.3
    stability_weight: 0.2
    avoid_poly_signal: true
    vienna_temperature: 37.0
```

---

## Prior Enhancements (1–5, previously completed)

| # | Enhancement | Key files |
|---|-------------|-----------|
| 1 | TCGA data integration | `01_fetch_mmrf_data.py`, `config.yaml → tcga` |
| 2 | Ensemble binding prediction | `03_predict_binding.py`, `config.yaml → ensemble` |
| 3 | dNdScov driver scoring | `config.yaml → mutations.dndscov` |
| 4 | HLA typing | `config.yaml → hla_typing` |
| 5 | Tumor expression filtering | `expression_filter.py`, `config.yaml → expression_filter` |