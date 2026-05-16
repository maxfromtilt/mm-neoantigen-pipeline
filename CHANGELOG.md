# Changelog

All notable changes to the MMRF CoMMpass Neoantigen Vaccine Pipeline are documented here.

Entries are grouped by enhancement number and sorted by date (newest first).

---

## [Unreleased] — Enhancement 11: ClinVar Pathogenicity, TMB, Survival Analysis, HLA Typing, Clinical PDF Report

**Date:** 2026-05-16
**Files created:** `11_pathogenicity.py`, `12_survival_analysis.py`, `hla_typing.py`, `13_clinical_report.py`
**Files modified:** `app.py`, `config.yaml`, `requirements.txt`, `CHANGELOG.md`

### 11a — ClinVar / InterVar Pathogenicity Scoring (`11_pathogenicity.py`)

- New standalone module querying ClinVar public API (E-utilities) for each gene/mutation combination
- Uses `esearch.fcgi` to find ClinVar records by gene + amino acid change, then `esummary.fcgi` to fetch clinical significance
- Falls back to **InterVar-style rules** when ClinVar data is unavailable:
  - Truncating mutations (frameshift, stop-gain) in tumour suppressors = "Pathogenic"
  - Known oncogenes (KRAS, NRAS, BRAF) with missense = "Likely Pathogenic"
  - Silent/non-coding = "Benign/Likely Benign"
- Added `pathogenicity_class` column to neoantigen candidates: Pathogenic | Likely Pathogenic | VUS | Likely Benign | Benign
- Outputs `output/pathogenicity.csv` with gene, aa_change, class, clinvar_id, explanation
- JSON cache at `output/pathogenicity_cache.json` (reused on subsequent runs, respecting NCBI rate limits)

### 11b — TMB (Tumour Mutational Burden) Display (`app.py`)

- Added `calculate_tmb()` helper function computing mutations per Mb from patient prediction data
- TMB metric added to the Overview tab key metrics row (6th metric)
- Displays mutation count, capture size, and category (Very High/High/Moderate/Low) relative to MM population norm (~2-5 mut/MB)
- Expandable detail section explaining MM TMB norms and clinical relevance
- Category thresholds: very high >=10, high >=5, moderate >=2, low <2 mut/MB

### 11c — Survival Analysis (`12_survival_analysis.py`)

- Extracts per-patient survival from `data/mmrf_cases.csv` (days_to_death, days_to_last_follow_up, vital_status)
- Computes cohort-level KM statistics: median OS, mean OS, n patients, n deceased
- Per-gene survival breakdown: which genes are associated with better/worse outcomes in the MMRF cohort
- Outputs `output/survival.csv` (patient-level) and `output/gene_survival_km.csv` (gene-level)
- Added `load_patient_survival_data()` helper to `app.py` for displaying patient OS in sidebar

### 11d — arCasHLA-Style HLA Typing Enhancement (`hla_typing.py`)

- Extracts HLA alleles from MMRF patient clinical data (WES-based if available)
- Falls back to population frequency priors (European ancestry defaults, configurable)
- Adds HLA confidence score (High/Medium/Low) based on data source
- Outputs `data/hla_typing.json` with per-patient HLA class I/II alleles
- Displays patient HLA type + confidence in sidebar expander in `app.py`
- `load_patient_hla()` helper function integrated into app.py overview section

### 11e — Per-Patient Clinical PDF Report (`13_clinical_report.py`)

- Generates a formatted one-page PDF clinical report for any specific patient
- Sections: mutation summary, top 5 epitopes table, vaccine construct summary (epitopes, length, GC content), mutation frequency in MM cohort, clinical trial matching (top 3), limitations/disclaimer
- Uses `fpdf2` library for pure-Python PDF generation (no external dependencies)
- Integrated into `app.py` Vaccine Construct tab with "Generate Clinical PDF Report" button and direct PDF download
- Output path: `output/{patient_id}_clinical_report.pdf`

**config.yaml additions:**
- `clinvar` section: API endpoint, rate limits, output paths
- `tmb` section: capture sizes, MM median norm, category thresholds
- `survival` section: MMRF cases path, cohort median OS
- `hla_typing` section: population priors (European/global), confidence levels, output path
- `clinical_report` section: output dir, top epitopes count, trial match count

**requirements.txt additions:**
- `fpdf2>=2.7.0` (for clinical PDF report generation)

---

## [Unreleased] — Enhancement 10: TCR Repertoire Integration

**Date:** 2026-05-16
**Files created:** `09_tcr_repertoire.py`
**Files modified:** `config.yaml`, `05_run_pipeline.py`, `CHANGELOG.md`

- New pipeline step querying three public TCR databases:
  - **VDJdb** (`https://vdjdb.cvrgr.org/`) — annotated TCR–epitope pairs via GitHub release dataset
  - **McPAS-TCR** (`http://friedmanburg.com/McPAS-TCR/`) — pathology-associated TCR sequences
  - **TCRdb** (`http://tbis.tech/`) — general T-cell receptor repository
- For each epitope in `output/selected_epitopes.csv`, searches all three DBs for known reactive TCRs
- **Exact epitope matching** via sequence-indexed lookup (O(1) per epitope)
- **Cross-reactivity detection** via positional amino-acid identity (>=7 shared AA) and Levenshtein edit distance (<=2), flagging molecular mimicry analogues
- **TCR_reactivity_score** boost applied to `vaccine_priority_score`: +15 for known reactive TCR, +10 for cross-reactive analogue
- **Outputs:**
  - `output/tcr_epitope_links.csv` — epitope, tcr_alpha, tcr_beta, v_gene, epitope_source, affinityKd, reactiveness_score, match_type, database, cross_reactive flag
  - `output/tcr_validation_report.csv` — per-epitope summary: has_known_tcr, tcr_count, cross_reactive_analogs, clinical_relevance
  - `output/selected_epitopes_tcr_enriched.csv` — original epitopes with TCR score columns merged
  - `output/tcr_validation_report.txt` — human-readable summary
- **Fallback curated data** for all three DBs when APIs are unreachable (MM-relevant TCR–epitope pairs for KRAS G12V/D, TP53 hotspot, BRAF V600E, FGFR3, NY-ESO-1, MAGE-A3, WT1, and common viral epitopes)
- VDJdb segments fetched from GitHub raw URL as backup when release zip is unavailable
- `cross_reactivity_flag` column added to enriched epitope output for downstream vaccine design
- `vaccine_priority_score` updated to include TCR contribution

**config.yaml — new `tcr` section:**
```yaml
tcr:
  enabled: true
  vdjdb_api: "https://vdjdb.cvrgr.org/"
  mcpas_api: "http://friedmanburg.com/McPAS-TCR/"
  tcrdb_api: "http://tbis.tech/"
  min_tcr_count: 1
  tcr_boost: 15
  cross_reactive_boost: 10
  sim_match_min_shared: 7
```

---

## [Unreleased] — Enhancement 10: PeptideAtlas / PRIDE Proteomics Validation

**Date:** 2026-05-16
**Files created:** `10_peptide_atlas.py`
**Files modified:** `config.yaml` (added `peptide_atlas` section), `05_run_pipeline.py` (added step 7)

- New standalone step added to the pipeline (step 7)
- Queries **PeptideAtlas** (https://www.peptideatlas.org/api/v1) for protein detection
  status across hundreds of shotgun proteomics experiments, confirming whether
  gene products containing predicted neoepitopes are actually detected in
  real human tissue by mass spectrometry
- Queries **PRIDE** (https://www.ebi.ac.uk/pride/ws/archive/v2) as a secondary
  proteomics resource, supplementing PeptideAtlas with additional experiment coverage
- Performs **epitope-level proteome coverage check** -- validates that the exact
  mutant peptide sequence appears in PeptideAtlas's non-redundant peptide library
- **Expression confidence tiers** (High / Medium / Low / Not Detected) derived from
  peptide count and experiment count per gene
- **Priority score boost** applied to `vaccine_priority_score` for proteins with
  confirmed expression (configurable via `expression_confidence_boost`, default 8 points)
- Includes retry logic and rate limiting (0.25s delay between queries) to respect
  public API resources

**Outputs:**
- `output/proteomics/proteomics_validation.csv` -- per-gene protein detection summary
  with tissue types, experiment counts, and confidence scores
- `output/proteomics/epitope_proteome_coverage.csv` -- per-epitope proteome evidence
  showing which mutant peptides are confirmed in the proteome
- `output/proteomics/proteomics_report.txt` -- human-readable ranked report with
  tissue breakdown

**Config additions (`config.yaml`):**
```yaml
peptide_atlas:
  enabled: true
  base_url: "https://www.peptideatlas.org/api/v1"
  pride_base_url: "https://www.ebi.ac.uk/pride/ws/archive/v2"
  protein_detection_threshold: 1
  expression_confidence_boost: 8
```

**Pipeline integration:** Added as step 7 in `05_run_pipeline.py`, to be run
after the Ligandomics validation step. Can be run standalone:
`python 10_peptide_atlas.py --input output/selected_epitopes.csv`

---

## [Unreleased] — Enhancement 9: cBioPortal Integration

**Date:** 2026-05-16
**Files created:** `08_cbioportal.py`
**Files modified:** `config.yaml`, `05_run_pipeline.py`

#### New module: `08_cbioportal.py`

- Queries the cBioPortal public API v2 (`https://api.cbioportal.org/v2.0.0`) to fetch real-world MM patient cohort data:
  - Study listing and patient counts
  - Per-gene mutation counts across studies (paginated, capped at `max_patients`)
  - Overall survival (OS) and progression-free survival (PFS) per patient via clinical records endpoint
- Builds a **mutation-frequency database**: genes × frequency across cohorts, weighted by cohort size
- **Validates predicted epitopes** against the cohort: for each epitope's gene, retrieves real-world MM mutation frequency and survival signal
- Computes `clinical_relevance_score` = `frequency * 100 * survival_modifier` (clamped 0.5-2.0x) -- epitopes in frequently mutated, long-survival genes score highest
- Survival modifier: genes associated with longer-than-median OS get up to 2x boost (immune-responsive patients tend to be better vaccine candidates)
- Falls back to known MM driver gene list if API is unreachable

#### Outputs

| File | Description |
|------|-------------|
| `output/cbioportal_mutations.csv` | gene, mutation_frequency, patient_count, median_survival_months |
| `output/cbioportal_validation.csv` | epitope_sequence, gene, hla_allele, frequency_in_mm_cohort, clinical_relevance_score, survival_signal |

#### `config.yaml` -- new `cbioportal` section

```yaml
cbioportal:
  enabled: true
  api_base: "https://api.cbioportal.org/v2.0.0"
  cancer_study_ids:
    - "mm"          # Multiple Myeloma
    - "laml"        # Acute Myeloid Leukemia proxy
  max_patients: 500
  survival_data: true
  mutation_frequency_boost: true
```

#### `05_run_pipeline.py` changes

- Added step 6: "cBioPortal Validation" -- runs `08_cbioportal.py` after pipeline validation
- Outputs wired into `output/cbioportal_mutations.csv` and `output/cbioportal_validation.csv`

---

## [Unreleased] — Enhancement 8: HLA Ligandomics Mass-Spec Data Integration

**Date:** 2026-05-16
**Files created:** `07_ligandomics.py`
**Files modified:** `config.yaml`, `05_run_pipeline.py`

### New module: `07_ligandomics.py`

- Fetches mass-spec validated HLA-bound peptides from **PRIDE** (European Proteomics
  Research Institute, `https://www.ebi.ac.uk/pride/ws/archive/v2`) and **HMB**
  (Human Myelome Database, `https://www.hmadb.org`) via their public APIs.
- Builds a sequence-indexed lookup of experimental HLA-I ligand data for multiple
  myeloma and hematological malignancies.
- **Validates computationally predicted epitopes** from `binding_predictions.csv`
  against the mass-spec index: checks if any predicted MHC-I-binding peptide appears
  in the experimental PRIDE/HMB dataset.
- **Score boost**: epitopes with ≥2 independent MS experiments supporting HLA binding
  receive a configurable +20 point boost to `vaccine_priority_score` (configurable via
  `ligandomics.validation_boost` in `config.yaml`).
- Re-ranks all candidates after boosting.
- **JSON cache** at `output/ligandomics_cache/pride_peptides.json` and
  `output/ligandomics_cache/hmb_peptides.json` — re-used on subsequent runs unless
  `--force-refresh` is passed.
- **Standalone mode**: `--standalone` flag fetches and caches MS data without requiring
  a `binding_predictions.csv` input.

### `config.yaml` — new `ligandomics` section

```yaml
ligandomics:
  enabled: true
  pridesearch_api: "https://www.ebi.ac.uk/pride/ws/archive/v2"
  hmb_api: "https://www.hmadb.org"
  cancer_types:
    - "multiple myeloma"
    - "hematological malignancy"
  min_mass_spec_evidence: 2  # minimum peptide count to include
  validation_boost: 20  # extra score points for mass-spec validated epitopes
```

### `05_run_pipeline.py` — new step 6 added

Step 6 runs `07_ligandomics.py` after pipeline validation. Disabled via
`ligandomics.enabled: false` in config to skip without editing the orchestrator.

---

## [Enhancement 6 & 7] — 2026-05-16

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