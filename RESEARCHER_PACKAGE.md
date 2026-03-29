# Multiple Myeloma Personalized Neoantigen Vaccine Pipeline

## For Research Use Only

This package contains a computational pipeline for designing personalized mRNA
cancer vaccines from somatic mutation data, developed using the MMRF CoMMpass
multiple myeloma dataset.

---

## Quick Start

```bash
pip install -r requirements.txt
python 05_run_pipeline.py --patient data/mmrf_1251_mutations.csv
```

---

## Pipeline Architecture

```
Patient Somatic Mutations (VCF/MAF/CSV)
        │
        ▼
┌─────────────────────────────┐
│ 01_fetch_mmrf_data.py       │  Data acquisition from MMRF CoMMpass
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 02_parse_mutations.py       │  Filter mutations → generate 8-11mer peptides
│                             │  Missense, frameshift, in-frame indels
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 03_predict_binding.py       │  MHC-I binding prediction
│   • PSSM model (built-in)  │  Supports HLA-A*02:01, A*01:01, A*03:01,
│   • MHCflurry (optional)   │  A*24:02, B*07:02, B*08:01
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Enrichment & Safety         │
│   • external_validation.py  │  COSMIC/IEDB cross-reference
│   • expression_filter.py    │  Gene expression enrichment (MM context)
│   • safety_screen.py        │  Autoimmunity/toxicity screening
│   • uniprot_lookup.py       │  Protein annotation
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 04_design_vaccine.py        │  mRNA construct design
│   • Epitope selection       │  Signal peptide + epitopes + linkers
│   • Codon optimization      │  5'UTR/3'UTR + poly-A tail
│   • mRNA construct          │  GC optimization, MW calculation
└──────────────┬──────────────┘
               ▼
  Vaccine FASTA + Report + CSVs
```

---

## Patient Data: MMRF_1251

**Demographics:** White male, diagnosed age 43, alive at 4.4 years follow-up

### Results Summary (PSSM Model)

| Metric | Value |
|--------|-------|
| Total mutations parsed | 2,005 |
| Peptide candidates generated | 12,000 (2,000 mutations x 6 HLA alleles) |
| Binders (IC50 < 500nM) | 17 (0.1%) |
| Strong binders (IC50 < 50nM) | 0 |
| Safe candidates (passed screening) | ~11,800 |
| Epitopes in final vaccine | 5 |
| mRNA construct length | 568 nt |
| GC content | 43.0% |

### Selected Vaccine Epitopes

| # | Gene | Mutation | Peptide | HLA | IC50 (nM) | Priority Score |
|---|------|----------|---------|-----|-----------|---------------|
| 1 | OR4K5 | E196K | YIIKILIVV | HLA-A*02:01 | 225.8 | 48.1 |
| 2 | TMEM131 | G195A | TSNHAVFTY | HLA-A*01:01 | 478.1 | 45.0 |
| 3 | AHR | N123Y | LLQALYGFV | HLA-A*24:02 | 262.4 | 44.2 |
| 4 | OR2T33 | T132N | HPLRYPNLM | HLA-B*07:02 | 411.5 | 42.5 |
| 5 | KALRN | S1866I | LLAARQAIT | HLA-A*02:01 | 478.1 | 47.3 |

### Vaccine Construct Architecture

```
5'Cap ── 5'UTR ── [Signal Peptide]─[Ep1]─[GGSGGGGSGG]─[Ep2]─...─[Ep5] ── STOP ── 3'UTR ── polyA(120)
```

**RNA Modifications:**
- 5' Cap: m7GpppN (Cap1 structure)
- All uridines: N1-methylpseudouridine (m1Ψ)
- Codon optimization: Human codon usage bias

---

## Output Files

### `output/mmrf_1251/` — PSSM-based results
| File | Description |
|------|-------------|
| `vaccine_report.txt` | Full vaccine design report with manufacturing notes |
| `vaccine.fasta` | mRNA sequence in FASTA format |
| `selected_epitopes.csv` | Top epitopes selected for vaccine construct |
| `all_candidates_screened.csv` | All 12,000 candidates with binding, safety, expression data |

### `output/mmrf_1251_mhcflurry/` — MHCflurry neural network results (if available)
Same file structure as above, but using MHCflurry's neural network ensemble
(trained on 350K+ experimental binding measurements) instead of PSSM.

### `output/validation/` — Pipeline validation
| File | Description |
|------|-------------|
| `benchmark_results.csv` | Benchmark against known binders |
| `validation_binding_predictions.csv` | Validation predictions |
| `validation_vaccine.fasta` | Validation construct |

---

## Pipeline Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `01_fetch_mmrf_data.py` | Data fetcher | Downloads MMRF CoMMpass mutation data |
| `02_parse_mutations.py` | Mutation parser | Filters variants, generates peptide windows |
| `03_predict_binding.py` | Binding predictor | PSSM + MHCflurry MHC-I binding prediction |
| `04_design_vaccine.py` | Vaccine designer | mRNA construct with UTRs, signal peptide, codon optimization |
| `05_run_pipeline.py` | Pipeline runner | End-to-end orchestration |
| `06_validate_pipeline.py` | Validation | Benchmarks against known epitopes |
| `external_validation.py` | COSMIC/IEDB | Cross-references cancer gene databases |
| `expression_filter.py` | Expression | MM-specific gene expression enrichment |
| `safety_screen.py` | Safety | Autoimmunity, toxicity, essential gene screening |
| `uniprot_lookup.py` | UniProt | Protein annotation and functional data |
| `config.yaml` | Configuration | All pipeline parameters |

---

## Binding Prediction Models

### PSSM (Built-in)
- Position-Specific Scoring Matrix for 4 common HLA alleles
- Fast (~12,000 predictions/second), approximate
- Based on published anchor residue preferences
- Suitable for initial screening

### MHCflurry (Optional, Recommended)
- Neural network ensemble trained on 350K+ experimental measurements
- Includes presentation score (processing + binding)
- Much more accurate than PSSM
- Install: `pip install mhcflurry && mhcflurry-downloads fetch`

---

## Scoring System

The **Vaccine Priority Score** (0-100) combines:

| Component | Weight | Source |
|-----------|--------|--------|
| MHC binding affinity | 35% | IC50 from PSSM or MHCflurry |
| Agretopicity | 20% | Mutant vs wildtype binding ratio |
| Immunogenicity | 20% | Mutation type, position, conservation |
| Foreignness | 15% | Sequence divergence from self |
| Driver gene status | 10% | Known cancer driver gene bonus |

---

## Configuration

Key parameters in `config.yaml`:

```yaml
neoantigen:
  mhc_class_i:
    binding_threshold: 500    # IC50 nM cutoff for binder classification
    strong_binding: 50        # IC50 nM cutoff for strong binder
  peptide_lengths: [8, 9, 10, 11]
  default_hla_alleles:
    class_i:
      - HLA-A*02:01
      - HLA-A*01:01
      - HLA-A*03:01
      - HLA-A*24:02
      - HLA-B*07:02
      - HLA-B*08:01
```

---

## Testing

```bash
python -m pytest tests/ -v
python 06_validate_pipeline.py
```

---

## Dependencies

```
pandas, numpy, pyyaml, biopython
mhcflurry (optional — for production-grade binding predictions)
```

---

## Limitations & Next Steps

1. **PSSM binding predictions are approximate** — MHCflurry or NetMHCpan recommended for clinical-grade work
2. **HLA typing assumed** — Real patients need HLA genotyping (e.g., OptiType from WGS/RNA-seq)
3. **Expression data is population-level** — Patient-specific RNA-seq would improve epitope selection
4. **No MHC-II predictions yet** — CD4+ T helper epitopes would strengthen immune response
5. **No clonality analysis** — Subclonal mutations may not be present in all tumor cells
6. **Experimental validation required** — ELISpot, multimer staining, in vivo immunogenicity

---

## References

- MMRF CoMMpass Study: https://research.themmrf.org
- MHCflurry: O'Donnell et al., Cell Systems 2020
- mRNA vaccine design: Sahin et al., Nature Reviews Drug Discovery 2014
- Neoantigen prediction: Schumacher & Schreiber, Science 2015

---

**DISCLAIMER:** This pipeline is for RESEARCH PURPOSES ONLY. Not validated for
clinical use. Any therapeutic application requires extensive preclinical and
clinical testing under appropriate regulatory oversight (FDA IND, IRB approval).
