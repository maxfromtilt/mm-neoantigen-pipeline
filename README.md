# Multiple Myeloma Neoantigen Vaccine Pipeline

A computational pipeline for designing personalized mRNA cancer vaccines targeting multiple myeloma (MM), using open-access genomic data from the [MMRF CoMMpass Study](https://themmrf.org/we-are-curing-multiple-myeloma/mmrf-commpass-study/) via the [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/).

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  01: Fetch Data  │────▶│  02: Parse Muts   │────▶│  03: Predict     │────▶│  04: Design      │
│  (GDC API)       │     │  & Gen Candidates │     │  MHC Binding     │     │  mRNA Vaccine    │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_fetch_mmrf_data.py` | Fetches somatic mutations, clinical data, and gene expression metadata from GDC |
| 2 | `02_parse_mutations.py` | Filters protein-altering mutations, generates mutant peptides, scores immunogenicity |
| 3 | `03_predict_binding.py` | Predicts MHC-I binding affinity (PSSM or MHCflurry), ranks by vaccine priority |
| 4 | `04_design_vaccine.py` | Selects epitopes, optimizes ordering, builds full mRNA construct with UTRs/signal peptide |
| — | `05_run_pipeline.py` | Orchestrator that runs all steps end-to-end |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python 05_run_pipeline.py

# Or run individual steps
python 01_fetch_mmrf_data.py --max-cases 50
python 02_parse_mutations.py
python 03_predict_binding.py
python 04_design_vaccine.py
```

### Pipeline Options

```bash
# Start from a specific step (skips earlier steps)
python 05_run_pipeline.py --step 3

# Skip data fetch if you already have the data
python 05_run_pipeline.py --skip-fetch

# Dry run (show what would execute)
python 05_run_pipeline.py --dry-run

# Custom case/mutation limits
python 05_run_pipeline.py --max-cases 100 --max-mutations 20000
```

## Output Files

| File | Description |
|------|-------------|
| `data/mmrf_cases.csv` | Clinical data (demographics, diagnoses) |
| `data/mmrf_mutations.csv` | Raw somatic mutations from GDC |
| `data/neoantigen_candidates.csv` | Filtered candidates with immunogenicity scores |
| `data/binding_predictions.csv` | MHC binding predictions and rankings |
| `output/vaccine_report.txt` | Full vaccine design report |
| `output/vaccine_mrna.fasta` | mRNA sequence in FASTA format |
| `output/vaccine_construct.json` | Construct metadata (epitopes, properties) |
| `output/selected_epitopes.csv` | Final selected epitopes |

## Configuration

Edit `config.yaml` to customize:

- **GDC settings** — project, data types, case limits
- **Mutation filters** — VAF threshold, consequence types, driver genes
- **Neoantigen prediction** — peptide lengths, HLA alleles, binding thresholds
- **Vaccine design** — max epitopes, linker sequence, UTRs, signal peptide, poly-A length

## How It Works

### Data Source
The pipeline uses the **MMRF CoMMpass Study** (~995 newly diagnosed MM patients) with whole genome/exome sequencing, RNA-seq, and clinical data — all accessible through the GDC open-access API.

### Neoantigen Selection
Mutations are filtered to protein-altering variants (missense, frameshift, indels) and scored by:
- Physicochemical distance between wildtype and mutant amino acids
- Known MM driver gene status (KRAS, NRAS, BRAF, TP53, etc.)
- MHC-I binding affinity (IC50) across common HLA alleles
- Agretopicity (mutant vs wildtype binding ratio)
- Foreignness score

### mRNA Vaccine Design
The construct follows established mRNA vaccine architecture:
- **5' Cap** — m7G Cap1 structure
- **5' UTR** — Human alpha-globin derived
- **Signal peptide** — Human tPA leader sequence
- **Epitope cassette** — Ordered epitopes joined by GSG linkers
- **3' UTR** — AES-mtRNR1 combination (BioNTech design)
- **Poly-A tail** — 120 nucleotides
- **Modification** — N1-methylpseudouridine (m1Ψ) replacing all uridines
- **Codon optimization** — Human codon usage bias

## Dependencies

- Python 3.8+
- pandas, numpy, biopython
- requests (GDC API access)
- scikit-learn
- PyYAML, tqdm
- **Optional:** [MHCflurry](https://github.com/openvax/mhcflurry) for production-grade binding predictions

## Disclaimer

This pipeline is for **research and educational purposes only**. The vaccine designs are computationally generated and have not been experimentally validated. Clinical application requires extensive preclinical testing, regulatory approval, and patient-specific HLA typing. This tool is not a medical device and should not be used for clinical decision-making.

## References

- MMRF CoMMpass Study: https://themmrf.org/we-are-curing-multiple-myeloma/mmrf-commpass-study/
- GDC Data Portal: https://portal.gdc.cancer.gov/
- GDC API Documentation: https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/
- MHCflurry: O'Donnell et al., Cell Systems 2020
- BioNTech individualized neoantigen vaccines: Sahin et al., Nature 2017
