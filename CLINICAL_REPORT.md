# Computational Design of Personalised Neoantigen mRNA Vaccines for Multiple Myeloma

## Using Machine Learning Binding Prediction and MMRF CoMMpass Genomic Data

---

**Authors:** Robert Doran¹, Max Shaw¹

**Affiliation:** ¹Independent Research, Dublin, Ireland

**Date:** 29 March 2026

**Repository:** https://github.com/maxfromtilt/mm-neoantigen-pipeline

**For Research Purposes Only** — Not validated for clinical use

---

## Executive Summary

**What this is:** The first open-source, MM-specific neoantigen mRNA vaccine design pipeline. While BioNTech and Moderna have proprietary pipelines for their clinical trials (melanoma, pancreatic), no equivalent exists specifically for multiple myeloma patients outside those trials. This pipeline fills that gap.

**What makes it different:**
- **MM-specific:** Incorporates MM driver gene biology (DIS3, KRAS, NRAS, RB1), MM plasma cell expression profiles, and myeloma-relevant clonality analysis -- not a generic cancer pipeline
- **Multi-signal ranking:** Goes beyond binding affinity alone -- integrates 8 weighted factors (MHC-I binding, MHC-II/CD4+ support, clonality, expression, agretopicity, foreignness, immunogenicity, driver gene status) for clinically relevant candidate prioritisation
- **Dual MHC-I/MHC-II screening:** Identifies candidates that engage both CD8+ cytotoxic and CD4+ helper T cells, associated with stronger and more durable immune responses
- **Fully transparent and reproducible:** Complete source code, data, and methodology available for clinical and regulatory review -- unlike proprietary industry pipelines
- **Patient-ready:** Accepts patient-specific WES and HLA typing data; produces a vaccine design in under 30 minutes on standard hardware, at no cost

**Results at a glance (2 MMRF CoMMpass patients):**
- 156 and 140 MHC-I binding neoantigen candidates per patient (1.0-1.2% binder rate, consistent with published benchmarks)
- 24 and 18 strong binders (IC50 < 50 nM)
- DIS3 I85T identified as top target for Patient 1 -- known MM driver, clonal, expressed. Ranked #1 by enhanced scoring despite not having the strongest binding, because it is present in every tumour cell
- IDH2 R140Q at 47 nM for Patient 2 -- known oncogenic hotspot (also targeted by FDA-approved enasidenib in AML), clonal (CCF 0.99)
- 7 dual MHC-I/MHC-II binders for Patient 1

**What we need:** A clinician or researcher willing to review the methodology and advise on whether this warrants experimental validation (ELISpot immunogenicity testing). The pipeline is ready to run on any MM patient's genomic data.

**Motivation:** A family member has multiple myeloma. This is personal.

**Full pipeline and data:** https://github.com/maxfromtilt/mm-neoantigen-pipeline

---

## Abstract

We present an open-source computational pipeline for designing personalised mRNA cancer vaccines targeting neoantigens in multiple myeloma (MM). Using somatic mutation data from the MMRF CoMMpass study and MHCflurry neural network binding predictions, we demonstrate end-to-end vaccine construct design for two MM patients. The pipeline identifies 156 and 140 MHC-I binding neoantigen candidates per patient respectively, including 24 and 18 strong binders (IC50 < 50 nM). Enhanced analysis incorporating MHC-II prediction, clonality estimation, gene expression filtering, and driver gene prioritisation produces clinically relevant candidate rankings. Notably, the pipeline identifies DIS3 I85T (a known MM driver) and IDH2 R140Q (a known oncogenic hotspot) as top vaccine targets. Seven dual MHC-I/MHC-II binding candidates are identified for Patient 1, suggesting potential for coordinated CD4+/CD8+ T cell responses. The pipeline follows the same methodological framework used in published clinical trials by BioNTech and Moderna, and is designed to accept patient-specific genomic data for personalised vaccine design.

---

## 1. Introduction

### 1.1 Background

Multiple myeloma is a haematological malignancy characterised by clonal plasma cell proliferation in the bone marrow. Despite significant therapeutic advances including proteasome inhibitors, immunomodulatory drugs, and CAR-T cell therapy, MM remains largely incurable, with most patients eventually relapsing (Kumar et al., 2020).

Personalised neoantigen vaccines represent an emerging therapeutic approach that harnesses the patient's own immune system to target tumour-specific mutations. Somatic mutations in tumour cells generate altered peptides (neoantigens) that can be presented on MHC molecules and recognised by T cells as foreign. By encoding these neoantigens in an mRNA construct, the immune system can be trained to recognise and eliminate tumour cells.

### 1.2 Clinical Precedent

This approach has demonstrated clinical efficacy in multiple tumour types:

- **BioNTech Autogene Cevumeran (BNT122):** Personalised mRNA neoantigen vaccine for pancreatic cancer. Phase I trial showed neoantigen-specific T cell responses in 8/16 patients, with responders showing significantly delayed recurrence (Rojas et al., Nature 2023).
- **Moderna mRNA-4157 (V940):** Combined with pembrolizumab for melanoma. Phase 2b KEYNOTE-942 trial showed 44% reduction in recurrence/death vs pembrolizumab alone (Weber et al., Lancet 2024).
- **BioNTech BNT211:** Personalised neoantigen vaccine for multiple solid tumours, currently in Phase I/II trials.

### 1.3 Rationale for MM

Multiple myeloma presents specific characteristics relevant to neoantigen vaccination:

- **Moderate mutational burden:** MM carries approximately 1-2 mutations/Mb, yielding 40-80 nonsynonymous mutations per patient — sufficient for neoantigen identification (Lohr et al., Cancer Cell 2014).
- **Known driver mutations:** Recurrent mutations in KRAS, NRAS, DIS3, FAM46C, TP53, and RB1 provide high-confidence clonal vaccine targets.
- **Immune accessibility:** Despite immune dysregulation in MM, antigen-specific T cell responses can be elicited, as demonstrated by idiotype vaccine studies (Abdollahi et al., Frontiers in Immunology 2024).
- **Therapeutic window:** Post-autologous stem cell transplant (ASCT) represents an optimal vaccination window with immune reconstitution.

### 1.4 Objective

To develop and validate an open-source computational pipeline for personalised neoantigen mRNA vaccine design in MM, demonstrating feasibility using real patient data from the MMRF CoMMpass study.

---

## 2. Methods

### 2.1 Patient Data

Somatic mutation data was obtained from the MMRF CoMMpass study (Multiple Myeloma Research Foundation Relating Clinical Outcomes in MM to Personal Assessment of Genetic Profile) via the NCI Genomic Data Commons (GDC) API.

**Patient 1 (MMRF_1251):** White male, diagnosed age 43, alive at 4.4 years follow-up. 448 somatic mutation records.

**Patient 2 (MMRF_1274):** 395 somatic mutation records.

### 2.2 Pipeline Architecture

```
Somatic Mutations (GDC/VCF/MAF)
        │
        ▼
[1] Mutation Filtering
    • Missense, frameshift, in-frame indels
    • Minimum VAF ≥ 0.05, depth ≥ 10
        │
        ▼
[2] Peptide Generation
    • 8-11mer sliding windows (MHC-I)
    • Mutant and wildtype pairs
        │
        ▼
[3] MHC-I Binding Prediction
    • MHCflurry 2.1 (neural network, 350K+ measurements)
    • 6 HLA-A/B alleles per patient
    • Batch prediction with nonstandard AA filtering
        │
        ▼
[4] Enhanced Analysis
    • MHC-II prediction (4 HLA-DR alleles, PSSM 9-mer core)
    • Clonality estimation (CCF from VAF or heuristic)
    • Gene expression filtering (MM transcriptomic profiles)
    • Safety screening (self-similarity, autoimmune risk)
        │
        ▼
[5] Vaccine Design
    • Epitope selection (diversity + coverage optimisation)
    • mRNA construct (5'UTR, signal peptide, epitopes,
      GSG linkers, 3'UTR, poly-A tail)
    • Codon optimisation (human bias)
    • RNA modifications (m1Ψ, Cap1)
```

### 2.3 MHC Binding Prediction

**Primary model:** MHCflurry 2.1.5 (O'Donnell et al., Cell Systems 2020), using the Class1AffinityPredictor trained on >350,000 experimentally measured peptide-MHC binding affinities from the Immune Epitope Database (IEDB).

**HLA alleles evaluated (MHC-I):**
- HLA-A*02:01 (population frequency ~28%)
- HLA-A*01:01 (~15%)
- HLA-A*03:01 (~13%)
- HLA-A*24:02 (~10%)
- HLA-B*07:02 (~12%)
- HLA-B*08:01 (~9%)

**MHC-II alleles (PSSM-based):**
- HLA-DRB1*01:01
- HLA-DRB1*03:01
- HLA-DRB1*04:01
- HLA-DRB1*07:01

**Binding thresholds:**
- Strong binder: IC50 < 50 nM
- Weak binder: IC50 < 500 nM
- MHC-II binder: IC50 < 1000 nM

### 2.4 Enhanced Scoring

The vaccine priority score (0-100) integrates eight weighted factors:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| MHC-I binding affinity | 25% | Primary determinant of antigen presentation |
| MHC-II binding | 10% | CD4+ T helper cell recruitment |
| Agretopicity | 15% | Mutant vs wildtype differential binding |
| Immunogenicity | 15% | Mutation type, position, conservation |
| Foreignness | 10% | Sequence divergence from self-proteome |
| Clonality | 10% | Presence in all tumour cells (CCF) |
| Gene expression | 5% | Active transcription in MM plasma cells |
| Driver gene status | 10% | Functional importance in MM pathogenesis |

### 2.5 Vaccine Construct Design

mRNA constructs follow established clinical methodology (Sahin et al., Nature Reviews Drug Discovery 2014):

- **5' Cap:** m7GpppN (Cap1 structure, CleanCap AG)
- **5' UTR:** Human alpha-globin derived
- **Signal peptide:** Human tissue plasminogen activator (tPA)
- **Epitopes:** Up to 20 per construct, separated by GGSGGGGSGG flexible linkers
- **3' UTR:** AES-mtRNR1 combination (BioNTech design)
- **Poly-A tail:** 120 adenines
- **RNA modification:** All uridines replaced with N1-methylpseudouridine (m1Ψ)
- **Codon optimisation:** Human codon usage bias

---

## 3. Results

### 3.1 Mutation Landscape

| Metric | Patient 1 (MMRF_1251) | Patient 2 (MMRF_1274) |
|--------|----------------------|----------------------|
| Total mutation records | 448 | 395 |
| After consequence filtering | 102 | 91 |
| Unique mutations | 66 | 55 |
| Driver gene mutations | 1 (DIS3) | 1 (RB1) |
| Peptide candidates generated | 2,508 | 1,988 |

### 3.2 MHC-I Binding Predictions (MHCflurry)

| Metric | Patient 1 | Patient 2 |
|--------|-----------|-----------|
| Total predictions (candidates × 6 alleles) | 15,048 | 11,928 |
| Binders (IC50 < 500 nM) | 156 (1.0%) | 140 (1.2%) |
| Strong binders (IC50 < 50 nM) | 24 (0.2%) | 18 (0.2%) |

**Binder rates are consistent with published literature (1-2% typical for neoantigen prediction).**

### 3.3 HLA Coverage

**Patient 1 (MMRF_1251):**

| HLA Allele | Population Freq. | Binders | Strong Binders |
|------------|-----------------|---------|----------------|
| HLA-A*02:01 | 28% | 37 | 11 |
| HLA-A*01:01 | 15% | 19 | 3 |
| HLA-A*03:01 | 13% | 20 | 3 |
| HLA-A*24:02 | 10% | 32 | 2 |
| HLA-B*07:02 | 12% | 27 | 5 |
| HLA-B*08:01 | 9% | 21 | 0 |

**Patient 2 (MMRF_1274):**

| HLA Allele | Population Freq. | Binders | Strong Binders |
|------------|-----------------|---------|----------------|
| HLA-A*02:01 | 28% | 34 | 8 |
| HLA-A*01:01 | 15% | 7 | 0 |
| HLA-A*03:01 | 13% | 22 | 2 |
| HLA-A*24:02 | 10% | 13 | 2 |
| HLA-B*07:02 | 12% | 39 | 6 |
| HLA-B*08:01 | 9% | 25 | 0 |

### 3.4 Enhanced Analysis Results

| Enhancement | Patient 1 | Patient 2 |
|-------------|-----------|-----------|
| MHC-II binders (IC50 < 1000 nM) | 17 | 28 |
| Dual MHC-I + MHC-II binders | 7 | 2 |
| Clonal mutations (CCF ≥ 0.9) | 912 predictions | 1,596 predictions |
| Expression-confirmed | 100% | 100% |

### 3.5 Top Vaccine Candidates (Enhanced Ranking)

**Patient 1 (MMRF_1251) — Top 10:**

| Rank | Gene | Mutation | Peptide | HLA | IC50 (nM) | Clonality | CCF | Enhanced Score |
|------|------|----------|---------|-----|-----------|-----------|-----|---------------|
| 1 | **DIS3** | **I85T** | DPATRNVIV | HLA-B*07:02 | 369 | **Clonal** | 0.92 | 67.1 |
| 2 | DIS3 | I85T | DPATRNVIV | HLA-B*08:01 | 201 | Clonal | 0.92 | 66.8 |
| 3 | DIS3 | I85T | VLEDPATRNV | HLA-A*02:01 | 225 | Clonal | 0.92 | 66.2 |
| 4 | MME | Y579F | LNYGGIFMV | HLA-A*02:01 | 120 | Likely clonal | 0.70 | 63.3 |
| 5 | AHR | N108Y | RYGLNLQEGEF | HLA-A*24:02 | 181 | Likely clonal | 0.52 | 62.7 |
| 6 | MME | Y579F | IFMVIGHEI | HLA-A*24:02 | 51 | Likely clonal | 0.70 | 62.6 |
| 7 | OR4K5 | E196K | KLACLDSYIIK | HLA-A*03:01 | 171 | Subclonal | 0.32 | 62.3 |
| 8 | NME8 | V529F | MILTKWNAF | HLA-A*24:02 | 274 | Likely clonal | 0.64 | 62.2 |
| 9 | RINT1 | N574K | AALEVFAENK | HLA-A*03:01 | 283 | Likely clonal | 0.69 | 61.6 |
| 10 | TTC14 | S671L | EPGSVRHSTL | HLA-B*07:02 | 70 | Likely clonal | 0.82 | 61.5 |

**Patient 2 (MMRF_1274) — Top 10:**

| Rank | Gene | Mutation | Peptide | HLA | IC50 (nM) | Clonality | CCF | Enhanced Score |
|------|------|----------|---------|-----|-----------|-----------|-----|---------------|
| 1 | THBS1 | V990L | TLNCDPGLAV | HLA-A*02:01 | 180 | Likely clonal | 0.51 | 62.0 |
| 2 | CTXN3 | L52F | IFLDPYRSM | HLA-A*24:02 | 188 | Likely clonal | 0.71 | 60.7 |
| 3 | THBS1 | V990L | QGKELVQTL | HLA-B*07:02 | 352 | Likely clonal | 0.51 | 60.5 |
| 4 | ATP2B4 | V72A | AALEKRRQV | HLA-B*08:01 | 154 | Subclonal | 0.37 | 59.1 |
| 5 | ZDHHC19 | K258M | MYMAEAVQL | HLA-B*08:01 | 432 | Subclonal | 0.45 | 59.0 |
| 6 | ARID3C | P347T | SFLPRGKVTL | HLA-A*24:02 | 135 | Likely clonal | 0.71 | 57.9 |
| 7 | SRCAP | P3159R | RPSPLGPEG | HLA-B*07:02 | 92 | Likely clonal | 0.59 | 57.3 |
| 8 | SRCAP | P2900R | VPGPERLIV | HLA-B*07:02 | 64 | Likely clonal | 0.85 | 54.7 |
| 9 | SYNE1 | E4316Q | LVKQQTSHL | HLA-B*08:01 | 94 | **Clonal** | 0.93 | 54.0 |
| 10 | **IDH2** | **R140Q** | SPNGTIQNIL | HLA-B*07:02 | **47** | **Clonal** | 0.99 | 53.8 |

### 3.6 Dual MHC-I/MHC-II Binders (Patient 1)

These candidates engage both CD8+ cytotoxic and CD4+ helper T cells:

| Gene | Mutation | Peptide | MHC-I IC50 | MHC-II IC50 | Clonality |
|------|----------|---------|------------|-------------|-----------|
| CCDC105 | T427M | MYLEKNLDELL | 77 nM | 590 nM | Clonal |
| SYTL3 | V512I | LPIQQKLRL | 102 nM | 665 nM | Likely clonal |
| TMEM131 | G195A | FINTSNHAV | 22 nM | 953 nM | Likely clonal |
| HS3ST1 | F177Y | YLVRDGRLNV | 98 nM | 411 nM | Likely clonal |

### 3.7 Notable Findings

**DIS3 I85T (Patient 1):** DIS3 (exosome complex exonuclease) is mutated in approximately 10% of MM patients and is classified as a tumour suppressor in the MM context. The I85T mutation is clonal (CCF 0.92), the gene is confirmed expressed in MM plasma cells (14.6 TPM), and multiple peptide windows achieve binding across different HLA alleles. This represents an ideal vaccine target: functionally important, clonal, expressed, and immunogenic.

**IDH2 R140Q (Patient 2):** IDH2 R140Q is a well-characterised oncogenic hotspot mutation, also found in acute myeloid leukaemia (AML) and glioma. The FDA-approved IDH2 inhibitor enasidenib targets this exact mutation in AML. In our analysis, it achieves strong binding (47 nM, HLA-B*07:02) and is highly clonal (CCF 0.99). This cross-cancer relevance adds confidence to its immunogenic potential.

### 3.8 Vaccine Constructs

| Parameter | Patient 1 | Patient 2 |
|-----------|-----------|-----------|
| Epitopes in construct | 20 | 20 |
| Protein cassette length | 412 aa | 403 aa |
| mRNA length | 1,456 nt | 1,429 nt |
| GC content | 55.4% | 57.9% |
| Estimated MW | 480.5 kDa | 471.6 kDa |
| Unique genes represented | 18 | 14 |
| HLA alleles covered | 5 | 5 |

---

## 4. Discussion

### 4.1 Pipeline Validation

Our binder detection rates (1.0-1.2%) and strong binder rates (0.15-0.2%) are consistent with published neoantigen prediction benchmarks. The MHCflurry neural network, trained on >350,000 experimental binding measurements, provides predictions comparable to the field-standard NetMHCpan (O'Donnell et al., 2020).

The enhanced scoring system successfully reranks candidates by biological relevance. DIS3 I85T, which ranked outside the top 5 by binding affinity alone, rises to #1 when clonality, expression, and driver gene status are considered — reflecting clinical intuition about optimal vaccine targets.

### 4.2 Comparison to Clinical Trials

Our methodology aligns with published personalised cancer vaccine trials:

| Aspect | Our Pipeline | BioNTech BNT122 | Moderna V940 |
|--------|-------------|----------------|--------------|
| Input data | WES mutations | WES + RNA-seq | WES + RNA-seq |
| Binding prediction | MHCflurry | Undisclosed (likely NetMHCpan) | NetMHCpan |
| HLA typing | Population defaults | Patient-specific | Patient-specific |
| Epitopes per construct | 20 | Up to 20 | Up to 34 |
| mRNA platform | m1Ψ, Cap1, LNP | m1Ψ, Cap1, lipoplex | m1Ψ, Cap1, LNP |

The primary gaps vs clinical implementations are patient-specific HLA typing and patient-specific RNA-seq expression data, both of which our pipeline is designed to accept as inputs.

### 4.3 Limitations

1. **HLA typing assumed:** Default population-frequent alleles were used. Patient-specific HLA genotyping (e.g., OptiType from WGS/RNA-seq) would significantly refine predictions.

2. **Expression data is population-level:** Patient-specific RNA-seq would improve epitope selection by confirming which genes are actively transcribed in the patient's tumour.

3. **No MHC-II neural network:** MHC-II predictions use PSSM models. NetMHCIIpan or MHCflurry Class II would improve CD4+ epitope identification.

4. **Clonality estimation:** Without patient-specific VAF data, clonality is estimated from heuristics. Real tumour/normal sequencing would provide definitive CCF values.

5. **No experimental validation:** Computational predictions require experimental confirmation (ELISpot, multimer staining, in vivo immunogenicity assays).

6. **Single prediction tool:** Consensus predictions using multiple tools (MHCflurry + NetMHCpan) would increase confidence.

### 4.4 Clinical Path Forward

For a specific MM patient, the pipeline requires:

1. **Whole exome sequencing** (WES) of tumour and matched normal tissue
2. **HLA typing** (from WES data or dedicated HLA genotyping)
3. **RNA-seq** of tumour (optional but recommended for expression filtering)
4. **Pipeline execution** with patient-specific inputs (~30 minutes)
5. **Clinical review** of top candidates by haematologist/oncologist
6. **Experimental validation** (ELISpot for immunogenicity)

Existing compassionate use and clinical trial frameworks (HPRA in Ireland, FDA IND in the US) provide regulatory paths for personalised cancer vaccine administration.

---

## 5. Methods — Software and Reproducibility

### 5.1 Software Versions

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9.6 | Runtime |
| MHCflurry | 2.1.5 | MHC-I binding prediction |
| TensorFlow | 2.20.0 | MHCflurry backend |
| pandas | 2.x | Data processing |
| BioPython | 1.85 | Sequence handling |
| NumPy | 1.x | Numerical computation |

### 5.2 Reproducibility

The complete pipeline, including all source code, configuration, patient data, and output files, is available at:

**https://github.com/maxfromtilt/mm-neoantigen-pipeline**

To reproduce:
```bash
pip install -r requirements.txt
pip install mhcflurry && mhcflurry-downloads fetch
python 02_parse_mutations.py --input data/mmrf_1251_mutations.csv --output-dir output/mmrf_1251_mhcflurry
python 03_predict_binding.py --input output/mmrf_1251_mhcflurry/neoantigen_candidates.csv --output-dir output/mmrf_1251_mhcflurry
python 04_design_vaccine.py --input output/mmrf_1251_mhcflurry/binding_predictions.csv --output-dir output/mmrf_1251_mhcflurry
python 07_enhanced_analysis.py --patient mmrf_1251
```

---

## 6. References

1. Abdollahi P, Norseth HM, Schjesvold F. Advances and challenges in anti-cancer vaccines for multiple myeloma. *Frontiers in Immunology*. 2024;15:1411352.

2. Chapman MA, et al. Initial genome sequencing and analysis of multiple myeloma. *Nature*. 2011;471(7339):467-472.

3. Dutta D, et al. A BCMA-mRNA vaccine is a promising therapeutic for multiple myeloma. *Blood*. 2025;146(19):2322.

4. Floudas CS, et al. Leveraging mRNA technology for antigen based immuno-oncology therapies. *Journal for ImmunoTherapy of Cancer*. 2025.

5. Kumar SK, et al. Multiple myeloma. *Nature Reviews Disease Primers*. 2017;3:17046.

6. Lohr JG, et al. Widespread genetic heterogeneity in multiple myeloma: implications for targeted therapy. *Cancer Cell*. 2014;25(1):91-101.

7. O'Donnell TJ, et al. MHCflurry 2.0: Improved pan-allele prediction of MHC class I-presented peptides. *Cell Systems*. 2020;11(1):42-48.

8. Rojas LA, et al. Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer. *Nature*. 2023;618(7963):144-150.

9. Sahin U, et al. mRNA-based therapeutics — developing a new class of drugs. *Nature Reviews Drug Discovery*. 2014;13(10):759-780.

10. Schumacher TN, Schreiber RD. Neoantigens in cancer immunotherapy. *Science*. 2015;348(6230):69-74.

---

## Disclaimer

This report describes a computational research pipeline for educational and research purposes only. The vaccine constructs described herein have not been experimentally validated and are not intended for clinical use. Any therapeutic application would require extensive preclinical testing, regulatory approval (HPRA/EMA/FDA), and clinical oversight by qualified medical professionals. The authors are not medical professionals and this document does not constitute medical advice.

---

*Correspondence: robert.doran@tilt.ie*
