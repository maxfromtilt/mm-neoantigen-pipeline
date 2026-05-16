#!/usr/bin/env python3
"""
16_generate_ind_package.py — IND Documentation Package Generator
===============================================================
Generates a complete IND-enabling regulatory documentation package
for a specific patient. Calls 15_ind_documentation.py for templates.

Usage:
    python 16_generate_ind_package.py --patient mmrf_1251
    python 16_generate_ind_package.py --patient mmrf_1251 --output /path/to/output

Output:
    output/ind_package/{patient_id}/
        pipeline_description.txt
        validation_summary.txt
        limitations_statement.txt
        dataset_manifest.json
        docker_environment.txt
        bioinformatics_methods.txt
        clinical_metadata.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Import shared templates and constants from 15_ind_documentation
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
from ind_documentation import (
    PIPELINE_VERSION,
    SOFTWARE_INVENTORY,
    DATABASE_SOURCES,
    PIPELINE_METHODOLOGY,
    VALIDATION_METHODS,
    LIMITATIONS_STATEMENT,
    DOCKERFILE_CONTENT,
    REQUIREMENTS_PIPELINE_CONTENT,
)


def build_software_table() -> str:
    lines = []
    lines.append(f"{'Package':<20} {'Version':<15} {'License':<15} {'Install'}")
    lines.append("-" * 75)
    for name, info in sorted(SOFTWARE_INVENTORY.items()):
        lines.append(f"{name:<20} {info['version']:<15} {info.get('license', 'N/A'):<15} pip install {info.get('pip', name)}")
    return "\n".join(lines)


def build_data_table() -> str:
    lines = []
    lines.append(f"{'Source':<20} {'Name':<30} {'Access':<15} {'Data Type'}")
    lines.append("-" * 95)
    for key, info in sorted(DATABASE_SOURCES.items()):
        lines.append(
            f"{key:<20} {info['name']:<30} "
            f"{info.get('access_model', 'Public'):<15} {info.get('data_type', 'N/A')}"
        )
    return "\n".join(lines)


def generate_dataset_manifest(patient_id: str) -> dict:
    """Build a JSON manifest of all datasets used for this patient."""
    manifest = {
        "manifest_id": f"IND-{patient_id.upper()}-{datetime.now().strftime('%Y%m%d')}",
        "pipeline_version": PIPELINE_VERSION,
        "generated_date": datetime.now().isoformat(),
        "patient_id": patient_id,
        "datasets": [],
    }

    # MMRF patient mutation data
    mutation_path = f"data/{patient_id}_mutations.csv"
    if os.path.exists(mutation_path):
        import pandas as pd
        df = pd.read_csv(mutation_path)
        manifest["datasets"].append({
            "dataset_id": f"MMRF-{patient_id.upper()}",
            "source": "MMRF_CoMMpass",
            "description": f"Somatic mutations for patient {patient_id}",
            "file": mutation_path,
            "n_mutations": len(df),
            "consequence_types": list(df.get("consequence_type", df.get("variant_class", "unknown")).unique())
            if "consequence_type" in df.columns or "variant_class" in df.columns else ["missense"],
            "access_date": "2024-2026",
            "version": "IA13",
        })

    # Pipeline output datasets
    output_dirs = [
        f"output/{patient_id}_mhcflurry",
        f"output/{patient_id}_enhanced",
        f"output/{patient_id}_wgs",
        f"output/{patient_id}_proteomics",
        f"output/{patient_id}_structures",
    ]
    for odir in output_dirs:
        if os.path.exists(odir) and os.path.isdir(odir):
            files = os.listdir(odir)
            for f in files:
                fpath = os.path.join(odir, f)
                if os.path.isfile(fpath):
                    manifest["datasets"].append({
                        "dataset_id": f"OUTPUT-{odir}",
                        "source": "MMRF_Neoantigen_Pipeline",
                        "description": f"Pipeline output: {f}",
                        "file": fpath,
                        "access_date": datetime.now().strftime("%Y-%m-%d"),
                        "version": PIPELINE_VERSION,
                    })

    # External databases
    for key, info in DATABASE_SOURCES.items():
        manifest["datasets"].append({
            "dataset_id": key,
            "source": key,
            "description": info["name"],
            "url": info["url"],
            "data_type": info.get("data_type", "unknown"),
            "access_date": info.get("access_date", "2024-2026"),
            "access_model": info.get("access_model", "Public"),
        })

    return manifest


def generate_clinical_metadata(patient_id: str) -> dict:
    """Generate clinical metadata for IND submission."""
    import pandas as pd

    meta = {
        "patient_id": patient_id,
        "pipeline_version": PIPELINE_VERSION,
        "generated_date": datetime.now().isoformat(),
        "study_identifier": "MMRF-CoMMpass-IA13",
        "indication": "Multiple Myeloma",
        "vaccine_type": "Personalised mRNA neoantigen vaccine",
        "antigen_class": "Mutation-derived peptide epitopes (neoantigens)",
        "delivery_platform": "mRNA (N1-methylpseudouridine modified)",
        "route_of_administration": "Intramuscular or intradermal (TBD with CMO)",
        "dosing_regimen": "TBD (preclinical/Phase I)",
    }

    # Try to extract from MMRF data
    cases_path = "data/mmrf_cases.csv"
    if os.path.exists(cases_path):
        try:
            cases = pd.read_csv(cases_path)
            match = cases[cases["submitter_id"].str.upper() == patient_id.upper().replace("_", "_").replace("MMRF", "MMRF_")]
            if match.empty:
                pid_short = patient_id.replace("mmrf_", "").upper()
                match = cases[cases["submitter_id"].str.upper().str.contains(pid_short, na=False)]
            if not match.empty:
                row = match.iloc[0]
                meta["age_at_enrollment"] = row.get("age_at_enrollment")
                meta["sex"] = row.get("sex")
                meta["race"] = row.get("race")
                meta["disease_stage"] = row.get("disease_stage", row.get("iss_stage"))
                meta["prior_lines_of_therapy"] = row.get("prior_lines_of_therapy", 0)
                meta["treatment_arm"] = row.get("treatment_arm", row.get("arm", "Standard"))
        except Exception:
            pass

    # Load vaccine construct info if available
    vaccine_json_path = f"output/{patient_id}_mhcflurry/vaccine_construct.json"
    if os.path.exists(vaccine_json_path):
        try:
            with open(vaccine_json_path) as f:
                vc = json.load(f)
                meta["vaccine_construct"] = {
                    "n_epitopes": vc.get("n_epitopes", "unknown"),
                    "cassette_length_aa": vc.get("cassette_length_aa", "unknown"),
                    "mrna_length_nt": vc.get("mrna_length_nt", "unknown"),
                    "gc_content": vc.get("gc_content", "unknown"),
                    "codon_optimisation_level": vc.get("codon_optimisation_level", "standard"),
                }
        except Exception:
            pass

    return meta


def build_bioinformatics_methods() -> str:
    """Build detailed bioinformatics methods text for IND."""
    return """
BIOINFORMATICS METHODS
Pipeline Version: {version}
Generated: {date}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MUTATION CALLING AND FILTERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 Primary Data
  - Somatic variant calls from MMRF CoMMpass IA13 release
  - Variant calling: Strelka + MuTect2 (GDC pipelines)
  - Reference genome: GRCh38
  - Annotation: Variant Effect Predictor (VEP)

1.2 Consequence Filtering
  Retained consequence types:
    - missense_variant / Missense_Mutation
    - frameshift_variant / Frame_Shift_Del / Frame_Shift_Ins
    - inframe_deletion / inframe_insertion / In_Frame_Del / In_Frame_Ins
    - nonstop_variant / Nonstop_Mutation (treated as frameshift)

  Excluded:
    - Synonymous variants (no protein change)
    - Splice site / intron variants (no MHC binding)
    - 5'/3' UTR, intergenic, upstream/downstream (no coding epitope)

1.3 Amino Acid Change Parsing
  - Input format: p.E196K or E196K (one-letter amino acid codes)
  - Single amino acid substitutions only
  - Stop codon (*) excluded from epitope generation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PEPTIDE EPITOPE GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 Window-Based Epitope Extraction
  - Context window: 17 amino acids centred on the mutant residue
  - Flanking sequence: UniProt protein sequence when available
    (fetched by 02_parse_mutations.py via UniProt REST API)
  - When UniProt sequence unavailable: approximate flanking using
    position-seeded random amino acid selection

2.2 Epitope Length
  - Generated lengths: 8, 9, 10, 11 amino acids
  - Rationale: HLA-I binds 8-11mer peptides
  - Mutant position anchored at central residue (position 5 for 9-mer)

2.3 Wildtype/Mutant Pairing
  - Each mutation generates a paired WT/MUT peptide pair
  - Required for agretopicity index calculation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. MHC BINDING PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 PSSM Model (Built-in, Default)
  - Position-specific scoring matrices for 6 HLA-I alleles
  - Anchor positions defined per allele based on IEDB MHC-eluted peptide data
  - IC50 prediction: exponential decay function of PSSM score
    IC50 = 50000 * exp(-1.5 * score)
  - Score = sum(anchor_scores[i][aa] for i in peptide + general_scores[aa])
  - Classification thresholds:
      Strong binder: IC50 < 50 nM
      Weak binder: 50 <= IC50 < 500 nM
      Non-binder: IC50 >= 500 nM

3.2 Neural Network Method (Optional, Recommended for IND)
  - MHCflurry 2.1 (OpenVax) via: pip install mhcflurry
  - Trained on 350,000+ MHC-I binding measurements from IEDB
  - Requires: python 3.8+, tensorflow (or tensorflow-lite)
  - Run: mhcflurry-predict --help

3.3 MHC-II Predictions
  - Secondary predictions for CD4+ helper T-cell epitopes
  - MHC-II binding predicted for 4 HLA-DR alleles using
    comparable PSSM matrices
  - Used for dual binder identification (MHC-I + MHC-II)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. EPITOPE RANKING AND SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4.1 Multi-Criteria Vaccine Priority Score

  vaccine_priority_score = (
      binding_norm * 0.30 +       # IC50 normalised to 0-1
      agreto_norm * 0.25 +         # Agretopicity normalised
      immuno_norm * 0.20 +         # Immunogenicity score normalised
      foreign_norm * 0.15 +        # Foreignness vs WT normalised
      driver_norm * 0.10           # Driver gene flag
  )

  Where:
    binding_norm = max(0, 1 - min(ic50, 50000) / 50000)
    agreto_norm  = min(agretopicity / 10.0, 1.0)
    immuno_norm  = immunogenicity_score / 100.0
    foreign_norm = foreignness_score / 10.0
    driver_norm  = 1.0 if is_driver_gene else 0.3

4.2 Optional Score Modifiers
    TCR_boost:          +15 (known reactive TCR in VDJdb/McPAS)
    Cross_react_boost:  +10 (cross-reactive analogue detected)
    Proteomics_boost:   +25 (MS-confirmed in PRIDE/PeptideAtlas)
    Ligandomics_boost:  +20 (HLA ligand mass-spec validation)

4.3 Final Selection
  - Top N epitopes (default: 20) by vaccine_priority_score
  - Diversity filter: max 3 epitopes per gene
  - HLA balance: at least 2 HLA alleles represented
  - Binding threshold: IC50 < 500 nM mandatory

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. mRNA VACCINE CONSTRUCT DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5.1 mRNA Architecture
  5' cap (Cap1 m7GpppN) → 5' UTR → Signal peptide →
  [Epitope1]-[Linker]-[Epitope2]-...[EpitopeN] →
  STOP → 3' UTR → Poly-A tail (120 nt)

5.2 Signal Peptide
  - Sequence: MDAMKRGLCCVLLLCGAVFVSPSQEIHARFR (MHC-I trafficking)
  - Included to direct translated protein to MHC-I presentation pathway

5.3 Linker Sequence
  - GGSGGGGSGG (10 aa)
  - Flexible Gly-Ser linker for independent epitope processing

5.4 Nucleoside Modifications
  - All uridine → N1-methylpseudouridine (m1Ψ)
  - Rationale: reduces innate immune activation, increases mRNA half-life

5.5 Codon Optimisation
  - Human codon usage bias (CUB) maximisation
  - GC content: 40-70% target, ideal 58%
  - Homopolymer runs: max 7 consecutive identical nucleotides
  - Polyadenylation signal (AAUAAA): avoided in coding sequence

5.6 5' and 3' UTRs
  - 5' UTR: Kozak context (GCCACCATG) for efficient translation initiation
  - 3' UTR: Human β-globin 3' UTR for stability

5.7 Poly-A Tail
  - 120 adenine residues
  - Mimics natural mRNA structure for ribosome engagement

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. QUALITY CONTROL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - GC content bounds: 0.40 – 0.70
  - No internal stop codons in reading frame
  - No known toxic peptide motifs (precomputed blacklist)
  - mRNA secondary structure MFE: estimated via RNAfold (ViennaRNA)
    or pure-Python nearest-neighbor fallback
  - Allergenicity: BLAST against allergen database (optional)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. STATISTICAL METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7.1 Binding Affinity
  - Pearson correlation vs experimental IC50 for PSSM model validation
  - RMSE, MAE reported for held-out IEDB benchmark set

7.2 Survival Analysis
  - Kaplan-Meier estimator for overall survival
  - Log-rank test for gene-level survival comparison
  - Cox proportional hazards for multivariate analysis
  - All survival data from MMRF CoMMpass clinical follow-up

7.3 Cohort Comparison
  - Fisher's exact test for mutation frequency vs published TCGA/MMRF
  - Bonferroni-corrected for multiple testing

""".strip().format(version=PIPELINE_VERSION, date=datetime.now().strftime("%Y-%m-%d"))


def generate_ind_package(patient_id: str, output_dir: str = None) -> str:
    """
    Generate the complete IND documentation package for a patient.
    Returns path to output directory.
    """
    if output_dir is None:
        output_dir = f"output/ind_package/{patient_id}"

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"[*] Generating IND package for {patient_id}...")
    print(f"[*] Output: {output_dir}/")

    # 1. Pipeline description
    pipeline_text = PIPELINE_METHODOLOGY.format(
        version=PIPELINE_VERSION,
        date=date_str,
        software_table=build_software_table(),
        data_table=build_data_table(),
    )
    with open(os.path.join(output_dir, "pipeline_description.txt"), "w") as f:
        f.write(pipeline_text)
    print(f"  [+] pipeline_description.txt")

    # 2. Validation summary
    validation_text = VALIDATION_METHODS.format(
        version=PIPELINE_VERSION,
        date=date_str,
    )
    with open(os.path.join(output_dir, "validation_summary.txt"), "w") as f:
        f.write(validation_text)
    print(f"  [+] validation_summary.txt")

    # 3. Limitations statement
    limitations_text = LIMITATIONS_STATEMENT.format(
        version=PIPELINE_VERSION,
        date=date_str,
    )
    with open(os.path.join(output_dir, "limitations_statement.txt"), "w") as f:
        f.write(limitations_text)
    print(f"  [+] limitations_statement.txt")

    # 4. Dataset manifest JSON
    manifest = generate_dataset_manifest(patient_id)
    with open(os.path.join(output_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [+] dataset_manifest.json")

    # 5. Docker environment
    with open(os.path.join(output_dir, "docker_environment.txt"), "w") as f:
        f.write("DOCKER ENVIRONMENT FOR MMRF NEOANTIGEN PIPELINE\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Pipeline Version: {PIPELINE_VERSION}\n")
        f.write(f"Generated: {date_str}\n\n")
        f.write("DOCKERFILE:\n")
        f.write("-" * 40 + "\n")
        f.write(DOCKERFILE_CONTENT)
        f.write("\n\nrequirements_pipeline.txt:\n")
        f.write("-" * 40 + "\n")
        f.write(REQUIREMENTS_PIPELINE_CONTENT)
    print(f"  [+] docker_environment.txt")

    # 6. Bioinformatics methods
    bio_methods = build_bioinformatics_methods()
    with open(os.path.join(output_dir, "bioinformatics_methods.txt"), "w") as f:
        f.write(bio_methods)
    print(f"  [+] bioinformatics_methods.txt")

    # 7. Clinical metadata
    clinical_meta = generate_clinical_metadata(patient_id)
    with open(os.path.join(output_dir, "clinical_metadata.json"), "w") as f:
        json.dump(clinical_meta, f, indent=2)
    print(f"  [+] clinical_metadata.json")

    # 8. Generate a README for the package
    readme = f"""IND DOCUMENTATION PACKAGE
Patient: {patient_id}
Pipeline Version: {PIPELINE_VERSION}
Generated: {date_str}
Package ID: IND-{patient_id.upper()}-{datetime.now().strftime('%Y%m%d')}

Contents:
  1. pipeline_description.txt   — Full computational methodology and software versions
  2. validation_summary.txt     — Internal validation vs IEDB benchmark, cross-DB validation
  3. limitations_statement.txt  — Known limitations, failure modes, required mitigations
  4. dataset_manifest.json      — All datasets used, sources, access dates, versions
  5. docker_environment.txt     — Dockerfile + requirements for reproducible execution
  6. bioinformatics_methods.txt — Detailed methods suitable for regulatory submission
  7. clinical_metadata.json     — Patient clinical context for IND filing

Regulatory Status: RESEARCH USE ONLY

For IND submission, this package must be supplemented with:
  - GMP manufacturing documentation (CMO)
  - Preclinical toxicology data (GLP studies)
  - Investigator's Brochure (IB)
  - Clinical protocol
  - IRB approval documentation
  - Patient informed consent (for sample use)

Contact: {patient_id} pipeline output files in output/{patient_id}_mhcflurry/
"""
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(readme)
    print(f"  [+] README.txt")

    print(f"\n[+] IND package complete: {output_dir}/")
    print(f"[+] Package ID: IND-{patient_id.upper()}-{datetime.now().strftime('%Y%m%d')}")

    return output_dir


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate IND documentation package for a patient"
    )
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. mmrf_1251)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: output/ind_package/{patient_id})",
    )
    args = parser.parse_args()

    out_dir = generate_ind_package(args.patient, args.output)
    print(f"\n[✓] Package ready: {out_dir}")