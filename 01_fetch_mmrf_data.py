#!/usr/bin/env python3
"""
MMRF CoMMpass Data Fetcher
==========================
Fetches multiple myeloma genomic data (mutations, gene expression, clinical)
from the NCI Genomic Data Commons (GDC) API.

The MMRF CoMMpass study contains ~995 newly diagnosed MM patients with:
- Whole Genome Sequencing (WGS)
- Whole Exome Sequencing (WXS)
- RNA Sequencing
- Clinical data (demographics, treatments, outcomes)

Usage:
    python 01_fetch_mmrf_data.py [--config config.yaml] [--max-cases 50]
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm


GDC_API_BASE = "https://api.gdc.cancer.gov"
PROJECT_ID = "MMRF-COMMPASS"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_project_summary() -> dict:
    """Fetch summary statistics for the MMRF-COMMPASS project."""
    url = f"{GDC_API_BASE}/projects/{PROJECT_ID}"
    params = {
        "expand": "summary,summary.data_categories,summary.experimental_strategies"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_cases(max_cases: Optional[int] = None, page_size: int = 100) -> pd.DataFrame:
    """
    Fetch clinical case data for all MMRF-COMMPASS patients.

    Returns DataFrame with demographics, diagnoses, and treatment info.
    """
    url = f"{GDC_API_BASE}/cases"

    filters = {
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": PROJECT_ID
        }
    }

    fields = [
        "case_id",
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "demographic.year_of_birth",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.classification_of_tumor",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_recurrence",
        "diagnoses.tissue_or_organ_of_origin",
        "exposures.alcohol_history",
    ]

    all_cases = []
    offset = 0
    total = max_cases or 9999

    with tqdm(desc="Fetching cases", unit="cases") as pbar:
        while offset < total:
            size = min(page_size, total - offset) if max_cases else page_size
            params = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": str(size),
                "from": str(offset),
            }

            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()["data"]

            hits = data["hits"]
            if not hits:
                break

            all_cases.extend(hits)
            pbar.update(len(hits))

            if not max_cases:
                total = data["pagination"]["total"]

            offset += len(hits)
            time.sleep(0.5)  # Rate limiting

    # Flatten nested JSON into DataFrame
    records = []
    for case in all_cases:
        record = {
            "case_id": case.get("case_id"),
            "submitter_id": case.get("submitter_id"),
        }
        # Demographics
        demo = case.get("demographic", {})
        if isinstance(demo, list):
            demo = demo[0] if demo else {}
        record["gender"] = demo.get("gender")
        record["race"] = demo.get("race")
        record["ethnicity"] = demo.get("ethnicity")
        record["vital_status"] = demo.get("vital_status")
        record["days_to_death"] = demo.get("days_to_death")

        # Diagnoses
        diags = case.get("diagnoses", [])
        if diags:
            diag = diags[0]
            record["age_at_diagnosis"] = diag.get("age_at_diagnosis")
            record["primary_diagnosis"] = diag.get("primary_diagnosis")
            record["tumor_stage"] = diag.get("tumor_stage")
            record["days_to_last_follow_up"] = diag.get("days_to_last_follow_up")
            record["days_to_recurrence"] = diag.get("days_to_recurrence")

        records.append(record)

    return pd.DataFrame(records)


def fetch_mutation_files(max_files: Optional[int] = None, page_size: int = 100) -> list:
    """
    Fetch metadata for somatic mutation files (MAF format) from MMRF-COMMPASS.

    These are open-access Masked Somatic Mutation files processed by the
    GDC's somatic mutation calling pipeline.
    """
    url = f"{GDC_API_BASE}/files"

    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": PROJECT_ID
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_category",
                    "value": "Simple Nucleotide Variation"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Masked Somatic Mutation"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "access",
                    "value": "open"
                }
            }
        ]
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "data_category",
        "data_type",
        "data_format",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "analysis.workflow_type",
    ]

    all_files = []
    offset = 0
    total = max_files or 9999

    with tqdm(desc="Fetching mutation files", unit="files") as pbar:
        while offset < total:
            size = min(page_size, total - offset) if max_files else page_size
            params = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": str(size),
                "from": str(offset),
            }

            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()["data"]

            hits = data["hits"]
            if not hits:
                break

            all_files.extend(hits)
            pbar.update(len(hits))

            if not max_files:
                total = data["pagination"]["total"]

            offset += len(hits)
            time.sleep(0.5)

    return all_files


def fetch_expression_files(max_files: Optional[int] = None, page_size: int = 100) -> list:
    """
    Fetch metadata for RNA-seq gene expression files from MMRF-COMMPASS.
    """
    url = f"{GDC_API_BASE}/files"

    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": PROJECT_ID
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_category",
                    "value": "Transcriptome Profiling"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Gene Expression Quantification"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "access",
                    "value": "open"
                }
            }
        ]
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "data_format",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "analysis.workflow_type",
    ]

    all_files = []
    offset = 0
    total = max_files or 9999

    with tqdm(desc="Fetching expression files", unit="files") as pbar:
        while offset < total:
            size = min(page_size, total - offset) if max_files else page_size
            params = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": str(size),
                "from": str(offset),
            }

            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()["data"]

            hits = data["hits"]
            if not hits:
                break

            all_files.extend(hits)
            pbar.update(len(hits))

            if not max_files:
                total = data["pagination"]["total"]

            offset += len(hits)
            time.sleep(0.5)

    return all_files


def download_file(file_id: str, output_dir: str, filename: str = None) -> str:
    """
    Download a single file from GDC by file UUID.

    Open-access files can be downloaded without authentication.
    Controlled-access files require a dbGaP token.
    """
    url = f"{GDC_API_BASE}/data/{file_id}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    # Determine filename
    if not filename:
        cd = resp.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            filename = cd.split("filename=")[1].strip('"')
        else:
            filename = f"{file_id}.dat"

    filepath = output_dir / filename
    total_size = int(resp.headers.get("content-length", 0))

    with open(filepath, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True,
                  desc=f"Downloading {filename}") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return str(filepath)


def fetch_ssm_occurrences(max_results: int = 10000) -> pd.DataFrame:
    """
    Fetch Simple Somatic Mutation (SSM) occurrence data directly from GDC.

    This is the aggregated mutation data across all MMRF-COMMPASS cases,
    which is open-access and doesn't require downloading individual MAF files.
    """
    url = f"{GDC_API_BASE}/ssm_occurrences"

    filters = {
        "op": "=",
        "content": {
            "field": "case.project.project_id",
            "value": PROJECT_ID
        }
    }

    fields = [
        "ssm.ssm_id",
        "ssm.genomic_dna_change",
        "ssm.chromosome",
        "ssm.start_position",
        "ssm.end_position",
        "ssm.reference_allele",
        "ssm.tumor_allele",
        "ssm.mutation_subtype",
        "ssm.consequence.transcript.gene.symbol",
        "ssm.consequence.transcript.gene.gene_id",
        "ssm.consequence.transcript.consequence_type",
        "ssm.consequence.transcript.aa_change",
        "ssm.consequence.transcript.annotation.hgvsc",
        "case.case_id",
        "case.submitter_id",
    ]

    all_mutations = []
    offset = 0

    with tqdm(desc="Fetching mutations", total=max_results, unit="mutations") as pbar:
        while offset < max_results:
            size = min(500, max_results - offset)
            params = {
                "filters": json.dumps(filters),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": str(size),
                "from": str(offset),
            }

            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()["data"]

            hits = data["hits"]
            if not hits:
                break

            all_mutations.extend(hits)
            pbar.update(len(hits))
            total = data["pagination"]["total"]
            pbar.total = min(max_results, total)

            offset += len(hits)
            time.sleep(0.3)

    # Flatten mutation data
    records = []
    for occ in all_mutations:
        ssm = occ.get("ssm", {})
        case = occ.get("case", {})
        consequences = ssm.get("consequence", [])

        for csq in consequences:
            transcript = csq.get("transcript", {})
            gene = transcript.get("gene", {})
            record = {
                "ssm_id": ssm.get("ssm_id"),
                "case_id": case.get("case_id"),
                "submitter_id": case.get("submitter_id"),
                "chromosome": ssm.get("chromosome"),
                "start_position": ssm.get("start_position"),
                "end_position": ssm.get("end_position"),
                "reference_allele": ssm.get("reference_allele"),
                "tumor_allele": ssm.get("tumor_allele"),
                "genomic_dna_change": ssm.get("genomic_dna_change"),
                "mutation_subtype": ssm.get("mutation_subtype"),
                "gene_symbol": gene.get("symbol"),
                "gene_id": gene.get("gene_id"),
                "consequence_type": transcript.get("consequence_type"),
                "aa_change": transcript.get("aa_change"),
            }
            records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch MMRF CoMMpass data from GDC for neoantigen analysis"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--max-cases", type=int, default=50,
                        help="Max number of cases to fetch")
    parser.add_argument("--max-mutations", type=int, default=10000,
                        help="Max number of SSM occurrences to fetch")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory for downloaded data")
    parser.add_argument("--download-mafs", action="store_true",
                        help="Download individual MAF files (requires more storage)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MMRF CoMMpass Data Fetcher")
    print("Multiple Myeloma Neoantigen Vaccine Pipeline - Step 1")
    print("=" * 70)

    # 1. Fetch project summary
    print("\n[1/4] Fetching project summary...")
    try:
        summary = fetch_project_summary()
        summary_path = output_dir / "project_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Project: {summary.get('name', PROJECT_ID)}")
        print(f"  Summary saved to: {summary_path}")
    except Exception as e:
        print(f"  Warning: Could not fetch project summary: {e}")

    # 2. Fetch clinical case data
    print(f"\n[2/4] Fetching clinical data (up to {args.max_cases} cases)...")
    cases_df = fetch_cases(max_cases=args.max_cases)
    cases_path = output_dir / "mmrf_cases.csv"
    cases_df.to_csv(cases_path, index=False)
    print(f"  Retrieved {len(cases_df)} cases")
    print(f"  Saved to: {cases_path}")

    if len(cases_df) > 0:
        print(f"\n  --- Clinical Data Summary ---")
        if "gender" in cases_df.columns:
            print(f"  Gender distribution:")
            print(f"    {cases_df['gender'].value_counts().to_dict()}")
        if "vital_status" in cases_df.columns:
            print(f"  Vital status:")
            print(f"    {cases_df['vital_status'].value_counts().to_dict()}")
        if "age_at_diagnosis" in cases_df.columns:
            ages = cases_df["age_at_diagnosis"].dropna()
            if len(ages) > 0:
                # GDC reports age in days
                ages_years = ages / 365.25
                print(f"  Age at diagnosis: median {ages_years.median():.0f} years "
                      f"(range: {ages_years.min():.0f}-{ages_years.max():.0f})")

    # 3. Fetch mutation data (SSM occurrences - open access)
    print(f"\n[3/4] Fetching somatic mutations (up to {args.max_mutations})...")
    mutations_df = fetch_ssm_occurrences(max_results=args.max_mutations)
    mutations_path = output_dir / "mmrf_mutations.csv"
    mutations_df.to_csv(mutations_path, index=False)
    print(f"  Retrieved {len(mutations_df)} mutation records")
    print(f"  Saved to: {mutations_path}")

    if len(mutations_df) > 0:
        print(f"\n  --- Mutation Summary ---")
        print(f"  Unique patients with mutations: "
              f"{mutations_df['case_id'].nunique()}")
        if "consequence_type" in mutations_df.columns:
            print(f"  Consequence types:")
            for ct, count in mutations_df["consequence_type"].value_counts().head(10).items():
                print(f"    {ct}: {count}")
        if "gene_symbol" in mutations_df.columns:
            print(f"\n  Most frequently mutated genes:")
            gene_counts = mutations_df.groupby("gene_symbol")["case_id"].nunique()
            for gene, count in gene_counts.sort_values(ascending=False).head(15).items():
                print(f"    {gene}: {count} patients")

    # 4. Fetch file manifests
    print(f"\n[4/4] Fetching file manifests...")
    mut_files = fetch_mutation_files(max_files=20)
    expr_files = fetch_expression_files(max_files=20)

    manifest = {
        "mutation_files": mut_files,
        "expression_files": expr_files,
    }
    manifest_path = output_dir / "file_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Found {len(mut_files)} mutation files, {len(expr_files)} expression files")
    print(f"  Manifest saved to: {manifest_path}")

    # Optional: Download MAF files
    if args.download_mafs and mut_files:
        print(f"\n[Optional] Downloading MAF files...")
        maf_dir = output_dir / "maf_files"
        for file_info in tqdm(mut_files[:5], desc="Downloading MAFs"):
            try:
                download_file(
                    file_info["file_id"],
                    str(maf_dir),
                    file_info.get("file_name")
                )
            except Exception as e:
                print(f"  Warning: Failed to download {file_info.get('file_name')}: {e}")
            time.sleep(1)

    print("\n" + "=" * 70)
    print("Data fetch complete!")
    print(f"All data saved to: {output_dir}/")
    print("\nNext step: Run 02_parse_mutations.py to process mutations")
    print("=" * 70)


if __name__ == "__main__":
    main()
