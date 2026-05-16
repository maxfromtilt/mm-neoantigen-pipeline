#!/usr/bin/env python3
"""
Step 10 — PeptideAtlas & PRIDE Proteomics Validation
====================================================
Queries PeptideAtlas and PRIDE (PRoteomics IDEntifications) databases to validate
whether genes/proteins containing predicted neoepitopes have been detected by
mass spectrometry in real human tissue samples.

PeptideAtlas aggregates shotgun proteomics experiments from hundreds of studies,
providing a non-redundant human proteome library. This step confirms that a
computationally predicted mutant peptide corresponds to a protein that is
actually expressed and detected in human samples — not just in silico.

Usage:
    python 10_peptide_atlas.py [--config config.yaml] [--input output/selected_epitopes.csv]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUTPUT_DIR = Path("output")
PROTEOMICS_DIR = OUTPUT_DIR / "proteomics"
PROTEOMICS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API clients with retry logic
# ---------------------------------------------------------------------------

def make_session(retries: int = 3, backoff: float = 2.0) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# PeptideAtlas API
# https://www.peptideatlas.org/api/v1/
# ---------------------------------------------------------------------------

PEPTIDEATLAS_BASE = "https://www.peptideatlas.org/api/v1"

# Canonical human UniProt tax ID used by PeptideAtlas
HUMAN_TAX_ID = 9606


def query_protein_at_peptideatlas(gene_symbol: str, session: requests.Session,
                                   timeout: float = 15.0) -> dict:
    """
    Query PeptideAtlas API for a gene's protein detection status.

    Returns dict:
        protein_detected: bool
        experiment_count: int
        tissue_types: list[str]
        expression_confidence: str  # High | Medium | Low | Not Detected
        peptide_count: int
        coverage_percent: float | None
        message: str
    """
    result = {
        "protein_detected": False,
        "experiment_count": 0,
        "tissue_types": [],
        "expression_confidence": "Not Detected",
        "peptide_count": 0,
        "coverage_percent": None,
        "message": "",
    }

    # PeptideAtlas resource endpoint for protein存在感
    # Try SwissProt accession search first
    url = f"{PEPTIDEATLAS_BASE}/proteins/"

    headers = {"Accept": "application/json"}

    try:
        # Search by gene symbol in the proteins endpoint
        # PeptideAtlas uses various query params; try symbol search
        resp = session.get(
            url,
            params={
                "search": gene_symbol,
                "tax": HUMAN_TAX_ID,
                "count": 1,
            },
            timeout=timeout,
        )

        if resp.status_code == 404:
            result["message"] = "Gene not found in PeptideAtlas"
            return result
        if resp.status_code == 429:
            result["message"] = "Rate limited by PeptideAtlas"
            return result
        if not resp.ok:
            result["message"] = f"HTTP {resp.status_code}"
            return result

        data = resp.json()

        # PeptideAtlas returns a list of protein entries
        resources = data.get("results", []) or data.get("response", {}).get("results", [])

        if not resources:
            result["message"] = "No proteins found for gene"
            return result

        # Take the top match
        protein = resources[0]

        result["protein_detected"] = True
        result["experiment_count"] = protein.get("nr_peptides", protein.get("peptide_count", 0))
        result["peptide_count"] = result["experiment_count"]

        # PeptideAtlas provides tissue classification via the "ts" (tissue) field
        tissues = protein.get("tissue", protein.get("tissues", []))
        if isinstance(tissues, str):
            tissues = [tissues]
        result["tissue_types"] = tissues[:20]  # cap at 20 tissues

        # Expression confidence derived from peptide count and experiment coverage
        pep_cnt = result["peptide_count"]
        if pep_cnt >= 10:
            result["expression_confidence"] = "High"
        elif pep_cnt >= 3:
            result["expression_confidence"] = "Medium"
        elif pep_cnt >= 1:
            result["expression_confidence"] = "Low"
        else:
            result["expression_confidence"] = "Not Detected"

        coverage = protein.get("coverage_percent") or protein.get("sequence_coverage")
        if coverage is not None:
            result["coverage_percent"] = float(coverage)

        result["message"] = "Found in PeptideAtlas"

    except requests.exceptions.Timeout:
        result["message"] = "PeptideAtlas timeout"
    except requests.exceptions.RequestException as e:
        result["message"] = f"PeptideAtlas error: {e}"

    return result


def query_protein_by_uniprot(uniprot_acc: str, session: requests.Session,
                              timeout: float = 10.0) -> dict:
    """
    Fallback: query by UniProt accession to get protein details from PeptideAtlas.
    """
    result = {
        "protein_detected": False,
        "experiment_count": 0,
        "tissue_types": [],
        "expression_confidence": "Not Detected",
        "peptide_count": 0,
        "coverage_percent": None,
        "message": "",
    }

    url = f"{PEPTIDEATLAS_BASE}/proteins/{uniprot_acc}"
    try:
        resp = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
        if resp.ok:
            protein = resp.json()
            result["protein_detected"] = True
            result["peptide_count"] = protein.get("nr_peptides", 0)
            result["experiment_count"] = result["peptide_count"]
            tissues = protein.get("tissue", [])
            result["tissue_types"] = tissues[:20] if isinstance(tissues, list) else []
            pep_cnt = result["peptide_count"]
            if pep_cnt >= 10:
                result["expression_confidence"] = "High"
            elif pep_cnt >= 3:
                result["expression_confidence"] = "Medium"
            elif pep_cnt >= 1:
                result["expression_confidence"] = "Low"
            result["message"] = "Found via UniProt accession"
        else:
            result["message"] = f"UniProt accession not found (HTTP {resp.status_code})"
    except Exception as e:
        result["message"] = f"UniProt lookup error: {e}"

    return result


# ---------------------------------------------------------------------------
# PRIDE API
# https://www.ebi.ac.uk/pride/ws/archive/v2
# ---------------------------------------------------------------------------

PRIDE_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"


def query_protein_at_pride(gene_symbol: str, session: requests.Session,
                           timeout: float = 15.0) -> dict:
    """
    Query PRIDE for protein detection data.

    Returns dict:
        protein_detected: bool
        experiment_count: int
        tissue_types: list[str]
        expression_confidence: str
        publication_count: int
        message: str
    """
    result = {
        "protein_detected": False,
        "experiment_count": 0,
        "tissue_types": [],
        "expression_confidence": "Not Detected",
        "publication_count": 0,
        "message": "",
    }

    try:
        # Search PRIDE projects for gene/protein
        url = f"{PRIDE_BASE}/search/summary/protein"
        resp = session.get(
            url,
            params={"gene": gene_symbol, "species": "Homo sapiens", "pageSize": 5},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

        if not resp.ok:
            result["message"] = f"PRIDE HTTP {resp.status_code}"
            return result

        data = resp.json()
        results_list = data.get("result", []) or data.get("results", [])

        if not results_list:
            result["message"] = "No PRIDE entries found"
            return result

        result["protein_detected"] = True
        result["publication_count"] = data.get("numHits", 0) or len(results_list)

        # Collect tissue information from PRIDE entries
        tissues = set()
        for entry in results_list[:10]:
            for kw in entry.get("keywords", []):
                k = kw.lower()
                if any(t in k for t in ["tissue", "cell line", "organ"]):
                    tissues.add(kw)
        result["tissue_types"] = sorted(list(tissues))[:15]

        hits = result["publication_count"]
        if hits >= 20:
            result["expression_confidence"] = "High"
        elif hits >= 5:
            result["expression_confidence"] = "Medium"
        elif hits >= 1:
            result["expression_confidence"] = "Low"
        else:
            result["expression_confidence"] = "Not Detected"

        result["experiment_count"] = hits
        result["message"] = "Found in PRIDE"

    except requests.exceptions.Timeout:
        result["message"] = "PRIDE timeout"
    except Exception as e:
        result["message"] = f"PRIDE error: {e}"

    return result


# ---------------------------------------------------------------------------
# Epitope proteome coverage check
# ---------------------------------------------------------------------------

def check_epitope_in_proteome(epitope_sequence: str, session: requests.Session,
                                timeout: float = 10.0) -> dict:
    """
    Check if an exact peptide sequence (mutant or wildtype) has been
    detected in PeptideAtlas's non-redundant peptide library.

    Returns:
        peptide_detected: bool
        database_source: str
        confidence: str
    """
    result = {
        "peptide_detected": False,
        "database_source": "",
        "confidence": "Not Found",
    }

    # PeptideAtlas non-redundant library search
    # The API may support peptide-level search
    try:
        url = f"{PEPTIDEATLAS_BASE}/peptides/"
        resp = session.get(
            url,
            params={"sequence": epitope_sequence, "tax": HUMAN_TAX_ID},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

        if resp.ok:
            data = resp.json()
            hits = data.get("results", []) or data.get("response", {}).get("results", [])
            if hits:
                result["peptide_detected"] = True
                result["database_source"] = "PeptideAtlas"
                result["confidence"] = "Confirmed"
                return result

        # Fallback: try Pride
        url2 = f"{PRIDE_BASE}/search/summary/peptide"
        resp2 = session.get(
            url2,
            params={"sequence": epitope_sequence},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

        if resp2.ok:
            data2 = resp2.json()
            hits2 = data2.get("result", []) or data2.get("results", [])
            if hits2:
                result["peptide_detected"] = True
                result["database_source"] = "PRIDE"
                result["confidence"] = "Confirmed"
                return result

        result["confidence"] = "Not Found"
        result["database_source"] = "None"

    except Exception as e:
        result["database_source"] = f"Error: {e}"

    return result


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate_proteomics(epitopes_df: pd.DataFrame, config: dict,
                         session: requests.Session) -> pd.DataFrame:
    """
    For each gene in the epitopes dataframe, query PeptideAtlas and PRIDE
    for protein expression confirmation.

    Adds columns:
        protein_detected (PeptideAtlas)
        experiment_count
        tissue_types
        expression_confidence
        pride_detected
        pride_experiment_count
        peptide_proteome_detected
        proteomics_confidence  # Combined score
    """
    genes = epitopes_df["gene_symbol"].dropna().unique()
    gene_list = list(genes)

    print(f"\n[PeptideAtlas] Querying {len(gene_list)} unique genes...")
    print(f"[PeptideAtlas] This may take several minutes due to API rate limits.")

    pa_config = config.get("peptide_atlas", {})
    confidence_boost = pa_config.get("expression_confidence_boost", 8)
    detection_threshold = pa_config.get("protein_detection_threshold", 1)

    # Store results keyed by gene
    gene_proteomics = {}

    for i, gene in enumerate(gene_list):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(gene_list)}] Querying {gene}...")

        # Query PeptideAtlas
        pa_result = query_protein_at_peptideatlas(gene, session)

        # Also try PRIDE
        pride_result = query_protein_at_pride(gene, session)

        gene_proteomics[gene] = {
            "pa": pa_result,
            "pride": pride_result,
        }

        # Respectful rate limiting — PeptideAtlas is a public resource
        time.sleep(0.25)

    print(f"[PeptideAtlas] Query complete. Processing {len(gene_list)} genes.")

    rows = []
    for _, epitope_row in epitopes_df.iterrows():
        gene = epitope_row.get("gene_symbol", "Unknown")
        epitope = epitope_row.get("mutant_peptide", epitope_row.get("peptide_sequence", ""))

        proteomics = gene_proteomics.get(gene, {})
        pa = proteomics.get("pa", {})
        pride = proteomics.get("pride", {})

        # Combined proteomics confidence
        # Prefer PeptideAtlas; supplement with PRIDE
        if pa["protein_detected"]:
            pa_conf = pa["expression_confidence"]
            pa_exp = pa["experiment_count"]
        else:
            pa_conf = "Not Detected"
            pa_exp = 0

        if pride["protein_detected"]:
            pr_conf = pride["expression_confidence"]
            pr_exp = pride["experiment_count"]
        else:
            pr_conf = "Not Detected"
            pr_exp = 0

        # Pick the best evidence
        conf_order = {"High": 4, "Medium": 3, "Low": 2, "Not Detected": 1}
        pa_score = conf_order.get(pa_conf, 0)
        pr_score = conf_order.get(pr_conf, 0)

        if pa_score >= pr_score:
            best_conf = pa_conf
            best_source = "PeptideAtlas"
            best_exp = pa_exp
            best_tissues = pa.get("tissue_types", [])
        else:
            best_conf = pr_conf
            best_source = "PRIDE"
            best_exp = pr_exp
            best_tissues = pride.get("tissue_types", [])

        # Epitope-level proteome check
        if epitope:
            epi_result = check_epitope_in_proteome(epitope, session)
        else:
            epi_result = {"peptide_detected": False, "database_source": "", "confidence": "Not Found"}

        # Build combined confidence
        if best_conf == "High" and epi_result.get("peptide_detected"):
            combined_conf = "High (proteome confirmed)"
        elif best_conf == "High":
            combined_conf = "High"
        elif best_conf == "Medium":
            combined_conf = "Medium"
        elif best_conf == "Low":
            combined_conf = "Low"
        else:
            combined_conf = "Not Detected"

        row = {
            "gene_symbol": gene,
            "epitope": epitope,
            "protein_detected": pa["protein_detected"] or pride["protein_detected"],
            "tissue_types": "; ".join(best_tissues[:10]) if best_tissues else "",
            "experiment_count": max(pa_exp, pr_exp),
            "expression_confidence": combined_conf,
            "peptide_proteome_detected": epi_result.get("peptide_detected", False),
            "peptide_database_source": epi_result.get("database_source", ""),
            "peptide_proteome_confidence": epi_result.get("confidence", "Not Found"),
            "peptide_atlas_detected": pa["protein_detected"],
            "peptide_atlas_experiment_count": pa_exp,
            "peptide_atlas_confidence": pa_conf,
            "pride_detected": pride["protein_detected"],
            "pride_experiment_count": pr_exp,
            "pride_confidence": pr_conf,
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Apply expression confidence boost to vaccine_priority_score if configured
    if confidence_boost > 0 and "vaccine_priority_score" in epitopes_df.columns:
        conf_boost_map = {
            "High (proteome confirmed)": confidence_boost,
            "High": confidence_boost * 0.9,
            "Medium": confidence_boost * 0.5,
            "Low": confidence_boost * 0.2,
            "Not Detected": 0,
        }
        boost = result_df["expression_confidence"].map(conf_boost_map).fillna(0)
        result_df["priority_score_boost"] = boost
        result_df["adjusted_vaccine_priority_score"] = (
            epitopes_df["vaccine_priority_score"].values + boost
        )

    return result_df


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------

def write_proteomics_validation(df: pd.DataFrame, output_dir: Path):
    """Write proteomics_validation.csv — per-gene protein detection summary."""

    # Deduplicate to gene level
    gene_df = (
        df[["gene_symbol", "protein_detected", "tissue_types",
            "experiment_count", "expression_confidence",
            "peptide_atlas_detected", "peptide_atlas_experiment_count",
            "pride_detected", "pride_experiment_count"]]
        .drop_duplicates(subset=["gene_symbol"])
        .sort_values(["protein_detected", "experiment_count"],
                     ascending=[False, False])
    )

    out_path = output_dir / "proteomics_validation.csv"
    gene_df.to_csv(out_path, index=False)
    print(f"[Output] {out_path}")
    return out_path


def write_epitope_proteome_coverage(df: pd.DataFrame, output_dir: Path):
    """Write epitope_proteome_coverage.csv — per-epitope proteome evidence."""

    cols = [
        "epitope", "gene_symbol", "protein_detected",
        "peptide_proteome_detected", "peptide_database_source",
        "expression_confidence", "tissue_types",
        "experiment_count",
    ]
    available = [c for c in cols if c in df.columns]
    epi_df = df[available].copy()

    out_path = output_dir / "epitope_proteome_coverage.csv"
    epi_df.to_csv(out_path, index=False)
    print(f"[Output] {out_path}")
    return out_path


def write_proteomics_report(df: pd.DataFrame, output_dir: Path):
    """Write a human-readable proteomics validation report."""

    total_genes = df["gene_symbol"].nunique()
    detected = df[df["protein_detected"] == True]["gene_symbol"].nunique()
    high_conf = df[df["expression_confidence"].str.contains("High", na=False)]["gene_symbol"].nunique()
    peptide_confirmed = df[df["peptide_proteome_detected"] == True]["gene_symbol"].nunique()

    pa_detected = df[df["peptide_atlas_detected"] == True]["gene_symbol"].nunique()
    pride_detected = df[df["pride_detected"] == True]["gene_symbol"].nunique()

    report_lines = [
        "=" * 70,
        "PEPTIDEATLAS / PRIDE PROTEOMICS VALIDATION REPORT",
        "=" * 70,
        "",
        f"Total genes validated:     {total_genes}",
        f"Protein detected (PA/PRIDE): {detected} ({detected/total_genes*100:.1f}%)",
        f"  - PeptideAtlas:        {pa_detected}",
        f"  - PRIDE:               {pride_detected}",
        f"High confidence:          {high_conf} ({high_conf/total_genes*100:.1f}%)",
        f"Peptide proteome confirmed: {peptide_confirmed}",
        "",
        "-" * 70,
        "GENE-LEVEL SUMMARY (sorted by expression confidence)",
        "-" * 70,
    ]

    # Top genes by confidence
    summary_cols = ["gene_symbol", "expression_confidence", "experiment_count",
                    "tissue_types", "peptide_proteome_detected"]
    available_cols = [c for c in summary_cols if c in df.columns]
    top_genes = (
        df[available_cols]
        .drop_duplicates(subset=["gene_symbol"])
        .sort_values("expression_confidence")
        .head(30)
    )

    conf_icons = {"High (proteome confirmed)": "***", "High": "**", "Medium": "*",
                   "Low": ".", "Not Detected": "-"}

    for _, row in top_genes.iterrows():
        conf = row.get("expression_confidence", "Unknown")
        icon = conf_icons.get(conf, "?")
        tissues = str(row.get("tissue_types", ""))[:50]
        pep_det = row.get("peptide_proteome_detected", False)
        pep_mark = "[PEPTIDE]" if pep_det else ""
        report_lines.append(
            f"  {icon:3} {row['gene_symbol']:<15} | {conf:<28} | "
            f"expts={row.get('experiment_count', 0):<4} | {tissues[:45]} {pep_mark}"
        )

    report_lines.extend(["", "-" * 70, "DATABASE COVERAGE DETAILS", "-" * 70])

    # Tissue breakdown
    all_tissues = []
    for tlist in df["tissue_types"].dropna():
        all_tissues.extend([t.strip() for t in str(tlist).split(";") if t.strip()])

    if all_tissues:
        from collections import Counter
        tissue_counts = Counter(all_tissues).most_common(20)
        report_lines.append("\nTop tissues with detected proteins:")
        for tissue, count in tissue_counts:
            report_lines.append(f"  {count:3}  {tissue}")

    report_lines.extend(["", "=" * 70])

    out_path = output_dir / "proteomics_report.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"[Output] {out_path}")
    print(f"\n[Summary] Genes: {total_genes} | Detected: {detected} ({detected/total_genes*100:.1f}%) "
          f"| High: {high_conf} | Peptide confirmed: {peptide_confirmed}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="PeptideAtlas & PRIDE proteomics validation for neoantigen pipeline"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument(
        "--input",
        default="output/selected_epitopes.csv",
        help="Selected epitopes CSV (from 06_validate_pipeline.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/proteomics",
        help="Output directory for proteomics results",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Step 10: PeptideAtlas & PRIDE Proteomics Validation")
    print("=" * 70)

    config = load_config(args.config)

    # Load epitopes
    epitope_path = Path(args.input)
    if not epitope_path.exists():
        # Try to find it in any patient output directory
        candidates = sorted(Path("output").glob("*/selected_epitopes.csv"))
        if candidates:
            epitope_path = candidates[-1]
            print(f"[Info] Using latest epitopes: {epitope_path}")
        else:
            print(f"[ERROR] {args.input} not found. Run pipeline steps 1-6 first.")
            sys.exit(1)

    print(f"[Input] Loading epitopes from: {epitope_path}")
    epitopes_df = pd.read_csv(epitope_path)
    print(f"[Input] {len(epitopes_df)} epitopes from {epitopes_df['gene_symbol'].nunique()} genes")

    # Create session
    session = make_session(retries=3, backoff=2.0)

    # Validate
    result_df = validate_proteomics(epitopes_df, config, session)

    # Outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_proteomics_validation(result_df, output_dir)
    write_epitope_proteome_coverage(result_df, output_dir)
    write_proteomics_report(result_df, output_dir)

    print(f"\n{'=' * 70}")
    print("  PeptideAtlas validation complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()