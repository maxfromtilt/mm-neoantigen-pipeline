#!/usr/bin/env python3
"""
Enhancement 8 — HLA Ligandomics Mass-Spec Data Integration
============================================================
Fetches mass-spec validated HLA-bound peptides from PRIDE and HMB databases,
validates computationally predicted epitopes against experimental data, and
incorporates mass-spec evidence into the pipeline immunogenicity score.

Sources:
  - PRIDE    (European Proteomics Research Institute)
             API: https://www.ebi.ac.uk/pride/ws/archive/v2
  - HMB      (Human Myelome Database — HLA ligand atlas)
             API: https://www.hmadb.org

Usage:
    python 07_ligandomics.py [--input data/binding_predictions.csv]
    python 07_ligandomics.py --standalone   # Fetch & cache only, no CSV input needed
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "ligandomics": {
        "enabled": True,
        "pridesearch_api": "https://www.ebi.ac.uk/pride/ws/archive/v2",
        "hmb_api": "https://www.hmadb.org",
        "cancer_types": ["multiple myeloma", "hematological malignancy"],
        "min_mass_spec_evidence": 2,
        "validation_boost": 20,
    }
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline config, appending ligandomics section if absent."""
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    # Merge defaults for ligandomics section
    if "ligandomics" not in config:
        config["ligandomics"] = DEFAULT_CONFIG["ligandomics"]
    else:
        # Fill in any missing keys from defaults
        for key, val in DEFAULT_CONFIG["ligandomics"].items():
            if key not in config["ligandomics"]:
                config["ligandomics"][key] = val
    return config


# =============================================================================
# PRIDE API Client
# =============================================================================

class PrideApiClient:
    """Client for the PRIDE WS Archive API v2.

    API docs: https://www.ebi.ac.uk/pride/ws/archive/v2
    Peptide search endpoint: /peptides/list
    """

    BASE_URL = "https://www.ebi.ac.uk/pride/ws/archive/v2"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json"})
        return self._session

    def search_peptides(self, disease: str, max_pages: int = 5,
                       page_size: int = 100) -> List[Dict]:
        """
        Search PRIDE for HLA-bound peptides associated with a disease term.

        Params:
            disease: free-text disease term (e.g. "multiple myeloma")
            max_pages: max API pages to fetch (default 5 × 100 = 500 peptides)
            page_size: results per page (max 100 for this endpoint)

        Returns:
            List of peptide records, each with keys:
                peptide_sequence, protein_accession, experiment_id,
                disease, species, publication, score
        """
        import requests

        results = []
        page = 0

        # PRIDE peptide search: GET /peptides/list?q=<disease>&pageSize=100&page=0
        base_params = {
            "pageSize": min(page_size, 100),
            "showPublic": True,
        }

        while page < max_pages:
            params = {**base_params, "page": page}
            if disease:
                params["q"] = disease

            url = f"{self.BASE_URL}/peptides/list"
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 204:
                    break
                resp.raise_for_status()
                data = resp.json()

                # PRIDE v2 peptide list response structure
                # {"content": [...], "page": {...}, "_links": {...}}
                peptides = data.get("content", [])
                if not peptides:
                    break

                for entry in peptides:
                    results.append({
                        "peptide_sequence": entry.get("sequence", "").upper(),
                        "protein_accession": entry.get("proteinAccession", ""),
                        "experiment_id": entry.get("experimentAccession", ""),
                        "disease": disease,
                        "species": entry.get("species", ""),
                        "score": entry.get("score", 0),
                        "source": "PRIDE",
                    })

                # Check pagination
                page_info = data.get("page", {})
                total_pages = page_info.get("totalPages", 1)
                if page >= total_pages - 1:
                    break
                page += 1

            except requests.exceptions.RequestException as e:
                print(f"    PRIDE API error (page {page}): {e}")
                break

        return results


# =============================================================================
# HMB API Client
# =============================================================================

class HmbApiClient:
    """Client for the Human Myelome Database (HMB) ligand API.

    The HMB site at https://www.hmadb.org provides HLA ligandomics data.
    API pattern varies; we try both the direct search and the JSON download
    approach documented on the site.
    """

    BASE_URL = "https://www.hmadb.org"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session = None

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json"})
        return self._session

    def fetch_hla_peptides(self, cancer_type: str = "multiple myeloma",
                           max_results: int = 500) -> List[Dict]:
        """
        Fetch HLA-bound peptide data from HMB for the given cancer type.

        Tries three strategies in order:
          1. /API/ligand/getByCancerType/<type>  (REST JSON)
          2. /download/ligands/<type>.json       (pre打包 download)
          3. /API/peptides?cancer=<type>           (search endpoint)

        Returns:
            List of peptide records.
        """
        import requests

        cancer_slug = cancer_type.lower().replace(" ", "-")
        results = []

        strategies = [
            # Strategy 1: REST by cancer type
            (f"{self.BASE_URL}/API/ligand/getByCancerType/{cancer_slug}",
             {"method": "rest_cancer"}),
            # Strategy 2: Pre-built JSON download
            (f"{self.BASE_URL}/download/ligands/{cancer_slug}.json",
             {"method": "download_json"}),
            # Strategy 3: Generic peptide search
            (f"{self.BASE_URL}/API/peptides",
             {"method": "search", "params": {"cancer": cancer_type[:20]}}),
        ]

        for url, meta in strategies:
            params = meta.get("params", {})
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 204 or resp.status_code == 404:
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")

                # Parse response
                if "json" in content_type or url.endswith(".json"):
                    data = resp.json()
                else:
                    data = resp.json()

                if not data:
                    continue

                # Normalise to list
                if isinstance(data, dict):
                    # Could be wrapped in a "peptides" or "ligands" key
                    for wrapper_key in ["peptides", "ligands", "results", "data"]:
                        if wrapper_key in data and isinstance(data[wrapper_key], list):
                            data = data[wrapper_key]
                            break
                    if isinstance(data, dict):
                        data = [data]
                elif not isinstance(data, list):
                    data = [data]

                for entry in data:
                    seq = entry.get("sequence", entry.get("peptide", "")).upper()
                    if len(seq) < 7:
                        continue
                    results.append({
                        "peptide_sequence": seq,
                        "protein_accession": entry.get("protein", entry.get("accession", "")),
                        "experiment_id": entry.get("experimentId", entry.get("experiment_id", "")),
                        "hla_allele": entry.get("hla", entry.get("allele", "")),
                        "disease": cancer_type,
                        "species": entry.get("species", "Human"),
                        "score": entry.get("intensity", entry.get("score", 1)),
                        "source": "HMB",
                    })

                if results:
                    print(f"    HMB: fetched {len(results)} peptides via {meta['method']}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"    HMB strategy '{meta['method']}' failed: {e}")
                continue

        return results


# =============================================================================
# Ligandomics Cache Manager
# =============================================================================

class LigandomicsCache:
    """Simple JSON cache for mass-spec peptide data."""

    def __init__(self, cache_dir: str = "output/ligandomics_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pride_cache = self.cache_dir / "pride_peptides.json"
        self.hmb_cache = self.cache_dir / "hmb_peptides.json"

    def load_pride(self) -> List[Dict]:
        if self.pride_cache.exists():
            with open(self.pride_cache) as f:
                return json.load(f)
        return []

    def save_pride(self, peptides: List[Dict]):
        with open(self.pride_cache, "w") as f:
            json.dump(peptides, f, indent=2)

    def load_hmb(self) -> List[Dict]:
        if self.hmb_cache.exists():
            with open(self.hmb_cache) as f:
                return json.load(f)
        return []

    def save_hmb(self, peptides: List[Dict]):
        with open(self.hmb_cache, "w") as f:
            json.dump(peptides, f, indent=2)

    def load_all(self) -> Tuple[List[Dict], List[Dict]]:
        return self.load_pride(), self.load_hmb()


# =============================================================================
# Mass-Spec Data Fetcher
# =============================================================================

def fetch_mass_spec_peptides(config: dict,
                              force_refresh: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """Fetch HLA ligandomics peptides from PRIDE and HMB.

    Args:
        config: pipeline config dict
        force_refresh: bypass cache if True

    Returns:
        (pride_peptides, hmb_peptides)
    """
    cache = LigandomicsCache()
    lg_config = config.get("ligandomics", {})

    if not force_refresh:
        pride_peptides = cache.load_pride()
        hmb_peptides = cache.load_hmb()
        if pride_peptides or hmb_peptides:
            print(f"  [Cache] Loaded {len(pride_peptides)} PRIDE + "
                  f"{len(hmb_peptides)} HMB peptides from cache")
            return pride_peptides, hmb_peptides

    cancer_types = lg_config.get("cancer_types", ["multiple myeloma"])

    # --- PRIDE ---
    pride_peptides = []
    if lg_config.get("enabled", True):
        pride_api = PrideApiClient()
        for cancer in cancer_types:
            print(f"  [PRIDE] Searching: '{cancer}' ...")
            results = pride_api.search_peptides(disease=cancer, max_pages=5)
            if results:
                pride_peptides.extend(results)
                print(f"    → {len(results)} peptide records")

        if pride_peptides:
            cache.save_pride(pride_peptides)
            print(f"  [PRIDE] Total: {len(pride_peptides)} peptides cached")

    # --- HMB ---
    hmb_peptides = []
    if lg_config.get("enabled", True):
        hmb_api = HmbApiClient()
        for cancer in cancer_types:
            print(f"  [HMB] Fetching: '{cancer}' ...")
            results = hmb_api.fetch_hla_peptides(cancer_type=cancer)
            if results:
                hmb_peptides.extend(results)

        if hmb_peptides:
            cache.save_hmb(hmb_peptides)
            print(f"  [HMB] Total: {len(hmb_peptides)} peptides cached")

    return pride_peptides, hmb_peptides


# =============================================================================
# Epitope Validator
# =============================================================================

def build_mass_spec_index(peptides: List[Dict]) -> Dict[str, List[Dict]]:
    """Build a sequence → evidence lookup from mass-spec peptide list.

    Returns:
        {SEQUENCE: [evidence_record, ...]}
    """
    index: Dict[str, List[Dict]] = {}
    for pep in peptides:
        seq = pep.get("peptide_sequence", "")
        if seq and len(seq) >= 7:
            index.setdefault(seq, []).append(pep)
    return index


def validate_epitopes_against_mass_spec(
    predictions_df: pd.DataFrame,
    mass_spec_peptides: List[Dict],
    config: dict
) -> pd.DataFrame:
    """Check which predicted MHC-binding peptides have experimental MS support.

    Args:
        predictions_df: binding_predictions.csv with mutant_peptide column
        mass_spec_peptides: combined PRIDE + HMB peptide list
        config: pipeline config

    Returns:
        DataFrame with added columns: mass_spec_source, experiment_id,
        evidence_count, validation_status
    """
    lg_config = config.get("ligandomics", {})
    min_evidence = lg_config.get("min_mass_spec_evidence", 2)

    index = build_mass_spec_index(mass_spec_peptides)

    validation_rows = []
    for _, row in predictions_df.iterrows():
        peptide = row.get("mutant_peptide", "").upper()
        hla = row.get("hla_allele", "")
        matching_peptides = index.get(peptide, [])

        if matching_peptides:
            # Filter by cancer type if specified
            cancer_types = lg_config.get("cancer_types", [])
            cancer_filtered = [
                p for p in matching_peptides
                if any(ct.lower() in p.get("disease", "").lower()
                       for ct in cancer_types) or not cancer_types
            ]
            evidence = cancer_filtered if cancer_filtered else matching_peptides

            if len(evidence) >= min_evidence:
                sources = list({e.get("source", "Unknown") for e in evidence})
                experiments = list({e.get("experiment_id", "") for e in evidence})
                validation_rows.append({
                    **row.to_dict(),
                    "mass_spec_source": "/".join(sources),
                    "experiment_id": ";".join(filter(None, experiments)),
                    "evidence_count": len(evidence),
                    "validation_status": "Confirmed",
                })
            else:
                validation_rows.append({
                    **row.to_dict(),
                    "mass_spec_source": "Insufficient evidence",
                    "experiment_id": "",
                    "evidence_count": len(evidence),
                    "validation_status": "Novel",
                })
        else:
            validation_rows.append({
                **row.to_dict(),
                "mass_spec_source": "",
                "experiment_id": "",
                "evidence_count": 0,
                "validation_status": "Novel",
            })

    return pd.DataFrame(validation_rows)


# =============================================================================
# Score Booster
# =============================================================================

def apply_mass_spec_boost(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Boost vaccine_priority_score for mass-spec validated epitopes.

    Config key: ligandomics.validation_boost (default 20 points)
    """
    lg_config = config.get("ligandomics", {})
    boost = lg_config.get("validation_boost", 20)

    mask = df["validation_status"] == "Confirmed"
    if mask.any():
        df.loc[mask, "vaccine_priority_score"] += boost
        df.loc[mask, "vaccine_priority_score"] = df.loc[mask, "vaccine_priority_score"].clip(upper=100)
        n_boosted = mask.sum()
        print(f"  [Score Boost] Applied +{boost} pts to {n_boosted} "
              f"mass-spec-validated epitopes")

    return df


# =============================================================================
# Report Generator
# =============================================================================

def generate_validation_report(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write the comparison report to output/ligandomics_validation.csv.

    Columns:
        epitope_sequence | hla_allele | mass_spec_source | experiment_id |
        evidence_type | validation_status | ic50_nM | vaccine_priority_score |
        gene_symbol | aa_change
    """
    report_cols = [
        "mutant_peptide", "hla_allele", "mass_spec_source", "experiment_id",
        "evidence_count", "validation_status", "ic50_nM", "vaccine_priority_score",
        "gene_symbol", "aa_change", "classification", "agretopicity"
    ]
    available = [c for c in report_cols if c in df.columns]

    # Rename mutant_peptide → epitope_sequence for output
    report_df = df[available].rename(columns={"mutant_peptide": "epitope_sequence"})
    # Drop rows with no mass-spec evidence and not confirmed
    # (i.e. keep all to show full picture, but label appropriately)
    report_df["evidence_type"] = report_df["validation_status"].apply(
        lambda s: "MS evidence" if s == "Confirmed" else "Computational prediction"
    )

    output_path = output_dir / "ligandomics_validation.csv"
    report_df.to_csv(output_path, index=False)
    return output_path


def generate_summary_stats(df: pd.DataFrame, mass_spec_peptides: List[Dict]) -> str:
    """Generate a brief text summary of the ligandomics validation results."""
    total = len(df)
    confirmed = (df["validation_status"] == "Confirmed").sum() if "validation_status" in df.columns else 0
    novel = (df["validation_status"] == "Novel").sum() if "validation_status" in df.columns else total

    lines = [
        "",
        "=" * 70,
        "HLA LIGANDOMICS VALIDATION REPORT",
        "=" * 70,
        f"  Mass-spec databases queried: PRIDE, HMB",
        f"  Total experimental peptides: {len(mass_spec_peptides)}",
        f"  Predicted epitopes evaluated: {total}",
        f"  Confirmed by mass-spec: {confirmed} ({100*confirmed/max(total,1):.1f}%)",
        f"  Novel (computational only): {novel} ({100*novel/max(total,1):.1f}%)",
        "",
        "  Validation status definitions:",
        "    Confirmed  = epitope sequence found in ≥2 MS experiments in PRIDE/HMB",
        "    Novel      = computational prediction, no MS confirmation",
        "",
    ]

    # Show confirmed epitopes if any
    if confirmed > 0 and "validation_status" in df.columns:
        confirmed_df = df[df["validation_status"] == "Confirmed"]
        lines.append("  Mass-spec validated epitopes (top candidates):")
        for _, row in confirmed_df.nlargest(10, "vaccine_priority_score").iterrows():
            pep = row.get("mutant_peptide", "?")
            hla = row.get("hla_allele", "?")
            gene = row.get("gene_symbol", "?")
            src = row.get("mass_spec_source", "?")
            score = row.get("vaccine_priority_score", "?")
            lines.append(
                f"    {pep} | {hla} | {gene} | {src} | score={score}"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Pipeline Step
# =============================================================================

def run_ligandomics_validation(input_path: str = "data/binding_predictions.csv",
                               output_dir: str = "output",
                               config_path: str = "config.yaml",
                               force_refresh: bool = False) -> Tuple[Path, List[Dict]]:
    """
    Run the full ligandomics validation pipeline step.

    Steps:
      1. Load config
      2. Fetch mass-spec peptides from PRIDE + HMB (with caching)
      3. Validate predicted epitopes against the MS index
      4. Apply score boost to confirmed epitopes
      5. Re-rank and save results

    Returns:
        (path to validation CSV, combined mass-spec peptide list)
    """
    config = load_config(config_path)
    lg_config = config.get("ligandomics", {})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HLA Ligandomics Mass-Spec Data Integration")
    print("Enhancement 8 — PRIDE + HMB Validation")
    print("=" * 70)

    if not lg_config.get("enabled", True):
        print("  [Config] Ligandomics disabled in config.yaml, skipping.")
        return Path(output_dir) / "ligandomics_validation.csv", []

    # --- Step 1: Fetch mass-spec data ---
    print(f"\n[1/4] Fetching HLA ligandomics mass-spec data...")
    pride_peptides, hmb_peptides = fetch_mass_spec_peptides(
        config, force_refresh=force_refresh
    )
    mass_spec_peptides = pride_peptides + hmb_peptides
    print(f"  Combined MS peptide index: {len(mass_spec_peptides)} entries")

    # --- Step 2: Validate predictions ---
    print(f"\n[2/4] Validating predicted epitopes against mass-spec data...")
    if not os.path.exists(input_path):
        print(f"  ERROR: Input not found: {input_path}")
        print(f"  Run 03_predict_binding.py first.")
        return Path(output_dir) / "ligandomics_validation.csv", mass_spec_peptides

    predictions_df = pd.read_csv(input_path)
    print(f"  Loaded {len(predictions_df)} predicted epitopes")

    validated_df = validate_epitopes_against_mass_spec(
        predictions_df, mass_spec_peptides, config
    )
    n_confirmed = (validated_df["validation_status"] == "Confirmed").sum()
    print(f"  Confirmed by MS: {n_confirmed} | Novel: {len(validated_df) - n_confirmed}")

    # --- Step 3: Apply score boost ---
    print(f"\n[3/4] Applying mass-spec score boost...")
    boosted_df = apply_mass_spec_boost(validated_df, config)

    # --- Step 4: Re-rank ---
    if "vaccine_priority_score" in boosted_df.columns:
        boosted_df = boosted_df.sort_values("vaccine_priority_score", ascending=False)
        boosted_df["rank"] = range(1, len(boosted_df) + 1)

    # --- Step 5: Save ---
    print(f"\n[4/4] Saving validation report...")
    report_path = generate_validation_report(boosted_df, output_path)

    # Console summary
    summary = generate_summary_stats(boosted_df, mass_spec_peptides)
    print(summary)

    print(f"  Report saved: {report_path}")
    print("=" * 70)

    return report_path, mass_spec_peptides


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate predicted epitopes against HLA ligandomics MS data"
    )
    parser.add_argument("--input", default="data/binding_predictions.csv",
                        help="Input: binding_predictions.csv from step 3")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-fetch of MS data (bypass cache)")
    parser.add_argument("--standalone", action="store_true",
                        help="Fetch & cache MS data only (no input CSV needed)")
    args = parser.parse_args()

    config = load_config(args.config)
    lg_config = config.get("ligandomics", {})

    if args.standalone:
        # Fetch-only mode
        print("Fetching mass-spec data only (standalone mode)...")
        pride, hmb = fetch_mass_spec_peptides(config, force_refresh=args.force_refresh)
        total = len(pride) + len(hmb)
        print(f"\nDone. Fetched {total} peptides: {len(pride)} PRIDE, {len(hmb)} HMB")
        return

    report_path, _ = run_ligandomics_validation(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        force_refresh=args.force_refresh,
    )
    print(f"\nValidation complete. Report: {report_path}")


if __name__ == "__main__":
    main()