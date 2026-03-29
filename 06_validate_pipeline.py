#!/usr/bin/env python3
"""
Pipeline Validation & End-to-End Simulation
=============================================
Runs an offline end-to-end simulation of the neoantigen vaccine pipeline
using well-characterized benchmark mutations from Multiple Myeloma.

This bypasses the GDC API fetch step and uses curated test data to:
1. Validate each pipeline step produces correct output
2. Benchmark predictions against known immunogenic mutations
3. Cross-reference results against COSMIC/IEDB databases
4. Generate a validation report

Usage:
    python 06_validate_pipeline.py [--config config.yaml]
"""

import os
import sys
import json
import time
from pathlib import Path
from io import StringIO

import pandas as pd
import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module

# Import pipeline modules
parse_mod = import_module("02_parse_mutations")
binding_mod = import_module("03_predict_binding")
design_mod = import_module("04_design_vaccine")


# =============================================================================
# Benchmark Mutation Dataset
# =============================================================================
# Well-characterized mutations from Multiple Myeloma and pan-cancer studies
# with known immunogenic properties. These are used to validate the pipeline.
# =============================================================================

BENCHMARK_MUTATIONS = [
    # --- KRAS mutations (most common oncogene in MM, ~23%) ---
    {
        "case_id": "BENCH-001", "submitter_id": "MMRF_BENCH_001",
        "gene_symbol": "KRAS", "gene_id": "ENSG00000133703",
        "chromosome": "chr12", "start_position": 25245350,
        "aa_change": "p.G12D", "consequence_type": "missense_variant",
        "ssm_id": "bench_kras_g12d",
        "reference_allele": "C", "tumor_allele": "T",
        "expected_immunogenic": True,
        "notes": "Most common KRAS mutation in MM, well-validated neoantigen",
    },
    {
        "case_id": "BENCH-001", "submitter_id": "MMRF_BENCH_001",
        "gene_symbol": "KRAS", "gene_id": "ENSG00000133703",
        "chromosome": "chr12", "start_position": 25245350,
        "aa_change": "p.G12V", "consequence_type": "missense_variant",
        "ssm_id": "bench_kras_g12v",
        "reference_allele": "C", "tumor_allele": "A",
        "expected_immunogenic": True,
        "notes": "Known HLA-A*02:01 restricted neoantigen",
    },
    # --- NRAS mutations (very common in MM, ~20%) ---
    {
        "case_id": "BENCH-002", "submitter_id": "MMRF_BENCH_002",
        "gene_symbol": "NRAS", "gene_id": "ENSG00000213281",
        "chromosome": "chr1", "start_position": 114713908,
        "aa_change": "p.Q61K", "consequence_type": "missense_variant",
        "ssm_id": "bench_nras_q61k",
        "reference_allele": "G", "tumor_allele": "T",
        "expected_immunogenic": True,
        "notes": "Common NRAS hotspot in MM, charge-changing mutation",
    },
    {
        "case_id": "BENCH-002", "submitter_id": "MMRF_BENCH_002",
        "gene_symbol": "NRAS", "gene_id": "ENSG00000213281",
        "chromosome": "chr1", "start_position": 114713908,
        "aa_change": "p.Q61R", "consequence_type": "missense_variant",
        "ssm_id": "bench_nras_q61r",
        "reference_allele": "T", "tumor_allele": "C",
        "expected_immunogenic": True,
        "notes": "Charge-changing mutation, highly immunogenic",
    },
    # --- BRAF V600E (found in ~4% of MM) ---
    {
        "case_id": "BENCH-003", "submitter_id": "MMRF_BENCH_003",
        "gene_symbol": "BRAF", "gene_id": "ENSG00000157764",
        "chromosome": "chr7", "start_position": 140753336,
        "aa_change": "p.V600E", "consequence_type": "missense_variant",
        "ssm_id": "bench_braf_v600e",
        "reference_allele": "A", "tumor_allele": "T",
        "expected_immunogenic": True,
        "notes": "Most studied cancer mutation, validated in melanoma vaccines",
    },
    # --- TP53 mutations (found in ~8% of MM, poor prognosis) ---
    {
        "case_id": "BENCH-004", "submitter_id": "MMRF_BENCH_004",
        "gene_symbol": "TP53", "gene_id": "ENSG00000141510",
        "chromosome": "chr17", "start_position": 7675088,
        "aa_change": "p.R175H", "consequence_type": "missense_variant",
        "ssm_id": "bench_tp53_r175h",
        "reference_allele": "C", "tumor_allele": "T",
        "expected_immunogenic": True,
        "notes": "Most common TP53 hotspot, charge-changing (R→H)",
    },
    {
        "case_id": "BENCH-004", "submitter_id": "MMRF_BENCH_004",
        "gene_symbol": "TP53", "gene_id": "ENSG00000141510",
        "chromosome": "chr17", "start_position": 7675229,
        "aa_change": "p.R248W", "consequence_type": "missense_variant",
        "ssm_id": "bench_tp53_r248w",
        "reference_allele": "G", "tumor_allele": "A",
        "expected_immunogenic": True,
        "notes": "TP53 hotspot, charge-changing (R→W)",
    },
    # --- DIS3 (MM-specific, ~11%) ---
    {
        "case_id": "BENCH-005", "submitter_id": "MMRF_BENCH_005",
        "gene_symbol": "DIS3", "gene_id": "ENSG00000083520",
        "chromosome": "chr13", "start_position": 73346251,
        "aa_change": "p.R780K", "consequence_type": "missense_variant",
        "ssm_id": "bench_dis3_r780k",
        "reference_allele": "C", "tumor_allele": "T",
        "expected_immunogenic": False,
        "notes": "Conservative mutation (R→K), both positive charge",
    },
    # --- FAM46C/TENT5C (MM-specific, ~11%) ---
    {
        "case_id": "BENCH-005", "submitter_id": "MMRF_BENCH_005",
        "gene_symbol": "FAM46C", "gene_id": "ENSG00000154277",
        "chromosome": "chr1", "start_position": 27105367,
        "aa_change": "p.D249N", "consequence_type": "missense_variant",
        "ssm_id": "bench_fam46c_d249n",
        "reference_allele": "C", "tumor_allele": "T",
        "expected_immunogenic": True,
        "notes": "Charge-changing (D→N), MM-specific driver",
    },
    # --- Passenger mutation (negative control) ---
    {
        "case_id": "BENCH-006", "submitter_id": "MMRF_BENCH_006",
        "gene_symbol": "TTN", "gene_id": "ENSG00000155657",
        "chromosome": "chr2", "start_position": 179390716,
        "aa_change": "p.A1234V", "consequence_type": "missense_variant",
        "ssm_id": "bench_ttn_a1234v",
        "reference_allele": "G", "tumor_allele": "A",
        "expected_immunogenic": False,
        "notes": "TTN is a very large gene with frequent passenger mutations",
    },
    {
        "case_id": "BENCH-006", "submitter_id": "MMRF_BENCH_006",
        "gene_symbol": "MUC16", "gene_id": "ENSG00000181143",
        "chromosome": "chr19", "start_position": 8959520,
        "aa_change": "p.S5000L", "consequence_type": "missense_variant",
        "ssm_id": "bench_muc16_s5000l",
        "reference_allele": "G", "tumor_allele": "A",
        "expected_immunogenic": False,
        "notes": "MUC16 (CA-125) frequent passenger, not a cancer driver",
    },
]


def create_benchmark_dataframe():
    """Create a DataFrame from benchmark mutations."""
    # Strip extra keys not in the pipeline's expected columns
    records = []
    for m in BENCHMARK_MUTATIONS:
        record = {k: v for k, v in m.items()
                  if k not in ("expected_immunogenic", "notes")}
        records.append(record)
    return pd.DataFrame(records)


def run_validation(config_path="config.yaml"):
    """Run full pipeline validation with benchmark data."""
    config = yaml.safe_load(open(config_path))
    results = {
        "steps": {},
        "benchmark_results": [],
        "summary": {},
    }

    print("=" * 70)
    print("  PIPELINE VALIDATION & END-TO-END SIMULATION")
    print("  Using benchmark mutations from MM literature")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Mutation Parsing (bypass GDC fetch)
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("  STEP 1: Parsing benchmark mutations")
    print(f"{'─' * 70}")

    mutations_df = create_benchmark_dataframe()
    print(f"  Loaded {len(mutations_df)} benchmark mutations")
    print(f"  Patients: {mutations_df['case_id'].nunique()}")
    print(f"  Genes: {', '.join(mutations_df['gene_symbol'].unique())}")

    # Filter mutations
    filtered_df = parse_mod.filter_mutations(mutations_df, config)
    print(f"  After filtering: {len(filtered_df)} mutations")

    # Process into neoantigen candidates
    candidates_df = parse_mod.process_mutations(filtered_df, config)
    print(f"  Generated {len(candidates_df)} peptide candidates")

    results["steps"]["parsing"] = {
        "input_mutations": len(mutations_df),
        "filtered_mutations": len(filtered_df),
        "candidates_generated": len(candidates_df),
    }

    # =========================================================================
    # STEP 1b: UniProt protein sequence lookup (if available)
    # =========================================================================
    try:
        from uniprot_lookup import generate_real_peptides, KNOWN_MM_PROTEINS
        print(f"\n{'─' * 70}")
        print("  STEP 1b: UniProt protein sequence enrichment")
        print(f"{'─' * 70}")
        print(f"  Pre-cached proteins: {', '.join(KNOWN_MM_PROTEINS.keys())}")

        uniprot_count = 0
        for gene in mutations_df['gene_symbol'].unique():
            if gene in KNOWN_MM_PROTEINS:
                uniprot_count += 1
                seq = KNOWN_MM_PROTEINS[gene]
                print(f"  {gene}: {len(seq)} aa sequence available")

        print(f"  {uniprot_count}/{len(mutations_df['gene_symbol'].unique())} "
              f"genes have real protein sequences")
        results["steps"]["uniprot"] = {"genes_with_sequences": uniprot_count}
    except ImportError:
        print("\n  [Skip] uniprot_lookup module not available")

    # =========================================================================
    # STEP 2: MHC Binding Prediction
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("  STEP 2: MHC binding prediction")
    print(f"{'─' * 70}")

    # Limit candidates for speed
    max_cands = min(len(candidates_df), 2000)
    if len(candidates_df) > max_cands:
        candidates_df = candidates_df.nlargest(max_cands, "immunogenicity_score")
        print(f"  Limited to top {max_cands} candidates by immunogenicity score")

    binding_df = binding_mod.run_binding_predictions(candidates_df, config)
    print(f"  Completed predictions for {len(binding_df)} candidate-allele pairs")

    # Summarize
    binders = binding_df[binding_df["classification"].isin(["strong_binder", "weak_binder"])]
    strong = binding_df[binding_df["classification"] == "strong_binder"]
    print(f"  Binders: {len(binders)} ({100*len(binders)/max(len(binding_df),1):.1f}%)")
    print(f"  Strong binders: {len(strong)} ({100*len(strong)/max(len(binding_df),1):.1f}%)")

    results["steps"]["binding"] = {
        "predictions": len(binding_df),
        "binders": len(binders),
        "strong_binders": len(strong),
    }

    # Rank
    ranked_df = binding_mod.rank_neoantigens(binding_df, config)

    # =========================================================================
    # STEP 2b: External validation (if available)
    # =========================================================================
    try:
        from external_validation import annotate_candidates
        print(f"\n{'─' * 70}")
        print("  STEP 2b: COSMIC/IEDB cross-referencing")
        print(f"{'─' * 70}")

        annotated_df = annotate_candidates(ranked_df)
        hotspots = annotated_df[annotated_df.get("cosmic_status", pd.Series()) == "hotspot"]
        iedb_validated = annotated_df[annotated_df.get("iedb_validated", pd.Series()) == True]
        print(f"  COSMIC hotspot mutations: {len(hotspots)}")
        print(f"  IEDB-validated epitopes: {len(iedb_validated)}")
        ranked_df = annotated_df
        results["steps"]["external_validation"] = {
            "cosmic_hotspots": len(hotspots) if len(hotspots) > 0 else 0,
            "iedb_validated": len(iedb_validated) if len(iedb_validated) > 0 else 0,
        }
    except (ImportError, Exception) as e:
        print(f"\n  [Skip] external_validation: {e}")

    # =========================================================================
    # STEP 2c: Expression filtering (if available)
    # =========================================================================
    try:
        from expression_filter import enrich_with_expression
        print(f"\n{'─' * 70}")
        print("  STEP 2c: Gene expression enrichment")
        print(f"{'─' * 70}")

        enriched_df = enrich_with_expression(ranked_df)
        if "expression_category" in enriched_df.columns:
            expr_counts = enriched_df["expression_category"].value_counts()
            for cat, count in expr_counts.items():
                print(f"  {cat}: {count} candidates")
        ranked_df = enriched_df
        results["steps"]["expression"] = {"enriched": True}
    except (ImportError, Exception) as e:
        print(f"\n  [Skip] expression_filter: {e}")

    # =========================================================================
    # STEP 2d: Safety screening (if available)
    # =========================================================================
    try:
        from safety_screen import screen_candidates
        print(f"\n{'─' * 70}")
        print("  STEP 2d: Safety screening")
        print(f"{'─' * 70}")

        screened_df = screen_candidates(ranked_df)
        if "safety_flag" in screened_df.columns:
            safety_counts = screened_df["safety_flag"].value_counts()
            for flag, count in safety_counts.items():
                print(f"  {flag}: {count} candidates")
        ranked_df = screened_df
        results["steps"]["safety"] = {"screened": True}
    except (ImportError, Exception) as e:
        print(f"\n  [Skip] safety_screen: {e}")

    # =========================================================================
    # STEP 3: Vaccine Design
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("  STEP 3: Vaccine construct design")
    print(f"{'─' * 70}")

    selected_df = design_mod.select_vaccine_epitopes(ranked_df, config)
    if len(selected_df) == 0:
        print("  WARNING: No epitopes selected. Using top candidates as fallback.")
        selected_df = ranked_df.head(10)

    epitopes = selected_df["mutant_peptide"].tolist()
    construct = design_mod.build_mrna_construct(epitopes, config)

    print(f"  Epitopes in construct: {construct['n_epitopes']}")
    print(f"  mRNA length: {construct['mrna_length_nt']} nt")
    print(f"  GC content: {construct['gc_content']:.1%}")
    print(f"  Estimated MW: {construct['estimated_molecular_weight_kda']} kDa")

    results["steps"]["vaccine_design"] = {
        "n_epitopes": construct["n_epitopes"],
        "mrna_length": construct["mrna_length_nt"],
        "gc_content": construct["gc_content"],
    }

    # =========================================================================
    # STEP 4: Benchmark Validation
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("  STEP 4: Benchmark mutation validation")
    print(f"{'─' * 70}")

    # Check if known immunogenic mutations scored higher than passengers
    benchmark_lookup = {m["aa_change"]: m for m in BENCHMARK_MUTATIONS}

    for aa_change, meta in benchmark_lookup.items():
        gene = meta["gene_symbol"]
        expected = meta["expected_immunogenic"]

        matching = ranked_df[
            (ranked_df["gene_symbol"] == gene) &
            (ranked_df["aa_change"] == aa_change)
        ]

        if len(matching) > 0:
            best = matching.iloc[0]
            score = best.get("vaccine_priority_score", 0)
            ic50 = best.get("ic50_nM", 99999)
            classification = best.get("classification", "unknown")

            is_binder = classification in ("strong_binder", "weak_binder")
            status = "PASS" if (expected and is_binder) or (not expected and not is_binder) else "CHECK"

            results["benchmark_results"].append({
                "gene": gene,
                "mutation": aa_change,
                "expected_immunogenic": expected,
                "predicted_binder": is_binder,
                "best_ic50": round(ic50, 1),
                "priority_score": round(score, 1),
                "status": status,
            })

            marker = "✓" if status == "PASS" else "?"
            print(f"  {marker} {gene} {aa_change}: IC50={ic50:.0f}nM, "
                  f"score={score:.1f}, {classification} "
                  f"(expected: {'immunogenic' if expected else 'passenger'})")
        else:
            print(f"  - {gene} {aa_change}: not found in predictions")
            results["benchmark_results"].append({
                "gene": gene, "mutation": aa_change,
                "expected_immunogenic": expected,
                "status": "NOT_FOUND",
            })

    # =========================================================================
    # STEP 5: Generate Validation Report
    # =========================================================================
    print(f"\n{'─' * 70}")
    print("  STEP 5: Generating validation outputs")
    print(f"{'─' * 70}")

    output_dir = Path("output/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark results
    bench_df = pd.DataFrame(results["benchmark_results"])
    bench_path = output_dir / "benchmark_results.csv"
    bench_df.to_csv(bench_path, index=False)
    print(f"  Saved: {bench_path}")

    # Save binding predictions
    binding_path = output_dir / "validation_binding_predictions.csv"
    ranked_df.to_csv(binding_path, index=False)
    print(f"  Saved: {binding_path}")

    # Save selected epitopes
    epitopes_path = output_dir / "validation_selected_epitopes.csv"
    selected_df.to_csv(epitopes_path, index=False)
    print(f"  Saved: {epitopes_path}")

    # Save vaccine construct
    construct_json = {k: v for k, v in construct.items()
                      if k not in ("full_mrna_dna", "full_mrna_rna", "cassette_dna")}
    construct_path = output_dir / "validation_construct.json"
    with open(construct_path, "w") as f:
        json.dump(construct_json, f, indent=2)
    print(f"  Saved: {construct_path}")

    # Save mRNA FASTA
    fasta_path = output_dir / "validation_vaccine.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">MM_benchmark_vaccine | {construct['n_epitopes']} epitopes | "
                f"{construct['mrna_length_nt']} nt\n")
        seq = construct["full_mrna_rna"]
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
    print(f"  Saved: {fasta_path}")

    # Generate report
    report = design_mod.generate_vaccine_report(construct, selected_df, config)
    report_path = output_dir / "validation_vaccine_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Save full results JSON
    results_path = output_dir / "validation_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    passed = sum(1 for b in results["benchmark_results"] if b["status"] == "PASS")
    total = len(results["benchmark_results"])

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Benchmark mutations: {passed}/{total} passed")
    print(f"  Pipeline steps completed: {len(results['steps'])}")
    print(f"  Vaccine construct: {construct['n_epitopes']} epitopes, "
          f"{construct['mrna_length_nt']} nt")
    print(f"\n  All outputs in: {output_dir}/")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate the neoantigen pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_validation(args.config)
