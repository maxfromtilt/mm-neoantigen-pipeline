"""
Comprehensive Test Suite for MM Neoantigen Vaccine Pipeline
=============================================================
Tests mutation parsing, binding prediction, vaccine design, and
validates benchmark mutations against known immunogenic properties.

Run with: cd mm-neoantigen-pipeline && python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure pipeline root is importable
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from importlib import import_module

parse_mod = import_module("02_parse_mutations")
binding_mod = import_module("03_predict_binding")
design_mod = import_module("04_design_vaccine")


# =============================================================================
# Fixtures
# =============================================================================

import yaml

@pytest.fixture
def config():
    config_path = os.path.join(PIPELINE_ROOT, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def driver_genes(config):
    return config["mutations"]["driver_genes"]


@pytest.fixture
def benchmark_mutations():
    """Well-characterized mutations for validation."""
    return pd.DataFrame([
        {"case_id": "B001", "submitter_id": "B001", "gene_symbol": "KRAS",
         "gene_id": "ENSG00000133703", "chromosome": "chr12",
         "start_position": 25245350, "aa_change": "p.G12D",
         "consequence_type": "missense_variant"},
        {"case_id": "B001", "submitter_id": "B001", "gene_symbol": "KRAS",
         "gene_id": "ENSG00000133703", "chromosome": "chr12",
         "start_position": 25245350, "aa_change": "p.G12V",
         "consequence_type": "missense_variant"},
        {"case_id": "B002", "submitter_id": "B002", "gene_symbol": "NRAS",
         "gene_id": "ENSG00000213281", "chromosome": "chr1",
         "start_position": 114713908, "aa_change": "p.Q61K",
         "consequence_type": "missense_variant"},
        {"case_id": "B003", "submitter_id": "B003", "gene_symbol": "BRAF",
         "gene_id": "ENSG00000157764", "chromosome": "chr7",
         "start_position": 140753336, "aa_change": "p.V600E",
         "consequence_type": "missense_variant"},
        {"case_id": "B004", "submitter_id": "B004", "gene_symbol": "TP53",
         "gene_id": "ENSG00000141510", "chromosome": "chr17",
         "start_position": 7675088, "aa_change": "p.R175H",
         "consequence_type": "missense_variant"},
        {"case_id": "B005", "submitter_id": "B005", "gene_symbol": "TTN",
         "gene_id": "ENSG00000155657", "chromosome": "chr2",
         "start_position": 179390716, "aa_change": "p.A1234V",
         "consequence_type": "missense_variant"},
    ])


# =============================================================================
# Mutation Parsing Tests
# =============================================================================

class TestParseAaChange:
    def test_single_letter_format(self):
        result = parse_mod.parse_aa_change("p.V600E")
        assert result == ("V", 600, "E")

    def test_single_letter_various(self):
        assert parse_mod.parse_aa_change("p.G12D") == ("G", 12, "D")
        assert parse_mod.parse_aa_change("p.Q61K") == ("Q", 61, "K")
        assert parse_mod.parse_aa_change("p.R175H") == ("R", 175, "H")

    def test_three_letter_format(self):
        result = parse_mod.parse_aa_change("p.Val600Glu")
        assert result == ("V", 600, "E")

    def test_three_letter_arg_his(self):
        result = parse_mod.parse_aa_change("p.Arg175His")
        assert result == ("R", 175, "H")

    def test_none_input(self):
        assert parse_mod.parse_aa_change(None) is None

    def test_empty_string(self):
        assert parse_mod.parse_aa_change("") is None

    def test_nan_input(self):
        assert parse_mod.parse_aa_change(float("nan")) is None

    def test_invalid_format(self):
        assert parse_mod.parse_aa_change("garbage") is None
        assert parse_mod.parse_aa_change("p.12") is None

    def test_stop_codon(self):
        result = parse_mod.parse_aa_change("p.R196*")
        assert result is not None
        assert result[2] == "*"


class TestGenerateMutantPeptides:
    def test_generates_peptides(self):
        peptides = parse_mod.generate_mutant_peptides("G", 12, "D", [9])
        assert len(peptides) > 0

    def test_correct_peptide_length(self):
        peptides = parse_mod.generate_mutant_peptides("G", 12, "D", [9])
        for p in peptides:
            assert len(p["mutant_peptide"]) == 9
            assert len(p["wildtype_peptide"]) == 9

    def test_mutation_present(self):
        peptides = parse_mod.generate_mutant_peptides("G", 12, "D", [9])
        for p in peptides:
            pos = p["mutation_position_in_peptide"]
            assert p["mutant_peptide"][pos] == "D"
            assert p["wildtype_peptide"][pos] == "G"

    def test_multiple_lengths(self):
        peptides = parse_mod.generate_mutant_peptides("V", 600, "E", [8, 9, 10, 11])
        lengths = set(p["peptide_length"] for p in peptides)
        assert lengths == {8, 9, 10, 11}

    def test_mutant_differs_from_wildtype(self):
        peptides = parse_mod.generate_mutant_peptides("G", 12, "D", [9])
        for p in peptides:
            assert p["mutant_peptide"] != p["wildtype_peptide"]


class TestImmunogenicityScore:
    def test_driver_gene_scores_higher(self, driver_genes):
        driver_score = parse_mod.calculate_immunogenicity_score(
            "G", "D", "KRAS", driver_genes, "missense_variant"
        )
        passenger_score = parse_mod.calculate_immunogenicity_score(
            "G", "D", "TTN", driver_genes, "missense_variant"
        )
        assert driver_score > passenger_score

    def test_charge_change_scores_higher(self, driver_genes):
        # R→H: charge change (positive → neutral)
        charge_score = parse_mod.calculate_immunogenicity_score(
            "R", "H", "TP53", driver_genes, "missense_variant"
        )
        # A→V: no charge change, small hydrophobicity change
        neutral_score = parse_mod.calculate_immunogenicity_score(
            "A", "V", "TTN", driver_genes, "missense_variant"
        )
        assert charge_score > neutral_score

    def test_frameshift_scores_higher_than_missense(self, driver_genes):
        frameshift = parse_mod.calculate_immunogenicity_score(
            "G", "D", "KRAS", driver_genes, "frameshift_variant"
        )
        missense = parse_mod.calculate_immunogenicity_score(
            "G", "D", "KRAS", driver_genes, "missense_variant"
        )
        assert frameshift > missense

    def test_score_capped_at_100(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "R", "D", "KRAS", driver_genes, "frameshift_variant"
        )
        assert score <= 100.0

    def test_score_non_negative(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "A", "A", "TTN", driver_genes, "missense_variant"
        )
        assert score >= 0.0


class TestFilterMutations:
    def test_filters_coding_mutations(self, config, benchmark_mutations):
        filtered = parse_mod.filter_mutations(benchmark_mutations, config)
        assert len(filtered) > 0
        assert len(filtered) <= len(benchmark_mutations)

    def test_removes_duplicates(self, config):
        df = pd.DataFrame([
            {"case_id": "C1", "gene_symbol": "KRAS", "aa_change": "p.G12D",
             "consequence_type": "missense_variant"},
            {"case_id": "C1", "gene_symbol": "KRAS", "aa_change": "p.G12D",
             "consequence_type": "missense_variant"},
        ])
        filtered = parse_mod.filter_mutations(df, config)
        assert len(filtered) == 1


# =============================================================================
# Binding Prediction Tests
# =============================================================================

class TestPredictBindingPSSM:
    def test_returns_valid_structure(self):
        result = binding_mod.predict_binding_pssm("GILGFVFTL", "HLA-A*02:01")
        assert "binding_score" in result
        assert "ic50_nM" in result
        assert "percentile_rank" in result
        assert "classification" in result

    def test_ic50_in_valid_range(self):
        result = binding_mod.predict_binding_pssm("GILGFVFTL", "HLA-A*02:01")
        assert 1 <= result["ic50_nM"] <= 50000

    def test_classification_values(self):
        # Test that classification is always one of the valid values
        result = binding_mod.predict_binding_pssm("GILGFVFTL", "HLA-A*02:01")
        assert result["classification"] in ("strong_binder", "weak_binder", "non_binder")
        # Our simplified PSSM is a rough approximation — actual binding
        # validation requires MHCflurry or NetMHCpan

    def test_anchor_preferences_a0201(self):
        # HLA-A*02:01 prefers L at P2, V at P9
        good = binding_mod.predict_binding_pssm("ALMAAAALV", "HLA-A*02:01")
        # P at P2 is not preferred for A*02:01
        bad = binding_mod.predict_binding_pssm("APMAAAALD", "HLA-A*02:01")
        assert good["ic50_nM"] < bad["ic50_nM"]

    def test_different_alleles_give_different_results(self):
        peptide = "GILGFVFTL"
        a0201 = binding_mod.predict_binding_pssm(peptide, "HLA-A*02:01")
        b0702 = binding_mod.predict_binding_pssm(peptide, "HLA-B*07:02")
        # Different alleles should give different scores
        assert a0201["binding_score"] != b0702["binding_score"]

    def test_unknown_allele_uses_default(self):
        result = binding_mod.predict_binding_pssm("GILGFVFTL", "HLA-Z*99:99")
        assert result["ic50_nM"] > 0  # Should still work with default model


class TestAgretopicity:
    def test_mutant_better_binder(self):
        # If mutant IC50=10 and wildtype IC50=100, agretopicity = 10
        agreto = binding_mod.calculate_agretopicity(10, 100)
        assert agreto == 10.0

    def test_wildtype_better_binder(self):
        agreto = binding_mod.calculate_agretopicity(100, 10)
        assert agreto == 0.1

    def test_equal_binding(self):
        agreto = binding_mod.calculate_agretopicity(50, 50)
        assert agreto == 1.0

    def test_zero_mutant_ic50(self):
        agreto = binding_mod.calculate_agretopicity(0, 100)
        assert agreto == 0.0


class TestForeignnessScore:
    def test_identical_peptides(self):
        score = binding_mod.calculate_foreignness_score("AAAAAAAAA", "AAAAAAAAA")
        assert score == 0.0

    def test_completely_different(self):
        score = binding_mod.calculate_foreignness_score("AAAAAAAAA", "RRRRRRRRR")
        assert score > 0.5

    def test_single_mutation(self):
        score = binding_mod.calculate_foreignness_score("AAGAAAAAA", "AADAAAAAA")
        assert 0 < score < 1.0

    def test_different_lengths_is_maximal(self):
        score = binding_mod.calculate_foreignness_score("AAAA", "AAAAA")
        assert score == 1.0  # Frameshift → maximally foreign


class TestRankNeoantigens:
    def test_ranking_produces_scores(self, config):
        df = pd.DataFrame({
            "ic50_nM": [10, 100, 1000],
            "agretopicity": [5.0, 2.0, 0.5],
            "immunogenicity_score": [80, 50, 20],
            "foreignness_score": [0.8, 0.5, 0.2],
            "is_driver_gene": [True, True, False],
        })
        ranked = binding_mod.rank_neoantigens(df, config)
        assert "vaccine_priority_score" in ranked.columns
        assert "rank" in ranked.columns
        # Best candidate should be rank 1
        assert ranked.iloc[0]["rank"] == 1

    def test_better_binding_ranks_higher(self, config):
        df = pd.DataFrame({
            "ic50_nM": [10, 10000],
            "agretopicity": [1.0, 1.0],
            "immunogenicity_score": [50, 50],
            "foreignness_score": [0.5, 0.5],
            "is_driver_gene": [False, False],
        })
        ranked = binding_mod.rank_neoantigens(df, config)
        # Lower IC50 (better binding) should rank higher
        assert ranked.iloc[0]["ic50_nM"] == 10


# =============================================================================
# Vaccine Design Tests
# =============================================================================

class TestCodonOptimize:
    def test_output_length(self):
        protein = "MTEYKLVVV"
        dna = design_mod.codon_optimize(protein)
        assert len(dna) == len(protein) * 3

    def test_valid_codons(self):
        protein = "MTEYKLVVV"
        dna = design_mod.codon_optimize(protein)
        for i in range(0, len(dna), 3):
            codon = dna[i:i+3]
            assert len(codon) == 3
            assert all(c in "ATCG" for c in codon)

    def test_deterministic(self):
        protein = "MTEYKLVVVGAGGVGKS"
        dna1 = design_mod.codon_optimize(protein)
        dna2 = design_mod.codon_optimize(protein)
        assert dna1 == dna2  # Same input → same output

    def test_start_codon(self):
        protein = "MVGA"
        dna = design_mod.codon_optimize(protein)
        assert dna[:3] == "ATG"  # M = ATG (only codon)


class TestBuildMrnaConstruct:
    def test_construct_structure(self, config):
        epitopes = ["GILGFVFTL", "NLVPMVATV", "GLCTLVAML"]
        construct = design_mod.build_mrna_construct(epitopes, config)

        assert construct["n_epitopes"] == 3
        assert construct["mrna_length_nt"] > 0
        assert 0 < construct["gc_content"] < 1.0
        assert construct["estimated_molecular_weight_da"] > 0

    def test_contains_utr_and_polya(self, config):
        epitopes = ["GILGFVFTL", "NLVPMVATV"]
        construct = design_mod.build_mrna_construct(epitopes, config)

        mrna = construct["full_mrna_dna"]
        assert mrna.endswith("A" * config["vaccine"]["poly_a_length"])

    def test_contains_signal_peptide(self, config):
        epitopes = ["GILGFVFTL"]
        construct = design_mod.build_mrna_construct(epitopes, config)
        assert config["vaccine"]["signal_peptide"] in construct["cassette_protein"]

    def test_rna_conversion(self, config):
        epitopes = ["GILGFVFTL"]
        construct = design_mod.build_mrna_construct(epitopes, config)
        rna = construct["full_mrna_rna"]
        assert "T" not in rna  # All T should be U
        assert "U" in rna


class TestOptimizeEpitopeOrder:
    def test_valid_permutation(self):
        epitopes = ["AAAA", "BBBB", "CCCC"]
        order = design_mod.optimize_epitope_order(epitopes, "GGSGGGGSGG")
        assert sorted(order) == [0, 1, 2]

    def test_single_epitope(self):
        order = design_mod.optimize_epitope_order(["AAAA"], "GGS")
        assert order == [0]

    def test_two_epitopes(self):
        order = design_mod.optimize_epitope_order(["AAAA", "BBBB"], "GGS")
        assert sorted(order) == [0, 1]


class TestSelectVaccineEpitopes:
    def test_respects_max_limit(self, config):
        # Create a large candidate set
        rows = []
        for i in range(100):
            rows.append({
                "gene_symbol": f"GENE{i % 20}",
                "aa_change": f"p.A{i}V",
                "mutant_peptide": f"AAAAAA{chr(65 + i%26)}AA",
                "hla_allele": "HLA-A*02:01",
                "classification": "strong_binder",
                "vaccine_priority_score": 90 - i,
                "ic50_nM": 20 + i,
            })
        df = pd.DataFrame(rows)
        selected = design_mod.select_vaccine_epitopes(df, config)
        assert len(selected) <= config["vaccine"]["max_neoantigens"]

    def test_gene_diversity(self, config):
        # All from same gene — should limit to 3
        rows = []
        for i in range(20):
            rows.append({
                "gene_symbol": "KRAS",
                "aa_change": f"p.G{i}D",
                "mutant_peptide": f"AAAAAA{chr(65 + i%26)}AA",
                "hla_allele": "HLA-A*02:01",
                "classification": "strong_binder",
                "vaccine_priority_score": 90 - i,
                "ic50_nM": 20 + i,
            })
        df = pd.DataFrame(rows)
        selected = design_mod.select_vaccine_epitopes(df, config)
        assert len(selected) <= 3


# =============================================================================
# Benchmark Mutation Validation Tests
# =============================================================================

class TestBenchmarkMutations:
    """
    Validate that known immunogenic mutations from MM literature
    are handled correctly by the pipeline.
    """

    def test_kras_g12d_is_driver(self, driver_genes):
        assert "KRAS" in driver_genes

    def test_kras_g12d_high_immunogenicity(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "G", "D", "KRAS", driver_genes, "missense_variant"
        )
        # G→D is a significant change (hydrophobic→charged)
        assert score >= 30

    def test_braf_v600e_high_immunogenicity(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "V", "E", "BRAF", driver_genes, "missense_variant"
        )
        # V→E is hydrophobic→charged, very immunogenic
        assert score >= 40

    def test_tp53_r175h_charge_change(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "R", "H", "TP53", driver_genes, "missense_variant"
        )
        # R→H: positive charge → neutral, significant change
        assert score >= 30

    def test_nras_q61k_charge_change(self, driver_genes):
        score = parse_mod.calculate_immunogenicity_score(
            "Q", "K", "NRAS", driver_genes, "missense_variant"
        )
        # Q→K: neutral → positive charge
        assert score >= 30

    def test_passenger_scores_lower(self, driver_genes):
        """TTN A→V (passenger) should score lower than KRAS G→D (driver)."""
        driver_score = parse_mod.calculate_immunogenicity_score(
            "G", "D", "KRAS", driver_genes, "missense_variant"
        )
        passenger_score = parse_mod.calculate_immunogenicity_score(
            "A", "V", "TTN", driver_genes, "missense_variant"
        )
        assert driver_score > passenger_score

    def test_driver_genes_in_benchmark_are_recognized(self, driver_genes):
        """All MM driver genes should be in the config."""
        for gene in ["KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C"]:
            assert gene in driver_genes, f"{gene} not in driver_genes"


# =============================================================================
# Integration Test
# =============================================================================

class TestEndToEndIntegration:
    """
    Run the full pipeline on benchmark mutations and validate output.
    """

    def test_full_pipeline(self, config, benchmark_mutations):
        # Step 1: Filter mutations
        filtered = parse_mod.filter_mutations(benchmark_mutations, config)
        assert len(filtered) > 0

        # Step 2: Generate candidates
        candidates = parse_mod.process_mutations(filtered, config)
        assert len(candidates) > 0
        assert "mutant_peptide" in candidates.columns
        assert "immunogenicity_score" in candidates.columns

        # Step 3: Predict binding (limit for speed)
        candidates_subset = candidates.head(500)
        binding_results = binding_mod.run_binding_predictions(candidates_subset, config)
        assert len(binding_results) > 0
        assert "ic50_nM" in binding_results.columns

        # Step 4: Rank
        ranked = binding_mod.rank_neoantigens(binding_results, config)
        assert "vaccine_priority_score" in ranked.columns
        assert ranked.iloc[0]["vaccine_priority_score"] >= ranked.iloc[-1]["vaccine_priority_score"]

        # Step 5: Select epitopes and build vaccine
        selected = design_mod.select_vaccine_epitopes(ranked, config)
        if len(selected) > 0:
            epitopes = selected["mutant_peptide"].tolist()
            construct = design_mod.build_mrna_construct(epitopes, config)
            assert construct["n_epitopes"] > 0
            assert construct["mrna_length_nt"] > 100
            assert 0.3 < construct["gc_content"] < 0.8

    def test_kras_mutations_produce_binders(self, config):
        """KRAS G12D and G12V should produce at least some binders."""
        mutations = pd.DataFrame([
            {"case_id": "T1", "submitter_id": "T1", "gene_symbol": "KRAS",
             "gene_id": "E1", "chromosome": "chr12", "start_position": 1,
             "aa_change": "p.G12D", "consequence_type": "missense_variant"},
            {"case_id": "T1", "submitter_id": "T1", "gene_symbol": "KRAS",
             "gene_id": "E1", "chromosome": "chr12", "start_position": 1,
             "aa_change": "p.G12V", "consequence_type": "missense_variant"},
        ])

        filtered = parse_mod.filter_mutations(mutations, config)
        candidates = parse_mod.process_mutations(filtered, config)
        assert len(candidates) > 0

        results = binding_mod.run_binding_predictions(candidates, config)
        binders = results[results["classification"].isin(["strong_binder", "weak_binder"])]
        assert len(binders) > 0, "KRAS mutations should produce at least some binders"


# =============================================================================
# External Validation Module Tests (if available)
# =============================================================================

class TestExternalValidation:
    @pytest.fixture(autouse=True)
    def _check_module(self):
        try:
            self.mod = import_module("external_validation")
        except ImportError:
            pytest.skip("external_validation module not available")

    def test_cosmic_hotspot_lookup(self):
        result = self.mod.check_cosmic_status("KRAS", "p.G12D")
        assert result is not None
        assert result["evidence_level"] == "validated"

    def test_cosmic_novel_mutation(self):
        result = self.mod.check_cosmic_status("FAKE_GENE", "p.X999Y")
        assert result is None

    def test_iedb_known_peptide(self):
        matches = self.mod.check_iedb_status("KLVVVGADGV")
        assert len(matches) > 0

    def test_mm_context(self):
        ctx = self.mod.get_mm_context("KRAS")
        assert ctx is not None
        assert ctx["frequency"] == 0.23

    def test_annotate_adds_columns(self):
        df = pd.DataFrame({
            "gene_symbol": ["KRAS", "TTN"],
            "aa_change": ["p.G12D", "p.A1234V"],
            "mutant_peptide": ["KLVVVGADGV", "AAAAAAAAA"],
            "ic50_nM": [50, 5000],
        })
        annotated = self.mod.annotate_candidates(df)
        assert "cosmic_status" in annotated.columns
        assert "confidence_score" in annotated.columns
        # KRAS G12D should have higher confidence than TTN passenger
        kras_conf = annotated[annotated["gene_symbol"] == "KRAS"]["confidence_score"].iloc[0]
        ttn_conf = annotated[annotated["gene_symbol"] == "TTN"]["confidence_score"].iloc[0]
        assert kras_conf > ttn_conf


# =============================================================================
# Safety Screen Module Tests (if available)
# =============================================================================

class TestSafetyScreen:
    @pytest.fixture(autouse=True)
    def _check_module(self):
        try:
            self.mod = import_module("safety_screen")
        except ImportError:
            pytest.skip("safety_screen module not available")

    def test_identical_self_peptide_flagged(self):
        # GILGFVFTL is in the self-peptide database (flu M1)
        hits = self.mod.screen_self_similarity("GILGFVFTL", threshold=0.9)
        assert len(hits) > 0

    def test_novel_peptide_passes(self):
        hits = self.mod.screen_self_similarity("XYZXYZXYZ", threshold=0.9)
        assert len(hits) == 0

    def test_autoimmune_risk_kras(self):
        risk = self.mod.check_known_autoimmune_targets("KRAS")
        assert risk["autoimmune_risk"] == "low"

    def test_screen_candidates_adds_columns(self):
        df = pd.DataFrame({
            "mutant_peptide": ["GILGFVFTL", "XYZXYZXYZ"],
            "gene_symbol": ["KRAS", "TP53"],
        })
        screened = self.mod.screen_candidates(df)
        assert "safety_flag" in screened.columns
        assert "self_similarity_max" in screened.columns


# =============================================================================
# UniProt Module Tests (if available)
# =============================================================================

class TestUniProtLookup:
    @pytest.fixture(autouse=True)
    def _check_module(self):
        try:
            self.mod = import_module("uniprot_lookup")
        except ImportError:
            pytest.skip("uniprot_lookup module not available")

    def test_known_kras_sequence(self):
        seq = self.mod.KNOWN_MM_PROTEINS.get("KRAS")
        assert seq is not None
        assert len(seq) > 100
        # KRAS starts with MTEYKLVVV
        assert seq.startswith("MTEYKLVVV")

    def test_kras_g12_is_glycine(self):
        seq = self.mod.KNOWN_MM_PROTEINS["KRAS"]
        # Position 12 (1-indexed) = index 11
        assert seq[11] == "G", f"Expected G at position 12, got {seq[11]}"

    def test_validate_mutation(self):
        assert self.mod.validate_mutation("KRAS", 12, "G") == True

    def test_generate_real_peptides(self):
        peptides = self.mod.generate_real_peptides("KRAS", "G", 12, "D", [9])
        assert len(peptides) > 0
        for p in peptides:
            assert p["sequence_source"] == "uniprot"
            assert len(p["mutant_peptide"]) == 9
