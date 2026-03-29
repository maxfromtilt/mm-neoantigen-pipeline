"""
Shared fixtures for the MM Neoantigen Vaccine Pipeline test suite.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import yaml

# Ensure the pipeline root is importable
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

CONFIG_PATH = os.path.join(PIPELINE_ROOT, "config.yaml")


@pytest.fixture
def config():
    """Load the real pipeline config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def driver_genes(config):
    """Return the list of driver genes from config."""
    return config["mutations"]["driver_genes"]


@pytest.fixture
def peptide_lengths(config):
    """Return peptide lengths from config."""
    return config["neoantigen"]["mhc_class_i"]["peptide_lengths"]


# ── Well-known MM benchmark mutations ──────────────────────────────────────

@pytest.fixture
def kras_g12d():
    """KRAS G12D — one of the most studied immunogenic driver mutations."""
    return {
        "aa_change": "p.G12D",
        "gene_symbol": "KRAS",
        "consequence_type": "missense_variant",
        "wt_aa": "G",
        "position": 12,
        "mut_aa": "D",
    }


@pytest.fixture
def kras_g12v():
    return {
        "aa_change": "p.G12V",
        "gene_symbol": "KRAS",
        "consequence_type": "missense_variant",
        "wt_aa": "G",
        "position": 12,
        "mut_aa": "V",
    }


@pytest.fixture
def nras_q61k():
    return {
        "aa_change": "p.Q61K",
        "gene_symbol": "NRAS",
        "consequence_type": "missense_variant",
        "wt_aa": "Q",
        "position": 61,
        "mut_aa": "K",
    }


@pytest.fixture
def braf_v600e():
    """BRAF V600E — strong charge-changing mutation in a driver gene."""
    return {
        "aa_change": "p.V600E",
        "gene_symbol": "BRAF",
        "consequence_type": "missense_variant",
        "wt_aa": "V",
        "position": 600,
        "mut_aa": "E",
    }


@pytest.fixture
def tp53_r175h():
    return {
        "aa_change": "p.R175H",
        "gene_symbol": "TP53",
        "consequence_type": "missense_variant",
        "wt_aa": "R",
        "position": 175,
        "mut_aa": "H",
    }


@pytest.fixture
def passenger_mutation():
    """A random passenger mutation in a non-driver gene with conservative AA change."""
    return {
        "aa_change": "p.A100G",
        "gene_symbol": "TTN",
        "consequence_type": "missense_variant",
        "wt_aa": "A",
        "position": 100,
        "mut_aa": "G",
    }


@pytest.fixture
def benchmark_mutations(kras_g12d, kras_g12v, nras_q61k, braf_v600e, tp53_r175h):
    """All five benchmark driver mutations as a list."""
    return [kras_g12d, kras_g12v, nras_q61k, braf_v600e, tp53_r175h]


# ── Synthetic datasets ─────────────────────────────────────────────────────

@pytest.fixture
def synthetic_mutations_df():
    """
    A small synthetic mutation DataFrame suitable for end-to-end testing.
    Includes driver mutations (KRAS, BRAF, NRAS, TP53) and passengers.
    """
    rows = [
        {"case_id": "CASE-001", "submitter_id": "S001", "gene_symbol": "KRAS",
         "gene_id": "ENSG00000133703", "chromosome": "12",
         "start_position": 25398284, "aa_change": "p.G12D",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-001", "submitter_id": "S001", "gene_symbol": "BRAF",
         "gene_id": "ENSG00000157764", "chromosome": "7",
         "start_position": 140453136, "aa_change": "p.V600E",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-002", "submitter_id": "S002", "gene_symbol": "NRAS",
         "gene_id": "ENSG00000213281", "chromosome": "1",
         "start_position": 115256529, "aa_change": "p.Q61K",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-002", "submitter_id": "S002", "gene_symbol": "TP53",
         "gene_id": "ENSG00000141510", "chromosome": "17",
         "start_position": 7577121, "aa_change": "p.R175H",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-003", "submitter_id": "S003", "gene_symbol": "KRAS",
         "gene_id": "ENSG00000133703", "chromosome": "12",
         "start_position": 25398284, "aa_change": "p.G12V",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-003", "submitter_id": "S003", "gene_symbol": "TTN",
         "gene_id": "ENSG00000155657", "chromosome": "2",
         "start_position": 179390716, "aa_change": "p.A100G",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-004", "submitter_id": "S004", "gene_symbol": "MUC16",
         "gene_id": "ENSG00000181143", "chromosome": "19",
         "start_position": 8999999, "aa_change": "p.S500T",
         "consequence_type": "missense_variant"},
        {"case_id": "CASE-004", "submitter_id": "S004", "gene_symbol": "DIS3",
         "gene_id": "ENSG00000083520", "chromosome": "13",
         "start_position": 73340000, "aa_change": "p.D488N",
         "consequence_type": "missense_variant"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_epitopes():
    """A small list of peptide epitopes for vaccine design tests."""
    return [
        "DILDTAGKE",
        "VVGAVGVGK",
        "KIGDFGLATV",
        "HMTEVVRRC",
        "KLVVVGADGV",
    ]
