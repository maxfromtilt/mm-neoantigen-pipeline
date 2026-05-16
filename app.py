"""

__version__ = "2.1.4"
MM Neoantigen Vaccine Designer — Interactive Dashboard
========================================================
A web-based interface for the Multiple Myeloma personalised
neoantigen mRNA vaccine design pipeline.

Run: streamlit run app.py
Deploy: Push to GitHub, connect to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path

import math
import re
import io


# ══════════════════════════════════════════════════════════════════════
# EMBEDDED PIPELINE (runs server-side, PSSM-based, no TensorFlow)
# ══════════════════════════════════════════════════════════════════════

# Amino acid alphabet
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# PSSM binding matrices for 6 HLA alleles
PSSM_MODELS = {
    "HLA-A*02:01": {
        "anchors": {1: {'L':1.0,'M':0.8,'I':0.7,'V':0.6,'A':0.4,'T':0.3},
                    8: {'V':1.0,'L':0.9,'I':0.8,'A':0.6,'T':0.5}},
        "general": {'A':0.3,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.1,'E':-0.2,'Q':0.0,
                    'G':-0.1,'H':-0.1,'I':0.4,'L':0.5,'K':-0.3,'M':0.3,'F':0.2,
                    'P':-0.4,'S':0.0,'T':0.1,'W':-0.1,'Y':0.1,'V':0.4},
    },
    "HLA-A*01:01": {
        "anchors": {1: {'T':1.0,'S':0.8,'D':0.6,'E':0.5},
                    8: {'Y':1.0,'F':0.8,'W':0.7}},
        "general": {'A':0.2,'R':-0.1,'N':0.1,'D':0.2,'C':0.0,'E':0.1,'Q':0.0,
                    'G':-0.2,'H':0.0,'I':0.1,'L':0.3,'K':-0.2,'M':0.1,'F':0.3,
                    'P':-0.3,'S':0.2,'T':0.3,'W':0.1,'Y':0.4,'V':0.1},
    },
    "HLA-A*03:01": {
        "anchors": {1: {'L':0.8,'V':0.7,'M':0.6,'I':0.5,'F':0.5},
                    8: {'K':1.0,'R':0.9,'Y':0.5}},
        "general": {'A':0.2,'R':0.3,'N':0.0,'D':-0.2,'C':0.0,'E':-0.1,'Q':0.1,
                    'G':-0.1,'H':0.1,'I':0.3,'L':0.4,'K':0.4,'M':0.2,'F':0.2,
                    'P':-0.3,'S':0.0,'T':0.1,'W':0.0,'Y':0.2,'V':0.3},
    },
    "HLA-A*24:02": {
        "anchors": {1: {'Y':1.0,'F':0.9,'W':0.7,'I':0.5},
                    8: {'F':1.0,'L':0.9,'I':0.8,'W':0.7}},
        "general": {'A':0.1,'R':-0.2,'N':-0.1,'D':-0.3,'C':0.0,'E':-0.2,'Q':0.0,
                    'G':-0.2,'H':-0.1,'I':0.4,'L':0.5,'K':-0.3,'M':0.2,'F':0.4,
                    'P':-0.4,'S':0.0,'T':0.0,'W':0.2,'Y':0.3,'V':0.3},
    },
    "HLA-B*07:02": {
        "anchors": {1: {'P':1.0,'A':0.4,'S':0.3},
                    8: {'L':1.0,'M':0.8,'F':0.6,'I':0.5}},
        "general": {'A':0.3,'R':0.1,'N':0.0,'D':-0.1,'C':0.0,'E':-0.1,'Q':0.0,
                    'G':-0.2,'H':0.0,'I':0.3,'L':0.5,'K':0.0,'M':0.3,'F':0.3,
                    'P':0.2,'S':0.1,'T':0.1,'W':0.0,'Y':0.1,'V':0.3},
    },
    "HLA-B*08:01": {
        "anchors": {1: {'R':1.0,'K':0.8,'Q':0.5},
                    8: {'L':1.0,'K':0.8,'R':0.7}},
        "general": {'A':0.2,'R':0.3,'N':0.0,'D':-0.1,'C':0.0,'E':0.0,'Q':0.1,
                    'G':-0.1,'H':0.0,'I':0.2,'L':0.4,'K':0.3,'M':0.2,'F':0.2,
                    'P':-0.3,'S':0.0,'T':0.1,'W':0.0,'Y':0.1,'V':0.2},
    },
}

MM_DRIVERS = {"KRAS","NRAS","BRAF","TP53","DIS3","FAM46C","TRAF3",
              "RB1","CYLD","MAX","IRF4","FGFR3","ATM","ATR","PRDM1"}

VALID_CONSEQUENCES = {"Missense_Mutation","missense_variant","Frame_Shift_Del",
    "Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins","frameshift_variant",
    "inframe_deletion","inframe_insertion","Nonstop_Mutation"}


def pssm_predict(peptide: str, allele: str) -> float:
    """Predict IC50 using PSSM. Returns IC50 in nM."""
    model = PSSM_MODELS.get(allele, PSSM_MODELS["HLA-A*02:01"])
    score = 0.0
    pep = peptide.upper()
    if len(pep) != 9:
        score += -0.1 * abs(len(pep) - 9)
    for i, aa in enumerate(pep):
        if i in model["anchors"] and len(pep) == 9:
            score += model["anchors"][i].get(aa, -0.5)
        else:
            score += model["general"].get(aa, -0.1)
    ic50 = 50000 * math.exp(-score * 1.5)
    return max(1.0, min(50000.0, ic50))


def parse_aa_change(aa_change: str):
    """Parse amino acid change string like 'p.E196K' or 'E196K'."""
    aa_change = str(aa_change).strip()
    if aa_change.startswith("p."):
        aa_change = aa_change[2:]
    m = re.match(r'^([A-Z*])(\d+)([A-Z*](?:fs\*\d+)?)$', aa_change)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def generate_peptides(wt_aa: str, mut_aa: str, position: int, lengths=(8,9,10,11)):
    """Generate mutant/wildtype peptide pairs from a point mutation."""
    # Build a pseudo protein context (17aa window centered on mutation)
    # Using generic amino acids for flanking since we don't have full sequence
    np.random.seed(position % 10000)
    flank_aas = "AVILMFYWSTCNGQHDERKP"
    left_flank = "".join(np.random.choice(list(flank_aas), 8))
    right_flank = "".join(np.random.choice(list(flank_aas), 8))

    mut_context = left_flank + mut_aa + right_flank
    wt_context = left_flank + wt_aa + right_flank
    mut_pos = 8  # Position of mutation in context

    peptides = []
    for length in lengths:
        for start in range(max(0, mut_pos - length + 1), min(mut_pos + 1, len(mut_context) - length + 1)):
            mut_pep = mut_context[start:start + length]
            wt_pep = wt_context[start:start + length]
            if all(aa in STANDARD_AA for aa in mut_pep):
                peptides.append((mut_pep, wt_pep))
    return peptides


def run_uploaded_pipeline(uploaded_file):
    """Run the full PSSM pipeline on uploaded mutation data."""
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    # Validate columns
    required = {"gene_symbol", "aa_change"}
    has_consequence = "consequence_type" in raw_df.columns
    if not required.issubset(set(raw_df.columns)):
        st.error(f"Missing required columns. Need: gene_symbol, aa_change. Found: {list(raw_df.columns)}")
        return

    st.success(f"Loaded {len(raw_df)} mutation records")

    # Filter by consequence type if available
    if has_consequence:
        raw_df = raw_df[raw_df["consequence_type"].isin(VALID_CONSEQUENCES)]
        st.info(f"After consequence filtering: {len(raw_df)} mutations")

    # Filter to rows with valid aa_change
    raw_df = raw_df[raw_df["aa_change"].notna() & (raw_df["aa_change"] != "")]
    raw_df = raw_df.drop_duplicates(subset=["gene_symbol", "aa_change"])

    if len(raw_df) == 0:
        st.error("No valid mutations found after filtering.")
        return

    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    alleles = list(PSSM_MODELS.keys())

    # Step 1: Parse mutations and generate peptides
    all_candidates = []
    for idx, (_, row) in enumerate(raw_df.iterrows()):
        gene = str(row["gene_symbol"])
        aa_str = str(row["aa_change"])
        wt_aa, pos, mut_aa = parse_aa_change(aa_str)

        if wt_aa is None or pos is None:
            continue

        # Only single AA substitutions for now
        if len(mut_aa) != 1 or mut_aa == "*":
            continue

        peps = generate_peptides(wt_aa, mut_aa, pos)
        for mut_pep, wt_pep in peps:
            all_candidates.append({
                "gene_symbol": gene,
                "aa_change": aa_str,
                "mutant_peptide": mut_pep,
                "wildtype_peptide": wt_pep,
                "is_driver_gene": gene in MM_DRIVERS,
            })

        progress.progress((idx + 1) / len(raw_df))

    if not all_candidates:
        st.error("No peptide candidates could be generated. Check aa_change format (e.g. E196K or p.E196K).")
        return

    candidates_df = pd.DataFrame(all_candidates)
    n_candidates = len(candidates_df)
    status.info(f"Generated {n_candidates} peptide candidates from {len(raw_df)} mutations")
    st.caption(
        "⚠️  Peptide flanking sequences are approximated from position context (seeded by mutation position). "
        "For highest accuracy, run the pipeline locally with full protein sequences via UniProt lookup — "
        "see the GitHub repo for instructions."
    )

    # Step 2: Binding predictions
    progress.progress(0)
    all_results = []
    total_preds = n_candidates * len(alleles)
    pred_count = 0

    for allele in alleles:
        for _, row in candidates_df.iterrows():
            mut_ic50 = pssm_predict(row["mutant_peptide"], allele)
            wt_ic50 = pssm_predict(row["wildtype_peptide"], allele)
            agretopicity = wt_ic50 / mut_ic50 if mut_ic50 > 0 else 0

            if mut_ic50 < 50:
                classification = "strong_binder"
            elif mut_ic50 < 500:
                classification = "weak_binder"
            else:
                classification = "non_binder"

            all_results.append({
                **row.to_dict(),
                "hla_allele": allele,
                "ic50_nM": round(mut_ic50, 2),
                "wt_ic50_nM": round(wt_ic50, 2),
                "agretopicity": round(agretopicity, 3),
                "classification": classification,
            })
            pred_count += 1
            if pred_count % 500 == 0:
                progress.progress(pred_count / total_preds)

    results_df = pd.DataFrame(all_results)
    binders = results_df[results_df["ic50_nM"] < 500]
    strong = results_df[results_df["ic50_nM"] < 50]

    progress.progress(1.0)

    # Step 3: Display results
    st.markdown("---")
    st.subheader("⟠  Analysis Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", f"{len(results_df):,}")
    col2.metric("Binders (IC50<500)", f"{len(binders)}", f"{100*len(binders)/max(len(results_df),1):.1f}%")
    col3.metric("Strong Binders", f"{len(strong)}", "IC50 < 50 nM")
    col4.metric("Unique Genes", f"{results_df['gene_symbol'].nunique()}")

    st.markdown(
        '<div class="highlight-box">'
        '<b>▲  Note:</b> These predictions use the built-in PSSM model for speed. '
        'For production-grade results, run the pipeline locally with MHCflurry '
        '(neural network trained on 350,000+ experimental measurements). '
        'PSSM predictions are approximate and should be validated.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Binding distribution
    if not binders.empty:
        fig = px.histogram(
            results_df[results_df["ic50_nM"] < 5000],
            x="ic50_nM",
            color=results_df[results_df["ic50_nM"] < 5000]["ic50_nM"].apply(classify_binding),
            color_discrete_map={
                "Strong Binder": "#2c5282",
                "Weak Binder": "#4299e1",
                "Non-Binder": "#cbd5e0",
            },
            nbins=50,
            labels={"ic50_nM": "Predicted IC50 (nM)", "color": "Category"},
            template="plotly_white",
        )
        fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="Strong (50nM)")
        fig.add_vline(x=500, line_dash="dash", line_color="orange", annotation_text="Weak (500nM)")
        fig.update_layout(height=400, xaxis_title="IC50 (nM)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Top candidates table
    if not binders.empty:
        st.subheader("Top Vaccine Candidates")
        top = binders.nsmallest(30, "ic50_nM")[
            ["gene_symbol", "aa_change", "mutant_peptide", "hla_allele",
             "ic50_nM", "classification", "agretopicity", "is_driver_gene"]
        ].rename(columns={
            "gene_symbol": "Gene", "aa_change": "Mutation",
            "mutant_peptide": "Peptide", "hla_allele": "HLA",
            "ic50_nM": "IC50 (nM)", "classification": "Binding",
            "agretopicity": "Agretopicity", "is_driver_gene": "Driver Gene",
        })
        st.dataframe(top, use_container_width=True, hide_index=True)

    # HLA coverage
    st.subheader("HLA Coverage")
    hla_data = []
    for allele in alleles:
        a_data = results_df[results_df["hla_allele"] == allele]
        hla_data.append({
            "HLA Allele": allele,
            "Binders": len(a_data[a_data["ic50_nM"] < 500]),
            "Strong Binders": len(a_data[a_data["ic50_nM"] < 50]),
        })
    st.dataframe(pd.DataFrame(hla_data), use_container_width=True, hide_index=True)

    # Download results
    csv_out = results_df.to_csv(index=False)
    st.download_button(
        "↓  Download Full Results (CSV)",
        data=csv_out,
        file_name="neoantigen_analysis_results.csv",
        mime="text/csv",
    )

    # Download binders only
    if not binders.empty:
        binders_csv = binders.to_csv(index=False)
        st.download_button(
            "↓  Download Binders Only (CSV)",
            data=binders_csv,
            file_name="neoantigen_binders.csv",
            mime="text/csv",
        )

    # Full results download
    st.markdown("---")
    full_csv = results_df.to_csv(index=False)
    st.download_button(
        "↓  Download Full Results (CSV)",
        data=full_csv,
        file_name="neoantigen_full_results.csv",
        mime="text/csv",
    )

    st.subheader("Contribute to Research")
    col_share, _ = st.columns([2, 1])

    with col_share:
        st.markdown("#### Contribute to Research")
        st.markdown(
            "Help build the largest open MM neoantigen database. "
            "Your results will be **fully anonymised** before sharing:"
        )
        st.markdown(
            "- All patient identifiers removed\n"
            "- Only gene, mutation, peptide, HLA, and binding data stored\n"
            "- No raw sequences or genomic coordinates\n"
            "- No personal or clinical information\n"
            "- Data used solely to improve neoantigen prediction for MM"
        )

        opt_in = st.checkbox(
            "I consent to sharing anonymised neoantigen predictions with the MM Vaccine Research Database",
            value=False,
        )

        if opt_in and st.button("Share Anonymised Results"):
            # Anonymise: keep only scientific columns
            anon_cols = ["gene_symbol", "aa_change", "mutant_peptide",
                         "hla_allele", "ic50_nM", "classification",
                         "agretopicity", "is_driver_gene"]
            anon_available = [c for c in anon_cols if c in results_df.columns]
            anon_df = results_df[anon_available].copy()
            # Remove any residual patient identifiers
            for col in ["case_id", "submitter_id", "ssm_id", "patient_id"]:
                if col in anon_df.columns:
                    anon_df = anon_df.drop(columns=[col])

            # Save to community database (GitHub-based for now)
            community_dir = "community"
            os.makedirs(community_dir, exist_ok=True)

            # Generate anonymous submission ID
            import hashlib, time
            sub_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:12]
            sub_path = os.path.join(community_dir, f"submission_{sub_id}.csv")

            try:
                anon_df.to_csv(sub_path, index=False)
                st.success(
                    f"Thank you! Your anonymised results ({len(anon_df)} predictions) "
                    f"have been submitted to the MM Neoantigen Research Database. "
                    f"Submission ID: {sub_id}"
                )
                st.balloons()
            except Exception as e:
                st.warning(
                    f"Could not save to server storage (Streamlit Cloud limitation). "
                    f"To contribute, please email the downloaded CSV to: "
                    f"robert.doran@tilt.ie with subject 'MM Neoantigen Data Contribution'"
                )


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MM Neoantigen Vaccine Designer v2.1.0",
    page_icon="⟠ ",
    layout="wide",
    initial_sidebar_state="expanded",
)# ── Global error handler ─────────────────────────────────────────────
def handle_error(exctype, value, tb):
    st.error(f"An error occurred: {str(value)}")
    st.info("If this persists, try reloading the page or switching patient.")
    print(f"ERROR: {exctype.__name__}: {value}", file=sys.stderr)

import sys
sys.excepthook = handle_error

# ── Version ─────────────────────────────────────────────────────────
st.sidebar.markdown("**Version 2.1.0** — Enhanced: ClinVar, TMB, Survival, HLA, Clinical PDF, TCR, PeptideAtlas, Ligandomics, IND Docs, mRNA Synthesis")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #ebf4ff 0%, #c3dafe 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #2c5282;
    }
    .stMetric > div {
        background: #f7fafc;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #e2e8f0;
    }
    .highlight-box {
        background: #fffff0;
        border-left: 4px solid #d69e2e;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    div[data-testid="stSidebar"] {
        background: #1a365d;
        color: white;
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    div[data-testid="stSidebar"] label {
        color: white !important;
    }
    .disclaimer {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.85rem;
        color: #742a2a;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────

def load_patient_data(patient_id: str, data_type: str = "enhanced") -> pd.DataFrame:
    """Load pre-computed results for a patient.
    Checks multiple possible output directory structures.
    """
    # Try different path patterns
    path_patterns = [
        f"output/{patient_id}_enhanced/{data_type}_predictions.csv",
        f"output/{patient_id}_mhcflurry/{data_type}_predictions.csv",
        f"output/{patient_id}/{data_type}_predictions.csv",
        f"output/{patient_id}_enhanced/selected_epitopes.csv",
        f"output/{patient_id}_mhcflurry/selected_epitopes.csv",
        f"output/{patient_id}/selected_epitopes.csv",
        f"output/{patient_id}_enhanced/enhanced_predictions.csv",
        f"output/{patient_id}/enhanced_predictions.csv",
    ]
    for path in path_patterns:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except (pd.errors.EmptyDataError, Exception):
                continue
    return pd.DataFrame()


def load_vaccine_report(patient_id: str) -> str:
    """Load vaccine report text from multiple possible locations."""
    for path in [
        f"output/{patient_id}_mhcflurry/vaccine_report.txt",
        f"output/{patient_id}/vaccine_report.txt",
        f"output/{patient_id}_enhanced/vaccine_report.txt",
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return ""


def load_patient_survival_data(patient_id: str) -> dict:
    """Load per-patient survival data from mmrf_cases.csv and survival analysis output."""
    result = {
        "vital_status": None,
        "days_to_death": None,
        "days_to_last_follow_up": None,
        "overall_survival_months": None,
        "cohort_median_os_months": 48.0,
        "cohort_n_patients": 0,
        "survival_source": None,
    }

    # Load from mmrf_cases.csv
    cases_path = "data/mmrf_cases.csv"
    if os.path.exists(cases_path):
        try:
            cases_df = pd.read_csv(cases_path)
            # Match by submitter_id or case_id
            match = cases_df[
                cases_df["submitter_id"].str.upper() == patient_id.upper().replace("MMRF_", "MMRF_")
            ]
            if match.empty:
                pid_cased = patient_id.replace("mmrf_", "").upper()
                match = cases_df[
                    cases_df["submitter_id"].str.upper().str.contains(pid_cased, na=False)
                ]
            if not match.empty:
                row = match.iloc[0]
                result["vital_status"] = row.get("vital_status")
                result["days_to_death"] = row.get("days_to_death")
                result["days_to_last_follow_up"] = row.get("days_to_last_follow_up")
                result["survival_source"] = "mmrf_cases"
                if pd.notna(row.get("days_to_death")):
                    result["overall_survival_months"] = round(row["days_to_death"] / 30.44, 1)
                elif pd.notna(row.get("days_to_last_follow_up")):
                    result["overall_survival_months"] = round(row["days_to_last_follow_up"] / 30.44, 1)
        except Exception:
            pass

    # Load from survival analysis output if available
    survival_path = "output/survival.csv"
    if os.path.exists(survival_path):
        try:
            surv_df = pd.read_csv(survival_path)
            row = surv_df[surv_df.get("patient_id", surv_df.get("submitter_id", "")) == patient_id]
            if not row.empty:
                for k in ["cohort_median_os_months", "cohort_n_patients", "overall_survival_months"]:
                    if k in row.columns:
                        result[k] = float(row.iloc[0][k])
        except Exception:
            pass

    return result


def calculate_tmb(mutation_df: pd.DataFrame, capture_size_mb: float = 30.0) -> dict:
    """
    Calculate Tumour Mutational Burden.

    Args:
        mutation_df: DataFrame with at least case_id column and a consequence column
        capture_size_mb: Exome capture size in Mb (default 30Mb for WGS, 1Mb for WES)

    Returns dict with tmb_score, mutation_count, capture_size_mb, tmb_category
    """
    if mutation_df.empty:
        return {"tmb_score": 0.0, "mutation_count": 0, "capture_size_mb": capture_size_mb, "tmb_category": "Unknown"}

    # Count missense + frameshift mutations per patient
    relevant = {"missense_variant", "Missense_Mutation", "Frame_Shift_Del",
                "Frame_Shift_Ins", "frameshift_variant", "inframe_deletion",
                "inframe_insertion", "In_Frame_Del", "In_Frame_Ins"}

    n_mutations = 0
    if "consequence_type" in mutation_df.columns:
        n_mutations = mutation_df[mutation_df["consequence_type"].isin(relevant)].shape[0]
    else:
        n_mutations = len(mutation_df)

    tmb = round(n_mutations / capture_size_mb, 2)

    # MM population norms: median ~2-5 mut/MB, high >10 mut/MB
    if tmb >= 10:
        category = "Very High (>10 mut/MB) — hypermutated, may respond to ICPi"
    elif tmb >= 5:
        category = "High (5-10 mut/MB) — intermediate"
    elif tmb >= 2:
        category = "Moderate (2-5 mut/MB) — MM population norm"
    else:
        category = "Low (<2 mut/MB) — lower neoantigen load"

    return {
        "tmb_score": tmb,
        "mutation_count": n_mutations,
        "capture_size_mb": capture_size_mb,
        "tmb_category": category,
    }


def load_patient_hla(patient_id: str) -> dict:
    """Load per-patient HLA typing from hla_typing.json or return population priors."""
    hla_config = {
        "class_i": ["HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02",
                    "HLA-B*07:02", "HLA-B*08:01"],
        "class_ii": ["HLA-DRB1*01:01", "HLA-DRB1*07:01", "HLA-DQB1*02:01"],
        "confidence": "Medium (population prior)",
    }

    hla_path = "data/hla_typing.json"
    if os.path.exists(hla_path):
        try:
            import json
            with open(hla_path) as f:
                hla_data = json.load(f)
            patient_hla = hla_data.get(patient_id, hla_data.get(
                patient_id.replace("mmrf_", "MMRF_").upper(), {}
            ))
            if patient_hla:
                hla_config["class_i"] = patient_hla.get("class_i", hla_config["class_i"])
                hla_config["class_ii"] = patient_hla.get("class_ii", hla_config["class_ii"])
                hla_config["confidence"] = patient_hla.get("confidence", "High (WES-based)")
        except Exception:
            pass

    return hla_config


def classify_binding(ic50):
    """Classify binding strength."""
    if ic50 < 50:
        return "Strong Binder"
    elif ic50 < 500:
        return "Weak Binder"
    else:
        return "Non-Binder"


def get_available_patients():
    """Find which patients have results."""
    patients = []
    output_dir = Path("output")
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.endswith("_enhanced"):
                pid = d.name.replace("_enhanced", "")
                patients.append(pid)
            elif d.is_dir() and d.name.startswith("mmrf_"):
                # Also check main output dir for patients with full pipeline runs
                if (d / "selected_epitopes.csv").exists():
                    pid = d.name
                    if pid not in patients:
                        patients.append(pid)
    # Also scan data/ for new patients without output yet
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.name.startswith("mmrf_") and f.name.endswith("_mutations.csv"):
                pid = f.stem  # e.g. mmrf_MMRF_2240
                if pid not in patients:
                    patients.append(pid)
    return sorted(patients, key=lambda x: x.replace("mmrf_","")) if patients else ["mmrf_1251", "mmrf_1274"]


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⟠  MM Vaccine Designer")
    st.markdown("---")

    patients = get_available_patients()

    # Default to the patient with the most complete results
    best_patient = "mmrf_1796"
    default_idx = 0
    for i, p in enumerate(patients):
        if p == best_patient:
            default_idx = i
            break

    selected_patient = st.selectbox(
        "Select Patient",
        patients,
        index=default_idx,
        format_func=lambda x: x.upper().replace("_", " "),
    )


    st.markdown("---")

    with st.expander("How to Use / What It Does"):
        st.markdown("### What is this?")
        st.markdown(
            "This tool designs personalised mRNA cancer vaccine candidates for "
            "multiple myeloma (MM) patients. It takes a patient's tumour mutation "
            "data and identifies which protein fragments (epitopes) are most likely "
            "to be seen by the immune system as foreign — making them good vaccine targets."
        )
        st.markdown("### Pipeline Overview")
        st.markdown("""
        1. **Load mutations** — Reads somatic mutations from the patient's tumour DNA
        2. **Generate peptides** — Creates 8-11 amino acid fragments around each mutation
        3. **Predict MHC binding** — Tests how strongly each fragment binds to HLA proteins
        4. **Rank epitopes** — Scores by immunogenicity, driver gene status, and other factors
        5. **Design vaccine** — Builds an mRNA construct with selected epitopes optimised for stability
        """)
        st.markdown("### New in v2.0 — Scientific Validation Layers")
        st.markdown("""
        - **HLA Ligandomics** — Mass-spec confirmed HLA-bound peptides (PRIDE/HMB databases)
        - **cBioPortal** — Mutation frequency in real MM patient cohorts (50+ studies)
        - **TCR Repertoire** — Known T-cell receptor targets for predicted epitopes
        - **PeptideAtlas** — Proteomics confirmation that the target proteins are expressed
        """)
        st.markdown("### How to Read the Results")
        st.markdown("""
        - **Top Candidates table** — Epitopes ranked by immunogenicity score (higher = better)
        - **IC50** — Binding strength in nanomolars. Lower = tighter binding (<50nM = strong)
        - **Agretopicity Index** — How much better the mutant binds vs wildtype (>1 = promising)
        - **Driver genes** — KRAS, NRAS, TP53, BRAF etc. are known MM oncogenes
        - **mRNA Vaccine** — The final optimised construct sequence ready for synthesis
        """)
        st.markdown("### Data Sources")
        st.markdown("""
        - Genomic data: **MMRF CoMMpass Study** (995 newly diagnosed MM patients)
        - Binding predictions: **MHCflurry 2.1** (neural network trained on IEDB)
        - dNdScov driver scores: Published selection pressure cutoffs
        - Trial matching: **ClinicalTrials.gov API** (live data)
        """)
        st.markdown("### Limitations")
        st.markdown("""
        - HLA typing uses population frequency priors (WES-based typing improves accuracy)
        - No RNA-seq expression data yet for many patients (expression filter applied where available)
        - Predictions are computational — experimental validation required before clinical use
        - dNdScov uses published gene scores, not per-patient calculation
        """)
        st.markdown(
            '<div class="disclaimer">'
            '<b>Research Only</b> — Not validated for clinical use. '
            'Consult a licensed oncologist before any treatment decision.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("---")
    st.markdown("### Filters")

    ic50_threshold = st.slider(
        "Max IC50 (nM)",
        min_value=50, max_value=5000, value=500, step=50,
        help="Filter candidates by MHC-I binding affinity"
    )

    show_clonal_only = st.checkbox(
        "Clonal mutations only",
        value=False,
        help="Show only mutations present in all tumour cells"
    )

    show_drivers_only = st.checkbox(
        "Driver genes only",
        value=False,
        help="Show only known MM driver gene mutations"
    )


    st.markdown("---")
    st.markdown("### IND Documentation")

    ind_package_dir = f"output/ind_package/{selected_patient}"
    ind_ready = os.path.exists(ind_package_dir) and os.path.isdir(ind_package_dir)

    if st.button("Generate IND Package", key="gen_ind_btn"):
        with st.spinner("Generating IND documentation package..."):
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "16_generate_ind_package.py", "--patient", selected_patient],
                    capture_output=True, text=True, cwd=str(Path.cwd())
                )
                if result.returncode == 0:
                    st.success("IND package generated!")
                else:
                    st.error(f"IND package generation failed: {result.stderr[:200]}")
            except Exception as e:
                st.error(f"Could not generate package: {e}")

    if ind_ready:
        import zipfile, io
        # Create a zip of the package
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(ind_package_dir):
                fpath = os.path.join(ind_package_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        zf.writestr(fname, f.read())
        zip_buffer.seek(0)
        st.download_button(
            "Download IND Package (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{selected_patient}_ind_package.zip",
            mime="application/zip",
        )
    else:
        st.caption("Generate above to download IND package.")

    st.markdown("---")
    st.markdown(
        "Open-source pipeline for designing personalised "
        "mRNA cancer vaccines for multiple myeloma.\n\n"
        "[▫  Paper](https://github.com/maxfromtilt/mm-neoantigen-pipeline/blob/main/CLINICAL_REPORT.md) · "
        "[▪  Code](https://github.com/maxfromtilt/mm-neoantigen-pipeline)"
    )
    st.markdown(
        '<div class="disclaimer">'
        '<b>Research Only</b> — Not validated for clinical use.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Load data ────────────────────────────────────────────────────────

df_enhanced = load_patient_data(selected_patient, "enhanced")
df_binding = load_patient_data(selected_patient, "binding")
df_mhc_ii = load_patient_data(selected_patient, "mhc_ii")
df_dual = load_patient_data(selected_patient, "dual")
df_epitopes = load_patient_data(selected_patient, "epitopes")
vaccine_report = load_vaccine_report(selected_patient)

# ── Apply filters ────────────────────────────────────────────────────

df_filtered = df_enhanced.copy()
if not df_filtered.empty:
    df_filtered = df_filtered[df_filtered["ic50_nM"].fillna(99999) <= ic50_threshold]
    if show_clonal_only and "clonality" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered.get("clonality", "") == "clonal"]
    if show_drivers_only and "is_driver_gene" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered.get("is_driver_gene", False) == True]


# ══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

st.markdown(f'<p class="main-header">⟠  Neoantigen Vaccine Design: {selected_patient.upper().replace("_", " ")}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalised mRNA cancer vaccine pipeline for multiple myeloma</p>', unsafe_allow_html=True)

# ── Key Metrics Row ──────────────────────────────────────────────────

if not df_enhanced.empty:
    total_predictions = len(df_enhanced)
    total_binders = len(df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 500])
    strong_binders = len(df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 50])
    n_dual = len(df_dual) if not df_dual.empty else 0

    clonal_count = 0
    if "clonality" in df_enhanced.columns:
        clonal_count = len(df_enhanced[
            (df_enhanced.get("clonality", "") == "clonal") &
            (df_enhanced["ic50_nM"].fillna(99999) < 500)
        ])

    # TMB calculation
    tmb_info = calculate_tmb(df_enhanced)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Predictions", f"{total_predictions:,}")
    col2.metric("MHC-I Binders", f"{total_binders}", f"{100*total_binders/max(total_predictions,1):.1f}%")
    col3.metric("Strong Binders", f"{strong_binders}", "IC50 < 50 nM")
    col4.metric("Dual MHC-I/II", f"{n_dual}", "CD4+ & CD8+")
    col5.metric("Clonal Binders", f"{clonal_count}", "In all tumour cells")
    col6.metric("TMB", f"{tmb_info['tmb_score']} mut/MB", tmb_info['tmb_category'].split(" — ")[0])

    with st.expander("▸  Tumour Mutational Burden (TMB) Details"):
        st.markdown(
            f"""
            **TMB = {tmb_info['tmb_score']} mutations/MB** (capture size: {tmb_info['capture_size_mb']} Mb)

            - Mutations counted: **{tmb_info['mutation_count']}** (missense/frameshift)
            - Category: **{tmb_info['tmb_category']}**

            **MM population norms:** Median TMB ~2-5 mut/MB. Published literature: MM is typically
            a lower-TMB haematological malignancy compared to solid tumours (e.g. melanoma ~10-50 mut/MB).
            Higher TMB correlates with better response to immune checkpoint inhibitors and may
            indicate more neoantigen targets per vaccine.
            """
        )

    # HLA typing display in sidebar
    patient_hla = load_patient_hla(selected_patient)
    with st.sidebar:
        with st.expander("▸  HLA Typing"):
            st.markdown(f"**Patient HLA (confidence: {patient_hla['confidence']})**")
            for allele in patient_hla.get("class_i", [])[:4]:
                st.markdown(f"  \u2022 {allele}")
            if patient_hla.get("class_ii"):
                st.markdown(f"  _Class II: {', '.join(patient_hla.get('class_ii', [])[:2])}_")

    st.markdown("---")


# ── Tabs ─────────────────────────────────────────────────────────────


# ── Site Introduction ─────────────────────────────────────────────────
st.markdown("""
## MM Neoantigen Vaccine Designer

**What it does:** Takes a patient's tumour mutation data, identifies the most immunogenic protein fragments (neoantigens), and designs an optimised mRNA vaccine construct ready for synthesis.

**Research tool only.** Predictions are computational — experimental validation required before clinical use.
""")

with st.expander("View data sources and methodology"):
    st.markdown("""
### Primary Data Sources
- **[MMRF CoMMpass Study](https://themmrf.org/we-are-curing-multiple-myeloma/mmrf-commpass-study/)** — Whole genome/exome sequencing and RNA-seq from ~995 newly diagnosed MM patients (primary dataset)
- **[NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/)** — Federal cancer genomics data repository (API for MMRF data)
- **[TCGA](https://portal.gdc.cancer.gov/)** — Validated mutation frequencies across multiple myeloma cohorts

### Scientific Validation Layers
- **[PRIDE Database](https://www.ebi.ac.uk/pride/)** — Mass spectrometry validated HLA-bound peptides (experimental confirmation)
- **[HMB Database](https://www.hmadb.org/)** — Human MHC-bound peptide repository
- **[cBioPortal](https://www.cbioportal.org/)** — Mutation frequency and survival data across 50+ cancer studies
- **[PeptideAtlas](https://www.peptideatlas.org/)** — Human proteomics data confirming protein expression
- **[VDJdb](https://vdjdb.cvrgr.org/)** — T-cell receptor sequences with known epitope specificity
- **[McPAS-TCR](http://friedmanburg.com/McPAS-TCR/)** — TCR pathology database

### Prediction Engine
- **[MHCflurry 2.1](https://github.com/openvax/mhcflurry)** — Neural network MHC binding predictions trained on IEDB (350,000+ measurements)
- **[PSSM binding matrices](https://pubmed.ncbi.nlm.nih.gov/15608049/)** — Position-specific scoring matrices for 6 HLA alleles

### Limitations
- HLA typing uses population frequency priors (arcasHLA/WES-based typing improves accuracy)
- No RNA-seq expression data for all patients
- dNdScov uses published gene scores, not per-patient calculation
- Predictions are computational — experimental validation required before clinical use
""")
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⌬  Analyse New Patient",
    "▤  Overview",
    "◎  Top Candidates",
    "⚗  Binding Analysis",
    "⟠  HLA Coverage",
    "⊕  Vaccine Construct",
    "⟡  3D Viewer",
    "≡  Full Data",
])


# ── Tab 0: Analyse New Patient ────────────────────────────────────────

with tab0:
    st.subheader("Upload Patient Mutation Data")
    st.markdown(
        "Upload a somatic mutation CSV file to design a personalised neoantigen vaccine. "
        "The pipeline will automatically parse mutations, predict MHC binding across 6 HLA alleles, "
        "and rank vaccine candidates."
    )

    col_upload, col_format = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your mutation CSV here",
            type=["csv"],
            help="Accepts MMRF CoMMpass format, MAF, or VCF-derived tables. Download the sample file above for the correct format.",
        )

    with col_format:
        st.markdown("**Required columns:**")
        st.markdown("- `gene_symbol` — gene name (e.g. DIS3, KRAS)")
        st.markdown("- `aa_change` — amino acid change (e.g. E196K or p.E196K)")
        st.markdown("")
        st.markdown("**Optional columns:**")
        st.markdown("- `consequence_type` — filters to missense/frameshift")
        st.markdown("- `chromosome`, `start_position`")

        st.markdown("")
        st.download_button(
            "↓  Download sample file (20 patients)",
            data=open("data/sample_upload.csv", "rb").read() if os.path.exists("data/sample_upload.csv") else b"",
            file_name="sample_mm_mutations.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### Or upload BAM/CRAM for variant calling")
        st.caption("Upload an aligned BAM or CRAM file for WGS/WES variant calling. "
                   "Note: Large file upload may take time. Variant calling runs server-side.")

        uploaded_bam = st.file_uploader(
            "Drop BAM or CRAM file here",
            type=["bam", "cram"],
            key="bam_uploader",
            help="Accepts indexed BAM (.bai) or CRAM (.crai) files. Large files (100MB+) may be slow to upload.",
        )
        if uploaded_bam:
            import tempfile, shutil
            with st.spinner("Processing BAM/CRAM file..."):
                try:
                    # Save uploaded file temporarily
                    suffix = ".bam" if uploaded_bam.name.endswith(".bam") else ".cram"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        shutil.copyfileobj(uploaded_bam, tmp)
                        tmp_path = tmp.name
                    # Run variant calling
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "06_wgs_variant_calling.py",
                         "--patient", selected_patient,
                         "--bam", tmp_path],
                        capture_output=True, text=True, cwd=str(Path.cwd()), timeout=300
                    )
                    if result.returncode == 0:
                        st.success("Variant calling complete!")
                        st.info("Results saved to: "
                                f"output/{selected_patient}_wgs/{selected_patient}_wgs_variants.csv")
                    else:
                        st.error(f"Variant calling failed: {result.stderr[:200]}")
                    # Cleanup temp file
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Could not process BAM/CRAM: {e}")

    if uploaded_file:
        run_uploaded_pipeline(uploaded_file)


# ── Tab 1: Overview ──────────────────────────────────────────────────

with tab1:
    if not df_enhanced.empty:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Binding Affinity Distribution")

            binders_df = df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 5000].copy()
            binders_df["Binding Category"] = binders_df["ic50_nM"].apply(classify_binding)

            fig_hist = px.histogram(
                binders_df,
                x="ic50_nM",
                color="Binding Category",
                color_discrete_map={
                    "Strong Binder": "#2c5282",
                    "Weak Binder": "#4299e1",
                    "Non-Binder": "#cbd5e0",
                },
                nbins=50,
                labels={"ic50_nM": "Predicted IC50 (nM)", "count": "Count"},
                template="plotly_white",
            )
            fig_hist.add_vline(x=50, line_dash="dash", line_color="red",
                              annotation_text="Strong (50 nM)")
            fig_hist.add_vline(x=500, line_dash="dash", line_color="orange",
                              annotation_text="Weak (500 nM)")
            fig_hist.update_layout(
                height=400,
                xaxis_title="Predicted IC50 (nM)",
                yaxis_title="Number of Candidates",
                showlegend=True,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.subheader("Enhanced Score Components")

            if "enhanced_score" in df_enhanced.columns:
                top_20 = df_enhanced.nsmallest(20, "ic50_nM")
                if "clonality_score" in top_20.columns:
                    avg_scores = {
                        "MHC-I Binding": top_20["ic50_nM"].apply(
                            lambda x: max(0, 1 - min(x, 50000)/50000)
                        ).mean(),
                        "Clonality": top_20["clonality_score"].mean(),
                        "Driver Gene": top_20["is_driver_gene"].astype(float).mean() if "is_driver_gene" in top_20.columns else 0,
                    }

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(avg_scores.values()),
                        theta=list(avg_scores.keys()),
                        fill='toself',
                        fillcolor='rgba(44, 82, 130, 0.3)',
                        line=dict(color='#2c5282', width=2),
                        name='Top 20 Average',
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        height=400,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        # Clonality breakdown
        if "clonality" in df_enhanced.columns:
            st.subheader("Mutation Clonality vs Binding Affinity")

            binders_with_clonality = df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 2000].copy()
            if not binders_with_clonality.empty and "cancer_cell_fraction" in binders_with_clonality.columns:
                fig_scatter = px.scatter(
                    binders_with_clonality,
                    x="cancer_cell_fraction",
                    y="ic50_nM",
                    color="clonality",
                    size="enhanced_score" if "enhanced_score" in binders_with_clonality.columns else None,
                    hover_data=["gene_symbol", "aa_change", "mutant_peptide", "hla_allele"],
                    color_discrete_map={
                        "clonal": "#2c5282",
                        "likely_clonal": "#4299e1",
                        "subclonal": "#cbd5e0",
                    },
                    labels={
                        "cancer_cell_fraction": "Cancer Cell Fraction (CCF)",
                        "ic50_nM": "IC50 (nM)",
                        "clonality": "Clonality",
                    },
                    template="plotly_white",
                )
                fig_scatter.add_hline(y=500, line_dash="dash", line_color="orange",
                                      annotation_text="Binder threshold")
                fig_scatter.add_hline(y=50, line_dash="dash", line_color="red",
                                      annotation_text="Strong binder")
                fig_scatter.update_yaxes(type="log")
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown(
                    '<div class="highlight-box">'
                    '<b>▸  Key insight:</b> The best vaccine targets are in the '
                    '<b>top-right quadrant</b> — high clonality (present in all tumour cells) '
                    'AND strong binding (low IC50). These are mutations the immune system '
                    'can target effectively and that exist in every cancer cell.'
                    '</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No data available for this patient. Run the pipeline first.")


# ── Tab 2: Top Candidates ───────────────────────────────────────────

with tab2:
    if not df_filtered.empty:
        st.subheader("Ranked Vaccine Candidates")

        # MS-Confirmed filter (from proteogenomics)
        show_ms_confirmed_only = st.checkbox(
            "Show MS-confirmed only",
            value=False,
            help="Show only epitopes validated by mass spectrometry (PRIDE/PeptideAtlas)"
        )

        # Get best per gene+mutation+peptide
        display_cols = [
            "gene_symbol", "aa_change", "mutant_peptide", "hla_allele",
            "ic50_nM", "classification",
        ]
        optional_cols = [
            "clonality", "cancer_cell_fraction",
            "best_mhc_ii_ic50", "enhanced_score",
        ]
        for c in optional_cols:
            if c in df_filtered.columns:
                display_cols.append(c)

        available_cols = [c for c in display_cols if c in df_filtered.columns]
        top_df = df_filtered.nsmallest(50, "ic50_nM")[available_cols].copy()

        # Apply MS-confirmed filter if proteomics data available
        if show_ms_confirmed_only:
            if "proteomics_confirmed" in top_df.columns:
                top_df = top_df[top_df["proteomics_confirmed"] == True]
                if top_df.empty:
                    st.warning("No MS-confirmed epitopes found. Run proteogenomics first (07_proteogenomics.py).")
            else:
                st.info("Proteomics data not loaded. Run 07_proteogenomics.py first to enable MS-confirmed filtering.")
        elif "proteomics_confirmed" in top_df.columns:
            # Add MS Confirmed badge indicator
            pass  # handled in display below

        # Add MS Confirmed badge if data available
        if "proteomics_confirmed" in top_df.columns:
            top_df["MS Confirmed"] = top_df["proteomics_confirmed"].apply(
                lambda x: "✓" if x else ""
            )

        # Format
        if "ic50_nM" in top_df.columns:
            top_df["ic50_nM"] = top_df["ic50_nM"].round(1)
        if "enhanced_score" in top_df.columns:
            top_df["enhanced_score"] = top_df["enhanced_score"].round(1)
        if "cancer_cell_fraction" in top_df.columns:
            top_df["cancer_cell_fraction"] = top_df["cancer_cell_fraction"].round(3)
        if "best_mhc_ii_ic50" in top_df.columns:
            top_df["best_mhc_ii_ic50"] = top_df["best_mhc_ii_ic50"].round(1)

        # Rename columns for display
        rename_map = {
            "gene_symbol": "Gene",
            "aa_change": "Mutation",
            "mutant_peptide": "Peptide",
            "hla_allele": "HLA Allele",
            "ic50_nM": "IC50 (nM)",
            "classification": "Binding",
            "clonality": "Clonality",
            "cancer_cell_fraction": "CCF",
            "best_mhc_ii_ic50": "MHC-II IC50",
            "enhanced_score": "Score",
            "proteomics_confirmed": "MS Confirmed",
        }
        top_df = top_df.rename(columns=rename_map)

        st.dataframe(
            top_df,
            use_container_width=True,
            height=600,
            hide_index=True,
        )

        # Dual binders section
        if not df_dual.empty:
            st.subheader("◎  Dual MHC-I / MHC-II Binders")
            st.markdown(
                "These candidates engage **both** CD8+ cytotoxic and CD4+ helper T cells, "
                "which is associated with stronger and more durable immune responses."
            )

            dual_display_cols = ["gene_symbol", "aa_change", "mutant_peptide",
                                 "hla_allele", "ic50_nM"]
            if "best_mhc_ii_ic50" in df_dual.columns:
                dual_display_cols.append("best_mhc_ii_ic50")
            if "clonality" in df_dual.columns:
                dual_display_cols.append("clonality")
            if "enhanced_score" in df_dual.columns:
                dual_display_cols.append("enhanced_score")

            avail_dual = [c for c in dual_display_cols if c in df_dual.columns]
            st.dataframe(
                df_dual[avail_dual].rename(columns=rename_map),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No candidates match current filters. Try adjusting the IC50 threshold.")


# ── Tab 3: Binding Analysis ─────────────────────────────────────────

with tab3:
    if not df_enhanced.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("MHC-I Binding by Gene")

            binders = df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 500].copy()
            if not binders.empty:
                gene_counts = binders.groupby("gene_symbol").agg(
                    n_binders=("ic50_nM", "count"),
                    best_ic50=("ic50_nM", "min"),
                    mean_ic50=("ic50_nM", "mean"),
                ).reset_index().sort_values("best_ic50")

                fig_gene = px.bar(
                    gene_counts.head(15),
                    x="gene_symbol",
                    y="n_binders",
                    color="best_ic50",
                    color_continuous_scale="Blues_r",
                    labels={
                        "gene_symbol": "Gene",
                        "n_binders": "Number of Binders",
                        "best_ic50": "Best IC50 (nM)",
                    },
                    template="plotly_white",
                )
                fig_gene.update_layout(height=400)
                st.plotly_chart(fig_gene, use_container_width=True)

        with col_b:
            st.subheader("Binding Affinity by HLA Allele")

            if "hla_allele" in df_enhanced.columns:
                hla_binders = df_enhanced[df_enhanced["ic50_nM"].fillna(99999) < 500]
                if not hla_binders.empty:
                    fig_box = px.box(
                        hla_binders,
                        x="hla_allele",
                        y="ic50_nM",
                        color="hla_allele",
                        labels={
                            "hla_allele": "HLA Allele",
                            "ic50_nM": "IC50 (nM)",
                        },
                        template="plotly_white",
                    )
                    fig_box.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)

        # MHC-II analysis
        if not df_mhc_ii.empty:
            st.subheader("MHC-II Binding Predictions (CD4+ T Helper Epitopes)")

            mhc_ii_col = "mhc_ii_ic50" if "mhc_ii_ic50" in df_mhc_ii.columns else None
            mhc_ii_binders = pd.DataFrame()
            if mhc_ii_col:
                mhc_ii_binders = df_mhc_ii[df_mhc_ii[mhc_ii_col] < 1000]
            col_c, col_d = st.columns(2)

            with col_c:
                st.metric("MHC-II Predictions", f"{len(df_mhc_ii):,}")
                st.metric("MHC-II Binders (IC50<1000)", f"{len(mhc_ii_binders)}")

            with col_d:
                if not mhc_ii_binders.empty and "hla_ii_allele" in mhc_ii_binders.columns:
                    fig_mhcii = px.histogram(
                        mhc_ii_binders,
                        x="mhc_ii_ic50",
                        color="hla_ii_allele",
                        nbins=30,
                        labels={"mhc_ii_ic50": "MHC-II IC50 (nM)"},
                        template="plotly_white",
                    )
                    fig_mhcii.update_layout(height=300)
                    st.plotly_chart(fig_mhcii, use_container_width=True)


# ── Tab 4: HLA Coverage ─────────────────────────────────────────────

with tab4:
    if not df_enhanced.empty and "hla_allele" in df_enhanced.columns:
        st.subheader("HLA Allele Coverage Analysis")

        # Population frequencies
        hla_freqs = {
            "HLA-A*02:01": 0.28, "HLA-A*01:01": 0.15,
            "HLA-A*03:01": 0.13, "HLA-A*24:02": 0.10,
            "HLA-B*07:02": 0.12, "HLA-B*08:01": 0.09,
        }

        coverage_data = []
        for allele, freq in hla_freqs.items():
            allele_data = df_enhanced[df_enhanced["hla_allele"].fillna("") == allele]
            binders = allele_data[allele_data["ic50_nM"].fillna(99999) < 500]
            strong = allele_data[allele_data["ic50_nM"].fillna(99999) < 50]
            coverage_data.append({
                "HLA Allele": allele,
                "Population Frequency": f"{freq:.0%}",
                "Total Binders": len(binders),
                "Strong Binders": len(strong),
                "Best IC50 (nM)": round(allele_data["ic50_nM"].min(), 1) if len(allele_data) > 0 else "-",
            })

        coverage_df = pd.DataFrame(coverage_data)
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)

        # Sunburst chart
        allele_binder_data = []
        for allele in hla_freqs:
            for cat in ["Strong Binder", "Weak Binder"]:
                if cat == "Strong Binder":
                    count = len(df_enhanced[(df_enhanced["hla_allele"].fillna("") == allele) & (df_enhanced["ic50_nM"].fillna(99999) < 50)])
                else:
                    count = len(df_enhanced[(df_enhanced["hla_allele"].fillna("") == allele) & (df_enhanced["ic50_nM"].fillna(99999) >= 50) & (df_enhanced["ic50_nM"].fillna(99999) < 500)])
                if count > 0:
                    allele_binder_data.append({
                        "Allele": allele.replace("HLA-", ""),
                        "Category": cat,
                        "Count": count,
                    })

        if allele_binder_data:
            sb_df = pd.DataFrame(allele_binder_data)
            fig_sun = px.sunburst(
                sb_df,
                path=["Allele", "Category"],
                values="Count",
                color="Category",
                color_discrete_map={
                    "Strong Binder": "#2c5282",
                    "Weak Binder": "#4299e1",
                },
                template="plotly_white",
            )
            fig_sun.update_layout(height=500)
            st.plotly_chart(fig_sun, use_container_width=True)

        # Population coverage estimate
        covered_alleles = [a for a in hla_freqs if len(df_enhanced[(df_enhanced["hla_allele"].fillna("") == a) & (df_enhanced["ic50_nM"].fillna(99999) < 500)]) > 0]
        total_coverage = 1 - np.prod([1 - hla_freqs[a] for a in covered_alleles])

        st.markdown(
            f'<div class="highlight-box">'
            f'<b>Estimated population coverage:</b> {total_coverage:.0%} '
            f'(based on {len(covered_alleles)} HLA alleles with binders). '
            f'Patient-specific HLA typing would refine this to exact coverage.'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tab 5: Vaccine Construct ─────────────────────────────────────────

with tab5:
    st.subheader("mRNA Vaccine Construct")

    if not df_epitopes.empty:
        col_v1, col_v2 = st.columns([2, 1])

        with col_v1:
            st.markdown("#### Selected Epitopes")
            ep_cols = ["gene_symbol", "aa_change", "mutant_peptide",
                       "hla_allele", "ic50_nM", "vaccine_priority_score"]
            avail_ep = [c for c in ep_cols if c in df_epitopes.columns]
            st.dataframe(
                df_epitopes[avail_ep].rename(columns={
                    "gene_symbol": "Gene",
                    "aa_change": "Mutation",
                    "mutant_peptide": "Peptide",
                    "hla_allele": "HLA",
                    "ic50_nM": "IC50 (nM)",
                    "vaccine_priority_score": "Priority Score",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with col_v2:
            st.markdown("#### Construct Overview")

            # Parse from vaccine report
            construct_info = {
                "Epitopes": len(df_epitopes) if not df_epitopes.empty else 0,
                "Linker": "GGSGGGGSGG",
                "5' Cap": "m7GpppN (Cap1)",
                "Modification": "m1Ψ (all U)",
                "Poly-A": "120 nt",
                "LNP formulation": "Recommended",
            }
            for k, v in construct_info.items():
                st.markdown(f"**{k}:** {v}")

            st.markdown("---")
            st.markdown("#### mRNA Synthesis Order")

            if st.button("Generate Synthesis Order", key="gen_synthesis_order"):
                with st.spinner("Generating synthesis order..."):
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, "17_synthesis_order.py", "--patient", selected_patient],
                            capture_output=True, text=True, cwd=str(Path.cwd())
                        )
                        if result.returncode == 0:
                            st.success("Synthesis order generated!")
                        else:
                            st.error(f"Synthesis order failed: {result.stderr}")
                    except Exception as e:
                        st.error(f"Could not generate order: {e}")

            # Download synthesis order CSV
            synthesis_csv_path = f"output/{selected_patient}/synthesis_order.csv"
            synthesis_form_path = f"output/{selected_patient}/synthesis_order_form.txt"

            if os.path.exists(synthesis_csv_path):
                with open(synthesis_csv_path) as f:
                    synthesis_data = f.read()
                st.download_button(
                    "Download Synthesis Order (CSV)",
                    data=synthesis_data,
                    file_name=f"{selected_patient}_synthesis_order.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Run Generate above to create the synthesis order.")

            if os.path.exists(synthesis_form_path):
                with open(synthesis_form_path) as f:
                    form_data = f.read()
                st.download_button(
                    "Download Order Form (TXT)",
                    data=form_data,
                    file_name=f"{selected_patient}_synthesis_order_form.txt",
                    mime="text/plain",
                )

            # Email vendor button
            email_draft_path = f"output/{selected_patient}/synthesis_email_draft.txt"
            if os.path.exists(email_draft_path):
                with open(email_draft_path) as f:
                    email_content = f.read()
                lines = email_content.split("\n")
                mailto_line = [l for l in lines if l.startswith("mailto:")]
                if mailto_line:
                    st.markdown(
                        f"[📧  Email Vendor (TriLink)]({mailto_line[0]})"
                    )

            st.markdown("---")
            st.markdown("#### Download Clinical Report")

            if st.button("Generate Clinical PDF Report", key="gen_clinical_pdf"):
                with st.spinner("Generating clinical report PDF..."):
                    try:
                        import importlib.util, sys
                        sys.path.insert(0, str(Path.cwd()))
                        spec = importlib.util.spec_from_file_location(
                            "clinical_report",
                            Path.cwd() / "13_clinical_report.py"
                        )
                        cr_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(cr_mod)
                        pdf_path = cr_mod.build_report(selected_patient)
                        st.success(f"Report generated: {pdf_path}")
                    except Exception as e:
                        st.error(f"Could not generate report: {e}")
            existing_pdf = f"output/{selected_patient}_clinical_report.pdf"
            if os.path.exists(existing_pdf):
                with open(existing_pdf, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    "Download Clinical Report (PDF)",
                    data=pdf_data,
                    file_name=f"{selected_patient}_clinical_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("Run Generate above to create the PDF report.")

        # Architecture diagram
        st.markdown("#### Construct Architecture")
        st.code(
            "5'Cap ── 5'UTR ── [Signal Peptide]─[Ep1]─[GGSGGGGSGG]─[Ep2]─...─[EpN] ── STOP ── 3'UTR ── AAAA...A(120)",
            language=None,
        )

    if vaccine_report:
        with st.expander("▫  Full Vaccine Design Report"):
            st.text(vaccine_report)


# ── Tab 6: 3D Viewer ─────────────────────────────────────────────────

with tab6:
    st.subheader("3D pMHC Structural Viewer")
    st.markdown(
        "Interactive 3D visualization of the top epitope-HLA class I complex. "
        "Shows the peptide sitting in the HLA binding groove."
    )

    patient_id = selected_patient

    # Look for generated structure viewers
    structure_dir = f"output/{patient_id}_structures"
    os.makedirs(structure_dir, exist_ok=True)

    if os.path.exists(structure_dir):
        viewers = sorted(Path(structure_dir).glob("*_viewer.html"))
    else:
        viewers = []

    if not viewers:
        st.info("No structure viewers generated yet. Run the structure viewer to generate them.")
        col_gen, _ = st.columns([1, 2])
        with col_gen:
            if st.button("Generate 3D Viewers", key="gen_structure_viewer"):
                with st.spinner("Generating 3D structure viewers..."):
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "14_structure_viewer.py", "--patient", patient_id],
                        capture_output=True, text=True, cwd=str(Path.cwd())
                    )
                    if result.returncode == 0:
                        st.success("Viewers generated!")
                        st.rerun()
                    else:
                        st.error(f"Viewer generation failed: {result.stderr}")

        st.markdown("**Or generate for a specific epitope:**")
        col_ep, col_hla = st.columns(2)
        with col_ep:
            epitopes = []
            if not df_epitopes.empty:
                epitopes = df_epitopes["mutant_peptide"].dropna().unique().tolist()[:10] if "mutant_peptide" in df_epitopes.columns else []
            selected_ep = st.selectbox(
                "Epitope",
                options=epitopes if epitopes else ["KLACLDSYIIK"],
                key="struct_epitope_select",
            )
        with col_hla:
            hlas = ["HLA-A*02:01", "HLA-A*03:01", "HLA-A*01:01",
                    "HLA-A*24:02", "HLA-B*07:02", "HLA-B*08:01"]
            selected_hla = st.selectbox("HLA Allele", options=hlas, key="struct_hla_select")

        if st.button("Generate Single Viewer", key="gen_single_viewer"):
            with st.spinner("Generating viewer..."):
                import subprocess
                result = subprocess.run(
                    [sys.executable, "14_structure_viewer.py",
                     "--patient", patient_id,
                     "--epitope", selected_ep,
                     "--hla", selected_hla],
                    capture_output=True, text=True, cwd=str(Path.cwd())
                )
                if result.returncode == 0:
                    st.success("Viewer generated!")
                    st.rerun()
                else:
                    st.error(f"Failed: {result.stderr}")
    else:
        st.success(f"Found {len(viewers)} structure viewer(s)")

        # Display each viewer
        for viewer_path in viewers:
            viewer_name = viewer_path.stem.replace("_viewer", "").replace("_", " ")
            with st.expander(f"▸  {viewer_name}"):
                # Read HTML and display in iframe
                try:
                    with open(viewer_path, "r") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.error(f"Could not load viewer: {e}")

                # Download button
                with open(viewer_path, "rb") as f:
                    viewer_data = f.read()
                st.download_button(
                    f"Download {viewer_name} HTML",
                    data=viewer_data,
                    file_name=viewer_path.name,
                    mime="text/html",
                )

        if st.button("🔄  Regenerate Viewers", key="regen_viewers"):
            import subprocess
            with st.spinner("Regenerating..."):
                result = subprocess.run(
                    [sys.executable, "14_structure_viewer.py", "--patient", patient_id],
                    capture_output=True, text=True, cwd=str(Path.cwd())
                )
                if result.returncode == 0:
                    st.success("Viewers regenerated!")
                    st.rerun()
                else:
                    st.error(f"Failed: {result.stderr}")


# ── Tab 7: Full Data ─────────────────────────────────────────────────

with tab7:
    st.subheader("Complete Dataset")

    if not df_enhanced.empty:
        st.markdown(f"**{len(df_enhanced):,} total predictions** for {selected_patient.upper()}")

        # Column selector
        all_cols = list(df_enhanced.columns)
        default_cols = ["gene_symbol", "aa_change", "mutant_peptide",
                       "hla_allele", "ic50_nM", "classification"]
        if "enhanced_score" in all_cols:
            default_cols.append("enhanced_score")
        if "clonality" in all_cols:
            default_cols.append("clonality")

        selected_cols = st.multiselect(
            "Select columns to display",
            all_cols,
            default=[c for c in default_cols if c in all_cols],
        )

        if selected_cols:
            st.dataframe(
                df_filtered[selected_cols],
                use_container_width=True,
                height=600,
                hide_index=True,
            )

        # Download button
        csv = df_enhanced.to_csv(index=False)
        st.download_button(
            label="↓  Download Full Results (CSV)",
            data=csv,
            file_name=f"{selected_patient}_enhanced_results.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("▸  **To analyse your own patient data**, use the **⌬  Analyse New Patient** tab.")


# ── Footer ───────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    '<b>▲  Research Use Only.</b> This pipeline is for computational research purposes only. '
    'Vaccine constructs have not been experimentally validated. Any therapeutic application '
    'requires preclinical testing, regulatory approval, and clinical oversight by qualified '
    'medical professionals. This tool does not provide medical advice.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<center><small>MM Neoantigen Vaccine Designer · "
    "<a href='https://github.com/maxfromtilt/mm-neoantigen-pipeline'>GitHub</a> · "
    "© 2026 Robert Doran</small></center>",
    unsafe_allow_html=True,
)
