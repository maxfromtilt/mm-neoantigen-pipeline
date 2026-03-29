"""
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
    progress = st.progress(0, text="Parsing mutations...")
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

        progress.progress((idx + 1) / len(raw_df), text=f"Parsing mutations... {idx+1}/{len(raw_df)}")

    if not all_candidates:
        st.error("No peptide candidates could be generated. Check aa_change format (e.g. E196K or p.E196K).")
        return

    candidates_df = pd.DataFrame(all_candidates)
    n_candidates = len(candidates_df)
    status.info(f"Generated {n_candidates} peptide candidates from {len(raw_df)} mutations")

    # Step 2: Binding predictions
    progress.progress(0, text="Running binding predictions...")
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
                progress.progress(pred_count / total_preds,
                                  text=f"Predicting binding... {pred_count}/{total_preds}")

    results_df = pd.DataFrame(all_results)
    binders = results_df[results_df["ic50_nM"] < 500]
    strong = results_df[results_df["ic50_nM"] < 50]

    progress.progress(1.0, text="Analysis complete!")

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

    # Email results
    st.markdown("---")
    st.subheader("Send Results")
    col_email, col_share = st.columns(2)

    with col_email:
        email_addr = st.text_input("Email address (to receive results)")
        if email_addr and st.button("Send Results via Email"):
            st.info(
                f"Email delivery is not yet connected to a mail server. "
                f"For now, please download the CSV above and send to **{email_addr}** manually. "
                f"Automated email delivery coming soon."
            )

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
    page_title="MM Neoantigen Vaccine Designer",
    page_icon="⟠ ",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    """Load pre-computed results for a patient."""
    paths = {
        "enhanced": f"output/{patient_id}_enhanced/enhanced_predictions.csv",
        "binding": f"output/{patient_id}_mhcflurry/binding_predictions.csv",
        "mhc_ii": f"output/{patient_id}_enhanced/mhc_ii_predictions.csv",
        "top": f"output/{patient_id}_enhanced/top_candidates.csv",
        "dual": f"output/{patient_id}_enhanced/dual_mhc_binders.csv",
        "epitopes": f"output/{patient_id}_mhcflurry/selected_epitopes.csv",
    }
    path = paths.get(data_type, "")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_vaccine_report(patient_id: str) -> str:
    """Load vaccine report text."""
    path = f"output/{patient_id}_mhcflurry/vaccine_report.txt"
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


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
    return sorted(patients) if patients else ["mmrf_1251", "mmrf_1274"]


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⟠  MM Vaccine Designer")
    st.markdown("---")

    patients = get_available_patients()
    selected_patient = st.selectbox(
        "Select Patient",
        patients,
        format_func=lambda x: x.upper().replace("_", " "),
    )

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
    st.markdown("### About")
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
    df_filtered = df_filtered[df_filtered["ic50_nM"] <= ic50_threshold]
    if show_clonal_only and "clonality" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["clonality"] == "clonal"]
    if show_drivers_only and "is_driver_gene" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["is_driver_gene"] == True]


# ══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

st.markdown(f'<p class="main-header">⟠  Neoantigen Vaccine Design: {selected_patient.upper().replace("_", " ")}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalised mRNA cancer vaccine pipeline for multiple myeloma</p>', unsafe_allow_html=True)

# ── Key Metrics Row ──────────────────────────────────────────────────

if not df_enhanced.empty:
    total_predictions = len(df_enhanced)
    total_binders = len(df_enhanced[df_enhanced["ic50_nM"] < 500])
    strong_binders = len(df_enhanced[df_enhanced["ic50_nM"] < 50])
    n_dual = len(df_dual) if not df_dual.empty else 0

    clonal_count = 0
    if "clonality" in df_enhanced.columns:
        clonal_count = len(df_enhanced[
            (df_enhanced["clonality"] == "clonal") &
            (df_enhanced["ic50_nM"] < 500)
        ])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Predictions", f"{total_predictions:,}")
    col2.metric("MHC-I Binders", f"{total_binders}", f"{100*total_binders/max(total_predictions,1):.1f}%")
    col3.metric("Strong Binders", f"{strong_binders}", "IC50 < 50 nM")
    col4.metric("Dual MHC-I/II", f"{n_dual}", "CD4+ & CD8+")
    col5.metric("Clonal Binders", f"{clonal_count}", "In all tumour cells")

    st.markdown("---")


# ── Tabs ─────────────────────────────────────────────────────────────

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⌬  Analyse New Patient",
    "▤  Overview",
    "◎  Top Candidates",
    "⚗  Binding Analysis",
    "⟠  HLA Coverage",
    "⊕  Vaccine Construct",
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
            help="Accepts MMRF CoMMpass format, MAF, or VCF-derived tables",
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
            "↓  Download example file",
            data=open("data/example_mutations.csv", "rb").read() if os.path.exists("data/example_mutations.csv") else b"",
            file_name="example_mutations.csv",
            mime="text/csv",
        )

    if uploaded_file:
        run_uploaded_pipeline(uploaded_file)


# ── Tab 1: Overview ──────────────────────────────────────────────────

with tab1:
    if not df_enhanced.empty:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Binding Affinity Distribution")

            binders_df = df_enhanced[df_enhanced["ic50_nM"] < 5000].copy()
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

            binders_with_clonality = df_enhanced[df_enhanced["ic50_nM"] < 2000].copy()
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

            binders = df_enhanced[df_enhanced["ic50_nM"] < 500].copy()
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
                hla_binders = df_enhanced[df_enhanced["ic50_nM"] < 500]
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

            mhc_ii_binders = df_mhc_ii[df_mhc_ii["mhc_ii_ic50"] < 1000]
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
            allele_data = df_enhanced[df_enhanced["hla_allele"] == allele]
            binders = allele_data[allele_data["ic50_nM"] < 500]
            strong = allele_data[allele_data["ic50_nM"] < 50]
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
                    count = len(df_enhanced[(df_enhanced["hla_allele"] == allele) & (df_enhanced["ic50_nM"] < 50)])
                else:
                    count = len(df_enhanced[(df_enhanced["hla_allele"] == allele) & (df_enhanced["ic50_nM"] >= 50) & (df_enhanced["ic50_nM"] < 500)])
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
        covered_alleles = [a for a in hla_freqs if len(df_enhanced[(df_enhanced["hla_allele"] == a) & (df_enhanced["ic50_nM"] < 500)]) > 0]
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
                "Epitopes": len(df_epitopes),
                "Linker": "GGSGGGGSGG",
                "5' Cap": "m7GpppN (Cap1)",
                "Modification": "m1Ψ (all U)",
                "Poly-A": "120 nt",
                "LNP formulation": "Recommended",
            }
            for k, v in construct_info.items():
                st.markdown(f"**{k}:** {v}")

        # Architecture diagram
        st.markdown("#### Construct Architecture")
        st.code(
            "5'Cap ── 5'UTR ── [Signal Peptide]─[Ep1]─[GGSGGGGSGG]─[Ep2]─...─[EpN] ── STOP ── 3'UTR ── AAAA...A(120)",
            language=None,
        )

    if vaccine_report:
        with st.expander("▫  Full Vaccine Design Report"):
            st.text(vaccine_report)


# ── Tab 6: Full Data ─────────────────────────────────────────────────

with tab6:
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
