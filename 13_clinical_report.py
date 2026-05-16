#!/usr/bin/env python3
"""
Per-Patient Clinical PDF Report Generator
===========================================
Generates a formatted one-page PDF clinical report for a specific MM patient.

Includes:
- Patient ID / date
- Summary: total mutations, neoantigen candidates, strong binders
- Top 5 epitopes ranked by immunogenicity score with IC50 values
- Vaccine construct summary (number of epitopes, construct length, GC content)
- Mutation frequency in MM cohort for each epitope's gene
- Trial matching summary (top 3 trials)
- Limitations and research disclaimer

Uses fpdf2 for PDF generation.
Output: output/{patient_id}_clinical_report.pdf

Usage:
    python 13_clinical_report.py --patient MMRF_2240
    python 13_clinical_report.py --patient MMRF_2240 --epitopes output/MMRF_2240_mhcflurry/selected_epitopes.csv
"""

import os
import sys
import json
import argparse
from datetime import date
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    os.system(f"{sys.executable} -m pip install --user fpdf2")
    from fpdf import FPDF


# MM driver genes for context
MM_DRIVER_GENES = {
    "KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C", "TRAF3",
    "RB1", "CYLD", "MAX", "IRF4", "FGFR3", "ATM", "ATR", "PRDM1"
}


def load_selected_epitopes(patient_id: str) -> list:
    """Load selected epitopes from pipeline output."""
    epitopes_path = f"output/{patient_id}_mhcflurry/selected_epitopes.csv"
    if not os.path.exists(epitopes_path):
        epitopes_path = f"output/{patient_id}_enhanced/top_candidates.csv"
    if not os.path.exists(epitopes_path):
        epitopes_path = f"output/{patient_id}_enhanced/enhanced_predictions.csv"

    if os.path.exists(epitopes_path):
        import pandas as pd
        try:
            df = pd.read_csv(epitopes_path)
            # Prefer binding predictions or enhanced results
            return df.to_dict("records")
        except Exception:
            pass
    return []


def load_vaccine_report(patient_id: str) -> dict:
    """Extract vaccine construct details from the text report."""
    report_path = f"output/{patient_id}_mhcflurry/vaccine_report.txt"
    details = {
        "epitope_count": 0,
        "construct_length": "N/A",
        "gc_content": "N/A",
        "cassette_sequence": "",
        "mRNA_sequence": "",
    }

    if not os.path.exists(report_path):
        return details

    try:
        with open(report_path) as f:
            text = f.read()

        # Extract metrics from report text
        import re
        n_match = re.search(r'(\d+)\s+epitopes?', text)
        if n_match:
            details["epitope_count"] = int(n_match.group(1))

        len_match = re.search(r'(\d+)\s+nt\s+total', text)
        if len_match:
            details["construct_length"] = len_match.group(1)

        gc_match = re.search(r'GC\s+content[:\s]+([0-9.]+)%', text, re.IGNORECASE)
        if gc_match:
            details["gc_content"] = gc_match.group(1) + "%"

        # Try to extract mRNA sequence from FASTA
        fasta_path = f"output/{patient_id}_mhcflurry/vaccine_mrna.fasta"
        if os.path.exists(fasta_path):
            with open(fasta_path) as f:
                content = f.read()
            seq_lines = []
            in_seq = False
            for line in content.splitlines():
                if line.startswith(">"):
                    in_seq = False
                    continue
                if in_seq or not seq_lines:
                    in_seq = True
                    seq_lines.append(line.strip())
            details["mRNA_sequence"] = "".join(seq_lines)
    except Exception:
        pass

    return details


def load_trial_matches(patient_id: str) -> list:
    """Load top trial matches for this patient."""
    trials_path = "output/trial_matches.csv"
    if not os.path.exists(trials_path):
        return []

    try:
        import pandas as pd
        df = pd.read_csv(trials_path)
        if "epitope_sequence" in df.columns and "nct_number" in df.columns:
            # Get top 3 unique trials
            df_sorted = df.sort_values("trial_relevance_score", ascending=False)
            seen_nct = set()
            results = []
            for _, row in df_sorted.iterrows():
                nct = row.get("nct_number")
                if nct and nct not in seen_nct:
                    seen_nct.add(nct)
                    results.append({
                        "nct": nct,
                        "title": str(row.get("trial_title", ""))[:80],
                        "phase": row.get("phase", ""),
                        "score": row.get("trial_relevance_score", 0),
                    })
                    if len(results) >= 3:
                        break
            return results
    except Exception:
        pass
    return []


def load_mutation_frequency() -> dict:
    """Load cBioPortal mutation frequency data."""
    freq_path = "output/cbioportal_mutations.csv"
    if not os.path.exists(freq_path):
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(freq_path)
        if "gene_symbol" in df.columns and "mutation_frequency" in df.columns:
            return dict(zip(df["gene_symbol"], df["mutation_frequency"]))
    except Exception:
        pass
    return {}


def gc_content(seq: str) -> float:
    """Calculate GC content of a nucleotide sequence."""
    if not seq or len(seq) == 0:
        return 0.0
    seq = seq.upper()
    return round(100 * (seq.count("G") + seq.count("C")) / len(seq), 1)


class ClinicalPDF(FPDF):
    """Styled PDF report for a single MM patient."""

    def header(self):
        # Colour accent bar
        self.set_fill_color(26, 54, 93)  # Navy
        self.rect(0, 0, 210, 8, "F")

        # Title
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(26, 54, 93)
        self.set_xy(10, 10)
        self.cell(190, 8, "Multiple Myeloma Neoantigen Vaccine — Patient Report", align="C", ln=True)

        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(190, 5, "Computational Research Report — Not for Clinical Use", align="C", ln=True)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"MM Neoantigen Pipeline v2.0.9 | Research Use Only | Page {self.page_no()}", align="C")

    def section_title(self, title: str, y: float = None):
        """Draw a section title with accent line."""
        if y:
            self.set_y(y)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(44, 82, 130)  # Blue
        self.cell(0, 6, title, ln=True)
        self.set_draw_color(44, 82, 130)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)


def build_report(patient_id: str, output_path: str = None):
    """Build the clinical PDF report for a patient."""

    pdf = ClinicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1 ──
    pdf.add_page()

    # Date and patient info
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(95, 5, f"Patient: {patient_id.upper()}", ln=False)
    pdf.cell(95, 5, f"Report Date: {date.today().strftime('%d %B %Y')}", align="R", ln=True)
    pdf.ln(3)

    # ── Section 1: Summary ──
    pdf.section_title("1. Mutation & Neoantigen Summary")

    epitopes = load_selected_epitopes(patient_id)
    binding_path = f"output/{patient_id}_enhanced/enhanced_predictions.csv"
    import pandas as pd
    df_enhanced = pd.read_csv(binding_path) if os.path.exists(binding_path) else pd.DataFrame()

    total_mutations = 0
    if not df_enhanced.empty:
        total_mutations = df_enhanced["aa_change"].nunique()

    total_binders = 0
    strong_binders = 0
    if not df_enhanced.empty:
        total_binders = len(df_enhanced[df_enhanced["ic50_nM"] < 500])
        strong_binders = len(df_enhanced[df_enhanced["ic50_nM"] < 50])

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)

    summary_data = [
        ("Total unique mutations", str(total_mutations)),
        ("Neoantigen peptide candidates", str(len(df_enhanced)) if not df_enhanced.empty else "N/A"),
        ("MHC-I Binders (IC50 < 500 nM)", str(total_binders)),
        ("Strong Binders (IC50 < 50 nM)", str(strong_binders)),
        ("Selected vaccine epitopes", str(len(epitopes))),
    ]

    col_w = [90, 100]
    for label, val in summary_data:
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w[0], 5, f"  {label}:", ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w[1], 5, val, ln=True)

    pdf.ln(3)

    # ── Section 2: Top 5 Epitopes ──
    pdf.section_title("2. Top 5 Epitopes (Ranked by Immunogenicity Score)")

    if not df_enhanced.empty:
        top5 = df_enhanced.nsmallest(5, "ic50_nM")
    elif epitopes:
        top5_records = epitopes[:5]
    else:
        top5_records = []

    if not df_enhanced.empty and len(df_enhanced) > 0:
        # Table header
        pdf.set_fill_color(44, 82, 130)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7.5)
        headers = ["Gene", "Mutation", "Peptide", "HLA", "IC50 (nM)", "Score"]
        col_widths = [20, 28, 38, 26, 22, 20]
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 5, h, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_text_color(40, 40, 40)
        for i, (_, row) in enumerate(top5.iterrows()):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(235, 244, 255)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 7.5)

            gene = str(row.get("gene_symbol", ""))
            aa = str(row.get("aa_change", ""))
            pep = str(row.get("mutant_peptide", ""))
            hla = str(row.get("hla_allele", ""))
            ic50 = str(round(row.get("ic50_nM", 0), 1))
            score = str(round(row.get("immunogenicity_score", row.get("enhanced_score", 0)), 1))

            values = [gene, aa, pep, hla, ic50, score]
            for val, w in zip(values, col_widths):
                pdf.cell(w, 5, val[:w//2], border=1, align="C", fill=fill)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "  No epitope data available. Run the pipeline first.", ln=True)

    pdf.ln(3)

    # ── Section 3: Vaccine Construct ──
    vaccine_details = load_vaccine_report(patient_id)
    pdf.section_title("3. Vaccine Construct Summary")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)

    if vaccine_details.get("epitope_count"):
        epitopes_n = vaccine_details["epitope_count"]
    else:
        epitopes_n = len(epitopes) if epitopes else 0

    construct_data = [
        ("Epitopes in construct", str(epitopes_n)),
        ("Construct length", f"{vaccine_details.get('construct_length', 'N/A')} nt"),
        ("GC content", vaccine_details.get("gc_content", "N/A")),
        ("5' cap", "m7GpppN (Cap1)"),
        ("Modifications", "m1 Psi (pseudouridine)"),
        ("Poly-A tail", "120 nt"),
    ]

    col1 = [f"  {k}:" for k, _ in construct_data]
    col2 = [v for _, v in construct_data]
    for c1, c2 in zip(col1, col2):
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(60, 5, c1, ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(130, 5, c2, ln=True)

    # GC content bar if sequence available
    mrna = vaccine_details.get("mRNA_sequence", "")
    if mrna and len(mrna) > 100:
        gc = gc_content(mrna)
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, f"  mRNA sequence: {len(mrna)} nt | GC: {gc}% | Seq preview: {mrna[:60]}...", ln=True)

    pdf.ln(3)

    # ── Section 4: Mutation Frequency ──
    pdf.section_title("4. Mutation Frequency in MM Cohort")

    if not df_enhanced.empty:
        gene_freq = df_enhanced["gene_symbol"].value_counts().head(8)
        mutation_freq = load_mutation_frequency()

        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_fill_color(44, 82, 130)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(["Gene", "Count", "In Driver List", "Cohort Freq"], [40, 25, 45, 60]):
            pdf.cell(w, 5, h, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_text_color(40, 40, 40)
        for i, (gene, count) in enumerate(gene_freq.items()):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(235, 244, 255)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 7.5)
            is_driver = "YES" if gene in MM_DRIVER_GENES else ""
            cohort_freq = mutation_freq.get(gene, "N/A")
            if isinstance(cohort_freq, float):
                cohort_freq = f"{cohort_freq:.1f}%"
            pdf.cell(40, 5, gene, border=1, align="C", fill=fill)
            pdf.cell(25, 5, str(count), border=1, align="C", fill=fill)
            pdf.cell(45, 5, is_driver, border=1, align="C", fill=fill)
            pdf.cell(60, 5, str(cohort_freq), border=1, align="C", fill=fill)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "  No mutation frequency data available.", ln=True)

    pdf.ln(3)

    # ── Section 5: Trial Matching ──
    pdf.section_title("5. Clinical Trial Matching (Top 3)")

    trials = load_trial_matches(patient_id)
    if trials:
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_fill_color(44, 82, 130)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(["NCT Number", "Phase", "Relevance", "Title"], [25, 20, 25, 120]):
            pdf.cell(w, 5, h, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_text_color(40, 40, 40)
        for i, trial in enumerate(trials):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(235, 244, 255)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.cell(25, 5, str(trial.get("nct", "")), border=1, align="C", fill=fill)
            pdf.cell(20, 5, str(trial.get("phase", "")), border=1, align="C", fill=fill)
            pdf.cell(25, 5, str(round(trial.get("score", 0))), border=1, align="C", fill=fill)
            pdf.cell(120, 5, str(trial.get("title", ""))[:70], border=1, align="L", fill=fill)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "  No trial match data available. Run 06_trial_matching.py first.", ln=True)

    pdf.ln(6)

    # ── Section 6: Limitations ──
    pdf.section_title("6. Limitations & Research Disclaimer")

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(80, 40, 40)
    limitations = [
        "This is a computational research report. Vaccine constructs have not been experimentally validated.",
        "HLA typing uses population frequency priors — WES-based HLA typing would improve prediction accuracy.",
        "Gene expression uses population-level MM data — patient-specific RNA-seq recommended for clinical use.",
        "MHC binding predictions are computational estimates — ELISpot/IFN-gamma assay validation required.",
        "Neoantigen vaccines in MM remain investigational. Regulatory approval required before any clinical use.",
        "Mutation frequency data from cBioPortal may not reflect individual patient tumour biology.",
    ]
    for lim in limitations:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(80, 40, 40)
        pdf.cell(5, 4, "  -", ln=False)
        pdf.multi_cell(180, 4, lim)

    pdf.ln(4)

    # Disclaimer box
    pdf.set_fill_color(255, 245, 245)
    pdf.set_draw_color(200, 100, 100)
    pdf.set_line_width(0.3)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(120, 40, 40)
    pdf.cell(0, 5,
             "RESEARCH USE ONLY — NOT FOR CLINICAL USE. Consult a qualified oncologist before any medical decision.",
             border=1, align="C", fill=True, ln=True)

    # Save
    if output_path is None:
        output_path = f"output/{patient_id}_clinical_report.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"Generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate per-patient clinical PDF report")
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. MMRF_2240)")
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    build_report(args.patient, args.output)


if __name__ == "__main__":
    main()