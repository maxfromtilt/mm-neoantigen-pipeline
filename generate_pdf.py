#!/usr/bin/env python3
"""Generate PDF from the clinical report using reportlab (pure Python)."""
import os
import sys

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
except ImportError:
    print("Installing reportlab...")
    os.system(f"{sys.executable} -m pip install --user reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


# Colours
NAVY = HexColor('#1a365d')
BLUE = HexColor('#2c5282')
LIGHT_BLUE = HexColor('#ebf4ff')
LIGHT_GREY = HexColor('#f7fafc')
BORDER_GREY = HexColor('#cbd5e0')
WHITE = HexColor('#ffffff')

def build_pdf():
    doc = SimpleDocTemplate(
        "CLINICAL_REPORT.pdf",
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        'Title2', parent=styles['Title'],
        fontSize=18, textColor=NAVY, spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=12, textColor=BLUE, alignment=TA_CENTER,
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=14, textColor=BLUE, spaceBefore=20, spaceAfter=10,
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontSize=11, textColor=NAVY, spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=HexColor('#666666'),
    ))
    styles.add(ParagraphStyle(
        'CellStyle', parent=styles['Normal'],
        fontSize=8, leading=10,
    ))
    styles.add(ParagraphStyle(
        'CellBold', parent=styles['Normal'],
        fontSize=8, leading=10, fontName='Helvetica-Bold',
    ))

    story = []

    # Title
    story.append(Paragraph(
        "Computational Design of Personalised Neoantigen<br/>mRNA Vaccines for Multiple Myeloma",
        styles['Title2']
    ))
    story.append(Paragraph(
        "Using Machine Learning Binding Prediction and MMRF CoMMpass Genomic Data",
        styles['Subtitle']
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 12))

    # Authors
    story.append(Paragraph(
        "<b>Authors:</b> Robert Doran, Max Shaw", styles['Body']
    ))
    story.append(Paragraph(
        "<b>Affiliation:</b> Independent Research, Dublin, Ireland", styles['Body']
    ))
    story.append(Paragraph(
        "<b>Date:</b> 29 March 2026", styles['Body']
    ))
    story.append(Paragraph(
        "<b>Repository:</b> https://github.com/maxfromtilt/mm-neoantigen-pipeline", styles['Body']
    ))
    story.append(Paragraph(
        "<b>Status:</b> For Research Purposes Only — Not validated for clinical use", styles['Body']
    ))
    story.append(Spacer(1, 16))

    # Executive Summary box
    story.append(Paragraph("Executive Summary", styles['H2']))

    summary_text = [
        ["<b>What this is</b>", "The first open-source, MM-specific neoantigen mRNA vaccine design pipeline. BioNTech and Moderna have proprietary pipelines for their trials (melanoma, pancreatic), but no equivalent exists specifically for multiple myeloma patients outside those trials. This fills that gap."],
        ["<b>What's different</b>", "<b>MM-specific:</b> Incorporates MM driver gene biology, plasma cell expression profiles, myeloma-relevant clonality. <b>Multi-signal ranking:</b> 8 weighted factors beyond binding alone. <b>Dual MHC-I/II:</b> Identifies candidates engaging both CD8+ and CD4+ T cells. <b>Transparent:</b> Fully auditable, unlike proprietary industry pipelines. <b>Patient-ready:</b> Runs in 30 min on any patient's WES + HLA data, at no cost."],
        ["<b>Key results</b>", "2 MMRF CoMMpass patients: 156 and 140 binders, 24 and 18 strong binders (IC50 &lt; 50 nM). DIS3 I85T ranked #1 (MM driver, clonal, expressed). IDH2 R140Q at 47 nM (oncogenic hotspot, CCF 0.99). 7 dual MHC-I/MHC-II binders found."],
        ["<b>What we need</b>", "A clinician or researcher willing to review the methodology and advise whether this warrants experimental validation (ELISpot immunogenicity testing)."],
        ["<b>How to use</b>", "Visit <b>mm-vaccine.streamlit.app</b> — upload a somatic mutation CSV (gene_symbol, aa_change columns) and the pipeline analyses it automatically. Pre-computed results for 5 MMRF CoMMpass patients available to explore immediately."],
        ["<b>Code &amp; data</b>", "github.com/maxfromtilt/mm-neoantigen-pipeline"],
    ]

    summary_table_data = [[Paragraph(row[0], styles['CellBold']), Paragraph(row[1], styles['CellStyle'])] for row in summary_text]
    ts = Table(summary_table_data, colWidths=[90, 370])
    ts.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BLUE),
        ('BOX', (0, 0), (-1, -1), 1.5, BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(ts)
    story.append(Spacer(1, 16))

    # Abstract
    story.append(Paragraph("Abstract", styles['H2']))
    story.append(Paragraph(
        "We present an open-source computational pipeline for designing personalised mRNA cancer vaccines "
        "targeting neoantigens in multiple myeloma (MM). Using somatic mutation data from the MMRF CoMMpass "
        "study and MHCflurry neural network binding predictions, we demonstrate end-to-end vaccine construct "
        "design for two MM patients. The pipeline identifies 156 and 140 MHC-I binding neoantigen candidates "
        "per patient respectively, including 24 and 18 strong binders (IC50 &lt; 50 nM). Enhanced analysis "
        "incorporating MHC-II prediction, clonality estimation, gene expression filtering, and driver gene "
        "prioritisation produces clinically relevant candidate rankings. Notably, the pipeline identifies "
        "DIS3 I85T (a known MM driver) and IDH2 R140Q (a known oncogenic hotspot) as top vaccine targets. "
        "Seven dual MHC-I/MHC-II binding candidates are identified for Patient 1, suggesting potential for "
        "coordinated CD4+/CD8+ T cell responses.",
        styles['Body']
    ))

    # Introduction
    story.append(Paragraph("1. Introduction", styles['H2']))
    story.append(Paragraph(
        "Multiple myeloma is a haematological malignancy characterised by clonal plasma cell proliferation "
        "in the bone marrow. Despite significant therapeutic advances, MM remains largely incurable. "
        "Personalised neoantigen vaccines represent an emerging approach that harnesses the patient's immune "
        "system to target tumour-specific mutations.",
        styles['Body']
    ))
    story.append(Paragraph("Clinical Precedent", styles['H3']))
    story.append(Paragraph(
        "<b>BioNTech Autogene Cevumeran (BNT122):</b> Personalised mRNA neoantigen vaccine for pancreatic cancer. "
        "Phase I trial: neoantigen-specific T cell responses in 8/16 patients (Rojas et al., Nature 2023).<br/><br/>"
        "<b>Moderna mRNA-4157 (V940):</b> Combined with pembrolizumab for melanoma. Phase 2b: 44% reduction "
        "in recurrence/death vs pembrolizumab alone (Weber et al., Lancet 2024).<br/><br/>"
        "<b>BCMA-mRNA vaccine for MM:</b> Preclinical work demonstrating mRNA vaccine feasibility specifically "
        "for multiple myeloma (Dutta et al., Blood 2025).",
        styles['Body']
    ))

    # Methods
    story.append(Paragraph("2. Methods", styles['H2']))
    story.append(Paragraph(
        "Somatic mutation data from the MMRF CoMMpass study was processed through a six-stage pipeline: "
        "mutation filtering, peptide generation (8-11mer windows), MHC-I binding prediction (MHCflurry 2.1, "
        "trained on 350,000+ experimental measurements), MHC-II prediction (4 HLA-DR alleles), "
        "clonality estimation, and mRNA vaccine construct design. Vaccine constructs use established "
        "clinical methodology: m1-pseudouridine modification, Cap1 structure, tPA signal peptide, "
        "GSG flexible linkers, and lipid nanoparticle formulation.",
        styles['Body']
    ))

    # Results
    story.append(Paragraph("3. Results", styles['H2']))
    story.append(Paragraph("3.1 Overview", styles['H3']))

    # Summary table
    summary_data = [
        [Paragraph('<b>Metric</b>', styles['CellBold']),
         Paragraph('<b>Patient 1 (MMRF_1251)</b>', styles['CellBold']),
         Paragraph('<b>Patient 2 (MMRF_1274)</b>', styles['CellBold'])],
        ['Total mutations', '448', '395'],
        ['Unique after filtering', '66', '55'],
        ['Peptide candidates', '2,508', '1,988'],
        ['Total predictions', '15,048', '11,928'],
        ['MHC-I binders (IC50<500nM)', '156 (1.0%)', '140 (1.2%)'],
        ['Strong binders (IC50<50nM)', '24 (0.2%)', '18 (0.2%)'],
        ['MHC-II binders (IC50<1000nM)', '17', '28'],
        ['Dual MHC-I+II binders', '7', '2'],
        ['Driver gene mutations', 'DIS3', 'RB1'],
        ['Vaccine epitopes', '20', '20'],
        ['mRNA construct length', '1,456 nt', '1,429 nt'],
    ]

    t = Table(summary_data, colWidths=[180, 140, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Top candidates Patient 1
    story.append(Paragraph("3.2 Top Vaccine Candidates — Patient 1 (MMRF_1251)", styles['H3']))

    p1_data = [
        [Paragraph('<b>Rank</b>', styles['CellBold']),
         Paragraph('<b>Gene</b>', styles['CellBold']),
         Paragraph('<b>Mutation</b>', styles['CellBold']),
         Paragraph('<b>Peptide</b>', styles['CellBold']),
         Paragraph('<b>HLA</b>', styles['CellBold']),
         Paragraph('<b>IC50 (nM)</b>', styles['CellBold']),
         Paragraph('<b>Clonality</b>', styles['CellBold']),
         Paragraph('<b>Score</b>', styles['CellBold'])],
        ['1', 'DIS3', 'I85T', 'DPATRNVIV', 'B*07:02', '369', 'Clonal', '67.1'],
        ['2', 'DIS3', 'I85T', 'DPATRNVIV', 'B*08:01', '201', 'Clonal', '66.8'],
        ['3', 'DIS3', 'I85T', 'VLEDPATRNV', 'A*02:01', '225', 'Clonal', '66.2'],
        ['4', 'MME', 'Y579F', 'LNYGGIFMV', 'A*02:01', '120', 'Likely', '63.3'],
        ['5', 'AHR', 'N108Y', 'RYGLNLQEGEF', 'A*24:02', '181', 'Likely', '62.7'],
        ['6', 'MME', 'Y579F', 'IFMVIGHEI', 'A*24:02', '51', 'Likely', '62.6'],
        ['7', 'TTC14', 'S671L', 'EPGSVRHSTL', 'B*07:02', '70', 'Likely', '61.5'],
        ['8', 'KALRN', 'S1866I', 'SLLAARQAI', 'A*02:01', '130', 'Subclonal', '60.8'],
        ['9', 'ETV1', 'N233S', 'SPVYEHNTM', 'B*07:02', '45', 'Likely', '61.3'],
        ['10', 'RINT1', 'N574K', 'AALEVFAENK', 'A*03:01', '283', 'Likely', '61.6'],
    ]

    t2 = Table(p1_data, colWidths=[30, 40, 45, 80, 52, 42, 55, 36])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (5, 0), (7, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ('BACKGROUND', (0, 1), (-1, 3), LIGHT_BLUE),  # Highlight DIS3
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Blue rows: DIS3 I85T — known MM driver gene, clonal (CCF 0.92), confirmed expressed (14.6 TPM)</i>",
        styles['Small']
    ))

    # Top candidates Patient 2
    story.append(Paragraph("3.3 Top Vaccine Candidates — Patient 2 (MMRF_1274)", styles['H3']))

    p2_data = [
        [Paragraph('<b>Rank</b>', styles['CellBold']),
         Paragraph('<b>Gene</b>', styles['CellBold']),
         Paragraph('<b>Mutation</b>', styles['CellBold']),
         Paragraph('<b>Peptide</b>', styles['CellBold']),
         Paragraph('<b>HLA</b>', styles['CellBold']),
         Paragraph('<b>IC50 (nM)</b>', styles['CellBold']),
         Paragraph('<b>Clonality</b>', styles['CellBold']),
         Paragraph('<b>Score</b>', styles['CellBold'])],
        ['1', 'THBS1', 'V990L', 'TLNCDPGLAV', 'A*02:01', '180', 'Likely', '62.0'],
        ['2', 'CTXN3', 'L52F', 'IFLDPYRSM', 'A*24:02', '188', 'Likely', '60.7'],
        ['3', 'ATP2B4', 'V72A', 'AALEKRRQV', 'B*08:01', '154', 'Subclonal', '59.1'],
        ['4', 'SRCAP', 'P3159R', 'RPSPLGPEG', 'B*07:02', '92', 'Likely', '57.3'],
        ['5', 'SRCAP', 'P2900R', 'VPGPERLIV', 'B*07:02', '64', 'Likely', '54.7'],
        ['6', 'SYNE1', 'E4316Q', 'LVKQQTSHL', 'B*08:01', '94', 'Clonal', '54.0'],
        ['7', 'IDH2', 'R140Q', 'SPNGTIQNIL', 'B*07:02', '47', 'Clonal', '53.8'],
    ]

    t3 = Table(p2_data, colWidths=[30, 40, 50, 78, 52, 42, 55, 36])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (5, 0), (7, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ('BACKGROUND', (0, 7), (-1, 7), LIGHT_BLUE),  # Highlight IDH2
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<i>Blue row: IDH2 R140Q — known oncogenic hotspot (also AML/glioma), strong binder (47 nM), clonal (CCF 0.99)</i>",
        styles['Small']
    ))

    # Dual binders
    story.append(Paragraph("3.4 Dual MHC-I/MHC-II Binders (Patient 1)", styles['H3']))
    story.append(Paragraph(
        "These candidates engage both CD8+ cytotoxic and CD4+ helper T cells, "
        "which is associated with stronger and more durable anti-tumour immune responses:",
        styles['Body']
    ))

    dual_data = [
        [Paragraph('<b>Gene</b>', styles['CellBold']),
         Paragraph('<b>Mutation</b>', styles['CellBold']),
         Paragraph('<b>Peptide</b>', styles['CellBold']),
         Paragraph('<b>MHC-I IC50</b>', styles['CellBold']),
         Paragraph('<b>MHC-II IC50</b>', styles['CellBold']),
         Paragraph('<b>Clonality</b>', styles['CellBold'])],
        ['CCDC105', 'T427M', 'MYLEKNLDELL', '77 nM', '590 nM', 'Clonal'],
        ['TMEM131', 'G195A', 'FINTSNHAV', '22 nM', '953 nM', 'Likely'],
        ['HS3ST1', 'F177Y', 'YLVRDGRLNV', '98 nM', '411 nM', 'Likely'],
        ['SYTL3', 'V512I', 'LPIQQKLRL', '102 nM', '665 nM', 'Likely'],
    ]

    t4 = Table(dual_data, colWidths=[55, 50, 85, 60, 65, 60])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (3, 0), (4, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)

    # Discussion
    story.append(PageBreak())
    story.append(Paragraph("4. Discussion", styles['H2']))
    story.append(Paragraph(
        "Our binder detection rates (1.0–1.2%) and strong binder rates (0.15–0.2%) are consistent with "
        "published neoantigen prediction benchmarks. The enhanced scoring system successfully reranks "
        "candidates by biological relevance: DIS3 I85T, which ranked outside the top 5 by binding affinity "
        "alone, rises to #1 when clonality, expression, and driver gene status are considered.",
        styles['Body']
    ))
    story.append(Paragraph(
        "The pipeline follows the same methodological framework used by BioNTech and Moderna in their "
        "published clinical trials. The primary gaps versus clinical implementations are patient-specific "
        "HLA typing and patient-specific RNA-seq expression data, both of which the pipeline is designed "
        "to accept as inputs.",
        styles['Body']
    ))

    story.append(Paragraph("Clinical Path Forward", styles['H3']))
    story.append(Paragraph(
        "For a specific MM patient, the pipeline requires: (1) whole exome sequencing of tumour and "
        "matched normal tissue, (2) HLA typing from WES or dedicated genotyping, (3) RNA-seq of tumour "
        "(recommended), (4) pipeline execution with patient-specific inputs (~30 minutes), "
        "(5) clinical review by haematologist/oncologist, and (6) experimental validation via ELISpot "
        "for immunogenicity confirmation.",
        styles['Body']
    ))
    story.append(Paragraph(
        "Existing compassionate use and clinical trial frameworks provide regulatory paths for personalised "
        "cancer vaccine administration. In Ireland, the HPRA manages compassionate use applications.",
        styles['Body']
    ))

    # Limitations
    story.append(Paragraph("Limitations", styles['H3']))
    story.append(Paragraph(
        "• HLA alleles are population defaults; patient-specific typing would refine predictions<br/>"
        "• Gene expression is population-level MM data; patient RNA-seq recommended<br/>"
        "• MHC-II predictions use PSSM models; neural network models would improve accuracy<br/>"
        "• Clonality is estimated without patient VAF data<br/>"
        "• No experimental validation has been performed<br/>"
        "• Computational predictions require clinical and laboratory confirmation",
        styles['Body']
    ))

    # References
    story.append(Paragraph("5. Key References", styles['H2']))
    refs = [
        "1. Rojas LA et al. Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer. <i>Nature</i>. 2023;618:144-150.",
        "2. O'Donnell TJ et al. MHCflurry 2.0: Improved pan-allele prediction. <i>Cell Systems</i>. 2020;11(1):42-48.",
        "3. Dutta D et al. A BCMA-mRNA vaccine for multiple myeloma. <i>Blood</i>. 2025;146(19):2322.",
        "4. Abdollahi P et al. Advances in anti-cancer vaccines for MM. <i>Frontiers Immunol</i>. 2024;15:1411352.",
        "5. Lohr JG et al. Genetic heterogeneity in multiple myeloma. <i>Cancer Cell</i>. 2014;25(1):91-101.",
        "6. Sahin U et al. mRNA-based therapeutics. <i>Nat Rev Drug Discov</i>. 2014;13(10):759-780.",
        "7. Schumacher TN, Schreiber RD. Neoantigens in cancer immunotherapy. <i>Science</i>. 2015;348:69-74.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['Small']))
        story.append(Spacer(1, 3))

    # Disclaimer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_GREY))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report describes a computational research pipeline for research purposes only. "
        "The vaccine constructs described herein have not been experimentally validated and are not intended "
        "for clinical use. Any therapeutic application requires preclinical testing, regulatory approval, "
        "and clinical oversight by qualified medical professionals.",
        styles['Small']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Correspondence:</b> robert.doran@tilt.ie  |  "
        "<b>Repository:</b> github.com/maxfromtilt/mm-neoantigen-pipeline",
        styles['Small']
    ))

    doc.build(story)
    print("Generated: CLINICAL_REPORT.pdf")


if __name__ == "__main__":
    build_pdf()
