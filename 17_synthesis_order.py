#!/usr/bin/env python3
"""
17_synthesis_order.py — mRNA Synthesis Vendor Integration
==========================================================
Formats a vaccine construct for order with GMP mRNA synthesis vendors
(TriLink BioTechnologies, Aldevra, ChemGenesis, etc.).

Usage:
    python 17_synthesis_order.py --patient mmrf_1251
    python 17_synthesis_order.py --input output/mmrf_1251_mhcflurry/vaccine_mrna.fasta

Output:
    output/{patient}/synthesis_order.csv
    output/{patient}/synthesis_order_form.txt
    output/{patient}/synthesis_sequence.fasta
"""

import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime


# ── Vendor specifications ─────────────────────────────────────────

VENDORS = {
    "TriLink BioTechnologies": {
        "contact": "orders@trilinkbio.com",
        "website": "https://www.trilinkbio.com",
        "min_length": 500,
        "max_length": 16000,
        "modifications": [
            "m1Ψ (N1-methylpseudouridine)",
            "m5C (5-methylcytidine)",
            "Cap1 (m7GpppN)",
            "Poly-A tail (configurable)",
        ],
        "notes": "CleanCap technology. Specify Cap1, m1Ψ, poly-A tail length.",
    },
    "Aldevra": {
        "contact": "info@aldevra.com",
        "website": "https://www.aldevra.com",
        "min_length": 500,
        "max_length": 15000,
        "modifications": [
            "m1Ψ (N1-methylpseudouridine)",
            "m5C (5-methylcytidine)",
            "Cap1",
            "Custom poly-A",
        ],
        "notes": "GMP mRNA synthesis. Provide clean sequence file.",
    },
    "ChemGenesis": {
        "contact": "orders@chemgenesis.com",
        "website": "https://www.chemgenesis.com",
        "min_length": 700,
        "max_length": 14000,
        "modifications": [
            "m1Ψ substitution",
            "Cap1",
            "Uracil-free (optional)",
            "Poly-A (100-150 nt)",
        ],
        "notes": "Good for long constructs. Specify all modifications.",
    },
}


def parse_mrna_sequence(fasta_path: str) -> dict:
    """Load and parse vaccine mRNA sequence from FASTA."""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    with open(fasta_path) as f:
        content = f.read()

    # Parse FASTA
    header = None
    sequence = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            header = line[1:]
        else:
            sequence.append(line)

    seq = "".join(sequence).upper()
    return {
        "header": header,
        "sequence": seq,
        "length_nt": len(seq),
    }


def compute_sequence_stats(seq: str) -> dict:
    """Compute GC content, length, and other stats for mRNA sequence."""
    if not seq:
        return {}

    seq_upper = seq.upper()
    length = len(seq_upper)

    # Count bases
    counts = {b: seq_upper.count(b) for b in "ACGU"}
    gc_count = counts["G"] + counts["C"]
    gc_content = gc_count / length if length > 0 else 0

    # Remove poly-A tail for codon analysis
    coding_seq = seq_upper.rstrip("A")
    if len(coding_seq) < 50:
        # Too short, use whole sequence
        coding_seq = seq_upper

    # Check for problematic motifs
    problems = []

    # Homopolymer runs
    homopolymer_runs = re.findall(r"(A{7,}|C{7,}|G{7,}|U{7,})", seq_upper)
    if homopolymer_runs:
        problems.append(f"Homopolymer runs: {homopolymer_runs}")

    # Polyadenylation signal
    if "AAUAAA" in seq_upper:
        problems.append("Contains AAUAAA polyadenylation signal")

    # Stop codons in-frame (other than final)
    stop_pattern = re.findall(r"U(?:AG|UAA|UGA)", coding_seq[:-3])  # Exclude tail
    if stop_pattern:
        problems.append(f"Internal stop codons: {stop_pattern}")

    # Long GC-rich regions (>10 consecutive G/C)
    gc_runs = re.findall(r"[GC]{10,}", seq_upper)
    if gc_runs:
        problems.append(f"Long GC runs: {gc_runs}")

    # Count restriction sites (common enzymes)
    common_sites = {
        "EcoRI": "GAATTC",
        "NotI": "GCGGCCGC",
        "XhoI": "CTCGAG",
        "SpeI": "ACTAGT",
        "BamHI": "GGATCC",
    }
    sites_found = {name: pat for name, pat in common_sites.items() if pat in seq_upper}
    if sites_found:
        problems.append(f"Restriction sites: {list(sites_found.keys())}")

    return {
        "length_nt": length,
        "a_count": counts["A"],
        "c_count": counts["C"],
        "g_count": counts["G"],
        "u_count": counts["U"],
        "gc_content": round(gc_content, 4),
        "gc_percent": round(gc_content * 100, 1),
        "poly_a_tail_nt": len(seq) - len(coding_seq) if coding_seq != seq_upper else 0,
        "coding_length_nt": len(coding_seq),
        "problems": problems,
        "is_acceptable": len(problems) == 0,
    }


def format_sequence_for_vendor(seq: str, vendor_name: str) -> str:
    """
    Format the mRNA sequence as a vendor-ready string.
    TriLink/Aldevra prefer plain text sequence, 80 chars per line.
    """
    seq_clean = seq.upper().replace("T", "U")  # Ensure RNA
    lines = []
    for i in range(0, len(seq_clean), 80):
        lines.append(seq_clean[i:i+80])
    return "\n".join(lines)


def generate_order_manifest(
    patient_id: str,
    sequence_data: dict,
    stats: dict,
    vendor_name: str = "TriLink BioTechnologies",
) -> dict:
    """Build the synthesis order manifest."""
    construct_id = f"MM-VAC-{patient_id.upper()}-{datetime.now().strftime('%Y%m%d')}"

    modifications = [
        "5' Cap (Cap1, m7GpppN)",
        "All uridine → N1-methylpseudouridine (m1Ψ)",
        "5-methylcytidine (m5C) at all C positions",
        f"Poly-A tail ({stats['poly_a_tail_nt']} nt)",
    ]

    vendor = VENDORS.get(vendor_name, VENDORS["TriLink BioTechnologies"])

    manifest = {
        "construct_id": construct_id,
        "patient_id": patient_id,
        "vaccine_name": f"Personalised MM Neoantigen Vaccine ({patient_id})",
        "sequence_5prime_to_3prime": stats.get("sequence", sequence_data.get("sequence", "")),
        "length_nt": stats["length_nt"],
        "gc_content": stats["gc_content"],
        "gc_percent": stats["gc_percent"],
        "coding_length_nt": stats["coding_length_nt"],
        "poly_a_tail_nt": stats["poly_a_tail_nt"],
        "modification_notes": "; ".join(modifications),
        "recommended_upl": vendor_name,
        "vendor_notes": vendor["notes"],
        "estimated_purity": "≥ 85% (HPLC purified)",
        "quality_specs": {
            "endotoxin": "< 10 EU/mL",
            "residual solvents": "USP <467> compliant",
            "identity": "CE or mass spectrometry",
            "potency": "In vitro transfection (HEK293)",
        },
        "special_instructions": "GMP grade, clinical trial use. Attach sequence file and modification spec.",
        "vendor_contact": vendor["contact"],
        "vendor_website": vendor["website"],
    }

    return manifest


def generate_email_draft(manifest: dict, vendor_name: str) -> str:
    """Generate a pre-filled email draft (mailto: URL) for the synthesis order."""
    vendor = VENDORS.get(vendor_name, VENDORS["TriLink BioTechnologies"])
    subject = f"mRNA Synthesis Order — {manifest['construct_id']}"

    body_lines = [
        f"Dear {vendor_name} team,",
        "",
        f"I am requesting a quote for GMP mRNA synthesis of the following construct:",
        "",
        f"Construct ID: {manifest['construct_id']}",
        f"Patient: {manifest['patient_id']}",
        f"Vaccine name: {manifest['vaccine_name']}",
        f"Length: {manifest['length_nt']} nucleotides",
        f"GC content: {manifest['gc_percent']}%",
        "",
        "Modifications required:",
    ]
    for mod in manifest["modification_notes"].split("; "):
        body_lines.append(f"  - {mod}")

    body_lines.extend([
        "",
        f"Recommended UPL: {manifest['recommended_upl']}",
        f"Estimated purity: {manifest.get('estimated_purity', '≥ 85%')}",
        "",
        "Quality specifications:",
        f"  - Endotoxin: {manifest['quality_specs']['endotoxin']}",
        f"  - Identity: {manifest['quality_specs']['identity']}",
        f"  - Potency: {manifest['quality_specs']['potency']}",
        "",
        "Please find the full sequence file attached.",
        "",
        "The construct sequence is available in the attached FASTA file.",
        "Please confirm pricing and lead time.",
        "",
        "Best regards,",
        "[Your Name]",
        "[Institution]",
        "[Contact Information]",
    ])

    body = "\n".join(body_lines)

    # Encode for mailto
    import urllib.parse
    encoded_subject = urllib.parse.quote(subject)
    encoded_body = urllib.parse.quote(body)
    mailto = f"mailto:{vendor['contact']}?subject={encoded_subject}&body={encoded_body}"

    return mailto, subject, body


def write_synthesis_files(
    patient_id: str,
    sequence_data: dict,
    stats: dict,
    manifest: dict,
    output_dir: str = None,
) -> dict:
    """Write all synthesis order output files."""
    if output_dir is None:
        output_dir = f"output/{patient_id}"

    os.makedirs(output_dir, exist_ok=True)

    files_written = {}

    # 1. CSV order manifest
    csv_path = os.path.join(output_dir, "synthesis_order.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(manifest.keys()))
        writer.writeheader()
        writer.writerow(manifest)
    files_written["csv"] = csv_path
    print(f"  [+] {csv_path}")

    # 2. Formatted order text file
    form_path = os.path.join(output_dir, "synthesis_order_form.txt")
    with open(form_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  mRNA SYNTHESIS ORDER FORM\n")
        f.write(f"  Construct: {manifest['construct_id']}\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONSTRUCT DETAILS\n")
        f.write("-" * 40 + "\n")
        for key, val in manifest.items():
            if key != "sequence_5prime_to_3prime":
                f.write(f"  {key:<30} {val}\n")
        f.write("\n")

        f.write("MODIFICATIONS (required)\n")
        f.write("-" * 40 + "\n")
        for mod in manifest["modification_notes"].split("; "):
            f.write(f"  • {mod}\n")
        f.write("\n")

        f.write("QUALITY SPECIFICATIONS\n")
        f.write("-" * 40 + "\n")
        for spec, val in manifest["quality_specs"].items():
            f.write(f"  {spec:<20} {val}\n")
        f.write("\n")

        f.write("SEQUENCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Length: {stats['length_nt']} nt\n")
        f.write(f"  GC content: {stats['gc_percent']}%\n")
        f.write(f"  Coding region: {stats['coding_length_nt']} nt\n")
        f.write(f"  Poly-A tail: {stats['poly_a_tail_nt']} nt\n")
        f.write("\n")
        f.write("  Full sequence (5'→3', RNA notation):\n\n")
        seq = manifest["sequence_5prime_to_3prime"]
        for i in range(0, len(seq), 80):
            f.write(f"    {seq[i:i+80]}\n")
        f.write("\n")

        f.write("VENDOR NOTES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {manifest['vendor_notes']}\n")

    files_written["form"] = form_path
    print(f"  [+] {form_path}")

    # 3. Sequence FASTA
    fasta_path = os.path.join(output_dir, "synthesis_sequence.fasta")
    seq = manifest["sequence_5prime_to_3prime"]
    with open(fasta_path, "w") as f:
        header = f">{manifest['construct_id']} | {stats['length_nt']} nt | GC={stats['gc_percent']}% | m1Ψ modified | Cap1 | polyA({stats['poly_a_tail_nt']})"
        f.write(header + "\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
    files_written["fasta"] = fasta_path
    print(f"  [+] {fasta_path}")

    # 4. Email draft
    mailto_url, subject, body = generate_email_draft(manifest, "TriLink BioTechnologies")
    email_path = os.path.join(output_dir, "synthesis_email_draft.txt")
    with open(email_path, "w") as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"To: {VENDORS['TriLink BioTechnologies']['contact']}\n")
        f.write(f"Vendor: TriLink BioTechnologies\n")
        f.write(f"mailto: {mailto_url}\n")
        f.write("\n" + "=" * 40 + "\n\n")
        f.write(body)
    files_written["email"] = email_path
    print(f"  [+] {email_path}")

    return files_written


# ── Main pipeline ──────────────────────────────────────────────────

def run_synthesis_order(
    patient_id: str = None,
    fasta_path: str = None,
    vendor: str = "TriLink BioTechnologies",
) -> dict:
    """
    Main entry point. Processes vaccine construct for GMP synthesis.

    Args:
        patient_id: Patient ID (looks for output/{patient_id}_mhcflurry/vaccine_mrna.fasta)
        fasta_path: Direct path to FASTA file (overrides patient_id lookup)
        vendor: Vendor name (TriLink, Aldevra, ChemGenesis)

    Returns:
        dict with paths to all generated files
    """
    if fasta_path is None and patient_id is None:
        raise ValueError("Must provide either patient_id or fasta_path")

    if fasta_path is None:
        # Look in standard output locations
        for candidate in [
            f"output/{patient_id}_mhcflurry/vaccine_mrna.fasta",
            f"output/{patient_id}_enhanced/vaccine_mrna.fasta",
            f"output/{patient_id}/vaccine_mrna.fasta",
        ]:
            if os.path.exists(candidate):
                fasta_path = candidate
                break

    if fasta_path is None or not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"Vaccine FASTA not found for {patient_id}. "
            f"Tried: output/{patient_id}_mhcflurry/vaccine_mrna.fasta"
        )

    print(f"[*] Loading vaccine sequence from: {fasta_path}")
    sequence_data = parse_mrna_sequence(fasta_path)
    print(f"[*] Sequence loaded: {sequence_data['length_nt']} nt")

    print(f"[*] Computing sequence statistics...")
    stats = compute_sequence_stats(sequence_data["sequence"])
    stats["sequence"] = sequence_data["sequence"]

    print(f"[*] Sequence stats: GC={stats['gc_percent']}%, Length={stats['length_nt']} nt")
    if stats.get("problems"):
        print(f"    [!] Issues: {stats['problems']}")
    else:
        print(f"    [✓] No problematic motifs detected")

    print(f"[*] Generating order manifest for vendor: {vendor}")
    manifest = generate_order_manifest(patient_id, sequence_data, stats, vendor)

    output_dir = f"output/{patient_id}"
    print(f"[*] Writing synthesis order files to: {output_dir}/")
    files = write_synthesis_files(patient_id, sequence_data, stats, manifest, output_dir)

    print(f"\n[+] Synthesis order complete!")
    print(f"    Order form: {files['form']}")
    print(f"    CSV manifest: {files['csv']}")
    print(f"    Sequence FASTA: {files['fasta']}")
    print(f"    Email draft: {files['email']}")
    print(f"\n    Construct ID: {manifest['construct_id']}")
    print(f"    Vendor: {vendor}")
    print(f"    Contact: {VENDORS[vendor]['contact']}")

    return files


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mRNA synthesis order for a vaccine construct"
    )
    parser.add_argument("--patient", default=None, help="Patient ID (e.g. mmrf_1251)")
    parser.add_argument("--input", default=None, help="Path to vaccine_mrna.fasta file")
    parser.add_argument(
        "--vendor",
        default="TriLink BioTechnologies",
        choices=list(VENDORS.keys()),
        help="Synthesis vendor",
    )
    args = parser.parse_args()

    if not args.patient and not args.input:
        print("[!] Must provide --patient or --input")
        sys.exit(1)

    files = run_synthesis_order(
        patient_id=args.patient,
        fasta_path=args.input,
        vendor=args.vendor,
    )
    print(f"\n[✓] All files written.")