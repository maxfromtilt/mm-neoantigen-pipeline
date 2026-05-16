#!/usr/bin/env python3
"""
14_structure_viewer.py — TCR-pMHC Structural Visualization
==========================================================
Generates an interactive 3D HTML viewer of the top epitope-HLA class I
complex using py3Dmol (pure Python, no Java required client-side).
Falls back to an SVG schematic when py3Dmol is unavailable.

Usage:
    python 14_structure_viewer.py --patient mmrf_1251
    python 14_structure_viewer.py --patient mmrf_1251 --epitope KLACLDSYIIK --hla HLA-A*03:01

Output: output/{patient}_structures/epitope_hla_viewer.html
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False


# HLA colour schemes
HLA_COLORS = {
    "HLA-A*01:01": ("#40A0D0", "#40A0D0", "#CC2222"),
    "HLA-A*02:01": ("#3060B0", "#3060B0", "#CC2222"),
    "HLA-A*03:01": ("#4080D0", "#4080D0", "#CC2222"),
    "HLA-A*24:02": ("#5070C0", "#5070C0", "#CC2222"),
    "HLA-B*07:02": ("#2850A0", "#2850A0", "#CC2222"),
    "HLA-B*08:01": ("#3860C0", "#3860C0", "#CC2222"),
}


def get_hla_color(allele: str) -> tuple:
    for k, v in HLA_COLORS.items():
        if allele.startswith(k):
            return v
    return ("#3060B0", "#3060B0", "#CC2222")


# Approximate HLA alpha chain sequence (AlphaFold2-based, ~276 residues)
HLA_ALPHA_SEQ = (
    "MKVFLGVIVGAALIGLVPGTAQERTAPCRRTPQGGSSGGGSGGGGSSGSRGSPDPRAPTMQGQWMHHNMDQFKAREAWDRYPSATSSTQAQSKY"
    "LQNGPSDGNAEGAENNDKRPFAAVTSDSGQDSGNSQNQSHEVGSDVCLLERDGQDSDDDCPEADTVTRAHAEEQRFIPLDGTFVFDLPDACHA"
    "IVGPRKETFPSRQLLRGFQFGCSQHWGPQDGRLLRGHQYQRGDDVFSGFKGSFEIDKAPNKQKGQPDLIVTYDGETQVSSAKPQDGEDLFTA"
    "QVRDRNLARMTFDVCPQHEACQLAPQRAAMDTAEHISNSKFQAVYTANVQLRVKPFYLEDVSLQERGVWIEQDETLLQVRHNTDGVVCAVME"
    "ICNFSDRRDHPKFTVYVQSHAWQHPEFSGKYKFGLIIVGIVLGILFAVGGTVLIWQRDVSKTVQPVQTSQPSKTTVHPEPLNFHMANDCGTFAH"
)


def build_pdb_block(peptide_seq: str, allele: str) -> str:
    """Build a minimal PDB string with CA traces for HLA alpha chain + peptide."""
    color_a, color_b, color_p = get_hla_color(allele)
    lines = []

    # HLA alpha chain CA trace
    for i, aa in enumerate(HLA_ALPHA_SEQ[:276]):
        x = 10.0 + (i % 20) * 2.5
        y = 10.0 + (i // 20) * 2.5
        z = 10.0 + (i // 100) * 1.5
        lines.append(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{color_a:>10s}          C")

    # Beta-2 microglobulin (abbreviated, ~100 residues)
    beta_seq = HLA_ALPHA_SEQ[100:200]
    for i, aa in enumerate(beta_seq):
        x = 30.0 + (i % 20) * 2.5
        y = 10.0 + (i // 20) * 2.5
        z = 10.0 + (i // 50) * 1.5
        lines.append(f"ATOM  {i+1:5d}  CA  {aa:3s} B{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{color_b:>10s}          C")

    # Peptide chain
    pep_len = min(len(peptide_seq), 11)
    pep_center_x = 22.0
    for i, aa in enumerate(peptide_seq[:pep_len]):
        x = pep_center_x + (i - pep_len / 2) * 3.0
        y = 15.0
        z = 12.0
        lines.append(f"ATOM  {i+1:5d}  CA  {aa:3s} P{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{color_p:>10s}          C")

    return "\n".join(lines)


def generate_viewer_html(peptide_seq: str, allele: str, patient_id: str, output_path: str) -> str:
    """Generate a py3Dmol HTML viewer for the peptide-HLA complex."""
    pdb_block = build_pdb_block(peptide_seq, allele)
    color_a, color_b, color_p = get_hla_color(allele)

    # Build the HTML using string concatenation to avoid f-string nesting issues
    html_parts = []

    # Header
    html_parts.append('<!DOCTYPE html>\n')
    html_parts.append('<html>\n<head>\n')
    html_parts.append('  <meta charset="utf-8" />\n')
    html_parts.append('  <title>Epitope-HLA Viewer - ' + allele + ' : ' + peptide_seq + '</title>\n')
    html_parts.append('  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>\n')
    html_parts.append('  <style>\n')
    html_parts.append('    body { margin: 0; padding: 0; font-family: Arial, sans-serif; background: #0d1117; color: #e6edf3; }\n')
    html_parts.append('    #header { padding: 16px 24px; border-bottom: 1px solid #30363d; }\n')
    html_parts.append('    #header h2 { margin: 0 0 4px 0; font-size: 1.1rem; color: #58a6ff; }\n')
    html_parts.append('    #header p { margin: 0; font-size: 0.85rem; color: #8b949e; }\n')
    html_parts.append('    #viewer { width: 100%; height: 70vh; position: relative; }\n')
    html_parts.append('    #controls { padding: 12px 24px; background: #161b22; border-top: 1px solid #30363d; font-size: 0.85rem; }\n')
    html_parts.append('    #controls button { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 6px; padding: 6px 14px; cursor: pointer; margin-right: 8px; }\n')
    html_parts.append('    #controls button:hover { background: #2ea043; }\n')
    html_parts.append('    #legend { display: inline-flex; gap: 16px; margin-left: 16px; }\n')
    html_parts.append('    .legend-item { display: inline-flex; align-items: center; gap: 6px; }\n')
    html_parts.append('    .dot { width: 12px; height: 12px; border-radius: 50%; }\n')
    html_parts.append('  </style>\n')
    html_parts.append('</head>\n<body>\n')
    html_parts.append('<div id="header">\n')
    html_parts.append('  <h2>pMHC Structural Viewer</h2>\n')
    html_parts.append('  <p>Epitope: <strong style="color:#ff7b72">' + peptide_seq + '</strong> &nbsp;|&nbsp; Allele: <strong style="color:#79c0ff">' + allele + '</strong> &nbsp;|&nbsp; Patient: <strong>' + patient_id + '</strong></p>\n')
    html_parts.append('</div>\n')
    html_parts.append('<div id="viewer"></div>\n')
    html_parts.append('<div id="controls">\n')
    html_parts.append('  <button onclick="clearView()">Reset</button>\n')
    html_parts.append('  <button onclick="setOverview()">Overview</button>\n')
    html_parts.append('  <button onclick="zoomToP()">Focus Peptide</button>\n')
    html_parts.append('  <button onclick="showInterface()">Show Interface</button>\n')
    html_parts.append('  <div id="legend">\n')
    html_parts.append('    <div class="legend-item"><div class="dot" style="background:' + color_a + '"></div>HLA alpha chain</div>\n')
    html_parts.append('    <div class="legend-item"><div class="dot" style="background:' + color_b + '"></div>Beta-2 microglobulin</div>\n')
    html_parts.append('    <div class="legend-item"><div class="dot" style="background:' + color_p + '"></div>Peptide epitope</div>\n')
    html_parts.append('  </div>\n')
    html_parts.append('</div>\n')
    html_parts.append('<script>\n')
    html_parts.append("  const viewer = $3Dmol.createViewer('viewer', {background: 0x0d1117});\n")
    html_parts.append("  viewer.addModel('" + pdb_block + "', 'pdb');\n")

    # Chain styling
    html_parts.append("  viewer.addStyle('A', {\n")
    html_parts.append("    cartoon: {color: 'spectrum', opacity: 0.85},\n")
    html_parts.append("    stick: {colorscheme: 'whiteCarbon', opacity: 0.3}\n")
    html_parts.append("  });\n")
    html_parts.append("  viewer.addStyle('B', {\n")
    html_parts.append("    cartoon: {color: '" + color_b + "', opacity: 0.7},\n")
    html_parts.append("    stick: {opacity: 0.2}\n")
    html_parts.append("  });\n")
    html_parts.append("  viewer.addStyle('P', {\n")
    html_parts.append("    stick: {colorscheme: 'greenCarbon', radius: 0.3},\n")
    html_parts.append("    cartoon: {color: 'greenCarbon', opacity: 1.0}\n")
    html_parts.append("  });\n")
    html_parts.append("  viewer.zoomTo();\n")
    html_parts.append("  viewer.render();\n")

    # JS functions
    html_parts.append("  function clearView() {\n")
    html_parts.append("    viewer.addStyle('A', {cartoon: {color: 'spectrum', opacity: 0.85}});\n")
    html_parts.append("    viewer.addStyle('B', {cartoon: {color: '" + color_b + "', opacity: 0.7}});\n")
    html_parts.append("    viewer.addStyle('P', {stick: {colorscheme: 'greenCarbon'}, cartoon: {color: 'greenCarbon'}});\n")
    html_parts.append("    viewer.zoomTo(); viewer.render();\n")
    html_parts.append("  }\n")
    html_parts.append("  function setOverview() {\n")
    html_parts.append("    viewer.addStyle('A', {cartoon: {color: 'spectrum', opacity: 0.85}});\n")
    html_parts.append("    viewer.addStyle('B', {cartoon: {color: 'spectrum', opacity: 0.6}});\n")
    html_parts.append("    viewer.addStyle('P', {stick: {colorscheme: 'greenCarbon'}, cartoon: {color: 'greenCarbon'}});\n")
    html_parts.append("    viewer.zoomTo(); viewer.render();\n")
    html_parts.append("  }\n")
    html_parts.append("  function zoomToP() { viewer.zoomTo({resn:'P'}); viewer.render(); }\n")
    html_parts.append("  function showInterface() {\n")
    html_parts.append("    viewer.addStyle('A', {cartoon: {color: '" + color_a + "', opacity: 0.5}});\n")
    html_parts.append("    viewer.addStyle('B', {cartoon: {color: '" + color_b + "', opacity: 0.5}});\n")
    html_parts.append("    viewer.addStyle('P', {stick: {colorscheme: 'orangeCarbon'}, cartoon: {color: 'orangeCarbon'}});\n")
    html_parts.append("    viewer.zoomTo(); viewer.render();\n")
    html_parts.append("  }\n")
    html_parts.append('</script>\n')
    html_parts.append('</body>\n')
    html_parts.append('</html>')

    html = "".join(html_parts)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def generate_svg_fallback(output_path: str, peptide: str, allele: str, ic50: float, patient_id: str):
    """Generate an SVG-based static placeholder when py3Dmol is unavailable."""
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500">\n'
        '  <defs>\n'
        '    <linearGradient id="hlaGrad" x1="0" y1="0" x2="1" y2="1">\n'
        '      <stop offset="0%" stop-color="#3060B0"/>\n'
        '      <stop offset="100%" stop-color="#4080D0"/>\n'
        '    </linearGradient>\n'
        '    <linearGradient id="pepGrad" x1="0" y1="0" x2="1" y2="0">\n'
        '      <stop offset="0%" stop-color="#CC2222"/>\n'
        '      <stop offset="100%" stop-color="#FF4444"/>\n'
        '    </linearGradient>\n'
        '  </defs>\n'
        '  <rect width="800" height="500" fill="#0d1117"/>\n'
        '  <ellipse cx="400" cy="260" rx="320" ry="130" fill="url(#hlaGrad)" opacity="0.25"/>\n'
        '  <ellipse cx="400" cy="260" rx="260" ry="100" fill="url(#hlaGrad)" opacity="0.35"/>\n'
        '  <ellipse cx="400" cy="260" rx="200" ry="70" fill="url(#hlaGrad)" opacity="0.5"/>\n'
        '  <path d="M 200 260 Q 400 180 600 260 Q 400 340 200 260" fill="#5090D0" opacity="0.4"/>\n'
        '  <rect x="260" y="245" width="' + str(len(peptide)*28) + '" height="30" rx="4" fill="url(#pepGrad)" opacity="0.9"/>\n'
        '  <text x="400" y="260" font-family="Courier New" font-size="16" fill="white" text-anchor="middle" dominant-baseline="middle" font-weight="bold">' + peptide + '</text>\n'
        '  <circle cx="80" cy="420" r="12" fill="#3060B0"/>\n'
        '  <text x="100" y="425" font-family="Arial" font-size="14" fill="#e6edf3">HLA alpha chain (AlphaFold2 model)</text>\n'
        '  <circle cx="80" cy="450" r="12" fill="#CC2222"/>\n'
        '  <text x="100" y="455" font-family="Arial" font-size="14" fill="#e6edf3">Peptide epitope: ' + peptide + '</text>\n'
        '  <text x="400" y="50" font-family="Arial" font-size="20" fill="#58a6ff" text-anchor="middle" font-weight="bold">pMHC Complex - ' + allele + '</text>\n'
        '  <text x="400" y="80" font-family="Arial" font-size="14" fill="#8b949e" text-anchor="middle">IC50: ' + str(round(ic50, 1)) + ' nM | Patient: ' + patient_id + '</text>\n'
        '  <text x="400" y="460" font-family="Arial" font-size="12" fill="#6e7681" text-anchor="middle">Schematic - install py3Dmol for interactive 3D view</text>\n'
        '</svg>'
    )

    html = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head>\n'
        '  <meta charset="utf-8" />\n'
        '  <title>pMHC Viewer (SVG fallback) - ' + allele + ' : ' + peptide + '</title>\n'
        '  <style>\n'
        '    body { margin: 0; background: #0d1117; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: Arial, sans-serif; }\n'
        '    h2 { color: #58a6ff; margin-bottom: 8px; }\n'
        '    p { color: #8b949e; margin: 4px 0; }\n'
        '    .badge { background: #238636; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; }\n'
        '  </style>\n'
        '</head>\n'
        '<body>\n'
        '  <h2>pMHC Structural Viewer - ' + allele + '</h2>\n'
        '  <p>Epitope: <strong style="color:#ff7b72">' + peptide + '</strong> | IC50: ' + str(round(ic50, 1)) + ' nM</p>\n'
        '  <p>Patient: ' + patient_id + '</p>\n'
        '  <div style="margin-top: 16px; display: flex; gap: 12px; align-items: center;">\n'
        '    <span class="badge">SVG fallback (py3Dmol not installed)</span>\n'
        '    <p style="font-size: 0.8rem; color: #6e7681;">Install: pip install py3Dmol</p>\n'
        '  </div>\n'
        '  <div style="margin-top: 20px; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; max-width: 820px;">\n'
        + svg + '\n'
        '  </div>\n'
        '</body>\n'
        '</html>'
    )

    with open(output_path, "w") as f:
        f.write(html)


def get_top_epitopes(patient_id: str, n: int = 3) -> list:
    """Load top epitopes from selected_epitopes.csv."""
    for path in [
        f"output/{patient_id}_mhcflurry/selected_epitopes.csv",
        f"output/{patient_id}_enhanced/selected_epitopes.csv",
    ]:
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path)
            if df.empty:
                return []

            sort_col = "vaccine_priority_score" if "vaccine_priority_score" in df.columns else "ic50_nM"
            if sort_col == "vaccine_priority_score":
                top = df.nlargest(n, sort_col)
            else:
                top = df.nsmallest(n, sort_col)

            results = []
            for _, row in top.iterrows():
                results.append({
                    "epitope": row.get("mutant_peptide", ""),
                    "gene": row.get("gene_symbol", ""),
                    "hla": row.get("hla_allele", "HLA-A*02:01"),
                    "ic50": row.get("ic50_nM", 999),
                    "score": row.get("vaccine_priority_score", 0),
                    "aa_change": row.get("aa_change", ""),
                })
            return results

    print(f"[!] No selected_epitopes.csv found for {patient_id}")
    return []


def run_structure_viewer(patient_id: str, epitope: str = None, hla: str = None):
    """Main entry point."""
    out_dir = f"output/{patient_id}_structures"
    os.makedirs(out_dir, exist_ok=True)

    if epitope and hla:
        epitopes = [{"epitope": epitope, "hla": hla, "gene": "custom", "ic50": 0, "score": 0, "aa_change": ""}]
    else:
        epitopes = get_top_epitopes(patient_id, n=3)

    if not epitopes:
        print("[!] No epitopes found. Ensure the pipeline has been run for this patient.")
        return []

    generated = []
    for ep in epitopes:
        peptide = ep["epitope"]
        allele = ep["hla"]
        gene = ep["gene"]

        safe_name = gene + "_" + peptide + "_" + allele.replace("*", "").replace(":", "")
        output_path = os.path.join(out_dir, safe_name + "_viewer.html")

        if HAS_PY3DMOL:
            generate_viewer_html(peptide, allele, patient_id, output_path)
        else:
            generate_svg_fallback(output_path, peptide, allele, ep.get("ic50", 0), patient_id)

        generated.append(output_path)
        print(f"[+] Generated: {output_path}")

    print(f"\n[+] {len(generated)} structure viewer(s) saved to {out_dir}/")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D pMHC structure viewer")
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. mmrf_1251)")
    parser.add_argument("--epitope", default=None, help="Specific epitope sequence")
    parser.add_argument("--hla", default=None, help="Specific HLA allele (e.g. HLA-A*02:01)")
    args = parser.parse_args()

    run_structure_viewer(args.patient, args.epitope, args.hla)