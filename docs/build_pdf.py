#!/usr/bin/env python3
"""
build_pdf.py — Build docs/paper.pdf from docs/content.md using pandoc.
Run: python3 docs/build_pdf.py   (from repo root)
"""
import re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
SRC  = HERE / "content.md"
OUT  = HERE / "paper.pdf"

def main():
    text = SRC.read_text()

    # Strip YAML frontmatter; build a clean title block for pandoc
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if m:
        import yaml
        meta = yaml.safe_load(m.group(1))
        body = text[m.end():]
    else:
        meta, body = {}, text

    # Remove slide / figure comment markers
    body = re.sub(r'<!--\s*(slide|figure):[^>]*-->', '', body)
    # Remove standalone horizontal rules that separate sections visually
    body = re.sub(r'\n---\n', '\n\n', body)
    # Strip bibtex fenced block (already in references section)
    body = re.sub(r'```bibtex.*?```', '', body, flags=re.DOTALL)

    authors = ' · '.join(a['name'] for a in meta.get('authors', []))
    title_block = f"""---
title: "{meta.get('title', '')}"
subtitle: "{meta.get('subtitle', '')}"
author: "{authors}"
date: "{meta.get('date', '')}"
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
header-includes:
  - \\usepackage{{amsmath}}
  - \\usepackage{{amssymb}}
---

"""
    full_md = title_block + body

    with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as tmp:
        tmp.write(full_md)
        tmp_path = tmp.name

    cmd = [
        'pandoc', tmp_path,
        '-o', str(OUT),
        '--pdf-engine=xelatex',
        '--highlight-style=tango',
        '-V', 'mainfont=DejaVu Serif',
        '-V', 'monofont=DejaVu Sans Mono',
    ]
    print(f"Building {OUT} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink()

    if result.returncode != 0:
        print("pandoc error:", result.stderr[:2000])
        sys.exit(1)
    print(f"  ✓  {OUT}  ({OUT.stat().st_size:,} bytes)")

if __name__ == '__main__':
    main()
