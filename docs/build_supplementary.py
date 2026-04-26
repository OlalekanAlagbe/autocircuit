#!/usr/bin/env python3
"""
build_supplementary.py — Build supplementary.html and supplementary.pdf from
docs/Supplementary-material.md.
Run: python3 docs/build_supplementary.py   (from repo root)
"""
import re, subprocess, sys, tempfile
from pathlib import Path

try:
    import yaml
    import markdown as md_lib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "markdown"])
    import yaml
    import markdown as md_lib

HERE     = Path(__file__).parent
SRC      = HERE / "Supplementary-material.md"
OUT_HTML = HERE / "supplementary.html"
OUT_PDF  = HERE / "supplementary.pdf"

SUPP_CSS = """<style>
:root{--bg:#0d1117;--bg2:#161b22;--border:#30363d;--text:#c9d1d9;--text-muted:#8b949e;
  --link:#58a6ff;--blue:#58a6ff;--green:#3fb950;--orange:#d29922;--purple:#bc8cff;}
[data-theme="light"]{--bg:#ffffff;--bg2:#f6f8fa;--border:#d0d7de;--text:#1f2328;
  --text-muted:#656d76;--link:#0969da;--blue:#0969da;--green:#1a7f37;
  --orange:#9a6700;--purple:#8250df;}
*{box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);max-width:860px;margin:0 auto;padding:2rem 1.5rem;
  font-size:16px;line-height:1.7;}
h1{font-size:1.9em;color:var(--blue);border-bottom:2px solid var(--border);padding-bottom:.5em;margin-bottom:.3em;}
h2{font-size:1.3em;color:var(--blue);border-bottom:1px solid var(--border);padding-bottom:.25em;margin-top:2em;}
h3{font-size:1.05em;color:var(--purple);margin-top:1.5em;}
p{margin:.6em 0;}
a{color:var(--link);text-decoration:none;}
a:hover{text-decoration:underline;}
code{font-family:'SFMono-Regular',Consolas,monospace;font-size:.85em;
  background:var(--bg2);border:1px solid var(--border);padding:1px 5px;border-radius:4px;color:var(--blue);}
pre{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:1em;overflow-x:auto;font-size:.82em;line-height:1.5;}
pre code{background:none;border:none;padding:0;color:var(--text);}
table{width:100%;border-collapse:collapse;font-size:.9em;margin:1em 0;}
th{background:var(--bg2);color:var(--text-muted);padding:7px 12px;border:1px solid var(--border);
  text-align:left;font-weight:600;}
td{padding:7px 12px;border:1px solid var(--border);}
tr:nth-child(even) td{background:var(--bg2);}
blockquote{border-left:3px solid var(--blue);margin:1em 0;padding:.5em 1em;
  color:var(--text-muted);background:var(--bg2);border-radius:0 4px 4px 0;}
hr{border:none;border-top:1px solid var(--border);margin:2em 0;}
.title-block{margin-bottom:2rem;}
.subtitle{font-size:1.05em;color:var(--text-muted);margin:.25em 0;}
.authors{font-size:.95em;color:var(--text-muted);margin:.5em 0;}
.date{font-size:.9em;color:var(--text-muted);}
.back-link{display:inline-block;margin-bottom:1.5rem;font-size:.9em;}
#theme-toggle{position:fixed;top:1rem;right:1rem;background:var(--bg2);border:1px solid var(--border);
  color:var(--text);border-radius:20px;padding:6px 14px;cursor:pointer;font-size:.85em;z-index:100;}
</style>"""

TOGGLE_JS = """<button id="theme-toggle" onclick="toggleTheme()">
  <span id="theme-icon">&#9728;&#65039;</span> <span id="theme-label">Light mode</span>
</button>
<script>
function toggleTheme(){
  var h=document.documentElement,d=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',d?'light':'dark');
  document.getElementById('theme-icon').textContent=d?'🌙':'☀️';
  document.getElementById('theme-label').textContent=d?'Dark mode':'Light mode';
  localStorage.setItem('theme',d?'light':'dark');
}
(function(){
  var s=localStorage.getItem('theme')||'dark';
  document.documentElement.setAttribute('data-theme',s);
  document.addEventListener('DOMContentLoaded',function(){
    var d=s==='dark';
    document.getElementById('theme-icon').textContent=d?'☀️':'🌙';
    document.getElementById('theme-label').textContent=d?'Light mode':'Dark mode';
  });
})();
</script>"""


def md_to_html(text):
    placeholders = {}

    def save_math(m):
        key = f"MATHBLOCK{len(placeholders)}ENDMATH"
        placeholders[key] = f'<div class="math-display">\\[{m.group(1)}\\]</div>'
        return f"\n\n{key}\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", save_math, text, flags=re.DOTALL)
    html = md_lib.markdown(text, extensions=["tables", "fenced_code", "nl2br", "attr_list"])
    for key, val in placeholders.items():
        html = html.replace(f"<p>{key}</p>", val)
        html = html.replace(key, val)
    return html


def build_html(meta, body):
    title    = meta.get("title", "Supplementary Material")
    subtitle = meta.get("subtitle", "")
    date_    = meta.get("date", "")
    authors  = " &nbsp;·&nbsp; ".join(
        f'<a href="{a["url"]}" target="_blank">{a["name"]}</a>' if a.get("url") else a["name"]
        for a in meta.get("authors", [])
    )
    body_html = md_to_html(body)
    body_html = body_html.replace("<table>", '<table class="data-table">')

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{SUPP_CSS}
<script>MathJax={{tex:{{inlineMath:[["$","$"],["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]}},svg:{{fontCache:"global"}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
{TOGGLE_JS}
<a href="index.html" class="back-link">← Back to paper</a>
<div class="title-block">
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <p class="authors">{authors}</p>
  <p class="date">{date_}</p>
</div>
{body_html}
</body>
</html>"""
    OUT_HTML.write_text(html)
    print(f"  ✓  {OUT_HTML}  ({OUT_HTML.stat().st_size:,} bytes)")


def build_pdf(meta, body):
    # Strip markdown comment markers and horizontal rules
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    body = re.sub(r"\n---\n", "\n\n", body)

    title    = meta.get("title", "Supplementary Material")
    subtitle = meta.get("subtitle", "")
    authors  = " · ".join(a["name"] for a in meta.get("authors", []))
    date_    = meta.get("date", "")

    title_block = f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{authors}"
date: "{date_}"
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

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as tmp:
        tmp.write(full_md)
        tmp_path = tmp.name

    cmd = [
        "pandoc", tmp_path,
        "-o", str(OUT_PDF),
        "--pdf-engine=xelatex",
        "--highlight-style=tango",
        "-V", "mainfont=DejaVu Serif",
        "-V", "monofont=DejaVu Sans Mono",
    ]
    print(f"Building {OUT_PDF} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(tmp_path).unlink()

    if result.returncode != 0:
        print("pandoc error:", result.stderr[:2000])
        sys.exit(1)
    print(f"  ✓  {OUT_PDF}  ({OUT_PDF.stat().st_size:,} bytes)")


def main():
    text = SRC.read_text()
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if m:
        meta = yaml.safe_load(m.group(1))
        body = text[m.end():]
    else:
        meta, body = {}, text

    print(f"Building supplementary from {SRC} …")
    build_html(meta, body)
    build_pdf(meta, body)


if __name__ == "__main__":
    main()
