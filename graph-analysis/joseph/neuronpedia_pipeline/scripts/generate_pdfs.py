#!/usr/bin/env python3
"""Generate PDFs of the paper and companion documents using reportlab.

Usage:
    python scripts/generate_pdfs.py
"""

import re
import sys
import io
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether, Preformatted,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from PIL import Image as PILImage

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE = Path(__file__).parent.parent
FIGURES_DIR = BASE / 'data' / 'stage_2_figures'
OUTPUT_DIR = BASE / 'docs' / 'papers' / 'pdf'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='PaperTitle', parent=styles['Title'],
        fontSize=18, leading=22, spaceAfter=12, alignment=TA_CENTER,
        textColor=HexColor('#1a1a1a'),
    ))
    styles.add(ParagraphStyle(
        name='PaperH1', parent=styles['Heading1'],
        fontSize=14, leading=18, spaceBefore=16, spaceAfter=8,
        textColor=HexColor('#1a1a1a'), keepWithNext=1,
    ))
    styles.add(ParagraphStyle(
        name='PaperH2', parent=styles['Heading2'],
        fontSize=12, leading=16, spaceBefore=12, spaceAfter=6,
        textColor=HexColor('#2a2a2a'), keepWithNext=1,
    ))
    styles.add(ParagraphStyle(
        name='PaperH3', parent=styles['Heading3'],
        fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
        textColor=HexColor('#3a3a3a'), keepWithNext=1,
    ))
    styles.add(ParagraphStyle(
        name='PaperBody', parent=styles['Normal'],
        fontSize=9.5, leading=13, spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name='PaperCaption', parent=styles['Normal'],
        fontSize=8.5, leading=11, spaceBefore=4, spaceAfter=12,
        alignment=TA_CENTER, textColor=HexColor('#555555'),
        fontName='Helvetica-Oblique',
    ))
    styles.add(ParagraphStyle(
        name='PaperCode', parent=styles['Code'],
        fontSize=8, leading=10, leftIndent=12,
        backColor=HexColor('#f4f4f4'), borderColor=HexColor('#dddddd'),
        borderWidth=0.5, borderPadding=4, spaceAfter=8,
    ))
    return styles


def md_inline_to_rl(text: str) -> str:
    """Convert markdown inline formatting to reportlab mini-HTML."""
    # Escape bare & < > that aren't already part of tags
    text = text.replace('&', '&amp;')
    # Protect code spans first
    codes = []
    def _code(m):
        codes.append(m.group(1))
        return f'\x00CODE{len(codes)-1}\x00'
    text = re.sub(r'`([^`]+)`', _code, text)
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    # Bold
    text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*([^\*]+)\*', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<i>\1</i>', text)
    # Links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Restore code spans
    for i, code in enumerate(codes):
        safe = code.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace(f'\x00CODE{i}\x00', f'<font name="Courier">{safe}</font>')
    return text


def parse_markdown_table(lines, i):
    """Parse a markdown table starting at line i. Returns (rows, new_i)."""
    rows = []
    while i < len(lines) and '|' in lines[i]:
        line = lines[i].strip()
        if not line:
            break
        # Skip separator row like |---|---|
        if re.match(r'^\|?\s*[\:\-]+[\s\|\:\-]*\|?\s*$', line):
            i += 1
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)
        i += 1
    return rows, i


def build_table(rows, styles):
    if not rows:
        return None
    # Normalize row lengths
    max_cols = max(len(r) for r in rows)
    normalized = []
    for r in rows:
        cells = r + [''] * (max_cols - len(r))
        normalized.append([Paragraph(md_inline_to_rl(c), styles['PaperBody']) for c in cells])

    page_width = 6.5 * inch
    col_width = page_width / max_cols
    t = Table(normalized, colWidths=[col_width] * max_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e8e8e8')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#999999')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def find_figure(fig_number: int) -> Path:
    """Find a figure file by number like 'fig1', 'fig23', etc."""
    if not FIGURES_DIR.exists():
        return None
    matches = list(FIGURES_DIR.glob(f'fig{fig_number}_*.png'))
    if matches:
        return matches[0]
    return None


def load_image(path: Path, max_width: float = 6.0 * inch, max_height: float = 6.5 * inch):
    """Load an image scaled to fit within max dimensions."""
    try:
        pil = PILImage.open(path)
        w, h = pil.size
        ratio = min(max_width / w, max_height / h)
        return Image(str(path), width=w * ratio, height=h * ratio)
    except Exception as e:
        print(f'  [WARN] Could not load {path}: {e}')
        return None


def markdown_to_flowables(md_text: str, styles, title: str, inline_figures: bool = False, md_dir: Path = None):
    """Convert a markdown document to a list of reportlab flowables."""
    flowables = []
    flowables.append(Paragraph(md_inline_to_rl(title), styles['PaperTitle']))
    flowables.append(Spacer(1, 0.2 * inch))

    lines = md_text.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = ' '.join(paragraph_lines).strip()
            if text:
                flowables.append(Paragraph(md_inline_to_rl(text), styles['PaperBody']))
            paragraph_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            code_text = '\n'.join(code_lines)
            flowables.append(Preformatted(code_text, styles['PaperCode']))
            code_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith('```'):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                flush_paragraph()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Inline images: ![alt](path)
        img_match = re.match(r'^!\[([^\]]*)\]\(([^\)]+)\)', stripped)
        if img_match:
            flush_paragraph()
            alt_text = img_match.group(1)
            img_path_str = img_match.group(2)
            # Resolve relative to the markdown file's directory
            if md_dir:
                img_path = (md_dir / img_path_str).resolve()
            else:
                img_path = Path(img_path_str)
            if img_path.exists():
                img = load_image(img_path, max_width=5.5 * inch, max_height=5.5 * inch)
                if img:
                    flowables.append(KeepTogether([img]))
            i += 1
            continue

        # Italic caption lines: *Figure N: ...*
        if stripped.startswith('*Figure') and stripped.endswith('*') and not stripped.startswith('**'):
            flush_paragraph()
            caption_text = stripped.strip('*')
            flowables.append(Paragraph(md_inline_to_rl(caption_text), styles['PaperCaption']))
            flowables.append(Spacer(1, 0.1 * inch))
            i += 1
            continue

        # Skip the first title line if it matches the doc title
        if stripped.startswith('# '):
            flush_paragraph()
            heading = stripped[2:].strip()
            if i < 5 and heading.lower() in title.lower():
                i += 1
                continue
            flowables.append(Paragraph(md_inline_to_rl(heading), styles['PaperH1']))
            i += 1
            continue

        if stripped.startswith('## '):
            flush_paragraph()
            flowables.append(Paragraph(md_inline_to_rl(stripped[3:].strip()), styles['PaperH2']))
            i += 1
            continue

        if stripped.startswith('### '):
            flush_paragraph()
            flowables.append(Paragraph(md_inline_to_rl(stripped[4:].strip()), styles['PaperH3']))
            i += 1
            continue

        # Horizontal rule
        if stripped in ('---', '***', '___'):
            flush_paragraph()
            flowables.append(Spacer(1, 0.1 * inch))
            i += 1
            continue

        # Table detection
        if '|' in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*[\:\-]+[\s\|\:\-]*\|?\s*$', lines[i + 1]):
            flush_paragraph()
            rows, i = parse_markdown_table(lines, i)
            table = build_table(rows, styles)
            if table:
                flowables.append(table)
                flowables.append(Spacer(1, 0.1 * inch))
            continue

        # Bullet list items
        bullet_match = re.match(r'^(\s*)[\-\*]\s+(.+)$', line)
        if bullet_match:
            flush_paragraph()
            indent_level = len(bullet_match.group(1)) // 2
            content = bullet_match.group(2)
            bullet_style = ParagraphStyle(
                name='Bullet', parent=styles['PaperBody'],
                leftIndent=12 + indent_level * 12, bulletIndent=indent_level * 12,
            )
            flowables.append(Paragraph(f'• {md_inline_to_rl(content)}', bullet_style))
            i += 1
            continue

        # Numbered list items
        num_match = re.match(r'^(\s*)(\d+)\.\s+(.+)$', line)
        if num_match:
            flush_paragraph()
            indent_level = len(num_match.group(1)) // 2
            num = num_match.group(2)
            content = num_match.group(3)
            num_style = ParagraphStyle(
                name='NumList', parent=styles['PaperBody'],
                leftIndent=14 + indent_level * 12, bulletIndent=indent_level * 12,
            )
            flowables.append(Paragraph(f'{num}. {md_inline_to_rl(content)}', num_style))
            i += 1
            continue

        # Blank line → paragraph break
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        paragraph_lines.append(stripped)
        i += 1

    flush_paragraph()
    flush_code()

    return flowables


def build_figures_appendix(styles):
    """Build an appendix with all Stage 2 figures organized by section."""
    flowables = [PageBreak()]
    flowables.append(Paragraph('Appendix: Statistical Figures', styles['PaperH1']))
    flowables.append(Paragraph(
        'The following figures visualize the statistical results discussed throughout the paper. '
        'All figures are generated by the scripts in <font name="Courier">scripts/stage_2_*_figures.py</font> '
        'from the data in <font name="Courier">data/stage_2_*/</font>.',
        styles['PaperBody']
    ))
    flowables.append(Spacer(1, 0.2 * inch))

    # Figure groups by theme
    groups = [
        ('Pipeline and Core Statistics', list(range(1, 10))),
        ('Layer Energy and Cross-Model Analysis', list(range(10, 24))),
        ('Category Breakdowns and Domain Effects', list(range(24, 34))),
        ('Regression and Statistical Deepdive', list(range(34, 49))),
    ]

    for group_name, fig_numbers in groups:
        flowables.append(Paragraph(group_name, styles['PaperH2']))

        for fig_num in fig_numbers:
            fig_path = find_figure(fig_num)
            if fig_path is None:
                continue
            img = load_image(fig_path, max_width=5.5 * inch, max_height=5.5 * inch)
            if img:
                caption = f'Figure {fig_num}: {fig_path.stem.replace(f"fig{fig_num}_", "").replace("_", " ").title()}'
                flowables.append(KeepTogether([
                    img,
                    Paragraph(caption, styles['PaperCaption']),
                ]))
                flowables.append(Spacer(1, 0.15 * inch))

    return flowables


def generate_pdf(md_path: Path, out_path: Path, title: str, include_figures: bool = False):
    print(f'Generating {out_path.name}...')

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    styles = build_styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title=title,
    )

    flowables = markdown_to_flowables(md_text, styles, title, md_dir=md_path.parent)

    if include_figures:
        flowables.extend(build_figures_appendix(styles))

    doc.build(flowables)
    size_kb = out_path.stat().st_size / 1024
    print(f'  -> {out_path} ({size_kb:.0f} KB)')


def main():
    papers = [
        {
            'md': BASE / 'docs' / 'papers' / 'CROSS_DOMAIN_CIRCUIT_PAPER.md',
            'pdf': OUTPUT_DIR / 'CROSS_DOMAIN_CIRCUIT_PAPER.pdf',
            'title': 'Cross-Domain Circuit Analysis of Factual Knowledge Retrieval in LLMs',
            'figures': True,
        },
        {
            'md': BASE / 'docs' / 'papers' / 'DATA_AVAILABILITY_AND_EXPERIMENTS.md',
            'pdf': OUTPUT_DIR / 'DATA_AVAILABILITY_AND_EXPERIMENTS.pdf',
            'title': 'Data Availability and Experimental Summary',
            'figures': False,
        },
        {
            'md': BASE.parent / 'supernode_detector' / 'docs' / 'SUPERNODE_PIPELINE_TOOL_PAPER.md',
            'pdf': OUTPUT_DIR / 'SUPERNODE_PIPELINE_TOOL_PAPER.pdf',
            'title': 'Supernode Detector: Automated Community Detection for Neuronpedia Attribution Graphs',
            'figures': False,
        },
    ]

    for paper in papers:
        if not paper['md'].exists():
            print(f'[SKIP] {paper["md"]} not found')
            continue
        generate_pdf(paper['md'], paper['pdf'], paper['title'], include_figures=paper['figures'])

    print(f'\nAll PDFs written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
