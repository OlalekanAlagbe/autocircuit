#!/usr/bin/env python3
"""Generate a visual data inventory chart showing all available data types."""

import sys
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'stage_2_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors
C_RAW = '#E3F2FD'       # light blue
C_AGG = '#E8F5E9'       # light green
C_STAT = '#FFF3E0'      # light orange
C_LAYER = '#F3E5F5'     # light purple
C_LIB = '#FCE4EC'       # light pink
C_FIG = '#ECEFF1'       # light grey
C_PAPER = '#FFFDE7'     # light yellow
C_BORDER = '#37474F'
C_HEADER = '#263238'

def draw_card(x, y, w, h, title, items, color, title_color='white'):
    """Draw a rounded card with title bar and bullet items."""
    # Card body
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.5)
    ax.add_patch(rect)
    # Title bar
    title_rect = FancyBboxPatch((x, y + h - 0.55), w, 0.55,
                                boxstyle="round,pad=0.05",
                                facecolor=C_HEADER, edgecolor='none')
    ax.add_patch(title_rect)
    ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color=title_color, family='sans-serif')

    # Items
    for i, item in enumerate(items):
        iy = y + h - 0.85 - i * 0.38
        if iy < y + 0.1:
            break
        ax.text(x + 0.15, iy, item, ha='left', va='center',
                fontsize=7.5, color='#212121', family='sans-serif')


# =========== ROW 1: Raw Circuit Data ===========
ax.text(11, 15.7, 'AUTOCIRCUIT DATA INVENTORY', ha='center', va='center',
        fontsize=18, fontweight='bold', color=C_HEADER, family='sans-serif')
ax.text(11, 15.3, '60 circuits  |  30 prompts x 2 models  |  3 knowledge domains (chemistry / geography / history)',
        ha='center', va='center', fontsize=10, color='#546E7A', family='sans-serif')

# Raw data cards
draw_card(0.3, 11.2, 4.8, 3.8, 'RAW CIRCUIT DATA (per circuit)', [
    'converted_graph.json',
    '  - Nodes: activation, layer, feature_id, ctx_idx',
    '  - Edges: source, target, weight',
    '  - Metadata: output_probability, top_predictions',
    '',
    'traceback_paths.json',
    '  - Critical paths (backward BFS, decay=0.8)',
    '  - Final-layer nodes + convergence scores',
    '',
    'circuit_analysis.json',
    '  - Community structure (Louvain)',
    '  - Bottleneck identification',
], C_RAW)

# Aggregated analysis
draw_card(5.5, 11.2, 5.3, 3.8, 'CROSS-DOMAIN ANALYSIS', [
    'cross_category_results.json',
    '  - Within-category Jaccard matrices',
    '  - Cross-category feature overlap',
    '  - Layer distributions per category',
    '  - Universal bottleneck features (6G, 15Q)',
    '',
    'enhanced_results.json',
    '  - Path topology (compression, decay, branching)',
    '  - Edge weight distributions + KS tests',
    '  - Community structure (Louvain, ARI)',
    '  - Output node sharing ratios',
    '  - Activation profile similarity (cosine)',
    '  - Bootstrap CIs, permutation tests, chi-square',
], C_AGG)

# Statistical deepdive
draw_card(11.2, 11.2, 5.3, 3.8, 'STATISTICAL DEEPDIVE', [
    'statistical_deepdive_results.json',
    '  - 16-metric correlation matrix (Bonferroni)',
    '  - OLS regression (R2=0.489, p<0.001)',
    '  - ANOVA / Kruskal-Wallis (6/18 survive)',
    '  - Mann-Whitney cross-model (13/16 sig)',
    '  - Cohen\'s d pairwise effect sizes',
    '  - Null results: convergence, concentration',
    '',
    'data_table.json (flat 60-row table)',
    '  - ~25 metrics per circuit:',
    '    confidence, energy, nodes, edges, bottleneck',
    '    stats, weight dist, path topology, peak layer',
], C_STAT)

# Layer energy
draw_card(16.9, 11.2, 4.8, 3.8, 'PER-LAYER ENERGY ANALYSIS', [
    'layer_energy_results.json',
    '  - Domain-averaged energy curves',
    '  - Profile similarity (within/cross/between)',
    '  - Cumulative energy (evidence accumulation)',
    '  - Bottleneck energy decomposition',
    '  - Per-layer confidence correlations',
    '  - Drift-diffusion metrics',
    '  - Domain overlap SNR',
    '',
    'per_circuit_layer_energy.json',
    '  - Full per-layer vectors (60 circuits):',
    '    energy, mean_act, max_act, node_count,',
    '    energy_fraction per layer',
], C_LAYER)


# =========== ROW 2: Libraries + Key Numbers ===========

draw_card(0.3, 7.0, 4.8, 3.8, 'LIBRARIES & PROFILES', [
    'stage_1_5_bottleneck_library.json',
    '  - 303 bottleneck features (220 unique)',
    '  - 51 cross-circuit features',
    '  - Neuronpedia API profiles',
    '',
    'neuronpedia_config.yaml',
    '  - API key + endpoints',
    '',
    'pipeline_constants.py',
    '  - GEMMA: 26 layers, 16384 SAE features',
    '  - QWEN: 36 layers, transcoder-hp',
    '  - Bottleneck threshold: 0.6 (60%)',
], C_LIB)

# Key findings card
draw_card(5.5, 7.0, 5.3, 3.8, 'KEY STATISTICAL FINDINGS', [
    'Architecture > Domain:',
    '  Within-model cosine: 0.978',
    '  Between-model cosine: 0.696  (14x gap)',
    '',
    'Bottleneck Tax (GEMMA L6):',
    '  r = -0.684 (Bonferroni sig)',
    '  7 layers survive correction',
    '',
    'Confidence = Total Energy:',
    '  GEMMA: r=0.527 (sole Bonferroni survivor)',
    '  Regression R2=0.489, p<0.001',
    '  BN convergence: r=0.270 (n.s.)',
    '  BN energy frac: r=0.256 (n.s.)',
], '#E0F7FA')

# Category-level findings
draw_card(11.2, 7.0, 5.3, 3.8, 'CATEGORY-LEVEL FINDINGS', [
    'Within-domain Jaccard:',
    '  Geography: 0.39 (highest)',
    '  Chemistry: 0.19',
    '  History:   0.11 (lowest)',
    '',
    'Permutation tests: p < 0.0001 (both models)',
    'Chi-square: layer dist != independent of cat',
    '',
    'ANOVA survivors (Bonferroni):',
    '  Total energy, output probability',
    '  (both models, eps2 = 0.62-0.65)',
    '',
    'Cohen\'s d outliers:',
    '  GEMMA geo vs chem energy: d=4.88',
], '#E0F2F1')

# Cross-model findings
draw_card(16.9, 7.0, 4.8, 3.8, 'CROSS-MODEL FINDINGS', [
    '13/16 metrics differ (Mann-Whitney):',
    '  Bottleneck layer: perfect separation',
    '  Peak activation layer: perfect separation',
    '',
    '3 architecture-INVARIANT constants:',
    '  Branching factor',
    '  Total edges',
    '  Community entropy',
    '',
    'Energy accumulation:',
    '  GEMMA: 56% in 1st half (front-loads)',
    '  QWEN:  32% in 1st half (back-loads)',
    '  50% threshold: GEMMA L11, QWEN L28',
], '#EDE7F6')


# =========== ROW 3: Figures + Paper ===========

draw_card(0.3, 2.8, 7.0, 3.8, 'FIGURES (28 total)', [
    'Figs 1-15 (Original + Statistical):',
    '  Pipeline, Jaccard heatmaps, layer dists, bootstrap CIs,',
    '  correlation matrix, cross-model comparison, ANOVA,',
    '  confidence scatter, regression, community, output sharing,',
    '  Cohen\'s d heatmap, activation profiles, regression coefs,',
    '  layer distribution comparison',
    '',
    'Figs 16-23 (Layer Energy):',
    '  Domain energy curves, cross-model normalized profiles,',
    '  per-layer confidence correlation, cumulative energy,',
    '  profile similarity, domain overlap SNR, summary panels',
    '',
    'Figs 24-28 (Category Breakdown):',
    '  Dashboard (12 metrics), per-cat energy profiles, cumulative,',
    '  scatter matrix, summary table',
], C_FIG)

draw_card(7.7, 2.8, 7.0, 3.8, 'PAPER DRAFT', [
    'CROSS_DOMAIN_CIRCUIT_PAPER.md',
    '',
    'Sections:',
    '  1. Introduction (motivation, RQs, 6 contributions)',
    '  2. Related Work (MechInterp, SAEs, steering)',
    '  3. Methods (attribution, traceback, stats framework)',
    '  4. Experimental Setup (30 prompts, 2 models)',
    '  5. Results (13 subsections incl. deepdive + layer energy)',
    '  6. Discussion (7 subsections + 8 limitations)',
    '  7. Conclusion (9 findings + future work)',
    '',
    'Two Paper Angles:',
    '  Angle 1: Architecture Over Knowledge',
    '  Angle 3: Evidence Accumulation, Not Compression',
], C_PAPER)

draw_card(15.1, 2.8, 6.6, 3.8, 'SCRIPTS', [
    'Pipeline scripts:',
    '  stage_2_cross_category_analysis.py',
    '  stage_2_enhanced_analysis.py',
    '  stage_2_statistical_deepdive.py',
    '  stage_2_layer_energy_analysis.py',
    '',
    'Figure scripts:',
    '  stage_2_paper_figures.py (figs 1-15)',
    '  stage_2_layer_energy_figures.py (figs 16-23)',
    '  stage_2_category_breakdown_figures.py (figs 24-28)',
    '',
    'Shared:',
    '  pipeline_constants.py',
    '  neuronpedia_config.yaml',
], '#EFEBE9')


# =========== Arrows showing data flow ===========
arrow_style = dict(arrowstyle='->', color='#78909C', linewidth=1.5,
                   connectionstyle='arc3,rad=0.1')

# Raw -> Cross-domain
ax.annotate('', xy=(5.5, 13.1), xytext=(5.1, 13.1), arrowprops=arrow_style)
# Cross-domain -> Statistical
ax.annotate('', xy=(11.2, 13.1), xytext=(10.8, 13.1), arrowprops=arrow_style)
# Statistical -> Layer energy
ax.annotate('', xy=(16.9, 13.1), xytext=(16.5, 13.1), arrowprops=arrow_style)

# Down arrows to findings
for x_pos in [2.7, 8.15, 13.85, 19.3]:
    ax.annotate('', xy=(x_pos, 10.8), xytext=(x_pos, 11.2),
                arrowprops=dict(arrowstyle='->', color='#B0BEC5', linewidth=1))

# Bottom row label
ax.text(11, 2.4, '60 raw circuits  -->  4 analysis stages  -->  28 figures  -->  9-finding paper',
        ha='center', va='center', fontsize=11, style='italic', color='#607D8B')

plt.savefig(OUTPUT_DIR / 'fig29_data_inventory.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig29_data_inventory.png")
