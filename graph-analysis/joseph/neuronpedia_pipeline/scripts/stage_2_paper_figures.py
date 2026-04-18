"""
Stage 2: Paper Figures Generator
Creates publication-quality matplotlib figures for the cross-domain circuit analysis paper.
Reads from: data_table.json, enhanced_results.json, cross_category_results.json, statistical_deepdive_results.json
Outputs to: data/stage_2_figures/
"""

import json
import os
import sys
import numpy as np

# Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(DATA_DIR, 'stage_2_figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

print("Loading data...")
data_table = load_json(os.path.join(DATA_DIR, 'stage_2_statistical_deepdive', 'data_table.json'))
enhanced = load_json(os.path.join(DATA_DIR, 'stage_2_enhanced_analysis', 'enhanced_results.json'))
cross_cat = load_json(os.path.join(DATA_DIR, 'stage_2_analysis', 'cross_category_results.json'))
deepdive = load_json(os.path.join(DATA_DIR, 'stage_2_statistical_deepdive', 'statistical_deepdive_results.json'))

# ── Style constants ──────────────────────────────────────────────────────────
CATEGORY_COLORS = {'chemistry': '#E74C3C', 'geography': '#3498DB', 'history': '#2ECC71'}
MODEL_COLORS = {'gemma-2-2b': '#9B59B6', 'qwen3-4b': '#E67E22'}
CATEGORY_LABELS = {'chemistry': 'Chemistry', 'geography': 'Geography', 'history': 'History'}
MODEL_LABELS = {'gemma-2-2b': 'GEMMA-2-2B', 'qwen3-4b': 'QWEN3-4B'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# ── Helper: extract per-model arrays from data table ─────────────────────────
def get_model_data(model):
    return [r for r in data_table if r['model'] == model]

def get_category_data(model, category):
    return [r for r in data_table if r['model'] == model and r['category'] == category]

def extract_metric(rows, metric):
    return np.array([r[metric] for r in rows if r.get(metric) is not None])

MODELS = ['gemma-2-2b', 'qwen3-4b']
CATEGORIES = ['chemistry', 'geography', 'history']


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Figure 1: Analysis Pipeline Architecture', fontsize=14, fontweight='bold', pad=15)

    stages = [
        ('Graph\nGeneration', 'Neuronpedia API\n→ attribution graph', '#3498DB'),
        ('Graph\nConversion', 'Standardize format\n→ nodes + edges', '#2ECC71'),
        ('Traceback\nAnalysis', 'Backward BFS\n→ paths + bottlenecks', '#E74C3C'),
        ('Cross-Circuit\nComparison', 'Jaccard similarity\n→ shared features', '#9B59B6'),
        ('Statistical\nDeepdive', 'Regression, ANOVA\n→ 60-row data table', '#E67E22'),
        ('Visualization', 'Figures + report\n→ paper-ready', '#1ABC9C'),
    ]

    x_positions = np.linspace(1.0, 13.5, len(stages))
    for i, (title, desc, color) in enumerate(stages):
        x = x_positions[i]
        # Box
        box = FancyBboxPatch((x - 0.9, 1.8), 1.8, 2.0,
                             boxstyle="round,pad=0.1", facecolor=color, alpha=0.2,
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 3.3, title, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
        ax.text(x, 2.3, desc, ha='center', va='center', fontsize=7, color='#333333')
        ax.text(x, 4.2, f'Step {i+1}', ha='center', va='center', fontsize=8, color='gray')

        # Arrow
        if i < len(stages) - 1:
            x_next = x_positions[i + 1]
            ax.annotate('', xy=(x_next - 1.0, 2.8), xytext=(x + 1.0, 2.8),
                       arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    # Data flow labels
    ax.text(7, 0.8, '30 prompts × 2 models = 60 circuits → 25 metrics per circuit → 8 statistical analyses',
            ha='center', va='center', fontsize=10, style='italic', color='#666')

    fig.savefig(os.path.join(FIG_DIR, 'fig1_pipeline.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 1: Pipeline architecture")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Within-Category Jaccard Heatmaps (3 panels per model)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_jaccard_heatmaps():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 2: Within-Category Jaccard Similarity Heatmaps', fontsize=14, fontweight='bold')

    for mi, model in enumerate(MODELS):
        for ci, cat in enumerate(CATEGORIES):
            ax = axes[mi][ci]

            # Find the matching within_category entry
            entry = None
            for item in cross_cat['within_category']:
                if item['model'] == model and item['category'] == cat:
                    entry = item
                    break

            if entry is None or 'jaccard_matrix' not in entry:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{MODEL_LABELS[model]} - {CATEGORY_LABELS[cat]}')
                continue

            jac = entry['jaccard_matrix']

            # Extract unique prompt slugs
            slugs = set()
            for key in jac:
                parts = key.split(' vs ')
                slugs.update(parts)
            slugs = sorted(slugs)
            n = len(slugs)

            # Build matrix
            matrix = np.eye(n)
            slug_idx = {s: i for i, s in enumerate(slugs)}
            for key, val in jac.items():
                parts = key.split(' vs ')
                if len(parts) == 2:
                    i, j = slug_idx.get(parts[0]), slug_idx.get(parts[1])
                    if i is not None and j is not None:
                        matrix[i][j] = val
                        matrix[j][i] = val

            # Short labels
            short_labels = []
            for s in slugs:
                # Extract key word
                parts = s.replace('the-chemical-symbol-for-', '').replace('the-capital-of-', '').replace('-is', '')
                parts = parts.split('-')
                label = parts[0].capitalize() if parts else s[:6]
                short_labels.append(label)

            im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=0.6, aspect='equal')
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(short_labels, fontsize=8)
            ax.set_title(f'{MODEL_LABELS[model]} - {CATEGORY_LABELS[cat]}\n(mean J = {entry["avg_jaccard_similarity"]:.3f})',
                        fontsize=11, color=CATEGORY_COLORS[cat])

            # Add text annotations
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center', fontsize=6,
                               color='white' if matrix[i][j] > 0.35 else 'black')

    fig.subplots_adjust(hspace=0.45, wspace=0.3, right=0.88, top=0.92)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Jaccard Similarity')
    fig.savefig(os.path.join(FIG_DIR, 'fig2_jaccard_heatmaps.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 2: Jaccard heatmaps")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Bottleneck Depth Comparison (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_bottleneck_depth():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 3: Bottleneck Depth by Knowledge Domain', fontsize=14, fontweight='bold')

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        means = []
        stds = []
        for cat in CATEGORIES:
            vals = extract_metric(get_category_data(model, cat), 'avg_bottleneck_layer')
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(len(CATEGORIES))
        bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                     color=[CATEGORY_COLORS[c] for c in CATEGORIES], alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        # Total layers line
        total = 26 if model == 'gemma-2-2b' else 36
        ax.axhline(y=total, color='gray', linestyle='--', alpha=0.5, label=f'Total layers ({total})')

        ax.set_xticks(x)
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
        ax.set_ylabel('Average Bottleneck Layer')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.legend(fontsize=9)

        # Annotate % depth
        for i, (m, s) in enumerate(zip(means, stds)):
            pct = m / total * 100
            ax.text(i, m + s + 0.5, f'{pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_bottleneck_depth.png'))
    plt.close(fig)
    print("  Figure 3: Bottleneck depth")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Confidence vs Total Energy Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_confidence_energy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 4: Prediction Confidence vs Total Activation Energy', fontsize=14, fontweight='bold')

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        for cat in CATEGORIES:
            rows = get_category_data(model, cat)
            x = extract_metric(rows, 'total_energy')
            y = extract_metric(rows, 'output_probability')
            ax.scatter(x, y, c=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat],
                      s=60, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)

        # Per-category fit lines
        for cat in CATEGORIES:
            rows = get_category_data(model, cat)
            x = extract_metric(rows, 'total_energy')
            y = extract_metric(rows, 'output_probability')
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p_fit(x_line), '-', color=CATEGORY_COLORS[cat], alpha=0.5, linewidth=1.5)

        # Overall fit line
        all_rows = get_model_data(model)
        x_all = extract_metric(all_rows, 'total_energy')
        y_all = extract_metric(all_rows, 'output_probability')
        if len(x_all) > 2:
            z = np.polyfit(x_all, y_all, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(x_line, p_fit(x_line), '--', color='black', alpha=0.8, linewidth=2, label='Overall fit')

        # Stats annotation
        corr_data = deepdive.get('expanded_correlation_matrix', {}).get(model, {}).get('correlations', {})
        te_data = corr_data.get('total_energy', {})
        r_val = te_data.get('spearman_r', 0)
        p_val = te_data.get('spearman_p', 1)
        bonf = str(te_data.get('survives_bonferroni', '')).lower() == 'true'
        sig_str = '***' if bonf else ('*' if p_val < 0.05 else 'ns')
        ax.text(0.05, 0.95, f'r = {r_val:.3f} (p = {p_val:.4f}) {sig_str}',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Total Activation Energy')
        ax.set_ylabel('Output Probability')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_confidence_energy.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 4: Confidence vs energy")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Confidence vs Median Bottleneck Layer Scatter
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_confidence_bottleneck():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 5: Prediction Confidence vs Median Bottleneck Layer', fontsize=14, fontweight='bold')

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        for cat in CATEGORIES:
            rows = get_category_data(model, cat)
            x = extract_metric(rows, 'median_bottleneck_layer')
            y = extract_metric(rows, 'output_probability')
            ax.scatter(x, y, c=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat],
                      s=60, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)

        # Stats
        bn_data = deepdive.get('confidence_vs_bottleneck_position', {}).get(model, {}).get('median_bottleneck_layer', {})
        r_val = bn_data.get('spearman_r', 0)
        p_val = bn_data.get('spearman_p', 1)
        sig_str = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns')
        ax.text(0.05, 0.95, f'r = {r_val:.3f} (p = {p_val:.4f}) {sig_str}',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Median Bottleneck Layer')
        ax.set_ylabel('Output Probability')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_confidence_bottleneck.png'))
    plt.close(fig)
    print("  Figure 5: Confidence vs bottleneck layer")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Correlation Matrix Heatmap (16 metrics vs confidence)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_correlation_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Figure 6: Correlation Matrix — Circuit Metrics vs Prediction Confidence', fontsize=14, fontweight='bold')

    metric_order = [
        'total_energy', 'weight_kurtosis', 'total_nodes', 'total_edges',
        'num_bottlenecks', 'avg_layers_visited', 'avg_compression_ratio',
        'avg_score_decay_rate', 'avg_branching_factor', 'top_10pct_weight_share',
        'weight_skewness', 'peak_layer', 'avg_bottleneck_layer', 'bottleneck_layer_std',
        'num_supernodes', 'size_entropy'
    ]
    metric_labels = [
        'Total Energy', 'Weight Kurtosis', 'Total Nodes', 'Total Edges',
        '# Bottlenecks', 'Avg Layers Visited', 'Compression Ratio',
        'Score Decay Rate', 'Branching Factor', 'Top 10% Weight Share',
        'Weight Skewness', 'Peak Layer', 'Avg BN Layer', 'BN Layer Std',
        '# Supernodes', 'Size Entropy'
    ]

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        corr_data = deepdive.get('expanded_correlation_matrix', {}).get(model, {}).get('correlations', {})

        r_values = []
        p_values = []
        for metric in metric_order:
            entry = corr_data.get(metric, {})
            r_values.append(entry.get('spearman_r', 0))
            p_values.append(entry.get('spearman_p', 1))

        r_arr = np.array(r_values).reshape(-1, 1)
        im = ax.imshow(r_arr, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect=0.3)

        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels, fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels(['Spearman r'], fontsize=10)
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')

        # Annotate with r values and significance stars
        bonf_alpha = 0.003125
        for i, (r, p) in enumerate(zip(r_values, p_values)):
            star = ''
            if p < bonf_alpha:
                star = ' ***'
            elif p < 0.01:
                star = ' **'
            elif p < 0.05:
                star = ' *'
            color = 'white' if abs(r) > 0.4 else 'black'
            ax.text(0, i, f'{r:.3f}{star}', ha='center', va='center', fontsize=9,
                   color=color, fontweight='bold' if p < bonf_alpha else 'normal')

    fig.subplots_adjust(right=0.85, top=0.92)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Spearman r')
    fig.savefig(os.path.join(FIG_DIR, 'fig6_correlation_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 6: Correlation heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Cross-Model Comparison (Mann-Whitney effect sizes)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_cross_model():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Figure 7: Cross-Model Differences (GEMMA vs QWEN)\nMann-Whitney U with Bonferroni Correction',
                fontsize=13, fontweight='bold')

    mw_data = deepdive.get('cross_model_comparison', {}).get('metrics', {})
    if not mw_data:
        ax.text(0.5, 0.5, 'No cross-model data', ha='center', va='center', transform=ax.transAxes)
        fig.savefig(os.path.join(FIG_DIR, 'fig7_cross_model.png'))
        plt.close(fig)
        return

    # Sort by absolute effect size
    metrics = sorted(mw_data.keys(), key=lambda m: abs(mw_data[m].get('rank_biserial_r', 0)), reverse=True)
    r_values = [mw_data[m].get('rank_biserial_r', 0) for m in metrics]
    p_values = [mw_data[m].get('p_value', 1) for m in metrics]
    bonf_alpha = 0.003125

    # Clean metric names
    clean_names = []
    for m in metrics:
        name = m.replace('_', ' ').replace('avg ', '').replace('pct', '%').title()
        clean_names.append(name)

    y = np.arange(len(metrics))
    colors = ['#E74C3C' if p < bonf_alpha else '#BDC3C7' for p in p_values]

    ax.barh(y, r_values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel('Rank-Biserial Correlation (r)', fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, label='Medium effect')
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', edgecolor='black', label=f'Significant (p < {bonf_alpha:.4f})'),
        mpatches.Patch(facecolor='#BDC3C7', edgecolor='black', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.0, -0.08), fontsize=9, ncol=2, frameon=True)

    ax.text(0.02, 0.98, '← GEMMA higher', transform=ax.transAxes, fontsize=9, color='#9B59B6', va='top')
    ax.text(0.98, 0.98, 'QWEN higher →', transform=ax.transAxes, fontsize=9, color='#E67E22', va='top', ha='right')

    ax.grid(True, axis='x', alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, 'fig7_cross_model.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 7: Cross-model comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: ANOVA Effect Size Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_anova_effects():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 8: Domain Effects on Circuit Metrics (ANOVA / Kruskal-Wallis)',
                fontsize=14, fontweight='bold')

    anova_data = deepdive.get('anova_kruskal', {})
    bonf_alpha = 0.00278

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        model_data = anova_data.get(model, {})
        if not model_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        metrics = sorted(model_data.keys())
        effect_sizes = [model_data[m].get('eta_squared', model_data[m].get('epsilon_squared', 0)) for m in metrics]
        p_values = [model_data[m].get('p_value', 1) for m in metrics]

        clean_names = [m.replace('_', ' ').title() for m in metrics]
        y = np.arange(len(metrics))
        colors = ['#E74C3C' if p < bonf_alpha else '#F39C12' if p < 0.05 else '#BDC3C7' for p in p_values]

        ax.barh(y, effect_sizes, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(clean_names, fontsize=9)
        ax.set_xlabel('Effect Size (η² or ε²)', fontsize=11)
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')

        # Threshold lines
        ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.4, label='Small (0.01)')
        ax.axvline(x=0.06, color='gray', linestyle='--', alpha=0.4, label='Medium (0.06)')
        ax.axvline(x=0.14, color='gray', linestyle='-', alpha=0.4, label='Large (0.14)')

        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, axis='x', alpha=0.3)

    # Shared legend for significance
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', edgecolor='black', label='Bonferroni-surviving'),
        mpatches.Patch(facecolor='#F39C12', edgecolor='black', label='Uncorrected p < 0.05'),
        mpatches.Patch(facecolor='#BDC3C7', edgecolor='black', label='Not significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, 'fig8_anova_effects.png'))
    plt.close(fig)
    print("  Figure 8: ANOVA effect sizes")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_bootstrap_ci():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 9: Bootstrap 95% CIs for Within-Category Jaccard Similarity',
                fontsize=14, fontweight='bold')

    bootstrap_data = enhanced.get('statistical_significance', {}).get('bootstrap_ci', {})

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        model_data = bootstrap_data.get(model, {})

        for ci, cat in enumerate(CATEGORIES):
            cat_data = model_data.get(cat, {})
            mean = cat_data.get('mean', 0)
            ci_lo = cat_data.get('ci_lower', 0)
            ci_hi = cat_data.get('ci_upper', 0)

            ax.errorbar(ci, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                       fmt='o', color=CATEGORY_COLORS[cat], capsize=8, capthick=2,
                       markersize=10, markeredgecolor='black', markeredgewidth=1,
                       linewidth=2, label=CATEGORY_LABELS[cat])

        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
        ax.set_ylabel('Within-Category Jaccard Similarity')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig9_bootstrap_ci.png'))
    plt.close(fig)
    print("  Figure 9: Bootstrap CIs")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Output vs Path Convergence
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_output_path():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle('Figure 10: Output Node Convergence vs Path Convergence',
                fontsize=14, fontweight='bold')

    output_data = enhanced.get('output_node_sharing', {})

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        model_data = output_data.get(model, {})
        comparison = model_data.get('output_vs_path_comparison', {})

        output_j = []
        path_j = []
        labels = []
        for cat in CATEGORIES:
            cat_data = comparison.get(cat, {})
            oj = cat_data.get('output_jaccard', 0)
            pj = cat_data.get('path_jaccard', 0)
            output_j.append(oj)
            path_j.append(pj)
            labels.append(CATEGORY_LABELS[cat])

        x = np.arange(len(CATEGORIES))
        width = 0.35
        bars1 = ax.bar(x - width/2, output_j, width, label='Output Jaccard',
                       color=[CATEGORY_COLORS[c] for c in CATEGORIES], alpha=0.9,
                       edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, path_j, width, label='Path Jaccard',
                       color=[CATEGORY_COLORS[c] for c in CATEGORIES], alpha=0.4,
                       edgecolor='black', linewidth=0.5, hatch='//')

        # Ratio annotations
        for i, (oj, pj) in enumerate(zip(output_j, path_j)):
            ratio = oj / pj if pj > 0 else 0
            ax.text(i, max(oj, pj) + 0.02, f'{ratio:.1f}x', ha='center', fontsize=10,
                   fontweight='bold', color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Jaccard Similarity')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig10_output_path.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 10: Output vs path convergence")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Path Topology Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_path_topology():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Figure 11: Path Topology Metrics by Domain', fontsize=14, fontweight='bold')

    metrics = ['avg_compression_ratio', 'avg_score_decay_rate', 'avg_branching_factor']
    metric_labels = ['Compression Ratio', 'Score Decay Rate', 'Branching Factor']

    for mi, model in enumerate(MODELS):
        for mci, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[mi][mci]

            data_per_cat = []
            for cat in CATEGORIES:
                vals = extract_metric(get_category_data(model, cat), metric)
                data_per_cat.append(vals)

            bp = ax.boxplot(data_per_cat, tick_labels=[CATEGORY_LABELS[c] for c in CATEGORIES],
                           patch_artist=True, widths=0.5)
            for patch, cat in zip(bp['boxes'], CATEGORIES):
                patch.set_facecolor(CATEGORY_COLORS[cat])
                patch.set_alpha(0.6)

            # Overlay individual points
            for ci, (cat, vals) in enumerate(zip(CATEGORIES, data_per_cat)):
                jitter = np.random.normal(0, 0.05, len(vals))
                ax.scatter(np.ones(len(vals)) * (ci + 1) + jitter, vals,
                          color=CATEGORY_COLORS[cat], alpha=0.6, s=25, zorder=3,
                          edgecolors='black', linewidth=0.3)

            ax.set_ylabel(mlabel)
            if mi == 0:
                ax.set_title(mlabel, fontsize=11, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

        # Model label on y-axis of first column
        axes[mi][0].set_ylabel(f'{MODEL_LABELS[model]}\n{metric_labels[0]}',
                              color=MODEL_COLORS[model], fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig11_path_topology.png'))
    plt.close(fig)
    print("  Figure 11: Path topology")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Cohen's d Effect Size Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_cohens_d():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Figure 12: Cohen's d Effect Sizes — Pairwise Category Comparisons",
                fontsize=14, fontweight='bold')

    # Structure: effect_sizes.pairwise_cohens_d.model.metric.pair.d
    effect_data = deepdive.get('effect_sizes', {}).get('pairwise_cohens_d', {})
    pairs = ['chemistry_vs_geography', 'chemistry_vs_history', 'geography_vs_history']
    pair_labels = ['Chem vs Geo', 'Chem vs Hist', 'Geo vs Hist']

    im = None
    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        model_data = effect_data.get(model, {})
        if not model_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
            continue

        # Get all metrics from the model data
        metrics = sorted(model_data.keys())
        clean_names = [m.replace('_', ' ').title() for m in metrics]

        # Build matrix: rows=metrics, cols=pairs
        matrix = np.zeros((len(metrics), len(pairs)))
        for mi2, metric in enumerate(metrics):
            metric_data = model_data.get(metric, {})
            for pi, pair in enumerate(pairs):
                pair_entry = metric_data.get(pair, {})
                matrix[mi2][pi] = pair_entry.get('d', 0)

        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pair_labels, fontsize=10)
        ax.set_yticks(range(len(metrics)))
        if mi == 0:
            ax.set_yticklabels(clean_names, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')

        # Text annotations
        for i in range(len(metrics)):
            for j in range(len(pairs)):
                val = matrix[i][j]
                color = 'white' if abs(val) > 2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color)

    fig.subplots_adjust(right=0.85, top=0.92, wspace=0.5)
    if im is not None:
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Cohen's d")
    fig.savefig(os.path.join(FIG_DIR, 'fig12_cohens_d.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 12: Cohen's d heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Activation Profile Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_activation_profiles():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 13: Activation Profile Comparison', fontsize=14, fontweight='bold')

    profile_data = enhanced.get('activation_profiles', {})

    for mi, model in enumerate(MODELS):
        # Left panel: Peak layers
        ax1 = axes[mi][0]
        model_data = profile_data.get(model, {})
        peak_data = model_data.get('peak_layer_distribution', {})

        for ci, cat in enumerate(CATEGORIES):
            cat_peaks = peak_data.get(cat, {}).get('peaks', [])
            if cat_peaks:
                jitter = np.random.normal(0, 0.1, len(cat_peaks))
                ax1.scatter(np.ones(len(cat_peaks)) * ci + jitter, cat_peaks,
                          color=CATEGORY_COLORS[cat], s=100, alpha=0.8,
                          edgecolors='white', linewidth=0.3, label=CATEGORY_LABELS[cat], zorder=3)
                mean_peak = np.mean(cat_peaks)
                ax1.hlines(mean_peak, ci - 0.3, ci + 0.3, color=CATEGORY_COLORS[cat],
                         linewidth=3, zorder=5)

        ax1.set_xticks(range(len(CATEGORIES)))
        ax1.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
        ax1.set_ylabel('Peak Activation Layer')
        ax1.set_title(f'{MODEL_LABELS[model]} — Peak Layers', color=MODEL_COLORS[model], fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)

        # Right panel: Total energy
        ax2 = axes[mi][1]
        energy_data = model_data.get('total_energy_comparison', {})

        means = []
        stds = []
        for cat in CATEGORIES:
            cat_energy = energy_data.get(cat, {})
            means.append(cat_energy.get('mean_energy', 0))
            stds.append(cat_energy.get('std_energy', 0))

        x = np.arange(len(CATEGORIES))
        bars = ax2.bar(x, means, yerr=stds, capsize=5, width=0.6,
                      color=[CATEGORY_COLORS[c] for c in CATEGORIES], alpha=0.8,
                      edgecolor='black', linewidth=0.5)

        ax2.set_xticks(x)
        ax2.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES])
        ax2.set_ylabel('Total Activation Energy')
        ax2.set_title(f'{MODEL_LABELS[model]} — Total Energy', color=MODEL_COLORS[model], fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig13_activation_profiles.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Figure 13: Activation profiles")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Regression Model Summary
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_regression():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Figure 14: Regression Model — Predicting Output Probability\n(Reduced Models)',
                fontsize=13, fontweight='bold')

    reg_data = deepdive.get('multiple_regression', {})

    models_shown = []
    r2_values = []
    adj_r2_values = []
    f_p_values = []
    labels = []

    for model in MODELS:
        reduced = reg_data.get(model, {}).get('reduced_model', {})
        if reduced:
            r2 = reduced.get('r_squared', 0)
            adj = reduced.get('adj_r_squared', 0)
            f_p = reduced.get('f_p_value', 1)
            features = reduced.get('features_used', [])
            models_shown.append(model)
            r2_values.append(r2)
            adj_r2_values.append(adj)
            f_p_values.append(f_p)
            labels.append(f'{MODEL_LABELS[model]}\n({", ".join(features[:3])})')

    x = np.arange(len(models_shown))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_values, width, label='R²', color=[MODEL_COLORS[m] for m in models_shown],
                  alpha=0.9, edgecolor='black')
    bars2 = ax.bar(x + width/2, adj_r2_values, width, label='Adjusted R²',
                  color=[MODEL_COLORS[m] for m in models_shown], alpha=0.5, edgecolor='black', hatch='//')

    # P-value annotations
    for i, (r2, adj, fp) in enumerate(zip(r2_values, adj_r2_values, f_p_values)):
        sig = '***' if fp < 0.001 else '**' if fp < 0.01 else '*' if fp < 0.05 else 'ns'
        ax.text(i, max(r2, adj) + 0.02, f'p = {fp:.4f} {sig}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Variance Explained')
    ax.set_ylim(0, 0.7)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig14_regression.png'))
    plt.close(fig)
    print("  Figure 14: Regression model")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Layer Distribution Contingency (Chi-Square visualization)
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_layer_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 15: Bottleneck Feature Distribution Across Processing Stages',
                fontsize=14, fontweight='bold')

    chi_data = enhanced.get('statistical_significance', {}).get('chi_square', {})

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        model_chi = chi_data.get(model, {})
        ct = model_chi.get('contingency_table', {})
        observed = ct.get('observed', [])
        rows = ct.get('rows', CATEGORIES)
        cols = ct.get('cols', [])

        if not observed or not cols:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        observed = np.array(observed, dtype=float)
        # Normalize to proportions per category
        row_sums = observed.sum(axis=1, keepdims=True)
        props = observed / row_sums

        x = np.arange(len(cols))
        width = 0.25
        for ci, cat in enumerate(rows):
            offset = (ci - 1) * width
            ax.bar(x + offset, props[ci], width, label=CATEGORY_LABELS.get(cat, cat),
                  color=CATEGORY_COLORS.get(cat, 'gray'), alpha=0.8,
                  edgecolor='black', linewidth=0.5)

        chi2 = model_chi.get('chi2', 0)
        p_val = model_chi.get('p_value', 1)
        ax.text(0.95, 0.95, f'χ² = {chi2:.1f}\np = {p_val:.4f}',
               transform=ax.transAxes, va='top', ha='right', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        col_labels = [c.replace('_', '\n') for c in cols]
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_ylabel('Proportion of Bottleneck Features')
        ax.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig15_layer_distribution.png'))
    plt.close(fig)
    print("  Figure 15: Layer distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  STAGE 2: PAPER FIGURES GENERATOR")
    print(f"  Output: {FIG_DIR}")
    print("=" * 70)
    print()

    fig1_pipeline()
    fig2_jaccard_heatmaps()
    fig3_bottleneck_depth()
    fig4_confidence_energy()
    fig5_confidence_bottleneck()
    fig6_correlation_heatmap()
    fig7_cross_model()
    fig8_anova_effects()
    fig9_bootstrap_ci()
    fig10_output_path()
    fig11_path_topology()
    fig12_cohens_d()
    fig13_activation_profiles()
    fig14_regression()
    fig15_layer_distribution()

    print()
    print("=" * 70)
    print(f"  DONE — 15 figures saved to {FIG_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
