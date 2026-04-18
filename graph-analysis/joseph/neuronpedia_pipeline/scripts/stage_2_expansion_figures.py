#!/usr/bin/env python3
"""Stage 2: Expansion Figures (Figs 37-44).

Generates publication-quality figures from D1-D4 analyses:
  Fig37: Interaction regression R² cascade (D1)
  Fig38: Unused column correlations heatmap (D1)
  Fig39: Multi-algorithm Jaccard distribution (D2)
  Fig40: Method-pair agreement matrix (D2)
  Fig41: Minimal pathway reduction by model/category (D3)
  Fig42: Bottleneck width vs weight retention (D3)
  Fig43: Steering effect by feature and layer (D4)
  Fig44: Convergence vs steering effect scatter (D4)

Output: data/stage_2_figures/fig37_*.png through fig44_*.png

Usage:
    python scripts/stage_2_expansion_figures.py
"""

import json
import sys
import io
import math
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
FIG_DIR = DATA_DIR / 'stage_2_figures'

# Input data paths
D1_RESULTS = DATA_DIR / 'stage_2_unused_columns' / 'unused_columns_results.json'
D2_RESULTS = DATA_DIR / 'stage_2_multi_algorithm' / 'multi_algorithm_results.json'
D3_RESULTS = DATA_DIR / 'stage_2_minimal_pathways' / 'minimal_pathway_results.json'
D4_RESULTS = DATA_DIR / 'stage_2_steering_validation' / 'steering_validation_results.json'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========================================================================
# Fig 37: Interaction Regression R² Cascade (D1)
# ========================================================================
def fig37_r2_cascade(d1):
    """Show how R² improves as we add predictors (additive → category → interactions)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(['gemma-2-2b', 'qwen3-4b']):
        ax = axes[idx]
        reg = d1.get('interaction_regression', {}).get(model, {})
        if not reg:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        # Build bars from available R² keys in the data
        r2_specs = [
            ('Additive\n(3 pred)', 'additive_r2'),
            ('+Category\ndummies', 'additive_plus_category_r2'),
            ('+Energy\ninteraction', 'energy_interaction_r2'),
            ('+Full\ninteractions', 'full_interaction_r2'),
            ('Expanded\n(6 pred)', 'expanded_predictors_r2'),
        ]
        labels = []
        values = []
        for lbl, key in r2_specs:
            if key in reg:
                labels.append(lbl)
                values.append(reg[key])
        if not labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('R-squared', fontsize=11)
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    fig.suptitle('Fig 37: Prediction Improvement with Category Interactions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig37_r2_cascade.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig37: {path.name}")


# ========================================================================
# Fig 38: Unused Column Correlation Heatmap (D1)
# ========================================================================
def fig38_unused_correlations(d1):
    """Heatmap of unused columns vs confidence correlation by model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(['gemma-2-2b', 'qwen3-4b']):
        ax = axes[idx]
        corrs = d1.get('unused_correlations', {}).get(model, {})
        if not corrs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        # Sort by absolute correlation
        items = sorted(corrs.items(), key=lambda x: abs(x[1].get('r', 0)), reverse=True)
        labels = [item[0].replace('_', '\n') for item in items]
        r_vals = [item[1].get('r', 0) for item in items]
        p_vals = [item[1].get('p', 1) for item in items]

        colors = ['#e15759' if r < 0 else '#4e79a7' for r in r_vals]
        alphas = [1.0 if p < 0.05 else 0.4 for p in p_vals]

        bars = ax.barh(range(len(labels)), r_vals, color=colors)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Pearson r vs confidence', fontsize=10)
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlim(-0.8, 0.8)
        ax.grid(axis='x', alpha=0.3)

        # Significance legend
        ax.text(0.98, 0.02, 'Solid = p<0.05\nFaded = n.s.',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Fig 38: Unused Column Correlations with Output Confidence',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig38_unused_correlations.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig38: {path.name}")


# ========================================================================
# Fig 39: Multi-Algorithm Jaccard Distribution (D2)
# ========================================================================
def fig39_jaccard_distribution(d2):
    """Violin/box plot of Jaccard scores by model and category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    circuit_results = d2.get('circuit_results', [])

    # By model
    ax = axes[0]
    gemma_j = [r['avg_jaccard'] for r in circuit_results if r['model'] == 'gemma-2-2b']
    qwen_j = [r['avg_jaccard'] for r in circuit_results if r['model'] == 'qwen3-4b']

    bp = ax.boxplot([gemma_j, qwen_j], tick_labels=['GEMMA-2-2B', 'QWEN3-4B'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#4e79a7')
    bp['boxes'][1].set_facecolor('#e15759')
    for box in bp['boxes']:
        box.set_alpha(0.6)

    # Add individual points
    for i, data in enumerate([gemma_j, qwen_j], 1):
        x = np.random.normal(i, 0.04, len(data))
        ax.scatter(x, data, alpha=0.4, s=15, color='black', zorder=3)

    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Valid threshold')
    ax.set_ylabel('Mean Pairwise Jaccard', fontsize=11)
    ax.set_title('By Model', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # By category
    ax = axes[1]
    cat_data = {}
    for cat in ['chemistry', 'geography', 'history']:
        cat_data[cat] = [r['avg_jaccard'] for r in circuit_results if r['category'] == cat]

    bp = ax.boxplot([cat_data['chemistry'], cat_data['geography'], cat_data['history']],
                    tick_labels=['Chemistry', 'Geography', 'History'],
                    patch_artist=True, widths=0.5)
    colors = ['#59a14f', '#f28e2b', '#76b7b2']
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.6)

    for i, (cat, data) in enumerate(cat_data.items(), 1):
        x = np.random.normal(i, 0.04, len(data))
        ax.scatter(x, data, alpha=0.4, s=15, color='black', zorder=3)

    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7)
    ax.set_ylabel('Mean Pairwise Jaccard', fontsize=11)
    ax.set_title('By Category', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 39: Multi-Algorithm Agreement Distribution (4 methods, 60 circuits)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig39_jaccard_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig39: {path.name}")


# ========================================================================
# Fig 40: Method-Pair Agreement Matrix (D2)
# ========================================================================
def fig40_method_pair_matrix(d2):
    """Heatmap of pairwise method agreement."""
    fig, ax = plt.subplots(figsize=(8, 7))

    methods = ['louvain', 'leiden', 'greedy', 'label_propagation']
    n = len(methods)
    matrix = np.ones((n, n))

    pair_means = d2.get('method_pair_means', {})
    for pair, val in pair_means.items():
        parts = pair.split('_vs_')
        if len(parts) == 2:
            m1, m2 = parts
            if m1 in methods and m2 in methods:
                i, j = methods.index(m1), methods.index(m2)
                matrix[i][j] = val
                matrix[j][i] = val

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    short_names = ['Louvain', 'Leiden', 'Greedy', 'LabelProp']
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=10)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = f'{matrix[i][j]:.3f}'
            color = 'white' if matrix[i][j] < 0.3 or matrix[i][j] > 0.8 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    plt.colorbar(im, label='Mean Jaccard Similarity')
    ax.set_title('Fig 40: Pairwise Method Agreement\n(averaged across 60 circuits)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig40_method_pair_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig40: {path.name}")


# ========================================================================
# Fig 41: Minimal Pathway Reduction by Model/Category (D3)
# ========================================================================
def fig41_reduction_breakdown(d3):
    """Grouped bar chart of circuit reduction % by model and category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ok_results = [r for r in d3.get('circuit_results', []) if r.get('status') == 'ok']

    # By model
    ax = axes[0]
    for i, model in enumerate(['gemma-2-2b', 'qwen3-4b']):
        mr = [r['reduction_pct'] for r in ok_results if r['model'] == model]
        if mr:
            bp = ax.boxplot([mr], positions=[i], widths=0.5, patch_artist=True)
            bp['boxes'][0].set_facecolor('#4e79a7' if i == 0 else '#e15759')
            bp['boxes'][0].set_alpha(0.6)
            x = np.random.normal(i, 0.04, len(mr))
            ax.scatter(x, mr, alpha=0.4, s=20, color='black', zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['GEMMA-2-2B', 'QWEN3-4B'], fontsize=10)
    ax.set_ylabel('Circuit Reduction %', fontsize=11)
    ax.set_title('By Model', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # By model x category (grouped bars)
    ax = axes[1]
    categories = ['chemistry', 'geography', 'history']
    models = ['gemma-2-2b', 'qwen3-4b']
    x_pos = np.arange(len(categories))
    width = 0.35

    for i, model in enumerate(models):
        means = []
        stds = []
        for cat in categories:
            vals = [r['reduction_pct'] for r in ok_results
                    if r['model'] == model and r['category'] == cat]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        color = '#4e79a7' if i == 0 else '#e15759'
        ax.bar(x_pos + i * width - width/2, means, width, yerr=stds,
               label=model.upper(), color=color, alpha=0.7, edgecolor='black',
               linewidth=0.5, capsize=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.capitalize() for c in categories], fontsize=10)
    ax.set_ylabel('Reduction %', fontsize=11)
    ax.set_title('By Model x Category', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 41: Minimal Pathway Circuit Reduction (60 circuits)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig41_reduction_breakdown.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig41: {path.name}")


# ========================================================================
# Fig 42: Bottleneck Width vs Weight Retention (D3)
# ========================================================================
def fig42_bn_vs_retention(d3):
    """Scatter: bottleneck width vs weight retention, colored by model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ok_results = [r for r in d3.get('circuit_results', []) if r.get('status') == 'ok']

    # BN layer histogram
    ax = axes[0]
    gemma_bn = [r['bottleneck_layer'] for r in ok_results if r['model'] == 'gemma-2-2b']
    qwen_bn = [r['bottleneck_layer'] for r in ok_results if r['model'] == 'qwen3-4b']

    bins_g = range(0, 27, 2)
    bins_q = range(0, 37, 2)

    ax.hist(gemma_bn, bins=list(bins_g), alpha=0.6, color='#4e79a7', label='GEMMA-2-2B', edgecolor='black')
    ax.hist(qwen_bn, bins=list(bins_q), alpha=0.6, color='#e15759', label='QWEN3-4B', edgecolor='black')
    ax.set_xlabel('Bottleneck Layer', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Bottleneck Layer Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Reduction vs path length
    ax = axes[1]
    for model, color, marker in [('gemma-2-2b', '#4e79a7', 'o'), ('qwen3-4b', '#e15759', 's')]:
        mr = [r for r in ok_results if r['model'] == model]
        if not mr:
            continue
        x = [r['reduction_pct'] for r in mr]
        y = [r['weight_retention'] for r in mr]
        ax.scatter(x, y, c=color, marker=marker, s=40, alpha=0.6, label=model.upper(), edgecolors='black', linewidth=0.5)

        # Add regression line if enough points
        if len(x) >= 2:
            x_arr, y_arr = np.array(x), np.array(y)
            coeffs = np.polyfit(x_arr, y_arr, 1)
            p = np.poly1d(coeffs)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            ax.plot(x_line, p(x_line), '--', color=color, linewidth=2, alpha=0.7)
            # Compute r-value
            ss_res = np.sum((y_arr - p(x_arr)) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r_val = math.copysign(math.sqrt(abs(r2)), coeffs[0])
            ax.annotate(f'r={r_val:.2f}', xy=(x_arr.mean(), p(x_arr.mean())),
                        fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Circuit Reduction %', fontsize=11)
    ax.set_ylabel('Weight Retention (fraction)', fontsize=11)
    ax.set_title('Reduction vs Weight Retention', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Fig 42: Minimal Pathway Structure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig42_bn_vs_retention.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig42: {path.name}")


# ========================================================================
# Fig 43: Steering Effect by Feature and Layer (D4)
# ========================================================================
def fig43_steering_effects(d4):
    """Bar chart of mean steering effect per feature."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    enriched = d4.get('enriched_results', [])
    if not enriched:
        print("  Fig43: No steering data, skipping")
        return

    # Per-feature mean effect
    ax = axes[0]
    feature_effects = defaultdict(list)
    for r in enriched:
        feature_effects[r['feature_label']].append(r['effect']['mean_logprob_shift'])

    features = sorted(feature_effects.keys())
    means = [np.mean(feature_effects[f]) for f in features]
    stds = [np.std(feature_effects[f]) for f in features]

    colors = plt.cm.Set2(np.linspace(0, 1, len(features)))
    bars = ax.bar(range(len(features)), means, yerr=stds, color=colors,
                  edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Mean Logprob Shift', fontsize=11)
    ax.set_title('Steering Effect by Feature', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Per-layer
    ax = axes[1]
    layer_effects = defaultdict(list)
    for r in enriched:
        layer_effects[r['layer']].append(r['effect']['mean_logprob_shift'])

    layers = sorted(layer_effects.keys())
    layer_means = [np.mean(layer_effects[l]) for l in layers]
    layer_stds = [np.std(layer_effects[l]) for l in layers]

    colors_l = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
    ax.bar(range(len(layers)), layer_means, yerr=layer_stds,
           color=colors_l[:len(layers)], edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=10)
    ax.set_ylabel('Mean Logprob Shift', fontsize=11)
    ax.set_title('Steering Effect by Layer', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Fig 43: Steering Effect Size Analysis (20 experiments)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig43_steering_effects.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig43: {path.name}")


# ========================================================================
# Fig 44: Convergence vs Steering Effect (D4)
# ========================================================================
def fig44_convergence_vs_effect(d4):
    """Scatter: avg_convergence vs KL divergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    enriched = d4.get('enriched_results', [])
    if not enriched:
        print("  Fig44: No steering data, skipping")
        return

    # Convergence vs KL
    ax = axes[0]
    x = [r['avg_convergence'] for r in enriched]
    y = [r['effect']['kl_divergence'] for r in enriched]
    strengths = [r['strength'] for r in enriched]

    # Color by strength direction
    colors = ['#e15759' if s < 0 else '#4e79a7' for s in strengths]
    ax.scatter(x, y, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Avg Convergence (bottleneck library)', fontsize=11)
    ax.set_ylabel('KL Divergence (steered vs default)', fontsize=11)
    ax.set_title('Convergence vs Steering Effect', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add correlation line
    if len(x) >= 3:
        mx = np.mean(x)
        my = np.mean(y)
        num = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
        denom = sum((x[i] - mx)**2 for i in range(len(x)))
        if denom > 0:
            slope = num / denom
            intercept = my - slope * mx
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e15759', label='Suppress (neg strength)'),
        Patch(facecolor='#4e79a7', label='Amplify (pos strength)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # circuits_appeared_in vs effect
    ax = axes[1]
    x2 = [r['circuits_appeared_in'] for r in enriched]
    y2 = [r['effect']['mean_logprob_shift'] for r in enriched]
    ax.scatter(x2, y2, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Circuits Appeared In', fontsize=11)
    ax.set_ylabel('Mean Logprob Shift', fontsize=11)
    ax.set_title('Feature Ubiquity vs Effect Size', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    fig.suptitle('Fig 44: Steering Prediction Validation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = FIG_DIR / 'fig44_convergence_vs_effect.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig44: {path.name}")


# ========================================================================
# Main
# ========================================================================
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: EXPANSION FIGURES (Fig 37-44)")
    print("=" * 70)

    # Load all results
    print("\n[1/2] Loading analysis results...")
    d1 = load_json(D1_RESULTS) if D1_RESULTS.exists() else {}
    d2 = load_json(D2_RESULTS) if D2_RESULTS.exists() else {}
    d3 = load_json(D3_RESULTS) if D3_RESULTS.exists() else {}
    d4 = load_json(D4_RESULTS) if D4_RESULTS.exists() else {}
    print(f"  D1: {'loaded' if d1 else 'missing'}")
    print(f"  D2: {'loaded' if d2 else 'missing'}")
    print(f"  D3: {'loaded' if d3 else 'missing'}")
    print(f"  D4: {'loaded' if d4 else 'missing'}")

    print("\n[2/2] Generating figures...")

    if d1:
        fig37_r2_cascade(d1)
        fig38_unused_correlations(d1)

    if d2:
        fig39_jaccard_distribution(d2)
        fig40_method_pair_matrix(d2)

    if d3:
        fig41_reduction_breakdown(d3)
        fig42_bn_vs_retention(d3)

    if d4:
        fig43_steering_effects(d4)
        fig44_convergence_vs_effect(d4)

    print("\n" + "=" * 70)
    print("  COMPLETE — 8 figures generated")
    print("=" * 70)
    print(f"  Output: {FIG_DIR}")
    print("\nDone!")


if __name__ == '__main__':
    main()
