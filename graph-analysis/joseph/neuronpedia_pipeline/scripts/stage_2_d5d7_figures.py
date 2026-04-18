#!/usr/bin/env python3
"""Generate figures for D5 (expanded steering), D6 (annotation), D7 (polysemanticity).

Figures:
  fig45: Expanded steering results heatmap (feature × circuit, colored by change)
  fig46: Steering success by domain and direction
  fig47: Polysemanticity: semantic category distribution
  fig48: Polysemanticity: layer-wise polysemanticity rate with category stacks

Usage:
    python scripts/stage_2_d5d7_figures.py
"""

import json
import sys
import io
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
FIG_DIR = DATA_DIR / 'stage_2_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_steering_results():
    with open(DATA_DIR / 'stage_3_steering' / 'steering_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_polysemanticity_results():
    with open(DATA_DIR / 'stage_2_polysemanticity' / 'polysemanticity_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def fig45_steering_heatmap(steering_data):
    """Heatmap of steering results: features × (circuit × strength)."""
    results = steering_data['results']

    # Get all D5 features
    d5_features = ['L2_F25751073', 'L5_F7993995', 'L9_F125286525', 'L1_F99962728', 'L7_F4828270']
    orig_features = ['L0_F1813559', 'L3_F5150441', 'L4_F110446948', 'L6_F2586668', 'L24_F88478228']
    all_features = orig_features + d5_features

    # Build matrix: rows = features, cols = experiments
    # For D5: 3 circuits × 2 strengths = 6 cols
    # For orig: variable circuits × 2 strengths
    # Let's focus on consistent format: all 10 features across the 3 D5 circuits

    circuits = ['sodium', 'japan', 'wwii']
    circuit_map = {
        'gemma-2-2b_the-chemical-symbol-for-sodium-is': 'sodium',
        'gemma-2-2b_the-capital-of-japan-is': 'japan',
        'gemma-2-2b_world-war-ii-ended-in': 'wwii',
    }
    strengths = [-20, 20]
    cols = []
    for c in circuits:
        for s in strengths:
            cols.append(f"{c}\n{'+' if s > 0 else ''}{s}")

    # Only D5 results (which use the 3 standard circuits)
    matrix = np.full((len(d5_features), len(cols)), np.nan)
    kl_matrix = np.full((len(d5_features), len(cols)), 0.0)

    for exp in results:
        fl = exp.get('feature_label', '')
        if fl not in d5_features:
            continue
        row = d5_features.index(fl)

        cid = exp.get('circuit_id', '')
        cname = circuit_map.get(cid, '')
        s = exp.get('strength', 0)

        col_label = f"{cname}\n{'+' if s > 0 else ''}{s}"
        if col_label in cols:
            col = cols.index(col_label)
            resp = exp.get('response', {})
            steered = resp.get('STEERED', '').strip()
            default = resp.get('DEFAULT', '').strip()
            changed = 1.0 if steered != default else 0.0
            matrix[row, col] = changed

            # Calculate KL
            slps = resp.get('steeredLogProbs', [])
            dlps = resp.get('defaultLogProbs', [])
            if slps and dlps:
                kl = 0.0
                for sp, dp in zip(slps[:5], dlps[:5]):
                    s_prob = math.exp(sp.get('logprob', -10))
                    d_prob = math.exp(dp.get('logprob', -10))
                    if d_prob > 0 and s_prob > 0:
                        kl += d_prob * math.log(d_prob / max(s_prob, 1e-10))
                kl_matrix[row, col] = abs(kl)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use KL divergence as color intensity, with change/no-change as overlay
    im = ax.imshow(kl_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)

    # Add text annotations
    for i in range(len(d5_features)):
        for j in range(len(cols)):
            if not np.isnan(matrix[i, j]):
                changed = matrix[i, j] > 0.5
                kl = kl_matrix[i, j]
                text = f"{'CHANGED' if changed else 'same'}\nKL={kl:.3f}"
                color = 'white' if kl > 0.2 else 'black'
                fontweight = 'bold' if changed else 'normal'
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                        color=color, fontweight=fontweight)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(d5_features)))
    ax.set_yticklabels(d5_features, fontsize=9)
    ax.set_title('Fig 45: D5 Expanded Steering Results\n(KL Divergence Heatmap, Text Change Overlay)', fontsize=11)
    ax.set_xlabel('Target Circuit × Strength')
    ax.set_ylabel('Steered Feature')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|KL Divergence|')

    plt.tight_layout()
    path = FIG_DIR / 'fig45_d5_steering_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig46_steering_by_domain(steering_data):
    """Bar chart: steering success rate by domain and direction."""
    results = steering_data['results']

    d5_features = ['L2_F25751073', 'L5_F7993995', 'L9_F125286525', 'L1_F99962728', 'L7_F4828270']

    categories = ['chemistry', 'geography', 'history']
    directions = ['suppress (-20)', 'amplify (+20)']

    counts = {cat: {d: {'changed': 0, 'total': 0} for d in directions} for cat in categories}

    for exp in results:
        fl = exp.get('feature_label', '')
        if fl not in d5_features:
            continue
        cat = exp.get('category', '')
        s = exp.get('strength', 0)
        direction = 'suppress (-20)' if s < 0 else 'amplify (+20)'

        resp = exp.get('response', {})
        changed = resp.get('STEERED', '').strip() != resp.get('DEFAULT', '').strip()

        if cat in counts:
            counts[cat][direction]['total'] += 1
            if changed:
                counts[cat][direction]['changed'] += 1

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(categories))
    width = 0.35
    colors = ['#2196F3', '#FF5722']

    for i, d in enumerate(directions):
        rates = []
        for cat in categories:
            total = counts[cat][d]['total']
            changed = counts[cat][d]['changed']
            rate = changed / total * 100 if total > 0 else 0
            rates.append(rate)

        bars = ax.bar(x + i * width, rates, width, label=d, color=colors[i], alpha=0.8)

        # Add count labels
        for j, (bar, cat) in enumerate(zip(bars, categories)):
            total = counts[cat][d]['total']
            changed = counts[cat][d]['changed']
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{changed}/{total}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Knowledge Domain')
    ax.set_ylabel('Text Change Rate (%)')
    ax.set_title('Fig 46: D5 Steering Success by Domain and Direction\n(5 New Features × 3 Circuits × ±20 Strength)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 80)
    ax.legend()
    ax.axhline(y=23.3, color='gray', linestyle='--', alpha=0.5, label='Overall D5 rate')
    ax.text(2.5, 25, 'Overall: 23.3%', fontsize=8, color='gray')

    plt.tight_layout()
    path = FIG_DIR / 'fig46_d5_steering_by_domain.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig47_semantic_categories(poly_data):
    """Horizontal bar chart of semantic category distribution."""
    cats = poly_data['overall']['category_counts']

    # Sort by count
    sorted_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)
    names = [c[0] for c in sorted_cats]
    values = [c[1] for c in sorted_cats]

    # Color mapping
    cat_colors = {
        'OTHER': '#9E9E9E',
        'CODE': '#2196F3',
        'LANGUAGE': '#4CAF50',
        'DOMAIN_SPECIFIC': '#FF9800',
        'SCIENCE': '#9C27B0',
        'MATH': '#F44336',
        'ENTITIES': '#00BCD4',
        'CONCEPTS': '#795548',
        'MULTILINGUAL': '#E91E63',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: horizontal bar chart
    colors = [cat_colors.get(n, '#666') for n in names]
    bars = ax1.barh(range(len(names)), values, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Feature Count')
    ax1.set_title('Semantic Category Distribution\n(130 Annotated GEMMA Features)')
    ax1.invert_yaxis()

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val} ({val/130*100:.0f}%)', va='center', fontsize=8)

    # Right: pie chart of mono vs poly
    poly_count = poly_data['overall']['polysemantic_count']
    mono_count = poly_data['overall']['monosemantic_count']
    ax2.pie([mono_count, poly_count],
            labels=[f'Monosemantic\n({mono_count}, {mono_count/130*100:.0f}%)',
                    f'Polysemantic\n({poly_count}, {poly_count/130*100:.0f}%)'],
            colors=['#4CAF50', '#FF5722'],
            autopct='', startangle=90, explode=[0, 0.05])
    ax2.set_title('Monosemantic vs Polysemantic\nBottleneck Features')

    plt.suptitle('Fig 47: Polysemanticity Analysis — Semantic Category Distribution', fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIG_DIR / 'fig47_semantic_categories.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig48_layer_polysemanticity(poly_data):
    """Layer-wise polysemanticity rate with stacked category bars."""
    layer_data = poly_data['layer_analysis']

    # Sort layers numerically
    layers = sorted(layer_data.keys(), key=int)
    layer_labels = [f'L{l}' for l in layers]

    cat_colors = {
        'OTHER': '#9E9E9E',
        'CODE': '#2196F3',
        'LANGUAGE': '#4CAF50',
        'DOMAIN_SPECIFIC': '#FF9800',
        'SCIENCE': '#9C27B0',
        'MATH': '#F44336',
        'ENTITIES': '#00BCD4',
        'CONCEPTS': '#795548',
        'MULTILINGUAL': '#E91E63',
    }

    all_cats = ['CODE', 'LANGUAGE', 'OTHER', 'DOMAIN_SPECIFIC', 'SCIENCE', 'MATH', 'ENTITIES', 'CONCEPTS', 'MULTILINGUAL']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1]})

    # Top: stacked bar chart of categories per layer
    bottoms = np.zeros(len(layers))
    for cat in all_cats:
        vals = []
        for l in layers:
            ld = layer_data[l]
            vals.append(ld['categories'].get(cat, 0))
        vals = np.array(vals)
        if vals.sum() > 0:
            ax1.bar(range(len(layers)), vals, bottom=bottoms,
                    label=cat, color=cat_colors.get(cat, '#666'), alpha=0.8)
            bottoms += vals

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layer_labels, fontsize=8, rotation=45)
    ax1.set_ylabel('Feature Count')
    ax1.set_title('Semantic Categories by Layer')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

    # Bottom: polysemanticity rate line
    poly_rates = [layer_data[l]['poly_rate'] * 100 for l in layers]
    totals = [sum(layer_data[l]['categories'].values()) for l in layers]

    ax2.bar(range(len(layers)), poly_rates, color='#FF5722', alpha=0.6)
    ax2.plot(range(len(layers)), poly_rates, 'o-', color='#D32F2F', markersize=5, linewidth=1.5)

    # Annotate with totals
    for i, (rate, total) in enumerate(zip(poly_rates, totals)):
        if rate > 0:
            ax2.text(i, rate + 2, f'{rate:.0f}%\n(n={total})', ha='center', fontsize=7)
        else:
            ax2.text(i, 2, f'n={total}', ha='center', fontsize=6, color='gray')

    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layer_labels, fontsize=8, rotation=45)
    ax2.set_ylabel('Polysemanticity Rate (%)')
    ax2.set_title('Polysemanticity Rate by Layer (Features with 2+ Categories)')
    ax2.set_ylim(0, 110)
    ax2.axhline(y=12.3, color='gray', linestyle='--', alpha=0.5)
    ax2.text(len(layers)-1, 14, 'Overall: 12.3%', fontsize=8, color='gray', ha='right')

    plt.suptitle('Fig 48: Layer-wise Polysemanticity Analysis', fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / 'fig48_layer_polysemanticity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("  D5-D7 FIGURE GENERATION")
    print("=" * 70)

    # Load data
    steering = load_steering_results()
    poly = load_polysemanticity_results()

    print("\n[1/4] Fig 45: D5 Steering Heatmap...")
    fig45_steering_heatmap(steering)

    print("[2/4] Fig 46: D5 Steering by Domain...")
    fig46_steering_by_domain(steering)

    print("[3/4] Fig 47: Semantic Categories...")
    fig47_semantic_categories(poly)

    print("[4/4] Fig 48: Layer Polysemanticity...")
    fig48_layer_polysemanticity(poly)

    print("\n" + "=" * 70)
    print("  ALL FIGURES GENERATED (fig45-fig48)")
    print("=" * 70)


if __name__ == '__main__':
    main()
