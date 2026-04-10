#!/usr/bin/env python3
"""Stage 2: Bolster Figures - Generate Figs 30-36 from C1-C4 analysis results.

Reads the output JSON from all four C-analysis scripts and creates
publication-quality figures for the paper.

Figures:
  Fig 30: Layer-to-layer connectivity heatmap (C1)
  Fig 31: Skip connection frequency by model (C1)
  Fig 32: Feature co-activation heatmap (C2)
  Fig 33: Feature cluster dendrogram (C2) [requires scipy]
  Fig 34: Output decomposition stacked bars by domain (C3)
  Fig 35: Bottleneck path contribution vs confidence scatter (C3)
  Fig 36: Cook's distance plot + outlier identification (C4)

Usage:
    python scripts/stage_2_bolster_figures.py
"""

import json
import sys
import io
import os
import math
import datetime
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ERROR] matplotlib/numpy not available. Cannot generate figures.")
    sys.exit(1)

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
FIGURES_DIR = DATA_DIR / 'stage_2_figures'

# Analysis result files
EDGE_FLOW_RESULTS = DATA_DIR / 'stage_2_edge_flow' / 'edge_flow_results.json'
COACTIVATION_RESULTS = DATA_DIR / 'stage_2_coactivation' / 'coactivation_results.json'
DECOMPOSITION_RESULTS = DATA_DIR / 'stage_2_output_decomposition' / 'output_decomposition_results.json'
OUTLIER_RESULTS = DATA_DIR / 'stage_2_outlier_analysis' / 'outlier_results.json'

MODELS = ['gemma-2-2b', 'qwen3-4b']
MODEL_LAYERS = {'gemma-2-2b': 26, 'qwen3-4b': 36}
CATEGORIES = ['chemistry', 'geography', 'history']
CAT_COLORS = {'chemistry': '#e74c3c', 'geography': '#2ecc71', 'history': '#3498db'}
MODEL_COLORS = {'gemma-2-2b': '#e67e22', 'qwen3-4b': '#9b59b6'}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fig30_connectivity_heatmap():
    """Fig 30: Layer-to-layer connectivity heatmap (from C1 edge flow)."""
    print("  Fig 30: Layer-to-layer connectivity heatmap...")
    results = load_json(EDGE_FLOW_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        pairs = results.get('model_top_pairs', {}).get(model, [])
        nl = MODEL_LAYERS[model]

        # Build matrix from top pairs
        matrix = np.zeros((nl, nl))
        for p in pairs:
            matrix[p['src']][p['tgt']] = p['weight']

        # Also fill from per-circuit data to get full matrix
        # Use aggregate flow data to build heatmap
        overall = results.get('model_overall', {}).get(model, {})
        flow = overall.get('avg_flow', [])

        # Create a meaningful heatmap: show flow direction
        # Use outgoing weight as columns, incoming as rows
        flow_data = np.zeros(nl)
        for entry in flow:
            l = entry['layer']
            if l < nl:
                flow_data[l] = entry['avg_outgoing']

        # Since we don't have full matrix in results, create flow profile heatmap
        # Show per-layer incoming vs outgoing
        incoming = np.array([entry['avg_incoming'] for entry in flow[:nl]])
        outgoing = np.array([entry['avg_outgoing'] for entry in flow[:nl]])

        # Normalize for visualization
        max_val = max(incoming.max(), outgoing.max())
        if max_val > 0:
            incoming_n = incoming / max_val
            outgoing_n = outgoing / max_val

        data_2d = np.vstack([incoming_n, outgoing_n])
        im = ax.imshow(data_2d, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Incoming', 'Outgoing'])
        ax.set_xlabel('Layer')
        ax.set_title(f'{model.upper()}\nLayer Flow Profile', fontsize=12)

        # Add value annotations for key layers
        for j in range(nl):
            if incoming_n[j] > 0.5:
                ax.text(j, 0, f'{incoming_n[j]:.1f}', ha='center', va='center', fontsize=5, color='white')
            if outgoing_n[j] > 0.5:
                ax.text(j, 1, f'{outgoing_n[j]:.1f}', ha='center', va='center', fontsize=5, color='white')

        plt.colorbar(im, ax=ax, label='Normalized Weight')

    fig.suptitle('Fig 30: Layer Flow Profiles (Incoming vs Outgoing Edge Weight)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig30_layer_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig31_skip_connections():
    """Fig 31: Skip connection frequency by model (from C1)."""
    print("  Fig 31: Skip connection frequency...")
    results = load_json(EDGE_FLOW_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Skip ratio comparison
    ax = axes[0]
    x_pos = []
    heights = []
    colors = []
    labels = []
    pos = 0
    for model in MODELS:
        mc = results.get('by_model_category', {}).get(model, {})
        for cat in CATEGORIES:
            data = mc.get(cat, {})
            if data:
                x_pos.append(pos)
                heights.append(data['avg_skip']['skip_ratio'])
                colors.append(CAT_COLORS[cat])
                labels.append(f"{model.split('-')[0]}\n{cat[:4]}")
                pos += 1
        pos += 0.5

    ax.bar(x_pos, heights, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Skip Connection Ratio')
    ax.set_title('Skip Ratio by Model & Domain')
    ax.set_ylim(0.7, 0.9)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

    # Right: Skip weight fraction
    ax = axes[1]
    x_pos2 = []
    heights2 = []
    colors2 = []
    labels2 = []
    pos = 0
    for model in MODELS:
        mc = results.get('by_model_category', {}).get(model, {})
        for cat in CATEGORIES:
            data = mc.get(cat, {})
            if data:
                x_pos2.append(pos)
                heights2.append(data['avg_skip']['skip_weight_fraction'])
                colors2.append(CAT_COLORS[cat])
                labels2.append(f"{model.split('-')[0]}\n{cat[:4]}")
                pos += 1
        pos += 0.5

    ax.bar(x_pos2, heights2, color=colors2, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos2)
    ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel('Skip Weight Fraction')
    ax.set_title('Weight Carried by Skip Connections')
    ax.set_ylim(0.6, 0.9)

    # Legend
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
    axes[0].legend(handles=patches, loc='lower left')

    fig.suptitle('Fig 31: Skip Connection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig31_skip_connections.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig32_coactivation_heatmap():
    """Fig 32: Feature co-activation summary (from C2)."""
    print("  Fig 32: Feature co-activation heatmap...")
    results = load_json(COACTIVATION_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Top co-activating pairs (bar chart)
    ax = axes[0]
    top_pairs = results.get('top_50_jaccard', [])[:20]
    if top_pairs:
        labels_p = [f"{p['feature_a'].split('_')[0]}&{p['feature_b'].split('_')[0]}" for p in top_pairs]
        jaccards = [p['jaccard'] for p in top_pairs]
        y_pos = range(len(top_pairs))
        ax.barh(y_pos, jaccards, color='#3498db', edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_p, fontsize=7)
        ax.set_xlabel('Jaccard Similarity')
        ax.set_title('Top 20 Co-Activating Feature Pairs')
        ax.invert_yaxis()

    # Right: Layer analysis - same vs cross layer
    ax = axes[1]
    la = results.get('layer_analysis', {})
    categories_bar = ['Same Layer', 'Cross Layer']
    values = [la.get('avg_same_layer_jaccard', 0), la.get('avg_cross_layer_jaccard', 0)]
    counts = [la.get('same_layer_pairs', 0), la.get('cross_layer_pairs', 0)]

    bars = ax.bar(categories_bar, values, color=['#e74c3c', '#2ecc71'],
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Average Jaccard Similarity')
    ax.set_title('Same-Layer vs Cross-Layer Co-Activation')

    # Add count annotations
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'n={count}', ha='center', fontsize=10)

    ratio = la.get('ratio', 0)
    ax.text(0.5, 0.85, f'Ratio: {ratio:.2f}x',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat'))

    fig.suptitle('Fig 32: Feature Co-Activation Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig32_coactivation.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig33_feature_clusters():
    """Fig 33: Feature cluster dendrogram (from C2, requires scipy)."""
    print("  Fig 33: Feature cluster dendrogram...")
    results = load_json(COACTIVATION_RESULTS)

    clustering = results.get('clustering', {})
    if not clustering:
        print("    [SKIP] No clustering data available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Show cluster membership as a grouped bar chart
    clusters = clustering.get('clusters_5', {})
    if not clusters:
        print("    [SKIP] No cluster data")
        return

    # Layer distribution per cluster
    cluster_layers = {}
    for cid, members in clusters.items():
        layers = []
        for m in members:
            parts = m.split('_')
            if len(parts) >= 1:
                try:
                    layer = int(parts[0].replace('L', ''))
                    layers.append(layer)
                except ValueError:
                    pass
        cluster_layers[cid] = layers

    # Plot layer distributions as violin/box
    data_for_box = []
    labels_box = []
    for cid in sorted(cluster_layers.keys()):
        if cluster_layers[cid]:
            data_for_box.append(cluster_layers[cid])
            labels_box.append(f'Cluster {cid}\n(n={len(cluster_layers[cid])})')

    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_box, patch_artist=True)
        colors_box = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Layer')
        ax.set_title('Feature Cluster Layer Distributions')
        ax.axhline(y=7, color='orange', linestyle='--', alpha=0.5, label='GEMMA bottleneck (L5-7)')
        ax.axhline(y=25, color='purple', linestyle='--', alpha=0.5, label='QWEN bottleneck (L22-25)')
        ax.legend()

    fig.suptitle('Fig 33: Hierarchical Feature Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig33_feature_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig34_output_decomposition():
    """Fig 34: Output decomposition stacked bars by domain (from C3)."""
    print("  Fig 34: Output decomposition stacked bars...")
    results = load_json(DECOMPOSITION_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        cs = results.get('category_stats', {}).get(model, {})
        if not cs:
            continue

        cats = sorted(cs.keys())
        bn_ratios = [cs[c]['avg_all_edges_bn_ratio'] for c in cats]
        non_bn_ratios = [1 - r for r in bn_ratios]

        x = np.arange(len(cats))
        width = 0.5

        ax.bar(x, bn_ratios, width, label='Bottleneck', color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax.bar(x, non_bn_ratios, width, bottom=bn_ratios, label='Non-Bottleneck',
               color='#3498db', edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cats])
        ax.set_ylabel('Fraction of Edge Weight')
        ax.set_title(f'{model.upper()}')
        ax.legend()
        ax.set_ylim(0, 1.05)

        # Annotate with confidence
        for i, cat in enumerate(cats):
            conf = cs[cat]['avg_confidence']
            ax.text(i, 1.01, f'conf={conf:.2f}', ha='center', fontsize=9, color='gray')

    fig.suptitle('Fig 34: Output Decomposition — Bottleneck vs Non-Bottleneck Contribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig34_output_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig35_bn_vs_confidence():
    """Fig 35: Bottleneck contribution vs confidence scatter (from C3)."""
    print("  Fig 35: Bottleneck contribution vs confidence scatter...")
    results = load_json(DECOMPOSITION_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        per_circuit = results.get('per_circuit', {})
        corr = results.get('correlations', {}).get(model, {})

        x_vals = []
        y_vals = []
        colors = []
        for cid, data in per_circuit.items():
            if data.get('model') == model:
                x_vals.append(data['all_edges_bn_ratio'])
                y_vals.append(data['confidence'])
                cat = data.get('category', 'unknown')
                colors.append(CAT_COLORS.get(cat, 'gray'))

        if x_vals:
            ax.scatter(x_vals, y_vals, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Add regression line
            if len(x_vals) >= 3:
                x_arr = np.array(x_vals)
                y_arr = np.array(y_vals)
                z = np.polyfit(x_arr, y_arr, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_line, p_line(x_line), 'k--', alpha=0.5)

        ax.set_xlabel('Bottleneck Edge Weight Fraction')
        ax.set_ylabel('Output Probability (Confidence)')

        r_info = corr.get('all_edges_bn_ratio_vs_confidence', {})
        r_val = r_info.get('r', 0)
        p_val = r_info.get('p', 1)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.set_title(f'{model.upper()}\nr={r_val:.3f}, p={p_val:.4f} {sig}')

    # Legend
    patches = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
    axes[0].legend(handles=patches, loc='best')

    fig.suptitle('Fig 35: Bottleneck Contribution vs Model Confidence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig35_bn_vs_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig36_cooks_distance():
    """Fig 36: Cook's distance plot + outlier identification (from C4)."""
    print("  Fig 36: Cook's distance plot...")
    results = load_json(OUTLIER_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        mdata = results.get(model, {})
        if not mdata:
            continue

        all_d = mdata.get('all_cooks_d', [])
        threshold = mdata.get('cooks_threshold', 0)

        if all_d:
            x = range(len(all_d))
            d_vals = [d['cooks_d'] for d in all_d]
            colors = [CAT_COLORS.get(d['category'], 'gray') for d in all_d]

            ax.bar(x, d_vals, color=colors, edgecolor='black', linewidth=0.3)
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold (4/n = {threshold:.3f})')

            # Label outliers
            for i, d in enumerate(all_d):
                if d['cooks_d'] > threshold:
                    slug = d['circuit_id'].split('_', 1)[-1]
                    ax.annotate(slug[:20], (i, d['cooks_d']),
                                fontsize=6, rotation=45, ha='left')

            ax.set_xlabel('Circuit Index')
            ax.set_ylabel("Cook's Distance")
            r2 = mdata['regression']['r_squared']
            ax.set_title(f"{model.upper()} (R²={r2:.3f})")
            ax.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()]
    fig.legend(handles=patches + [plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold')],
               loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fig 36: Cook's Distance — Outlier Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig36_cooks_distance.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GENERATING BOLSTER FIGURES (Figs 30-36)")
    print("=" * 70)

    fig30_connectivity_heatmap()
    fig31_skip_connections()
    fig32_coactivation_heatmap()
    fig33_feature_clusters()
    fig34_output_decomposition()
    fig35_bn_vs_confidence()
    fig36_cooks_distance()

    print("\n" + "=" * 70)
    print("  ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"  Output: {FIGURES_DIR}")
    figs = list(FIGURES_DIR.glob('fig3*.png'))
    for f in sorted(figs):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.0f} KB)")
    print(f"\n  Total new figures: {len(figs)}")
    print("\nDone!")


if __name__ == '__main__':
    main()
