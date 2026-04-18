#!/usr/bin/env python3
"""
Stage 2: Category Breakdown Dashboard

Generates per-category (chemistry/geography/history) breakdown charts
for all key metrics, split by model.

Usage:
    python stage_2_category_breakdown_figures.py
"""

import json
import sys
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
DEEPDIVE_DIR = DATA_DIR / 'stage_2_statistical_deepdive'
LAYER_DIR = DATA_DIR / 'stage_2_layer_energy'
OUTPUT_DIR = DATA_DIR / 'stage_2_figures'

MODELS = ['gemma-2-2b', 'qwen3-4b']
MODEL_LABELS = {'gemma-2-2b': 'GEMMA-2-2B', 'qwen3-4b': 'QWEN3-4B'}
CATEGORIES = ['chemistry', 'geography', 'history']
CAT_COLORS = {'chemistry': '#2196F3', 'geography': '#4CAF50', 'history': '#FF9800'}
CAT_LABELS = {'chemistry': 'Chem', 'geography': 'Geo', 'history': 'Hist'}
MODEL_COLORS = {'gemma-2-2b': '#E91E63', 'qwen3-4b': '#9C27B0'}

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data():
    with open(DEEPDIVE_DIR / 'data_table.json', 'r', encoding='utf-8') as f:
        data_table = json.load(f)
    with open(LAYER_DIR / 'per_circuit_layer_energy.json', 'r', encoding='utf-8') as f:
        layer_data = json.load(f)
    return data_table, layer_data


def get_vals(data_table, model, category, metric):
    return [r[metric] for r in data_table
            if r['model'] == model and r['category'] == category and metric in r]


def fig_category_dashboard(data_table, layer_data):
    """
    Big 4x3 dashboard: 4 rows of metrics, 3 grouped by category within each.
    Each panel = one metric, showing both models side by side with category grouping.
    """

    metrics = [
        ('output_probability', 'Output Probability\n(Confidence)', False),
        ('total_energy', 'Total Activation Energy', False),
        ('total_nodes', 'Total Nodes', False),
        ('total_edges', 'Total Edges', False),
        ('num_bottlenecks', 'Num Bottlenecks', False),
        ('avg_bottleneck_layer', 'Avg Bottleneck Layer', False),
        ('median_bottleneck_layer', 'Median Bottleneck Layer', False),
        ('avg_branching_factor', 'Avg Branching Factor', False),
        ('weight_kurtosis', 'Edge Weight Kurtosis', True),
        ('avg_compression_ratio', 'Compression Ratio', False),
        ('mean_bottleneck_convergence', 'Bottleneck Convergence', False),
        ('peak_layer', 'Peak Energy Layer', False),
    ]

    n_metrics = len(metrics)
    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 13))
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(len(CATEGORIES))

    for idx, (metric, label, use_log) in enumerate(metrics):
        ax = axes[idx]

        for mi, model in enumerate(MODELS):
            means = []
            stds = []
            individual_vals = []
            for cat in CATEGORIES:
                vals = get_vals(data_table, model, cat, metric)
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
                individual_vals.append(vals)

            offset = -bar_width/2 + mi * bar_width
            bars = ax.bar(x + offset, means, bar_width * 0.9,
                         yerr=stds, capsize=3,
                         color=MODEL_COLORS[model], alpha=0.7,
                         label=MODEL_LABELS[model] if idx == 0 else '',
                         edgecolor='white', linewidth=0.5)

            # Overlay individual data points
            for ci, vals in enumerate(individual_vals):
                jitter = np.random.default_rng(42 + mi).uniform(-0.06, 0.06, len(vals))
                ax.scatter([ci + offset] * len(vals) + jitter,
                          vals, color=MODEL_COLORS[model],
                          s=12, alpha=0.4, edgecolors='none', zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES])
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        if use_log:
            ax.set_yscale('log')

    # Remove unused axes if any
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Single legend at top
    handles = [mpatches.Patch(color=MODEL_COLORS[m], alpha=0.7, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12,
              bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('Circuit Metrics by Knowledge Domain\n'
                 'Each bar = 10 circuits; dots = individual circuits; error bars = 1 std',
                 fontsize=14, y=1.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig24_category_breakdown_dashboard.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig24_category_breakdown_dashboard.png")


def fig_category_energy_profiles(layer_data):
    """
    Per-category layer energy profiles: 3x2 grid (category × model).
    Shows all 10 individual circuit traces per panel + mean.
    """
    from pipeline_constants import GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS
    MODEL_LAYERS = {'gemma-2-2b': GEMMA_TOTAL_LAYERS, 'qwen3-4b': QWEN_TOTAL_LAYERS}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for ri, cat in enumerate(CATEGORIES):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            total_layers = MODEL_LAYERS[model]
            x = np.arange(total_layers)

            traces = []
            for slug, data in layer_data.get(model, {}).items():
                if data.get('category') == cat:
                    frac = data['layer_energy_fraction']
                    ax.plot(x, frac, color=CAT_COLORS[cat], alpha=0.25, linewidth=0.8)
                    traces.append(frac)

            if traces:
                mean_trace = np.mean(traces, axis=0)
                ax.plot(x, mean_trace, color=CAT_COLORS[cat], linewidth=2.5,
                       label=f'Mean (n={len(traces)})')

            ax.set_xlim(0, total_layers - 1)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper right', fontsize=8)

            if ri == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f'{cat.title()}\nEnergy Fraction')
            if ri == 2:
                ax.set_xlabel('Layer')

    fig.suptitle('Per-Category Layer Energy Profiles\n'
                 'Individual circuits (faint) + domain mean (bold)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig25_category_energy_profiles.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig25_category_energy_profiles.png")


def fig_category_cumulative(layer_data):
    """
    Per-category cumulative energy curves: 2 panels (one per model),
    all 3 categories overlaid with individual traces.
    """
    from pipeline_constants import GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS
    MODEL_LAYERS = {'gemma-2-2b': GEMMA_TOTAL_LAYERS, 'qwen3-4b': QWEN_TOTAL_LAYERS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        total_layers = MODEL_LAYERS[model]
        x = np.arange(total_layers)

        for cat in CATEGORIES:
            traces = []
            for slug, data in layer_data.get(model, {}).items():
                if data.get('category') == cat:
                    energy = np.array(data['layer_energy'])
                    total = data['total_energy']
                    if total > 0:
                        cum = np.cumsum(energy) / total
                        ax.plot(x, cum, color=CAT_COLORS[cat], alpha=0.15, linewidth=0.7)
                        traces.append(cum)

            if traces:
                mean_cum = np.mean(traces, axis=0)
                ax.plot(x, mean_cum, color=CAT_COLORS[cat], linewidth=2.5,
                       label=f'{cat.title()} (n={len(traces)})')

        ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5)
        ax.axhline(y=0.9, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cumulative Energy Fraction')
        ax.set_title(MODEL_LABELS[model])
        ax.legend(loc='lower right')
        ax.set_xlim(0, total_layers - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Cumulative Energy Accumulation by Category\n'
                 'Individual circuits (faint) + category means (bold)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig26_category_cumulative_energy.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig26_category_cumulative_energy.png")


def fig_category_scatter_matrix(data_table):
    """
    Key scatter plots: confidence vs major metrics, colored by category.
    2x2 grid of the most interesting relationships.
    """
    scatter_pairs = [
        ('total_energy', 'output_probability', 'Total Energy', 'Confidence'),
        ('avg_bottleneck_layer', 'output_probability', 'Avg Bottleneck Layer', 'Confidence'),
        ('total_nodes', 'total_energy', 'Total Nodes', 'Total Energy'),
        ('weight_kurtosis', 'output_probability', 'Edge Weight Kurtosis', 'Confidence'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
        ax = axes[idx]

        for model in MODELS:
            marker = 'o' if model == 'gemma-2-2b' else 's'
            for cat in CATEGORIES:
                x_vals = get_vals(data_table, model, cat, x_metric)
                y_vals = get_vals(data_table, model, cat, y_metric)
                if x_vals and y_vals:
                    ax.scatter(x_vals, y_vals, color=CAT_COLORS[cat],
                              marker=marker, s=40, alpha=0.6,
                              edgecolors='white', linewidth=0.5,
                              label=f'{CAT_LABELS[cat]} ({MODEL_LABELS[model][:5]})' if idx == 0 else '')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.2)

    # Build combined legend
    handles = []
    for cat in CATEGORIES:
        handles.append(mpatches.Patch(color=CAT_COLORS[cat], label=cat.title()))
    handles.append(plt.Line2D([0], [0], marker='o', color='grey', linestyle='None',
                              markersize=7, label='GEMMA'))
    handles.append(plt.Line2D([0], [0], marker='s', color='grey', linestyle='None',
                              markersize=7, label='QWEN'))
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=10,
              bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('Key Metric Relationships by Category\n'
                 'Circles = GEMMA, Squares = QWEN; Color = Domain',
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig27_category_scatter_matrix.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig27_category_scatter_matrix.png")


def fig_category_summary_table(data_table, layer_data):
    """
    Text-heavy summary table rendered as a figure.
    Shows mean +/- std for key metrics per model per category.
    """
    from pipeline_constants import GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS
    MODEL_LAYERS = {'gemma-2-2b': GEMMA_TOTAL_LAYERS, 'qwen3-4b': QWEN_TOTAL_LAYERS}

    key_metrics = [
        ('output_probability', 'Confidence', '.3f'),
        ('total_energy', 'Total Energy', '.0f'),
        ('total_nodes', 'Nodes', '.0f'),
        ('total_edges', 'Edges', '.0f'),
        ('num_bottlenecks', 'Bottlenecks', '.1f'),
        ('avg_bottleneck_layer', 'Avg BN Layer', '.1f'),
        ('median_bottleneck_layer', 'Med BN Layer', '.1f'),
        ('avg_branching_factor', 'Branching', '.2f'),
        ('weight_kurtosis', 'Wt Kurtosis', '.1f'),
        ('avg_compression_ratio', 'Compression', '.3f'),
        ('peak_layer', 'Peak Layer', '.1f'),
        ('mean_bottleneck_convergence', 'BN Convergence', '.3f'),
    ]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')

    # Build table data
    col_labels = ['Metric']
    for model in MODELS:
        for cat in CATEGORIES:
            col_labels.append(f'{MODEL_LABELS[model][:5]}\n{cat.title()[:4]}')

    cell_text = []
    cell_colors = []
    for metric, label, fmt in key_metrics:
        row = [label]
        row_colors = ['#f0f0f0']
        for model in MODELS:
            for cat in CATEGORIES:
                vals = get_vals(data_table, model, cat, metric)
                if vals:
                    m, s = np.mean(vals), np.std(vals)
                    row.append(f'{m:{fmt}}\n+/-{s:{fmt}}')
                else:
                    row.append('N/A')
                row_colors.append(CAT_COLORS[cat] + '20')  # light tint
        cell_text.append(row)
        cell_colors.append(row_colors)

    table = ax.table(cellText=cell_text,
                     colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)

    # Style metric names
    for i in range(len(cell_text)):
        cell = table[i + 1, 0]
        cell.set_text_props(fontweight='bold')

    fig.suptitle('Complete Category Breakdown: Mean +/- Std (n=10 per cell)',
                 fontsize=14, y=0.95)
    plt.savefig(OUTPUT_DIR / 'fig28_category_summary_table.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig28_category_summary_table.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CATEGORY BREAKDOWN FIGURES")
    print("=" * 60)

    print("\nLoading data...")
    data_table, layer_data = load_data()
    print(f"  {len(data_table)} circuits in data table")

    print("\nGenerating figures...\n")
    fig_category_dashboard(data_table, layer_data)
    fig_category_energy_profiles(layer_data)
    fig_category_cumulative(layer_data)
    fig_category_scatter_matrix(data_table)
    fig_category_summary_table(data_table, layer_data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
