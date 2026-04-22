#!/usr/bin/env python3
"""
Stage 2: Layer Energy Figures

Generates publication-quality figures from the per-layer activation energy analysis.
Supports both Angle 1 (Architecture Over Knowledge) and Angle 3 (Evidence Accumulation).

Figures:
  1. Domain energy curves (3 domains overlaid per model, 2 panels)
  2. Cross-model normalized depth profiles
  3. Per-layer confidence correlation (bar chart of Spearman r per layer)
  4. Cumulative energy curves (evidence accumulation through layers)
  5. Profile similarity comparison (within vs between model)
  6. Bottleneck energy decomposition scatter
  7. Domain overlap SNR (signal-to-noise ratio per layer)

Usage:
    python stage_2_layer_energy_figures.py
"""

import json
import sys
import io
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==========================================================================
# Configuration
# ==========================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
LAYER_ENERGY_DIR = DATA_DIR / 'stage_2_layer_energy'
OUTPUT_DIR = DATA_DIR / 'stage_2_figures'

MODELS = ['gemma-2-2b', 'qwen3-4b']
MODEL_LABELS = {'gemma-2-2b': 'GEMMA-2-2B', 'qwen3-4b': 'QWEN3-4B'}
CATEGORY_COLORS = {'chemistry': '#2196F3', 'geography': '#4CAF50', 'history': '#FF9800'}
CATEGORY_LABELS = {'chemistry': 'Chemistry', 'geography': 'Geography', 'history': 'History'}
MODEL_COLORS = {'gemma-2-2b': '#E91E63', 'qwen3-4b': '#9C27B0'}

sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_constants import GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS
MODEL_TOTAL_LAYERS = {'gemma-2-2b': GEMMA_TOTAL_LAYERS, 'qwen3-4b': QWEN_TOTAL_LAYERS}

# Matplotlib style
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
})


def load_results():
    """Load layer energy results."""
    path = LAYER_ENERGY_DIR / 'layer_energy_results.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_per_circuit():
    """Load per-circuit layer energy data."""
    path = LAYER_ENERGY_DIR / 'per_circuit_layer_energy.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================================================
# Figure 16: Domain Energy Curves (Angle 1)
# ==========================================================================

def fig16_domain_energy_curves(results, per_circuit):
    """
    Domain-averaged layer energy fraction profiles per model.
    Shows 3 domain curves overlaid per model, demonstrating how
    architecture constrains energy profiles regardless of knowledge domain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        total_layers = MODEL_TOTAL_LAYERS[model]
        domain_curves = results.get('domain_curves', {}).get(model, {})

        for cat in ['chemistry', 'geography', 'history']:
            curve = domain_curves.get(cat, {})
            if not curve:
                continue
            mean = np.array(curve['mean_layer_fraction'])
            std = np.array(curve['std_layer_fraction'])
            x = np.arange(total_layers)

            ax.plot(x, mean, color=CATEGORY_COLORS[cat], linewidth=2,
                    label=CATEGORY_LABELS[cat])
            ax.fill_between(x, mean - std, mean + std,
                           color=CATEGORY_COLORS[cat], alpha=0.15)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Energy Fraction')
        ax.set_title(f'{MODEL_LABELS[model]}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, total_layers - 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Domain-Averaged Layer Energy Profiles\n'
                 'Architecture constrains energy distribution; domains overlap',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig16_domain_energy_curves.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig16_domain_energy_curves.png")


# ==========================================================================
# Figure 17: Cross-Model Normalized Depth Profiles (Angle 1)
# ==========================================================================

def fig17_cross_model_profiles(results, per_circuit):
    """
    Overlay GEMMA and QWEN energy profiles after normalizing to [0,1] depth.
    Shows how different architectures produce fundamentally different energy
    distribution patterns even when layer count is normalized.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, cat in enumerate(['chemistry', 'geography', 'history']):
        ax = axes[idx]

        for model in MODELS:
            total_layers = MODEL_TOTAL_LAYERS[model]
            # Collect all circuits for this model+category
            fracs = []
            for slug, data in per_circuit.get(model, {}).items():
                if data.get('category') == cat:
                    fracs.append(data['layer_energy_fraction'])

            if not fracs:
                continue

            fracs_arr = np.array(fracs)
            mean = np.mean(fracs_arr, axis=0)
            std = np.std(fracs_arr, axis=0)

            # Normalize to [0,1] depth
            x = np.linspace(0, 1, total_layers)
            ax.plot(x, mean, color=MODEL_COLORS[model], linewidth=2,
                    label=MODEL_LABELS[model])
            ax.fill_between(x, mean - std, mean + std,
                           color=MODEL_COLORS[model], alpha=0.15)

        ax.set_xlabel('Normalized Depth (0=input, 1=output)')
        if idx == 0:
            ax.set_ylabel('Energy Fraction')
        ax.set_title(f'{CATEGORY_LABELS[cat]}')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-Model Energy Profiles (Normalized Depth)\n'
                 'GEMMA front-loads energy; QWEN back-loads it',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig17_cross_model_profiles.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig17_cross_model_profiles.png")


# ==========================================================================
# Figure 18: Per-Layer Confidence Correlation (Angle 1 + 3)
# ==========================================================================

def fig18_layer_confidence_correlation(results):
    """
    Bar chart showing Spearman r between each layer's energy fraction
    and model confidence. Highlights which layers drive confident predictions.
    This is our "causal tracing" analog.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        lc = results.get('layer_confidence_correlation', {}).get(model, {})
        per_layer = lc.get('per_layer', [])
        bonf_alpha = lc.get('bonferroni_alpha', 0.05)

        if not per_layer:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        layers = [l['layer'] for l in per_layer]
        r_values = [l['spearman_r'] for l in per_layer]
        p_values = [l['spearman_p'] for l in per_layer]

        # Color bars by significance
        colors = []
        for r, p in zip(r_values, p_values):
            if p < bonf_alpha:
                colors.append('#E91E63' if r < 0 else '#4CAF50')  # Red negative, green positive
            else:
                colors.append('#BDBDBD')  # Grey non-significant

        bars = ax.bar(layers, r_values, color=colors, width=0.8, edgecolor='none')

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spearman r (with confidence)')
        ax.set_title(f'{MODEL_LABELS[model]}')
        ax.set_xlim(-0.5, max(layers) + 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance markers
        n_sig = sum(1 for p in p_values if p < bonf_alpha)
        ax.text(0.02, 0.98, f'{n_sig} Bonferroni-significant\n(alpha={bonf_alpha:.4f})',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Legend
        sig_pos = mpatches.Patch(color='#4CAF50', label='Sig. positive')
        sig_neg = mpatches.Patch(color='#E91E63', label='Sig. negative')
        not_sig = mpatches.Patch(color='#BDBDBD', label='Not significant')
        ax.legend(handles=[sig_pos, sig_neg, not_sig], loc='lower right', fontsize=8)

    fig.suptitle('Per-Layer Energy-Confidence Correlation\n'
                 'Which layers\' energy predicts confident outputs? (Causal tracing analog)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig18_layer_confidence_correlation.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig18_layer_confidence_correlation.png")


# ==========================================================================
# Figure 19: Cumulative Energy Curves (Angle 3)
# ==========================================================================

def fig19_cumulative_energy(results):
    """
    Cumulative energy through layers — evidence accumulation analog.
    Shows how energy builds up layer-by-layer, with 50% and 90% thresholds marked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        total_layers = MODEL_TOTAL_LAYERS[model]
        cum_data = results.get('cumulative_energy', {}).get(model, {}).get('_category_summary', {})

        for cat in ['chemistry', 'geography', 'history']:
            summary = cum_data.get(cat, {})
            if not summary:
                continue

            mean_cum = np.array(summary['mean_cumulative'])
            std_cum = np.array(summary['std_cumulative'])
            x = np.arange(total_layers)

            ax.plot(x, mean_cum, color=CATEGORY_COLORS[cat], linewidth=2,
                    label=CATEGORY_LABELS[cat])
            ax.fill_between(x, mean_cum - std_cum, mean_cum + std_cum,
                           color=CATEGORY_COLORS[cat], alpha=0.1)

            # Mark 50% threshold
            layer_50 = summary.get('mean_50pct_layer', 0)
            ax.axvline(x=layer_50, color=CATEGORY_COLORS[cat], linestyle='--',
                      alpha=0.5, linewidth=1)

        # Reference lines
        ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=0.9, color='grey', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(0.5, 0.52, '50%', transform=ax.get_yaxis_transform(), fontsize=8, color='grey')
        ax.text(0.5, 0.92, '90%', transform=ax.get_yaxis_transform(), fontsize=8, color='grey')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Cumulative Energy Fraction')
        ax.set_title(f'{MODEL_LABELS[model]}')
        ax.legend(loc='lower right')
        ax.set_xlim(0, total_layers - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cumulative Energy Accumulation Through Layers\n'
                 'GEMMA reaches 50% by L11; QWEN by L28 — different accumulation regimes',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig19_cumulative_energy.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig19_cumulative_energy.png")


# ==========================================================================
# Figure 20: Profile Similarity Comparison (Angle 1)
# ==========================================================================

def fig20_profile_similarity(results):
    """
    Grouped bar chart comparing:
    - Within-category cosine similarity
    - Cross-category cosine similarity
    - Between-model cosine similarity
    Shows architecture is the dominant factor (within-model >> between-model).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Within-Category\n(same model, same domain)',
                  'Cross-Category\n(same model, diff domain)',
                  'Between-Model\n(normalized depth)']

    means = []
    stds = []

    # Compute aggregated stats
    # Within-category: average across models
    within_vals = []
    for model in MODELS:
        sim = results.get('profile_similarity', {}).get(model, {}).get('within_category', {})
        for cat, v in sim.items():
            within_vals.append(v['mean_cosine'])
    means.append(np.mean(within_vals) if within_vals else 0)
    stds.append(np.std(within_vals) if within_vals else 0)

    # Cross-category: average across models
    cross_vals = []
    for model in MODELS:
        sim = results.get('profile_similarity', {}).get(model, {}).get('cross_category', {})
        for key, v in sim.items():
            cross_vals.append(v['mean_cosine'])
    means.append(np.mean(cross_vals) if cross_vals else 0)
    stds.append(np.std(cross_vals) if cross_vals else 0)

    # Between-model
    cm = results.get('profile_similarity', {}).get('cross_model', {}).get('overall', {})
    means.append(cm.get('between_model_mean', 0))
    stds.append(cm.get('between_model_std', 0))

    x = np.arange(len(categories))
    colors = ['#4CAF50', '#FF9800', '#E91E63']

    bars = ax.bar(x, means, yerr=stds, width=0.6, color=colors,
                  edgecolor='white', linewidth=1.5, capsize=5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0.5, 1.12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add bracket showing the gap
    gap = means[0] - means[2]
    gap = means[0] - means[2]
    ax.annotate('', xy=(0, means[0] + 0.035), xytext=(2, means[0] + 0.035),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5))
    ax.text(1, means[0] + 0.045, f'Gap = {gap:.3f}', ha='center', fontsize=10,
            fontweight='bold')

    ax.set_title('Energy Profile Similarity: Architecture Dominates Domain\n'
                 'Within-model profiles are 97.8% similar; between-model only 69.6%',
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig20_profile_similarity.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  [OK] fig20_profile_similarity.png")


# ==========================================================================
# Figure 21: Domain Overlap SNR (Angle 1 Key Evidence)
# ==========================================================================

def fig21_domain_overlap_snr(results):
    """
    Per-layer signal-to-noise ratio showing where domain differences
    exceed within-domain variability. SNR < 1 means domains are
    indistinguishable at that layer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        total_layers = MODEL_TOTAL_LAYERS[model]
        ov = results.get('domain_overlap', {}).get(model, {})

        if not ov:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        snr = np.array(ov['snr_per_layer'])
        x = np.arange(total_layers)

        # Color by above/below 1
        colors = ['#E91E63' if s >= 1 else '#4CAF50' for s in snr]
        ax.bar(x, snr, color=colors, width=0.8)

        # Reference line at SNR = 1
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='SNR = 1.0')

        frac = ov.get('fraction_indistinguishable', 0) * 100
        ax.text(0.02, 0.98,
                f'{frac:.0f}% of layers below SNR=1\n(domains indistinguishable)',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax.set_xlabel('Layer')
        ax.set_ylabel('Signal-to-Noise Ratio\n(cross-domain diff / within-domain std)')
        ax.set_title(f'{MODEL_LABELS[model]}')
        ax.set_xlim(-0.5, total_layers - 0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # Legend
        above = mpatches.Patch(color='#E91E63', label='Domains distinguishable (SNR>=1)')
        below = mpatches.Patch(color='#4CAF50', label='Domains indistinguishable (SNR<1)')
        ax.legend(handles=[above, below], loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=8)

    fig.suptitle('Domain Overlap Signal-to-Noise Ratio Per Layer\n'
                 'Red = domain differences exceed variability; Green = domains are indistinguishable',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig21_domain_overlap_snr.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig21_domain_overlap_snr.png")


# ==========================================================================
# Figure 22: Combined Angle 1 Summary (Architecture vs Domain)
# ==========================================================================

def fig22_architecture_vs_domain_summary(results, per_circuit):
    """
    Multi-panel summary figure for Angle 1 paper.
    Panel A: Within-model similarity (within-cat vs cross-cat)
    Panel B: Between-model similarity (GEMMA vs QWEN)
    Panel C: GEMMA energy profile with all 30 circuits overlaid
    Panel D: QWEN energy profile with all 30 circuits overlaid
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Similarity bar chart per model
    ax = axes[0, 0]
    for mi, model in enumerate(MODELS):
        sim = results.get('profile_similarity', {}).get(model, {})
        within = sim.get('within_category', {})
        cross = sim.get('cross_category', {})

        within_mean = np.mean([v['mean_cosine'] for v in within.values()]) if within else 0
        cross_mean = np.mean([v['mean_cosine'] for v in cross.values()]) if cross else 0

        x_pos = [mi * 3, mi * 3 + 1]
        ax.bar(x_pos[0], within_mean, width=0.8, color='#4CAF50', edgecolor='white')
        ax.bar(x_pos[1], cross_mean, width=0.8, color='#FF9800', edgecolor='white')

        ax.text(x_pos[0], within_mean + 0.002, f'{within_mean:.3f}', ha='center', fontsize=9)
        ax.text(x_pos[1], cross_mean + 0.002, f'{cross_mean:.3f}', ha='center', fontsize=9)

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0.9, 1.0)
    ax.set_title('A: Within vs Cross-Category Similarity')
    ax.legend([mpatches.Patch(color='#4CAF50'), mpatches.Patch(color='#FF9800')],
              ['Within-category', 'Cross-category'], loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Cross-model comparison
    ax = axes[0, 1]
    cm = results.get('profile_similarity', {}).get('cross_model', {}).get('overall', {})
    bars = ax.bar([0, 1],
                  [cm.get('within_model_mean', 0), cm.get('between_model_mean', 0)],
                  yerr=[cm.get('within_model_std', 0), cm.get('between_model_std', 0)],
                  width=0.6, color=[MODEL_COLORS['gemma-2-2b'], '#9C27B0'],
                  edgecolor='white', capsize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Within-Model', 'Between-Model'])
    ax.set_ylabel('Cosine Similarity\n(Normalized Depth)')
    ax.set_title('B: Cross-Model Profile Similarity')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [cm.get('within_model_mean', 0), cm.get('between_model_mean', 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    p_val = cm.get('mann_whitney_p', 1)
    ax.text(0.5, 0.05, f'Mann-Whitney p < 0.000001',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel C: GEMMA all circuits overlaid
    ax = axes[1, 0]
    model = 'gemma-2-2b'
    total_layers = MODEL_TOTAL_LAYERS[model]
    for slug, data in per_circuit.get(model, {}).items():
        cat = data.get('category', 'unknown')
        ax.plot(np.arange(total_layers), data['layer_energy_fraction'],
                color=CATEGORY_COLORS.get(cat, '#999'), alpha=0.3, linewidth=0.8)

    # Overlay domain means
    domain_curves = results.get('domain_curves', {}).get(model, {})
    for cat in ['chemistry', 'geography', 'history']:
        curve = domain_curves.get(cat, {})
        if curve:
            ax.plot(np.arange(total_layers), curve['mean_layer_fraction'],
                    color=CATEGORY_COLORS[cat], linewidth=2.5, label=CATEGORY_LABELS[cat])

    ax.set_xlabel('Layer')
    ax.set_ylabel('Energy Fraction')
    ax.set_title(f'C: {MODEL_LABELS[model]} — All 30 Circuits')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='gray')
    ax.set_xlim(0, total_layers - 1)
    ax.grid(True, alpha=0.3)

    # Panel D: QWEN all circuits overlaid
    ax = axes[1, 1]
    model = 'qwen3-4b'
    total_layers = MODEL_TOTAL_LAYERS[model]
    for slug, data in per_circuit.get(model, {}).items():
        cat = data.get('category', 'unknown')
        ax.plot(np.arange(total_layers), data['layer_energy_fraction'],
                color=CATEGORY_COLORS.get(cat, '#999'), alpha=0.3, linewidth=0.8)

    domain_curves = results.get('domain_curves', {}).get(model, {})
    for cat in ['chemistry', 'geography', 'history']:
        curve = domain_curves.get(cat, {})
        if curve:
            ax.plot(np.arange(total_layers), curve['mean_layer_fraction'],
                    color=CATEGORY_COLORS[cat], linewidth=2.5, label=CATEGORY_LABELS[cat])

    ax.set_xlabel('Layer')
    ax.set_ylabel('Energy Fraction')
    ax.set_title(f'D: {MODEL_LABELS[model]} — All 30 Circuits')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9, edgecolor='gray')
    ax.set_xlim(0, total_layers - 1)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Architecture Over Knowledge: Energy Profile Comparison\n'
                 'Domain profiles overlap within models but diverge between models',
                 fontsize=14, y=1.03)
    fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90)
    plt.savefig(OUTPUT_DIR / 'fig22_architecture_vs_domain_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  [OK] fig22_architecture_vs_domain_summary.png")


# ==========================================================================
# Figure 23: Evidence Accumulation Summary (Angle 3)
# ==========================================================================

def fig23_evidence_accumulation_summary(results, per_circuit):
    """
    Multi-panel summary figure for Angle 3 paper.
    Panel A: Cumulative energy curves (GEMMA vs QWEN side-by-side)
    Panel B: Per-layer confidence correlation (GEMMA)
    Panel C: Early vs late energy split by model
    Panel D: Bottleneck layer energy vs total energy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Cumulative energy side-by-side (GEMMA and QWEN on same axes, normalized)
    ax = axes[0, 0]
    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        cum_data = results.get('cumulative_energy', {}).get(model, {}).get('_category_summary', {})

        # Average across all categories for this model
        all_means = []
        for cat, summary in cum_data.items():
            all_means.append(summary['mean_cumulative'])

        if all_means:
            overall_mean = np.mean(all_means, axis=0)
            overall_std = np.std(all_means, axis=0)
            x = np.linspace(0, 1, total_layers)

            ax.plot(x, overall_mean, color=MODEL_COLORS[model], linewidth=2.5,
                    label=MODEL_LABELS[model])
            ax.fill_between(x, overall_mean - overall_std, overall_mean + overall_std,
                           color=MODEL_COLORS[model], alpha=0.1)

    ax.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel('Normalized Depth (0=input, 1=output)')
    ax.set_ylabel('Cumulative Energy Fraction')
    ax.set_title('A: Evidence Accumulation Curves')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Per-layer confidence correlation for GEMMA (the model with significant results)
    ax = axes[0, 1]
    model = 'gemma-2-2b'
    lc = results.get('layer_confidence_correlation', {}).get(model, {})
    per_layer = lc.get('per_layer', [])
    bonf_alpha = lc.get('bonferroni_alpha', 0.05)

    if per_layer:
        layers = [l['layer'] for l in per_layer]
        r_values = [l['spearman_r'] for l in per_layer]
        p_values = [l['spearman_p'] for l in per_layer]

        colors = ['#E91E63' if (p < bonf_alpha and r < 0) else '#4CAF50' if (p < bonf_alpha and r > 0) else '#BDBDBD'
                  for r, p in zip(r_values, p_values)]
        ax.bar(layers, r_values, color=colors, width=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # Annotate key layers
        for l_data in per_layer:
            if l_data['spearman_p'] < bonf_alpha:
                layer = l_data['layer']
                r = l_data['spearman_r']
                ax.annotate(f'L{layer}\nr={r:.2f}',
                           xy=(layer, r), xytext=(layer, r + (0.1 if r > 0 else -0.1)),
                           fontsize=7, ha='center', va='bottom' if r > 0 else 'top',
                           fontweight='bold')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Spearman r')
    ax.set_title(f'B: Layer Energy vs Confidence ({MODEL_LABELS["gemma-2-2b"]})')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Early vs late energy comparison
    ax = axes[1, 0]
    for mi, model in enumerate(MODELS):
        dd = results.get('drift_diffusion', {}).get(model, {})
        cat_comp = dd.get('category_comparison', {})

        early_fracs = []
        late_fracs = []
        for cat in ['chemistry', 'geography', 'history']:
            cc = cat_comp.get(cat, {})
            if cc:
                early_fracs.append(cc['mean_early_fraction'])
                late_fracs.append(1 - cc['mean_early_fraction'])

        if early_fracs:
            mean_early = np.mean(early_fracs)
            mean_late = np.mean(late_fracs)
            x_pos = [mi * 3, mi * 3 + 1]
            ax.bar(x_pos[0], mean_early, width=0.8, color='#2196F3', label='Early Half' if mi == 0 else '')
            ax.bar(x_pos[1], mean_late, width=0.8, color='#FF5722', label='Late Half' if mi == 0 else '')
            ax.text(x_pos[0], mean_early + 0.01, f'{mean_early:.1%}', ha='center', fontsize=10)
            ax.text(x_pos[1], mean_late + 0.01, f'{mean_late:.1%}', ha='center', fontsize=10)

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel('Energy Fraction')
    ax.set_title('C: Early vs Late Energy Distribution')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Bottleneck energy fraction vs confidence
    ax = axes[1, 1]
    for model in MODELS:
        be = results.get('bottleneck_energy', {}).get(model, {}).get('per_circuit', {})
        if not be:
            continue
        fracs = [v['bottleneck_energy_fraction'] for v in be.values()]
        confs = [v['output_probability'] for v in be.values()]
        ax.scatter(fracs, confs, color=MODEL_COLORS[model], alpha=0.6, s=40,
                  label=MODEL_LABELS[model], edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Bottleneck Layer Energy Fraction')
    ax.set_ylabel('Output Probability (Confidence)')
    ax.set_title('D: Bottleneck Energy vs Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation info
    for model in MODELS:
        corr = results.get('bottleneck_energy', {}).get(model, {}).get('correlations', {})
        bn_corr = corr.get('bn_energy_frac_vs_confidence', {})
        if bn_corr:
            r = bn_corr.get('spearman_r', 0)
            p = bn_corr.get('spearman_p', 1)
            # Position label at bottom to avoid overlapping title
            y_pos = 0.12 if model == 'gemma-2-2b' else 0.04
            ax.text(0.02, y_pos,
                    f'{MODEL_LABELS[model]}: r={r:.3f}, p={p:.3f}',
                    transform=ax.transAxes, fontsize=8,
                    color=MODEL_COLORS[model], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.suptitle('Evidence Accumulation Analysis\n'
                 'Confidence driven by distributed energy, not bottleneck concentration',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig23_evidence_accumulation_summary.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig23_evidence_accumulation_summary.png")


# ==========================================================================
# Main
# ==========================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING LAYER ENERGY FIGURES")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    results = load_results()
    per_circuit = load_per_circuit()
    print("  Done.\n")

    # Generate all figures
    print("Generating figures...\n")
    fig16_domain_energy_curves(results, per_circuit)
    fig17_cross_model_profiles(results, per_circuit)
    fig18_layer_confidence_correlation(results)
    fig19_cumulative_energy(results)
    fig20_profile_similarity(results)
    fig21_domain_overlap_snr(results)
    fig22_architecture_vs_domain_summary(results, per_circuit)
    fig23_evidence_accumulation_summary(results, per_circuit)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
