#!/usr/bin/env python3
"""
Stage 1.4: Distribution Visualizations
Generates three key figures:
  1. Category distribution by layer (stacked bar)
  2. Bottleneck semantic profile (before/after comparison)
  3. Layer-wise semantic flow (Sankey diagram)
Plus: cross-prompt bottleneck convergence heatmap
"""

import json
import os
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))

from annotate_features_v2 import classify_from_explanation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
LIBRARY_PATH = BASE / "data" / "stage_1_5_bottleneck_library.json"
OUT_DIR = BASE / "data" / "stage_1_4_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette
COLORS = {
    'SEMANTICS:CODE':       '#1565C0',  # Blue
    'SEMANTICS:CONCEPT':    '#2E7D32',  # Green
    'SEMANTICS:ENTITY':     '#6A1B9A',  # Purple
    'SEMANTICS:GEOGRAPHIC': '#E65100',  # Orange
    'SEMANTICS:TEMPORAL':   '#00838F',  # Teal
    'POLYSEMANTIC':         '#C62828',  # Red
    'SYNTAX':               '#F9A825',  # Yellow
    'UNKNOWN':              '#9E9E9E',  # Gray
}

CATEGORY_ORDER = [
    'SEMANTICS:CODE', 'SEMANTICS:CONCEPT', 'SEMANTICS:ENTITY',
    'SEMANTICS:GEOGRAPHIC', 'SEMANTICS:TEMPORAL', 'POLYSEMANTIC',
    'SYNTAX', 'UNKNOWN'
]

SHORT_NAMES = {
    'SEMANTICS:CODE': 'Code',
    'SEMANTICS:CONCEPT': 'Concept',
    'SEMANTICS:ENTITY': 'Entity',
    'SEMANTICS:GEOGRAPHIC': 'Geographic',
    'SEMANTICS:TEMPORAL': 'Temporal',
    'POLYSEMANTIC': 'Polysemantic',
    'SYNTAX': 'Syntax',
    'UNKNOWN': 'Unknown',
}


def load_annotations():
    """Load features from the bottleneck library JSON and classify via explanation."""
    with open(LIBRARY_PATH, encoding='utf-8') as f:
        library = json.load(f)

    rows = []
    seen_labels = set()

    # 1. Cross-circuit features (appear in 2+ circuits)
    for label, entry in library.get('cross_circuit_features', {}).items():
        seen_labels.add(label)
        explanation = entry.get('explanation', '')
        layer = entry.get('layer', 0)
        category = classify_from_explanation(explanation, layer=layer)
        confidence = 'HIGH' if explanation else 'LOW'
        rows.append({
            'feature_label': label,
            'layer': str(layer),
            'manual_category': category,
            'confidence': confidence,
            'explanation': explanation,
        })

    # 2. Per-circuit bottlenecks (add any not already seen from cross-circuit)
    for circuit_name, bottlenecks in library.get('per_circuit_bottlenecks', {}).items():
        for bn in bottlenecks:
            label = bn.get('label', '')
            if label in seen_labels:
                continue
            seen_labels.add(label)
            layer = bn.get('layer', 0)
            # Per-circuit entries lack explanation; look up in cross_circuit if present
            explanation = ''
            if label in library.get('cross_circuit_features', {}):
                explanation = library['cross_circuit_features'][label].get('explanation', '')
            category = classify_from_explanation(explanation, layer=layer)
            confidence = 'MEDIUM' if explanation else 'LOW'
            rows.append({
                'feature_label': label,
                'layer': str(layer),
                'manual_category': category,
                'confidence': confidence,
                'explanation': explanation,
            })

    return rows


def load_bottleneck_data():
    """Load cross-prompt bottleneck convergence data."""
    traceback_files = {
        'GEMMA\nSouthern': BASE / "data/prompts/gemma-2-2b_the-southern-most-us-state-is/3_analysis/traceback_paths.json",
        'GEMMA\nPresident': BASE / "data/prompts/gemma-2-2b_the-president-of-the-united-states-lives-in/3_analysis/traceback_paths.json",
        'GEMMA\nWater': BASE / "data/prompts/gemma-2-2b_water-boils-at-100-degrees/3_analysis/traceback_paths.json",
        'QWEN\nPresident': BASE / "data/prompts/qwen3-4b_im-endthe-president-of-the-united-states-lives-in/3_analysis/traceback_paths.json",
        'QWEN\nWater': BASE / "data/prompts/qwen3-4b_water-boils-at-100-degrees/3_analysis/traceback_paths.json",
    }

    all_data = {}
    for name, path in traceback_files.items():
        if path.exists():
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            paths = data.get('critical_paths', [])
            n = len(paths)
            node_counts = Counter()
            for p in paths:
                for step in p.get('path_nodes', []):
                    label = step.get('label', step.get('node_id', ''))
                    node_counts[label] += 1
            # 100% convergence nodes
            conv = {k for k, v in node_counts.items() if v == n}
            all_data[name] = conv
    return all_data


def load_bottleneck_semantics():
    """Load the Neuronpedia query results for bottleneck features."""
    path = BASE / "data" / "bottleneck_deep_dive.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return []


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Stacked Bar Chart - Category Distribution by Layer
# ══════════════════════════════════════════════════════════════════════════
def plot_category_by_layer(rows):
    """Stacked bar chart showing semantic category distribution per layer."""
    layer_cats = defaultdict(Counter)
    for r in rows:
        layer = int(r['layer'])
        cat = r['manual_category']
        layer_cats[layer][cat] += 1

    layers = sorted(layer_cats.keys())
    layer_labels = [f"L{l}" for l in layers]

    fig, ax = plt.subplots(figsize=(14, 6))

    bottoms = np.zeros(len(layers))
    bars_by_cat = {}

    for cat in CATEGORY_ORDER:
        vals = [layer_cats[l].get(cat, 0) for l in layers]
        if sum(vals) == 0:
            continue
        color = COLORS.get(cat, '#999999')
        b = ax.bar(layer_labels, vals, bottom=bottoms, color=color,
                   label=SHORT_NAMES.get(cat, cat), edgecolor='white', linewidth=0.5)
        bars_by_cat[cat] = b
        bottoms += np.array(vals)

    # Add count labels on top of each bar
    for i, l in enumerate(layers):
        total = sum(layer_cats[l].values())
        ax.text(i, bottoms[i] + 0.15, str(total), ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Count', fontsize=12, fontweight='bold')
    n_total = sum(sum(layer_cats[l].values()) for l in layers)
    ax.set_title(f'Semantic Category Distribution by Layer\n'
                 f'Bottleneck library (N={n_total} features across {len(layers)} layers)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.set_ylim(0, max(bottoms) + 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "category_distribution_by_layer.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [1/4] Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Bottleneck Semantic Profile - Before/After Comparison
# ══════════════════════════════════════════════════════════════════════════
def plot_bottleneck_profile(rows):
    """
    Compare semantic composition before bottleneck (L0-L4) vs after (L6+).
    GEMMA bottleneck is at L5, so we split there.
    """
    before = Counter()  # L0-L4
    after = Counter()   # L6+
    at_bn = Counter()   # L5 (if we had it -- we don't in our sample, but L6 is close)

    for r in rows:
        layer = int(r['layer'])
        cat = r['manual_category']
        if layer <= 3:
            before[cat] += 1
        elif layer >= 9:
            after[cat] += 1
        else:  # L6 (our closest sample to bottleneck)
            at_bn[cat] += 1

    groups = {'Before BN\n(L0-L3)': before, 'At/Near BN\n(L6)': at_bn, 'After BN\n(L9+)': after}
    group_names = list(groups.keys())
    x = np.arange(len(group_names))
    width = 0.10
    cats_present = [c for c in CATEGORY_ORDER if any(groups[g].get(c, 0) > 0 for g in group_names)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cat in enumerate(cats_present):
        offset = (i - len(cats_present) / 2) * width + width / 2
        vals = [groups[g].get(cat, 0) for g in group_names]
        ax.bar(x + offset, vals, width, color=COLORS.get(cat, '#999'),
               label=SHORT_NAMES.get(cat, cat), edgecolor='white', linewidth=0.5)

    # Add bottleneck arrow
    ax.annotate('L5 BOTTLENECK', xy=(1, 0), xytext=(1, -1.8),
                fontsize=10, fontweight='bold', color='#C62828', ha='center',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Count', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Composition Before vs After GEMMA L5 Bottleneck\n'
                 'What types of features survive the information filter?',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "bottleneck_before_after.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [2/4] Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Semantic Flow Diagram (Alluvial/Sankey-style with matplotlib)
# ══════════════════════════════════════════════════════════════════════════
def plot_semantic_flow(rows):
    """
    Layer-by-layer semantic flow showing how categories shift.
    Uses a simplified alluvial plot (matplotlib-based).
    """
    layer_cats = defaultdict(Counter)
    for r in rows:
        layer = int(r['layer'])
        cat = r['manual_category']
        layer_cats[layer][cat] += 1

    layers = sorted(layer_cats.keys())
    cats_present = [c for c in CATEGORY_ORDER
                    if any(layer_cats[l].get(c, 0) > 0 for l in layers)]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot stacked area chart
    x_pos = list(range(len(layers)))
    bottom = np.zeros(len(layers))

    for cat in cats_present:
        vals = np.array([layer_cats[l].get(cat, 0) for l in layers], dtype=float)
        # Normalize to percentage
        totals = np.array([sum(layer_cats[l].values()) for l in layers], dtype=float)
        pcts = vals / totals * 100

        ax.fill_between(x_pos, bottom, bottom + pcts,
                         color=COLORS.get(cat, '#999'), alpha=0.8,
                         label=SHORT_NAMES.get(cat, cat))
        # Label in the middle of each band if large enough
        mid = bottom + pcts / 2
        for i, (p, m) in enumerate(zip(pcts, mid)):
            if p >= 15:  # Only label if >15%
                ax.text(i, m, f'{p:.0f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
        bottom += pcts

    # Mark bottleneck
    # L5 is between index 1 (L3) and 2 (L6) in our sample
    bn_x = 1.5  # Between L3 and L6
    ax.axvline(x=bn_x, color='#C62828', linewidth=2, linestyle='--', alpha=0.8)
    ax.text(bn_x, 102, 'L5 BOTTLENECK', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#C62828'))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Features (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Category Flow Across Layers\n'
                 'How does the semantic composition change through the network?',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2,
              bbox_to_anchor=(1.01, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "semantic_flow_by_layer.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [3/4] Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cross-Prompt Bottleneck Convergence Heatmap
# ══════════════════════════════════════════════════════════════════════════
def plot_convergence_heatmap(bottleneck_data, semantics_data):
    """
    Heatmap showing which bottleneck features appear across which prompts.
    Annotated with Neuronpedia semantic labels.
    """
    if not bottleneck_data:
        print("  [4/4] Skipped: no bottleneck data")
        return None

    prompts = list(bottleneck_data.keys())

    # Find features that appear in 2+ prompts
    all_features = Counter()
    for name, features in bottleneck_data.items():
        for f in features:
            all_features[f] += 1

    # Filter to cross-prompt features, limit to most interesting
    cross_features = {f for f, c in all_features.items() if c >= 2}

    # Also filter to features from different prompts (not just top/bottom of same)
    # Keep features that appear in at least 2 different base prompts
    prompt_base = {}
    for p in prompts:
        base = p.replace('\n', ' ').split()[-1]  # "Southern", "President", "Water"
        prompt_base[p] = base

    feature_bases = defaultdict(set)
    for name, features in bottleneck_data.items():
        base = prompt_base[name]
        for f in features:
            if f in cross_features:
                feature_bases[f].add(base)

    multi_prompt_features = sorted(
        [f for f, bases in feature_bases.items() if len(bases) >= 2],
        key=lambda f: (int(f.split('_')[0][1:]) if f.startswith('L') else 999, f)
    )

    if not multi_prompt_features:
        print("  [4/4] Skipped: no cross-prompt features found")
        return None

    # Limit to top 20 most shared
    multi_prompt_features = multi_prompt_features[:20]

    # Build the semantic label lookup from our Neuronpedia data
    np_labels = {}
    for entry in semantics_data:
        layer = entry.get('layer', '')
        cid = entry.get('circuit_id', '')
        examples = entry.get('examples', [])
        explanation = entry.get('explanation', '')
        key = f"L{layer}_F{cid}"
        if explanation:
            np_labels[key] = explanation[:40]
        elif examples:
            clean = [e.replace('▁', '').strip() for e in examples[:2] if e.replace('▁', '').strip()]
            np_labels[key] = ', '.join(clean)[:40]

    # Build heatmap matrix
    matrix = np.zeros((len(multi_prompt_features), len(prompts)))
    for j, name in enumerate(prompts):
        features = bottleneck_data[name]
        for i, f in enumerate(multi_prompt_features):
            if f in features:
                matrix[i, j] = 1

    # Feature labels with NP semantics
    y_labels = []
    for f in multi_prompt_features:
        label = f
        if f in np_labels:
            label = f"{f}\n({np_labels[f]})"
        y_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, max(6, len(multi_prompt_features) * 0.45)))

    cmap = plt.cm.colors.ListedColormap(['#FAFAFA', '#1565C0'])
    ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

    # Grid lines
    for i in range(len(multi_prompt_features) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(prompts) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)

    # Labels
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(multi_prompt_features)))
    ax.set_yticklabels(y_labels, fontsize=7)

    # Checkmarks in filled cells
    for i in range(len(multi_prompt_features)):
        for j in range(len(prompts)):
            if matrix[i, j] == 1:
                ax.text(j, i, '✓', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')

    ax.set_title('Cross-Prompt Bottleneck Convergence\n'
                 'Features appearing as 100% convergence points in multiple prompts',
                 fontsize=13, fontweight='bold', pad=15)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    out = OUT_DIR / "cross_prompt_convergence_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [4/4] Saved: {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("Stage 1.4: Distribution Visualizations")
    print("=" * 50)

    rows = load_annotations()
    print(f"Loaded {len(rows)} annotated features")

    bottleneck_data = load_bottleneck_data()
    print(f"Loaded bottleneck data for {len(bottleneck_data)} prompt analyses")

    semantics_data = load_bottleneck_semantics()
    print(f"Loaded {len(semantics_data)} Neuronpedia semantic profiles")
    print()

    # Generate all figures
    plot_category_by_layer(rows)
    plot_bottleneck_profile(rows)
    plot_semantic_flow(rows)
    plot_convergence_heatmap(bottleneck_data, semantics_data)

    print(f"\nAll figures saved to: {OUT_DIR}")

    # Print summary statistics for documentation
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    cats = Counter(r['manual_category'] for r in rows)
    print(f"\nOverall distribution (N={len(rows)}):")
    for cat in CATEGORY_ORDER:
        count = cats.get(cat, 0)
        if count > 0:
            print(f"  {SHORT_NAMES.get(cat, cat):15s}: {count:3d} ({count/len(rows)*100:5.1f}%)")

    confs = Counter(r['confidence'] for r in rows)
    print(f"\nConfidence distribution:")
    for c in ['HIGH', 'MEDIUM', 'LOW']:
        print(f"  {c:8s}: {confs.get(c, 0):3d}")

    # Layer-specific patterns
    print(f"\nLayer-specific highlights:")
    layer_cats = defaultdict(Counter)
    for r in rows:
        layer_cats[int(r['layer'])][r['manual_category']] += 1

    for l in sorted(layer_cats.keys()):
        total = sum(layer_cats[l].values())
        dominant = layer_cats[l].most_common(1)[0]
        print(f"  L{l:2d}: {total} features, dominant = {SHORT_NAMES.get(dominant[0], dominant[0])} "
              f"({dominant[1]}/{total} = {dominant[1]/total*100:.0f}%)")


if __name__ == '__main__':
    main()
