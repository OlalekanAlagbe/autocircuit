#!/usr/bin/env python3
"""
Stage 2: Cross-Category Circuit Analysis

Analyzes 60 circuits (30 prompts × 2 models) across 3 knowledge domains:
  - Chemistry: Chemical symbol lookups (element → symbol)
  - Geography: World capital lookups (country → capital)
  - History: Historical event date lookups (event → year)

Key analyses:
  1. Per-circuit bottleneck extraction from traceback_paths.json
  2. Within-category comparison (do same-domain circuits share bottlenecks?)
  3. Cross-category comparison (universal vs domain-specific features)
  4. Model comparison (GEMMA vs QWEN architectural differences)
  5. Statistical significance tests

Usage:
    python stage_2_cross_category_analysis.py
    python stage_2_cross_category_analysis.py --output-dir data/stage_2_analysis
"""

import json
import sys
import io
import os
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================================================
# Configuration
# ==========================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_analysis'

MODELS = ['gemma-2-2b', 'qwen3-4b']

# Prompt categories with slugified directory names
CATEGORIES = {
    'chemistry': [
        'the-chemical-symbol-for-gold-is',
        'the-chemical-symbol-for-iron-is',
        'the-chemical-symbol-for-lead-is',
        'the-chemical-symbol-for-mercury-is',
        'the-chemical-symbol-for-argon-is',
        'the-chemical-symbol-for-sodium-is',
        'the-chemical-symbol-for-silver-is',
        'the-chemical-symbol-for-potassium-is',
        'the-chemical-symbol-for-copper-is',
        'the-chemical-symbol-for-tungsten-is',
    ],
    'geography': [
        'the-capital-of-france-is',
        'the-capital-of-japan-is',
        'the-capital-of-germany-is',
        'the-capital-of-egypt-is',
        'the-capital-of-italy-is',
        'the-capital-of-spain-is',
        'the-capital-of-canada-is',
        'the-capital-of-thailand-is',
        'the-capital-of-turkey-is',
        'the-capital-of-south-korea-is',
    ],
    'history': [
        'world-war-ii-ended-in',
        'the-first-moon-landing-was-in',
        'the-berlin-wall-fell-in',
        'the-titanic-sank-in',
        'america-declared-independence-in',
        'the-french-revolution-began-in',
        'world-war-i-started-in',
        'columbus-reached-the-americas-in',
        'the-great-fire-of-london-occurred-in',
        'nelson-mandela-was-released-from-prison-in',
    ],
}

# Bottleneck threshold: feature must appear in this % of paths to count
BOTTLENECK_THRESHOLD = 0.6  # 60%+ of paths


# ==========================================================================
# Data Loading
# ==========================================================================

def load_traceback(model: str, prompt_slug: str) -> Optional[dict]:
    """Load traceback_paths.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    path = PROMPTS_DIR / dir_name / '3_analysis' / 'traceback_paths.json'
    if not path.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_bottlenecks(traceback_data: dict, threshold: float = BOTTLENECK_THRESHOLD) -> List[dict]:
    """
    Extract bottleneck features from traceback paths.

    A bottleneck is a feature that appears in >= threshold fraction of traced paths.
    """
    paths = traceback_data.get('critical_paths', [])
    if not paths:
        return []

    num_paths = len(paths)
    feature_appearances = Counter()
    feature_layers = {}
    feature_max_score = {}

    for path in paths:
        seen_in_path = set()
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label and label not in seen_in_path:
                feature_appearances[label] += 1
                seen_in_path.add(label)
                feature_layers[label] = node.get('layer', -1)
                current_max = feature_max_score.get(label, 0)
                feature_max_score[label] = max(current_max, node.get('score', 0))

    bottlenecks = []
    for feature, count in feature_appearances.most_common():
        convergence = count / num_paths
        if convergence >= threshold:
            bottlenecks.append({
                'label': feature,
                'layer': feature_layers[feature],
                'convergence': convergence,
                'path_count': count,
                'total_paths': num_paths,
                'max_score': feature_max_score[feature],
            })

    return bottlenecks


def extract_all_path_features(traceback_data: dict) -> Set[str]:
    """Extract ALL features from all paths (for Jaccard similarity)."""
    features = set()
    for path in traceback_data.get('critical_paths', []):
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label:
                features.add(label)
    return features


# ==========================================================================
# Analysis Functions
# ==========================================================================

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_within_category_metrics(
    category_data: Dict[str, dict],
    category_name: str,
    model: str,
) -> dict:
    """
    Compute within-category metrics for one category + one model.

    Args:
        category_data: {prompt_slug: traceback_data}

    Returns dict with:
        - shared_bottlenecks: features appearing as bottlenecks in 2+ circuits
        - jaccard_matrix: pairwise Jaccard similarity
        - layer_distribution: bottleneck counts per layer
        - consistency_score: % of circuits sharing the top bottleneck
    """
    # Extract bottlenecks for each circuit
    circuit_bottlenecks = {}  # prompt -> [bottleneck features]
    circuit_all_features = {}  # prompt -> set of all features

    for slug, data in category_data.items():
        bottlenecks = extract_bottlenecks(data)
        circuit_bottlenecks[slug] = bottlenecks
        circuit_all_features[slug] = extract_all_path_features(data)

    # Shared bottlenecks: features that are bottlenecks in multiple circuits
    bottleneck_circuit_count = Counter()
    bottleneck_info = {}
    for slug, bns in circuit_bottlenecks.items():
        for bn in bns:
            label = bn['label']
            bottleneck_circuit_count[label] += 1
            if label not in bottleneck_info:
                bottleneck_info[label] = {
                    'label': label,
                    'layer': bn['layer'],
                    'circuits': [],
                    'convergences': [],
                }
            bottleneck_info[label]['circuits'].append(slug)
            bottleneck_info[label]['convergences'].append(bn['convergence'])

    shared_bottlenecks = []
    for label, count in bottleneck_circuit_count.most_common():
        if count >= 2:
            info = bottleneck_info[label]
            shared_bottlenecks.append({
                'label': label,
                'layer': info['layer'],
                'circuit_count': count,
                'total_circuits': len(category_data),
                'share_pct': count / len(category_data) * 100,
                'avg_convergence': sum(info['convergences']) / len(info['convergences']),
                'circuits': info['circuits'],
            })

    # Jaccard similarity matrix
    slugs = sorted(category_data.keys())
    jaccard_matrix = {}
    for i, slug_a in enumerate(slugs):
        for j, slug_b in enumerate(slugs):
            if i < j:
                sim = jaccard_similarity(
                    circuit_all_features.get(slug_a, set()),
                    circuit_all_features.get(slug_b, set()),
                )
                jaccard_matrix[f"{slug_a} vs {slug_b}"] = sim

    avg_jaccard = (
        sum(jaccard_matrix.values()) / len(jaccard_matrix)
        if jaccard_matrix else 0.0
    )

    # Layer distribution
    layer_dist = Counter()
    for slug, bns in circuit_bottlenecks.items():
        for bn in bns:
            layer_dist[bn['layer']] += 1

    # Consistency score: what % of circuits share the most common bottleneck?
    top_bottleneck_share = 0
    if bottleneck_circuit_count:
        top_label, top_count = bottleneck_circuit_count.most_common(1)[0]
        top_bottleneck_share = top_count / len(category_data) * 100

    return {
        'category': category_name,
        'model': model,
        'num_circuits': len(category_data),
        'shared_bottlenecks': shared_bottlenecks,
        'avg_jaccard_similarity': avg_jaccard,
        'jaccard_matrix': jaccard_matrix,
        'layer_distribution': dict(sorted(layer_dist.items())),
        'consistency_score': top_bottleneck_share,
        'top_bottleneck': bottleneck_circuit_count.most_common(1)[0][0] if bottleneck_circuit_count else None,
        'bottlenecks_per_circuit': {
            slug: len(bns) for slug, bns in circuit_bottlenecks.items()
        },
    }


def compute_cross_category_metrics(
    all_within_results: List[dict],
    all_bottleneck_sets: Dict[str, Set[str]],
    model: str,
) -> dict:
    """
    Compute cross-category metrics for one model.

    Args:
        all_within_results: list of within-category result dicts
        all_bottleneck_sets: {category: set of bottleneck feature labels}
    """
    categories = list(all_bottleneck_sets.keys())

    # Universal bottlenecks: appear in ALL categories
    if len(categories) >= 2:
        universal = set.intersection(*all_bottleneck_sets.values())
    else:
        universal = set()

    # Category-specific: only in one category
    category_specific = {}
    for cat in categories:
        others = set()
        for other_cat in categories:
            if other_cat != cat:
                others |= all_bottleneck_sets[other_cat]
        specific = all_bottleneck_sets[cat] - others
        category_specific[cat] = specific

    # Pairwise category overlap
    category_overlap = {}
    for cat_a, cat_b in combinations(categories, 2):
        overlap = all_bottleneck_sets[cat_a] & all_bottleneck_sets[cat_b]
        category_overlap[f"{cat_a} vs {cat_b}"] = {
            'shared_features': list(overlap),
            'count': len(overlap),
            'jaccard': jaccard_similarity(
                all_bottleneck_sets[cat_a],
                all_bottleneck_sets[cat_b],
            ),
        }

    # Layer distribution comparison
    layer_by_category = {}
    for result in all_within_results:
        layer_by_category[result['category']] = result['layer_distribution']

    # Average bottleneck layer per category
    avg_layer_by_category = {}
    for result in all_within_results:
        layers = result['layer_distribution']
        if layers:
            total_count = sum(layers.values())
            weighted_sum = sum(int(layer) * count for layer, count in layers.items())
            avg_layer_by_category[result['category']] = weighted_sum / total_count
        else:
            avg_layer_by_category[result['category']] = None

    return {
        'model': model,
        'universal_bottlenecks': list(universal),
        'universal_count': len(universal),
        'category_specific': {
            cat: list(features) for cat, features in category_specific.items()
        },
        'category_specific_counts': {
            cat: len(features) for cat, features in category_specific.items()
        },
        'category_overlap': category_overlap,
        'avg_bottleneck_layer': avg_layer_by_category,
        'layer_distributions': layer_by_category,
        'avg_jaccard_within': {
            r['category']: r['avg_jaccard_similarity'] for r in all_within_results
        },
    }


# ==========================================================================
# Report Generation
# ==========================================================================

def generate_report(
    within_results: List[dict],
    cross_results: List[dict],
    model_comparison: dict,
    output_dir: Path,
) -> str:
    """Generate the markdown analysis report."""
    lines = []
    lines.append("# Stage 2: Cross-Category Circuit Analysis Report")
    lines.append(f"\n**Date**: {__import__('datetime').date.today()}")
    lines.append(f"**Circuits Analyzed**: 60 (30 prompts × 2 models)")
    lines.append(f"**Categories**: Chemistry (10), Geography (10), History (10)")
    lines.append(f"**Models**: GEMMA-2-2B, QWEN3-4B")
    lines.append("")
    lines.append("---")

    # Executive Summary
    lines.append("\n## 1. Executive Summary\n")
    for cr in cross_results:
        model = cr['model']
        lines.append(f"### {model.upper()}")
        lines.append(f"- **Universal bottlenecks** (all 3 categories): {cr['universal_count']}")
        for cat, count in cr['category_specific_counts'].items():
            lines.append(f"- **{cat.title()}-specific** features: {count}")
        for pair, data in cr['category_overlap'].items():
            lines.append(f"- **{pair}** overlap: {data['count']} shared features (Jaccard={data['jaccard']:.3f})")
        lines.append(f"- **Avg bottleneck layer**: {cr['avg_bottleneck_layer']}")
        lines.append("")

    # Within-Category Results
    lines.append("\n## 2. Within-Category Analysis\n")
    for wr in within_results:
        lines.append(f"### {wr['category'].title()} — {wr['model'].upper()}")
        lines.append(f"- Circuits: {wr['num_circuits']}")
        lines.append(f"- Avg Jaccard similarity: {wr['avg_jaccard_similarity']:.4f}")
        lines.append(f"- Top bottleneck consistency: {wr['consistency_score']:.1f}% ({wr['top_bottleneck']})")
        lines.append(f"- Shared bottlenecks (2+ circuits): {len(wr['shared_bottlenecks'])}")
        lines.append("")

        if wr['shared_bottlenecks']:
            lines.append("| Feature | Layer | Circuits | Share % | Avg Convergence |")
            lines.append("|---------|-------|----------|---------|-----------------|")
            for sb in wr['shared_bottlenecks'][:15]:
                lines.append(
                    f"| {sb['label']} | L{sb['layer']} | "
                    f"{sb['circuit_count']}/{sb['total_circuits']} | "
                    f"{sb['share_pct']:.0f}% | "
                    f"{sb['avg_convergence']:.0%} |"
                )
            lines.append("")

        # Layer distribution
        if wr['layer_distribution']:
            lines.append(f"**Layer distribution**: {wr['layer_distribution']}")
            lines.append("")

    # Cross-Category Results
    lines.append("\n## 3. Cross-Category Analysis\n")
    for cr in cross_results:
        model = cr['model']
        lines.append(f"### {model.upper()}\n")

        lines.append(f"#### Universal Bottlenecks ({cr['universal_count']})")
        if cr['universal_bottlenecks']:
            for feat in cr['universal_bottlenecks']:
                lines.append(f"- {feat}")
        else:
            lines.append("- None found")
        lines.append("")

        lines.append("#### Category-Specific Bottlenecks")
        for cat, features in cr['category_specific'].items():
            lines.append(f"- **{cat.title()}** ({len(features)}): {', '.join(features[:10])}")
            if len(features) > 10:
                lines.append(f"  ... and {len(features) - 10} more")
        lines.append("")

        lines.append("#### Average Bottleneck Layer by Category")
        for cat, avg_layer in cr['avg_bottleneck_layer'].items():
            if avg_layer is not None:
                lines.append(f"- **{cat.title()}**: L{avg_layer:.1f}")
        lines.append("")

    # Model Comparison
    lines.append("\n## 4. Model Comparison (GEMMA vs QWEN)\n")
    if model_comparison:
        lines.append(f"- Shared universal features: {model_comparison.get('shared_universal_count', 0)}")
        lines.append(f"- GEMMA-only universal: {model_comparison.get('gemma_only_universal_count', 0)}")
        lines.append(f"- QWEN-only universal: {model_comparison.get('qwen_only_universal_count', 0)}")
        lines.append("")

        lines.append("### Within-Category Jaccard Comparison")
        lines.append("| Category | GEMMA Avg Jaccard | QWEN Avg Jaccard |")
        lines.append("|----------|-------------------|------------------|")
        for cat in CATEGORIES:
            gemma_j = model_comparison.get('jaccard_comparison', {}).get(cat, {}).get('gemma-2-2b', 0)
            qwen_j = model_comparison.get('jaccard_comparison', {}).get(cat, {}).get('qwen3-4b', 0)
            lines.append(f"| {cat.title()} | {gemma_j:.4f} | {qwen_j:.4f} |")
        lines.append("")

        lines.append("### Avg Bottleneck Layer Comparison")
        lines.append("| Category | GEMMA Avg Layer | QWEN Avg Layer |")
        lines.append("|----------|-----------------|----------------|")
        for cat in CATEGORIES:
            gemma_l = model_comparison.get('layer_comparison', {}).get(cat, {}).get('gemma-2-2b', 'N/A')
            qwen_l = model_comparison.get('layer_comparison', {}).get(cat, {}).get('qwen3-4b', 'N/A')
            gemma_str = f"L{gemma_l:.1f}" if isinstance(gemma_l, (int, float)) else gemma_l
            qwen_str = f"L{qwen_l:.1f}" if isinstance(qwen_l, (int, float)) else qwen_l
            lines.append(f"| {cat.title()} | {gemma_str} | {qwen_str} |")
        lines.append("")

    return '\n'.join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Category Circuit Analysis')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: CROSS-CATEGORY CIRCUIT ANALYSIS")
    print("=" * 70)
    print(f"  Categories: {', '.join(CATEGORIES.keys())}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ---- Load all traceback data ----
    print("\n[1/4] Loading traceback data...")
    all_data = {}  # (model, category, prompt_slug) -> traceback_data
    loaded = 0
    missing = 0

    for model in MODELS:
        for category, prompts in CATEGORIES.items():
            for slug in prompts:
                data = load_traceback(model, slug)
                if data:
                    all_data[(model, category, slug)] = data
                    loaded += 1
                else:
                    missing += 1

    print(f"  Loaded: {loaded}, Missing: {missing}")
    if missing > 0:
        print(f"  [WARN] {missing} circuits missing traceback data!")

    # ---- Within-Category Analysis ----
    print("\n[2/4] Computing within-category metrics...")
    within_results = []
    all_bottleneck_sets = {}  # (model, category) -> set of bottleneck labels

    for model in MODELS:
        for category, prompts in CATEGORIES.items():
            category_data = {}
            for slug in prompts:
                key = (model, category, slug)
                if key in all_data:
                    category_data[slug] = all_data[key]

            if category_data:
                result = compute_within_category_metrics(category_data, category, model)
                within_results.append(result)

                # Collect all bottleneck labels for cross-category
                bn_labels = set()
                for sb in result['shared_bottlenecks']:
                    bn_labels.add(sb['label'])
                # Also include per-circuit bottlenecks
                for slug, data in category_data.items():
                    for bn in extract_bottlenecks(data):
                        bn_labels.add(bn['label'])
                all_bottleneck_sets[(model, category)] = bn_labels

                print(f"  {model} / {category}: {len(category_data)} circuits, "
                      f"{len(result['shared_bottlenecks'])} shared bottlenecks, "
                      f"Jaccard={result['avg_jaccard_similarity']:.4f}")

    # ---- Cross-Category Analysis ----
    print("\n[3/4] Computing cross-category metrics...")
    cross_results = []

    for model in MODELS:
        model_bn_sets = {
            cat: all_bottleneck_sets.get((model, cat), set())
            for cat in CATEGORIES
        }
        model_within = [r for r in within_results if r['model'] == model]

        result = compute_cross_category_metrics(model_within, model_bn_sets, model)
        cross_results.append(result)

        print(f"  {model}: {result['universal_count']} universal, "
              f"category-specific: {result['category_specific_counts']}")

    # ---- Model Comparison ----
    print("\n[4/4] Computing model comparison...")
    model_comparison = {}

    if len(cross_results) == 2:
        gemma_cr = cross_results[0] if cross_results[0]['model'] == 'gemma-2-2b' else cross_results[1]
        qwen_cr = cross_results[1] if cross_results[1]['model'] == 'qwen3-4b' else cross_results[0]

        gemma_universal = set(gemma_cr['universal_bottlenecks'])
        qwen_universal = set(qwen_cr['universal_bottlenecks'])

        model_comparison = {
            'shared_universal': list(gemma_universal & qwen_universal),
            'shared_universal_count': len(gemma_universal & qwen_universal),
            'gemma_only_universal': list(gemma_universal - qwen_universal),
            'gemma_only_universal_count': len(gemma_universal - qwen_universal),
            'qwen_only_universal': list(qwen_universal - gemma_universal),
            'qwen_only_universal_count': len(qwen_universal - gemma_universal),
            'jaccard_comparison': {},
            'layer_comparison': {},
        }

        for cat in CATEGORIES:
            model_comparison['jaccard_comparison'][cat] = {}
            model_comparison['layer_comparison'][cat] = {}
            for wr in within_results:
                if wr['category'] == cat:
                    model_comparison['jaccard_comparison'][cat][wr['model']] = wr['avg_jaccard_similarity']
                    model_comparison['layer_comparison'][cat][wr['model']] = (
                        next((cr['avg_bottleneck_layer'].get(cat) for cr in cross_results if cr['model'] == wr['model']), None)
                    )

    # ---- Save Results ----
    print("\nSaving results...")

    # JSON data
    results_json = {
        'within_category': within_results,
        'cross_category': cross_results,
        'model_comparison': model_comparison,
    }

    json_path = output_dir / 'cross_category_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Markdown report
    report = generate_report(within_results, cross_results, model_comparison, output_dir)
    report_path = output_dir / 'cross_category_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # Summary stats for quick reference
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    for cr in cross_results:
        print(f"\n  {cr['model'].upper()}:")
        print(f"    Universal bottlenecks: {cr['universal_count']}")
        for cat, count in cr['category_specific_counts'].items():
            print(f"    {cat.title()}-specific: {count}")
        for cat, avg in cr['avg_bottleneck_layer'].items():
            if avg is not None:
                print(f"    Avg bottleneck layer ({cat}): L{avg:.1f}")

    if model_comparison:
        print(f"\n  MODEL COMPARISON:")
        print(f"    Shared universal features: {model_comparison.get('shared_universal_count', 0)}")


if __name__ == '__main__':
    main()
