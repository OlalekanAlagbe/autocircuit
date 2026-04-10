#!/usr/bin/env python3
"""Stage 2: Feature Co-Activation Patterns - which bottleneck features appear together.

Analyzes co-occurrence of bottleneck features across 60 circuits. Cross-references
converted_graph.json nodes with the bottleneck library to find feature clusters.

Analyses:
  1. Co-occurrence matrix: which bottleneck features appear together across circuits
  2. Jaccard similarity between feature activation sets
  3. Within-layer vs cross-layer co-activation rates
  4. Feature clusters via hierarchical clustering (scipy optional)

Output: data/stage_2_coactivation/coactivation_results.json + coactivation_report.md

Usage:
    python scripts/stage_2_feature_coactivation.py
"""

import json
import sys
import io
import os
import argparse
import math
import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Tuple
from itertools import combinations

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_coactivation'
BOTTLENECK_LIBRARY = DATA_DIR / 'stage_1_5_bottleneck_library.json'

MODELS = ['gemma-2-2b', 'qwen3-4b']

CATEGORIES = {
    'chemistry': [
        'the-chemical-symbol-for-gold-is', 'the-chemical-symbol-for-iron-is',
        'the-chemical-symbol-for-lead-is', 'the-chemical-symbol-for-mercury-is',
        'the-chemical-symbol-for-argon-is', 'the-chemical-symbol-for-sodium-is',
        'the-chemical-symbol-for-silver-is', 'the-chemical-symbol-for-potassium-is',
        'the-chemical-symbol-for-copper-is', 'the-chemical-symbol-for-tungsten-is',
    ],
    'geography': [
        'the-capital-of-france-is', 'the-capital-of-japan-is', 'the-capital-of-germany-is',
        'the-capital-of-egypt-is', 'the-capital-of-italy-is', 'the-capital-of-spain-is',
        'the-capital-of-canada-is', 'the-capital-of-thailand-is', 'the-capital-of-turkey-is',
        'the-capital-of-south-korea-is',
    ],
    'history': [
        'world-war-ii-ended-in', 'the-first-moon-landing-was-in', 'the-berlin-wall-fell-in',
        'the-titanic-sank-in', 'america-declared-independence-in', 'the-french-revolution-began-in',
        'world-war-i-started-in', 'columbus-reached-the-americas-in',
        'the-great-fire-of-london-occurred-in', 'nelson-mandela-was-released-from-prison-in',
    ],
}


def load_bottleneck_library() -> dict:
    """Load the cross-circuit bottleneck feature library."""
    with open(BOTTLENECK_LIBRARY, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_circuit_features(model: str, slug: str) -> Set[str]:
    """Load converted_graph.json and return set of feature labels present."""
    dir_name = f"{model}_{slug}"
    conv_dir = PROMPTS_DIR / dir_name / '2_conversion'
    if not conv_dir.exists():
        return set()
    matches = list(conv_dir.glob('*converted_graph.json'))
    if not matches:
        return set()
    with open(matches[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {node['label'] for node in data.get('nodes', [])}


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Feature Co-Activation Patterns')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: FEATURE CO-ACTIVATION PATTERNS")
    print("=" * 70)

    # Step 1: Load bottleneck library
    print("\n[1/5] Loading bottleneck library...")
    library = load_bottleneck_library()
    cross_features = library.get('cross_circuit_features', {})
    print(f"  {len(cross_features)} cross-circuit features in library")

    # Build feature -> layer mapping
    feature_layers = {}
    for label, info in cross_features.items():
        feature_layers[label] = info.get('layer', -1)

    # Step 2: Build per-circuit feature sets
    print("\n[2/5] Loading circuit features...")
    circuit_features = {}  # circuit_id -> set of feature labels
    loaded = 0
    for model in MODELS:
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                cid = f"{model}_{slug}"
                feats = get_circuit_features(model, slug)
                if feats:
                    circuit_features[cid] = feats
                    loaded += 1
                else:
                    print(f"  [WARN] Missing: {cid}")
    print(f"  Loaded {loaded} circuits")

    # Step 3: Build feature -> circuit sets (only for bottleneck features)
    print("\n[3/5] Building co-occurrence data...")
    feature_circuits = {}  # feature_label -> set of circuit_ids
    for label in cross_features:
        circuits_with = set()
        for cid, feats in circuit_features.items():
            if label in feats:
                circuits_with.add(cid)
        if circuits_with:
            feature_circuits[label] = circuits_with

    print(f"  {len(feature_circuits)} bottleneck features found in circuits")

    # Filter to features appearing in 2+ circuits for meaningful co-activation
    active_features = {k: v for k, v in feature_circuits.items() if len(v) >= 2}
    print(f"  {len(active_features)} features appear in 2+ circuits")

    # Step 4: Compute co-occurrence and Jaccard
    print("\n[4/5] Computing co-occurrence matrix and Jaccard similarities...")
    feat_list = sorted(active_features.keys())
    n_feats = len(feat_list)
    feat_idx = {f: i for i, f in enumerate(feat_list)}

    # Co-occurrence: count circuits where both features appear
    cooccurrence = [[0] * n_feats for _ in range(n_feats)]
    top_jaccard = []
    same_layer_pairs = 0
    cross_layer_pairs = 0
    same_layer_jaccard_sum = 0.0
    cross_layer_jaccard_sum = 0.0

    for i in range(n_feats):
        cooccurrence[i][i] = len(active_features[feat_list[i]])

    pairs_computed = 0
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            fa, fb = feat_list[i], feat_list[j]
            sa, sb = active_features[fa], active_features[fb]
            intersection = len(sa & sb)
            union = len(sa | sb)
            cooccurrence[i][j] = intersection
            cooccurrence[j][i] = intersection

            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0

            if jaccard > 0:
                top_jaccard.append((fa, fb, jaccard, intersection))

            # Layer analysis
            la = feature_layers.get(fa, -1)
            lb = feature_layers.get(fb, -1)
            if la >= 0 and lb >= 0:
                if la == lb:
                    same_layer_pairs += 1
                    same_layer_jaccard_sum += jaccard
                else:
                    cross_layer_pairs += 1
                    cross_layer_jaccard_sum += jaccard

            pairs_computed += 1

    top_jaccard.sort(key=lambda x: x[2], reverse=True)
    print(f"  Computed {pairs_computed} pairs")
    print(f"  Top Jaccard: {top_jaccard[0][2]:.3f} ({top_jaccard[0][0]} & {top_jaccard[0][1]})" if top_jaccard else "  No co-occurrences found")

    # Layer analysis summary
    avg_same = same_layer_jaccard_sum / same_layer_pairs if same_layer_pairs > 0 else 0.0
    avg_cross = cross_layer_jaccard_sum / cross_layer_pairs if cross_layer_pairs > 0 else 0.0
    print(f"  Same-layer pairs: {same_layer_pairs}, avg Jaccard: {avg_same:.4f}")
    print(f"  Cross-layer pairs: {cross_layer_pairs}, avg Jaccard: {avg_cross:.4f}")

    # Step 5: Per-model analysis
    print("\n[5/5] Per-model analysis...")
    model_stats = {}
    for model in MODELS:
        model_circuits = {cid for cid in circuit_features if cid.startswith(model)}
        model_feats = {}
        for label, circuits in active_features.items():
            model_intersection = circuits & model_circuits
            if len(model_intersection) >= 2:
                model_feats[label] = model_intersection

        # Top co-occurring pairs for this model
        model_pairs = []
        mf_list = sorted(model_feats.keys())
        for i in range(len(mf_list)):
            for j in range(i + 1, len(mf_list)):
                fa, fb = mf_list[i], mf_list[j]
                sa, sb = model_feats[fa], model_feats[fb]
                inter = len(sa & sb)
                union = len(sa | sb)
                if inter > 0 and union > 0:
                    model_pairs.append((fa, fb, inter / union, inter))

        model_pairs.sort(key=lambda x: x[2], reverse=True)

        # Layer distribution of active features
        layer_dist = Counter()
        for label in model_feats:
            l = feature_layers.get(label, -1)
            if l >= 0:
                layer_dist[l] += 1

        model_stats[model] = {
            'n_active_features': len(model_feats),
            'top_20_pairs': [(a, b, j, n) for a, b, j, n in model_pairs[:20]],
            'layer_distribution': dict(sorted(layer_dist.items())),
        }
        print(f"  {model}: {len(model_feats)} active features, {len(model_pairs)} co-occurring pairs")

    # Hierarchical clustering (optional - requires scipy)
    clustering = None
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert co-occurrence to distance (1 - normalized co-occurrence)
        max_cooc = max(max(row) for row in cooccurrence) if cooccurrence else 1
        if max_cooc > 0 and n_feats > 2:
            dist_matrix = [[0.0] * n_feats for _ in range(n_feats)]
            for i in range(n_feats):
                for j in range(n_feats):
                    if i != j:
                        norm_cooc = cooccurrence[i][j] / max_cooc
                        dist_matrix[i][j] = 1.0 - norm_cooc

            # Convert to condensed form
            condensed = []
            for i in range(n_feats):
                for j in range(i + 1, n_feats):
                    condensed.append(dist_matrix[i][j])

            Z = linkage(condensed, method='ward')
            # Cut at different thresholds
            clusters_5 = fcluster(Z, t=5, criterion='maxclust')
            clusters_10 = fcluster(Z, t=10, criterion='maxclust')

            # Build cluster membership
            cluster_members_5 = defaultdict(list)
            cluster_members_10 = defaultdict(list)
            for idx, (c5, c10) in enumerate(zip(clusters_5, clusters_10)):
                cluster_members_5[int(c5)].append(feat_list[idx])
                cluster_members_10[int(c10)].append(feat_list[idx])

            clustering = {
                'n_clusters_5': len(cluster_members_5),
                'n_clusters_10': len(cluster_members_10),
                'clusters_5': {str(k): v for k, v in sorted(cluster_members_5.items())},
                'clusters_10': {str(k): v for k, v in sorted(cluster_members_10.items())},
            }
            print(f"  Clustering: {len(cluster_members_5)} clusters (k=5), "
                  f"{len(cluster_members_10)} clusters (k=10)")
    except ImportError:
        print("  [INFO] scipy not available - skipping hierarchical clustering")

    # Compile results
    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'circuits_loaded': loaded,
            'library_features': len(cross_features),
            'features_in_circuits': len(feature_circuits),
            'features_2plus': len(active_features),
        },
        'layer_analysis': {
            'same_layer_pairs': same_layer_pairs,
            'cross_layer_pairs': cross_layer_pairs,
            'avg_same_layer_jaccard': avg_same,
            'avg_cross_layer_jaccard': avg_cross,
            'ratio': avg_same / avg_cross if avg_cross > 0 else float('inf'),
        },
        'top_50_jaccard': [
            {'feature_a': a, 'feature_b': b, 'jaccard': j, 'shared_circuits': n}
            for a, b, j, n in top_jaccard[:50]
        ],
        'model_stats': model_stats,
        'clustering': clustering,
        'feature_circuit_counts': {
            label: len(circuits) for label, circuits in
            sorted(active_features.items(), key=lambda x: len(x[1]), reverse=True)[:50]
        },
    }

    # Save JSON
    json_path = output_dir / 'coactivation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Feature Co-Activation Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits**: {loaded}")
    lines.append(f"**Library features**: {len(cross_features)}")
    lines.append(f"**Active features** (2+ circuits): {len(active_features)}\n")
    lines.append("---\n")

    lines.append("## 1. Layer Analysis\n")
    la = results['layer_analysis']
    lines.append(f"- Same-layer co-occurring pairs: **{la['same_layer_pairs']}**")
    lines.append(f"- Cross-layer co-occurring pairs: **{la['cross_layer_pairs']}**")
    lines.append(f"- Avg same-layer Jaccard: **{la['avg_same_layer_jaccard']:.4f}**")
    lines.append(f"- Avg cross-layer Jaccard: **{la['avg_cross_layer_jaccard']:.4f}**")
    lines.append(f"- Ratio (same/cross): **{la['ratio']:.2f}x**\n")
    if la['avg_same_layer_jaccard'] > la['avg_cross_layer_jaccard']:
        lines.append("> Features in the **same layer** co-activate more strongly, "
                     "suggesting within-layer functional modules.\n")
    else:
        lines.append("> Features across **different layers** co-activate more strongly, "
                     "suggesting cross-layer functional chains.\n")

    lines.append("## 2. Top Co-Activating Feature Pairs\n")
    lines.append("| Rank | Feature A | Feature B | Jaccard | Shared Circuits |")
    lines.append("|------|-----------|-----------|---------|-----------------|")
    for rank, entry in enumerate(results['top_50_jaccard'][:25], 1):
        lines.append(f"| {rank} | {entry['feature_a']} | {entry['feature_b']} | "
                     f"{entry['jaccard']:.3f} | {entry['shared_circuits']} |")
    lines.append("")

    lines.append("## 3. Most Connected Features\n")
    lines.append("| Feature | Circuits |")
    lines.append("|---------|----------|")
    for label, count in list(results['feature_circuit_counts'].items())[:20]:
        lines.append(f"| {label} | {count} |")
    lines.append("")

    lines.append("## 4. Per-Model Analysis\n")
    for model in MODELS:
        ms = model_stats.get(model, {})
        if not ms:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append(f"- Active features: **{ms['n_active_features']}**")
        lines.append(f"- Layer distribution: {ms['layer_distribution']}\n")
        lines.append("**Top co-activating pairs:**\n")
        lines.append("| Feature A | Feature B | Jaccard | Shared |")
        lines.append("|-----------|-----------|---------|--------|")
        for a, b, j, n in ms['top_20_pairs'][:10]:
            lines.append(f"| {a} | {b} | {j:.3f} | {n} |")
        lines.append("")

    if clustering:
        lines.append("## 5. Feature Clusters (Hierarchical)\n")
        lines.append(f"**5 clusters:**\n")
        for cid, members in sorted(clustering['clusters_5'].items()):
            lines.append(f"- Cluster {cid} ({len(members)} features): "
                         f"{', '.join(members[:5])}{'...' if len(members) > 5 else ''}")
        lines.append("")

    report_path = output_dir / 'coactivation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Active features: {len(active_features)}")
    print(f"  Same-layer Jaccard: {avg_same:.4f}")
    print(f"  Cross-layer Jaccard: {avg_cross:.4f}")
    print(f"  Top pair: {top_jaccard[0][0]} & {top_jaccard[0][1]} (J={top_jaccard[0][2]:.3f})" if top_jaccard else "")
    print("\nDone!")


if __name__ == '__main__':
    main()
