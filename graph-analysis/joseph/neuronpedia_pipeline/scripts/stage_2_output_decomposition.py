#!/usr/bin/env python3
"""Stage 2: Output Logit Decomposition - bottleneck vs non-bottleneck path contribution.

Decomposes output node contributions by tracing paths backward through circuit graphs.
Tests whether output energy flowing through bottleneck layers predicts model confidence.

Key question: Does a higher fraction of output coming through bottleneck paths
predict confidence? Direct test of evidence accumulation hypothesis.

Output: data/stage_2_output_decomposition/output_decomposition_results.json + report.md

Usage:
    python scripts/stage_2_output_decomposition.py
"""

import json
import sys
import io
import os
import argparse
import math
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_output_decomposition'

MODELS = ['gemma-2-2b', 'qwen3-4b']
MODEL_LAYERS = {'gemma-2-2b': 26, 'qwen3-4b': 36}

# Bottleneck layer ranges (from Stage 2 findings)
BOTTLENECK_RANGES = {
    'gemma-2-2b': (5, 7),   # L5-L7
    'qwen3-4b': (22, 25),   # L22-L25
}

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


def pearson_r(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson correlation and approximate p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0, 1.0
    r = num / (dx * dy)
    # Approximate p-value via t-distribution
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # Approximate 2-tail p using normal for large n
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def load_graph(model: str, slug: str) -> Optional[dict]:
    """Load converted_graph.json."""
    d = PROMPTS_DIR / f"{model}_{slug}" / '2_conversion'
    if not d.exists():
        return None
    matches = list(d.glob('*converted_graph.json'))
    if not matches:
        return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def load_traceback(model: str, slug: str) -> Optional[dict]:
    """Load traceback_paths.json."""
    p = PROMPTS_DIR / f"{model}_{slug}" / '3_analysis' / 'traceback_paths.json'
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def decompose_circuit(graph_data: dict, traceback_data: dict, model: str) -> Optional[dict]:
    """Decompose a circuit's output into bottleneck vs non-bottleneck contributions.

    Approach:
    1. Build adjacency from edges (node_id -> [(target, weight)])
    2. For each critical path, walk node sequence and classify edges
    3. Edge is "bottleneck-passing" if source or target layer is in bottleneck range
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    metadata = graph_data.get('metadata', {})
    confidence = metadata.get('output_probability', 0.0)

    if not nodes or not edges:
        return None

    bn_lo, bn_hi = BOTTLENECK_RANGES[model]

    # Build node_id -> layer lookup
    node_layer = {}
    for node in nodes:
        node_layer[node['id']] = node.get('layer', 0)

    # Build label -> node_ids mapping (a label may map to multiple nodes)
    label_to_ids = defaultdict(list)
    for node in nodes:
        label_to_ids[node['label']].append(node['id'])

    # Build edge lookup: (source_id, target_id) -> weight
    edge_weights = {}
    # Also build adjacency for path walking
    adj_out = defaultdict(list)  # node_id -> [(target_id, weight)]
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        w = edge.get('weight', 0.0)
        edge_weights[(src, tgt)] = w
        adj_out[src].append((tgt, w))

    # Process critical paths
    paths = traceback_data.get('critical_paths', [])
    if not paths:
        return None

    path_results = []
    for path_data in paths:
        path_nodes = path_data.get('path_nodes', [])
        if len(path_nodes) < 2:
            continue

        # Walk consecutive pairs and look up edge weights
        bn_energy = 0.0
        non_bn_energy = 0.0
        total_energy = 0.0
        edges_found = 0
        edges_missed = 0

        for k in range(len(path_nodes) - 1):
            src_label = path_nodes[k].get('label', '')
            tgt_label = path_nodes[k + 1].get('label', '')
            src_layer = path_nodes[k].get('layer', -1)
            tgt_layer = path_nodes[k + 1].get('layer', -1)

            # Try to find the actual edge weight by matching node IDs
            best_weight = None
            for sid in label_to_ids.get(src_label, []):
                for tid in label_to_ids.get(tgt_label, []):
                    if (sid, tid) in edge_weights:
                        best_weight = edge_weights[(sid, tid)]
                        break
                if best_weight is not None:
                    break

            if best_weight is None:
                # Fallback: use score difference as proxy
                s1 = path_nodes[k].get('score', 0)
                s2 = path_nodes[k + 1].get('score', 0)
                best_weight = abs(s1 - s2) if s1 and s2 else 0.0
                edges_missed += 1
            else:
                edges_found += 1

            abs_w = abs(best_weight)
            total_energy += abs_w

            # Classify: bottleneck if either end is in bottleneck range
            if (bn_lo <= src_layer <= bn_hi) or (bn_lo <= tgt_layer <= bn_hi):
                bn_energy += abs_w
            else:
                non_bn_energy += abs_w

        bn_ratio = bn_energy / total_energy if total_energy > 0 else 0.0

        path_results.append({
            'path_length': len(path_nodes),
            'total_energy': total_energy,
            'bottleneck_energy': bn_energy,
            'non_bottleneck_energy': non_bn_energy,
            'bottleneck_ratio': bn_ratio,
            'edges_found': edges_found,
            'edges_missed': edges_missed,
        })

    if not path_results:
        return None

    # Aggregate across paths
    avg_bn_ratio = sum(p['bottleneck_ratio'] for p in path_results) / len(path_results)
    total_bn = sum(p['bottleneck_energy'] for p in path_results)
    total_all = sum(p['total_energy'] for p in path_results)
    overall_bn_ratio = total_bn / total_all if total_all > 0 else 0.0

    # Also compute edge-level decomposition across ALL edges (not just paths)
    all_bn_weight = 0.0
    all_non_bn_weight = 0.0
    for edge in edges:
        src_id = edge['source']
        tgt_id = edge['target']
        w = abs(edge.get('weight', 0.0))
        src_l = node_layer.get(src_id, -1)
        tgt_l = node_layer.get(tgt_id, -1)
        if (bn_lo <= src_l <= bn_hi) or (bn_lo <= tgt_l <= bn_hi):
            all_bn_weight += w
        else:
            all_non_bn_weight += w

    all_total = all_bn_weight + all_non_bn_weight
    all_bn_ratio = all_bn_weight / all_total if all_total > 0 else 0.0

    return {
        'confidence': confidence,
        'n_paths': len(path_results),
        'path_avg_bn_ratio': avg_bn_ratio,
        'overall_bn_ratio': overall_bn_ratio,
        'total_path_energy': total_all,
        'total_bn_energy': total_bn,
        'all_edges_bn_ratio': all_bn_ratio,
        'all_edges_bn_weight': all_bn_weight,
        'all_edges_total_weight': all_total,
        'paths': path_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Output Logit Decomposition')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: OUTPUT LOGIT DECOMPOSITION")
    print("=" * 70)

    # Process all circuits
    print("\n[1/3] Processing circuits...")
    per_circuit = {}
    by_model = {m: [] for m in MODELS}
    by_model_cat = {m: {c: [] for c in CATEGORIES} for m in MODELS}
    loaded = 0
    missing = 0

    for model in MODELS:
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                graph = load_graph(model, slug)
                traceback = load_traceback(model, slug)
                if graph is None or traceback is None:
                    print(f"  [WARN] Missing: {model}/{slug}")
                    missing += 1
                    continue

                result = decompose_circuit(graph, traceback, model)
                if result is None:
                    missing += 1
                    continue

                result['model'] = model
                result['category'] = cat
                result['slug'] = slug
                cid = f"{model}_{slug}"
                per_circuit[cid] = result
                by_model[model].append(result)
                by_model_cat[model][cat].append(result)
                loaded += 1

    print(f"  Loaded: {loaded}, Missing: {missing}")

    # Correlation analysis
    print("\n[2/3] Correlation analysis...")
    correlations = {}
    for model in MODELS:
        data = by_model[model]
        if len(data) < 5:
            continue

        # Path-level bottleneck ratio vs confidence
        x_path = [d['path_avg_bn_ratio'] for d in data]
        y_conf = [d['confidence'] for d in data]
        r_path, p_path = pearson_r(x_path, y_conf)

        # All-edges bottleneck ratio vs confidence
        x_all = [d['all_edges_bn_ratio'] for d in data]
        r_all, p_all = pearson_r(x_all, y_conf)

        # Total path energy vs confidence
        x_energy = [d['total_path_energy'] for d in data]
        r_energy, p_energy = pearson_r(x_energy, y_conf)

        correlations[model] = {
            'path_bn_ratio_vs_confidence': {'r': r_path, 'p': p_path, 'n': len(data)},
            'all_edges_bn_ratio_vs_confidence': {'r': r_all, 'p': p_all, 'n': len(data)},
            'total_energy_vs_confidence': {'r': r_energy, 'p': p_energy, 'n': len(data)},
        }

        print(f"  {model}:")
        print(f"    Path BN ratio vs confidence:  r={r_path:.3f}, p={p_path:.4f}")
        print(f"    All-edges BN ratio vs conf:   r={r_all:.3f}, p={p_all:.4f}")
        print(f"    Total energy vs confidence:    r={r_energy:.3f}, p={p_energy:.4f}")

    # Per-category breakdown
    print("\n[3/3] Per-category breakdown...")
    category_stats = {}
    for model in MODELS:
        category_stats[model] = {}
        for cat in CATEGORIES:
            data = by_model_cat[model][cat]
            if not data:
                continue
            avg_bn = sum(d['path_avg_bn_ratio'] for d in data) / len(data)
            avg_all_bn = sum(d['all_edges_bn_ratio'] for d in data) / len(data)
            avg_conf = sum(d['confidence'] for d in data) / len(data)
            category_stats[model][cat] = {
                'n': len(data),
                'avg_path_bn_ratio': avg_bn,
                'avg_all_edges_bn_ratio': avg_all_bn,
                'avg_confidence': avg_conf,
            }
            print(f"  {model}/{cat}: path_bn={avg_bn:.3f}, "
                  f"all_bn={avg_all_bn:.3f}, conf={avg_conf:.3f}")

    # Strip per-path details for JSON size
    per_circuit_slim = {}
    for cid, data in per_circuit.items():
        slim = {k: v for k, v in data.items() if k != 'paths'}
        per_circuit_slim[cid] = slim

    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'loaded': loaded,
            'missing': missing,
            'bottleneck_ranges': {k: list(v) for k, v in BOTTLENECK_RANGES.items()},
        },
        'per_circuit': per_circuit_slim,
        'correlations': correlations,
        'category_stats': category_stats,
        'model_summary': {},
    }

    # Model-level summary
    for model in MODELS:
        data = by_model[model]
        if not data:
            continue
        results['model_summary'][model] = {
            'n': len(data),
            'avg_path_bn_ratio': sum(d['path_avg_bn_ratio'] for d in data) / len(data),
            'avg_all_edges_bn_ratio': sum(d['all_edges_bn_ratio'] for d in data) / len(data),
            'avg_confidence': sum(d['confidence'] for d in data) / len(data),
            'avg_total_energy': sum(d['total_path_energy'] for d in data) / len(data),
        }

    # Save JSON
    json_path = output_dir / 'output_decomposition_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Output Logit Decomposition Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits**: {loaded}")
    lines.append(f"**Bottleneck ranges**: GEMMA L5-L7, QWEN L22-L25\n")
    lines.append("---\n")

    lines.append("## 1. Model Summary\n")
    lines.append("| Model | N | Avg Path BN Ratio | Avg All-Edge BN Ratio | Avg Confidence |")
    lines.append("|-------|---|-------------------|-----------------------|----------------|")
    for model in MODELS:
        ms = results['model_summary'].get(model, {})
        if not ms:
            continue
        lines.append(f"| {model} | {ms['n']} | {ms['avg_path_bn_ratio']:.3f} | "
                     f"{ms['avg_all_edges_bn_ratio']:.3f} | {ms['avg_confidence']:.3f} |")
    lines.append("")

    lines.append("## 2. Correlation: Bottleneck Contribution vs Confidence\n")
    lines.append("**Key test**: Does higher bottleneck contribution predict higher confidence?\n")
    for model in MODELS:
        corr = correlations.get(model, {})
        if not corr:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append("| Metric | r | p | N |")
        lines.append("|--------|---|---|---|")
        for name, vals in corr.items():
            sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else ""
            lines.append(f"| {name} | {vals['r']:.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")
        lines.append("")

    lines.append("## 3. Per-Category Breakdown\n")
    lines.append("| Model | Category | N | Path BN Ratio | All-Edge BN Ratio | Confidence |")
    lines.append("|-------|----------|---|---------------|-------------------|------------|")
    for model in MODELS:
        cs = category_stats.get(model, {})
        for cat in sorted(CATEGORIES.keys()):
            data = cs.get(cat, {})
            if not data:
                continue
            lines.append(f"| {model} | {cat} | {data['n']} | {data['avg_path_bn_ratio']:.3f} | "
                         f"{data['avg_all_edges_bn_ratio']:.3f} | {data['avg_confidence']:.3f} |")
    lines.append("")

    lines.append("## 4. Interpretation\n")
    for model in MODELS:
        corr = correlations.get(model, {})
        if not corr:
            continue
        r_path = corr.get('path_bn_ratio_vs_confidence', {}).get('r', 0)
        p_path = corr.get('path_bn_ratio_vs_confidence', {}).get('p', 1)
        lines.append(f"### {model.upper()}\n")
        if p_path < 0.05:
            if r_path > 0:
                lines.append(f"**Positive correlation** (r={r_path:.3f}, p={p_path:.4f}): "
                             "Higher bottleneck contribution **predicts higher confidence**. "
                             "Supports evidence accumulation hypothesis.")
            else:
                lines.append(f"**Negative correlation** (r={r_path:.3f}, p={p_path:.4f}): "
                             "Higher bottleneck contribution **predicts LOWER confidence**. "
                             "Consistent with 'bottleneck tax' finding — compression costs accuracy.")
        else:
            lines.append(f"**No significant correlation** (r={r_path:.3f}, p={p_path:.4f}): "
                         "Bottleneck contribution does not predict confidence. "
                         "Contribution is distributed rather than concentrated.")
        lines.append("")

    report_path = output_dir / 'output_decomposition_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    for model in MODELS:
        ms = results['model_summary'].get(model, {})
        corr = correlations.get(model, {})
        if not ms:
            continue
        r = corr.get('path_bn_ratio_vs_confidence', {}).get('r', 0)
        p = corr.get('path_bn_ratio_vs_confidence', {}).get('p', 1)
        print(f"  {model.upper()}: BN ratio={ms['avg_path_bn_ratio']:.3f}, "
              f"corr with confidence: r={r:.3f}, p={p:.4f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
