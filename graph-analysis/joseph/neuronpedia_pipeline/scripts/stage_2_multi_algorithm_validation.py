#!/usr/bin/env python3
"""Stage 2: Multi-Algorithm Validation at Scale.

Runs 4 independent community detection algorithms on ALL 60 circuits
and measures pairwise agreement (Jaccard) to validate that community
structure is real, not an artifact of any single algorithm.

Methods:
  1. Louvain (modularity optimization)
  2. Leiden (guaranteed well-connected)
  3. Greedy Modularity (fast CNM)
  4. Label Propagation (neighbor voting)

Output: data/stage_2_multi_algorithm/multi_algorithm_results.json + report.md

Usage:
    python scripts/stage_2_multi_algorithm_validation.py
"""

import json
import sys
import io
import os
import argparse
import datetime
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import networkx as nx
import community as louvain_community

# Try importing Leiden dependencies
LEIDEN_AVAILABLE = False
try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_multi_algorithm'
DATA_TABLE = DATA_DIR / 'stage_2_statistical_deepdive' / 'data_table.json'


# ========================================================================
# Community Detection Methods (adapted from 3_analyze_circuit_multi.py)
# ========================================================================

def detect_louvain(G, resolution=1.0):
    """Louvain community detection."""
    G_u = G.to_undirected()
    return louvain_community.best_partition(G_u, weight='weight', resolution=resolution)


def detect_leiden(G, resolution=0.01):
    """Leiden community detection (requires igraph + leidenalg)."""
    if not LEIDEN_AVAILABLE:
        return None
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    edges = []
    weights = []
    for u, v in G.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(abs(G[u][v].get('weight', 1.0)))
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    try:
        partition = la.find_partition(
            ig_graph, la.CPMVertexPartition,
            weights='weight', resolution_parameter=resolution
        )
        return {node_list[i]: partition.membership[i] for i in range(len(node_list))}
    except Exception:
        return None


def detect_greedy(G):
    """Greedy modularity maximization."""
    G_u = G.to_undirected()
    comms = nx.community.greedy_modularity_communities(G_u, weight='weight', resolution=1.0)
    result = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            result[node] = cid
    return result


def detect_label_prop(G):
    """Asynchronous label propagation."""
    G_u = G.to_undirected()
    comms = nx.community.asyn_lpa_communities(G_u, weight='weight')
    result = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            result[node] = cid
    return result


def calc_modularity(G, partition):
    """Calculate modularity for a partition."""
    unique_comms = set(partition.values())
    comm_sets = [set(n for n, c in partition.items() if c == cid) for cid in unique_comms]
    G_u = G.to_undirected()
    return nx.community.modularity(G_u, comm_sets, weight='weight')


def pairwise_jaccard(p1, p2):
    """Compute Jaccard similarity between two partitions using best-match strategy."""
    def to_comm_sets(p):
        comms = defaultdict(set)
        for node, cid in p.items():
            comms[cid].add(node)
        return list(comms.values())

    c1 = to_comm_sets(p1)
    c2 = to_comm_sets(p2)
    if not c1 or not c2:
        return 0.0

    total = 0.0
    for s1 in c1:
        best = 0.0
        for s2 in c2:
            inter = len(s1 & s2)
            union = len(s1 | s2)
            if union > 0:
                best = max(best, inter / union)
        total += best
    return total / len(c1)


# ========================================================================
# Graph Loading
# ========================================================================

def load_graph(graph_path):
    """Load a converted_graph.json into a NetworkX DiGraph."""
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get('nodes', []):
        G.add_node(node['id'], layer=node.get('layer', -1),
                   activation=node.get('activation', 0),
                   influence=node.get('influence', 0))
    for edge in data.get('edges', []):
        G.add_edge(edge['source'], edge['target'],
                   weight=abs(edge.get('weight', 1.0)))
    return G


def find_graph_file(circuit_id):
    """Find the converted_graph.json for a circuit."""
    prompt_dir = PROMPTS_DIR / circuit_id / '2_conversion'
    if not prompt_dir.exists():
        return None
    matches = list(prompt_dir.glob('*converted_graph.json'))
    return matches[0] if matches else None


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Multi-Algorithm Validation')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: MULTI-ALGORITHM VALIDATION AT SCALE")
    print("=" * 70)
    print(f"  Leiden available: {LEIDEN_AVAILABLE}")

    # Load data table
    print("\n[1/4] Loading circuit metadata...")
    with open(DATA_TABLE, 'r', encoding='utf-8') as f:
        data_table = json.load(f)
    print(f"  {len(data_table)} circuits in table")

    # Process each circuit
    print("\n[2/4] Running 4 algorithms on each circuit...")
    METHOD_NAMES = ['louvain', 'leiden', 'greedy', 'label_propagation']
    circuit_results = []
    errors = []

    for idx, row in enumerate(data_table):
        cid = row['circuit_id']
        model = row['model']
        category = row['category']
        slug = row.get('prompt_slug', '')

        graph_path = find_graph_file(cid)
        if graph_path is None:
            errors.append(f"Graph not found: {cid}")
            continue

        try:
            G = load_graph(graph_path)
        except Exception as e:
            errors.append(f"Load error {cid}: {e}")
            continue

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes < 3:
            errors.append(f"Too few nodes ({n_nodes}): {cid}")
            continue

        # Run each algorithm
        partitions = {}
        modularities = {}

        # Louvain
        try:
            p = detect_louvain(G)
            partitions['louvain'] = p
            modularities['louvain'] = calc_modularity(G, p)
        except Exception as e:
            errors.append(f"Louvain failed {cid}: {e}")

        # Leiden
        try:
            p = detect_leiden(G)
            if p is not None:
                partitions['leiden'] = p
                modularities['leiden'] = calc_modularity(G, p)
        except Exception as e:
            errors.append(f"Leiden failed {cid}: {e}")

        # Greedy
        try:
            p = detect_greedy(G)
            partitions['greedy'] = p
            modularities['greedy'] = calc_modularity(G, p)
        except Exception as e:
            errors.append(f"Greedy failed {cid}: {e}")

        # Label Propagation
        try:
            p = detect_label_prop(G)
            partitions['label_propagation'] = p
            modularities['label_propagation'] = calc_modularity(G, p)
        except Exception as e:
            errors.append(f"LabelProp failed {cid}: {e}")

        # Pairwise Jaccard
        methods_available = list(partitions.keys())
        pair_jaccards = {}
        for i, m1 in enumerate(methods_available):
            for m2 in methods_available[i+1:]:
                j = pairwise_jaccard(partitions[m1], partitions[m2])
                pair_jaccards[f"{m1}_vs_{m2}"] = j

        avg_jaccard = sum(pair_jaccards.values()) / len(pair_jaccards) if pair_jaccards else 0.0

        # Classify agreement
        if avg_jaccard > 0.7:
            quality = 'VALID'
        elif avg_jaccard > 0.5:
            quality = 'MODERATE'
        else:
            quality = 'WEAK'

        n_communities = {m: len(set(p.values())) for m, p in partitions.items()}

        circuit_results.append({
            'circuit_id': cid,
            'model': model,
            'category': category,
            'prompt_slug': slug,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'methods_ran': len(partitions),
            'modularities': modularities,
            'n_communities': n_communities,
            'pairwise_jaccards': pair_jaccards,
            'avg_jaccard': avg_jaccard,
            'quality': quality,
        })

        status = f"  [{idx+1:2d}/60] {cid[:45]:45s} J={avg_jaccard:.3f} [{quality}]"
        print(status)

    # Aggregate
    print(f"\n[3/4] Aggregating results...")
    print(f"  Processed: {len(circuit_results)}, Errors: {len(errors)}")

    # Overall stats
    all_jaccards = [r['avg_jaccard'] for r in circuit_results]
    n_valid = sum(1 for r in circuit_results if r['quality'] == 'VALID')
    n_moderate = sum(1 for r in circuit_results if r['quality'] == 'MODERATE')
    n_weak = sum(1 for r in circuit_results if r['quality'] == 'WEAK')
    overall_avg = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0

    print(f"\n  OVERALL:")
    print(f"    Mean Jaccard: {overall_avg:.3f}")
    print(f"    VALID (>0.7):   {n_valid} ({100*n_valid/len(circuit_results):.0f}%)")
    print(f"    MODERATE (0.5-0.7): {n_moderate} ({100*n_moderate/len(circuit_results):.0f}%)")
    print(f"    WEAK (<0.5):    {n_weak} ({100*n_weak/len(circuit_results):.0f}%)")

    # Per-model breakdown
    model_stats = {}
    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        model_circuits = [r for r in circuit_results if r['model'] == model_name]
        if not model_circuits:
            continue
        mj = [r['avg_jaccard'] for r in model_circuits]
        mv = sum(1 for r in model_circuits if r['quality'] == 'VALID')
        mm = sum(1 for r in model_circuits if r['quality'] == 'MODERATE')
        mw = sum(1 for r in model_circuits if r['quality'] == 'WEAK')
        avg_mod = {}
        for method in METHOD_NAMES:
            vals = [r['modularities'].get(method) for r in model_circuits
                    if method in r['modularities']]
            vals = [v for v in vals if v is not None]
            if vals:
                avg_mod[method] = sum(vals) / len(vals)
        model_stats[model_name] = {
            'n': len(model_circuits),
            'mean_jaccard': sum(mj) / len(mj),
            'std_jaccard': math.sqrt(sum((j - sum(mj)/len(mj))**2 for j in mj) / len(mj)) if len(mj) > 1 else 0,
            'n_valid': mv,
            'n_moderate': mm,
            'n_weak': mw,
            'avg_modularities': avg_mod,
        }
        print(f"\n  {model_name.upper()}:")
        print(f"    Mean Jaccard: {model_stats[model_name]['mean_jaccard']:.3f} "
              f"(std={model_stats[model_name]['std_jaccard']:.3f})")
        print(f"    VALID: {mv}, MODERATE: {mm}, WEAK: {mw}")
        for m, v in avg_mod.items():
            print(f"    Avg {m} modularity: {v:.4f}")

    # Per-category breakdown
    cat_stats = {}
    for cat in ['chemistry', 'geography', 'history']:
        cat_circuits = [r for r in circuit_results if r['category'] == cat]
        if not cat_circuits:
            continue
        cj = [r['avg_jaccard'] for r in cat_circuits]
        cv = sum(1 for r in cat_circuits if r['quality'] == 'VALID')
        cm = sum(1 for r in cat_circuits if r['quality'] == 'MODERATE')
        cw = sum(1 for r in cat_circuits if r['quality'] == 'WEAK')
        cat_stats[cat] = {
            'n': len(cat_circuits),
            'mean_jaccard': sum(cj) / len(cj),
            'n_valid': cv,
            'n_moderate': cm,
            'n_weak': cw,
        }
        print(f"\n  {cat.upper()}:")
        print(f"    Mean Jaccard: {cat_stats[cat]['mean_jaccard']:.3f}")
        print(f"    VALID: {cv}, MODERATE: {cm}, WEAK: {cw}")

    # Per-model-per-category
    model_cat_stats = {}
    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        model_cat_stats[model_name] = {}
        for cat in ['chemistry', 'geography', 'history']:
            mc = [r for r in circuit_results
                  if r['model'] == model_name and r['category'] == cat]
            if not mc:
                continue
            mcj = [r['avg_jaccard'] for r in mc]
            model_cat_stats[model_name][cat] = {
                'n': len(mc),
                'mean_jaccard': sum(mcj) / len(mcj),
                'n_valid': sum(1 for r in mc if r['quality'] == 'VALID'),
                'n_moderate': sum(1 for r in mc if r['quality'] == 'MODERATE'),
                'n_weak': sum(1 for r in mc if r['quality'] == 'WEAK'),
            }

    # Method-pair agreement averaged across all circuits
    pair_avg = defaultdict(list)
    for r in circuit_results:
        for pair, j in r['pairwise_jaccards'].items():
            pair_avg[pair].append(j)
    method_pair_means = {pair: sum(vals)/len(vals) for pair, vals in pair_avg.items()}

    print(f"\n  METHOD PAIR AVERAGES (across all circuits):")
    for pair, val in sorted(method_pair_means.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pair}: {val:.3f}")

    # Best and worst circuits
    sorted_results = sorted(circuit_results, key=lambda x: x['avg_jaccard'])
    bottom5 = sorted_results[:5]
    top5 = sorted_results[-5:]

    print(f"\n  TOP 5 (highest agreement):")
    for r in reversed(top5):
        print(f"    {r['circuit_id'][:50]:50s} J={r['avg_jaccard']:.3f}")

    print(f"\n  BOTTOM 5 (lowest agreement):")
    for r in bottom5:
        print(f"    {r['circuit_id'][:50]:50s} J={r['avg_jaccard']:.3f}")

    # Build results dict
    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_circuits': len(circuit_results),
            'n_errors': len(errors),
            'leiden_available': LEIDEN_AVAILABLE,
        },
        'overall': {
            'mean_jaccard': overall_avg,
            'n_valid': n_valid,
            'n_moderate': n_moderate,
            'n_weak': n_weak,
            'pct_valid_or_moderate': 100 * (n_valid + n_moderate) / len(circuit_results) if circuit_results else 0,
        },
        'model_stats': model_stats,
        'category_stats': cat_stats,
        'model_category_stats': model_cat_stats,
        'method_pair_means': method_pair_means,
        'circuit_results': circuit_results,
        'errors': errors,
    }

    # Save JSON
    json_path = output_dir / 'multi_algorithm_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    print("\n[4/4] Generating report...")
    lines = []
    lines.append("# Stage 2: Multi-Algorithm Validation Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits analyzed**: {len(circuit_results)}")
    lines.append(f"**Methods**: Louvain, Leiden, Greedy Modularity, Label Propagation")
    lines.append(f"**Leiden available**: {LEIDEN_AVAILABLE}\n")
    lines.append("---\n")

    lines.append("## Overall Results\n")
    lines.append(f"- **Mean Jaccard agreement**: {overall_avg:.3f}")
    lines.append(f"- **VALID** (Jaccard > 0.7): {n_valid} circuits ({100*n_valid/len(circuit_results):.0f}%)")
    lines.append(f"- **MODERATE** (0.5-0.7): {n_moderate} circuits ({100*n_moderate/len(circuit_results):.0f}%)")
    lines.append(f"- **WEAK** (< 0.5): {n_weak} circuits ({100*n_weak/len(circuit_results):.0f}%)")
    lines.append(f"- **Valid + Moderate**: {n_valid+n_moderate} circuits "
                 f"({results['overall']['pct_valid_or_moderate']:.0f}%)\n")

    lines.append("## Per-Model Breakdown\n")
    lines.append("| Model | N | Mean Jaccard | Std | Valid | Moderate | Weak |")
    lines.append("|-------|---|--------------|-----|-------|----------|------|")
    for model_name, ms in model_stats.items():
        lines.append(f"| {model_name} | {ms['n']} | {ms['mean_jaccard']:.3f} | "
                     f"{ms['std_jaccard']:.3f} | {ms['n_valid']} | {ms['n_moderate']} | {ms['n_weak']} |")
    lines.append("")

    lines.append("## Per-Category Breakdown\n")
    lines.append("| Category | N | Mean Jaccard | Valid | Moderate | Weak |")
    lines.append("|----------|---|--------------|-------|----------|------|")
    for cat, cs in cat_stats.items():
        lines.append(f"| {cat} | {cs['n']} | {cs['mean_jaccard']:.3f} | "
                     f"{cs['n_valid']} | {cs['n_moderate']} | {cs['n_weak']} |")
    lines.append("")

    lines.append("## Per-Model-Per-Category\n")
    lines.append("| Model | Category | N | Mean Jaccard | Valid | Moderate | Weak |")
    lines.append("|-------|----------|---|--------------|-------|----------|------|")
    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        for cat in ['chemistry', 'geography', 'history']:
            mc = model_cat_stats.get(model_name, {}).get(cat, {})
            if mc:
                lines.append(f"| {model_name} | {cat} | {mc['n']} | {mc['mean_jaccard']:.3f} | "
                             f"{mc['n_valid']} | {mc['n_moderate']} | {mc['n_weak']} |")
    lines.append("")

    lines.append("## Method Pair Agreement (averaged across all circuits)\n")
    lines.append("| Method Pair | Mean Jaccard |")
    lines.append("|-------------|-------------|")
    for pair, val in sorted(method_pair_means.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {pair} | {val:.3f} |")
    lines.append("")

    lines.append("## Average Modularity by Method and Model\n")
    lines.append("| Model | Louvain | Leiden | Greedy | Label Prop |")
    lines.append("|-------|---------|--------|--------|------------|")
    for model_name, ms in model_stats.items():
        mods = ms.get('avg_modularities', {})
        lines.append(f"| {model_name} | "
                     f"{mods.get('louvain', 0):.4f} | "
                     f"{mods.get('leiden', 0):.4f} | "
                     f"{mods.get('greedy', 0):.4f} | "
                     f"{mods.get('label_propagation', 0):.4f} |")
    lines.append("")

    lines.append("## Top 5 Highest Agreement Circuits\n")
    lines.append("| Circuit | Model | Category | Avg Jaccard | Quality |")
    lines.append("|---------|-------|----------|-------------|---------|")
    for r in reversed(top5):
        lines.append(f"| {r['prompt_slug'][:35]} | {r['model']} | {r['category']} | "
                     f"{r['avg_jaccard']:.3f} | {r['quality']} |")
    lines.append("")

    lines.append("## Bottom 5 Lowest Agreement Circuits\n")
    lines.append("| Circuit | Model | Category | Avg Jaccard | Quality |")
    lines.append("|---------|-------|----------|-------------|---------|")
    for r in bottom5:
        lines.append(f"| {r['prompt_slug'][:35]} | {r['model']} | {r['category']} | "
                     f"{r['avg_jaccard']:.3f} | {r['quality']} |")
    lines.append("")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    report_path = output_dir / 'multi_algorithm_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Mean Jaccard: {overall_avg:.3f}")
    print(f"  VALID: {n_valid}, MODERATE: {n_moderate}, WEAK: {n_weak}")
    print(f"  Valid+Moderate: {n_valid+n_moderate}/{len(circuit_results)} "
          f"({results['overall']['pct_valid_or_moderate']:.0f}%)")
    print("\nDone!")


if __name__ == '__main__':
    main()
