#!/usr/bin/env python3
"""Stage 2: Minimal Pathway Extraction at Scale.

For each of the 60 circuits, extracts the minimal subgraph connecting
input features to output features, measuring circuit redundancy and
identifying structural bottlenecks.

Adapts layer thresholds per model:
  GEMMA-2-2B (26 layers): input L0-5, output L21-25
  QWEN3-4B (36 layers):   input L0-7, output L29-35

Output: data/stage_2_minimal_pathways/minimal_pathway_results.json + report.md

Usage:
    python scripts/stage_2_minimal_pathways.py
"""

import json
import sys
import io
import math
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import networkx as nx

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_minimal_pathways'
DATA_TABLE = DATA_DIR / 'stage_2_statistical_deepdive' / 'data_table.json'

# Model-specific layer thresholds
LAYER_CONFIG = {
    'gemma-2-2b': {'total': 26, 'input_max': 5, 'output_min': 21},
    'qwen3-4b':   {'total': 36, 'input_max': 7, 'output_min': 29},
}


def load_graph(graph_path):
    """Load converted_graph.json into NetworkX DiGraph."""
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
    """Find converted_graph.json for a circuit."""
    prompt_dir = PROMPTS_DIR / circuit_id / '2_conversion'
    if not prompt_dir.exists():
        return None
    matches = list(prompt_dir.glob('*converted_graph.json'))
    return matches[0] if matches else None


def extract_minimal_pathway(G, model):
    """Extract minimal pathway from a circuit graph.

    Returns dict with metrics and minimal subgraph info.
    """
    cfg = LAYER_CONFIG.get(model, LAYER_CONFIG['gemma-2-2b'])
    input_max = cfg['input_max']
    output_min = cfg['output_min']

    n_nodes_orig = G.number_of_nodes()
    n_edges_orig = G.number_of_edges()

    # Identify input features (early layers, high out-degree)
    input_features = []
    for node in G.nodes():
        layer = G.nodes[node].get('layer', 999)
        if layer <= input_max and G.out_degree(node) > 0:
            input_features.append((node, G.out_degree(node), layer))
    input_features.sort(key=lambda x: x[1], reverse=True)
    input_features = input_features[:10]  # top 10

    # Identify output features (late layers, high in-degree)
    output_features = []
    for node in G.nodes():
        layer = G.nodes[node].get('layer', 0)
        if layer >= output_min and G.in_degree(node) > 0:
            output_features.append((node, G.in_degree(node), layer))
    output_features.sort(key=lambda x: x[1], reverse=True)
    output_features = output_features[:10]

    if not input_features or not output_features:
        return {
            'status': 'no_io_features',
            'n_input': len(input_features),
            'n_output': len(output_features),
        }

    # Find shortest paths between inputs and outputs
    sources = [n for n, _, _ in input_features]
    targets = [n for n, _, _ in output_features]

    all_paths = []
    for src in sources:
        for tgt in targets:
            try:
                gen = nx.shortest_simple_paths(G, src, tgt)
                count = 0
                for path in gen:
                    if count >= 3:
                        break
                    if len(path) <= 20:
                        all_paths.append(path)
                    count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if not all_paths:
        return {
            'status': 'no_paths',
            'n_input': len(input_features),
            'n_output': len(output_features),
        }

    # Extract essential edges (appear in >= 20% of paths)
    edge_counts = defaultdict(int)
    for path in all_paths:
        for i in range(len(path) - 1):
            edge_counts[(path[i], path[i+1])] += 1

    threshold = max(1, int(0.2 * len(all_paths)))
    essential_edges = {e for e, c in edge_counts.items() if c >= threshold}

    # Fallback: if no edges pass threshold, use top 50% of edges by frequency
    if not essential_edges:
        sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
        n_keep = max(1, len(sorted_edges) // 2)
        essential_edges = {e for e, c in sorted_edges[:n_keep]}

    # Build minimal subgraph
    minimal_nodes = set()
    for s, t in essential_edges:
        minimal_nodes.add(s)
        minimal_nodes.add(t)

    n_nodes_min = len(minimal_nodes)
    n_edges_min = len(essential_edges)
    reduction_pct = (1 - n_nodes_min / n_nodes_orig) * 100 if n_nodes_orig > 0 else 0

    # Layer distribution of minimal circuit
    layer_counts = defaultdict(int)
    for node in minimal_nodes:
        layer = G.nodes[node].get('layer', -1)
        layer_counts[layer] += 1

    # Bottleneck = layer with fewest nodes in minimal circuit
    if layer_counts:
        bottleneck_layer = min(layer_counts, key=layer_counts.get)
        bottleneck_width = layer_counts[bottleneck_layer]
    else:
        bottleneck_layer = -1
        bottleneck_width = 0

    # Path length stats
    path_lengths = [len(p) for p in all_paths]
    avg_path_len = sum(path_lengths) / len(path_lengths)
    unique_paths = len(set(tuple(p) for p in all_paths))

    # Edge weight in minimal vs full
    minimal_weight = sum(G[s][t].get('weight', 0) for s, t in essential_edges
                         if G.has_edge(s, t))
    total_weight = sum(d.get('weight', 0) for _, _, d in G.edges(data=True))
    weight_retention = minimal_weight / total_weight if total_weight > 0 else 0

    return {
        'status': 'ok',
        'original_nodes': n_nodes_orig,
        'original_edges': n_edges_orig,
        'minimal_nodes': n_nodes_min,
        'minimal_edges': n_edges_min,
        'reduction_pct': reduction_pct,
        'bottleneck_layer': bottleneck_layer,
        'bottleneck_width': bottleneck_width,
        'n_paths_found': len(all_paths),
        'unique_paths': unique_paths,
        'avg_path_length': avg_path_len,
        'weight_retention': weight_retention,
        'n_input': len(input_features),
        'n_output': len(output_features),
        'layers_in_minimal': dict(layer_counts),
    }


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: MINIMAL PATHWAY EXTRACTION AT SCALE")
    print("=" * 70)

    # Load data table
    print("\n[1/3] Loading circuit metadata...")
    with open(DATA_TABLE, 'r', encoding='utf-8') as f:
        data_table = json.load(f)
    print(f"  {len(data_table)} circuits")

    # Process each circuit
    print("\n[2/3] Extracting minimal pathways...")
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

        try:
            metrics = extract_minimal_pathway(G, model)
        except Exception as e:
            errors.append(f"Extraction error {cid}: {e}")
            continue

        metrics['circuit_id'] = cid
        metrics['model'] = model
        metrics['category'] = category
        metrics['prompt_slug'] = slug
        circuit_results.append(metrics)

        if metrics['status'] == 'ok':
            tag = f"R={metrics['reduction_pct']:.0f}% BN=L{metrics['bottleneck_layer']}w{metrics['bottleneck_width']}"
        else:
            tag = metrics['status']
        print(f"  [{idx+1:2d}/60] {cid[:45]:45s} {tag}")

    # Aggregate
    print(f"\n[3/3] Aggregating results...")
    ok_results = [r for r in circuit_results if r['status'] == 'ok']
    print(f"  Successful: {len(ok_results)}, Failed: {len(circuit_results) - len(ok_results)}, Errors: {len(errors)}")

    if not ok_results:
        print("  No successful extractions!")
        return

    # Overall stats
    reductions = [r['reduction_pct'] for r in ok_results]
    bn_widths = [r['bottleneck_width'] for r in ok_results]
    path_lengths = [r['avg_path_length'] for r in ok_results]
    weight_rets = [r['weight_retention'] for r in ok_results]
    unique_paths_list = [r['unique_paths'] for r in ok_results]

    def mean(lst): return sum(lst) / len(lst) if lst else 0
    def std(lst):
        m = mean(lst)
        return math.sqrt(sum((x - m)**2 for x in lst) / len(lst)) if len(lst) > 1 else 0

    print(f"\n  OVERALL ({len(ok_results)} circuits):")
    print(f"    Reduction: {mean(reductions):.1f}% +/- {std(reductions):.1f}%")
    print(f"    Bottleneck width: {mean(bn_widths):.1f} +/- {std(bn_widths):.1f}")
    print(f"    Avg path length: {mean(path_lengths):.1f} +/- {std(path_lengths):.1f}")
    print(f"    Weight retention: {mean(weight_rets):.3f} +/- {std(weight_rets):.3f}")
    print(f"    Unique paths: {mean(unique_paths_list):.0f} +/- {std(unique_paths_list):.0f}")

    # Per-model
    model_stats = {}
    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        mr = [r for r in ok_results if r['model'] == model_name]
        if not mr:
            continue
        red = [r['reduction_pct'] for r in mr]
        bw = [r['bottleneck_width'] for r in mr]
        pl = [r['avg_path_length'] for r in mr]
        wr = [r['weight_retention'] for r in mr]
        up = [r['unique_paths'] for r in mr]
        bn_layers = [r['bottleneck_layer'] for r in mr]

        model_stats[model_name] = {
            'n': len(mr),
            'mean_reduction': mean(red),
            'std_reduction': std(red),
            'mean_bn_width': mean(bw),
            'mean_path_length': mean(pl),
            'mean_weight_retention': mean(wr),
            'mean_unique_paths': mean(up),
            'median_bn_layer': sorted(bn_layers)[len(bn_layers)//2],
        }
        print(f"\n  {model_name.upper()} ({len(mr)}):")
        print(f"    Reduction: {mean(red):.1f}% +/- {std(red):.1f}%")
        print(f"    Bottleneck width: {mean(bw):.1f}, median BN layer: {sorted(bn_layers)[len(bn_layers)//2]}")
        print(f"    Path length: {mean(pl):.1f}, weight retention: {mean(wr):.3f}")

    # Per-category
    cat_stats = {}
    for cat in ['chemistry', 'geography', 'history']:
        cr = [r for r in ok_results if r['category'] == cat]
        if not cr:
            continue
        red = [r['reduction_pct'] for r in cr]
        cat_stats[cat] = {
            'n': len(cr),
            'mean_reduction': mean(red),
            'std_reduction': std(red),
        }
        print(f"\n  {cat.upper()} ({len(cr)}): reduction={mean(red):.1f}% +/- {std(red):.1f}%")

    # Correlation: reduction vs confidence
    from_table = {}
    for row in data_table:
        from_table[row['circuit_id']] = row

    paired = []
    for r in ok_results:
        tbl = from_table.get(r['circuit_id'], {})
        conf = tbl.get('output_probability')
        if conf is not None:
            paired.append((r['reduction_pct'], r['weight_retention'],
                           r['bottleneck_width'], conf, r['model']))

    if len(paired) >= 5:
        # Pearson r: reduction vs confidence
        for model_name in ['gemma-2-2b', 'qwen3-4b']:
            mp = [(red, wr, bw, conf) for red, wr, bw, conf, m in paired if m == model_name]
            if len(mp) < 5:
                continue
            reds = [x[0] for x in mp]
            confs = [x[3] for x in mp]
            # Reduction vs confidence
            n = len(reds)
            mr_ = mean(reds)
            mc_ = mean(confs)
            num = sum((reds[i] - mr_) * (confs[i] - mc_) for i in range(n))
            d1 = math.sqrt(sum((reds[i] - mr_)**2 for i in range(n)))
            d2 = math.sqrt(sum((confs[i] - mc_)**2 for i in range(n)))
            r_val = num / (d1 * d2) if d1 > 0 and d2 > 0 else 0
            # t-test
            if abs(r_val) < 1:
                t_stat = r_val * math.sqrt((n-2) / (1 - r_val**2))
            else:
                t_stat = float('inf')
            print(f"\n  {model_name.upper()}: reduction vs confidence r={r_val:.3f} (t={t_stat:.2f}, n={n})")

            # Weight retention vs confidence
            wrs = [x[1] for x in mp]
            mw_ = mean(wrs)
            num2 = sum((wrs[i] - mw_) * (confs[i] - mc_) for i in range(n))
            d3 = math.sqrt(sum((wrs[i] - mw_)**2 for i in range(n)))
            r_wr = num2 / (d3 * d2) if d3 > 0 and d2 > 0 else 0
            print(f"             weight_retention vs confidence r={r_wr:.3f}")

    # Build results
    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_circuits': len(circuit_results),
            'n_ok': len(ok_results),
            'n_errors': len(errors),
        },
        'overall': {
            'mean_reduction': mean(reductions),
            'std_reduction': std(reductions),
            'mean_bn_width': mean(bn_widths),
            'mean_path_length': mean(path_lengths),
            'mean_weight_retention': mean(weight_rets),
            'mean_unique_paths': mean(unique_paths_list),
        },
        'model_stats': model_stats,
        'category_stats': cat_stats,
        'circuit_results': circuit_results,
        'errors': errors,
    }

    # Save JSON
    json_path = output_dir / 'minimal_pathway_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Minimal Pathway Extraction Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits extracted**: {len(ok_results)}/{len(circuit_results)}\n")
    lines.append("---\n")

    lines.append("## Overall Results\n")
    lines.append(f"- **Mean reduction**: {mean(reductions):.1f}% (+/- {std(reductions):.1f}%)")
    lines.append(f"- **Mean bottleneck width**: {mean(bn_widths):.1f}")
    lines.append(f"- **Mean path length**: {mean(path_lengths):.1f}")
    lines.append(f"- **Mean weight retention**: {mean(weight_rets):.3f} "
                 f"({mean(weight_rets)*100:.1f}% of total weight)")
    lines.append(f"- **Mean unique paths**: {mean(unique_paths_list):.0f}\n")

    lines.append("## Per-Model Breakdown\n")
    lines.append("| Model | N | Reduction% | BN Width | Path Length | Weight Retention | Median BN Layer |")
    lines.append("|-------|---|-----------|----------|-------------|-----------------|-----------------|")
    for model_name, ms in model_stats.items():
        lines.append(f"| {model_name} | {ms['n']} | {ms['mean_reduction']:.1f}% | "
                     f"{ms['mean_bn_width']:.1f} | {ms['mean_path_length']:.1f} | "
                     f"{ms['mean_weight_retention']:.3f} | L{ms['median_bn_layer']} |")
    lines.append("")

    lines.append("## Per-Category Breakdown\n")
    lines.append("| Category | N | Mean Reduction% | Std |")
    lines.append("|----------|---|----------------|-----|")
    for cat, cs in cat_stats.items():
        lines.append(f"| {cat} | {cs['n']} | {cs['mean_reduction']:.1f}% | {cs['std_reduction']:.1f}% |")
    lines.append("")

    lines.append("## Individual Circuit Results\n")
    lines.append("| Circuit | Model | Category | Reduction% | BN Layer | BN Width | Paths | Weight Ret |")
    lines.append("|---------|-------|----------|-----------|----------|----------|-------|------------|")
    for r in sorted(ok_results, key=lambda x: x['reduction_pct'], reverse=True):
        lines.append(f"| {r['prompt_slug'][:30]} | {r['model']} | {r['category']} | "
                     f"{r['reduction_pct']:.0f}% | L{r['bottleneck_layer']} | "
                     f"{r['bottleneck_width']} | {r['unique_paths']} | {r['weight_retention']:.3f} |")
    lines.append("")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- {e}")

    report_path = output_dir / 'minimal_pathway_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Reduction: {mean(reductions):.1f}% +/- {std(reductions):.1f}%")
    print(f"  Weight retention: {mean(weight_rets):.3f}")
    print(f"  Bottleneck width: {mean(bn_widths):.1f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
