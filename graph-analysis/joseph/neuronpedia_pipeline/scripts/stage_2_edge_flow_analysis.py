#!/usr/bin/env python3
"""Stage 2: Edge Flow Analysis - layer-to-layer connectivity patterns across 60 circuits.

Analyzes edge flow patterns across 60 circuits (30 prompts x 2 models x 3 domains).
Reads converted_graph.json files and computes layer-to-layer connectivity metrics.

Analyses:
  1. Layer-to-layer edge weight matrix: M[i][j] = sum |weight| from layer i to layer j
  2. Skip connection detection: edges spanning 2+ layers vs adjacent-only
  3. Per-layer flow analysis: incoming vs outgoing edge weight, forward flow ratio
  4. Aggregation: averages per model per category, architecture-invariance test

Output: data/stage_2_edge_flow/edge_flow_results.json + edge_flow_report.md

Usage:
    python scripts/stage_2_edge_flow_analysis.py
    python scripts/stage_2_edge_flow_analysis.py --output-dir data/stage_2_edge_flow
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
from typing import Dict, List, Optional, Tuple

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
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_edge_flow'

MODELS = ['gemma-2-2b', 'qwen3-4b']
MODEL_LAYERS = {'gemma-2-2b': 26, 'qwen3-4b': 36}

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


# ==========================================================================
# Data Loading
# ==========================================================================

def load_converted_graph(model: str, prompt_slug: str) -> Optional[dict]:
    """Load converted_graph.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    conv_dir = PROMPTS_DIR / dir_name / '2_conversion'
    if not conv_dir.exists():
        return None
    matches = list(conv_dir.glob('*converted_graph.json'))
    if not matches:
        return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================================================
# Core Computation
# ==========================================================================

def compute_edge_flow(graph_data: dict, num_layers: int) -> Optional[dict]:
    """Compute edge flow metrics for a single circuit.

    Returns dict with layer_matrix, skip_stats, per_layer_flow, summary.
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    if not nodes or not edges:
        return None

    # Build node_id -> layer lookup from nodes array
    node_layer = {}
    for node in nodes:
        node_layer[node['id']] = node.get('layer', 0)

    # Initialize accumulators
    layer_matrix = [[0.0] * num_layers for _ in range(num_layers)]
    layer_incoming = [0.0] * num_layers
    layer_outgoing = [0.0] * num_layers
    total_edges = 0
    adjacent_edges = 0
    skip_edges = 0
    skip_spans = []
    skipped = 0

    for edge in edges:
        src_id = edge.get('source', '')
        tgt_id = edge.get('target', '')
        weight = abs(edge.get('weight', 0.0))

        # Lookup layers
        if src_id in node_layer:
            src_l = node_layer[src_id]
        else:
            try:
                src_l = int(src_id.split('_')[0])
            except (ValueError, IndexError):
                skipped += 1
                continue

        if tgt_id in node_layer:
            tgt_l = node_layer[tgt_id]
        else:
            try:
                tgt_l = int(tgt_id.split('_')[0])
            except (ValueError, IndexError):
                skipped += 1
                continue

        src_l = max(0, min(src_l, num_layers - 1))
        tgt_l = max(0, min(tgt_l, num_layers - 1))

        layer_matrix[src_l][tgt_l] += weight
        layer_outgoing[src_l] += weight
        layer_incoming[tgt_l] += weight
        total_edges += 1

        span = abs(tgt_l - src_l)
        if span <= 1:
            adjacent_edges += 1
        else:
            skip_edges += 1
            skip_spans.append((span, weight))

    if total_edges == 0:
        return None

    # Skip stats
    total_weight = sum(layer_outgoing)
    skip_weight = sum(s[1] for s in skip_spans)

    skip_stats = {
        'total_edges': total_edges,
        'adjacent_edges': adjacent_edges,
        'skip_edges': skip_edges,
        'skip_ratio': skip_edges / total_edges,
        'avg_skip_span': sum(s[0] for s in skip_spans) / len(skip_spans) if skip_spans else 0.0,
        'max_skip_span': max((s[0] for s in skip_spans), default=0),
        'skip_weight_fraction': skip_weight / total_weight if total_weight > 0 else 0.0,
    }

    # Per-layer flow
    per_layer_flow = []
    for i in range(num_layers):
        inc = layer_incoming[i]
        out = layer_outgoing[i]
        total = inc + out
        per_layer_flow.append({
            'layer': i,
            'incoming': inc,
            'outgoing': out,
            'forward_flow_ratio': out / total if total > 0 else 0.0,
        })

    # Top layer pairs
    top_pairs = []
    for i in range(num_layers):
        for j in range(num_layers):
            if layer_matrix[i][j] > 0:
                top_pairs.append({'src': i, 'tgt': j, 'weight': layer_matrix[i][j], 'span': abs(j - i)})
    top_pairs.sort(key=lambda x: x['weight'], reverse=True)

    return {
        'layer_matrix': layer_matrix,
        'skip_stats': skip_stats,
        'per_layer_flow': per_layer_flow,
        'total_weight': total_weight,
        'top_10_pairs': top_pairs[:10],
    }


# ==========================================================================
# Aggregation
# ==========================================================================

def avg_matrices(flows: List[dict], num_layers: int) -> List[List[float]]:
    """Element-wise average of layer matrices."""
    n = len(flows)
    avg = [[0.0] * num_layers for _ in range(num_layers)]
    for f in flows:
        m = f['layer_matrix']
        for i in range(num_layers):
            for j in range(num_layers):
                avg[i][j] += m[i][j]
    for i in range(num_layers):
        for j in range(num_layers):
            avg[i][j] /= n
    return avg


def avg_skip(flows: List[dict]) -> dict:
    """Average skip stats."""
    n = len(flows)
    keys = ['total_edges', 'adjacent_edges', 'skip_edges', 'skip_ratio',
            'avg_skip_span', 'max_skip_span', 'skip_weight_fraction']
    agg = {k: 0.0 for k in keys}
    for f in flows:
        for k in keys:
            agg[k] += f['skip_stats'].get(k, 0.0)
    return {k: v / n for k, v in agg.items()}


def avg_flow(flows: List[dict], num_layers: int) -> List[dict]:
    """Average per-layer flow."""
    n = len(flows)
    inc = [0.0] * num_layers
    out = [0.0] * num_layers
    ffr = [0.0] * num_layers
    for f in flows:
        for e in f['per_layer_flow']:
            l = e['layer']
            if l < num_layers:
                inc[l] += e['incoming']
                out[l] += e['outgoing']
                ffr[l] += e['forward_flow_ratio']
    return [{'layer': i, 'avg_incoming': inc[i]/n, 'avg_outgoing': out[i]/n,
             'avg_ffr': ffr[i]/n} for i in range(num_layers)]


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def flow_profile(flows: List[dict], num_layers: int) -> List[float]:
    """Get averaged forward-flow-ratio profile."""
    n = len(flows)
    ffr = [0.0] * num_layers
    for f in flows:
        for e in f['per_layer_flow']:
            l = e['layer']
            if l < num_layers:
                ffr[l] += e['forward_flow_ratio']
    return [v / n for v in ffr]


# ==========================================================================
# Main Analysis
# ==========================================================================

def analyze_all(output_dir: Path) -> dict:
    """Main analysis loop over all 60 circuits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: EDGE FLOW ANALYSIS")
    print("=" * 70)

    # Step 1: Load and compute per-circuit
    print("\n[1/4] Loading circuits...")
    per_circuit = {}
    by_model_cat = {m: {c: [] for c in CATEGORIES} for m in MODELS}
    loaded = 0
    missing = 0

    for model in MODELS:
        nl = MODEL_LAYERS[model]
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                graph = load_converted_graph(model, slug)
                if graph is None:
                    print(f"  [WARN] Missing: {model}/{slug}")
                    missing += 1
                    continue
                flow = compute_edge_flow(graph, nl)
                if flow is None:
                    missing += 1
                    continue
                flow['model'] = model
                flow['slug'] = slug
                flow['category'] = cat
                flow['confidence'] = graph.get('metadata', {}).get('output_probability', 0.0)
                cid = f"{model}_{slug}"
                per_circuit[cid] = flow
                by_model_cat[model][cat].append(flow)
                loaded += 1

    print(f"  Loaded: {loaded}, Missing: {missing}")

    # Step 2: Aggregate per model per category
    print("\n[2/4] Aggregating per model/category...")
    agg_mc = {}
    for model in MODELS:
        nl = MODEL_LAYERS[model]
        agg_mc[model] = {}
        for cat in CATEGORIES:
            flows = by_model_cat[model][cat]
            if not flows:
                continue
            agg_mc[model][cat] = {
                'n': len(flows),
                'avg_skip': avg_skip(flows),
                'avg_flow': avg_flow(flows, nl),
            }
            print(f"  {model}/{cat}: {len(flows)} circuits, "
                  f"skip_ratio={agg_mc[model][cat]['avg_skip']['skip_ratio']:.3f}")

    # Step 3: Per-model overall
    print("\n[3/4] Per-model overall...")
    model_all = {}
    for model in MODELS:
        nl = MODEL_LAYERS[model]
        all_flows = []
        for cat_flows in by_model_cat[model].values():
            all_flows.extend(cat_flows)
        if not all_flows:
            continue
        model_all[model] = {
            'n': len(all_flows),
            'avg_skip': avg_skip(all_flows),
            'avg_flow': avg_flow(all_flows, nl),
            'avg_matrix': avg_matrices(all_flows, nl),
        }
        ss = model_all[model]['avg_skip']
        print(f"  {model}: {len(all_flows)} circuits, skip_ratio={ss['skip_ratio']:.3f}, "
              f"skip_weight_frac={ss['skip_weight_fraction']:.3f}")

    # Step 4: Architecture invariance test
    print("\n[4/4] Architecture invariance test...")
    invariance = {}
    for model in MODELS:
        nl = MODEL_LAYERS[model]
        cats = sorted(CATEGORIES.keys())
        cross_sims = {}
        all_sims = []
        for i in range(len(cats)):
            for j in range(i+1, len(cats)):
                fa = by_model_cat[model][cats[i]]
                fb = by_model_cat[model][cats[j]]
                if fa and fb:
                    pa = flow_profile(fa, nl)
                    pb = flow_profile(fb, nl)
                    sim = cosine_sim(pa, pb)
                    key = f"{cats[i]} vs {cats[j]}"
                    cross_sims[key] = sim
                    all_sims.append(sim)

        # Within-category split-half
        within_sims = {}
        for cat in cats:
            flows = by_model_cat[model][cat]
            if len(flows) >= 4:
                half = len(flows) // 2
                pa = flow_profile(flows[:half], nl)
                pb = flow_profile(flows[half:], nl)
                within_sims[cat] = cosine_sim(pa, pb)

        avg_cross = sum(all_sims) / len(all_sims) if all_sims else 0.0
        avg_within = sum(within_sims.values()) / len(within_sims) if within_sims else 0.0
        gap = avg_within - avg_cross

        invariance[model] = {
            'cross_category': cross_sims,
            'within_category': within_sims,
            'avg_cross': avg_cross,
            'avg_within': avg_within,
            'gap': gap,
            'invariant': gap < 0.05,
        }
        print(f"  {model}: cross={avg_cross:.4f}, within={avg_within:.4f}, "
              f"gap={gap:.4f} -> {'INVARIANT' if gap < 0.05 else 'DOMAIN-DEPENDENT'}")

    # Build results (strip large matrices from per-circuit for JSON size)
    per_circuit_slim = {}
    for cid, flow in per_circuit.items():
        slim = {k: v for k, v in flow.items() if k != 'layer_matrix'}
        per_circuit_slim[cid] = slim

    # Extract top pairs from model-level matrices
    model_top_pairs = {}
    for model, data in model_all.items():
        nl = MODEL_LAYERS[model]
        pairs = []
        m = data['avg_matrix']
        for i in range(nl):
            for j in range(nl):
                if m[i][j] > 0:
                    pairs.append({'src': i, 'tgt': j, 'weight': m[i][j], 'span': abs(j-i)})
        pairs.sort(key=lambda x: x['weight'], reverse=True)
        model_top_pairs[model] = pairs[:20]

    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'loaded': loaded,
            'missing': missing,
            'models': MODELS,
        },
        'per_circuit': per_circuit_slim,
        'by_model_category': {
            model: {cat: agg for cat, agg in cats.items()}
            for model, cats in agg_mc.items()
        },
        'model_overall': {
            model: {k: v for k, v in data.items() if k != 'avg_matrix'}
            for model, data in model_all.items()
        },
        'model_top_pairs': model_top_pairs,
        'invariance': invariance,
    }

    return results


# ==========================================================================
# Report
# ==========================================================================

def generate_report(results: dict, output_dir: Path):
    """Write markdown report."""
    lines = []
    lines.append("# Stage 2: Edge Flow Analysis Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits**: {results['metadata']['loaded']} loaded, "
                 f"{results['metadata']['missing']} missing\n")
    lines.append("---\n")

    # Section 1: Skip connections
    lines.append("## 1. Skip Connection Summary\n")
    lines.append("| Model | Circuits | Avg Edges | Skip Ratio | Avg Skip Span | "
                 "Skip Weight Frac |")
    lines.append("|-------|----------|-----------|------------|---------------|"
                 "------------------|")
    for model in MODELS:
        ov = results.get('model_overall', {}).get(model, {})
        if not ov:
            continue
        ss = ov['avg_skip']
        lines.append(f"| {model} | {ov['n']} | {ss['total_edges']:.0f} | "
                     f"{ss['skip_ratio']:.3f} | {ss['avg_skip_span']:.2f} | "
                     f"{ss['skip_weight_fraction']:.3f} |")
    lines.append("")

    # Per category
    lines.append("### Per-Category Breakdown\n")
    lines.append("| Model | Category | N | Skip Ratio | Skip Weight Frac |")
    lines.append("|-------|----------|---|------------|------------------|")
    for model in MODELS:
        mc = results.get('by_model_category', {}).get(model, {})
        for cat in sorted(CATEGORIES.keys()):
            data = mc.get(cat, {})
            if not data:
                continue
            ss = data['avg_skip']
            lines.append(f"| {model} | {cat} | {data['n']} | "
                         f"{ss['skip_ratio']:.3f} | {ss['skip_weight_fraction']:.3f} |")
    lines.append("")

    # Section 2: Per-layer flow
    lines.append("## 2. Per-Layer Flow Profiles\n")
    for model in MODELS:
        ov = results.get('model_overall', {}).get(model, {})
        if not ov:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append("| Layer | Avg Incoming | Avg Outgoing | Forward Flow Ratio |")
        lines.append("|-------|--------------|--------------|--------------------|")
        for e in ov['avg_flow']:
            if e['avg_incoming'] > 0.01 or e['avg_outgoing'] > 0.01:
                lines.append(f"| L{e['layer']} | {e['avg_incoming']:.2f} | "
                             f"{e['avg_outgoing']:.2f} | {e['avg_ffr']:.3f} |")
        lines.append("")

    # Section 3: Top layer pairs
    lines.append("## 3. Dominant Layer-to-Layer Connections\n")
    for model in MODELS:
        pairs = results.get('model_top_pairs', {}).get(model, [])
        if not pairs:
            continue
        lines.append(f"### {model.upper()} (Top 15)\n")
        lines.append("| Rank | Source | Target | Span | Avg Weight |")
        lines.append("|------|--------|--------|------|------------|")
        for rank, p in enumerate(pairs[:15], 1):
            lines.append(f"| {rank} | L{p['src']} | L{p['tgt']} | {p['span']} | "
                         f"{p['weight']:.2f} |")
        lines.append("")

    # Section 4: Invariance
    lines.append("## 4. Architecture Invariance Test\n")
    lines.append("Tests whether edge flow patterns are determined by architecture "
                 "(similar across domains) rather than knowledge domain.\n")
    inv = results.get('invariance', {})
    for model in MODELS:
        mi = inv.get(model, {})
        if not mi:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append(f"- **Avg cross-category similarity**: {mi['avg_cross']:.4f}")
        lines.append(f"- **Avg within-category similarity**: {mi['avg_within']:.4f}")
        lines.append(f"- **Gap**: {mi['gap']:.4f}")
        lines.append(f"- **Result**: {'ARCHITECTURE-INVARIANT' if mi['invariant'] else 'DOMAIN-DEPENDENT'}\n")

        lines.append("| Pair | Cosine Similarity |")
        lines.append("|------|-------------------|")
        for pair, sim in mi.get('cross_category', {}).items():
            lines.append(f"| {pair} | {sim:.4f} |")
        lines.append("")
        if mi.get('within_category'):
            lines.append("| Category | Split-Half Similarity |")
            lines.append("|----------|----------------------|")
            for cat, sim in mi['within_category'].items():
                lines.append(f"| {cat} | {sim:.4f} |")
            lines.append("")

    # Section 5: Key findings
    lines.append("## 5. Key Findings\n")
    for model in MODELS:
        ov = results.get('model_overall', {}).get(model, {})
        mi = inv.get(model, {})
        if not ov:
            continue
        ss = ov['avg_skip']
        lines.append(f"### {model.upper()}\n")
        lines.append(f"- **{ss['skip_ratio']:.1%}** of edges are skip connections (span 2+ layers)")
        lines.append(f"- **{ss['skip_weight_fraction']:.1%}** of total weight carried by skip connections")
        lines.append(f"- Average skip span: **{ss['avg_skip_span']:.2f}** layers")
        if mi:
            status = 'architecture-invariant' if mi['invariant'] else 'domain-dependent'
            lines.append(f"- Flow patterns are **{status}** (gap={mi['gap']:.4f})")
        lines.append("")

    report = '\n'.join(lines)
    path = output_dir / 'edge_flow_report.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Edge Flow Analysis')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = analyze_all(output_dir)

    # Save JSON
    json_path = output_dir / 'edge_flow_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    generate_report(results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    for model in MODELS:
        ov = results.get('model_overall', {}).get(model, {})
        if not ov:
            continue
        ss = ov['avg_skip']
        mi = results.get('invariance', {}).get(model, {})
        print(f"\n  {model.upper()} ({ov['n']} circuits):")
        print(f"    Skip ratio:       {ss['skip_ratio']:.3f}")
        print(f"    Skip weight frac: {ss['skip_weight_fraction']:.3f}")
        print(f"    Avg skip span:    {ss['avg_skip_span']:.2f}")
        if mi:
            print(f"    Invariant:        {'YES' if mi['invariant'] else 'NO'} (gap={mi['gap']:.4f})")
    print("\nDone!")


if __name__ == '__main__':
    main()
