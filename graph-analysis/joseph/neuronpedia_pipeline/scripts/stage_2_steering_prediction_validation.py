#!/usr/bin/env python3
"""Stage 2: Steering Prediction Validation.

Compares graph-structural metrics (betweenness centrality, activation,
influence, degree) with actual steering effect sizes from the 20
Neuronpedia steering experiments.

Key question: Can we predict which features will have the strongest
causal effect from graph topology alone?

Output: data/stage_2_steering_validation/steering_validation_results.json + report.md

Usage:
    python scripts/stage_2_steering_prediction_validation.py
"""

import json
import sys
import io
import math
import datetime
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import networkx as nx

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_steering_validation'
STEERING_FILE = DATA_DIR / 'stage_3_steering' / 'steering_results.json'


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


def compute_steering_effect(result):
    """Compute the steering effect magnitude from logprob data.

    Returns:
        dict with effect metrics:
          - kl_divergence: KL(steered || default) over first 5 tokens
          - mean_logprob_shift: mean |steered_lp - default_lp|
          - max_logprob_shift: max |steered_lp - default_lp|
          - token_change_count: how many of top-1 tokens changed
          - text_changed: boolean whether generated text differs
    """
    resp = result.get('response', {})
    steered_lps = resp.get('steeredLogProbs', [])
    default_lps = resp.get('defaultLogProbs', [])

    if not steered_lps or not default_lps:
        return None

    n = min(len(steered_lps), len(default_lps), 5)  # first 5 tokens

    shifts = []
    token_changes = 0
    kl_sum = 0

    for i in range(n):
        s_lp = steered_lps[i].get('logprob', 0)
        d_lp = default_lps[i].get('logprob', 0)
        s_tok = steered_lps[i].get('token', '')
        d_tok = default_lps[i].get('token', '')

        shift = abs(s_lp - d_lp)
        shifts.append(shift)

        if s_tok != d_tok:
            token_changes += 1

        # KL contribution (approximate): p_default * (log p_default - log p_steered)
        # We have log probs directly
        p_d = math.exp(d_lp)
        p_s = math.exp(s_lp) if s_lp > -20 else 1e-9
        if p_d > 0 and p_s > 0:
            kl_sum += p_d * (d_lp - s_lp)

    steered_text = resp.get('STEERED', '')
    default_text = resp.get('DEFAULT', '')
    text_changed = steered_text != default_text

    return {
        'mean_logprob_shift': sum(shifts) / len(shifts) if shifts else 0,
        'max_logprob_shift': max(shifts) if shifts else 0,
        'kl_divergence': kl_sum,
        'token_change_count': token_changes,
        'text_changed': text_changed,
        'n_tokens_compared': n,
    }


def find_feature_in_graph(G, feature_label):
    """Find nodes in graph matching a feature label (L{layer}_F{id}).

    Feature label format: L3_F5150441
    Node ID format: {layer}_{position}_{ctx_idx}

    We match by layer and check if any node on that layer has
    matching circuit_tracer_id (position) mapping to the feature's NP ID.
    Since NP ID = circuit_tracer_id % 16384, we need to find nodes where
    position % 16384 matches the feature's NP ID.
    """
    parts = feature_label.split('_')
    if len(parts) != 2:
        return []

    target_layer = int(parts[0][1:])  # "L3" -> 3
    target_ct_id = int(parts[1][1:])  # "F5150441" -> 5150441
    target_np_id = target_ct_id % 16384

    matching_nodes = []
    for node in G.nodes():
        node_parts = node.split('_')
        if len(node_parts) >= 2:
            try:
                node_layer = int(node_parts[0])
                node_pos = int(node_parts[1])
                if node_layer == target_layer and (node_pos % 16384) == target_np_id:
                    matching_nodes.append(node)
            except ValueError:
                continue

    return matching_nodes


def compute_graph_metrics_for_feature(G, feature_nodes):
    """Compute graph-structural metrics for a set of feature nodes.

    Returns dict with:
        - betweenness: mean betweenness centrality
        - in_degree: mean in-degree
        - out_degree: mean out-degree
        - activation: mean activation attribute
        - influence: mean influence attribute
        - weighted_degree: sum of adjacent edge weights
        - pagerank: mean PageRank score
    """
    if not feature_nodes:
        return None

    # Betweenness (expensive but we cache)
    # Use undirected for betweenness since attribution graphs are DAGs
    G_u = G.to_undirected()

    try:
        bc = nx.betweenness_centrality(G_u, weight='weight')
    except Exception:
        bc = {n: 0 for n in G.nodes()}

    try:
        pr = nx.pagerank(G, weight='weight', max_iter=50)
    except Exception:
        pr = {n: 1.0/G.number_of_nodes() for n in G.nodes()}

    betweenness_vals = [bc.get(n, 0) for n in feature_nodes]
    in_deg = [G.in_degree(n) for n in feature_nodes]
    out_deg = [G.out_degree(n) for n in feature_nodes]
    activation_vals = [G.nodes[n].get('activation', 0) for n in feature_nodes]
    influence_vals = [G.nodes[n].get('influence', 0) for n in feature_nodes]
    pagerank_vals = [pr.get(n, 0) for n in feature_nodes]

    # Weighted degree: sum of all adjacent edge weights
    w_deg = []
    for n in feature_nodes:
        w = sum(G[u][v].get('weight', 0) for u, v in G.in_edges(n))
        w += sum(G[u][v].get('weight', 0) for u, v in G.out_edges(n))
        w_deg.append(w)

    def mean(lst): return sum(lst) / len(lst) if lst else 0

    return {
        'n_nodes_matched': len(feature_nodes),
        'betweenness': mean(betweenness_vals),
        'in_degree': mean(in_deg),
        'out_degree': mean(out_deg),
        'activation': mean(activation_vals),
        'influence': mean(influence_vals),
        'pagerank': mean(pagerank_vals),
        'weighted_degree': mean(w_deg),
    }


def pearson_r(x, y):
    """Compute Pearson correlation with p-value approximation."""
    n = len(x)
    if n < 3:
        return 0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    d1 = math.sqrt(sum((x[i] - mx)**2 for i in range(n)))
    d2 = math.sqrt(sum((y[i] - my)**2 for i in range(n)))
    if d1 == 0 or d2 == 0:
        return 0, 1.0
    r = num / (d1 * d2)
    if abs(r) >= 1:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r**2))
    # Approximate p-value from t distribution
    df = n - 2
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: STEERING PREDICTION VALIDATION")
    print("=" * 70)

    # Load steering results
    print("\n[1/4] Loading steering results...")
    with open(STEERING_FILE, 'r', encoding='utf-8') as f:
        steering_data = json.load(f)

    experiments = steering_data.get('results', [])
    print(f"  {len(experiments)} experiments")

    # Group by circuit to avoid reloading graphs
    circuit_groups = defaultdict(list)
    for exp in experiments:
        circuit_groups[exp['circuit_id']].append(exp)

    print(f"  {len(circuit_groups)} unique circuits")

    # Process each circuit
    print("\n[2/4] Computing graph metrics for steered features...")
    enriched_results = []
    graph_cache = {}

    for cid, exps in circuit_groups.items():
        graph_path = find_graph_file(cid)
        if graph_path is None:
            print(f"  [SKIP] {cid}: graph not found")
            continue

        if cid not in graph_cache:
            try:
                G = load_graph(graph_path)
                graph_cache[cid] = G
            except Exception as e:
                print(f"  [ERROR] {cid}: {e}")
                continue
        else:
            G = graph_cache[cid]

        for exp in exps:
            feature_label = exp['feature_label']
            strength = exp['strength']

            # Compute steering effect
            effect = compute_steering_effect(exp)
            if effect is None:
                continue

            # Find feature in graph
            feature_nodes = find_feature_in_graph(G, feature_label)

            # Compute graph metrics
            graph_metrics = compute_graph_metrics_for_feature(G, feature_nodes)

            enriched = {
                'feature_label': feature_label,
                'circuit_id': cid,
                'strength': strength,
                'layer': exp.get('layer', -1),
                'circuits_appeared_in': exp.get('circuits_appeared_in', 0),
                'avg_convergence': exp.get('avg_convergence', 0),
                'is_universal': exp.get('is_universal', False),
                'effect': effect,
                'graph_metrics': graph_metrics,
                'n_nodes_in_graph': G.number_of_nodes(),
            }
            enriched_results.append(enriched)

            found_tag = f"matched {graph_metrics['n_nodes_matched']}" if graph_metrics else "NOT found"
            effect_tag = f"shift={effect['mean_logprob_shift']:.3f}"
            print(f"  {feature_label:15s} {cid[:35]:35s} str={strength:+3d} {found_tag:12s} {effect_tag}")

    print(f"\n  Enriched results: {len(enriched_results)}")

    # Analyze correlations
    print("\n[3/4] Analyzing correlations...")

    # Filter to experiments where we found the feature in the graph
    with_metrics = [r for r in enriched_results if r['graph_metrics'] is not None]
    without_metrics = [r for r in enriched_results if r['graph_metrics'] is None]
    print(f"  With graph metrics: {len(with_metrics)}")
    print(f"  Without (feature not in graph): {len(without_metrics)}")

    # Predictor-effect correlations
    predictors = ['betweenness', 'in_degree', 'out_degree', 'activation',
                  'influence', 'pagerank', 'weighted_degree']
    effects = ['mean_logprob_shift', 'max_logprob_shift', 'kl_divergence']

    # Also use circuit-level predictors
    circuit_predictors = ['circuits_appeared_in', 'avg_convergence']

    correlations = {}

    print(f"\n  Graph metric vs steering effect correlations (n={len(with_metrics)}):")
    print(f"  {'Predictor':25s} {'mean_lp_shift':>15s} {'max_lp_shift':>15s} {'kl_divergence':>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    for pred in predictors:
        x = [r['graph_metrics'][pred] for r in with_metrics]
        row = {}
        vals = []
        for eff in effects:
            y = [r['effect'][eff] for r in with_metrics]
            r_val, p_val = pearson_r(x, y)
            star = '*' if p_val < 0.05 else ''
            row[eff] = {'r': r_val, 'p': p_val}
            vals.append(f"r={r_val:+.3f}{star}")
        correlations[pred] = row
        print(f"  {pred:25s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s}")

    # Circuit-level predictors (available for all experiments)
    print(f"\n  Circuit-level predictors vs effect (n={len(enriched_results)}):")
    for pred in circuit_predictors:
        x = [r[pred] for r in enriched_results]
        row = {}
        vals = []
        for eff in effects:
            y = [r['effect'][eff] for r in enriched_results]
            r_val, p_val = pearson_r(x, y)
            star = '*' if p_val < 0.05 else ''
            row[eff] = {'r': r_val, 'p': p_val}
            vals.append(f"r={r_val:+.3f}{star}")
        correlations[pred] = row
        print(f"  {pred:25s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s}")

    # Strength magnitude vs effect
    x_str = [abs(r['strength']) for r in enriched_results]
    for eff in effects:
        y_eff = [r['effect'][eff] for r in enriched_results]
        r_val, p_val = pearson_r(x_str, y_eff)
        star = '*' if p_val < 0.05 else ''
        correlations.setdefault('abs_strength', {})[eff] = {'r': r_val, 'p': p_val}
    print(f"  {'abs_strength':25s} " +
          ' '.join(f"r={correlations['abs_strength'][e]['r']:+.3f}{'*' if correlations['abs_strength'][e]['p']<0.05 else '':1s}" for e in effects))

    # Layer analysis
    print(f"\n  Per-layer mean effects:")
    layer_effects = defaultdict(list)
    for r in enriched_results:
        layer_effects[r['layer']].append(r['effect']['mean_logprob_shift'])

    for layer in sorted(layer_effects.keys()):
        vals = layer_effects[layer]
        m = sum(vals) / len(vals)
        print(f"    L{layer}: mean_shift={m:.4f} (n={len(vals)})")

    # Per-feature analysis
    print(f"\n  Per-feature mean effects:")
    feature_effects = defaultdict(list)
    for r in enriched_results:
        feature_effects[r['feature_label']].append(r['effect']['mean_logprob_shift'])

    for feat in sorted(feature_effects.keys()):
        vals = feature_effects[feat]
        m = sum(vals) / len(vals)
        print(f"    {feat:20s}: mean_shift={m:.4f} (n={len(vals)})")

    # Summary statistics
    print("\n[4/4] Generating summary...")

    # Best predictor
    best_pred = None
    best_r = 0
    for pred, eff_dict in correlations.items():
        for eff, vals in eff_dict.items():
            if abs(vals['r']) > abs(best_r):
                best_r = vals['r']
                best_pred = (pred, eff)

    all_shifts = [r['effect']['mean_logprob_shift'] for r in enriched_results]
    all_kl = [r['effect']['kl_divergence'] for r in enriched_results]
    text_change_rate = sum(1 for r in enriched_results if r['effect']['text_changed']) / len(enriched_results)

    def mean(lst): return sum(lst)/len(lst) if lst else 0

    print(f"\n  Summary:")
    print(f"    Mean logprob shift: {mean(all_shifts):.4f}")
    print(f"    Mean KL divergence: {mean(all_kl):.4f}")
    print(f"    Text change rate: {text_change_rate:.0%}")
    if best_pred:
        print(f"    Best predictor: {best_pred[0]} -> {best_pred[1]} (r={best_r:.3f})")

    # Build results
    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_experiments': len(experiments),
            'n_enriched': len(enriched_results),
            'n_with_graph_metrics': len(with_metrics),
            'n_circuits': len(circuit_groups),
        },
        'summary': {
            'mean_logprob_shift': mean(all_shifts),
            'mean_kl_divergence': mean(all_kl),
            'text_change_rate': text_change_rate,
            'best_predictor': {
                'predictor': best_pred[0] if best_pred else None,
                'effect': best_pred[1] if best_pred else None,
                'r': best_r,
            },
        },
        'correlations': correlations,
        'per_layer': {str(k): {'mean_shift': mean(v), 'n': len(v)}
                      for k, v in layer_effects.items()},
        'per_feature': {k: {'mean_shift': mean(v), 'n': len(v)}
                        for k, v in feature_effects.items()},
        'enriched_results': enriched_results,
    }

    # Save JSON
    json_path = output_dir / 'steering_validation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Steering Prediction Validation Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Experiments**: {len(experiments)}")
    lines.append(f"**Features with graph metrics**: {len(with_metrics)}/{len(enriched_results)}\n")
    lines.append("---\n")

    lines.append("## Summary\n")
    lines.append(f"- **Mean logprob shift**: {mean(all_shifts):.4f}")
    lines.append(f"- **Mean KL divergence**: {mean(all_kl):.4f}")
    lines.append(f"- **Text change rate**: {text_change_rate:.0%}")
    if best_pred:
        lines.append(f"- **Best predictor**: {best_pred[0]} -> {best_pred[1]} (r={best_r:.3f})\n")

    lines.append("## Graph Metric vs Steering Effect Correlations\n")
    lines.append("| Predictor | mean_lp_shift | max_lp_shift | KL_divergence |")
    lines.append("|-----------|--------------|-------------|---------------|")
    for pred in predictors + circuit_predictors + ['abs_strength']:
        if pred in correlations:
            row = correlations[pred]
            cells = []
            for eff in effects:
                if eff in row:
                    r_val = row[eff]['r']
                    p_val = row[eff]['p']
                    star = '*' if p_val < 0.05 else ''
                    cells.append(f"{r_val:+.3f}{star}")
                else:
                    cells.append("-")
            lines.append(f"| {pred} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines.append("## Per-Feature Effects\n")
    lines.append("| Feature | Mean Shift | N Experiments |")
    lines.append("|---------|-----------|---------------|")
    for feat in sorted(feature_effects.keys()):
        vals = feature_effects[feat]
        lines.append(f"| {feat} | {mean(vals):.4f} | {len(vals)} |")
    lines.append("")

    lines.append("## Per-Layer Effects\n")
    lines.append("| Layer | Mean Shift | N |")
    lines.append("|-------|-----------|---|")
    for layer in sorted(layer_effects.keys()):
        vals = layer_effects[layer]
        lines.append(f"| L{layer} | {mean(vals):.4f} | {len(vals)} |")
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append("This analysis tests whether graph-structural metrics can predict")
    lines.append("causal intervention effects. A strong correlation between betweenness/")
    lines.append("influence and logprob shift would validate the circuit analysis approach.")
    lines.append(f"With only {len(experiments)} experiments on {len(circuit_groups)} circuits,")
    lines.append("statistical power is limited — this is an initial feasibility test.\n")

    report_path = output_dir / 'steering_validation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  {len(enriched_results)} experiments enriched with graph metrics")
    if best_pred:
        print(f"  Best predictor: {best_pred[0]} -> {best_pred[1]} (r={best_r:.3f})")
    print("\nDone!")


if __name__ == '__main__':
    main()
