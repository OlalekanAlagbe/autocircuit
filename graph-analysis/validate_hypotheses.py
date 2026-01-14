#!/usr/bin/env python3
"""
Validate feature hypotheses by sampling key nodes and preparing them for lookup.

This script identifies key features across different computational phases of a circuit
and prepares them for validation against expected semantic interpretations.

Usage:
    python validate_hypotheses.py <graph_data.json>
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_graph_data(filepath: str) -> dict:
    """Load graph data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def organize_nodes_by_layer_ctx(nodes: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    """Organize nodes by (layer, context_position) pairs."""
    organized = defaultdict(list)
    for node in nodes:
        if node['feature_type'] == 'cross layer transcoder':
            key = (str(node['layer']), node['ctx_idx'])
            organized[key].append(node)
    return organized


def calculate_in_degree(links: List[dict]) -> Dict[str, int]:
    """Calculate in-degree for each node."""
    in_degree = defaultdict(int)
    for link in links:
        in_degree[link['target']] += 1
    return in_degree


def sample_domain_features(nodes_by_layer_ctx: dict, ctx_pos: int,
                           layers: List[str], n: int = 5) -> List[dict]:
    """Sample top features from early layers at specific context position."""
    features = []
    for layer in layers:
        features.extend(nodes_by_layer_ctx.get((layer, ctx_pos), []))

    features = [f for f in features if f.get('influence') is not None]
    features.sort(key=lambda x: x['influence'], reverse=True)
    return features[:n]


def sample_hub_features(nodes_by_layer_ctx: dict, in_degree: dict,
                       layer: str, ctx_pos: int, n: int = 5) -> List[Tuple[dict, int]]:
    """Sample hub nodes with highest in-degree."""
    nodes = nodes_by_layer_ctx.get((layer, ctx_pos), [])
    nodes_with_degree = []
    for node in nodes:
        if node.get('influence') is not None:
            degree = in_degree[node['node_id']]
            nodes_with_degree.append((node, degree))

    nodes_with_degree.sort(key=lambda x: x[1], reverse=True)
    return nodes_with_degree[:n]


def sample_output_features(nodes: List[dict], links: List[dict],
                           output_node: str, layer: str, ctx_pos: int,
                           n: int = 5) -> List[Tuple[dict, float]]:
    """Sample features with strongest connections to output logit."""
    output_edges = []
    for link in links:
        if link['target'] == output_node:
            src = next((n for n in nodes if n['node_id'] == link['source']), None)
            if (src and src.get('layer') == layer and
                src.get('ctx_idx') == ctx_pos and src.get('influence') is not None):
                output_edges.append((src, abs(link['weight'])))

    output_edges.sort(key=lambda x: x[1], reverse=True)
    return output_edges[:n]


def print_hypothesis_results(hypothesis_name: str, features: List,
                            description: str, location: str):
    """Print formatted hypothesis results."""
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS: {hypothesis_name}")
    print(f"{'='*80}")
    print(f"Expected: {description}")
    print(f"Location: {location}\n")

    if isinstance(features[0], tuple):
        # Handle (node, metric) tuples
        for i, (node, metric) in enumerate(features, 1):
            inf = node.get('influence', 0)
            act = node.get('activation', 0)
            layer = node.get('layer', '?')
            print(f"{i}. Feature {node['feature']:>10} | "
                  f"Layer: {layer:<2} | Metric: {metric:.3f} | "
                  f"Inf: {inf:.3f} | Act: {act:.2f}")
    else:
        # Handle plain nodes
        for i, node in enumerate(features, 1):
            inf = node.get('influence', 0)
            act = node.get('activation', 0)
            layer = node.get('layer', '?')
            print(f"{i}. Feature {node['feature']:>10} | "
                  f"Layer: {layer:<2} | Inf: {inf:.3f} | Act: {act:.2f}")


def main(graph_path: str):
    """Main validation workflow."""
    print("="*80)
    print("FEATURE HYPOTHESIS VALIDATION")
    print("="*80)
    print(f"\nLoading graph data from: {graph_path}\n")

    # Load and organize data
    data = load_graph_data(graph_path)
    nodes_by_layer_ctx = organize_nodes_by_layer_ctx(data['nodes'])
    in_degree = calculate_in_degree(data['links'])

    # Extract metadata
    metadata = data.get('metadata', {})
    prompt = metadata.get('prompt', 'unknown')
    print(f"Prompt: {prompt}")
    print(f"Total nodes: {len(data['nodes'])}")
    print(f"Total edges: {len(data['links'])}\n")

    # Collect all sampled features
    all_features = set()

    # HYPOTHESIS 1: Domain Features
    h1_features = sample_domain_features(nodes_by_layer_ctx, 1, ['0', '1', '2'], n=5)
    print_hypothesis_results(
        "Domain/Topic Features",
        h1_features,
        "Scientific/biology domain context detectors",
        "L0-L2, Context position 1"
    )
    all_features.update(f['feature'] for f in h1_features)

    # HYPOTHESIS 2: Structural Features
    h2_features_c2 = sample_domain_features(nodes_by_layer_ctx, 2, ['1', '2'], n=3)
    h2_features_c3 = sample_domain_features(nodes_by_layer_ctx, 3, ['2', '3'], n=3)
    h2_features = h2_features_c2 + h2_features_c3
    print_hypothesis_results(
        "Syntactic/Structural Features",
        h2_features,
        "Definitional structure detectors ('X stands for Y')",
        "L1-L3, Context positions 2-3"
    )
    all_features.update(f['feature'] for f in h2_features)

    # HYPOTHESIS 3: Morphological Features
    h3_features = sample_domain_features(nodes_by_layer_ctx, 4, ['6', '7', '8', '9', '10'], n=5)
    print_hypothesis_results(
        "Morphological/Prefix Features",
        h3_features,
        "Chemical prefix detectors ('deoxy-')",
        "L6-L10, Context position 4"
    )
    all_features.update(f['feature'] for f in h3_features)

    # HYPOTHESIS 4: Integration Hubs
    h4_features = sample_hub_features(nodes_by_layer_ctx, in_degree, '23', 6, n=5)
    print_hypothesis_results(
        "Integration Hub Features",
        h4_features,
        "High in-degree nodes combining multiple evidence sources",
        "L23, Context position 6"
    )
    all_features.update(f[0]['feature'] for f in h4_features)

    # HYPOTHESIS 5: Output Features
    # Try to find output node
    output_node = next((n['node_id'] for n in data['nodes'] if n.get('feature_type') == 'logit'), None)
    if output_node:
        h5_features = sample_output_features(data['nodes'], data['links'], output_node, '25', 6, n=5)
        print_hypothesis_results(
            "Token-Specific Boosting Features",
            h5_features,
            "Features specifically boosting output token",
            "L25, Context position 6"
        )
        all_features.update(f[0]['feature'] for f in h5_features)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal unique features sampled: {len(all_features)}")
    print("\nFeature IDs for Neuronpedia lookup:")
    for feat in sorted(all_features):
        print(f"  {feat}")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("""
1. Look up each feature on Neuronpedia to verify semantic interpretation
2. Check if activation patterns match hypothesis predictions
3. Calculate hypothesis confirmation rate
4. Document unexpected or surprising findings

Neuronpedia URL format:
https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k/{FEATURE_ID}
    """)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_hypotheses.py <graph_data.json>")
        sys.exit(1)

    main(sys.argv[1])
