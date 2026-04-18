#!/usr/bin/env python3
"""
Query bottleneck features from circuit analysis using modulo 16384 mapping.

This script:
1. Loads circuit analysis JSON (from 3_analysis)
2. Extracts bottleneck features identified by traceback analysis
3. Maps Circuit Tracer global IDs to Neuronpedia per-layer IDs
4. Queries semantic descriptions from Neuronpedia API
5. Saves results for taxonomy building

Usage:
    python query_bottleneck_semantics.py \\
        --circuit ../data/prompts/gemma-2-2b_the-southern-most-us-state-is \\
        --output ../data/bottleneck_semantics.json
"""

import sys
import json
import argparse
from pathlib import Path
import io

# Fix Windows console encoding
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add scripts directory to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also add current directory so query_feature_semantics (sibling in advanced_analysis/) can be found
sys.path.insert(0, str(Path(__file__).parent))

from query_feature_semantics import FeatureSemanticAnalyzer


def map_circuit_to_neuronpedia(circuit_id: int) -> int:
    """
    Map Circuit Tracer global feature ID to Neuronpedia per-layer ID.

    Formula: neuronpedia_id = circuit_id % 16384
    Validated: 40/40 test cases successful (100%)

    Args:
        circuit_id: Global feature ID from Circuit Tracer raw graphs

    Returns:
        Per-layer feature ID for Neuronpedia API (0-16,383)
    """
    return circuit_id % 16384


def load_circuit_analysis(circuit_dir: Path) -> dict:
    """Load circuit analysis JSON from 3_analysis directory."""
    analysis_file = circuit_dir / '3_analysis' / 'circuit_analysis.json'

    if not analysis_file.exists():
        raise FileNotFoundError(f"Circuit analysis not found: {analysis_file}")

    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_raw_graph(circuit_dir: Path) -> dict:
    """Load raw graph JSON from 1_generation directory."""
    generation_dir = circuit_dir / '1_generation'

    # Try both naming patterns: raw_graph.json and *_raw_graph.json
    raw_graph_file = generation_dir / 'raw_graph.json'
    if raw_graph_file.exists():
        with open(raw_graph_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Fallback to pattern matching
    raw_graph_files = list(generation_dir.glob('*_raw_graph.json'))
    if not raw_graph_files:
        raise FileNotFoundError(f"No raw graph found in: {generation_dir}")

    with open(raw_graph_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_bottleneck_features(circuit_analysis: dict, raw_graph: dict) -> list:
    """
    Extract bottleneck features from circuit analysis.

    Returns:
        list of dicts with {node_id, layer, circuit_feature_id, neuronpedia_id}
    """
    bottlenecks = []

    # Get bottleneck nodes from flow_analysis
    flow_analysis = circuit_analysis.get('flow_analysis', {})
    bottleneck_nodes = flow_analysis.get('bottleneck_nodes', [])

    print(f"Found {len(bottleneck_nodes)} bottleneck nodes in analysis")

    # Build node_id -> feature mapping from raw graph
    node_feature_map = {}
    for node in raw_graph.get('nodes', []):
        node_id = node.get('node_id')
        feature = node.get('feature')
        layer = node.get('layer')

        if node_id and feature is not None and layer not in ['E', None]:
            node_feature_map[node_id] = {
                'feature': feature,
                'layer': int(layer) if str(layer).isdigit() else None
            }

    # Map bottleneck nodes to features
    for bottleneck in bottleneck_nodes:
        node_id = bottleneck.get('node_id')

        if node_id in node_feature_map:
            node_info = node_feature_map[node_id]
            circuit_feature_id = node_info['feature']
            layer = node_info['layer']

            if layer is not None and circuit_feature_id is not None:
                # Apply mapping
                neuronpedia_id = map_circuit_to_neuronpedia(circuit_feature_id)

                bottlenecks.append({
                    'node_id': node_id,
                    'layer': layer,
                    'circuit_feature_id': circuit_feature_id,
                    'neuronpedia_id': neuronpedia_id,
                    'betweenness': bottleneck.get('betweenness'),
                    'degree': bottleneck.get('degree')
                })

    return bottlenecks


def query_bottleneck_semantics(circuit_dir: Path, output_file: Path):
    """
    Main function to query bottleneck semantics.

    Args:
        circuit_dir: Path to circuit directory
        output_file: Path to save results JSON
    """
    print("=" * 70)
    print("BOTTLENECK SEMANTIC QUERY")
    print("=" * 70)
    print()
    print(f"Circuit: {circuit_dir.name}")
    print()

    # Load data
    print("Loading circuit analysis...")
    analysis = load_circuit_analysis(circuit_dir)

    print("Loading raw graph...")
    raw_graph = load_raw_graph(circuit_dir)

    # Extract bottlenecks
    print()
    print("Extracting bottleneck features...")
    bottlenecks = extract_bottleneck_features(analysis, raw_graph)

    if not bottlenecks:
        print("⚠ No bottlenecks found with valid feature IDs")
        return

    print(f"Found {len(bottlenecks)} bottlenecks to query")
    print()

    # Query semantics
    analyzer = FeatureSemanticAnalyzer()
    results = []

    for i, bottleneck in enumerate(bottlenecks, 1):
        node_id = bottleneck['node_id']
        layer = bottleneck['layer']
        circuit_id = bottleneck['circuit_feature_id']
        neuronpedia_id = bottleneck['neuronpedia_id']

        print(f"[{i}/{len(bottlenecks)}] {node_id}")
        print(f"  Circuit ID: {circuit_id:,}")
        print(f"  Mapped ID:  {neuronpedia_id:,}")
        print(f"  Layer:      {layer}")

        try:
            # Query using mapped ID
            feature_result = analyzer.query_single_feature(layer, neuronpedia_id)

            # Combine bottleneck info with semantic data
            result = {
                **bottleneck,
                'semantic_category': feature_result.get('semantic_category'),
                'description': feature_result.get('description'),
                'positive_examples': feature_result.get('positive_examples'),
                'activation_count': feature_result.get('activation_count')
            }

            results.append(result)

            # Print summary
            print(f"  ✓ Category: {result['semantic_category']}")
            if result['positive_examples']:
                examples_str = ', '.join(result['positive_examples'][:3])
                print(f"  ✓ Examples: {examples_str}")

        except Exception as e:
            print(f"  ✗ Query failed: {e}")
            # Add placeholder
            results.append({
                **bottleneck,
                'semantic_category': 'ERROR',
                'description': None,
                'positive_examples': [],
                'activation_count': 0,
                'error': str(e)
            })

        print()

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'circuit': circuit_dir.name,
            'total_bottlenecks': len(results),
            'successful_queries': sum(1 for r in results if r.get('semantic_category') not in ['ERROR', 'UNKNOWN']),
            'bottlenecks': results
        }, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Output: {output_file}")
    print(f"Total bottlenecks: {len(results)}")
    print(f"Successful queries: {sum(1 for r in results if r.get('semantic_category') not in ['ERROR', 'UNKNOWN'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Query bottleneck feature semantics using modulo 16384 mapping"
    )
    parser.add_argument(
        '--circuit',
        required=True,
        help='Path to circuit directory (e.g., ../data/prompts/gemma-2-2b_the-southern-most-us-state-is)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save results JSON (e.g., ../data/bottleneck_semantics.json)'
    )

    args = parser.parse_args()

    circuit_dir = Path(args.circuit)
    output_file = Path(args.output)

    if not circuit_dir.exists():
        print(f"Error: Circuit directory not found: {circuit_dir}")
        sys.exit(1)

    query_bottleneck_semantics(circuit_dir, output_file)


if __name__ == '__main__':
    main()
