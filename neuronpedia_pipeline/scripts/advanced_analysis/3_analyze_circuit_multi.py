#!/usr/bin/env python3
"""
Multi-Algorithm Circuit Analysis with Validation

Runs 4 independent community detection methods and compares results
to prove circuit structure is real, not an algorithmic artifact.

Methods:
1. Louvain (current baseline)
2. Leiden (improved modularity)
3. Greedy Modularity (NetworkX fast)
4. Label Propagation (neighbor voting)

Output:
- Individual results for each method
- Comparison statistics (Jaccard similarity, modularity)
- Convergence analysis (which bottlenecks all methods agree on)

Usage:
    python 3_analyze_circuit_multi.py --file PATH_TO_GRAPH.json --output OUTPUT_DIR

Author: AI Safety Camp 2025 - Project #24
Date: February 2026
"""

import networkx as nx
import community as louvain_community  # python-louvain
from pathlib import Path
import json
from collections import defaultdict
import sys


def detect_communities_louvain(G, resolution=1.0):
    """
    Louvain community detection (CURRENT BASELINE METHOD)

    Uses greedy modularity optimization with configurable resolution parameter.

    Args:
        G (nx.DiGraph): NetworkX directed graph with 'weight' edge attribute
        resolution (float): Modularity resolution parameter (default: 1.0)
            - Higher values → more communities
            - Lower values → fewer communities

    Returns:
        dict: {node_id: community_id} mapping

    Complexity: O(n log n) where n = number of nodes

    Reference: "Fast unfolding of communities in large networks"
               Blondel et al. (2008)
    """
    # Convert directed graph to undirected for community detection
    G_undirected = G.to_undirected()

    communities = louvain_community.best_partition(
        G_undirected,
        weight='weight',
        resolution=resolution
    )

    return communities


def detect_communities_leiden(G, resolution=0.01):
    """
    Leiden community detection (IMPROVED METHOD)

    Guarantees well-connected communities, fixing Louvain's disconnected
    community problem.

    Args:
        G (nx.DiGraph): NetworkX directed graph with 'weight' edge attribute
        resolution (float): CPM resolution parameter (default: 0.01)
            - Different scale than Louvain resolution
            - 0.01 is typical for sparse graphs

    Returns:
        dict: {node_id: community_id} mapping
        None if igraph/leidenalg not installed

    Complexity: O(n log n) similar to Louvain but with better guarantees

    Reference: "From Louvain to Leiden: guaranteeing well-connected communities"
               Traag et al. (2019)
    """
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError:
        print("\n[WARNING] igraph/leidenalg not installed, skipping Leiden")
        print("Install with: pip install igraph leidenalg")
        return None

    # Convert NetworkX to igraph
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Convert edges (ignore direction for community detection)
    edges = []
    weights = []
    for u, v in G.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(abs(G[u][v]['weight']))

    # Create igraph
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    ig_graph.es['weight'] = weights

    # Run Leiden algorithm
    try:
        partition = la.find_partition(
            ig_graph,
            la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution
        )

        # Convert back to dict
        communities = {
            node_list[i]: partition.membership[i]
            for i in range(len(node_list))
        }

        return communities

    except Exception as e:
        print(f"\n[WARNING] Leiden failed: {e}")
        return None


def detect_communities_greedy(G):
    """
    Greedy modularity maximization (FAST METHOD)

    Uses NetworkX's built-in greedy_modularity_communities which
    iteratively merges communities to maximize modularity.

    Args:
        G (nx.DiGraph): NetworkX directed graph with 'weight' edge attribute

    Returns:
        dict: {node_id: community_id} mapping

    Complexity: O(m log n) where m = edges, n = nodes

    Reference: "Finding community structure in very large networks"
               Clauset et al. (2004)
    """
    # Convert to undirected
    G_undirected = G.to_undirected()

    # Run greedy modularity
    communities_list = nx.community.greedy_modularity_communities(
        G_undirected,
        weight='weight',
        resolution=1.0
    )

    # Convert to dict
    communities = {}
    for comm_id, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            communities[node] = comm_id

    return communities


def detect_communities_label_propagation(G):
    """
    Label propagation (VOTING METHOD)

    Completely different principle: nodes iteratively adopt the most common
    label among their neighbors. No modularity optimization.

    Args:
        G (nx.DiGraph): NetworkX directed graph with 'weight' edge attribute

    Returns:
        dict: {node_id: community_id} mapping

    Complexity: O(m) - very fast, near-linear time

    Reference: "Near linear time algorithm to detect community structures"
               Raghavan et al. (2007)
    """
    # Convert to undirected
    G_undirected = G.to_undirected()

    # Run asynchronous label propagation
    communities_list = nx.community.asyn_lpa_communities(
        G_undirected,
        weight='weight'
    )

    # Convert to dict
    communities = {}
    for comm_id, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            communities[node] = comm_id

    return communities


def calculate_modularity(G, communities):
    """
    Calculate modularity score for a partition

    Modularity measures how well the partition separates the graph into
    communities vs a random null model.

    Args:
        G (nx.DiGraph): NetworkX graph
        communities (dict): {node: community_id} mapping

    Returns:
        float: Modularity score in range [-0.5, 1.0]
            - > 0.3: Good community structure
            - > 0.5: Strong community structure
            - < 0.1: Weak or no community structure

    Formula:
        Q = (1/2m) Σ [A_ij - (k_i * k_j)/2m] δ(c_i, c_j)

    Where:
        - A_ij = adjacency matrix
        - k_i = degree of node i
        - m = total edge weight
        - δ(c_i, c_j) = 1 if nodes in same community, 0 otherwise
    """
    # Convert dict to list of sets
    unique_comms = set(communities.values())
    communities_list = [
        set(node for node, comm in communities.items() if comm == c)
        for c in unique_comms
    ]

    # Convert to undirected for modularity calculation
    G_undirected = G.to_undirected()

    return nx.community.modularity(G_undirected, communities_list, weight='weight')


def compare_partitions(partition1, partition2):
    """
    Calculate Jaccard similarity between two partitions

    Measures overlap of communities between methods. Uses best matching
    strategy to align communities.

    Args:
        partition1, partition2 (dict): {node: community_id} mappings

    Returns:
        float: Jaccard similarity in range [0, 1]
            - 1.0: Perfect agreement
            - >0.7: Strong agreement
            - 0.5-0.7: Moderate agreement
            - <0.5: Weak agreement

    Algorithm:
        For each community in partition1, find best matching community
        in partition2 (highest Jaccard overlap), then average.
    """
    def partition_to_communities(partition):
        """Convert {node: comm_id} to list of sets"""
        comms = defaultdict(set)
        for node, comm_id in partition.items():
            comms[comm_id].add(node)
        return list(comms.values())

    comms1 = partition_to_communities(partition1)
    comms2 = partition_to_communities(partition2)

    if not comms1 or not comms2:
        return 0.0

    # For each community in partition1, find best match in partition2
    total_similarity = 0.0
    for c1 in comms1:
        best_match = 0.0
        for c2 in comms2:
            # Jaccard = |intersection| / |union|
            intersection = len(c1 & c2)
            union = len(c1 | c2)
            if union > 0:
                jaccard = intersection / union
                best_match = max(best_match, jaccard)
        total_similarity += best_match

    # Normalize by number of communities
    return total_similarity / len(comms1)


def identify_consensus_bottlenecks(all_traceback_paths, convergence_threshold=0.75):
    """
    Identify bottlenecks that appear in MANY paths with HIGH scores

    Bottleneck = node that appears in ≥75% of paths (convergence),
    indicating all paths flow through it.

    Args:
        all_traceback_paths (list): List of path dictionaries from traceback analysis
        convergence_threshold (float): Fraction of paths node must appear in (default: 0.75)

    Returns:
        list: Sorted list of (node_id, metrics) tuples for consensus bottlenecks

    Metrics per bottleneck:
        - convergence: Fraction of paths containing this node
        - avg_score: Average score across paths
        - max_score: Maximum score seen
        - layer: Which transformer layer
    """
    if not all_traceback_paths:
        return []

    # Count appearances and accumulate scores
    node_stats = defaultdict(lambda: {
        'layer': None,
        'appearances': 0,
        'total_score': 0,
        'max_score': 0,
        'min_score': float('inf'),
        'path_ids': []
    })

    num_paths = len(all_traceback_paths)

    for path in all_traceback_paths:
        if 'path_nodes' not in path:
            continue

        for node in path['path_nodes']:
            node_id = node.get('label', node.get('id', ''))
            score = node.get('score', 0)
            layer = node.get('layer', -1)

            if node_stats[node_id]['layer'] is None:
                node_stats[node_id]['layer'] = layer

            node_stats[node_id]['appearances'] += 1
            node_stats[node_id]['total_score'] += score
            node_stats[node_id]['max_score'] = max(
                node_stats[node_id]['max_score'], score
            )
            node_stats[node_id]['min_score'] = min(
                node_stats[node_id]['min_score'], score
            )
            node_stats[node_id]['path_ids'].append(path.get('rank', 0))

    # Filter by convergence threshold
    bottlenecks = []
    for node_id, stats in node_stats.items():
        convergence = stats['appearances'] / num_paths
        if convergence >= convergence_threshold:
            avg_score = stats['total_score'] / stats['appearances']
            bottlenecks.append((node_id, {
                **stats,
                'convergence': convergence,
                'avg_score': avg_score
            }))

    # Sort by average score descending
    bottlenecks.sort(key=lambda x: x[1]['avg_score'], reverse=True)

    return bottlenecks


def run_multi_algorithm_analysis(graph_file, output_dir):
    """
    Main analysis: Run all 4 methods and compare

    Args:
        graph_file (str): Path to converted graph JSON
        output_dir (str): Directory to save results

    Returns:
        dict: Complete analysis results including validation metrics

    Workflow:
        1. Load graph from JSON
        2. Run all 4 community detection methods
        3. Calculate modularity for each method
        4. Compute pairwise Jaccard similarities
        5. Determine consensus bottlenecks (if traceback available)
        6. Save comprehensive JSON with validation metrics

    Output JSON structure:
        {
          "methods": {...},  # Results for each method
          "comparisons": {...},  # Pairwise Jaccard scores
          "consensus_bottlenecks": [...],  # Nodes all methods agree on
          "validation": {
            "all_methods_ran": bool,
            "avg_jaccard": float,
            "result_quality": str
          }
        }
    """
    print("=" * 70)
    print("MULTI-ALGORITHM CIRCUIT ANALYSIS")
    print("=" * 70)
    print(f"\nInput: {graph_file}")
    print(f"Output: {output_dir}")

    # Load graph
    graph_path = Path(graph_file)
    if not graph_path.exists():
        print(f"\n[ERROR] Graph file not found: {graph_file}")
        sys.exit(1)

    with open(graph_path) as f:
        data = json.load(f)

    # Build NetworkX graph
    G = nx.DiGraph()

    print("\n[1/6] Loading graph...")
    for node in data['nodes']:
        G.add_node(
            node['id'],
            layer=node['layer'],
            activation=node.get('activation', 0),
            influence=node.get('influence', 0)
        )

    for edge in data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            weight=abs(edge['weight'])
        )

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")

    # Run all methods
    results = {}

    # Method 1: Louvain
    print("\n[2/6] Running Louvain (baseline)...")
    communities_louvain = detect_communities_louvain(G, resolution=1.0)
    modularity_louvain = calculate_modularity(G, communities_louvain)
    num_comms_louvain = len(set(communities_louvain.values()))
    print(f"  Communities: {num_comms_louvain}")
    print(f"  Modularity: {modularity_louvain:.4f}")
    results['louvain'] = {
        'communities': communities_louvain,
        'modularity': modularity_louvain,
        'num_communities': num_comms_louvain
    }

    # Method 2: Leiden
    print("\n[3/6] Running Leiden (improved)...")
    communities_leiden = detect_communities_leiden(G, resolution=0.01)
    if communities_leiden:
        modularity_leiden = calculate_modularity(G, communities_leiden)
        num_comms_leiden = len(set(communities_leiden.values()))
        print(f"  Communities: {num_comms_leiden}")
        print(f"  Modularity: {modularity_leiden:.4f}")
        results['leiden'] = {
            'communities': communities_leiden,
            'modularity': modularity_leiden,
            'num_communities': num_comms_leiden
        }
    else:
        print("  Skipped (dependencies not installed)")

    # Method 3: Greedy Modularity
    print("\n[4/6] Running Greedy Modularity (fast)...")
    communities_greedy = detect_communities_greedy(G)
    modularity_greedy = calculate_modularity(G, communities_greedy)
    num_comms_greedy = len(set(communities_greedy.values()))
    print(f"  Communities: {num_comms_greedy}")
    print(f"  Modularity: {modularity_greedy:.4f}")
    results['greedy'] = {
        'communities': communities_greedy,
        'modularity': modularity_greedy,
        'num_communities': num_comms_greedy
    }

    # Method 4: Label Propagation
    print("\n[5/6] Running Label Propagation (voting)...")
    communities_label = detect_communities_label_propagation(G)
    modularity_label = calculate_modularity(G, communities_label)
    num_comms_label = len(set(communities_label.values()))
    print(f"  Communities: {num_comms_label}")
    print(f"  Modularity: {modularity_label:.4f}")
    results['label_propagation'] = {
        'communities': communities_label,
        'modularity': modularity_label,
        'num_communities': num_comms_label
    }

    # Compare methods
    print("\n[6/6] Computing cross-method comparisons...")
    methods = list(results.keys())
    comparisons = {}

    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            similarity = compare_partitions(
                results[method1]['communities'],
                results[method2]['communities']
            )
            key = f"{method1}_vs_{method2}"
            comparisons[key] = similarity
            print(f"  {method1} vs {method2}: Jaccard = {similarity:.3f}")

    # Calculate average Jaccard
    avg_jaccard = sum(comparisons.values()) / len(comparisons) if comparisons else 0

    # Determine result quality
    if avg_jaccard > 0.7:
        quality = "VALID - Strong convergence"
    elif avg_jaccard > 0.5:
        quality = "MODERATE - Likely real but less robust"
    else:
        quality = "WEAK - May be algorithmic artifacts"

    # Check for traceback results to identify consensus bottlenecks
    consensus_bottlenecks = []
    traceback_file = Path(output_dir) / "traceback_paths.json"
    if traceback_file.exists():
        print("\n[BONUS] Identifying consensus bottlenecks from traceback...")
        with open(traceback_file) as f:
            traceback_data = json.load(f)
        if 'critical_paths' in traceback_data:
            bottlenecks = identify_consensus_bottlenecks(
                traceback_data['critical_paths'],
                convergence_threshold=0.75
            )
            consensus_bottlenecks = [
                {'node_id': node_id, **stats}
                for node_id, stats in bottlenecks
            ]
            print(f"  Found {len(consensus_bottlenecks)} consensus bottlenecks (75%+ convergence)")
            for i, bn in enumerate(consensus_bottlenecks[:5]):  # Show top 5
                print(f"    {i+1}. {bn['node_id']} (L{bn['layer']}, {bn['convergence']*100:.0f}% paths)")

    # Prepare output
    output = {
        'graph_file': str(graph_file),
        'graph_stats': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G)
        },
        'methods': results,
        'comparisons': comparisons,
        'consensus_bottlenecks': consensus_bottlenecks,
        'validation': {
            'all_methods_ran': len(results) >= 3,  # At least 3 of 4 methods
            'num_methods': len(results),
            'avg_jaccard': avg_jaccard,
            'consensus_count': len(consensus_bottlenecks),
            'result_quality': quality
        }
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'multi_algorithm_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Methods ran: {len(results)}/4")
    print(f"Average Jaccard similarity: {avg_jaccard:.3f}")
    print(f"Result quality: {quality}")
    print(f"\n[OK] Results saved to: {output_file}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-algorithm circuit analysis for validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on GEMMA southern state
    python 3_analyze_circuit_multi.py \\
        --file ../data/prompts/gemma-2-2b_the-southern-most-us-state-is/2_conversion/*_converted_graph.json \\
        --output ../data/prompts/gemma-2-2b_the-southern-most-us-state-is/3_analysis/

Interpretation:
    - Avg Jaccard > 0.7: Strong agreement = results are REAL
    - Avg Jaccard 0.5-0.7: Moderate agreement = likely real
    - Avg Jaccard < 0.5: Weak agreement = may be artifacts
        """
    )
    parser.add_argument(
        '--file',
        required=True,
        help='Converted graph JSON file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Handle wildcards in file path
    file_path = Path(args.file)
    if '*' in str(file_path):
        # Expand wildcard
        parent = file_path.parent
        pattern = file_path.name
        matches = list(parent.glob(pattern))
        if not matches:
            print(f"[ERROR] No files match pattern: {args.file}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"[WARNING] Multiple files match, using first: {matches[0]}")
        file_path = matches[0]

    run_multi_algorithm_analysis(str(file_path), args.output)
