"""
Supernode Detector
Groups features into high-level supernodes for circuit analysis
Uses community detection algorithms (Louvain, Leiden)
"""

import json
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("WARNING: python-louvain not installed. Run: pip install python-louvain")


class SupernodeDetector:
    """Detect and analyze supernodes in attribution graphs"""

    def __init__(self, min_supernode_size: int = 3, max_supernode_size: int = 50):
        self.min_size = min_supernode_size
        self.max_size = max_supernode_size

    def detect_supernodes_louvain(self, graph: nx.Graph, resolution: float = 1.0) -> Dict[int, List[str]]:
        """
        Use Louvain algorithm to detect communities (supernodes)

        Args:
            graph: NetworkX graph (undirected for community detection)
            resolution: Resolution parameter for Louvain (higher = more communities)

        Returns:
            Dictionary mapping community_id -> list of node IDs
        """
        if not LOUVAIN_AVAILABLE:
            print("ERROR: Louvain algorithm not available")
            return self._fallback_layer_grouping(graph)

        # Ensure undirected for community detection
        if graph.is_directed():
            graph = graph.to_undirected()

        # Run Louvain algorithm with resolution parameter
        partition = community_louvain.best_partition(graph, resolution=resolution)

        # Group nodes by community
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)

        # Filter by size constraints
        filtered = {}
        for comm_id, nodes in communities.items():
            if self.min_size <= len(nodes) <= self.max_size:
                filtered[comm_id] = nodes

        print(f"Detected {len(filtered)} supernodes (filtered from {len(communities)} communities)")
        print(f"  Resolution parameter: {resolution}")
        return filtered

    def detect_supernodes_layer_based(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """
        Simple layer-based grouping as fallback

        Groups all features in same layer together
        """
        # Extract layer from node attributes
        layers = defaultdict(list)

        for node in graph.nodes():
            node_data = graph.nodes[node]
            layer = node_data.get('layer', 'unknown')
            layers[layer].append(node)

        print(f"Created {len(layers)} layer-based supernodes")
        return {i: nodes for i, (layer, nodes) in enumerate(layers.items())}

    def _fallback_layer_grouping(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Fallback to layer-based grouping"""
        return self.detect_supernodes_layer_based(graph)

    def detect_supernodes_leiden(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """
        Use Leiden algorithm (better than Louvain for some cases)

        Requires: pip install leidenalg igraph
        """
        try:
            import igraph as ig
            import leidenalg as la

            # Convert NetworkX to igraph
            edge_list = [(u, v) for u, v in graph.edges()]
            g = ig.Graph.TupleList(edge_list, directed=False)

            # Run Leiden algorithm
            partition = la.find_partition(g, la.ModularityVertexPartition)

            # Convert back to node IDs
            node_list = list(graph.nodes())
            communities = defaultdict(list)

            for node_idx, comm_id in enumerate(partition.membership):
                communities[comm_id].append(node_list[node_idx])

            print(f"Leiden detected {len(communities)} communities")
            return dict(communities)

        except ImportError:
            print("Leiden not available, falling back to Louvain")
            return self.detect_supernodes_louvain(graph)

    def analyze_supernode(self, graph: nx.Graph, supernode_nodes: List[str]) -> Dict:
        """
        Analyze properties of a supernode

        Args:
            graph: Full graph
            supernode_nodes: List of node IDs in this supernode

        Returns:
            Dictionary with supernode statistics
        """
        subgraph = graph.subgraph(supernode_nodes)

        # Get node attributes
        activations = [graph.nodes[n].get('activation', 0) for n in supernode_nodes]
        layers = [graph.nodes[n].get('layer') for n in supernode_nodes if graph.nodes[n].get('layer') is not None]
        labels = [graph.nodes[n].get('label', '') for n in supernode_nodes]

        # Compute statistics
        stats = {
            'size': len(supernode_nodes),
            'nodes': supernode_nodes,
            'mean_activation': np.mean(activations) if activations else 0,
            'max_activation': max(activations) if activations else 0,
            'layers': list(set(layers)),
            'layer_distribution': {l: layers.count(l) for l in set(layers)},
            'internal_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph) if len(supernode_nodes) > 1 else 0,
            'labels': labels[:5],  # Sample of labels
        }

        # External connectivity
        external_edges = 0
        for node in supernode_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor not in supernode_nodes:
                    external_edges += 1

        stats['external_edges'] = external_edges
        stats['external_ratio'] = external_edges / (external_edges + stats['internal_edges']) if (external_edges + stats['internal_edges']) > 0 else 0

        return stats

    def rank_supernodes(self, graph: nx.Graph, supernodes: Dict[int, List[str]]) -> List[Tuple[int, float]]:
        """
        Rank supernodes by importance

        Importance criteria:
        - High mean activation
        - High betweenness centrality
        - Many external connections (hub behavior)

        Returns:
            List of (supernode_id, importance_score) sorted by score
        """
        scores = []

        for supernode_id, nodes in supernodes.items():
            stats = self.analyze_supernode(graph, nodes)

            # Compute importance score (weighted combination)
            activation_score = stats['mean_activation']
            connectivity_score = stats['external_edges'] / 100  # Normalize
            density_score = stats['density']

            importance = (
                0.4 * activation_score +
                0.4 * connectivity_score +
                0.2 * density_score
            )

            scores.append((supernode_id, importance))

        # Sort by importance (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def describe_supernode(self, graph: nx.Graph, supernode_nodes: List[str]) -> str:
        """
        Generate human-readable description of supernode

        Returns:
            String description
        """
        stats = self.analyze_supernode(graph, supernode_nodes)

        desc = f"Supernode with {stats['size']} features\n"
        desc += f"  Layers: {', '.join(map(str, stats['layers']))}\n"
        desc += f"  Mean activation: {stats['mean_activation']:.3f}\n"
        desc += f"  Max activation: {stats['max_activation']:.3f}\n"
        desc += f"  Internal edges: {stats['internal_edges']}\n"
        desc += f"  External edges: {stats['external_edges']}\n"
        desc += f"  Density: {stats['density']:.3f}\n"

        if stats['labels']:
            desc += f"  Sample labels: {', '.join(stats['labels'][:3])}\n"

        return desc

    def save_supernodes(self, supernodes: Dict[int, List[str]], output_path: Path):
        """Save supernode assignments to JSON"""
        with open(output_path, 'w') as f:
            json.dump(supernodes, f, indent=2)
        print(f"Saved supernodes to {output_path}")


def main():
    """Test supernode detection"""
    print("Supernode Detector - Test")

    # Create sample graph
    G = nx.karate_club_graph()  # Classic test graph

    detector = SupernodeDetector(min_supernode_size=2, max_supernode_size=20)

    # Detect supernodes
    supernodes = detector.detect_supernodes_louvain(G)

    print(f"\nDetected {len(supernodes)} supernodes")
    for supernode_id, nodes in list(supernodes.items())[:3]:
        print(f"\nSupernode {supernode_id}:")
        print(f"  Size: {len(nodes)}")
        print(f"  Nodes: {nodes[:5]}...")

    # Rank supernodes
    print("\n" + "="*60)
    print("Ranking supernodes...")
    ranked = detector.rank_supernodes(G, supernodes)

    print("\nTop 3 most important supernodes:")
    for supernode_id, score in ranked[:3]:
        print(f"  Supernode {supernode_id}: importance score = {score:.3f}")


if __name__ == "__main__":
    main()
