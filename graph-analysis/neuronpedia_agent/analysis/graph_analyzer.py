"""Graph analysis module for attribution graphs"""

from typing import Dict, List, Tuple
import networkx as nx


class Path:
    """Represents a computational path through the graph"""
    def __init__(self, nodes: List[str], total_influence: float, bottleneck_nodes: List[str]):
        self.nodes = nodes
        self.total_influence = total_influence
        self.bottleneck_nodes = bottleneck_nodes


class GraphAnalyzer:
    """Analyze raw attribution graph structure to identify important computational patterns"""

    def __init__(self, graph_data: Dict):
        """Initialize with graph JSON data"""
        self.nodes = graph_data['nodes']
        self.edges = graph_data['edges']
        self.logit_nodes = graph_data.get('logit_nodes', [])

    def compute_node_importance(self) -> Dict[str, float]:
        """
        Compute importance score for each node based on:
        - Direct influence on target logits
        - Betweenness centrality (how many paths flow through this node)
        - Downstream influence (how much this node affects other important nodes)

        Returns: {node_id: importance_score}
        """
        importance = {}

        # 1. Direct logit influence
        for node in self.nodes:
            direct_influence = sum(
                edge['weight']
                for edge in self.edges
                if edge['source'] == node['id'] and edge['target'].startswith('logit_')
            )
            importance[node['id']] = direct_influence

        # 2. Add betweenness centrality
        graph = nx.DiGraph()
        graph.add_edges_from([
            (e['source'], e['target'], {'weight': e['weight']})
            for e in self.edges
        ])
        centrality = nx.betweenness_centrality(graph, weight='weight')

        for node_id, cent_score in centrality.items():
            importance[node_id] = importance.get(node_id, 0) + cent_score * 0.5

        # 3. Normalize
        max_score = max(importance.values()) if importance else 1.0
        return {k: v/max_score for k, v in importance.items()}

    def identify_input_features(self, layer_threshold: int = 5) -> List[str]:
        """
        Find features in early layers (0-5) that activate on specific input tokens
        These are typically the start of computational pathways

        Returns: List of node IDs
        """
        input_features = []
        for node in self.nodes:
            if node.get('layer', 0) <= layer_threshold:
                input_features.append(node['id'])
        return input_features

    def identify_output_features(self, layer_threshold: int = 16) -> List[str]:
        """
        Find features in late layers (16+) that promote specific output tokens
        These are typically the end of computational pathways

        Returns: List of node IDs
        """
        output_features = []
        for node in self.nodes:
            if node.get('layer', 0) >= layer_threshold:
                output_features.append(node['id'])
        return output_features

    def trace_pathways(self, source_nodes: List[str], target_nodes: List[str]) -> List[Path]:
        """
        Find strongest paths from source nodes (inputs) to target nodes (outputs)

        Path = {
            'nodes': [node_id1, node_id2, ...],
            'total_influence': float,
            'bottleneck_nodes': [nodes with high centrality]
        }

        Returns: List of paths sorted by total_influence
        """
        # Build graph
        graph = nx.DiGraph()
        graph.add_edges_from([
            (e['source'], e['target'], {'weight': e['weight']})
            for e in self.edges
        ])

        paths = []
        centrality = nx.betweenness_centrality(graph, weight='weight')

        # Find paths between each source-target pair
        for source in source_nodes:
            for target in target_nodes:
                if source in graph and target in graph:
                    try:
                        # Get shortest weighted path
                        path_nodes = nx.shortest_path(
                            graph,
                            source,
                            target,
                            weight=lambda u, v, d: 1.0 / (d['weight'] + 1e-10)
                        )

                        # Calculate total influence along path
                        total_influence = sum(
                            graph[path_nodes[i]][path_nodes[i+1]]['weight']
                            for i in range(len(path_nodes) - 1)
                        )

                        # Identify bottleneck nodes (high centrality nodes in path)
                        bottlenecks = [
                            node for node in path_nodes
                            if centrality.get(node, 0) > 0.5
                        ]

                        paths.append(Path(
                            nodes=path_nodes,
                            total_influence=total_influence,
                            bottleneck_nodes=bottlenecks
                        ))
                    except nx.NetworkXNoPath:
                        continue

        # Sort by total influence
        paths.sort(key=lambda p: p.total_influence, reverse=True)
        return paths

    def get_node(self, node_id: str) -> Dict:
        """Get node data by ID"""
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None
