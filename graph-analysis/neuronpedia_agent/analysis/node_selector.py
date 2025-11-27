"""Node selection module for choosing which nodes to pin"""

from typing import List
from .graph_analyzer import GraphAnalyzer


class NodeSelector:
    """Automatically select which nodes to pin based on importance and interpretability"""

    def __init__(self, analyzer: GraphAnalyzer, max_nodes: int = 30):
        self.analyzer = analyzer
        self.max_nodes = max_nodes

    def select_nodes_for_pinning(self, strategy: str = "pathway") -> List[str]:
        """
        Select nodes to pin using one of several strategies:

        - "pathway": Follow strongest paths from input to output
        - "importance": Select top-N most important nodes globally
        - "balanced": Mix of input features, middle processing, output features

        Returns: List of node IDs to pin
        """
        if strategy == "pathway":
            return self._pathway_strategy()
        elif strategy == "importance":
            return self._importance_strategy()
        elif strategy == "balanced":
            return self._balanced_strategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _pathway_strategy(self) -> List[str]:
        """
        1. Identify top 3-5 target logits
        2. Find input features in early layers
        3. Trace strongest paths between them
        4. Select nodes on these paths, prioritizing bottlenecks
        """
        # Get input and output features
        input_features = self.analyzer.identify_input_features(layer_threshold=5)
        output_features = self.analyzer.identify_output_features(layer_threshold=16)

        # Trace paths
        paths = self.analyzer.trace_pathways(
            input_features[:10],  # Sample top 10 input features
            output_features[:10]   # Sample top 10 output features
        )

        # Collect nodes from strongest paths
        selected_nodes = set()
        for path in paths:
            # Prioritize bottleneck nodes
            selected_nodes.update(path.bottleneck_nodes)
            # Add other nodes from path
            selected_nodes.update(path.nodes)

            if len(selected_nodes) >= self.max_nodes:
                break

        return list(selected_nodes)[:self.max_nodes]

    def _importance_strategy(self) -> List[str]:
        """Select top-N most important nodes globally"""
        importance = self.analyzer.compute_node_importance()

        # Sort by importance and take top N
        sorted_nodes = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [node_id for node_id, _ in sorted_nodes[:self.max_nodes]]

    def _balanced_strategy(self) -> List[str]:
        """
        Ensure representation across layers:
        - 30% input features (layers 0-5)
        - 40% middle processing (layers 6-15)
        - 30% output features (layers 16+)
        """
        importance = self.analyzer.compute_node_importance()

        # Categorize nodes by layer
        input_nodes = []
        middle_nodes = []
        output_nodes = []

        for node_id, imp_score in importance.items():
            node = self.analyzer.get_node(node_id)
            if node:
                layer = node.get('layer', 0)
                if layer <= 5:
                    input_nodes.append((node_id, imp_score))
                elif layer <= 15:
                    middle_nodes.append((node_id, imp_score))
                else:
                    output_nodes.append((node_id, imp_score))

        # Sort each category by importance
        input_nodes.sort(key=lambda x: x[1], reverse=True)
        middle_nodes.sort(key=lambda x: x[1], reverse=True)
        output_nodes.sort(key=lambda x: x[1], reverse=True)

        # Select according to percentages
        num_input = int(self.max_nodes * 0.30)
        num_middle = int(self.max_nodes * 0.40)
        num_output = int(self.max_nodes * 0.30)

        selected = []
        selected.extend([node_id for node_id, _ in input_nodes[:num_input]])
        selected.extend([node_id for node_id, _ in middle_nodes[:num_middle]])
        selected.extend([node_id for node_id, _ in output_nodes[:num_output]])

        return selected
