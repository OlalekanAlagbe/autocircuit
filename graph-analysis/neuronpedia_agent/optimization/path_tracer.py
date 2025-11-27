"""Path tracing module for understanding computational pathways"""

from typing import List
from dataclasses import dataclass
from ..analysis.graph_analyzer import GraphAnalyzer
from ..analysis.grouping_engine import Supernode


@dataclass
class ComputationPath:
    """Represents a computational pathway through supernodes"""
    supernode_sequence: List[Supernode]
    edge_weights: List[float]
    bottlenecks: List[Supernode]
    narrative: str


class PathTracer:
    """Trace and visualize computational pathways through the cleaned subgraph"""

    def __init__(self, graph_analyzer: GraphAnalyzer, supernodes: List[Supernode]):
        self.analyzer = graph_analyzer
        self.supernodes = supernodes

    def trace_computation(self, input_token: str, output_logit: str) -> ComputationPath:
        """
        Trace how information flows from input_token to output_logit through supernodes

        Returns: ComputationPath with:
        - supernode_sequence: List of supernodes in order
        - edge_weights: Influence between each supernode pair
        - bottlenecks: Critical supernodes that route most influence
        - narrative: Text description of the computation
        """
        # Identify which supernodes contain input/output features
        input_supernodes = []
        output_supernodes = []

        for snode in self.supernodes:
            # Check if any nodes in this supernode are input features
            for node_id in snode.node_ids:
                node = self.analyzer.get_node(node_id)
                if node and node.get('layer', 0) <= 5:
                    input_supernodes.append(snode)
                    break

            # Check if any nodes boost the target output
            for node_id in snode.node_ids:
                for edge in self.analyzer.edges:
                    if edge['source'] == node_id and output_logit in edge['target']:
                        output_supernodes.append(snode)
                        break

        # Build sequence by following influence paths
        sequence = []
        if input_supernodes:
            sequence.append(input_supernodes[0])

        # Add middle supernodes sorted by layer
        middle_supernodes = [
            s for s in self.supernodes
            if s not in input_supernodes and s not in output_supernodes
        ]
        middle_supernodes.sort(key=lambda s: s.layer_range[0])
        sequence.extend(middle_supernodes)

        if output_supernodes:
            sequence.append(output_supernodes[0])

        # Calculate edge weights between consecutive supernodes
        edge_weights = []
        for i in range(len(sequence) - 1):
            weight = self._calculate_influence_between_supernodes(sequence[i], sequence[i+1])
            edge_weights.append(weight)

        # Identify bottlenecks (supernodes with high influence)
        bottlenecks = [s for s in sequence if s.total_influence > 0.3]

        # Generate narrative
        narrative = self.generate_narrative(ComputationPath(
            supernode_sequence=sequence,
            edge_weights=edge_weights,
            bottlenecks=bottlenecks,
            narrative=""
        ))

        return ComputationPath(
            supernode_sequence=sequence,
            edge_weights=edge_weights,
            bottlenecks=bottlenecks,
            narrative=narrative
        )

    def _calculate_influence_between_supernodes(self, source: Supernode, target: Supernode) -> float:
        """Calculate total influence from source supernode to target supernode"""
        total_influence = 0.0

        for source_node in source.node_ids:
            for target_node in target.node_ids:
                for edge in self.analyzer.edges:
                    if edge['source'] == source_node and edge['target'] == target_node:
                        total_influence += edge['weight']

        return total_influence

    def generate_narrative(self, path: ComputationPath) -> str:
        """
        Create a human-readable explanation of the computational pathway

        Example: "The model detects 'Texas' in the input (supernode: Texas detection),
        retrieves the capital-state relationship (supernode: capital-of relation),
        and promotes 'Austin' in the output (supernode: say [capital city])."
        """
        if not path.supernode_sequence:
            return "No clear computational pathway found."

        narrative_parts = ["The model"]

        for i, snode in enumerate(path.supernode_sequence):
            if i == 0:
                narrative_parts.append(f"starts with {snode.label}")
            elif i == len(path.supernode_sequence) - 1:
                narrative_parts.append(f"and finally {snode.label}")
            else:
                narrative_parts.append(f"then {snode.label}")

        return " ".join(narrative_parts) + "."
