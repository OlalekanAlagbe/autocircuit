"""Metrics calculation module for evaluating subgraph quality"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of subgraph validation"""
    passed: bool
    replacement_score: float
    completeness_score: float
    num_supernodes: int
    suggestions: list


class MetricsCalculator:
    """Compute quality metrics for the cleaned subgraph"""

    def __init__(self, full_graph: Dict, subgraph: Dict):
        self.full_graph = full_graph
        self.subgraph = subgraph

    def compute_replacement_score(self) -> float:
        """
        Replacement = (end-to-end influence through pinned features) / (total influence)

        Algorithm:
        1. Compute total influence from input tokens to target logits in full graph
        2. Compute influence that flows through pinned nodes in subgraph
        3. Return ratio
        """
        # Get pinned node IDs
        pinned_ids = set(self.subgraph.get('pinned_node_ids', []))

        # Calculate total influence in full graph to logits
        total_influence = 0.0
        for edge in self.full_graph.get('edges', []):
            if edge['target'].startswith('logit_'):
                total_influence += edge['weight']

        # Calculate influence through pinned nodes
        pinned_influence = 0.0
        for edge in self.full_graph.get('edges', []):
            if edge['source'] in pinned_ids and edge['target'].startswith('logit_'):
                pinned_influence += edge['weight']
            elif edge['source'] in pinned_ids and edge['target'] in pinned_ids:
                # Indirect influence through other pinned nodes
                for edge2 in self.full_graph.get('edges', []):
                    if edge2['source'] == edge['target'] and edge2['target'].startswith('logit_'):
                        pinned_influence += edge['weight'] * edge2['weight']

        if total_influence == 0:
            return 0.0

        return min(pinned_influence / total_influence, 1.0)

    def compute_completeness_score(self) -> float:
        """
        Completeness = (incoming edge influence explained by subgraph) / (total incoming influence)

        Algorithm:
        1. For each pinned node, sum incoming edge weights
        2. For each pinned node, sum incoming edge weights from OTHER pinned nodes
        3. Return ratio averaged across all pinned nodes
        """
        pinned_ids = set(self.subgraph.get('pinned_node_ids', []))

        if not pinned_ids:
            return 0.0

        total_completeness = 0.0
        num_nodes = 0

        for node_id in pinned_ids:
            # Sum all incoming edges to this node
            total_incoming = 0.0
            explained_incoming = 0.0

            for edge in self.full_graph.get('edges', []):
                if edge['target'] == node_id:
                    total_incoming += edge['weight']
                    if edge['source'] in pinned_ids:
                        explained_incoming += edge['weight']

            if total_incoming > 0:
                total_completeness += explained_incoming / total_incoming
                num_nodes += 1

        if num_nodes == 0:
            return 0.0

        return total_completeness / num_nodes

    def validate_subgraph(self, min_replacement: float = 0.5, min_completeness: float = 0.7) -> ValidationResult:
        """
        Check if subgraph meets quality thresholds:
        - Replacement > 0.5
        - Completeness > 0.7
        - Number of supernodes: 3-7
        - Interpretability: Can generate coherent narrative

        Returns: ValidationResult with pass/fail and suggestions
        """
        replacement = self.compute_replacement_score()
        completeness = self.compute_completeness_score()
        num_supernodes = len(self.subgraph.get('supernodes', []))

        passed = True
        suggestions = []

        if replacement < min_replacement:
            passed = False
            suggestions.append(
                f"Replacement score ({replacement:.2f}) below threshold ({min_replacement}). "
                "Consider pinning more influential nodes."
            )

        if completeness < min_completeness:
            passed = False
            suggestions.append(
                f"Completeness score ({completeness:.2f}) below threshold ({min_completeness}). "
                "Consider adding nodes that connect existing pinned nodes."
            )

        if num_supernodes < 3:
            suggestions.append(
                f"Only {num_supernodes} supernodes. Consider breaking large groups into smaller units."
            )
        elif num_supernodes > 7:
            suggestions.append(
                f"{num_supernodes} supernodes may be too many for interpretability. "
                "Consider merging related groups."
            )

        return ValidationResult(
            passed=passed,
            replacement_score=replacement,
            completeness_score=completeness,
            num_supernodes=num_supernodes,
            suggestions=suggestions
        )
