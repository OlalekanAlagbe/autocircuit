#!/usr/bin/env python3
"""
Query Neuronpedia API for detailed feature semantics.

This script fetches activation examples, descriptions, and patterns for specific features
to build semantic understanding of bottleneck features.

Usage:
    python query_feature_semantics.py --feature L5_F7993995
    python query_feature_semantics.py --layer 5 --feature-id 7993995
    python query_feature_semantics.py --analyze-bottleneck
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add scripts directory to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_description_fetcher import FeatureDescriptionFetcher


class FeatureSemanticAnalyzer:
    """Analyzes feature semantics from Neuronpedia API."""

    def __init__(self):
        self.fetcher = FeatureDescriptionFetcher()
        self.model = self.fetcher.model_name

    def query_single_feature(self, layer: int, feature_id: int) -> dict:
        """
        Query detailed information about a single feature.

        Returns dict with:
            - layer: Layer number
            - feature_id: Feature ID
            - label: Human-readable label (e.g., "L5_F7993995")
            - description: Text description from API
            - positive_examples: List of tokens that activate this feature
            - activation_count: Number of activation examples
            - semantic_category: Preliminary categorization (needs manual review)
        """
        print(f"\n{'='*70}")
        print(f"QUERYING FEATURE: L{layer}_F{feature_id}")
        print(f"{'='*70}\n")

        # Fetch from API
        description = self.fetcher.fetch_feature_description(layer, feature_id)

        # Parse activation examples from description
        positive_examples = []
        if description and "Activates on:" in description:
            # Extract tokens after "Activates on:"
            examples_str = description.split("Activates on:")[-1].strip()
            positive_examples = [ex.strip() for ex in examples_str.split(",")]

        # Preliminary semantic categorization
        semantic_category = self._categorize_feature(positive_examples, description)

        result = {
            "layer": layer,
            "feature_id": feature_id,
            "label": f"L{layer}_F{feature_id}",
            "description": description,
            "positive_examples": positive_examples,
            "activation_count": len(positive_examples),
            "semantic_category": semantic_category,
            "notes": []
        }

        # Print summary
        self._print_feature_summary(result)

        return result

    def _categorize_feature(self, examples: list, description: str) -> str:
        """
        Preliminary categorization based on activation patterns.

        Categories:
            - SYNTAX: Punctuation, code tokens, special characters
            - SEMANTICS: Meaningful words, concepts, entities
            - POLYSEMANTIC: Mixed patterns (both syntax and semantics)
            - UNKNOWN: Cannot determine from examples
        """
        if not examples:
            return "UNKNOWN"

        # Check for syntax patterns
        syntax_indicators = [
            '(', ')', '[', ']', '{', '}', ';', ':',
            'class', 'def', 'import', 'return',
            '▁', '_', '...', '"""', "'''"
        ]

        syntax_count = sum(1 for ex in examples
                          if any(ind in ex.lower() for ind in syntax_indicators))

        # Check for semantic patterns
        semantic_indicators = [
            # Geographic
            'country', 'city', 'state', 'capital', 'region',
            # Entities
            'person', 'organization', 'location', 'name',
            # Concepts
            'idea', 'concept', 'theory', 'principle'
        ]

        semantic_count = sum(1 for ex in examples
                            if any(ind in ex.lower() for ind in semantic_indicators))

        # Categorize
        if syntax_count > len(examples) * 0.7:
            return "SYNTAX"
        elif semantic_count > len(examples) * 0.5:
            return "SEMANTICS"
        elif syntax_count > 0 and semantic_count > 0:
            return "POLYSEMANTIC"
        else:
            # Check actual content
            has_meaningful_words = any(len(ex.strip()) > 3 and ex.isalpha()
                                      for ex in examples)
            return "SEMANTICS" if has_meaningful_words else "SYNTAX"

    def _print_feature_summary(self, result: dict):
        """Print formatted summary of feature."""
        print(f"Feature: {result['label']}")
        print(f"Category: {result['semantic_category']}")
        print(f"\nDescription:")
        desc = result['description'] or "No description available"
        print(f"  {desc}")
        print(f"\nActivation Examples ({result['activation_count']} total):")
        for i, ex in enumerate(result['positive_examples'][:10], 1):
            print(f"  {i}. {ex}")

        if result['activation_count'] > 10:
            print(f"  ... and {result['activation_count'] - 10} more")

    def analyze_layer_features(self, layer: int, feature_ids: list) -> list:
        """Analyze multiple features in a layer."""
        print(f"\n{'='*70}")
        print(f"ANALYZING {len(feature_ids)} FEATURES IN LAYER {layer}")
        print(f"{'='*70}")

        results = []
        for feature_id in feature_ids:
            result = self.query_single_feature(layer, feature_id)
            results.append(result)
            print()  # Spacing between features

        # Summary statistics
        self._print_layer_summary(layer, results)

        return results

    def _print_layer_summary(self, layer: int, results: list):
        """Print summary statistics for a layer."""
        print(f"\n{'='*70}")
        print(f"LAYER {layer} SUMMARY")
        print(f"{'='*70}\n")

        # Count by category
        categories = {}
        for r in results:
            cat = r['semantic_category']
            categories[cat] = categories.get(cat, 0) + 1

        print("Semantic Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            pct = (count / len(results)) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")

    def analyze_bottleneck(self, bottleneck_file: str):
        """
        Analyze bottleneck feature from traceback analysis.

        Args:
            bottleneck_file: Path to traceback_paths.json
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING BOTTLENECK FROM: {bottleneck_file}")
        print(f"{'='*70}\n")

        # Load traceback data
        with open(bottleneck_file, 'r') as f:
            data = json.load(f)

        # Find bottleneck feature(s)
        # Look for features that appear in all paths
        if 'critical_paths' not in data:
            print("ERROR: No critical_paths found in traceback file")
            return None

        # Count feature appearances across paths
        feature_counts = {}
        total_paths = len(data['critical_paths'])

        for path in data['critical_paths']:
            path_nodes = path.get('path_nodes', [])
            for node in path_nodes:
                # Node format: ["node_id", layer, feature_id, ...]
                if len(node) >= 3:
                    layer = node[1]
                    feature_id = node[2]
                    key = (layer, feature_id)
                    feature_counts[key] = feature_counts.get(key, 0) + 1

        # Find features in ALL paths (100% convergence)
        bottleneck_features = [(layer, fid) for (layer, fid), count in feature_counts.items()
                               if count == total_paths]

        print(f"Found {len(bottleneck_features)} bottleneck features (100% convergence)")
        print()

        # Analyze each bottleneck
        results = []
        for layer, feature_id in sorted(bottleneck_features):
            result = self.query_single_feature(layer, feature_id)
            result['convergence'] = 1.0  # 100%
            result['paths_containing'] = total_paths
            results.append(result)

        return results

    def export_results(self, results: list, output_file: str):
        """Export results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[SAVED] Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Neuronpedia API for feature semantics"
    )
    parser.add_argument(
        '--feature',
        help='Feature label (e.g., L5_F7993995)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        help='Layer number'
    )
    parser.add_argument(
        '--feature-id',
        type=int,
        help='Feature ID within layer'
    )
    parser.add_argument(
        '--analyze-bottleneck',
        action='store_true',
        help='Analyze bottleneck features from traceback'
    )
    parser.add_argument(
        '--traceback-file',
        default=str(Path(__file__).parent.parent.parent / 'data' / 'prompts' / 'gemma-2-2b_water-boils-at-100-degrees' / '3_analysis' / 'traceback_paths.json'),
        help='Path to traceback_paths.json'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file for results'
    )

    args = parser.parse_args()

    analyzer = FeatureSemanticAnalyzer()

    # Parse feature label if provided
    if args.feature:
        # Format: L5_F7993995
        parts = args.feature.replace('L', '').split('_F')
        if len(parts) == 2:
            args.layer = int(parts[0])
            args.feature_id = int(parts[1])

    # Execute based on arguments
    results = None

    if args.analyze_bottleneck:
        results = analyzer.analyze_bottleneck(args.traceback_file)
    elif args.layer is not None and args.feature_id is not None:
        result = analyzer.query_single_feature(args.layer, args.feature_id)
        results = [result]
    else:
        parser.print_help()
        sys.exit(1)

    # Export if requested
    if results and args.output:
        analyzer.export_results(results, args.output)


if __name__ == '__main__':
    main()
