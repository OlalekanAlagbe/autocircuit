#!/usr/bin/env python3
"""
Batch query features from Neuronpedia API for semantic taxonomy building.

This script queries multiple features across layers and saves results to CSV.
"""

import sys
import os
import json
import csv
import io
from pathlib import Path

# Fix Windows console encoding (only if not already wrapped)
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add scripts directory to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also add current directory so query_feature_semantics (sibling in advanced_analysis/) can be found
sys.path.insert(0, str(Path(__file__).parent))

from query_feature_semantics import FeatureSemanticAnalyzer


def batch_query_features(layers, feature_ids, output_csv):
    """
    Query multiple features and save to CSV.

    Args:
        layers: List of layer numbers to query
        feature_ids: List of feature IDs to query per layer
        output_csv: Path to output CSV file
    """
    analyzer = FeatureSemanticAnalyzer()

    results = []
    total = len(layers) * len(feature_ids)
    current = 0

    print(f"Querying {total} features ({len(layers)} layers × {len(feature_ids)} features/layer)...")
    print()

    for layer in layers:
        for feat_id in feature_ids:
            current += 1
            print(f"[{current}/{total}] Querying L{layer}_F{feat_id}...", end=" ")

            try:
                result = analyzer.query_single_feature(layer, feat_id)

                # Flatten for CSV
                row = {
                    'feature_label': result['label'],
                    'layer': result['layer'],
                    'feature_id': result['feature_id'],
                    'activation_examples': '; '.join(result['positive_examples']),
                    'activation_count': result['activation_count'],
                    'auto_category': result['semantic_category'],
                    'manual_category': '',  # To be filled in manually
                    'confidence': '',  # To be filled in manually
                    'notes': ''  # To be filled in manually
                }

                results.append(row)
                print(f"✓ {result['semantic_category']}")

            except Exception as e:
                print(f"✗ Error: {e}")
                # Add placeholder row for failed queries
                results.append({
                    'feature_label': f'L{layer}_F{feat_id}',
                    'layer': layer,
                    'feature_id': feat_id,
                    'activation_examples': 'ERROR',
                    'activation_count': 0,
                    'auto_category': 'UNKNOWN',
                    'manual_category': '',
                    'confidence': '',
                    'notes': f'Query failed: {e}'
                })

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print()
    print(f"✓ Saved {len(results)} features to: {output_csv}")
    print()
    print("Summary:")

    # Category distribution
    from collections import Counter
    categories = Counter(r['auto_category'] for r in results)
    for cat, count in categories.most_common():
        pct = (count / len(results)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    return results


if __name__ == '__main__':
    # Define sampling strategy - EXPANDED TO 80 FEATURES
    # Stage 1.1: Query 10 features per layer × 8 layers = 80 features total
    layers = [0, 3, 6, 9, 12, 15, 20, 25]  # 8 layers (increased density)
    # Sample across full 16k range
    feature_ids = [100, 500, 1000, 3000, 5000, 8000, 10000, 12000, 14000, 15000]  # 10 features per layer

    # Use absolute path based on script location (scripts/advanced_analysis/ -> scripts/)
    output_csv = str(Path(__file__).parent.parent / 'semantic_taxonomy_annotations.csv')

    print("=" * 70)
    print("BATCH FEATURE QUERY FOR SEMANTIC TAXONOMY")
    print("=" * 70)
    print()

    results = batch_query_features(layers, feature_ids, output_csv)

    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review the CSV file and manually verify auto_category assignments")
    print("2. Fill in manual_category column with your classifications")
    print("3. Add confidence ratings (LOW/MEDIUM/HIGH)")
    print("4. Add notes for edge cases or interesting findings")
    print()
    print(f"CSV file: {output_csv}")
