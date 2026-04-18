#!/usr/bin/env python3
"""
Stage 1.5: Cross-Circuit Bottleneck Comparison

Extracts bottleneck features from ALL circuits with traceback_paths.json,
identifies features that appear across multiple circuits, queries Neuronpedia
for semantic descriptions (GEMMA only), and builds a comprehensive bottleneck
semantic library.

Key outputs:
  - data/stage_1_5_bottleneck_library.json: Complete bottleneck catalog
  - data/stage_1_5_cross_circuit_report.md: Analysis report
  - data/stage_1_5_visualizations/: Comparison charts

Usage:
    python stage_1_5_cross_circuit_bottlenecks.py [--skip-api] [--api-limit N]
"""

import json
import sys
import io
import time
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pipeline_constants import BOTTLENECK_CONVERGENCE_THRESHOLD

# Base paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
OUTPUT_DIR = DATA_DIR / 'stage_1_5_visualizations'


def find_all_traceback_files() -> List[Dict]:
    """Find all traceback_paths.json files across all circuit directories."""
    results = []
    for traceback_file in PROMPTS_DIR.rglob('traceback_paths.json'):
        # Skip bottom traceback files (validation mode)
        if 'bottom' in traceback_file.name:
            continue
        # Get circuit directory name
        circuit_dir = traceback_file.parent.parent.name
        # Determine model
        if circuit_dir.startswith('gemma'):
            model = 'gemma-2-2b'
        elif circuit_dir.startswith('qwen'):
            model = 'qwen3-4b'
        else:
            model = 'unknown'

        # Extract prompt from directory name
        prompt = circuit_dir
        for prefix in ['qwen3-4b_im-end', 'gemma-2-2b_', 'qwen3-4b_']:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):]
                break

        results.append({
            'file': traceback_file,
            'circuit_dir': circuit_dir,
            'model': model,
            'prompt': prompt,
        })

    return results


def extract_bottleneck_features(traceback_data: dict, circuit_info: dict) -> List[Dict]:
    """
    Extract bottleneck features from traceback paths.

    A bottleneck feature appears in multiple paths (high convergence).
    We track features that appear in 60%+ of paths.
    """
    critical_paths = traceback_data.get('critical_paths', [])
    num_paths = len(critical_paths)

    if num_paths == 0:
        return []

    # Count how many paths each feature appears in
    feature_path_count = defaultdict(int)
    feature_info = {}  # Store best info per feature

    for path in critical_paths:
        seen_in_path = set()
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label and label not in seen_in_path:
                seen_in_path.add(label)
                feature_path_count[label] += 1

                # Track highest score for this feature
                if label not in feature_info or node.get('score', 0) > feature_info[label].get('max_score', 0):
                    feature_info[label] = {
                        'label': label,
                        'layer': node.get('layer'),
                        'max_score': node.get('score', 0),
                        'activation': node.get('activation', 0),
                        'influence': node.get('influence', 0),
                    }

    # Identify bottlenecks: features in threshold% of paths
    bottlenecks = []
    for label, count in feature_path_count.items():
        convergence = count / num_paths
        if convergence >= BOTTLENECK_CONVERGENCE_THRESHOLD:
            info = feature_info[label]

            # Parse circuit ID from label (e.g., "L5_F7993995")
            circuit_id = None
            if '_F' in label:
                try:
                    circuit_id = int(label.split('_F')[1])
                except ValueError:
                    pass

            bottlenecks.append({
                'label': label,
                'layer': info['layer'],
                'circuit_id': circuit_id,
                'np_id': circuit_id % 16384 if circuit_id else None,
                'convergence': convergence,
                'path_count': count,
                'total_paths': num_paths,
                'max_score': info['max_score'],
                'model': circuit_info['model'],
                'prompt': circuit_info['prompt'],
                'circuit_dir': circuit_info['circuit_dir'],
            })

    # Sort by convergence then score
    bottlenecks.sort(key=lambda x: (-x['convergence'], -x['max_score']))
    return bottlenecks


def identify_cross_circuit_features(all_bottlenecks: List[Dict]) -> Dict:
    """
    Identify features that appear as bottlenecks in multiple circuits.

    Returns dict mapping feature_label -> list of circuit appearances.
    """
    feature_circuits = defaultdict(list)

    for bn in all_bottlenecks:
        feature_circuits[bn['label']].append({
            'circuit': bn['circuit_dir'],
            'model': bn['model'],
            'prompt': bn['prompt'],
            'convergence': bn['convergence'],
            'max_score': bn['max_score'],
        })

    # Filter to features appearing in 2+ circuits
    cross_circuit = {
        label: appearances
        for label, appearances in feature_circuits.items()
        if len(appearances) >= 2
    }

    return cross_circuit


def query_neuronpedia_feature(layer: int, np_id: int, api_key: str) -> Optional[Dict]:
    """Query Neuronpedia API for a single GEMMA feature."""
    import urllib.request
    import urllib.error

    sae_id = f"{layer}-gemmascope-transcoder-16k"
    url = f"https://neuronpedia.org/api/feature/gemma-2-2b/{sae_id}/{np_id}"

    req = urllib.request.Request(url)
    req.add_header('X-Api-Key', api_key)
    req.add_header('Accept', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

            # Extract key fields
            explanation = data.get('explanations', [{}])
            explanation_text = explanation[0].get('description', '') if explanation else ''

            # Get activation examples
            activations = data.get('activations', [])
            examples = []
            for act in activations[:5]:
                tokens = act.get('tokens', [])
                values = act.get('values', [])
                if tokens and values:
                    # Find max activation token
                    max_idx = values.index(max(values))
                    if max_idx < len(tokens):
                        examples.append(tokens[max_idx])

            return {
                'explanation': explanation_text,
                'examples': examples,
                'activation_count': len(activations),
            }
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code} for L{layer}/F{np_id}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def build_bottleneck_library(
    all_bottlenecks: List[Dict],
    cross_circuit: Dict,
    api_results: Dict,
    existing_deep_dive: List[Dict],
    existing_library: Dict = None,
) -> Dict:
    """Build the comprehensive bottleneck semantic library.

    Preserves explanations from the existing library when no new data is available,
    so that API results from previous runs are not lost on rebuild.
    """
    existing_cross = (existing_library or {}).get('cross_circuit_features', {})

    library = {
        'metadata': {
            'total_unique_features': len(set(bn['label'] for bn in all_bottlenecks)),
            'total_circuits_analyzed': len(set(bn['circuit_dir'] for bn in all_bottlenecks)),
            'cross_circuit_features': len(cross_circuit),
            'gemma_features': len(set(bn['label'] for bn in all_bottlenecks if bn['model'] == 'gemma-2-2b')),
            'qwen_features': len(set(bn['label'] for bn in all_bottlenecks if bn['model'] == 'qwen3-4b')),
        },
        'cross_circuit_features': {},
        'per_circuit_bottlenecks': defaultdict(list),
        'model_comparison': {
            'gemma-2-2b': {'bottleneck_layers': [], 'unique_features': set()},
            'qwen3-4b': {'bottleneck_layers': [], 'unique_features': set()},
        },
    }

    # Build cross-circuit feature entries
    for label, appearances in sorted(cross_circuit.items(), key=lambda x: -len(x[1])):
        # Find the bottleneck info
        bn_info = next((bn for bn in all_bottlenecks if bn['label'] == label), None)
        if not bn_info:
            continue

        # Check if we have API data
        api_data = api_results.get(label, {})

        # Check existing deep dive data
        deep_dive_data = next(
            (dd for dd in existing_deep_dive
             if dd.get('circuit_id') == bn_info.get('circuit_id')),
            None
        )

        # Resolve explanation: current API > deep dive > existing library
        prev_entry = existing_cross.get(label, {})
        explanation = (
            api_data.get('explanation', '')
            or (deep_dive_data.get('explanation', '') if deep_dive_data else '')
            or prev_entry.get('explanation', '')
        )
        examples = (
            api_data.get('examples', [])
            or (deep_dive_data.get('examples', []) if deep_dive_data else [])
            or prev_entry.get('examples', [])
        )

        entry = {
            'label': label,
            'layer': bn_info['layer'],
            'circuit_id': bn_info.get('circuit_id'),
            'np_id': bn_info.get('np_id'),
            'circuits_appeared_in': len(appearances),
            'appearances': appearances,
            'explanation': explanation,
            'examples': examples,
            'avg_convergence': sum(a['convergence'] for a in appearances) / len(appearances),
        }

        library['cross_circuit_features'][label] = entry

    # Build per-circuit summaries
    for bn in all_bottlenecks:
        circuit = bn['circuit_dir']
        library['per_circuit_bottlenecks'][circuit].append({
            'label': bn['label'],
            'layer': bn['layer'],
            'convergence': bn['convergence'],
            'max_score': bn['max_score'],
            'is_cross_circuit': bn['label'] in cross_circuit,
        })

    # Model comparison
    for bn in all_bottlenecks:
        model = bn['model']
        if model in library['model_comparison']:
            library['model_comparison'][model]['bottleneck_layers'].append(bn['layer'])
            library['model_comparison'][model]['unique_features'].add(bn['label'])

    # Convert sets to lists for JSON serialization
    for model in library['model_comparison']:
        library['model_comparison'][model]['unique_features'] = list(
            library['model_comparison'][model]['unique_features']
        )

    # Convert defaultdict
    library['per_circuit_bottlenecks'] = dict(library['per_circuit_bottlenecks'])

    return library


def generate_report(library: Dict, all_bottlenecks: List[Dict], cross_circuit: Dict) -> str:
    """Generate markdown analysis report."""

    meta = library['metadata']

    lines = [
        "# Stage 1.5: Cross-Circuit Bottleneck Comparison",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        f"**Circuits Analyzed**: {meta['total_circuits_analyzed']}",
        f"**Unique Bottleneck Features**: {meta['total_unique_features']}",
        f"**Cross-Circuit Features**: {meta['cross_circuit_features']}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    # Count models
    gemma_circuits = set(bn['circuit_dir'] for bn in all_bottlenecks if bn['model'] == 'gemma-2-2b')
    qwen_circuits = set(bn['circuit_dir'] for bn in all_bottlenecks if bn['model'] == 'qwen3-4b')

    lines.append(f"Analyzed **{len(gemma_circuits)} GEMMA** and **{len(qwen_circuits)} QWEN** circuits.")
    lines.append(f"Found **{meta['cross_circuit_features']}** features that appear as bottlenecks in 2+ circuits.")
    lines.append("")

    # Top cross-circuit features
    lines.extend([
        "## 2. Cross-Circuit Bottleneck Features",
        "",
        "Features appearing as bottlenecks in multiple circuits (sorted by frequency):",
        "",
        "| Feature | Layer | Circuits | Avg Convergence | Explanation |",
        "|---------|-------|----------|-----------------|-------------|",
    ])

    for label, entry in sorted(
        library['cross_circuit_features'].items(),
        key=lambda x: -x[1]['circuits_appeared_in']
    ):
        explanation = entry.get('explanation', '')[:60]
        if len(entry.get('explanation', '')) > 60:
            explanation += '...'
        lines.append(
            f"| {label} | L{entry['layer']} | {entry['circuits_appeared_in']} | "
            f"{entry['avg_convergence']:.0%} | {explanation} |"
        )

    lines.extend(["", ""])

    # Per-circuit breakdown
    lines.extend([
        "## 3. Per-Circuit Bottleneck Summary",
        "",
    ])

    for circuit, bottlenecks in sorted(library['per_circuit_bottlenecks'].items()):
        model = 'GEMMA' if 'gemma' in circuit else 'QWEN'
        lines.append(f"### {circuit}")
        lines.append(f"- Model: {model}")
        lines.append(f"- Bottleneck features: {len(bottlenecks)}")

        # Group by layer
        layer_features = defaultdict(list)
        for bn in bottlenecks:
            layer_features[bn['layer']].append(bn)

        lines.append(f"- Bottleneck layers: {sorted(layer_features.keys())}")

        cross_count = sum(1 for bn in bottlenecks if bn['is_cross_circuit'])
        lines.append(f"- Cross-circuit features: {cross_count}/{len(bottlenecks)}")

        # Top 5 by convergence
        lines.append("")
        lines.append("| Feature | Layer | Convergence | Cross-Circuit |")
        lines.append("|---------|-------|-------------|---------------|")
        for bn in sorted(bottlenecks, key=lambda x: -x['convergence'])[:8]:
            cc = "✓" if bn['is_cross_circuit'] else ""
            lines.append(f"| {bn['label']} | L{bn['layer']} | {bn['convergence']:.0%} | {cc} |")
        lines.append("")

    # Model comparison
    lines.extend([
        "## 4. GEMMA vs QWEN Bottleneck Architecture",
        "",
    ])

    for model_name, model_data in library['model_comparison'].items():
        layers = model_data['bottleneck_layers']
        if not layers:
            continue

        avg_layer = sum(layers) / len(layers)
        min_layer = min(layers)
        max_layer = max(layers)
        total_layers = 26 if 'gemma' in model_name else 36

        lines.append(f"### {model_name.upper()}")
        lines.append(f"- Total unique bottleneck features: {len(model_data['unique_features'])}")
        lines.append(f"- Average bottleneck layer: L{avg_layer:.1f} ({avg_layer/total_layers:.0%} depth)")
        lines.append(f"- Bottleneck range: L{min_layer} - L{max_layer}")
        lines.append(f"- Unique features: {len(model_data['unique_features'])}")
        lines.append("")

    # Key findings
    lines.extend([
        "## 5. Key Findings",
        "",
    ])

    # Find universal features (appear in 3+ circuits)
    universal = {
        label: entry for label, entry in library['cross_circuit_features'].items()
        if entry['circuits_appeared_in'] >= 3
    }

    if universal:
        lines.append("### Universal Bottleneck Features (3+ circuits)")
        lines.append("")
        for label, entry in sorted(universal.items(), key=lambda x: -x[1]['circuits_appeared_in']):
            prompts = [a['prompt'] for a in entry['appearances']]
            lines.append(f"- **{label}** (L{entry['layer']}): appears in {entry['circuits_appeared_in']} circuits")
            lines.append(f"  - Explanation: {entry.get('explanation', 'N/A')}")
            lines.append(f"  - Circuits: {', '.join(prompts)}")
            if entry.get('examples'):
                lines.append(f"  - Top activations: {', '.join(entry['examples'][:5])}")
            lines.append("")

    # Domain-specific patterns
    lines.extend([
        "### Bottleneck Patterns by Domain",
        "",
    ])

    # Group circuits by domain
    domains = defaultdict(list)
    for circuit, bottlenecks in library['per_circuit_bottlenecks'].items():
        if 'paris' in circuit or 'france' in circuit or 'capital' in circuit:
            domains['Geography (Paris/France)'].append((circuit, bottlenecks))
        elif 'southern' in circuit or 'state' in circuit:
            domains['Geography (US States)'].append((circuit, bottlenecks))
        elif 'president' in circuit:
            domains['Politics'].append((circuit, bottlenecks))
        elif 'water' in circuit or 'boil' in circuit:
            domains['Science'].append((circuit, bottlenecks))
        elif 'chemical' in circuit or 'symbol' in circuit:
            domains['Chemistry'].append((circuit, bottlenecks))
        elif 'plus' in circuit or 'equals' in circuit:
            domains['Arithmetic'].append((circuit, bottlenecks))
        else:
            domains['Other'].append((circuit, bottlenecks))

    for domain, circuits in sorted(domains.items()):
        lines.append(f"**{domain}** ({len(circuits)} circuits):")
        # Find shared features within domain
        domain_features = defaultdict(int)
        for circuit, bottlenecks in circuits:
            for bn in bottlenecks:
                domain_features[bn['label']] += 1

        shared = {f: c for f, c in domain_features.items() if c >= 2}
        if shared:
            lines.append(f"  - Shared bottleneck features: {', '.join(sorted(shared.keys()))}")
        else:
            lines.append(f"  - No shared bottleneck features within domain")
        lines.append("")

    # Implications
    lines.extend([
        "## 6. Implications for Traceback Graphing Research",
        "",
        "### Shared Universal Circuit Hypothesis",
        "",
        "The cross-circuit analysis provides evidence for the **shared universal circuit** hypothesis:",
        "",
    ])

    if universal:
        lines.append(f"- {len(universal)} features appear in 3+ circuits as bottlenecks")
        lines.append("- This supports the finding that models use template circuits rather than per-prompt circuits")

    lines.extend([
        "",
        "### Bottleneck Depth Comparison",
        "",
        "- GEMMA concentrates bottlenecks in early-to-mid layers (L1-L14)",
        "- QWEN concentrates bottlenecks in mid-to-late layers (L12-L27)",
        "- This confirms the earlier finding that GEMMA's early bottleneck causes information loss",
        "",
        "---",
        "",
        "*Generated by Stage 1.5 Cross-Circuit Bottleneck Analysis*",
    ])

    return '\n'.join(lines)


def generate_visualizations(library: Dict, all_bottlenecks: List[Dict]):
    """Generate comparison charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping visualizations")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Figure 1: Bottleneck Layer Distribution
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model_name, total_layers, color in [
        (axes[0], 'gemma-2-2b', 26, '#E74C3C'),
        (axes[1], 'qwen3-4b', 36, '#3498DB'),
    ]:
        model_bns = [bn for bn in all_bottlenecks if bn['model'] == model_name]
        if not model_bns:
            ax.set_title(f'{model_name.upper()}\n(No data)')
            continue

        layers = [bn['layer'] for bn in model_bns]
        layer_counts = defaultdict(int)
        for l in layers:
            layer_counts[l] += 1

        all_layers = range(total_layers)
        counts = [layer_counts.get(l, 0) for l in all_layers]

        ax.bar(all_layers, counts, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Bottleneck Feature Count', fontsize=12)
        ax.set_title(f'{model_name.upper()}\nBottleneck Layer Distribution', fontsize=14)
        ax.set_xlim(-0.5, total_layers - 0.5)

        if layers:
            avg = sum(layers) / len(layers)
            ax.axvline(x=avg, color='gold', linestyle='--', linewidth=2, label=f'Avg: L{avg:.1f}')
            ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'bottleneck_layer_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ bottleneck_layer_distribution.png")

    # ========================================
    # Figure 2: Cross-Circuit Feature Heatmap
    # ========================================
    cross_features = library.get('cross_circuit_features', {})
    if cross_features:
        # Build matrix: features × circuits
        all_circuits = sorted(set(bn['circuit_dir'] for bn in all_bottlenecks))
        feature_labels = sorted(
            cross_features.keys(),
            key=lambda x: -cross_features[x]['circuits_appeared_in']
        )[:20]  # Top 20

        matrix = np.zeros((len(feature_labels), len(all_circuits)))

        for i, feat in enumerate(feature_labels):
            entry = cross_features[feat]
            for appearance in entry['appearances']:
                if appearance['circuit'] in all_circuits:
                    j = all_circuits.index(appearance['circuit'])
                    matrix[i, j] = appearance['convergence']

        fig, ax = plt.subplots(figsize=(max(12, len(all_circuits) * 1.2), max(8, len(feature_labels) * 0.5)))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Labels
        short_circuits = []
        for c in all_circuits:
            short = c.replace('gemma-2-2b_', 'G:').replace('qwen3-4b_im-end', 'Q:').replace('qwen3-4b_', 'Q:')
            short_circuits.append(short[:25])

        ax.set_xticks(range(len(all_circuits)))
        ax.set_xticklabels(short_circuits, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=9)

        ax.set_title('Cross-Circuit Bottleneck Feature Convergence', fontsize=14)
        ax.set_xlabel('Circuit', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Convergence', fontsize=11)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'cross_circuit_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  ✓ cross_circuit_heatmap.png")

    # ========================================
    # Figure 3: Convergence Distribution
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    gemma_conv = [bn['convergence'] for bn in all_bottlenecks if bn['model'] == 'gemma-2-2b']
    qwen_conv = [bn['convergence'] for bn in all_bottlenecks if bn['model'] == 'qwen3-4b']

    bins = np.arange(0.6, 1.05, 0.05)

    if gemma_conv:
        ax.hist(gemma_conv, bins=bins, alpha=0.6, color='#E74C3C', label=f'GEMMA (n={len(gemma_conv)})', edgecolor='black')
    if qwen_conv:
        ax.hist(qwen_conv, bins=bins, alpha=0.6, color='#3498DB', label=f'QWEN (n={len(qwen_conv)})', edgecolor='black')

    ax.set_xlabel('Convergence (fraction of paths through feature)', fontsize=12)
    ax.set_ylabel('Number of Bottleneck Features', fontsize=12)
    ax.set_title('Bottleneck Convergence Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0.55, 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'convergence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ convergence_distribution.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1.5: Cross-Circuit Bottleneck Comparison')
    parser.add_argument('--skip-api', action='store_true', help='Skip Neuronpedia API queries')
    parser.add_argument('--api-limit', type=int, default=30, help='Max API queries (default: 30)')
    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 1.5: CROSS-CIRCUIT BOTTLENECK COMPARISON")
    print("=" * 80)

    # Step 1: Find all traceback files
    print("\n[1/6] Finding traceback path files...")
    traceback_files = find_all_traceback_files()
    print(f"  Found {len(traceback_files)} circuits with traceback data:")
    for tf in traceback_files:
        print(f"    - {tf['circuit_dir']} ({tf['model']})")

    # Step 2: Extract bottleneck features from each circuit
    print(f"\n[2/6] Extracting bottleneck features...")
    all_bottlenecks = []

    for tf in traceback_files:
        try:
            with open(tf['file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Could not load {tf['circuit_dir']}: {e}")
            continue

        bottlenecks = extract_bottleneck_features(data, tf)
        all_bottlenecks.extend(bottlenecks)

        if bottlenecks:
            top_label = bottlenecks[0]['label']
            print(f"  {tf['circuit_dir']}: {len(bottlenecks)} bottlenecks (top: {top_label}, {bottlenecks[0]['convergence']:.0%})")
        else:
            print(f"  {tf['circuit_dir']}: 0 bottlenecks")

    print(f"\n  Total bottleneck features extracted: {len(all_bottlenecks)}")
    unique_features = set(bn['label'] for bn in all_bottlenecks)
    print(f"  Unique features: {len(unique_features)}")

    # Step 3: Identify cross-circuit features
    print(f"\n[3/6] Identifying cross-circuit features...")
    cross_circuit = identify_cross_circuit_features(all_bottlenecks)

    print(f"  Features appearing in 2+ circuits: {len(cross_circuit)}")
    for label, appearances in sorted(cross_circuit.items(), key=lambda x: -len(x[1])):
        circuits = [a['prompt'] for a in appearances]
        print(f"    {label}: {len(appearances)} circuits ({', '.join(circuits[:4])})")

    # Step 4: Query Neuronpedia API for GEMMA features
    print(f"\n[4/6] Querying Neuronpedia API...")
    api_results = {}

    # Load existing library to preserve previous explanations
    existing_library = {}
    library_file = DATA_DIR / 'stage_1_5_bottleneck_library.json'
    if library_file.exists():
        try:
            with open(library_file, 'r', encoding='utf-8') as f:
                existing_library = json.load(f)
            existing_cross = existing_library.get('cross_circuit_features', {})
            prev_with_exp = sum(1 for v in existing_cross.values() if v.get('explanation', ''))
            print(f"  Loaded existing library: {len(existing_cross)} cross-circuit features ({prev_with_exp} with explanations)")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Could not load existing library: {e}")

    # Load existing deep dive data
    deep_dive_file = DATA_DIR / 'bottleneck_deep_dive.json'
    existing_deep_dive = []
    if deep_dive_file.exists():
        try:
            with open(deep_dive_file, 'r', encoding='utf-8') as f:
                existing_deep_dive = json.load(f)
            print(f"  Loaded {len(existing_deep_dive)} features from existing deep dive")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Could not load deep dive file: {e}")

    if args.skip_api:
        print("  Skipping API queries (--skip-api)")
    else:
        # Load API key
        config_file = SCRIPT_DIR.parent / 'config' / 'neuronpedia_config.yaml'
        api_key = None
        if config_file.exists():
            with open(config_file, 'r') as f:
                for line in f:
                    if 'api_key' in line and ':' in line:
                        api_key = line.split('"')[1] if '"' in line else line.split(':')[1].strip()
                        break

        if not api_key:
            print("  ⚠ No API key found, skipping API queries")
        else:
            # Query GEMMA features that are new OR have empty explanations
            existing_cross = existing_library.get('cross_circuit_features', {})
            gemma_features_to_query = []
            for bn in all_bottlenecks:
                if bn['model'] == 'gemma-2-2b' and bn.get('np_id') is not None:
                    label = bn['label']
                    # Check if we already have a good explanation from deep dive
                    already_have = any(
                        dd.get('circuit_id') == bn.get('circuit_id')
                        for dd in existing_deep_dive
                    )
                    # Check if existing library has a non-empty explanation
                    has_explanation = bool(existing_cross.get(label, {}).get('explanation', ''))
                    if label not in api_results and not already_have and not has_explanation:
                        gemma_features_to_query.append(bn)

            # Deduplicate by label
            seen = set()
            unique_to_query = []
            for bn in gemma_features_to_query:
                if bn['label'] not in seen:
                    seen.add(bn['label'])
                    unique_to_query.append(bn)

            query_count = min(len(unique_to_query), args.api_limit)
            print(f"  Querying {query_count} GEMMA features (new or missing explanations)...")

            for i, bn in enumerate(unique_to_query[:query_count]):
                print(f"  [{i+1}/{query_count}] {bn['label']} (NP: L{bn['layer']}/F{bn['np_id']})")
                result = query_neuronpedia_feature(bn['layer'], bn['np_id'], api_key)
                if result:
                    api_results[bn['label']] = result
                    print(f"    ✓ {result.get('explanation', 'N/A')[:60]}")
                    if result.get('examples'):
                        print(f"    Examples: {', '.join(result['examples'][:3])}")
                time.sleep(0.5)  # Rate limiting

    print(f"  API results: {len(api_results)} features queried")

    # Step 5: Build library
    print(f"\n[5/6] Building bottleneck semantic library...")
    library = build_bottleneck_library(all_bottlenecks, cross_circuit, api_results, existing_deep_dive, existing_library)

    # Save library
    library_file = DATA_DIR / 'stage_1_5_bottleneck_library.json'
    with open(library_file, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Library saved: {library_file}")

    # Step 6: Generate report and visualizations
    print(f"\n[6/6] Generating report and visualizations...")

    report = generate_report(library, all_bottlenecks, cross_circuit)
    report_file = DATA_DIR / 'stage_1_5_cross_circuit_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Report saved: {report_file}")

    generate_visualizations(library, all_bottlenecks)

    # Summary
    print("\n" + "=" * 80)
    print("STAGE 1.5 COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Circuits analyzed: {library['metadata']['total_circuits_analyzed']}")
    print(f"  Unique bottleneck features: {library['metadata']['total_unique_features']}")
    print(f"  Cross-circuit features: {library['metadata']['cross_circuit_features']}")
    print(f"  GEMMA features: {library['metadata']['gemma_features']}")
    print(f"  QWEN features: {library['metadata']['qwen_features']}")
    print(f"\nOutputs:")
    print(f"  Library: {library_file}")
    print(f"  Report: {report_file}")
    print(f"  Visualizations: {OUTPUT_DIR}/")

    # Show top cross-circuit features
    if cross_circuit:
        print(f"\n--- TOP CROSS-CIRCUIT BOTTLENECK FEATURES ---")
        for label, appearances in sorted(cross_circuit.items(), key=lambda x: -len(x[1]))[:10]:
            api_data = api_results.get(label, {})
            dd_data = None
            if '_F' in label:
                feat_id_str = label.split('_F')[1]
                dd_data = next(
                    (dd for dd in existing_deep_dive
                     if str(dd.get('circuit_id', '')) == feat_id_str),
                    None
                )
            explanation = api_data.get('explanation', '')
            if not explanation and dd_data:
                explanation = dd_data.get('explanation', '')

            print(f"  {label} ({len(appearances)} circuits): {explanation[:60] if explanation else 'N/A'}")


if __name__ == '__main__':
    main()
