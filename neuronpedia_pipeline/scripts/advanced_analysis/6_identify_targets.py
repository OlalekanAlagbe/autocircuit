"""
Identify Steering Intervention Targets
Reads comprehensive circuit analysis and outputs intervention candidates
Supports interactive selection or command-line file specification
"""

import json
from pathlib import Path
import argparse
import sys

def find_available_analysis_files():
    """Find all comprehensive analysis files in the data directory"""
    # parent.parent.parent because we're in scripts/advanced_analysis/
    graphs_dir = Path(__file__).parent.parent.parent / "data" / "graphs"

    if not graphs_dir.exists():
        return []

    # Find all *_analysis.json files
    analysis_files = list(graphs_dir.glob("*_analysis.json"))

    # Sort by modification time (newest first)
    analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return analysis_files

def display_available_files(files):
    """Display numbered list of available analysis files"""
    print("\n" + "=" * 60)
    print("AVAILABLE ANALYSIS FILES")
    print("=" * 60)

    if not files:
        print("No analysis files found!")
        print("Please run '/circuit-tracer-analyze' first.")
        return

    for i, file_path in enumerate(files, 1):
        # Get file size
        size_kb = file_path.stat().st_size / 1024
        print(f"{i}. {file_path.name} ({size_kb:.1f} KB)")

    print("=" * 60)

def get_user_selection(files):
    """Get user's file selection interactively"""
    print("\nSelect analysis file:")
    print("-" * 60)

    while True:
        try:
            choice = input("Select file (enter number, or press Enter for latest): ").strip()

            if choice == "":
                # Use latest (first in list)
                return files[0]

            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number or press Enter for latest")

def display_targets(analysis_data):
    """Display steering targets in a user-friendly format"""

    # Check if comprehensive analysis format
    if 'flow_analysis' not in analysis_data:
        print("\n[ERROR] Legacy analysis format detected!")
        print("This file does not contain flow_analysis.")
        print("Please re-run '/circuit-tracer-analyze' to generate comprehensive analysis.")
        return False

    flow_analysis = analysis_data['flow_analysis']
    metadata = analysis_data.get('metadata', {})
    prompt = metadata.get('prompt', 'Unknown')

    print("\n" + "=" * 60)
    print("STEERING INTERVENTION TARGETS")
    print("=" * 60)
    print(f"Prompt: \"{prompt}\"")
    print("=" * 60)

    # ========================================================================
    # INPUT AMPLIFICATION TARGETS
    # ========================================================================
    print("\n" + "=" * 60)
    print("INPUT AMPLIFICATION TARGETS")
    print("=" * 60)
    print("Strategy: Amplify these early-layer nodes to boost signal propagation")
    print("-" * 60)

    input_nodes = flow_analysis.get('input_nodes', [])

    if not input_nodes:
        print("[WARNING] No input targets found")
    else:
        print(f"\nTop {len(input_nodes)} input nodes (early layers, high fan-out):\n")
        for i, node in enumerate(input_nodes, 1):
            label = node['label']
            out_deg = node['out_degree']
            activation = node['activation']
            layer = node['layer']
            betweenness = node.get('betweenness', 0.0)

            print(f"{i:2d}. {label:20s} | Layer {layer:2d} | Out-degree: {out_deg:3d} | Activation: {activation:6.2f}")
            if i == 1:
                print(f"    ^ HIGHEST PRIORITY - Spreads to {out_deg} downstream nodes")

        print(f"\nTotal input targets: {len(input_nodes)}")
        print("Use case: Amplify activations to strengthen early circuit activation")

    # ========================================================================
    # OUTPUT SUPPRESSION TARGETS
    # ========================================================================
    print("\n" + "=" * 60)
    print("OUTPUT SUPPRESSION TARGETS")
    print("=" * 60)
    print("Strategy: Suppress these late-layer nodes to prevent wrong answers")
    print("-" * 60)

    output_nodes = flow_analysis.get('output_nodes', [])

    if not output_nodes:
        print("[WARNING] No output targets found")
    else:
        print(f"\nTop {len(output_nodes)} output nodes (late layers, high activation):\n")
        for i, node in enumerate(output_nodes, 1):
            label = node['label']
            activation = node['activation']
            in_deg = node['in_degree']
            layer = node['layer']
            influence = node.get('influence', 0.0)

            print(f"{i:2d}. {label:20s} | Layer {layer:2d} | Activation: {activation:7.2f} | In-degree: {in_deg:3d}")
            if i == 1:
                print(f"    ^ HIGHEST PRIORITY - Most active output node (likely wrong answer)")

        print(f"\nTotal output targets: {len(output_nodes)}")
        print("Use case: Suppress to reduce confidence in incorrect predictions")

    # ========================================================================
    # BOTTLENECK INTERVENTION TARGETS
    # ========================================================================
    print("\n" + "=" * 60)
    print("BOTTLENECK INTERVENTION TARGETS")
    print("=" * 60)
    print("Strategy: Ablate these critical pathway nodes to disrupt reasoning")
    print("-" * 60)

    bottleneck_nodes = flow_analysis.get('bottleneck_nodes', [])

    if not bottleneck_nodes:
        print("[WARNING] No bottleneck targets found")
    else:
        print(f"\nTop {len(bottleneck_nodes)} bottleneck nodes (high betweenness centrality):\n")
        for i, node in enumerate(bottleneck_nodes, 1):
            label = node['label']
            betweenness = node['betweenness']
            layer = node['layer']
            activation = node['activation']
            in_deg = node['in_degree']
            out_deg = node['out_degree']

            print(f"{i:2d}. {label:20s} | Layer {layer:2d} | Betweenness: {betweenness:.6f} | In/Out: {in_deg:3d}/{out_deg:3d}")
            if i == 1:
                print(f"    ^ HIGHEST PRIORITY - Critical information bottleneck")

        print(f"\nTotal bottleneck targets: {len(bottleneck_nodes)}")
        print("Use case: Ablate to test necessity for circuit function")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_targets = len(input_nodes) + len(output_nodes) + len(bottleneck_nodes)
    print(f"Total intervention candidates: {total_targets}")
    print(f"  - Input amplification targets: {len(input_nodes)}")
    print(f"  - Output suppression targets: {len(output_nodes)}")
    print(f"  - Bottleneck ablation targets: {len(bottleneck_nodes)}")

    return True

def export_targets_json(analysis_data, output_path):
    """Export targets to JSON for programmatic use"""
    flow_analysis = analysis_data.get('flow_analysis', {})
    metadata = analysis_data.get('metadata', {})

    targets = {
        'metadata': metadata,
        'input_amplification': flow_analysis.get('input_nodes', []),
        'output_suppression': flow_analysis.get('output_nodes', []),
        'bottleneck_ablation': flow_analysis.get('bottleneck_nodes', [])
    }

    with open(output_path, 'w') as f:
        json.dump(targets, f, indent=2)

    print(f"\n[OK] Targets exported to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Identify steering intervention targets from circuit analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python 6_identify_targets.py  (interactive mode)
  python 6_identify_targets.py --file path/to/analysis.json
  python 6_identify_targets.py --file path/to/analysis.json --export
        '''
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to specific analysis file (if not provided, will use interactive mode)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export targets to JSON file'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CIRCUIT TRACER - IDENTIFY TARGETS")
    print("=" * 60)

    # Determine input file
    if args.file:
        # Command-line mode
        analysis_file = Path(args.file)
        if not analysis_file.exists():
            print(f"\n[ERROR] File not found: {analysis_file}")
            sys.exit(1)
        print(f"\nUsing file from command-line: {analysis_file.name}")
    else:
        # Interactive mode
        print("\nEntering interactive mode...")

        # Find available files
        available_files = find_available_analysis_files()

        if not available_files:
            print("\n[ERROR] No analysis files found!")
            print("Please run '/circuit-tracer-analyze' first.")
            sys.exit(1)

        # Display available files
        display_available_files(available_files)

        # Get user selection
        analysis_file = get_user_selection(available_files)
        print(f"\nSelected: {analysis_file.name}")

    # Load analysis
    print("\n" + "=" * 60)
    print("LOADING ANALYSIS")
    print("=" * 60)

    with open(analysis_file) as f:
        analysis_data = json.load(f)

    print(f"Loaded: {analysis_file.name}")

    # Display targets
    success = display_targets(analysis_data)

    if not success:
        sys.exit(1)

    # Export if requested
    if args.export:
        output_dir = analysis_file.parent.parent / "steering"
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = analysis_file.stem.replace('_analysis', '')
        output_path = output_dir / f"targets_{base_name}.json"

        export_targets_json(analysis_data, output_path)

    print("\n" + "=" * 60)
    print("[SUCCESS] Target Identification Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Design steering experiments using these targets")
    print("  2. Compare targets across prompts to find universal nodes")
    print("  3. Run ablation studies to validate target importance")

    print("\n" + "=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)
