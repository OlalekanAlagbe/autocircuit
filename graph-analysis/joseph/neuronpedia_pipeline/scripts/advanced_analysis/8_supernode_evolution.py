"""
Supernode Evolution Across Layers
Tracks how information transforms within supernodes as it flows through layers

This analysis reveals:
- How concepts enter and exit supernodes
- Which layers show the most transformation
- Whether supernodes split/merge across network depth
- The semantic progression within functional modules
"""

import json
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Add scripts to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from path_manager import PathManager


class SupernodeEvolutionAnalyzer:
    """Analyze how supernodes evolve across transformer layers"""

    def __init__(self, analysis_path):
        """
        Args:
            analysis_path: Path to analysis JSON file
        """
        with open(analysis_path) as f:
            self.analysis = json.load(f)

        self.metadata = self.analysis['metadata']
        self.prompt = self.metadata['prompt']
        self.supernodes = self.analysis.get('louvain_supernodes', {}).get('supernodes', {})

        print(f"[OK] Loaded circuit: {self.prompt}")
        print(f"  Supernodes: {len(self.supernodes)}")

    def track_supernode_layers(self, supernode_id):
        """
        Track which features belong to a supernode at each layer

        Returns:
            dict: {layer: [feature_ids]}
        """
        if str(supernode_id) not in self.supernodes:
            return {}

        sn_data = self.supernodes[str(supernode_id)]
        nodes = sn_data.get('nodes', [])

        layer_features = defaultdict(list)

        for node_id in nodes:
            # Parse node format: "layer_feature_ctx"
            parts = node_id.split('_')
            if len(parts) >= 2:
                layer = int(parts[0])
                layer_features[layer].append(node_id)

        return dict(layer_features)

    def compute_layer_statistics(self, layer_features):
        """
        Compute statistics for each layer in supernode

        Returns:
            dict: {layer: {num_features, entry_features, exit_features}}
        """
        layers = sorted(layer_features.keys())
        layer_stats = {}

        for i, layer in enumerate(layers):
            features = set(layer_features[layer])

            # Entry features: new in this layer
            if i > 0:
                prev_features = set(layer_features[layers[i-1]])
                # Note: We can't track feature identity across layers perfectly
                # since format is "layer_feature_ctx" - feature ID changes
                entry_count = len(features)  # All features are "entries" per layer
            else:
                entry_count = len(features)

            # Exit features: don't appear in next layer
            if i < len(layers) - 1:
                next_features = set(layer_features[layers[i+1]])
                exit_count = len(features)  # All features "exit" per layer
            else:
                exit_count = len(features)

            layer_stats[layer] = {
                'num_features': len(features),
                'entry_count': entry_count,
                'exit_count': exit_count,
                'features': list(features)[:5]  # Sample for inspection
            }

        return layer_stats

    def identify_transformation_points(self, layer_stats):
        """
        Identify layers where significant transformation occurs

        Transformation = large change in feature count

        Returns:
            List of (layer, change_magnitude, change_type)
        """
        layers = sorted(layer_stats.keys())
        transformations = []

        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i+1]

            curr_count = layer_stats[curr_layer]['num_features']
            next_count = layer_stats[next_layer]['num_features']

            change = next_count - curr_count
            pct_change = (change / curr_count * 100) if curr_count > 0 else 0

            # Significant if >20% change
            if abs(pct_change) > 20:
                change_type = 'expansion' if change > 0 else 'contraction'
                transformations.append({
                    'from_layer': curr_layer,
                    'to_layer': next_layer,
                    'change': change,
                    'pct_change': pct_change,
                    'type': change_type
                })

        return transformations

    def analyze_all_supernodes(self):
        """Analyze evolution for all supernodes"""

        evolution_data = {}

        for sn_id in self.supernodes.keys():
            sn_id_int = int(sn_id)

            # Track layers
            layer_features = self.track_supernode_layers(sn_id_int)

            if not layer_features:
                continue

            # Compute statistics
            layer_stats = self.compute_layer_statistics(layer_features)

            # Identify transformations
            transformations = self.identify_transformation_points(layer_stats)

            evolution_data[sn_id_int] = {
                'layer_features': {k: len(v) for k, v in layer_features.items()},
                'layer_stats': layer_stats,
                'transformations': transformations,
                'layer_range': [min(layer_features.keys()), max(layer_features.keys())],
                'total_layers': len(layer_features)
            }

        return evolution_data

    def generate_evolution_report(self, evolution_data, output_path):
        """Generate text report of supernode evolution"""

        report = []
        report.append("=" * 80)
        report.append("SUPERNODE EVOLUTION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        report.append(f"Circuit: {self.prompt}")
        report.append(f"Total Supernodes: {len(evolution_data)}")
        report.append("")

        # Summary statistics
        report.append("=" * 80)
        report.append("EVOLUTION SUMMARY")
        report.append("=" * 80)

        total_transformations = sum(len(data['transformations']) for data in evolution_data.values())
        report.append(f"Total transformation points: {total_transformations}")
        report.append("")

        # Supernode-by-supernode analysis
        report.append("=" * 80)
        report.append("SUPERNODE-BY-SUPERNODE EVOLUTION")
        report.append("=" * 80)

        for sn_id, data in sorted(evolution_data.items()):
            report.append(f"\nSupernode {sn_id}:")
            report.append(f"  Layer Range: L{data['layer_range'][0]}-L{data['layer_range'][1]} ({data['total_layers']} layers)")

            # Feature count progression
            layer_features = data['layer_features']
            progression = [f"L{l}:{c}" for l, c in sorted(layer_features.items())[:5]]
            if len(layer_features) > 5:
                report.append(f"  Feature Progression: {', '.join(progression)}, ...")
            else:
                report.append(f"  Feature Progression: {', '.join(progression)}")

            # Transformations
            if data['transformations']:
                report.append(f"  Transformation Points: {len(data['transformations'])}")
                for trans in data['transformations']:
                    report.append(f"    - L{trans['from_layer']}->L{trans['to_layer']}: {trans['change']:+d} features ({trans['pct_change']:+.1f}%) [{trans['type']}]")
            else:
                report.append("  Transformation Points: None (stable across layers)")

        # Key findings
        report.append("")
        report.append("=" * 80)
        report.append("KEY INSIGHTS")
        report.append("=" * 80)

        # Which supernodes span most layers?
        span_sorted = sorted(evolution_data.items(), key=lambda x: x[1]['total_layers'], reverse=True)
        report.append("\n1. WIDEST SPAN (most layers):")
        for sn_id, data in span_sorted[:3]:
            report.append(f"   SN{sn_id}: {data['total_layers']} layers (L{data['layer_range'][0]}-L{data['layer_range'][1]})")

        # Which have most transformations?
        trans_sorted = sorted(evolution_data.items(), key=lambda x: len(x[1]['transformations']), reverse=True)
        report.append("\n2. MOST DYNAMIC (most transformations):")
        for sn_id, data in trans_sorted[:3]:
            if data['transformations']:
                report.append(f"   SN{sn_id}: {len(data['transformations'])} transformation points")

        # Which layers show most transformation overall?
        layer_transformation_counts = defaultdict(int)
        for data in evolution_data.values():
            for trans in data['transformations']:
                layer_transformation_counts[trans['from_layer']] += 1

        if layer_transformation_counts:
            report.append("\n3. HOTSPOT LAYERS (most transformations across all supernodes):")
            for layer, count in sorted(layer_transformation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"   Layer {layer}: {count} transformations")

        report.append("")
        report.append("=" * 80)

        # Write report
        output_text = "\n".join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"[OK] Evolution report saved to: {output_path}")
        return output_text

    def visualize_evolution(self, evolution_data, output_path):
        """Create Sankey-style visualization of supernode evolution"""

        # Since we can't easily create a Sankey diagram with matplotlib,
        # we'll create a heatmap showing feature counts across layers

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Chart 1: Feature count heatmap
        supernode_ids = sorted(evolution_data.keys())
        all_layers = set()
        for data in evolution_data.values():
            all_layers.update(data['layer_features'].keys())
        all_layers = sorted(all_layers)

        # Build matrix: supernodes x layers
        matrix = np.zeros((len(supernode_ids), len(all_layers)))

        for i, sn_id in enumerate(supernode_ids):
            layer_features = evolution_data[sn_id]['layer_features']
            for j, layer in enumerate(all_layers):
                matrix[i, j] = layer_features.get(layer, 0)

        # Plot heatmap
        im = ax1.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_yticks(range(len(supernode_ids)))
        ax1.set_yticklabels([f"SN{sn_id}" for sn_id in supernode_ids], fontsize=9)
        ax1.set_xticks(range(0, len(all_layers), 5))
        ax1.set_xticklabels([f"L{all_layers[i]}" for i in range(0, len(all_layers), 5)], fontsize=9)
        ax1.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Supernode', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Count by Supernode and Layer', fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Number of Features', fontsize=10)

        # Chart 2: Layer transformation frequency
        layer_transformation_counts = defaultdict(int)
        for data in evolution_data.values():
            for trans in data['transformations']:
                layer_transformation_counts[trans['from_layer']] += 1

        if layer_transformation_counts:
            layers = sorted(layer_transformation_counts.keys())
            counts = [layer_transformation_counts[l] for l in layers]

            ax2.bar(layers, counts, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Layer', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Number of Transformations', fontsize=11, fontweight='bold')
            ax2.set_title('Transformation Hotspots (layers with >20% feature count change)', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No significant transformations detected\n(>20% feature count change)',
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.axis('off')

        plt.suptitle(f'Supernode Evolution Across Layers\n"{self.prompt}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Evolution visualization saved to: {output_path}")
        plt.close()

    def analyze_evolution(self, output_dir):
        """Main method to analyze supernode evolution"""

        print("\n" + "="*60)
        print("STEP 1: Track Evolution Across Layers")
        print("="*60)

        evolution_data = self.analyze_all_supernodes()
        print(f"Analyzed {len(evolution_data)} supernodes")

        print("\n" + "="*60)
        print("STEP 2: Identify Transformation Points")
        print("="*60)

        total_trans = sum(len(data['transformations']) for data in evolution_data.values())
        print(f"Found {total_trans} transformation points across all supernodes")

        print("\n" + "="*60)
        print("STEP 3: Generate Report")
        print("="*60)

        report_path = output_dir / "supernode_evolution.txt"
        self.generate_evolution_report(evolution_data, report_path)

        print("\n" + "="*60)
        print("STEP 4: Create Visualizations")
        print("="*60)

        viz_path = output_dir / "evolution_heatmap.png"
        self.visualize_evolution(evolution_data, viz_path)

        # Save data
        data_path = output_dir / "evolution_data.json"
        with open(data_path, 'w') as f:
            json.dump(evolution_data, f, indent=2, default=str)
        print(f"[OK] Evolution data saved to: {data_path}")

        return evolution_data


def main():
    parser = argparse.ArgumentParser(description='Analyze supernode evolution across layers')
    parser.add_argument('--analysis', type=str,
                       help='Path to analysis JSON file (if not provided, uses latest)')
    parser.add_argument('--prompt', type=str,
                       help='Prompt text to analyze')

    args = parser.parse_args()

    pm = PathManager()

    # Auto-discover analysis file if not provided
    if args.analysis:
        analysis_path = Path(args.analysis)
    elif args.prompt:
        analysis_path = pm.circuit_analysis_path(args.prompt)
        if not analysis_path.exists():
            print(f"[ERROR] Analysis not found for prompt: {args.prompt}")
            return
    else:
        # Find available analyses and let user choose
        prompts = pm.get_all_analyzed_prompts()
        if not prompts:
            print("[ERROR] No analysis files found!")
            return

        # Display available prompts
        print("\nAvailable circuit analyses:")
        print("=" * 60)
        for i, prompt_text in enumerate(prompts, 1):
            analysis_file = pm.circuit_analysis_path(prompt_text)
            size_mb = analysis_file.stat().st_size / (1024 * 1024)
            print(f"{i}. {prompt_text} ({size_mb:.2f} MB)")
        print("=" * 60)

        # Get user selection
        print("\nSelect analysis to process:")
        print("-" * 60)

        while True:
            try:
                choice = input("Enter number (or press Enter for latest): ").strip()

                if choice == "":
                    prompt = prompts[0]
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(prompts):
                    prompt = prompts[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(prompts)}")
            except ValueError:
                print("Please enter a valid number or press Enter for latest")

        analysis_path = pm.circuit_analysis_path(prompt)
        print(f"\n[INFO] Selected: {prompt}")

    # Extract prompt from analysis
    with open(analysis_path) as f:
        analysis = json.load(f)
        prompt = analysis['metadata']['prompt']

    # Create output directory using PathManager
    output_dir = pm.evolution_dir(prompt)
    print(f"\nOutput directory: {output_dir}")

    # Analyze evolution
    analyzer = SupernodeEvolutionAnalyzer(analysis_path)
    evolution_data = analyzer.analyze_evolution(output_dir)

    print("\n" + "="*60)
    print("SUPERNODE EVOLUTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Report: {output_dir / 'supernode_evolution.txt'}")
    print(f"Visualization: {output_dir / 'evolution_heatmap.png'}")
    print(f"Data: {output_dir / 'evolution_data.json'}")


if __name__ == "__main__":
    main()
