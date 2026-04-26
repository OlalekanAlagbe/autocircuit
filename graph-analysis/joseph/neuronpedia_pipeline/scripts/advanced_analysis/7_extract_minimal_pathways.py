"""
Minimal Pathway Extraction - Find Essential Circuit
Extracts the minimum viable circuit connecting input features to output features

This analysis identifies:
- Critical features necessary for the output
- Essential edges that must be present
- Redundant connections that can be removed
- Circuit efficiency metrics
"""

import json
from pathlib import Path
import networkx as nx
import numpy as np
import argparse
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.parse_math'] = False
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Add scripts to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from path_manager import PathManager


class MinimalPathwayExtractor:
    """Extract minimal sufficient circuit from full attribution graph"""

    def __init__(self, analysis_path):
        """
        Args:
            analysis_path: Path to analysis JSON file
        """
        self.analysis_path = Path(analysis_path)

        with open(self.analysis_path) as f:
            self.analysis = json.load(f)

        self.metadata = self.analysis['metadata']
        self.prompt = self.metadata['prompt']

        # Load feature descriptions
        self.feature_descriptions = self._load_feature_descriptions()

        # Build NetworkX graph
        self.G = self._build_graph()

        print(f"[OK] Loaded circuit: {self.prompt}")
        print(f"  Original: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
        print(f"  Feature descriptions: {len(self.feature_descriptions)} available")

    def _load_feature_descriptions(self):
        """Load feature descriptions from analysis JSON"""
        descriptions = {}

        # Load from layer groups (top features with descriptions)
        for group_name, group_data in self.analysis.get('layer_groups', {}).items():
            for feat in group_data.get('top_features_with_descriptions', []):
                feature_id = feat.get('feature', '')
                desc = feat.get('description', '')
                if feature_id and desc:
                    descriptions[feature_id] = desc

        return descriptions

    def _build_graph(self):
        """Build NetworkX directed graph from analysis data"""
        G = nx.DiGraph()

        # Add nodes from supernodes
        supernodes = self.analysis.get('louvain_supernodes', {}).get('supernodes', {})
        for sn_id, sn_data in supernodes.items():
            node_list = sn_data.get('nodes', [])
            for node_id in node_list:
                # Parse node format: "layer_feature_ctx"
                parts = node_id.split('_')
                if len(parts) >= 2:
                    layer = int(parts[0])
                    G.add_node(node_id, layer=layer)

        # For edges, we need to load from converted graph
        # (analysis JSON doesn't have edges, only supernode structure)
        # Use PathManager to find converted graph
        pm = PathManager()
        converted_path = pm.converted_graph_path(self.prompt)

        if converted_path.exists():
            print(f"[OK] Loading edges from: {converted_path.name}")
            with open(converted_path) as f:
                converted = json.load(f)
                for edge in converted.get('edges', []):
                    source = edge['source']
                    target = edge['target']
                    weight = edge.get('weight', 1.0)
                    if source in G.nodes() and target in G.nodes():
                        G.add_edge(source, target, weight=weight)
        else:
            print(f"[WARNING] Converted file not found: {converted_path}")
            print("[WARNING] Building graph without edges")

        return G

    def identify_input_features(self, top_k=20):
        """
        Identify input features (early layers with high fan-out)

        Returns:
            List of (node_id, out_degree, layer) tuples
        """
        input_features = []

        for node in self.G.nodes():
            layer = self.G.nodes[node].get('layer', 999)
            out_degree = self.G.out_degree(node)

            # Input features: layers 0-5, high out-degree
            if layer <= 5 and out_degree > 0:
                input_features.append((node, out_degree, layer))

        # Sort by out-degree
        input_features.sort(key=lambda x: x[1], reverse=True)

        return input_features[:top_k]

    def identify_output_features(self, top_k=20):
        """
        Identify output features (late layers with high in-degree)

        Returns:
            List of (node_id, in_degree, layer) tuples
        """
        output_features = []

        for node in self.G.nodes():
            layer = self.G.nodes[node].get('layer', 0)
            in_degree = self.G.in_degree(node)

            # Output features: layers 21-25, high in-degree
            if layer >= 21 and in_degree > 0:
                output_features.append((node, in_degree, layer))

        # Sort by in-degree
        output_features.sort(key=lambda x: x[1], reverse=True)

        return output_features[:top_k]

    def find_all_paths_bfs(self, sources, targets, max_paths_per_pair=3):
        """
        Find shortest paths from sources to targets (optimized)

        Args:
            sources: List of source node IDs
            targets: List of target node IDs
            max_paths_per_pair: Maximum paths to find per source-target pair

        Returns:
            List of paths (each path is a list of node IDs)
        """
        all_paths = []
        total_pairs = len(sources) * len(targets)
        processed = 0

        print(f"[INFO] Finding paths for {len(sources)} sources x {len(targets)} targets = {total_pairs} pairs")

        for source in sources:
            for target in targets:
                processed += 1
                if processed % 10 == 0:
                    print(f"  Progress: {processed}/{total_pairs} pairs ({processed/total_pairs*100:.1f}%)")

                try:
                    # Use shortest_simple_paths instead of all_simple_paths (much faster)
                    # This finds paths in order of increasing length
                    path_generator = nx.shortest_simple_paths(self.G, source, target)

                    # Take only first max_paths_per_pair paths
                    paths = []
                    for i, path in enumerate(path_generator):
                        if i >= max_paths_per_pair:
                            break
                        if len(path) <= 15:  # Skip very long paths
                            paths.append(path)

                    all_paths.extend(paths)

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path exists between this source-target pair
                    continue

        print(f"[OK] Found {len(all_paths)} total paths")
        return all_paths

    def extract_essential_edges(self, paths, threshold=0.5):
        """
        Extract edges that appear in many paths (essential edges)

        Args:
            paths: List of paths
            threshold: Minimum fraction of paths an edge must appear in

        Returns:
            Set of essential edges (source, target) tuples
        """
        edge_counts = defaultdict(int)

        # Count edge appearances
        for path in paths:
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                edge_counts[edge] += 1

        # Filter to essential edges
        min_count = int(threshold * len(paths))
        essential_edges = {edge for edge, count in edge_counts.items() if count >= min_count}

        print(f"[OK] Found {len(essential_edges)} essential edges (appear in >={threshold:.0%} of paths)")
        return essential_edges

    def construct_minimal_circuit(self, essential_edges):
        """
        Build minimal circuit graph with only essential edges

        Returns:
            NetworkX graph of minimal circuit
        """
        minimal_G = nx.DiGraph()

        # Add nodes from essential edges
        for source, target in essential_edges:
            if source not in minimal_G:
                minimal_G.add_node(source, **self.G.nodes[source])
            if target not in minimal_G:
                minimal_G.add_node(target, **self.G.nodes[target])

            # Add edge with original weight
            if self.G.has_edge(source, target):
                minimal_G.add_edge(source, target, **self.G[source][target])

        print(f"[OK] Minimal circuit: {len(minimal_G.nodes())} nodes, {len(minimal_G.edges())} edges")
        return minimal_G

    def compute_pathway_metrics(self, minimal_G, paths):
        """Compute metrics about the minimal circuit"""

        reduction_pct = (1 - len(minimal_G.nodes()) / len(self.G.nodes())) * 100

        # Bottleneck width: narrowest layer in circuit
        layer_counts = defaultdict(int)
        for node in minimal_G.nodes():
            layer = minimal_G.nodes[node].get('layer', 0)
            layer_counts[layer] += 1

        bottleneck_width = min(layer_counts.values()) if layer_counts else 0
        bottleneck_layer = min(layer_counts, key=layer_counts.get) if layer_counts else 0

        # Path diversity: how many parallel pathways?
        path_diversity = len(set(tuple(path) for path in paths))

        metrics = {
            'original_nodes': len(self.G.nodes()),
            'original_edges': len(self.G.edges()),
            'minimal_nodes': len(minimal_G.nodes()),
            'minimal_edges': len(minimal_G.edges()),
            'reduction_pct': reduction_pct,
            'bottleneck_width': bottleneck_width,
            'bottleneck_layer': bottleneck_layer,
            'path_diversity': path_diversity,
            'avg_path_length': np.mean([len(p) for p in paths]) if paths else 0
        }

        return metrics

    def extract_minimal_pathway(self, output_path):
        """Main method to extract minimal pathway"""

        print("\n" + "="*60)
        print("STEP 1: Identify Input Features")
        print("="*60)
        input_features = self.identify_input_features(top_k=10)
        print(f"Top input features (by out-degree):")
        for node, out_deg, layer in input_features[:5]:
            print(f"  {node} (L{layer}): out_degree={out_deg}")

        print("\n" + "="*60)
        print("STEP 2: Identify Output Features")
        print("="*60)
        output_features = self.identify_output_features(top_k=10)
        print(f"Top output features (by in-degree):")
        for node, in_deg, layer in output_features[:5]:
            print(f"  {node} (L{layer}): in_degree={in_deg}")

        print("\n" + "="*60)
        print("STEP 3: Find All Paths (Input -> Output)")
        print("="*60)
        sources = [node for node, _, _ in input_features]
        targets = [node for node, _, _ in output_features]
        paths = self.find_all_paths_bfs(sources, targets, max_paths_per_pair=3)

        if not paths:
            print("[ERROR] No paths found! Cannot extract minimal circuit.")
            return None

        print("\n" + "="*60)
        print("STEP 4: Extract Essential Edges")
        print("="*60)
        essential_edges = self.extract_essential_edges(paths, threshold=0.2)

        # If threshold too high, try lower
        if len(essential_edges) == 0:
            print("[WARNING] No edges found at 20% threshold, trying 10%...")
            essential_edges = self.extract_essential_edges(paths, threshold=0.1)

        # If still nothing, use top N most frequent edges
        if len(essential_edges) == 0:
            print("[WARNING] Extreme path diversity detected. Using top 500 most frequent edges...")
            edge_counts = defaultdict(int)
            for path in paths:
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_counts[edge] += 1

            # Get top 500 edges by frequency
            sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
            essential_edges = {edge for edge, count in sorted_edges[:500]}
            print(f"[OK] Using {len(essential_edges)} most frequent edges")

        print("\n" + "="*60)
        print("STEP 5: Construct Minimal Circuit")
        print("="*60)
        minimal_G = self.construct_minimal_circuit(essential_edges)

        print("\n" + "="*60)
        print("STEP 6: Compute Metrics")
        print("="*60)
        metrics = self.compute_pathway_metrics(minimal_G, paths)

        print(f"Circuit Reduction: {metrics['reduction_pct']:.1f}%")
        print(f"  Original: {metrics['original_nodes']:,} nodes -> Minimal: {metrics['minimal_nodes']:,} nodes")
        print(f"Bottleneck: Layer {metrics['bottleneck_layer']} with {metrics['bottleneck_width']} features")
        print(f"Path Diversity: {metrics['path_diversity']} unique paths")
        print(f"Average Path Length: {metrics['avg_path_length']:.1f} features")

        # Save results
        result = {
            'metadata': self.metadata,
            'minimal_circuit': {
                'nodes': list(minimal_G.nodes()),
                'edges': [(u, v, minimal_G[u][v]) for u, v in minimal_G.edges()]
            },
            'metrics': metrics,
            'input_features': [{'node': n, 'out_degree': d, 'layer': l} for n, d, l in input_features],
            'output_features': [{'node': n, 'in_degree': d, 'layer': l} for n, d, l in output_features],
            'num_paths': len(paths),
            'sample_paths': [[str(node) for node in path] for path in paths[:5]]
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[OK] Minimal pathway saved to: {output_path}")
        return minimal_G, metrics

    def visualize_comparison(self, minimal_G, metrics, output_path):
        """Create side-by-side comparison visualization"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Bar chart: Node counts
        categories = ['Nodes', 'Edges']
        original = [metrics['original_nodes'], metrics['original_edges']]
        minimal = [metrics['minimal_nodes'], metrics['minimal_edges']]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, original, width, label='Original Circuit', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, minimal, width, label='Minimal Circuit', color='coral', alpha=0.8)

        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Circuit Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, (orig, mini) in enumerate(zip(original, minimal)):
            reduction = (1 - mini/orig) * 100
            ax1.text(i, max(orig, mini) * 1.05, f'-{reduction:.1f}%',
                    ha='center', fontsize=10, fontweight='bold', color='green')

        # Metrics summary
        ax2.axis('off')
        metrics_text = f"""
MINIMAL PATHWAY METRICS

Circuit Reduction:
  {metrics['reduction_pct']:.1f}% fewer features

Original Circuit:
  {metrics['original_nodes']:,} nodes
  {metrics['original_edges']:,} edges

Minimal Circuit:
  {metrics['minimal_nodes']:,} nodes
  {metrics['minimal_edges']:,} edges

Bottleneck:
  Layer {metrics['bottleneck_layer']} (width: {metrics['bottleneck_width']})

Path Analysis:
  {metrics['path_diversity']} unique pathways
  {metrics['avg_path_length']:.1f} avg features per path

Interpretation:
  The minimal circuit contains only the
  ESSENTIAL features needed to connect
  inputs to outputs. All redundant
  connections have been pruned.
        """

        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add action items below metrics
        action_text = f"""
ACTION ITEMS:

1. Test Bottleneck Layer (L{metrics['bottleneck_layer']}):
   • Ablate all {metrics['bottleneck_width']} feature(s)
   • Expected: Circuit breaks if critical

2. Validate Minimal Circuit:
   • Run model with only minimal nodes active
   • Expected: Output should match full circuit

3. Identify Redundancy:
   • {metrics['reduction_pct']:.0f}% of circuit was redundant
   • Investigate: Why so many parallel paths?
        """

        # Place action items BELOW the plot area
        fig.text(0.5, -0.08, action_text, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'),
                 family='monospace', transform=fig.transFigure, verticalalignment='top')

        plt.suptitle(f'Minimal Pathway Analysis\n"{self.prompt}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make more room for action items below
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Comparison visualization saved to: {output_path}")
        plt.close()

    def visualize_minimal_graph_structure(self, minimal_G, output_path, metrics):
        """
        Visualize the actual minimal pathway graph with nodes and edges

        Creates:
        - Node positioning: x=layer, y=betweenness_centrality
        - Node size: proportional to activation
        - Node color: by layer group (INPUT/MIDDLE/OUTPUT)
        - Edge width: proportional to weight
        - Labels: show top features
        """

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        # 1. Calculate node positions (hierarchical by layer)
        print("[INFO] Computing node positions...")
        pos = {}
        betweenness = nx.betweenness_centrality(minimal_G)

        # Group nodes by layer for better vertical spacing
        layer_nodes = defaultdict(list)
        for node in minimal_G.nodes():
            layer = minimal_G.nodes[node]['layer']
            layer_nodes[layer].append(node)

        # Position nodes
        for layer, nodes_in_layer in layer_nodes.items():
            # Sort by betweenness for consistent ordering
            nodes_sorted = sorted(nodes_in_layer, key=lambda n: betweenness.get(n, 0), reverse=True)
            num_nodes = len(nodes_sorted)

            for i, node in enumerate(nodes_sorted):
                # x = layer, y = spread vertically based on betweenness and index
                y_offset = (i - num_nodes/2) * 2  # Spread around center
                y_importance = betweenness.get(node, 0) * 50  # Scale by importance
                pos[node] = (layer, y_offset + y_importance)

        # 2. Node sizes (by activation, with minimum size)
        node_sizes = []
        for node in minimal_G.nodes():
            activation = minimal_G.nodes[node].get('activation', 0.1)
            size = max(100, activation * 100)  # Minimum size 100
            node_sizes.append(size)

        # 3. Node colors (by layer group)
        def get_layer_group_color(layer):
            if layer <= 5:
                return '#66c2a5'      # INPUT (green)
            elif layer <= 15:
                return '#fc8d62'   # MIDDLE (orange)
            else:
                return '#8da0cb'   # OUTPUT (blue)

        node_colors = [
            get_layer_group_color(minimal_G.nodes[node]['layer'])
            for node in minimal_G.nodes()
        ]

        # 4. Edge widths (by weight magnitude)
        edge_widths = []
        for u, v in minimal_G.edges():
            weight = abs(minimal_G[u][v].get('weight', 1.0))
            width = max(0.5, weight * 2)  # Minimum width 0.5
            edge_widths.append(width)

        # 5. Draw graph
        print("[INFO] Drawing network...")
        nx.draw_networkx_nodes(minimal_G, pos, node_size=node_sizes,
                               node_color=node_colors, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(minimal_G, pos, width=edge_widths,
                               alpha=0.3, edge_color='gray', arrows=True,
                               arrowsize=10, ax=ax)

        # 6. Add labels for top nodes with descriptions
        # Prioritize nodes that have descriptions
        nodes_with_desc = [n for n in minimal_G.nodes() if n in self.feature_descriptions]
        nodes_without_desc = [n for n in minimal_G.nodes() if n not in self.feature_descriptions]

        # Sort both by betweenness
        nodes_with_desc_sorted = sorted(nodes_with_desc, key=lambda n: betweenness.get(n, 0), reverse=True)
        nodes_without_desc_sorted = sorted(nodes_without_desc, key=lambda n: betweenness.get(n, 0), reverse=True)

        # Take top 15 with descriptions, top 5 without
        top_nodes = nodes_with_desc_sorted[:15] + nodes_without_desc_sorted[:5]

        labels = {}
        for node in top_nodes:
            layer = minimal_G.nodes[node]['layer']
            parts = node.split('_')
            feature_id = parts[1][:4] if len(parts) >= 2 else 'unknown'

            # Get description if available
            if node in self.feature_descriptions:
                desc = self.feature_descriptions[node]
                # Get first two activation tokens (e.g., "Paris, France")
                # Descriptions are in format: "Activates on: token1, token2, token3"
                if "Activates on:" in desc:
                    tokens_part = desc.split("Activates on:")[1].strip()
                    # Get first 2 tokens, clean them
                    tokens = [t.strip() for t in tokens_part.split(",")[:2]]
                    short_desc = ", ".join(tokens)
                    # Truncate if too long
                    if len(short_desc) > 25:
                        short_desc = short_desc[:25] + "..."
                else:
                    short_desc = desc[:25] + "..." if len(desc) > 25 else desc
                labels[node] = f"L{layer}_F{feature_id}\n{short_desc}"
            else:
                labels[node] = f"L{layer}_F{feature_id}"

        nx.draw_networkx_labels(minimal_G, pos, labels, font_size=6, ax=ax)

        # 7. Add layer group boundaries
        for layer in [5.5, 15.5]:  # boundaries
            ax.axvline(x=layer, color='black', linestyle='--', alpha=0.3, linewidth=2)

        # 8. Add layer group labels
        ax.text(2.5, ax.get_ylim()[1] * 0.95, 'INPUT', fontsize=12,
                fontweight='bold', color='#66c2a5', ha='center')
        ax.text(10.5, ax.get_ylim()[1] * 0.95, 'MIDDLE', fontsize=12,
                fontweight='bold', color='#fc8d62', ha='center')
        ax.text(20, ax.get_ylim()[1] * 0.95, 'OUTPUT', fontsize=12,
                fontweight='bold', color='#8da0cb', ha='center')

        # 9. Title and labels
        ax.set_title(f'Minimal Pathway Network Structure\n"{self.prompt}"\n'
                     f'{len(minimal_G.nodes())} essential nodes | '
                     f'{len(minimal_G.edges())} essential edges | '
                     f'{metrics["reduction_pct"]:.1f}% reduction',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance (vertical spread)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # 10. Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#66c2a5', label='Input (L0-5)'),
            Patch(facecolor='#fc8d62', label='Middle (L6-15)'),
            Patch(facecolor='#8da0cb', label='Output (L16+)')
        ]
        # Place legend OUTSIDE plot area on the right
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=10, framealpha=0.9, edgecolor='black', borderaxespad=0)

        # Add action items
        # Get top nodes by betweenness for recommendations
        betweenness = nx.betweenness_centrality(minimal_G)
        top_nodes = sorted(minimal_G.nodes(),
                           key=lambda n: betweenness.get(n, 0),
                           reverse=True)[:3]

        action_text = "CIRCUIT MANIPULATION:\n\n"
        action_text += "1. Target High-Centrality Nodes:\n"
        for i, node in enumerate(top_nodes[:2], 1):
            layer = minimal_G.nodes[node]['layer']
            parts = node.split('_')
            feat_id = parts[1][:6] if len(parts) >= 2 else 'unknown'
            action_text += f"   • L{layer}_F{feat_id} (betweenness={betweenness[node]:.3f})\n"
        action_text += "\n2. Test Layer Boundaries:\n"
        action_text += "   • Input→Middle (L5-6) transition\n"
        action_text += "   • Middle→Output (L15-16) transition\n"
        action_text += f"\n3. Bottleneck Analysis:\n"
        action_text += f"   • L{metrics['bottleneck_layer']} has only {metrics['bottleneck_width']} node(s)\n"
        action_text += "   • Critical choke point - test ablation"

        # Place action items BELOW the plot area
        fig.text(0.5, -0.05, action_text, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'),
                 family='monospace', transform=fig.transFigure, verticalalignment='top')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, right=0.85)  # Make room for legend on right and action items below
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[OK] Network structure visualization saved to: {output_path.name}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Extract minimal pathway from circuit')
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
            print("Please run Scripts 1-3 first.")
            return
    else:
        # Find available analyses and let user choose
        prompts = pm.get_all_analyzed_prompts()
        if not prompts:
            print("[ERROR] No analysis files found!")
            print("Please run Scripts 1-3 first to generate circuit analyses.")
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
                    # Use latest (first in list)
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
    output_dir = pm.minimal_pathways_dir(prompt)
    print(f"\nOutput directory: {output_dir}")

    # Extract minimal pathway
    extractor = MinimalPathwayExtractor(analysis_path)

    pathway_path = pm.minimal_pathways_result_path(prompt)
    minimal_G, metrics = extractor.extract_minimal_pathway(pathway_path)

    if minimal_G:
        # Create visualizations
        viz_path = pm.minimal_pathways_viz_path(prompt)
        extractor.visualize_comparison(minimal_G, metrics, viz_path)

        # Create network structure visualization
        struct_viz_path = pm.minimal_pathways_structure_viz_path(prompt)
        extractor.visualize_minimal_graph_structure(minimal_G, struct_viz_path, metrics)

        print("\n" + "="*60)
        print("MINIMAL PATHWAY EXTRACTION COMPLETE")
        print("="*60)
        print(f"Results: {pathway_path.name}")
        print(f"Comparison: {viz_path.name}")
        print(f"Network Structure: {struct_viz_path.name}")
        print(f"Location: {output_dir}")
    else:
        print("\n[ERROR] Failed to extract minimal pathway")


if __name__ == "__main__":
    main()
