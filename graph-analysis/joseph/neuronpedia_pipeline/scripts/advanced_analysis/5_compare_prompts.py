"""
Compare Multiple Attribution Graphs
Systematically compare circuits across different prompts
Supports 2-graph or multi-graph comparison with interactive selection
"""

import json
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from itertools import combinations

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def load_graph(graph_path):
    """Load converted graph and create NetworkX object"""
    with open(graph_path) as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    return G, data

def compare_graphs(graph1_path, graph2_path, prompt1_name, prompt2_name):
    """Compare two attribution graphs"""

    print("="*80)
    print(f"COMPARING: {prompt1_name} vs {prompt2_name}")
    print("="*80)

    # Load graphs
    G1, data1 = load_graph(graph1_path)
    G2, data2 = load_graph(graph2_path)

    print(f"\n{prompt1_name}:")
    print(f"  Nodes: {G1.number_of_nodes()}")
    print(f"  Edges: {G1.number_of_edges()}")
    print(f"  Density: {nx.density(G1):.6f}")

    print(f"\n{prompt2_name}:")
    print(f"  Nodes: {G2.number_of_nodes()}")
    print(f"  Edges: {G2.number_of_edges()}")
    print(f"  Density: {nx.density(G2):.6f}")

    # Feature overlap
    features1 = set(G1.nodes[n]['feature_id'] for n in G1.nodes())
    features2 = set(G2.nodes[n]['feature_id'] for n in G2.nodes())
    overlap = features1 & features2

    print(f"\n## FEATURE OVERLAP")
    print("-"*80)
    print(f"{prompt1_name} unique features: {len(features1)}")
    print(f"{prompt2_name} unique features: {len(features2)}")
    print(f"Overlapping features: {len(overlap)} ({len(overlap)/min(len(features1), len(features2))*100:.1f}%)")

    # Top features
    print(f"\n## TOP 5 FEATURES BY ACTIVATION")
    print("-"*80)

    print(f"\n{prompt1_name}:")
    top1 = sorted(G1.nodes(), key=lambda n: G1.nodes[n]['activation'], reverse=True)[:5]
    for i, node in enumerate(top1, 1):
        data = G1.nodes[node]
        print(f"  {i}. {data['label']}: {data['activation']:.3f} (L{data['layer']}, F{data['feature_id']})")

    print(f"\n{prompt2_name}:")
    top2 = sorted(G2.nodes(), key=lambda n: G2.nodes[n]['activation'], reverse=True)[:5]
    for i, node in enumerate(top2, 1):
        data = G2.nodes[node]
        print(f"  {i}. {data['label']}: {data['activation']:.3f} (L{data['layer']}, F{data['feature_id']})")

    # Check for shared top features
    top1_features = set(G1.nodes[n]['feature_id'] for n in top1)
    top2_features = set(G2.nodes[n]['feature_id'] for n in top2)
    shared_top = top1_features & top2_features

    if shared_top:
        print(f"\n⭐ SHARED TOP FEATURES: {shared_top}")
        for fid in shared_top:
            node1 = [n for n in G1.nodes() if G1.nodes[n]['feature_id'] == fid][0]
            node2 = [n for n in G2.nodes() if G2.nodes[n]['feature_id'] == fid][0]
            act1 = G1.nodes[node1]['activation']
            act2 = G2.nodes[node2]['activation']
            print(f"  F{fid}: {act1:.3f} vs {act2:.3f}")

    # Layer analysis
    print(f"\n## LAYER DISTRIBUTION")
    print("-"*80)

    layers1 = {}
    for node in G1.nodes():
        layer = G1.nodes[node]['layer']
        layers1[layer] = layers1.get(layer, 0) + 1

    layers2 = {}
    for node in G2.nodes():
        layer = G2.nodes[node]['layer']
        layers2[layer] = layers2.get(layer, 0) + 1

    print(f"\n{prompt1_name} - Top 5 active layers:")
    for layer, count in sorted(layers1.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Layer {layer}: {count} nodes")

    print(f"\n{prompt2_name} - Top 5 active layers:")
    for layer, count in sorted(layers2.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Layer {layer}: {count} nodes")

    # Inhibitory connections
    print(f"\n## EDGE WEIGHT ANALYSIS")
    print("-"*80)

    weights1 = [G1[u][v]['weight'] for u, v in G1.edges()]
    weights2 = [G2[u][v]['weight'] for u, v in G2.edges()]

    pos1 = sum(1 for w in weights1 if w > 0)
    neg1 = sum(1 for w in weights1 if w < 0)
    pos2 = sum(1 for w in weights2 if w > 0)
    neg2 = sum(1 for w in weights2 if w < 0)

    print(f"\n{prompt1_name}:")
    print(f"  Excitatory (positive): {pos1} ({pos1/len(weights1)*100:.1f}%)")
    print(f"  Inhibitory (negative): {neg1} ({neg1/len(weights1)*100:.1f}%)")

    print(f"\n{prompt2_name}:")
    print(f"  Excitatory (positive): {pos2} ({pos2/len(weights2)*100:.1f}%)")
    print(f"  Inhibitory (negative): {neg2} ({neg2/len(weights2)*100:.1f}%)")

    # Generate visualization
    print(f"\n## GENERATING COMPARISON VISUALIZATION")
    print("-"*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Layer distribution
    ax = axes[0, 0]
    all_layers = sorted(set(list(layers1.keys()) + list(layers2.keys())))
    counts1 = [layers1.get(l, 0) for l in all_layers]
    counts2 = [layers2.get(l, 0) for l in all_layers]

    x = np.arange(len(all_layers))
    width = 0.35
    ax.bar(x - width/2, counts1, width, label=prompt1_name, alpha=0.8, color='#FF6B6B')
    ax.bar(x + width/2, counts2, width, label=prompt2_name, alpha=0.8, color='#4ECDC4')

    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Number of Nodes', fontweight='bold')
    ax.set_title('Layer Distribution Comparison', fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(all_layers[::2], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Activation distribution
    ax = axes[0, 1]
    acts1 = [G1.nodes[n]['activation'] for n in G1.nodes()]
    acts2 = [G2.nodes[n]['activation'] for n in G2.nodes()]

    ax.hist(acts1, bins=50, alpha=0.6, label=prompt1_name, color='#FF6B6B', density=True)
    ax.hist(acts2, bins=50, alpha=0.6, label=prompt2_name, color='#4ECDC4', density=True)
    ax.set_xlabel('Activation', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Activation Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Edge weight distribution
    ax = axes[1, 0]
    ax.hist(weights1, bins=50, alpha=0.6, label=prompt1_name, color='#FF6B6B', density=True)
    ax.hist(weights2, bins=50, alpha=0.6, label=prompt2_name, color='#4ECDC4', density=True)
    ax.set_xlabel('Edge Weight', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Edge Weight Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Summary stats
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
COMPARISON SUMMARY

Prompt 1: {prompt1_name}
Prompt 2: {prompt2_name}

COMPLEXITY:
• Nodes: {G1.number_of_nodes()} vs {G2.number_of_nodes()}
• Edges: {G1.number_of_edges()} vs {G2.number_of_edges()}
• Density: {nx.density(G1):.4f} vs {nx.density(G2):.4f}

ACTIVATION:
• Mean: {np.mean(acts1):.3f} vs {np.mean(acts2):.3f}
• Max: {max(acts1):.1f} vs {max(acts2):.1f}

OVERLAP:
• Shared features: {len(overlap)} ({len(overlap)/min(len(features1), len(features2))*100:.1f}%)
• Shared top-5: {len(shared_top)}

INHIBITION:
• {prompt1_name}: {neg1/len(weights1)*100:.1f}% negative
• {prompt2_name}: {neg2/len(weights2)*100:.1f}% negative
"""

    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save
    # Use absolute path based on script location (scripts/advanced_analysis/ -> neuronpedia_pipeline/)
    output_path = Path(__file__).parent.parent.parent / "data" / "visualizations" / f"comparison_{prompt1_name}_vs_{prompt2_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return {
        'overlap': len(overlap),
        'overlap_pct': len(overlap)/min(len(features1), len(features2))*100,
        'shared_top_features': shared_top,
        'nodes': (G1.number_of_nodes(), G2.number_of_nodes()),
        'edges': (G1.number_of_edges(), G2.number_of_edges())
    }

def find_available_graphs():
    """Find all converted graphs in the data directory"""
    # parent.parent.parent because we're in scripts/advanced_analysis/
    graphs_dir = Path(__file__).parent.parent.parent / "data" / "graphs"

    if not graphs_dir.exists():
        return []

    # Find all *_converted.json files
    converted_graphs = sorted(graphs_dir.glob("*_converted.json"))
    return converted_graphs

def display_available_graphs(graphs):
    """Display numbered list of available graphs"""
    print("\n" + "=" * 60)
    print("AVAILABLE CONVERTED GRAPHS")
    print("=" * 60)

    if not graphs:
        print("No converted graphs found!")
        print("Please run '/circuit-tracer-fetch' and '/circuit-tracer-convert' first.")
        return

    for i, graph_path in enumerate(graphs, 1):
        # Get file size
        size_mb = graph_path.stat().st_size / (1024 * 1024)
        print(f"{i}. {graph_path.name} ({size_mb:.2f} MB)")

    print("=" * 60)

def get_user_selection(graphs, mode='2-graph'):
    """Get user's graph selection interactively"""

    if mode == '2-graph':
        print("\nSelect two graphs to compare:")
        print("-" * 60)

        # Get first graph
        while True:
            try:
                choice1 = input("Select first graph (enter number): ").strip()
                idx1 = int(choice1) - 1
                if 0 <= idx1 < len(graphs):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(graphs)}")
            except ValueError:
                print("Please enter a valid number")

        # Get second graph
        while True:
            try:
                choice2 = input("Select second graph (enter number): ").strip()
                idx2 = int(choice2) - 1
                if 0 <= idx2 < len(graphs):
                    if idx2 != idx1:
                        break
                    else:
                        print("Please select a different graph")
                else:
                    print(f"Please enter a number between 1 and {len(graphs)}")
            except ValueError:
                print("Please enter a valid number")

        return [(graphs[idx1], graphs[idx2])]

    else:  # multi-graph mode
        print("\nSelect multiple graphs to compare:")
        print("-" * 60)
        print("Enter numbers separated by commas (e.g., 1,2,3)")

        while True:
            try:
                choices = input("Select graphs: ").strip()
                indices = [int(x.strip()) - 1 for x in choices.split(',')]

                # Validate all indices
                if all(0 <= idx < len(graphs) for idx in indices):
                    if len(indices) < 2:
                        print("Please select at least 2 graphs")
                        continue
                    if len(set(indices)) != len(indices):
                        print("Please don't select the same graph multiple times")
                        continue

                    # Return all pairs (combinations)
                    selected_graphs = [graphs[i] for i in indices]
                    pairs = list(combinations(selected_graphs, 2))

                    print(f"\nWill compare {len(pairs)} pairs:")
                    for g1, g2 in pairs:
                        print(f"  - {g1.name} vs {g2.name}")

                    confirm = input("\nProceed? (y/n): ").strip().lower()
                    if confirm == 'y':
                        return pairs
                    else:
                        continue
                else:
                    print(f"Please enter numbers between 1 and {len(graphs)}")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas")

def get_graph_name(graph_path):
    """Extract a clean name from graph filename"""
    # Remove 'real_' prefix and '_converted.json' suffix
    name = graph_path.stem.replace('real_', '').replace('_converted', '')
    # Replace underscores with spaces and title case
    name = name.replace('_', ' ').title()
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare attribution graphs across multiple prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python 5_compare_prompts.py  (interactive mode)
  python 5_compare_prompts.py graph1.json graph2.json --name1 "Japan" --name2 "France"
        '''
    )
    parser.add_argument(
        'graphs',
        nargs='*',
        help='Paths to graphs (if not provided, will use interactive mode)'
    )
    parser.add_argument('--name1', help='Name for first graph')
    parser.add_argument('--name2', help='Name for second graph')

    args = parser.parse_args()

    print("=" * 60)
    print("CIRCUIT TRACER - GRAPH COMPARISON")
    print("=" * 60)

    # Check if command-line graphs were provided
    if args.graphs and len(args.graphs) >= 2:
        # Command-line mode
        print("\nUsing graphs from command-line arguments")
        graph1_path = Path(args.graphs[0])
        graph2_path = Path(args.graphs[1])
        name1 = args.name1 or get_graph_name(graph1_path)
        name2 = args.name2 or get_graph_name(graph2_path)

        results = compare_graphs(graph1_path, graph2_path, name1, name2)

    else:
        # Interactive mode
        print("\nEntering interactive mode...")

        # Find available graphs
        available_graphs = find_available_graphs()

        if not available_graphs:
            print("\n[ERROR] No converted graphs found!")
            print("Please run the following first:")
            print("  1. /circuit-tracer-fetch")
            print("  2. /circuit-tracer-convert")
            sys.exit(1)

        if len(available_graphs) < 2:
            print("\n[ERROR] Need at least 2 converted graphs to compare!")
            print(f"Found only {len(available_graphs)} graph(s).")
            print("Please fetch and convert more graphs first.")
            sys.exit(1)

        # Display available graphs
        display_available_graphs(available_graphs)

        # Ask user: 2-graph or multi-graph comparison?
        print("\n" + "=" * 60)
        print("COMPARISON MODE")
        print("=" * 60)
        print("1. Two-graph comparison (compare 2 graphs)")
        print("2. Multi-graph comparison (compare multiple graphs, all pairs)")
        print("=" * 60)

        while True:
            mode_choice = input("Select mode (1 or 2): ").strip()
            if mode_choice in ['1', '2']:
                break
            print("Please enter 1 or 2")

        comparison_mode = '2-graph' if mode_choice == '1' else 'multi-graph'

        # Get user's graph selections
        graph_pairs = get_user_selection(available_graphs, mode=comparison_mode)

        # Perform comparisons
        print("\n" + "=" * 60)
        print(f"STARTING {len(graph_pairs)} COMPARISON(S)")
        print("=" * 60)

        all_results = []
        for i, (graph1_path, graph2_path) in enumerate(graph_pairs, 1):
            name1 = get_graph_name(graph1_path)
            name2 = get_graph_name(graph2_path)

            print(f"\n[{i}/{len(graph_pairs)}] Comparing: {name1} vs {name2}")
            print("-" * 60)

            results = compare_graphs(graph1_path, graph2_path, name1, name2)
            all_results.append({
                'graphs': (name1, name2),
                'results': results
            })

            if i < len(graph_pairs):
                print("\n" + "=" * 60)

        # Summary for multi-graph mode
        if comparison_mode == 'multi-graph' and len(all_results) > 1:
            print("\n" + "=" * 60)
            print("MULTI-GRAPH COMPARISON SUMMARY")
            print("=" * 60)

            for i, item in enumerate(all_results, 1):
                g1, g2 = item['graphs']
                res = item['results']
                print(f"\n{i}. {g1} vs {g2}:")
                print(f"   Overlap: {res['overlap_pct']:.1f}% ({res['overlap']} features)")
                print(f"   Shared top-5: {len(res['shared_top_features'])}")
                print(f"   Nodes: {res['nodes'][0]} vs {res['nodes'][1]}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
