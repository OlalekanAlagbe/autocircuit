"""
Visualize Real Neuronpedia Data
Creates PNG visualizations of converted graphs and supernodes
Supports interactive selection or command-line file specification
"""

import json
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering
matplotlib.rcParams['text.parse_math'] = False  # Disable math parsing entirely
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import argparse
import sys
from collections import Counter, defaultdict
import re

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from path_manager import PathManager

def escape_matplotlib_text(text):
    r"""
    Remove special characters that matplotlib interprets as LaTeX math mode.
    Removes: $ { } _ ^ # % & \ to prevent math mode parsing errors
    """
    if not isinstance(text, str):
        return str(text)

    # Just remove problematic characters entirely
    return (text
            .replace('$', '')
            .replace('{', '(')
            .replace('}', ')')
            .replace('_', '-')
            .replace('^', '')
            .replace('#', '')
            .replace('%', 'pct')
            .replace('&', 'and')
            .replace('\\', ''))

def infer_supernode_theme(supernode_nodes, G, top_n=10):
    """
    Infer the semantic theme of a supernode based on its feature descriptions

    Args:
        supernode_nodes: List of node IDs in this supernode
        G: NetworkX graph with node attributes
        top_n: Number of top features to analyze

    Returns:
        String describing the supernode's theme
    """
    # Collect descriptions from nodes in this supernode
    descriptions = []
    activations = []

    for node in supernode_nodes:
        if node in G.nodes():
            desc = G.nodes[node].get('description', '')
            act = G.nodes[node].get('activation', 0)
            if desc and isinstance(desc, str) and len(desc) > 5:
                descriptions.append(desc)
                activations.append(act)

    if not descriptions:
        return "Unknown"

    # Sort by activation and take top features
    sorted_pairs = sorted(zip(descriptions, activations), key=lambda x: x[1], reverse=True)
    top_descriptions = [desc for desc, _ in sorted_pairs[:top_n]]

    # Extract keywords from descriptions
    all_text = ' '.join(top_descriptions).lower()

    # Remove common prefixes like "Activates on:"
    all_text = all_text.replace('activates on:', '').replace('activates on', '')

    # Define theme keywords
    theme_patterns = {
        'Geographic': ['city', 'cities', 'state', 'states', 'country', 'countries', 'capitol', 'capital',
                       'located', 'location', 'place', 'region', 'area', 'territory', 'geographic'],
        'Political': ['president', 'political', 'party', 'republican', 'democratic', 'election',
                      'government', 'congress', 'senator', 'politics', 'administration'],
        'Numerical': ['number', 'numbers', 'digit', 'digits', 'count', 'quantity', 'amount',
                      'calculate', 'math', 'numeric', 'integer'],
        'Temporal': ['time', 'date', 'year', 'years', 'month', 'day', 'when', 'period',
                     'century', 'decade', 'era', 'season', 'temporal'],
        'Entity': ['name', 'names', 'person', 'people', 'entity', 'entities', 'who',
                   'individual', 'character'],
        'Syntactic': ['token', 'tokens', 'word', 'words', 'syntax', 'grammar', 'parsing',
                      'structure', 'sentence', 'phrase'],
        'Semantic': ['meaning', 'concept', 'idea', 'semantic', 'understanding', 'interpretation',
                     'context', 'relation', 'relationship']
    }

    # Count theme matches
    theme_scores = {}
    for theme, keywords in theme_patterns.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        if score > 0:
            theme_scores[theme] = score

    # Return top theme or generic description
    if theme_scores:
        top_theme = max(theme_scores, key=theme_scores.get)
        return top_theme

    # Fallback: try to extract most common meaningful words
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    word_counts = Counter(words)
    # Filter out very common words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'they', 'been', 'were', 'their',
                  'would', 'there', 'could', 'which', 'these', 'those', 'about', 'after'}
    meaningful_words = [(w, c) for w, c in word_counts.most_common(5) if w not in stop_words]

    if meaningful_words and meaningful_words[0][1] >= 2:
        return meaningful_words[0][0].capitalize()

    return "Processing"


def load_feature_descriptions_from_analysis(analysis):
    """
    Extract all feature descriptions from analysis JSON

    Args:
        analysis: Dict containing analysis results

    Returns:
        Dict mapping feature IDs to descriptions
    """
    descriptions = {}

    # First, check if we have ALL features descriptions (from comprehensive fetch)
    all_features = analysis.get('feature_descriptions', {}).get('all_features', {})
    if all_features:
        print(f"[INFO] Loading comprehensive feature descriptions ({len(all_features)} features)")
        descriptions.update(all_features)

    # From layer groups (top features with descriptions)
    for group_name, group_data in analysis.get('layer_groups', {}).items():
        for feat in group_data.get('top_features_with_descriptions', []):
            feature_id = feat.get('feature', '')
            desc = feat.get('description', '')
            if feature_id and desc:
                descriptions[feature_id] = desc

    # From steering targets
    for target_id, target_data in analysis.get('steering_targets', {}).items():
        if 'description' in target_data:
            feature_id = target_data.get('feature', target_id)
            descriptions[feature_id] = target_data['description']

    return descriptions


def get_neuronpedia_url(node_id):
    """
    Generate Neuronpedia URL for a feature node

    Args:
        node_id: Feature ID in format "L{layer}_F{feature}"

    Returns:
        URL string to Neuronpedia page for this feature
    """
    # Extract layer and feature from node_id
    # Format: L12_F1234 -> layer=12, feature=1234
    try:
        parts = node_id.split('_')
        layer = parts[0][1:]  # Remove 'L' prefix
        feature = parts[1][1:]  # Remove 'F' prefix
        # Neuronpedia URL format: https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature}
        return f"https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{feature}"
    except (IndexError, ValueError):
        return None


def extract_clean_tokens(description, max_tokens=3):
    """
    Extract clean, readable tokens from Neuronpedia description

    Args:
        description: Feature description from Neuronpedia
        max_tokens: Maximum number of tokens to extract

    Returns:
        Clean comma-separated tokens or shortened description
    """
    if not description:
        return "No description"

    # Extract tokens from "Activates on: token1, token2, token3" format
    if "Activates on:" in description:
        tokens_part = description.split("Activates on:")[1].strip()
        # Split by comma and clean up
        tokens = [t.strip().strip('"').strip("'") for t in tokens_part.split(",")]
        # Filter out empty and very long tokens
        tokens = [t for t in tokens if t and len(t) < 30]
        # Take first N tokens
        if tokens:
            return ", ".join(tokens[:max_tokens])

    # Fallback: return first 50 chars at word boundary
    if len(description) > 50:
        # Find last space before char 50
        truncated = description[:50]
        last_space = truncated.rfind(' ')
        if last_space > 20:  # Only truncate at space if it's not too early
            return truncated[:last_space] + "..."
        return truncated + "..."

    return description


def infer_theme_from_descriptions(feature_descriptions_list, min_descriptions=5):
    """
    Infer semantic theme from a list of feature descriptions

    Args:
        feature_descriptions_list: List of description strings
        min_descriptions: Minimum number of descriptions required for inference

    Returns:
        Inferred theme string (e.g., "Geographic", "Syntax", "Names") or "Insufficient Data"
    """
    if not feature_descriptions_list:
        return "Insufficient Data"

    # If we have too few descriptions, don't try to infer
    if len(feature_descriptions_list) < min_descriptions:
        return "Insufficient Data"

    # Extract all tokens from descriptions
    all_tokens = []
    for desc in feature_descriptions_list[:10]:  # Look at top 10 features
        if desc and "Activates on:" in desc:
            tokens_part = desc.split("Activates on:")[1].strip()
            tokens = [t.strip().strip('"').strip("'").lower() for t in tokens_part.split(",")[:5]]
            all_tokens.extend(tokens)

    if not all_tokens:
        return "Insufficient Data"

    # Theme detection patterns
    geographic_keywords = ['paris', 'france', 'london', 'city', 'capital', 'country', 'state', 'york', 'albany']
    syntax_keywords = ['the', 'of', 'is', 'in', 'on', 'at', 'to', 'and', 'or', 'a', 'an']
    punctuation_keywords = ['.', ',', ';', ':', '!', '?', '"', "'"]
    number_keywords = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'zero', 'one', 'two']

    # Count matches
    geo_count = sum(1 for t in all_tokens if any(kw in t for kw in geographic_keywords))
    syntax_count = sum(1 for t in all_tokens if t in syntax_keywords)
    punct_count = sum(1 for t in all_tokens if t in punctuation_keywords)
    number_count = sum(1 for t in all_tokens if any(kw in t for kw in number_keywords))

    # Determine theme based on highest count
    counts = [
        (geo_count, "Geographic"),
        (syntax_count, "Syntax"),
        (punct_count, "Punctuation"),
        (number_count, "Numbers")
    ]

    max_count, theme = max(counts, key=lambda x: x[0])

    if max_count > 0:
        return theme

    # Fallback: use first few tokens as theme
    unique_tokens = list(dict.fromkeys(all_tokens))[:3]  # Remove duplicates, keep order
    if unique_tokens:
        return ", ".join(unique_tokens).title()

    return "Insufficient Data"


def get_top_features_for_supernode(sn_id, analysis, feature_descriptions, n=3):
    """
    Get top N features from a supernode with their descriptions

    Args:
        sn_id: Supernode ID (as string)
        analysis: Analysis dictionary
        feature_descriptions: Dict of feature ID -> description
        n: Number of top features to return

    Returns:
        List of (feature_id, activation, description) tuples
    """
    supernodes = analysis.get('louvain_supernodes', {}).get('supernodes', {})
    sn_data = supernodes.get(str(sn_id), {})

    # Get node list with activations
    nodes = sn_data.get('nodes', [])
    if not nodes:
        return []

    # Sort by activation (stored in analysis)
    # Feature activations are in layer_groups
    feature_acts = {}
    for group_name, group_data in analysis.get('layer_groups', {}).items():
        for feat in group_data.get('features', []):
            feature_acts[feat['feature']] = feat['activation']

    # Get top N features by activation that HAVE descriptions
    top_features = []
    for node_id in nodes:
        # Only include features that have descriptions
        if node_id in feature_descriptions:
            activation = feature_acts.get(node_id, 0)
            description = feature_descriptions[node_id]
            top_features.append((node_id, activation, description))

    # Sort by activation and take top N
    top_features.sort(key=lambda x: x[1], reverse=True)
    return top_features[:n]


def find_available_converted_graphs():
    """Find all converted graphs in the new PathManager structure"""
    pm = PathManager()

    converted_graphs = []

    # Look in new structure: data/prompts/*/2_conversion/converted_graph.json
    prompts_dir = pm.prompts_dir
    if prompts_dir.exists():
        for prompt_dir in prompts_dir.iterdir():
            if prompt_dir.is_dir():
                conv_dir = prompt_dir / '2_conversion'
                if conv_dir.exists():
                    # Search for both naming patterns
                    conv_graphs = list(conv_dir.glob('*converted_graph.json'))
                    if not conv_graphs:
                        conv_graphs = list(conv_dir.glob('converted_graph.json'))
                    # Use the first match if any
                    if conv_graphs:
                        converted_graphs.append(conv_graphs[0])

    # Also check old location for backward compatibility
    old_graphs_dir = pm.base_dir / "graphs"
    if old_graphs_dir.exists():
        old_converted = list(old_graphs_dir.glob("*_converted.json"))
        converted_graphs.extend(old_converted)

    # Sort by modification time (newest first)
    converted_graphs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return converted_graphs

def display_available_graphs(graphs):
    """Display numbered list of available converted graphs"""
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

        # Check if supernodes exist
        supernode_path = graph_path.parent / f"{graph_path.stem.replace('_converted', '')}_supernodes.json"
        status = "[HAS SUPERNODES]" if supernode_path.exists() else "[NO SUPERNODES]"

        print(f"{i}. {graph_path.name} ({size_mb:.2f} MB) {status}")

    print("=" * 60)

def get_user_selection(graphs):
    """Get user's graph selection interactively"""
    print("\nSelect graph to visualize:")
    print("-" * 60)

    while True:
        try:
            choice = input("Select graph (enter number, or press Enter for latest): ").strip()

            if choice == "":
                # Use latest (first in list)
                return graphs[0]

            idx = int(choice) - 1
            if 0 <= idx < len(graphs):
                return graphs[idx]
            else:
                print(f"Please enter a number between 1 and {len(graphs)}")
        except ValueError:
            print("Please enter a valid number or press Enter for latest")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Visualize Circuit Tracer graph with supernodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
    Examples:
      python 4_visualize.py  (interactive mode)
      python 4_visualize.py --file path/to/graph_converted.json
        '''
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to specific converted graph file (if not provided, will use interactive mode)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CIRCUIT TRACER - VISUALIZATION")
    print("=" * 60)

    # Determine input file
    if args.file:
        # Command-line mode
        graph_file = Path(args.file)
        if not graph_file.exists():
            print(f"\n[ERROR] File not found: {graph_file}")
            sys.exit(1)
        print(f"\nUsing file from command-line: {graph_file.name}")
    else:
        # Interactive mode
        print("\nEntering interactive mode...")

        # Find available graphs
        available_graphs = find_available_converted_graphs()

        if not available_graphs:
            print("\n[ERROR] No converted graphs found!")
            print("Please run '/circuit-tracer-fetch' and '/circuit-tracer-convert' first.")
            sys.exit(1)

        # Display available graphs
        display_available_graphs(available_graphs)

        # Get user selection
        graph_file = get_user_selection(available_graphs)
        print(f"\nSelected: {graph_file.name}")

    # Try to load metadata to determine subfolder structure
    with open(graph_file) as f:
        graph_data_preview = json.load(f)
        metadata = graph_data_preview.get('metadata', {})
        prompt_slug = metadata.get('slug', 'unknown')

    # Use PathManager to find analysis files
    pm = PathManager()
    prompt = metadata.get('prompt', '')
    model_id = metadata.get('model', metadata.get('modelId', ''))

    # If no prompt in metadata, try to extract from slug
    if not prompt:
        prompt = prompt_slug.replace('-', ' ').title()

    # Get analysis file paths using PathManager
    analysis_file = pm.circuit_analysis_path(prompt, model_id=model_id)
    supernode_file = pm.supernodes_path(prompt, model_id=model_id)

    # Fall back to old structure if not found
    if not analysis_file.exists():
        base_filename = graph_file.stem.replace('_converted', '')
        output_subdir = graph_file.parent / prompt_slug
        analysis_file = output_subdir / f"{base_filename}_analysis.json"
        if not analysis_file.exists():
            analysis_file = graph_file.parent / f"{base_filename}_analysis.json"

    if not supernode_file.exists():
        base_filename = graph_file.stem.replace('_converted', '')
        output_subdir = graph_file.parent / prompt_slug
        supernode_file = output_subdir / f"{base_filename}_supernodes.json"
        if not supernode_file.exists():
            supernode_file = graph_file.parent / f"{base_filename}_supernodes.json"

    # Check for comprehensive analysis file first, fallback to legacy supernodes
    has_comprehensive_analysis = analysis_file.exists()

    if not has_comprehensive_analysis and not supernode_file.exists():
        print(f"\n[ERROR] Analysis file not found: {analysis_file.name}")
        print("Please run '/circuit-tracer-analyze' first to generate analysis.")
        sys.exit(1)

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    print(f"Graph: {graph_file.name}")

    with open(graph_file) as f:
        graph_data = json.load(f)

    # Load comprehensive analysis or fallback to legacy supernodes
    layer_groups = None
    flow_analysis = None

    if has_comprehensive_analysis:
        print(f"Analysis: {analysis_file.name} (comprehensive)")
        with open(analysis_file) as f:
            analysis_data = json.load(f)
        supernodes = {k: v['nodes'] for k, v in analysis_data['louvain_supernodes']['supernodes'].items()}
        layer_groups = analysis_data.get('layer_groups', None)
        flow_analysis = analysis_data.get('flow_analysis', None)

        # Load feature descriptions from analysis
        feature_descriptions = load_feature_descriptions_from_analysis(analysis_data)
        print(f"Loaded {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges, {len(feature_descriptions)} feature descriptions")
        print(f"Loaded {len(supernodes)} supernodes, {len(layer_groups) if layer_groups else 0} layer groups")
    else:
        print(f"Supernodes: {supernode_file.name} (legacy)")
        with open(supernode_file) as f:
            supernodes = json.load(f)
        feature_descriptions = {}  # No descriptions in legacy mode
        analysis_data = {}  # Empty for legacy mode
        print(f"Loaded {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
        print(f"Loaded {len(supernodes)} supernodes")
        print("[WARNING] Legacy format detected. Run '/circuit-tracer-analyze' again for layer-based visualizations.")

    # Extract prompt for titles
    prompt_text = graph_data.get('metadata', {}).get('prompt', 'Unknown prompt')
    # Truncate if too long
    if len(prompt_text) > 60:
        prompt_text = prompt_text[:57] + "..."

    # Create NetworkX graph
    G = nx.DiGraph()

    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            layer=node['layer'],
            activation=node['activation'],
            influence=node.get('influence', 0.0),
            label=node['label']
        )

    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    # Assign supernode membership
    node_to_supernode = {}
    supernode_ids = []
    for sn_id, node_list in supernodes.items():
        sn_id_int = int(sn_id)
        supernode_ids.append(sn_id_int)
        for node in node_list:
            node_to_supernode[node] = sn_id_int

    supernode_ids = sorted(supernode_ids)
    print(f"Supernode IDs: {supernode_ids}")

    # Generate colors dynamically for all supernodes
    def generate_colors(n):
        """Generate n distinct colors"""
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
        return colors

    color_list = generate_colors(len(supernode_ids))
    supernode_colors = {sn_id: color_list[i] for i, sn_id in enumerate(supernode_ids)}

    print("\n" + "="*60)
    print("VISUALIZATION 1: Supernode Overview (Compressed)")
    print("="*60)

    # For large graphs, we'll create a compressed view showing supernode-level connections
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create supernode graph (compressed view)
    SG = nx.DiGraph()
    supernode_stats = {}

    for sn_id, nodes in supernodes.items():
        sn_id_int = int(sn_id)

        # Calculate supernode statistics
        activations = [G.nodes[n]['activation'] for n in nodes if n in G.nodes()]
        influences = [G.nodes[n].get('influence', 0.0) for n in nodes if n in G.nodes()]
        layers = [G.nodes[n]['layer'] for n in nodes if n in G.nodes()]

        mean_layer = np.mean(layers) if layers else 0
        mean_activation = np.mean(activations) if activations else 0

        # Infer theme for this supernode
        theme = infer_supernode_theme(nodes, G, top_n=15)

        supernode_stats[sn_id_int] = {
            'size': len(nodes),
            'mean_layer': mean_layer,
            'mean_activation': mean_activation,
            'mean_influence': np.mean(influences) if influences else 0,
            'layers': sorted(set(layers)),
            'theme': theme
        }

        SG.add_node(sn_id_int, **supernode_stats[sn_id_int])

    # Add edges between supernodes
    supernode_edges = {}
    for u, v in G.edges():
        if u in node_to_supernode and v in node_to_supernode:
            sn_u = node_to_supernode[u]
            sn_v = node_to_supernode[v]
            if sn_u != sn_v:
                key = (sn_u, sn_v)
                supernode_edges[key] = supernode_edges.get(key, 0) + 1

    for (sn_u, sn_v), count in supernode_edges.items():
        SG.add_edge(sn_u, sn_v, weight=count)

    # Layout: position by mean layer with better vertical spacing
    pos = {}
    # Group supernodes by layer range for better spacing
    layer_bins = defaultdict(list)
    for sn_id in SG.nodes():
        mean_layer = supernode_stats[sn_id]['mean_layer']
        layer_bin = int(mean_layer / 5)  # Group into bins of 5 layers
        layer_bins[layer_bin].append((sn_id, mean_layer))

    # Position nodes with spacing
    for sn_id in SG.nodes():
        mean_layer = supernode_stats[sn_id]['mean_layer']
        layer_bin = int(mean_layer / 5)

        # Find index within bin for vertical offset
        bin_nodes = sorted(layer_bins[layer_bin], key=lambda x: x[1])
        idx = next(i for i, (sid, _) in enumerate(bin_nodes) if sid == sn_id)

        # Spread vertically with more space
        y_offset = (idx - len(bin_nodes)/2) * 4  # Increased spacing
        pos[sn_id] = (mean_layer, y_offset)

    # Draw supernodes
    node_sizes = [supernode_stats[sn_id]['size'] * 50 for sn_id in SG.nodes()]
    node_colors = [supernode_colors.get(sn_id, '#CCCCCC') for sn_id in SG.nodes()]

    nx.draw_networkx_nodes(
        SG, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )

    # Draw edges with weights
    edge_widths = [SG[u][v]['weight'] / 100 for u, v in SG.edges()]
    nx.draw_networkx_edges(
        SG, pos,
        width=edge_widths,
        alpha=0.3,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        ax=ax
    )

    # Infer themes for all supernodes based on their feature descriptions
    supernode_themes = {}
    for sn_id in SG.nodes():
        if has_comprehensive_analysis and feature_descriptions:
            top_features = get_top_features_for_supernode(sn_id, analysis_data, feature_descriptions, n=10)
            descriptions = [desc for _, _, desc in top_features]
            theme = infer_theme_from_descriptions(descriptions)
            supernode_themes[sn_id] = theme
        else:
            supernode_themes[sn_id] = "Unknown"

    # Labels - highlight output supernodes (L21-25) and include theme and feature descriptions
    labels = {}
    for sn_id in SG.nodes():
        stats = supernode_stats[sn_id]
        max_layer = max(stats['layers'])
        theme = supernode_themes.get(sn_id, 'Unknown')

        # Get top features for this supernode (with descriptions)
        if has_comprehensive_analysis and feature_descriptions:
            top_features = get_top_features_for_supernode(sn_id, analysis_data, feature_descriptions, n=2)
        else:
            top_features = []

        # Build label with feature descriptions
        # Escape matplotlib special characters in theme
        safe_theme = escape_matplotlib_text(theme)
        label = f"SN{sn_id}: {safe_theme}\n"

        # Add feature descriptions if available
        if top_features:
            for node_id, _, desc in top_features[:2]:  # Top 2 features
                # Extract clean tokens using new function
                clean_tokens = extract_clean_tokens(desc, max_tokens=2)
                # Escape matplotlib special characters
                clean_tokens = escape_matplotlib_text(clean_tokens)
                label += f"• {clean_tokens}\n"
        else:
            # No feature descriptions available - add note
            label += f"({stats['size']} features)\n"

        # Check if this is an output supernode
        if max_layer >= 21:
            # This is an output supernode - find the target logit token
            target_token = None
            target_prob = None

            # Look for target logit node in the graph
            for node in graph_data.get('nodes', []):
                if node.get('is_target_logit') and node.get('clerp'):
                    # Extract token and prob from clerp: 'Output " the" (p=0.335)'
                    clerp = node['clerp']
                    if 'Output' in clerp and '(p=' in clerp:
                        # Extract token between quotes
                        start = clerp.find('"') + 1
                        end = clerp.find('"', start)
                        target_token = clerp[start:end]
                        # Extract probability
                        target_prob = node.get('token_prob', 0)
                    break

            label += f"[OUTPUT] {stats['size']} nodes\nL{min(stats['layers'])}-{max(stats['layers'])} | Influence: {stats['mean_activation']:.1f}"
            if target_token:
                # Escape matplotlib special characters in output token
                safe_token = escape_matplotlib_text(target_token)
                label += f"\n>>> \"{safe_token}\" (p={target_prob:.1%})"
        else:
            label += f"{stats['size']} nodes\nL{min(stats['layers'])}-{max(stats['layers'])} | Act: {stats['mean_activation']:.1f}"

        labels[sn_id] = label

    nx.draw_networkx_labels(SG, pos, labels, font_size=7, font_weight='bold', ax=ax)

    ax.set_title(f'Supernode-Level Circuit\n"{prompt_text}"',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Layer', fontsize=12)
    ax.axis('off')

    # Add Model Predictions Box in upper left
    if has_comprehensive_analysis and 'metadata' in analysis_data:
        metadata = analysis_data['metadata']
        model_output = metadata.get('model_output', 'Unknown')
        output_prob = metadata.get('output_probability', 0)
        top_preds = metadata.get('top_predictions', [])

        # Build predictions text
        pred_text = "MODEL PREDICTIONS:\n\n"
        pred_text += f"Prompt: {prompt_text}\n\n"

        if top_preds:
            pred_text += "Top 5 Outputs:\n"
            for i, pred in enumerate(top_preds[:5], 1):
                token = pred.get('token', '?')
                prob = pred.get('probability', 0)
                # Highlight the actual output
                if token == model_output:
                    marker = ">>> "
                else:
                    marker = f" {i}. "
                # Color code by confidence
                pred_text += f"{marker}\"{token}\" ({prob:.1%})\n"
        else:
            # Fallback if no top_predictions
            pred_text += f"Output: \"{model_output}\" ({output_prob:.1%})\n"

        # Determine box color based on output probability
        if output_prob > 0.5:
            box_color = 'lightgreen'
        elif output_prob > 0.2:
            box_color = 'lightyellow'
        else:
            box_color = 'lightcoral'

        # Place box in upper left of plot area
        ax.text(0.02, 0.98, pred_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9, edgecolor='black'),
                family='monospace')

    # Legend - dynamically generate based on actual supernodes with inferred themes
    legend_elements = []
    for sn_id in sorted(supernode_stats.keys())[:10]:  # Show up to 10 supernodes
        stats = supernode_stats[sn_id]
        color = supernode_colors.get(sn_id, '#CCCCCC')
        theme = supernode_themes.get(sn_id, 'Unknown')
        label = f"SN{sn_id} [{theme}]: {stats['size']} nodes (L{min(stats['layers'])}-{max(stats['layers'])})"
        legend_elements.append(mpatches.Patch(color=color, label=label))

    if legend_elements:
        # Place legend OUTSIDE plot area on the right
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0.9, edgecolor='black', borderaxespad=0)

    # Add action items box with semantic meaning
    # Find output supernodes (L21+)
    output_supernodes = [sn_id for sn_id in supernode_stats.keys()
                         if max(supernode_stats[sn_id]['layers']) >= 21]

    # Find most connected supernodes (highest degree)
    most_connected = sorted(SG.nodes(), key=lambda n: SG.degree(n), reverse=True)[:3]

    action_text = "NEXT STEPS:\n\n"

    # Action 1: Output supernodes with semantic info
    action_text += f"1. Investigate Output Supernodes:\n"
    if output_supernodes:
        out_sn = output_supernodes[0]
        out_theme = supernode_themes.get(out_sn, 'Unknown')
        out_layers = f"L{min(supernode_stats[out_sn]['layers'])}-{max(supernode_stats[out_sn]['layers'])}"

        # Get top features for output supernode
        if has_comprehensive_analysis and feature_descriptions:
            out_features = get_top_features_for_supernode(out_sn, analysis_data, feature_descriptions, n=2)
            if out_features:
                out_tokens = extract_clean_tokens(out_features[0][2], max_tokens=2)
                action_text += f"   • SN{out_sn} ({out_theme}): \"{out_tokens}\" - drives final token\n"
            else:
                action_text += f"   • SN{out_sn} ({out_theme}) - drives final token\n"
        else:
            action_text += f"   • SN{out_sn} - drives final token\n"

        action_text += f"   • Test ablation to measure impact on output\n"
        if has_comprehensive_analysis and feature_descriptions and out_features:
            top_node = out_features[0][0]
            url = get_neuronpedia_url(top_node)
            if url:
                action_text += f"   • Inspect: {url}\n\n"
            else:
                action_text += "\n"
        else:
            action_text += "\n"
    else:
        action_text += f"   • No output-layer supernodes detected\n\n"

    # Action 2: Hub supernodes with semantic info
    action_text += f"2. Test Hub Supernodes:\n"
    for sn_id in most_connected[:2]:
        hub_theme = supernode_themes.get(sn_id, 'Unknown')
        hub_layers = f"L{min(supernode_stats[sn_id]['layers'])}-{max(supernode_stats[sn_id]['layers'])}"

        # Get semantic description
        if has_comprehensive_analysis and feature_descriptions:
            hub_features = get_top_features_for_supernode(sn_id, analysis_data, feature_descriptions, n=1)
            if hub_features:
                hub_tokens = extract_clean_tokens(hub_features[0][2], max_tokens=2)
                action_text += f"   • SN{sn_id} ({hub_theme}): \"{hub_tokens}\" - {SG.degree(sn_id)} connections\n"
            else:
                action_text += f"   • SN{sn_id} ({hub_theme}) - {SG.degree(sn_id)} connections\n"
        else:
            action_text += f"   • SN{sn_id} - {SG.degree(sn_id)} connections\n"
    action_text += f"   • Critical bottleneck in information flow\n\n"

    # Action 3: Specific pathway tracing
    action_text += f"3. Trace Information Flow:\n"
    action_text += f"   • Run Script 7 to extract minimal pathways\n"
    action_text += f"   • Identify which supernodes are essential\n"
    action_text += f"   • Compare with Script 8 for layer evolution"

    # Add interpretation guide
    interpret_text = "HOW TO INTERPRET:\n\n"
    interpret_text += "• Node size = number of features in supernode\n"
    interpret_text += "• Node color = supernode theme (see legend)\n"
    interpret_text += "• X-axis = mean layer position (L0-L25)\n"
    interpret_text += "• Edge width = connection strength\n"
    interpret_text += "• Green box = high confidence output\n"
    interpret_text += "• Yellow/Red box = low confidence (potential failure)\n\n"
    interpret_text += "WHY THIS MATTERS:\n\n"
    if has_comprehensive_analysis and 'metadata' in analysis_data:
        metadata = analysis_data['metadata']
        output_prob = metadata.get('output_probability', 0)
        if output_prob > 0.5:
            interpret_text += f"✓ Model is confident ({output_prob:.1%}) - circuit is working\n"
            interpret_text += "  Use this to understand how correct reasoning works\n"
        elif output_prob > 0.2:
            interpret_text += f"⚠ Model is uncertain ({output_prob:.1%}) - mixed signals\n"
            interpret_text += "  Circuit shows competing pathways\n"
        else:
            interpret_text += f"✗ Model is failing ({output_prob:.1%}) - circuit broken\n"
            interpret_text += "  Compare with working circuit to find failure mode\n"
    else:
        interpret_text += "Visualizes feature circuits through network layers\n"

    # Place action items and interpretation guide BELOW the plot area
    fig.text(0.3, -0.05, action_text, ha='center', fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'),
             family='monospace', transform=fig.transFigure)

    fig.text(0.7, -0.05, interpret_text, ha='center', fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='black'),
             family='monospace', transform=fig.transFigure)

    # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
    plt.subplots_adjust(bottom=0.15, right=0.85)  # Make room for legend on right and action items below
    # Create output directory using subfolder structure (organized by prompt slug)
    # Use PathManager for organized output
    pm = PathManager()
    prompt = metadata.get('prompt', '')
    model_id = metadata.get('model', metadata.get('modelId', ''))

    # If no prompt in metadata, try to extract from slug
    if not prompt:
        print(f"[WARNING] No prompt found in metadata, using slug: {prompt_slug}")
        prompt = prompt_slug.replace('-', ' ').title()

    output_dir = pm.visualizations_dir(prompt, model_id=model_id)
    print(f"\nPrompt: {prompt}")
    print(f"Model: {model_id}")
    print(f"Organizing visualizations in: {output_dir}")

    output_path = pm.visualization_path(prompt, 'supernode_overview', model_id=model_id)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path.name}")
    plt.close()

    print("\n" + "="*60)
    print("VISUALIZATION 2: Layer-by-Layer Supernode Distribution")
    print("="*60)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Count nodes per layer per supernode
    layer_supernode_counts = {}
    for sn_id, nodes in supernodes.items():
        sn_id_int = int(sn_id)
        for node in nodes:
            if node in G.nodes():
                layer = G.nodes[node]['layer']
                if layer not in layer_supernode_counts:
                    layer_supernode_counts[layer] = {}
                layer_supernode_counts[layer][sn_id_int] = layer_supernode_counts[layer].get(sn_id_int, 0) + 1

    # Prepare data for stacked bar chart
    layers = sorted(layer_supernode_counts.keys())
    # Use actual supernode IDs
    sn_ids = supernode_ids

    data = {sn_id: [layer_supernode_counts.get(layer, {}).get(sn_id, 0) for layer in layers] for sn_id in sn_ids}

    # Create stacked bar chart
    x = np.arange(len(layers))
    width = 0.8

    bottom = np.zeros(len(layers))
    for sn_id in sn_ids:
        ax.bar(x, data[sn_id], width, label=f'SN{sn_id}',
               bottom=bottom, color=supernode_colors.get(sn_id, '#CCCCCC'), alpha=0.8)
        bottom += data[sn_id]

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_title(f'Supernode Distribution Across Layers\n"{prompt_text}"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x[::2])  # Show every 2nd layer
    ax.set_xticklabels(layers[::2])
    # Place legend OUTSIDE plot area on the right
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='black', borderaxespad=0)
    ax.grid(axis='y', alpha=0.3)

    # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
    output_path = pm.visualization_path(prompt, 'layer_distribution', model_id=model_id)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path.name}")
    plt.close()

    print("\n" + "="*60)
    print("VISUALIZATION 3: Activation Heatmap by Layer")
    print("="*60)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Organize activations by layer and supernode
    layer_sn_activations = {}
    for sn_id, nodes in supernodes.items():
        sn_id_int = int(sn_id)
        for node in nodes:
            if node in G.nodes():
                layer = G.nodes[node]['layer']
                activation = G.nodes[node]['activation']

                if layer not in layer_sn_activations:
                    # Initialize with all supernode IDs
                    layer_sn_activations[layer] = {sn: [] for sn in supernode_ids}

                layer_sn_activations[layer][sn_id_int].append(activation)

    # Calculate mean activations
    layers = sorted(layer_sn_activations.keys())
    mean_activations = {sn_id: [] for sn_id in supernode_ids}

    for layer in layers:
        for sn_id in supernode_ids:
            acts = layer_sn_activations[layer][sn_id]
            mean_act = np.mean(acts) if acts else 0
            mean_activations[sn_id].append(mean_act)

    # Plot
    x = np.arange(len(layers))
    for sn_id in supernode_ids:
        ax.plot(x, mean_activations[sn_id], marker='o', linewidth=2,
                label=f'SN{sn_id}', color=supernode_colors.get(sn_id, '#CCCCCC'), markersize=6)

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Activation', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean Activation by Layer and Supernode\n"{prompt_text}"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(layers[::2])
    # Place legend OUTSIDE plot area on the right
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='black', borderaxespad=0)
    ax.grid(True, alpha=0.3)

    # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
    output_path = pm.visualization_path(prompt, 'activation_heatmap', model_id=model_id)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path.name}")
    plt.close()

    print("\n" + "="*60)
    print("VISUALIZATION 4: Top Features by Activation")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 20 nodes by activation
    nodes_by_activation = sorted(
        [(n, G.nodes[n]['activation'], G.nodes[n]['layer'], node_to_supernode.get(n, -1))
         for n in G.nodes()],
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # Extract data
    labels = [G.nodes[n]['label'] for n, _, _, _ in nodes_by_activation]
    activations = [act for _, act, _, _ in nodes_by_activation]
    colors = [supernode_colors.get(sn, '#CCCCCC') for _, _, _, sn in nodes_by_activation]

    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, activations, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Activation', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 20 Features by Activation\n"{prompt_text}"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add activation values as text
    for i, (_, act, layer, sn) in enumerate(nodes_by_activation):
        ax.text(act + 2, i, f'{act:.1f} (L{layer}, SN{sn})',
                va='center', fontsize=8)

    # Legend - dynamically generate based on actual supernodes
    legend_elements = []
    for sn_id in supernode_ids:
        color = supernode_colors.get(sn_id, '#CCCCCC')
        legend_elements.append(mpatches.Patch(color=color, label=f'SN{sn_id}'))

    if legend_elements:
        # Place legend OUTSIDE plot area on the right
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, framealpha=0.9, edgecolor='black', borderaxespad=0)

    # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
    output_path = pm.visualization_path(prompt, 'feature_importance', model_id=model_id)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path.name}")
    plt.close()

    print("\n" + "="*60)
    print("VISUALIZATION 5: Edge Weight Distribution")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get all edge weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    positive_weights = [w for w in weights if w > 0]
    negative_weights = [w for w in weights if w < 0]

    # Histogram of all weights
    ax1.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.set_xlabel('Edge Weight', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Edge Weight Distribution\n(All Connections)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart: positive vs negative
    sizes = [len(positive_weights), len(negative_weights)]
    labels = [f'Excitatory\n({len(positive_weights)} edges)',
              f'Inhibitory\n({len(negative_weights)} edges)']
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Excitatory vs Inhibitory Connections', fontsize=12, fontweight='bold')

    # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
    output_path = pm.visualization_path(prompt, 'information_flow', model_id=model_id)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path.name}")
    plt.close()

    # ============================================================================
    # LAYER-BASED VISUALIZATIONS (only if comprehensive analysis available)
    # ============================================================================

    if layer_groups and flow_analysis:
        print("\n" + "="*60)
        print("VISUALIZATION 6: Layer Group Flow Diagram")
        print("="*60)

        fig, ax = plt.subplots(figsize=(16, 8))

        # Define layer group order and colors
        group_order = ['input', 'early_proc', 'middle_proc', 'late_proc', 'output']
        group_colors = {
            'input': '#3498db',       # Blue
            'early_proc': '#2ecc71',  # Green
            'middle_proc': '#f39c12', # Orange
            'late_proc': '#e74c3c',   # Red
            'output': '#9b59b6'       # Purple
        }
        group_labels = {
            'input': 'Input\n(L0-5)',
            'early_proc': 'Early Proc\n(L6-10)',
            'middle_proc': 'Middle Proc\n(L11-15)',
            'late_proc': 'Late Proc\n(L16-20)',
            'output': 'Output\n(L21-25)'
        }

        # Filter to only groups that exist
        existing_groups = [g for g in group_order if g in layer_groups]

        # Create nodes for each layer group
        x_positions = np.linspace(0, 10, len(existing_groups))
        node_positions = {}

        for i, group_name in enumerate(existing_groups):
            group_data = layer_groups[group_name]
            x = x_positions[i]
            y = 5  # Center vertically

            # Node size based on number of nodes
            size = group_data['num_nodes'] * 50

            # Draw node
            circle = plt.Circle((x, y), radius=0.8,
                               color=group_colors.get(group_name, '#CCCCCC'),
                               alpha=0.7, zorder=2)
            ax.add_patch(circle)

            # Add label
            label = group_labels.get(group_name, group_name)
            ax.text(x, y+0.1, label, ha='center', va='center',
                   fontsize=11, fontweight='bold', zorder=3)

            # Add statistics
            stats_text = f"{group_data['num_nodes']} nodes\n{group_data['max_activation']:.1f} max act"
            ax.text(x, y-0.4, stats_text, ha='center', va='center',
                   fontsize=9, zorder=3)

            # Add top feature description if available (NEW!)
            if 'top_features_with_descriptions' in group_data and len(group_data['top_features_with_descriptions']) > 0:
                top_feat = group_data['top_features_with_descriptions'][0]
                description = top_feat.get('description', '')

                if description and description != 'None':
                    # Truncate long descriptions
                    if len(description) > 50:
                        description = description[:47] + '...'

                    # Extract layer and feature number from feature ID
                    feat_parts = top_feat['feature'].split('_')
                    if len(feat_parts) >= 2:
                        layer_num, feat_num = feat_parts[0], feat_parts[1]
                        desc_label = f"L{layer_num}_F{feat_num}:\n{description}"
                    else:
                        desc_label = description

                    # Add description below the stats
                    ax.text(x, y-1.1, desc_label, ha='center', va='top',
                           fontsize=7, style='italic', zorder=3,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7, edgecolor='gray'))

            node_positions[group_name] = (x, y)

        # Draw edges between consecutive groups
        for i in range(len(existing_groups) - 1):
            group1 = existing_groups[i]
            group2 = existing_groups[i+1]

            x1, y1 = node_positions[group1]
            x2, y2 = node_positions[group2]

            # Edge count is the outgoing edges from group1
            edge_count = layer_groups[group1]['outgoing_edges']

            # Draw arrow
            arrow = mpatches.FancyArrowPatch(
                (x1+0.8, y1), (x2-0.8, y2),
                arrowstyle='->', mutation_scale=30,
                linewidth=2 + (edge_count / 1000),  # Width based on edge count
                color='#34495e', alpha=0.6, zorder=1
            )
            ax.add_patch(arrow)

            # Add edge count label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.5
            ax.text(mid_x, mid_y, f"{edge_count} edges",
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(-1, 11)
        ax.set_ylim(2, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Information Flow Through Layer Groups\n"{prompt_text}"',
                    fontsize=16, fontweight='bold', pad=20)

        # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
        output_path = pm.visualization_path(prompt, 'thought_progression', model_id=model_id)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path.name}")
        plt.close()

        # ========================================================================
        print("\n" + "="*60)
        print("VISUALIZATION 7: Layer Group Comparison")
        print("="*60)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Prepare data
        groups = existing_groups
        group_names_short = [g.replace('_', ' ').title() for g in groups]

        # 1. Node count comparison
        node_counts = [layer_groups[g]['num_nodes'] for g in groups]
        colors_list = [group_colors.get(g, '#CCCCCC') for g in groups]

        ax1.bar(group_names_short, node_counts, color=colors_list, alpha=0.8)
        ax1.set_ylabel('Number of Nodes', fontsize=11, fontweight='bold')
        ax1.set_title('Node Distribution', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=15)

        # 2. Activation comparison
        mean_acts = [layer_groups[g]['mean_activation'] for g in groups]
        max_acts = [layer_groups[g]['max_activation'] for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        ax2.bar(x - width/2, mean_acts, width, label='Mean', alpha=0.8, color='#3498db')
        ax2.bar(x + width/2, max_acts, width, label='Max', alpha=0.8, color='#e74c3c')
        ax2.set_ylabel('Activation', fontsize=11, fontweight='bold')
        ax2.set_title('Activation Statistics', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(group_names_short, rotation=15)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Edge flow comparison
        incoming = [layer_groups[g]['incoming_edges'] for g in groups]
        internal = [layer_groups[g]['internal_edges'] for g in groups]
        outgoing = [layer_groups[g]['outgoing_edges'] for g in groups]

        x = np.arange(len(groups))
        width = 0.25

        ax3.bar(x - width, incoming, width, label='Incoming', alpha=0.8, color='#2ecc71')
        ax3.bar(x, internal, width, label='Internal', alpha=0.8, color='#f39c12')
        ax3.bar(x + width, outgoing, width, label='Outgoing', alpha=0.8, color='#e74c3c')
        ax3.set_ylabel('Edge Count', fontsize=11, fontweight='bold')
        ax3.set_title('Edge Flow Patterns', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(group_names_short, rotation=15)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Influence comparison
        mean_influences = [layer_groups[g]['mean_influence'] for g in groups]
        max_influences = [layer_groups[g]['max_influence'] for g in groups]

        ax4.bar(x - width/2, mean_influences, width, label='Mean', alpha=0.8, color='#9b59b6')
        ax4.bar(x + width/2, max_influences, width, label='Max', alpha=0.8, color='#e67e22')
        ax4.set_ylabel('Influence', fontsize=11, fontweight='bold')
        ax4.set_title('Influence Statistics', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(group_names_short, rotation=15)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Layer Group Analysis\n"{prompt_text}"',
                    fontsize=16, fontweight='bold', y=0.995)
        # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
        output_path = pm.visualization_path(prompt, 'supernode_connections', model_id=model_id)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path.name}")
        plt.close()

        # ========================================================================
        print("\n" + "="*60)
        print("VISUALIZATION 8: Steering Target Candidates")
        print("="*60)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Input nodes (high fan-out)
        input_nodes = flow_analysis['input_nodes'][:10]
        if input_nodes:
            labels = [n['label'] for n in input_nodes]
            values = [n['out_degree'] for n in input_nodes]
            colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(values)))

            y_pos = np.arange(len(labels))
            ax1.barh(y_pos, values, color=colors_gradient, alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.invert_yaxis()
            ax1.set_xlabel('Out-Degree', fontsize=10, fontweight='bold')
            ax1.set_title('Input Nodes\n(Early amplification targets)',
                         fontsize=11, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)

        # 2. Output nodes (high activation)
        output_nodes = flow_analysis['output_nodes'][:10]
        if output_nodes:
            labels = [n['label'] for n in output_nodes]
            values = [n['activation'] for n in output_nodes]
            colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.9, len(values)))

            y_pos = np.arange(len(labels))
            ax2.barh(y_pos, values, color=colors_gradient, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.invert_yaxis()
            ax2.set_xlabel('Activation', fontsize=10, fontweight='bold')
            ax2.set_title('Output Nodes\n(Late suppression targets)',
                         fontsize=11, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

        # 3. Bottleneck nodes (high betweenness)
        bottleneck_nodes = flow_analysis['bottleneck_nodes'][:10]
        if bottleneck_nodes:
            labels = [f"{n['label']} (L{n['layer']})" for n in bottleneck_nodes]
            values = [n['betweenness'] for n in bottleneck_nodes]
            colors_gradient = plt.cm.Purples(np.linspace(0.4, 0.9, len(values)))

            y_pos = np.arange(len(labels))
            ax3.barh(y_pos, values, color=colors_gradient, alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels, fontsize=8)
            ax3.invert_yaxis()
            ax3.set_xlabel('Betweenness Centrality', fontsize=10, fontweight='bold')
            ax3.set_title('Bottleneck Nodes\n(Critical pathway targets)',
                         fontsize=11, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)

        plt.suptitle(f'Steering Intervention Candidates\n"{prompt_text}"',
                    fontsize=16, fontweight='bold', y=0.98)

        # Add action items below the plots
        action_text = "INTERVENTION STRATEGY:\n\n"
        action_text += "1. Test Input Amplification:\n"
        if input_nodes:
            action_text += f"   • Amplify {input_nodes[0]['label']} (highest fan-out)\n"
            action_text += f"   • Expected: Strengthen early signal propagation\n\n"

        action_text += "2. Test Output Ablation:\n"
        if output_nodes:
            action_text += f"   • Ablate {output_nodes[0]['label']} (highest fan-in)\n"
            action_text += f"   • Expected: Block final output formation\n\n"

        action_text += "3. Test Bottleneck Manipulation:\n"
        if bottleneck_nodes:
            action_text += f"   • Modify {bottleneck_nodes[0]['label']} (highest betweenness)\n"
            action_text += f"   • Expected: Redirect information flow"

        fig.text(0.5, -0.05, action_text, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'),
                 family='monospace', transform=fig.transFigure)

        # plt.tight_layout()  # Skip to avoid matplotlib math parsing errors
        plt.subplots_adjust(bottom=0.15)  # Make room for action items
        output_path = pm.visualization_path(prompt, 'summary_dashboard', model_id=model_id)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path.name}")
        plt.close()

        print("\n" + "="*60)
        print("[SUCCESS] ALL VISUALIZATIONS COMPLETE!")
        print("="*60)
        print("\nGenerated files (Louvain-based):")
        print(f"  1. supernode_overview.png - Supernode-level circuit diagram")
        print(f"  2. layer_distribution.png - Nodes per layer per supernode")
        print(f"  3. activation_heatmap.png - Mean activation across layers")
        print(f"  4. feature_importance.png - Top 20 features by activation")
        print(f"  5. information_flow.png - Edge weight distribution")
        print(f"\nGenerated files (Layer-based):")
        print(f"  6. thought_progression.png - Information flow through layer groups")
        print(f"  7. supernode_connections.png - Layer group statistics comparison")
        print(f"  8. summary_dashboard.png - Steering intervention candidates")
        print(f"\nAll saved to: {output_dir}")
        print(f"\nNext step: Run cross-prompt analysis to compare multiple graphs")

    else:
        print("\n" + "="*60)
        print("[SUCCESS] LOUVAIN VISUALIZATIONS COMPLETE!")
        print("="*60)
        print("\nGenerated files (Louvain-based):")
        print(f"  1. supernode_overview.png - Supernode-level circuit diagram")
        print(f"  2. layer_distribution.png - Nodes per layer per supernode")
        print(f"  3. activation_heatmap.png - Mean activation across layers")
        print(f"  4. feature_importance.png - Top 20 features by activation")
        print(f"  5. information_flow.png - Edge weight distribution")
        print(f"\nAll saved to: {output_dir}")
        print(f"\n[INFO] Layer-based visualizations not available (legacy format)")
        print(f"Run circuit analysis again to enable 3 additional layer-based visualizations")
        print(f"\nNext step: Run cross-prompt analysis to compare multiple graphs")

    print("\n" + "=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
