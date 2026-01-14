"""CLI interface for Neuronpedia Attribution Graph Cleanup Automation Agent"""

import click
from pathlib import Path
import json
from neuronpedia_agent.analysis.graph_analyzer import GraphAnalyzer
from neuronpedia_agent.analysis.node_selector import NodeSelector
from neuronpedia_agent.analysis.grouping_engine import GroupingEngine
from neuronpedia_agent.labeling.auto_labeler import AutoLabeler
from neuronpedia_agent.optimization.metrics import MetricsCalculator


@click.group()
def cli():
    """Neuronpedia Attribution Graph Cleanup Automation Agent"""
    pass


@cli.command()
@click.option('--prompt', required=True, help='Input prompt for graph generation')
@click.option('--model', default='gemma-2-2b', help='Model to use')
@click.option('--output', default='cleaned_graph.json', help='Output file path')
@click.option('--strategy', default='pathway', type=click.Choice(['pathway', 'importance', 'balanced']))
@click.option('--max-nodes', default=30, type=int, help='Maximum nodes to pin')
@click.option('--grouping', default='functional', type=click.Choice(['functional', 'semantic', 'layer', 'hybrid']))
def cleanup(prompt, model, output, strategy, max_nodes, grouping):
    """
    Generate and cleanup an attribution graph for a given prompt

    Example:
    python main.py cleanup --prompt "The capital of Texas is" --strategy pathway --grouping functional
    """
    click.echo(f"Generating graph for prompt: {prompt}")
    click.echo(f"Using model: {model}")

    # TODO: Implement full graph generation and cleanup pipeline
    click.echo("Note: Full implementation requires API integration")
    click.echo(f"Would use strategy: {strategy}, max_nodes: {max_nodes}, grouping: {grouping}")
    click.echo(f"Output would be saved to: {output}")


@cli.command()
@click.option('--graph-file', required=True, type=click.Path(exists=True), help='Path to graph JSON')
@click.option('--output', default='cleaned_graph.json', help='Output file path')
@click.option('--strategy', default='pathway')
@click.option('--max-nodes', default=30, type=int)
@click.option('--grouping', default='functional')
@click.option('--api-key', envvar='ANTHROPIC_API_KEY', help='Anthropic API key for labeling')
def cleanup_existing(graph_file, output, strategy, max_nodes, grouping, api_key):
    """
    Cleanup an existing graph JSON file

    Example:
    python main.py cleanup-existing --graph-file my_graph.json --strategy balanced
    """
    click.echo(f"Loading graph from: {graph_file}")

    try:
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)

        # Analyze graph
        analyzer = GraphAnalyzer(graph_data)
        click.echo(f"Graph loaded: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")

        # Select nodes
        selector = NodeSelector(analyzer, max_nodes=max_nodes)
        pinned_nodes = selector.select_nodes_for_pinning(strategy=strategy)
        click.echo(f"Selected {len(pinned_nodes)} nodes using {strategy} strategy")

        # Group into supernodes
        grouper = GroupingEngine(analyzer, pinned_nodes)
        supernodes = grouper.create_supernodes(strategy=grouping)
        click.echo(f"Created {len(supernodes)} supernodes using {grouping} grouping")

        # Generate labels if API key provided
        if api_key:
            labeler = AutoLabeler(api_key=api_key)
            for snode in supernodes:
                label = labeler.generate_label(snode, graph_data, prompt="", target_logit="")
                snode.label = label
                click.echo(f"  - {label} ({len(snode.node_ids)} nodes, layers {snode.layer_range[0]}-{snode.layer_range[1]})")

        # Save output
        output_data = {
            'pinned_node_ids': pinned_nodes,
            'supernodes': [
                {
                    'label': snode.label,
                    'node_ids': snode.node_ids,
                    'layer_range': snode.layer_range,
                    'functional_role': snode.functional_role,
                    'total_influence': snode.total_influence
                }
                for snode in supernodes
            ],
            'original_graph': graph_data
        }

        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)

        click.echo(f"✓ Saved cleaned graph to: {output}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise


@cli.command()
@click.option('--prompt', required=True)
@click.option('--model', default='gemma-2-2b')
def generate_only(prompt, model):
    """
    Generate a graph without cleanup (for inspection)

    Example:
    python main.py generate-only --prompt "The capital of Texas is"
    """
    click.echo(f"Generating graph for: {prompt}")
    click.echo(f"Using model: {model}")
    click.echo("Note: Requires API integration")


@cli.command()
@click.option('--graph-file', required=True, type=click.Path(exists=True))
def analyze(graph_file):
    """
    Analyze a graph and print statistics without cleanup

    Shows:
    - Node count, edge count
    - Layer distribution
    - Top influential nodes
    - Suggested pinning strategy

    Example:
    python main.py analyze --graph-file my_graph.json
    """
    click.echo(f"Analyzing graph from: {graph_file}")

    try:
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)

        analyzer = GraphAnalyzer(graph_data)

        # Basic statistics
        num_nodes = len(graph_data.get('nodes', []))
        num_edges = len(graph_data.get('edges', []))
        click.echo(f"\nGraph Statistics:")
        click.echo(f"  Nodes: {num_nodes}")
        click.echo(f"  Edges: {num_edges}")

        # Layer distribution
        layers = [node.get('layer', 0) for node in graph_data.get('nodes', [])]
        if layers:
            click.echo(f"  Layers: {min(layers)}-{max(layers)}")

        # Top influential nodes
        importance = analyzer.compute_node_importance()
        top_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        click.echo(f"\nTop 10 Most Important Nodes:")
        for i, (node_id, score) in enumerate(top_nodes, 1):
            node = analyzer.get_node(node_id)
            explanation = node.get('explanation', 'No explanation') if node else 'Unknown'
            click.echo(f"  {i}. {node_id} (influence: {score:.3f}) - {explanation[:50]}")

        click.echo(f"\nSuggested Strategy:")
        click.echo(f"  - Consider 'pathway' strategy for clear computational paths")
        click.echo(f"  - Expect ~5 supernodes from functional grouping")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise


if __name__ == '__main__':
    cli()
