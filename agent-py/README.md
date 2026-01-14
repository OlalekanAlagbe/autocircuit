# Neuronpedia Attribution Graph Cleanup Automation Agent

An automated tool for cleaning and organizing attribution graphs generated from prompts in Neuronpedia's Circuit Tracer.

## Overview

This agent automates the manual process of:
- Pinning important nodes from attribution graphs
- Grouping nodes into interpretable supernodes
- Generating human-readable labels for functional modules
- Computing quality metrics (replacement and completeness scores)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Analyze an existing graph

```bash
python main.py analyze --graph-file my_graph.json
```

### Cleanup an existing graph

```bash
export ANTHROPIC_API_KEY=your_api_key
python main.py cleanup-existing \
    --graph-file my_graph.json \
    --strategy pathway \
    --grouping functional \
    --output cleaned_graph.json
```

## Project Structure

```
graph-analysis/
├── main.py                      # CLI entry point
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
├── neuronpedia_agent/
│   ├── analysis/
│   │   ├── graph_analyzer.py       # Graph structure analysis
│   │   ├── node_selector.py        # Node selection strategies
│   │   └── grouping_engine.py      # Supernode creation
│   ├── labeling/
│   │   └── auto_labeler.py         # LLM-based label generation
│   └── optimization/
│       ├── path_tracer.py          # Computational pathway tracing
│       └── metrics.py              # Quality metrics
```

## Features

### Node Selection Strategies
- **Pathway**: Follow strongest computational paths from input to output
- **Importance**: Select globally most important nodes
- **Balanced**: Distribute selection across layers

### Grouping Strategies
- **Functional**: Group by computational role (input detectors, processors, output promoters)
- **Semantic**: Group by semantic similarity (requires embeddings)
- **Layer**: Group by layer proximity
- **Hybrid**: Combination of functional and semantic

### Quality Metrics
- **Replacement Score**: Fraction of end-to-end influence through pinned features (target: >0.5)
- **Completeness Score**: Fraction of incoming edge influence explained (target: >0.7)

## Configuration

Edit `config.yaml` to customize:
- Node selection thresholds
- Grouping parameters
- LLM settings for labeling
- Quality metric thresholds

## Requirements

- Python 3.8+
- Anthropic API key (for automated labeling)
- See `requirements.txt` for full dependencies

## Examples

See the specification document for detailed examples and usage patterns.

## License

[License information to be added]
