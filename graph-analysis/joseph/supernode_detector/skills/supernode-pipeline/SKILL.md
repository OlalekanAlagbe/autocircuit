---
name: supernode-pipeline
description: This skill should be used when the user wants to run the full supernode detection pipeline on a prompt. Use when the user says "run the pipeline", "detect supernodes", "analyze a prompt", or wants end-to-end supernode detection.
---

# Supernode Pipeline

Run the full supernode detection pipeline: generate an attribution graph, detect communities, label supernodes, and export an annotated JSON for Neuronpedia upload.

## Overview

This skill runs `supernode_pipeline.py` which:
- Generates an attribution graph via the Neuronpedia Circuit Tracer API
- Detects communities using multi-algorithm clustering (Louvain, Leiden, Greedy Modularity)
- Applies influence-based trimming (keeps top 15% of nodes by default)
- Identifies bottleneck nodes via betweenness centrality
- Labels supernodes with Neuronpedia feature explanations
- Exports `annotated_graph.json` ready for Neuronpedia upload

## Instructions

### 1. Run the pipeline

```bash
cd supernode_detector
python scripts/supernode_pipeline.py "YOUR PROMPT HERE"
```

Options:
- `--model MODEL` -- Model ID (default: `gemma-2-2b`)
- `--raw-graph PATH` -- Use an existing raw graph JSON (skips API generation)
- `--skip-generation` -- Skip generation if a cached `raw_graph.json` exists
- `--rate-delay SECONDS` -- Delay between Neuronpedia API calls (default: 0.5)
- `--target-pct FLOAT` -- Fraction of nodes to retain after trimming (default: 0.15)

### 2. Wait for completion (typically 30-90 seconds)

The pipeline:
- Calls Neuronpedia API to generate the attribution graph (~10-20s)
- Downloads raw graph from S3
- Runs community detection algorithms and selects best partition
- Recursively splits oversized communities (>30% of graph)
- Trims low-influence nodes
- Fetches feature explanations from Neuronpedia (rate-limited)
- Exports annotated JSON

### 3. Report results to user

Include:
- Prompt used and model
- Number of communities detected
- Number of nodes retained after trimming
- Bottleneck nodes identified
- Output location
- Next step: upload `annotated_graph.json` to Neuronpedia validator

## Output Structure

Files saved to: `supernode_detector/output/{model}_{prompt_slug}/`

- **annotated_graph.json** -- Upload to neuronpedia.org/graph/validator
- **supernode_report.md** -- Human-readable analysis
- **community_visualization.png** -- Colored community layout
- **raw_graph.json** -- Original unmodified graph

## Example Interaction

**User**: "Run the pipeline for 'The Titanic sank in the year'"

```bash
cd supernode_detector
python scripts/supernode_pipeline.py "The Titanic sank in the year"
```

Report:
```
Supernode pipeline complete!

Prompt: <bos>The Titanic sank in the year
Model: gemma-2-2b
Communities: 7 detected (Louvain selected, modularity 0.42)
Nodes retained: 148 of 980 (15.1%)
Bottlenecks: 5 pinned nodes
Output: output/gemma-2-2b_bos-the-titanic-sank-in-the-year/

Next step: Upload annotated_graph.json to neuronpedia.org/graph/validator
```

## Important Notes

- **Feature ID mapping**: `neuronpedia_id = circuit_tracer_id % 16384`
- **GEMMA SAE path**: `{layer}-gemmascope-transcoder-16k`
- **QWEN**: Graph generation works but feature API labels are not yet available (structural labels only)
- **Windows encoding**: Script handles UTF-8 encoding automatically
- **API key**: Must be set in `config/neuronpedia_config.yaml`

## Supported Models

| Model | Generation | Semantic Labels | Notes |
|-------|-----------|-----------------|-------|
| gemma-2-2b | Yes | Yes | Full support via gemmascope-transcoder-16k SAE |
| qwen3-4b | Yes | Structural only | Feature API not yet available on Neuronpedia |

## Common Issues

- **"Connection timeout"**: Neuronpedia API may be busy, retry
- **"API rate limit"**: Feature labeling slows down, increase `--rate-delay`
- **"No communities detected"**: Graph may be too small or disconnected
- **"python-louvain not installed"**: Run `pip install python-louvain`

## When to Use

- User asks to "run the pipeline" or "detect supernodes"
- User wants to analyze a specific prompt
- Starting a new supernode analysis
- User wants to generate a Neuronpedia-uploadable graph

## Next Steps

After running the pipeline:
1. **Upload** (`/supernode-upload`) -- Upload annotated_graph.json to Neuronpedia
2. **Report** (`/supernode-report`) -- Generate a detailed analysis report
