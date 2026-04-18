---
name: supernode-detect
description: Use when the user wants to detect supernodes, cluster a circuit graph, or generate an annotated attribution graph from a text prompt. Runs the end-to-end supernode detection pipeline (graph generation -> community detection -> semantic labeling -> influence trimming -> export).
---

# Supernode Detection

Run the automated supernode detection pipeline on a text prompt or existing raw graph. Produces an annotated JSON ready for upload to Neuronpedia, plus a markdown report and community visualization.

## Instructions

1. **Verify the config has an API key:**

```bash
grep "api_key" config/neuronpedia_config.yaml
```

2. **Single prompt (most common):**

```bash
python scripts/supernode_pipeline.py "The capital of France is"
```

3. **Specify a model:**

```bash
python scripts/supernode_pipeline.py "Water boils at 100 degrees" --model gemma-2-2b
```

Supported: `gemma-2-2b` (full semantic labels) and `qwen3-4b` (structural labels only — Neuronpedia feature API not yet available).

4. **Reuse an existing raw graph (skip API generation):**

```bash
python scripts/supernode_pipeline.py --raw-graph output/gemma-2-2b_prior-run/raw_graph.json
```

5. **Tune trimming aggressiveness:**

```bash
python scripts/supernode_pipeline.py "prompt" --target-pct 0.10   # Keep 10% of nodes (tighter)
python scripts/supernode_pipeline.py "prompt" --target-pct 0.20   # Keep 20% of nodes (looser)
```

Default is 15%. Lower values produce cleaner but smaller supernodes; higher values preserve more context.

## Output

```
output/{model}_{prompt-slug}/
  annotated_graph.json        <- ready for Neuronpedia upload
  supernode_report.md         <- human-readable analysis
  community_visualization.png <- colored community layout
  raw_graph.json              <- original unmodified graph
```

The markdown report lists each supernode with its label, node count, layer range, and member features. The annotated JSON embeds `qParams.supernodes`, `qParams.pinnedIds`, and `metadata.feature_details` for direct Neuronpedia upload.

## Example Interaction

**User:** Detect supernodes for "The chemical symbol for water is"

**Command:**

```bash
python scripts/supernode_pipeline.py "The chemical symbol for water is"
```

**Expected output:**

```
[1/6] Generating attribution graph...
  -> Graph downloaded: 1316 nodes, 67343 links
[2/6] Community detection...
  -> Louvain selected (modularity=0.4498, 14 communities)
[3/6] Bottleneck identification...
  -> Top 10 bottlenecks pinned
[4/6] Semantic labeling...
  -> Labeled 14/14 communities
[5/6] Influence trimming (target: 15%)...
  -> Kept 178 / 1205 nodes (14.8%)
[6/6] Exporting annotated graph...
  -> Wrote: output/gemma-2-2b_the-chemical-symbol-for-water-is/annotated_graph.json
```

## Common Issues

- **Rate limiting during semantic labeling:** Increase delay with `--rate-delay 1.0` (default 0.5s).
- **Oversized communities (>30% of graph):** Recursive splitting handles this automatically, but very dense graphs may produce many small sub-communities. Try `--target-pct 0.10` for a tighter result.
- **Missing semantic labels on QWEN:** Expected — the Neuronpedia feature API does not yet support QWEN SAEs. Structural labels (layer range + size) are produced as fallback.

## When to Use

- Turning a text prompt into a publishable circuit visualization
- Clustering an existing raw attribution graph into interpretable groups
- Generating supernode reports for a new prompt before deeper analysis

## Related Skills

- `/supernode-upload` — push the annotated graph to Neuronpedia programmatically
- `/neuronpedia-analyze` — run structural circuit analysis on the underlying graph
- `/neuronpedia-visualize` — generate additional PNG visualizations

## Prerequisites

- Python 3.8+ with `networkx`, `python-louvain`, `leidenalg`, `python-igraph`, `matplotlib`, `requests`, `pyyaml`, `numpy`
- Neuronpedia API key in `config/neuronpedia_config.yaml`
