# Supernode Detector

Automated supernode detection for [Neuronpedia](https://neuronpedia.org) circuit attribution graphs. Generates an attribution graph from a text prompt, detects functional communities via multi-algorithm graph clustering, applies influence-based trimming, labels supernodes with Neuronpedia feature explanations, and exports an annotated JSON ready for upload.

## Quick Start

```bash
pip install -r requirements.txt
# Add your Neuronpedia API key to config/neuronpedia_config.yaml

python scripts/supernode_pipeline.py "The Titanic sank in the year"
```

Output lands in `output/{model}_{prompt_slug}/`. Upload `annotated_graph.json` to [neuronpedia.org/graph/validator](https://neuronpedia.org/graph/validator).

## Pipeline

1. **Graph Generation** &mdash; Calls the Neuronpedia circuit tracer API to produce an attribution graph, then downloads the raw JSON.

2. **Community Detection** &mdash; Runs Louvain, Leiden, and Greedy Modularity on the graph's adjacency structure. Selects the best partition by modularity with a penalty for oversized communities. Recursively splits any community exceeding 30% of graph size.

3. **Influence Trimming** &mdash; Keeps only the highest-influence nodes per community, targeting a configurable percentage of total nodes (default 15%). Bottleneck nodes are always retained.

4. **Bottleneck Identification** &mdash; Computes betweenness centrality to find critical routing nodes. Top bottlenecks are pinned in the Neuronpedia viewer.

5. **Semantic Labeling** &mdash; Queries the Neuronpedia feature API for community members and generates a descriptive label per supernode. Falls back to structural labels when explanations are unavailable.

6. **Export** &mdash; Writes `annotated_graph.json` with `qParams.supernodes`, `qParams.pinnedIds`, and `metadata.feature_details` for direct Neuronpedia upload.

## CLI Options

```
python scripts/supernode_pipeline.py [prompt] [options]

Arguments:
  prompt                  Text prompt to analyze

Options:
  --model MODEL           Model ID (default: gemma-2-2b)
  --raw-graph PATH        Use an existing raw graph JSON (skips API generation)
  --skip-generation       Skip generation if a cached raw_graph.json exists
  --rate-delay SECONDS    Delay between Neuronpedia API calls (default: 0.5)
  --target-pct FLOAT      Fraction of nodes to retain after trimming (default: 0.15)
  --upload                Upload the annotated graph to Neuronpedia after pipeline completes
  --upload-only PATH      Skip the pipeline and upload an existing annotated_graph.json
  --batch-file PATH       Process multiple prompts from a file (one per line, # comments)
```

## Output

```
output/{model}_{prompt_slug}/
    annotated_graph.json        # Upload to Neuronpedia (cantor-paired features)
    supernode_report.md         # Human-readable analysis
    community_visualization.png # Colored community layout
    raw_graph.json              # Original unmodified graph
    upload_result.json          # Created after --upload (contains viewable URL)
```

## Uploading to Neuronpedia

**Recommended: use the `--upload` flag** (programmatic API, bypasses the web UI):

```bash
python scripts/supernode_pipeline.py "The Titanic sank in the year" --upload
```

This runs the full pipeline, then uploads via the Neuronpedia API and prints the viewable URL. Features are automatically cantor-paired for Neuronpedia compatibility and slugs include a timestamp to avoid collisions.

**Manual upload** (if you prefer the web UI):

1. Go to [neuronpedia.org/graph/validator](https://neuronpedia.org/graph/validator)
2. Paste the contents of `annotated_graph.json` into the text area
3. Click **Validate JSON** — verify the green checkmark
4. Skip the Feature Details section (already configured via `neuronpedia_source_set`)
5. Use the upload button if available, or use `--upload-only` to upload via the API:

```bash
python scripts/supernode_pipeline.py --upload-only output/gemma-2-2b_my-prompt/annotated_graph.json
```

**Note:** The web UI upload button may be disabled for some models. The `--upload` flag uses the API directly and works regardless.

## Batch Mode

Process multiple prompts in one command:

```bash
# Inline
python scripts/supernode_pipeline.py "prompt 1" "prompt 2" "prompt 3" --upload

# From file
python scripts/supernode_pipeline.py --batch-file prompts.txt --upload
```

Prompt file format: one prompt per line, `#` for comments, blank lines skipped. Failed prompts don't abort the batch. Results saved to `output/batch_results.json`.

## Supported Models

| Model | Generation | Semantic Labels | Notes |
|-------|-----------|-----------------|-------|
| gemma-2-2b | Yes | Yes | Full support via gemmascope-transcoder-16k SAE |
| qwen3-4b | Yes | Structural only | Feature API not yet available on Neuronpedia |

## Configuration

Copy the example config and add your API key:

```yaml
# config/neuronpedia_config.yaml
api:
  api_key: "your-neuronpedia-api-key"
```

## Dependencies

Python 3.8+ with: `networkx`, `python-louvain`, `leidenalg`, `python-igraph`, `matplotlib`, `requests`, `pyyaml`, `numpy`. See `requirements.txt`.

## Related Work

This tool is part of the [autocircuit](https://github.com/jbl10/autocircuit) research project on mechanistic interpretability of language model circuits. The companion `neuronpedia_pipeline/` contains the full research pipeline for cross-circuit bottleneck analysis and steering validation.

## License

MIT
