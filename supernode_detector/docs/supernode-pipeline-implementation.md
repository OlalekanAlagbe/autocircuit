# Supernode Pipeline Implementation Reference

**Goal:** End-to-end CLI tool that takes a text prompt, generates a Neuronpedia attribution graph, automatically detects supernodes via graph community detection, labels them using NP feature explanations, and outputs a ready-to-upload annotated JSON file.

**Architecture:** Single Python script (`scripts/supernode_pipeline.py`) with modular functions for each pipeline step. Operates on raw Neuronpedia graph format (4 top-level keys: `metadata`, `qParams`, `nodes`, `links`). Injects community assignments into `qParams.supernodes` as `[["label", "node_id_1", ...], ...]` arrays. No modifications to nodes or links.

**Tech Stack:** Python 3, networkx, python-louvain (community), leidenalg/igraph, matplotlib, requests, pyyaml

---

## Module Structure

The pipeline is organized into 6 modules within a single file:

### 1. Utilities
- `load_api_key()` - Reads API key from `config/neuronpedia_config.yaml`
- `slugify(text)` - Converts prompt text to filesystem-safe slug

### 2. Graph Generation (`generate_graph`)
- POST to Neuronpedia `/api/graph/generate`
- Downloads raw graph JSON from returned S3 URL
- Supports model-specific source sets (GEMMA uses `gemmascope-transcoder-16k`)

### 3. Community Detection (`detect_communities`)
- `build_graph()` - Converts raw JSON to NetworkX DiGraph, filtering embedding/error nodes
- Runs Louvain (resolution=2.0), Leiden, and Greedy Modularity
- Selects best by modularity score with singleton tiebreaker
- Post-processing merges communities with <3 nodes into nearest neighbor
- Fallback: layer-based grouping for graphs with <20 nodes

### 4. Bottleneck Identification (`identify_bottlenecks`)
- Betweenness centrality (sampled for graphs >500 nodes)
- Top-10 nodes pinned as bottleneck candidates
- Communities with 2+ bottleneck nodes flagged
- Optional cross-reference with bottleneck library (`data/stage_1_5_bottleneck_library.json`)

### 5. Semantic Labeling (`label_communities`)
- Strategy 1: Top-3 feature explanation summarization (keyword overlap)
- Strategy 2: Top-1 direct explanation (truncated)
- Selects Strategy 1 if clear theme (2+ shared keywords), else Strategy 2, else structural fallback
- QWEN: structural labels only (feature API unavailable)
- Rate: ~2 calls/sec, handles 429 with 60s wait

### 6. Export & Visualization
- `export_annotated_graph()` - Injects `qParams.supernodes` and `qParams.pinnedIds`
- `generate_report()` - Markdown report with community table, algorithm comparison, bottleneck list
- `generate_visualization()` - Spring layout PNG, colored by community, star markers for bottlenecks

---

## Running the Pipeline

### Basic usage
```bash
python scripts/supernode_pipeline.py "The capital of France is"
```

### With existing graph (no API generation)
```bash
python scripts/supernode_pipeline.py --raw-graph path/to/raw_graph.json
```

### With cached graph (skip if already generated)
```bash
python scripts/supernode_pipeline.py "The capital of France is" --skip-generation
```

### With custom model
```bash
python scripts/supernode_pipeline.py "The capital of France is" --model gemma-2-2b
```

---

## Key Implementation Notes

### Node ID format
Raw graph nodes use `{layer}_{feature}_{ctx_idx}` format (e.g., `5_7993_3`).

### Influence field can be null
Raw graph nodes may have `"influence": null`. All code uses `(value or 0)` pattern to handle this safely.

### Feature ID mapping
In raw Neuronpedia graphs, the `feature` field is the direct NP feature ID. No modular arithmetic needed (unlike circuit tracer IDs which use `circuit_tracer_id % 16384`).

### SAE paths
- GEMMA: `{layer}-gemmascope-transcoder-16k`
- QWEN: Not available (structural labels only)

### Windows compatibility
Script includes UTF-8 encoding wrapper for Windows stdout/stderr.

---

## Validation

1. **Offline test:** Run with `--raw-graph` pointing to an existing graph JSON
2. **Verify output:** Check `annotated_graph.json` has `qParams.supernodes` populated
3. **Neuronpedia validator:** Upload to https://neuronpedia.org/graph/validator
4. **Report check:** Verify `supernode_report.md` has community table and algorithm comparison
5. **Visualization:** Verify `community_visualization.png` shows colored clusters with bottleneck stars
