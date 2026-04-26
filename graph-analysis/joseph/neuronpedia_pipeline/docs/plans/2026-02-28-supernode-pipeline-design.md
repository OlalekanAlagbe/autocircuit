# Supernode Pipeline Design

**Date**: 2026-02-28
**Status**: Approved
**Goal**: Build an end-to-end pipeline that takes a text prompt, generates a circuit attribution graph, automatically detects and labels supernodes (communities), and outputs a Neuronpedia-compatible JSON file ready for upload and interactive exploration.

---

## 1. Motivation

Neuronpedia's circuit tracer generates attribution graphs with hundreds to thousands of nodes. Manually identifying functional groups ("supernodes") takes ~2 hours per circuit. The only existing automated tool (`attribution-graph-probing`) uses activation-pattern-based grouping with no graph-theoretic methods, limited to 5 circuits on 1 model.

Our pipeline uses **graph topology** (community detection on the circuit's adjacency structure) as the primary grouping mechanism — a fundamentally different and complementary approach. Combined with our cross-circuit bottleneck library, Neuronpedia semantic explanations, and causal steering validation data, this creates the most comprehensive automated supernode tool available.

### Differentiators from existing tools
- **Graph-topology-based grouping** (Louvain/Leiden/Greedy) vs activation-pattern-based
- **Multi-algorithm selection** — pick best by modularity, not hardcoded
- **Self-contained** — only requires a Neuronpedia API key, no local databases
- **Multi-model support** — GEMMA-2-2B and QWEN3-4B (structural labels for QWEN since feature API unavailable)
- **Optional cross-circuit enrichment** — if bottleneck library present, enrich labels with cross-circuit frequency and known roles
- **Dual labeling strategy** — compare top-3 summarization vs top-1 direct explanation, use whichever is cleaner

---

## 2. Pipeline Architecture

```
INPUT: "The capital of France is" --model gemma-2-2b
         |
         v
+---------------------+
|  Step 1: Generate   |  POST /api/graph/generate
|  Attribution Graph  |  -> raw_graph.json (from S3 URL)
+---------------------+
         |
         v
+---------------------+
|  Step 2: Community  |  Louvain + Leiden + Greedy Modularity
|  Detection          |  -> pick best by modularity score
|                     |  -> community assignments per node
+---------------------+
         |
         v
+---------------------+
|  Step 3: Bottleneck |  Betweenness centrality on graph
|  Identification     |  + optional library lookup
|                     |  -> flag bottleneck communities
+---------------------+
         |
         v
+---------------------+
|  Step 4: Semantic   |  GET /api/feature/{model}/{sae}/{id}
|  Labeling           |  -> NP explanations for top features
|                     |  -> generate community labels
+---------------------+
         |
         v
+---------------------+
|  Step 5: Format &   |  Inject into qParams.supernodes
|  Export             |  -> annotated_graph.json
|                     |  -> supernode_report.md
|                     |  -> community_visualization.png
+---------------------+
         |
         v
OUTPUT: annotated_graph.json (upload to neuronpedia.org/graph/validator)
        supernode_report.md
        community_visualization.png
```

### Single command
```bash
python supernode_pipeline.py "The capital of France is" --model gemma-2-2b
```

---

## 3. Step Details

### Step 1: Generate Attribution Graph

**API call:**
```json
POST https://www.neuronpedia.org/api/graph/generate
{
    "prompt": "The capital of France is",
    "modelId": "gemma-2-2b",
    "sourceSetName": "gemmascope-transcoder-16k",
    "maxNLogits": 10,
    "desiredLogitProb": 0.95,
    "nodeThreshold": 0.8,
    "edgeThreshold": 0.85,
    "maxFeatureNodes": 5000
}
```

**Response:** Returns a `slug` and S3 URL for the graph JSON.

**Download:** Fetch the raw graph JSON from S3 and save locally.

**Rate limiting:** Single API call, no rate concerns.

### Step 2: Community Detection

**Input:** Raw graph JSON (nodes + links arrays)

**Process:**
1. Build NetworkX DiGraph from nodes/links
2. Filter embedding layer nodes (layer="E") and error nodes (feature=-1) from community detection
3. Run 3 algorithms:
   - **Louvain** (resolution=2.0) via python-louvain
   - **Leiden** (resolution=1.0) via leidenalg/igraph
   - **Greedy Modularity** via networkx
4. Compute modularity score for each partition
5. Select partition with highest modularity
6. If two are within 5%, prefer fewer single-node communities

**Post-processing:**
- Minimum community size: 3 nodes (smaller groups merged into nearest neighbor community by edge weight)
- Maximum community size: 30% of total nodes (split large communities by running Louvain recursively at higher resolution)
- Disconnected components get independent community sets

**Fallback:** If graph has <20 nodes, use layer-based grouping (input/processing/output) instead.

### Step 3: Bottleneck Identification

**Primary method (always runs):**
1. Compute betweenness centrality for all nodes
2. Identify top-10 betweenness nodes as bottleneck candidates
3. Flag communities containing 2+ bottleneck candidates as "bottleneck communities"
4. Pin bottleneck nodes in `qParams.pinnedIds` for visibility

**Optional enrichment (if bottleneck library exists):**
1. Load `data/stage_1_5_bottleneck_library.json`
2. Match graph features to library entries via `np_id = circuit_tracer_id % 16384`
3. Annotate matching features with cross-circuit frequency, convergence, known roles
4. Append library metadata to community labels

### Step 4: Semantic Labeling

**For each community, run two strategies in parallel:**

**Strategy 1 — Top-3 Summarization:**
1. Rank community features by influence score (descending)
2. Take top 3 features
3. Query NP feature API: `GET /api/feature/{model}/{sae_path}/{np_id}`
4. Extract explanations
5. Find common theme across explanations (keyword overlap, semantic similarity)
6. Generate label: "theme (L{min}-L{max}, N nodes)"

**Strategy 2 — Top-1 Direct:**
1. Take the single highest-influence feature
2. Query NP feature API
3. Use its explanation directly, truncated to ~40 chars
4. Generate label: "explanation (L{min}-L{max})"

**Selection:** Output both in the report. For the JSON, use Strategy 1 if a clear theme exists (2+ explanations share keywords), otherwise fall back to Strategy 2.

**QWEN fallback:** No feature API available. Use structural labels:
- "L{min}-L{max} cluster ({N} nodes, betweenness={score})"

**Rate limiting:** ~2 calls/sec. Typical graph: 5-10 communities x 3 features = 15-30 calls. ~15-30 seconds.

**SAE path mapping:**
- GEMMA: `{layer}-gemmascope-transcoder-16k`
- QWEN: N/A (structural labels only)

### Step 5: Format & Export

**Primary output — annotated_graph.json:**
- Start with the raw graph JSON exactly as downloaded from S3
- Inject `qParams.supernodes` array: `[["label", "node_id_1", "node_id_2", ...], ...]`
- Set `qParams.pinnedIds` to bottleneck node IDs
- Do NOT modify nodes or links arrays — keep them exactly as Neuronpedia generated them

**Secondary output — supernode_report.md:**
```markdown
# Supernode Analysis Report
## Prompt: "The capital of France is"
## Model: gemma-2-2b
## Algorithm: Louvain (modularity=0.48, best of 3)

### Communities Found: 7
| # | Label (Strategy 1) | Label (Strategy 2) | Nodes | Layers | Bottleneck? |
|---|-------|-------|-------|--------|-------------|
| 1 | capital cities / place names | place names and surrounding words | 15 | L4-L7 | Yes |
| 2 | formatting infrastructure | HTML formatting tags | 23 | L0-L2 | No |
...

### Algorithm Comparison
| Algorithm | Modularity | Communities | Single-node |
|-----------|-----------|-------------|-------------|
| Louvain   | 0.48      | 7           | 0           |
| Leiden    | 0.45      | 9           | 2           |
| Greedy   | 0.43      | 6           | 0           |

### Bottleneck Nodes (pinned)
- L5_F7993 (betweenness=0.023, community 1)
- L6_F4662 (betweenness=0.019, community 1)
```

**Tertiary output — community_visualization.png:**
- NetworkX spring layout colored by community assignment
- Node size proportional to influence
- Bottleneck nodes highlighted with star markers
- Legend with community labels
- Cap at 200 highest-influence nodes for readability

**File structure:**
```
output/
  {model}_{prompt_slug}/
    annotated_graph.json
    supernode_report.md
    community_visualization.png
    raw_graph.json              (original, unmodified)
```

---

## 4. Configuration

```yaml
# supernode_config.yaml (or CLI args)
model: gemma-2-2b
source_set: gemmascope-transcoder-16k

# Community detection
algorithms: [louvain, leiden, greedy]
louvain_resolution: 2.0
leiden_resolution: 1.0
min_community_size: 3
max_community_pct: 0.30

# Labeling
label_strategy: auto  # auto, top3, top1
max_label_length: 50

# Graph generation
node_threshold: 0.8
edge_threshold: 0.85
max_feature_nodes: 5000
max_n_logits: 10
desired_logit_prob: 0.95

# API
rate_delay: 0.5  # seconds between feature API calls
api_key_path: config/neuronpedia_config.yaml
```

---

## 5. Dependencies

**Required (Python packages):**
- `networkx` — graph operations, betweenness centrality, greedy modularity
- `python-louvain` (community) — Louvain community detection
- `leidenalg` + `igraph` — Leiden community detection
- `matplotlib` — community visualization
- `requests` — Neuronpedia API calls
- `pyyaml` — config loading

**All already installed** in our current pipeline environment.

**External dependency:**
- Neuronpedia API key (in `config/neuronpedia_config.yaml`)

**No local database required** — the pipeline is fully self-contained. Optional bottleneck library enrichment if `data/stage_1_5_bottleneck_library.json` exists.

---

## 6. Error Handling

| Scenario | Handling |
|----------|----------|
| API rate limit (429) | Wait 60s + retry with exponential backoff |
| API timeout | Retry up to 3x with 5s delay |
| Graph generation fails | Error message with API response, exit |
| Feature API unavailable (QWEN) | Fall back to structural labels |
| No explanations for features | Fall back to structural labels |
| Graph <20 nodes | Skip community detection, use layer grouping |
| Disconnected graph | Independent community sets per component |
| Very large graph (>2000 nodes) | Run community detection on full graph but cap visualization at 200 nodes |
| Checkpoint/resume | Save raw_graph.json after Step 1; if it exists, skip generation on re-run |

---

## 7. Future Enhancements (Phase 2)

- **Upload to Neuronpedia**: Add `--upload` flag using signed-put + save-to-db API
- **Probe validation**: Add probe prompting step (inspired by attribution-graph-probing) to validate community coherence
- **Batch mode**: Process multiple prompts in sequence
- **Interactive mode**: Preview communities and let user rename/merge before export
- **Cross-circuit comparison**: Generate supernodes for N prompts and highlight shared communities
- **Causal enrichment**: If steering results exist, annotate features with known causal effects
