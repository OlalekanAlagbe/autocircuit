---
name: supernode-upload
description: This skill should be used when the user wants to upload an annotated graph to Neuronpedia. Use when the user says "upload to Neuronpedia", "validate the graph", or "upload the graph".
---

# Supernode Upload

Upload an annotated attribution graph to Neuronpedia's graph validator for visualization and sharing.

## Overview

This skill handles uploading the pipeline output (`annotated_graph.json`) to Neuronpedia:
- Validates the JSON structure before upload
- Guides upload via the Neuronpedia graph validator web interface
- Verifies the graph renders correctly

## Instructions

### 1. Locate the annotated graph

Find the output from a previous pipeline run:

```bash
ls supernode_detector/output/
```

The annotated graph is at: `output/{model}_{prompt_slug}/annotated_graph.json`

### 2. Validate the JSON structure locally

Before uploading, verify the JSON contains required fields:

```bash
python -c "
import json
with open('supernode_detector/output/{model}_{prompt_slug}/annotated_graph.json') as f:
    data = json.load(f)
print('Nodes:', len(data.get('nodes', [])))
print('Links:', len(data.get('links', [])))
print('Has supernodes:', 'supernodes' in data.get('qParams', {}))
print('Has pinnedIds:', 'pinnedIds' in data.get('qParams', {}))
print('Has metadata:', 'metadata' in data)
"
```

All fields should be present. `qParams.supernodes` contains the community groupings, `qParams.pinnedIds` contains bottleneck nodes.

### 3. Upload to Neuronpedia

1. Go to [neuronpedia.org/graph/validator](https://neuronpedia.org/graph/validator)
2. Upload `annotated_graph.json`
3. Verify the green "Valid Attribution Graph" checkmark appears
4. Click **Upload the graph** (skip the Feature Details section)
5. Copy the resulting URL

### 4. Report results to user

Include:
- Validation status (valid/invalid)
- Number of nodes and edges
- Number of supernodes/communities
- Number of pinned bottleneck nodes
- The Neuronpedia URL (if uploaded)

## JSON Structure

The annotated graph JSON must contain:

```json
{
  "nodes": [...],
  "links": [...],
  "qParams": {
    "supernodes": {
      "Community 0: Label": ["node_id_1", "node_id_2", ...],
      "Community 1: Label": [...]
    },
    "pinnedIds": ["bottleneck_node_1", "bottleneck_node_2"]
  },
  "metadata": {
    "slug": "model_prompt-slug",
    "feature_details": {...}
  }
}
```

## Common Issues

- **"Invalid Attribution Graph"**: Check that node IDs in supernodes match actual node IDs in the graph
- **"Missing fields"**: Ensure `qParams.supernodes` and `qParams.pinnedIds` exist
- **"Graph too large"**: Very large graphs (>2000 nodes) may be slow to render
- **Red validation**: Usually means node ID mismatch between supernodes and graph nodes

## When to Use

- After running `/supernode-pipeline`
- User asks to "upload to Neuronpedia" or "validate the graph"
- User wants to share a circuit visualization
- Testing that the pipeline output is Neuronpedia-compatible

## Next Steps

After uploading:
1. Share the Neuronpedia URL
2. Explore the interactive graph visualization
3. Run `/supernode-report` to generate a written analysis
