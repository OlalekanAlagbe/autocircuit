---
name: neuronpedia-compare
description: This skill should be used when the user wants to compare two or more circuit graphs across different prompts. Use when the user says "compare circuits", "find universal features", or "what's similar between these prompts".
---

# Neuronpedia Compare

Compare attribution graphs across different prompts to find universal vs task-specific features.

## Instructions

### 1. Extract the graph names

Identify which graphs the user wants to compare (e.g., "Japan" and "France"). Locate the corresponding `*_converted.json` files.

### 2. Navigate to the pipeline directory

```bash
cd neuronpedia_pipeline
```

### 3. Run the comparison script

```bash
python scripts/5_compare_prompts.py \
  data/graphs/real_graph1_converted.json \
  data/graphs/real_graph2_converted.json \
  --name1 "Descriptive Name 1" \
  --name2 "Descriptive Name 2"
```

Takes 5-10 seconds for 2 graphs. Calculates feature overlap and complexity differences.

### 4. Report the comparison results

Include:
- **Feature overlap percentage** (e.g., "35.7% overlap")
- **Shared top features** -- especially note if any have identical activations
- **Complexity differences**: node counts, edge counts, percentage differences
- **Layer distribution patterns** -- similarities or differences
- **Inhibitory connection ratios** -- how consistent they are
- **Visualization saved** -- location of comparison PNG

### 5. Highlight key discoveries

- Universal features (appear in both with same/similar activation)
- Task-specific features (unique to each graph)
- Complexity insights (why one might be more complex)

## Output

- **Console**: Detailed comparison statistics
- **File created**: `data/visualizations/comparison_{name1}_vs_{name2}.png`
- **4-panel visualization** showing:
  1. Layer distribution comparison
  2. Activation distributions
  3. Edge weight distributions
  4. Summary statistics table

## Example Interaction

**User**: "Compare the Japan and France circuits"

1. Identify the two graphs
2. Navigate to neuronpedia_pipeline
3. Run comparison script
4. Report feature overlap, shared top features, complexity differences, and key insights

## Common Issues

- **"Graphs not found"**: Both graphs need to be converted first with `/neuronpedia-convert`
- **"No overlap"**: Very different task types may have no shared features (this is normal)
- **"Memory error"**: Compare 2-3 graphs at a time maximum

## When to Use

- User asks to "compare two graphs/circuits"
- User wants to find "universal features"
- After analyzing 2+ different prompts
- User asks "what's similar between these prompts?"
- Building a multi-prompt dataset

## Next Steps

After comparing:
1. Investigate universal features more deeply
2. Design steering experiments to test causal relationships
3. Generate more prompt comparisons to validate patterns
