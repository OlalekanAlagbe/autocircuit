---
name: neuronpedia-visualize
description: This skill should be used when the user wants to visualize a circuit. Use when the user says "visualize the circuit", "see the graph", or "generate figures". Creates 5 publication-quality PNG visualizations.
---

# Neuronpedia Visualize

Generate 5 types of publication-quality PNG visualizations of circuit structure.

## Instructions

### 1. Navigate to the pipeline directory

```bash
cd neuronpedia_pipeline
```

### 2. Verify analysis has been run

Check that supernode data exists from `/neuronpedia-analyze`. If not, inform user to run analysis first.

### 3. Run the visualization script

```bash
python scripts/4_visualize.py
```

Takes 10-20 seconds. Generates 5 different visualization types, all at 300 DPI (publication quality).

### 4. Report what was created

List all 5 PNG files with their purposes:
1. **Supernode overview** -- Circuit diagram showing communities
2. **Layer distribution** -- Bar chart of nodes per layer
3. **Activation heatmap** -- Layer x feature intensity map
4. **Top features** -- Top 20 features ranked by activation
5. **Edge weights** -- Excitatory vs inhibitory distribution

Include total file size, location (`data/visualizations/`), and key insights visible in the visualizations.

## Output

**5 PNG files created** (~5-8 MB total):

1. `real_supernodes_overview.png` -- Circuit diagram with color-coded supernodes
2. `real_layer_distribution.png` -- Bar chart showing which layers are most active
3. `real_activation_heatmap.png` -- Heatmap revealing layer specialization
4. `real_top_features.png` -- Ranked bar chart of most important features
5. `real_edge_weights.png` -- Histogram of positive/negative connections

All images are 300 DPI, suitable for publications and presentations.

## Example Interaction

**User**: "Visualize the circuit"

1. Navigate to neuronpedia_pipeline
2. Run visualization script
3. Report:

```
Generated 5 visualizations in data/visualizations/:

1. Supernode overview - Shows 3 communities with clear connections
2. Layer distribution - Layer 0 has most nodes (212), Layer 15 highest activations
3. Activation heatmap - Reveals Layer 15 specialization (red/hot zone)
4. Top features - L15_F376262 dominates with 150.29 activation
5. Edge weights - 57% excitatory (positive), 43% inhibitory (negative)

Total size: 5.4 MB (all 300 DPI, publication quality)
Key insight: Layer 15 appears to be the factual knowledge hub.
```

## Common Issues

- **"No supernode data found"**: Run `/neuronpedia-analyze` first
- **"Matplotlib backend error"**: Set environment variable `MPLBACKEND=Agg`
- **"Memory error"**: Normal for very large graphs (1000+ nodes), visualizations will still complete

## When to Use

- User asks to "visualize the circuit"
- User wants to "see the graph"
- After running `/neuronpedia-analyze`
- User needs figures for presentations or papers
- User wants to understand activation patterns visually

## Next Steps

After visualizing:
1. Review the PNG files to understand circuit structure
2. Run `/neuronpedia-compare` to compare with other prompts
3. Use visualizations in presentations or research papers
