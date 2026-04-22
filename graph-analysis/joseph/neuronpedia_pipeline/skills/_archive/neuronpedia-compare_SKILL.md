# Neuronpedia Compare

Compare attribution graphs across different prompts to find universal vs task-specific features.

## Instructions

When the user wants to compare two or more circuit graphs:

1. **Extract the graph names** from the user's request:
   - Identify which graphs they want to compare (e.g., "Japan" and "France")
   - Look for the corresponding `*_converted.json` files

2. **Navigate to the pipeline directory**:
   ```bash
   cd neuronpedia_pipeline
   ```

3. **Run the comparison script** with both graphs:
   ```bash
   python scripts/5_compare_prompts.py \
     data/graphs/real_graph1_converted.json \
     data/graphs/real_graph2_converted.json \
     --name1 "Descriptive Name 1" \
     --name2 "Descriptive Name 2"
   ```
   - Takes 5-10 seconds for 2 graphs
   - Calculates feature overlap and complexity differences

4. **Report the comparison results**:
   - **Feature overlap percentage** (e.g., "35.7% overlap")
   - **Shared top features** - especially note if any have identical activations
   - **Complexity differences**:
     - Node counts (which graph is larger)
     - Edge counts (which is more complex)
     - Percentage differences
   - **Layer distribution patterns** - similarities or differences
   - **Inhibitory connection ratios** - how consistent they are
   - **Visualization saved** - location of comparison PNG

5. **Highlight key discoveries**:
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

## Example interaction

**User**: "Compare the Japan and France circuits"

**You should**:
1. Identify the two graphs
2. Navigate to neuronpedia_pipeline
3. Run comparison script
4. Report: "✓ Comparison complete: Japan (yen) vs France (Paris)

   **Feature overlap**: 35.7% (281 shared features)

   **Shared top features**: 4 of top 5 are identical!
   - L15_F376262: 150.290 in BOTH (universal factual feature!)
   - L15_F294512: 105.398 in BOTH (identical)
   - L13_F33150139: 101.254 in BOTH (identical)
   - L24_F88478228: 124.506 vs 100.470 (similar role, different intensity)

   **Complexity difference**:
   - Japan: 858 nodes, 22,687 edges
   - France: 985 nodes, 34,025 edges (+50% more complex)

   **Inhibition consistency**: 42.7% vs 41.1% (very consistent suppression)

   **Key insight**: L15_F376262 appears to be a universal feature for factual retrieval - it activates identically (150.29) regardless of the specific fact being recalled!

   Visualization saved: data/visualizations/comparison_Japan_vs_France.png"

## Common issues

- **"Graphs not found"**: Both graphs need to be converted first with `/neuronpedia-convert`
- **"No overlap"**: Very different task types may have no shared features (this is normal)
- **"Memory error"**: Compare 2-3 graphs at a time maximum

## When to use this skill

- User asks to "compare two graphs/circuits"
- User wants to find "universal features"
- After analyzing 2+ different prompts
- User asks "what's similar between these prompts?"
- Building a multi-prompt dataset

## Next steps after this skill

After comparing:
1. Investigate universal features more deeply (e.g., test L15_F376262 on more prompts)
2. Design steering experiments to test causal relationships
3. Generate more prompt comparisons to validate patterns

## Key discovery

This skill revealed **L15_F376262 as a universal factual feature** with identical 150.29 activation in both Japan and France prompts - a major scientific finding!
