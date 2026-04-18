# Neuronpedia Convert

Convert raw Circuit Tracer graphs to pipeline-compatible NetworkX format.

## Instructions

When the user wants to convert a fetched graph:

1. **Navigate to the pipeline directory**:
   ```bash
   cd neuronpedia_pipeline
   ```

2. **Identify the graph to convert**:
   - Auto-detects the latest `real_*.json` file in `data/graphs/`
   - Or use a specific file if the user mentions one

3. **Run the conversion script**:
   ```bash
   python scripts/2_convert_graph.py
   ```
   - The script will automatically find and process the latest raw graph
   - Takes less than 1 second typically

4. **Report the conversion results**:
   - Original node count vs filtered node count
   - Original edge count vs filtered edge count
   - Number of nodes removed (typically 10-15%)
   - Output file location
   - File size before and after

## Output

- **File created**: `data/graphs/real_{name}_converted.json`
- **File size**: 2-4 MB (typically 40% smaller than raw)
- **Changes made**:
  - Nodes without activation data removed
  - Format converted to NetworkX compatibility
  - Negative edge weights preserved (inhibitory connections)
  - Metadata retained

## Example interaction

**User**: "Convert the graph"

**You should**:
1. Navigate to neuronpedia_pipeline
2. Run conversion script
3. Report: "✓ Conversion complete:
   - Input: real_japan_currency.json (962 nodes, 35,561 edges)
   - Output: real_japan_currency_converted.json (858 nodes, 22,687 edges)
   - Removed: 104 invalid nodes (10.8%)
   - Size: 4.3 MB → 2.7 MB

   Next step: Run `/neuronpedia-analyze` to detect supernodes"

## Common issues

- **"File not found"**: User needs to run `/neuronpedia-fetch` first
- **"No valid nodes after filtering"**: Raw graph may be corrupted, re-fetch
- **"Too many nodes removed"**: If >20% removed, may indicate data issues

## When to use this skill

- User says "convert the graph"
- After running `/neuronpedia-fetch`
- User asks to "prepare for analysis"
- Before running analysis or visualization

## Next steps after this skill

After converting:
1. Run `/neuronpedia-analyze` - Detect supernodes and analyze structure
2. Run `/neuronpedia-validate` - Verify conversion quality
