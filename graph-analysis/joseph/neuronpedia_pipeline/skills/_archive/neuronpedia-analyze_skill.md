# Neuronpedia Analyze

Detect supernodes and analyze circuit structure using Louvain community detection.

## Overview

This skill runs **Script 3** (`3_analyze_circuit.py`) which:
- Detects supernodes using Louvain community detection
- Analyzes layer groups (INPUT, EARLY, MIDDLE, LATE, OUTPUT)
- Fetches feature descriptions from Neuronpedia API
- Identifies steering targets
- Computes graph statistics

## Instructions

When the user wants to analyze a circuit:

1. **Navigate to scripts directory**:
   ```bash
   cd neuronpedia_pipeline/scripts
   ```

2. **Run the analysis script**:
   ```bash
   python 3_analyze_circuit.py
   ```
   - Script will show available converted graphs
   - User selects which graph to analyze (or press Enter for latest)
   - **Choose fetch strategy**:
     - **Option 3 (RECOMMENDED)**: Smart - Top per supernode (~90 features, 30-60 seconds)
     - Option 1: Quick - Top features only (~15 features, 10 seconds)
     - Option 2: Complete - ALL features (~900 features, 5-10 minutes)

3. **Wait for completion**:
   - **Smart fetch (Option 3)**: 30-60 seconds - best balance of speed and quality
   - **Quick fetch (Option 1)**: 10 seconds - minimal descriptions
   - **Complete fetch (Option 2)**: 5-10 minutes - exhaustive coverage

   During execution:
   - Louvain community detection runs
   - Feature descriptions fetched from Neuronpedia (targeted to top features per supernode)
   - Layer groups analyzed
   - Steering targets identified

4. **Report results to user**:
   - Number of supernodes detected (typically 5-15)
   - Largest supernodes by size
   - Layer coverage
   - Feature descriptions fetched
   - Output location

## Output Structure

Files saved to: `neuronpedia_pipeline/data/prompts/{prompt-slug}/3_analysis/`

- **circuit_analysis.json** - Complete analysis with:
  - Louvain supernodes
  - Layer groups
  - Feature descriptions (cached)
  - Steering targets
  - Graph statistics
- **supernodes.json** - Legacy format (for compatibility)

## Analysis Components

### 1. Louvain Supernodes
- Community detection algorithm
- Groups related features
- Auto-tunes resolution parameter
- Typical result: 5-15 communities

### 2. Layer Groups
- INPUT (L0-5)
- EARLY (L6-10)
- MIDDLE (L11-15)
- LATE (L16-20)
- OUTPUT (L21-25)

### 3. Feature Descriptions

**Three fetch strategies available**:

1. **Smart Fetch (Option 3 - RECOMMENDED)**:
   - Fetches top 10 features per supernode by activation
   - ~90 features total (for typical 8-9 supernodes)
   - Takes 30-60 seconds
   - Provides enough data for accurate theme inference
   - Best balance of speed and quality

2. **Quick Fetch (Option 1)**:
   - Fetches top 3 features per layer group
   - ~15 features total
   - Takes 10 seconds
   - Minimal descriptions for quick exploration

3. **Complete Fetch (Option 2)**:
   - Fetches ALL features in the graph
   - ~900 features total
   - Takes 5-10 minutes
   - Exhaustive coverage (rarely needed)

**Description format**: "Activates on: token1, token2, token3"
**Caching**: Descriptions are cached for performance on subsequent runs

### 4. Steering Targets
- Input amplification candidates
- Output ablation candidates
- Bottleneck manipulation targets

## Example Interaction

**User**: "Analyze the circuit for 'Paris is the capital of'"

**You should**:
```bash
cd neuronpedia_pipeline/scripts
python 3_analyze_circuit.py
# Select the prompt: paris-is-the-capital-of
# Choose: 3 (Smart fetch - RECOMMENDED)
```

**Report**:
```
✓ Analysis complete!

Prompt: <bos>Paris is the capital of
Output: " France" (85.7%)

Supernodes: 9 communities detected
  SN0: 154 nodes | L0-L10
  SN5: 146 nodes | L0-L20
  SN6: 133 nodes | L0-L17

Feature descriptions: 80 fetched and cached (Smart fetch)

Location: neuronpedia_pipeline/data/prompts/paris-is-the-capital-of/3_analysis/

Next step: Run Script 4 to visualize the circuit

Note: Smart fetch targeted top 10 features per supernode for optimal theme inference
```

## Important Notes

- **Smart fetch (Option 3) is recommended** - fetches top features per supernode for accurate themes
- **Feature descriptions are cached** - subsequent runs are faster
- **Supernode count varies** - depends on circuit complexity (typically 5-15)
- **Analysis is deterministic** - same input always produces same supernodes
- **Theme quality**: Smart fetch targets most activated features, which are most relevant for theme inference
- **Performance**: Smart fetch is 10× faster than Complete fetch with similar theme quality

## Common Issues

- **"No converted graphs found"**: Run Scripts 1 & 2 first
- **"API rate limit"**: Feature fetching may slow down, this is normal
- **"Connection error"**: Check internet for Neuronpedia API access

## When to Use This Skill

- After running `/neuronpedia-convert`
- User wants to "analyze the circuit"
- Before visualization or pathway extraction
- When exploring circuit structure

## Next Steps

After analysis, run:
1. **Script 4** (`/neuronpedia-visualize`) - Create 8 visualizations
2. **Script 7** - Extract minimal pathways
3. **Script 8** - Analyze supernode evolution
4. **Script 9** - Identify steering targets
