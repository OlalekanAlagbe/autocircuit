---
name: neuronpedia-analyze
description: This skill should be used when the user wants to analyze a circuit's structure. Use when the user says "analyze the circuit", "detect supernodes", or "run analysis". Runs Louvain community detection, betweenness centrality, and feature description fetching.
---

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

### 1. Navigate to scripts directory

```bash
cd neuronpedia_pipeline/scripts
```

### 2. Run the analysis script

```bash
python 3_analyze_circuit.py
```

The script shows available converted graphs. Select which graph to analyze (or press Enter for latest).

**Choose fetch strategy** (see `references/fetch-strategies.md` for details):
- **Option 3 (RECOMMENDED)**: Smart -- Top per supernode (~90 features, 30-60 seconds)
- Option 1: Quick -- Top features only (~15 features, 10 seconds)
- Option 2: Complete -- ALL features (~900 features, 5-10 minutes)

### 3. Wait for completion

During execution:
- Louvain community detection runs
- Feature descriptions fetched from Neuronpedia (targeted to top features per supernode)
- Layer groups analyzed
- Steering targets identified

### 4. Report results to user

Include:
- Number of supernodes detected (typically 5-15)
- Largest supernodes by size
- Layer coverage
- Feature descriptions fetched
- Output location

## Output Structure

Files saved to: `neuronpedia_pipeline/data/prompts/{prompt-slug}/3_analysis/`

- **circuit_analysis.json** -- Complete analysis with Louvain supernodes, layer groups, feature descriptions, steering targets, graph statistics
- **supernodes.json** -- Legacy format (for compatibility)

## Analysis Components

### 1. Louvain Supernodes
Community detection algorithm that groups related features. Auto-tunes resolution parameter. Typical result: 5-15 communities.

### 2. Layer Groups
- INPUT (L0-5)
- EARLY (L6-10)
- MIDDLE (L11-15)
- LATE (L16-20)
- OUTPUT (L21-25)

### 3. Steering Targets
- Input amplification candidates
- Output ablation candidates
- Bottleneck manipulation targets

## Example Interaction

**User**: "Analyze the circuit for 'Paris is the capital of'"

```bash
cd neuronpedia_pipeline/scripts
python 3_analyze_circuit.py
# Select the prompt: paris-is-the-capital-of
# Choose: 3 (Smart fetch - RECOMMENDED)
```

Report:
```
Prompt: <bos>Paris is the capital of
Output: " France" (85.7%)

Supernodes: 9 communities detected
  SN0: 154 nodes | L0-L10
  SN5: 146 nodes | L0-L20
  SN6: 133 nodes | L0-L17

Feature descriptions: 80 fetched and cached (Smart fetch)
Location: neuronpedia_pipeline/data/prompts/paris-is-the-capital-of/3_analysis/
```

## Important Notes

- **Smart fetch (Option 3) is recommended** -- fetches top features per supernode for accurate themes
- **Feature descriptions are cached** -- subsequent runs are faster
- **Supernode count varies** -- depends on circuit complexity (typically 5-15)
- **Analysis is deterministic** -- same input always produces same supernodes

## Common Issues

- **"No converted graphs found"**: Run Scripts 1 & 2 first
- **"API rate limit"**: Feature fetching may slow down, this is normal
- **"Connection error"**: Check internet for Neuronpedia API access

## When to Use

- After running `/neuronpedia-convert`
- User wants to "analyze the circuit"
- Before visualization or pathway extraction
- When exploring circuit structure

## Next Steps

After analysis:
1. **Script 4** (`/neuronpedia-visualize`) -- Create visualizations
2. **Script 3b** (`3b_traceback_paths.py`) -- Extract critical paths
3. **Script 7** -- Extract minimal pathways
