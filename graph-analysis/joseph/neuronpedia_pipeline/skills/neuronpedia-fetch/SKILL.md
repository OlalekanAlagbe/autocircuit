---
name: neuronpedia-fetch
description: This skill should be used when the user wants to generate or fetch an attribution graph from Neuronpedia. Use when the user says "fetch a graph", "generate a graph", or wants to start a new circuit analysis for a prompt.
---

# Neuronpedia Fetch

Generate attribution graphs from the Neuronpedia Circuit Tracer API for circuit analysis.

## Overview

This skill runs **Script 1** (`1_generate_graph.py`) which:
- Connects to Neuronpedia API
- Generates circuit graphs for any prompt
- Downloads results from S3
- Organizes files using PathManager

## Instructions

### 1. Navigate to scripts directory

```bash
cd neuronpedia_pipeline/scripts
```

### 2. Run the generation script

```bash
python 1_generate_graph.py
```

The script prompts for the text to analyze. Alternatively, provide via command line:

```bash
python 1_generate_graph.py --prompt "Your text here"
```

### 3. Wait for completion (typically 10-20 seconds)

The script:
- Connects to Neuronpedia API
- Submits prompt for Circuit Tracer analysis
- Downloads resulting graph from S3
- Saves to organized directory structure

### 4. Report results to user

Include:
- Prompt used
- Output location (organized by prompt slug)
- Number of nodes (typically 900-1100)
- Number of edges (typically 20K-55K)
- File size (typically 2-6 MB)
- **Model output prediction** and probability

## Output Structure

Files are saved to: `neuronpedia_pipeline/data/prompts/{prompt-slug}/1_generation/`

- **raw_graph.json** -- Complete Circuit Tracer graph
- **metadata.json** -- Prompt info, model, timestamps

## Example Interaction

**User**: "Fetch a graph for 'The capital of France is'"

```bash
cd neuronpedia_pipeline/scripts
echo "The capital of France is" | python 1_generate_graph.py
```

Report:
```
Graph generated successfully!

Prompt: <bos>The capital of France is
Output: " a" (p=20.7%)
Location: neuronpedia_pipeline/data/prompts/the-capital-of-france-is/1_generation/
Nodes: 1088
Edges: 54382
Size: 6.15 MB

Next step: Run Script 2 to convert the graph
```

## Important Notes

- **Model predictions vary by prompt wording** -- "The capital of X is" vs "X is the capital of" can produce different results
- **No target token override** -- Neuronpedia always uses the model's top prediction
- **API is public** -- No authentication required for graph generation
- **Files are organized by prompt** -- Each prompt gets its own directory

## Common Issues

- **"Connection timeout"**: Neuronpedia API may be busy, retry
- **"Graph generation failed"**: Check internet connection
- **Large graphs**: Some prompts generate 1000+ nodes, this is normal

## When to Use

- User asks to "fetch a graph" or "generate a graph"
- User wants to analyze a specific prompt
- Starting a new circuit analysis
- Testing how a model processes text

## Next Steps

After fetching:
1. **Script 2** (`/neuronpedia-convert`) -- Convert to standard format
2. **Script 3** (`/neuronpedia-analyze`) -- Detect supernodes and analyze
