# Circuit Report

Run the full Neuronpedia pipeline on a prompt and generate a comprehensive analysis report as a Markdown file.

## Overview

This skill:
1. Runs the full pipeline orchestrator (`run_full_pipeline.py`)
2. Reads all output data files (circuit_analysis.json, traceback_paths.json, cross-circuit report)
3. Analyzes the circuit: information flow, bottlenecks, cross-circuit patterns, steering targets
4. Writes a `circuit_analysis_report.md` into the prompt's output directory

## Instructions

### Step 1: Run the Pipeline

Run the orchestrator for the user's prompt. Default to `gemma-2-2b` unless the user specifies a model.

```bash
cd neuronpedia_pipeline
python run_full_pipeline.py --prompt "<USER_PROMPT>" --model <MODEL>
```

- Timeout: 10 minutes. The pipeline typically completes in 3-8 minutes.
- If the pipeline has already been run (output directory exists with all 4 step directories), skip this step and go straight to analysis.
- If only some steps completed, re-run the full pipeline (it will overwrite).

### Step 2: Read the Data Files

After the pipeline completes, read these files from `data/prompts/{model}_{prompt-slug}/`:

1. **`3_analysis/*_circuit_analysis.json`** -- Contains:
   - `metadata`: prompt, model, predictions, node/edge counts
   - `metadata.top_predictions`: array of {token, probability}
   - `metadata.model_output`: predicted token
   - `louvain_supernodes`: community detection results with size, layers, activation, influence, degree
   - `layer_groups`: INPUT/EARLY_PROC/MIDDLE_PROC/LATE_PROC/OUTPUT with node counts, activations, top features
   - `steering_targets`: features with descriptions identified as input/output/bottleneck candidates
   - `all_features`: all fetched feature descriptions keyed by node_id
   - `graph_stats`: density, degree, connectivity, activation/influence/weight statistics

2. **`3_analysis/traceback_paths.json`** -- Contains:
   - `final_layer_nodes`: all output nodes ranked by contribution
   - `critical_paths`: top 5 paths traced backward from output, each with ordered path_nodes containing layer, activation, influence, score, label

3. **`data/stage_1_5_cross_circuit_report.md`** -- Cross-circuit comparison data. Search for the current prompt's slug to find:
   - Per-circuit bottleneck summary (how many features, which layers, cross-circuit overlap %)
   - Universal bottleneck features table

4. **`3_analysis/*_supernodes.json`** and **`3_analysis/*_layer_groups.json`** -- Supplementary data if needed.

### Step 3: Analyze and Write the Report

Generate a `circuit_analysis_report.md` in the prompt's output directory. The report should follow this structure exactly:

```markdown
# Circuit Analysis: "<PROMPT>"

**Model:** <model>
**Date:** <today>
**Predicted Token:** "<token>" (<probability>%)
**Circuit Size:** <nodes> nodes, <edges> edges, <num_layers> layers (L0-L<max>)
**Supernodes:** <count> (Louvain, resolution=<res>)
**Layer Groups:** 5

---

## 1. Model Output
Table of top 10 predictions with token and probability.
Note what the model "knows" -- look at tokens beyond #1 for semantic clues.

## 2. Circuit Architecture
### Layer Group Summary
Table: Layer Group | Layers | Nodes | Mean Act | Max Act | Top Feature

### Supernode Rankings
Table of top 6 supernodes by importance score with size, layers, mean activation, and inferred role.

## 3. Information Flow (Input to Output)
Walk through each layer group (INPUT, EARLY_PROC, MIDDLE_PROC, LATE_PROC, OUTPUT):
- List 3-5 key features per group with activation, description, and cross-circuit status
- Note activation jumps between groups
- Identify where prompt-specific semantics first emerge
- Highlight any features whose activation examples directly relate to the prompt topic

## 4. Critical Traceback Paths
### Path 1 (highest contribution)
Show the full path as an ASCII tree with feature labels and descriptions.
Provide one-line interpretation of the path's semantic meaning.

### Top 5 Final Layer Nodes
Table: Rank | Feature | Contribution | Activation | Influence

## 5. Bottleneck Features
### Key Bottlenecks (ranked by traceback score)
Table: Feature | Layer | Score | Activation | Cross-Circuit | NP Description | Interpretation

### Prompt-Specific Features (not cross-circuit)
Table of features unique to this circuit with semantic role interpretation.

### Steering Targets
Table: Feature | Layer | Betweenness | What It Does | Steering Potential
Rank by betweenness centrality. Note the best surgical target.

## 6. Cross-Circuit Comparison
### This Circuit vs Others
Table comparing bottleneck count, cross-circuit %, and depth range.
Pull data from the stage_1_5_cross_circuit_report.md.

### GEMMA vs QWEN (if both models were run)
Compare bottleneck depth distributions between models.

### Shared Universal Bottlenecks
Table of cross-circuit features found in this circuit.

## 7. Key Findings
3-6 bullet points summarizing the most important discoveries:
- Semantic layering pattern
- Where prompt-specific knowledge emerges
- Cross-circuit overlap rate and what it means
- Which bottlenecks are misnamed/misleading vs genuinely interpretable
- Best steering target and why

## 8. Graph Statistics
Table of density, average degree, components, activation/influence/weight stats, features fetched.
```

### Interpretation Guidelines

When writing the report:

1. **Feature descriptions from Neuronpedia are often misleading.** Activation examples like "Drupal views" or "HTML tags" on a universal bottleneck don't mean the feature literally encodes Drupal. Cross-circuit universality is the clue -- if a feature appears in 5+ circuits across different prompts, it encodes something abstract (relational structure, emphasis, routing) not the literal surface pattern.

2. **Look for directly semantic features.** These are features whose activation examples directly relate to the prompt topic (e.g., "Africa, Africa" or "river, streams, Mississippi" for a river prompt). These typically emerge in L16-L20 (LATE_PROC) and are the most interpretable.

3. **Traceback scores indicate information routing importance.** A score of 2 billion vs 500 million means that feature carries 4x more information toward the output. The highest-scoring feature in the traceback is the single most critical relay.

4. **Cross-circuit overlap rate tells you how unique a circuit is.** Chemistry prompts share 60-75% of bottlenecks. Unusual factual queries may share only 20-30%. Low overlap = specialized pathway.

5. **Betweenness centrality identifies chokepoints.** High betweenness = information must flow through this node. Best steering targets are high-betweenness, prompt-specific features (not universal ones that would disrupt everything).

## Output

The report is saved to:
```
data/prompts/{model}_{prompt-slug}/circuit_analysis_report.md
```

## Example Interaction

**User**: `/circuit-report The chemical symbol for gold is`

**You should**:
1. Run `python run_full_pipeline.py --prompt "The chemical symbol for gold is" --model gemma-2-2b`
2. Read the 3 data files from the output directory
3. Analyze the circuit following the template and guidelines above
4. Write `circuit_analysis_report.md`
5. Report to user: location, key findings, and a 2-3 sentence summary

## When to Use This Skill

- User asks for a "full analysis" or "circuit report" for a prompt
- User says `/circuit-report`
- User wants to analyze a new prompt end-to-end
- User wants an analysis report for an already-run prompt (skip pipeline, just generate .md)

## Notes

- If the user specifies `--model both`, generate separate reports for each model
- If the pipeline was already run, check for existing output directories before re-running
- The cross-circuit report updates automatically as part of the pipeline (Stage 1.5)
- Reports accumulate over time -- each prompt gets its own .md for future meta-analysis
