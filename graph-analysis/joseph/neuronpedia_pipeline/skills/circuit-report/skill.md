---
name: circuit-report
description: This skill should be used when the user wants a full circuit analysis report for a prompt. Use when the user says "circuit report", "full analysis", or wants to analyze a new prompt end-to-end. Runs the pipeline and generates a comprehensive markdown report.
---

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

1. **`3_analysis/*_circuit_analysis.json`** -- Contains metadata, louvain_supernodes, layer_groups, steering_targets, all_features, graph_stats
2. **`3_analysis/traceback_paths.json`** -- Contains final_layer_nodes and critical_paths (top 5 backward-traced paths)
3. **`data/stage_1_5_cross_circuit_report.md`** -- Cross-circuit comparison data. Search for the current prompt's slug.
4. **`3_analysis/*_supernodes.json`** and **`3_analysis/*_layer_groups.json`** -- Supplementary data if needed.

### Step 3: Analyze and Write the Report

Generate `circuit_analysis_report.md` in the prompt's output directory. Follow the report template in `references/report-template.md` exactly. Apply the interpretation guidelines from `references/interpretation-guidelines.md` when writing analysis sections.

The report covers 8 sections: Model Output, Circuit Architecture, Information Flow, Critical Traceback Paths, Bottleneck Features, Cross-Circuit Comparison, Key Findings, and Graph Statistics.

## Output

The report is saved to:
```
data/prompts/{model}_{prompt-slug}/circuit_analysis_report.md
```

## Example Interaction

**User**: `/circuit-report The chemical symbol for gold is`

1. Run `python run_full_pipeline.py --prompt "The chemical symbol for gold is" --model gemma-2-2b`
2. Read the 3 data files from the output directory
3. Analyze the circuit following the template and interpretation guidelines
4. Write `circuit_analysis_report.md`
5. Report to user: location, key findings, and a 2-3 sentence summary

## When to Use

- User asks for a "full analysis" or "circuit report" for a prompt
- User says `/circuit-report`
- User wants to analyze a new prompt end-to-end
- User wants an analysis report for an already-run prompt (skip pipeline, just generate .md)

## Notes

- If the user specifies `--model both`, generate separate reports for each model
- If the pipeline was already run, check for existing output directories before re-running
- The cross-circuit report updates automatically as part of the pipeline (Stage 1.5)
- Reports accumulate over time -- each prompt gets its own .md for future meta-analysis
