---
name: steering-validate
description: This skill should be used when the user wants to test whether bottleneck features actually steer model output. Use when the user says "test steering", "validate bottlenecks", or "steering validate". Uses the Neuronpedia Steering API.
---

# Steering Validate

Validate bottleneck features by steering them via the Neuronpedia API and measuring output changes.

## Overview

This skill runs **Stage 3** (`5_steering_validation.py`) which:
- Tests whether amplifying or suppressing bottleneck features changes model output
- Uses the Neuronpedia Steering API (`POST /api/steer`)
- Measures Steering Impact Score (SIS) and Disruption Score (DS)
- Correlates cross-circuit frequency with steering effectiveness
- Generates a comprehensive validation report

## Instructions

### Quick Test (~20 minutes, ~30 API calls)

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --quick
```

Tests top 5 universal features across 2 circuits each with strengths [-20, +20].

### Single Feature Test

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --feature L3_F5150441
```

Tests one specific feature across all its GEMMA circuits with all 6 strengths.

### Full Run (~10 hours, ~950 API calls)

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py
```

Tests all cross-circuit GEMMA features. Use `--resume` to continue after interruption.

### Resume Interrupted Run

```bash
python 5_steering_validation.py --resume
```

### Analyze Existing Results

```bash
python 5_steering_validation.py --analyze-only
```

Regenerates metrics and report from saved results (no API calls).

### Custom Rate Limiting

```bash
python 5_steering_validation.py --quick --rate-delay 20
```

Override the default 36-second delay between API calls.

## Output

Files saved to: `neuronpedia_pipeline/data/stage_3_steering/`

- **`steering_baselines.json`** -- Unsteered baseline outputs for all GEMMA circuits
- **`steering_results.json`** -- All experiment results with metadata
- **`steering_analysis.json`** -- Computed metrics (SIS, DS, correlations)
- **`steering_validation_report.md`** -- Human-readable analysis report
- **`steering_checkpoint.json`** -- Progress checkpoint for resume

## Example Interaction

**User**: "Test if our bottleneck features actually steer the model"

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --quick
```

Report key findings: which features changed output, suppression vs amplification rates, frequency-disruption correlation, and early vs late layer effectiveness.

## Important Notes

- **Rate limit**: Neuronpedia allows ~100 steering calls/hour. The script enforces 36s delays by default.
- **GEMMA only**: Steering API supports gemma-2-2b. QWEN features are not tested.
- **Deterministic**: Temperature=0, seed=42 ensures reproducible results.
- **Checkpoint recovery**: Progress saved after each feature. Use `--resume` if interrupted.

## Prerequisites

1. At least 2 GEMMA circuits analyzed (full pipeline run)
2. Stage 1.5 cross-circuit analysis complete (`stage_1_5_bottleneck_library.json` exists)
3. API key configured in `config/neuronpedia_config.yaml`

## Common Issues

- **"No bottleneck library found"**: Run the full pipeline first to build the cross-circuit library
- **"No GEMMA circuits found"**: Need at least one circuit with `3_analysis/*_circuit_analysis.json`
- **"Rate limited"**: Script auto-retries after 60s. Consider increasing `--rate-delay`

## When to Use

- User asks to "test steering" or "validate bottlenecks"
- User says `/steering-validate`
- After running multiple circuit analyses to validate which features actually matter
- Before writing about steering results in a paper or report
