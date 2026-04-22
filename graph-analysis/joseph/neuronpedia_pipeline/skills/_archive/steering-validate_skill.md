# Steering Validate

Validate bottleneck features by steering them via the Neuronpedia API and measuring output changes.

## Overview

This skill runs **Stage 3** (`5_steering_validation.py`) which:
- Tests whether amplifying or suppressing bottleneck features changes model output
- Uses the Neuronpedia Steering API (`POST /api/steer`)
- Measures Steering Impact Score (SIS) and Disruption Score (DS)
- Correlates cross-circuit frequency with steering effectiveness
- Generates a comprehensive validation report

## Research Question

Do universal bottleneck features (identified via cross-circuit convergence in Stage 1.5) have predictable steering effects, and does a feature's cross-circuit frequency predict its steering impact?

## Instructions

When the user wants to validate steering targets or test bottleneck features:

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

Tests all cross-circuit GEMMA features with strengths [-50, -20, -10, +10, +20, +50]. Use `--resume` to continue after interruption.

### Resume Interrupted Run

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --resume
```

Loads checkpoint and continues from where it left off.

### Analyze Existing Results

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --analyze-only
```

Regenerates metrics and report from saved results (no API calls).

### Custom Rate Limiting

```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --quick --rate-delay 20
```

Override the default 36-second delay between API calls. Default is 36s (100 calls/hour).

## Output Structure

Files saved to: `neuronpedia_pipeline/data/stage_3_steering/`

- **`steering_baselines.json`** - Unsteered baseline outputs for all GEMMA circuits
- **`steering_results.json`** - All steering experiment results with metadata
- **`steering_analysis.json`** - Computed metrics (SIS, DS, correlations)
- **`steering_validation_report.md`** - Human-readable analysis report
- **`steering_checkpoint.json`** - Progress checkpoint for resume

## Metrics Computed

### Steering Impact Score (SIS)
Fraction of amplification experiments that changed the model's output vs baseline. Higher = feature has more influence when boosted.

### Disruption Score (DS)
Fraction of suppression experiments that changed the model's output vs baseline. Higher = feature is more critical to the prediction.

### Frequency-Disruption Correlation
Pearson correlation between a feature's cross-circuit appearance count and its mean disruption score. Positive correlation supports the thesis that universal bottlenecks have universal steering effects.

### Layer Analysis
Compares steering effectiveness of early-layer (L0-L10) vs late-layer (L11+) bottlenecks.

## Example Interaction

**User**: "Test if our bottleneck features actually steer the model"

**You should**:
```bash
cd neuronpedia_pipeline/scripts
python 5_steering_validation.py --quick
```

**Report**:
```
Stage 3: Steering Validation (Quick Mode)

Tested 5 universal features across 2 circuits each:
  - L1_F99962728: 75% change rate (suppression disrupts 3/4 experiments)
  - L3_F5150441: 50% change rate (amplification shifts predictions)
  - L5_F7993995: 25% change rate (weak steering effect)

Frequency-disruption correlation: r=0.72 (strong positive)

Key finding: Features appearing in 4+ circuits have 3x higher disruption
scores than single-circuit features.

Output: data/stage_3_steering/
  - steering_validation_report.md (full report)
  - steering_analysis.json (metrics)
```

## Important Notes

- **Rate limit**: Neuronpedia allows ~100 steering calls/hour. The script enforces 36s delays by default.
- **GEMMA only**: Steering API supports gemma-2-2b. QWEN features are not tested.
- **Deterministic**: Temperature=0, seed=42 ensures reproducible results.
- **Checkpoint recovery**: Progress is saved after each feature. Use `--resume` if interrupted.
- **Prerequisites**: Requires Stage 1.5 bottleneck library (`stage_1_5_bottleneck_library.json`) and at least 2 analyzed GEMMA circuits.

## Common Issues

- **"No bottleneck library found"**: Run the full pipeline first (`run_full_pipeline.py`) to build the cross-circuit library
- **"No GEMMA circuits found"**: Need at least one circuit with `3_analysis/*_circuit_analysis.json`
- **"Rate limited"**: Script auto-retries after 60s. Consider increasing `--rate-delay`
- **"Config not found"**: Ensure `config/neuronpedia_config.yaml` has your API key

## When to Use This Skill

- User asks to "test steering" or "validate bottlenecks"
- User says `/steering-validate`
- After running multiple circuit analyses and wanting to validate which features actually matter
- Before writing about steering results in a paper or report

## Prerequisites

Before running this skill, ensure:
1. At least 2 GEMMA circuits analyzed (full pipeline run)
2. Stage 1.5 cross-circuit analysis complete (`stage_1_5_bottleneck_library.json` exists)
3. API key configured in `config/neuronpedia_config.yaml`

## Next Steps

After steering validation:
1. **Update circuit reports** (`/circuit-report`) - Include steering validation results
2. **Paper update** - Add Stage 3 findings to the research paper
3. **Full run** - If quick test looks promising, run the complete validation overnight
