---
name: essential-pathway-steer
description: Use when the user wants to run essential-pathway steering experiments, test features on the minimal viable pathway, or validate causal influence via topology-based feature selection. Replicates Section 5.25 of the cross-domain paper.
---

# Essential-Pathway Steering Validation

Run causal steering experiments on features that appear on the essential (minimal viable) pathway of GEMMA circuits, and compare results to frequency-based feature selection. This is the experiment that produced the "three-tier dissociation" finding: topology predicts distributional perturbation, but circuit redundancy absorbs text-level changes.

## Instructions

1. **Dry run (see which features will be tested before spending API calls):**

```bash
python scripts/stage_2_essential_pathway_steering.py --dry-run
```

Reconstructs the essential pathway for all 30 GEMMA circuits, selects 5 high-frequency pathway features not previously tested, and prints the planned experiment count (default: 5 features x 3 circuits x 2 strengths = 30 experiments, ~18 min).

2. **Full run (30 experiments, ~18 minutes at 36s rate limit):**

```bash
python scripts/stage_2_essential_pathway_steering.py
```

Uses the same 3 target circuits as the D5 expansion (sodium / japan / wwii) for fair comparison. Tests at strengths ±20 only.

3. **Analyze existing results without running new experiments:**

```bash
python scripts/stage_2_essential_pathway_steering.py --analyze-only
```

Re-reads `data/stage_2_essential_pathway_steering/essential_pathway_steering_results.json` and re-runs the comparison analysis against `data/stage_3_steering/steering_results.json`.

4. **Adjust rate limiting or feature count:**

```bash
python scripts/stage_2_essential_pathway_steering.py --rate-delay 40 --n-features 5
```

Default rate delay is 36s (100 calls/hour). Increase if you see 429 errors.

## Output

```
data/stage_2_essential_pathway_steering/
  essential_pathway_steering_results.json   <- per-experiment results + analysis
```

The JSON includes:
- `metadata`: timestamp, target circuits, strengths, already-tested features
- `selected_features`: the 5 essential-pathway features chosen
- `feature_pathway_status`: which of the 10 D5 features ARE vs ARE NOT on essential pathways
- `results`: per-experiment baseline text, steered text, text_changed flag, KL divergence, logprob shift
- `analysis`: aggregate statistics and comparison to D5

## Example Interaction

**User:** Run the essential pathway steering experiment and tell me what you find

**Command:**

```bash
python scripts/stage_2_essential_pathway_steering.py
```

**Expected output (abbreviated):**

```
[1/5] Extracting essential-pathway features from converted graphs...
  Processed 36 GEMMA circuits
  Total unique features on essential pathways: 1000

[2/5] Selecting 5 candidate features...
  Selected: L0_F64712375, L1_F1736314, L21_F5479683, L25_F50014975, L24_F18002975

[3/5] Running 30 steering experiments...
  Collecting baselines...
  [1/30] SUPPRESS -20 on chemistry...  KL=0.4413, shift=0.1117
  ...
  [5/30] SUPPRESS -20 on history...    KL=2.9418, shift=0.2465 CHANGED
  ...

[4/5] Analyzing results...
  ESSENTIAL-PATHWAY AGGREGATE:
    Text change rate: 8/30 (26.7%)
    Mean KL divergence: 1.4476

  === KEY FINDING ===
  Essential-pathway features: 26.7% text change rate, 1.4476 mean KL
  -> Essential-pathway features show LOWER text-change rate than frequency-only
     but HIGHER mean KL divergence (stronger distributional perturbation)
```

## Common Issues

- **"No essential-pathway features found":** The script reconstructs pathways from `data/prompts/<circuit>/2_conversion/*.json`. If converted graphs are missing, run the main pipeline first on the 30 GEMMA prompts.
- **Rate limiting (429):** Increase `--rate-delay` to 40 or 45 seconds. The default 36s is tight for 100 calls/hour.
- **Text changes concentrated in history:** Expected. The three-tier finding shows history prompts are most susceptible (60% change rate), geography intermediate (20%), chemistry completely resistant (0%). This is output determinism, not circuit topology.

## When to Use

- Reproducing Section 5.25 of the cross-domain circuit paper
- Testing whether a new feature selection strategy (topology, frequency, or something else) predicts causal influence
- Extending the essential-pathway analysis to additional features, circuits, or strengths
- Validating new features before including them in the paper

## Related Skills

- `/steering-validate` — run the original D4 steering experiments (10 features, broader sweep)
- `/neuronpedia-analyze` — inspect the essential pathway for a single circuit

## Prerequisites

- 30 GEMMA circuits with converted graphs in `data/prompts/gemma-2-2b_*/2_conversion/`
- Neuronpedia API key with steering endpoint access
- ~18 minutes of wall-clock time per 30-experiment run

## Key Finding (paper Section 5.25)

The experiment revealed a three-tier dissociation:

1. **Essential-pathway features** produce the highest distributional perturbation (mean KL = 1.448) but only 26.7% text change rate.
2. **D5 frequency-based features** that happen to be on essential pathways: 18.8% text change rate.
3. **D5 frequency-based features NOT on essential pathways:** 33.3% text change rate (highest of any group).

Neither pathway position nor cross-circuit frequency reliably predicts text-level causal influence. Circuit redundancy (94.1%, from Section 5.20) absorbs perturbations before the output layer. Output determinism (chemistry 0%, geography 20%, history 60%) is the dominant factor governing steering susceptibility.
