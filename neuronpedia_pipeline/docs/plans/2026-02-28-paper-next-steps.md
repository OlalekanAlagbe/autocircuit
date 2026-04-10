# Paper Next Steps — Prioritized

**Date**: 2026-02-28
**Paper**: `docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md`

---

## Priority 1: Essential-Pathway Steering

**Goal**: Test whether features on the minimal viable pathway (from D3) produce stronger steering effects than frequency-based features.

**Why**: Our biggest finding gap. We showed frequency doesn't predict causal influence. The obvious follow-up is: does graph *topology* (being on the essential path) predict causal influence? Either outcome is a major finding.

**Steps**:
1. Extract essential-pathway features from `data/stage_2_minimal_pathways/minimal_pathway_results.json`
2. Map to NP IDs (`circuit_tracer_id % 16384`)
3. Select 5 essential-pathway features that differ from our existing 10
4. Run steering experiments (same format as D5: 3 circuits × ±20 = 30 API calls)
5. Compare change rate and KL divergence against frequency-based features
6. Add results as Section 5.25 and update Discussion/Conclusion

**Estimated time**: ~20 min API calls + analysis

---

## Priority 2: Dose-Response Curve

**Goal**: Build full dose-response curves for steering at ±5, ±10, ±20, ±50 strength.

**Why**: Reviewers will want to know if effects scale linearly, have a threshold, or saturate. Currently we only test ±20.

**Steps**:
1. Select 3-5 features that showed changes (L7_F4828270, L6_F2586668, L0_F1813559, L24_F88478228)
2. Run at strengths [-50, -20, -10, -5, +5, +10, +20, +50] across 2-3 circuits
3. Plot dose-response curves per feature
4. Analyze: linear vs threshold vs saturation?
5. Add as Section 5.26

**Estimated time**: ~60-90 API calls at 36s each = ~40-55 min

---

## Priority 3: Varied Prompt Formats

**Goal**: Test whether circuit conservation is driven by domain or prompt format.

**Why**: All geography prompts are "The capital of X is" — maybe circuits are similar because prompts are similar, not because geography knowledge is conserved.

**Steps**:
1. Create 10 alternative-format prompts (varied sentence structures, same facts)
   - e.g., "What city serves as Japan's capital?" / "Japan's seat of government is"
   - Mix across all 3 domains
2. Run circuit tracing via Neuronpedia API for these 10 prompts × 2 models = 20 new circuits
3. Compare Jaccard similarity: same-fact-different-format vs same-format-different-fact
4. If circuits remain similar despite format change → genuine domain effect
5. Add as Section 5.27

**Estimated time**: 20 API calls for circuit generation + analysis

---

## Priority 4: Bootstrap Confidence Intervals

**Goal**: Add proper uncertainty quantification to all key statistics.

**Why**: With n=30 per model, point estimates are noisy. Bootstrap CIs are free and make the paper statistically stronger.

**Steps**:
1. Implement bootstrap (10,000 resamples) for all reported correlations, R² values, and effect sizes
2. Report 95% CIs alongside point estimates
3. Identify which findings survive bootstrapping and which have wide CIs
4. Update paper tables/text with CI ranges

**Estimated time**: Pure computation, ~30 min to implement and run

---

## Priority 5: Algorithm-Invariant Communities

**Goal**: Identify "hard core" communities that all 4 algorithms agree on.

**Why**: Turns the community degeneracy limitation into a positive finding.

**Steps**:
1. From D2 multi-algorithm data, find node pairs that are co-assigned by ALL 4 algorithms
2. Build invariant community graph from these consensus pairs
3. Characterize: how many invariant communities? How large? Do they correspond to bottleneck regions?
4. Add as Section 5.28

**Estimated time**: Analysis on existing data, ~1 hour

---

## Lower Priority (If Time Permits)

- **Multicollinearity-robust regression**: Ridge/LASSO/PCA regression on existing data
- **Sub-threshold feature recovery**: Vary circuit tracer threshold to find steered features
- **Wait for QWEN API**: Re-run D6 annotation and steering when available
