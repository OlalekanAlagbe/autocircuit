# Feature Hypotheses for "<bos>The state containing Seattle is"

This document presents a comprehensive framework for identifying semantic features in a neural circuit that explains how the model completes the phrase to predict "Washington".

---

## Overview

We predict **6 categories of features** active in this circuit, organized by layer and function.

---

## 1. Domain/Topic Features (L0-L10)

**Predicted:** Features detecting the semantic domain (Geography, US States, Cities).

**Layer Range:** Early to middle layers (L0-L10)

**Sample Features:**
- `7_462_1` (Layer 7, Context 1)
- `10_14174_1` (Layer 10, Context 1)

**Expected Activations:**
- US state names (Washington, California, Oregon, etc.)
- US city names (Seattle, Portland, San Francisco, etc.)
- Geographic terminology (state, city, capital, region)

**Testable Predictions:**
- These features should activate strongly on geographic prompts
- Should NOT activate on non-geographic domains (math, cooking, etc.)
- Maximum activation on "Seattle" and "state" tokens

**Counter-Hypotheses (What We Shouldn't Find):**
- Generic noun detectors with no geographic specialization
- Features that activate equally on all proper nouns

---

## 2. Syntactic/Structural Features (L1-L5)

**Predicted:** Detectors for sentence structure and spatial relation phrases.

**Layer Range:** Early layers (L1-L5)

**Sample Features:**
- `4_11742_4` (Layer 4, Seattle context)

**Expected Activations:**
- "containing" - spatial containment relation
- "located in" - location relation
- "is in" - membership/location
- "belongs to" - set membership

**Testable Predictions:**
- Features should activate on relational phrases
- Should generalize across different entity types (not just geography)
- Activation should be strongest on "containing" token (C3)

**Counter-Hypotheses:**
- Pure lexical features that only match "containing" literally
- Features that don't generalize to synonymous relations

---

## 3. Morphological/Prefix Features (L4-L12)

**Predicted:** Features processing word parts or proper noun patterns.

**Layer Range:** Middle layers (L4-L12)

**Expected Activations:**
- Proper noun capitalization patterns
- Multi-word city/state names
- Geographic naming conventions

**Testable Predictions:**
- Should activate on proper nouns in geographic context
- May show sub-word processing for "Washington" prediction

**Counter-Hypotheses:**
- Pure case-sensitivity features with no semantic content
- Features that activate on all capitalized words equally

---

## 4. Token Prediction Features (L20-L27)

**Predicted:** Specific boosters for the output token "Washington".

**Layer Range:** Late layers (L20-L27)

**Sample Features (highest influence):**
- `27_7049_5` (Layer 27, target position)
- `25_553_5`, `25_583_5` (Layer 25, target position)

**Expected Activations:**
- Direct logit contribution for "Washington"
- May also activate for related tokens (state names)

**Testable Predictions:**
- These features should have the highest total influence in the circuit
- Should be almost exclusively at context position C5 (target)
- Activation should correlate with "Washington" probability

**Counter-Hypotheses:**
- Generic late-layer features with no token-specific function
- Features that boost multiple unrelated completions equally

---

## 5. Information Routing Features (L15-L20)

**Predicted:** "Mover" heads/features that transport context from subject to target.

**Layer Range:** Mid-to-late layers (L15-L20)

**Sample Features (key routers):**
- `19_467_5` (Layer 19, Context 5) - Influence: 343.29
- `19_467_4` (Layer 19, Context 4) - Influence: 335.67

**Expected Activations:**
- Strong weights from C4 (Seattle) to C5 (is)
- Cross-position attention patterns
- Should "move" Seattle context to prediction position

**Testable Predictions:**
- Should show strong cross-attention from C4 to C5
- Influence should be high when subject-target distance is involved
- May show different routing for different query types

**Counter-Hypotheses:**
- Pure self-attention features with no cross-position flow
- Features that only process within-token information

---

## 6. Integration/Decision Features (L21-L27)

**Predicted:** Features that combine multiple streams (Domain + Syntax + Entity).

**Layer Range:** Deepest layers (L21-L27)

**Sample Features:**
- Layer 23-25 features at C5 position

**Expected Activations:**
- Combine evidence: "Seattle" (city) + "containing" (relation) + "state" (category)
- High in-degree (many incoming connections)
- Final decision-making computation

**Testable Predictions:**
- Should receive inputs from multiple earlier feature types
- High weighted in-degree in late layers
- Should be polysemantic (combining multiple concepts)

**Counter-Hypotheses:**
- Simple pass-through features with no integration
- Features with only single input source

---

## Validation Methodology

### Step 1: Sample Features

Select 2-5 representative features from each hypothesis category based on:
- Influence score (prefer high influence)
- Layer position (match hypothesis layer range)
- Context position (match expected token focus)

### Step 2: Neuronpedia Lookup

For each sampled feature, query Neuronpedia:
```
https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k/{LAYER}_{INDEX}
```

Examine:
- Top activating examples
- Activation patterns
- Feature description (if available)

### Step 3: Categorize Results

For each feature, record:
- **Confirmed**: Activations match hypothesis
- **Partial**: Some match, some unexpected
- **Refuted**: Activations contradict hypothesis
- **Polysemantic**: Feature serves multiple purposes

### Step 4: Calculate Confirmation Rate

```
Confirmation Rate = (Confirmed + 0.5 × Partial) / Total Features Sampled
```

Target: >60% confirmation rate suggests valid hypothesis framework.

### Step 5: Document Unexpected Findings

Record any:
- Polysemantic features (multiple roles)
- Surprising activation patterns
- Features that don't fit any category
- New hypothesis categories discovered

---

## Expected Results Summary

| Category | Layer Range | Expected Confirmation | Key Features |
|----------|-------------|----------------------|--------------|
| Domain/Topic | L0-L10 | High (>70%) | 7_462_1, 10_14174_1 |
| Syntactic | L1-L5 | Medium (50-70%) | 4_11742_4 |
| Morphological | L4-L12 | Medium (50-70%) | TBD |
| Token Prediction | L20-L27 | High (>70%) | L27 features |
| Routing | L15-L20 | High (>70%) | 19_467_5, 19_467_4 |
| Integration | L21-L27 | Medium (50-70%) | L23-L25 features |

---

## References

- Neuronpedia: https://neuronpedia.org/gemma-2-2b
- Graph API: https://www.neuronpedia.org/api/graph/
- Model: gemma-2-2b with gemmascope-transcoder-16k SAE
