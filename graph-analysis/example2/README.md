# Analysis of: "<bos>The state containing Seattle is"

## Overview

This directory contains a complete circuit analysis for the prompt:
**"<bos>The state containing Seattle is"**

**Model:** gemma-2-2b
**Circuit Size:** 1,045 nodes, 37,978 edges
**Task:** Identify the state containing Seattle (Target: "Washington")

---

## Files

### Graph Information
- **`graph_info.txt`** - Complete metadata about the graph including:
  - API endpoints and download commands
  - Prompt and token information
  - Generation settings and thresholds

### Analysis Documents
1. **`CIRCUIT_ANALYSIS_COMPLETE.md`** - **START HERE**
   - Executive summary of all findings
   - Answers to three key questions about the circuit
   - Feature hypotheses and validation strategy
   - Recommended as entry point for understanding the analysis

2. **`circuit_description.md`** - Detailed technical analysis
   - Complete circuit flow breakdown
   - Information flow pathways with weights
   - Token-level routing patterns
   - Phase-by-phase computational strategy
   - ASCII circuit diagrams

3. **`feature_hypotheses.md`** - Comprehensive hypothesis document
   - 6 categories of predicted feature types
   - Expected activation patterns
   - Testable predictions for each category
   - Counter-hypotheses (what we shouldn't find)
   - Validation methodology

4. **`hypothesis_features_summary.txt`** - Quick reference guide
   - Key features sampled for each hypothesis
   - Feature IDs with influence and activation scores
   - Neuronpedia lookup URLs
   - Expected vs. unexpected findings

### Data Files
- **`supernodes.json`** - Supernode groupings for graph visualization
- **`analyze_graph.py`** - Python script for reproducing the analysis

---

## Key Findings

### Circuit Architecture
The model uses a **hierarchical target-focused circuit** with distinct computational phases:

1. **Embedding Injection** (Layer E)
   - Strongest influence comes from direct embeddings (Weight: 2,168 to L0)
   - Top Nodes: `E_2329_2` (state), `E_26996_4` (Seattle)

2. **Distributed Early Processing** (L0-L5)
   - High parallelism across all tokens
   - Focus on "Seattle" (C4) and "containing" (C3) entities

3. **Routing & Context Shifting** (L6-L20)
   - Information routing from Context tokens to Target
   - Node `19_467_5` acts as a key late-stage router

4. **Decision Integration** (L21-L27)
   - Final integration at the target position (C5)
   - Aggregates evidence to predict the completion "Washington"

### Information Flow Patterns

**Top Flows by Total Weight:**
1. L-1 (Embedding) → L0: 2,168 weight
2. L-1 → L2: 794 weight
3. L-1 → L1: 651 weight
4. L-1 → L3: 466 weight
5. L-1 → L4: 465 weight

**Token-Level Routing:**
- C5 → C5: 4,331 edges (Self-processing at target)
- C1 → C1: 2,056 edges
- C3 → C3: 1,542 edges
- C4 → C4: 1,474 edges

### Critical Insight: Embedding Dominance
The analysis reveals that the **Embedding Layer (L-1)** is the single largest source of influence, directly feeding early layers (L0-L4) with over 4,500 total weight. This suggests the model relies heavily on the initial semantic values of "Seattle" and "state".

---

## Feature Hypotheses

We predict **6 categories of features** active in this circuit:

### 1. Domain/Topic Features (L0-L10)
**Predicted:** Geography/US States detectors
**Sample Features:** `7_462_1`, `10_14174_1`
**Expected activations:** State names, City names

### 2. Syntactic/Structural Features (L1-L5)
**Predicted:** "containing" relation detectors
**Sample Features:** `4_11742_4` (Seattle context)
**Expected activations:** "containing", "located in"

### 3. Morphological/Prefix Features (Middle Layers)
**Predicted:** Token parts processing
**Expected activations:** Sub-words of "Washington"

### 4. Integration Routing Features (L15-L20)
**Predicted:** Moving "Seattle" context to "is" position
**Sample Features:** `19_467_5`, `19_467_4`
**Expected:** Strong cross-attention from C4 to C5

---

## Validation Strategy

### Step 1: Feature Lookup
Identify key features from each category (top features extracted in `hypothesis_features_summary.txt`)

### Step 2: Semantic Verification
For each feature, check on Neuronpedia:
```
https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_index}
```

Verify:
- Do activation examples match predictions?
- Is the feature semantically interpretable?
- Does it serve the hypothesized role?

---

## How to Reproduce

### 1. Download Graph Data

```bash
# Get metadata
curl -X GET "https://www.neuronpedia.org/api/graph/gemma-2-2b/thestatecontaini-1768998665701" \
  -H "x-api-key: YOUR_API_KEY"

# Download full graph
curl -s -X GET "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmknzdk5r0000x491tcid8y2s/thestatecontaini-1768998665701.json" \
  -o graph_data.json
```

### 2. Run Analysis Scripts

```bash
# Run the full analysis suite:
python3 analyze_graph.py --api-key YOUR_API_KEY --slug thestatecontaini-1768998665701
```

### 3. Explore Results

Start with `CIRCUIT_ANALYSIS_COMPLETE.md` for the executive summary, then dive into specific documents based on your interests:
- Circuit flow details → `circuit_description.md`
- Feature predictions → `feature_hypotheses.md`
- Quick reference → `hypothesis_features_summary.txt`

---

## Research Questions

This example helps address:

1. **How does the model retrieve factual knowledge?**
   - Answer: Strong reliance on embedding injection of key entities ("Seattle").

2. **Where does the "relation" processing happen?**
   - Answer: Distributed across L0-L5 ("containing").

3. **How is the target predicted?**
   - Answer: Late-stage integration at C5 ("is") combining context from C4.

---

## Acknowledgments

- **Graph Generator:** circuit-tracer
- **Platform:** Neuronpedia (neuronpedia.org)
- **Model:** gemma-2-2b
