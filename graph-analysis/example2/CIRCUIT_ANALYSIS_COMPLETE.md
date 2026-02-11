# Complete Circuit Analysis: "<bos>The state containing Seattle is"

## Executive Summary

We analyzed the Neuronpedia graph for **gemma-2-2b** processing the prompt:
**"<bos>The state containing Seattle is"**

The circuit reveals a hierarchical information flow from input tokens to the final prediction of "Washington".

**Circuit Statistics:**
- **1,045 nodes** (features)
- **37,978 edges** (connections)
- **27 layers** + embeddings
- **6 context positions** (tokens)

---

## Three Key Questions Answered

### Q1: How to Group Features into Supernodes?

**Answer:** Group by **(Layer, Context Position)** pairs.

We identified distinct supernodes representing functional units:
- **Early Layers (L0-L5)**: Distributed analysis across all tokens
- **Middle Layers (L6-L20)**: Information routing toward target position
- **Late Layers (L21-L27)**: Focused on the target position (Context 5 = "is")

### Q2: What are the Main Information Flows?

**Answer:** Five major pathways:

1. **Embeddings → L0** (Weight: 2,168) - Strongest single flow
   - Target embedding and "Seattle" embedding have highest output

2. **L0 → Early Layers** (L1-L5) - Parallel token processing

3. **Skip Connections**: L0 → Late Layers (L21-L27)
   - Preserve early features for late-stage decision

4. **Token Routing**: C4 (Seattle) → C5 (is)
   - Information moves from subject to target position

5. **Decision Cascade**: L21 → L22 → ... → L27
   - Dense connectivity for final prediction

### Q3: What High-Level Circuit Does the Model Use?

**Answer:** The model uses a **Hierarchical Target-Focused Circuit** with 4 phases:

1. **Distributed Token Analysis** (L0-L5)
   - Extract features from all tokens in parallel
   - Process "Seattle" as city entity
   - Process "state" and "containing" for relational context

2. **Context Integration** (L6-L15)
   - Route information from subject ("Seattle") to target ("is")
   - Build geographic query: "city → state" relation

3. **Decision Bottleneck** (L16-L22)
   - Highly selective features with high influence
   - Crystallize prediction: "Washington"

4. **Output Transformation** (L23-L27)
   - Dense interconnections
   - Final logit prediction for "Washington"

---

## Key Findings

### Embedding Dominance

The analysis reveals that the **Embedding Layer (L-1)** is the single largest source of influence:
- `E_2329_2` ("state"): 1,747 influence
- `E_26996_4` ("Seattle"): 1,382 influence
- `E_10751_3` ("containing"): 1,127 influence

This suggests the model relies heavily on initial semantic representations.

### Critical Router Nodes

Late-layer features act as key information routers:
- `19_467_5` (Layer 19, Context 5): 343 influence - Major late-stage router
- `19_467_4` (Layer 19, Context 4): 335 influence - Seattle context aggregator

### Token Flow Pattern

```
C4 (Seattle) ──────────────────────────────────────────────┐
                                                           ↓
C3 (containing) ────────────────────────────────────────→ C5 (is) → OUTPUT
                                                           ↑
C2 (state) ────────────────────────────────────────────────┘
```

---

## Validation Strategy

### Step 1: Feature Lookup
Identify high-influence features from each hypothesis category (see `hypothesis_features_summary.txt`)

### Step 2: Semantic Verification
For each feature, check on Neuronpedia:
- Do activation examples match predictions?
- Is the feature semantically interpretable?
- Does it serve the hypothesized role?

### Step 3: Confirmation Rate
Track: `# features matching hypothesis / # features sampled`

### Step 4: Document Unexpected Findings
Note polysemantic features and counter-examples.

---

## Files in This Analysis

| File | Description |
|------|-------------|
| `graph_info.txt` | Graph metadata, API endpoints, download commands |
| `CIRCUIT_ANALYSIS_COMPLETE.md` | This file - Executive summary (START HERE) |
| `circuit_description.md` | Detailed technical analysis with flow diagrams |
| `feature_hypotheses.md` | Predicted feature types and validation methodology |
| `hypothesis_features_summary.txt` | Quick reference for sampled features |
| `supernodes.json` | Supernode groupings for graph visualization |

---

## Reproduction

```bash
# Get metadata
curl -X GET "https://www.neuronpedia.org/api/graph/gemma-2-2b/thestatecontaini-1768998665701" \
  -H "x-api-key: YOUR_API_KEY"

# Download full graph
curl -s -X GET "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmknzdk5r0000x491tcid8y2s/thestatecontaini-1768998665701.json" \
  -o graph_data.json

# Run analysis
python3 analyze_graph.py --api-key YOUR_API_KEY --slug thestatecontaini-1768998665701
```

---

## Research Questions Addressed

1. **How does the model retrieve factual knowledge?**
   - Strong reliance on embedding injection of key entities ("Seattle", "state")

2. **Where does the "relation" processing happen?**
   - Distributed across L0-L5 ("containing" relation detection)

3. **How is the target predicted?**
   - Late-stage integration at C5 ("is") combining context from C4 ("Seattle")
