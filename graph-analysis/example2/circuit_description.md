# Detailed Circuit Description: "<bos>The state containing Seattle is"

## Overview

- **Model**: gemma-2-2b
- **Prompt**: "<bos>The state containing Seattle is"
- **Target Output**: Logit for "Washington" at position C5
- **Circuit Size**: 1,045 nodes, 37,978 edges

---

## 1. GROUPING FEATURES INTO SUPERNODES

### Methodology

Features are grouped into **supernodes** based on two dimensions:
- **Layer**: Computational depth (E, 0-27)
- **Context Position**: Which token they process (C0-C5)

Tokens: `["<bos>", "The", " state", " containing", " Seattle", " is"]`

### Supernode Distribution

| Layer Range | Focus | Feature Count | Description |
|-------------|-------|---------------|-------------|
| E (Embeddings) | All tokens | 6 | Initial token representations |
| L0-L5 | Distributed | ~300 | Parallel token processing |
| L6-L15 | Shifting to C5 | ~100 | Information routing |
| L16-L22 | Mostly C5 | ~50 | Decision preparation |
| L23-L27 | C5 only | ~30 | Final prediction |

---

## 2. MAIN INFORMATION FLOWS

### Top 5 Layer-to-Layer Flows

| Source | Target | Weight | Interpretation |
|--------|--------|--------|----------------|
| L-1 (E) | L0 | 2,168 | Embedding injection |
| L-1 (E) | L2 | 794 | Skip to early processing |
| L-1 (E) | L1 | 651 | Sequential processing |
| L-1 (E) | L3 | 466 | Distributed processing |
| L-1 (E) | L4 | 465 | Distributed processing |

**Key Insight**: Embeddings dominate early information flow with direct connections to layers 0-4.

### Token-Level Routing Patterns

| Source Token | Target Token | Edges | Interpretation |
|--------------|--------------|-------|----------------|
| C5 (is) | C5 (is) | 4,331 | Self-attention at target |
| C1 (The) | C1 (The) | 2,056 | Article processing |
| C3 (containing) | C3 (containing) | 1,542 | Relation processing |
| C4 (Seattle) | C4 (Seattle) | 1,474 | Entity processing |
| C2 (state) | C2 (state) | 1,055 | Noun processing |

**Key Pattern**: Self-attention dominates, but cross-token flows (C4→C5) carry semantic content to the prediction site.

---

## 3. COMPUTATIONAL PHASES

### Phase 1: Distributed Token Analysis (Layers 0-5)

```
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDINGS (Layer E)                                           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────┐ ┌─────────┐ ┌────┐       │
│  │<bos>│ │ The │ │state│ │containing│ │ Seattle │ │ is │       │
│  │ C0  │ │ C1  │ │ C2  │ │    C3    │ │   C4    │ │ C5 │       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └────┬─────┘ └────┬────┘ └─┬──┘       │
│     │       │       │         │            │        │           │
│     ▼       ▼       ▼         ▼            ▼        ▼           │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              LAYER 0 (335+ features)                │        │
│  │   Parallel processing of all token positions        │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**What's being computed:**
- "Seattle" recognized as city entity
- "state" identified as geographic noun
- "containing" detected as spatial relation
- "is" marked as prediction target

### Phase 2: Context Integration (Layers 6-15)

```
┌─────────────────────────────────────────────────────────────────┐
│  MIDDLE LAYERS: Information Routing                             │
│                                                                 │
│    C1 (The)  C2 (state)  C3 (containing)  C4 (Seattle)         │
│       │          │             │               │                │
│       │          │             │               │                │
│       └──────────┴─────────────┴───────────────┘                │
│                          │                                      │
│                          ▼                                      │
│                 ┌─────────────────┐                             │
│                 │   C5 ("is")     │                             │
│                 │  TARGET TOKEN   │                             │
│                 └─────────────────┘                             │
│                                                                 │
│  Pattern: All context flows toward target position              │
└─────────────────────────────────────────────────────────────────┘
```

**What's being computed:**
- Geographic query: "Which state contains Seattle?"
- Relation binding: "containing" links "state" to "Seattle"
- Context aggregation at C5

### Phase 3: Decision Bottleneck (Layers 16-22)

```
┌─────────────────────────────────────────────────────────────────┐
│  LATE LAYERS: High-Confidence Selection                         │
│                                                                 │
│  Features narrow to C5 only:                                    │
│                                                                 │
│  L16: ████ (C5)                                                 │
│  L17: ███████ (C5)                                              │
│  L18: █████████ (C5)                                            │
│  L19: ██████████████ (C5)  ← Key router nodes                   │
│  L20: █████████████████ (C5)                                    │
│  L21: ████████████████████ (C5)                                 │
│  L22: ██████████████████████████ (C5)                           │
│                                                                 │
│  High influence features: 19_467_5, 19_467_4                    │
└─────────────────────────────────────────────────────────────────┘
```

**What's being computed:**
- Crystallizing prediction: "Washington"
- High-confidence feature selection
- Suppressing alternative completions

### Phase 4: Output Transformation (Layers 23-27)

```
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT MODULE: Dense Decision Network                          │
│                                                                 │
│  L23_C5 ──────┬──────────────────┐                              │
│               │                  │                              │
│               ▼                  ▼                              │
│           L24_C5 ─────────→ L25_C5                              │
│               │                  │                              │
│               │                  │                              │
│               └────────┬─────────┘                              │
│                        ▼                                        │
│                    L27_C5                                       │
│                  (OUTPUT LOGIT)                                 │
│                 "Washington"                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**What's being computed:**
- Final combination of evidence
- Logit boosting for "Washington"
- Probability assignment

---

## 4. CRITICAL INSIGHT: Skip Connections

The circuit utilizes direct connections from Layer 0 to decision modules:

```
LAYER 0 ─────────────────────────────────────────────┐
    │                                                │
    ├─→ L1 ─→ L2 ─→ L3 ─→ ... ─→ L20 ─→ L21 ──┐    │
    │                                          │    │
    └──────────────────────────────────────────┼────┤
                                               ▼    ▼
                                            L23/L24/L25
                                            (Decision)
```

This suggests that early, raw token features are preserved and used directly for the final decision, bypassing intermediate processing.

---

## 5. FUNCTIONAL CIRCUIT SUMMARY

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: <bos> The state containing Seattle is                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDINGS (E) - Strong output from "Seattle" and "state"      │
│  E_2329_2 (state): 1747 influence                               │
│  E_26996_4 (Seattle): 1382 influence                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  EARLY LAYERS (L0-L5): Parallel Token Processing                │
│  - Distributed analysis across all 6 tokens                     │
│  - Token-specific analysis (Seattle as city, state as noun)     │
│  - Begin building geographic context                            │
└─────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  MIDDLE LAYERS (L6-L15) │   │  SKIP CONNECTIONS       │
│  Target-Focused Routing │   │  L0 → L23/L24/L25       │
│  C4 (Seattle) → C5 (is) │   │  Preserve early features│
└─────────────────────────┘   └─────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  BOTTLENECK (L16-L22): High-Confidence Feature Selection        │
│  - Features concentrate at C5 (target position)                 │
│  - Key nodes: 19_467_5, 19_467_4                                │
│  - Crystallize prediction: "Washington"                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT MODULE (L23-L27): Dense Decision Network                │
│  - Tightly interconnected layers                                │
│  - Final logit computation                                      │
│  - Output: "Washington"                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. KEY COMPUTATIONAL PRINCIPLES

1. **Hierarchical Processing**: Early layers analyze tokens independently, late layers integrate at target position

2. **Skip Connections**: Layer 0 connects to layers 23-25, preserving low-level features for high-level decision

3. **Target-Focused Funnel**: Information progressively routes toward C5 ("is") through middle layers

4. **Dense Output Module**: Layers 23-27 form tightly connected decision network

5. **Multi-Evidence Integration**: Circuit combines:
   - "Seattle" (city entity)
   - "state" (geographic category)
   - "containing" (spatial relation)
   - Position 5 embedding (target location)
