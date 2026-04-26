# Circuit Analysis Report Template

Generate a `circuit_analysis_report.md` in the prompt's output directory following this structure exactly:

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
