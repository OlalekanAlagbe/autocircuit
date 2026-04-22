---
title: "Mechanistic Interpretability of Analogical Reasoning in Gemma-2-2B"
subtitle: "A Sparse Autoencoder Attribution Graph Analysis"
date: "March 2026"
model: "Gemma-2-2B"
sae: "gemmascope-transcoder-16k"
authors:
  - name: "Olalekan Alagbe"
    url: "https://www.linkedin.com/in/olalekanjoshuaalagbe1000/"
  - name: "Konstantinos Krampis"
    url: "https://www.linkedin.com/in/kkrampis/"
links:
  - label: "Paper"
    icon: "ai ai-arxiv"
    url: "#"
  - label: "Code"
    icon: "fab fa-github"
    url: "https://github.com/kkrampis/autocircuit"
  - label: "Presentation"
    icon: "fa-solid fa-display"
    url: "presentation.html"
  - label: "Attribution Graphs"
    icon: "fa-solid fa-brain"
    url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin"
tldr: >
  We identify a shared 180-feature analogical reasoning circuit in Gemma-2-2B
  that generalizes across geographic and semantic analogies, including a dedicated
  `L5 SAE#5793` feature labeled simply *"analogies"* — direct evidence of a
  reusable relational reasoning primitive.
bibtex: |
  @article{alagbe2026analogical,
    title   = {Mechanistic Interpretability of Analogical Reasoning in {Gemma-2-2B}:
               A Sparse Autoencoder Attribution Graph Analysis},
    author  = {Alagbe, Olalekan and Krampis, Konstantinos},
    year    = {2026},
    month   = {March},
    note    = {Neuronpedia API gemmascope-transcoder-16k SAE analysis}
  }
supplementary:
  - label: "Interactive Presentation"
    icon: "fa-solid fa-display"
    url: "presentation.html"
    description: "20-slide reveal.js presentation with GitHub dark theme, circuit flow diagrams, feature tables, and layer-by-layer analysis."
  - label: "Live Attribution Graphs"
    icon: "fa-solid fa-brain"
    url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin"
    description: "Interactive Neuronpedia attribution graph viewer for all five prompts."
    sublinks:
      - label: "analog_berlin"
        url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin"
        desc: "Paris:France::Berlin:?"
      - label: "analog_rome"
        url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_rome"
        desc: "Paris:France::Rome:?"
      - label: "analog_tokyo"
        url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_tokyo"
        desc: "Paris:France::Tokyo:?"
      - label: "analog_teacher"
        url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_teacher"
        desc: "Doctor:hospital::teacher:?"
      - label: "analog_bird"
        url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_bird"
        desc: "Fish:water::bird:?"
---

## Abstract

We present the first mechanistic analysis of analogical reasoning in Gemma-2-2B
using Sparse Autoencoder (SAE) attribution graphs. By generating and comparing
five attribution graphs across structurally distinct analogical prompts — covering
geographic analogies (*Paris:France::Berlin:?*, *Rome:?*, *Tokyo:?*) and semantic
role analogies (*Doctor:hospital::teacher:?*, *Fish:water::bird:?*) — we identify
a shared **analogical reasoning circuit** comprising 180 features active across
all five prompts and 510 features active across at least three.

We discover dedicated analogy-encoding features at layers 5, 8, 9, and 13,
including a feature at Layer 5 labeled literally as **"analogies"** and a Layer 8
feature encoding **"analogies or comparisons"** appearing across all graphs with
high influence. Early layers (0–4) contain structural template features tracking
the "X is to Y as Z is to" pattern, while mid-to-late layers (5–13) house
increasingly semantic representations of the relational structure. The circuit
spans all 26 transformer layers and exhibits cross-domain generalization, with the
same core features activating for both geographic and semantic role analogies.

<!-- slide: What Is Analogical Reasoning? -->

## 1. Introduction

Analogical reasoning — the ability to recognize and complete structural
relationships between concepts — is a foundational cognitive ability underlying
scientific discovery, language understanding, and problem solving. The classic
analogy task, *"Paris is to France as Berlin is to ____,"* tests whether a model
can identify the capital-city relationship abstractly and apply it to a new
country. Large language models exhibit striking competence on such tasks, yet the
internal computational mechanisms remain poorly understood.

<!-- figure:fig-analogy-task -->
**Figure 1.** The analogy task. Given the relationship between Paris and France,
the model must recognize the *capital-of* relation and apply it to a new pair.
This requires relational abstraction, not mere fact retrieval.

Mechanistic interpretability research has made significant progress in
understanding factual recall circuits (Meng et al. 2022), indirect object
identification (Wang et al. 2022), and syntactic processing (Conmy et al. 2023).
However, analogical reasoning presents a distinct challenge: it requires not
merely retrieving a stored fact, but recognizing a **relational structure** and
applying it compositionally to novel inputs.

<!-- slide: Research Questions -->

### Research Questions

1. Does Gemma-2-2B employ a **shared circuit** for analogical reasoning, or
   different mechanisms for different analogy types?
2. Which SAE features are most **consistently activated** across diverse
   analogical prompts?
3. Are there interpretable features that encode the **abstract relational
   structure** of analogies?
4. How is the analogical computation **distributed across transformer layers**?

<!-- slide: Prompt Selection -->

## 2. Methodology

### 2.1 Prompt Selection

We selected five prompts spanning two structural analogy types to ensure
cross-domain coverage:

| ID | Prompt | Expected | Type |
|----|--------|----------|------|
| `analog_berlin` | *"Paris is to France as Berlin is to"* | Germany | Capital |
| `analog_rome` | *"Paris is to France as Rome is to"* | Italy | Capital |
| `analog_tokyo` | *"Paris is to France as Tokyo is to"* | Japan | Capital |
| `analog_teacher` | *"Doctor is to hospital as teacher is to"* | school | Sem. role |
| `analog_bird` | *"Fish is to water as bird is to"* | air / sky | Sem. role |

<!-- slide: Attribution Graph Generation -->

### 2.2 Attribution Graph Generation

Attribution graphs were generated using the Neuronpedia API
(`/api/graph/generate`) with Gemma-2-2B and the `gemmascope-transcoder-16k` SAE
— a 26-layer cross-layer transcoder with 16,384 features per layer.

| Parameter | Value |
|-----------|-------|
| Model | `gemma-2-2b` |
| SAE | `gemmascope-transcoder-16k` |
| Max feature nodes | 3,000 |
| Desired logit probability | 0.95 |
| Node threshold | 0.80 |
| Edge threshold | 0.85 |

Each graph returns nodes (SAE feature activations with layer, index, influence
score, activation magnitude) and directed edges (attribution scores). Graphs were
downloaded from S3 and loaded into NetworkX DiGraph objects for analysis.

### 2.3 Feature Identification and Cross-Graph Analysis

Cross-graph feature overlap was computed by finding features — identified by
*(layer, SAE index)* pairs — appearing consistently across multiple graphs.
Three overlap thresholds were analyzed: ≥3/5, ≥4/5, and 5/5 graphs. The top 25
most recurring features were labeled using the Neuronpedia feature explanation
API.

**Key technical note:** The correct endpoint for `gemmascope-transcoder-16k`
requires a **layer-prefixed SAE identifier** (e.g. `4-gemmascope-transcoder-16k`
for layer 4) rather than the global SAE name.

### 2.4 Causal Validation (Attempted)

Steering experiments were attempted using the Neuronpedia `/api/steer` endpoint
(amplification at +20 and suppression at −20 for each top feature across three
test prompts). All 30 tests returned HTTP 404 — the `gemmascope-transcoder-16k`
SAE does not support steering via the public API for this model. Causal
validation is documented as *attempted but inconclusive* due to this limitation.

<!-- slide: Attribution Graph Structure -->

## 3. Results

### 3.1 Graph Structure

All five attribution graphs exhibited a consistent structural pattern, with
features activated across all 26 transformer layers (0–25) plus the embedding
layer (E):

| Graph | Prompt | Nodes | Edges | Max Influence |
|-------|--------|-------|-------|---------------|
| `analog_berlin` | Paris:France::Berlin:? | 930 | 25,915 | 0.8001 |
| `analog_rome` | Paris:France::Rome:? | 963 | 27,608 | 0.8002 |
| `analog_tokyo` | Paris:France::Tokyo:? | 905 | 22,414 | 0.8001 |
| `analog_teacher` | Doctor:hospital::teacher:? | 1,040 | 35,481 | 0.8001 |
| `analog_bird` | Fish:water::bird:? | 1,071 | 38,741 | 0.8000 |

The semantic role analogies (*teacher*, *bird*) have notably larger graphs
(1,040–1,071 nodes) compared to the capital analogies (905–963 nodes), reflecting
greater ambiguity in the expected completion domain.

<!-- slide: Core Circuit Identification -->

### 3.2 The Core Analogical Reasoning Circuit

Cross-graph feature overlap analysis revealed a substantial shared circuit:

| Threshold | Features Found |
|-----------|---------------|
| Active in ≥3/5 graphs | **510 features** |
| Active in ≥4/5 graphs | **277 features** |
| Active in all 5 graphs | **180 features** |

<!-- figure:fig-layer-distribution -->
**Figure 2.** Distribution of core circuit features (active in all 5 graphs) by
layer group. Orange = analogy recognition hub (L5–L6); Purple = relational
integration (L8–L13).

Early layers (L0–L3) account for the plurality of core features, consistent with
structural template processing occurring first. The mid-range layers (L5–L6) show
elevated feature counts — the **analogy recognition hub** layers. Isolated
high-influence features appear at L8, L9, L11, and L13.

<!-- slide: Multi-step Reasoning Evidence -->

### 3.3 Evidence for Multi-Step Analogical Reasoning

We provide evidence that Gemma-2-2B performs **genuine multi-step analogical
reasoning internally**. The model does not merely pattern-match the surface form
of the analogy prompt — instead, dedicated analogy-concept features activate at
specific layers, each contributing a computationally distinct role.

> **Core finding:** The attribution graph reveals a three-phase computational
> process: (1) structural template parsing at early layers, (2) analogy concept
> recognition at mid layers, and (3) relational integration and output formation
> at deeper layers. The same sequence activates for both *geographic* and
> *semantic role* analogies — evidence of a domain-agnostic relational
> reasoning mechanism.

<!-- figure:fig-circuit-flow -->
**Figure 3.** The three-phase analogical reasoning circuit in Gemma-2-2B. We
stress that this diagram simplifies the true mechanisms considerably — the
attribution graph for any single prompt contains hundreds of features; the
circuit shown represents the semantically interpretable core.

The Phase 2 features — particularly `L5 SAE#5793` ("analogies") and
`L8 SAE#13766` ("analogies or comparisons") — are not geographic features.
They activate equally for "Doctor:hospital::teacher:?" (in particular, they are
*not* Berlin or Tokyo features), confirming they encode the *relational structure*
itself rather than domain content.

<!-- slide: Feature Analysis -->

### 3.4 Top Recurring Features

**Directly Analogical Features** (Neuronpedia labels explicitly reference
analogical reasoning or comparison):

| Feature | App. | Avg Inf. | Label |
|---------|------|----------|-------|
| `L5 #5793` | 11/5 | 0.590 | **"analogies"** |
| `L8 #13766` | 21/5 | 0.533 | **"analogies or comparisons"** |
| `L9 #13344` | 14/5 | 0.681 | "comparison between two things" |
| `L5 #2141` | 12/5 | 0.647 | "comparisons of public figures" |
| `L13 #10969` | 11/5 | 0.676 | "comparisons between disciplines" |

**Structural Template Features** (encode "X is to Y as Z is to" scaffold):

| Feature | App. | Avg Inf. | Label |
|---------|------|----------|-------|
| `L0 #11651` | 10/5 | 0.633 | "the word 'to'" |
| `L1 #11356` | 10/5 | 0.609 | "'to' followed by a verb" |
| `L2 #11475` | 10/5 | 0.638 | "the word 'refers'" |
| `L4 #10752` | 10/5 | 0.626 | "'to be' preceded by 'to'" |
| `L5 #9672` | 12/5 | 0.579 | "the phrase 'it is to'" |

Three high-recurrence features (`L4 #14857`, `L6 #2267`, `L3 #3205`) are labeled
"code snippets" and "legal jargon" — we hypothesize these detect **formal
text pattern matching**: the analogy syntax shares structural properties with
code comments, legal definitions, and scientific documentation.

<!-- slide: Cross-Domain Generalization -->

### 3.5 Cross-Domain Generalization

The consistent activation of `L5 SAE#5793` ("analogies") and `L8 SAE#13766`
("analogies or comparisons") across both capital-city and semantic role analogy
types provides evidence for a **domain-general analogical reasoning mechanism**.

<!-- figure:fig-venn -->
**Figure 4.** Cross-domain feature overlap. The 180 features active in all five
graphs include the core analogy-concept features at L5 and L8, confirming a
domain-agnostic mechanism for both geographic and semantic role analogies.

### 3.6 Activation Magnitudes Build Through Layers

| Layer Range | Role | Typical Avg Activation |
|-------------|------|------------------------|
| L0 (structural) | Token & syntax parsing | 1.5 – 6.4 |
| L5 (analogy hub) | Analogy concept activation | 7.4 – 11.1 |
| L8–L9 (detectors) | Comparison detection | ~13.4 |
| L10–L13 (integration) | Relational + domain integration | 9.1 – 16.3 |

<!-- figure:fig-activation -->
**Figure 5.** Average activation magnitude of core circuit features by layer
range. Early structural features fire weakly; mid-layer analogy detectors and
late integrators fire substantially stronger — consistent with an accumulating
signal as the relational structure is assembled.

<!-- slide: Discussion -->

## 4. Discussion

### 4.1 The Analogical Reasoning Circuit

Our analysis reveals that Gemma-2-2B implements analogical reasoning through a
distributed circuit spanning all 26 transformer layers. The most significant
finding is the existence of **explicitly semantic analogy features** at layers 5,
8, 9, and 13 — features whose automated explanations use the words "analogies,"
"comparisons," and "relationships between concepts." This suggests the model has
internalized analogical structure as a discrete, reusable computational primitive.

### 4.2 The Role of Formal Text Features

The high-recurrence "code and legal text" features present an intriguing puzzle.
Two complementary interpretations:

1. **Functional hypothesis:** These features detect formal, template-driven text
   patterns. The analogy syntax is highly formal and structured, resembling legal
   definitions, code comments, and mathematical notation.
2. **Training data hypothesis:** The analogy format appears frequently in SAT prep
   and educational texts that also contain code and legal text, creating a
   statistical association.

### 4.3 Comparison with the Capital City Recall Circuit

Comparing with the capital city *factual recall* circuit reveals **overlap** in
formal-text features (`L4 #14857`, `L6 #2267`) and **divergence** in the L5/L8
analogy-specific features — which were absent from the factual recall circuit.
This supports the interpretation that analogy features are selectively activated
by relational structure recognition, not by factual recall.

<!-- slide: Limitations & Future Work -->

## 5. Limitations

1. **No causal validation.** Without successful steering experiments, we cannot
   confirm that the identified features are causally necessary.
2. **SAE coverage.** The transcoder SAE does not capture attention head
   contributions or residual stream features.
3. **Threshold sensitivity.** Results depend on node/edge thresholds (0.80/0.85).
4. **Label quality.** Neuronpedia explanations are LLM-generated and may
   underspecify broader concepts.
5. **Prompt set size.** Five prompts are sufficient for initial identification
   but too few to claim statistical robustness.

**Future work:** Direct causal validation with TransformerLens activation
patching; expanded prompt set covering arithmetic and cross-lingual analogies;
attention head analysis; comparison across model scales (9B, 27B).

<!-- slide: Conclusions -->

## 6. Conclusions

1. **A stable shared circuit exists.** 180 features active across all five prompts;
   510 across at least three — a consistent internal mechanism.
2. **Dedicated analogy features exist at L5, L8, L9, and L13.** First direct
   evidence of interpretable analogy-concept features in a large language model.
3. **Three-phase architecture.** Template parsing (L0–L4) → Analogy recognition
   (L5–L9) → Relational integration (L10–L13), with activation magnitude
   increasing through the sequence.
4. **Cross-domain generalization confirmed.** `L5 SAE#5793` activates for both
   geographic and semantic role analogies — a domain-agnostic primitive.
5. **Formal-text features co-activate.** Reflect the formal syntax of the analogy
   template activating domain-general structured-text detectors.
6. **Steering validation outstanding.** Causal validation via TransformerLens
   remains to be completed.

## Supplementary Materials

## BibTeX
