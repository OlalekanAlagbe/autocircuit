---
title: "Mechanistic Interpretability of Analogical Reasoning in Gemma-2-2B"
subtitle: "A Sparse Autoencoder Attribution Graph Analysis"
date: "March 2026"
model: "Gemma-2-2B"
sae: "gemmascope-transcoder-16k"
authors:
  - name: "Olalekan Alagbe"
    url: "https://www.linkedin.com/in/olalekanjoshuaalagbe1000/"
  - name: "Joseph Lawrence"
    url: ""
  - name: "Anish Maheshwar"
    url: ""
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
    author  = {Alagbe, Olalekan and Lawrence, Joseph and Maheshwar, Anish and Krampis, Konstantinos},
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



### 3.7 Causal Validation via Feature Steering

<p>The attribution graph analysis above identifies recurring features and causal path structures, but does not establish whether these features are causally necessary for the model's predictions. A feature that is highly active in the graph may be a byproduct of the computation rather than a driver of it. To distinguish load-bearing circuit components from high-activation but functionally inert nodes, we performed systematic causal steering experiments using the Neuronpedia <code>/api/steer</code> endpoint.</p>

<p><strong>Methodology.</strong> For each of the five prompts in this paper, we generated a Neuronpedia attribution graph (§2.2) and traced the top-5 causal paths from the logit node backward through the graph (greedy walk following highest-weight incoming edges at each step) and the top-5 paths forward from the highest-influence embedding nodes. All transcoder features appearing on at least one of these 10 paths constitute the circuit's <em>backbone</em>. We then applied four experimental paradigms, each targeting a different causal question:</p>
<ol>
<li><strong>Necessity (individual suppression):</strong> Suppress a single backbone feature at strength −20 and check whether the model's first predicted token changes. Tests whether each feature is individually a single point of failure.</li>
<li><strong>Necessity (full backbone suppression):</strong> Suppress all backbone features simultaneously. Tests whether the backbone is collectively necessary.</li>
<li><strong>Sufficiency (hub boost):</strong> Boost a backbone hub feature at strength +20 on an altered prompt. Tests whether a single feature can induce the original prediction in a new context.</li>
<li><strong>Specificity (non-backbone suppression):</strong> Suppress high-activation features that do <em>not</em> appear on any of the top-10 causal paths. Tests whether the path tracing correctly identifies load-bearing features.</li>
</ol>
<p>All experiments use <code>modelId: "gemma-2-2b"</code>, <code>layer: "{N}-gemmascope-transcoder-16k"</code>, and <code>strength_multiplier: 4</code>. Two additional analogical circuits from the expanded 30-prompt analysis (Cairo→Kenya, Puppy→cat) were included for cross-validation. In total, we report 159 individual steering API calls across 7 circuits.</p>

<h4>3.7.1 Late-Layer Backbone Necessity (Individual Suppression)</h4>

<p>Late-layer features (L17+) sit on the direct causal chain from mid-layer integration hubs to the logit node. We suppressed each one individually at strength −20 and recorded whether the model's first predicted token changed. Full per-circuit results follow.</p>

<p style="font-weight:700;margin-top:1.2em;">analog_berlin — "Paris is to France as Berlin is to" → Germany (p=0.973)</p>

<table>
<thead><tr><th>Feature</th><th>Layer</th><th>Index</th><th>Role in Causal Path</th><th>Steered Token</th><th>Necessary?</th></tr></thead>
<tbody>
<tr><td>Science hub</td><td>21</td><td>4827</td><td>Strongest path entry (edge +198.0)</td><td>Germany</td><td>no</td></tr>
<tr><td>Relay</td><td>22</td><td>15670</td><td>Path 1 relay</td><td>Germany</td><td>no</td></tr>
<tr><td>Output driver A</td><td>25</td><td>4717</td><td>Final amplifier (shared across circuits)</td><td>Germany</td><td>no</td></tr>
<tr><td>Location encoder</td><td>16</td><td>6491</td><td>Location/direction feature, path 2 entry</td><td>Germany</td><td>no</td></tr>
<tr><td>Relay</td><td>17</td><td>14546</td><td>Mid-cascade relay</td><td>Germany</td><td>no</td></tr>
<tr><td>Relay</td><td>19</td><td>5773</td><td>Late relay</td><td>Germany</td><td>no</td></tr>
<tr><td>Integrator</td><td>21</td><td>7482</td><td>Integration hub (paths 2–4)</td><td>Germany</td><td>no</td></tr>
<tr style="font-weight:700"><td>Output driver B</td><td>25</td><td>2725</td><td>Secondary output driver (edge −2.09)</td><td>the</td><td>YES</td></tr>
<tr><td>Relation applier</td><td>19</td><td>855</td><td>Relation application node</td><td>Germany</td><td>no</td></tr>
</tbody>
</table>

<p><strong>1/9 necessary.</strong> Only L25/2725 is a single point of failure. Suppressing it shifts the prediction from "Germany" to "the" — the model loses the entity but retains grammatical continuation. The primary path through L21→L22→L25_4717 provides fully redundant coverage for the remaining 8 features. The highest-weight feature in the graph (L21/4827, edge +198.0) is not individually necessary, demonstrating that attribution weight alone does not predict causal necessity.</p>

<p style="font-weight:700;margin-top:1.2em;">analog_rome — "Paris is to France as Rome is to" → Italy (p=0.974)</p>

<table>>
<thead><tr><th>Feature</th><th>Layer</th><th>Index</th><th>Role</th><th>Steered Token</th><th>Necessary?</th></tr></thead>
<tbody>
<tr><td>Relay</td><td>20</td><td>15360</td><td>Backward path from logit</td><td>Italy</td><td>no</td></tr>
<tr style="font-weight:700"><td>Late gate</td><td>24</td><td>16122</td><td>Backward path, L24 suppression gate</td><td>the</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Output driver</td><td>25</td><td>286</td><td>Backward path, output driver</td><td>the</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Final amplifier</td><td>25</td><td>4717</td><td>Shared final amplifier (act=265.2)</td><td>the</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Output driver C</td><td>25</td><td>10521</td><td>Tertiary output driver</td><td>the</td><td>YES</td></tr>
<tr><td>Relay</td><td>17</td><td>14546</td><td>Mid-cascade relay (shared with Berlin)</td><td>Italy</td><td>no</td></tr>
<tr><td>Relay A</td><td>22</td><td>12202</td><td>Late relay</td><td>Italy</td><td>no</td></tr>
<tr><td>Relay B</td><td>22</td><td>14727</td><td>Late relay</td><td>Italy</td><td>no</td></tr>
<tr><td>Relay</td><td>23</td><td>5917</td><td>Late relay</td><td>Italy</td><td>no</td></tr>
<tr><td>Secondary gate</td><td>24</td><td>13277</td><td>Late gate</td><td>Italy</td><td>no</td></tr>
</tbody>
</table>

<p><strong>4/10 necessary.</strong> All four necessary features sit at L24–L25, forming a tight bottleneck. L24/16122 is the suppression gate through which the dominant backward path passes; L25/286, L25/4717, and L25/10521 are three independent output drivers, each individually necessary. The Rome circuit has more single points of failure than Berlin despite similar confidence (p=0.974 vs 0.973), indicating that path redundancy varies even among structurally similar geographic analogies. All four failure modes produce "the" — the model loses entity selection but retains article generation.</p>

<p style="font-weight:700;margin-top:1.2em;">analog_tokyo — "Paris is to France as Tokyo is to" → Japan (p=0.990)</p>

<table>>
<thead><tr><th>Feature</th><th>Layer</th><th>Index</th><th>Role</th><th>Steered Token</th><th>Necessary?</th></tr></thead>
<tbody>
<tr><td>Relay</td><td>20</td><td>15360</td><td>Backward path from logit</td><td>Japan</td><td>no</td></tr>
<tr style="font-weight:700"><td>Output driver</td><td>25</td><td>286</td><td>Backward path, output driver</td><td>the</td><td>YES</td></tr>
<tr><td>Output driver B</td><td>25</td><td>12223</td><td>Backward path, secondary output</td><td>Japan</td><td>no</td></tr>
<tr><td>Relay</td><td>17</td><td>14546</td><td>Mid-cascade relay (shared)</td><td>Japan</td><td>no</td></tr>
<tr style="font-weight:700"><td>Late relay A</td><td>23</td><td>850</td><td>Late relay</td><td>the</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Late relay B</td><td>23</td><td>13914</td><td>Late relay (also necessary in Cairo circuit)</td><td>the</td><td>YES</td></tr>
<tr><td>Gate</td><td>24</td><td>13277</td><td>Late gate (shared with Rome)</td><td>Japan</td><td>no</td></tr>
<tr><td>Output driver C</td><td>25</td><td>10152</td><td>Tertiary output</td><td>Japan</td><td>no</td></tr>
<tr><td>Hub</td><td>20</td><td>6648</td><td>L20 convergence hub</td><td>Japan</td><td>no</td></tr>
<tr><td>Integration</td><td>21</td><td>7764</td><td>Late integration</td><td>Japan</td><td>no</td></tr>
</tbody>
</table>

<p><strong>3/10 necessary.</strong> The necessary features span L23–L25 rather than concentrating at L25 alone. Notably, <strong>L23/13914 is necessary in both the Tokyo and Cairo circuits</strong> — this feature is a shared bottleneck for country-prediction analogies, consistent with a late-layer "geographic entity selector" role. L25/286 recurs as necessary here and in Rome and Cairo, making it the single most critical output driver across geographic analogies (necessary in 3/4 tested capital circuits). Despite having the highest confidence of any circuit (p=0.990), Tokyo has 3 necessary features — indicating that prediction confidence and path redundancy are not perfectly correlated.</p>

<p style="font-weight:700;margin-top:1.2em;">analog_teacher — "Doctor is to hospital as teacher is to" → school (p=0.486)</p>

<table>>
<thead><tr><th>Feature</th><th>Layer</th><th>Index</th><th>Role</th><th>Steered Token</th><th>Necessary?</th></tr></thead>
<tbody>
<tr style="font-weight:700"><td>Embedding</td><td>0</td><td>17</td><td>Backward path, embedding-level</td><td>the</td><td>YES</td></tr>
<tr><td>Gateway</td><td>18</td><td>6532</td><td>Backward path, mid-late gateway</td><td>school</td><td>no</td></tr>
<tr><td>Hub</td><td>20</td><td>6179</td><td>Backward path, convergence hub</td><td>school</td><td>no</td></tr>
<tr style="font-weight:700"><td>Output driver</td><td>25</td><td>4975</td><td>Backward path, output driver</td><td>...</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Final amplifier</td><td>25</td><td>4717</td><td>Shared final amplifier (act=135.6)</td><td>a</td><td>YES</td></tr>
<tr><td>Relay</td><td>22</td><td>15670</td><td>Late relay (shared)</td><td>school</td><td>no</td></tr>
<tr><td>Relay B</td><td>18</td><td>11952</td><td>Mid-late relay</td><td>school</td><td>no</td></tr>
<tr><td>Legal docs</td><td>18</td><td>13586</td><td>Legal docs feature</td><td>school</td><td>no</td></tr>
<tr><td>Convergence</td><td>21</td><td>2655</td><td>Late convergence hub</td><td>school</td><td>no</td></tr>
<tr><td>Gate</td><td>24</td><td>15259</td><td>Late suppression gate</td><td>school</td><td>no</td></tr>
</tbody>
</table>

<p><strong>3/10 necessary.</strong> The teacher circuit is the only one where an <strong>L0 embedding-level feature</strong> (L0/17) is individually necessary — suppressing it changes the prediction from "school" to "the." This suggests the semantic role analogy relies on an early feature that is not redundantly encoded by later layers, unlike the capital analogies where embedding features are always compensated. Both L25 output drivers (4975 and 4717) are independently necessary, each producing a different failure token ("..." and "a"), indicating they carry non-overlapping information to the logit. The three L18 features (gateway, relay, legal docs) are all dispensable, consistent with the mid-layer features serving as redundant relays rather than bottlenecks.</p>

<p style="font-weight:700;margin-top:1.2em;">analog_bird — "Fish is to water as bird is to" → air (p=0.117)</p>

<table>>
<thead><tr><th>Feature</th><th>Layer</th><th>Index</th><th>Role</th><th>Steered Token</th><th>Necessary?</th></tr></thead>
<tbody>
<tr style="font-weight:700"><td>Backward A</td><td>22</td><td>4252</td><td>Backward path from logit</td><td>(space)</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Backward B</td><td>24</td><td>8106</td><td>Backward path, late gate</td><td>____</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Final amplifier</td><td>25</td><td>4717</td><td>Shared final amplifier (act=122.3)</td><td>the</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Output driver</td><td>25</td><td>11801</td><td>Output driver</td><td>?</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Relay A</td><td>22</td><td>15670</td><td>Late relay (shared)</td><td>________</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Relay B</td><td>22</td><td>14727</td><td>Late relay</td><td>(space)</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Relay C</td><td>22</td><td>13619</td><td>Late relay</td><td>(space)</td><td>YES</td></tr>
<tr><td>Gate A</td><td>24</td><td>4383</td><td>Suppression gate</td><td>air</td><td>no</td></tr>
<tr style="font-weight:700"><td>Gate B</td><td>24</td><td>12559</td><td>Suppression gate</td><td>the</td><td>YES</td></tr>
<tr><td>Hub</td><td>20</td><td>3094</td><td>Integration hub</td><td>air</td><td>no</td></tr>
</tbody>
</table>

<p><strong>8/10 necessary.</strong> This is the most fragile circuit in the dataset — nearly every feature is a single point of failure. Failure modes include whitespace, underscores, question marks, and "the" — the model loses the ability to produce any coherent completion, not just the target "air." Three L22 relay features (15670, 14727, 13619) are all independently necessary despite occupying the same layer, indicating they carry non-redundant information through parallel channels at L22. Only L24/4383 and L20/3094 are dispensable. This fragility is consistent with the circuit's very low prediction confidence (p=0.117): the model barely resolves "air" over competitors ("sky", "the", "nest"), and removing almost any contributing feature tips the balance.</p>

<div class="finding-box">
<p><strong>Cross-circuit pattern: Necessity inversely correlates with prediction confidence.</strong> Berlin (p=0.973): 1/9 necessary. Rome (p=0.974): 4/10. Tokyo (p=0.990): 3/10. Teacher (p=0.486): 3/10. Bird (p=0.117): 8/10. High-confidence predictions route through redundant parallel causal paths; low-confidence predictions rely on fragile, non-redundant chains. Three features recur as necessary across multiple circuits: <strong>L25/286</strong> (Rome, Tokyo, Cairo), <strong>L25/4717</strong> (Rome, Teacher, Bird), and <strong>L23/13914</strong> (Tokyo, Cairo).</p>
</div>

<h4>3.7.2 Full Backbone Suppression</h4>

<p>To test whether the backbone features are <em>collectively</em> necessary — rather than just individually redundant — we suppressed all late-layer backbone features (L17+) simultaneously for each circuit.</p>

<table>
<thead><tr><th>Circuit</th><th>p</th><th>N feat.</th><th>Default Output</th><th>Steered Output</th><th>Disrupted?</th></tr></thead>
<tbody>
<tr style="font-weight:700;"><td><code>analog_berlin</code></td><td>0.973</td><td>9</td><td>Germany. It is the</td><td>of of of of of</td><td>YES</td></tr>
<tr style="font-weight:700;"><td><code>analog_rome</code></td><td>0.974</td><td>10</td><td>Italy. It is the</td><td>pleaſure pleaſure plea</td><td>YES</td></tr>
<tr style="font-weight:700;"><td><code>analog_tokyo</code></td><td>0.990</td><td>10</td><td>Japan. It is the</td><td>country country count</td><td>YES</td></tr>
<tr style="font-weight:700;"><td><code>analog_teacher</code></td><td>0.486</td><td>10</td><td>school. The doc</td><td>1111</td><td>YES</td></tr>
<tr style="font-weight:700;"><td><code>analog_bird</code></td><td>0.117</td><td>10</td><td>air. The fish</td><td>(newline) The the the</td><td>YES</td></tr>
<tr style="font-weight:700;"><td>Cairo→Kenya</td><td>0.963</td><td>9</td><td>Kenya. It is the</td><td>(whitespace)</td><td>YES</td></tr>
<tr><td>Puppy→cat</td><td>0.756</td><td>4</td><td>cat. I'</td><td>cat. I think</td><td>no</td></tr>
</tbody>
</table>

<p><strong>6/7 circuits fully disrupted</strong>, including all five prompts from this paper. The failure modes are qualitatively distinct and informative:</p>
<ul>
<li><strong>Capital analogies</strong> degenerate to repetitive or archaic text. Berlin produces "of of of of of" (function word repetition); Rome produces "pleaſure pleaſure" (the model falls into an archaic English register with long-s characters); Tokyo produces "country country count" (the model retrieves the category "country" but cannot resolve which country). These failure modes suggest the backbone features are required for entity selection, while the prompt structure alone activates a "country" category representation without specific entity resolution.</li>
<li><strong>Teacher</strong> collapses to "1111" — numeric output with no semantic content. The model loses both the category and the entity, consistent with the semantic role analogy requiring more backbone computation than the geographic analogies where "country" is at least partially retained.</li>
<li><strong>Bird</strong> produces "(newline) The the the" — the model falls through to generic text continuation, consistent with the circuit's already-fragile state (8/10 individually necessary features).</li>
<li><strong>Puppy→cat is the sole exception</strong> — its prediction survives full backbone suppression. This circuit has only 4 tested backbone features (the fewest) and a moderate confidence (p=0.756), suggesting the prediction is carried primarily by direct embedding→logit connections outside the multi-hop backbone. The top-5 causal path tracing misses the critical nodes for this circuit.</li>
</ul>

<h4>3.7.3 Phase 1 and Phase 2 Feature Necessity</h4>

<p>The three-phase architecture proposed in §3.3 — template parsing (Phase 1, L0–L4), analogy recognition (Phase 2, L5–L9), and relational integration (Phase 3, L10–L13) — is based on feature co-occurrence and semantic labels. To test whether these phases are <em>causally</em> necessary, we suppressed each of the 9 key phase features identified in §3.3 individually across all five prompts (9 features × 5 prompts = 45 tests).</p>

<table>>
<thead><tr><th>Feature</th><th>Phase</th><th>Neuronpedia Label</th><th>Berlin</th><th>Rome</th><th>Tokyo</th><th>Teacher</th><th>Bird</th></tr></thead>
<tbody>
<tr style="font-weight:700"><td>L0/11651</td><td>1</td><td>"the word 'to'"</td><td>Berlin</td><td>Rome</td><td>Tokyo</td><td>school</td><td>water</td></tr>
<tr><td>L1/11356</td><td>1</td><td>"'to' followed by a verb"</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>L4/10752</td><td>1</td><td>"'to be' preceded by 'to'"</td><td>—</td><td>—</td><td>—</td><td style="font-weight:700;">classroom</td><td style="font-weight:700;">sky</td></tr>
<tr><td>L5/9672</td><td>1</td><td>"the phrase 'it is to'"</td><td>—</td><td>—</td><td>—</td><td>—</td><td style="font-weight:700;">sky</td></tr>
<tr><td>L5/5793</td><td>2</td><td><strong>"analogies"</strong></td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>L5/2141</td><td>2</td><td>"comparisons of public figures"</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>L8/13766</td><td>2</td><td>"analogies or comparisons"</td><td>—</td><td>—</td><td>—</td><td>—</td><td style="font-weight:700;">fish</td></tr>
<tr><td>L9/13344</td><td>2</td><td>"comparison between two things"</td><td>—</td><td>—</td><td>—</td><td>—</td><td style="font-weight:700;">sky</td></tr>
<tr><td>L13/10969</td><td>3</td><td>"comparisons between disciplines"</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>
<p class="figure-caption">Cells show the steered first token when the feature is suppressed at strength −20. "—" = prediction unchanged (feature is individually redundant). Bold = prediction disrupted.</p>

<p><strong>Phase 1 results.</strong> <code>L0/11651</code> ("the word 'to'") is the most causally important early-layer feature, necessary in <strong>4/5 circuits</strong>. Its failure mode is diagnostic: suppressing it causes all three capital analogies to predict the <em>city name</em> itself (Berlin, Rome, Tokyo) rather than the target country. The model loses the "X is to Y as Z is to ___" template structure and instead completes with the most recently mentioned entity — it no longer recognizes the prompt as an analogy. For the bird circuit, suppressing L0/11651 produces "water" (the source-pair element), further confirming that the model reverts to association with the earlier part of the prompt when the template is disrupted. <code>L4/10752</code> ("'to be' preceded by 'to'") is necessary for 2/5 circuits: the teacher circuit predicts "classroom" (a near-synonym) and the bird circuit predicts "sky" (a competitor). <code>L5/9672</code> ("the phrase 'it is to'") is necessary only for the bird circuit.</p>

<p><strong>Phase 2 results.</strong> The headline feature <code>L5/5793</code> ("analogies") is <strong>never individually necessary</strong> in any circuit. Neither is <code>L5/2141</code> ("comparisons of public figures"). This does not mean Phase 2 is inert — it means Phase 2 features are individually redundant for high-confidence circuits. For the fragile bird circuit (p=0.117), Phase 2 features <em>are</em> individually necessary: <code>L8/13766</code> ("analogies or comparisons") changes "air" to "fish" — the model retrieves the source-domain animal instead of the target-domain habitat — and <code>L9/13344</code> ("comparison between two things") changes "air" to "sky." These failures show that Phase 2 features contribute to relational mapping but are only individually critical when the prediction has minimal redundancy.</p>

<p><strong>Phase 3 results.</strong> <code>L13/10969</code> ("comparisons between disciplines") is not individually necessary for any circuit.</p>

<h4>3.7.4 Collective Phase Suppression</h4>

<p>The individual redundancy of Phase 2 features for high-confidence circuits raises the question: are they <em>collectively</em> necessary? We suppressed all features within each phase simultaneously, then tested combined Phase 1+2 and Phase 1+2+3 suppression.</p>

<table>>
<thead><tr><th>Experiment</th><th>Features Suppressed</th><th>Berlin</th><th>Rome</th><th>Tokyo</th><th>Teacher</th><th>Bird</th></tr></thead>
<tbody>
<tr style="font-weight:700">
<td>All Phase 2 (4 feat.)</td>
<td>L5/5793, L5/2141, L8/13766, L9/13344</td>
<td>France</td><td>France</td><td>France</td><td>be</td><td>fish</td>
</tr>
<tr style="font-weight:700;">
<td>All Phase 1 (5 feat.)</td>
<td>L0/11651, L1/11356, L4/10752, L5/9672 + L2/11475</td>
<td>(empty)</td><td>(empty)</td><td>(empty)</td><td>to</td><td>to</td>
</tr>
<tr style="font-weight:700;">
<td>Phase 1+2 (9 feat.)</td>
<td>All Phase 1 + Phase 2</td>
<td>:</td><td>:</td><td>:</td><td>:</td><td>:</td>
</tr>
<tr style="font-weight:700;">
<td>Phase 1+2+3 (10 feat.)</td>
<td>All Phase 1 + Phase 2 + L13/10969</td>
<td>:</td><td>:</td><td>:</td><td>be</td><td>:</td>
</tr>
</tbody>
</table>
<p class="figure-caption">Every cell is disrupted — all four experiments disrupt all 5 circuits (20/20).</p>

<div class="finding-box">
<p><strong>Phase 2 collective suppression is the most informative experiment in this paper.</strong> Suppressing all four Phase 2 features simultaneously disrupts all 5 circuits — including the high-confidence capital analogies (p>0.97) that were immune to <em>every</em> individual Phase 2 suppression. The failure mode is striking and consistent: all three capital analogies predict <strong>"France"</strong> — the <em>source</em> country — instead of the target country. The model retains the factual association "Paris is to France" and completes with "France" because it can no longer compute the relational transfer "as X is to ___." It echoes the source pair's answer rather than mapping to the target pair. This is direct causal evidence that Phase 2 features collectively implement the relational transfer operation — the core of analogical reasoning — rather than merely detecting the analogy template.</p>

<p><strong>Phase 1 collective suppression</strong> produces empty outputs for all three capital analogies — the model cannot even generate a coherent token. For the semantic role analogies (teacher, bird), it produces "to" — the model is stuck on the syntactic connective it can no longer parse. This is a more severe failure than Phase 2 suppression, consistent with Phase 1 (template parsing) being a prerequisite for Phase 2 (analogy recognition).</p>

<p><strong>Combined Phase 1+2 suppression</strong> produces ":" for 4/5 circuits. This punctuation output mirrors the format-override failure documented in the linguistic analysis (§4.5 of the expanded paper), where the model's format-completion circuit outcompetes the semantic-retrieval circuit. With both template parsing and analogy recognition disabled, the prompt resembles a list header ("Paris is to France as Berlin is to:") and the model defaults to list-formatting punctuation.</p>

<p>These results establish a <strong>causal hierarchy</strong>: Phase 1 (template parsing) → Phase 2 (relational transfer) → Phase 3 + late layers (entity selection). Each phase is collectively necessary, and earlier phases are prerequisites for later ones.</p>
</div>

<h4>3.7.5 Sufficiency (Hub Boost on Altered Prompts)</h4>

<p>Necessity tests establish which features are required for the prediction. Sufficiency tests ask the converse: can a single backbone feature <em>induce</em> the target prediction in a new context? For each circuit, we boosted the primary backward-path hub at strength +20 on two altered prompts — one retaining the target entity and one crossing domain boundaries.</p>

<table>>
<thead><tr><th>Circuit</th><th>Hub Boosted</th><th>Altered Prompt</th><th>Baseline → Boosted</th><th>Induced?</th></tr></thead>
<tbody>
<tr><td><code>analog_berlin</code></td><td>L21/4827</td><td>"Cairo is to Egypt as Nairobi is to"</td><td>Kenya → Kenya</td><td>no</td></tr>
<tr style="font-weight:700"><td><code>analog_berlin</code></td><td>L21/4827</td><td>"Madrid is to Spain as Berlin is to"</td><td>→ Germany</td><td>YES</td></tr>
<tr><td><code>analog_rome</code></td><td>L20/15360</td><td>"Paris is to France as Tokyo is to"</td><td>→ Japan</td><td>no</td></tr>
<tr style="font-weight:700"><td><code>analog_rome</code></td><td>L20/15360</td><td>"Madrid is to Spain as Rome is to"</td><td>→ Italy</td><td>YES</td></tr>
<tr><td><code>analog_tokyo</code></td><td>L20/15360</td><td>"Paris is to France as Rome is to"</td><td>→ Italy</td><td>no</td></tr>
<tr style="font-weight:700"><td><code>analog_tokyo</code></td><td>L20/15360</td><td>"Beijing is to China as Tokyo is to"</td><td>→ Japan</td><td>YES</td></tr>
<tr><td><code>analog_teacher</code></td><td>L0/17</td><td>"Nurse is to hospital as teacher is to"</td><td>→ be</td><td>no</td></tr>
<tr><td><code>analog_teacher</code></td><td>L0/17</td><td>"Doctor is to hospital as chef is to"</td><td>→ hospital</td><td>no</td></tr>
<tr><td><code>analog_bird</code></td><td>L22/4252</td><td>"Cat is to land as bird is to"</td><td>→ (newline)</td><td>no</td></tr>
<tr style="font-weight:700"><td><code>analog_bird</code></td><td>L22/4252</td><td>"Fish is to water as eagle is to"</td><td>→ air</td><td>YES</td></tr>
<tr style="font-weight:700"><td>Cairo→Kenya</td><td>L15/15954</td><td>"Lagos is to Nigeria as Nairobi is to"</td><td>→ Kenya</td><td>YES</td></tr>
</tbody>
</table>

<p><strong>5/11 tests succeed.</strong> The pattern is consistent: sufficiency succeeds when the altered prompt retains the target entity or a semantically close substitute, and fails when it crosses domain boundaries. Specifically:</p>
<ul>
<li><strong>Capital analogies:</strong> Each hub induces the correct country when the target city appears in the altered prompt (Berlin→Germany, Rome→Italy, Tokyo→Japan). But the Berlin hub cannot inject "Germany" into the Cairo/Nairobi context, and the Rome hub cannot override Tokyo→Japan. These hubs encode <em>domain-specific geographic associations</em> rather than general-purpose "answer slot" activators.</li>
<li><strong>Teacher:</strong> Boosting L0/17 on a near-paraphrase ("Nurse is to hospital as teacher is to") produces "be" rather than "school." The L0 embedding feature does not carry enough domain information to override the altered context — it is necessary (§3.7.1) but not sufficient.</li>
<li><strong>Bird:</strong> Boosting L22/4252 on "Fish is to water as <em>eagle</em> is to" successfully produces "air." Substituting "eagle" for "bird" retains the target habitat domain, and the hub carries enough signal to resolve "air" even with the entity change. But "Cat is to land as bird is to" fails — the source-pair domain (land mammal) is too distant from the hub's encoded association.</li>
</ul>

<h4>3.7.6 Specificity (Non-Backbone Feature Suppression)</h4>

<p>Specificity tests confirm that the causal path tracing correctly identifies load-bearing features by testing whether high-activation features <em>outside</em> the backbone affect the prediction. For each circuit, we suppressed 2 non-backbone features — nodes with high activation or influence scores that do not appear on any of the top-10 causal paths.</p>

<table>>
<thead><tr><th>Circuit</th><th>Feature</th><th>Neuronpedia Label</th><th>Steered Token</th><th>Disrupted?</th></tr></thead>
<tbody>
<tr><td><code>analog_berlin</code></td><td>L6/3335</td><td>"difficulty/challenges"</td><td>Germany</td><td>no</td></tr>
<tr><td><code>analog_berlin</code></td><td>L13/4435</td><td>"opera-related terms"</td><td>Germany</td><td>no</td></tr>
<tr><td><code>analog_rome</code></td><td>L6/2267</td><td>"formal text/code" (§3.4)</td><td>Italy</td><td>no</td></tr>
<tr><td><code>analog_rome</code></td><td>L4/14857</td><td>"code snippets" (§3.4)</td><td>Italy</td><td>no</td></tr>
<tr><td><code>analog_tokyo</code></td><td>L6/2267</td><td>"formal text/code" (§3.4)</td><td>Japan</td><td>no</td></tr>
<tr><td><code>analog_tokyo</code></td><td>L3/10018</td><td>early structural feature</td><td>Japan</td><td>no</td></tr>
<tr><td><code>analog_teacher</code></td><td>L4/14857</td><td>"code snippets" (§3.4)</td><td>school</td><td>no</td></tr>
<tr><td><code>analog_teacher</code></td><td>L8/13766</td><td>"analogies or comparisons" (§3.3)</td><td>school</td><td>no</td></tr>
<tr style="font-weight:700;background:var(--smoking-bg);"><td><code>analog_bird</code></td><td>L6/2267</td><td>"formal text/code" (§3.4)</td><td>sky</td><td>YES</td></tr>
<tr><td><code>analog_bird</code></td><td>L5/5793</td><td>"analogies" (§3.3)</td><td>air</td><td>no</td></tr>
<tr><td>Cairo→Kenya</td><td>L5/5500</td><td>"profanity and comparisons"</td><td>Kenya</td><td>no</td></tr>
<tr><td>Cairo→Kenya</td><td>L0/8</td><td>"are/research"</td><td>Kenya</td><td>no</td></tr>
<tr><td>Puppy→cat</td><td>L9/2909</td><td>"formulas/ratios"</td><td>cat</td><td>no</td></tr>
</tbody>
</table>

<p><strong>12/13 pass specificity.</strong> The sole exception is instructive: suppressing <code>L6/2267</code> ("words in programming code, legal jargon, or scientific texts") in the <code>analog_bird</code> circuit changes "air" to "sky." Both are semantically valid completions — the model's choice between "air" (p=0.117) and "sky" (likely the next-ranked competitor) is barely resolved, and this formal-text feature tips the balance. Critically, <code>L6/2267</code> is confirmed <strong>causally inert for all high-confidence circuits</strong> — it does not affect Berlin, Rome, Tokyo, or Cairo. This supports the interpretation from §4.2: formal-text features process template syntax in Phase 1 and do not drive the relational completion in Phase 3, except at the extreme margin where token competition is unresolved.</p>

<p>The <code>L5/5793</code> ("analogies") feature — the paper's "smoking gun" analogy-concept feature — passes specificity for the bird circuit (prediction unchanged at "air"), consistent with the individual necessity result (§3.7.3) where it is never necessary. This feature is part of the Phase 2 collective that is necessary when suppressed together (§3.7.4) but is individually dispensable in all tested contexts.</p>

<h4>3.7.7 Summary of Causal Validation</h4>

<table>>
<thead><tr><th>Circuit</th><th>p</th><th>Type</th><th>Individual Necessity</th><th>Full Suppress</th><th>Phase 2 Collective</th><th>Sufficiency</th><th>Specificity</th></tr></thead>
<tbody>
<tr><td><code>analog_berlin</code></td><td>0.973</td><td>Capital</td><td>1/9</td><td>DISRUPTED</td><td>→ France</td><td>1/2</td><td>PASS (0/2)</td></tr>
<tr><td><code>analog_rome</code></td><td>0.974</td><td>Capital</td><td>4/10</td><td>DISRUPTED</td><td>→ France</td><td>1/2</td><td>PASS (0/2)</td></tr>
<tr><td><code>analog_tokyo</code></td><td>0.990</td><td>Capital</td><td>3/10</td><td>DISRUPTED</td><td>→ France</td><td>1/2</td><td>PASS (0/2)</td></tr>
<tr><td><code>analog_teacher</code></td><td>0.486</td><td>Sem. role</td><td>3/10</td><td>DISRUPTED</td><td>→ be</td><td>0/2</td><td>PASS (0/2)</td></tr>
<tr><td><code>analog_bird</code></td><td>0.117</td><td>Sem. role</td><td>8/10</td><td>DISRUPTED</td><td>→ fish</td><td>1/2</td><td>1/2 (air→sky)</td></tr>
<tr><td>Cairo→Kenya</td><td>0.963</td><td>Capital</td><td>2/9</td><td>DISRUPTED</td><td>—</td><td>1/2</td><td>PASS (0/2)</td></tr>
<tr><td>Puppy→cat</td><td>0.756</td><td>Sem. role</td><td>0/4</td><td>intact</td><td>—</td><td>—</td><td>PASS (0/1)</td></tr>
</tbody>
</table>

<p>Across 159 steering experiments on 7 circuits, the causal validation establishes five principal findings:</p>
<ol>
<li><strong>The late-layer backbone is collectively necessary.</strong> Full backbone suppression disrupts 6/7 circuits across both geographic and semantic role analogies, confirming that the features identified by causal path tracing drive the prediction.</li>
<li><strong>Individual necessity scales with circuit fragility.</strong> High-confidence circuits have 1–4 necessary features; the lowest-confidence circuit has 8/10. Redundancy through parallel causal paths is a property of well-learned associations.</li>
<li><strong>Phase 2 is collectively necessary but individually redundant.</strong> No individual Phase 2 feature disrupts any high-confidence circuit, but simultaneous suppression of all four Phase 2 features disrupts every circuit — capital analogies revert to the source-pair answer ("France"), demonstrating that Phase 2 implements relational transfer.</li>
<li><strong>Phase 1 template features are individually necessary.</strong> <code>L0/11651</code> ("the word 'to'") alone disrupts 4/5 circuits, with capital analogies repeating the city name. Template parsing at the earliest layers is a prerequisite for relational reasoning at later layers.</li>
<li><strong>Formal-text features are causally inert.</strong> The high-recurrence features at L4–L6 (§3.4) — <code>L4/14857</code> "code snippets", <code>L6/2267</code> "formal text/code" — do not affect any high-confidence prediction when suppressed, confirming they process template syntax rather than driving entity selection. The sole exception (bird: air→sky at p=0.117) occurs at the margin of unresolved token competition.</li>
</ol>



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
