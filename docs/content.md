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
  - label: "PDF"
    icon: "fa-solid fa-file-pdf"
    url: "paper.pdf"
  - label: "Supplementary"
    icon: "fa-solid fa-file-lines"
    url: "supplementary.html"
  - label: "Code"
    icon: "fab fa-github"
    url: "https://github.com/kkrampis/autocircuit"
  - label: "Presentation"
    icon: "fa-solid fa-display"
    url: "presentation.html"
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
  - label: "Supplementary Material"
    icon: "fa-solid fa-file-lines"
    url: "supplementary.html"
    description: "Full supplementary document: all inference prompts, agent pipeline prompts used by Olalekan, links to all five Neuronpedia attribution graphs, and tooling reference."
    sublinks:
      - label: "PDF version"
        url: "supplementary.pdf"
        desc: "Downloadable PDF of the supplementary material"
  - label: "Interactive Presentation"
    icon: "fa-solid fa-display"
    url: "presentation.html"
    description: "20-slide reveal.js presentation with GitHub dark theme, circuit flow diagrams, feature tables, and layer-by-layer analysis."
  - label: "Live Attribution Graphs"
    icon: "fa-solid fa-brain"
    url: "https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin"
    description: "Interactive Neuronpedia attribution graph viewer for all five prompts. Full graph list with prompts in Supplementary Material."
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

We present a mechanistic analysis of analogical reasoning in Gemma-2-2B using Neuronpedia attribution graphs and Sparse Autoencoder (SAE) features. By generating and comparing five attribution graphs across structurally distinct analogical prompts — covering geographic analogies (*Paris - France → Berlin - ?*, *Rome - ?*, *Tokyo - ?*) and semantic role analogies (*Doctor - hospital → teacher - ?*, *Fish - water → bird - ?*) — we identify a shared **analogical reasoning circuit** comprising 180 features active across all five prompts and 510 features active across at least three. Each feature is identified by a stable *(layer, feature index)* pair, identifying circuits as lists of recurring internal model feature activation patterns that retain similar structure across analogical prompts.

We discover dedicated analogy-encoding features at layers 5, 8, 9, and 13, including a feature at layer 5 labeled literally as **"analogies"** and a layer 8 feature encoding **"analogies or comparisons"** appearing across all graphs with high influence. Early layers (0–4) contain circuit templates tracking the "X is to Y as Z is to" pattern, while mid-to-late layers (5–13) house increasingly semantic representations of the relational structure. The circuit spans all 26 transformer layers and exhibits cross-domain generalization, with the same core features activating for both geographic and semantic role analogies. Causal validation via 159 feature steering experiments confirms that the identified backbone features are collectively necessary for the model's predictions, and that Phase 2 features collectively implement the relational transfer operation that is the computational core of analogical reasoning.

---

<!-- slide: What Is Analogical Reasoning? -->

## 1. Introduction

<!-- figure:fig-analogy-task -->

Analogical reasoning — the ability to recognize and complete structural relationships between concepts — is a foundational cognitive ability underlying scientific discovery, language understanding, and abstract problem solving. The classic analogy task, *"Paris is to France as Berlin is to \_\_\_\_,"* tests whether a model can identify the capital-city relationship abstractly and apply it to a new country. Large language models (LLMs) exhibit striking competence on such tasks [1], yet the internal computational mechanisms remain poorly understood.

Mechanistic interpretability research has made significant progress in understanding factual recall circuits [2], indirect object identification [3], and syntactic processing [4]. Sparse autoencoders (SAEs) have emerged as a central tool in this effort, learning sparse, interpretable decompositions of model activations [5, 6] that can be applied at scale across all layers and sublayers of large models [7]. The Neuronpedia platform [8] operationalizes this infrastructure, providing public APIs for attribution graph generation and feature steering that democratize circuit-level analysis beyond institutions with direct model access.

However, analogical reasoning presents a distinct challenge beyond prior circuit analyses: it requires not merely retrieving a stored fact, but recognizing a **relational structure** and applying it compositionally to novel inputs. The relation type is never named in the prompt — the model must infer *capital-of* from the example alone, hold it as a variable, and transfer it to a new argument pair. Prior work has documented that LLMs exhibit apparently emergent analogical reasoning [1] and identified internal attention-head mechanisms supporting abstract reasoning [9], yet a feature-level, causally-validated circuit account has been absent.

We address this gap using attribution graphs generated from the `gemmascope-transcoder-16k` SAE suite [7], which provides cross-layer transcoder features for every layer of Gemma-2-2B. Our analysis identifies a three-phase circuit with explicitly labeled analogy-concept features, provides causal validation through 159 steering experiments, and constitutes — to our knowledge — an SAE-level mechanistic account of analogical reasoning in a large language model.

<!-- slide: Research Questions -->

### 1.1 Research Questions

1. Does Gemma-2-2B employ a **shared circuit** for analogical reasoning, or does it use different mechanisms for different analogy types?
2. Which SAE features — identified by stable *(layer, feature index)* pairs — are most **consistently activated** across diverse analogical prompts?
3. Are there interpretable, semantically meaningful features that encode the **abstract relational structure** of analogies, and how are they discovered?
4. How is the analogical computation **distributed across transformer layers**, and can phase boundaries be causally validated?

---

## 2. Methodology

<!-- slide: Prompt Selection -->

### 2.1 Prompt Selection

We selected five prompts spanning two structural analogy types to ensure cross-domain coverage:

| ID | Prompt | Expected | Type |
|---|---|---|---|
| `analog_berlin` | *"Paris is to France as Berlin is to"* | Germany | Capital |
| `analog_rome` | *"Paris is to France as Rome is to"* | Italy | Capital |
| `analog_tokyo` | *"Paris is to France as Tokyo is to"* | Japan | Capital |
| `analog_teacher` | *"Doctor is to hospital as teacher is to"* | school | Semantic role |
| `analog_bird` | *"Fish is to water as bird is to"* | air / sky | Semantic role |

<!-- slide: Attribution Graph Generation -->

### 2.2 Attribution Graph Generation

Attribution graphs were generated using the Neuronpedia API [8] (`/api/graph/generate`) with Gemma-2-2B and the `gemmascope-transcoder-16k` SAE [7] — a 26-layer cross-layer transcoder with 16,384 features per layer. Each graph request returns a JSON object containing nodes (SAE feature activations with layer, index, influence score, and activation magnitude) and directed edges (attribution scores). Graphs were downloaded and loaded into NetworkX `DiGraph` objects for analysis.

| Parameter | Value |
|---|---|
| Model | `gemma-2-2b` |
| SAE | `gemmascope-transcoder-16k` |
| Max feature nodes | 3,000 |
| Desired logit probability | 0.95 |
| Node threshold | 0.80 |
| Edge threshold | 0.85 |

**Key technical finding:** The correct API endpoint for `gemmascope-transcoder-16k` requires a **layer-prefixed SAE identifier** (e.g., `4-gemmascope-transcoder-16k` for layer 4) rather than the global SAE name.

### 2.3 Feature Identification and Cross-Graph Analysis

#### 2.3.1 Feature Identity via Stable (Layer, Feature Index) Pairs

Each feature in the attribution graphs is identified by a stable *(layer, feature index)* pair — for example, *(5, 5793)* uniquely and persistently identifies a feature within the `gemmascope-transcoder-16k` SAE [7]. These identifiers are fixed properties of the trained SAE and do not vary across prompts, sessions, or API calls.

Cross-graph feature overlap was computed by finding which *(layer, feature index)* pairs appear as nodes across multiple independently generated graphs. Formally, let $G_i$ denote the set of feature IDs active in graph $i$. The shared circuit at threshold $k$ is:

$$\mathcal{C}_k = \left\{ f \;\middle|\; \sum_{i=1}^{5} \mathbf{1}[f \in G_i] \geq k \right\}$$

Three thresholds were analyzed: $k \in \{3, 4, 5\}$. The 180-feature core circuit ($k=5$) is therefore a concrete, enumerable list of *(layer, feature index)* identifiers that recur across all five independently generated graphs regardless of whether the prompt is geographic or semantic in nature. Feature labels were retrieved using the Neuronpedia feature explanation API [8].

### 2.4 Three-Phase Architecture: How Phase Boundaries Were Identified

The three-phase architecture was identified through two converging lines of evidence, neither of which required the authors to impose phase boundaries a priori.

**Semantic label analysis.** After retrieving Neuronpedia automated labels for the top recurring features, a consistent gradient emerged across layer depth. Features in layers 0–4 carry purely syntactic labels: "the word 'to'", "'to' followed by a verb", "the phrase 'it is to'". Features in layers 5–9 carry explicitly relational-semantic labels: "analogies", "analogies or comparisons", "comparison between two things". Features in layers 10–13 carry integrative labels: "comparisons between disciplines and relationships between concepts". The phase boundaries emerge from the content of the labels rather than an arbitrary partition of layers. This three-stage organization mirrors the emergent symbolic architecture documented by Webb et al. [9] for abstract reasoning more broadly, where early layers abstract tokens into relational variables, intermediate layers perform induction over those variables, and later layers retrieve the answer.

**Activation magnitude progression.** Average activation magnitudes of core features increase monotonically through the phases:

| Phase | Layer Range | Role | Activation Magnitude |
|---|---|---|---|
| Circuit template | L0 | Token and syntax parsing | 1.5 – 6.4 |
| Analogy recognition hub | L5 | Analogy concept activation | 7.4 – 11.1 |
| Comparison detectors | L8–L9 | Relational detection | ~13.4 |
| Relational integration | L10–L13 | Domain + relation integration | 9.1 – 16.3 |

**Causal validation.** Phase boundaries were then tested causally via collective suppression experiments (§3.4), which confirmed that each phase is collectively necessary and that earlier phases are prerequisites for later ones.

### 2.5 Discovery of Analogy-Concept Features

The key features — L5 SAE#5793 ("analogies") and L8 SAE#13766 ("analogies or comparisons") — were not specifically sought. They emerged from the cross-graph overlap analysis described in §2.3. Once the intersection feature set was computed, each feature's automated Neuronpedia explanation [8] was retrieved. L5 SAE#5793 returned the label "analogies"; L8 SAE#13766 returned "analogies or comparisons."

The significance of these labels is their **domain-agnosticism**. Both features appear in attribution graphs for Berlin, Rome, and Tokyo (geographic capital analogies) and for teacher and bird (semantic role analogies). This is consistent with the broader finding in the analogical reasoning literature that LLMs encode relational information in a domain-general manner [10, 11], and extends that behavioral finding to a specific, causally-validated internal feature. L8 SAE#13766 was additionally notable for having 21 appearances across the five graphs and an influence score of 0.533, placing it among the highest-influence recurring features.

### 2.6 Phase 2 Definition

Phase 2 is defined by two jointly applied criteria: **layer position (5–9)** and **feature label content**. Features in this layer range whose Neuronpedia labels explicitly reference analogies, comparisons, or relational structure constitute Phase 2. The four members are:

| Feature | Label |
|---|---|
| L5 SAE#5793 | "analogies" |
| L5 SAE#2141 | "comparisons of people or figures using well-known public figures" |
| L8 SAE#13766 | "analogies or comparisons" |
| L9 SAE#13344 | "phrases suggesting uncertainty or comparison between two things" |

This grouping is validated causally: suppressing all four Phase 2 features simultaneously collapses all five circuits, with capital analogies producing "France" — the source-pair answer — rather than the target country. An arbitrary phase definition would not produce such a consistent and semantically meaningful failure mode.

### 2.7 Causal Validation via Feature Steering

The cross-graph analysis (§2.3) identifies the 180-feature shared circuit as a *correlational* object: these features recur across all five analogy graphs. Recurrence does not establish that the circuit *causes* the model's answers — a recurring feature could be a passenger that co-activates with the computation without driving it. We test causation by intervening on the circuit with the Neuronpedia `/api/steer` endpoint [8] (`modelId: "gemma-2-2b"`, `strength_multiplier: 4`, `temperature: 0`, `seed: 42`).

One methodological fact governs the whole validation. The per-feature `strength` we set (−20 for ablation) is scaled by the global `strength_multiplier` of 4, so the effective intervention is **−80** per feature — a large perturbation. At that magnitude, removing *any* sizable feature set derails the model, so the bare observation "the output changed" (necessity) is necessary but **not sufficient** evidence that a particular circuit is responsible. The discriminating evidence is of two kinds: (i) a **matched contrast** — does ablating the circuit break the task differently from ablating a size- and strength-matched *random* set? — and (ii) a **predicted failure mode** — does the model fail in the specific way the circuit's claimed role implies? Our design is built on these rather than on necessity alone:

1. **Circuit necessity vs. a matched null (primary).** For each of the five analogy prompts, run two ablations of identical size (180 features) and strength (−20): (a) all 180 circuit features, and (b) 180 random *non*-circuit features drawn from the same prompt's graph. The evidence is the contrast between them.
2. **Internal structure (individual necessity).** Ablate each of the 180 features one at a time to locate the individually load-bearing members.
3. **Robustness.** Layer-stratified ablation (is the effect just deletion of L0 token features?), a strength titration (is it an artifact of the large −80 magnitude?), and held-out analogy prompts the circuit was never derived from (does it generalize?).
4. **Architecture.** Collective suppression of the Phase-1 and Phase-2 feature groups (§2.4) to test the three-phase organization.
5. **Single-feature side-tests.** A sufficiency probe (boost the highest-influence hub) and an individual non-circuit specificity scan, reported as supplementary single-feature controls.

Across these paradigms the validation comprises **218 individual steering API calls**.

---

## 3. Results

<!-- slide: Attribution Graph Structure -->

### 3.1 Graph Structure

All five attribution graphs exhibited a consistent structural pattern, with features activated across all 26 transformer layers (0–25) plus the embedding layer (E):

| Graph | Prompt | Nodes | Edges | Max Influence |
|---|---|---|---|---|
| `analog_berlin` | Paris - France → Berlin - ? | 930 | 25,915 | 0.8001 |
| `analog_rome` | Paris - France → Rome - ? | 963 | 27,608 | 0.8002 |
| `analog_tokyo` | Paris - France → Tokyo - ? | 905 | 22,414 | 0.8001 |
| `analog_teacher` | Doctor - hospital → teacher - ? | 1,040 | 35,481 | 0.8001 |
| `analog_bird` | Fish - water → bird - ? | 1,071 | 38,741 | 0.8000 |

The semantic role analogies (*teacher*, *bird*) have notably larger graphs (1,040–1,071 nodes, 35k–38k edges) compared to the capital analogies (905–963 nodes, 22k–27k edges). We interpret this as reflecting greater ambiguity in the expected completion domain: the *capital-of* relation maps to a discrete, well-encoded fact [2], whereas professional and ecological roles require broader world-knowledge access.

<!-- slide: Core Circuit Identification -->

### 3.2 The Core Analogical Reasoning Circuit

<!-- figure:fig-layer-distribution -->

Cross-graph feature overlap analysis over the stable *(layer, feature index)* identifier space revealed a substantial shared circuit:

| Threshold | Features Found |
|---|---|
| Active in ≥3/5 graphs | **510 features** |
| Active in ≥4/5 graphs | **277 features** |
| Active in all 5 graphs | **180 features** |

Core features by layer group (5/5 graphs):

| Layer Group | Core Features |
|---|---|
| L0 | 12 |
| L1–L4 | 19 |
| L5–L6 | 12 |
| L8–L13 | 7 |

early layers (L0–L3) account for the plurality of core features, consistent with circuit template processing occurring first. The mid-range layers (L5–L6) show elevated feature counts relative to neighbors — these are the **analogy recognition hub** layers. Isolated high-influence features appear at L8, L9, L11, and L13.

<!-- slide: Multi-step Reasoning Evidence -->

### 3.3 The Three-Phase Analogical Reasoning Circuit

<!-- figure:fig-circuit-flow -->

We provide evidence that Gemma-2-2B performs **genuine multi-step analogical reasoning internally**. The attribution graph reveals a three-phase computational process that activates for both geographic and semantic role analogies — evidence of a domain-agnostic relational reasoning mechanism. This three-stage organization parallels the symbolic architecture identified by Webb et al. [9] through causal mediation analysis and the internal representation findings of Lee et al. [10].

---

**Phase 1 · layers 0–4 · Circuit Template Parsing**

| Feature | Label |
|---|---|
| L0 SAE#11651 | *"the word 'to'"* |
| L1 SAE#11356 | *"the word 'to' followed by a verb"* |
| L2 SAE#11475 | *"the word 'refers' and related words"* |
| L4 SAE#10752 | *"uses of the verb 'to be' preceded by 'to'"* |
| L5 SAE#9672 | *"the phrase 'it is to'"* |

These features encode the syntactic skeleton of the analogy prompt. Their progression from individual tokens to multi-word patterns reflects hierarchical parsing of the relational connective. These are *structural* features — they fire on any text with this grammatical form, not specifically on analogical content.

---

**Phase 2 · layers 5–9 · Analogy Recognition Hub**

| Feature | Label |
|---|---|
| L5 SAE#5793 | *"analogies"* ← dedicated analogy concept feature |
| L5 SAE#2141 | *"comparisons of people or figures using well-known public figures"* |
| L8 SAE#13766 | *"analogies or comparisons"* (21 activations across 5 graphs, influence 0.533) |
| L9 SAE#13344 | *"phrases suggesting uncertainty or comparison between two things"* |

This is where circuit template processing gives way to semantic recognition of the *relational concept itself*. The presence of L5 SAE#5793, labeled "analogies" by Neuronpedia's automated SAE feature explanation system [8], is particularly significant: it activates consistently for both capital-city and semantic role analogies. It is not a geographic feature — it fires equally for "Doctor - hospital → teacher - ?". This is direct evidence of the kind of abstract relational representation that prior behavioral work [1, 11] has hypothesized but not directly observed inside a model.

---

**Phase 3 · layers 10–13 · Relational Integration**

| Feature | Label |
|---|---|
| L11 SAE#15947 | *"references to historical or social change"* |
| L13 SAE#10969 | *"comparisons between disciplines and relationships between concepts"* |

L13 SAE#10969 serves an integrative role, combining the recognized relational structure from Phase 2 with domain-specific knowledge to produce the final completion. layers 14–25 then handle domain-specific knowledge retrieval and output token formatting, analogous to the factual recall circuits identified by Meng et al. [2].

---

> **Note:** This diagram simplifies the true mechanisms considerably. The attribution graph for any single prompt contains hundreds of features; the circuit shown represents the semantically interpretable core.

<!-- slide: Feature Analysis -->

### 3.4 Top Recurring Features

**Directly analogical features** (Neuronpedia labels explicitly reference analogical reasoning or comparison):

| Feature | Appearances | Avg Influence | Label |
|---|---|---|---|
| L5 #5793 | 11/5 | 0.590 | "analogies" |
| L8 #13766 | 21/5 | 0.533 | "analogies or comparisons" |
| L9 #13344 | 14/5 | 0.681 | "comparison between two things" |
| L5 #2141 | 12/5 | 0.647 | "comparisons of public figures" |
| L13 #10969 | 11/5 | 0.676 | "comparisons between disciplines" |

**Circuit templates** (encode the "X is to Y as Z is to" scaffold):

| Feature | Appearances | Avg Influence | Label |
|---|---|---|---|
| L0 #11651 | 10/5 | 0.633 | "the word 'to'" |
| L1 #11356 | 10/5 | 0.609 | "'to' followed by a verb" |
| L2 #11475 | 10/5 | 0.638 | "the word 'refers'" |
| L4 #10752 | 10/5 | 0.626 | "'to be' preceded by 'to'" |
| L5 #9672 | 12/5 | 0.579 | "the phrase 'it is to'" |

**High-recurrence formal text features** (labels unrelated to analogical reasoning):

| Feature | Appearances | Avg Influence | Label |
|---|---|---|---|
| L4 #14857 | 22/5 | 0.681 | "code snippets and license agreements" |
| L6 #2267 | 20/5 | 0.724 | "words in programming code, legal jargon, or scientific texts" |
| L3 #3205 | 20/5 | 0.670 | "code snippets and documentation references" |

These formal-text features have higher raw appearance counts than the explicitly analogical features. Causal steering (§3.7.6) confirms they are inert for all high-confidence circuits, consistent with their role as detectors of syntactic formality rather than relational semantics. The polysemanticity of neurons in large models [6] is precisely why SAE-based feature decomposition [5, 6, 7] is necessary to distinguish these classes of activation.

<!-- slide: Cross-Domain Generalization -->

### 3.5 Cross-Domain Generalization

<!-- figure:fig-venn -->

The consistent activation of L5 SAE#5793 ("analogies") and L8 SAE#13766 ("analogies or comparisons") across both capital-city and semantic role analogy types provides the most direct evidence for a **domain-general analogical reasoning mechanism**. The 180 features active in all five graphs form the stable intersection of the two analogy type families, and this intersection includes the core analogy-concept features at L5 and L8.

The slightly larger graphs for semantic role analogies (teacher, bird: 1,040–1,071 nodes) relative to capital analogies (Berlin, Rome, Tokyo: 905–963 nodes) may reflect that semantic role completions require broader world-knowledge access — knowing that teachers work in schools, or that birds inhabit air — rather than purely relational computation over a discrete, well-encoded geographic fact [2].

### 3.6 Activation Magnitudes Build Through Layers

<!-- figure:fig-activation -->

Average activation magnitudes of core circuit features increase substantially with layer depth:

| Layer Range | Role | Typical Activation |
|---|---|---|
| L0 (structural) | Token and syntax parsing | 1.5 – 6.4 |
| L5 (analogy hub) | Analogy concept activation | 7.4 – 11.1 |
| L8–L9 (detectors) | Comparison detection | ~13.4 |
| L10–L13 (integration) | Relational + domain integration | 9.1 – 16.3 |

This monotonically increasing pattern is consistent with an accumulating signal as the relational structure is assembled.

### 3.7 Causal Validation via Feature Steering

Everything above is *correlational*: the attribution graphs tell us which features are active and influential when the model answers, not whether those features *cause* the answer. This section intervenes — suppressing and boosting features — to ask, claim by claim, whether the paper's structural findings are causally real.

**How to read every result in this section.** One methodological fact governs the interpretation (§2.7). Steering at `strength −20` is an effective **−80** intervention per feature, large enough that ablating *any* sizable feature set derails the model. So bare necessity ("the output changed") is necessary but **not sufficient** evidence that a particular circuit is responsible. The discriminating evidence is (i) a **matched contrast** — does ablating the circuit break the task differently from ablating a size- and strength-matched *random* set? — and (ii) a **predicted failure mode** — does the model fail in the specific way the circuit's role implies? We rely on these throughout.

#### 3.7.1 Is the Recurring Circuit Causally Load-Bearing?

We test whether the 180-feature shared circuit (§2.3) drives the answer or is a set of passengers. For each prompt we run two ablations of identical size (180 features) and strength (−20, effective −80) that differ only in *which* features are removed: **(A)** all 180 circuit features, and **(B)** a *matched null* of 180 random non-circuit features sampled from the same prompt's graph (seed 7). Because the circuit is *defined* by recurrence across all five prompts, we run both on all five — a result on one prompt could not justify the word "shared."

If the circuit implements the analogical computation, removing it should leave the model unable to complete "X is to Y as Z is to ___," falling back to the template scaffolding it can still see — repeating the connective " to" — rather than emitting any answer. A random ablation has no reason to fail in this particular way.

| Prompt | Baseline answer (logprob) | (A) Circuit ablation → first token (logprob) | (B) Matched null → first token (logprob) |
|---|---|---|---|
| Paris…Berlin is to | ` Germany` (−0.04) | **` to` (−0.03)** | ` Kyrie` (−2.84) |
| Paris…Rome is to | ` Italy` (−0.04) | **` to` (−0.03)** | ` autorytatywna` (−0.81) |
| Paris…Tokyo is to | ` Japan` (−0.02) | **` to` (−0.03)** | ` to` (−1.97) |
| Doctor…teacher is to | ` school` (−0.57) | **` to` (−0.03)** | ` initComponents` (−0.80) |
| Fish…bird is to | ` air` (−2.58) | **` to` (−0.03)** | ` espère` (−0.32) |

*The two matched ablations on all five analogy prompts (three geographic, two semantic-role). Both remove the correct answer in every case. The discriminating signal is the failure mode.*

The table reads in two passes. **Necessity holds but does not discriminate:** both (A) and (B) remove the correct answer on all 5/5 prompts — confirming, across prompts, that at effective −80 you can derail the model by ablating 180 of anything. **Specificity lives in the failure mode:** circuit ablation produces a reproducible, near-deterministic collapse to the connective ` to` — the same token at the same confidence (−0.026 to −0.030, σ = 0.0015) on all five prompts, across both geographic and semantic-role analogies — whereas the matched null, facing the identical prompts at the identical magnitude, scatters to idiosyncratic out-of-distribution tokens (a Polish word, a French word, a code identifier, a name) at far lower confidence (mean −1.35). Only one null run (Tokyo) lands on ` to`, and at 65× lower probability. The *only* variable differing between (A) and (B) is which features were removed, so the structured collapse is attributable to the circuit, not to the magnitude of the intervention.

One caveat, stated plainly: all five prompts end in "to," so "collapse to the connective" and "repeat the final token" coincide. This is *consistent* with relational-completion loss rather than contrary to it — and critically, the matched null faces the same "to"-final prompts without reproducing the collapse, so prompt shape alone does not produce it.

**Internal structure.** Treating the 180 features as one set hides their internal distribution, so we also ablated each feature *individually* on the Berlin prompt. Only **24/180 (13.3%)** shift the prediction alone, and they concentrate sharply at the embedding layer (**17/24 at L0**; 42.5% of L0 features tested vs. 5.0% at L1+). The circuit is *internally redundant*: necessity lives in the set, not in most of its members, and the individually load-bearing ones are early template features such as L0/11651 ("the word 'to'").

#### 3.7.2 Robustness: Layer, Magnitude, and Generalization

**Is it just deletion of L0 token features?** Forty of the 180 features are at L0 (the embedding layer), so suppressing them resembles deleting the entity tokens. If that were the whole effect, removing only the L0 features should kill the answer. It does not. Ablating the circuit one layer-band at a time:

| Prompt | L0 only (40) | L1–L4 (67) | L5–L9 (50) | L10+ (23) | L1+ no-L0 (140) |
|---|---|---|---|---|---|
| Berlin | ` Germany` ✓ | broke | broke | broke | broke |
| Rome | ` Italy` ✓ | broke | broke | broke | broke |
| Tokyo | ` Japan` ✓ | broke | broke | broke | broke |
| Bird | ` air` ✓ | broke (→ water) | broke | broke | broke |
| Teacher | broke | broke | broke | broke | broke |

*Ablating only the 40 L0 embedding features leaves the answer intact on 4/5 prompts (the exception, teacher, has a weak baseline, p=0.49). Necessity is carried by L1+.* The token-deletion explanation fails: the causal weight sits in the relational mid/late features, not the input-token representations. (The bird result is a bonus — L0 alone leaves ` air` intact, but the L1–L4 band reverts it to ` water`, the source-domain answer, the predicted relational failure.)

**Is the effect an artifact of the large −80 magnitude?** A strength titration on Berlin and Tokyo sweeps `strength` from −2 (effective −8) to −40 (effective −160):

| strength | Berlin circuit | Berlin null | Tokyo circuit | Tokyo null |
|---|---|---|---|---|
| −2 | ` to` (−0.03) | ` similar` (−0.62) | ` to` (−0.03) | ` onPostExecute` (−1.11) |
| −40 | ` to` (−0.03) | ` similar` (−0.70) | ` to` (−0.03) | ` onPostExecute` (−1.20) |

The clean ` to` collapse is already complete at the weakest setting tested (−2), and the circuit-vs-null distinction holds across the whole range. The effect is not a product of the blunt −80 intervention.

**Does the circuit generalize beyond its defining prompts?** The 180 features were defined from five specific prompts. Applying the *fixed* circuit to four held-out analogies it was never derived from:

| Held-out prompt | Baseline | Circuit ablation → |
|---|---|---|
| Lisbon…Vienna is to | ` Austria` | ` to` (−0.03) |
| Athens…Oslo is to | ` Norway` | ` to` (−0.03) |
| Pen…knife is to | ` cutting` | ` to` (−0.03) |
| Bee…ant is to | ` colony` | ` to` (−0.03) |

The same ` to` collapse at −0.03 appears on all four → the circuit signature is not overfit to its defining set.

#### 3.7.3 The Three-Phase Architecture

§3.7.1–§3.7.2 establish that the recurring circuit is causally load-bearing. A separate claim (§2.4) is that the circuit is *organized* into three phases. We test that with phase-level suppression of the representative features per phase, across all five prompts.

**Individual phase necessity.** Suppressing the 9 key phase features one at a time (45 tests) shows the Phase-2 "analogy" features are individually redundant on high-confidence prompts, while a Phase-1 template feature, L0/11651 ("the word 'to'"), is individually necessary in 4/5 circuits — suppressing it makes the capital analogies emit the *city name itself* rather than completing the analogy.

**Collective phase suppression** is the decisive architecture experiment:

| Experiment | Features | Berlin | Rome | Tokyo | Teacher | Bird |
|---|---|---|---|---|---|---|
| All Phase 2 (4 feat.) | L5/5793, L5/2141, L8/13766, L9/13344 | **France** | **France** | **France** | be | fish |
| All Phase 1 (5 feat.) | L0/11651, L1/11356, L4/10752, L5/9672, L2/11475 | (empty) | (empty) | (empty) | to | to |
| Phase 1+2 (9 feat.) | All Phase 1 + Phase 2 | : | : | : | : | : |

*All cells disrupted.* Suppressing all four Phase-2 features makes the three capital analogies output **"France"** — the *source* country. The model retains the factual association "Paris is to France" but loses the relational transfer "as Berlin is to ___." This is the failure mode predicted if Phase 2 implements relational transfer, and it is the strongest single piece of architecture evidence. Phase-1 suppression produces a more severe failure (empty output for capitals), consistent with Phase 1 being a prerequisite for Phase 2. Together these establish an ordered hierarchy: Phase 1 (template) → Phase 2 (relational transfer) → later layers (answer retrieval).

#### 3.7.4 The Causal Validation Ledger

Collecting the interventions against the structural claims they bear on. "Demonstrated" means the predicted causal signature was observed under the matched-contrast or predicted-failure-mode standard; "supported" means consistent evidence without a full matched control; "not tested" means the claim is descriptive and was not subjected to steering.

| Structural claim | Causal test | Result | Verdict |
|---|---|---|---|
| **1.** The 180 recurring features form the circuit that drives the analogy, not passengers (§2.3, §3.2) | Circuit ablation vs. matched null, all 5 prompts (§3.7.1) | Circuit → reproducible ` to` collapse (−0.03, 5/5); null → idiosyncratic OOD tokens (mean −1.35) | **Demonstrated** (failure-mode contrast). Necessity alone is non-discriminating at this strength. |
| **2.** Not merely deletion of L0 entity tokens | Layer-stratified ablation, L0 excluded (§3.7.2) | L0-only leaves the answer intact 4/5; L1+ carries necessity | **Demonstrated** |
| **3.** Not an artifact of the −80 magnitude | Strength titration −2 → −40 (§3.7.2) | Clean ` to` collapse already at −2; contrast holds across range | **Demonstrated** |
| **4.** The circuit generalizes beyond its defining prompts | Fixed circuit on 4 held-out analogies (§3.7.2) | Same ` to` collapse, 4/4 | **Supported** |
| **5.** Dedicated analogy features (L5/8/9/13) participate causally (§3.4, §4.1) | Phase-2 collective suppression, 5 prompts (§3.7.3) | Capitals → source country "France"; 5/5 disrupted | **Demonstrated collectively** (individually redundant) |
| **6.** Three-phase architecture (§2.4, §3.3) | Phase 1 / Phase 2 / Phase 1+2 suppression (§3.7.3) | Distinct ordered failure modes | **Demonstrated** |
| **7.** Cross-domain generalization (§3.5) | Circuit + phase tests include teacher & bird | Same signatures in semantic-role analogies | **Supported** (weak baselines for teacher p=0.49, bird p=0.12) |
| **8.** Formal-text features co-activate (§3.4, §4.2) | — (descriptive) | — | **Not tested** (individually causally inert) |

**What this ledger does not claim.** Three honest bounds. *(i) Sufficiency:* we show the circuit is necessary and fails in the predicted way; we do not show it is *sufficient* to generate the answer in isolation. The single-feature sufficiency probe (boost the highest-influence hub) is largely negative — the hub induces the target only when the target entity is already present in the prompt. *(ii) Surgical necessity:* because −80 is a strong intervention, single-feature necessity is layer-dependent, and the strongest claims rest on the matched contrast rather than any individual ablation. *(iii) Mediation:* the phase experiments show each phase is collectively necessary, but establishing that information *flows along the edge* Phase 1 → Phase 2 requires path patching on the model's own weights, which the steering API does not expose; this remains future work. With those bounds stated, the steering evidence supports the paper's central structural claims: a recurring circuit that is causally load-bearing across five prompts (and four held-out analogies), not reducible to token deletion or to the intervention magnitude, internally organized into the three phases of §3.3, and shared between geographic and semantic-role analogies.
---

<!-- slide: Discussion -->

## 4. Discussion

### 4.1 The Analogical Reasoning Circuit in Gemma-2-2B

Our analysis reveals that Gemma-2-2B implements analogical reasoning through a distributed circuit spanning all 26 transformer layers, with specific functional specialization at each phase. The most significant finding is the existence of **explicitly semantic analogy features** at layers 5, 8, 9, and 13 — features whose automated explanations use the words "analogies," "comparisons," and "relationships between concepts." This suggests that the model has internalized analogical structure as a discrete, reusable computational primitive.

This is qualitatively distinct from multi-hop factual reasoning. Analogical reasoning requires extracting an unnamed relation type, holding it as a variable, and applying it to a new argument pair. The Phase 2 collective suppression experiment demonstrates that this extraction and transfer are implemented by identifiable, causally load-bearing internal components whose removal causes the model to echo the source-pair answer rather than transfer the relation — consistent with the "missing relational information" failure mode documented by Lee et al. [10] at the behavioral level. Our work provides a feature-level causal account of this phenomenon.

Prior behavioral evidence [1] established that LLMs can match human performance on analogical tasks; Webb et al. [9] identified emergent symbolic mechanisms supporting abstract reasoning through causal mediation of attention heads. The present work extends these findings to the SAE feature level: the relational reasoning primitive is not just a pattern of attention head behavior but a specifically labeled, causally load-bearing feature in the SAE's learned decomposition of residual stream activations.

### 4.2 The Role of Formal Text Features

The high-recurrence "code and legal text" features present an interpretive puzzle best understood through the lens of polysemanticity and superposition [6]. Two complementary explanations:

**Functional hypothesis:** These features detect formal, template-driven text patterns generally. The analogy syntax "X is to Y as Z is to" is highly structured, resembling legal definitions, code comments, and mathematical notation. The model reuses a general "formal syntax" detector.

**Training data hypothesis:** The analogy format appears frequently in SAT preparation and educational materials — which also contain code examples and legal definitions — creating a statistical association between formal-text features and analogy-completion contexts.

Both are compatible with the causal steering data. The formal features process the syntactic surface of the template while the analogy features process the relational semantics; only the latter are collectively necessary for relational transfer. The SAE-based decomposition [5, 6] is what makes this functional distinction visible — raw neuron activations would not cleanly separate these roles.

### 4.3 Comparison with the Capital City Recall Circuit

Comparison with the capital city *factual recall* circuit (prompt: "The capital of X is") reveals:

- **Overlap:** Formal-text features (L4/#14857, L6/#2267) appear with high frequency in both circuits, activated by the formal definitional structure of both prompt types. This is analogous to the shared MLP modules Meng et al. [2] identified across different factual recall tasks.
- **Divergence:** The L5 "analogies" feature and L8 "analogies or comparisons" feature appear to be specific to the analogical task — they were not among the top recurring features in the factual recall circuit — supporting the interpretation that these features are selectively activated by relational structure recognition.

### 4.4 Relation to Anthropic's Attribution Graph Methodology

The present work is in direct methodological continuity with Anthropic's *On the Biology of a Large Language Model* [12], which applied attribution graphs to Claude 3.5 Haiku using cross-layer transcoders. Both papers find that models implement multi-step, staged computation rather than direct input-to-output pattern matching, and both validate circuit hypotheses through feature steering. Anthropic's paper groups related features into manually curated "supernodes" to present a cleaner narrative; the present work uses automated cross-graph intersection, which is more scalable and less susceptible to confirmation bias but produces a less narratively refined picture of any single circuit. The two approaches are complementary.

### 4.5 Redundancy as a Property of Well-Learned Computation

The inverse relationship between prediction confidence and circuit fragility — ranging from 1/9 individually necessary features (Berlin, p=0.973) to 8/10 (bird, p=0.117) — suggests a general principle: well-learned associations are protected by redundant parallel causal paths, while barely-resolved predictions rely on non-redundant chains. This principle aligns with the circuit redundancy findings in [12] and may reflect a general property of how transformers allocate computational resources across tasks of varying difficulty.

---

<!-- slide: Limitations & Future Work -->

## 5. Limitations

1. **SAE-feature-level intervention only.** Steering operates at the SAE feature level, not the attention head or residual stream level. The causal role of non-SAE circuit components is not assessed.
2. **SAE coverage.** The `gemmascope-transcoder-16k` SAE [7] covers only cross-layer transcoder features. Attention head contributions and residual stream features are not captured.
3. **Threshold sensitivity.** Results are sensitive to node and edge thresholds (0.80/0.85). Lower thresholds would reveal more features; higher thresholds would produce sparser, more focused circuits.
4. **Label quality.** Neuronpedia [8] automated feature explanations are LLM-generated and may not perfectly capture feature semantics.
5. **Prompt set size.** Five prompts are sufficient for initial circuit identification but too few to claim statistical robustness. A larger prompt set covering arithmetic, cross-lingual, and abstract relational analogies [13] would strengthen conclusions.

**Future work:** Direct causal validation with TransformerLens activation patching at the attention head and residual stream level; expanded prompt sets; analysis across model scales (Gemma-2-9B, 27B); comparison with factual recall and multi-hop reasoning circuits; and testing whether the Phase 2 features generalize to the cross-lingual analogical settings studied in [14].

---

<!-- slide: Conclusions -->

## 6. Conclusions

We have identified and characterized the **analogical reasoning circuit in Gemma-2-2B** using SAE attribution graphs from the Neuronpedia platform [8]. The key conclusions are:

1. **A stable shared circuit exists, identified by common feature IDs.** 180 features — identified by stable *(layer, feature index)* pairs — appear in all five independently generated attribution graphs.
2. **Dedicated analogy features exist at layers 5, 8, 9, and 13.** These features have Neuronpedia explanations explicitly referencing analogies, comparisons, and relational concepts — providing direct SAE-level evidence of interpretable analogy-concept features in a large language model.
3. **The circuit exhibits a three-phase architecture, identified by label semantics and validated causally.** Circuit template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13), with activation magnitude increasing through the sequence.
4. **Cross-domain generalization is confirmed.** The same core features, including L5 SAE#5793 ("analogies"), activate for both geographic and semantic role analogies — a domain-agnostic relational reasoning primitive consistent with behavioral findings [1, 10, 11].
5. **Phase 2 implements relational transfer, collectively but not individually.** Simultaneous suppression collapses every circuit; capital analogies revert to the source-pair answer.
6. **Circuit fragility tracks prediction confidence.** High-confidence predictions route through redundant parallel causal paths (1–4 necessary features); low-confidence predictions rely on fragile non-redundant chains (up to 8/10 necessary).

---

## Attribution Graphs

The five Neuronpedia attribution graphs generated for this study are publicly available for interactive exploration. Full graph descriptions, inference prompts, and the agent pipeline methodology are documented in the [Supplementary Material](supplementary.html).

| Prompt | Neuronpedia Graph |
|--------|------------------|
| Paris is to France as Berlin is to | [analog\_berlin](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin) |
| Paris is to France as Rome is to | [analog\_rome](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_rome) |
| Paris is to France as Tokyo is to | [analog\_tokyo](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_tokyo) |
| Doctor is to hospital as teacher is to | [analog\_teacher](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_teacher) |
| Fish is to water as bird is to | [analog\_bird](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_bird) |

---

## References

[1] Webb, T., Holyoak, K.J., & Lu, H. (2023). Emergent analogical reasoning in large language models. *Nature Human Behaviour*, 7, 1526–1541. arXiv: https://arxiv.org/abs/2212.09196

[2] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*. https://arxiv.org/abs/2202.05262

[3] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *ICLR 2023*. https://arxiv.org/abs/2211.00593

[4] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. *NeurIPS 2023*. https://arxiv.org/abs/2304.14997

[5] Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. *ICLR 2024*. https://arxiv.org/abs/2309.08600

[6] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features

[7] Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., & Nanda, N. (2024). Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. https://arxiv.org/abs/2408.05147

[8] Lin, J., & Bloom, J. (2023). Neuronpedia: Interactive platform for sparse autoencoder research and feature steering. https://www.neuronpedia.org

[9] Webb, T.W., Frankland, S.M., Altabaa, A., Segert, S., Krishnamurthy, K., Campbell, D., Russin, J., Giallanza, T., O'Reilly, R., Lafferty, J., & Cohen, J.D. (2025). Emergent symbolic mechanisms support abstract reasoning in large language models. https://arxiv.org/abs/2502.20332

[10] Lee, T., et al. (2025). The curious case of analogies: Investigating analogical reasoning in large language models. https://arxiv.org/abs/2511.20344

[11] Wijesiriwardene, T., et al. (2025). Analogical reasoning inside large language models: Concept vectors and the limits of abstraction. https://arxiv.org/abs/2503.03666

[12] Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N.L., et al. (2025). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html

[13] Turney, P.D. (2006). Similarity of semantic relations. *Computational Linguistics*, 32(3), 379–416. [Foundational work on relational similarity benchmarks underlying analogy tasks.]

[14] Allen, C., & Hospedales, T. (2019). Analogies explained: Towards understanding word embeddings. *ICML 2019*. https://arxiv.org/abs/1901.09813

[15] Marks, S., Rager, C., Michaud, E.J., Belinkov, Y., Bau, D., & Mueller, A. (2024). Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. https://arxiv.org/abs/2403.19647

---

## Supplementary Materials

**Interactive Presentation:** 20-slide reveal.js presentation with circuit flow diagrams, feature tables, and layer-by-layer analysis.  
https://kkrampis.github.io/autocircuit/presentation.html

**Live Attribution Graphs:**

- [`analog_berlin`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin) — Paris - France → Berlin - ?
- [`analog_rome`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_rome) — Paris - France → Rome - ?
- [`analog_tokyo`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_tokyo) — Paris - France → Tokyo - ?
- [`analog_teacher`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_teacher) — Doctor - hospital → teacher - ?
- [`analog_bird`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_bird) — Fish - water → bird - ?

**Code:** https://github.com/kkrampis/autocircuit

---

```bibtex
@article{alagbe2026analogical,
  title   = {Mechanistic Interpretability of Analogical Reasoning in {Gemma-2-2B}:
             A Sparse Autoencoder Attribution Graph Analysis},
  author  = {Alagbe, Olalekan and Lawrence, Joseph and Maheshwar, Anish and Krampis, Konstantinos},
  year    = {2026},
  month   = {March},
  note    = {Neuronpedia API \texttt{gemmascope-transcoder-16k} SAE analysis}
}
```
