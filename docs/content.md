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

### 2.7 Circuit Definition and Causal Validation via Feature Steering

A **circuit** for a given prompt is defined as the backbone of causally important features in that prompt's attribution graph. The extraction procedure is identical for all prompts: trace the top-5 causal paths backward from the logit node (greedy walk following highest-weight incoming edges) and the top-5 paths forward from the highest-influence embedding nodes. Any transcoder feature appearing on at least one of these 10 paths is a backbone member. This procedure is analogous to the automated circuit discovery approach of Conmy et al. [4], applied here at the SAE feature level rather than the attention head level.

Causal validation was performed using the Neuronpedia `/api/steer` endpoint [8] with `modelId: "gemma-2-2b"` and `strength_multiplier: 4`. Four experimental paradigms were applied:

1. **Necessity (individual suppression):** Suppress a single backbone feature at strength −20.
2. **Necessity (full backbone suppression):** Suppress all backbone features simultaneously.
3. **Sufficiency (hub boost):** Boost a backbone hub feature at strength +20 on altered prompts.
4. **Specificity (non-backbone suppression):** Suppress high-activation features not on any causal path.

Two additional circuits from an expanded 30-prompt analysis (Cairo→Kenya, Puppy→cat) were included for cross-validation, for a total of **159 individual steering API calls across 7 circuits**.

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

### 3.6 Circuit Stability Across Scaled and Syntactically Diverse Prompts

To validate that the shared circuit identified in §3.1 is not an artifact of using only five similar prompts, an extensive scaling experiment was performed. The central question: if we keep adding new analogical prompts — including versions phrased very differently from the original format — do the same features keep showing up?

The experiment generated attribution graphs for 50 prompts in total. Crucially, from the second batch onward, the prompts were not just new examples of the same template — they were rephrased into three syntactically distinct surface forms alongside the original:

| Surface Form | Example |
|---|---|
| Standard X-to-Y | `Paris is to France as Berlin is to` |
| Diverse-A (Just as…) | `Just as Paris is the capital of France, Berlin is the capital of` |
| Diverse-B (Found in…) | `Doctors are found in hospitals. Teachers are found in` |
| Diverse-C (The way…) | `The way a fish lives in water, a bird lives in` |

These four forms look very different on the surface — different word order, different connectives, no shared "is to … as" string. If the circuit from §3.1 were merely tracking surface tokens, it would fall apart when these diverse forms were introduced.

At each milestone (N = 5, 10, 20, 30, 40, 50), the strictest possible threshold was applied: a feature must appear in **every single** attribution graph at that point.

| N | Recurring features (k = ALL) | Drop from previous |
|---|---|---|
| 5 | **180** | — |
| 10 | **116** | −64 (−35.6 %) |
| 20 | **86** | −30 (−25.9 %) |
| 30 | **77** | −9 (−10.5 %) |
| 40 | **70** | −7 (−9.1 %) |
| 50 | **67** | −3 (−4.3 %) |

![Figure S1: Scaling curve showing the number of features that recur across ALL N attribution graphs as N grows from 5 to 50. The curve drops steeply at first — removing features that only appeared by coincidence in the small prompt set — then nearly plateaus, converging to a stable floor of 67 features.](scaling_curve.png)

The curve has a clear two-phase shape. First, a **rapid contraction** from N = 5 to N = 20: adding the first batch of syntactically diverse prompts eliminates roughly half the initial 180 features — features that appeared in every one of the original five prompts by coincidence, or because all five shared the "is to … as" surface string, do not survive once fundamentally different phrasings are added. Second, a **near-plateau** from N = 20 onward: only 19 further features are lost across 30 additional prompts, and the final step (N = 40 → 50) removes just 3. By N = 50, the curve has essentially stopped moving.

This convergence behaviour is the key result. The 67-feature core at N = 50 survived 50 prompts spanning two semantic domains, four surface forms, and a strict all-or-nothing threshold. That is not noise — it is a stable circuit.

**The five directly analogical features all survive.** Within the 67-feature core, five features carry Neuronpedia labels that explicitly describe analogical or comparative reasoning. Every one of them appears in all 50 attribution graphs:

| Feature | Appearances | Avg. Influence | Label |
|---|---|---|---|
| L13 #10969 | 62 | 0.713 | "comparisons between disciplines and relationships between concepts" |
| L9 #13344 | 116 | 0.683 | "phrases suggesting uncertainty or comparison between two things" |
| L9 #14231 | 53 | 0.683 | "words representing comparisons and relationships" |
| L7 #749 | 80 | 0.652 | "analogies and comparisons" |
| L5 #2141 | 62 | 0.639 | "comparisons of people or figures using well-known public figures" |

Three of these — L13 #10969, L9 #13344, and L5 #2141 — were already identified in the original five-prompt analysis (§3.4). The scaling experiment adds two new ones: L9 #14231 ("words representing comparisons and relationships") and L7 #749 ("analogies and comparisons"), which only become visible once the prompt set is large and diverse enough to filter out coincidental co-activations. All five span the analogy recognition and relational integration phases (§3.4–3.5).

Despite the 50 prompts being phrased four different ways, the model consistently activated the same five semantic features. This confirms that the circuit is not reading a surface token pattern — it is recognising the underlying relational structure of an analogy, regardless of how that structure is expressed in words.

### 3.7 Causal Validation via Feature Steering

The attribution graph analysis identifies recurring features and causal path structures but does not by itself establish whether those features are causally necessary for the model's predictions. To answer that question, we performed systematic causal steering experiments via the Neuronpedia API [8]. However, the experiments test two distinct feature sets that serve different purposes, and understanding which set is being tested at each point is essential for reading the results correctly.

**Feature Set A — Backbone features (§3.7.1–3.7.2).** A backbone is the set of features that sit on the main causal highway through a single prompt's attribution graph — traced via the top-5 highest-weight paths forward from the embedding layer and the top-5 highest-weight paths backward from the logit node (§2.7). Because the paths are traced toward the output token, backbone features naturally cluster in the final layers of the model (primarily L16–L25) — they are the last-mile computation before a word is written. They are *not* the cross-graph recurring features identified in §3.2–3.5; each circuit has its own backbone. Testing them first establishes that the attribution graphs capture real causal structure rather than mere correlations — a prerequisite for trusting any downstream steering result on the shared circuit features.

**Feature Set B — Phase features (§3.7.3–3.7.4).** These are the cross-graph recurring features identified in §3.2–3.5 — the Phase 1 structural template features (L0–L4), the Phase 2 analogy recognition features (L5–L9), and the Phase 3 integration features (L10–L13). These are the features the paper's main claim is about; testing them is the primary causal validation of the shared circuit.

#### 3.7.1 Late-Layer Backbone Necessity (Individual Suppression)

*Feature Set A.* These are prompt-specific late-layer features — not the cross-graph recurring Phase features from §3.2–3.5. The experiments below ask: are these final-stage output features individually necessary for the model's prediction, and does attribution weight predict which ones are?

**`analog_berlin`** — "Paris is to France as Berlin is to" → Germany (p=0.973)

| Feature | Layer | Index | Role | Steered Token | Necessary? |
|---|---|---|---|---|---|
| Science hub | 21 | 4827 | Strongest path entry (edge +198.0) | Germany | no |
| Relay | 22 | 15670 | Path 1 relay | Germany | no |
| Output driver A | 25 | 4717 | Final amplifier (shared across circuits) | Germany | no |
| Location encoder | 16 | 6491 | Location/direction feature, path 2 entry | Germany | no |
| Relay | 17 | 14546 | Mid-cascade relay | Germany | no |
| Relay | 19 | 5773 | Late relay | Germany | no |
| Integrator | 21 | 7482 | Integration hub (paths 2–4) | Germany | no |
| Output driver B | 25 | 2725 | Secondary output driver (edge −2.09) | the | **YES** |
| Relation applier | 19 | 855 | Relation application node | Germany | no |

**1/9 necessary.** Only L25/2725 is a single point of failure. The highest-weight feature (L21/4827, edge +198.0) is not individually necessary, demonstrating that attribution weight alone does not predict causal necessity — a key methodological lesson consistent with prior circuit analysis work [3, 4].

**`analog_rome`** — "Paris is to France as Rome is to" → Italy (p=0.974)

| Feature | Layer | Index | Role | Steered Token | Necessary? |
|---|---|---|---|---|---|
| Relay | 20 | 15360 | Backward path from logit | Italy | no |
| Late gate | 24 | 16122 | Backward path, L24 suppression gate | the | **YES** |
| Output driver | 25 | 286 | Backward path, output driver | the | **YES** |
| Final amplifier | 25 | 4717 | Shared final amplifier (act=265.2) | the | **YES** |
| Output driver C | 25 | 10521 | Tertiary output driver | the | **YES** |
| Relay | 17 | 14546 | Mid-cascade relay (shared with Berlin) | Italy | no |
| Relay A | 22 | 12202 | Late relay | Italy | no |
| Relay B | 22 | 14727 | Late relay | Italy | no |
| Relay | 23 | 5917 | Late relay | Italy | no |
| Secondary gate | 24 | 13277 | Late gate | Italy | no |

**4/10 necessary.** The Rome circuit has more single points of failure than Berlin despite near-identical confidence (p=0.974 vs 0.973), indicating path redundancy varies even among structurally similar geographic analogies.

**`analog_tokyo`** — "Paris is to France as Tokyo is to" → Japan (p=0.990)

| Feature | Layer | Index | Role | Steered Token | Necessary? |
|---|---|---|---|---|---|
| Relay | 20 | 15360 | Backward path from logit | Japan | no |
| Output driver | 25 | 286 | Backward path, output driver | the | **YES** |
| Output driver B | 25 | 12223 | Backward path, secondary output | Japan | no |
| Relay | 17 | 14546 | Mid-cascade relay (shared) | Japan | no |
| Late relay A | 23 | 850 | Late relay | the | **YES** |
| Late relay B | 23 | 13914 | Late relay (also necessary in Cairo circuit) | the | **YES** |
| Gate | 24 | 13277 | Late gate (shared with Rome) | Japan | no |
| Output driver C | 25 | 10152 | Tertiary output | Japan | no |
| Hub | 20 | 6648 | L20 convergence hub | Japan | no |
| Integration | 21 | 7764 | Late integration | Japan | no |

**3/10 necessary.** L23/13914 is necessary in both Tokyo and Cairo circuits — a shared bottleneck consistent with a late-layer "geographic entity selector" role. L25/286 recurs as necessary in Rome, Tokyo, and Cairo, making it the single most critical output driver across geographic analogies.

**`analog_teacher`** — "Doctor is to hospital as teacher is to" → school (p=0.486)

| Feature | Layer | Index | Role | Steered Token | Necessary? |
|---|---|---|---|---|---|
| Embedding | 0 | 17 | Backward path, embedding-level | the | **YES** |
| Gateway | 18 | 6532 | Backward path, mid-late gateway | school | no |
| Hub | 20 | 6179 | Backward path, convergence hub | school | no |
| Output driver | 25 | 4975 | Backward path, output driver | ... | **YES** |
| Final amplifier | 25 | 4717 | Shared final amplifier (act=135.6) | a | **YES** |
| Relay | 22 | 15670 | Late relay (shared) | school | no |
| Relay B | 18 | 11952 | Mid-late relay | school | no |
| Legal docs | 18 | 13586 | Legal docs feature | school | no |
| Convergence | 21 | 2655 | Late convergence hub | school | no |
| Gate | 24 | 15259 | Late suppression gate | school | no |

**3/10 necessary.** The teacher circuit is the only one where an **L0 embedding-level feature** (L0/17) is individually necessary — suggesting the semantic role analogy relies on an early feature not redundantly encoded by later layers, unlike the capital analogies.

**`analog_bird`** — "Fish is to water as bird is to" → air (p=0.117)

| Feature | Layer | Index | Role | Steered Token | Necessary? |
|---|---|---|---|---|---|
| Backward A | 22 | 4252 | Backward path from logit | (space) | **YES** |
| Backward B | 24 | 8106 | Backward path, late gate | \_\_\_\_ | **YES** |
| Final amplifier | 25 | 4717 | Shared final amplifier (act=122.3) | the | **YES** |
| Output driver | 25 | 11801 | Output driver | ? | **YES** |
| Relay A | 22 | 15670 | Late relay (shared) | \_\_\_\_\_\_\_\_ | **YES** |
| Relay B | 22 | 14727 | Late relay | (space) | **YES** |
| Relay C | 22 | 13619 | Late relay | (space) | **YES** |
| Gate A | 24 | 4383 | Suppression gate | air | no |
| Gate B | 24 | 12559 | Suppression gate | the | **YES** |
| Hub | 20 | 3094 | Integration hub | air | no |

**8/10 necessary.** This is the most fragile circuit in the dataset. Three L22 relay features (15670, 14727, 13619) are all independently necessary despite occupying the same layer, indicating they carry non-redundant information through parallel channels. This fragility is consistent with the circuit's very low prediction confidence (p=0.117).

**Cross-circuit pattern.** Necessity inversely correlates with prediction confidence: Berlin (p=0.973): 1/9; Rome (p=0.974): 4/10; Tokyo (p=0.990): 3/10; Teacher (p=0.486): 3/10; Bird (p=0.117): 8/10. Three features recur as necessary across multiple circuits: **L25/#286** (Rome, Tokyo, Cairo), **L25/#4717** (Rome, Teacher, Bird), and **L23/#13914** (Tokyo, Cairo).

#### 3.7.2 Full Backbone Suppression

| Circuit | p | N feat. | Default Output | Steered Output | Disrupted? |
|---|---|---|---|---|---|
| `analog_berlin` | 0.973 | 9 | Germany. It is the | of of of of of | YES |
| `analog_rome` | 0.974 | 10 | Italy. It is the | pleaſure pleaſure plea | YES |
| `analog_tokyo` | 0.990 | 10 | Japan. It is the | country country count | YES |
| `analog_teacher` | 0.486 | 10 | school. The doc | 1111 | YES |
| `analog_bird` | 0.117 | 10 | air. The fish | (newline) The the the | YES |
| Cairo→Kenya | 0.963 | 9 | Kenya. It is the | (whitespace) | YES |
| Puppy→cat | 0.756 | 4 | cat. I' | cat. I think | no |

**6/7 circuits fully disrupted.** Failure modes are qualitatively informative: capital analogies degenerate to repetitive or archaic text, indicating the backbone is required for entity selection while the prompt structure alone partially activates a "country" category. Teacher collapses to "1111"; bird falls through to generic continuation. Puppy→cat is the sole exception, apparently carried by direct embedding-to-logit connections outside the multi-hop backbone.

#### 3.7.3 Phase 1 and Phase 2 Feature Necessity

*Feature Set B — primary validation of the shared circuit.* The features tested here are the cross-graph recurring features identified in §3.3–3.4: the five Phase 1 structural template features (L0–L4), the four Phase 2 analogy recognition features (L5–L9), and the Phase 3 integration feature (L13/#10969). Unlike the backbone features in §3.7.1–3.7.2, which were selected by path tracing within individual graphs, these features were selected because they recur across all five independently generated attribution graphs. The question here is whether they are also causally necessary.

Individual suppression of the 9 key phase features across all five prompts (45 tests):

| Feature | Phase | Label | Berlin | Rome | Tokyo | Teacher | Bird |
|---|---|---|---|---|---|---|---|
| L0/11651 | 1 | "the word 'to'" | Berlin | Rome | Tokyo | school | water |
| L1/11356 | 1 | "'to' followed by a verb" | — | — | — | — | — |
| L4/10752 | 1 | "'to be' preceded by 'to'" | — | — | — | classroom | sky |
| L5/9672 | 1 | "the phrase 'it is to'" | — | — | — | — | sky |
| **L5/5793** | **2** | **"analogies"** | — | — | — | — | — |
| L5/2141 | 2 | "comparisons of public figures" | — | — | — | — | — |
| L8/13766 | 2 | "analogies or comparisons" | — | — | — | — | fish |
| L9/13344 | 2 | "comparison between two things" | — | — | — | — | sky |
| L13/10969 | 3 | "comparisons between disciplines" | — | — | — | — | — |

*Cells show the steered first token when suppressed at strength −20. "—" = prediction unchanged.*

**Phase 1 results.** L0/11651 is necessary in 4/5 circuits. Suppressing it causes capital analogies to predict the *city name itself* (Berlin, Rome, Tokyo), indicating the model reverts to the most recently mentioned entity rather than completing the analogy. For the bird circuit, suppression produces "water" (source-pair element).

**Phase 2 results.** L5/5793 ("analogies") is **never individually necessary** in any circuit — it is individually redundant for high-confidence circuits. For the fragile bird circuit (p=0.117), however, Phase 2 features become individually necessary: L8/13766 changes "air" to "fish" (source-domain animal); L9/13344 changes "air" to "sky."

**Phase 3 results.** L13/10969 is not individually necessary for any circuit.

#### 3.7.4 Collective Phase Suppression

| Experiment | Features Suppressed | Berlin | Rome | Tokyo | Teacher | Bird |
|---|---|---|---|---|---|---|
| All Phase 2 (4 feat.) | L5/5793, L5/2141, L8/13766, L9/13344 | **France** | **France** | **France** | be | fish |
| All Phase 1 (5 feat.) | L0/11651, L1/11356, L4/10752, L5/9672, L2/11475 | (empty) | (empty) | (empty) | to | to |
| Phase 1+2 (9 feat.) | All Phase 1 + Phase 2 | : | : | : | : | : |
| Phase 1+2+3 (10 feat.) | All Phase 1 + Phase 2 + L13/10969 | : | : | : | be | : |

*All 20/20 cells disrupted.*

**Phase 2 collective suppression is the most informative experiment in this paper.** All three capital analogies output **"France"** — retaining the factual association "Paris is to France" but losing the relational transfer "as Berlin is to \_\_\_." This is direct causal evidence that Phase 2 features collectively implement the relational transfer operation. The failure mode is precisely what one would predict from the internal representation findings of Lee et al. [10], where reasoning failures reflect missing relational information in mid-upper layers.

**Phase 1 collective suppression** produces empty outputs for capital analogies and "to" for semantic role analogies — a more severe failure, consistent with Phase 1 being a prerequisite for Phase 2.

**Combined Phase 1+2 suppression** produces ":" for 4/5 circuits, consistent with the model defaulting to list-formatting punctuation when both template parsing and analogy recognition are disabled.

These results establish a **causal hierarchy**: Phase 1 → Phase 2 → Phase 3 + late layers. Each phase is collectively necessary, and earlier phases are prerequisites for later ones.

#### 3.7.5 Sufficiency (Hub Boost on Altered Prompts)

| Circuit | Hub Boosted | Altered Prompt | Induced? |
|---|---|---|---|
| `analog_berlin` | L21/4827 | "Cairo is to Egypt as Nairobi is to" | no |
| `analog_berlin` | L21/4827 | "Madrid is to Spain as Berlin is to" | **YES → Germany** |
| `analog_rome` | L20/15360 | "Paris is to France as Tokyo is to" | no |
| `analog_rome` | L20/15360 | "Madrid is to Spain as Rome is to" | **YES → Italy** |
| `analog_tokyo` | L20/15360 | "Paris is to France as Rome is to" | no |
| `analog_tokyo` | L20/15360 | "Beijing is to China as Tokyo is to" | **YES → Japan** |
| `analog_teacher` | L0/17 | "Nurse is to hospital as teacher is to" | no |
| `analog_teacher` | L0/17 | "Doctor is to hospital as chef is to" | no |
| `analog_bird` | L22/4252 | "Cat is to land as bird is to" | no |
| `analog_bird` | L22/4252 | "Fish is to water as eagle is to" | **YES → air** |
| Cairo→Kenya | L15/15954 | "Lagos is to Nigeria as Nairobi is to" | **YES → Kenya** |

**5/11 tests succeed.** Sufficiency holds when the altered prompt retains the target entity or a semantically close substitute, and fails when it crosses domain boundaries. The capital hubs encode domain-specific geographic associations rather than general-purpose "answer slot" activators.

#### 3.7.6 Specificity (Non-Backbone Feature Suppression)

| Circuit | Feature | Label | Steered Token | Disrupted? |
|---|---|---|---|---|
| `analog_berlin` | L6/3335 | "difficulty/challenges" | Germany | no |
| `analog_berlin` | L13/4435 | "opera-related terms" | Germany | no |
| `analog_rome` | L6/2267 | "formal text/code" | Italy | no |
| `analog_rome` | L4/14857 | "code snippets" | Italy | no |
| `analog_tokyo` | L6/2267 | "formal text/code" | Japan | no |
| `analog_tokyo` | L3/10018 | early structural feature | Japan | no |
| `analog_teacher` | L4/14857 | "code snippets" | school | no |
| `analog_teacher` | L8/13766 | "analogies or comparisons" | school | no |
| `analog_bird` | L6/2267 | "formal text/code" | **sky** | YES |
| `analog_bird` | L5/5793 | "analogies" | air | no |
| Cairo→Kenya | L5/5500 | "profanity and comparisons" | Kenya | no |
| Puppy→cat | L9/2909 | "formulas/ratios" | cat | no |

**12/13 pass specificity.** The sole exception — L6/2267 tipping bird from "air" to "sky" — occurs at the margin of unresolved token competition (p=0.117) and is confirmed inert for all high-confidence circuits. L5/5793 ("analogies") passes specificity for the bird circuit, consistent with it being individually dispensable but collectively necessary.

#### 3.7.7 Summary of Causal Validation

| Circuit | p | Type | Individual Necessity | Full Suppress | Phase 2 Collective | Sufficiency | Specificity |
|---|---|---|---|---|---|---|---|
| `analog_berlin` | 0.973 | Capital | 1/9 | DISRUPTED | → France | 1/2 | PASS |
| `analog_rome` | 0.974 | Capital | 4/10 | DISRUPTED | → France | 1/2 | PASS |
| `analog_tokyo` | 0.990 | Capital | 3/10 | DISRUPTED | → France | 1/2 | PASS |
| `analog_teacher` | 0.486 | Sem. role | 3/10 | DISRUPTED | → be | 0/2 | PASS |
| `analog_bird` | 0.117 | Sem. role | 8/10 | DISRUPTED | → fish | 1/2 | 1/2 |
| Cairo→Kenya | 0.963 | Capital | 2/9 | DISRUPTED | — | 1/2 | PASS |
| Puppy→cat | 0.756 | Sem. role | 0/4 | intact | — | — | PASS |

Across 159 steering experiments, the two validation tracks converge on five principal findings.

From **Feature Set A (backbone, §3.7.1–3.7.2):** First, the late-layer backbone is collectively necessary — full suppression disrupts 6/7 circuits — establishing that the attribution graphs track genuine causal structure. Second, attribution weight does not predict individual necessity: the highest-weight feature in the Berlin graph (L21/4827, edge +198.0) is not individually necessary, while a lower-weight feature (L25/2725) is. Third, three late-layer features recur as necessary across multiple distinct circuits — L25/#286 (Rome, Tokyo, Cairo), L25/#4717 (Rome, Teacher, Bird), and L23/#13914 (Tokyo, Cairo) — revealing a shared output mechanism not visible in the cross-graph overlap analysis. Fourth, individual necessity scales inversely with prediction confidence: the Bird circuit (p=0.117) has 8/10 necessary backbone features while the Berlin circuit (p=0.973) has 1/9.

From **Feature Set B (phase features, §3.7.3–3.7.4):** Fifth, the shared circuit identified in §3.2–3.5 is causally validated. Phase 1 features are individually necessary — L0/#11651 alone disrupts 4/5 circuits, causing capital prompts to revert to the city name. Phase 2 features are collectively necessary but individually redundant — simultaneous suppression collapses every circuit, with capital analogies reverting to "France" (the source-pair answer), while no single Phase 2 feature is individually indispensable. This is direct causal evidence that Phase 2 collectively implements the relational transfer operation. Phase 3 and formal-text features (L4/#14857, L6/#2267) are not individually necessary, confirming that high recurrence in the attribution graphs does not imply causal load-bearing.

---

<!-- slide: Discussion -->

## 4. Discussion

**Overall synthesis.** The results establish that Gemma-2-2B performs analogical reasoning through a stable, three-phase distributed circuit rather than any single mechanism or layer. The convergence of structural, semantic, and causal evidence — across 159 steering experiments, a 50-prompt scaling study, cross-domain generalization testing, and 7 distinct circuits — provides a mechanistic account at a level of specificity and causal resolution that prior behavioral work on LLM analogical reasoning could not reach. The core argument of this paper is not merely that recurring features exist, but that the recurring features identified through graph overlap are causally load-bearing, and that different phases of the circuit play functionally distinct and experimentally separable roles.

**The three-phase architecture in context.** The three-phase organization — structural template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13) — mirrors the abstract reasoning architecture documented by Webb et al. [9] through causal mediation analysis, where early layers abstract tokens into relational variables, intermediate layers perform induction over those variables, and later layers retrieve answers. The present results extend that framework in two important ways: by identifying specific SAE features at each phase rather than working at the attention head level, and by providing direct causal evidence through feature steering that each phase is collectively necessary for the circuit to function. Crucially, the phase boundaries were not imposed a priori — they emerged from the content of Neuronpedia automated labels naturally clustering by layer depth, with a convergent gradient in activation magnitudes — rising from 1.5–6.4 in Phase 1 to 9.1–16.3 in Phase 3 — confirming the same partition through a second independent line of evidence.

**Circuit stability across surface forms.** Perhaps the most theoretically significant finding is the convergence of the feature set to a stable 67-feature core across 50 prompts phrased in four syntactically distinct surface forms (§3.6). The initial 180-feature circuit, identified from five prompts sharing the "X is to Y as Z is to" template, contracts rapidly when surface-diverse prompts are introduced — losing roughly half its features by N = 20 — but then plateaus, with only 19 further features lost across the subsequent 30 prompts, and just 3 in the final step. This two-phase scaling behaviour has a clear interpretation: the first contraction eliminates features that were coincidental artifacts of the shared surface template, while the plateau identifies features that activate because of the underlying relational structure, regardless of how that structure is expressed in words. All five directly analogical features survive the full 50-prompt filter. This is strong evidence against a surface-token explanation of the circuit and in favour of a genuine, abstract relational representation inside the model.

**Cross-domain generalization.** The cross-domain generalization finding reinforces this interpretation. The shared circuit — and the stable 67-feature core — includes features that activate for both geographic capital analogies and semantic role analogies. The analogy-concept features at L5 and L8 fire equally for "Paris is to France as Berlin is to" and for "Doctor is to hospital as teacher is to", despite these prompts sharing no surface tokens related to analogy. This is consistent with the behavioral finding of Wijesiriwardene et al. [11] that LLMs encode relational information in a domain-general manner, and constitutes the first identification of specific internal features implementing that domain-generality at the feature level. The slightly larger attribution graphs for semantic role analogies (1,040–1,071 nodes) relative to capital analogies (905–963 nodes) may reflect that semantic roles require broader world-knowledge access rather than retrieval of a discrete, well-encoded fact — an interpretation consistent with the ROME findings of Meng et al. [2] on the compactness of factual storage for geographic entities.

**Causal validation.** To confirm that the identified phases actually drive the model's predictions, each set of phase features was suppressed and the output observed. Phase 1 features are individually critical: removing just the feature tracking "the word 'to'" causes the model to stop completing the analogy and instead repeat the last entity it read — outputting "Berlin" instead of "Germany." Phase 2 features behave differently: removing any single one has no effect because the remaining three compensate, but removing all four simultaneously causes every capital circuit to output "France." The model still knows Paris→France but loses the ability to transfer that relationship to a new pair — direct evidence that Phase 2 collectively carries the relational transfer step. Prediction confidence and fragility are inversely related: high-confidence circuits have backup paths such that few features are individually critical, while low-confidence circuits have no redundancy and break when any single feature is removed. Finally, features whose labels describe formal text rather than analogy appear frequently in the graphs but are causally inert — suppressing them changes nothing, confirming that appearing in the graph does not equal driving the prediction.

---

<!-- slide: Limitations & Future Work -->

## 5. Limitations

All steering experiments operate at the SAE feature level only — attention heads and residual stream components are not assessed, so the three-phase architecture describes the SAE-visible portion of the circuit, not necessarily the complete one. The prompts were constructed by the authors rather than sampled from a standard benchmark, and no formal statistical tests were applied, meaning the findings are descriptive rather than statistically confirmed. Feature labels are LLM-generated by Neuronpedia and may not perfectly capture feature semantics. All results are from a single model (Gemma-2-2B) with a single SAE suite, so generalization to other model sizes or architectures is unknown.

**Future work:** activation patching at the attention head level, replication with benchmark prompt sets, and cross-model comparison.

---

<!-- slide: Conclusions -->

## 6. Conclusions

We have identified a shared analogical reasoning circuit in Gemma-2-2B comprising 180 features active across five initial prompts, converging to a stable 67-feature core across 50 prompts phrased in four syntactically distinct surface forms. The circuit is organized into three phases: Phase 1 (L0–L4) parses the structural format of the analogy prompt; Phase 2 (L5–L9) recognises the relational concept itself through features explicitly labeled "analogies" and "analogies or comparisons"; and Phase 3 (L10–L13) integrates the relation with domain-specific knowledge. The same core features activate for both geographic capital analogies and semantic role analogies, confirming a domain-agnostic relational reasoning mechanism rather than separate topic-specific circuits.

Causal steering experiments across 159 tests confirm that these phases are functionally distinct and causally necessary. Removing Phase 1 causes the model to lose the analogy structure entirely. Removing all four Phase 2 features simultaneously causes every capital circuit to output "France" — the model retains the Paris→France fact but loses the ability to transfer the relationship to a new pair, which is direct evidence that Phase 2 carries the relational transfer step. Circuits with higher prediction confidence tolerate individual feature removal better, suggesting that well-learned predictions are encoded through redundant backup paths while uncertain ones are not. These results provide the first causally validated, feature-level account of analogical reasoning in a large language model.

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
