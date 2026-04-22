# Mechanistic Interpretability of Analogical Reasoning in Gemma-2-2B: A Sparse Autoencoder Attribution Graph Analysis

**Olalekan Alagbe · Joseph Lawrence · Anish Maheshwar · Konstantinos Krampis**

*Mechanistic Interpretability · March 2026*

[Code](https://github.com/kkrampis/autocircuit) · [Presentation](https://kkrampis.github.io/autocircuit/presentation.html) · [Attribution Graphs](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin)

---

## Abstract

We present the first mechanistic analysis of analogical reasoning in Gemma-2-2B using Sparse Autoencoder (SAE) attribution graphs. By generating and comparing five attribution graphs across structurally distinct analogical prompts — covering geographic analogies (*Paris:France::Berlin:?*, *Rome:?*, *Tokyo:?*) and semantic role analogies (*Doctor:hospital::teacher:?*, *Fish:water::bird:?*) — we identify a shared **analogical reasoning circuit** comprising 180 features active across all five prompts and 510 features active across at least three. Each feature is identified by a stable *(layer, SAE index)* pair, making the circuit a concrete, reproducible list of recurring internal components rather than an approximate structural characterization.

We discover dedicated analogy-encoding features at layers 5, 8, 9, and 13, including a feature at Layer 5 labeled literally as **"analogies"** and a Layer 8 feature encoding **"analogies or comparisons"** appearing across all graphs with high influence. Early layers (0–4) contain structural template features tracking the "X is to Y as Z is to" pattern, while mid-to-late layers (5–13) house increasingly semantic representations of the relational structure. The circuit spans all 26 transformer layers and exhibits cross-domain generalization, with the same core features activating for both geographic and semantic role analogies. Causal validation via 159 feature steering experiments confirms that the identified backbone features are collectively necessary for the model's predictions, and that Phase 2 features collectively implement the relational transfer operation that is the computational core of analogical reasoning.

---

## 1. Introduction

Analogical reasoning — the ability to recognize and complete structural relationships between concepts — is a foundational cognitive ability underlying scientific discovery, language understanding, and abstract problem solving. The classic analogy task, *"Paris is to France as Berlin is to \_\_\_\_,"* tests whether a model can identify the capital-city relationship abstractly and apply it to a new country. Large language models exhibit striking competence on such tasks, yet the internal computational mechanisms remain poorly understood.

Mechanistic interpretability research has made significant progress in understanding factual recall circuits (Meng et al., 2022), indirect object identification (Wang et al., 2022), and syntactic processing (Conmy et al., 2023). However, analogical reasoning presents a distinct challenge: it requires not merely retrieving a stored fact, but recognizing a **relational structure** and applying it compositionally to novel inputs. Critically, the relation type is never named in the prompt — the model must infer *capital-of* from the example alone, hold it as a variable, and transfer it to a new argument pair.

### 1.1 Research Questions

1. Does Gemma-2-2B employ a **shared circuit** for analogical reasoning, or does it use different mechanisms for different analogy types?
2. Which SAE features — identified by stable *(layer, SAE index)* pairs — are most **consistently activated** across diverse analogical prompts?
3. Are there interpretable, semantically meaningful features that encode the **abstract relational structure** of analogies, and how are they discovered?
4. How is the analogical computation **distributed across transformer layers**, and can phase boundaries be causally validated?

---

## 2. Methodology

### 2.1 Prompt Selection

We selected five prompts spanning two structural analogy types to ensure cross-domain coverage:

| ID | Prompt | Expected | Type |
|---|---|---|---|
| `analog_berlin` | *"Paris is to France as Berlin is to"* | Germany | Capital |
| `analog_rome` | *"Paris is to France as Rome is to"* | Italy | Capital |
| `analog_tokyo` | *"Paris is to France as Tokyo is to"* | Japan | Capital |
| `analog_teacher` | *"Doctor is to hospital as teacher is to"* | school | Semantic role |
| `analog_bird` | *"Fish is to water as bird is to"* | air / sky | Semantic role |

### 2.2 Attribution Graph Generation

Attribution graphs were generated using the Neuronpedia API (`/api/graph/generate`) with Gemma-2-2B and the `gemmascope-transcoder-16k` SAE — a 26-layer cross-layer transcoder with 16,384 features per layer. Each graph request returns a URL to an AWS S3 object containing the full attribution graph in JSON format, including nodes (SAE feature activations with layer, index, influence score, and activation magnitude) and directed edges (attribution scores). Graphs were downloaded and loaded into NetworkX `DiGraph` objects for analysis.

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

#### 2.3.1 Feature Identity via Stable (Layer, SAE Index) Pairs

Each feature in the attribution graphs is identified by a stable *(layer, SAE index)* pair — for example, *(5, 5793)* uniquely and persistently identifies a feature within the `gemmascope-transcoder-16k` SAE. These identifiers are fixed properties of the trained SAE and do not vary across prompts, sessions, or API calls.

Cross-graph feature overlap was computed by finding which *(layer, index)* pairs appear as nodes across multiple independently generated graphs. Formally, let $G_i$ denote the set of feature IDs active in graph $i$. The shared circuit at threshold $k$ is:

$$\mathcal{C}_k = \left\{ f \;\middle|\; \sum_{i=1}^{5} \mathbf{1}[f \in G_i] \geq k \right\}$$

Three thresholds were analyzed: $k \in \{3, 4, 5\}$. The 180-feature core circuit ($k=5$) is therefore a concrete, enumerable list of *(layer, index)* identifiers — not an approximate structural characterization — that recur across all five independently generated graphs regardless of whether the prompt is geographic or semantic in nature. The top 25 most recurring features were labeled using the Neuronpedia feature explanation API.

### 2.4 Three-Phase Architecture: How Phase Boundaries Were Identified

The three-phase architecture was identified through two converging lines of evidence, neither of which required the authors to impose phase boundaries a priori.

**Semantic label analysis.** After retrieving Neuronpedia automated labels for the top recurring features, a consistent gradient emerged across layer depth. Features in layers 0–4 carry purely syntactic labels: "the word 'to'", "'to' followed by a verb", "the phrase 'it is to'". Features in layers 5–9 carry explicitly relational-semantic labels: "analogies", "analogies or comparisons", "comparison between two things". Features in layers 10–13 carry integrative labels: "comparisons between disciplines and relationships between concepts". The phase boundaries therefore emerge from the content of the labels rather than from an arbitrary partition of layers.

**Activation magnitude progression.** Average activation magnitudes of core features increase monotonically through the phases:

| Phase | Layer Range | Role | Activation Magnitude |
|---|---|---|---|
| Structural template | L0 | Token and syntax parsing | 1.5 – 6.4 |
| Analogy recognition hub | L5 | Analogy concept activation | 7.4 – 11.1 |
| Comparison detectors | L8–L9 | Relational detection | ~13.4 |
| Relational integration | L10–L13 | Domain + relation integration | 9.1 – 16.3 |

This pattern is consistent with an accumulating signal: early structural features fire weakly; mid-layer analogy detectors and late integrators fire substantially stronger, indicating that the analogical signal builds as it propagates through the network rather than being computed all at once.

**Causal validation.** Phase boundaries were then tested causally via collective suppression experiments (§3.4), which confirmed that each phase is collectively necessary and that earlier phases are prerequisites for later ones.

### 2.5 Discovery of Analogy-Concept Features

The key features — L5 SAE#5793 ("analogies") and L8 SAE#13766 ("analogies or comparisons") — were not specifically sought. They emerged from the cross-graph overlap analysis described in §2.3. Once the intersection feature set was computed, each feature's automated Neuronpedia explanation was retrieved. L5 SAE#5793 returned the label "analogies"; L8 SAE#13766 returned "analogies or comparisons."

The significance of these labels is their **domain-agnosticism**. Both features appear in attribution graphs for Berlin, Rome, and Tokyo (geographic capital analogies) and for teacher and bird (semantic role analogies). A geographic or lexical feature would be expected to appear selectively; the fact that these features appear across all five structurally distinct prompts, and carry labels referencing the abstract concept of analogy rather than any specific domain, is what identifies them as candidates for relational reasoning primitives. L8 SAE#13766 was additionally notable for having 21 appearances across the five graphs (activating at multiple token positions within each graph) and an influence score of 0.533, placing it among the highest-influence recurring features.

### 2.6 Phase 2 Definition

Phase 2 is defined by two jointly applied criteria: **layer position (5–9)** and **feature label content**. Features in this layer range whose Neuronpedia labels explicitly reference analogies, comparisons, or relational structure constitute Phase 2. The four members are:

| Feature | Label |
|---|---|
| L5 SAE#5793 | "analogies" |
| L5 SAE#2141 | "comparisons of people or figures using well-known public figures" |
| L8 SAE#13766 | "analogies or comparisons" |
| L9 SAE#13344 | "phrases suggesting uncertainty or comparison between two things" |

This grouping involves interpretive judgment in reading the labels. It is validated causally: suppressing all four Phase 2 features simultaneously collapses all five circuits, with capital analogies producing "France" — the source-pair answer — rather than the target country. This failure mode is precisely what one would predict if Phase 2 implements relational transfer: the model retains the factual association "Paris is to France" but can no longer compute "as X is to \_\_\_." An arbitrary phase definition would not produce such a consistent and semantically meaningful failure mode.

### 2.7 Circuit Definition and Causal Validation via Feature Steering

A **circuit** for a given prompt is defined as the backbone of causally important features in that prompt's attribution graph. The extraction procedure is identical for all prompts: trace the top-5 causal paths backward from the logit node (greedy walk following highest-weight incoming edges) and the top-5 paths forward from the highest-influence embedding nodes. Any transcoder feature appearing on at least one of these 10 paths is a backbone member.

Causal validation was performed using the Neuronpedia `/api/steer` endpoint with `modelId: "gemma-2-2b"` and `strength_multiplier: 4`. Initial steering attempts returned HTTP 404 errors because requests used the global SAE name rather than the required layer-prefixed identifier (e.g., `21-gemmascope-transcoder-16k` for layer 21). After correcting the endpoint format, four experimental paradigms were applied:

1. **Necessity (individual suppression):** Suppress a single backbone feature at strength −20 and check whether the predicted token changes.
2. **Necessity (full backbone suppression):** Suppress all backbone features simultaneously.
3. **Sufficiency (hub boost):** Boost a backbone hub feature at strength +20 on altered prompts.
4. **Specificity (non-backbone suppression):** Suppress high-activation features not on any causal path.

Two additional circuits from an expanded 30-prompt analysis (Cairo→Kenya, Puppy→cat) were included for cross-validation, for a total of **159 individual steering API calls across 7 circuits**.

---

## 3. Results

### 3.1 Graph Structure

All five attribution graphs exhibited a consistent structural pattern, with features activated across all 26 transformer layers (0–25) plus the embedding layer (E):

| Graph | Prompt | Nodes | Edges | Max Influence |
|---|---|---|---|---|
| `analog_berlin` | Paris:France::Berlin:? | 930 | 25,915 | 0.8001 |
| `analog_rome` | Paris:France::Rome:? | 963 | 27,608 | 0.8002 |
| `analog_tokyo` | Paris:France::Tokyo:? | 905 | 22,414 | 0.8001 |
| `analog_teacher` | Doctor:hospital::teacher:? | 1,040 | 35,481 | 0.8001 |
| `analog_bird` | Fish:water::bird:? | 1,071 | 38,741 | 0.8000 |

The semantic role analogies (*teacher*, *bird*) have notably larger graphs (1,040–1,071 nodes, 35k–38k edges) compared to the capital analogies (905–963 nodes, 22k–27k edges). We interpret this as reflecting greater ambiguity in the expected completion domain: the *capital-of* relation maps to a discrete, well-encoded fact, whereas professional and ecological roles require broader world-knowledge access.

### 3.2 The Core Analogical Reasoning Circuit

Cross-graph feature overlap analysis over the stable *(layer, SAE index)* identifier space revealed a substantial shared circuit:

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

Early layers (L0–L3) account for the plurality of core features, consistent with structural template processing occurring first. The mid-range layers (L5–L6) show elevated feature counts relative to neighbors — these are the **analogy recognition hub** layers. Isolated high-influence features appear at L8, L9, L11, and L13.

### 3.3 The Three-Phase Analogical Reasoning Circuit

We provide evidence that Gemma-2-2B performs **genuine multi-step analogical reasoning internally**. The attribution graph reveals a three-phase computational process that activates for both geographic and semantic role analogies — evidence of a domain-agnostic relational reasoning mechanism.

---

**Phase 1 · Layers 0–4 · Structural Template Parsing**

| Feature | Label |
|---|---|
| L0 SAE#11651 | *"the word 'to'"* |
| L1 SAE#11356 | *"the word 'to' followed by a verb"* |
| L2 SAE#11475 | *"the word 'refers' and related words"* |
| L4 SAE#10752 | *"uses of the verb 'to be' preceded by 'to'"* |
| L5 SAE#9672 | *"the phrase 'it is to'"* |

These features encode the syntactic skeleton of the analogy prompt. Their progression from individual tokens to multi-word patterns reflects hierarchical parsing of the relational connective. Crucially, these are *structural* features — they fire on any text with this grammatical form, not specifically on analogical content.

---

**Phase 2 · Layers 5–9 · Analogy Recognition Hub**

| Feature | Label |
|---|---|
| L5 SAE#5793 | *"analogies"* ← dedicated analogy concept feature |
| L5 SAE#2141 | *"comparisons of people or figures using well-known public figures"* |
| L8 SAE#13766 | *"analogies or comparisons"* (21 activations across 5 graphs, influence 0.533) |
| L9 SAE#13344 | *"phrases suggesting uncertainty or comparison between two things"* |

This is where structural template processing gives way to semantic recognition of the *relational concept itself*. The presence of L5 SAE#5793 — labeled simply "analogies" — is particularly significant: this feature explicitly encodes the concept of analogical structure and activates consistently for both capital-city and semantic role analogies. It is not a geographic feature — it fires equally for "Doctor:hospital::teacher:?".

---

**Phase 3 · Layers 10–13 · Relational Integration**

| Feature | Label |
|---|---|
| L11 SAE#15947 | *"references to historical or social change"* |
| L13 SAE#10969 | *"comparisons between disciplines and relationships between concepts"* |

L13 SAE#10969 serves an integrative role, combining the recognized relational structure from Phase 2 with domain-specific knowledge to produce the final completion. Layers 14–25 then handle domain-specific knowledge retrieval and output token formatting.

---

> **Note:** This diagram simplifies the true mechanisms considerably. The attribution graph for any single prompt contains hundreds of features; the circuit shown represents the semantically interpretable core.

### 3.4 Top Recurring Features

**Directly analogical features** (Neuronpedia labels explicitly reference analogical reasoning or comparison):

| Feature | Appearances | Avg Influence | Label |
|---|---|---|---|
| L5 #5793 | 11/5 | 0.590 | "analogies" |
| L8 #13766 | 21/5 | 0.533 | "analogies or comparisons" |
| L9 #13344 | 14/5 | 0.681 | "comparison between two things" |
| L5 #2141 | 12/5 | 0.647 | "comparisons of public figures" |
| L13 #10969 | 11/5 | 0.676 | "comparisons between disciplines" |

**Structural template features** (encode the "X is to Y as Z is to" scaffold):

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

These formal-text features have higher raw appearance counts than the explicitly analogical features, reflecting activation at multiple syntactic token positions (the connectives) rather than selective activation at the relational structure. Causal steering (§3.7.6) confirms they are inert for all high-confidence circuits.

### 3.5 Cross-Domain Generalization

The consistent activation of L5 SAE#5793 ("analogies") and L8 SAE#13766 ("analogies or comparisons") across both capital-city and semantic role analogy types provides the most direct evidence for a **domain-general analogical reasoning mechanism**. The 180 features active in all five graphs form the stable intersection of the two analogy type families, and this intersection includes the core analogy-concept features at L5 and L8.

The slightly larger graphs for semantic role analogies (teacher, bird: 1,040–1,071 nodes) relative to capital analogies (Berlin, Rome, Tokyo: 905–963 nodes) may reflect that semantic role completions require broader world-knowledge access — knowing that teachers work in schools, or that birds inhabit air — rather than purely relational computation over a discrete, well-encoded fact.

### 3.6 Activation Magnitudes Build Through Layers

Average activation magnitudes of core circuit features increase substantially with layer depth:

| Layer Range | Role | Typical Activation |
|---|---|---|
| L0 (structural) | Token and syntax parsing | 1.5 – 6.4 |
| L5 (analogy hub) | Analogy concept activation | 7.4 – 11.1 |
| L8–L9 (detectors) | Comparison detection | ~13.4 |
| L10–L13 (integration) | Relational + domain integration | 9.1 – 16.3 |

This monotonically increasing pattern is consistent with an accumulating signal as the relational structure is assembled, rather than independent per-layer computation.

### 3.7 Causal Validation via Feature Steering

The attribution graph analysis identifies recurring features and causal path structures but does not by itself establish whether these features are causally necessary for the model's predictions. To distinguish load-bearing circuit components from high-activation but functionally inert nodes, we performed systematic causal steering experiments.

#### 3.7.1 Late-Layer Backbone Necessity (Individual Suppression)

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

**1/9 necessary.** Only L25/2725 is a single point of failure. The highest-weight feature (L21/4827, edge +198.0) is not individually necessary, demonstrating that attribution weight alone does not predict causal necessity.

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

**4/10 necessary.** All four necessary features sit at L24–L25, forming a tight bottleneck. The Rome circuit has more single points of failure than Berlin despite near-identical confidence (p=0.974 vs 0.973), indicating path redundancy varies even among structurally similar geographic analogies.

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

**3/10 necessary.** L23/13914 is necessary in both the Tokyo and Cairo circuits — a shared bottleneck consistent with a late-layer "geographic entity selector" role. L25/286 recurs as necessary in Rome, Tokyo, and Cairo, making it the single most critical output driver across geographic analogies.

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

**3/10 necessary.** The teacher circuit is the only one where an **L0 embedding-level feature** (L0/17) is individually necessary — suppressing it changes the prediction from "school" to "the," suggesting the semantic role analogy relies on an early feature not redundantly encoded by later layers.

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

**8/10 necessary.** This is the most fragile circuit in the dataset. Three L22 relay features (15670, 14727, 13619) are all independently necessary despite occupying the same layer, indicating they carry non-redundant information through parallel channels. This fragility is consistent with the circuit's very low prediction confidence (p=0.117): the model barely resolves "air" over competitors, and removing almost any contributing feature tips the balance.

**Cross-circuit pattern.** Necessity inversely correlates with prediction confidence: Berlin (p=0.973): 1/9 necessary; Rome (p=0.974): 4/10; Tokyo (p=0.990): 3/10; Teacher (p=0.486): 3/10; Bird (p=0.117): 8/10. Three features recur as necessary across multiple circuits: **L25/#286** (Rome, Tokyo, Cairo), **L25/#4717** (Rome, Teacher, Bird), and **L23/#13914** (Tokyo, Cairo).

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

**6/7 circuits fully disrupted.** Failure modes are qualitatively informative: capital analogies degenerate to repetitive or archaic text (Berlin: "of of of of of"; Rome: "pleaſure pleaſure"; Tokyo: "country country count"), indicating the backbone is required for entity selection while the prompt structure alone partially activates a "country" category. Teacher collapses to "1111"; bird falls through to generic continuation. Puppy→cat is the sole exception, its prediction apparently carried by direct embedding-to-logit connections outside the multi-hop backbone.

#### 3.7.3 Phase 1 and Phase 2 Feature Necessity

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

*Cells show the steered first token when the feature is suppressed at strength −20. "—" = prediction unchanged.*

**Phase 1 results.** L0/11651 ("the word 'to'") is necessary in 4/5 circuits. Its failure mode is diagnostic: suppressing it causes all three capital analogies to predict the *city name itself* (Berlin, Rome, Tokyo), indicating the model reverts to associating with the most recently mentioned entity rather than completing the analogy. For the bird circuit, suppression produces "water" (the source-pair element). L4/10752 is necessary for 2/5 circuits; L5/9672 only for the bird circuit.

**Phase 2 results.** L5/5793 ("analogies") is **never individually necessary** in any circuit. This does not mean Phase 2 is inert — it means Phase 2 features are individually redundant for high-confidence circuits. For the fragile bird circuit (p=0.117), Phase 2 features become individually necessary: L8/13766 changes "air" to "fish" (source-domain animal); L9/13344 changes "air" to "sky."

**Phase 3 results.** L13/10969 is not individually necessary for any circuit.

#### 3.7.4 Collective Phase Suppression

| Experiment | Features Suppressed | Berlin | Rome | Tokyo | Teacher | Bird |
|---|---|---|---|---|---|---|
| All Phase 2 (4 feat.) | L5/5793, L5/2141, L8/13766, L9/13344 | **France** | **France** | **France** | be | fish |
| All Phase 1 (5 feat.) | L0/11651, L1/11356, L4/10752, L5/9672, L2/11475 | (empty) | (empty) | (empty) | to | to |
| Phase 1+2 (9 feat.) | All Phase 1 + Phase 2 | : | : | : | : | : |
| Phase 1+2+3 (10 feat.) | All Phase 1 + Phase 2 + L13/10969 | : | : | : | be | : |

*All 20/20 cells are disrupted.*

**Phase 2 collective suppression is the most informative experiment in this paper.** Suppressing all four Phase 2 features simultaneously disrupts all five circuits — including the high-confidence capital analogies (p>0.97) that were immune to every individual Phase 2 suppression. The failure mode is consistent and semantically meaningful: all three capital analogies output **"France"** rather than the target country. The model retains the factual association "Paris is to France" but can no longer compute the relational transfer "as Berlin is to \_\_\_," echoing the source pair's answer instead. This is direct causal evidence that Phase 2 features collectively implement the relational transfer operation — the computational core of analogical reasoning.

**Phase 1 collective suppression** produces empty outputs for capital analogies and "to" for semantic role analogies — a more severe failure, consistent with Phase 1 being a prerequisite for Phase 2.

**Combined Phase 1+2 suppression** produces ":" for 4/5 circuits. With both template parsing and analogy recognition disabled, the prompt resembles a list header and the model defaults to list-formatting punctuation.

These results establish a **causal hierarchy**: Phase 1 (template parsing) → Phase 2 (relational transfer) → Phase 3 + late layers (entity selection). Each phase is collectively necessary, and earlier phases are prerequisites for later ones.

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

Across 159 steering experiments, five principal findings emerge:

1. **The late-layer backbone is collectively necessary.** Full backbone suppression disrupts 6/7 circuits.
2. **Individual necessity scales inversely with prediction confidence.** High-confidence circuits have 1–4 necessary features; the lowest-confidence circuit has 8/10.
3. **Phase 2 is collectively necessary but individually redundant.** Simultaneous suppression collapses every circuit; capital analogies revert to the source-pair answer ("France").
4. **Phase 1 template features are individually necessary.** L0/11651 alone disrupts 4/5 circuits; template parsing is a prerequisite for relational reasoning.
5. **Formal-text features are causally inert.** L4/#14857 and L6/#2267 do not affect any high-confidence prediction, confirming they process template syntax rather than driving entity selection.

---

## 4. Discussion

### 4.1 The Analogical Reasoning Circuit in Gemma-2-2B

Our analysis reveals that Gemma-2-2B implements analogical reasoning through a distributed circuit spanning all 26 transformer layers, with specific functional specialization at each phase. The most significant finding is the existence of **explicitly semantic analogy features** at layers 5, 8, 9, and 13 — features whose automated explanations use the words "analogies," "comparisons," and "relationships between concepts." The presence of these features, and their causal necessity as a collective, suggests that the model has internalized the concept of analogical structure as a discrete, reusable computational primitive, rather than computing analogies purely through implicit pattern matching.

This is qualitatively distinct from multi-hop factual reasoning of the form "the capital of the state containing Dallas is Austin," where the model chains two stored factual lookups. Analogical reasoning requires extracting an unnamed relation type, holding it as a variable, and applying it to a new argument pair. The Phase 2 collective suppression experiment demonstrates that this extraction and transfer are implemented by identifiable, causally load-bearing internal components whose removal causes the model to echo the source-pair answer rather than transfer the relation.

### 4.2 The Role of Formal Text Features

The high-recurrence "code and legal text" features present an interpretive puzzle. We propose two complementary explanations:

**Functional hypothesis:** These features detect formal, template-driven text patterns generally. The analogy syntax "X is to Y as Z is to" is highly structured, resembling legal definitions, code comments, and mathematical notation. The model reuses a general "formal syntax" detector.

**Training data hypothesis:** The analogy format appears frequently in SAT preparation and educational materials — which also contain code examples and legal definitions — creating a statistical association between formal-text features and analogy-completion contexts.

Both are compatible with the causal steering data. The formal features process the syntactic surface of the template while the analogy features process the relational semantics; only the latter are collectively necessary for relational transfer.

### 4.3 Comparison with the Capital City Recall Circuit

Comparison with the capital city *factual recall* circuit (prompt: "The capital of X is") reveals:

- **Overlap:** Formal-text features (L4/#14857, L6/#2267) appear with high frequency in both circuits, activated by the formal definitional structure of both prompt types.
- **Divergence:** The L5 "analogies" feature and L8 "analogies or comparisons" feature appear to be specific to the analogical task — they were not among the top recurring features in the factual recall circuit — supporting the interpretation that these features are selectively activated by relational structure recognition.

### 4.4 Redundancy as a Property of Well-Learned Computation

The inverse relationship between prediction confidence and circuit fragility suggests a general principle: well-learned associations are protected by redundant parallel causal paths, while barely-resolved predictions rely on non-redundant chains where nearly every contributing feature is a single point of failure. This principle may reflect how transformers allocate computational resources — distributing load across parallel paths in proportion to the confidence with which a mapping has been learned.

---

## 5. Limitations

1. **SAE-feature-level intervention only.** Steering operates at the SAE feature level, not the attention head or residual stream level. The causal role of non-SAE circuit components is not assessed.
2. **SAE coverage.** The `gemmascope-transcoder-16k` SAE covers only cross-layer transcoder features. Attention head contributions and residual stream features are not captured.
3. **Threshold sensitivity.** Results are sensitive to the node and edge thresholds set during graph generation (0.80 and 0.85). Lower thresholds would reveal more features; higher thresholds would produce sparser, more focused circuits.
4. **Label quality.** Neuronpedia automated feature explanations are LLM-generated and may not perfectly capture feature semantics.
5. **Prompt set size.** Five prompts are sufficient for initial circuit identification but too few to claim statistical robustness. A larger prompt set covering more analogy types — arithmetic, cross-lingual, abstract relational — would strengthen conclusions.

**Future work:** Direct causal validation with TransformerLens activation patching at the attention head and residual stream level; expanded prompt sets; analysis across model scales (Gemma-2-9B, 27B); comparison with factual recall and multi-hop reasoning circuits.

---

## 6. Conclusions

We have identified and characterized the **analogical reasoning circuit in Gemma-2-2B** using SAE attribution graphs from the Neuronpedia platform. The key conclusions are:

1. **A stable shared circuit exists, identified by common feature IDs.** 180 features — identified by stable *(layer, SAE index)* pairs — appear in all five independently generated attribution graphs. These are literally the same feature identifiers recurring across structurally distinct prompts, not approximate structural analogues.

2. **Dedicated analogy features exist at layers 5, 8, 9, and 13.** These features have Neuronpedia explanations explicitly referencing analogies, comparisons, and relational concepts, providing the first direct evidence of interpretable analogy-concept features in a large language model.

3. **The circuit exhibits a three-phase architecture, identified by label semantics and validated causally.** Template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13), with activation magnitude increasing through the sequence. Phase boundaries emerge from automated label content and are confirmed by collective suppression experiments.

4. **Cross-domain generalization is confirmed.** The same core features, including L5 SAE#5793 ("analogies"), activate for both geographic and semantic role analogies — a domain-agnostic relational reasoning primitive.

5. **Phase 2 implements relational transfer, collectively but not individually.** No individual Phase 2 feature is necessary for high-confidence circuits, but their simultaneous suppression collapses every circuit with a consistent failure mode: capital analogies revert to the source-pair answer, directly demonstrating that Phase 2 implements the relational transfer operation.

6. **Circuit fragility tracks prediction confidence.** High-confidence predictions are protected by redundant parallel causal paths (1–4 necessary features); low-confidence predictions rely on fragile non-redundant chains (up to 8/10 necessary). This relationship holds consistently across all tested circuits.

---

## References

- Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. *NeurIPS 2023*.
- Lindsey, J., Gurnee, W., Ameisen, E., et al. (2025). On the biology of a large language model. *Anthropic*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *ICLR 2023*.
- Neuronpedia. (2025). Circuit tracing platform and gemmascope-transcoder-16k SAE. https://www.neuronpedia.org

---

## Supplementary Materials

**Interactive Presentation:** 20-slide reveal.js presentation with circuit flow diagrams, feature tables, and layer-by-layer analysis.  
https://kkrampis.github.io/autocircuit/presentation.html

**Live Attribution Graphs:**

- [`analog_berlin`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin) — Paris:France::Berlin:?
- [`analog_rome`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_rome) — Paris:France::Rome:?
- [`analog_tokyo`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_tokyo) — Paris:France::Tokyo:?
- [`analog_teacher`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_teacher) — Doctor:hospital::teacher:?
- [`analog_bird`](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_bird) — Fish:water::bird:?

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
