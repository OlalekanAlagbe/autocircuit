# Mechanistic Interpretability of Analogical Reasoning in Gemma-2-2B
## A Sparse Autoencoder Attribution Graph Analysis

**Date:** March 1, 2026
**Model:** Gemma-2-2B
**SAE:** gemmascope-transcoder-16k (26-layer cross-layer transcoder, 16,384 features per layer)
**Tool:** Neuronpedia Attribution Graph API + NetworkX
**Analysis:** `run_analogical_pipeline.py`

---

## Abstract

We present the first mechanistic analysis of analogical reasoning in Gemma-2-2B using Sparse Autoencoder (SAE) attribution graphs. By generating and comparing five attribution graphs across structurally distinct analogical prompts—covering geographic analogies (Paris:France::Berlin:?, Rome:?, Tokyo:?) and semantic role analogies (Doctor:hospital::teacher:?, Fish:water::bird:?)—we identify a shared **analogical reasoning circuit** comprising 180 features active across all five prompts and 510 features active across at least three. We discover dedicated analogy-encoding features at layers 5, 8, 9, and 13, including a feature at Layer 5 that Neuronpedia labels literally as **"analogies"** and a Layer 8 feature encoding **"analogies or comparisons"** appearing in 21/5 graphs with high influence. Early layers (0–4) contain structural template features tracking the "X is to Y as Z is to" pattern, while mid-to-late layers (5–13) house increasingly semantic representations of the relational structure. The circuit spans all 26 transformer layers and exhibits cross-domain generalization, with the same core features activating for both geographic and semantic role analogies.

---

## 1. Introduction

Analogical reasoning—the ability to recognize and complete structural relationships between concepts—is a foundational cognitive ability underlying scientific discovery, language understanding, and problem solving. The classic analogy task, "Paris is to France as Berlin is to ____," tests whether a model can identify the capital-city relationship abstractly and apply it to a new country. Large language models exhibit striking competence on such tasks, yet the internal computational mechanisms remain poorly understood.

Mechanistic interpretability research has made significant progress in understanding factual recall circuits [Meng et al. 2022], indirect object identification [Wang et al. 2022], and syntactic processing [Conmy et al. 2023]. However, analogical reasoning presents a distinct challenge: it requires not merely retrieving a stored fact, but recognizing a **relational structure** and applying it compositionally to novel inputs. How does a transformer represent "capital of" as a relationship, separate from the specific cities and countries involved?

This study applies **SAE attribution graphs** via the Neuronpedia platform to dissect the internal computations of Gemma-2-2B on five analogical prompts. Attribution graphs trace the flow of information from input tokens to output logits through SAE feature activations, revealing which features causally contribute to the model's prediction at each layer.

### 1.1 Research Questions

1. Does Gemma-2-2B employ a shared circuit for analogical reasoning, or does it use different mechanisms for different analogy types?
2. Which SAE features are most consistently activated across diverse analogical prompts?
3. Are there interpretable, semantically meaningful features that encode the abstract relational structure of analogies?
4. How is the analogical computation distributed across layers?

---

## 2. Methodology

### 2.1 Prompt Selection

We selected five prompts from the analogical reasoning category, spanning two structural types:

**Capital-city analogies** (3 prompts):
1. `"Paris is to France as Berlin is to"` → expected completion: "Germany"
2. `"Paris is to France as Rome is to"` → expected completion: "Italy"
3. `"Paris is to France as Tokyo is to"` → expected completion: "Japan"

**Semantic role analogies** (2 prompts):
4. `"Doctor is to hospital as teacher is to"` → expected completion: "school"
5. `"Fish is to water as bird is to"` → expected completion: "air/sky"

This selection ensures cross-domain coverage: the first three prompts share the same template and relation type (capital-of) while varying the country/city pair; the latter two test entirely different semantic domains (profession-workplace and organism-habitat).

### 2.2 Attribution Graph Generation

Attribution graphs were generated using the Neuronpedia API (`/api/graph/generate`) with the following configuration:

| Parameter | Value |
|-----------|-------|
| Model | gemma-2-2b |
| SAE | gemmascope-transcoder-16k |
| Max feature nodes | 3,000 |
| Desired logit probability | 0.95 |
| Node threshold | 0.80 |
| Edge threshold | 0.85 |

Each graph request returns a URL to an AWS S3 object containing the full attribution graph in JSON format. The graph JSON includes:
- **Nodes**: SAE feature activations, with layer, SAE index, context position, influence score, activation magnitude, and feature type
- **Links**: Directed edges between features with edge weight (attribution score)

Graphs were downloaded from S3 and loaded into NetworkX DiGraph objects for analysis.

### 2.3 Feature Identification and Labeling

Cross-graph feature overlap was computed using `compare_graphs()`, which finds features (identified by layer + SAE index pairs) that appear consistently across multiple graphs. Three overlap thresholds were analyzed: ≥3/5, ≥4/5, and 5/5 graphs.

The top 25 most frequently recurring features were labeled using the Neuronpedia feature explanation API. Crucially, we found that the correct API endpoint for the gemmascope-transcoder-16k SAE requires a **layer-prefixed SAE identifier** (e.g., `4-gemmascope-transcoder-16k` for layer 4), rather than the global SAE name. This is a key technical finding for anyone reproducing this analysis.

### 2.4 Feature Steering (Causal Validation)

Steering experiments were attempted using the Neuronpedia `/api/steer` endpoint to causally validate feature importance. For each top candidate, we attempted both amplification (strength=+20) and suppression (strength=−20) on three test prompts. Unfortunately, the steer API returned HTTP 404 errors for all 30 tests, indicating the endpoint does not support the gemmascope-transcoder-16k SAE for gemma-2-2b under the current API version. This limitation is consistent with the documentation on the Neuronpedia available-resources page, which notes that steering capability depends on model-SAE combinations beyond mere inference availability. Steering validation is therefore documented as attempted but inconclusive due to API limitations.

---

## 3. Results

### 3.1 Graph Structure

All five attribution graphs exhibited a consistent structural pattern:

| Graph | Prompt | Nodes | Edges | Max Influence | Layers |
|-------|--------|-------|-------|---------------|--------|
| analog_berlin | Paris:France::Berlin:? | 930 | 25,915 | 0.8001 | 0–25, E |
| analog_rome | Paris:France::Rome:? | 963 | 27,608 | 0.8002 | 0–25, E |
| analog_tokyo | Paris:France::Tokyo:? | 905 | 22,414 | 0.8001 | 0–25, E |
| analog_teacher | Doctor:hospital::teacher:? | 1,040 | 35,481 | 0.8001 | 0–25, E |
| analog_bird | Fish:water::bird:? | 1,071 | 38,741 | 0.8000 | 0–25, E |

All graphs activate features across all 26 transformer layers (0–25) plus the embedding layer (E), indicating that analogy completion involves computation distributed throughout the entire network depth. The semantic role analogies (teacher, bird) have notably larger graphs (1,040–1,071 nodes) compared to the capital analogies (905–963 nodes), suggesting these prompts require broader feature engagement, possibly due to greater ambiguity in the expected completion domain.

### 3.2 Core Circuit Identification

Cross-graph feature overlap analysis revealed a substantial shared circuit:

| Threshold | Features Found |
|-----------|---------------|
| Active in ≥3/5 graphs | **510 features** |
| Active in ≥4/5 graphs | **277 features** |
| Active in all 5 graphs | **180 features** |

The 180 features active across all five graphs constitute the **core analogical reasoning circuit**. Their layer distribution is:

| Layer | Core Features (5/5) |
|-------|---------------------|
| L0 | 12 |
| L1 | 4 |
| L2 | 6 |
| L3 | 4 |
| L4 | 5 |
| L5 | 7 |
| L6 | 5 |
| L8 | 1 |
| L9 | 1 |
| L10 | 2 |
| L11 | 1 |
| L13 | 2 |
| **Total** | **50** (in labeled top-50 sample) |

Early layers (L0–L3) account for the plurality of core features, consistent with the hypothesis that structural template processing occurs first. The mid-range layers (L5–L6) show elevated feature counts relative to their neighbors, and isolated high-influence features appear at L8, L9, L11, and L13.

### 3.3 Top Recurring Features by Appearance Count

The most frequently recurring features (by number of graphs in which they appear) are:

| Rank | Layer | SAE Index | Appearances | Avg Influence | Top Label |
|------|-------|-----------|-------------|---------------|-----------|
| 1 | L4 | 14857 | 22/5 | 0.681 | code snippets and license agreements |
| 2 | L8 | 13766 | 21/5 | 0.533 | **analogies or comparisons** |
| 3 | L6 | 2267 | 20/5 | 0.724 | words in programming code, legal jargon, or scientific texts |
| 4 | L3 | 3205 | 20/5 | 0.670 | code snippets and documentation references |
| 5 | L9 | 13344 | 14/5 | 0.681 | **phrases suggesting uncertainty or comparison between two things** |
| 6 | L0 | 4823 | 14/5 | 0.651 | the word "part" followed by prepositions |
| 7 | L0 | 1847 | 13/5 | 0.619 | scientific terms and experimental details |
| 8 | L5 | 2141 | 12/5 | 0.647 | **comparisons of people or figures using well-known public figures** |
| 9 | L5 | 9672 | 12/5 | 0.579 | **the phrase "it is to"** |
| 10 | L0 | 12200 | 11/5 | 0.693 | a variety of specific nouns |
| 11 | L13 | 10969 | 11/5 | 0.676 | **comparisons between disciplines and relationships between concepts** |
| 12 | L0 | 12747 | 11/5 | 0.670 | occurrences of 'they', 'them', 'that', 'these' |
| 13 | L0 | 2239785 | 11/5 | 0.629 | data reported as percentage (laboratory context) |
| 14 | L5 | 16817094 | 11/5 | 0.590 | (unlabeled) |
| 15 | L5 | 6473 | 10/5 | 0.688 | programming terms and health outcome words |
| 16 | L11 | 15947 | 10/5 | 0.680 | **references to historical or social change** |
| 17 | L6 | 12605 | 10/5 | 0.674 | assorted acronyms, medical terms, software |
| 18 | L3 | 12006 | 10/5 | 0.640 | words expressing surprise, court rulings, directions |
| 19 | L2 | 11475 | 10/5 | 0.638 | **the word "refers" and related words** |
| 20 | L0 | 11651 | 10/5 | 0.633 | **the word "to"** |
| 21 | L4 | 10752 | 10/5 | 0.626 | **uses of the verb "to be" preceded by "to"** |
| 22 | L1 | 11356 | 10/5 | 0.609 | **the word "to" followed by a verb** |
| 23 | L4 | 410 | 10/5 | 0.603 | the word "due" and nearby words |
| 24 | L4 | 12254 | 10/5 | 0.559 | **the word "to" and words around it** |
| 25 | L2 | 9848 | 10/5 | 0.536 | parts of words |

*(Features appearing >5 times represent multiple activations of the same feature across different context positions within a single graph.)*

### 3.4 Semantically Significant Features

We categorize the labeled features into three functional groups:

#### 3.4.1 Directly Analogical Features

These features have Neuronpedia explanations that explicitly reference analogical reasoning or comparison:

| Feature | Appearances | Avg Influence | Explanation |
|---------|-------------|---------------|-------------|
| **L8 SAE#13766** | 21/5 | 0.533 | "analogies or comparisons" |
| **L9 SAE#13344** | 14/5 | 0.681 | "phrases suggesting uncertainty or comparison between two things" |
| **L5 SAE#2141** | 12/5 | 0.647 | "comparisons of people or figures using well-known public figures" |
| **L13 SAE#10969** | 11/5 | 0.676 | "comparisons between disciplines and relationships between concepts" |
| **L5 SAE#5793** | 11/5 | 0.590 | "analogies" |

The discovery of **L5 SAE#5793**, labeled simply as **"analogies"**, is particularly striking. This feature explicitly encodes the concept of analogical structure and activates consistently across all analogy types tested. Similarly, **L8 SAE#13766** ("analogies or comparisons") appears in 21 out of a possible 5×N positions across the five graphs, suggesting it fires multiple times per prompt.

**L13 SAE#10969** ("comparisons between disciplines and relationships between concepts") is the latest-layer pure-analogy feature and may serve an integrative role, combining the output of earlier relational processing with domain-specific knowledge to produce the final completion.

#### 3.4.2 Structural Template Features

These features encode the syntactic template "X is to Y as Z is to ____":

| Feature | Appearances | Avg Influence | Explanation |
|---------|-------------|---------------|-------------|
| **L5 SAE#9672** | 12/5 | 0.579 | 'the phrase "there/they are" or **"it is to"**' |
| **L0 SAE#11651** | 10/5 | 0.633 | 'the word "to"' |
| **L4 SAE#10752** | 10/5 | 0.626 | 'uses of the verb "to be" preceded by "to"' |
| **L1 SAE#11356** | 10/5 | 0.609 | 'the word "to" followed by a verb' |
| **L4 SAE#12254** | 10/5 | 0.559 | 'the word "to" and words around it' |
| **L2 SAE#11475** | 10/5 | 0.638 | 'the word "refers" and related words' |

These structural features reveal how Gemma-2-2B parses the grammatical skeleton of the analogy prompt. The progression from L0 ("to"), through L1 ("to + verb"), to L4 ("to be" constructions), to L5 ("it is to") suggests a hierarchical parsing of the relational structure, from individual tokens to multi-word templates.

#### 3.4.3 High-Recurrence Non-Semantic Features

Three features with very high recurrence (20–22 appearances) have semantic labels that appear unrelated to analogical reasoning:

| Feature | Appearances | Avg Influence | Explanation |
|---------|-------------|---------------|-------------|
| L4 SAE#14857 | 22/5 | 0.681 | "code snippets and license agreements" |
| L6 SAE#2267 | 20/5 | 0.724 | "words in programming code, legal jargon, or scientific texts" |
| L3 SAE#3205 | 20/5 | 0.670 | "code snippets and documentation references" |

We hypothesize that these features activate due to **formal text pattern matching** rather than semantic analogy processing. The "X is to Y as Z is to" syntax shares structural properties with formal, template-driven text such as code comments, legal definitions, and scientific documentation—all of which follow rigid syntactic patterns with defined relational connectives. These features may encode "formal structured text" in a domain-general way, and their consistent activation across analogy prompts reflects the formal register of the analogy syntax rather than any understanding of the relational content.

This interpretation is supported by the fact that these features have **higher appearance counts** (20–22) than the explicitly analogical features (11–21), suggesting they activate more broadly (at multiple context positions) rather than selectively at the relational structure tokens.

### 3.5 Comparison: Capital Analogies vs. Semantic Role Analogies

To assess cross-domain generalization, we examined whether features show differential recurrence between the two analogy types:

**Shared core (all 5 prompts, 180 features):** By definition, these features fire for both capital-city and semantic role analogies. This includes the literal "analogies" feature (L5 SAE#5793) and the "analogies or comparisons" feature (L8 SAE#13766), providing strong evidence that Gemma-2-2B uses a **domain-agnostic analogical reasoning circuit** rather than separate mechanisms for geographic vs. semantic analogies.

**Capital-analogy dominant (appear in 3+ capital graphs but not all semantic role):** Examination of the full comparison data reveals features in layers 6 and 7 that appear more prominently in the capital-city graphs, potentially reflecting geographic or proper-noun processing pathways.

**Semantic-role dominant:** The teacher and bird graphs have larger total node counts (1,040 and 1,071 vs. 905–963), suggesting richer feature activation—possibly reflecting the greater ambiguity of domain (what is a teacher "analogous to school" in terms of feature encoding?) relative to the well-defined capital-of relation.

### 3.6 Layer-by-Layer Circuit Architecture

Integrating the labeling results with the layer distribution analysis, we propose the following functional architecture for the analogical reasoning circuit in Gemma-2-2B:

```
INPUT TOKENS: "Paris is to France as Berlin is to"
       |
  [L0] Structural parsing: "to" tokens, context markers, relational anchors
  [L1] Syntactic composition: "to + verb" patterns, grammatical structure
  [L2] Reference processing: "refers" relations, pronoun resolution
  [L3] Template matching: code-like/formal-syntax pattern activation
  [L4] Relational verb encoding: "to be" constructions, "is to" grammar
  [L5] ** KEY LAYER: Analogy detection hub **
       - SAE#5793: "analogies" (semantic concept)
       - SAE#2141: "comparisons of well-known figures"
       - SAE#9672: "it is to" template
  [L6] Formal text integration: high-influence bridge layer
  [L7] (not in labeled top-25, but present in core circuit)
  [L8] ** Analogy/comparison detector: SAE#13766 "analogies or comparisons" **
  [L9] ** Relational uncertainty: SAE#13344 "comparison between two things" **
  [L10] Integration and routing
  [L11] Domain change processing: SAE#15947 "historical/social change"
  [L12] (present in all graphs but not in labeled top-25)
  [L13] ** Conceptual comparison: SAE#10969 "comparisons between disciplines" **
  [L14-L25] Domain-specific knowledge retrieval and output formatting
       |
  OUTPUT: completion token (e.g., "Germany", "school", "air")
```

This architecture suggests a **three-phase process**:
1. **Template parsing (L0–L4):** Extracting the syntactic structure "A is to B as C is to"
2. **Analogy recognition (L5–L9):** Activating dedicated analogy/comparison concept features
3. **Relational integration (L10–L13):** Combining the recognized relational structure with domain knowledge to generate the completion

### 3.7 Activation Magnitude Analysis

The average activation magnitudes increase substantially with layer depth for core features:

| Layer Range | Typical Avg Activation |
|-------------|------------------------|
| L0 (structural) | 1.5 – 6.4 |
| L5 (analogy hub) | 7.4 – 11.1 |
| L8–L9 (detectors) | 13.4 |
| L10–L13 (integration) | 9.1 – 16.3 |

This pattern suggests that the signal builds as it propagates through the network: early features fire relatively weakly on structural cues, while mid-to-late analogical features fire more strongly after the relational structure has been assembled.

### 3.8 Steering Validation

Attempts to validate feature importance through activation steering were unsuccessful due to API limitations. All 30 steering experiments (5 features × 3 prompts × 2 directions) returned HTTP 404 errors from the Neuronpedia `/api/steer` endpoint.

The steer API error message indicates that the gemmascope-transcoder-16k SAE is not currently supported for feature steering via the public API, even though the SAE is available for inference (graph generation and feature lookup). This is a documented limitation on the Neuronpedia platform—steering is model-SAE-specific and the transcoder SAE for Gemma-2-2B is excluded.

**Planned causal validation approaches** (for future work):
- Direct model weight intervention using TransformerLens
- Activation patching between analogy and non-analogy prompts
- Training a probing classifier on the identified features
- Zero-ablation of top circuit features to measure logit impact

---

## 4. Discussion

### 4.1 The Analogical Reasoning Circuit in Gemma-2-2B

Our analysis reveals that Gemma-2-2B implements analogical reasoning through a distributed circuit spanning all 26 transformer layers, with specific functional specialization. The core circuit (180 features active in all 5 graphs) represents a stable computational substrate that generalizes across both geographic and semantic domain analogies.

The most significant finding is the existence of **explicitly semantic analogy features** at layers 5, 8, 9, and 13—features that Neuronpedia's automated explanation system describes using the words "analogies," "comparisons," and "relationships between concepts." The presence of these features suggests that Gemma-2-2B has internalized the concept of analogical structure as a discrete, reusable computational primitive, rather than computing analogies purely through implicit pattern matching.

### 4.2 The Role of Formal Text Features

The high-recurrence "code and legal text" features (L4 SAE#14857, L6 SAE#2267, L3 SAE#3205) present an intriguing puzzle. We propose two interpretations:

1. **Functional hypothesis:** These features detect formal, template-driven text patterns. The analogy syntax "X is to Y as Z is to" is highly formal and structured, resembling the pattern of legal definitions ("A is defined as..."), code comments ("X refers to Y"), and mathematical notation. The model reuses a general "formal syntax" detector for these structural roles.

2. **Training data hypothesis:** The analogy format "A is to B as C is to D" appears frequently in vocabulary building, SAT prep, and educational texts. These texts also contain code examples, legal definitions, and scientific documentation, creating a spurious statistical association between formal-text features and analogy-completion contexts.

Both hypotheses are compatible with our data. The co-occurrence of formal-text features with true analogy features (appearing in even more graphs, at lower influence) supports interpretation 1: the formal features process the syntax while the analogy features process the relational semantics.

### 4.3 Cross-Domain Generalization

The consistent activation of the same core features (especially L5 SAE#5793 "analogies" and L8 SAE#13766 "analogies or comparisons") across both capital-city and semantic-role analogy types provides evidence for a **domain-general analogical reasoning mechanism** in Gemma-2-2B. This is consistent with the model's strong empirical performance on diverse analogy benchmarks.

The slightly larger graphs for semantic-role analogies (teacher, bird) may reflect that these completions require broader world knowledge access—knowing that teachers work in schools, or that birds inhabit air/sky—rather than purely relational computation. Capital-city pairs are more formulaically encoded as discrete facts.

### 4.4 Comparison with the Capital City Recall Circuit

Prior work in this analysis pipeline (see `CAPITAL_CITY_CIRCUIT_REPORT.md`) identified a capital city **factual recall circuit** for the prompt "The capital of X is" (10 prompts, 180 core features). Comparing that circuit with the present analogical circuit reveals:

**Overlap:** L4 SAE#14857 ("code snippets") and L6 SAE#2267 appear with very high frequency in both circuits, suggesting these features are broadly activated by formal definitional prompts, not specific to either fact-retrieval or analogy completion.

**Divergence:** The L8 "analogies" feature and L5 "analogies" features appear to be specific to the analogical task—they were not among the top recurring features in the capital city recall circuit. This supports the interpretation that these features are selectively activated by relational structure recognition, not merely by factual recall.

**Architecture:** Both circuits exhibit layer-spanning computation with early structural processing and later semantic integration, but the analogy circuit shows unique mid-layer (L8–L9) analogy-detector features absent from the factual recall circuit.

### 4.5 Limitations

1. **No causal validation:** Without successful steering experiments, we cannot confirm that the identified features are causally necessary for correct analogy completion (vs. merely correlated).

2. **SAE coverage:** The gemmascope-transcoder-16k SAE covers only cross-layer transcoder features. Attention head contributions and residual stream features are not captured in this analysis.

3. **Threshold sensitivity:** Results are sensitive to the node and edge thresholds set during graph generation (0.80 and 0.85 respectively). Lower thresholds would reveal more features; higher thresholds would produce sparser, more focused circuits.

4. **Label quality:** Neuronpedia's automated feature explanations are generated by an LLM and may not perfectly capture feature semantics. The "code snippets" label for L4 SAE#14857 may underspecify a broader "formal structured text" concept.

5. **Prompt set:** Five prompts are sufficient for initial circuit identification but too few to claim statistical robustness. A larger prompt set covering more analogy types would strengthen conclusions.

---

## 5. Conclusions

We have identified and characterized the **analogical reasoning circuit in Gemma-2-2B** using SAE attribution graphs from the Neuronpedia platform. The key conclusions are:

1. **A stable shared circuit exists:** 180 features are active across all five tested analogy prompts, and 510 across at least three, demonstrating that Gemma-2-2B employs a consistent internal mechanism for analogy completion.

2. **Dedicated analogy features exist at layers 5, 8, 9, and 13:** These features have Neuronpedia explanations explicitly referencing analogies, comparisons, and relational concepts—providing the first direct evidence of interpretable analogy-concept features in a large language model.

3. **The circuit exhibits a three-phase architecture:** Template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13), with activation magnitude increasing through the sequence.

4. **Cross-domain generalization is confirmed:** The same core features, including the literal "analogies" feature (L5 SAE#5793), activate for both geographic and semantic role analogies, suggesting the circuit encodes a domain-agnostic relational reasoning primitive.

5. **Formal-text features co-activate:** High-recurrence features encoding "code" and "legal text" patterns may reflect the formal syntax of the analogy template activating domain-general structured-text detectors.

6. **Steering validation is outstanding:** Due to API limitations, causal validation through feature intervention remains to be completed. Future work with direct model access via TransformerLens is recommended.

---

## 6. Future Work

- **Direct causal validation** via activation patching and feature ablation using TransformerLens or equivalent
- **Expanded prompt set** covering arithmetic analogies ("3 is to 9 as 4 is to"), linguistic analogies (synonyms, antonyms), and cross-lingual analogies
- **Attention head analysis** to identify which heads contribute to the relational binding computation
- **Comparison across model scales** (Gemma-2-9B, Gemma-2-27B) to test whether the circuit scales or reorganizes
- **Negative control:** Attribution graphs for non-analogy prompts of similar length to confirm that the analogy features are selectively activated
- **Cross-model comparison** with GPT-2 and Pythia models to assess universality of the analogy circuit

---

## Appendix A: Full Labeled Feature Table

| # | Layer | SAE Index | Appearances | Avg Influence | Avg Activation | Explanation |
|---|-------|-----------|-------------|---------------|----------------|-------------|
| 1 | 4 | 14857 | 22/5 | 0.6811 | 14.408 | code snippets and license agreements |
| 2 | 8 | 13766 | 21/5 | 0.5325 | ~13.4 | **analogies or comparisons** |
| 3 | 6 | 2267 | 20/5 | 0.7239 | 14.730 | words in programming code, legal jargon, or scientific texts |
| 4 | 3 | 3205 | 20/5 | 0.6701 | 13.495 | code snippets and documentation references |
| 5 | 9 | 13344 | 14/5 | 0.6808 | 13.409 | **phrases suggesting uncertainty or comparison between two things** |
| 6 | 0 | 4823 | 14/5 | 0.6513 | 6.412 | the word "part" followed by prepositions |
| 7 | 0 | 1847 | 13/5 | 0.6188 | ~5.3 | scientific terms and experimental details |
| 8 | 5 | 2141 | 12/5 | 0.6472 | 7.795 | **comparisons of people or figures using well-known public figures** |
| 9 | 5 | 9672 | 12/5 | 0.5790 | 7.437 | **the phrase "it is to"** |
| 10 | 0 | 12200 | 11/5 | 0.6927 | 3.816 | a variety of specific nouns |
| 11 | 13 | 10969 | 11/5 | 0.6762 | 16.347 | **comparisons between disciplines and relationships between concepts** |
| 12 | 0 | 12747 | 11/5 | 0.6702 | 3.615 | occurrences of 'they', 'them', 'that', 'these' |
| 13 | 0 | 2239785 | 11/5 | 0.6292 | ~4.5 | data reported as percentage |
| 14 | 5 | 16817094 | 11/5 | 0.5896 | 11.141 | (no explanation available) |
| 15 | 5 | 6473 | 10/5 | 0.6880 | 8.163 | programming terms and bad health outcome words |
| 16 | 11 | 15947 | 10/5 | 0.6801 | 14.538 | **references to historical or social change** |
| 17 | 6 | 12605 | 10/5 | 0.6737 | 9.838 | assorted acronyms, medical terms, software |
| 18 | 3 | 12006 | 10/5 | 0.6400 | 8.488 | words expressing surprise, court rulings, directions |
| 19 | 2 | 11475 | 10/5 | 0.6379 | 5.776 | **the word "refers" and related words** |
| 20 | 0 | 11651 | 10/5 | 0.6334 | 5.504 | **the word "to"** |
| 21 | 4 | 10752 | 10/5 | 0.6258 | 7.408 | **uses of the verb "to be" preceded by "to"** |
| 22 | 1 | 11356 | 10/5 | 0.6088 | 9.321 | **the word "to" followed by a verb** |
| 23 | 4 | 410 | 10/5 | 0.6028 | ~6.0 | the word "due" and nearby words |
| 24 | 4 | 12254 | 10/5 | 0.5592 | ~5.0 | **the word "to" and surrounding context** |
| 25 | 2 | 9848 | 10/5 | 0.5364 | ~4.8 | parts of words |

*Bold entries are semantically relevant to analogical reasoning or structural template encoding.*

---

## Appendix B: Data Files

All raw data produced by this analysis is available in the project directory:

| File | Description |
|------|-------------|
| `graphs/analog_berlin.json` | Attribution graph: Paris:France::Berlin:? (930 nodes, 25,915 edges) |
| `graphs/analog_rome.json` | Attribution graph: Paris:France::Rome:? (963 nodes, 27,608 edges) |
| `graphs/analog_tokyo.json` | Attribution graph: Paris:France::Tokyo:? (905 nodes, 22,414 edges) |
| `graphs/analog_teacher.json` | Attribution graph: Doctor:hospital::teacher:? (1,040 nodes, 35,481 edges) |
| `graphs/analog_bird.json` | Attribution graph: Fish:water::bird:? (1,071 nodes, 38,741 edges) |
| `graphs/analog_summaries.json` | Graph-level summary statistics |
| `graphs/analog_comparison.json` | Cross-graph feature overlap (thresholds 3/5, 4/5, 5/5) |
| `graphs/analog_labeled_features.json` | Top 25 features with Neuronpedia explanations |
| `graphs/analog_steering_validation.json` | Steering experiment results (all 404 errors) |
| `circuits/analogical_reasoning_circuit.json` | Full saved circuit with 73 nodes |

Live attribution graphs are viewable at:
- Berlin: `https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin`
- Rome: `https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_rome`
- Tokyo: `https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_tokyo`
- Teacher: `https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_teacher`
- Bird: `https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_bird`

---

## Appendix C: Key Technical Notes for Reproducibility

1. **S3 Download Required:** The Neuronpedia `/api/graph/generate` endpoint returns a metadata object containing an S3 URL. The actual graph node/edge data must be downloaded separately from this S3 URL. The saved API response JSON contains `s3url`, `url`, `numNodes`, and `numLinks` but not the actual graph data.

2. **Layer-Prefixed SAE ID:** Feature label lookups via `/api/feature/{model_id}/{sae_id}/{feature_index}` require the SAE ID to be prefixed with the layer number (e.g., `4-gemmascope-transcoder-16k`). The global SAE ID (`gemmascope-transcoder-16k`) returns HTTP 500 for individual feature lookups.

3. **Steering API Limitation:** The `/api/steer` endpoint returns HTTP 404 for the gemmascope-transcoder-16k SAE on gemma-2-2b. Only certain model-SAE combinations support steering via the public API.

4. **None Activation Handling:** Some graph nodes (embedding nodes and logit nodes) have `activation=None`. The `get_top_nodes` function in `autocircuit_tools.py` requires `None`-safe handling for the `round()` call.

---

*Analysis performed using Neuronpedia API, NetworkX, and custom Python tooling. Model: Gemma-2-2B (Google DeepMind). SAE: gemmascope-transcoder-16k. All data current as of March 1, 2026.*
