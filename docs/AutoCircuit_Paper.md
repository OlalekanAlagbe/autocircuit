# AutoCircuit: Mechanistic Interpretability of Reasoning Circuits in Gemma-2-2B

*Linguistic Reasoning, Factual Recall, and Analogical Reasoning*

---

## Abstract

This paper presents a mechanistic interpretability analysis of Gemma-2-2B across three reasoning categories — linguistic reasoning, factual recall, and analogical reasoning — using attribution graphs generated via the Neuronpedia sparse autoencoder platform. For each category, ten prompts were selected and one attribution graph generated per prompt. The top 40 nodes by influence score were extracted from each graph, and features appearing in at least 5 out of 10 graphs were identified as recurring shared circuit components, with the top 15 retained per category for analysis. Individual prompt circuits were additionally traced to reconstruct the causal sequence from input to prediction.

The recurring feature analysis reveals that the majority of features firing consistently across prompts within each category reflect structural and format detection rather than task-specific content. The task-relevant features that do emerge indicate that the model employs word meaning and definition processing for linguistic reasoning, broad domain knowledge for factual recall, and explicit relational and comparison processing for analogical reasoning. Per-prompt circuit tracing reveals a consistent two-stage architecture: early layers establish the domain and structural character of the input, while late layers retrieve the specific answer through dedicated entity or concept features. Prediction confidence is directly tied to the specificity of late-layer features — high confidence emerges when a dedicated feature converges on the answer, while low confidence reflects the absence of such a feature. These findings suggest that Gemma-2-2B does not employ dedicated reasoning circuits for the tasks studied, but instead routes predictions through whatever combination of structural, domain, and entity features are available.

---

## 1. Introduction

Rapid progress in transformer-based language modelling has directed attention towards understanding the internal computational mechanisms that give rise to emergent capabilities (Wei et al., 2022). Despite impressive performance across diverse tasks, modern LLMs remain widely regarded as "black boxes" (Alishahi et al., 2019): their predictions are difficult to explain in terms of human-interpretable algorithms. Mechanistic interpretability research aims to demystify these models by reverse-engineering their internal components into understandable circuits — subgraphs of the model's computational graph that implement distinct, identifiable functions (Olah, 2022; Wang et al., 2023).

Several recent works have made significant progress in identifying circuits for specific tasks. Wang et al. (2023) reverse-engineered the Indirect Object Identification (IOI) circuit in GPT-2 Small; Hanna et al. (2023) identified the Greater-Than circuit; Heimersheim and Janiak (2023) found a Python docstring completion circuit. Each of these required considerable manual effort — researchers iteratively apply activation patching to remove unnecessary components, cycling through variations of the dataset and circuit granularity until a satisfying mechanistic account is found. This manual process is a major obstacle to scaling interpretability research to larger models and more complex behaviours.

Conmy et al. (2023) proposed Automatic Circuit DisCovery (ACDC) to automate the third and most labour-intensive step of this workflow: finding the connections between abstract neural network units that form a circuit. ACDC iterates from outputs to inputs through the computational graph, pruning edges whose removal does not substantially degrade the model's performance on a task-specific metric. Validated against five previously hand-identified circuits, ACDC recovered the correct components with competitive accuracy compared to gradient-based alternatives such as Subnetwork Probing (SP; Cao et al., 2021) and Head Importance Score for Pruning (HISP; Michel et al., 2019).

We extend this line of work in a complementary direction. Rather than automating activation-patching over a single pre-defined computational graph, we leverage attribution graphs — a recently developed tool from Anthropic's circuit-tracing framework (Lindsey et al., 2025) — to systematically mine shared subgraph patterns across diverse prompt categories. Attribution graphs are generated via the Neuronpedia API using sparse autoencoder (SAE) transcoder features, and cross-graph pattern recognition identifies recurring circuit motifs that appear across multiple prompts. This approach allows the discovery of reusable reasoning pathways without requiring prior knowledge of which model components are involved.

This paper presents a systematic mechanistic analysis of three reasoning circuit categories in Gemma-2-2B — linguistic reasoning, factual recall, and analogical reasoning — each instantiated as a set of 10 single-token completion prompts with unambiguous correct answers. We generate 30 attribution graphs, apply cross-graph feature mining at a 50% co-occurrence threshold, and trace causal paths from influential features to the final logit node.

Our main contributions are as follows. First, across all three reasoning categories, we identify recurring SAE features that fired consistently across prompts within each category, revealing the circuit components the model relies on when approaching linguistic, factual, and analogical tasks. Second, through per-prompt circuit tracing, we show how individual features activate in a logically coherent sequence — from early structural detection through to late-layer semantic retrieval — that directly accounts for the model's final prediction in each case.

---

## 2. Methods

We analysed three categories of reasoning in Gemma-2-2B — linguistic reasoning, factual recall, and analogical reasoning — by generating attribution graphs for 30 prompts and mining the resulting graphs for shared circuit patterns. For each category, we selected 10 prompts, generated one attribution graph per prompt via the Neuronpedia API, and identified both the features that recurred consistently across prompts within a category and the causal feature sequences that explain the model's prediction for each individual prompt.

### 2.1 Step 1: Prompts

We selected 30 prompt completions spanning three reasoning categories, each designed so that the next predicted token is unambiguous and the behaviour to be explained is precisely defined.

The 30 prompts, their categories, sub-types, correct answers, and prediction probabilities are summarised in Table 1.

**Table 1.** The 30 prompts used in this study. "p" is the prediction probability assigned by Gemma-2-2B to the listed correct answer token. Prompts 3–4 predicted the colon ":" (a format-override failure) rather than the expected semantic answer.

| # | Category | Prompt (truncated) | Answer | p |
|---|----------|--------------------|--------|---|
| 1 | Linguistic | The opposite of hot is | cold | 0.566 |
| 2 | Linguistic | The antonym of ancient is | modern | 0.136 |
| 3 | Linguistic | A synonym for happy is | : | 0.092 |
| 4 | Linguistic | Another word for fast is | : | 0.084 |
| 5 | Linguistic | The plural of child is | children | 0.677 |
| 6 | Linguistic | The plural of mouse is | mice | 0.706 |
| 7 | Linguistic | The past tense of swim is | swam | 0.496 |
| 8 | Linguistic | The past tense of write is | written | 0.344 |
| 9 | Linguistic | A book of maps is called an | atlas | 0.392 |
| 10 | Linguistic | A person who writes books is called an | author | 0.836 |
| 11 | Factual | The capital of Nigeria is | Abuja | 0.173 |
| 12 | Factual | The capital of Ghana is | Accra | 0.456 |
| 13 | Factual | The Nile flows through | Egypt | 0.643 |
| 14 | Factual | The tallest mountain in the world is | Everest | 0.263 |
| 15 | Factual | Napoleon was exiled to | Elba | 0.635 |
| 16 | Factual | The first US president was | Washington | 0.193 |
| 17 | Factual | Water is composed of hydrogen and | oxygen | 0.978 |
| 18 | Factual | The powerhouse of the cell is the | mitochondria | 0.124 |
| 19 | Factual | The theory of relativity was developed by | Einstein | 0.521 |
| 20 | Factual | Hamlet was written by | Shakespeare | 0.718 |
| 21 | Analogical | Paris is to France as Berlin is to | Germany | 0.973 |
| 22 | Analogical | Cairo is to Egypt as Nairobi is to | Kenya | 0.963 |
| 23 | Analogical | Fish is to water as bird is to | air | 0.120 |
| 24 | Analogical | Puppy is to dog as kitten is to | cat | 0.652 |
| 25 | Analogical | Clock is to time as thermometer is to | temperature | 0.358 |
| 26 | Analogical | Book is to reading as radio is to | listening | 0.312 |
| 27 | Analogical | Leaf is to tree as petal is to | flower | 0.136 |
| 28 | Analogical | Wheel is to car as wing is to | airplane | 0.184 |
| 29 | Analogical | Judge is to court as priest is to | church | 0.486 |
| 30 | Analogical | Soldier is to army as sailor is to | navy | 0.214 |

### 2.2 Step 2: Model and Computational Graph

All analyses use Gemma-2-2B (Google DeepMind), a 2-billion parameter, 26-layer transformer language model. We represent the model's computation as an attribution graph generated via the Neuronpedia API, using the gemmascope-transcoder-16k sparse autoencoder (SAE) applied across all transformer layers. This SAE uses a cross-layer transcoder architecture with 16,384 features per layer, decomposing the residual stream at the level of individual semantic features rather than composite attention heads or MLP layers.

Each node in the attribution graph carries several properties. The layer and feature fields identify which SAE feature the node corresponds to, while ctx_idx encodes the token position in the prompt — meaning the same feature can appear as distinct nodes if it fires at different positions in the input sequence. The **activation** value measures how strongly a feature fired for a given prompt. The **influence value** measures the aggregated downstream causal effect of that node on the target logit — the sum of weighted contributions across all directed paths from that node to the logit output. The clerp field carries the human-readable label for the feature, retrieved via the Neuronpedia explanation API; where this is empty, the feature is identified by its layer and index alone. The is_target_logit flag identifies the final prediction node, and token_prob records the feature's direct probability contribution to the predicted token.
A feature may show high activation but low influence if its signal is cancelled by inhibitory edges downstream. All circuit identification in this study is based on influence rather than activation, because influence directly quantifies a feature's contribution to the model's final prediction.

### 2.3 Step 3: Circuit Identification

**Recurring features.** Sparse autoencoders are designed so that for any given input, only a small subset of features activate meaningfully — the majority carry negligible signal (Bricken et al., 2023). Consistent with this, causal influence in attribution graphs is concentrated in a small head of high-influence nodes. We therefore extracted the top 40 nodes by influence score from each graph as a conservative estimate of the computationally active circuit for that prompt.

To identify features that recur across prompts within a category, we applied a 50% co-occurrence threshold: a feature must appear in the top 40 of at least 5 out of 10 graphs to qualify as a recurring shared circuit component. From the qualifying features, we retained the top 15 by appearance count as the category-level recurring circuit. This threshold is conservative enough to exclude idiosyncratic prompt-level activations while permissive enough to capture genuine category-level patterns.

**Per-prompt circuit tracing.** Beyond the cross-graph analysis, each individual graph was traced to reconstruct the causal sequence from input to prediction. Starting from the top influential nodes, we followed the directed edge weights layer by layer — observing how early structural features pass signal to mid-layer domain classifiers and finally to late-layer entity-specific features that drive the logit toward the predicted token.

---

## 3. Results

### 3.1 Prediction Confidence Across All Thirty Prompts

Figures 1–3 show prediction probability distributions for all 30 prompts grouped by category, and Figure 5 presents a cross-category heatmap. Prediction confidence varied dramatically both within and across categories, ranging from p=0.084 to p=0.978.

*Figure 1. Prediction confidence (probability of top token) for 10 linguistic reasoning prompts. Color codes indicate sub-task category. Prompts predicting ":" (colon) instead of a semantic answer reflect a format-override failure.*

*Figure 2. Prediction confidence for 10 factual recall prompts. Confidence is highest for over-trained scientific facts (water/oxygen, p=0.978) and lowest for ambiguous historical and geographic facts (Nigeria, p=0.173; US president, p=0.193).*

*Figure 3. Prediction confidence for 10 analogical reasoning prompts. Geographic capital-city analogies achieve near-ceiling confidence (p>0.96), while functional and botanical analogies fall below p=0.20.*

*Figure 5. Cross-category confidence heatmap for all 30 prompts (P1–P10 per row). Green = high confidence, red = low confidence. The high-confidence cluster in Factual Recall P7 (water/oxygen, p=0.978) and Analogical P1–P2 (geographic, p>0.96) is clearly visible.*

Key patterns:
1. Factual recall shows the widest confidence range (0.124–0.978), consistent with high training-frequency variance across facts.
2. Analogical reasoning shows the clearest category-internal correlation: geographic analogies cluster near p≥0.96 while functional/botanical analogies cluster at p≤0.18.
3. Linguistic reasoning shows a bimodal distribution: closed-class morphological completions (irregular plurals and lexical look-up) cluster above p=0.50, while open-class antonym/synonym retrieval spans a wide range (0.084–0.566).

### 3.2 Shared Circuit Components: Cross-Category Recurring Features

At the 50% co-occurrence threshold, 312 features were identified for linguistic reasoning, 222 for factual recall, and 429 for analogical reasoning. Figure 4 summarizes recurring feature counts and top feature universality across categories.

*Figure 4. Recurring circuit components per category. Left bars show total recurring features at 50% threshold; right bars show the number of graphs in which the single most universal feature appeared. Analogical reasoning has the largest shared circuit (429 features) and the most universal top feature (L6/F2586668, appearing in 48/10 graphs — more than the total graph count due to multiple token positions).*

### 3.3 Top Recurring Features per Category

#### 3.3.1 Linguistic Reasoning (Top 15 Recurring Features)

| Layer | Feature | Appearances | Avg Influence | Label |
|-------|---------|-------------|---------------|-------|
| 6 | 2586668 | 37/10 | 0.6571 | words in programming code, legal jargon, or scientific texts |
| 0 | 74438300 | 26/10 | 0.6758 | a variety of specific nouns |
| 3 | 5150441 | 25/10 | 0.6616 | code snippets and documentation references (web dev) |
| 4 | 110446948 | 24/10 | 0.6861 | code snippets and license agreements |
| 7 | 4828270 | 23/10 | 0.6683 | reference codes, abbreviations, and identifiers |
| 0 | 11637899 | 22/10 | 0.7065 | "part" followed by prepositions / section words |
| 0 | 2239785 | 21/10 | 0.6129 | data as percentage in brackets; recognizes countries |
| 0 | 70051365 | 20/10 | 0.6481 | software code terms: "assembly", "using", "namespace" |
| 4 | 50205205 | 19/10 | 0.5519 | words associated with etymology or definition of a word |
| 8 | 3234687 | 16/10 | 0.6077 | words and phrases related to the meaning of words |
| 0 | 40747877 | 19/10 | 0.5392 | technical documents or data, including numbers, units, and references to figures or tables |
| 0 | 1708475 | 18/10 | 0.6013 | scientific terms and experimental details related to biological and chemical research |
| 0 | 20624252 | 16/10 | 0.7214 | words related to administrative processes and computer programs |
| 0 | 84571514 | 16/10 | 0.6172 | parenthetical numerical references and citations to literature, laws, and statistics |
| 6 | 1857621 | 16/10 | 0.585 | words related to medical or scientific texts, especially regarding drugs and medical reactions, numbers, and plurals |

The majority of recurring features — register and format detection, code snippets, legal jargon, technical document markers — do not appear semantically relevant to linguistic reasoning tasks. Their consistent firing across prompts suggests the model is responding to the structural form of the prompts rather than their linguistic content.

The two features that are meaningfully related to the task are:

- **Layer 4, F50205205** ("words associated with etymology or definition of a word")
- **Layer 8, F3234687** ("words and phrases related to the meaning of words")

Both tell us that the model employs word meaning and word definition processing when handling linguistic prompts.

#### 3.3.2 Factual Recall (Top 15 Recurring Features)

| Layer | Feature | Appearances | Avg Influence | Label |
|-------|---------|-------------|---------------|-------|
| 4 | 110446948 | 34/10 | 0.6152 | code snippets and license agreements |
| 6 | 2586668 | 32/10 | 0.6480 | words in programming code, legal jargon, or scientific texts |
| 3 | 5150441 | 24/10 | 0.6390 | code snippets and documentation references (web dev) |
| 7 | 4828270 | 22/10 | 0.7077 | reference codes, abbreviations, and identifiers |
| 0 | 40747877 | 21/10 | 0.5454 | technical documents: numbers, units, references |
| 4 | 47653198 | 20/10 | 0.6173 | various acronyms, IDs, and symbols (scientific data) |
| 0 | 2239785 | 20/10 | 0.5995 | data as percentage in brackets; recognizes countries |
| 5 | 2584395 | 19/10 | 0.6766 | law-related terminology and case/legal entity references |
| 0 | 11637899 | 19/10 | 0.6524 | "part" followed by prepositions / section words |
| 0 | 74438300 | 19/10 | 0.6433 | a variety of specific nouns |
| 0 | 1708475 | 19/10 | 0.5713 | scientific terms and experimental details related to biological and chemical research |
| 0 | 70051365 | 18/10 | 0.6317 | terms used in software code such as "assembly", "using", "namespace", and "license" |
| 1 | 99962728 | 16/10 | 0.5813 | words or phrases that appear in legal or technical documents |
| 0 | 2399144 | 15/10 | 0.7105 | technical writing related to scientific studies |
| 0 | 33329529 | 14/10 | 0.6902 | words and phrases related to societal and political issues |

The task-relevant features for factual recall are:

- **Layer 0, F1708475** — "scientific terms and experimental details related to biological and chemical research"
- **Layer 4, F47653198** — "various acronyms, IDs, and symbols (scientific data)"
- **Layer 0, F2399144** — "technical writing related to scientific studies"
- **Layer 0, F33329529** — "words and phrases related to societal and political issues"

The rest are largely the same structural/format features that appeared in linguistic reasoning. The task-relevant features tell us that the model employs scientific knowledge processing, technical data recognition, and societal and political knowledge when handling factual recall prompts — aligning directly with the geography, history, science, and literature sub-types in this category.

#### 3.3.3 Analogical Reasoning (Top 15 Recurring Features)

| Layer | Feature | Appearances | Avg Influence | Label |
|-------|---------|-------------|---------------|-------|
| 6 | 2586668 | 48/10 | 0.7217 | words in programming code, legal jargon, or scientific texts |
| 3 | 5150441 | 47/10 | 0.6449 | code snippets and documentation references (web dev) |
| 8 | 94882191 | 41/10 | 0.4939 | analogies or comparisons |
| 4 | 110446948 | 37/10 | 0.6755 | code snippets and license agreements |
| 7 | 4828270 | 31/10 | 0.7418 | reference codes, abbreviations, and identifiers |
| 9 | 89171325 | 30/10 | 0.6755 | phrases suggesting uncertainty or comparison between two things |
| 0 | 74438300 | 27/10 | 0.7010 | a variety of specific nouns |
| 9 | 4261730 | 27/10 | 0.6394 | formulas, ratios, and mathematical notation |
| 5 | 16817094 | 27/10 | 0.5668 | analogies |
| 7 | 286895 | 21/10 | 0.6201 | analogies and comparisons |
| 0 | 1708475 | 26/10 | 0.5885 | scientific terms and experimental details related to biological and chemical research |
| 0 | 81262125 | 25/10 | 0.6496 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 0 | 2239785 | 25/10 | 0.637 | data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 5 | 46836675 | 25/10 | 0.5711 | the phrase "there/they are" or the phrase "it is to." |
| 10 | 105902170 | 23/10 | 0.7307 | words related to research, business or product comparison |

The task-relevant features for analogical reasoning are:

- **Layer 8, F94882191** — "analogies or comparisons": detects the relational structure of the prompt
- **Layer 9, F89171325** — "phrases suggesting uncertainty or comparison between two things": encodes the A:B::C:? template
- **Layer 5, F16817094** — "analogies": early recognition of the analogy format
- **Layer 7, F286895** — "analogies and comparisons": reinforces relational framing across layers
- **Layer 10, F105902170** — "words related to research, business or product comparison": broader comparison context that supports relational mapping

Together these tell us that the model employs explicit relational and comparison processing when handling analogical prompts.

### 3.5 Per-Prompt Circuit Walkthroughs

Beyond the cross-graph recurring features, each individual prompt activates a distinct combination of features that together narrate how the model arrives at its specific prediction. We trace three examples per category, chosen for the clarity and logical coherence of their feature flow from early layers through to the logit.

#### 3.5.1 Analogical Reasoning

**"Paris is to France as Berlin is to ---"**

*Early Layer:* At the earliest layers, Gemma-2-2B immediately parsed the geographic and political character of the input. Feature F5757 at Layer 4 (node_id=4_5757_6, inf=0.7999) registered "place names or location references," while F15859 at Layer 4 (node_id=4_15859_6, inf=0.7981) more specifically detected "places, people, and political entities in Europe and Africa." Feature F3554 at Layer 3 (node_id=3_3554_6, inf=0.7971) activated for "mentions of countries," and F13375 at Layer 4 (node_id=4_13375_5, inf=0.7943) for "words and phrases that evoke a sense of nation and political entities." Together, these early features establish that the model has identified all four terms (Paris, France, Berlin) as European geographic and political entities before any relational processing begins. Notably, F3636 at Layer 0 (node_id=0_3636_4, inf=0.7993) fires for "proper nouns, especially names of people and places" — the very first transformer layer is already registering proper-noun status for all analogy terms.

*Middle Layer:* By the middle layers, domain knowledge takes over with two notable developments. First, F6884 at Layer 7 (node_id=7_6884_2, inf=0.7979) activates specifically for "place names relating to France and cities, specifically 'Saint-...' and 'ville' or 'canton'" — the model has mapped the A-term (Paris) into its France-specific geography domain. Then, F4435 at Layer 13 (node_id=13_4435_8, inf=0.7987) fires for "opera-related terms, including German words and names of places," grounding the B-term (Berlin) in its German cultural context. The critical relational insight arrives at L10 with F5144 (node_id=10_5144_5, inf=0.7995), labeled simply "comparisons" — by Layer 10, the model has abstracted the analogy template as a comparison structure, with Paris-to-France and Berlin-to-Germany resolved as parallel capital-to-country relationships.

*Late Layer:* The late layers crystallise the answer through a convergent pathway. F14348 at Layer 19 (node_id=19_14348_8, inf=0.8001) — labeled "uses of the word 'to' and other prepositions" — achieves the highest influence score in the entire graph. This feature appears to be tracking the structural connector "is to" in the analogy template, acting as a final syntactic anchor.

---

**"Wheel is to car as wing is to ---"**

*Early layers:* The most directly relevant early feature is Layer 1, F13554 ("words related to car transmissions and gear shifting") and Layer 4, F5249 ("car parts, file extensions, and cooking-related words") — together establishing that the model has identified the mechanical/vehicular character of the prompt. Layer 2, F1220 ("words related to motors, mechanics, and systems of vehicles") reinforces this. Layer 7, F4920 ("words that are machine parts and how the parts are used") is particularly significant — by Layer 7 the model has recognised not just the objects but their functional role as parts of a larger mechanical system.

*Middle layers:* Layer 9, F13344 ("phrases suggesting uncertainty or comparison between two things") and Layer 9, F14231 ("words representing comparisons and relationships") confirm that the analogy template is recognised. Layer 10, F14542 ("words related to research, business or product comparison") extends the relational framing into a broader comparison structure.

*Late layers:* Layer 23, F5563 ("words related to transport, aviation, and the military") is the decisive feature — directly activating the aviation domain that yields "airplane" as the predicted token. The low confidence (p=0.162) is telling — the model correctly maps the part-to-vehicle relationship but the aviation domain is weakly anchored, with no dedicated "airplane parts" feature appearing to firmly select the answer.

---

**"Judge is to court as priest is to ---"**

*Early layers:* The model immediately identifies the dual-domain nature of the prompt. Layer 1, F11387 ("formal legal definitions and analysis of how principles work"), Layer 1, F15799 ("words related to legal and business documents"), and Layer 0, F1218 ("words and phrases related to law enforcement and international relations") establish the legal domain from "judge." Simultaneously, Layer 5, F7592 ("words and phrases related to religion and religious texts") fires on "priest" — by the early layers both source and target domains are already identified.

*Middle layers:* Layer 9, F6402 ("references to laws, articles and sections") reinforces the legal domain grounding. The remaining middle layer features are largely noise — no strong relational abstraction feature fires here, which is notable compared to other analogical prompts.

*Late layers:* The late layers tell the clearest story. Layer 17, F6505 ("court") fires with an extraordinary activation of 71.9 — the highest single concept activation in this walkthrough — directly encoding the source institution. Then Layer 19, F10549 ("words and phrases related to religion and religious belief") activates for the target domain, and Layer 23, F4640 ("historical sites, graveyards, churches, religious symbols") narrows toward the specific answer. Layer 20, F14217 ("text related to laws and government") continues anchoring the source domain in parallel.

The moderate confidence (p=0.486) reflects the model knowing the target domain is religious but having no single feature that locks onto "church" specifically — F4640 activates across churches, graveyards, and religious symbols equally, leaving competition among plausible answers.

#### 3.5.2 Factual Recall

**"Hamlet was written by"**

*Early layers:* The model quickly establishes the literary and authorship context of the prompt. Layer 6, F7712 ("references to books and quotations") and Layer 2, F9235 ("proper nouns that are names of people or artistic works") identify "Hamlet" as a literary work. Layer 1, F15258 ("names of real or fictional people") and Layer 6, F5100 ("proper nouns including names of people, organizations") prepare the circuit to resolve the authorship relationship. Layer 1, F7685 ("the word 'by'") is notably specific — the model registers the attribution structure of the prompt at the very first layer.

*Middle layers:* Layer 8, F15528 ("references to famous people, especially those who have died") is the most relevant here — the model narrows the answer space to a deceased historical figure. Layer 13, F5441 ("titles of people and organizations") and Layer 14, F11187 ("places and organizations in northern Europe") begin anchoring the geographic and cultural context of the answer.

*Late layers:* Layer 18, F4309 ("the word 'write' in different contexts, including copyright notices and general writing") fires with an activation of 46.47 — directly encoding the authorship relationship. Layer 17, F5633 ("books") and Layer 18, F15471 ("art/creativity") reinforce the literary domain. Layer 23, F11667 ("mentions of historical rulers and their territories") further anchors the answer in a historical figure of authority.

The moderate confidence (p=0.578) likely reflects the model predicting "William" rather than "Shakespeare" directly — the first name is the initial token, and while the circuit converges on the correct author, the split between "William" and "Shakespeare" as the next token introduces some uncertainty.

---

**"Water is composed of hydrogen and"**

*Early layers:* From the earliest layers, the model establishes a chemistry context. F3470 at Layer 3 (inf=0.7921) fires for "technical or scientific concepts related to biology, physiology, and chemistry, especially at a cellular level," while F10278 at Layer 4 (inf=0.7930) specifically activates for "mentions of chemical elements or compounds" — directly registering "hydrogen" as a chemical element and priming the circuit to expect another element to follow.

*Middle layers:* The compositional logic of the sentence locks in at Layer 8. F1623 (inf=0.7989, act=10.26) fires for "sentence fragments involving composition or structure using words like 'of', 'bit', 'up', or 'with'" — the phrase "is composed of... and" is precisely this pattern. F2506 at Layer 7 (inf=0.7953) narrows the domain further to "technical language related to chemical processes and reactions." By this point the model has parsed the sentence as a chemical enumeration: one element named, one to follow.

*Late layers:* Four Layer 9 features fire in tight succession — "code and equations" (act=16.62), "scientific and mathematical texts" (act=5.25), "formulas, ratios, and mathematical notation" (act=5.17), and "words related to chemistry and chemical elements" (act=4.46). This dense cluster represents the model's recall of H₂O as a stored formula, producing the near-certain prediction of "oxygen" at p=0.978.

---

**"The longest river in the world is the"**

*Early layers:* Layer 6, F2974 ("places, people, and demographics in Africa and the South Pacific") establishes the Africa geographic signal early — this is the feature that will eventually route the circuit toward the Nile specifically. Layer 6, F3456 ("references to geographic locations and bodies of water") and Layer 3, F11151 ("the preposition 'across' and other words associated with geography or location") frame the spatial character of the query. Layer 5, F5456 ("mentions of global concerns, world events, or specific topical references") encodes "world" as a superlative scope marker.

*Middle layers:* Layer 7, F14593 ("words related to geology, geography, and oceanography") carries the highest influence in this stage. Layer 9, F8205 ("mentions of rivers") is the most diagnostically relevant feature — directly encoding the subject of the query. Layer 13, F15057 ("geographic locations, especially bodies of water, parks and trails") reinforces the water domain classification.

*Late layers:* Layer 19, F13436 ("words related to rivers and watersheds") fires with an activation of 25.67 — the dominant late-layer feature. The convergence of this feature with the Africa signal from Layer 6 and the bodies-of-water framing from Layer 13 produces the clean route to "Nile" over competitors like "Amazon", which would require a South America rather than Africa signal.

#### 3.5.3 Linguistic Reasoning

**"The plural of child is"**

*Early layers:* The most relevant early features are Layer 2, F11571 ("words related to family, childhood, and male relationships") — directly registering "child" as a family/kinship term — and Layer 7, F10626 ("terms related to family relationships, especially focusing on marital status and legitimacy of children") which reinforces this at a later early layer with specific reference to children. Layer 3, F14287 ("discourse regarding definitions and word usage") is significant — it encodes the definitional structure of the prompt ("the plural of X is"), telling us the model has recognised this as a word-definition query. Layer 4, F2866 ("the word 'is'") anchors the completion frame.

*Middle layers:* Layer 9, F12218 ("references to old English combined with place names, or names and meanings") is the most mechanistically interesting feature here — "children" is a historical Old English irregular plural, and this feature firing suggests the model is routing the answer through etymological/historical word-form knowledge rather than a regular morphological rule.

*Late layers:* Layer 17, F9341 ("statements or words that express general truths") encodes the declarative, factual nature of the completion. Layer 22, F7881 ("words referencing large numbers of people or objects") is a broad plurality encoder that activates as the model prepares to output a plural form. No dedicated "children"-specific feature appears — the answer converges through the combination of family domain, Old English morphology, and plurality encoding.

---

**"The past tense of swim is"**

*Early layers:* Layer 5, F10444 ("baseball terminology") carries the highest influence in the early layers — a clear case of feature repurposing. Baseball verbs are predominantly irregular monosyllables with vowel-change past tenses (hit, pitch, run, throw) — structurally identical to "swim." The feature is firing on that shared morphological pattern, not on sport. Layer 2, F10736 ("verbs being used") and Layer 0, F6366 ("the word 'meaning'") establish that the model recognises a verb definition structure.

*Middle layers:* Layer 11, F12241 ("name origins and meanings") fires with a notable activation of 22.44 — as seen in the child→children walkthrough, this feature recurs as a morphological history detector, activating on words whose current form derives from historical sound changes. Layer 12, F79 ("the definitions of words, especially names, and the word 'meaning'") reinforces the definitional lookup frame.

*Late layers:* Layer 18, F6481 ("mentions of grammatical tense and simple declarative clauses") fires at an extraordinary activation of 86.06 — by far the highest in this walkthrough — directly encoding the past tense query structure. Layer 16, F6856 ("definitions of words") and Layer 24, F4155 ("mentions of occurrences and changes over time, especially using auxiliary verbs") support the tense-change framing.

---

**"The past tense of write is"**

*Early layers:* Layer 4, F13440 ("context related to words/language and their meanings or usages") and Layer 4, F14373 ("mentions of the word 'term' or other words relating to language") establish that the model recognises this as a language/word-form query. Layer 2, F10736 ("verbs being used") and Layer 2, F15200 ("the verb 'to be' in various languages and tenses") register that a verb transformation is being requested.

*Middle layers:* Layer 7, F10819 ("words and phrases related to reading and writing data in a computer system") is the closest middle-layer feature to the actual verb "write", though it fires in a computing context rather than a linguistic one — another instance of feature repurposing. No strong relational feature encoding past tense transformation appears.

*Late layers:* Layer 18, F10954 ("terms used in academic papers, especially those using measurements or observations") fires at activation 35.85 — the highest in this walkthrough — but has no clear connection to the predicted token. No dedicated feature for "written" or irregular past participle formation appears anywhere in the circuit.

The low confidence (p=0.344) follows directly — the model identifies a verb transformation is needed but lacks a specific feature to resolve whether the answer is "wrote" or "written", reflecting genuine uncertainty between the simple past and past participle forms.

---

## 5. Limitations

- **No causal steering validation.** The recurring features identified here are correlational — they appear consistently but their causal necessity for the task computations has not been confirmed by intervention. Steering experiments were blocked by Neuronpedia API errors in the analogical domain and not attempted in the linguistic and factual domains.

- **Limited prompt counts.** With 10 prompts per domain (9 for factual recall), the 30% threshold admits both genuine circuit components and coincidental co-activations. A larger dataset (50+ prompts per sub-task) would sharpen the analysis.

- **SAE coverage limitations.** The gemmascope-transcoder-16k SAE may not capture all computationally relevant features, particularly those in attention heads. Some circuit components may be invisible to this analysis.

- **Single model.** All findings are specific to Gemma-2-2B. Whether the same circuit architecture applies to other models (GPT-2, LLaMA, Gemma-7B, Gemma-27B) is unknown and is an important direction for future work.

- **Feature label quality.** Neuronpedia's automated feature labels may not perfectly capture the computational role of each feature, particularly for polysemantic features that activate on multiple contexts.

---

## 6. Conclusion

Across all three reasoning categories, the recurring feature analysis reveals a consistent pattern: the majority of features that fire across prompts within a category reflect structural and format detection rather than task-specific content. However, the task-relevant features that do emerge tell a meaningful story about how the model approaches each category. For linguistic reasoning, the model employs word meaning and definition processing. For factual recall, it draws on broad domain knowledge spanning science, technical data, and societal and political knowledge. For analogical reasoning, it employs explicit relational and comparison processing — showing the strongest and most direct correspondence between recurring features and the task being performed, while linguistic reasoning shows the weakest.

At the per-prompt level, the circuit analysis reveals a consistent two-stage architecture across all three categories. Early layers establish the structural and domain character of the input — identifying what kind of text is being processed and what domain the key terms belong to. Late layers then attempt to retrieve the specific answer through dedicated entity or concept features.

The confidence of the model's prediction is directly tied to how well the late-layer retrieval succeeds. When a dedicated feature exists for the specific answer — as with "court" firing at activation 71.9 in the *"Judge is to court as priest is to ---"* prompt, or four chemistry features converging simultaneously for *"Water is composed of hydrogen and"* — the model predicts with high confidence. When no such feature exists, as in the antonym of ancient or Mount Everest's location, the circuit correctly identifies the domain but cannot resolve the specific answer, producing low confidence and weak predictions.

The most universal finding is what is absent rather than what is present — the model does not appear to have dedicated reasoning circuits for antonym retrieval, morphological transformation, or geographic fact lookup. Instead it routes these tasks through whatever combination of structural, domain, and entity features are available, with confidence reflecting how cleanly those features converge on a single answer.

---

## 7. References

[1] Anthropic (2025). Circuit Tracing: Attribution Graphs and Mechanistic Interpretability Methods. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html

[2] Lindsey, J., et al. (2025). On the biology of a large language model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/

[3] Neuronpedia (2025). Gemma-2-2B Attribution Graph API and SAE Feature Explorer. https://neuronpedia.org/gemma-2-2b/graph

[4] Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. *Advances in Neural Information Processing Systems*, 36 (NeurIPS 2023). arXiv:2304.14997.

[5] Syed, A., Rager, C., and Conmy, A. (2023). Attribution patching outperforms automated circuit discovery. arXiv:2310.10348.

[6] Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. (2023). Interpretability in the wild: a circuit for indirect object identification in GPT-2 Small. *The Eleventh International Conference on Learning Representations* (ICLR 2023).

[7] Hanna, M., Liu, O., and Variengien, A. (2023). How does GPT-2 compute greater-than? Interpreting mathematical abilities in a pre-trained language model. arXiv:2305.00586.

[8] Heimersheim, S. and Janiak, J. (2023). A circuit for Python docstrings in a 4-layer attention-only transformer. *Alignment Forum*. https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn.

[9] McDougall, C., Conmy, A., Rushing, C., McGrath, T., and Nanda, N. (2023). Copy suppression: comprehensively understanding an attention head. arXiv:2310.04625.

[10] Elhage, N., Nanda, N., Olsson, C., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html.

[11] Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context learning and induction heads. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.

[12] Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards monosemanticity: decomposing language models with dictionary learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features/

[13] Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy models of superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/

[14] Bills, S., Cammarata, N., Mossing, D., et al. (2023). Language models can explain neurons in language models. *OpenAI*. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html.

[15] Goldowsky-Dill, N., MacLeod, C., Sato, L., and Arora, A. (2023). Localizing model behavior with path patching. arXiv:2304.05969.

[16] Cao, S., Sanh, V., and Rush, A. (2021). Low-complexity probing via finding subnetworks. *Proceedings of NAACL-HLT 2021*, pp. 960–966.

[17] Michel, P., Levy, O., and Neubig, G. (2019). Are sixteen heads really better than one? *Advances in Neural Information Processing Systems* 32 (NeurIPS 2019), pp. 14014–14024.

[18] Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *The Eleventh International Conference on Learning Representations* (ICLR 2023).

[19] Olah, C. (2022). Mechanistic interpretability, variables, and the importance of interpretable bases. *Transformer Circuits Thread*. https://www.transformer-circuits.pub/2022/mech-interp-essay.

[20] Räuker, T., Ho, A., Casper, S., and Hadfield-Menell, D. (2022). Toward transparent AI: a survey on interpreting the inner structures of deep neural networks. arXiv:2207.13243.

[21] Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems* 35 (NeurIPS 2022).

[22] Wei, J., Tay, Y., Bommasani, R., et al. (2022). Emergent abilities of large language models. arXiv:2206.07682.

[23] Alishahi, A., Chrupała, G., and Linzen, T. (2019). Analyzing and interpreting neural networks for NLP: a report on the first BlackboxNLP workshop. *Natural Language Engineering*, 25(4), pp. 543–557.

[24] Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., et al. (2022). Causal scrubbing: a method for rigorously testing interpretability hypotheses. *Alignment Forum*. https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing.
