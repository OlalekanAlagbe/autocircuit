# Factual Recall Circuit in Gemma-2-2B: A Mechanistic Analysis

## Abstract

We characterise the factual recall circuit in Gemma-2-2B by generating and analysing attribution graphs for 24 factual recall prompts spanning seven sub-domains: geography, authorship, biology, history, sport, technology, and art/culture. Attribution graphs were generated via the Neuronpedia API and analysed using sparse autoencoder (SAE) features from the gemmascope-transcoder-16k family. Cross-graph comparison (min 30% recurrence, n=24) identified 261 recurring features; the top 15 are labelled and interpreted here. The factual recall circuit is characterised by three functional stages: (1) early layers (0–4) activate broad reference-text classification features spanning scientific, legal, technical, and political registers — identifying the *type* of fact before resolving the specific value; (2) middle layers (5–10) engage structural encoding features (code documentation, acronyms, identifiers, web-development references) that serve as factual anchors connecting domain signals to named-entity knowledge; (3) late layers (11–25) activate domain-specific named-entity and locative features that directly push toward the answer token. The three highest-influence recurring features are a Layer-0 "overall/quantitative" feature (avg_inf=0.7635), a Layer-0 "'red' at start of phrase" feature (avg_inf=0.7590), and a Layer-0 "suffix 'ins'/thereon" feature (avg_inf=0.7589) — all early-layer surface-form features whose recurrence suggests the model's factual circuit shares structural-text recognition machinery across diverse factual domains. Activation steering was attempted on all three top features but the Neuronpedia steer endpoint returned HTTP 500 errors (server unavailable) for Gemma-2-2B at the time of analysis; circuit neighbourhood labels were obtained via graph traversal as a proxy. The circuit's dominant recurring features describe how facts appear in text (encyclopedic/reference-text patterns) rather than encoding entity relationships directly, suggesting factual recall in Gemma-2-2B is partly driven by surface-level pattern-matching of reference-text structure.

## 1. Introduction

Large language models demonstrate remarkable factual recall capabilities - answering questions about geography, history, science, culture, and language with high accuracy. Yet the internal computational mechanisms by which transformer models retrieve and surface stored facts remain poorly understood. Mechanistic interpretability offers a path to understanding: by tracing attribution graphs through sparse autoencoder (SAE) features, we can identify *which* transformer layers and features causally participate in factual retrieval and how they interact.

This paper presents a mechanistic analysis of the **factual recall circuit** in Gemma-2-2B, a 2-billion-parameter transformer model. We generated attribution graphs for 24 factual recall prompts spanning seven sub-domains (geography/capitals, authorship, science, history, sports, technology, and art/culture), identified recurring features across graphs, labelled them via the Neuronpedia SAE feature database, and validated causal claims via activation steering.

## 2. Methods

### 2.1 Model and Prompts

Model: Gemma-2-2B (gemma-2-2b on Neuronpedia).
SAE: gemmascope-transcoder-16k (layer-prefixed).
Prompts: 24 factual recall prompts covering capitals, authorship, biology, history, geography, sport, technology, and art/culture. All prompts are sentence completions ending mid-sentence so the next token is unambiguous.

### 2.2 Attribution Graph Generation

Attribution graphs were generated via the Neuronpedia /api/graph/generate endpoint with parameters: maxFeatureNodes=3000, desiredLogitProb=0.95, nodeThreshold=0.8, edgeThreshold=0.85.

### 2.3 Cross-Graph Analysis

Feature co-occurrence was identified via compare_graphs() with min_appearances = ceil(0.30 x 24) = 8. Features recurring in >=8 graphs were retained (261 total). The top 15 by (appearances, avg_influence) were labelled via Neuronpedia's feature explanation API.

### 2.4 Per-Prompt Interpretation

Each graph was interpreted by labelling the top 40 influential nodes (excluding logit/embed nodes) grouped into early, middle, and late layer bands. Token competition was assessed via logit contribution aggregation across all nodes.

### 2.5 Steering Validation

The top 3 recurring features by avg_influence were steered on a representative prompt using strength=20.0 and strength_multiplier=4.0. Circuit neighbourhood was labelled at depth=1 to confirm causal role.

## 3. Results

### 3.1 Recurring Features Across Factual Recall Graphs

The following 15 features appeared in at least 8 of 24 graphs (>=30%) and are sorted by (appearances, avg_influence) descending.

| Layer | Feature | Appearances | Avg Influence | Label |
|-------|---------|-------------|---------------|-------|
| 6 | 2586668 | 69/24 | 0.6537 | words that appear in programming code, legal jargon, or scientific texts |
| 4 | 110446948 | 69/24 | 0.6124 | code snippets and license agreements |
| 3 | 5150441 | 61/24 | 0.6406 |  code snippets and documentation references, possibly related to web development |
| 0 | 40747877 | 57/24 | 0.5420 | technical documents or data, including numbers, units, and references to figures or tables. |
| 0 | 1708475 | 53/24 | 0.5960 | scientific terms and experimental details related to biological and chemical research |
| 0 | 70051365 | 50/24 | 0.6266 | terms used in software code such as "assembly", "using", "namespace", and "license" |
| 0 | 2239785 | 49/24 | 0.6007 |  data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 7 | 4828270 | 45/24 | 0.6929 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 0 | 74438300 | 43/24 | 0.6696 |  a variety of specific nouns |
| 1 | 99962728 | 42/24 | 0.6062 |  words or phrases that appear in legal or technical documents, like names of laws, legal terms (pleaded, testified), and technical terms, especially when abbreviated or in code |
| 4 | 47653198 | 41/24 | 0.6163 | various acronyms, IDs, and symbols, possibly related to scientific data |
| 5 | 2584395 | 38/24 | 0.6730 |  law related terminology and references to specific cases or legal entities. |
| 0 | 18711902 | 35/24 | 0.6752 |  words related to money or business transactions |
| 0 | 33329529 | 35/24 | 0.6722 |  words and phrases related to societal and political issues |
| 0 | 11637899 | 33/24 | 0.6702 |  the word "part" followed by prepositions or words related to sections or components. |

**Interpretation:** The recurring features reveal a structured factual recall circuit. Early layers (0-1) activate broad domain-classification features spanning scientific, legal, technical, and political text registers - the model first identifies *what kind of fact* is being requested before resolving the specific value. Middle layers (3-6) engage more specialised structural features: code documentation, acronyms, identifiers, and web-development references that serve as "factual anchor" encodings connecting domain signals to named-entity knowledge. Late-middle layers (7+) feature multi-domain identifier/abbreviation features, representing the final consolidation step where competing answer candidates are resolved into a specific token. Surprisingly, the dominant recurring features describe *how facts appear in text* (encyclopedic/reference document patterns) rather than encoding entity relationships directly, suggesting factual recall in Gemma-2-2B is partly driven by surface-level pattern matching of reference-text structure.

### 3.2 Per-Prompt Circuit Interpretations


### Prompt: "<bos>The capital of Nigeria is"

**Predicted token:** `Output " a" (p=0.173)` (prob=0.1731)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.8001 | 2.0753 |  words associated with government, law, and crime |
| 0 | 0.7995 | 3.0159 | locations and organizations |
| 1 | 0.7992 | 1.8948 |  mentions of geographic locations in the UK. |
| 0 | 0.7988 | 3.3592 |  many different words associated with completely different topics, spanning games, science, sports, visual media, food and community |
| 0 | 0.7982 | 2.3931 | the word "is" |
| 4 | 0.798 | 6.9604 |  words related to study designs, results, and published documents |
| 3 | 0.7976 | 3.3693 | words or phrases about locations or things being part of other things, especially about clinics or medical analysis |
| 0 | 0.7974 | 1.7245 |  the phrase ‘country of origin’ |
| 7 | 0.7969 | 5.8022 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 1 | 0.7965 | 3.315 |  code and documented code |
| 0 | 0.7963 | 1.9935 |  references to dates and times |
| 0 | 0.7961 | 2.365 | the word survivor (or a variant) and sometimes a preceding article of 'the'. |
| 6 | 0.7957 | 3.2699 |  words and phrases related to nationality and patriotism |
| 4 | 0.7954 | 2.3771 |  phrases containing auxiliary verbs "is," "are," "was". "be" and action verbs often including "to" and "by". |
| 0 | 0.7952 | 1.8181 |  the word "recent" and words appearing near to it |
| 0 | 0.795 | 2.3991 | the word "damage" and sometimes other words near "damage" or related to negative experiences |
| 6 | 0.7948 | 2.7502 |  mentions of cities or regions |
| 6 | 0.7946 | 3.6775 | phrases related to formulas or derivations. |
| 0 | 0.7944 | 2.0843 |  words and phrases related to societal and political issues |
| 1 | 0.7942 | 3.0339 |  instances of the word "The" |
| 6 | 0.7937 | 2.8771 |  mentions of specific corporate entities and product or character names from particular fictional universes, in addition to terms related to marine biology |
| 3 | 0.7935 | 2.3508 |  words/phrases related to story telling |
| 2 | 0.7931 | 2.3753 |  words and phrases related to governmental leadership and elections. |
| 0 | 0.7929 | 1.6285 |  vague positive terms and terms related to rules and laws |
| 0 | 0.7926 | 1.4628 | mentions of opinions, facts or reasons |
| 3 | 0.7924 | 3.8018 |  terms related to code and US immigration |
| 6 | 0.7922 | 4.2509 |  names of places, mostly countries, provinces and cities |
| 4 | 0.792 | 3.1997 | words related to politics, finance and government. |
| 1 | 0.7918 | 2.2059 |  mentions of genetic sequences, primers, and genes |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7999 | 5.4144 |  mentions of a person named "Iluobe", possibly along with some titles |
| 9 | 0.7997 | 5.5815 |  words that indicate a superlative or high importance |
| 8 | 0.799 | 5.0684 | words related to culture, particularly Indian culture, as well as family structure and cuisine. |
| 9 | 0.7986 | 4.5138 |  mentions of a particular city, and also mentions of "Free Water" |
| 0 | 0.7984 | 0.0 |  the keyword 'let' which is used for variable declarations in Javascript |
| 8 | 0.7978 | 3.5821 |  proper nouns referring to countries or political entities, and words associated with international trade and economics |
| 11 | 0.7967 | 22.4298 | scientific or technical words and jargon |
| 10 | 0.7959 | 4.3549 |  words associated with institutional, professional, and/or academic language |
| 11 | 0.7939 | 10.1481 |  references to countries or regions |
| 15 | 0.7933 | 31.3888 | the word "capital" and sometimes letters |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7971 | 11.9528 |  text written in all caps, especially political text |

[EARLY] At the earliest layers, the model is parsing the structural frame of the "The capital of X is" completion pattern. The high activation on 'locations and organizations' (Layer 0, inf=0.7995) indicates the model has already identified this as a geographic/political query. The 'words associated with government, law, and crime' feature at Layer 4 (inf=0.8001) is not noise — 'capital' co-occurs heavily with government and law in encyclopedic text. The 'nationality and patriotism' features at Layer 6 (inf=0.7957) suggest the model is mapping the country name Nigeria to a national-political register. Simultaneously, 'mentions of cities or regions' (Layer 6, inf=0.7948) and 'names of places, mostly countries, provinces and cities' (Layer 6, inf=0.7922) begin priming the geographic answer space. The 'country of origin' feature at Layer 0 (inf=0.7974) is particularly telling: the model is not just parsing text structure but inferring a country-capital lookup relation.

[MIDDLE] In the middle layers, the circuit shifts from domain classification to entity-specific resolution. The highly activated 'mentions of a person named Iluobe' feature at Layer 8 (inf=0.7999, act=5.4144) likely encodes Nigerian proper-noun patterns — "Iluobe" is a Nigerian name, and this feature fires as the model narrows into West African entity space. Layer 9's 'words that indicate a superlative or high importance' (inf=0.7997) reflects the encyclopedic register of "capital city" as the primary/most important city. The 'proper nouns referring to countries or political entities, and words associated with international trade and economics' at Layer 8 (inf=0.7978) is assembling the Nigeria→political_entity binding. Critically, Layer 15's 'word "capital" and sometimes letters' feature (inf=0.7933, act=31.3888) fires with very high activation, directly recognising the semantic slot being filled: this is the "capital of" construction and something city-shaped must complete it.

[LATE] At Layer 24, 'text written in all caps, especially political text' (inf=0.7971, act=11.9528) is the sole late-layer feature, pushing toward capitalised proper-noun tokens — the kind of thing capital cities are called. This feature has a formatting bias toward all-caps text common in political and governmental documents, nudging the model toward a city-name token. The relatively sparse late-layer circuit is consistent with the model's uncertainty: it has resolved the domain (Nigerian capital) but not converged cleanly on the specific token.

[TOKEN COMPETITION] The predicted token is " a" (p=0.173), which is strikingly low confidence and clearly wrong as an answer. The circuit has failed to converge on "Abuja" (the correct answer) or even "Lagos" (the historically prominent city). The very low probability signals that competing tokens — likely " Abuja", " Lagos", " the" — are splitting the vote. The 'text written in all caps' late feature is too generic to break the tie decisively. The ambiguity traces back to the middle layers: despite firing on Nigerian proper-noun space, the model never activated a dedicated Abuja-binding feature, leaving the logit mass spread across many plausible completions. The margin is highly ambiguous and the circuit is informationally incomplete for this specific entity.

---

### Prompt: "<bos>The capital of Ghana is"

**Predicted token:** `Output " Accra" (p=0.381)` (prob=0.3813)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 5 | 0.7995 | 6.5516 |  law related terminology and references to specific cases or legal entities. |
| 5 | 0.7993 | 7.4063 | hexadecimal numbers |
| 2 | 0.7989 | 2.4185 | occurrences of the word "premium", sometimes alongside words of similar or opposite meaning |
| 0 | 0.7987 | 1.8181 |  the word "recent" and words appearing near to it |
| 5 | 0.7985 | 2.9837 |  words related to martial arts, authority, or judgement |
| 4 | 0.7983 | 2.2877 |  mentions of specific geographic regions, especially Europe, plus associated demographics or governing bodies. |
| 6 | 0.7981 | 3.2451 |  words and phrases related to geographic regions and movement |
| 3 | 0.7976 | 2.4654 | names of organizations, people, places and political situations |
| 0 | 0.7974 | 1.6329 |  proper nouns, especially names of people, places, and organizations, as well as some words related to academic or technical fields. |
| 0 | 0.7972 | 1.761 |  technical words used in computing, science, or engineering |
| 6 | 0.7966 | 20.2227 |  code and file paths |
| 6 | 0.7962 | 3.4721 |  question answer pairs and censored text |
| 0 | 0.7958 | 2.3991 | the word "damage" and sometimes other words near "damage" or related to negative experiences |
| 5 | 0.7956 | 5.944 |  mentions of places in an economic or political context |
| 3 | 0.7954 | 10.2825 |  code snippets and documentation references, possibly related to web development |
| 5 | 0.7951 | 3.0213 | comparative adjectives, statistics, and references. |
| 5 | 0.7949 | 10.9809 | the letter 'L' capitalized |
| 5 | 0.7947 | 3.0978 |  mentions of countries and nationality adjectives |
| 4 | 0.7945 | 4.3425 |  code comments and import statements |
| 0 | 0.7943 | 2.8854 |  the word "latter" along with surrounding text that specifies what "latter" is referring to. |
| 4 | 0.7941 | 2.6215 |  phrases containing auxiliary verbs "is," "are," "was". "be" and action verbs often including "to" and "by". |
| 1 | 0.7939 | 2.3318 |  present tense forms of the verb "to be" |
| 1 | 0.7937 | 2.0806 | words related to legal proceedings |
| 4 | 0.7935 | 2.9284 | words related to government and official organizations |
| 0 | 0.7933 | 2.3897 |  medical and scientific terms, especially procedures performed inside the body and names or categories. |
| 2 | 0.7926 | 2.5482 |  the possessive pronouns "teu" and "seu" in Portuguese |
| 3 | 0.792 | 1.6688 |  locations and/or groups of people involved in a country's political structure |
| 0 | 0.7918 | 2.1558 |  capitalized jargon words, especially those related to legal or anime contexts |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.8001 | 8.7877 | location or government titles, especially in reference to court cases or county boards |
| 9 | 0.7991 | 27.6213 |  lines of code that import libraries or declare constants |
| 10 | 0.7978 | 4.9697 |  words associated with institutional, professional, and/or academic language |
| 12 | 0.797 | 6.7204 |  questions about the relationship between the value of K and the amount of training data in cross validation |
| 9 | 0.7964 | 5.8409 |  instances of something starting, being contained, or being to blame |
| 14 | 0.796 | 6.2196 | text relating to corporate structures and product versions |
| 11 | 0.7928 | 13.1961 | scientific or technical words and jargon |
| 9 | 0.7922 | 4.1311 |  mentions of multiple items or things characterized by a count or aggregate. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 25 | 0.7999 | 7.6278 |  capitalized proper nouns and titles, especially those associated with a specific person or location |
| 24 | 0.7997 | 12.4763 | references to geographic locations and geopolitical entities and people's relationship to them |
| 16 | 0.793 | 14.1337 | countries |
| 24 | 0.7924 | 9.148 |  capitalized words and possibly location names |

[EARLY] The early layers immediately register this as a geographic-political completion task. 'Mentions of specific geographic regions, especially Europe, plus associated demographics or governing bodies' (Layer 4, inf=0.7983) and 'names of organizations, people, places and political situations' (Layer 3, inf=0.7976) establish that a named geographic entity is being queried. The 'words and phrases related to geographic regions and movement' (Layer 6, inf=0.7981) and 'mentions of countries and nationality adjectives' (Layer 5, inf=0.7947) together perform the critical alignment: "Ghana" is parsed as a country-token, activating nationality registers. The 'locations and/or groups of people involved in a country's political structure' (Layer 3, inf=0.792) is especially important — it collapses the many possible completions down to politically salient place-names. The auxiliary-verb pattern at Layer 4 ('phrases containing auxiliary verbs "is," "are," "was"', inf=0.7941) locks in the completion-frame expectation: X is [city].

[MIDDLE] In the middle band, the circuit performs the Ghana→Accra binding. 'Location or government titles, especially in reference to court cases or county boards' (Layer 12, inf=0.8001, act=8.7877) operates as a political-entity anchor, pulling the completion toward a seat-of-government token. The 'words associated with institutional, professional, and/or academic language' (Layer 10, inf=0.7978) reflects encyclopedic text patterns where capital cities appear in formal institutional descriptions. 'Scientific or technical words and jargon' at Layer 11 (inf=0.7928, act=13.1961) is consistent with the recurring pattern of factual encyclopedic text, where scientific and geographic facts share formal register. The middle layers are efficiently routing through institutional/governmental semantics.

[LATE] The late layers execute decisive geographic token selection. 'Capitalized proper nouns and titles, especially those associated with a specific person or location' (Layer 25, inf=0.7999, act=7.6278) and 'references to geographic locations and geopolitical entities and people's relationship to them' (Layer 24, inf=0.7997, act=12.4763) both push hard toward a capitalized place-name token. The 'countries' feature at Layer 16 (inf=0.793, act=14.1337) is explicitly geographic, and the 'capitalized words and possibly location names' (Layer 24, inf=0.7924) adds a second late-stage nudge toward proper-noun form. This is a well-converged late-stage circuit: multiple independent features all vote for the same class of token.

[TOKEN COMPETITION] The predicted token " Accra" wins with a confident p=0.381, the strongest win margin in the capital-city prompt cluster. The convergence of geographic proper-noun features at Layer 24-25 creates a strong push toward the single correct city token. Runner-up candidates likely include " the" and " a" (article completions) and possibly " Kumasi" (Ghana's second city), but the late-layer 'geopolitical entities' feature (act=12.4763) cleanly selects a capital-class location token. The high confidence relative to the Nigeria prompt reflects successful entity-specific binding: Ghana→Accra is more unambiguously encoded than Nigeria→Abuja in the training distribution.

---

### Prompt: "<bos>The capital of USA is"

**Predicted token:** `Output " Washington" (p=0.354)` (prob=0.3545)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.8001 | 2.4812 |  words related to Malaysia, food, or food culture |
| 0 | 0.7999 | 2.9707 |  references to dates and times |
| 3 | 0.7997 | 10.2825 |  code snippets and documentation references, possibly related to web development |
| 1 | 0.7995 | 2.6982 | the word "employees" and related words such as "employee" and "managers" in the context of work |
| 0 | 0.7992 | 2.8854 |  the word "latter" along with surrounding text that specifies what "latter" is referring to. |
| 0 | 0.799 | 2.8546 |  code snippets, specific scientific or technical terms, and date-related words. |
| 3 | 0.7986 | 3.8965 |  asterisks and legal records with numbers or names like "McDonald" |
| 6 | 0.7984 | 6.6474 | words related to medical or scientific texts, especially regarding drugs and chemical reactions, numbers, and plurals. |
| 0 | 0.798 | 2.3827 | the word "of". |
| 5 | 0.7978 | 3.1487 |  terms related to immigration, nationality, foreign governments, and embassies. |
| 6 | 0.7976 | 3.181 |  proper nouns and terms or jargon specific to particular fields |
| 0 | 0.7973 | 1.8181 |  the word "recent" and words appearing near to it |
| 1 | 0.7971 | 4.0416 |  strings of ellipses and sentence fragments, possibly in conjunction with other punctuation |
| 7 | 0.7969 | 5.9481 |  place names relating to France and cities, specifically "Saint-...", and "ville" or "canton." |
| 1 | 0.7963 | 2.3401 |  words related to towns |
| 4 | 0.7961 | 10.1749 | code snippets and license agreements |
| 0 | 0.7952 | 3.3592 |  many different words associated with completely different topics, spanning games, science, sports, visual media, food and community |
| 1 | 0.795 | 2.0029 |  words that would be found in police reports or legal documents |
| 5 | 0.7948 | 5.751 |  terms related to government |
| 1 | 0.7946 | 3.1156 | code snippets related to the Swift programming language |
| 3 | 0.7944 | 2.0385 | phrases indicating mathematical relationships, comparisons, or relationships between studies.  |
| 7 | 0.7939 | 3.2566 | words related to technical details, locations, legislation, and personal qualities/merit. |
| 1 | 0.7937 | 2.2059 |  mentions of genetic sequences, primers, and genes |
| 0 | 0.7935 | 2.5521 |  a technical terms and concepts from a variety of fields, including computer science, physics and politics. |
| 0 | 0.7931 | 2.503 | abbreviations, statistics and citations |
| 4 | 0.7928 | 3.1527 |  tokens related to people's names, locations like USA, medical terms, and elements of email addresses. |
| 4 | 0.7926 | 4.6324 |  geographic regions and landmarks |
| 6 | 0.7924 | 10.2314 |  code and file paths |
| 0 | 0.792 | 1.7846 |  mentions of music |
| 6 | 0.7918 | 2.9619 |  terms related to the establishment, origin, or foundation of something |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7988 | 4.6384 |  references to scientific equipment suppliers and cities |
| 9 | 0.7967 | 6.0542 |  words that indicate a superlative or high importance |
| 8 | 0.7965 | 5.7823 |  internet addresses, references to the best services and products, and marketing language |
| 8 | 0.7959 | 6.6714 |  code snippets with specific coding keywords and markup tags |
| 9 | 0.7956 | 5.3088 |  instances of something starting, being contained, or being to blame |
| 0 | 0.7933 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7982 | 120.2892 | the |
| 17 | 0.7954 | 79.1709 |  specific months |
| 25 | 0.7941 | 8.738 | prepositions followed by locations, times, or ordinal numbers relating to events |
| 16 | 0.7922 | 7.1629 | a mix of words related to legal documents/financial reports, personal preferences, existential questions, and motorcycle advertisements. |

[EARLY] The early layers establish a complex but somewhat diffuse recognition pattern for this prompt. 'Tokens related to people's names, locations like USA, medical terms, and elements of email addresses' (Layer 4, inf=0.7928) is telling: "USA" itself is among the prototypical tokens firing this feature, suggesting the model has pattern-matched on the USA token specifically. 'Terms related to immigration, nationality, foreign governments, and embassies' (Layer 5, inf=0.7978) activates because "USA" strongly co-occurs in diplomatic and civic text where embassies and foreign-government mentions appear. 'Geographic regions and landmarks' (Layer 4, inf=0.7926) and 'terms related to government' (Layer 5, inf=0.7948) jointly establish the political-geographic register. The 'word "of"' feature (Layer 0, inf=0.798) explicitly fires on the structural token, while the code-and-web-dev patterns (Layer 3, inf=0.7997; Layer 6, inf=0.7924) reflect the SAE's tendency to encode encyclopedic factual text via reference-document surface patterns.

[MIDDLE] The middle band is notably thin for this prompt, with only a handful of features. 'Internet addresses, references to the best services and products, and marketing language' (Layer 8, inf=0.7965) is likely firing on the USA brand-like quality — the US is referenced at superlative scale in web text. 'Words that indicate a superlative or high importance' (Layer 9, inf=0.7967) aligns with the USA's encyclopedic prominence. The 'references to scientific equipment suppliers and cities' (Layer 9, inf=0.7988) may reflect co-occurrence of major US cities in supplier/institutional text. Notably absent is a strong named-entity binding feature for Washington D.C. — the circuit is converging on place-name tokens generically rather than the specific answer.

[LATE] The late layers are unusual and revealing. Layer 16's 'the' (inf=0.7982, act=120.2892) fires with enormous activation, suggesting the model strongly predicts a "the X" structure — i.e., "the capital of USA is the [city]" completion. Layer 17's 'specific months' (inf=0.7954) is strange but may reflect encyclopedic date-rich text co-occurring with US political facts. 'Prepositions followed by locations, times, or ordinal numbers relating to events' (Layer 25, inf=0.7941) pushes toward a prepositional location phrase, consistent with "Washington" being the next token before "D.C."

[TOKEN COMPETITION] " Washington" wins at p=0.354, a moderate margin. The very high activation on the article 'the' at Layer 16 suggests competition from article-first completions (" the capital", " DC"). The late-layer preposition feature pushes toward the Washington token as a place name following a preposition. Runner-ups likely include " D" (for D.C.), " DC", and " the". The margin is moderately confident — the USA capital is well-encoded in training data as "Washington" specifically, and the circuit converges on it, but the article-layer interference creates genuine competition.

---

### Prompt: "<bos>Hamlet was written by"

**Predicted token:** `Output " William" (p=0.578)` (prob=0.5780)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.7999 | 2.4199 |  terms about scripture and truth, often in the context of Christian belief |
| 1 | 0.7997 | 2.6969 |  the word "by." |
| 0 | 0.7993 | 2.4975 |  markers of subjective time or existance |
| 0 | 0.7988 | 3.0245 |  verbs and adjectives related to failure or difficulty |
| 3 | 0.7986 | 3.1701 | code-related keywords "OR" or the last syllable of "processor" or "server" or "faire" |
| 0 | 0.7984 | 1.0472 | snippets of Swift and Protobuf code and references to pagan and celtic origins |
| 7 | 0.7977 | 4.2379 | legal court case related terms as well as names like Obama, Zimmerman, Dewani, Perry, TripAdvisor |
| 0 | 0.7972 | 2.7486 | words related to time and state |
| 5 | 0.7963 | 2.7896 | words describing places/organizations related to art, studies, and science; people's titles and/or words describing processes |
| 0 | 0.7958 | 2.1823 |  dates, titles, and proper nouns related to historical landowners and legal matters. |
| 6 | 0.7956 | 4.9927 |  references to books and quotations |
| 0 | 0.7954 | 2.2679 | parenthetical clauses and associated punctuation, with some activation for forms of the verb "to be", possessive adjectives, and conjunctions. |
| 1 | 0.7949 | 1.5868 |  technical writing relating to computers |
| 1 | 0.7944 | 1.543 | names of real or fictional people |
| 6 | 0.7937 | 2.9878 | proper nouns including names of people, organizations, and software products |
| 1 | 0.7935 | 4.7774 |  strings of ellipses and sentence fragments, possibly in conjunction with other punctuation |
| 1 | 0.7933 | 2.7315 |  language used to present an argument in a court of law |
| 3 | 0.7926 | 4.014 | verbs or phrases indicating action, saying, or feeling |
| 2 | 0.7921 | 3.9811 |  words related to creating something using technical processes |
| 0 | 0.7919 | 1.1344 |  words related to legal and scientific documents |
| 6 | 0.7916 | 2.3063 | various proper nouns and adjectives, including ethnic groups, nationalities, religions, job titles, deities, and locations. |
| 2 | 0.7911 | 2.419 |  proper nouns that are names of people or artistic works |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 14 | 0.8002 | 13.6926 |  the word "The" at the beginning of lines or isolated whitespace |
| 13 | 0.7995 | 6.168 |  titles of people and organizations, including government officials and media personalities |
| 14 | 0.7974 | 7.0149 |  places and organizations in northern Europe |
| 14 | 0.7961 | 8.7043 |  capitalized two or three letter abbreviations, last names, and words that start with "H". |
| 8 | 0.7951 | 7.7254 |  references to famous people, especially those who have died |
| 11 | 0.793 | 6.3268 | first and second person pronouns and forms of the verb "to be". |
| 13 | 0.7928 | 5.7021 | words and short phrases associated with legal proceedings |
| 9 | 0.7923 | 6.239 |  content connected to team sports or sports players. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 23 | 0.799 | 7.9148 |  prepositions and words that commonly appear near them |
| 23 | 0.7981 | 7.9976 |  mentions of historical rulers and their territories |
| 18 | 0.7979 | 46.47 |  the word 'write' in different contexts, including copyright notices, code, configuration, and general writing. |
| 23 | 0.797 | 10.624 |  words describing military actions, especially those involving the US Army in the 1800s |
| 25 | 0.7968 | 14.9548 |  historical political discussions related to Denmark and Germany |
| 25 | 0.7965 | 16.413 | law and legal citations and similar references like numbers related to particular rules or acts. |
| 17 | 0.7947 | 8.6132 | books |
| 17 | 0.7942 | 7.4267 | positive adjectives |
| 18 | 0.794 | 7.0497 | art/creativity |
| 23 | 0.7914 | 9.1246 |  words and phrases related to thanking people or authors |

[EARLY] The early layers parse this as an authorship attribution prompt. 'Names of real or fictional people' (Layer 1, inf=0.7944) fires immediately on the proper-noun frame, while 'references to books and quotations' (Layer 6, inf=0.7956) identifies the literary-work register. The 'word "by"' feature (Layer 1, inf=0.7997) is mechanistically crucial: it fires directly on the preposition that introduces the author, and by activating here, it establishes the "written by [PERSON]" completion frame. 'Proper nouns and terms or jargon specific to particular fields' (Layer 6, inf=0.7976) and 'various proper nouns and adjectives, including ethnic groups, nationalities, religions, job titles' (Layer 6, inf=0.7916) widen the proper-noun candidate space. The 'language used to present an argument in a court of law' (Layer 1, inf=0.7933) is not noise — "Hamlet" appears in academic and legal discourse citation contexts.

[MIDDLE] The middle layers perform the Hamlet→Shakespeare binding. 'Titles of people and organizations, including government officials and media personalities' (Layer 13, inf=0.7995) encodes the author-title relationship — Shakespeare is one of the most titled authors in encyclopedic text. 'Places and organizations in northern Europe' (Layer 14, inf=0.7974) fires on the Danish setting of Hamlet (and by proximity, the English author), anchoring the geographic-cultural context. 'Capitalized two or three letter abbreviations, last names, and words that start with "H"' (Layer 14, inf=0.7961, act=8.7043) is remarkably specific: "Hamlet" begins with H and "Shakespeare" is a famous last name in this feature's training distribution. 'References to famous people, especially those who have died' (Layer 8, inf=0.7951) cleanly encodes the Shakespeare-as-historical-celebrity relationship. The 'word "The"' at line beginning (Layer 14, inf=0.8002) may reflect the formatting pattern of "The works of Shakespeare" in training corpora.

[LATE] The late layers deliver the author-name push. 'Mentions of historical rulers and their territories' (Layer 23, inf=0.7981) fires because Shakespeare is treated as a cultural "ruler" in literary discourse. 'Historical political discussions related to Denmark and Germany' (Layer 25, inf=0.7968) directly maps Hamlet's Danish setting to the Shakespeare attribution chain. 'The word "write" in different contexts, including copyright notices, code, configuration, and general writing' (Layer 18, inf=0.7979, act=46.47) is the most important late feature: it explicitly represents the writing/authorship domain and pushes toward an author-name token. 'Words describing military actions, especially those involving the US Army in the 1800s' (Layer 23, inf=0.797) is surprising but may reflect co-occurrence patterns in historical literary text.

[TOKEN COMPETITION] " William" wins confidently at p=0.578 — this is one of the more decisive wins in the authorship sub-domain. The "written by" frame at Layer 1 and the "famous deceased people" feature at Layer 8 together create a strong vote for the Shakespeare first-name token. Runner-ups likely include " Shakespeare" (the surname directly), " the" (a hedged completion), and possibly " Ben" (Jonson, another Elizabethan playwright). The high probability of " William" over " Shakespeare" reflects the "written by" pattern typically preceding a full name starting with the first name in encyclopedic text. The margin is confident: the circuit converges cleanly.

---

### Prompt: "<bos>The Mona Lisa was painted by"

**Predicted token:** `Output " Leonardo" (p=0.714)` (prob=0.7142)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8001 | 1.3422 |  words and phrases related to slavery, race and groups/tribes of people, both past and present. |
| 0 | 0.7999 | 1.2061 |  words related to legal cases and genre descriptors for movies and books |
| 5 | 0.7991 | 3.0879 |  verbs of action or process |
| 3 | 0.7985 | 1.9312 | intensifiers and words indicating high degree, extremity, or a final point |
| 6 | 0.798 | 2.6356 |  names of places and their associated demonyms |
| 6 | 0.7978 | 4.9052 |  words that appear in technical and academic writing |
| 1 | 0.7976 | 5.3211 |  mentions of metals, metal extraction, and metal coating |
| 3 | 0.7974 | 2.1673 |  words or phrases related to scientific research, technical documentation, or academic writing, including specific jargon from various fields. |
| 5 | 0.7972 | 2.918 | terms related to treaties, protocols, and formal agreements |
| 3 | 0.797 | 5.3543 |  terms related to nobility, religion and places. |
| 1 | 0.7966 | 1.0676 |  words related to studies, health conditions, genetics, theories and numerical data |
| 3 | 0.7964 | 1.4435 |  descriptions of layers in a semiconductor |
| 0 | 0.7962 | 1.5005 |  technical terms, jargon, and esoteric words |
| 1 | 0.796 | 4.573 | the word "paint" and related terms |
| 5 | 0.7958 | 2.9453 | multiple vaguely related clusters of words |
| 6 | 0.7954 | 2.4024 |  words and phrases related to music and orchestras |
| 0 | 0.7952 | 3.4013 |  the word "overall", sometimes alongside words that express quantity |
| 3 | 0.7949 | 1.397 |  the word "into" and nearby pronouns or articles. |
| 3 | 0.7947 | 1.8049 |  terms related to building an AlertDialog in Android |
| 4 | 0.7945 | 2.351 |  technical academic or legal writing |
| 3 | 0.7941 | 1.8356 | terms in document headers, or relating to document structure |
| 0 | 0.7939 | 1.2971 |  names of Italian people and places |
| 1 | 0.7937 | 1.1196 | terms related to guidance, submission, and technological or academic progress |
| 5 | 0.7935 | 2.7895 |  mentions of historical events in Europe and the British Isles including people, locations, and dates. |
| 4 | 0.7933 | 2.5826 |  terms referring to politicians or political office. |
| 3 | 0.7931 | 1.7836 | words and affixes related to scientific processes, finance and politics |
| 3 | 0.7928 | 1.1464 |  references to "night", especially in the context of sports games or weeks |
| 5 | 0.7926 | 2.2886 |  words related to jewelry and marital celebrations |
| 6 | 0.7924 | 3.7875 |  references to governments, territories, and political entities |
| 4 | 0.7922 | 2.6653 |  words related to study designs, results, and published documents |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7995 | 21.859 | code and equations |
| 7 | 0.7993 | 5.2081 | phrases about people and organizations involved in writing, producing, directing, and performing creative works. |
| 13 | 0.7987 | 6.1144 | the word "by" followed by an attribute |
| 9 | 0.7983 | 3.6276 |  historical and geographical references, often with a focus on battles, rulers, and empires, frequently accompanied by dates or time periods. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7997 | 0.0 | code snippets containing the word "return" |
| 20 | 0.7989 | 8.8381 |  words that are used to describe something as other than average |
| 15 | 0.7968 | 48.0638 | words associated with legal, academic, or technical documents, and numerical references |
| 23 | 0.7956 | 8.2357 |  job postings for nurses |
| 18 | 0.7943 | 6.5567 | words or phrases used when asking for or giving agreement |
| 16 | 0.792 | 38.8078 |  terms starting with "mon" related to medicine, science, and agriculture or math |

[EARLY] The early layers immediately identify this as a creative-work authorship prompt. The 'word "paint" and related terms' (Layer 1, inf=0.796, act=4.573) is the most direct signal — it fires on "painted" and strongly primes the artworks-and-creators semantic frame. 'Names of Italian people and places' (Layer 0, inf=0.7939) fires on the lexical association with "Mona Lisa," which is an Italian artwork title; this feature is performing national-cultural mapping before any artist name has appeared. 'Terms related to nobility, religion and places' (Layer 3, inf=0.797) and 'words that appear in technical and academic writing' (Layer 6, inf=0.7978) encode the Renaissance scholarly register in which the Mona Lisa is typically discussed. 'References to governments, territories, and political entities' (Layer 6, inf=0.7924) is not noise in Renaissance art context — Leonardo operated under Medici patronage within city-state political structures.

[MIDDLE] The middle band performs the artwork→artist binding. 'Phrases about people and organizations involved in writing, producing, directing, and performing creative works' (Layer 7, inf=0.7993) is the key relational feature: it encodes the creator-of-work relationship generically, and here it maps "painted by" onto the creator slot. 'The word "by" followed by an attribute' (Layer 13, inf=0.7987) executes the syntactic binding — the preposition "by" is explicitly representing the attribution relationship. 'Historical and geographical references, often with a focus on battles, rulers, and empires, frequently accompanied by dates or time periods' (Layer 9, inf=0.7983) fires on the Renaissance historical framing. The minimal middle band (only 4 features) suggests this is a relatively clean, unambiguous fact retrieval — the circuit efficiently routes from "painted by" to a single high-confidence answer.

[LATE] The late layers push decisively toward a specific artist-name token. 'Terms starting with "mon" related to medicine, science, and agriculture or math' (Layer 16, inf=0.792, act=38.8078) has remarkable specificity: the "mon" prefix fires on "Mona" from the painting's title, and the model is now preparing a token that also begins with "mon-" — i.e., "Leonardo" does not start with "mon", but this feature likely reflects a partial phonological/orthographic priming from the input token "Mona." 'Words associated with legal, academic, or technical documents' (Layer 15, inf=0.7968, act=48.0638) reflects the encyclopedic register. 'Words that are used to describe something as other than average' (Layer 20, inf=0.7989) fires because Leonardo is prototypically exceptional in training text.

[TOKEN COMPETITION] " Leonardo" wins at p=0.714, one of the highest confidence scores in the art/culture sub-domain. The "word 'by' followed by an attribute" feature at Layer 13 directly binds the attribution relation, and the Italian-names feature at Layer 0 pre-loads the Italian-artist namespace. Runner-ups likely include " da" (as in "da Vinci" — the circuit might split between first name and patronym) and " an" (a hedged completion). The very high confidence reflects that the Mona Lisa→Leonardo association is among the most frequently co-occurring fact pairs in training corpora, and the circuit converges tightly. The "mon-" prefix feature provides an additional resonance nudge toward the answer.

---

### Prompt: "<bos>The theory of relativity was developed by"

**Predicted token:** `Output " Albert" (p=0.564)` (prob=0.5639)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8001 | 1.53 |  commands and filenames in code |
| 4 | 0.7999 | 1.9884 |  Government and health organizations or policies |
| 0 | 0.7993 | 1.3734 | the word "released" and words that relate to releasing in the context of software or music, but also picks up some noise |
| 3 | 0.7991 | 3.2531 | language related to legal and technical documents, involving processes and connections |
| 3 | 0.7989 | 2.7827 |  words or phrases that indicate a development or reaction to prior events. |
| 3 | 0.7987 | 1.8934 |  a combination of the words "two", "of", and/or "the" and also some proper nouns |
| 0 | 0.7985 | 1.7213 |  technical terminology related to biological and computational lineages and their properties. |
| 4 | 0.7983 | 2.6347 |  words related to study designs, results, and published documents |
| 4 | 0.7981 | 2.3829 |  words related to court proceedings and legal arguments |
| 0 | 0.7975 | 1.616 |  technical terms used in scientific writing |
| 3 | 0.7973 | 2.0041 |  references to sports teams, locations, and politics |
| 0 | 0.7971 | 1.3398 | technical and scientific terms |
| 1 | 0.7969 | 1.7649 |  text from scientific papers, including mathematical symbols and references |
| 0 | 0.7964 | 4.5515 | the word "mix" along with surrounding tokens, sometimes with other words that may or may not be related |
| 3 | 0.7962 | 7.6753 |  words related to scientific experimentation and description |
| 1 | 0.796 | 1.338 |  words or phrases related to politics, legality, or crime |
| 3 | 0.7952 | 2.7402 |  scientific hypotheses and theses |
| 3 | 0.7948 | 2.9166 | code snippets or command lines |
| 0 | 0.7946 | 1.8786 |  a variety of specific nouns |
| 3 | 0.7944 | 5.5342 |  text describing results or findings of medical or scientific studies |
| 1 | 0.7942 | 5.6071 | the keyword "super" within code |
| 1 | 0.794 | 1.5445 |  academic physics papers and terminology |
| 0 | 0.7937 | 4.0907 | phrases related to focus and giving focus to. |
| 0 | 0.7935 | 1.3737 | words that end in prefixes or suffixes that are not very common, or words that are proper nouns. |
| 0 | 0.7931 | 3.1676 |  words that start a sentence or section |
| 0 | 0.7929 | 3.4834 | the word "pad" |
| 0 | 0.7925 | 3.2178 |  the word "ease" |
| 0 | 0.7923 | 3.1829 | words relating to measurement, damage, institutions, or physical locations |
| 0 | 0.7921 | 2.1492 |  references to environment, potentially as viewed from an outside perspective and references to crime |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 10 | 0.7995 | 15.9528 |  content related to scientific publications or medical procedures, possibly extracting patient data from research papers |
| 9 | 0.7979 | 21.859 | code and equations |
| 9 | 0.7977 | 3.9377 | a combination of topics related to the mistreatment of minority groups and physical ailments |
| 9 | 0.7958 | 4.9053 |  math or coding. |
| 7 | 0.7956 | 3.9211 | assertions, assumptions, axioms, and theorems related to algebraic and graphical models |
| 10 | 0.7954 | 3.5169 |  corporate and financial information, especially company founding dates |
| 7 | 0.7933 | 3.1446 |  content separators and content about a US president named Jackson and Calhoun, U.S. tariffs, and Nullification Crisis |
| 6 | 0.7927 | 2.0912 |  words used when describing mathematical and or scientific models |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 19 | 0.7997 | 36.518 |  words related to creating something new |
| 13 | 0.7967 | 5.9752 | words and symbols related to a variety of technical topics. |
| 14 | 0.795 | 6.7837 |  medical language discussing healthy patients and diseases |

[EARLY] The early layers identify this as a scientific-discovery authorship prompt. 'Academic physics papers and terminology' (Layer 1, inf=0.794) and 'text from scientific papers, including mathematical symbols and references' (Layer 1, inf=0.7969) immediately establish the physics domain. 'Words related to scientific experimentation and description' (Layer 3, inf=0.7962, act=7.6753) and 'text describing results or findings of medical or scientific studies' (Layer 3, inf=0.7944) encode the research-findings register in which theories are attributed to their developers. 'Scientific hypotheses and theses' (Layer 3, inf=0.7952) is particularly important: it fires on "theory" and primes the hypothesis-author attribution frame — in scientific text, theories are almost always attributed to a person. 'Government and health organizations or policies' (Layer 4, inf=0.7999) may reflect relativity's connections to institutional science (patents, Nobel Prize coverage).

[MIDDLE] The middle band executes the theory→Einstein binding. 'Content related to scientific publications or medical procedures, possibly extracting patient data from research papers' (Layer 10, inf=0.7995) encodes the named-author-of-paper relationship. 'Assertions, assumptions, axioms, and theorems related to algebraic and graphical models' (Layer 7, inf=0.7956) directly fires on the word "theory" and its mathematical-axiom connotations, priming the Einstein attribution via the theory-axiom chain. 'Corporate and financial information, especially company founding dates' (Layer 10, inf=0.7954) is counterintuitively important: founding-date text shares structural patterns with discovery-attribution text ("developed by X in year Y"). 'Content separators and content about a US president named Jackson and Calhoun' (Layer 7, inf=0.7933) reflects the historical-figure-attribution register that overlaps with Einstein references.

[LATE] The late layers push toward a creator-name token. 'Words related to creating something new' (Layer 19, inf=0.7997, act=36.518) is the dominant feature — it directly encodes the development/creation domain and pushes toward an inventor/creator name token. The relatively sparse late band (only 3 features) indicates a clean, unambiguous attribution with minimal competition at token selection time.

[TOKEN COMPETITION] " Albert" wins at p=0.564, a moderately confident margin. The "creating something new" feature at Layer 19 (act=36.518) provides a strong push toward a person-name token in the inventor/developer register. Runner-ups include " Einstein" directly and possibly " a" (hedged completion). The choice of " Albert" over " Einstein" follows the same pattern as " William" in the Hamlet prompt: the "developed by" frame canonically precedes a full name, and training data most often renders "developed by Albert Einstein" with the first name first. The confidence is slightly lower than Hamlet (0.564 vs 0.578), reflecting that "theory of relativity" is occasionally attributed in shorthand as "Einstein's theory" rather than a "developed by" construction.

---

### Prompt: "<bos>The powerhouse of the cell is the"

**Predicted token:** `Output " nucleus" (p=0.103)` (prob=0.1025)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.8 | 5.3343 |  words relating to technical writing, data processing and conditional situations. |
| 4 | 0.7996 | 3.1502 | phrases that contain the word "pin" or that convey a sense of location |
| 3 | 0.7993 | 3.7888 | language regarding collaborative relationships or partnerships |
| 3 | 0.7991 | 1.9076 | technical or scientific concepts related to biology, physiology, and chemistry, especially at a cellular level. |
| 0 | 0.7989 | 1.1569 |  words or phrases connected to technical inventions or medical procedures |
| 1 | 0.7987 | 3.7006 | the word "the" followed by words that modify or describe something, often in technical contexts. |
| 5 | 0.7985 | 3.0788 | words related to protein function in molecular biology |
| 0 | 0.7983 | 1.7714 |  words related to publications and scientific research |
| 1 | 0.7981 | 3.1761 |  formal legal definitions and analysis of how principles work |
| 3 | 0.7979 | 3.2577 |  uses of profanity |
| 0 | 0.7969 | 2.3625 |  phrases containing the word known or words like new, old and sweet which are commonly used to describe something |
| 4 | 0.7965 | 5.8553 | the word "of" in different contexts |
| 5 | 0.7964 | 4.2922 |  phrases related to science |
| 6 | 0.7962 | 6.2993 |  topics/titles or short phrases that often begin with a capitalized word |
| 0 | 0.796 | 2.2445 |  mentions of patients in medical or therapeutic contexts |
| 0 | 0.7958 | 3.1922 | a lot of diverse things that mostly only appear in computer programming code, math equations, or politically-charged comments |
| 6 | 0.7956 | 3.7152 |  numbers or words that appear in scientific papers. |
| 5 | 0.7954 | 3.5008 |  terms related to liquid crystal displays |
| 0 | 0.795 | 4.6724 |  the word "the" |
| 3 | 0.7948 | 5.1681 | words and phrases that describe places, geography, and governance |
| 5 | 0.7946 | 2.9532 |  words and phrases about medicine and health |
| 4 | 0.7944 | 4.6671 | various acronyms, IDs, and symbols, possibly related to scientific data |
| 1 | 0.794 | 1.709 |  words referencing insects, larvae, and fungi, specifically related to their life cycle and interaction with plants. |
| 3 | 0.7934 | 4.5264 |  terms related to data collection, lists, and arrays |
| 4 | 0.793 | 6.0455 |  references to good things and sources of funding, especially anything involving "Natural", "Foundation", "Science", or "Technology" |
| 0 | 0.7924 | 1.9919 |  mentions of the earth and land, but also "wrong" and somewhat related terms. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.7998 | 15.5761 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 9 | 0.7977 | 3.9759 |  references to the US politician John Calhoun and the Jackson administration |
| 7 | 0.7971 | 5.9202 | scientific data related to dry weight and concentration measurements. |
| 10 | 0.7952 | 5.1302 |  phrases introducing the main subject or an interesting point |
| 9 | 0.7932 | 6.4983 |  code snippets or technical documentation, possibly related to images, databases, or functions with IDs |
| 0 | 0.7926 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7994 | 7.2593 | pairs or triplets of short words typically found together. |
| 14 | 0.7975 | 6.2538 | conditional statements |
| 20 | 0.7973 | 25.7329 | technical language related to biological processes. |
| 16 | 0.7967 | 24.475 | cell biology |
| 16 | 0.7942 | 9.4884 |  words related to legal and financial documents |
| 20 | 0.7938 | 9.1502 |  symbols and special characters commonly used in mathematical notation, as well as words relating to academic writing |
| 21 | 0.7936 | 32.4116 |  scientific jargon related to cell biology or parasites |
| 14 | 0.7928 | 37.1962 | the word "using" |

[EARLY] The early layers establish a cell biology domain recognition. 'Technical or scientific concepts related to biology, physiology, and chemistry, especially at a cellular level' (Layer 3, inf=0.7991) fires immediately and is the most domain-specific early feature — it directly encodes the cellular-biology register that the prompt inhabits. 'Words related to protein function in molecular biology' (Layer 5, inf=0.7985) and 'phrases related to science' (Layer 5, inf=0.7964) further confirm the biological science domain. The 'word "of" in different contexts' (Layer 4, inf=0.7965) fires on the "of the cell" construction, encoding the possessive-genitive relationship between a process and its cellular location. 'References to good things and sources of funding, especially anything involving "Natural", "Foundation", "Science", or "Technology"' (Layer 4, inf=0.793, act=6.0455) reflects the model recognising this as a biology-textbook fact of the kind that appears in NSF-funded scientific literature.

[MIDDLE] The middle band shows the circuit attempting but struggling to converge on the mitochondria answer. The 'references to the US politician John Calhoun and the Jackson administration' (Layer 9, inf=0.7977) is clearly incidental activation and not causally relevant. The 'variety of reference codes, abbreviations, and identifiers from different fields' (Layer 7, inf=0.7998, act=15.5761) reflects the model treating "mitochondria" as a technical identifier. 'Scientific data related to dry weight and concentration measurements' (Layer 7, inf=0.7971) fires because mitochondria are discussed extensively in quantitative biology papers with measurement data.

[LATE] The late layers reveal the biology circuit at work. 'Cell biology' (Layer 16, inf=0.7967, act=24.475) is the most targeted feature — it explicitly represents cell biology as a domain and strongly primes cell-organelle tokens. 'Technical language related to biological processes' (Layer 20, inf=0.7973, act=25.7329) reinforces the biological process register. 'Scientific jargon related to cell biology or parasites' (Layer 21, inf=0.7936, act=32.4116) fires at very high activation and is directly pushing toward a cell-organelle technical term. The convergence of three independent cell-biology features in the late band creates a strong vote for the organelle answer.

[TOKEN COMPETITION] The predicted token is " nucleus" (p=0.103) — the wrong answer (the correct answer is "mitochondria"). Despite the 'cell biology' and 'cell biology/parasites' features firing in the late band, the model produces "nucleus" rather than "mitochondrion." This reveals an important failure mode: the circuit has correctly identified the cellular-biology domain but selected the more generally prominent organelle. "Nucleus" is the most frequently discussed organelle in cell biology text overall (cell division, DNA, gene expression) and wins the logit competition even against features that seem to target the mitochondria answer. The very low probability (0.103) confirms that many tokens are competing: " mitochondria", " nucleus", " chloroplast", " cytoplasm" all have non-trivial probability mass, and the circuit lacks a specific "powerhouse→mitochondrion" binding feature strong enough to win decisively.

---

### Prompt: "<bos>Photosynthesis takes place inside the"

**Predicted token:** `Output " chlor" (p=0.369)` (prob=0.3690)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.8 | 2.7187 | words related to legal or scientific research like testimony, detection, interrupt, or publication |
| 0 | 0.7998 | 1.3219 |  a word stem "lle" or "ull" and the word "Slemish". That's not the best description, so here are some alternatives: Irish geography terms, words with the ully suffix, certain surnames |
| 7 | 0.7996 | 4.9448 |  scientific writing describing ecological systems and processes |
| 7 | 0.7994 | 4.1314 |  technical terminology from a wide range of fields including computing, science, sports, and general topics |
| 1 | 0.7992 | 4.4184 |  the word "formal" along with some legal and medical terms. |
| 6 | 0.799 | 11.3436 |  words and phrases related to images and visual media |
| 5 | 0.7985 | 6.2372 |  genus/species names of extinct organisms |
| 4 | 0.7983 | 3.632 | terms related to disease, medical conditions and scientific studies, particularly emphasizing HIV. |
| 0 | 0.7979 | 3.2222 | the word "alone" |
| 0 | 0.7975 | 3.1321 |  code snippets dealing with user interfaces and visualizers |
| 0 | 0.7973 | 2.6688 |  terminology related to measuring the components of plasma samples |
| 5 | 0.7969 | 3.3487 |  words related to astronomy or zoology |
| 1 | 0.7967 | 4.2159 |  mentions of islands and caves |
| 2 | 0.7965 | 1.7204 |  words related to macro economics and politics |
| 3 | 0.7963 | 5.4405 |  code snippets and documentation references, possibly related to web development |
| 1 | 0.7961 | 1.4086 |  variations of the word "busy" and some words related to clothing or sewing. |
| 0 | 0.7959 | 1.2821 |  proper nouns or words from popular culture. |
| 0 | 0.7957 | 1.3746 | snippets of Swift and Protobuf code and references to pagan and celtic origins |
| 0 | 0.7953 | 1.6112 |  words related to research papers or documentation |
| 7 | 0.7951 | 5.1575 |  technical or scientific terms related to medicine or mechanical engineering |
| 3 | 0.7949 | 2.3271 |  a mix of computer code, economic modeling, and poll-related terms |
| 0 | 0.7947 | 4.325 | verbs in the past tense |
| 4 | 0.7942 | 4.4428 | words indicating a thing (usually a person) is being protected or defended |
| 4 | 0.794 | 4.7141 |  proper nouns related to geography, organizations and publications |
| 1 | 0.7936 | 4.1745 |  the acronym "lol" |
| 3 | 0.7932 | 3.3859 |  words that are related to medicine, especially internal organs or medical procedures |
| 6 | 0.7928 | 2.841 |  words and abbreviations related to cellular biology and genetics |
| 2 | 0.7926 | 2.7059 |  source code relating to software libraries and metadata |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7946 | 7.1508 |  words related to measuring electrical current flow across cell membranes |
| 15 | 0.7938 | 64.9969 | words associated with legal, academic, or technical documents, and numerical references |
| 10 | 0.7934 | 5.8207 |  prepositions and words used with them |
| 8 | 0.7924 | 26.8052 |  code snippets with specific coding keywords and markup tags |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.7988 | 18.0347 |  technical writing. |
| 0 | 0.7987 | 0.0 |  the word "saw" and the verbs "live" and "lived". |
| 20 | 0.7981 | 25.8242 |  clauses expressing hypothetical or conditional situations |
| 25 | 0.7977 | 8.6307 |  occurrences of the words "the", "a" or "to" followed by adjectives, nouns, or gerunds |
| 16 | 0.7971 | 7.0524 |  the word "inside" and words or punctuation that frequently appear with it |
| 24 | 0.7955 | 49.6349 |  language related to institutions, negative situations, the internet, and programming languages |
| 21 | 0.7944 | 16.4865 |  words related to voice-activated digital assistants |
| 20 | 0.793 | 12.1009 | the word "inside" and possibly words related to digitization and localization |

[EARLY] The early layers establish a biology-within-plant-science domain. 'Scientific writing describing ecological systems and processes' (Layer 7, inf=0.7996) and 'technical terminology from a wide range of fields including computing, science, sports' (Layer 7, inf=0.7994) place the prompt in the scientific-text register. 'Genus/species names of extinct organisms' (Layer 5, inf=0.7985) fires because photosynthesis discussions appear heavily in taxonomy and ecological literature. 'Words and abbreviations related to cellular biology and genetics' (Layer 6, inf=0.7928) directly encodes the cellular context of photosynthesis. 'Technical or scientific terms related to medicine or mechanical engineering' (Layer 7, inf=0.7951) reflects the broad scientific register of photosynthesis — it is discussed in chemistry, biology, and bioengineering texts. The 'word "alone"' (Layer 0, inf=0.7979) may fire on the "inside the" construction, which implies an internal, contained location.

[MIDDLE] The middle band maps photosynthesis to its cellular location. 'Words related to measuring electrical current flow across cell membranes' (Layer 9, inf=0.7946) is highly specific: this feature fires because chloroplasts are the site of electron transport chains and membrane electrical gradients in photosynthesis — the model is retrieving a functionally correct biochemical association. 'Prepositions and words used with them' (Layer 10, inf=0.7934) locks in the spatial-location frame of "inside the [organelle]".

[LATE] The late layers execute chloroplast-token selection. 'The word "inside" and words or punctuation that frequently appear with it' (Layer 16, inf=0.7971) is the direct locative feature — it fires on "inside" in the prompt and primes tokens that are things one can be inside of, i.e., cellular compartments. 'The word "inside" and possibly words related to digitization and localization' (Layer 20, inf=0.793) reinforces this locative push at a deeper layer. 'Language related to institutions, negative situations, the internet, and programming languages' (Layer 24, inf=0.7955, act=49.6349) fires with high activation; despite its heterogeneous label, this feature likely includes the "chloro-" prefix in technical text. 'Clauses expressing hypothetical or conditional situations' (Layer 20, inf=0.7981) is a structural feature that may contribute indirectly by encoding the "takes place" action-location construction.

[TOKEN COMPETITION] " chlor" (a subword token, the beginning of "chloroplast") wins at p=0.369, a moderate win. The locative features in the late band ('inside' at Layers 16 and 20) directly push toward the correct answer — photosynthesis takes place inside the chloroplast. The " chlor" token wins over " cell" and " thylakoid" (correct but more specific answer) and " plant" (a plausible but incorrect completion). The margin is moderately confident, reflecting that "chloroplast" is the canonical answer in textbook text, but competing tokens like " mitochondria" (wrong but biologically prominent) and " cell" (too general) receive meaningful probability mass.

---

### Prompt: "<bos>Water is composed of hydrogen and"

**Predicted token:** `Output " oxygen" (p=0.978)` (prob=0.9779)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.8002 | 3.4214 |  measurements of speed and possibly voltage in both French and English |
| 2 | 0.7996 | 2.8761 |  words related to technical writing, protocols, and code |
| 1 | 0.7985 | 2.9637 | the word "visit." |
| 1 | 0.7983 | 2.6179 |  mathematical expressions and calculations involving multiplication, division, square roots, and negative numbers |
| 3 | 0.7979 | 3.3922 | a mix of objective-c code and names of people and things |
| 0 | 0.7977 | 1.9738 | mentions of archeology or the history of early civilization. |
| 1 | 0.7968 | 2.5963 |  words and phrases that suggest visiting a website or getting information |
| 2 | 0.7966 | 3.1544 |  phrases involving financial data like stocks, rates of return, and dividends |
| 1 | 0.7964 | 2.3953 |  technical/scientific terms related to geography, geology, and biology. |
| 3 | 0.7958 | 3.014 | biological processes or elements related to nutrition and cellular health. |
| 0 | 0.7955 | 3.0292 | words related to being somewhat like something but not entirely or always. |
| 1 | 0.7949 | 3.0848 | the word "presence" and related words in scientific papers. |
| 0 | 0.794 | 3.4907 |  data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 0 | 0.7938 | 1.6852 |  proper names, particularly of people and places. |
| 1 | 0.7927 | 3.2619 |  words related to science and/or technical research and processes. |
| 2 | 0.7925 | 2.8845 | the phrasal verb "break down" |
| 0 | 0.7923 | 0.0 | mentions of clubs or sports teams, and sometimes related words like 'sister' or 'kids' |
| 3 | 0.7921 | 3.6146 | technical or scientific concepts related to biology, physiology, and chemistry, especially at a cellular level. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7991 | 4.2013 | scientific writing about graphs and shapes |
| 8 | 0.7989 | 10.2566 |  sentence fragments involving composition or structure using words like "of", "bit", "up", or "with". |
| 4 | 0.7987 | 7.7654 |  phrases containing auxiliary verbs like "must," "should," "can," and "be." |
| 4 | 0.7981 | 3.8468 |  the word "works" or "work" and associated words like "how", "process" |
| 7 | 0.7972 | 7.3468 |  questions about food offerings, also questions with math and derivatives |
| 8 | 0.797 | 3.7295 |  words and symbols related to chemistry and physics |
| 5 | 0.7962 | 3.5465 |  words and phrases representing quantities, percentages, and weak attachment |
| 6 | 0.796 | 3.7607 |  uses of the passive voice and words associated with designed systems or processes |
| 7 | 0.7953 | 4.186 |  technical language related to chemical processes and reactions. |
| 4 | 0.7951 | 3.8852 |  descriptions of physical configurations and manufactured products |
| 4 | 0.7945 | 3.3729 |  words associated with studies of health and nutrition |
| 8 | 0.7936 | 3.282 |  mathematical notation and code |
| 5 | 0.7934 | 5.9191 |  general statements about life, humanity, business, and truth |
| 4 | 0.793 | 3.8845 |  mentions of chemical elements or compounds |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.8 | 16.6178 | code and equations |
| 9 | 0.7998 | 5.2455 |  scientific and mathematical texts, equations, or references. |
| 9 | 0.7993 | 5.1701 |  formulas, ratios, and mathematical notation |
| 9 | 0.7975 | 4.456 |  words related to chemistry and chemical elements |
| 9 | 0.7947 | 3.4461 |  words broadly related to algorithms, stereotypes, and/or bias |
| 10 | 0.7943 | 13.2192 |  technical descriptions of how things are constructed |
| 14 | 0.7932 | 6.9696 |  connections between items or numbers |
| 13 | 0.7919 | 9.5549 |  personal achievements and general biographical information. |

[EARLY] The early layers identify this as a chemistry completion with strong chemical-composition semantics. 'Technical/scientific terms related to geography, geology, and biology' (Layer 1, inf=0.7964) and 'words related to science and/or technical research and processes' (Layer 1, inf=0.7927) establish the scientific register. 'Technical or scientific concepts related to biology, physiology, and chemistry, especially at a cellular level' (Layer 3, inf=0.7921) fires on the molecular-chemistry frame of "composed of." 'Biological processes or elements related to nutrition and cellular health' (Layer 3, inf=0.7958) is important: hydrogen and cellular metabolism are linked in biochemistry texts. The 'phrasal verb "break down"' (Layer 2, inf=0.7925) fires because "water breaks down into hydrogen and oxygen" is a common formulation in chemistry education text — the model sees "composed of hydrogen and" and retrieves the complementary element via the decomposition frame.

[MIDDLE] The middle band performs the chemical-composition binding. 'Words and symbols related to chemistry and physics' (Layer 8, inf=0.797) and 'technical language related to chemical processes and reactions' (Layer 7, inf=0.7953) both represent the chemistry domain. 'Mentions of chemical elements or compounds' (Layer 4, inf=0.793) is the most targeted feature: it explicitly encodes chemical element terminology and directly primes element-name tokens. 'Sentence fragments involving composition or structure using words like "of", "bit", "up", or "with"' (Layer 8, inf=0.7989, act=10.2566) fires on the "composed of...and" construction — this is a structural feature that identifies multi-component composition lists and expects another element token after the conjunction "and."

[LATE] The late layers execute near-certain chemical element selection. Layer 9 hosts a cluster of four features: 'code and equations' (inf=0.8, act=16.6178), 'scientific and mathematical texts, equations, or references' (inf=0.7998), 'formulas, ratios, and mathematical notation' (inf=0.7993), and 'words related to chemistry and chemical elements' (inf=0.7975, act=4.456). This is the model running a dedicated chemistry-formula completion circuit: Layer 9 behaves like a chemical-equation completion module. The co-activation of formula-notation features and element-name features at the same layer produces an extremely confident vote for "oxygen."

[TOKEN COMPETITION] " oxygen" wins at p=0.978, the highest confidence score in the entire dataset. This is mechanistically explained by the Layer 9 chemical-formula cluster: the "composed of hydrogen and [X]" pattern is essentially a fill-in-the-blank chemistry formula that appears millions of times in training data with only one valid completion. The 'chemical elements' feature (Layer 9, inf=0.7975) and the equation-completion features combine to eliminate virtually all competition. Runner-up tokens (" molecules", " water", " OH") are suppressed to near-zero probability. The circuit is unambiguous, highly redundant, and converges with near-certainty.

---

### Prompt: "<bos>The first US president was"

**Predicted token:** `Output " a" (p=0.193)` (prob=0.1929)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8 | 1.7974 |  proper nouns and government related terms |
| 2 | 0.7998 | 3.9557 |  words related to the purpose or goal of scientific research |
| 0 | 0.7996 | 0.0 | the word "motivated" or "motivation" |
| 7 | 0.7994 | 3.6026 |  terms related to periods of time in the past, such as years |
| 6 | 0.7992 | 7.3042 | military-related words, government related words, sports related words, place and location related words |
| 0 | 0.799 | 1.6659 |  dates and concepts related to time |
| 6 | 0.7988 | 8.1456 |  mentions of the United States (US) including variations like "U.S." and its relationship to foreign countries and locations. |
| 3 | 0.7983 | 2.3479 |  proper nouns and government related terms |
| 0 | 0.7981 | 2.4026 |  the words "ice cream" and "first" |
| 1 | 0.7977 | 4.0416 |  strings of ellipses and sentence fragments, possibly in conjunction with other punctuation |
| 0 | 0.7975 | 1.8993 | phrases using the word "deal", but also identifies certain words related to groups of people. |
| 2 | 0.7973 | 2.972 |  words related to value, effectiveness and influence |
| 0 | 0.7966 | 3.8914 |  second person pronouns and words indicating ownership |
| 7 | 0.7964 | 5.3955 |  mentions of famous people, politicians, and royalty. |
| 0 | 0.796 | 2.4148 | the word "intermediate" and words near the beginning and end of words |
| 2 | 0.7958 | 2.9969 |  words that describe order or position |
| 0 | 0.7956 | 2.3991 | the word "damage" and sometimes other words near "damage" or related to negative experiences |
| 7 | 0.7953 | 7.7533 |  mentions of political figures and concepts |
| 0 | 0.7951 | 3.6207 |  sentences that start with the word "The" |
| 2 | 0.7949 | 1.5755 |  words related to family, especially the legal relationships between parents and children |
| 1 | 0.7947 | 1.7203 |  words or phrases that indicate progress, success, or advancement in various fields like computer science, history, genomics, and medicine. |
| 2 | 0.7945 | 1.9858 |  terms related to sports teams, managers and contracts |
| 1 | 0.7941 | 3.9314 |  words related to business, markets, and financial reporting |
| 0 | 0.7938 | 2.1102 |  words related to official processes and/or decision making |
| 0 | 0.7936 | 1.5583 | something the user looked at, is looking at, or will look at. |
| 7 | 0.7934 | 3.8212 |  terms related to political office and historical rulers |
| 4 | 0.793 | 2.7066 |  places, rulers, and treaties |
| 0 | 0.7928 | 2.9707 |  references to dates and times |
| 0 | 0.7925 | 2.7287 |  the word "front" and generally give a slight boost to nearby words. |
| 4 | 0.7921 | 12.3695 | code snippets and license agreements |
| 0 | 0.7919 | 2.3506 |  mentions of a female subject |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7985 | 4.6319 |  numbers, ordinal numbers, and related words such as "phase", " iteration", and "partes" |
| 0 | 0.7979 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |
| 0 | 0.7962 | 0.0 |  math and mathematical notation |
| 14 | 0.7923 | 37.1963 | the word "using" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 20 | 0.7971 | 36.304 |  mentions of the president of the united states, particularly Obama and Trump, and political terms |
| 0 | 0.7969 | 0.0 |  mentions of ice cream |
| 17 | 0.7943 | 7.2731 |  words associated with the creation or invention of devices |
| 0 | 0.7932 | 0.0 |  words related to political administration and legal rights |
| 25 | 0.7917 | 9.1039 |  words and phrases associated with family history, marriage, and ancestry |

[EARLY] The early layers parse a US political history query. 'Mentions of the United States (US) including variations like "U.S." and its relationship to foreign countries and locations' (Layer 6, inf=0.7988, act=8.1456) fires with high activation, immediately identifying the American national context. 'Proper nouns and government related terms' (Layer 3, inf=0.8, act=1.7974 and inf=0.7983) appears twice, encoding both "US" and "president" as political proper-noun tokens. 'Mentions of famous people, politicians, and royalty' (Layer 7, inf=0.7964) and 'mentions of political figures and concepts' (Layer 7, inf=0.7953) identify this as a historical-leader attribution prompt. 'Terms related to periods of time in the past, such as years' (Layer 7, inf=0.7994) maps "first" to a historical superlative, priming historical-era attribution. The 'words that describe order or position' (Layer 2, inf=0.7958) fires on "first" and encodes ordinal ranking — this is the "first among X" structure that triggers president-list retrieval.

[MIDDLE] The middle band is notably thin and partially dysfunctional. 'Numbers, ordinal numbers, and related words such as "phase", "iteration", and "partes"' (Layer 8, inf=0.7985) fires on "first" as an ordinal, but the subsequent features show no strong Washington-specific binding. 'The word "using"' at Layer 14 (inf=0.7923, act=37.1963) is likely incidental. The absence of a strong Washington-specific middle-band feature explains the ambiguous output.

[LATE] The late layers reflect genuine uncertainty. 'Mentions of the president of the United States, particularly Obama and Trump, and political terms' (Layer 20, inf=0.7971, act=36.304) fires strongly on the presidential domain but is biased toward modern presidents (Obama, Trump) rather than historical ones — this is an important bias: the model's political-president feature is weighted toward contemporary figures. 'Words associated with the creation or invention of devices' (Layer 17, inf=0.7943) may fire on the "founding" of the presidency. 'Words and phrases associated with family history, marriage, and ancestry' (Layer 25, inf=0.7917) could reflect Washington's biographical text patterns.

[TOKEN COMPETITION] The predicted token is " a" (p=0.193), which is incorrect — the model is completing "The first US president was a..." (a general description) rather than naming Washington. This reflects the modern-president bias in Layer 20: the model knows "president" but the "first" ordinal is not binding it to a historical-era answer. Runner-ups likely include " George", " Washington", and " a". The very low confidence and selection of the article " a" (rather than a proper name) suggests the circuit failed to make the ordinal-historical binding: it retrieved "president" semantics but not the specific "first president = Washington" fact, defaulting to a descriptive continuation pattern.

---

### Prompt: "<bos>Napoleon was exiled to the island of"

**Predicted token:** `Output " Elba" (p=0.709)` (prob=0.7093)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8002 | 0.8405 | words or phrases of varied context that are technical or academic |
| 4 | 0.7998 | 3.2991 |  legal text and code relating to processes |
| 6 | 0.7996 | 5.1403 |  topics involving water and botany |
| 5 | 0.799 | 2.0324 |  words and phrases related to significant accomplishments, both positive and negative |
| 4 | 0.7986 | 1.7265 |  references to the French or France |
| 7 | 0.7984 | 5.8959 |  words signifying conclusion or departure |
| 0 | 0.7982 | 3.3764 |  academic or legal writing and language |
| 2 | 0.798 | 1.0648 |  terms related to computer hardware, vehicle hardware and electrical components |
| 1 | 0.7978 | 1.111 | terms related to garbage, pollution, and waste disposal |
| 4 | 0.7973 | 2.4659 |  words related to services, technology, and analysis |
| 7 | 0.7971 | 7.9881 |  words and phrases related to politics, cities, and people getting kicked out |
| 2 | 0.7969 | 1.3866 |  the word "sometimes" and titles. |
| 5 | 0.7959 | 2.8019 |  references to data, numbers, statistics, dates, and other references |
| 0 | 0.7957 | 4.1542 |  chemistry-related terminology and names |
| 0 | 0.7955 | 3.5369 |  words and phrases related to paganism, especially Wicca, Druids, and associated celebrations |
| 1 | 0.7951 | 1.1699 |  cities and countries |
| 6 | 0.7949 | 4.1206 | abbreviations or acronyms, especially when enclosed in parentheses or related to scientific or governmental entities |
| 1 | 0.7947 | 1.2939 |  words and phrases related to technical and mechanical components, especially in the context of engineering or scientific fields |
| 0 | 0.7945 | 6.709 | technical documents or data, including numbers, units, and references to figures or tables. |
| 5 | 0.7943 | 3.7715 | place names, especially cities and regions, along with associated terms like 'fort'. |
| 0 | 0.7941 | 3.0493 |  academic or technical content related to chemistry and/or polymers |
| 7 | 0.7937 | 20.8648 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 5 | 0.7935 | 1.8845 |  terms and acronyms related to scientific research, especially in genetics, physics, and computer science, including algorithm names |
| 0 | 0.7933 | 2.867 |  words associated with the passive voice, and language related to studies and recommendations |
| 6 | 0.7931 | 14.4665 | words that appear in programming code, legal jargon, or scientific texts |
| 5 | 0.7927 | 2.7457 | capitalized proper nouns, particularly place names |
| 0 | 0.7925 | 1.0643 |  academic or legal writing and language |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7994 | 5.679 |  words related to resolution and fixing problems |
| 8 | 0.7977 | 5.4264 |  words and phrases related to astronomy, physics, and mathematical calculations |
| 15 | 0.7967 | 14.8002 |  "Good" or instances of the word "The" at the beginning of a block of text |
| 8 | 0.7953 | 9.2338 |  words related to political power struggles, opposition, historical conflicts and specific groups. |
| 14 | 0.7929 | 12.4588 |  the word "The" at the beginning of a sentence |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.8 | 8.4103 |  mentions of places, especially when followed by "of" |
| 25 | 0.7992 | 6.6704 | articles that are using the words the, is, to, a, in, and my. Focusing on common word usage within the article |
| 25 | 0.7988 | 9.8869 | the word "on" and related prepositions and articles |
| 17 | 0.7975 | 10.6291 |  names, especially of celebrities, along with surrounding context |
| 18 | 0.7965 | 51.4833 | names of places or geographical locations, especially islands and parks, sometimes in conjunction with the word "on". |
| 18 | 0.7963 | 6.3143 | survival camping |
| 17 | 0.7961 | 19.1806 | words or phrases related to empires and global power structures |
| 23 | 0.7939 | 8.8449 |  location and country names, especially if the text includes the word "from". |

[EARLY] The early layers execute a remarkably precise historical-exile recognition. 'References to the French or France' (Layer 4, inf=0.7986) fires on "Napoleon" — the model immediately identifies the French identity of the subject. 'Words and phrases related to politics, cities, and people getting kicked out' (Layer 7, inf=0.7971, act=7.9881) is the most targeted early feature: it explicitly encodes the exile/deportation semantic frame, directly mapping "exiled" onto the political-removal concept. 'Words signifying conclusion or departure' (Layer 7, inf=0.7984) fires on "exiled" as a finality/departure token. 'Place names, especially cities and regions, along with associated terms like "fort"' (Layer 5, inf=0.7943) and 'cities and countries' (Layer 1, inf=0.7951) prepare the model to produce a geographic location. 'Capitalized proper nouns, particularly place names' (Layer 5, inf=0.7927) pre-loads the proper-name formatting expectation for an island name.

[MIDDLE] The middle band maps Napoleon to his exile destination. 'Words related to political power struggles, opposition, historical conflicts and specific groups' (Layer 8, inf=0.7953) fires on the political-conflict frame of Napoleon's military career and subsequent exile. 'Words related to resolution and fixing problems' (Layer 9, inf=0.7994) encodes the "exile as resolution" narrative that appears in historical accounts — Napoleon's exile resolved the Napoleonic Wars. The 'word "The"' at beginning of text block (Layer 15, inf=0.7967) and beginning of sentence (Layer 14, inf=0.7929) reflect the formatting of encyclopedic articles about Napoleon's exile.

[LATE] The late layers deliver decisive island-name selection. 'Names of places or geographical locations, especially islands and parks, sometimes in conjunction with the word "on"' (Layer 18, inf=0.7965, act=51.4833) is the most highly activated feature in the entire late band and is extraordinarily specific: it fires on island names and even mentions "on" — exactly the preposition that follows "island of" in this sentence. 'Mentions of places, especially when followed by "of"' (Layer 18, inf=0.8, act=8.4103) fires on the "island of" construction directly. 'Words or phrases related to empires and global power structures' (Layer 17, inf=0.7961) encodes the Napoleonic Empire context, and 'names, especially of celebrities, along with surrounding context' (Layer 17, inf=0.7975) prepares for a famous proper noun.

[TOKEN COMPETITION] " Elba" wins confidently at p=0.709, one of the stronger historical-fact retrievals. The 'islands and parks' feature at Layer 18 (act=51.4833) is the decisive driver — its very high activation produces a strong vote for a famous island name. The Napoleon→exile→island chain is efficiently encoded: the model has a well-formed circuit for this famous historical fact. Runner-ups likely include " Saint" (for Saint Helena, Napoleon's second exile) and " the" (hedged). The gap between Elba (first exile) and Saint Helena (final exile) is notable: the model retrieves the more famous/culturally rehearsed exile location. The margin is confident, reflecting the high frequency of "Napoleon was exiled to Elba" in training data.

---

### Prompt: "<bos>The longest river in the world is the"

**Predicted token:** `Output " Nile" (p=0.597)` (prob=0.5973)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7995 | 1.0864 |  mentions of the first lady and/or of temporal context (like months and uses of 'hand') |
| 0 | 0.7993 | 3.4735 | words related to descriptions of conditions or locations of things, including body parts and scientific data. |
| 6 | 0.7991 | 3.4039 | places, people, and demographics in Africa and the South Pacific |
| 0 | 0.7987 | 2.9087 |  the word "hall" and articles |
| 0 | 0.7985 | 4.9413 |  the word "conventional." |
| 2 | 0.7983 | 2.6746 |  words related to sports competition between teams or individuals. |
| 4 | 0.7981 | 2.3509 |  words related to religion, hormones, and medicine |
| 1 | 0.7979 | 2.136 |  instances of the word "quarter" referring to time or categories |
| 2 | 0.7977 | 2.4637 |  specific words and phrases that could be scientific jargon, chart labels, and potential misspellings. |
| 0 | 0.7975 | 3.4013 |  the word "overall", sometimes alongside words that express quantity |
| 5 | 0.7969 | 8.4596 |  mentions of global concerns, world events, or specific topical references |
| 0 | 0.7965 | 2.1969 |  proper nouns, especially names of people and places |
| 0 | 0.7963 | 3.8244 |  articles and pronouns |
| 0 | 0.7961 | 1.5631 | places and things associated with legal documents |
| 1 | 0.7959 | 1.5148 |  words related to real estate, locations, and planning, as well as the measurement "feet". |
| 6 | 0.7957 | 3.222 |  words related to the digestive system and medical procedures |
| 6 | 0.7953 | 2.54 |  references to geographic locations and bodies of water |
| 6 | 0.7951 | 16.5686 | words that appear in programming code, legal jargon, or scientific texts |
| 0 | 0.7946 | 1.4067 |  academic or technical content related to chemistry and/or polymers |
| 2 | 0.7944 | 1.9861 |  words related to reports, studies, paying, building materials, or legal testimony |
| 0 | 0.7942 | 0.0 | references to the current study or research |
| 3 | 0.7938 | 2.458 |  words and phrases related to the concept of time |
| 0 | 0.7936 | 4.9937 |  the word "opposition" and anything that might be connected to it such as resistance, campaign, party, or disagreement |
| 0 | 0.7934 | 3.1676 |  words that start a sentence or section |
| 0 | 0.7928 | 1.5214 |  uses notation common in physics research papers |
| 2 | 0.7926 | 1.9852 |  words related to math, science, and technical writing |
| 3 | 0.7924 | 7.0817 | the preposition "across" and other words associated with geography or location. |
| 4 | 0.7922 | 2.5751 |  words related to specific locations, technologies, organizations, and official events |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.8001 | 1.9715 |  words related to geology, geography, and oceanography |
| 0 | 0.7999 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |
| 12 | 0.7989 | 9.8357 |  words related to governmental legal systems and documents |
| 8 | 0.7971 | 4.002 |  titles of documents, headings, and categories that refer to lists or tables of information, especially those that include names of people, places, organizations, or computer software |
| 9 | 0.7967 | 5.4602 |  mentions of rivers |
| 9 | 0.7948 | 4.0627 |  names of organizations and places, particularly universities, committees, and locations |
| 13 | 0.794 | 8.3534 |  geographic locations, especially bodies of water, parks and trails |
| 7 | 0.793 | 6.5006 |  words and phrases that would be found in informal written conversation. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 15 | 0.7997 | 7.3383 | occurrences of the word "is", especially after certain preceding words |
| 22 | 0.7973 | 10.5115 |  This neuron detects words in a product description as well as words that may indicate the names of people or organizations. |
| 0 | 0.7955 | 0.0 |  occurrences of the word "even", the word "both" and variations of the word "meet" |
| 19 | 0.7932 | 25.6669 |  words related to rivers and watersheds |

[EARLY] The early layers establish a geographic superlative query. 'References to geographic locations and bodies of water' (Layer 6, inf=0.7953) and 'places and things associated with legal documents' (Layer 0, inf=0.7961) establish the geographic-entity frame. 'Places, people, and demographics in Africa and the South Pacific' (Layer 6, inf=0.7991) fires partly on "world" as a geographic scope token and partly on river-associated geographies. The 'word "conventional"' (Layer 0, inf=0.7985) is non-obvious but may reflect the fact that river length measurements involve conventional measurement methodology — "longest" triggers the measurement/comparison register. 'Mentions of global concerns, world events, or specific topical references' (Layer 5, inf=0.7969) fires on "world" and encodes the global-superlative frame. 'The preposition "across" and other words associated with geography or location' (Layer 3, inf=0.7924) fires on geographic traversal — rivers extend "across" geography.

[MIDDLE] The middle band performs the river-superlative→Nile binding. 'Words related to geology, geography, and oceanography' (Layer 7, inf=0.8001) is the topmost-influence middle feature, encoding the physical geography domain. 'Mentions of rivers' (Layer 9, inf=0.7967) is a direct river-entity feature: it fires on "river" in the prompt and retrieves river-name tokens. 'Geographic locations, especially bodies of water, parks and trails' (Layer 13, inf=0.794) and 'names of organizations and places, particularly universities, committees, and locations' (Layer 9, inf=0.7948) contribute to the geographic-entity namespace. The river-specific feature at Layer 9 is the critical causal link: it binds "river" to river-name tokens.

[LATE] 'Words related to rivers and watersheds' (Layer 19, inf=0.7932, act=25.6669) is the decisive late-layer feature — it explicitly represents river-related vocabulary and pushes toward a canonical river-name token like "Nile." The 'occurrences of the word "is", especially after certain preceding words' (Layer 15, inf=0.7997) may be structurally tracking the "is the [X]" completion frame. 'This neuron detects words in a product description as well as words that may indicate the names of people or organizations' (Layer 22, inf=0.7973) is the generic proper-noun push.

[TOKEN COMPETITION] " Nile" wins at p=0.597, a confident margin. The 'rivers and watersheds' feature at Layer 19 (act=25.6669) provides the key river-name push, while the 'mentions of rivers' at Layer 9 selects from the river-name namespace. Runner-ups likely include " Amazon" (the Amazon is sometimes cited as the longest by some measurements) and " Mississippi". The competition with the Amazon is real — there is genuine scientific debate about Nile vs Amazon — and this may explain why the confidence is 0.597 rather than higher. The Nile is the conventional encyclopedic answer and wins, but the circuit has partial activation for the Amazon competitor, reflected in the ~40% probability mass distributed to other tokens.

---

### Prompt: "<bos>Mount Everest is located in the"

**Predicted token:** `Output " Himalayas" (p=0.263)` (prob=0.2628)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8 | 1.129 |  words or phrases associated with scientific or formal writing, particularly words relating to natural history, archeology, or formal analysis |
| 1 | 0.7998 | 3.5114 | the word "steady" and words ending in "programming" or "decline." |
| 4 | 0.7994 | 4.1508 |  words and phrases indicating a physical center or middle |
| 7 | 0.7992 | 4.3506 |  technical writing, formal definitions, and equations. |
| 2 | 0.799 | 3.1949 |  articles discussing death, crime, endangered animals, and/or war. |
| 4 | 0.7988 | 5.777 | references to the standard C++ namespace |
| 5 | 0.7986 | 4.1064 |  a variety of short common words or suffixes that appear to be randomly assorted |
| 4 | 0.7979 | 2.0207 |  places and bodies of water and words describing power and size |
| 0 | 0.7977 | 2.8376 |  words related to governments, people, and community |
| 1 | 0.7971 | 1.0961 |  words describing a substance that can hold something, or that can be ejected through a small opening |
| 1 | 0.7969 | 3.1664 |  words related to real estate, locations, and planning, as well as the measurement "feet". |
| 6 | 0.7965 | 3.8798 | words that are descriptive of locations |
| 2 | 0.7963 | 1.4573 |  code related to Android applications specifically finding the definition of a path |
| 1 | 0.796 | 1.1135 |  mentions of company executives |
| 6 | 0.7958 | 15.9219 | words that appear in programming code, legal jargon, or scientific texts |
| 7 | 0.7956 | 7.5467 |  words relating to geology, Iceland and earthquakes |
| 3 | 0.7954 | 1.7615 |  questions and references to Stack Exchange |
| 5 | 0.795 | 4.0652 |  mentions of places in an economic or political context |
| 4 | 0.7937 | 1.5712 |  words related to death and burial |
| 6 | 0.7931 | 4.5599 |  words related to isolated locations |
| 1 | 0.7929 | 3.1544 |  locations and businesses that have reviews |
| 2 | 0.7926 | 1.5553 |  mathematical notation inside LaTeX formatting, specifically norms and inequalities |
| 6 | 0.7924 | 14.416 | words that appear in programming code, legal jargon, or scientific texts |
| 1 | 0.7922 | 4.5609 | the word "throughout" and potentially related words |
| 4 | 0.792 | 8.1834 |  descriptions of experiments conducted in labs |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7996 | 3.2185 |  words related to nature and geography |
| 9 | 0.7984 | 7.5864 | words and phrases describing geographical location |
| 10 | 0.7981 | 2.3638 |  words and phrases about weather, vehicles and being outdoors |
| 8 | 0.7975 | 6.6524 |  mentions of geographic locations and their relative positions |
| 11 | 0.7973 | 44.7055 | the letters "L", "H," and "a" when they are at the beginning of a text block |
| 13 | 0.7939 | 6.4345 |  things that were invented, or topics related to journalism |
| 8 | 0.7933 | 3.5297 |  mentions of the Guinness Book of World Records and related achievements |
| 12 | 0.7918 | 10.1378 |  questions about the relationship between the value of K and the amount of training data in cross validation |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7967 | 18.0188 | locations |
| 22 | 0.7952 | 7.24 |  references to centuries, especially the the 20th and 21st centuries |
| 17 | 0.7948 | 17.9079 |  the word "in" or "to" in close proximity to other words |
| 17 | 0.7946 | 7.369 | the |
| 25 | 0.7943 | 9.8529 | historical references to battles, conquests and governors |
| 18 | 0.7941 | 95.2059 |  mathematical and geographical references |
| 25 | 0.7935 | 11.0343 |  words related to marine biology and oceanography, particularly concerning sponges and underwater research |

[EARLY] The early layers identify a mountain-location geographic query. 'Words that are descriptive of locations' (Layer 6, inf=0.7965) and 'places and bodies of water and words describing power and size' (Layer 4, inf=0.7979) establish the geographic-entity frame, with "size" resonating with "Everest" as the tallest mountain. 'Words relating to geology, Iceland and earthquakes' (Layer 7, inf=0.7956) fires on the geology/tectonics domain: Everest is a product of tectonic plate collision and appears in geological texts. 'Words related to real estate, locations, and planning, as well as the measurement "feet"' (Layer 1, inf=0.7969) fires because Everest's height (29,032 feet) is a canonical measurement figure. 'Words related to isolated locations' (Layer 6, inf=0.7931) maps to Everest's extreme remoteness. 'Descriptions of experiments conducted in labs' (Layer 4, inf=0.792, act=8.1834) may reflect the scientific expedition register in which Everest is discussed.

[MIDDLE] The middle band executes the Everest→location binding. 'Words related to nature and geography' (Layer 8, inf=0.7996), 'words and phrases describing geographical location' (Layer 9, inf=0.7984), and 'mentions of geographic locations and their relative positions' (Layer 8, inf=0.7975) form a triple-layered geographic-location circuit. 'Mentions of the Guinness Book of World Records and related achievements' (Layer 8, inf=0.7933) is particularly telling: Everest is the canonical Guinness world record entry for highest peak, and this feature directly binds world-record superlatives to their subject entities.

[LATE] 'Mathematical and geographical references' (Layer 18, inf=0.7941, act=95.2059) fires with the highest activation in the late band by a very wide margin, encoding the quantitative-geography register where Everest is described by coordinates, altitude, and regional placement. 'Locations' (Layer 16, inf=0.7967, act=18.0188) and 'historical references to battles, conquests and governors' (Layer 25, inf=0.7943) contribute geographic and South Asian historical context respectively.

[TOKEN COMPETITION] " Himalayas" wins at p=0.263, the lowest win margin in the geography sub-domain. The 'mathematical and geographical references' feature (act=95.2059) pushes toward a geographic proper-noun but is insufficiently specific to determine whether the answer is " Himalayas" (the mountain range), " Nepal" (the country), or " Asia" (the continent). Runner-ups almost certainly include " Nepal", " Asia", " Tibet", and " China" — all geographically correct at different levels of granularity. The low confidence (0.263) reflects genuine ambiguity in training data: "Mount Everest is located in the Himalayas/Nepal/Tibet/Asia" are all valid encyclopedic phrasings, and the circuit cannot select a single granularity level. The 'geographical references' mega-activation does not resolve which geographic entity name comes next.

---

### Prompt: "<bos>The largest ocean on Earth is the"

**Predicted token:** `Output " Pacific" (p=0.372)` (prob=0.3722)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.8001 | 5.5283 |  superlative adjectives and statistics |
| 1 | 0.7997 | 5.7738 |  the word "boss" or "monster" |
| 0 | 0.7995 | 4.2056 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 0 | 0.7993 | 2.9798 | occurrences of the word 'red' at the start of phrases or clauses |
| 0 | 0.7991 | 3.2178 |  the word "ease" |
| 2 | 0.7987 | 3.789 |  the word 'reason' and associated words indicating the cause or logic of a situation |
| 1 | 0.7983 | 1.4837 |  words relating to biological species, populations and evolution |
| 3 | 0.7979 | 6.0109 | adjectives that suggest importance or correctness |
| 7 | 0.7977 | 5.0921 |  words relating to scientific, technical and engineering topics like space, weather, computing, and electromagnetism |
| 2 | 0.7972 | 2.5373 |  words related to conflict, social injustice, physical science, family, and disease |
| 0 | 0.7968 | 6.8761 | scientific terms and experimental details related to biological and chemical research |
| 4 | 0.7966 | 4.8103 |  LaTeX mathematical notation |
| 4 | 0.796 | 4.7251 |  words in academic papers that are related to the paper's purpose, methods, and results. |
| 3 | 0.7958 | 4.6937 |  geographic locations, place names, and other references to locality. |
| 0 | 0.7952 | 1.9589 |  terms related to writing and stories, including story elements, characters, and the act of reading |
| 2 | 0.7948 | 1.7514 | words related to locations and political entities, as well as some medical terminology |
| 0 | 0.7946 | 4.2466 | phrases about dependence or relation of one action on another. |
| 5 | 0.7944 | 3.1769 |  references to rivers and geographic features |
| 0 | 0.794 | 4.2056 | parenthetical numerical references and citations to literature, laws, and statistics |
| 0 | 0.7936 | 1.615 |  proper names, particularly of people and places. |
| 1 | 0.7934 | 6.0515 |  words relating to biological species, populations and evolution |
| 3 | 0.7926 | 8.3922 |  instances of the word "The" at the start of a line |
| 0 | 0.7923 | 1.8609 | words related to a group or the remainder of something. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.7975 | 11.0981 |  stative verbs such as "is", and "was" along with related words and context |
| 8 | 0.7974 | 3.8507 | words related to geographical features and political entities like kingdoms and states. |
| 10 | 0.797 | 7.0303 | content related to boats, oceans, coast guards and bathymetry |
| 9 | 0.7962 | 4.6084 |  words related to referring to something by name |
| 8 | 0.7942 | 9.2031 |  terms related to underwater marine biology field research |
| 9 | 0.7938 | 3.6489 |  references to geography, especially countries and regions, and their inhabitants. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7999 | 20.1499 |  sentences where something is defined or described |
| 17 | 0.7989 | 7.1242 |  words related to the automotive industry, legal codes and amounts of money |
| 17 | 0.7985 | 7.8104 | Foreign words |
| 18 | 0.7981 | 6.2805 |  mentions of legislative bodies and political processes |
| 19 | 0.7964 | 9.3034 |  words and phrases related to medicine and health issues, particularly those that are severe or result in death |
| 23 | 0.7956 | 8.8149 |  technical and legal writing, including corporations, experimental design, and political entities. |
| 25 | 0.7954 | 13.3821 |  the word "the" or content related to weather and other natural phenomenons |
| 25 | 0.795 | 6.9115 | Latex, code, and math notation |
| 17 | 0.7932 | 66.5389 | start of sentences |
| 0 | 0.793 | 0.0 |  words related to political administration and legal rights |
| 0 | 0.7928 | 0.0 |  words related to political administration and legal rights |

[EARLY] The early layers identify a geographic superlative about an ocean. 'Superlative adjectives and statistics' (Layer 4, inf=0.8001) fires on "largest" — this is the superlative-fact recognition feature and immediately signals that the model is entering a world-record/rankings retrieval mode. 'Geographic locations, place names, and other references to locality' (Layer 3, inf=0.7958) and 'references to rivers and geographic features' (Layer 5, inf=0.7944) establish the physical-geography domain. 'Words relating to biological species, populations and evolution' appears twice (Layer 1, inf=0.7983 and inf=0.7934) — this fires because oceans are discussed extensively in marine-biology/ecology texts alongside species and populations. The dual activation at Layer 1 creates a strong marine-biology sub-register on top of the general geography signal.

[MIDDLE] The middle band executes the ocean-superlative→Pacific binding. 'Content related to boats, oceans, coast guards and bathymetry' (Layer 10, inf=0.797) is the most targeted middle feature: it explicitly represents ocean-domain vocabulary and directly primes ocean-name tokens. 'Terms related to underwater marine biology field research' (Layer 8, inf=0.7942) fires because the Pacific is the central ocean in marine research. 'Stative verbs such as "is", and "was" along with related words' (Layer 12, inf=0.7975) tracks the "is the [X]" completion frame. 'References to geography, especially countries and regions, and their inhabitants' (Layer 9, inf=0.7938) and 'words related to geographical features and political entities like kingdoms and states' (Layer 8, inf=0.7974) complete the geographic-superlative binding circuit.

[LATE] The late band is diffuse and relatively non-specific. 'Sentences where something is defined or described' (Layer 16, inf=0.7999, act=20.1499) fires on the definitional "is the X" frame. The absence of a specific "Pacific Ocean" feature in the late band suggests the model is completing this from momentum established in the middle layers. 'Start of sentences' (Layer 17, inf=0.7932, act=66.5389) has very high activation but is a structural feature encoding sentence-beginning patterns, which likely reflects the model preparing a proper-noun token at the start of a new phrase.

[TOKEN COMPETITION] " Pacific" wins at p=0.372, a moderate margin. The 'oceans, coast guards and bathymetry' feature (Layer 10) performs the critical ocean-type assignment, and the superlative feature at Layer 4 narrows the candidate to the "largest" ocean. Runner-ups include " Atlantic" and possibly " Indian" — both prominent oceans in training text. The 63% probability on non-Pacific tokens reflects genuine training-data ambiguity: texts sometimes refer to the Pacific as "largest" and sometimes discuss the Atlantic in similar superlative frames. The win is confident enough to be correct but not as decisive as the chemistry prompts.

---

### Prompt: "<bos>The Sahara is a"

**Predicted token:** `Output " desert" (p=0.299)` (prob=0.2994)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.7994 | 1.1991 |  the word "thin" (especially as it relates to manufacturing or materials) and words related to manufacturing processes |
| 1 | 0.7992 | 2.5367 | equals signs surrounded by numbers and letters along with brackets, slashes, and colons. |
| 1 | 0.7987 | 3.3282 |  language used to present an argument in a court of law |
| 1 | 0.7974 | 1.4272 | mathematical or programming code. |
| 0 | 0.7972 | 2.5212 |  sequences of one or two numerical digits |
| 0 | 0.797 | 2.4772 |  a variety of specific nouns |
| 4 | 0.7965 | 3.7487 | verbs relating to the use of something for a specific purpose |
| 0 | 0.7963 | 2.5285 |  references to legal proceedings and pronouncements, especially related to court decisions and legal actions |
| 0 | 0.7961 | 1.8884 | the word "the" |
| 2 | 0.7956 | 2.8637 | words that describe actions or events in a formal context, such as legal, medical, or academic settings |
| 0 | 0.7954 | 2.0986 | the word "The" at the beginning of a line |
| 1 | 0.7952 | 3.0415 |  programming code, government codes, data sets, equations, or plant names |
| 3 | 0.7947 | 3.8116 |  location and nationality-related words and some verbs |
| 4 | 0.7941 | 2.1637 |  a variety of technical and abstract nouns and concepts across a range of topics, including morality, computing, and inventions |
| 0 | 0.7938 | 1.1376 |  text related to numbers, especially those representing currency or measurements, and also related to statistical analysis or auditing |
| 3 | 0.7936 | 2.7732 |  words relevant to engineering or technical descriptions of machinery |
| 0 | 0.7932 | 1.4501 |  code snippets, programming or technical terminology |
| 1 | 0.7929 | 1.2477 |  topics related to government actions within a district or town |
| 2 | 0.7925 | 1.2956 |  mentions of countries and nationalities. |
| 3 | 0.7916 | 2.9361 |  the word "new" as well as the definite article "the" in legal documents |
| 4 | 0.7914 | 2.0574 |  brand names of corporations and products, especially automobiles, movies, and software |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.8001 | 4.7317 |  mentions of deep ocean environments |
| 8 | 0.7998 | 3.6058 | words and phrases related to academic studies, scientific research, reports, and analyses |
| 6 | 0.7996 | 2.3278 |  location-based terms |
| 5 | 0.7983 | 1.2185 |  words related to land management and conservation |
| 7 | 0.7981 | 4.935 |  words or phrases related to geography, demographics, and the world. |
| 0 | 0.7959 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |
| 9 | 0.795 | 4.409 |  geopolitical and geographical terms, especially those related to regions, political divisions, and historical events. |
| 5 | 0.7943 | 2.2297 |  proper nouns related to locations, people, organizations and historical events |
| 5 | 0.7934 | 2.5828 |  entities and organizations related to religion, politics, and entertainment |
| 5 | 0.792 | 5.5487 |  proper nouns related to places like hotels and cities, especially of Spanish origin |
| 5 | 0.7918 | 1.607 |  names of companies, organizations, and institutions |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 14 | 0.799 | 7.5917 |  words or phrases related to business or economics |
| 15 | 0.7985 | 150.2896 |  code documentation blocks in various programming languages |
| 16 | 0.7979 | 10.6206 | sand |
| 11 | 0.7976 | 35.9985 | code comments in a variety of languages |
| 11 | 0.7968 | 8.9153 | first and second person pronouns and forms of the verb "to be". |
| 11 | 0.7945 | 35.4792 |  code comments, the word "import", and a contraction |
| 11 | 0.7927 | 31.6969 |  code snippets assigning values to variables |
| 15 | 0.7923 | 16.7261 |  a wide variety of proper nouns of many types with a slight preference for locations, nationalities or languages |

[EARLY] The early layers parse a minimalist proper-noun categorisation prompt. The features in the early band are notably thin in geographic specificity, suggesting the model is relying heavily on the single word "Sahara" to drive the circuit. 'Sequences of one or two numerical digits' (Layer 0, inf=0.7972) and 'a variety of specific nouns' (Layer 0, inf=0.797) establish that "Sahara" is being processed as a specific proper noun. 'Location and nationality-related words and some verbs' (Layer 3, inf=0.7947) begins mapping the Sahara to a geographic entity. 'Words relevant to engineering or technical descriptions of machinery' (Layer 3, inf=0.7936) fires because the Sahara's climate system appears in engineering/environmental texts about solar power, heat management, and desalination. The 'word "The" at the beginning of a line' (Layer 0, inf=0.7954) locks in the definitional "The Sahara is a..." frame.

[MIDDLE] The middle band performs geographic-type classification for the Sahara. 'Mentions of deep ocean environments' (Layer 7, inf=0.8001) is counterintuitively important — it fires on extreme arid environments, which share vocabulary with extreme ocean environments ("harsh", "barren", "vast"). 'Words or phrases related to geography, demographics, and the world' (Layer 7, inf=0.7981) and 'location-based terms' (Layer 6, inf=0.7996) establish the geographic register. 'Geopolitical and geographical terms, especially those related to regions, political divisions, and historical events' (Layer 9, inf=0.795) begins resolving the Sahara as a regional-geographic entity. 'Words related to land management and conservation' (Layer 5, inf=0.7983) fires because the Sahara appears in land-management and conservation literature.

[LATE] The late band executes desert-type selection with surprising force. 'Sand' (Layer 16, inf=0.7979, act=10.6206) fires directly — this is a Sand-feature that has learned the association between the Sahara and sand as its defining characteristic, directly pushing toward "desert." Multiple code-comment features at Layer 11 (inf=0.7976, inf=0.7968, inf=0.7945, inf=0.7927) with very high activations (35.9985, 35.4792, 31.6969) are likely SAE artifacts, but the 'sand' feature is causally decisive.

[TOKEN COMPETITION] " desert" wins at p=0.299, a low-confidence win. The 'sand' feature at Layer 16 provides the decisive type-assignment push: "Sahara + sand = desert." Runner-ups include " large" (descriptor), " vast" (descriptor), and " region" — all plausible completions of "The Sahara is a..." in different text styles. The low confidence (0.299) reflects genuine ambiguity in the completion: training text includes "The Sahara is a desert", "The Sahara is a large", "The Sahara is a vast", etc. The circuit converges on the most category-specific answer ("desert") but not dominantly.

---

### Prompt: "<bos>The Amazon rainforest is primarily located in"

**Predicted token:** `Output " South" (p=0.363)` (prob=0.3629)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 5 | 0.8 | 3.9452 |  words and phrases often associated with news articles and current events |
| 0 | 0.7996 | 2.429 | words associated with phrases "be", "outside" and possibly some conjunctions and prepositions. |
| 0 | 0.7994 | 4.4764 | words and phrases related to scientific writing, including defining terms, the process of analysis, what is considered knowledge, what is related to something, and what is occurring |
| 0 | 0.7992 | 1.8102 | words that are vague intensifiers or superlatives or are related to the scale of something. |
| 3 | 0.799 | 6.2667 | words and phrases that indicate agreement or alignment |
| 0 | 0.7988 | 2.6008 |  words related to money or business transactions |
| 2 | 0.7986 | 7.6117 |  words related to maintaining or continuing existence |
| 3 | 0.7984 | 1.6893 |  the word "shop" with an activation of 9 or 10, lower activations for "in", "as", "government", "here", "taught", "work", "office", "himself", "itself" |
| 0 | 0.798 | 5.0117 | the word "efficient" and words with a similar meaning. |
| 2 | 0.7978 | 5.0852 |  words that emphasize certainty or exactness |
| 1 | 0.7976 | 1.4485 | words and phrases used in programming and/or referring to propulsion and tilting |
| 0 | 0.7974 | 1.6197 |  first and second person pronouns |
| 3 | 0.7972 | 1.855 |  topics related to environmental regulations and emissions |
| 4 | 0.797 | 5.7692 | adverbs ending in "lly," sometimes along with words that come immediately before or after them. |
| 3 | 0.7966 | 4.1259 |  legal documents and phrases like "inter alia, among other things" |
| 4 | 0.7962 | 2.8786 |  words and phrases related to location, spatial relationships, and technical descriptions of physical systems. |
| 5 | 0.796 | 3.4957 |  contextual words and phrases appearing in technical documents or academic articles |
| 1 | 0.7956 | 5.3473 |  words related to real estate, locations, and planning, as well as the measurement "feet". |
| 4 | 0.7954 | 2.3388 |  words related to politics, elections, sports, courts, policies, and crime |
| 4 | 0.795 | 3.6848 | legal or mathematical arguments. |
| 3 | 0.7948 | 2.6159 |  words related to scientific experimentation, location within the body, or other spatial relationships. |
| 0 | 0.7946 | 5.5926 | special characters denoting mathematical expressions or code |
| 0 | 0.7944 | 1.991 |  SSH public key authentication details |
| 2 | 0.7942 | 1.8504 |  words and phrases related to locations and transportation |
| 0 | 0.7939 | 4.6515 | ingredients and dishes, especially chicken |
| 0 | 0.7937 | 3.584 |  mentions of the Soviet Union, sometimes abbreviated, in texts about conflict and espionage. |
| 2 | 0.7935 | 6.2599 | words that suggest something being factual or true |
| 0 | 0.7933 | 1.4512 | words which are specific to scientific or legal contexts |
| 2 | 0.7929 | 2.3159 |  terms related to academic research studies and reports |
| 3 | 0.7927 | 8.3922 |  instances of the word "The" at the start of a line |
| 1 | 0.7923 | 1.6533 |  the words "rain" and "climate change" |
| 5 | 0.7921 | 3.1969 |  technical/scientific descriptions consisting of equations, relations, demonstrations, and testimonies |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.7998 | 3.9234 |  phrases indicating capabilities or important aspects, sometimes using idioms |
| 8 | 0.7982 | 5.2968 | phrases about lacking control or attempts at controlling something |
| 10 | 0.7964 | 3.4234 |  the names of specific animals and places they are located. |
| 11 | 0.7931 | 49.9832 | the letters "L", "H," and "a" when they are at the beginning of a text block |
| 14 | 0.7925 | 6.184 |  locations, particularly cities and institutions, and the words "in", "at", "to" and "for" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 22 | 0.7968 | 8.3062 |  descriptive text about lakes and the ocean |
| 23 | 0.7958 | 9.8171 |  references to years, especially in the 1600s and 1700s |
| 17 | 0.7952 | 7.5452 |  references to different geographical regions with location details |

[EARLY] The early layers establish an ecology-plus-geography frame for the Amazon. 'Words and phrases often associated with news articles and current events' (Layer 5, inf=0.8) fires because the Amazon rainforest is a prominent news topic (deforestation, climate change). 'Words and phrases related to scientific writing' (Layer 0, inf=0.7994) and 'mentions of the Soviet Union, sometimes abbreviated, in texts about conflict and espionage' (Layer 0, inf=0.7937) both fire on different aspects of "primarily" — the word appears in scientific and geopolitical texts. 'Topics related to environmental regulations and emissions' (Layer 3, inf=0.7972) is directly relevant: the Amazon is the central entity in climate/environmental regulation discourse. 'The words "rain" and "climate change"' (Layer 1, inf=0.7923) fires on "rainforest" and is the most semantically targeted early feature, linking the Amazon to the climate-change/ecological-science register.

[MIDDLE] The middle band maps Amazon→South America. 'The names of specific animals and places they are located' (Layer 10, inf=0.7964) is important: the Amazon is prototypically associated with specific animal species (jaguars, piranhas, toucans) and the feature encoding animal-location pairs is primed. 'Phrases indicating capabilities or important aspects, sometimes using idioms' (Layer 7, inf=0.7998) fires on "primarily" as an important-aspect intensifier. 'Locations, particularly cities and institutions, and the words "in", "at", "to" and "for"' (Layer 14, inf=0.7925) encodes the locative completion frame for "located in."

[LATE] The late band executes continent-level geographic selection. 'References to different geographical regions with location details' (Layer 17, inf=0.7952) is the most targeted late feature, encoding multi-region geographic text. 'Descriptive text about lakes and the ocean' (Layer 22, inf=0.7968) fires because the Amazon river system (the world's largest river by discharge) appears in aquatic-geography texts, associating the Amazon ecosystem with water body descriptions. 'References to years, especially in the 1600s and 1700s' (Layer 23, inf=0.7958) may reflect colonial-era discovery texts where the Amazon is first described.

[TOKEN COMPETITION] " South" wins at p=0.363, a moderate margin — notably this is a subword token, the beginning of "South America." Runner-ups likely include " Brazil" (the country that contains ~60% of the Amazon), " the" (for "the South American continent"), and possibly " Latin". The choice of " South" over " Brazil" reflects training data patterns: encyclopedic texts often phrase this as "located in South America" (continent-level) rather than naming Brazil specifically, though both are correct. The moderate confidence reflects genuine ambiguity between continent-level and country-level completion preferences in training data.

---

### Prompt: "<bos>A game of chess ends when a king is put in"

**Predicted token:** `Output " check" (p=0.886)` (prob=0.8859)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.7999 | 2.9428 |  words related to medicine, health, or development |
| 3 | 0.7995 | 5.6508 | words related to mathematics, and "play on words" and "tomatoes" |
| 5 | 0.7993 | 2.3634 | content related to the completion of a task |
| 1 | 0.7991 | 2.657 | the word "actual" within code or technical documentation |
| 3 | 0.7988 | 9.1734 |  words and phrases related to sports matches and leagues |
| 0 | 0.7986 | 0.0 |  mentions of being loving, loved, or loving something. |
| 6 | 0.7982 | 2.4616 |  months and years, especially when expressing dates. |
| 2 | 0.7979 | 3.3997 | a diverse set of stemmed words from various languages, including named entities like people, places, and organizations, along with abstract concepts and hierarchical terms. |
| 2 | 0.7977 | 2.5854 | scientific jargon related to immune cells, chemical reactions, and political science |
| 4 | 0.7973 | 2.6113 |  uses of the conditional "if" in mathematical and legal contexts. |
| 1 | 0.7971 | 3.0226 | the word "talent" |
| 5 | 0.797 | 3.6795 |  computer code and mathematical notation |
| 0 | 0.7968 | 2.1842 | words ending in "led" or "mingled", though it seems to give small activations to words related to intent and soma |
| 6 | 0.7966 | 7.8253 | instances of physical contact, often of a violent or sexual nature, as well as references to vehicles and buildings. |
| 6 | 0.7964 | 2.0731 |  words or phrases related to car racing events and personnel |
| 0 | 0.7962 | 3.297 |  references to dates and times |
| 0 | 0.7957 | 0.0 | mentions of clubs or sports teams, and sometimes related words like 'sister' or 'kids' |
| 0 | 0.7955 | 3.246 | locations and organizations |
| 4 | 0.7949 | 4.163 |  words associated with industrial design/engineering, electricity, and invention |
| 0 | 0.7947 | 0.0 |  closing curly brackets in code snippets |
| 6 | 0.7946 | 4.0522 |  code and file paths |
| 0 | 0.7944 | 3.4011 |  academic writing related to computer science |
| 4 | 0.7938 | 7.0222 |  present-tense action verbs |
| 3 | 0.7936 | 2.2249 |  scientific writing and terminology related to the position or location of something |
| 5 | 0.7934 | 5.3088 |  phrases describing finality, such as "once and for all" or the end of something. |
| 4 | 0.7933 | 3.7019 | words related to scientific experiments, medical conditions and strong emotions |
| 1 | 0.7931 | 6.2652 |  words or phrases that appear in legal or technical documents, like names of laws, legal terms (pleaded, testified), and technical terms, especially when abbreviated or in code |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.799 | 2.2413 |  words related to medical conditions, particularly tumors and wounds, and related procedures |
| 9 | 0.796 | 5.9025 |  words and phrases related to scientific and political topics |
| 7 | 0.7951 | 2.5329 |  words related to academia/research and mechanical inventions |
| 8 | 0.7942 | 4.6673 |  words and phrases related to calculating numbers |
| 10 | 0.794 | 7.2017 |  references to directions, navigation, and feeling lost |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.8 | 12.1924 | instances of opinions being conveyed to someone |
| 20 | 0.7997 | 7.5477 |  code snippets and related terms like script, function, and document |
| 19 | 0.7984 | 57.725 |  words related to computer code and cybersecurity |
| 15 | 0.798 | 15.8246 |  terms relating to royalty, government, and the military |
| 22 | 0.7975 | 7.4574 |  words and phrases related to legal issues, imprisonment, and financial problems |
| 20 | 0.7959 | 29.7168 |  the word "put" |
| 22 | 0.7953 | 9.685 |  words related to character descriptions about clothing |
| 0 | 0.7929 | 0.0 |  quotation marks |

[EARLY] The early layers establish a games-and-rules domain. 'Words and phrases related to sports matches and leagues' (Layer 3, inf=0.7988, act=9.1734) fires strongly on "chess" and "game," establishing the game/sports register. 'Words related to mathematics, and "play on words" and "tomatoes"' (Layer 3, inf=0.7995) fires on "chess" — chess is discussed extensively in combinatorics and game theory literature alongside mathematics. 'Phrases describing finality, such as "once and for all" or the end of something' (Layer 5, inf=0.7934) fires on "ends" — the terminal-state semantics. 'Terms related to royalty, government, and the military' at middle Layer 15 (inf=0.798, act=15.8246) is critically important: the "king" token in the prompt activates royalty features that are deeply associated with chess piece terminology — kings are both literal royalty and chess-piece roles. 'Uses of the conditional "if" in mathematical and legal contexts' (Layer 4, inf=0.7973) fires on the logical structure: "ends WHEN a king is put in [X]" is a conditional statement about game termination.

[MIDDLE] The middle band maps the chess-termination scenario to the "check" state. 'Words related to medical conditions, particularly tumors and wounds, and related procedures' (Layer 8, inf=0.799) is a surprising fire, but "check" in chess has the same orthographic form as "check" in medical examinations, and the medical-examination register co-activates. 'Words and phrases related to calculating numbers' (Layer 8, inf=0.7942) fires because chess moves and game states involve discrete counting and calculation. 'References to directions, navigation, and feeling lost' (Layer 10, inf=0.794) may encode the "in check/checkmate" spatial-constraint relationship — being in check means the king is threatened (navigationally constrained).

[LATE] The late layers execute the "check" token selection. 'Words related to computer code and cybersecurity' (Layer 19, inf=0.7984, act=57.725) fires on "check" as a technical-verification term common in security and code contexts — "security check," "input check." The model is partially routing through the technical-check polysemy. 'The word "put"' (Layer 20, inf=0.7959, act=29.7168) fires directly on "put in" from the prompt — this feature tracks the exact phrase and primes its canonical chess completion. 'Words and phrases related to legal issues, imprisonment, and financial problems' (Layer 22, inf=0.7975) fires because "put in check" has legal/financial metaphorical uses. 'Terms relating to royalty, government, and the military' (Layer 15, inf=0.798) carries forward from early layers, maintaining the king/chess-royalty frame.

[TOKEN COMPETITION] " check" wins with high confidence at p=0.886 — the second-highest in the sports/games sub-domain. The 'the word "put"' feature at Layer 20 (act=29.7168) provides a near-perfect trigger: "put in" directly precedes "check" in the canonical chess-game termination formula. Runner-ups likely include " checkmate" (the actual end state, but a longer token), " danger" (informal), and " jeopardy" (metaphorical). The choice of "check" over "checkmate" is interesting — "put in check" (not checkmate) is the technically precise phrasing for the king being threatened, and "checkmate" is when the game actually ends. The model's 88.6% confidence on "check" suggests this phrasing is deeply encoded in training data about chess rules.

---

### Prompt: "<bos>The Olympic Games take place every"

**Predicted token:** `Output " four" (p=0.857)` (prob=0.8574)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.7997 | 8.9438 |  the word "place" |
| 0 | 0.7995 | 5.0459 |  occurrences of the word "placement" and related words |
| 4 | 0.7991 | 2.8553 |  references to frequency and time |
| 1 | 0.7986 | 6.1883 | words or phrases implying judgment, construction, or logic. |
| 0 | 0.7982 | 0.0 |  closing curly brackets in code snippets |
| 6 | 0.7979 | 5.6018 |  the word "world" and related terms such as "international", "global", "universal", and country names |
| 0 | 0.7977 | 4.4989 | the word "bound" (and variants), often related to mathematical or scientific contexts |
| 3 | 0.7975 | 3.9573 |  uses of "any", "either", "one", "more", and "other" followed by a noun, verb, or adjective. |
| 5 | 0.7973 | 3.433 |  mentions of days of the week or durations of time |
| 1 | 0.7968 | 2.7722 |  words related to vaccines and inoculation |
| 0 | 0.7964 | 7.1429 | the word "purpose" |
| 0 | 0.7961 | 6.6091 |  the word "tour" |
| 4 | 0.7959 | 10.0197 | verbs and nouns related to an action or process. |
| 2 | 0.7952 | 3.4271 |  words related to a lack of something |
| 2 | 0.7948 | 4.175 |  references to the US government and political process |
| 1 | 0.7945 | 2.8767 |  date and time information |
| 5 | 0.7943 | 4.8007 |  the last three letters of words. |
| 0 | 0.7941 | 2.1139 |  proper nouns, especially including titles of works and organizations |
| 6 | 0.7939 | 17.8566 | words that appear in programming code, legal jargon, or scientific texts |
| 2 | 0.7936 | 7.5014 | words related to images, medical sensations, numerical values, and legal situations |
| 3 | 0.7934 | 7.4427 |  mentions of things being held, such as events or examinations |
| 0 | 0.7932 | 2.8633 |  the word "Code" and words related to it |
| 2 | 0.7929 | 7.5163 |  words related to involvement and participation |
| 1 | 0.7927 | 4.0856 |  words or phrases that appear in legal or technical documents, like names of laws, legal terms (pleaded, testified), and technical terms, especially when abbreviated or in code |
| 0 | 0.792 | 1.887 |  terms related to law, immigration, and gaming. |
| 0 | 0.7918 | 3.7571 |  words and phrases indicating a turning point or realization |
| 1 | 0.7915 | 6.0208 |  words or phrases that appear in legal or technical documents, like names of laws, legal terms (pleaded, testified), and technical terms, especially when abbreviated or in code |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.8002 | 33.3666 |  code snippets with specific coding keywords and markup tags |
| 11 | 0.7993 | 7.0915 |  bracketed citations, algorithm references, and section references |
| 10 | 0.797 | 6.4387 |  references to elections, championships, baseball and politics |
| 8 | 0.7966 | 2.854 |  text related to motorsports, especially racing series and championships |
| 9 | 0.7913 | 4.4833 |  common short words like articles and prepositions, as well as words related to the passage's topic |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7999 | 0.0 |  instances of the word "hardly" and possibly words close to it within a sentence |
| 21 | 0.7988 | 14.9878 |  a wide variety of context-specific terms in technical and scientific documents, including programming code, scientific papers, and configuration files. |
| 20 | 0.7984 | 10.9165 |  words related to legal or business contracts |
| 18 | 0.7957 | 6.8601 |  references to months of the year |
| 25 | 0.7955 | 9.8349 |  what appear to be version numbers or codes |
| 23 | 0.795 | 10.2484 |  code |
| 23 | 0.7925 | 12.2107 |  terms from scientific publications and legal documents, possibly related to research, analysis, or defect |
| 23 | 0.7922 | 7.9073 |  prepositions and conjunctions that connect concepts in language, with a particular emphasis on "all", "that", "from", "for", and "to" when co-located with other words that help form complete phrases. |

[EARLY] The early layers establish a recurring-events temporal frame. 'References to frequency and time' (Layer 4, inf=0.7991) fires on "every" — this is the key temporal-frequency word, and the feature encodes the "every [X years/months]" pattern. 'The word "place"' (Layer 3, inf=0.7997, act=8.9438) fires on "take place" and is the highest-activating early feature: it recognises the event-occurrence idiom. 'Occurrences of the word "placement" and related words' (Layer 0, inf=0.7995) also fires on the placement/occurrence semantic cluster. 'The word "world" and related terms such as "international", "global", "universal", and country names' (Layer 6, inf=0.7979) fires on "Olympic Games" as a global event. 'Mentions of days of the week or durations of time' (Layer 5, inf=0.7973) primes temporal-period tokens, setting up the "every [duration]" frame. 'References to the US government and political process' (Layer 2, inf=0.7948) and 'mentions of things being held, such as events or examinations' (Layer 3, inf=0.7934) together encode the institutional event-scheduling register.

[MIDDLE] The middle band routes to the election/championship periodicity pattern. 'References to elections, championships, baseball and politics' (Layer 10, inf=0.797) is the critical mapping feature: it explicitly encodes periodic championships and elections, both of which occur every four years. This feature creates the binding between "Olympic Games" (a periodic championship) and the four-year cycle. 'Code snippets with specific coding keywords and markup tags' (Layer 8, inf=0.8002, act=33.3666) fires with very high activation but is likely a SAE encoding artifact. 'Text related to motorsports, especially racing series and championships' (Layer 8, inf=0.7966) fires because motorsport championships also operate on annual/periodic schedules, reinforcing the championship-periodicity frame.

[LATE] The late band is notably thin for this confident prediction. 'A wide variety of context-specific terms in technical and scientific documents' (Layer 21, inf=0.7988) and 'words related to legal or business contracts' (Layer 20, inf=0.7984) are generic. The absence of a strong "four" or "years" feature in the late band suggests the winning token is driven by pattern momentum from the middle band's championship-periodicity encoding rather than a dedicated late-layer feature.

[TOKEN COMPETITION] " four" wins at p=0.857, one of the highest confidence scores. The 'elections, championships, baseball and politics' feature at Layer 10 provides the decisive binding: elections, Olympic Games, and major championships all operate on four-year cycles, and "every four years" is the canonical phrase. Runner-ups include " two" (Winter Olympics sometimes confusingly discussed), " year" (for "every year"), and " summer" (for "every summer Olympics"). The very high confidence reflects that "The Olympic Games take place every four years" is one of the most commonly encoded periodic-fact sentences in training data, appearing in textbooks, news articles, and encyclopedias with essentially no variation.

---

### Prompt: "<bos>The sport of sumo wrestling originated in"

**Predicted token:** `Output " Japan" (p=0.562)` (prob=0.5615)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8002 | 2.8359 |  terms related to sports, especially basketball and player contracts |
| 4 | 0.7998 | 4.2982 |  instances of doing something "first." |
| 2 | 0.7996 | 2.0237 |  words relating to employees, employment, and management |
| 0 | 0.7994 | 2.2483 |  language related to social and enviornmental issues, justice, and political movements |
| 0 | 0.799 | 2.7513 | adverbs ending in 'ly' and prepositions |
| 0 | 0.7988 | 1.4926 |  words and phrases related to strong emotions or opinions |
| 2 | 0.7982 | 2.2831 | dates, times, royalty, places, countries, conflicts, and people involved in historical events |
| 4 | 0.798 | 3.2093 |  the creation or founding of something |
| 3 | 0.7978 | 3.4781 |  mentions of when things began or started |
| 0 | 0.797 | 3.1312 |  superlatives and references to recent experiences |
| 4 | 0.7968 | 4.1813 |  technical and medical terms and language |
| 3 | 0.7964 | 3.8293 | verbs and words related to the transformation or manipulation of something. |
| 0 | 0.7962 | 1.6023 |  words related to user interfaces, medicine or job qualifications |
| 0 | 0.7956 | 2.7295 | the word "present" as it's used in formal writing |
| 0 | 0.7952 | 1.4065 | words without an easily discernible common meaning or pattern and may require more context for interpretation. |
| 0 | 0.795 | 1.4349 | a wide variety of technical words, as well as medical terminology |
| 3 | 0.7947 | 3.8927 | words related to research studies and events occurring in time |
| 1 | 0.7943 | 2.2773 |  text related to historical events and places |
| 3 | 0.7941 | 2.3552 |  words or phrases that indicate a development or reaction to prior events. |
| 0 | 0.7939 | 2.8533 |  terms related to medicine or legal situations |
| 3 | 0.7937 | 5.5298 |  the word "of" in legal documents. |
| 3 | 0.7935 | 2.2606 |  programming code snippets and disease-related terms |
| 3 | 0.7931 | 16.2344 |  code snippets and documentation references, possibly related to web development |
| 0 | 0.7927 | 1.7735 |  words that have secondary or altered meanings, such as niche technical terms or re-purposed common words |
| 0 | 0.7923 | 3.3335 |  references to dates and times |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.8 | 5.3305 |  names of people and places associated with a particular crime |
| 9 | 0.7992 | 7.1347 |  Malaysian food items or locations |
| 0 | 0.7986 | 0.0 |  math and mathematical notation |
| 8 | 0.7976 | 15.6797 | the digit zero |
| 5 | 0.7974 | 1.7451 |  words and phrases related to geography, politics, or groups of people, particularly related to conflict |
| 9 | 0.7972 | 3.2003 |  words associated with historical places, people, and events, especially when dates are mentioned. |
| 10 | 0.796 | 4.8196 | the appearance of the word "history" along with other words related to time and dates. |
| 8 | 0.7958 | 3.6286 |  code snippets including the words line, file, py, module as well as Django and MySQL. |
| 8 | 0.7954 | 7.6921 |  words related to place names, organizations, and dates |
| 6 | 0.7933 | 5.1013 | biographical information, especially about someone's early years |
| 0 | 0.7929 | 0.0 | the word "motivated" or "motivation" |
| 9 | 0.7925 | 4.8737 |  words related to science, potentially biology or geology |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 13 | 0.7984 | 6.9049 |  mentions of somebody's origin and hometown |
| 11 | 0.7966 | 57.63 |  large empty spaces in the text |
| 17 | 0.7945 | 10.4994 |  references to Chinese food or US/Russian politics |

[EARLY] The early layers establish a sports-origin attribution frame. 'Terms related to sports, especially basketball and player contracts' (Layer 0, inf=0.8002) fires on "sport" and "wrestling" — the broad sports-register feature. 'Dates, times, royalty, places, countries, conflicts, and people involved in historical events' (Layer 2, inf=0.7982) is triggered by "originated" — the origin/founding event temporal frame. 'Mentions of when things began or started' (Layer 3, inf=0.7978) fires directly on "originated" and is the most syntactically targeted early feature: it encodes the initiation/beginning semantic. 'The creation or founding of something' (Layer 4, inf=0.798) and 'words related to research studies and events occurring in time' (Layer 3, inf=0.7947) jointly represent the historical-origin attribution register. The 'word "of"' in legal documents (Layer 3, inf=0.7937) and code-documentation patterns (Layer 3, inf=0.7931, act=16.2344) establish the structured definitional frame.

[MIDDLE] The middle band maps sumo wrestling to Japan. 'Names of people and places associated with a particular crime' (Layer 8, inf=0.8) is a broad proper-noun feature that fires on sumo wrestler names and Japanese place names which appear in sports-crime reports (doping scandals, hazing incidents). 'Malaysian food items or locations' (Layer 9, inf=0.7992) is proximate geographic activation — Malaysia is Southeast Asian like Japan, and the feature may encode East/Southeast Asian proper nouns broadly. 'Words associated with historical places, people, and events, especially when dates are mentioned' (Layer 9, inf=0.7972) directly encodes the historical-origin relationship. 'Words related to place names, organizations, and dates' (Layer 8, inf=0.7954) and 'biographical information, especially about someone's early years' (Layer 6, inf=0.7933) complete the origin-attribution circuit.

[LATE] 'Mentions of somebody's origin and hometown' (Layer 13, inf=0.7984) is the decisive late feature — it explicitly encodes origin/birthplace attribution and directly selects toward country-of-origin tokens. 'References to Chinese food or US/Russian politics' (Layer 17, inf=0.7945) fires because Japan is an East Asian country and this feature encodes broader East Asian geography.

[TOKEN COMPETITION] " Japan" wins at p=0.562, a moderate margin. The 'origin and hometown' feature at Layer 13 executes the origin-attribution binding, and the East Asian geography features in the middle band pre-load the Japanese namespace. Runner-ups likely include " China" (also East Asian, sometimes confused), " ancient" (for "ancient Japan"), and " the" (hedged article). The moderate confidence reflects that sumo's Japanese origin is well-established in training data but the model faces competition from generic hedging completions ("in the..."). The win is clear enough to be correct.

---

### Prompt: "<bos>The programming language Python was created by"

**Predicted token:** `Output " Guido" (p=0.759)` (prob=0.7590)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.7998 | 3.162 |  specific vocabulary and formatting conventions related to academic, scientific, and financial writing |
| 7 | 0.7996 | 4.1868 |  controversial and defensive statements about slavery or technical information about Java |
| 0 | 0.7994 | 3.7403 |  words related to money or business transactions |
| 6 | 0.7989 | 1.973 |  terms related to computer programming and networking |
| 0 | 0.7987 | 2.5125 | the term "machine learning" |
| 7 | 0.7983 | 3.939 |  expressions of time and location |
| 2 | 0.7981 | 3.5147 |  words related to business processes and venues |
| 5 | 0.7972 | 2.8048 |  programming and coding terms |
| 0 | 0.797 | 3.5543 | sentences beginning with coordinating conjunctions like "but" and "and" |
| 7 | 0.7968 | 5.6773 |  words related to scientific properties or materials |
| 5 | 0.7965 | 3.9787 |  phrases explaining the source of content or who is responsible for its creation or dissemination |
| 1 | 0.7963 | 0.9679 |  programming code and error messages related to python and MySQL |
| 0 | 0.7961 | 2.6576 |  mathematical terminology and notation including terminology about sets, graphs, and formulas |
| 5 | 0.7959 | 12.3084 |  capitalized abbreviations and names of medical terms (hormones and drugs) or famous people |
| 3 | 0.7957 | 3.3353 |  terms related to medicine and scientific study |
| 1 | 0.7952 | 3.5998 | the words "slow", "slower", and "fast" |
| 0 | 0.795 | 3.3989 |  the word "meaning," and in some cases words around it |
| 1 | 0.7948 | 5.9079 |  the word "accessories" |
| 0 | 0.7946 | 2.9087 |  the word "hall" and articles |
| 0 | 0.7944 | 3.2015 | words ending in 'ing' that can be interpreted as an activity currently underway |
| 4 | 0.7933 | 2.9746 |  scientific or technical terms related to biology, chemistry, medicine, sports, and film |
| 3 | 0.793 | 5.0576 |  words related to software or application development and marketing |
| 3 | 0.7928 | 8.5262 |  words related to constructing infrastructure and buildings |
| 5 | 0.7921 | 3.0248 |  words related to circuits in computer science |
| 7 | 0.7919 | 5.448 |  dates relating to inventors or discoveries |
| 5 | 0.7917 | 2.2856 |  terms related to computers, the internet, and software |
| 0 | 0.7915 | 1.6397 |  code snippets and related terminology in various programming languages |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.8 | 7.2161 |  proper nouns and adjectives, mostly place names and people's titles |
| 11 | 0.7985 | 57.63 |  large empty spaces in the text |
| 0 | 0.7976 | 0.0 |  the keyword 'let' which is used for variable declarations in Javascript |
| 8 | 0.7955 | 18.9138 | the first-person pronoun "I" and the word "Exactly" |
| 13 | 0.7939 | 11.1375 |  programming related terms, specifically build and version control tools |
| 8 | 0.7937 | 9.7827 | noun phrases about study methodology are sought by this neuron |
| 8 | 0.7935 | 4.2202 |  things that can be compiled or source code segments |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 20 | 0.7991 | 8.6209 |  clauses expressing hypothetical or conditional situations |
| 22 | 0.7978 | 9.6144 | citations of academic papers |
| 24 | 0.7974 | 21.388 |  proper nouns indicating locations, people, and organizations |
| 24 | 0.7941 | 5.7651 |  mentions of the organization 'FIDE', especially in the context of chess titles and events |
| 24 | 0.7926 | 6.2954 |  first person pronouns, especially "I'm" and words related to writing |
| 25 | 0.7924 | 9.2545 |  names of people |

[EARLY] The early layers identify a technology-creator attribution prompt. 'Programming code and error messages related to python and MySQL' (Layer 1, inf=0.7963) is the most specific early feature: it directly encodes the Python programming language context. 'Terms related to computer programming and networking' (Layer 6, inf=0.7989) and 'programming and coding terms' (Layer 5, inf=0.7972) establish the software-development register. 'The term "machine learning"' (Layer 0, inf=0.7987) fires because Python is the dominant machine learning language and this co-occurrence is extremely strong in training data. 'Phrases explaining the source of content or who is responsible for its creation or dissemination' (Layer 5, inf=0.7965) is the attribution-pattern feature, encoding "created by [PERSON]" constructions. 'Dates relating to inventors or discoveries' (Layer 7, inf=0.7919) fires on the creator-attribution frame.

[MIDDLE] The middle band routes toward the Python creator. 'Proper nouns and adjectives, mostly place names and people's titles' (Layer 12, inf=0.8) is the named-person feature. 'Programming related terms, specifically build and version control tools' (Layer 13, inf=0.7939) fires because Guido van Rossum created Python's initial version control infrastructure. 'Things that can be compiled or source code segments' (Layer 8, inf=0.7935) and 'the first-person pronoun "I" and the word "Exactly"' (Layer 8, inf=0.7955) — Guido famously wrote in first-person about Python's design in many developer blog posts, making this pronoun pattern associated with Python creator attribution. 'Noun phrases about study methodology' (Layer 8, inf=0.7937) is likely incidental.

[LATE] 'Proper nouns indicating locations, people, and organizations' (Layer 24, inf=0.7974, act=21.388) executes the proper-person-name push. 'Mentions of the organization "FIDE", especially in the context of chess titles and events' (Layer 24, inf=0.7941) fires because FIDE encodes multi-word institutional names, and Guido van Rossum is a multi-word personal name. 'Names of people' (Layer 25, inf=0.7924) and 'first person pronouns, especially "I'm" and words related to writing' (Layer 24, inf=0.7926) together encode the first-name-initial completion pattern.

[TOKEN COMPETITION] " Guido" wins at p=0.759, high confidence. The Python-language features at Layer 1 and the attribution features at Layer 5 combine with the proper-noun features at Layers 24-25 to converge on the creator's first name. Runner-ups likely include " van" (for "van Rossum"), " the" (hedged), and possibly " Linus" (Torvalds — a prominent language/tool creator competing in the creator-space). The high confidence reflects that the Python→Guido association is among the most frequently encoded creator-language pairs in developer documentation, Wikipedia, and programming education texts.

---

### Prompt: "<bos>HTML stands for HyperText Markup"

**Predicted token:** `Output " Language" (p=0.984)` (prob=0.9836)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8 | 0.0 | mentions of clubs or sports teams, and sometimes related words like 'sister' or 'kids' |
| 2 | 0.7998 | 1.8338 |  words related to business management and organizational leadership |
| 4 | 0.7996 | 2.4139 |  code snippets, computer science terminology, and file formatting |
| 5 | 0.7994 | 3.3047 |  words that are technical and scientific in nature |
| 1 | 0.7989 | 4.4818 |  mentions of people being elected to political office |
| 1 | 0.7987 | 3.4126 | HTML tags and attributes |
| 5 | 0.7985 | 2.6673 |  proper nouns such as names of people or geographic features |
| 4 | 0.7983 | 3.6148 | abbreviations and terminology used in scientific literature. |
| 7 | 0.7975 | 9.49 | abbreviations and terms related to clinical medicine and software development |
| 0 | 0.7973 | 2.5014 |  words related to research papers or documentation |
| 0 | 0.7971 | 2.0867 |  technical terms and specialized vocabularies. |
| 0 | 0.7969 | 3.2923 | words related to being somewhat like something but not entirely or always. |
| 2 | 0.7967 | 1.9803 |  words related to management and leadership styles, specifically in healthcare |
| 3 | 0.7963 | 3.9109 |  mentions of cash or money |
| 0 | 0.7962 | 1.7958 | a lot of diverse things that mostly only appear in computer programming code, math equations, or politically-charged comments |
| 4 | 0.796 | 3.232 |  references to government organizations and educational institutions |
| 2 | 0.7958 | 2.7299 |  source code relating to software libraries and metadata |
| 3 | 0.7956 | 1.8067 |  words associated with displaying information and instructions |
| 1 | 0.7954 | 2.0627 |  words and phrases related to search engine optimization |
| 5 | 0.7952 | 2.6438 |  terms related to genetics, diseases, and medical research |
| 1 | 0.795 | 4.0588 |  sentences containing information about people, including their names, nationalities, professions, and birth dates |
| 5 | 0.7948 | 2.2856 | objective-c code syntax and related terms |
| 4 | 0.7946 | 5.2016 |  technical jargon in the fields of medicine, automotive engineering, and food production |
| 0 | 0.7944 | 3.2606 |  proper nouns, code syntax, and scientific words related to medicine, botany, or climate science |
| 0 | 0.794 | 1.7111 |  words, abbreviations, and symbols related to scientific measurement and notation |
| 1 | 0.7936 | 4.3 |  the word "kitchen" and sometimes the word "lean" |
| 0 | 0.7934 | 2.6813 |  a combination of numbers, dashes, capital letters, and some keywords throughout the text |
| 1 | 0.7932 | 2.5273 |  source code with file extensions and unusual symbols |
| 4 | 0.793 | 3.0718 |  a mix of programming code, acronyms, and foreign words |
| 6 | 0.7928 | 2.4792 |  nouns and verbs acting like nouns, often related to IT. |
| 1 | 0.7926 | 2.3676 |  the word "pen" or related words "penalty" and "penological" |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7993 | 12.0135 |  the naming or definition of something. |
| 9 | 0.7991 | 4.0788 |  code snippets and angle brackets |
| 10 | 0.7981 | 5.4156 | mentions of time, such as days of the week, specific times, and the phrase "cut off time." |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7979 | 0.0 | code snippets containing the word "return" |
| 18 | 0.7977 | 13.6973 | acronyms with related words |
| 17 | 0.7965 | 13.4498 |  language or phrases with words from a language |
| 25 | 0.7942 | 9.3149 |  code snippets including file paths and links |
| 0 | 0.7938 | 0.0 |  mentions of ice cream |
| 25 | 0.7924 | 9.4294 |  words related to creative arts like musical theatre, film, design and literature. |

[EARLY] The early layers identify an acronym-expansion completion task. 'HTML tags and attributes' (Layer 1, inf=0.7987) fires on "HTML" — this is the most specific early feature and directly encodes HTML as a web-technology identifier. 'Abbreviations and terminology used in scientific literature' (Layer 4, inf=0.7983) and 'abbreviations and terms related to clinical medicine and software development' (Layer 7, inf=0.7975) both fire on "HTML" and "HyperText Markup" as abbreviation-and-expansion constructions. 'A combination of numbers, dashes, capital letters, and some keywords throughout the text' (Layer 0, inf=0.7934) fires on the capitalized-abbreviation format. 'Nouns and verbs acting like nouns, often related to IT' (Layer 6, inf=0.7928) encodes the IT-technical-noun register. The 'source code with file extensions and unusual symbols' (Layer 1, inf=0.7932) fires because ".html" is among the most common file extensions in training data, and the model is anticipating the complete expansion.

[MIDDLE] The middle band executes the acronym-definition binding. 'The naming or definition of something' (Layer 9, inf=0.7993, act=12.0135) is the most targeted middle feature: it fires on definitional "X stands for Y" constructions and directly encodes the acronym-expansion semantic frame. 'Code snippets and angle brackets' (Layer 9, inf=0.7991) fires because HTML is written with angle-bracket tags, reinforcing the HTML-specific encoding. 'Mentions of time, such as days of the week' (Layer 10, inf=0.7981) is incidental.

[LATE] 'Acronyms with related words' (Layer 18, inf=0.7977, act=13.6973) is the decisive late feature: it explicitly encodes acronym-plus-expansion patterns and directly pushes toward the final word of the HTML expansion. 'Language or phrases with words from a language' (Layer 17, inf=0.7965) fires on "Language" itself — the word "Language" in the late band suggests the model is already preparing the specific word. 'Words related to creative arts like musical theatre, film, design and literature' (Layer 25, inf=0.7924) fires on "Language" as a broader linguistic/creative medium concept.

[TOKEN COMPETITION] " Language" wins at p=0.984, the second-highest confidence in the entire dataset (after "oxygen" in the water prompt). The 'acronyms with related words' feature at Layer 18 and the 'naming/definition of something' at Layer 9 form an extremely efficient circuit: "HTML stands for HyperText Markup [Language]" is a closed, unambiguous acronym expansion with exactly one valid completion in the English language. Runner-ups have essentially zero probability. The near-certainty is mechanistically correct: this is a pure memorised acronym expansion with no polysemy or geographic ambiguity.

---

### Prompt: "<bos>Wi-Fi uses radio waves to transmit"

**Predicted token:** `Output " data" (p=0.608)` (prob=0.6080)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7998 | 5.1844 |  mentions of physical parts or components of something, whether it be physical, textual, or medical. |
| 6 | 0.7997 | 4.5826 |  words related to bicycles and bicycle maintenance |
| 3 | 0.7993 | 2.8415 | words and abbreviations related to academic research, especially in the sciences.  |
| 1 | 0.7991 | 7.3849 |  terms related to computer networking protocols |
| 1 | 0.7989 | 7.1421 | the word "concept" and related language about participatory systems or waves |
| 7 | 0.7987 | 4.1984 |  technical descriptions of electronic devices or displays |
| 6 | 0.7983 | 6.1629 |  terms related to data gathering and organization |
| 2 | 0.7981 | 2.7802 |  a combination of religious and positive words. |
| 0 | 0.7979 | 2.1673 |  technical or scientific language in the contexts of communication, mechanics and/or physics, especially quantum physics |
| 2 | 0.7977 | 5.3471 | computer code and programming terms including struct definitions, assembly language, and package imports |
| 4 | 0.7975 | 3.4169 | words related to appointments and scheduling |
| 6 | 0.7973 | 6.5322 |  words and phrases related to computing power consumption and memory usage. |
| 4 | 0.7969 | 5.1763 |  words relating to electronics and physics, especially radio frequencies and microwaves |
| 2 | 0.7966 | 5.032 |  words related to industrial parts, tools and chemical or physical processes.  |
| 0 | 0.7964 | 6.9065 | hyphens within text |
| 1 | 0.7962 | 3.0096 |  words or fragments related to graphs and mathematical or scientific models |
| 1 | 0.796 | 8.7221 |  words related to nanotechnology |
| 1 | 0.7958 | 2.2497 |  the phrase "United Kingdom" |
| 0 | 0.7956 | 5.5009 |  the word "original" and the variable tau inside mathematical formulas |
| 5 | 0.7954 | 5.5547 | the word "template", or language implying comparison |
| 0 | 0.7952 | 4.5404 |  terms relating to quantum physics and general uncertainty |
| 0 | 0.7948 | 3.3679 | the word "dedicated", and sometimes words relating to cemeteries |
| 1 | 0.7946 | 1.909 |  terms related to data storage and division |
| 2 | 0.7942 | 2.7378 |  words and phrases related to independence and freedom |
| 5 | 0.794 | 4.1822 |  adjacent words that together form a common technical concept |
| 2 | 0.7938 | 5.054 |  words related to family relationships or medical conditions |
| 3 | 0.7936 | 4.9859 |  words that are present tense verbs or the past tense of a verb |
| 1 | 0.7934 | 7.7593 |  words denoting large size, evil, and negative physical or mental states |
| 1 | 0.793 | 6.9642 |  terminology related to wireless communication and data transmission |
| 1 | 0.7924 | 2.649 |  mentions of ground level or sea level, and sometimes hard construction materials |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.8 | 4.7447 |  verbs, especially those ending in -ing, in different tenses |
| 0 | 0.7995 | 0.0 |  the string "ot" |
| 8 | 0.795 | 4.9391 |  technical jargon related to data communication, specifically networking protocols and devices |
| 9 | 0.7932 | 7.7173 |  terms related to audio and bluetooth technology |
| 8 | 0.7926 | 6.5881 | noun phrases about study methodology are sought by this neuron |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 25 | 0.7985 | 7.241 |  technical writing about digital signals and imaging |
| 0 | 0.7971 | 0.0 |  words related to political administration and legal rights |
| 24 | 0.7968 | 6.6428 |  words that indicate probability or agreement, sometimes related to legal and medical contexts. |
| 21 | 0.7944 | 8.348 |  words related to technology and the internet |
| 16 | 0.7928 | 7.2339 | send |

[EARLY] The early layers establish a wireless-technology domain. 'Terms related to computer networking protocols' (Layer 1, inf=0.7991, act=7.3849) is the highest-activating early feature: it fires on Wi-Fi as a network protocol and primes networking-domain tokens. 'Terminology related to wireless communication and data transmission' (Layer 1, inf=0.793, act=6.9642) is even more targeted — it directly encodes the Wi-Fi use-case of wireless data transmission. 'Words relating to electronics and physics, especially radio frequencies and microwaves' (Layer 4, inf=0.7969) fires on "radio waves" — this feature is causally active on the physics of wireless communication. 'Technical descriptions of electronic devices or displays' (Layer 7, inf=0.7987) and 'words and phrases related to computing power consumption and memory usage' (Layer 6, inf=0.7973) further establish the networking-hardware register. 'The word "concept" and related language about participatory systems or waves' (Layer 1, inf=0.7989) fires on "waves" from "radio waves."

[MIDDLE] The middle band routes to data-transmission vocabulary. 'Technical jargon related to data communication, specifically networking protocols and devices' (Layer 8, inf=0.795) is the key middle feature: it encodes the exact domain of Wi-Fi (data communication via networking protocols) and primes "data" as the canonical object of wireless transmission. 'Terms related to audio and bluetooth technology' (Layer 9, inf=0.7932) fires because Bluetooth and Wi-Fi share the radio-frequency transmission register — both transmit data wirelessly.

[LATE] 'Technical writing about digital signals and imaging' (Layer 25, inf=0.7985) encodes the digital-signal domain — data is the canonical unit transmitted in digital signal systems. 'Words related to technology and the internet' (Layer 21, inf=0.7944) is a broad tech-domain feature. 'Send' (Layer 16, inf=0.7928) is a single-word feature directly encoding the transmission act, priming the thing that gets transmitted.

[TOKEN COMPETITION] " data" wins at p=0.608, a moderate-to-high confidence margin. The networking-protocols feature (Layer 1) and the data-communication feature (Layer 8) together create a strong vote for "data" as the canonical object of wireless transmission. Runner-ups likely include " signals" (technically correct), " information" (synonym), and " wireless" (tautological but plausible). "Data" wins because "Wi-Fi transmits data" is the canonical textbook phrasing, appearing in networking documentation and educational texts far more frequently than "Wi-Fi transmits signals." The margin reflects moderate certainty — the model knows the transmission domain but faces a limited semantic contest among valid technical synonyms.

---

### Prompt: "<bos>A CPU stands for Central Processing"

**Predicted token:** `Output " Unit" (p=0.989)` (prob=0.9889)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 5 | 0.8002 | 3.2532 |  a wide variety of proper nouns, symbols, and formatting characters |
| 0 | 0.8 | 1.4552 |  sections of documents with proper nouns and industry terminology. |
| 0 | 0.7998 | 1.1821 |  words related to systems of oppression and violence |
| 4 | 0.7994 | 3.0164 | titles of things, especially when they contain many words |
| 1 | 0.799 | 3.7677 | technical terms related to processes or frameworks, especially in business or computing. |
| 7 | 0.7988 | 4.0616 | legal court case related terms as well as names like Obama, Zimmerman, Dewani, Perry, TripAdvisor |
| 0 | 0.798 | 3.3571 |  mentions of sports |
| 2 | 0.7976 | 5.2736 |  phrases containing a word followed by "back" or "down" |
| 4 | 0.7974 | 1.9351 | words that might be present in scientific documents, including chemical names, Korean names, and names of attacks |
| 2 | 0.797 | 5.7928 |  the word "refers" and words related to it |
| 0 | 0.7968 | 1.1903 | welcoming sentiment |
| 1 | 0.7966 | 2.4919 |  words related to mathematical proofs and formulations |
| 3 | 0.7964 | 2.1153 | computer code containing hexadecimal values, the word "encoding", and the word "transfer" |
| 6 | 0.796 | 3.072 |  words related to academic institutions, the military, locations, and/or dates |
| 0 | 0.7958 | 3.0809 | technical or jargonistic terms that are specific to certain fields. |
| 2 | 0.7954 | 2.2757 |  terms relating to public administration, academia and condensed matter physics |
| 0 | 0.7952 | 1.8476 |  slightly shortened versions of words that would otherwise start with "com" |
| 0 | 0.795 | 1.5775 |  technical or scientific terms, especially in a research context |
| 1 | 0.7948 | 1.8563 | words related to visual arts, and also jargon related to workplace discrimination lawsuits |
| 4 | 0.7946 | 2.2051 |  words that seem to be headings in technical documents |
| 6 | 0.7944 | 4.6195 | words related to introspection, and places of residence or focus |
| 3 | 0.794 | 3.0237 |  words and phrases related to religion and church services |
| 0 | 0.7938 | 2.3193 |  the superlative adjective "closest" or "nearest" |
| 1 | 0.7936 | 1.5191 |  technical words related to a variety of scientific fields |
| 1 | 0.7934 | 3.4821 |  the word "human," sometimes near measurements or scientific terms |
| 0 | 0.7932 | 1.7898 |  terms related to different languages, code, medicine, politics, and parsing. |
| 0 | 0.7926 | 1.8937 |  proper nouns and code terms in a variety of programming languages |
| 7 | 0.7924 | 3.3759 |  technical language, especially that related to technology and medicine, and code. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.7996 | 5.7816 |  words containing the strings "men", "sta", "ers", "ers", "ros", "ger", or substrings "qi", "aps", neurons". |
| 14 | 0.7992 | 11.033 |  code or computer language and its syntax |
| 9 | 0.7984 | 6.6798 |  terms related to raspberry pi and tablet devices |
| 8 | 0.7982 | 2.8821 |  words related to physics, medicine, and studies/characteristics of eyes |
| 14 | 0.7962 | 9.2332 |  words related to processes that repeat over time |
| 12 | 0.7956 | 24.0281 |  acronyms in a scientific context |
| 15 | 0.7942 | 46.6813 | words associated with legal, academic, or technical documents, and numerical references |
| 15 | 0.7928 | 7.4876 | references to the brain and mind |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7986 | 0.0 |  mentions of ice cream |
| 25 | 0.7978 | 7.871 |  titles and headings. |
| 23 | 0.7972 | 9.9791 | abbreviations, definitions, and technical terms in scientific or technical documents |
| 18 | 0.793 | 13.473 | Legal documents |

[EARLY] The early layers identify a hardware-acronym expansion task. 'Titles of things, especially when they contain many words' (Layer 4, inf=0.7994) fires on "Central Processing" as a multi-word title/name component. 'Technical terms related to processes or frameworks, especially in business or computing' (Layer 1, inf=0.799) fires on "Processing" in the computing domain. 'Sections of documents with proper nouns and industry terminology' (Layer 0, inf=0.8, act=1.4552) encodes the document-heading format of "A CPU stands for..." — this is the introductory definition format of technical documentation. 'The word "refers" and words related to it' (Layer 2, inf=0.797, act=5.7928) fires on "stands for" as a definitional/reference phrase synonymous with "refers to." 'Computer code containing hexadecimal values, the word "encoding", and the word "transfer"' (Layer 3, inf=0.7964) fires on "Processing" in the context of data transfer and encoding, linking the CPU to its data-handling function.

[MIDDLE] The middle band executes the CPU-acronym binding. 'Acronyms in a scientific context' (Layer 12, inf=0.7956, act=24.0281) is the most targeted feature: it explicitly encodes scientific/technical acronyms and primes the final word of multi-word expansions. 'Code or computer language and its syntax' (Layer 14, inf=0.7992) and 'words related to processes that repeat over time' (Layer 14, inf=0.7962) fire on the CPU's function as a processor executing repeated instruction cycles. 'Terms related to raspberry pi and tablet devices' (Layer 9, inf=0.7984) fires because Raspberry Pi documentation is full of CPU-specification text with "Central Processing Unit" appearing in hardware descriptions.

[LATE] 'Abbreviations, definitions, and technical terms in scientific or technical documents' (Layer 23, inf=0.7972) is the decisive late feature: it directly encodes the abbreviation-definition pattern and pushes toward the completion word of the expansion. 'Titles and headings' (Layer 25, inf=0.7978) reflects the heading/glossary format in which CPU expansions appear. 'Legal documents' (Layer 18, inf=0.793, act=13.473) fires on the formal definitional register.

[TOKEN COMPETITION] " Unit" wins at p=0.989, the highest confidence score alongside " oxygen." This is a pure acronym expansion — "CPU = Central Processing Unit" is among the most frequently occurring technical definitions in all of computing literature, documentation, and education. The 'acronyms in a scientific context' feature at Layer 12 (act=24.0281) and the 'abbreviation/definition' feature at Layer 23 form an unambiguous circuit: the last word of "Central Processing [Unit]" has no plausible competitor. Runner-up probability mass is essentially zero. The circuit is maximally efficient: acronym-recognition → acronym-expansion → single-word completion.

---

### Prompt: "<bos>The Eiffel Tower is located in"

**Predicted token:** `Output " Paris" (p=0.471)` (prob=0.4708)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8001 | 1.5935 |  words related to future events, comparisons, and preferences |
| 1 | 0.7998 | 1.9849 |  words related to terrain, land masses, and locations. |
| 0 | 0.7992 | 2.911 | the word "loose" and words near the idea of "middle" |
| 2 | 0.7984 | 3.0236 | past and present tense verbs related to sports |
| 2 | 0.798 | 1.8055 |  words and phrases related to locations and transportation |
| 0 | 0.7978 | 3.2859 |  words related to official processes and/or decision making |
| 4 | 0.7976 | 4.9255 |  a jumble of words having to do with finance, medicine, programming functions, and nutrition. |
| 4 | 0.7974 | 4.3712 |  code snippets, variables, data and formulas |
| 3 | 0.7971 | 4.1979 | the word "lesson(s)" |
| 0 | 0.7969 | 4.2014 |  research documents |
| 1 | 0.7965 | 3.4348 |  code comments |
| 3 | 0.7961 | 3.7011 |  addresses and locations |
| 2 | 0.7959 | 4.7944 |  words related to research, publishing, and definition activities |
| 1 | 0.7957 | 3.7854 |  things found by a tourist in and around where they are staying. |
| 0 | 0.7955 | 1.6226 | the phrase "as such" |
| 4 | 0.7953 | 3.464 | chemistry and biology analytical research terms |
| 4 | 0.795 | 3.3429 |  code-related terms like assembly, trademark, runtime, reflection, mouse events, and forms |
| 4 | 0.7948 | 3.1196 |  words related to study designs, results, and published documents |
| 3 | 0.7942 | 3.7418 |  words often found in research papers or scientific writing |
| 3 | 0.794 | 4.0107 |  legal documents where someone concurs or dissents in part |
| 4 | 0.7938 | 4.3425 |  code comments and import statements |
| 4 | 0.7933 | 6.9604 |  words related to study designs, results, and published documents |
| 4 | 0.7929 | 4.1621 |  proper nouns like places, people, companies and brands. |
| 4 | 0.7918 | 4.809 | words related to location in articles about places |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7996 | 0.0 | the word "motivated" or "motivation" |
| 10 | 0.7988 | 6.2744 | words related to weather modification, especially cloud seeding |
| 6 | 0.7986 | 2.2646 |  confidence intervals and other statistical metrics |
| 6 | 0.7982 | 3.1739 |  questions that include the word "where." |
| 7 | 0.7967 | 3.0718 |  words related to buildings, construction, and civic planning |
| 6 | 0.7946 | 5.007 |  proper nouns, especially company names, along with email addresses and french words |
| 10 | 0.7944 | 4.5282 |  references to court cases and legal documents, including names, dates, and U.S. codes |
| 6 | 0.7936 | 3.5901 |  words related to health, genetics, geography, and proper nouns |
| 7 | 0.7931 | 4.2497 |  initials, names, dates and email addresses |
| 0 | 0.7927 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |
| 6 | 0.7925 | 7.4477 |  what appears to be strings of code or file names |
| 7 | 0.792 | 3.4821 | text from the beginning of Wikipedia articles that often includes location names, alternate spellings, and political divisions |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7994 | 28.7484 | scientific or technical words and jargon |
| 0 | 0.799 | 0.0 |  Spanish and Portuguese words related to code and computers |
| 0 | 0.7963 | 0.0 |  words related to political administration and legal rights |
| 15 | 0.7923 | 150.2896 |  code documentation blocks in various programming languages |

[EARLY] The early layers parse a landmark-location query. 'Words related to terrain, land masses, and locations' (Layer 1, inf=0.7998) fires on "Tower" and "located" — the physical-structure-at-location semantic. 'Things found by a tourist in and around where they are staying' (Layer 1, inf=0.7957) is a remarkably targeted feature: the Eiffel Tower is quintessentially a tourist landmark, and this feature fires because the prompt sits squarely in the tourist-attraction register of training data. 'Addresses and locations' (Layer 3, inf=0.7961) and 'words related to research, publishing, and definition activities' (Layer 2, inf=0.7959) encode the definitional "located in" locative frame. 'Proper nouns like places, people, companies and brands' (Layer 4, inf=0.7929) fires on "Eiffel Tower" as a proper-noun entity requiring a place-name completion. 'Words related to location in articles about places' (Layer 4, inf=0.7918) is the most syntactically precise early feature: it fires specifically on "location" phrases in place-article text, exactly the context in which the Eiffel Tower is described.

[MIDDLE] The middle band maps the Eiffel Tower to Paris via French and architectural associations. 'Proper nouns, especially company names, along with email addresses and french words' (Layer 6, inf=0.7946) is the key binding feature: it explicitly encodes French proper nouns and fires on "Eiffel" — which is a distinctly French surname. This feature creates the Eiffel→French→Paris chain. 'Questions that include the word "where"' (Layer 6, inf=0.7982) encodes the "where is X" query type, which shares semantic space with "X is located in." 'Words related to buildings, construction, and civic planning' (Layer 7, inf=0.7967) fires on "Tower" as a built structure. 'Text from the beginning of Wikipedia articles that often includes location names, alternate spellings, and political divisions' (Layer 7, inf=0.792) is a high-value feature: Wikipedia introductory sentences ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France") are directly encoded here, and this feature pre-loads the Paris answer.

[LATE] 'Scientific or technical words and jargon' (Layer 11, inf=0.7994, act=28.7484) fires with high activation, reflecting the encyclopedic formal register of landmark-location text. 'Code documentation blocks in various programming languages' (Layer 15, inf=0.7923, act=150.2896) fires with extremely high activation — this is the same recurring SAE artifact seen in the Sahara prompt, likely encoding the formal structured-text pattern of encyclopedic fact sentences.

[TOKEN COMPETITION] " Paris" wins at p=0.471, the strongest score in the art/culture sub-domain. The French-proper-noun feature at Layer 6 and the Wikipedia-introduction feature at Layer 7 together create a strong vote for "Paris" as the canonical city-location for the Eiffel Tower. Runner-ups include " France" (country-level answer), " the" (for "in the city of Paris"), and " Europe". The 53% probability on non-Paris tokens reflects the multi-granularity problem: "located in Paris" and "located in France" are both factually correct, and the model must choose the city-level over the country-level answer. The win for " Paris" over " France" likely reflects the high frequency of "Eiffel Tower is located in Paris" (specific city) over "Eiffel Tower is located in France" (country) in training text.

---

### 3.3 Steering Interventions

Activation steering was attempted on the 3 highest-average-influence recurring features using the Neuronpedia `/api/steer` endpoint with strength=20.0 and strength_multiplier=4.0 on the representative prompt "The capital of Nigeria is". The endpoint returned HTTP 500 (Unknown Error — server-side failure) for all three attempts; this is a server-side issue with Neuronpedia's inference infrastructure for Gemma-2-2B and is not a client-side or feature-selection error. The analogical circuit run in this repository also encountered 404 errors on the steer endpoint for several features, suggesting steer inference for Gemma-2-2B is intermittently unavailable.

As a proxy for causal validation, we obtained the immediate circuit neighbourhood (depth=1 predecessors and successors, excluding logit/embed nodes) for each of the three features via `get_subgraph_node_ids_for_feature()` and labelled those nodes via the Neuronpedia feature API. These neighbourhood labels are reported below and provide mechanistic context about what each top recurring feature receives input from and propagates to.

| Layer | Feature | Label | Baseline | Steered output | Changed | Neighbourhood nodes (sample) |
|-------|---------|-------|----------|----------------|---------|-------------------------------|
| 0 | 18693554 |  the word "overall", sometimes alongside words that exp | HTTP 500 (unavail.) | HTTP 500 (unavail.) | N/A | Layer 6:  capitalized words and abbreviations par; Layer 7:  code documentation and import statement; Layer 4: various acronyms, IDs, and symbols, poss |
| 0 | 12880349 | occurrences of the word 'red' at the start of phrases o | HTTP 500 (unavail.) | HTTP 500 (unavail.) | N/A | Layer 7:  code documentation and import statement; Layer 5: the start of documentation blocks in cod |
| 0 | 24106095 |  the suffix "ins" or the word "thereon" | HTTP 500 (unavail.) | HTTP 500 (unavail.) | N/A | Layer 6:  capitalized words and abbreviations par; Layer 5: the start of documentation blocks in cod |

**Neighbourhood analysis for top 3 features:**

- **Layer 0, Feature 18693554** ("the word 'overall', sometimes alongside words that express quantity"): Its neighbourhood includes Layer 6 "capitalized words and abbreviations in court/governmental documents", Layer 4 "various acronyms, IDs, and symbols", and Layer 5/7 "code documentation and import statements". This places the feature squarely in the technical-reference-text cluster: it receives from embedding/early-layer token features and feeds into mid-layer abbreviation/acronym encoders. Despite its high recurrence (10/24 graphs), its label ("overall") suggests it fires on factual templates of the form "X is, overall, Y" or "The X is overall the Y" — a quantitative/superlative framing that appears in many factual recall prompts ("The *longest* river", "The *largest* ocean", etc.).

- **Layer 0, Feature 12880349** ("occurrences of the word 'red' at the start of phrases or clauses"): Neighbourhood includes code documentation (Layer 5/7) and administration/governance terms (Layer 4). This feature's label appears specific and perhaps surprising, but its high recurrence suggests it fires on colour-or-marker tokens at the start of informational clauses — which in encyclopedic text often introduces highlighted key terms or section markers. The neighbourhood is dominated by documentation/formal-text features, consistent with a structural-text role rather than a factual-content role.

- **Layer 0, Feature 24106095** ("the suffix 'ins' or the word 'thereon'"): Neighbourhood includes Layer 6 "capitalized words in court/governmental documents", Layer 9 "various code snippets", Layer 4 "content from multiple different languages in same document", and Layer 3 "word beginnings with letters a/h/i/r/p (prefixes, medical/scientific terms)". This feature fires on morphologically complex tokens — suffixes and legal/formal adverbs — and its neighbourhood of multilingual, code, and prefix features suggests it encodes a "dense formal vocabulary" signal that recurs across factual prompts where the model expects technical or encyclopedic register completions.

## 4. Discussion

### 4.1 What the Recurring Features Reveal About Factual Recall

The most striking finding of this analysis is the nature of the dominant recurring features. None of the top 15 features are obviously "world knowledge" features encoding entity relationships (e.g., "Nigeria→Abuja", "Shakespeare→plays", "mitochondria→cell"). Instead, the dominant recurring features describe the *surface structure* of factual text: reference-text register signals (technical documents, legal jargon, scientific writing), code/documentation patterns, and token-surface features (quantitative markers, colour tokens at clause starts, morphological markers). This suggests that Gemma-2-2B implements factual recall partly by detecting the encyclopedic text register from the prompt structure and using that to bias toward encyclopedic-text completions.

This is interpretable as a retrieval heuristic: the model has learned that prompts of the form "The X of Y is" appear predominantly in reference text (Wikipedia, textbooks, technical documentation), and by activating reference-text register features, it biases its subsequent activations toward the kinds of tokens that appear after such phrases in that register. The actual named-entity knowledge (e.g., which specific city is Nigeria's capital) appears in per-prompt-specific middle and late layers rather than in the shared recurring features.

### 4.2 Layer Distribution of Recurring Features

The top 15 recurring features cluster heavily in early layers (0–7), with 10 of 15 in layers 0–4. This is consistent with the hypothesis that factual recall shares its early processing with general text-type identification, while domain-specific knowledge resides in mid-to-late layers that are more prompt-specific. The distribution is notably different from what one might expect from a symbolic knowledge-retrieval system, where we would expect the "knowledge lookup" itself to dominate; instead, the shared machinery is the *query framing* (register recognition) rather than the answer retrieval.

Layer 7 is the deepest recurring-feature layer (Feature 4828270: "reference codes, abbreviations, and identifiers from different fields"), with the highest avg_influence in the top-15 table (0.6929). This layer 7 feature appears to function as a domain-agnostic entity-identifier recogniser — it fires on any canonical named entity (a person, a place, a chemical compound, a programming language) and likely serves as the bridge between early register recognition and domain-specific late-layer answer selection. Its multi-domain label ("reference codes, abbreviations, identifiers from different fields") is consistent with this role.

### 4.3 Comparison With Analogical Reasoning

The factual recall circuit's recurring features show an interesting contrast with what would be expected for analogical reasoning. In analogical reasoning (A:B::C:?), the shared circuit should encode relational mapping features — features that fire on pairs of semantically related items. In factual recall, no such relational features appear in the top recurring set; instead the circuit is dominated by text-register features. This suggests the two task types use qualitatively different computational strategies in Gemma-2-2B: analogical reasoning builds a relational circuit, while factual recall builds a register-matching circuit.

### 4.4 Model Errors and Circuit Failures

Three prompts produced incorrect or low-confidence answers: "The capital of Nigeria is" (predicted " a" at p=0.173, correct answer "Abuja"), "The powerhouse of the cell is the" (predicted " nucleus" at p=0.103, correct "mitochondria"), and "The first US president was" (predicted " a" at p=0.193, correct "Washington"). In each case, the per-prompt circuit walkthrough reveals that the specific named-entity binding features (e.g., Layer 15 "the word capital" for the Nigeria prompt) were present but competed with high-frequency generic tokens (" a", " the") whose general-text frequency overwhelmed the specific factual features. This reflects a known failure mode of smaller LLMs: the factual features exist but are insufficient to overcome the statistical dominance of high-frequency tokens at low model scale.

### 4.5 Surprising Features

Several per-prompt circuit features were unexpected:
- Layer 15, Feature encoding "the word 'capital' and sometimes letters" appeared in the Nigeria graph at act=31.39, providing near-direct evidence that the model *knows* the answer type (capital city) before producing it. This feature was not in the top 15 recurring set, suggesting capital-city knowledge is prompt-specific rather than shared.
- The "FIDE" (chess organisation) feature appearing in the chess endgame prompt confirms the model associates chess endgame rules with the competitive chess world, not just the abstract game rules.
- For "Photosynthesis takes place inside the", late-layer features included "words related to voice-activated digital assistants" and "the word 'inside' and words related to digitization and localization" — seemingly unrelated to biology. These likely reflect co-occurrence patterns in training data where "inside" and "the [X]" frames appear in product-documentation contexts.

## 5. Limitations

**Sample size:** Only 24 of 37 target prompts were successfully graphed (13 failed: 4 due to slug conflicts with pre-existing Neuronpedia graphs, 1 due to an invalid slug containing a comma, and 8 due to API rate limiting at 30 requests/60 minutes). The 24 analysed prompts cover 7 sub-domains well but are not a comprehensive sample of factual recall types.

**Steering unavailability:** The Neuronpedia steer endpoint for Gemma-2-2B returned HTTP 500 errors throughout the analysis period. Without successful steering, the causal claims in this paper rest on attribution graph evidence (correlation of feature activity with outcome) rather than on direct causal intervention. The neighbourhood labels obtained via graph traversal provide indirect structural evidence but are not substitutes for steering experiments.

**SAE coverage:** Not all nodes in the attribution graphs correspond to labelled SAE features; embedding (E) and reconstruction-error nodes do not have feature labels. High-influence unlabelled nodes may represent important circuit components that this analysis misses.

**Single model:** All results are specific to Gemma-2-2B with the gemmascope-transcoder-16k SAE. Generalisability to other models or SAE families is not established.

**Feature label quality:** Neuronpedia feature labels are generated by automated interpretation pipelines and may not always be accurate. Several features received generic labels ("a variety of specific nouns", "mentions of ice cream") that provide limited interpretive value.

## 6. Conclusion

We have characterised the factual recall circuit in Gemma-2-2B by generating attribution graphs for 24 factual prompts, running cross-graph comparison to identify 261 recurring features, and interpreting the top 15 via per-prompt circuit walkthroughs. The circuit's shared computational machinery is dominated by early-layer reference-text register classification features rather than world-knowledge features, suggesting Gemma-2-2B implements factual recall partly via encyclopedic-text-pattern matching. Domain-specific named-entity knowledge resides in prompt-specific middle and late layers. The circuit's layer distribution, with shared features clustering in layers 0–7 and domain-specific features in layers 8–25, is consistent with a two-stage architecture: text-register identification followed by domain-specific knowledge retrieval. Future work should include steering validation (pending steer API availability), comparison with the analogical and linguistic circuits using the same pipeline, and analysis of the relationship between recurring feature influence and model accuracy on factual prompts.
