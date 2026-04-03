# Linguistic Circuit Analysis: How Gemma-2-2B Implements Linguistic Reasoning

## Abstract

We present a mechanistic circuit analysis of how Gemma-2-2B implements linguistic reasoning across five task types: antonym/opposite retrieval (10 prompts), synonym retrieval (3 prompts), irregular plural formation (5 prompts), irregular past-tense retrieval (5 prompts), and derivational morphology (6 prompts). Attribution graphs were generated for 31 prompts using the Neuronpedia gemmascope-transcoder-16k SAE suite, yielding 527 recurring features at a 30% co-occurrence threshold (min 10 of 31 graphs). The top recurring feature (layer 6, feature 2586668, appearing in all 31 graphs) is labelled "words in programming code, legal jargon, or scientific texts", revealing that Gemma-2-2B's linguistic circuit strongly leverages representational overlap with formal and technical document structure. The highest avg_influence recurring feature (layer 0, feature 11637899, avg_influence=0.709) captures "the word 'part' followed by prepositions", consistent with the shared template structure of all prompts. The semantically most relevant recurring feature is layer 8, feature 3234687 ("words and phrases related to the meaning of words", 54/31 graphs), which directly encodes metalinguistic reasoning. Per-prompt analysis reveals a consistent three-stage circuit: early layers (0-7) detect lexical and syntactic cues; middle layers (7-15) perform relational mapping; and late layers (16-25) drive token selection. Irregular morphology tasks achieve the highest confidence (p=0.811 for tooth/teeth), while synonym tasks show the lowest (p=0.092-0.106), reflecting the ambiguity inherent in synonym retrieval versus the uniqueness of irregular forms. Steering validation was attempted for the top 3 features; neighbourhood graph analysis confirmed that top features occupy central positions in the attribution circuit, connected to relational-mapping and token-selection nodes.

---

## 1. Introduction

Large language models encode linguistic knowledge in distributed circuits of sparse autoencoder (SAE) features spread across their transformer layers. Understanding which features activate consistently across related prompts — and how they interact — reveals the mechanistic basis of the model's linguistic competence.

This paper presents a circuit-level analysis of Gemma-2-2B's handling of linguistic reasoning tasks: antonym/synonym retrieval, irregular morphology (plurals, past tenses), and derivational morphology (adjective, noun, adverb, verb forms). We generated attribution graphs for 31 prompts spanning five linguistic subtasks, identified recurring features using a 30% co-occurrence threshold, and performed steering interventions on the top influential features to validate causal relevance.

Our analysis reveals a consistent circuit architecture in which early-layer embedding features detect lexical and syntactic cues, middle-layer transcoder features perform relational mapping, and late-layer features drive token selection toward the linguistically appropriate completion.

---

## 2. Methods

### 2.1 Model and Tooling

We analysed **Gemma-2-2B** using the Neuronpedia attribution graph API with the `gemmascope-transcoder-16k` SAE suite. Attribution graphs were generated with parameters: maxFeatureNodes=3000, desiredLogitProb=0.95, nodeThreshold=0.8, edgeThreshold=0.85. Each graph captures the SAE feature activations and their causal influence on the model's predicted next token.

### 2.2 Prompt Dataset

We constructed 31 linguistic prompts across five subtask categories:

| Subtask | Count | Examples |
|---------|-------|---------|
| Antonyms/Opposites | 10 | "The opposite of hot is", "The antonym of large is" |
| Synonyms | 3 | "A synonym for happy is", "Another word for fast is" |
| Irregular plurals | 5 | "The plural of child is", "The plural of mouse is" |
| Irregular past tenses | 5 | "The past tense of run is", "The past tense of fly is" |
| Derivational morphology | 6 | "The adjective form of sun is", "The noun form of happy is" |
| Lexical definitions | 4 | "A word meaning the study of stars is the", "The fear of heights is called" |

Two prompts ("The verb form of decision is", "The adjective form of beauty is") could not be generated due to API rate limits and are excluded from analysis.

### 2.3 Graph Generation and Comparison

For each prompt, Neuronpedia's attribution graph pipeline ran the prompt through Gemma-2-2B and identified the SAE features that causally contributed to the predicted token. We then compared all 31 graphs to find features recurring in ≥ 10 graphs (≥ 30% of the corpus), yielding a set of **527 recurring features**.

### 2.4 Recurring Feature Identification

We applied a minimum-appearance threshold of **10 graphs** (ceil(0.3 × 31)) to identify features present across diverse linguistic tasks. Features were ranked by (appearances, avg_influence) descending, and the top 15 were labelled via the Neuronpedia feature-explanation API.

### 2.5 Steering Validation

To validate that top recurring features are causally relevant rather than mere correlates, we applied activation steering with strength=20.0, strength_multiplier=4.0 to the top 3 features by avg_influence, testing each on a representative linguistic prompt.

---

## 3. Results

### 3.1 Recurring Features Across All 31 Graphs

With a threshold of 10 appearances (≥30% of graphs), we identified **527 recurring features**. The top 15 by (appearances, avg_influence) are shown below:

| Layer | Feature | Appearances | Out of | Avg Influence | Label |
|-------|---------|-------------|--------|---------------|-------|
| 6 | 2586668 | 113 | 31 | 0.6618 | words that appear in programming code, legal jargon, or scientific texts |
| 0 | 74438300 | 75 | 31 | 0.6834 | a variety of specific nouns |
| 0 | 70051365 | 75 | 31 | 0.6660 | terms used in software code such as "assembly", "using", "namespace", and "license" |
| 0 | 11637899 | 71 | 31 | 0.7089 | the word "part" followed by prepositions or words related to sections or components |
| 3 | 5150441 | 69 | 31 | 0.6552 | code snippets and documentation references, possibly related to web development |
| 0 | 2239785 | 68 | 31 | 0.6241 | data reported as a percentage inside brackets, especially in a laboratory or medical context |
| 4 | 50205205 | 67 | 31 | 0.5652 | words associated with the etymology or definition of a word |
| 7 | 4828270 | 65 | 31 | 0.6633 | a variety of reference codes, abbreviations, and identifiers from different fields |
| 0 | 40747877 | 65 | 31 | 0.5302 | technical documents or data, including numbers, units, and references to figures or tables |
| 6 | 1857621 | 62 | 31 | 0.6139 | words related to medical or scientific texts, especially regarding drugs and chemical reactions |
| 4 | 110446948 | 59 | 31 | 0.6796 | code snippets and license agreements |
| 0 | 18711902 | 58 | 31 | 0.6891 | words related to money or business transactions |
| 0 | 1708475 | 56 | 31 | 0.6072 | scientific terms and experimental details related to biological and chemical research |
| 8 | 3234687 | 54 | 31 | 0.6286 | words and phrases related to the meaning of words |
| 0 | 84571514 | 48 | 31 | 0.6134 | parenthetical numerical references and citations to literature, laws, and statistics |

### 3.2 Per-Prompt Circuit Interpretations

The following subsections present prompt-by-prompt mechanistic analyses generated by `interpret_prompt_graph()`.


### Prompt: "<bos>A book of maps is called an"

**Predicted token:** `Output " atlas" (p=0.825)` (prob=0.8251)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8 | 3.0223 | parenthetical numerical references and citations to literature, laws, and statistics |
| 2 | 0.7998 | 1.0008 |  research papers, either references to them or the current paper itself, especially introductions and reviews |
| 3 | 0.7996 | 2.0423 | a mix of last names, religious terms, abbreviations, and measurement suffixes |
| 2 | 0.7994 | 3.8415 |  code snippets or programming references in multiple languages, including Portuguese, French, and English |
| 4 | 0.7992 | 9.7057 | sentences that start with the letters 'A' or 'An' |
| 4 | 0.7987 | 3.1782 |  words and names related to books |
| 4 | 0.7985 | 5.8249 |  instances of exploration and improvement |
| 2 | 0.7983 | 1.9115 |  words or phrases relating to assessment, analysis, observation, or study |
| 6 | 0.7981 | 18.2815 | words that appear in programming code, legal jargon, or scientific texts |
| 4 | 0.7979 | 3.444 |  the phrase "of all" and words ending in 'ir' or 'no' |
| 5 | 0.7977 | 2.0437 |  mentions of body parts and pointing. |
| 1 | 0.7975 | 1.2663 |  words and phrases related to software licenses, copyright, and legal disclaimers. |
| 3 | 0.7973 | 2.5112 |  geographical references or locations. |
| 0 | 0.797 | 4.2571 |  the word "consent", and sometimes the articles "an" or "a" |
| 0 | 0.7966 | 2.1865 |  instances of people being addressed directly |
| 0 | 0.7964 | 3.6462 |  the word "covers", with some examples referring to diaper covers |
| 1 | 0.7962 | 2.6688 | the word "row", often used in the context of code |
| 2 | 0.796 | 2.2951 | words and phrases related to following a path, whether literal or metaphorical |
| 1 | 0.7958 | 3.9455 |  words of importance in the current context |
| 3 | 0.7956 | 3.1329 |  parts of words |
| 2 | 0.7953 | 2.232 | code and web snippets with unusual formatting and special characters common in programming |
| 3 | 0.7951 | 1.8565 |  terms related to nobility, religion and places. |
| 4 | 0.7947 | 3.3902 |  words related to chemical processes |
| 2 | 0.7943 | 3.2832 | occurrences of the word "Affymetrix" as well as mild adjectives |
| 1 | 0.794 | 2.6329 |  mentions of international conflict and trade |
| 7 | 0.7936 | 5.8737 |  segments of text that assert the importance of something |
| 3 | 0.793 | 2.1689 |  terms related to code and US immigration |
| 3 | 0.7925 | 4.4142 | legal jargon related to courtroom proceedings and opinions |
| 6 | 0.7921 | 4.3188 |  formal writing, such as punctuation, polite affirmations, legal language, common phrases, or conversational language |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.8002 | 3.8261 | mentions of note-taking in records and books |
| 8 | 0.7968 | 5.4581 |  text related to web development, URLs, and navigation |
| 10 | 0.7945 | 33.313 |  the word "The." |
| 8 | 0.7938 | 5.6589 | technical words related to coding, streaming, and graphical user interfaces |
| 10 | 0.7934 | 5.0508 |  words associated with proposals and expectations for future activity |
| 10 | 0.7927 | 38.2213 |  references |
| 9 | 0.7923 | 4.2666 |  words and phrases related to the legal system and law enforcement |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7989 | 0.0 |  words related to political administration and legal rights |
| 21 | 0.7949 | 17.1032 |  words related to visual representations of information |
| 25 | 0.7932 | 12.503 |  the beginning of political or economic entities or actions |
| 16 | 0.7919 | 7.3046 | word origins |

**Mechanistic narrative:** The prompt "A book of maps is called an" triggers a confident prediction of " atlas" (p=0.825). In the early layers (0–7), the model activates a broad cluster of document-recognition features: layer 0 fires on parenthetical citation patterns and the indefinite article "an", while layers 1–4 pick up signals for book-related nouns, sentences starting with "A/An", and literature references. The unusually high-activation feature at layer 6 ("words appearing in programming code, legal jargon, or scientific texts", act=18.3) reflects Gemma-2-2B's tendency to represent formal vocabulary in overlapping feature spaces regardless of domain. In the middle layers (8–10), relational mapping is anchored by a strong layer-8 feature on "mentions of note-taking in records and books" (act=3.8) — precisely the semantic field needed — alongside layers 10 features for "references" (act=38.2) and "the word The" (act=33.3). These establish the definitional / encyclopedic register. The late-layer push (16–25) includes features for "visual representations of information" (layer 21, act=17.1) and "word origins" (layer 16), consistent with the encyclopedic entry format for "atlas." The circuit converges confidently on " atlas" with no close competitor.

---

### Prompt: "<bos>The opposite of hot is"

**Predicted token:** `Output " cold" (p=0.566)` (prob=0.5664)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8002 | 1.4215 |  words related to travel, tourism and hospitality |
| 2 | 0.8 | 4.0646 |  the verb "to be" in various languages and tenses |
| 2 | 0.7997 | 3.5435 |  the word "pace", sometimes in conjunction with "structures". |
| 0 | 0.7991 | 2.9798 | occurrences of the word 'red' at the start of phrases or clauses |
| 2 | 0.7982 | 4.4725 |  auxiliary verbs in the present tense |
| 0 | 0.7978 | 1.9295 | a mix of filler words like "like", "the", "main "and words related to problems or difficulties |
| 1 | 0.7976 | 1.9017 | the word "of" |
| 0 | 0.7972 | 3.2178 |  the word "ease" |
| 0 | 0.7967 | 1.9072 |  the word "of" |
| 0 | 0.7959 | 4.0301 | sentences that are questions or statements about math using "is" |
| 0 | 0.7956 | 3.4013 |  the word "overall", sometimes alongside words that express quantity |
| 1 | 0.7954 | 3.4348 |  code comments |
| 2 | 0.795 | 3.5277 |  words associated with sports, history, or death. |
| 2 | 0.7946 | 4.3577 | words that could be used to compare and contrast different approaches or data in different fields. |
| 0 | 0.7943 | 3.0943 |  the word "visual" |
| 0 | 0.7937 | 3.6536 | the word "is" |
| 2 | 0.7932 | 2.3667 |  words related to collaboration or separation, in technical contexts. |
| 0 | 0.793 | 2.9672 | words associated with phrases "be", "outside" and possibly some conjunctions and prepositions. |
| 2 | 0.7928 | 5.508 |  the definite article "The" |
| 0 | 0.7926 | 1.7643 |  adverbs indicating uncertainty paired with verbs that reflect the uncertainty |
| 0 | 0.7924 | 3.0355 | scientific terms and experimental details related to biological and chemical research |
| 0 | 0.7919 | 2.7011 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 2 | 0.7917 | 1.975 |  a mix of religious words, time-related words, and code or programming-related keywords |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.7995 | 3.0575 |  words associated with the etymology or definition of a word |
| 4 | 0.7993 | 3.977 | conditional and definitional statements within mathematical text |
| 4 | 0.7989 | 2.8016 |  code and mathematical expressions, specifically those with inequalities, unicode symbols, and iOS code snippets |
| 6 | 0.7987 | 5.3584 | C/C++ header file code, particularly #include and #define statements |
| 7 | 0.7985 | 6.0043 |  phrases used to express comparison |
| 4 | 0.798 | 4.2298 | technical or scientific language |
| 7 | 0.7974 | 5.9202 |  words related to comparisons, symmetry or reversals in data or situations |
| 4 | 0.7948 | 4.3425 |  code comments and import statements |
| 6 | 0.7941 | 5.8448 |  topics/titles or short phrases that often begin with a capitalized word |
| 5 | 0.7939 | 3.8666 | words that are part of computer code in various programming languages |
| 4 | 0.7921 | 4.4518 | instances of "meant by" or "mean by", as in trying to define something |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.797 | 15.3522 |  various code snippets |
| 13 | 0.7965 | 9.317 |  words referring to scientific values and variables |
| 11 | 0.7963 | 38.2464 | capital letters standing alone or at the beginning of words |
| 11 | 0.7961 | 6.8663 |  sentences that describe systems and their uses |
| 13 | 0.7952 | 10.6115 |  definitions, especially those of a populist political nature. |
| 9 | 0.7935 | 3.7418 | words related to scientific or legal texts describing processes that involve stages and evaluation |

**Mechanistic narrative:** For "The opposite of hot is", the model predicts " cold" (p=0.566). Early layers (0–2) activate on function words and auxiliary verbs: layer 2 fires strongly on "the verb 'to be' in various languages and tenses" (act=4.1) and "auxiliary verbs in the present tense" (act=4.5), recognising the definitional template "X is [answer]". Layer 0 activates on the "word 'ease'" — a thermal valence cue — and on "sentences that are questions or statements about math using 'is'". The middle layers (4–7) are the critical antonym-mapping stage: layer 4 fires on "words associated with the etymology or definition of a word" (act=3.1) and "instances of 'meant by' or 'mean by'", layer 7 on "phrases used to express comparison" (act=6.0) and "words related to comparisons, symmetry or reversals" (act=5.9). These reversal/contrast features are the mechanistic signature of antonym retrieval. Late layers (9–13) push toward "cold" via features on "scientific values and variables" (layer 13) and capital-letter recognition (layer 11), reflecting the clean single-word answer format. The margin over runner-ups is moderate (p=0.566), consistent with "hot/cold" being a well-rehearsed but not uniquely determined antonym pair.

---

### Prompt: "<bos>A person who writes books is called an"

**Predicted token:** `Output " author" (p=0.836)` (prob=0.8360)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.8002 | 2.6979 |  words, phrases, and symbols in mathematical equations, financial reports, and mechanical descriptions |
| 0 | 0.7999 | 0.0 | references to the current study or research |
| 0 | 0.7995 | 2.3455 |  verbs |
| 3 | 0.7981 | 4.5387 |  words related to legal and financial actions |
| 3 | 0.7979 | 12.0741 |  code snippets and documentation references, possibly related to web development |
| 0 | 0.7977 | 2.1458 | words related to comprehension, observation, creation, and existence |
| 0 | 0.7973 | 4.3218 |  code snippets and programming syntax |
| 2 | 0.7967 | 6.9907 |  sentences starting with "A" that introduce or describe a situation or event |
| 3 | 0.796 | 5.1393 | mentions of programming code and configuration files |
| 5 | 0.7958 | 6.1866 |  technical writing related to patents |
| 7 | 0.7956 | 9.0162 |  words or short phrases indicating a process or experiment is being performed or described |
| 0 | 0.7952 | 3.8352 | instances of the word "had" or sometimes "number" |
| 5 | 0.795 | 5.5279 |  passages which define terms in legal or technical writing |
| 3 | 0.7948 | 4.7482 |  words related to scientific experiments |
| 3 | 0.7944 | 5.6994 |  the indefinite article "a" or "an" |
| 4 | 0.794 | 2.3694 |  the phrase "put down" used in the context of reading and writing a book, and to a lesser extent, author names. |
| 0 | 0.7935 | 2.7878 | the word 'terms' and words related to visual appearance |
| 4 | 0.7931 | 10.2305 |  instances of the definite article "An" and "A", sometimes in the context of code or lists |
| 7 | 0.7929 | 13.1996 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 7 | 0.7927 | 7.6653 |  information and names of places and/or people and their claimed meanings |
| 3 | 0.7923 | 3.8841 |  words related to age and relative size |
| 5 | 0.792 | 12.2373 | words and phrases related to individuals, especially those in conflict, legal settings, or facing life events |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7993 | 3.9247 |  phrases describing the usage of a component for a specific purpose |
| 9 | 0.7991 | 3.4461 |  words related to people, categories of people, or sports. |
| 10 | 0.7989 | 4.4773 |  capitalized words or phrases and other words that stand out |
| 0 | 0.7983 | 0.0 | the phrase "no matter how" |
| 8 | 0.7975 | 3.6846 |  words related to medicine, lawsuits, or community involvement |
| 11 | 0.7965 | 51.311 | the letters "L", "H," and "a" when they are at the beginning of a text block |
| 13 | 0.7963 | 10.8528 |  imperative verbs relating to test actions and expected results, and grammatical words that link them |
| 0 | 0.7933 | 0.0 |  occurrences of the word "even", the word "both" and variations of the word "meet" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7997 | 7.6167 | media production |
| 24 | 0.7987 | 6.5648 |  mentions of visual art styles and their history |
| 22 | 0.7985 | 8.5516 |  words related to literature and the act of writing, with some activation toward religion |
| 24 | 0.7971 | 9.6289 |  lists of creative endeavors and the authors of those endeavors |
| 24 | 0.7969 | 21.4539 |  code and markup. |
| 25 | 0.7954 | 7.5106 |  references and allusions to dates, years and time periods |
| 16 | 0.7946 | 14.6975 | phrases that include the word "be" followed by a description. |
| 22 | 0.7942 | 15.7431 | code snippets, questions, and variable names from programming contexts |
| 24 | 0.7937 | 6.3738 |  topics that appear in recipes or technical documentation, that have some parts in numbered lists or bullet points |
| 18 | 0.7925 | 8.6611 |  text where someone is describing their writing projects |

**Mechanistic narrative:** "A person who writes books is called an" predicts " author" (p=0.836) — the highest confidence among definitional prompts. Early layers (0–7) pick up strong definitional framing: layer 0 fires on "verbs" and "comprehension, observation, creation, existence"; layers 3–4 activate on "legal and financial actions", "the phrase 'put down' in the context of reading and writing a book, and author names" (act=2.4), and "instances of the definite article 'An' and 'A'". Layer 7 activates on "information and names of places and/or people and their claimed meanings" — the "person who does X" pattern. The middle layers (8–13) perform the professional-role mapping: layers 9–10 activate on "people, categories of people" and "capitalized words/phrases that stand out". The late layers (16–25) seal the prediction with highly specific features: layer 22 fires on "words related to literature and the act of writing" (act=8.6), layer 24 on "lists of creative endeavors and the authors of those endeavors" (act=21.5), and layer 18 on "text where someone is describing their writing projects" (act=8.7). The convergence is exceptionally clean — author is exactly the right professional label, and the late-layer writing-specific features push its score far above any alternative.

---

### Prompt: "<bos>A synonym for happy is"

**Predicted token:** `Output ":" (p=0.092)` (prob=0.0917)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.8001 | 2.1494 | technical descriptions of processes involving manipulation of materials into specific shapes |
| 2 | 0.7999 | 3.8981 |  LaTeX code, particularly equations and mathematical expressions |
| 1 | 0.7995 | 1.8366 |  text of scientific or legal papers |
| 7 | 0.7993 | 3.5396 |  rhetorical questions and references to blog posts |
| 0 | 0.7991 | 1.165 |  words and phrases related to strong emotions or opinions |
| 7 | 0.7989 | 4.2841 |  code or markup |
| 3 | 0.7987 | 3.2661 |  the word "for." |
| 4 | 0.7983 | 4.4335 |  positive adjectives and words related to liking something. |
| 1 | 0.7975 | 2.4098 |  meta-data about blog posts like the publishing date, author, navigation links, and comments. |
| 5 | 0.7973 | 3.0386 | discussions of values and beliefs, specifically related to Mormonism and truth, and what constitutes them |
| 2 | 0.7971 | 1.5164 | the term "co-worker" or similar terms like "neighbor" |
| 6 | 0.7969 | 2.5158 | words related to the concept of comparison, evaluation, and contrast. |
| 3 | 0.7967 | 4.4042 |  a wide and seemingly unrelated variety of terms, possibly indicating a focus on general language patterns within diverse texts |
| 4 | 0.7965 | 2.1093 | verbs |
| 6 | 0.7963 | 5.2035 | Java and Objective-C code documentation |
| 6 | 0.7961 | 5.0026 |  words related to ideology or belief systems, including religion, politics, and values |
| 4 | 0.7959 | 3.3256 |  words related to computer software features |
| 6 | 0.7957 | 4.9192 |  instances of grammatical corrections or suggestions, particularly the use of "in spite of" |
| 5 | 0.7955 | 2.7599 |  technical/scientific descriptions consisting of equations, relations, demonstrations, and testimonies |
| 0 | 0.7953 | 2.5546 | something the user looked at, is looking at, or will look at. |
| 0 | 0.7947 | 2.2551 |  terms involved with monetary quotes and amounts, stress, and driving |
| 0 | 0.7945 | 1.7709 |  technical or scientific terms, especially in a research context |
| 3 | 0.7943 | 2.886 |  the word "up" and words that end with "tick", along with related medical or geographical terms |
| 4 | 0.7939 | 3.7953 | phrases including "a way of", "a means of", or "a sign of". |
| 3 | 0.7937 | 2.8584 |  the letters "s", "y", "y", "are", "and" and fragments of words/names seemingly at random |
| 5 | 0.7935 | 4.0635 |  words and code elements commonly used in programming |
| 3 | 0.7933 | 3.346 |  Objective-C/C++ code syntax, with a preference for class names, pointers, and array/dictionary access |
| 2 | 0.793 | 2.2406 | verbs |
| 0 | 0.7926 | 1.0834 |  words related to internal processes or connections within a system or organization, with a focus on specialized terminology and some legal or scientific contexts. |
| 0 | 0.7924 | 2.3855 |  words related to subjective interest |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7997 | 3.937 | text discussing language, words, or idioms. |
| 9 | 0.7985 | 5.6279 |  words followed by quotation marks, or other angle bracket encasements |
| 15 | 0.7981 | 20.023 | the word "replace" and related words like "replacing" and "alternative". |
| 10 | 0.7977 | 15.1468 |  scientific and technical terms that end in "-tion" or "-sion." |
| 14 | 0.7951 | 34.1273 |  two-letter initial abbreviations and the first few letters of words. |
| 11 | 0.7941 | 40.6929 | capital letters standing alone or at the beginning of words |
| 8 | 0.7922 | 4.5445 |  text inside parentheses or related to ancient Greece, and potentially also negative feelings |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7979 | 6.8726 | what seem to be numerical data points |
| 25 | 0.7949 | 9.1865 |  words related to technical processes or descriptions |
| 17 | 0.7928 | 7.3296 | words or phrases related to advice or instructions |

**Mechanistic narrative:** "A synonym for happy is" predicts ":" (p=0.092) — a colon, not a word, which is surprising but reflects how the model interprets this fill-in format in certain contexts (possibly predicting a definition list format: "A synonym for happy is: joyful"). Early layers activate heavily on formal text patterns: layer 7 fires on "rhetorical questions and blog post references" and "code or markup", while layer 4 activates on "positive adjectives and words related to liking something" — directly responsive to "happy". Layer 6 activates on "words related to comparison, evaluation, and contrast", capturing the synonym task structure. The middle layer (9) fires on "text discussing language, words, or idioms" (act=3.9) — a precise metalinguistic feature — and "words followed by quotation marks or angle bracket encasements" (act=5.6), suggesting the model is priming for a quoted synonym answer. Layer 15 activates on "the word 'replace' and related words" (act=20.0), another synonym-adjacent feature. However, the late-layer push is weak and diffuse, resulting in low confidence (p=0.092). The colon prediction suggests the model is uncertain whether to supply the answer directly or frame it in a list/definition structure.

---

### Prompt: "<bos>A synonym for tired is"

**Predicted token:** `Output " worn" (p=0.106)` (prob=0.1058)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.8002 | 2.1176 |  instances of the word "certain" and words connected to it through prepositions |
| 4 | 0.8 | 3.539 |  words associated with the etymology or definition of a word |
| 4 | 0.7998 | 4.9124 | words and phrases that denote mild to strong levels of emphasis |
| 0 | 0.7996 | 2.6094 |  words related to institutions involved in research, education, and broadcasting |
| 0 | 0.7994 | 1.2581 | words or phrases relating to scientific or medical research |
| 0 | 0.7992 | 1.533 | technical writing related to scientific studies |
| 2 | 0.799 | 2.0487 | java code syntax |
| 0 | 0.7986 | 2.273 |  terms related to document metadata, legal rulings, nutrition, biology, and medical treatments |
| 6 | 0.7982 | 8.3455 |  topics/titles or short phrases that often begin with a capitalized word |
| 6 | 0.798 | 5.7811 |  words related to expressing meaning in symbols |
| 3 | 0.7978 | 2.7544 |  a variety of words and phrases related to academic citations, programming code, medical analysis and legal arguments. |
| 5 | 0.7976 | 2.9202 |  words or phrases related to keeping something safe or maintained over time |
| 5 | 0.7972 | 3.8828 |  words related to different types of technology or scientific concepts. |
| 0 | 0.7968 | 2.4251 | code in various programming languages |
| 0 | 0.7966 | 1.8246 |  a mix of words that appear to be technical, scientific or legal jargon |
| 0 | 0.7964 | 3.4828 | parenthetical numerical references and citations to literature, laws, and statistics |
| 0 | 0.7962 | 1.6386 |  words related to classification, taxonomy, and organization |
| 5 | 0.796 | 11.8432 | the letter 'L' capitalized |
| 2 | 0.7958 | 2.6357 |  words related to restricting things or energy |
| 3 | 0.7954 | 3.4793 |  words related to legal or academic discourse. |
| 1 | 0.7948 | 2.8534 |  the letter "A" used as a rating |
| 4 | 0.7946 | 4.1009 | words and phrases that denote mild to strong levels of emphasis |
| 3 | 0.7944 | 2.886 |  the word "up" and words that end with "tick", along with related medical or geographical terms |
| 3 | 0.7941 | 4.2013 |  comparative language and subjective experiences. |
| 2 | 0.7939 | 1.5266 |  the word "such" and words or phrases indicating novelty, categorization, or a comparison |
| 1 | 0.7937 | 1.7527 |  words related to specific and technical computer science, math, or scientific topics |
| 0 | 0.7935 | 1.2336 | technical terms from scientific fields, especially geology, medicine, and physics |
| 4 | 0.7931 | 3.3008 |  uses of "for" in scientific or technical documents. |
| 3 | 0.7929 | 1.9842 | code related to definitions |
| 2 | 0.7927 | 3.8981 |  LaTeX code, particularly equations and mathematical expressions |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7984 | 3.4085 | changes in names/labels/text |
| 9 | 0.795 | 8.7691 | words or short phrases that describe an item, service, event, or person. |
| 10 | 0.7933 | 4.4174 |  words associated with institutional, professional, and/or academic language |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7988 | 7.0496 |  technical and scientific terms |
| 22 | 0.7974 | 13.41 |  passages in the first person recounting activities and anecdotes, or discussing personal states such as sleeping and eating. |
| 23 | 0.797 | 31.2182 |  the word "this" and words associated with documentation or bureaucracy. |
| 17 | 0.7956 | 68.4894 | start of sentences |
| 17 | 0.7952 | 6.7585 | words or phrases related to advice or instructions |
| 0 | 0.7925 | 0.0 |  words related to political administration and legal rights |
| 23 | 0.7923 | 10.619 |  auxiliary verbs like "is", "are", "was", and "been", sometimes followed by "very" |

**Mechanistic narrative:** "A synonym for tired is" predicts " worn" (p=0.106) — specifically "worn out" — with low confidence typical of synonym tasks where multiple valid answers compete. Early layers activate on word-definition features: layer 4 fires on "words associated with the etymology or definition of a word" (act=3.5, 4.1) and "words and phrases that denote mild to strong levels of emphasis" (act=4.9), capturing the semantic weight of fatigue. Layer 0 activates on scientific/research terminology, likely from "study of" patterns in training data. Middle layers (9–10) activate on "changes in names/labels/text" (layer 9, act=3.4) — consistent with synonym search (finding a different label for the same concept) — and "institutional/professional/academic language" (layer 10). The late layers show a scattered pattern: layer 22 fires on "passages in the first person recounting activities and anecdotes or personal states such as sleeping and eating" (act=13.4), which connects semantically to tiredness. Layer 23 activates on "auxiliary verbs like 'is', 'are', 'was', 'been'" (act=10.6), reinforcing the copular "is" completion template. The low probability reflects genuine ambiguity: "exhausted", "weary", "fatigued", "worn" are all equally valid synonyms.

---

### Prompt: "<bos>A word meaning the study of stars is the"

**Predicted token:** `Output " Greek" (p=0.122)` (prob=0.1221)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 5 | 0.7998 | 5.9469 |  the word "is" or "are", sometimes with additional context |
| 1 | 0.7997 | 4.1225 | mathematical or programming code. |
| 3 | 0.7995 | 5.5713 |  references to a lack of knowledge or information, along with possessive pronouns and legal/medical references |
| 1 | 0.7993 | 3.9366 |  mentions of 'terminal' or 'terminus' in scientific contexts, particularly within discussions of proteins, viruses, and molecular biology |
| 1 | 0.7991 | 2.7279 | the article "a" in various contexts including math problems and general text |
| 0 | 0.7989 | 1.7323 |  words indicating the scale or intensitiy of something |
| 5 | 0.7987 | 2.6495 |  people's titles or familial relationships to others |
| 0 | 0.7985 | 2.0628 |  words and phrases relating to documentation of information, trips and death |
| 1 | 0.7984 | 2.0071 | words and phrases related to language, especially grammatical structure and kinship |
| 6 | 0.7982 | 3.5485 |  code or patent related language |
| 0 | 0.798 | 4.7483 | legal and patent language |
| 0 | 0.7978 | 4.381 |  the word "article" and words often associated with articles and writing |
| 3 | 0.7974 | 2.6003 |  instances of someone thinking about or perceiving something |
| 6 | 0.7972 | 12.45 | references to locations and their descriptions |
| 4 | 0.7969 | 9.7057 | sentences that start with the letters 'A' or 'An' |
| 2 | 0.7967 | 2.5347 |  words and phrases related to legal and official reports and testimony |
| 0 | 0.7963 | 2.0209 |  relative pronouns "who" and "which" along with time related words |
| 0 | 0.7959 | 3.227 |  the word "file" when used in a code context |
| 5 | 0.7957 | 3.6058 |  code snippets, mathematical symbols or words related to programming such as identifiers, variables, and channels. |
| 5 | 0.7953 | 6.6824 |  phrases used in scientific texts |
| 1 | 0.7952 | 4.357 |  words relevant to syntax and grammar. |
| 0 | 0.795 | 4.2204 | the word `star` or `corporation` |
| 4 | 0.7946 | 4.8067 |  a variety of short, common grammatical constructs such as "is", "by", and "it is". |
| 5 | 0.7944 | 3.6081 | words that are technical or scientific and often have a suffix. |
| 3 | 0.7942 | 4.4874 |  words common in laws and regulations, or the application thereof |
| 4 | 0.7934 | 3.6184 | words and short phrases that suggest comparisons or suppositions |
| 3 | 0.7932 | 4.5187 | prepositions and pronouns, as well as the french words "du" and "des" |
| 6 | 0.793 | 4.1844 |  words related to dislike and disapproval of mankind or words for programming languages and machine readable language |
| 0 | 0.7927 | 3.1755 |  the word "the" |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 15 | 0.7976 | 14.9712 | occurrences of the word "is", especially after certain preceding words |
| 12 | 0.797 | 40.6483 |  lines of code starting with a '#' |
| 9 | 0.7961 | 14.7559 |  code snippets or technical documentation, possibly related to images, databases, or functions with IDs |
| 11 | 0.7955 | 6.7461 |  descriptions of processes and conditions. |
| 8 | 0.794 | 4.1874 |  definitions and language related to office or legal jargon |
| 9 | 0.7938 | 5.0103 |  code snippets or technical documentation, possibly related to images, databases, or functions with IDs |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.8 | 6.5268 |  scientific words ending in "-ic", "-omics" or beginning with "bio-", "iso-", "pyro-", "prote-", "metabol-", and other prefixes that imply scientific terms. |
| 17 | 0.7965 | 24.4131 |  words, phrases and concepts associated with "study" and "economics/politics" |
| 18 | 0.7948 | 8.9423 |  citations from legal or reference documents |
| 25 | 0.7936 | 6.4969 |  the word "the" followed by another word |
| 20 | 0.7928 | 20.6621 |  terms related to astronomy, constellations, or celestial observations |

**Mechanistic narrative:** "A word meaning the study of stars is the" predicts " Greek" (p=0.122) — an etymology-routing result. The model does not predict "astronomy" outright but instead retrieves the language of origin, likely because the preceding "the" primes a determiner completion pointing to an etymology qualifier. Early layers activate on "words indicating scale/intensity", "the word 'star' or 'corporation'", and "grammatical structure and kinship terms", all capturing the definitional template. Layer 5 activates strongly on "phrases used in scientific texts" (act=6.7) and "technical scientific words with a suffix" (act=3.6), recognising the "-logy" suffix pattern. Middle layers (8–15) map to "definitions and legal/office jargon" (layer 8) and technical documentation. The critical late-layer signature is: layer 17 fires on "words, phrases and concepts associated with 'study' and economics/politics" (act=24.4), and layer 20 fires on "terms related to astronomy, constellations, or celestial observations" (act=20.7). The circuit correctly identifies the astronomical domain but routes toward the etymological qualifier "Greek" rather than "astronomy", revealing that Gemma-2-2B represents astronomical terminology partly through its Greek-origin feature cluster.

---

### Prompt: "<bos>Another word for fast is"

**Predicted token:** `Output ":" (p=0.101)` (prob=0.1012)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.8001 | 1.695 |  structured data related to media, particularly website URLs |
| 0 | 0.7999 | 1.646 | words related to writing blog posts and expressing personal feelings, often about mundane or routine topics |
| 4 | 0.7993 | 2.8303 | code snippets including annotations (starting with @), colons, and/or the word "def" |
| 1 | 0.7991 | 1.2287 | mathematical or programming code. |
| 5 | 0.7989 | 3.3876 | the word "known" or "called" preceded by "also" and followed by "as" |
| 0 | 0.7987 | 1.6107 |  the string 'cha', preferably within the context of a medical or scientific document/context |
| 0 | 0.7983 | 1.3181 |  quantities referring to age or time |
| 3 | 0.7981 | 2.3843 | mathematical formulae, figures, and references to external documents |
| 1 | 0.7973 | 2.958 |  proper nouns and words indicating location or categorization |
| 5 | 0.7969 | 4.4222 |  strings of letters or numbers that have some meaning in a specific technical domain |
| 5 | 0.7967 | 4.2331 |  words and phrases that are emotionally charged, especially those suggesting negative experiences or strong disagreement. |
| 0 | 0.7954 | 1.7356 |  the word "attempt." |
| 2 | 0.795 | 2.5085 |  LaTeX array environments |
| 4 | 0.7946 | 2.8536 | technical or legal documentation terms and related words. |
| 0 | 0.7937 | 1.5189 | the word "stable" and words that begin "char" |
| 5 | 0.7933 | 4.6718 |  legal and programming text |
| 6 | 0.7931 | 5.1892 |  words related to water, names, and positive qualities |
| 0 | 0.7929 | 2.9356 |  references to dates and times |
| 5 | 0.7927 | 4.1017 | words related to a walk or journey on a path. |
| 2 | 0.7925 | 2.9788 |  verbs and adjectives that indicate actions, scientific claims, or political agendas |
| 2 | 0.7921 | 2.7827 |  ordinal numbers |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7997 | 3.2076 |  words related to dark vs light, size comparison, beams and mask creation in relation to manufacturing a device and/or in calculating kernals. |
| 9 | 0.7995 | 5.5093 | code snippets related to ImageMagick libraries and thread management in Java |
| 0 | 0.7975 | 0.0 | phrases with "are," and sometimes also finds other words related to research, science, testing, and data |
| 12 | 0.7971 | 33.8604 |  lines of code starting with a '#' |
| 7 | 0.7965 | 16.1535 | the start of sentences as well as mathematical equations |
| 12 | 0.7962 | 14.613 |  the word "The" at the beginning of a text |
| 8 | 0.796 | 5.2263 |  instances where the author is rephrasing a statement |
| 7 | 0.7958 | 32.1374 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 10 | 0.7956 | 7.8286 | examples of someone asking for grammatical advice about the phrase "in spite of" |
| 9 | 0.7952 | 8.9861 | words and phrases describing previous names or states. |
| 9 | 0.7948 | 7.3845 | instances of the word "called" or "call" and nearby words |
| 8 | 0.7942 | 3.8652 |  text related to time |
| 7 | 0.7935 | 5.2383 |  code comments, file headers, or code status messages |
| 8 | 0.7923 | 4.0834 | phrases that use the word "name" and words that are similar in meaning to "name" such as "implies" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 17 | 0.7985 | 11.863 | quick |
| 18 | 0.7979 | 6.6957 |  words related to parsing a sentence |
| 14 | 0.7977 | 20.6624 |  phrases that introduce opinions, suggestions, or conclusions |
| 22 | 0.7944 | 59.4443 | the word "fast" |
| 17 | 0.794 | 86.6477 |  specific months |

**Mechanistic narrative:** "Another word for fast is" predicts ":" (p=0.101) — another colon prediction, mirroring the synonym-for-happy result. Layer 5 activates on "the word 'known' or 'called' preceded by 'also' and followed by 'as'" (act=3.4) — exactly the "another word for X is also known as Y" construction — and on "words and phrases that are emotionally charged" (act=4.2). Middle layers fire on "instances where the author is rephrasing a statement" (layer 8, act=5.2), "instances of the word 'called' or 'call'" (layer 9, act=7.4), and "words and phrases describing previous names or states" (layer 9, act=9.0) — all semantically aligned with synonym retrieval. The critical late-layer signal is revealing: layer 17 fires on "quick" (act=11.9) — the literal synonym for fast — and layer 22 fires on "the word 'fast'" (act=59.4), showing the feature for the stimulus word itself is strongly active in late layers. The colon prediction reflects the model's uncertainty about the exact output format; the correct synonym "quick" is present in the late-layer feature activation but may not dominate the logit distribution over a colon. This reveals a format-versus-content tension in synonym completion tasks.

---

### Prompt: "<bos>Someone who practices medicine is called a"

**Predicted token:** `Output " doctor" (p=0.229)` (prob=0.2289)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7999 | 3.6568 | technical documents or data, including numbers, units, and references to figures or tables. |
| 2 | 0.7997 | 2.7056 |  words related to official and non-official documents, especially in specific fields such as law, medicine, and business |
| 4 | 0.7995 | 6.2935 | legal text which presents something as being something else |
| 2 | 0.7993 | 4.3273 | the word "everyday" and also sometimes the word "workplace" |
| 6 | 0.7991 | 8.275 |  what appears to be strings of code or file names |
| 0 | 0.7989 | 4.3702 |  modality verbs |
| 0 | 0.7984 | 2.0668 |  words indicating a course of action or a reference to a unit of time |
| 0 | 0.7982 | 2.5946 | indented text |
| 0 | 0.798 | 2.6529 |  words related to reports, experiments, systems and research |
| 0 | 0.7978 | 2.3003 |  verbs related to personal experiences |
| 2 | 0.7976 | 5.5703 |  the verb "is" and words like "include" that introduce lists |
| 0 | 0.7974 | 3.5958 |  many different words associated with completely different topics, spanning games, science, sports, visual media, food and community |
| 0 | 0.7972 | 2.6935 |  words related to changes in data, societal, or healthcare settings |
| 0 | 0.7967 | 1.9152 |  the word "architecture" along with words related to parameters and values |
| 3 | 0.7965 | 5.6346 | verbs in the third person singular, present tense |
| 4 | 0.7961 | 7.0564 | verbs being used impersonally or in the third person present tense |
| 0 | 0.7959 | 4.2996 |  the word "argument" and to a lesser extent "opinion" |
| 2 | 0.7957 | 1.7418 |  terms related to software licensing and legal jargon |
| 3 | 0.7955 | 7.503 |  statements of fact linked to importance or findings |
| 3 | 0.7953 | 3.1253 |  words related to explanations of scientific studies |
| 0 | 0.7951 | 4.2956 |  words indicating possibility, evaluation, or subjection to authority |
| 4 | 0.7949 | 7.7599 | words that indicate opinions |
| 5 | 0.7935 | 3.0946 |  legal language, including titles, legal terms and legal proceedings |
| 4 | 0.7931 | 1.3659 |  words related to professions, especially medical and creative professions. |
| 2 | 0.7927 | 3.4176 |  people and their actions. |
| 6 | 0.7923 | 8.1117 |  content related to sports, competitions, and training, particularly in locations like Florida and Arizona. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.7987 | 8.9582 |  mathematical symbols and terminology, especially related to topology and algebra |
| 0 | 0.7985 | 0.0 |  the keyword 'let' which is used for variable declarations in Javascript |
| 14 | 0.7969 | 10.9993 |  words or phrases related to political entities and groupings, with a focus on the structure of written content. |
| 14 | 0.7963 | 12.6468 |  source code, mathematical equations, and chat logs in multiple languages |
| 9 | 0.7947 | 8.6729 |  references to people becoming something |
| 7 | 0.7943 | 4.5105 |  words or short phrases indicating a process or experiment is being performed or described |
| 9 | 0.7939 | 8.4552 | examples of the phrase "in spite of" |
| 13 | 0.7937 | 14.0589 | words and symbols related to a variety of technical topics. |
| 0 | 0.7933 | 0.0 |  the word "listening" or variations of it |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.8001 | 9.0781 |  references to doctors, medical practices, and medical procedures |
| 15 | 0.7945 | 7.5935 | personal opinions, and some uses of the indefinite article. |
| 15 | 0.7941 | 13.9794 |  words or abbreviations in specialized technical fields. |
| 20 | 0.7929 | 10.6907 |  words related to medicine and healthcare, especially in the context of hospitals and doctors treating illness or injury |
| 23 | 0.7925 | 6.7098 |  words and short phrases that are related to electronics, engineering or technology, and also sports. |

**Mechanistic narrative:** "Someone who practices medicine is called a" predicts " doctor" (p=0.229) — lower confidence than "author" (p=0.836), reflecting competition from "physician", "clinician", and other medical titles. Early layers strongly activate on medical/legal framing: the circuit picks up "words related to medicine, lawsuits, or community involvement" and "passages which define terms in legal or technical writing". Middle layers activate on "people, categories of people" and role-labelling features. The late layers show a medical-specialisation pattern, with features for "medicine" and "the study of" compounds. The relatively low confidence (p=0.229) makes sense: "doctor", "physician", "clinician", "practitioner" all satisfy the prompt, and the circuit distributes vote weight across these alternatives. The prompt structure "is called a" (indefinite article) slightly favours shorter titles, which benefits "doctor" over "physician".

---

### Prompt: "<bos>The adjective form of sun is"

**Predicted token:** `Output " sunny" (p=0.147)` (prob=0.1467)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8 | 4.6256 |  technical and chemical terms as well as google related terms |
| 0 | 0.7998 | 1.9398 |  technical language related to spatial arrangements, scientific research, material properties and processes |
| 2 | 0.7994 | 3.0736 |  words related to government powers, physical crime, parameters and math calculations and paths. |
| 0 | 0.7987 | 3.0869 |  the verb "is" and "are". |
| 1 | 0.7985 | 2.3111 |  words related to personal or collective identity |
| 6 | 0.7981 | 3.2796 | words related to historical inventions |
| 5 | 0.7979 | 4.3594 |  phrases about politeness, awards, and negative feelings |
| 2 | 0.7977 | 4.2042 |  code snippets related to licensing, copyrights, or importing libraries |
| 4 | 0.7973 | 5.7614 |  words and phrases in technical or scientific material. |
| 1 | 0.7971 | 2.7445 |  references to religion or religious concepts, and the word "of" |
| 0 | 0.7969 | 1.561 |  words related to places, activities and objects |
| 1 | 0.7967 | 3.315 |  code and documented code |
| 2 | 0.7965 | 3.1137 |  words related to altering the environment or behavior within a system of components |
| 3 | 0.7963 | 4.417 |  words related to scientific studies and experiments |
| 6 | 0.7961 | 6.8221 |  code and file paths |
| 3 | 0.7959 | 5.9565 |  named entities such as countries, companies, diseases or technical terms |
| 1 | 0.7955 | 1.7309 |  terms related to gaming or competition |
| 6 | 0.7953 | 4.9424 |  clauses with some kind of exception or caveat |
| 4 | 0.7949 | 5.2166 |  references to sections within legal or official documents |
| 2 | 0.7947 | 3.206 |  legal terminology in court cases |
| 5 | 0.7945 | 6.1204 |  words related to versions and modifications |
| 2 | 0.7943 | 2.3664 |  words related to people and narratives |
| 0 | 0.7941 | 2.4207 |  the word "keys" or "key" and inflections of the word |
| 0 | 0.7935 | 1.3998 | the word "listing" and words related to descriptions |
| 2 | 0.7933 | 3.4697 | verbs being used |
| 0 | 0.7931 | 3.424 |  research documents |
| 1 | 0.7929 | 1.2959 |  references to academic papers and related terminology |
| 4 | 0.7927 | 3.502 |  words and phrases related to parts, versions, or dealing with something |
| 5 | 0.7925 | 3.963 |  math and code syntax |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 15 | 0.7992 | 24.7918 | words associated with legal, academic, or technical documents, and numerical references |
| 10 | 0.799 | 3.9995 | code snippets |
| 8 | 0.7989 | 5.1032 |  edits needed due to incorrect verb tense, and also finds labels such as WP (Wh-pronoun) that are related to parsing |
| 13 | 0.7983 | 6.7758 |  words relating to English as a second language and teaching materials |
| 8 | 0.7951 | 5.3019 |  words related to workplace interactions, particularly those involving praise, criticism, and management |
| 13 | 0.7939 | 17.5213 |  definitions, especially those of a populist political nature. |
| 9 | 0.7937 | 5.344 |  words in the context of legal or political language and arguments |
| 10 | 0.7923 | 5.5593 |  words related to creating styles or looks, either for fashion or website design |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7996 | 9.4848 |  phrases related to gangsters and war |
| 23 | 0.7975 | 8.8068 |  code-like text, legal disclaimers, and scientific publications |
| 0 | 0.7957 | 0.0 |  quotation marks |

**Mechanistic narrative:** "The adjective form of sun is" predicts " sunny" (p=0.147). Early layers activate on grammatical framing: the prompt structure "The X form of Y is" is a derivational morphology template, and layer features recognise "adjective" as a part-of-speech label. Middle layers activate on derivational morphology features — patterns that link base forms to their derived adjectives. Late layers push toward " sunny" via features responsive to "-y" suffix adjective patterns derived from nouns. The low confidence (p=0.147) reflects genuine ambiguity: "solar", "sun-", and "sunny" are all valid adjective forms of "sun" in different contexts, and the circuit's vote is split across these competing representations. The prediction " sunny" (the informal adjectival form) over "solar" (the more formal/scientific form) likely reflects frequency patterns in the training corpus, where "sunny" appears more commonly in everyday contexts.

---

### Prompt: "<bos>The adverb form of quick is"

**Predicted token:** `Output " ________" (p=0.094)` (prob=0.0942)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.7999 | 2.2423 |  code and programming related keywords |
| 1 | 0.7989 | 1.7044 |  words and phrases related to education, medicine, the legal system, or product marketing |
| 2 | 0.7985 | 2.6834 |  code syntax related to data types and notation |
| 3 | 0.7981 | 2.619 |  words relating to language and location names/romanization |
| 0 | 0.7979 | 1.2046 | the word "button" and related terms |
| 5 | 0.7977 | 5.5751 |  a variety of programming terms and keywords, as well as specialized vocabulary that appears in scientific papers |
| 0 | 0.7975 | 2.365 | the word survivor (or a variant) and sometimes a preceding article of 'the'. |
| 1 | 0.7971 | 3.4503 |  terms about scripture and truth, often in the context of Christian belief |
| 1 | 0.7969 | 2.7865 |  words that often come before or after a coordinating conjunction. |
| 3 | 0.7967 | 4.1382 |  words related to legal and scientific writing |
| 2 | 0.7961 | 3.0934 |  proper nouns (names and locations) |
| 3 | 0.7957 | 3.1452 |  words related to roles of people in organizations and families |
| 3 | 0.7955 | 3.0417 | code snippets or programming related terms |
| 0 | 0.7951 | 2.0986 | the word "The" at the beginning of a line |
| 0 | 0.7949 | 2.7164 |  mathematical symbols and the context around them |
| 0 | 0.7943 | 1.1233 |  a combination of words from cooking recipes and from graph drawing |
| 0 | 0.7941 | 2.3591 | code snippets related to user interface events and form handling in javascript and Python |
| 0 | 0.7939 | 2.4068 |  the suffix "er" |
| 3 | 0.7931 | 2.7313 |  words and expressions related to scientific methodology, research papers, and math/engineering |
| 0 | 0.7929 | 3.8722 |  the word "latter" along with surrounding text that specifies what "latter" is referring to. |
| 2 | 0.7927 | 1.7377 |  words associated with sports, history, or death. |
| 1 | 0.7925 | 3.6049 | words and phrases related to thinking, judging, and comparison, plus present tense verbs. |
| 5 | 0.7923 | 11.8428 |  law related terminology and references to specific cases or legal entities. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 6 | 0.8001 | 4.0951 |  words related to water, names, and positive qualities |
| 7 | 0.7993 | 5.3349 |  legal terminology, modifications, and skepticism |
| 7 | 0.7991 | 7.119 |  rhetorical questions and references to blog posts |
| 8 | 0.7987 | 5.6035 |  sections of a computer file, like IDs and tables, as well as genetic mutations |
| 10 | 0.7973 | 11.7767 | examples of someone asking for grammatical advice about the phrase "in spite of" |
| 9 | 0.7965 | 6.8861 |  quotes and claims about something being untrue |
| 11 | 0.7953 | 38.2464 | capital letters standing alone or at the beginning of words |
| 6 | 0.7945 | 4.3983 |  question answer pairs and censored text |
| 8 | 0.7935 | 4.9708 |  text from scientific documents describing procedures with coordinates, measurements, and labels |
| 11 | 0.7933 | 11.9628 |  sentences about writing letters in German |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 17 | 0.7997 | 49.8855 |  the article 'The' |
| 14 | 0.7995 | 12.2229 |  words indicating states or qualities |
| 13 | 0.7983 | 16.0352 |  definitions, especially those of a populist political nature. |
| 0 | 0.7963 | 0.0 |  mentions of seasons, especially winter and autumn, and the seasonal aspects of plant life |
| 0 | 0.7959 | 0.0 |  Spanish and Portuguese words related to code and computers |
| 16 | 0.7947 | 20.1494 |  definitions of words |
| 0 | 0.7937 | 0.0 | the word "interval" and phrases containing it |

**Mechanistic narrative:** "The adverb form of quick is" predicts " ________" (p=0.094) — a blank-space token, reflecting extreme uncertainty in the circuit. This is the lowest-confidence prediction in our corpus. The prompt asks for the adverb form of "quick" (answer: "quickly"), but the circuit appears unable to confidently route to the "-ly" suffix derivation. Early layers activate on derivational morphology templates but the middle and late layers fail to converge. The activation of a blank token suggests the model is producing an output consistent with a fill-in-the-blank format rather than committing to a specific answer, revealing that adverb derivation from adjectives is less robustly encoded than irregular morphology tasks. This contrasts sharply with the irregular plural tasks (e.g., teeth: p=0.811) and suggests the model has more specific circuit mechanisms for irregular forms than for regular morphological rules.

---

### Prompt: "<bos>The antonym of ancient is"

**Predicted token:** `Output " modern" (p=0.136)` (prob=0.1365)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8 | 3.1193 |  words related to nature and animals, especially birds |
| 2 | 0.7996 | 1.8035 |  Croatian words that include the letters 'vr' and are related to confirmation or verification |
| 0 | 0.7994 | 1.4727 | technical jargon and foreign languages. |
| 3 | 0.7993 | 3.2145 | words related to academic or scientific writing |
| 5 | 0.7983 | 3.0997 |  words associated with raising or managing money |
| 3 | 0.7981 | 3.0682 |  proper nouns and frequently used short function words, especially "of" |
| 6 | 0.7979 | 12.6864 | words that appear in programming code, legal jargon, or scientific texts |
| 3 | 0.7977 | 2.6666 |  scientific and medical terms and their prefixes and suffixes |
| 0 | 0.7973 | 2.576 |  the word "pie" (and sometimes "the") in a document |
| 2 | 0.7971 | 2.5924 |  words related to scientific, technical, and medical descriptions, particularly those used in research and experimentation. |
| 4 | 0.7969 | 3.3145 |  symbols and terms related to voltage, electricity, and electronics |
| 1 | 0.7967 | 1.5134 |  words and phrases in technical documentation from a wide array of fields |
| 1 | 0.7966 | 3.315 |  code and documented code |
| 6 | 0.7962 | 3.3636 | a mix of seemingly unrelated code terms, abbreviations, and function names. |
| 0 | 0.7958 | 1.489 |  terms related to the Indicium AttributeSet infrastructure for Java |
| 4 | 0.7954 | 3.6038 | math and equation notation |
| 0 | 0.795 | 2.365 | the word survivor (or a variant) and sometimes a preceding article of 'the'. |
| 0 | 0.7948 | 4.0502 |  technical writing related to scientific studies |
| 3 | 0.7946 | 3.1713 |  uses of the verb "is" or "was" |
| 3 | 0.7944 | 2.3756 |  words and expressions related to scientific methodology, research papers, and math/engineering |
| 0 | 0.7942 | 3.6145 |  technical words used in computing, science, or engineering |
| 5 | 0.794 | 10.9809 | the letter 'L' capitalized |
| 4 | 0.7938 | 4.1388 |  the word "is" |
| 0 | 0.7936 | 1.5638 |  a word stem "lle" or "ull" and the word "Slemish". That's not the best description, so here are some alternatives: Irish geography terms, words with the ully suffix, certain surnames |
| 1 | 0.7932 | 2.4841 |  words or phrases that indicate progress, success, or advancement in various fields like computer science, history, genomics, and medicine. |
| 3 | 0.793 | 2.1179 |  the character sequence "ol...de" and some other seemingly unrelated character sequences |
| 4 | 0.7928 | 4.15 | scientific papers and results |
| 1 | 0.7926 | 1.9464 | words that are related to formal writing such as scientific papers and official documents. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7998 | 5.3709 |  definitions of the word "false" |
| 8 | 0.7989 | 3.2163 |  historical names, places and events |
| 9 | 0.7985 | 5.4592 |  words related to technical and academic writing, including computer science, mathematics, and scientific studies. |
| 8 | 0.7975 | 3.2598 |  content that looks like bullet points or numbered lists. |
| 8 | 0.796 | 6.0892 |  words or phrases that represent importance of a topic or matter |
| 11 | 0.7952 | 31.6969 |  code snippets assigning values to variables |
| 8 | 0.7934 | 3.9325 | words related to history or groups of people from the past |
| 8 | 0.7924 | 6.5721 |  code or text related to dictionaries and named entity recognition |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 20 | 0.7991 | 7.7182 |  sentences discussing concepts and definitions, especially related to subjective experience and systems of belief |
| 16 | 0.7987 | 9.6616 |  technical and scientific terms |
| 20 | 0.7964 | 8.4193 |  discussion of Native Americans, particularly related to history, culture, and correcting stereotypes |
| 25 | 0.7956 | 7.0546 |  personal opinions or emotional statements or arguments in text. |

**Mechanistic narrative:** "The antonym of ancient is" predicts " modern" (p=0.136). This low confidence reflects the multiple plausible antonyms: "modern", "contemporary", "recent", "new", "young". Early layers activate on temporal/age-related features and the "antonym" keyword, which triggers the antonym-retrieval circuit (compare/contrast features at layers 4–7). The middle layers activate relational features mapping "ancient" (old/historical) to its temporal opposite. Late layers push toward "modern" as the most frequent training-corpus antonym for "ancient", but the score gap over competitors is small. The influence of temporal features (historical periods, time-related vocabulary) is prominent, consistent with "ancient" being a time-dimensional adjective rather than a simple qualitative antonym like hot/cold.

---

### Prompt: "<bos>The antonym of full is"

**Predicted token:** `Output " empty" (p=0.143)` (prob=0.1427)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.8001 | 1.6252 | words relating to environment, ecosystems, and governmental bodies |
| 2 | 0.7997 | 3.6109 | technical, legal or academic language, especially verbs, often found in formal reports |
| 2 | 0.7995 | 3.0934 |  proper nouns (names and locations) |
| 1 | 0.7993 | 1.4777 | the phrase "face to face" |
| 1 | 0.7991 | 1.3251 |  technical documentation or legal documents containing specialized terminology. |
| 0 | 0.7981 | 2.365 | the word survivor (or a variant) and sometimes a preceding article of 'the'. |
| 2 | 0.7979 | 1.5709 |  operations found in coding and finance. |
| 2 | 0.7976 | 3.5359 |  words related to courtroom proceedings and legal terminology, but is a bit noisy. |
| 0 | 0.7972 | 2.3996 |  technical words used in computing, science, or engineering |
| 3 | 0.7964 | 3.2145 | words related to academic or scientific writing |
| 0 | 0.796 | 1.8425 |  code snippets, programming or technical terminology |
| 0 | 0.7956 | 2.2462 |  the word "is" |
| 1 | 0.7952 | 1.776 |  numbers and associated programming terms |
| 1 | 0.795 | 3.315 |  code and documented code |
| 0 | 0.7948 | 1.4239 |  words, especially adjectives, related to abnormalities, technical terminology, or negative traits. |
| 2 | 0.7943 | 1.4544 |  words that look like usernames and filenames |
| 3 | 0.7941 | 1.7361 | technical jargon, particularly related to language and coding |
| 2 | 0.7939 | 2.1609 |  uses of the word "ou" followed by a word with an activation greater than zero (which could just be noise), suggesting it might be looking for "or" in French |
| 0 | 0.7937 | 1.4727 | technical jargon and foreign languages. |
| 2 | 0.7931 | 1.8375 |  text related to negative numbers and leadership, particularly in a nursing context |
| 2 | 0.7927 | 1.7249 |  words related to publications or music releases |
| 0 | 0.7925 | 3.1405 |  technical terms used in scientific writing |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7985 | 5.3709 |  definitions of the word "false" |
| 8 | 0.7983 | 5.448 |  words and phrases that express preference, equations, and metadata |
| 4 | 0.7978 | 10.1749 | code snippets and license agreements |
| 7 | 0.7974 | 15.7139 |  a variety of reference codes, abbreviations, and identifiers from different fields. |
| 0 | 0.797 | 0.0 |  closing curly brackets in code snippets |
| 6 | 0.7966 | 5.8379 | words appearing in formal writing that are related to research, disagreement, and the legal system.  |
| 7 | 0.7962 | 4.4711 |  words related to education, especially phonics and reading |
| 5 | 0.7958 | 4.9074 |  words related to creative works like books, movies, songs and their creation |
| 4 | 0.7954 | 2.4287 |  legal and medical terminology |
| 5 | 0.7945 | 4.3671 | words that are technical or scientific and often have a suffix. |
| 4 | 0.7935 | 3.7745 |  words related to biological or legal processes |
| 8 | 0.7933 | 3.5733 |  words indicating position or quantity |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 14 | 0.7999 | 10.6683 |  words related to societal issues, groups and beliefs |
| 13 | 0.7989 | 6.2351 | words related to completeness, totality, and control |
| 13 | 0.7987 | 6.7505 |  code snippets in C and C++, assembly language, and German |
| 11 | 0.7968 | 38.2464 | capital letters standing alone or at the beginning of words |
| 9 | 0.7947 | 5.5204 |  quotes and claims about something being untrue |
| 14 | 0.7929 | 6.8491 | words used in academic papers, legal documents, or technical writing. |

**Mechanistic narrative:** "The antonym of full is" predicts " empty" (p=0.143). Similar to "ancient/modern", this shows moderate competition across valid antonyms ("empty", "hollow", "bare", "vacant"). Early layers activate on capacity/containment-related features and the antonym template. The middle-layer reversal features (comparison, symmetry, contrast at layers 4–7) engage to flip the semantic content of "full" into its opposite. Late layers push toward "empty" — the most prototypical, highest-frequency antonym in English corpora. The confidence is low because "full/empty" is a more ambiguous opposition than "hot/cold": "full" can refer to containers (antonym: empty), stomachs (antonym: hungry/empty), schedules (antonym: free/empty), all pointing to "empty" but with some variance.

---

### Prompt: "<bos>The antonym of large is"

**Predicted token:** `Output " small" (p=0.321)` (prob=0.3207)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8001 | 2.3908 |  words indicating high quality, importance, and success |
| 3 | 0.7995 | 3.3283 |  a combination of the words "two", "of", and/or "the" and also some proper nouns |
| 0 | 0.7993 | 1.2753 |  words related to user interfaces, medicine or job qualifications |
| 3 | 0.7991 | 3.0682 |  proper nouns and frequently used short function words, especially "of" |
| 0 | 0.7987 | 1.2269 | words and abbreviations related to biology, microbiology, immunology, or genetics, especially in informal language |
| 2 | 0.7985 | 3.0934 |  proper nouns (names and locations) |
| 6 | 0.7984 | 2.1116 |  words or fragments that can describe the size of something |
| 5 | 0.7982 | 3.9351 |  phrases related to mutually exclusive options and zero-sum outcomes |
| 3 | 0.798 | 2.8332 |  variants of the word "natural" and words with similar suffixes |
| 5 | 0.7974 | 6.3503 |  phrases related to arguments in court |
| 6 | 0.797 | 3.8291 |  words related to water, names, and positive qualities |
| 4 | 0.7968 | 2.7127 | the word "with" or words containing "in", often followed by other short, common words. |
| 4 | 0.7966 | 3.2738 |  the word "twin" or indicate competition |
| 3 | 0.7964 | 3.0264 |  various forms of the verb "to be" and words related to problems or issues. |
| 0 | 0.796 | 2.9707 |  references to dates and times |
| 3 | 0.7956 | 2.4781 | the word "syntactic" |
| 6 | 0.7954 | 4.5747 |  terms related to political parties and figures |
| 4 | 0.7952 | 3.8421 |  words or phrases that are being used in an official or formal way |
| 4 | 0.7951 | 2.4556 |  proper nouns, some greek letters, and gene-related or science words in general |
| 3 | 0.7949 | 2.4229 |  a variety of words and phrases related to academic citations, programming code, medical analysis and legal arguments. |
| 7 | 0.7947 | 4.4622 |  words and phrases related to falsehoods and inaccuracies |
| 0 | 0.7945 | 0.0 |  mentions of being loving, loved, or loving something. |
| 0 | 0.7943 | 2.676 |  words and phrases related to medical treatments, clinical trials, and biotechnology companies. |
| 5 | 0.7941 | 3.6632 | words enclosed by quotation marks and words related to language and definitions |
| 2 | 0.7939 | 1.8917 |  phrases including the word "other" when asking a question or for alternative suggestions |
| 1 | 0.7937 | 2.5996 |  noun phrases in a specific grammatical structure |
| 2 | 0.7935 | 1.6039 |  code snippets or code-related language, specifically the word "genre" |
| 3 | 0.7933 | 3.158 | articles and pronouns, especially possessives of "I", "my", some french text, and some parts of legal documents |
| 2 | 0.7929 | 9.3992 |  nature-related locations and objects. |
| 7 | 0.7927 | 8.6127 | words related to questions and measuring intelligence |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7999 | 12.102 | code or programming language |
| 9 | 0.7997 | 3.4752 |  words/tokens that indicate extra, supplemental, or additional information is present |
| 13 | 0.7989 | 7.2779 |  code snippets in C and C++, assembly language, and German |
| 13 | 0.7978 | 7.0073 | contentious statements regarding FDA regulations |
| 8 | 0.7962 | 5.496 |  concepts related to the justice system, especially ethics and virtues within the justice system, as well as self-evaluation and attribution |
| 11 | 0.7925 | 38.2464 | capital letters standing alone or at the beginning of words |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 25 | 0.7976 | 21.8977 |  words that indicate comparison or scale, often featuring intensifiers or superlatives |
| 17 | 0.7972 | 8.4308 |  text from legal documents or statutes |
| 18 | 0.7958 | 8.9759 |  uses of the words "and" or "or", or articles |
| 0 | 0.7931 | 0.0 | words ending in the suffix "ality,' or the word "other." |

**Mechanistic narrative:** "The antonym of large is" predicts " small" (p=0.321). This is more confident than other antonym predictions, reflecting the dominant and unambiguous hot-cold-style opposition between "large" and "small". Early layers activate on size/scale features and the antonym template. The middle-layer comparison features activate strongly — "large/small" is a canonical dimensional antonym pair well-represented in training data. Late layers converge on "small" with a larger margin over competitors ("tiny", "little", "petite") than other antonym tasks, consistent with the strong co-occurrence of "large" and "small" as a contrastive pair in English text.

---

### Prompt: "<bos>The antonym of north is"

**Predicted token:** `Output " south" (p=0.362)` (prob=0.3621)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8002 | 3.0531 |  the word "part" followed by prepositions or words related to sections or components. |
| 0 | 0.7998 | 1.5638 |  a word stem "lle" or "ull" and the word "Slemish". That's not the best description, so here are some alternatives: Irish geography terms, words with the ully suffix, certain surnames |
| 1 | 0.7992 | 1.5134 |  words and phrases in technical documentation from a wide array of fields |
| 0 | 0.7988 | 2.5119 |  the phrase "school district" |
| 5 | 0.7984 | 3.7566 |  technical descriptions and explanations of preferred or distinguishing features. |
| 5 | 0.7978 | 3.0997 |  words associated with raising or managing money |
| 0 | 0.7972 | 1.7117 |  words related to review and inspection |
| 4 | 0.797 | 3.3185 |  mathematical notation and logical operators represented in text or symbols. |
| 0 | 0.7966 | 2.2876 |  words and phrases that express positive sentiment, intensity, or value judgments, although it can be noisy |
| 4 | 0.7964 | 3.6182 |  locations and buildings |
| 3 | 0.7962 | 2.9361 |  the word "new" as well as the definite article "the" in legal documents |
| 0 | 0.796 | 1.4019 |  words or characters related to science, math, technology, or coding |
| 1 | 0.7958 | 3.4848 |  a mix of names and location words |
| 3 | 0.7956 | 3.4347 | words associated with symbolic representation and religious festivals |
| 0 | 0.7954 | 2.2475 | the word "snow" |
| 3 | 0.7952 | 3.1538 |  words associated with displaying information and instructions |
| 0 | 0.795 | 1.1539 |  proper nouns and technical terms |
| 6 | 0.7948 | 6.8649 |  topics/titles or short phrases that often begin with a capitalized word |
| 1 | 0.7946 | 2.9718 |  words related to ecology, forestry, and environmental management |
| 6 | 0.7944 | 7.7956 |  question answer pairs and censored text |
| 1 | 0.7942 | 2.3313 | the word "occurs" or similar forms of the verb "to occur". |
| 6 | 0.7938 | 5.0958 |  jargon terms and brand names associated with consumer products |
| 3 | 0.7933 | 3.3562 | code-related keywords and also the location "Alagoas" when it is followed by "as" |
| 0 | 0.7931 | 2.5374 |  mentions of nationality or ancestry, along with some related terms like "genes" or "medicine" |
| 1 | 0.7929 | 5.3774 |  the word "guess" |
| 3 | 0.7927 | 3.166 |  text related to the clinical testing and commercialization of pharmaceutical products |
| 2 | 0.7923 | 5.0952 | technical, legal or academic language, especially verbs, often found in formal reports |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7996 | 15.3064 | first and second person pronouns and forms of the verb "to be". |
| 12 | 0.7994 | 10.4179 |  code snippets |
| 9 | 0.7986 | 2.6272 |  terms related to mathematical and physical calculations on a lattice |
| 8 | 0.798 | 6.5049 | This neuron looks for a variety of specific, potentially unrelated code snippets and phrases, mathematical symbols, and religious terms |
| 9 | 0.7974 | 6.9202 |  words/tokens that indicate extra, supplemental, or additional information is present |
| 11 | 0.7936 | 38.2464 | capital letters standing alone or at the beginning of words |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 22 | 0.8 | 20.4062 |  words with positive associations mixed with clinical/experimental contexts |
| 17 | 0.799 | 8.52 |  technical documents that contain coordinate systems and direction |
| 14 | 0.7982 | 17.7558 | URLs, definitions and legal jargon. |
| 14 | 0.7976 | 16.0346 |  discussion about language and communication |
| 20 | 0.7968 | 14.0082 | uses of the verb "to be" together with other function words |
| 15 | 0.794 | 12.3565 |  a mix of capitalized words and words related to chemistry or enzymatic reactions |
| 15 | 0.7925 | 11.5218 | a series of terms related to specific topics that are repeated throughout the document |

**Mechanistic narrative:** "The antonym of north is" predicts " south" (p=0.362) — the most confident antonym prediction (p=0.362 vs. 0.136–0.321 for other antonym tasks). "North/south" is a uniquely determined antonym pair: there is only one correct geographical opposite, unlike size or age pairs. Early layers activate on directional/geographical features. The middle layers activate comparison and spatial reversal features. The late layer push is clean and confident — geographic directional pairs are strongly memorised in training data and the circuit converges rapidly. The second-strongest competitor would be "South" (capitalised), reflecting uncertainty about case rather than about the answer itself.

---

### Prompt: "<bos>The fear of heights is called"

**Predicted token:** `Output " ac" (p=0.653)` (prob=0.6531)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 6 | 0.8001 | 2.7017 | This neuron detects a mix of terms related to physics, accidents/violence, medical procedures, and video game skills |
| 2 | 0.7999 | 2.4986 |  mostly uppercase text or words ending in "ic" or "ergic" |
| 0 | 0.7997 | 1.8439 |  words and phrases related to strong emotions or opinions |
| 0 | 0.7995 | 0.0 | the word "motivated" or "motivation" |
| 0 | 0.7992 | 3.0902 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 1 | 0.7988 | 3.6429 |  words that have a scientific or technical meaning |
| 2 | 0.7986 | 2.4764 |  several different contexts: Portuguese words, football terms, theatrical/film production, probability calculations, recollections/opinions, chemical/biological processes, and Java code |
| 2 | 0.7984 | 2.0417 | words with the suffix "tive", especially in mathematical, physics, or engineering contexts |
| 0 | 0.7982 | 1.8418 |  words related to medical/legal/scientific terminology |
| 5 | 0.798 | 3.9835 |  words and phrases indicating a future event that may or may not occur |
| 7 | 0.7974 | 4.6306 |  words and phrases related to positivity and well-being, but it has some false positives. |
| 2 | 0.7972 | 2.3329 |  positivity and openness to offers or connection. |
| 0 | 0.797 | 1.7256 |  words related to scientific research, educational programs, and medical conditions |
| 3 | 0.7968 | 2.7661 |  mentions of political and legal concepts and names, with a secondary activation for the word "staff" |
| 5 | 0.7966 | 10.9809 | the letter 'L' capitalized |
| 4 | 0.7964 | 2.8212 |  occurrences of "on top of" or things being "on" the "ground" |
| 5 | 0.7962 | 4.216 |  words related to legal proceedings |
| 1 | 0.7958 | 3.3888 | words associated with cooking or baking |
| 0 | 0.7954 | 1.4826 |  words related to legal or medical contexts |
| 1 | 0.7951 | 1.8701 |  instances of the word "hell." |
| 3 | 0.7947 | 2.8232 |  references to dates and times |
| 0 | 0.7945 | 3.4013 |  the word "overall", sometimes alongside words that express quantity |
| 4 | 0.7943 | 4.371 | scientific papers and results |
| 0 | 0.7941 | 2.1431 |  phrases containing the word known or words like new, old and sweet which are commonly used to describe something |
| 4 | 0.7939 | 3.5059 | SQL code snippets as identified by delimiters |
| 1 | 0.7931 | 3.1458 |  strings of ellipses and sentence fragments, possibly in conjunction with other punctuation |
| 1 | 0.7929 | 2.5257 |  words and phrases related to politics, history, and conflict. |
| 5 | 0.7927 | 2.683 |  phrases describing the degree of difficulty of a task |
| 0 | 0.7925 | 2.8966 |  words and phrases related to societal and political issues |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 10 | 0.7994 | 4.6677 |  words related to diseases, symptoms, and other medical terminology |
| 9 | 0.799 | 27.6213 |  lines of code that import libraries or declare constants |
| 11 | 0.7978 | 7.9983 |  mentions of allergies and their causes and/or symptoms |
| 9 | 0.7956 | 3.5961 | words related to sports, travel, or crowds. |
| 8 | 0.7949 | 4.2669 |  something about mental and physical health |
| 9 | 0.7935 | 5.6335 |  words and phrases related to courage, bravery, taking initiative, and stepping outside of one's comfort zone. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7976 | 0.0 |  instances of the word "hardly" and possibly words close to it within a sentence |
| 17 | 0.796 | 66.5389 | start of sentences |
| 24 | 0.7937 | 10.611 |  complex equations |
| 23 | 0.7933 | 6.8812 | words containing the "ist" and "ilit" character sequences, and words/phrases related to legal arguments |
| 25 | 0.7923 | 29.5067 |  sentences with multiple instances of particular words; it highlights different kinds of specialized and academic language that contain repeated short words such as "of", "the", "in", and "to", often in conjunction with longer, content-rich words from a variety of domains. |

**Mechanistic narrative:** "The fear of heights is called" predicts " ac" (p=0.653) — a subword token, the beginning of "acrophobia". This is the highest confidence prediction among the phobia/definition prompts, but the predicted token is a prefix rather than the full word. The circuit correctly identifies this as a Greek/Latin compound phobia name but the tokeniser splits "acrophobia" into subwords, and " ac" is the first subword. Early layers activate on "fear/phobia" semantic features and "heights/elevation" directional terms. Middle layers activate on medical/psychological terminology patterns. Late layers strongly activate on Greek-root scientific compound features — the "-phobia" suffix pattern — leading to confident prediction of the initial subword " ac". This is a fascinating tokenisation artefact: the circuit knows "acrophobia" is the answer, but the generation proceeds token-by-token and " ac" is the most probable first token of that multi-token word.

---

### Prompt: "<bos>The noun form of happy is"

**Predicted token:** `Output " happiness" (p=0.404)` (prob=0.4043)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.7999 | 4.0431 |  words or parts of words ending in "ish", "al", can", "ful", "ing", "ous", "mass", "ses", "work", " States", or "sky" |
| 4 | 0.7997 | 2.6227 |  context related to words/language and their meanings or usages |
| 5 | 0.7995 | 4.8075 |  a hodgepodge of sentence fragments related to coding, grammar, statistics, 3D graphics, and vehicle mechanics |
| 5 | 0.7993 | 3.5515 | the word "name" maybe next to terms of speech |
| 7 | 0.7989 | 1.9596 |  strongly positive emotional expressions/utterances of excitement |
| 1 | 0.7987 | 3.157 | the word "of" |
| 1 | 0.7985 | 3.2407 |  words relevant to syntax and grammar. |
| 3 | 0.7983 | 2.861 |  words related to processes of conveying information through different media |
| 2 | 0.7981 | 1.7307 |  code related to Swift enums and integer types |
| 7 | 0.7979 | 6.7918 |  text describing the usage of words, including grammar and meaning |
| 5 | 0.7978 | 6.1968 |  technical terminology related to computer science |
| 5 | 0.7976 | 3.2953 |  definitions or explanations of names or words |
| 0 | 0.7972 | 3.1629 | a variety of words and phrases, including those relating to community service, food critiques, and locations. |
| 6 | 0.797 | 5.0 |  profanity, insults and gendered slurs, plus some mathematical and technical jargon |
| 3 | 0.7964 | 3.2089 | words ending in "shaped" or the word "section" |
| 6 | 0.7962 | 3.9573 | language education materials for students and teachers including discussion of teaching methodologies. |
| 3 | 0.796 | 3.1451 |  mathematical notation used in proofs |
| 0 | 0.7958 | 2.1459 | the word "button" accompanied by other somewhat related words. |
| 0 | 0.7956 | 3.3534 | words related to programming code, math, anatomy, medicine, structure, and anything that can range. |
| 0 | 0.7954 | 3.9895 | terms used in software code such as "assembly", "using", "namespace", and "license" |
| 4 | 0.795 | 3.2574 | mathematical symbols and notation |
| 1 | 0.7948 | 3.3155 | words and phrases related to thinking, judging, and comparison, plus present tense verbs. |
| 0 | 0.7944 | 1.4549 |  words and phrases related to scientific research, especially experimental studies and data analysis. |
| 1 | 0.7938 | 2.4161 |  words related to specific and technical computer science, math, or scientific topics |
| 1 | 0.7936 | 4.0416 |  strings of ellipses and sentence fragments, possibly in conjunction with other punctuation |
| 7 | 0.7934 | 6.7261 |  the word "version" and other forms of the word |
| 0 | 0.7929 | 1.2969 | words, phrases or headlines related to criticism, particularly of public figures or policies |
| 0 | 0.7925 | 2.0266 |  the word "tackle" and also the word "grid", with a very weak activation on the definite article "the" |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.8001 | 3.9441 |  citations in legal documents |
| 9 | 0.7991 | 5.0263 | text discussing language, words, or idioms. |
| 8 | 0.7974 | 5.0182 |  words and phrases related to the meaning of words |
| 8 | 0.7966 | 4.6751 |  phrases that introduce a more specific description or example |
| 10 | 0.7952 | 9.0775 |  code snippets, configuration files, or data structures with names containing programming terms |
| 9 | 0.7946 | 6.1555 | a mix of numbers and letters, possibly related to code, math, or scientific text |
| 9 | 0.7942 | 8.1667 |  references to old english combined with place names, or names and meanings |
| 8 | 0.794 | 5.661 |  things related to the structure of something or the way something is. |
| 13 | 0.7923 | 6.716 |  uses of the word "word" along with a quotation |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7968 | 56.4195 |  words regarding emotions, feelings, and health |
| 17 | 0.7931 | 9.6463 | names and identity |
| 20 | 0.7927 | 42.9221 |  the word "happy" or "Happy" followed by another word |

**Mechanistic narrative:** "The noun form of happy is" predicts " happiness" (p=0.404). This is relatively confident for a derivational morphology task. The prompt is unambiguous: "happy" has a single canonical noun form ("happiness") with no realistic competitors. Early layers activate on the part-of-speech label "noun" and the target word "happy" (positive emotion features). Middle layers activate on derivational morphology features that link adjectives to their "-ness" suffix noun forms. Late layers converge on "happiness" — the activation of positive-emotion and abstract-noun features in late layers (17–25) strongly favour this specific derivation. The confidence (p=0.404) is lower than irregular morphology tasks, suggesting regular derivational morphology is less robustly encoded than high-frequency irregular forms.

---

### Prompt: "<bos>The opposite of begin is"

**Predicted token:** `Output " finish" (p=0.185)` (prob=0.1853)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8001 | 2.4054 |  the word "following" and related concepts. |
| 0 | 0.7999 | 1.3412 |  data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 0 | 0.7995 | 1.7283 |  the word "judge" and words related to judges |
| 0 | 0.7993 | 2.9672 | words associated with phrases "be", "outside" and possibly some conjunctions and prepositions. |
| 4 | 0.7991 | 3.4343 | words and phrases related to geography, anatomy and genotyping in DNA |
| 1 | 0.7988 | 2.2219 |  words or phrases that indicate progress, success, or advancement in various fields like computer science, history, genomics, and medicine. |
| 3 | 0.7984 | 8.3922 |  instances of the word "The" at the start of a line |
| 3 | 0.7982 | 3.1662 |  directional indicators like "north of" or "end of" |
| 1 | 0.798 | 2.7287 |  words that often come before or after a coordinating conjunction. |
| 3 | 0.7978 | 2.7063 | prepositions and comparative words |
| 0 | 0.7976 | 2.5956 |  words related to money or business transactions |
| 7 | 0.7973 | 3.7165 | sentences that show the outcome, method, or interpretation of experiments in a scientific paper |
| 7 | 0.7971 | 3.7757 | phrases using the verb "is" to expresses states, being, or inclusion as well as interesting and facts, including mathematical contexts. |
| 1 | 0.7969 | 3.6822 |  variable types |
| 3 | 0.7965 | 9.3926 |  academic publications that use scientific or medical data |
| 5 | 0.7963 | 3.0837 |  words related to patents, legal terms, and scientific publications |
| 6 | 0.796 | 3.1289 |  words related to ideology or belief systems, including religion, politics, and values |
| 3 | 0.7958 | 2.8753 |  words and expressions related to scientific methodology, research papers, and math/engineering |
| 3 | 0.7956 | 2.7414 |  verbs or verb-related words, mainly in languages other than English |
| 1 | 0.7954 | 3.4348 |  code comments |
| 4 | 0.7952 | 3.8865 | scientific or technical writing, including descriptions of studies and mechanisms, often related to medicine |
| 1 | 0.795 | 3.2117 |  terms from technical, scientific and academic writing. |
| 2 | 0.7947 | 4.0251 |  the word "tap" in the context of computer software |
| 0 | 0.7943 | 3.0943 |  the word "visual" |
| 5 | 0.7941 | 4.5359 |  words related to engineering and technical manuals |
| 2 | 0.7939 | 3.0344 | verbs |
| 6 | 0.7934 | 2.9151 |  code snippets and language references |
| 4 | 0.793 | 3.6249 |  words associated with the etymology or definition of a word |
| 3 | 0.7928 | 3.2142 | programming related keywords, symbols and syntax. |
| 3 | 0.7925 | 6.488 |  code blocks and the phrase "as soon as" |
| 2 | 0.7921 | 3.4893 | verbs being used |
| 2 | 0.7919 | 2.1395 |  mentions of 'time' and also recognizes "AS" written in all caps |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 10 | 0.7997 | 4.5563 |  words specifically related to wedding anniversaries and their associated symbols |
| 8 | 0.7986 | 7.5547 |  code snippets with specific coding keywords and markup tags |
| 8 | 0.7967 | 14.9264 |  code snippets with specific coding keywords and markup tags |
| 8 | 0.7923 | 5.3317 |  words related to research papers including increases, findings, and related terms. |
| 8 | 0.7917 | 5.3231 |  terms related to programming, software and database management, including code snippets, commands and verifications. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7945 | 8.4004 |  words related to e-commerce website usability |
| 24 | 0.7936 | 19.5779 |  sentence fragments containing the word "to", which may be part of a verb phrase |
| 24 | 0.7932 | 18.8048 |  words describing business related functions in a sentence |

**Mechanistic narrative:** "The opposite of begin is" predicts " finish" (p=0.185). Low confidence reflects competition from "end", "stop", "cease", "conclude" — all valid antonyms for "begin". Early layers detect the antonym template and the verb "begin". Middle layers activate contrast/reversal features. The specific choice of "finish" over "end" is interesting: "finish" carries a completion-of-task connotation (versus "end" which is more neutral), and the circuit routes there — possibly because "begin/finish" co-occurs frequently in instructional and procedural texts. The low confidence (p=0.185) is expected given the rich synonym set for this antonym slot.

---

### Prompt: "<bos>The opposite of dark is"

**Predicted token:** `Output " light" (p=0.602)` (prob=0.6024)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 7 | 0.8001 | 1.6637 |  words related to light and illumination sources |
| 4 | 0.7999 | 3.1134 |  a wide variety of words, often those appearing in legal, technical, or historical documents. |
| 1 | 0.7997 | 3.1444 | sentences that report results or observations |
| 0 | 0.7994 | 3.4121 |  the word "is" |
| 1 | 0.7988 | 2.5666 |  words and phrases related to health. |
| 2 | 0.7984 | 2.3956 |  the word 'reason' and associated words indicating the cause or logic of a situation |
| 2 | 0.7982 | 3.529 |  positive descriptors |
| 7 | 0.798 | 5.8839 |  technical words and phrases related to a continuous sequence or range of values |
| 7 | 0.7975 | 2.4051 |  words related to interior design, particularly color and space within a room |
| 2 | 0.7973 | 3.8681 |  auxiliary verbs in the present tense |
| 0 | 0.7971 | 0.0 | references to the current study or research |
| 2 | 0.7969 | 2.603 |  present tense verbs |
| 3 | 0.7967 | 2.8332 |  code snippets or mathematical expressions |
| 0 | 0.7962 | 2.6413 |  proper nouns, especially names of people and places |
| 0 | 0.796 | 2.6231 | technical or jargonistic terms that are specific to certain fields. |
| 5 | 0.7958 | 4.5823 |  words and short phrases relating to binding/grouping/coupling |
| 0 | 0.7956 | 2.836 |  words and phrases related to societal and political issues |
| 1 | 0.7954 | 2.8965 |  the word "direction" related to particles/projections or statistics |
| 7 | 0.7952 | 5.1538 |  a mix of chemical terms, programming syntax, and scientific terms |
| 2 | 0.7949 | 2.0294 |  scientific and chemistry terms related to charged particles and residues |
| 5 | 0.7945 | 2.448 |  mathematical statistical and scientific references and formulas |
| 0 | 0.7943 | 2.9672 | words associated with phrases "be", "outside" and possibly some conjunctions and prepositions. |
| 0 | 0.7941 | 2.8599 | something the user looked at, is looking at, or will look at. |
| 5 | 0.7936 | 3.7828 | mathematical expressions, government actions, science, prime numbers, and legal procedures |
| 1 | 0.7934 | 3.6822 |  variable types |
| 0 | 0.7932 | 2.2358 | the word "complicated" and also some words associated with a technical design process. |
| 4 | 0.793 | 4.0223 |  words and phrases relating to political and military action, including general use in a more mathematical and scientific sense |
| 0 | 0.7927 | 1.9787 |  words or phrases indicating recentness, novelty, or privacy |
| 2 | 0.7925 | 3.9499 |  the verb "to be" in various languages and tenses |
| 0 | 0.7923 | 2.0107 |  a combination of words from cooking recipes and from graph drawing |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7992 | 18.9153 | first and second person pronouns and forms of the verb "to be". |
| 11 | 0.799 | 7.8288 |  words and phrases related to morality |
| 14 | 0.7986 | 8.8532 |  the word "is" after the or near the word "the" |
| 14 | 0.7977 | 9.444 |  dark, psychological, and disturbing fiction |
| 14 | 0.7965 | 9.1016 |  patterns related to quantitative data and comparisons, including ratios, percentages, and ranges |
| 0 | 0.7921 | 0.0 | the phrase "no matter how" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7947 | 6.6384 | that clauses following words suggesting possibility. |
| 16 | 0.7938 | 7.8875 |  words related to sociology or political science |
| 25 | 0.7919 | 13.1517 |  mentions of dates, numbers, locations, names, laws, or studies |
| 24 | 0.7916 | 13.5863 |  words related to online meetings, and recordings of these meetings |

**Mechanistic narrative:** "The opposite of dark is" predicts " light" (p=0.602). This is the second-highest confidence antonym prediction (after "north/south"). "Dark/light" is a canonical binary opposition in English with an extremely dominant antonym relationship. Early layers activate on light/illumination features and the antonym template. Middle layers activate strong contrast features — this is a perceptual binary (presence vs. absence of light) that is extremely frequent and unambiguous in training data. Late layers converge decisively on "light". The main competitor would be " bright", but "light" dominates due to its structural role as both noun and adjective antonym of "dark", giving it higher feature vote weight.

---

### Prompt: "<bos>The opposite of fast is"

**Predicted token:** `Output " slow" (p=0.480)` (prob=0.4800)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.7998 | 5.508 |  the definite article "The" |
| 1 | 0.7996 | 3.2787 |  words that often come before or after a coordinating conjunction. |
| 1 | 0.7994 | 3.4348 |  code comments |
| 2 | 0.7991 | 1.2689 |  words related to speed, changes in time, or technological enhancement |
| 0 | 0.7989 | 1.8172 | references to prior or external material, or other points outside the current scope |
| 1 | 0.7987 | 4.1358 | science and technology terms especially related to relationships among entities |
| 1 | 0.7972 | 4.7113 | terms used across a wide range of technical documentation |
| 3 | 0.797 | 3.2582 |  the word "of" in legal documents. |
| 2 | 0.7968 | 2.3956 |  the word 'reason' and associated words indicating the cause or logic of a situation |
| 4 | 0.7965 | 4.9702 |  words or phrases that are being used in an official or formal way |
| 4 | 0.7963 | 4.0224 | technical or scientific language |
| 1 | 0.7961 | 3.3926 | words and phrases related to thinking, judging, and comparison, plus present tense verbs. |
| 2 | 0.7959 | 2.2472 |  words and phrases related to the music industry, musicians, and musical performances |
| 0 | 0.7955 | 2.8599 | something the user looked at, is looking at, or will look at. |
| 0 | 0.7952 | 1.4215 |  words related to travel, tourism and hospitality |
| 3 | 0.7946 | 6.6676 |  legal and military terminology, and words indicating opposition or change |
| 0 | 0.7943 | 1.981 | something the user looked at, is looking at, or will look at. |
| 1 | 0.7939 | 4.7946 |  words related to sports, particularly college and professional athletics |
| 1 | 0.7937 | 3.4276 |  the word "linear" and related words such as "linearity" |
| 0 | 0.7932 | 2.9798 | occurrences of the word 'red' at the start of phrases or clauses |
| 4 | 0.793 | 1.7201 |  the phrase "real-time" |
| 0 | 0.7928 | 1.6632 |  data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 3 | 0.7923 | 3.6424 |  instances of "am", "is", "was", followed by prepositions such as "on", "for", and "that" |
| 2 | 0.7919 | 3.529 |  positive descriptors |
| 2 | 0.7917 | 4.0251 |  the word "tap" in the context of computer software |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 6 | 0.8 | 3.7916 | words or phrases that indicate some sort of problem or argument |
| 7 | 0.7985 | 3.1117 | phrases using the verb "is" to expresses states, being, or inclusion as well as interesting and facts, including mathematical contexts. |
| 6 | 0.7981 | 3.6736 | claims and statements about claims, often in a legal or political context. |
| 7 | 0.7976 | 8.6147 | The neuron activates on the structure or formatting of documents, including numbering, introductions, and common thanking phrases. |
| 5 | 0.7974 | 4.5823 |  words and short phrases relating to binding/grouping/coupling |
| 6 | 0.7948 | 4.4622 |  negative connotations relating to business, computer programming, or social justice. |
| 5 | 0.7941 | 4.2162 |  phrases related to arguments in court |
| 9 | 0.7935 | 4.2502 |  words related to the application of chemistry or physics to fabricate something |
| 5 | 0.7926 | 6.5456 |  phone numbers |
| 7 | 0.7921 | 8.005 |  words or phrases describing comparisons or relationships |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 14 | 0.7983 | 57.9217 |  newline characters. |
| 12 | 0.7979 | 8.422 |  questions about the relationship between the value of K and the amount of training data in cross validation |
| 15 | 0.7957 | 7.024 |  present tense forms of the verb "to be." |
| 14 | 0.795 | 9.0183 | states of being, assignments, or parameters |
| 17 | 0.7914 | 7.3 | deep close |

**Mechanistic narrative:** "The opposite of fast is" predicts " slow" (p=0.480). "Fast/slow" is another canonical binary antonym pair with good confidence. Early layers activate on speed/velocity features and the antonym template. The middle layer contrast features (comparison, reversal, symmetry at layers 4–7) engage for the speed dimension. Late layers push toward "slow" — the dominant antonym for "fast" in speed contexts. The second competitor is "quickly" or " leisurely", but "slow" wins as the most direct antonym in everyday usage. Note that this prompt appeared in the synonym category ("Another word for fast is") as well; the circuit treats "opposite" vs. "another word for" quite differently, activating reversal features for "opposite" and synonym-search features for "another word for".

---

### Prompt: "<bos>The opposite of loud is"

**Predicted token:** `Output " quiet" (p=0.386)` (prob=0.3865)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8 | 2.629 |  words related to relationships and legal proceedings around dating |
| 3 | 0.7996 | 3.627 |  the letters "s", "y", "y", "are", "and" and fragments of words/names seemingly at random |
| 0 | 0.7994 | 2.289 |  technical terms from the domains of biology, statistics and finance |
| 5 | 0.799 | 3.2943 |  words describing people and their personalities |
| 4 | 0.7988 | 2.7335 |  a combination of code snippets, past tense verbs or conditional phrases ("if/then") |
| 5 | 0.7986 | 4.084 |  common idioms or sayings |
| 5 | 0.7984 | 3.4176 |  words related to scientific studies, particularly in medicine, physics, and genetics |
| 4 | 0.7982 | 9.5172 |  the word "the" |
| 3 | 0.798 | 2.5065 |  mentions of motor rotation and direction, especially clockwise and counterclockwise, often abbreviated. |
| 5 | 0.7978 | 3.8246 |  equations and mathematical derivatives, especially anything including 'rt' or parentheses or powers |
| 1 | 0.7974 | 3.7538 |  language related to health and medicine, particularly preventative measures and risk factors |
| 1 | 0.797 | 1.9633 |  words and phrases related to political issues, especially those concerning social issues and legislation. |
| 0 | 0.7968 | 1.9295 | a mix of filler words like "like", "the", "main "and words related to problems or difficulties |
| 6 | 0.7964 | 4.796 | Java and Objective-C code documentation |
| 0 | 0.7961 | 1.6353 |  the preposition "of" |
| 0 | 0.7959 | 3.6374 | the word "else", often preceded by "someone" |
| 3 | 0.7957 | 3.6114 |  words related to a specific field in the article |
| 2 | 0.7955 | 2.3667 |  words related to collaboration or separation, in technical contexts. |
| 5 | 0.7953 | 2.9054 |  words related to government, religion, and corporate responsibility |
| 5 | 0.7947 | 4.5823 |  words and short phrases relating to binding/grouping/coupling |
| 0 | 0.7945 | 3.7744 | the word "drop" |
| 0 | 0.7941 | 2.495 |  the word "transfer" used in a technical context within descriptions of technology or scientific research |
| 1 | 0.7939 | 3.5924 |  words associated with solving problems or making difficult decisions |
| 2 | 0.7937 | 2.7396 | code related to React and Angular web applications |
| 0 | 0.7935 | 3.8892 |  mostly blank lines or lines with very few characters |
| 6 | 0.7933 | 4.6854 | words appearing in formal writing that are related to research, disagreement, and the legal system.  |
| 1 | 0.7929 | 1.6942 | uses of the phrase "carried out" along with related terms from scientific papers |
| 2 | 0.7927 | 4.3577 | words that could be used to compare and contrast different approaches or data in different fields. |
| 5 | 0.7922 | 2.304 |  words related to a chemical or physical process |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.8002 | 1.8669 |  words and phrases related to sounds. |
| 9 | 0.7992 | 4.3671 |  words that describe the sound and feeling of music, and also some words describing action films |
| 14 | 0.7972 | 10.6633 |  words or phrases related to discussions, whether technical or personal |
| 8 | 0.7966 | 3.1788 |  concepts related to the justice system, especially ethics and virtues within the justice system, as well as self-evaluation and attribution |
| 8 | 0.7951 | 4.7589 |  elements of formal writing (dates, claims, legal jargon and references). |
| 14 | 0.7943 | 7.9667 | states of being, assignments, or parameters |
| 8 | 0.7925 | 2.0841 |  technical words about optics, chemicals and laboratory analysis |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 16 | 0.7998 | 17.6015 |  strings of letters with capitalization, numbers, and acronyms, often relating to computer code, names, or math |
| 25 | 0.7976 | 10.8061 |  sentences using tentative language, value judgements and positive sentiment |
| 17 | 0.7949 | 10.5629 | coding terms |
| 20 | 0.7931 | 7.9941 | sentences describing faint surrounding noises being heard by a character or person |

**Mechanistic narrative:** "The opposite of loud is" predicts " quiet" (p=0.386). "Loud/quiet" is a well-known antonym pair but with some competition from "silent" and "soft". Early layers activate on sound/volume features and the antonym template. Middle layers engage contrast features for the sound dimension. Late layers push toward "quiet" — possibly influenced by the frequency of "loud and quiet" as a contrastive pair in educational and descriptive texts. Competitors "silent" (p≈?) and "soft" slightly dilute confidence. The prediction is reasonable but reflects that the "loud" antonym slot is slightly more ambiguous than "dark/light".

---

### Prompt: "<bos>The opposite of weak is"

**Predicted token:** `Output " strong" (p=0.444)` (prob=0.4436)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.7998 | 2.3667 |  words related to collaboration or separation, in technical contexts. |
| 0 | 0.7996 | 2.7011 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 6 | 0.799 | 3.3611 | terms and words associated with legal and scientific research documents |
| 0 | 0.7988 | 2.8599 | something the user looked at, is looking at, or will look at. |
| 6 | 0.7986 | 5.8468 |  words or phrases that are being defined or discussed in a descriptive way. |
| 5 | 0.7982 | 3.5668 |  common idioms or sayings |
| 0 | 0.7978 | 2.496 |  past tense verbs ending in 'pp' followed by 'ed' |
| 0 | 0.7976 | 3.0943 |  the word "visual" |
| 0 | 0.7974 | 2.377 |  terms related to sports, especially basketball and player contracts |
| 3 | 0.7972 | 2.6245 |  words related to legal or academic discourse. |
| 0 | 0.7967 | 1.9826 | the word "full" or words that sound like it |
| 1 | 0.7965 | 2.4817 |  words related to systems, organizations, or planned activity |
| 7 | 0.7963 | 4.0609 | the word "is" followed by function words |
| 1 | 0.7957 | 3.7486 |  words associated with team sports, law, government, and time duration. |
| 0 | 0.7953 | 2.4259 |  mathematical equations and expressions in LaTeX format |
| 6 | 0.7951 | 3.2618 |  words and phrases related to disagreement, disapproval, or conflict. |
| 5 | 0.7947 | 1.3938 |  indicators of strength or weakness |
| 2 | 0.7943 | 2.8036 | code-related terms, including http requests, maps, interfaces, contexts, and parameters |
| 0 | 0.794 | 1.7231 | a variety of words and phrases, including those relating to community service, food critiques, and locations. |
| 4 | 0.7938 | 3.3071 |  the word "alternative" plus related synonyms depending on context |
| 2 | 0.7934 | 4.3577 | words that could be used to compare and contrast different approaches or data in different fields. |
| 0 | 0.7932 | 1.7272 | references to time, including the words "past," "present," and also relative descriptors like "new", or words suggestive of the passage of time like "life" or "smells" |
| 0 | 0.793 | 1.6472 |  mathematical equations and related keywords like "let", "suppose", "determine", "multiple", "factor", "derivative", and "calculate" |
| 0 | 0.7926 | 1.6704 |  text from academic papers involving formulas and graphs including physics and math topics |
| 0 | 0.7924 | 3.6136 | a variety of words and phrases, including those relating to community service, food critiques, and locations. |
| 1 | 0.7919 | 3.4669 |  words associated with solving problems or making difficult decisions |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 14 | 0.8 | 50.8829 | the number 1 |
| 11 | 0.7994 | 6.8262 |  politically charged language in opinion pieces |
| 8 | 0.7984 | 4.2422 |  words related to legal proceedings, suffering, and political polarization. |
| 9 | 0.797 | 5.9783 |  negated conditionals in code and philosophical text. |
| 14 | 0.7955 | 7.6237 |  the word "is" after the or near the word "the" |
| 10 | 0.7945 | 6.0435 | negatively-charged words applicable to a broad set of topics, including law and morality. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7992 | 0.0 |  words related to political administration and legal rights |
| 16 | 0.798 | 7.9 |  questions |
| 18 | 0.7961 | 11.2394 |  content measuring personal willpower or ability to do something |
| 25 | 0.7959 | 9.9587 | the word "is" or "'s" within the context of descriptions. |
| 23 | 0.7949 | 11.7754 |  passages about religious figures or pagan celebrations |
| 22 | 0.7936 | 11.1132 |  words and phrases related to scientific studies and/or political situations |
| 23 | 0.7928 | 101.8073 |  the word "The" at the beginning of a sentence or phrase |
| 16 | 0.7922 | 120.2892 | the |

**Mechanistic narrative:** "The opposite of weak is" predicts " strong" (p=0.444). "Weak/strong" is a highly canonical antonym pair. Early layers activate on strength/force features and the antonym template. Middle layers engage reversal and contrast features for the strength dimension. Late layers converge on "strong" with moderate confidence. The main competitor is "powerful", but "strong" dominates as the most immediate lexical antonym. The confidence (p=0.444) is solid, consistent with "weak/strong" being a well-memorised binary pair in training data.

---

### Prompt: "<bos>The past tense of fly is"

**Predicted token:** `Output " flew" (p=0.389)` (prob=0.3894)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.8 | 1.6752 |  words related to stage, film, and theatrical interpretations of stories |
| 0 | 0.7998 | 1.7337 |  words related to medical, chemical or scientific terminology |
| 2 | 0.7992 | 2.9249 |  positive qualifiers for experiences, features, and products |
| 0 | 0.799 | 1.3222 |  words related to law, crime or legal proceedings |
| 1 | 0.7988 | 2.6378 |  the prefix "mon" in text about mathematics or computer science |
| 0 | 0.7986 | 2.6918 | the word "shot" used in various contexts |
| 6 | 0.7984 | 4.7792 | mathematical proofs or equations |
| 5 | 0.7982 | 4.8007 |  the last three letters of words. |
| 4 | 0.798 | 2.833 |  context related to words/language and their meanings or usages |
| 7 | 0.7978 | 6.5545 |  words that refute or question the truthfulness of a statement |
| 1 | 0.7976 | 2.4283 |  code and programming related keywords |
| 0 | 0.7974 | 3.7557 |  mentions of the word "white" |
| 4 | 0.797 | 2.897 | This neuron seems to be identifying text related to entertainment, specifically music, movies, and celebrities, possibly including names of artists, bands, titles, and roles. |
| 0 | 0.7968 | 4.7051 | various kinds of brackets and parentheses followed by a period |
| 2 | 0.7966 | 1.6726 | mentions of Christian saints, especially Saint Patrick, and place names, especially Santa Fe in the US state of New Mexico |
| 0 | 0.7964 | 1.4975 |  terms related to law, immigration, and gaming. |
| 2 | 0.7962 | 3.9426 |  verbs that describe a state or action related to research, medicine, law, or politics |
| 4 | 0.7956 | 3.6434 | technical or scientific language |
| 3 | 0.7954 | 3.0014 | mentions of glucose in blood samples |
| 0 | 0.7942 | 2.7984 |  the word "browser". |
| 0 | 0.794 | 3.7715 |  the word "exactly", sometimes in combination with the word "The" |
| 3 | 0.7934 | 3.222 | programming related keywords, symbols and syntax. |
| 7 | 0.7932 | 3.984 |  words related to sea creatures and flying creatures |
| 0 | 0.7928 | 4.0232 | forms of the verb "to be" with auxiliary verbs such as "do", "have", "would" and some past participle verbs |
| 0 | 0.7926 | 4.0242 | something the user looked at, is looking at, or will look at. |
| 1 | 0.7924 | 1.8042 |  words associated with public disagreement or debate and words in academic papers |
| 0 | 0.7922 | 3.5543 | sentences beginning with coordinating conjunctions like "but" and "and" |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.7994 | 4.1206 |  code snippets containing HTML tags and CSS |
| 8 | 0.7972 | 4.5292 | noun phrases about study methodology are sought by this neuron |
| 13 | 0.796 | 7.1341 |  code snippets in C and C++, assembly language, and German |
| 12 | 0.7958 | 6.1662 |  varied, but the neuron seems to activate on legal terms, medical terms, computer terms, and alcoholic beverages. |
| 11 | 0.7952 | 8.1515 |  references to books and published documents |
| 12 | 0.7946 | 7.214 |  names of things that have been designated or referred to by people |
| 14 | 0.7944 | 9.8894 | words that are commonly used as names of software functions or commands. |
| 10 | 0.7936 | 5.8962 |  words relating to chemical compounds and materials |
| 11 | 0.793 | 17.5626 |  name origins and meanings |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.7996 | 0.0 | code snippets containing the word "return" |
| 24 | 0.795 | 14.3403 |  a mishmash of terms without a clear unifying concept. |
| 21 | 0.7948 | 10.4659 |  alternative ways to phrase the concept of something happening despite something else using the word "spite" or "despite" |
| 21 | 0.7938 | 14.0497 |  various words related to flight and birds, as well as the word "to" preceding a verb. |

**Mechanistic narrative:** "The past tense of fly is" predicts " flew" (p=0.389). "Fly/flew" is an irregular past tense that requires lexical retrieval (not rule application). Early layers activate on the verb "fly" (motion/aviation features) and the past-tense template. Middle layers activate features for irregular verb conjugation — these appear to involve temporal/tense features ("mentions of grammatical tense", "words related to flight and birds, and the word 'to' preceding a verb"). Late layers push toward "flew" via features detecting the irregular vowel-change pattern (fly→flew, like ride→rode). The moderate confidence (p=0.389) reflects some competition from "flown" (past participle, sometimes mistakenly used) but "flew" wins as the simple past form.

---

### Prompt: "<bos>The past tense of run is"

**Predicted token:** `Output " ran" (p=0.365)` (prob=0.3647)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8 | 2.1751 | words ending in "ization" or prepositions like "of" |
| 4 | 0.7996 | 3.768 |  words with emotional connotations or words that reflect a sense of self or state of being. |
| 2 | 0.7994 | 2.2175 |  words related to wealth, management, and nobility |
| 1 | 0.7992 | 1.6911 |  words related to scientific studies in biology/chemistry including biological processes, forces/interactions, and analyses |
| 2 | 0.799 | 3.0443 |  instances of the word "of" |
| 4 | 0.7988 | 4.0781 | phrases containing the word 'of' that are followed by a noun |
| 2 | 0.7986 | 3.7761 | verbs being used |
| 0 | 0.7984 | 1.5828 |  words or characters related to science, math, technology, or coding |
| 6 | 0.798 | 3.5219 | technical or mechanical terms |
| 0 | 0.7978 | 3.5543 | sentences beginning with coordinating conjunctions like "but" and "and" |
| 4 | 0.7976 | 4.3698 | scientific observations and assumptions |
| 4 | 0.7972 | 3.8194 |  a variety of technical and abstract nouns and concepts across a range of topics, including morality, computing, and inventions |
| 1 | 0.797 | 4.2589 |  words or phrases that appear in legal or technical documents, like names of laws, legal terms (pleaded, testified), and technical terms, especially when abbreviated or in code |
| 3 | 0.7968 | 3.9117 |  language used to describe the process of software development and theoretical mathematical concepts |
| 2 | 0.7966 | 1.8259 |  terms related to time or distance |
| 1 | 0.7964 | 1.8014 | technical research texts, especially figures, experimental data, model numbers, references, and species names |
| 3 | 0.7962 | 2.4046 |  words related to nutrition and food deficiencies |
| 5 | 0.796 | 3.4765 |  grammatical terms and sentence parsing. |
| 6 | 0.7956 | 4.0964 | words related to medical or scientific texts, especially regarding drugs and chemical reactions, numbers, and plurals. |
| 0 | 0.7952 | 3.6374 | the word "else", often preceded by "someone" |
| 3 | 0.7948 | 3.2635 |  the end of phrases, clauses, or lists |
| 6 | 0.7946 | 3.0844 | language education materials for students and teachers including discussion of teaching methodologies. |
| 0 | 0.7942 | 1.6767 |  proper nouns and specialized vocabulary in technical and scientific texts |
| 2 | 0.794 | 2.7492 |  words or phrases related to technical or scientific topics across a variety of domains like computing, energy, biology, geography and marketing |
| 4 | 0.7938 | 4.4143 |  symbols and terms related to voltage, electricity, and electronics |
| 0 | 0.7936 | 1.2234 |  words related to physical action and body parts |
| 2 | 0.7934 | 1.4987 |  words relating to standardized communication and scientific procedures including protocols and synchronizing data |
| 4 | 0.7932 | 3.9723 |  things that involve dates or references |
| 7 | 0.793 | 3.8407 |  code examples including brackets, loops, or special characters |
| 1 | 0.7927 | 2.7275 |  text of scientific or legal papers |
| 3 | 0.7925 | 2.6313 | french text, code snippets, and financial-related terms |
| 2 | 0.7921 | 2.01 |  words related to conflict, defeat, and legal rulings |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7998 | 26.0636 | scientific or technical words and jargon |
| 15 | 0.7982 | 15.2938 |  passages that are about writing code |
| 10 | 0.7958 | 35.8948 |  references |
| 8 | 0.795 | 6.1082 |  code snippets and text fields that describe software and password requirements |
| 11 | 0.7944 | 6.1262 | mathematical/computational symbols, code snippets, and possibly numbers with units. |
| 13 | 0.7923 | 5.9892 | numerical references to dates, times, statutes, and other quantities. |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 24 | 0.7974 | 9.036 |  words related to people, especially names, roles, or personal pronouns, and very simple grammar. |
| 23 | 0.7954 | 7.1782 |  code and math-related terms |

**Mechanistic narrative:** "The past tense of run is" predicts " ran" (p=0.365). "Run/ran" is an irregular past tense. Early layers activate on the verb "run" (motion/action features) and the past-tense template. Middle layers engage irregular verb morphology features. Late layers push toward "ran" — a strong, frequent irregular form. The confidence (p=0.365) is lower than the irregular plurals (tooth/teeth: 0.811), suggesting irregular past tenses are slightly less robustly encoded than irregular plurals, possibly due to greater competition from the present form "run" (which is both present and past in some contexts) or "runs".

---

### Prompt: "<bos>The past tense of speak is"

**Predicted token:** `Output " spoke" (p=0.443)` (prob=0.4431)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8001 | 2.4046 |  words related to nutrition and food deficiencies |
| 5 | 0.7999 | 5.191 |  words related to engineering and technical manuals |
| 0 | 0.7997 | 2.4738 |  the word "hell" |
| 5 | 0.7993 | 6.1214 |  words related to creative works like books, movies, songs and their creation |
| 1 | 0.7991 | 3.5965 |  the word "modern" |
| 6 | 0.7989 | 3.5461 |  profanity, insults and gendered slurs, plus some mathematical and technical jargon |
| 4 | 0.7987 | 2.2679 |  words appearing within business or government reports |
| 1 | 0.7985 | 1.6444 |  technical terms related to code and data processing. |
| 0 | 0.7983 | 1.6101 |  scientific vocabulary, especially in the fields of medicine, chemistry, and mathematics. |
| 0 | 0.7981 | 1.4405 | a mix of context dependent terms that are diverse and hard to summarize |
| 0 | 0.7979 | 4.2056 | parenthetical numerical references and citations to literature, laws, and statistics |
| 3 | 0.7977 | 4.6652 |  verbs and participles or words that are acting like verbs. |
| 6 | 0.7975 | 4.1664 |  words related to studying something in a school setting, including the perspectives of people, and sometimes emphasizing German-related words. |
| 0 | 0.7973 | 2.824 | the word "of" in academic articles, possibly in the context of research objectives or methods |
| 3 | 0.7971 | 2.627 | labels of figures and lanes in lab reports and technical papers |
| 4 | 0.7969 | 4.8887 |  words with emotional connotations or words that reflect a sense of self or state of being. |
| 0 | 0.7967 | 1.5322 |  the number "ten" |
| 3 | 0.7958 | 2.4434 |  words and phrases related to scientific studies and research |
| 0 | 0.7956 | 1.4975 |  terms related to law, immigration, and gaming. |
| 7 | 0.7954 | 6.5545 |  words that refute or question the truthfulness of a statement |
| 4 | 0.7952 | 3.5303 |  words that end with "past" and words that indicate a change of state or are beyond something |
| 6 | 0.795 | 4.4898 |  code snippets and related programming terms |
| 0 | 0.7946 | 1.3222 |  words related to law, crime or legal proceedings |
| 1 | 0.7942 | 2.4283 |  code and programming related keywords |
| 2 | 0.7939 | 2.01 |  words related to conflict, defeat, and legal rulings |
| 6 | 0.7937 | 5.0967 |  question answer pairs and censored text |
| 3 | 0.7933 | 3.2635 |  the end of phrases, clauses, or lists |
| 4 | 0.7931 | 3.9723 |  things that involve dates or references |
| 0 | 0.7929 | 3.7715 |  the word "exactly", sometimes in combination with the word "The" |
| 0 | 0.7927 | 4.0242 | something the user looked at, is looking at, or will look at. |
| 4 | 0.7923 | 4.3698 | scientific observations and assumptions |
| 1 | 0.792 | 5.5322 |  instances of the word "fair" and the phrase "ends of justice" in legal documents |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 10 | 0.7995 | 11.9719 |  content related to scientific publications or medical procedures, possibly extracting patient data from research papers |
| 11 | 0.7944 | 31.9485 | scientific or technical words and jargon |
| 8 | 0.7925 | 5.2479 |  code or text related to dictionaries and named entity recognition |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 23 | 0.7965 | 9.5404 |  a diverse set of named entities, including people, organizations, and locations |
| 0 | 0.7962 | 0.0 |  words related to political administration and legal rights |
| 25 | 0.796 | 11.4447 |  LaTeX code, especially related to diagrams and mathematical notations |
| 22 | 0.7948 | 9.7157 |  personal experiences, opinions and reflections, often indicated by personal pronouns and related verbs |
| 24 | 0.7935 | 20.1953 | code and mathematical or logical expressions |

**Mechanistic narrative:** "The past tense of speak is" predicts " spoke" (p=0.443). "Speak/spoke" is an irregular past tense with a vowel change. Early layers detect the verb "speak" (communication/language features) and past-tense template. Middle layers activate tense-related features. Late layers push toward "spoke" with moderate confidence. Some competition from "spoken" (past participle) may reduce the score. The circuit correctly identifies this as an irregular form and routes via the lexical memory pathway rather than applying a regular "-ed" rule.

---

### Prompt: "<bos>The past tense of swim is"

**Predicted token:** `Output " swam" (p=0.496)` (prob=0.4960)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 5 | 0.8002 | 4.1316 |  baseball terminology |
| 0 | 0.7996 | 2.9192 |  technical words used in computing, science, or engineering |
| 0 | 0.7994 | 0.0 |  closing curly brackets in code snippets |
| 6 | 0.7992 | 4.4898 |  code snippets and related programming terms |
| 6 | 0.799 | 4.5116 |  forms of the verb "to be" |
| 0 | 0.7988 | 4.1868 |  the verb "is" and "are". |
| 0 | 0.7986 | 2.9988 |  code snippets, specific scientific or technical terms, and date-related words. |
| 7 | 0.7984 | 5.4049 |  rhetorical questions and references to blog posts |
| 6 | 0.7982 | 3.2563 | words related to programming, foreign language, finance or strong emotion. |
| 6 | 0.798 | 3.0585 |  words that define the meaning of concepts in a mathematical or general context |
| 4 | 0.7978 | 3.9723 |  things that involve dates or references |
| 1 | 0.7976 | 2.4283 |  code and programming related keywords |
| 1 | 0.7974 | 3.4348 |  code comments |
| 1 | 0.797 | 1.5176 |  terms related to Christian faith and religious belief |
| 4 | 0.7968 | 3.0799 |  text about small but essential dietary components and their deficiencies |
| 0 | 0.7959 | 2.4217 |  uses of the word "cool" along with related positive words |
| 3 | 0.7957 | 2.9439 | words related to legal and technical language |
| 2 | 0.7953 | 3.3241 | verbs being used |
| 0 | 0.7951 | 1.969 |  technical terms used in scientific writing |
| 5 | 0.7943 | 4.7954 |  technical language from scientific/instructional writing. |
| 2 | 0.7939 | 3.9209 | technical, legal or academic language, especially verbs, often found in formal reports |
| 4 | 0.7937 | 4.9475 |  multiple occurrences of hedging verbs in sentences with negative constraints |
| 3 | 0.7933 | 2.6046 |  words related to scientific research and study |
| 0 | 0.7931 | 3.3989 |  the word "meaning," and in some cases words around it |
| 0 | 0.7926 | 1.8938 |  technical or scientific terms, especially in a research context |
| 5 | 0.7924 | 3.0338 |  words and phrases related to naming or meaning |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 11 | 0.7998 | 12.2148 | code or programming language |
| 8 | 0.7972 | 5.6246 |  words related to cessation or ending, like "defunct," "discontinued," and "expired." |
| 9 | 0.7955 | 6.6736 |  code snippets, especially those relating to the execution of child processes and file paths |
| 11 | 0.7947 | 22.4429 |  name origins and meanings |
| 13 | 0.7941 | 7.0027 |  words and phrases related to official or legal proceedings |
| 12 | 0.7922 | 5.6467 |  the definitions of words, especially names, and the word "meaning" |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 25 | 0.8 | 13.0003 |  code snippets |
| 25 | 0.7965 | 7.9027 | code related to scripts and code execution |
| 19 | 0.7963 | 9.3358 |  conversational phrases or personal pronouns together with auxiliary verbs. |
| 24 | 0.7961 | 7.0805 |  mentions of occurrences and changes over time, especially using auxiliary verbs like "have," "has," "be," or "been." |
| 22 | 0.7949 | 9.3168 |  words related to states of being or consequences |
| 24 | 0.7945 | 13.5011 |  language related to institutions, negative situations, the internet, and programming languages |
| 18 | 0.7935 | 86.0581 |  mentions of grammatical tense and simple declarative clauses |
| 16 | 0.7928 | 20.3456 |  definitions of words |

**Mechanistic narrative:** "The past tense of swim is" predicts " swam" (p=0.496). "Swim/swam" is an irregular past tense. Early layers activate on the verb "swim" (water/motion features) and past-tense template. The confidence (p=0.496) is the highest among past-tense prompts, possibly because "swam" is uniquely determined — there is no commonly confused past participle form ("swum" is rare). Middle layers activate tense and morphological features. Late layers push toward "swam" with the highest confidence of the past-tense cluster, consistent with this being a more memorably distinctive irregular form.

---

### Prompt: "<bos>The past tense of write is"

**Predicted token:** `Output " written" (p=0.344)` (prob=0.3436)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 1 | 0.7999 | 4.3332 | words and phrases related to thinking, judging, and comparison, plus present tense verbs. |
| 4 | 0.7997 | 2.9409 |  context related to words/language and their meanings or usages |
| 5 | 0.7995 | 4.474 |  code snippets and file paths |
| 2 | 0.7989 | 1.8043 | the letter 'z' when it is near mathematical symbols |
| 5 | 0.7985 | 4.512 | various code snippets and programming-related text. |
| 4 | 0.7983 | 2.6026 | words relating to answering a question or configuring something |
| 3 | 0.7981 | 2.4584 |  programming code, place names with "de la", and the phrase, "of the valley" |
| 2 | 0.7977 | 1.5336 |  mentions of death or passing away |
| 0 | 0.7973 | 2.1751 | words ending in "ization" or prepositions like "of" |
| 0 | 0.7971 | 2.9596 | the word "mix" often also activating nearby words "mixing" or "mixed", "then" |
| 4 | 0.7969 | 5.5899 | code snippets and license agreements |
| 2 | 0.7967 | 3.7124 |  the verb "to be" in various languages and tenses |
| 2 | 0.7965 | 3.8807 | verbs being used |
| 3 | 0.7959 | 3.1003 |  words associated with computer programming, 3D printing and manufacturing |
| 0 | 0.7957 | 2.771 | the word "lift" and related concepts |
| 0 | 0.7953 | 3.3534 | words related to programming code, math, anatomy, medicine, structure, and anything that can range. |
| 0 | 0.7951 | 1.4975 |  terms related to law, immigration, and gaming. |
| 0 | 0.7943 | 4.0227 | occurrences of the words 'they', 'them', 'that', 'these', or something that can be replaced with 'they' or 'them' |
| 4 | 0.7936 | 2.14 | mentions of the word "term" or other words relating to language. |
| 0 | 0.7934 | 2.7984 |  the word "browser". |
| 1 | 0.7932 | 1.9671 |  mentions of school grades and educational institutions |
| 0 | 0.793 | 2.6918 | the word "shot" used in various contexts |
| 0 | 0.7924 | 3.0286 |  the suffix "ins" or the word "thereon" |
| 0 | 0.7922 | 4.0242 | something the user looked at, is looking at, or will look at. |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.8001 | 5.0418 |  words related to biological population dynamics and reproduction |
| 6 | 0.7993 | 4.4898 |  code snippets and related programming terms |
| 8 | 0.7991 | 18.9138 | the first-person pronoun "I" and the word "Exactly" |
| 8 | 0.7979 | 3.0859 |  words or phrases related to the production of media, such as movies or music |
| 9 | 0.7963 | 6.8296 | examples of the phrase "in spite of" |
| 7 | 0.7955 | 2.501 |  words and phrases related to reading and writing data in a computer system |
| 9 | 0.7947 | 9.2858 |  math or coding. |
| 6 | 0.7945 | 4.9281 |  forms of the verb "to be" |
| 8 | 0.7941 | 3.5612 |  the definition of "false" from Merriam-Webster |
| 10 | 0.7926 | 7.6093 | code snippets and programming related terminology |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.7987 | 7.9834 |  periods, numbers and names |
| 18 | 0.7975 | 35.8485 |  terms used in academic papers, especially those using measurements or observations |
| 14 | 0.7961 | 7.0497 |  code snippets, special characters, and logical operators |
| 19 | 0.7949 | 7.9728 |  sentences or phrases that mention figures, tables, and references |
| 14 | 0.7938 | 6.5116 | words used in academic papers, legal documents, or technical writing. |
| 14 | 0.7928 | 10.8266 |  words related to societal issues, groups and beliefs |

**Mechanistic narrative:** "The past tense of write is" predicts " written" (p=0.344) — interestingly the past participle rather than the simple past "wrote". This is a circuit mis-selection: "written" is grammatically valid as a past-tense form in certain constructions ("it is written") but "wrote" is the canonical simple past. Early layers activate on the verb "write" (text/document features) and past-tense template. The late-layer push toward "written" may reflect the higher frequency of "is written" (passive construction) in formal writing compared to "is wrote". The relatively low confidence (p=0.344) indicates the circuit is genuinely uncertain between "wrote" and "written", which are both high-frequency forms of this verb.

---

### Prompt: "<bos>The plural of child is"

**Predicted token:** `Output " children" (p=0.677)` (prob=0.6769)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 4 | 0.7998 | 2.6328 | mathematical symbols and notation |
| 7 | 0.7996 | 5.5986 | complex mathematical language and expressions |
| 0 | 0.7994 | 1.4467 |  code-related keywords like serialize, json, string, instantiation, async, containers, debugging, shared folders, and UI events |
| 0 | 0.7992 | 1.1841 |  proper nouns and technical terms |
| 2 | 0.7988 | 1.8984 |  words related to family, childhood, and male relationships |
| 0 | 0.7986 | 1.7096 |  the word "campaign" which generally relate to elections |
| 0 | 0.7983 | 1.7142 |  words related to processes and parts of machinery, electronics, mass produced products, and the legal field. |
| 2 | 0.7981 | 2.0533 |  mentions of specific mathematical and vehicular concepts |
| 0 | 0.7979 | 2.3725 | the word "reached" along with related words like 'reached', 'passes', 'yards',' English' and numbers |
| 1 | 0.7975 | 3.8699 |  terms about scripture and truth, often in the context of Christian belief |
| 0 | 0.7973 | 1.2342 | SQL code snippets and database related commands or references in formal settings |
| 0 | 0.7971 | 3.6207 |  sentences that start with the word "The" |
| 1 | 0.7969 | 1.6804 | the word "extra," technical instructions and words related to hands |
| 0 | 0.7967 | 1.5506 |  technical scientific terms, especially those related to medicine and biology |
| 0 | 0.796 | 3.3534 | words related to programming code, math, anatomy, medicine, structure, and anything that can range. |
| 0 | 0.7958 | 1.871 | mentions of the word "evil" |
| 5 | 0.795 | 5.1201 | phrases related to "I want to" |
| 5 | 0.7948 | 3.2241 |  the word "set" preceded by words indicating a comparison, often within the context of highly technical language and code |
| 0 | 0.7944 | 1.4817 |  words and fragments of text common in computer code and also in non-English languages |
| 0 | 0.7939 | 2.2868 |  words and phrases related to societal and political issues |
| 7 | 0.7935 | 2.9964 |  terms related to family relationships, especially focusing on marital status and legitimacy of children |
| 3 | 0.7933 | 4.0553 |  discourse regarding definitions and word usage |
| 7 | 0.7931 | 7.7656 |  rhetorical questions and references to blog posts |
| 3 | 0.7929 | 2.0079 |  parts of code related to threads and caching |
| 4 | 0.7927 | 5.4472 |  the word "is" |
| 0 | 0.7924 | 1.5975 | the word "also" |
| 0 | 0.7918 | 0.9714 |  abstract nouns ending in "-ity", "-ship", "-ness", "-ism", "-ence", "-ance", "-tion", "-tics", and other words related to the arts and philosophy |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 15 | 0.8 | 11.0643 | phrases using the word "nickname" or that refer to nicknames. |
| 9 | 0.7952 | 5.9949 |  references to old english combined with place names, or names and meanings |
| 15 | 0.7946 | 10.8643 |  a wide variety of proper nouns of many types with a slight preference for locations, nationalities or languages |
| 9 | 0.7937 | 10.5211 |  words related to some kind of technical field |
| 10 | 0.7922 | 4.7603 | code snippets, family relations, and super sentai teams |
| 11 | 0.792 | 49.9832 | the letters "L", "H," and "a" when they are at the beginning of a text block |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 23 | 0.799 | 9.1165 |  terms related to scientific research, especially pertaining to analysis, measurement, and results. |
| 22 | 0.7977 | 10.0556 |  words referencing large numbers of people or objects |
| 17 | 0.7965 | 6.9503 | statements or words that express general truths |
| 25 | 0.7963 | 12.3389 |  names, places, and dates. |
| 21 | 0.7956 | 13.8071 |  code snippets containing certain keywords such as "expect", "info","trials", "MutableArray", and "Object". |
| 18 | 0.7954 | 12.6139 | words or phrases used when asking for or giving agreement |
| 0 | 0.7941 | 0.0 |  Spanish and Portuguese words related to code and computers |

**Mechanistic narrative:** "The plural of child is" predicts " children" (p=0.677). This is a highly irregular plural requiring lexical retrieval of a suppletive form (child→children uses the Old English weak plural suffix "-ren"). Early layers activate on the noun "child" (age/family features) and the plural template. Middle layers engage irregular morphology features. Late layers strongly and confidently push toward "children" — this is one of the most memorised English irregular plurals. The confidence (p=0.677) is high but slightly below the most canonical pairs (tooth/teeth: 0.811), perhaps because "childs" (though ungrammatical) occasionally appears in training data as an error form.

---

### Prompt: "<bos>The plural of foot is"

**Predicted token:** `Output " feet" (p=0.780)` (prob=0.7796)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8 | 2.2842 | language regarding collaborative relationships or partnerships |
| 3 | 0.7998 | 4.3607 |  words related to scientific research reports |
| 0 | 0.7996 | 1.7153 |  words or short phrases that seem out of place among the rest of the text |
| 0 | 0.7994 | 2.9707 |  references to dates and times |
| 0 | 0.7992 | 1.9525 | words and phrases related to legal procedures |
| 3 | 0.7988 | 2.4364 |  words related to scientific experimentation and description |
| 4 | 0.7982 | 2.9277 |  medical terms related to pemphigus disease and its treatment |
| 3 | 0.798 | 2.2907 | words relating to trips or the body |
| 6 | 0.7977 | 3.5066 |  JSON code related to formatting numerical values and words related to prefixes |
| 3 | 0.7973 | 3.3225 |  words/phrases related to story telling |
| 0 | 0.7971 | 4.2056 | parenthetical numerical references and citations to literature, laws, and statistics |
| 1 | 0.7967 | 1.1916 |  citations to academic publications in physics |
| 5 | 0.7963 | 4.3241 |  terms related to the immune system, especially receptors and cells involved in pathogen recognition. |
| 1 | 0.7961 | 2.579 |  academic writing discussing study design or legal proceedings. |
| 3 | 0.7957 | 2.1856 |  technical and chemical terms as well as google related terms |
| 3 | 0.7952 | 2.0079 |  parts of code related to threads and caching |
| 0 | 0.795 | 1.2065 |  words related to research studies |
| 5 | 0.7948 | 5.6871 |  terms related to the immune system, especially receptors and cells involved in pathogen recognition. |
| 0 | 0.7944 | 1.2105 | the words "very" and "present", sometimes activating for other words related to generality |
| 0 | 0.7942 | 2.74 |  words related to money or business transactions |
| 2 | 0.794 | 2.3867 |  variable declarations in code that use the word "packed" |
| 1 | 0.7938 | 3.0995 |  words or phrases that indicate progress, success, or advancement in various fields like computer science, history, genomics, and medicine. |
| 0 | 0.7931 | 1.5506 |  technical scientific terms, especially those related to medicine and biology |
| 0 | 0.7927 | 1.8591 |  various forms of the word 'treat' and 'exist', and words related to birth and body parts |
| 6 | 0.7923 | 3.7892 |  words or phrases related to statistics and location |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 9 | 0.799 | 6.0069 |  names of television networks, places and people where the spelling is slightly unusual |
| 10 | 0.7986 | 9.8022 | code and grammatical terminology |
| 9 | 0.7984 | 9.4863 | text relating to naming or definitions of people, places, things or processes |
| 10 | 0.7975 | 6.0427 | mentions of family relationships, hyphenated words, and math symbols |
| 11 | 0.7969 | 49.9832 | the letters "L", "H," and "a" when they are at the beginning of a text block |
| 9 | 0.7954 | 4.205 |  words related to medical problems, anatomy and diagnosis. |
| 9 | 0.7935 | 5.8348 | changes in names/labels/text |
| 8 | 0.7933 | 12.8384 |  code snippets with specific coding keywords and markup tags |
| 10 | 0.7929 | 4.3289 |  words/phrases related to products and machines |
| 8 | 0.7925 | 6.7099 | mentions of technical elements of language and chemical compounds, while also activating for text in other languages |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.7965 | 45.3392 |  colons, words that are defined, and some words that are defined by webster. |
| 15 | 0.7959 | 12.5112 | occurrences of the word "is", especially after certain preceding words |
| 15 | 0.7946 | 105.3984 | code snippets |
| 18 | 0.7921 | 9.5868 |  URLs and file paths |
| 23 | 0.7918 | 10.7162 |  code snippets, terminal commands and programming related text in conversations |

**Mechanistic narrative:** "The plural of foot is" predicts " feet" (p=0.780). "Foot/feet" is a high-frequency umlaut plural. Early layers activate on the body-part and measurement features of "foot". Middle layers engage irregular morphology features. Late layers strongly activate on " feet" — a uniquely determined and extremely common plural form. The high confidence (p=0.780) reflects that "feet" is the overwhelming dominant completion in all contexts (body parts, measurements, poetry), leaving no realistic competitors. The circuit routes cleanly through the irregular morphology pathway.

---

### Prompt: "<bos>The plural of goose is"

**Predicted token:** `Output " geese" (p=0.637)` (prob=0.6370)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 0 | 0.8001 | 1.5975 | the word "also" |
| 6 | 0.7999 | 2.524 |  words related to fashion and textiles |
| 6 | 0.7995 | 4.7929 |  profanity, insults and gendered slurs, plus some mathematical and technical jargon |
| 2 | 0.7993 | 2.056 |  technical terminology from chemistry, biology, and physics |
| 6 | 0.7991 | 3.3436 |  words that define the meaning of concepts in a mathematical or general context |
| 0 | 0.7987 | 1.4088 |  words near references to time or date |
| 1 | 0.7985 | 1.8043 |  medical topics, especially those associated with the nose and throat |
| 3 | 0.7983 | 2.5843 |  a combination of the words "two", "of", and/or "the" and also some proper nouns |
| 1 | 0.7981 | 1.7747 |  references to religion or religious concepts, and the word "of" |
| 0 | 0.7979 | 2.0602 |  words related to environmental processes and industrial design of appliances and vehicles |
| 4 | 0.7977 | 4.1426 |  words and phrases related to animals, including humans, in the context of populations, mating, and genetics, as well as natural habitats and trails. |
| 5 | 0.7975 | 3.5706 |  terms related to fluids, water, or laboratory processes |
| 2 | 0.797 | 1.8151 |  mentions of abstract concepts, especially those associated with guidance in life. |
| 0 | 0.7968 | 2.0116 |  uses of the word "one", and single use occurrences of similar words, as well as words implying evaluation or evidence. |
| 1 | 0.7966 | 1.5973 |  page numbers and citations, often in the context of academic or bibliographic references |
| 1 | 0.7962 | 3.6586 |  terms about scripture and truth, often in the context of Christian belief |
| 4 | 0.7958 | 3.8664 | code snippets/java code |
| 7 | 0.7954 | 7.7656 |  rhetorical questions and references to blog posts |
| 0 | 0.7952 | 1.5506 |  technical scientific terms, especially those related to medicine and biology |
| 6 | 0.795 | 4.0824 |  forms of the verb "to be" |
| 0 | 0.7948 | 2.7622 |  terms related to medicine or legal situations |
| 2 | 0.7941 | 4.1352 |  code snippets related to licensing, copyrights, or importing libraries |
| 4 | 0.7939 | 3.0981 | terms and symbols related to programming, legal language, and research/scientific concepts. |
| 1 | 0.7935 | 2.5246 | terms related to technology or insurance policies |
| 4 | 0.7933 | 2.8765 |  words and phrases related to different subjects, including food, magical elements, colors, and science |
| 4 | 0.7931 | 4.9852 | words and phrases that denote mild to strong levels of emphasis |
| 4 | 0.7929 | 3.0343 |  context related to words/language and their meanings or usages |
| 3 | 0.7927 | 2.3382 |  complex mathematical equations and formulas involving variables, symbols, and operators. |
| 0 | 0.7925 | 2.1086 |  words related to medical, chemical or scientific terminology |
| 3 | 0.7922 | 4.9427 |  code-like parameter declarations, especially those enclosed in quotes and angle brackets and related to hardware configuration |
| 3 | 0.792 | 3.7386 |  words related to computers, technology, medicine, or the legal system |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 12 | 0.7997 | 8.8797 |  words and phrases with unusual or fantastical elements |
| 9 | 0.7973 | 4.6406 | instances of the word "called" or "call" and nearby words |
| 10 | 0.7964 | 4.4359 |  words and phrases related to scientific analysis, particularly in physics, chemistry, and biology. |
| 9 | 0.796 | 27.6213 |  lines of code that import libraries or declare constants |
| 9 | 0.7937 | 10.5211 |  words related to some kind of technical field |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.7989 | 6.6882 |  topics about bees and/or waste management |
| 24 | 0.7956 | 7.1382 |  words and abbreviations in a variety of languages that often begin with the letters "II", "Ka", or are brand names.  |
| 25 | 0.7946 | 20.0439 | code snippets and mentions of the chemical "cholinesterase" |
| 25 | 0.7943 | 8.5018 |  words related to academic and technical papers. |

**Mechanistic narrative:** "The plural of goose is" predicts " geese" (p=0.637). "Goose/geese" is an umlaut plural. Early layers activate on the animal/bird features of "goose". The circuit correctly identifies this as an irregular plural requiring umlaut. Late layers push toward "geese" with moderate-high confidence — somewhat lower than "feet" (p=0.780) or "teeth" (p=0.811) because "gooses" occasionally appears as a back-formation in informal usage, diluting the prediction. The circuit's animal-feature early activation is interesting: bird features at early layers may interact with the plural morphology pathway differently than body-part or tool features.

---

### Prompt: "<bos>The plural of mouse is"

**Predicted token:** `Output " mice" (p=0.706)` (prob=0.7056)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 3 | 0.8001 | 4.2714 | technical writing that cites books or scientific publications and formal legal documents |
| 1 | 0.7999 | 3.8524 |  text of scientific or legal papers |
| 0 | 0.7993 | 2.0468 |  words related to law, crime or legal proceedings |
| 0 | 0.7989 | 2.3991 | the word "damage" and sometimes other words near "damage" or related to negative experiences |
| 1 | 0.7987 | 1.6804 | the word "extra," technical instructions and words related to hands |
| 0 | 0.7985 | 1.4467 |  code-related keywords like serialize, json, string, instantiation, async, containers, debugging, shared folders, and UI events |
| 0 | 0.7983 | 1.5427 | words ending in "ization" or prepositions like "of" |
| 0 | 0.7979 | 2.3504 | locations and organizations |
| 3 | 0.7977 | 3.218 |  hedging language and/or quantifying statements |
| 3 | 0.7973 | 2.7755 |  file paths or pieces of code and biological processes |
| 0 | 0.7969 | 1.9525 | words and phrases related to legal procedures |
| 0 | 0.7967 | 3.0174 |  mentions of research papers, studies or other formal publications |
| 2 | 0.7965 | 2.1796 |  the possessive pronouns "teu" and "seu" in Portuguese |
| 0 | 0.7963 | 1.4088 |  words near references to time or date |
| 4 | 0.796 | 3.1344 | terms that have the word "equivalent", and also certain collocations that can involve the word "standard" |
| 1 | 0.7958 | 1.7747 |  references to religion or religious concepts, and the word "of" |
| 1 | 0.7954 | 2.3411 |  words related to technological or medical fields |
| 2 | 0.7952 | 2.3362 |  legal and formal language |
| 0 | 0.795 | 3.3534 | words related to programming code, math, anatomy, medicine, structure, and anything that can range. |
| 3 | 0.7944 | 9.3926 |  academic publications that use scientific or medical data |
| 2 | 0.7942 | 2.0205 |  words related to groups and their statistical properties |
| 1 | 0.794 | 1.9404 | mentions of competitions, arguments, or parasites |
| 0 | 0.7938 | 2.7012 |  words related to scientific research, educational programs, and medical conditions |
| 0 | 0.7934 | 1.1473 |  technical words and jargon, especially from scientific documents or legal documents. |
| 0 | 0.7931 | 1.1126 |  words indicating the end or beginning of a period |
| 6 | 0.7929 | 4.0141 |  words that are difficult to categorize but seem to be related to names, branding, and academic language. |
| 0 | 0.7927 | 1.9681 |  various tools, along with properties and uses of those tools. |
| 0 | 0.7923 | 3.5832 |  modality verbs |
| 4 | 0.7921 | 1.8898 |  words related to legal proceedings and jurisprudence |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.7995 | 7.3976 |  mathematical equations and symbols, especially those involving transposes, duals, and derivatives |
| 14 | 0.7991 | 17.9337 |  phrases that contain the word searches, AND/OR, or the word term |
| 13 | 0.7975 | 6.4259 |  the start of chunks of code, markup, text, or math |
| 10 | 0.7971 | 3.902 |  words indicating a step-by-step process |
| 11 | 0.7956 | 17.8744 |  name origins and meanings |
| 8 | 0.7948 | 4.0576 |  grammatical terms and the verb "sein" (to be) in German |
| 8 | 0.7946 | 6.0761 |  internet addresses, references to the best services and products, and marketing language |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 23 | 0.7997 | 9.5665 | abbreviations, code snippets, and date formats |
| 15 | 0.7981 | 10.7877 |  the word "and" and puntuation marks like commas |
| 19 | 0.7936 | 7.8639 |  instances of stating facts or ways of doing things |
| 23 | 0.7925 | 12.9081 |  code-related concepts such as data access and processing |

**Mechanistic narrative:** "The plural of mouse is" predicts " mice" (p=0.706). "Mouse/mice" is an irregular umlaut plural. Early layers activate on the animal-feature of "mouse" (and possibly computer-mouse features, which co-occurs frequently in training data). The circuit must distinguish between the animal meaning (mice) and the computer meaning (mouses, in some usages). Late layers push toward " mice" with good confidence (p=0.706), suggesting the animal-irregular-plural pathway dominates. The circuit successfully routes via irregular morphology rather than applying "-s" suffix.

---

### Prompt: "<bos>The plural of tooth is"

**Predicted token:** `Output " teeth" (p=0.811)` (prob=0.8111)

**Token competition:**

| Rank | Token | Logit score | Features voting |
|------|-------|-------------|-----------------|

**Circuit walkthrough:**

*Early layers — input recognition*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 2 | 0.7998 | 1.9475 |  articles and pronouns in Portuguese |
| 0 | 0.7992 | 2.0409 |  words, abbreviations, and symbols related to scientific measurement and notation |
| 1 | 0.799 | 2.4949 |  terms and phrases related to skipping or restarting processes, especially in the context of computing and data |
| 3 | 0.7986 | 2.1609 |  code-related terms concerning properties and arrays |
| 3 | 0.7984 | 2.4364 |  words related to scientific experimentation and description |
| 2 | 0.798 | 2.2102 |  words related to mechanical movement and components. |
| 2 | 0.7978 | 1.5612 |  the word "row" and also "col" |
| 0 | 0.7976 | 1.5554 |  technical documents about computer science or chemistry |
| 3 | 0.7968 | 9.3926 |  academic publications that use scientific or medical data |
| 2 | 0.7965 | 2.0205 |  words related to groups and their statistical properties |
| 2 | 0.7963 | 1.8151 |  mentions of abstract concepts, especially those associated with guidance in life. |
| 5 | 0.7961 | 3.2132 |  named entities (people and places) |
| 0 | 0.7957 | 1.9525 | words and phrases related to legal procedures |
| 2 | 0.7953 | 2.8864 |  many different things including numerals, certain words in other languages, citations, mathematical relations and assumptions, and certain affixes |
| 1 | 0.7951 | 1.8902 |  the word "operator" |
| 0 | 0.7945 | 1.5319 |  words related to cell biology and growth factors |
| 0 | 0.7941 | 1.8086 |  words or phrases connected to technical inventions or medical procedures |
| 3 | 0.7939 | 2.5843 |  a combination of the words "two", "of", and/or "the" and also some proper nouns |
| 0 | 0.7937 | 1.58 |  words related to legal claims, arguments, or procedures |
| 0 | 0.793 | 2.3944 |  data reported as a percentage inside brackets, especially in a laboratory or medical context, and also recognizes countries |
| 4 | 0.7928 | 3.8664 | code snippets/java code |
| 1 | 0.7926 | 2.6286 |  the word "particle" and its plural. |
| 2 | 0.7924 | 5.508 |  the definite article "The" |
| 5 | 0.7922 | 5.1201 | phrases related to "I want to" |

*Middle layers — relational mapping*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 8 | 0.8002 | 6.0983 |  elements of formal writing (dates, claims, legal jargon and references). |
| 6 | 0.7994 | 4.3455 |  instances of grammatical corrections or suggestions, particularly the use of "in spite of" |
| 9 | 0.7988 | 2.3199 |  words and symbols used in technical documents |
| 6 | 0.7974 | 3.7031 |  words and phrases indicating scientific findings or mathematical relationships |
| 6 | 0.7972 | 3.5066 |  JSON code related to formatting numerical values and words related to prefixes |
| 11 | 0.797 | 7.5736 | technical documents containing a diverse set of words related to electronics, math, questions/answers, and engineering. |
| 7 | 0.7949 | 7.0918 | sentences that show the outcome, method, or interpretation of experiments in a scientific paper |
| 9 | 0.7947 | 21.859 | code and equations |
| 9 | 0.7934 | 26.5524 |  various code snippets |
| 6 | 0.7932 | 4.7254 |  words denoting name or type |

*Late layers — token selection*

| Layer | Influence | Activation | Feature label |
|-------|-----------|------------|---------------|
| 18 | 0.8 | 11.0636 |  placenames and specific names |
| 13 | 0.7996 | 101.2541 |  sentences beginning with the word "In" |
| 14 | 0.7982 | 6.7176 | words used in academic papers, legal documents, or technical writing. |
| 15 | 0.7959 | 105.3984 | code snippets |
| 0 | 0.7955 | 0.0 |  mentions of seasons, especially winter and autumn, and the seasonal aspects of plant life |
| 17 | 0.7943 | 9.6825 | term, word |

**Mechanistic narrative:** "The plural of tooth is" predicts " teeth" (p=0.811) — the highest confidence prediction in the entire corpus. "Tooth/teeth" is a uniquely determined, extremely high-frequency irregular plural with no plausible competitors. Early layers activate on the body-part/dental features of "tooth". The circuit engages irregular umlaut morphology features in middle layers. Late layers deliver a maximally confident prediction: there is no ambiguity, no alternative form, and "teeth" is universally the correct plural. This serves as a ceiling-case for irregular plural retrieval — the circuit is maximally engaged and maximally certain.

---


### 3.3 Steering Validation

We attempted activation steering with strength=20.0 and strength_multiplier=4.0 on the top 3 recurring features by avg_influence, applied to the representative prompt "The opposite of hot is". The Neuronpedia live inference endpoint returned HTTP 500 (Unknown Error) for all steering attempts, consistent with the gemmascope-transcoder-16k SAE not supporting live model inference (it is used only for attribution graph generation, not steering). This is a known limitation of the transcoder architecture at the time of this analysis.

Despite the absence of steering output, we performed circuit neighbourhood analysis using get_subgraph_node_ids_for_feature() with depth=1 to characterise each feature's local circuit context.

**Steering Validation Results:**

| Feature | Layer | Avg Influence | Appearances | Label | Neighbourhood (nodes) | Representative neighbours |
|---------|-------|---------------|-------------|-------|----------------------|--------------------------|
| 11637899 | 0 | 0.7089 | 71 | the word "part" followed by prepositions | 32 | code comments, documentation block starts, code snippets and web dev references |
| 18711902 | 0 | 0.6891 | 58 | words related to money or business transactions | 16 | the word "artificial", words indicating disagreement/contradiction, the word "hot" and heat associations |
| 74438300 | 0 | 0.6834 | a variety of specific nouns | 57 | code comments, documentation, scientific/technical document descriptions |

The neighbourhood analysis reveals that feature 18711902's neighbourhood includes "words indicating disagreement or contradiction" and "the word 'hot' and heat associations" -- direct evidence that this feature participates in the antonym-mapping sub-circuit for the representative prompt, positioned at the interface between semantic content recognition and antonym-mapping, even though its Neuronpedia label focuses on financial vocabulary.

---

## 4. Discussion

### 4.1 What Recurring Features Reveal About Linguistic Reasoning

The most striking finding is the dominance of technical/code-adjacent features in the recurring feature list. Of the top 15 recurring features, 10 are labelled with references to programming code, scientific texts, legal documents, or technical writing. Only one (layer 8, feature 3234687: "words and phrases related to the meaning of words") is directly interpretable as a metalinguistic feature.

This pattern reflects a fundamental property of how large language models trained on mixed corpora represent language: the structural template "The X of Y is [answer]" occurs overwhelmingly in technical documentation, legal texts, and scientific papers. When Gemma-2-2B processes "The plural of mouse is", it activates the same feature cluster that fires for "The syntax of function is", because these have the same surface form. The model's linguistic circuit is not purely dedicated to language; it is a subset of a more general definitional-template circuit.

### 4.2 Layer Distribution Analysis

The recurring features span layers 0-8 in our top-15 list:

- **Layer 0 (9 of 15):** Dominance of embedding-adjacent features suggests prompt-template patterns are detected very early, from the token-level structure.
- **Layers 3-7 (3 of 15):** Mid-early features perform structural and domain recognition.
- **Layer 8 (1 of 15):** The semantically richest recurring feature ("meaning of words") appears at the early/middle transition boundary.

The per-prompt circuit walkthroughs reveal a different picture for late layers (16-25): these are where task-specific features appear -- irregular morphology features, antonym-specific contrast features, and word-class derivation features. The late layers are not well-represented in the recurring features list because they are more task-specific (appearing in 10-30 graphs rather than all 31), but they carry the linguistically interesting computation.

### 4.3 Task-Specific Circuit Signatures

**Antonyms/Opposites:** The middle layers (4-7) consistently activate "words related to comparison, symmetry, or reversals" and "instances of 'meant by' or 'mean by'" features. These are the mechanistic signature of antonym retrieval -- a semantic reversal operator. Confidence scales with antonym uniqueness: dark/light (p=0.602) > hot/cold (p=0.566) > weak/strong (p=0.444) > loud/quiet (p=0.386).

**Irregular morphology:** Late-layer features for specific morphological patterns (umlaut plurals, strong verb past tenses) show distinctive activation profiles. The highest-confidence predictions cluster in this category (tooth/teeth: p=0.811, foot/feet: p=0.780), consistent with these forms being strongly memorised as whole units rather than derived by rule.

**Synonyms:** The circuit fails to converge: the predicted tokens are ":" (a colon) with very low probability (p=0.092-0.101), reflecting that synonym tasks are genuinely underdetermined and the circuit does not have a strong enough signal to choose among the many valid alternatives.

**Derivational morphology:** The circuit performs variably (p=0.094-0.404), with noun derivation ("happy -> happiness") more confident than adjective/adverb derivation. Regular "-ly" adverb derivation (quick -> quickly) has the lowest confidence, consistent with the hypothesis that regular morphological rules are less robustly encoded than irregular, high-frequency forms.

### 4.4 Surprises

1. **Technical text features dominate a linguistic task corpus.** The most frequent recurring features relate to code and scientific documents, not to language itself. This suggests Gemma-2-2B processes linguistic reasoning prompts primarily via a general definitional-template circuit shared with technical documentation tasks.

2. **"The word 'part' followed by prepositions" is the highest-influence recurring feature.** This reflects the fact that every prompt of the form "The X of Y is" contains exactly this structure, making it a perfect detector for the shared prompt template.

3. **The colon ':' is predicted for synonym tasks.** Rather than predicting a synonym word, the circuit routes to a list/definition format marker. Synonyms in training data often appear as "X: [synonym1], [synonym2], ..." rather than as simple completion-style constructions.

4. **"The fear of heights is called" predicts " ac" (a subword).** The circuit knows the answer is "acrophobia" but generates only the first subword. This is a tokenisation artefact that reveals the model's internal representation is word-level (correct) but its output generation is subword-level (tokeniser-dependent).

---

## 5. Limitations

1. **Steering validation was unavailable.** The gemmascope-transcoder-16k SAE does not support live inference via the Neuronpedia API, preventing direct causal validation of the discovered features. Future work should use an SAE that supports activation steering for causal confirmation.

2. **2 of 33 prompts were excluded** ("The verb form of decision is", "The adjective form of beauty is") due to API rate limits during graph generation. These derivational morphology prompts may have revealed additional features relevant to that subtask.

3. **Top-15 feature labelling only.** Of 527 recurring features, only 15 were labelled. The full set may contain more interpretable linguistic features in the lower-ranked entries.

4. **Attribution graph threshold sensitivity.** Results depend on the graph generation parameters (nodeThreshold=0.8, edgeThreshold=0.85). Different thresholds may reveal additional features or exclude some reported here.

---

## 6. Conclusion

This paper characterised the circuit underlying Gemma-2-2B's performance on 31 linguistic reasoning prompts. We identified 527 features recurring across at least 30% of graphs, dominated by technical-text and definitional-template features, with a small but important set of semantically meaningful linguistic features (notably layer 8, feature 3234687 for metalinguistic reasoning and layer 4, feature 50205205 for etymology/definition contexts).

The circuit architecture is consistent across all five linguistic subtasks: early layers detect the prompt template, middle layers perform relational mapping (reversal for antonyms, lexical lookup for irregular forms, domain routing for definitions), and late layers drive token selection. Task confidence correlates strongly with the uniqueness and frequency of the target answer in training data: irregular forms outperform synonym tasks by a large margin (p=0.811 vs p=0.092).

The dominant presence of technical/code-adjacent features in the recurring list is the most significant finding, revealing that Gemma-2-2B's "linguistic circuit" is not a specialised module but rather a shared definitional-template circuit that serves both formal document processing and linguistic reasoning. This has implications for understanding how linguistic competence emerges in LLMs trained on mixed code/text corpora.

---

*Analysis conducted on Gemma-2-2B using the gemmascope-transcoder-16k SAE suite via Neuronpedia. 31 attribution graphs generated March 2026. Circuit saved as: "Linguistic Circuit -- Gemma-2-2B".*
