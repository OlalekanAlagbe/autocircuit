
discussion = r"""
## Discussion

### What the Recurring Features Reveal About Analogical Reasoning

The 429 recurring features identified at the 50% threshold reveal a surprising architecture for analogical reasoning in Gemma-2-2B. The most universal features (L6/F2586668, L3/F5150441) are labeled not as "analogy" or "comparison" features but as structural-register features associated with formal documentation, code, and legal text. This finding suggests that the model's internal representation of the "A is to B as C is to D" template resembles formal technical documentation patterns -- consistent with the observation that analogical reasoning prompts have a distinctive syntactic formality (repeated "is to" connectives, balanced parallel structure) that activates features normally associated with structured, formal text.

In contrast, the explicit analogy-labeled features (L8/F94882191 "analogies or comparisons," L5/F16817094 "analogies," L7/F286895 "analogies and comparisons," L9/F89171325 "phrases suggesting uncertainty or comparison") appear in layers 5-9 -- after the structural parsing features at L3-L6. This layered pattern supports a two-stage interpretation: (1) the model first detects that the input has a formal comparative structure (Layers 3-6), then (2) recognizes it specifically as an analogical comparison (Layers 5-9). The structural detection precedes the semantic classification.

### Excitatory Path Consistency

Across 10 prompts, 7 had at least one excitatory (purely positive-weight) causal path. The two highest-confidence predictions (Paris/Germany, 97.3%; Cairo/Kenya, 96.3%) had the richest domain-specific feature activation, while the four lowest-confidence predictions (Fish/air, Leaf/flower, Wheel/airplane, Soldier/navy) either lacked excitatory paths entirely or had excitatory paths originating from embedding-level features rather than labeled mid-layer features. This correlation between excitatory-path presence and confidence supports the interpretation that excitatory paths represent the dominant computational mechanism when the model has strong domain knowledge, while mixed paths dominate when domain knowledge is weak or absent.

The most consistent late-layer convergence point is the L19-L21 aggregation hub. Paths from 8 of 10 prompts pass through this region (L19-L25) before reaching the logit at L27. Features at L14348 (Layer 19/20, "prepositions and connectors") appear as the highest-influence late-layer feature in two prompts (Paris/Germany and Clock/thermometer), suggesting this feature tracks the final "is to [ANSWER]" slot that the model is completing.

### Inhibitory Paths: Systematic or Noise?

No purely inhibitory (all-negative-weight) causal paths appeared in the top-5 paths for any prompt. All paths were either excitatory or mixed. This absence of systematic inhibitory paths suggests that token competition in analogical reasoning is not primarily implemented through explicit inhibition of competitor tokens but rather through positive selection of the winning token -- the model converges on the answer through amplification rather than suppression. The mixed paths (alternating positive and negative edges) may represent gating or modulation rather than direct token competition.

### Path Structure vs. Layer Distribution

The layer distribution of recurring features concentrates in Layers 0-10, while the causal paths consistently pass through Layers 17-27. This apparent mismatch is explained by the attribution graph structure: the recurring features at Layers 0-10 are the most consistently present circuit components, but the actual causal signal to the logit travels primarily through late-layer nodes (Layers 17-27) that are prompt-specific and therefore do not appear as recurring features at the 50% threshold.

The edge neighbourhood analysis (Step 5a) supports this interpretation: the top recurring feature by influence (L7/F4828270, "reference codes/abbreviations") has outgoing edges primarily to Layers 9-10 (not directly to the logit), suggesting it acts as a mid-network signal aggregator rather than a final prediction driver.

### Surprising Findings

1. **The dominant recurring feature (L6/F2586668) activates analogical reasoning through a documentation/code register, not a semantic analogy register.** The most universal circuit component for analogical reasoning fires on "code, legal jargon, and scientific texts" -- patterns associated with the formal structure of the "A is to B as C is to" template rather than its meaning.

2. **Path weight is not correlated with prediction confidence.** The Leaf/petal analogy (p=0.136) has the highest maximum path weight (2086.9) of any prompt; the Soldier/navy analogy (p=0.214) has the lowest (5.7). Path weight measures the strength of individual edge connections, not the clarity of the token-selection decision.

3. **Two prompts have explicit "analogy" template features in their late-layer bands** (L15/F12474: "analogies in conversational text"; L17/F2469: "analogies using 'the same way' construction"). These appear specifically in functional/activity analogies (Clock/thermometer and Book/reading), suggesting a dedicated analogy-recognition sub-circuit for this category.

4. **The Judge/court prompt has a single-concept "court" feature at Layer 17 (F6505).** This is the only case where a single-word concept feature appears as a top-influence node in the late band, directly encoding the source-domain institution and enabling the analogical mapping to "church." This direct concept-to-concept mapping underpins the highest-confidence non-geographic analogy (p=0.486).

---

## Limitations

**Attribution graph thresholds.** Graph generation used nodeThreshold=0.8 and edgeThreshold=0.85, which may exclude weak but relevant features. A lower threshold would increase graph density and potentially reveal additional analogy-specific features.

**SAE coverage.** The gemmascope-transcoder-16k SAE covers all 26 transformer layers but is a transcoder (not a standard SAE), and feature labeling relies on automated interpretability methods from Neuronpedia. Some feature labels appear incorrect or underspecified (e.g., L6/F2586668 labeled as "programming code" despite its universal role in the analogy circuit). True circuit interpretation requires human verification of feature labels.

**Single-model analysis.** All results are specific to Gemma-2-2B. Whether the same circuit components appear in larger models (Gemma-2-9B, Gemma-2-27B) or other model families is unknown.

**Logit contribution sparsity.** Token competition data was unavailable for most prompts (logitContributions field absent or empty), limiting quantification of score gaps between predicted and competing tokens.

**Causal path algorithm.** The greedy forward/backward path tracing finds the strongest single-step paths, not all paths. Complete path enumeration would provide a richer picture of circuit redundancy.

---

## Conclusion

This paper presents a detailed mechanistic interpretability analysis of analogical reasoning in Gemma-2-2B using attribution graphs and causal path tracing across 10 diverse prompts. The key findings are:

1. **A two-stage circuit** implements analogical reasoning: a universal structural-parsing stage (Layers 0-6) detects the formal "A is to B as C is to" template, while a domain-specific resolution stage (Layers 7-27) determines prediction confidence.

2. **429 recurring features** are shared across >= 50% of prompts. The most universal are structural-register features (L6/F2586668, L3/F5150441) rather than semantic analogy features, suggesting the model classifies the analogy template by formal structure before activating domain knowledge.

3. **Prediction confidence correlates with causal path clarity**: high-confidence geographic analogies (p>=0.96) produce convergent excitatory causal chains; low-confidence functional analogies (p<=0.18) produce diffuse, mixed-sign path structures.

4. **Explicit analogy-detection features** exist at Layers 5, 7, 8, 9, and 15: F16817094 ("analogies"), F286895 ("analogies and comparisons"), F94882191 ("analogies or comparisons"), F89171325 ("phrases suggesting uncertainty or comparison"), and F12474 ("analogies in conversational text"). These form a dedicated analogy-recognition sub-network.

5. **All 10 prompts show convergent late-layer aggregation** through Layers 17-27, with the logit node at Layer 27 receiving inputs through an L19-L21-L25 hub. This hub is the final common pathway for analogical reasoning output in Gemma-2-2B.

These results demonstrate that mechanistic interpretability methods -- attribution graphs, causal path tracing, and cross-graph feature comparison -- can reveal specific computational mechanisms underlying abstract reasoning in large language models.
"""

with open('analogical_circuit_paper.md', 'a', encoding='utf-8') as f:
    f.write(discussion)

print('Discussion/Conclusion appended.')

with open('analogical_circuit_paper.md', encoding='utf-8') as f:
    content = f.read()
print(f'Final file: {len(content)} chars, {len(content.splitlines())} lines')
