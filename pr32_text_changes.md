# PR #32 — Text Changes: "Added experiment with 50 prompts"

**Branch:** `OlalekanAlagbe:claude/neuronpedia-graph-agent-main` → `KKrampis:claude/neuronpedia-graph-agent-main`  
**File changed:** `docs/autocircuit_manuscript_v5 - Olalekan.md`  
**Date generated:** 2026-06-04

---

## Summary of Changes

| Section                                    | Status                | Description                                                                      |
| ------------------------------------------ | --------------------- | -------------------------------------------------------------------------------- |
| §2.5 Discovery of Analogy-Concept Features | **Rewritten**         | Figure reference changed; new two-figure structure with Appendix A pointer       |
| §2.6 Phase 2 Definition                    | **Removed**           | Entire section deleted (iframe embed gone)                                       |
| §2.7 → §2.6 Circuit Definition             | **Renumbered**        | Content unchanged, section number shifted                                        |
| §3.5 Cross-Domain Generalization           | **Minor edit**        | Sentence trimmed; "as shown in Figure 1 below" removed                           |
| §3.6 Activation Magnitudes                 | **Replaced**          | Old section removed; new §3.6 on 50-prompt scaling experiment added              |
| §4 Discussion                              | **Heavily condensed** | Five subsections (4.1–4.5, ~600 words) collapsed into one paragraph (~100 words) |
| §6 Conclusions                             | **Rewritten**         | Incorporates 50-prompt scaling result and 67-feature stable core                 |
| Appendix A                                 | **Added**             | New appendix with 10 UI-screenshot figures across all five prompts               |

---

## Section-by-Section Changes

---

### §2.5 Discovery of Analogy-Concept Features — Rewritten

#### REMOVED text (current version):

> The most striking finding of the cross-graph analysis was not planned — it emerged from the data. The key features — L5 SAE#5793 ("analogies"), L8 SAE#13766
> ("analogies or comparisons"), and L9 SAE#13344 ("phrases suggesting uncertainty or comparison between two things") — were not specifically sought. They emerged
> from the cross-graph overlap analysis described in Section 2.3. Once the intersection feature set was computed, each feature's automated Neuronpedia
> explanation [8] was retrieved.
> The significance of these labels is their domain-agnosticism: all three features appear in attribution graphs for Berlin, Rome, and Tokyo (geographic capital
> analogies) and for teacher and bird (semantic role analogies) — **as shown in Figure 1 below**. This is consistent with the broader finding in the analogical
> reasoning literature that LLMs encode relational information in a domain-general manner [10, 11], and extends that behavioral finding to specific,
> causally-validated internal features. L8 SAE#13766 was additionally notable for having 21 appearances across the five graphs and an influence score of 0.533,
> placing it among the highest-influence recurring features.
> 
> ![Figure 1: L8 SAE#13766 active across all five analogy prompts — two semantic role analogies (top row) and three geographic capital analogies (bottom row). The consistent activation of this comparison-detection feature across both domains is direct evidence of a domain-general analogical reasoning mechanism.](feature_13766_all_prompts.png)

#### ADDED text (PR version):

> The most striking finding of the cross-graph analysis was not planned — it emerged from the data. The key features — L5 SAE#5793 ("analogies"), L8 SAE#13766 ("analogies or comparisons"), and L9 SAE#13344 ("phrases suggesting uncertainty or comparison between two things") — were not specifically sought. They emerged from the cross-graph overlap analysis described in Section 2.3. Once the intersection feature set was computed, each feature's automated Neuronpedia explanation [8] was retrieved.
> 
> The significance of these labels is their domain-agnosticism: all three features appear in attribution graphs for Berlin, Rome, and Tokyo (geographic capital analogies) and for teacher and bird (semantic role analogies). This is consistent with the broader finding in the analogical reasoning literature that LLMs encode relational information in a domain-general manner [10, 11], and extends that behavioral finding to specific, causally-validated internal features. **To illustrate, Figures 1a and 1b show L9 SAE#13344 and L8 SAE#13766 respectively, each captured inside the attribution graph UI for the *teacher* prompt — one representative from the five. The complete set of both features across all five prompts is provided in Appendix A.**
> 
> ![Figure 1a: L9 SAE#13344 ("phrases suggesting uncertainty or comparison between two things") active in the attribution graph for the teacher analogy prompt.](UI-GRAPHS/13344_doctor-is-to-hospital-as-teacher-is-to.png)
> 
> ![Figure 1b: L8 SAE#13766 ("analogies or comparisons") active in the attribution graph for the teacher analogy prompt.](UI-GRAPHS/13766_doctor-is-to-hospital-as-teacher-is-to.png)

**Key difference:** The single composite figure showing all five prompts at once (`feature_13766_all_prompts.png`) is replaced by two separate UI screenshots of the *teacher* prompt only (one per feature), with the full set moved to the new Appendix A. The sentence about L8 SAE#13766 having 21 appearances and influence 0.533 is removed (that information was specific to the 5-prompt analysis and is superseded by the scaling experiment data in §3.6).

---

### §2.6 Phase 2 Definition — REMOVED ENTIRELY

The following section no longer exists in the PR version:

> **§2.6 Phase 2 Definition**
> 
> Phase 2 is defined by two jointly applied criteria: layer position (5–9) and feature label content. Features in this layer range whose Neuronpedia labels
> explicitly reference analogies, comparisons, or relational structure constitute Phase 2. The four members are L5 SAE#5793 ("analogies"), L5 SAE#2141
> ("comparisons of people or figures using well-known public figures"), L8 SAE#13766 ("analogies or comparisons"), and L9 SAE#13344 ("phrases suggesting
> uncertainty or comparison between two things") — all four can be inspected interactively below:
> 
> `<iframe src="https://www.neuronpedia.org/list/cmoo57kqn001hut5fl6djy2fu?embed=true" title="Phase 2 – Direct Analogical Features" style="height: 400px; width: 100%;"></iframe>`
> 
> T

**Note:** The stray "T" at the end of the old §2.6 (an apparent draft artifact) is also removed.

**Consequence:** The old §2.7 "Circuit Definition and Causal Validation via Feature Steering" becomes the new **§2.6**, with content unchanged.

---

### §3.5 Cross-Domain Generalization — Minor edit

#### REMOVED phrase:

> "...and for teacher and bird (semantic role analogies) **— as shown in Figure 1 below**."

#### REPLACED with:

> "...and for teacher and bird (semantic role analogies)."

The back-reference to "Figure 1 below" is dropped because the figure presentation was restructured in §2.5.

---

### §3.6 — Replaced: Old section removed, major new section added

#### REMOVED — §3.6 Activation Magnitudes Build Through Layers:

> Average activation magnitudes of core circuit features increase substantially with layer depth. L0 structural features show activations of 1.5–6.4; the L5 analogy hub features reach 7.4–11.1; L8–L9 comparison detectors reach approximately 13.4; and L10–L13 integration features span 9.1–16.3. This monotonically increasing pattern is consistent with an accumulating signal as the relational structure is assembled rather than independent per-layer computation.

#### ADDED — §3.6 Circuit Stability Across Scaled and Syntactically Diverse Prompts (entire new section):

> To validate that the shared circuit identified in Section 3.1 is not an artifact of using only five similar prompts, an extensive scaling experiment was performed. The central question is this: if we keep adding new analogical prompts — including versions phrased very differently from the original format — do the same features keep showing up?
> 
> The experiment generated attribution graphs for **50 prompts** in total. Crucially, from the second batch onward, the prompts were not just new examples of the same template — they were rephrased into three syntactically distinct surface forms alongside the original:
> 
> | Surface Form          | Example                                                            |
> | --------------------- | ------------------------------------------------------------------ |
> | Standard X-to-Y       | `Paris is to France as Berlin is to`                               |
> | Diverse-A (Just as…)  | `Just as Paris is the capital of France, Berlin is the capital of` |
> | Diverse-B (Found in…) | `Doctors are found in hospitals. Teachers are found in`            |
> | Diverse-C (The way…)  | `The way a fish lives in water, a bird lives in`                   |
> 
> At each milestone (N = 5, 10, 20, 30, 40, 50), the strictest possible threshold was applied: a feature must appear in **every single** attribution graph at that point.
> 
> | N   | Recurring features (k = ALL) | Drop from previous |
> | --- | ---------------------------- | ------------------ |
> | 5   | **180**                      | —                  |
> | 10  | **116**                      | −64 (−35.6 %)      |
> | 20  | **86**                       | −30 (−25.9 %)      |
> | 30  | **77**                       | −9 (−10.5 %)       |
> | 40  | **70**                       | −7 (−9.1 %)        |
> | 50  | **67**                       | −3 (−4.3 %)        |
> 
> The curve has a clear two-phase shape: **rapid contraction** from N=5→20 (surface-token features drop out), then **near-plateau** from N=20→50 (only 19 further features lost across 30 additional prompts).
> 
> **The five directly analogical features all survive.** Within the 67-feature core, five features carry Neuronpedia labels that explicitly describe analogical or comparative reasoning, all present in all 50 graphs:
> 
> | Feature    | Appearances | Avg. Influence | Label                                                                |
> | ---------- | ----------- | -------------- | -------------------------------------------------------------------- |
> | L13 #10969 | 62          | 0.713          | "comparisons between disciplines and relationships between concepts" |
> | L9 #13344  | 116         | 0.683          | "phrases suggesting uncertainty or comparison between two things"    |
> | L9 #14231  | 53          | 0.683          | "words representing comparisons and relationships"                   |
> | L7 #749    | 80          | 0.652          | "analogies and comparisons"                                          |
> | L5 #2141   | 62          | 0.639          | "comparisons of people or figures using well-known public figures"   |
> 
> Two of these — L9 #14231 and L7 #749 — are **newly identified** by the scaling experiment (not visible in the 5-prompt analysis). The other three were already known.

**Note:** The activation magnitude content from the old §3.6 is not lost — it was already covered in §2.4 (Table 1) and the Introduction's description of Evidence 2. It is not duplicated in the PR version.

---

### §4 Discussion — Heavily Condensed

#### REMOVED — Five subsections totaling ~600 words:

**§4.1 The Analogical Reasoning Circuit in Gemma-2-2B** (~130 words)  
Discussed the circuit spanning all 26 layers, the analogy features at layers 5/8/9/13, the Phase 2 collective suppression result, and the extension to SAE feature level.

**§4.2 The Role of Formal Text Features** (~130 words)  
Explained the "code and legal text" features via two hypotheses: functional (formal syntax detector) and training data (analogy format co-occurs with legal/code text in SAT prep materials).

**§4.3 Comparison with the Capital City Recall Circuit** (~90 words)  
Noted overlap (formal-text features shared with factual recall) and divergence (L5/L8 analogy features specific to analogical task, absent from factual recall circuit).

**§4.4 Relation to Anthropic's Attribution Graph Methodology** (~100 words)  
Drew parallels with Anthropic's *On the Biology of a Large Language Model* [12]: both find staged computation, both use steering validation; contrasted manual supernodes (Anthropic) vs. automated intersection (this work).

**§4.5 Redundancy as a Property of Well-Learned Computation** (~60 words)  
Argued that inverse confidence/fragility relationship reflects a general principle of how transformers allocate computational resources.

#### ADDED — Single condensed paragraph (~100 words):

> Gemma-2-2B implements analogical reasoning through a stable, distributed circuit with a clear three-phase architecture — structural template parsing, analogy recognition, and relational integration — anchored by five features that activate consistently regardless of how the analogy is phrased. The collective suppression experiments show that the analogy recognition phase is causally necessary for relational transfer: remove it, and the model echoes the source-pair answer instead of completing the new one. The inverse relationship between prediction confidence and circuit fragility further suggests that well-learned associations are protected by redundant parallel causal paths, while low-confidence predictions rely on fragile non-redundant chains — a pattern that may reflect a general principle of how transformers allocate computational resources across tasks of varying difficulty.

---

### §6 Conclusions — Rewritten

#### REMOVED (current version):

> We have identified and characterized the **analogical reasoning circuit in Gemma-2-2B** using SAE attribution graphs from the Neuronpedia platform [8]. Six key conclusions follow. First, a stable shared circuit exists, identified by common feature IDs: 180 features — identified by stable *(layer, feature index)* pairs — appear in all five independently generated attribution graphs. Second, dedicated analogy features exist at layers 5, 8, 9, and 13: these features have Neuronpedia explanations explicitly referencing analogies, comparisons, and relational concepts, providing direct SAE-level evidence of interpretable analogy-concept features in a large language model. Third, the circuit exhibits a three-phase architecture, identified by label semantics and validated causally: circuit template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13), with activation magnitude increasing through the sequence. Fourth, cross-domain generalization is confirmed: the same core features, including L5 SAE#5793 ("analogies"), activate for both geographic and semantic role analogies — a domain-agnostic relational reasoning primitive consistent with behavioral findings [1, 10, 11]. Fifth, Phase 2 implements relational transfer, collectively but not individually: simultaneous suppression collapses every circuit, with capital analogies reverting to the source-pair answer. Sixth, circuit fragility tracks prediction confidence: high-confidence predictions route through redundant parallel causal paths (1–4 necessary features) while low-confidence predictions rely on fragile non-redundant chains (up to 8/10 necessary).

#### ADDED (PR version):

> We have identified and characterized the analogical reasoning circuit in Gemma-2-2B using SAE attribution graphs from the Neuronpedia platform [8]. Across five initial prompts, 180 features — identified by stable (layer, feature index) pairs — appear in every attribution graph, forming the basis of the circuit. **A further scaling experiment tested whether those features would survive 50 prompts phrased across four syntactically distinct surface forms; the circuit converged to a stable 67-feature core, with five features carrying explicitly analogical labels present across all 50 graphs.** The circuit exhibits a three-phase architecture — structural template parsing (L0–L4), analogy recognition (L5–L9), and relational integration (L10–L13) — with activation magnitude increasing through each phase. Cross-domain generalization is confirmed: the same core features activate for both geographic and semantic role analogies, consistent with a domain-agnostic relational reasoning primitive. Phase 2 implements relational transfer collectively but not individually: simultaneous suppression collapses every circuit, with capital analogies reverting to the source-pair answer. Finally, circuit fragility tracks prediction confidence — high-confidence predictions route through redundant parallel causal paths while low-confidence predictions rely on fragile non-redundant chains.

**Key differences:**

- "Six key conclusions follow" numbered list structure → single flowing paragraph
- Bold on "analogical reasoning circuit in Gemma-2-2B" removed
- Scaling experiment result (50 prompts → 67-feature stable core) added as a new finding
- Specific feature IDs removed from conclusions (e.g., "L5 SAE#5793") for cleaner prose
- Quantitative detail on fragility ("up to 8/10 necessary") removed

---

### Appendix A — New (entire section added)

The PR adds a new appendix after the BibTeX block:

> ## Appendix A: Analogy-Concept Features Across All Five Prompts
> 
> The figures below show L9 SAE#13344 and L8 SAE#13766 as they appear inside the Neuronpedia attribution graph UI for each of the five original prompts.
> 
> ### A.1 L9 SAE#13344 — "phrases suggesting uncertainty or comparison between two things"
> 
> [5 figures: berlin, rome, tokyo, teacher, bird]
> 
> ### A.2 L8 SAE#13766 — "analogies or comparisons"
> 
> [5 figures: berlin, rome, tokyo, teacher, bird]

This consolidates the full cross-prompt visualization that was previously attempted with the single composite `feature_13766_all_prompts.png`, replacing it with individual UI screenshots stored in `docs/UI-GRAPHS/`.

---

## New Binary Files Added

| File                                           | Purpose                                         |
| ---------------------------------------------- | ----------------------------------------------- |
| `docs/UI-GRAPHS/13344_berlin.png`              | L9 SAE#13344 in Berlin graph UI                 |
| `docs/UI-GRAPHS/13344_rome.png`                | L9 SAE#13344 in Rome graph UI                   |
| `docs/UI-GRAPHS/13344_tokyo.png`               | L9 SAE#13344 in Tokyo graph UI                  |
| `docs/UI-GRAPHS/13344_teacher.png`             | L9 SAE#13344 in teacher graph UI                |
| `docs/UI-GRAPHS/13344_bird.png`                | L9 SAE#13344 in bird graph UI                   |
| `docs/UI-GRAPHS/13766_berlin.png`              | L8 SAE#13766 in Berlin graph UI                 |
| `docs/UI-GRAPHS/13766_rome.png`                | L8 SAE#13766 in Rome graph UI                   |
| `docs/UI-GRAPHS/13766_tokyo.png`               | L8 SAE#13766 in Tokyo graph UI                  |
| `docs/UI-GRAPHS/13766_teacher.png`             | L8 SAE#13766 in teacher graph UI                |
| `docs/UI-GRAPHS/13766_bird.png`                | L8 SAE#13766 in bird graph UI                   |
| `docs/UI-GRAPHS/feature_13344_all_prompts.png` | Updated composite for feature 13344             |
| `docs/scaling_curve.png`                       | New: plot of feature count vs. N prompts (5→50) |
| `docs/feature_6261_location and place.png`     | New feature visualization                       |
