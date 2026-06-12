# PR Review Session — autocircuit

---

## PR List

Latest open PRs on the autocircuit repository:

1. **PR #33** — "Restructure §3.7 causal validation as a claim-by-claim ledger + cross-prompt circuit test" by `amhw460`, opened 2026-06-10
2. **PR #32** — "Added experiment with 50 prompts" by `OlalekanAlagbe`, opened 2026-05-10

---

## PR #33 Review — "Restructure §3.7 causal validation as a claim-by-claim ledger + cross-prompt circuit test"

**Author:** amhw460 | **Files changed:** 27 | **+38,090 / −215**

### Overview

This PR does two distinct things: (1) adds new experiment code and data for validating the 180-feature analogical circuit across all five prompts, and (2) rewrites §3.7 of the paper to be epistemically more honest about what the steering evidence does and does not establish. The framing is strong and the experimental design is sound.

### Strengths

- **Epistemically honest rewrite.** The claim-by-claim ledger in §3.7.4 (demonstrated / supported / not tested) is exactly the right format. Explicitly recording that sufficiency is *not established* is commendable and reviewer-proof.
- **Cross-prompt design is well-motivated.** Running the same circuit ablation + matched-null contrast on all five prompts, rather than just Berlin, directly addresses the biggest generalization objection.
- **Crash-safe, resumable experiment scripts.** Writing JSON after every API call and skipping already-completed prompts is good engineering for fragile external APIs.
- **Retry logic is appropriate.** Handling 429 (rate limit) and transient 500s separately, with different strategies (indefinite vs. bounded), is correct.
- **`deep_validation.py` is well-scoped.** Each stage maps to a specific reviewer objection, and the docstring is clear about what each stage tests and why.
- **`.gitignore` was missing and is now added.**

### Issues & Suggestions

**Code quality**

- `deep_validation.py:load_circuit()` uses `json.load(open(CIRCUIT_JSON))` without a context manager — the file handle leaks. Should be `with open(CIRCUIT_JSON) as f: json.load(f)`.
- `deep_validation.py` defines `is_error`, `is_429`, `is_500` as one-liners on a single line (`def is_error(r): return ...`) — fine functionally but inconsistent with the rest of the codebase style.
- `safe_steer` in `deep_validation.py` accepts a `strength_multiplier` parameter but `cross_prompt_validation.py`'s version does not — the two implementations have diverged. A shared utility in `run_steering.py` or a shared module would prevent drift.
- In `cross_prompt_validation.py`, the `predicts_expected` function does case-insensitive comparison on just the first token — this will silently fail if the expected answer is multi-token (e.g. `"cutting"` might tokenize to `[" cut", "ting"]`). Worth a comment acknowledging this.

**Data files committed to the repo**

- `analogical_circuit_180features.json`, `cross_prompt_results.json`, `null_control.json`, `results.json`, `failed_features.json` — large JSON data files committed directly. These are fine for reproducibility but will bloat the repo over time. Consider whether `*.log` files (`run.log`, `retry.log`, etc.) need to be tracked — they add noise without adding reproducibility value.
- `exp4_scale/circuits/*.json` — these look like pre-existing files that were re-committed or modified. Worth confirming these aren't accidentally overwriting results from a prior experiment.

**`path_patching.py` (exp9)**

- The file is clearly marked `SCAFFOLD / UNTESTED` in its docstring, which is honest. However, it is included in this PR's diff. Untested scaffold code in a research paper repo can be misleading if someone runs it without reading the header. Consider moving it to a `_wip/` directory or adding a top-of-file `raise NotImplementedError` guard so it can't be accidentally run.
- `TRANSCODER_RELEASE` and `TRANSCODER_SAE_ID` are marked `# VERIFY` — the correct IDs should be looked up before this is merged, or the file should stay out of the paper's supporting code.

**Paper (docs/index.html)**

- The PR description says §3.7.2/§3.7.3 are "reframed" and §3.7.4 is "replaced" — these changes are in a large HTML diff that's hard to review. It would be worth confirming the claim-by-claim ledger in §3.7.4 is complete (all major claims from the abstract/introduction appear in it) and that the Appendix A.3/A.4/A.6 rewrites match the new experiment results.

**Missing experiment (exp7)**

- `cross_prompt_validation.py` references `experiments/exp7_remaining_analogy_validation/graphs/` for `analog_rome.json`, `analog_tokyo.json`, etc. — but exp7 doesn't appear to be in this PR's diff. If those graph files don't exist in the base branch, this script will fail on checkout. Worth verifying exp7 is already merged.

### Summary

This is a solid, well-motivated PR that substantially strengthens the paper's causal validation section. The main asks before merging:

1. Fix the file-handle leak in `deep_validation.py`.
2. Verify the exp7 graph files exist on the base branch.
3. Decide what to do with `path_patching.py` (guard it or move it to `_wip/`).
4. Confirm the `*.log` files are intentionally tracked.

---

## PR #32 Review — "Added experiment with 50 prompts"

**Author:** OlalekanAlagbe | **Files changed:** 14 | **+77 / −49**

### Overview

This PR adds the results of a 50-prompt scaling experiment to the manuscript. It extends the original 5-prompt circuit analysis by testing 50 prompts across four syntactically distinct surface forms, showing the 180-feature circuit converges to a stable 67-feature core. It also adds attribution-graph UI screenshots for the two key analogy features across all five original prompts, and collapses the §4 Discussion into a single-paragraph summary.

### Strengths

- **The scaling experiment is the right experiment.** Showing the circuit stabilizes under syntactic diversity (not just more instances of the same template) is a strong generalization argument. The two-phase convergence curve (rapid drop, then plateau) is a compelling result on its own.
- **The table format for surface forms is clear** and makes it easy to see how diverse the 50-prompt set actually is.
- **Moving the UI screenshots to an Appendix** is the right call — they're evidence, not narrative.
- **The convergence of all 5 directly-analogical features** through all 50 prompts is the headline result and is stated clearly.

### Issues & Suggestions

**Manuscript content**

- **§4 Discussion was removed entirely** (4.1–4.5 deleted, replaced with a single paragraph). This is a major content loss. Sections 4.2 (formal text features / polysemanticity), 4.3 (comparison with factual recall circuit), 4.4 (relation to Anthropic's attribution graph methodology), and 4.5 (redundancy principle) contained substantive scientific discussion that is not preserved in the one-paragraph replacement. This feels like an accidental deletion rather than a deliberate editorial decision — confirm this is intentional before merging.
- **§3.6 was also replaced** — the original "Activation Magnitudes Build Through Layers" section (the monotonically increasing activation pattern finding) is gone, replaced by the new scaling experiment section. The activation magnitude finding was a concrete quantitative result; if it's being dropped, the data should still appear somewhere (table or Appendix).
- **Missing `scaling_curve.png` reference** — the manuscript references `scaling_curve.png` as Figure S1, but the image path in the diff is `docs/scaling_curve.png`. If the manuscript is also in `docs/`, the relative path should work, but this should be verified by rendering the document.
- **Missing experiment code.** This PR adds results (tables, images, text) but no code for how the 50 prompts were generated, how the attribution graphs were fetched, or how the N=5/10/20/30/40/50 milestone counts were computed. Without the code, the experiment is not reproducible. At minimum, a script or notebook should accompany the results.
- **No PR description.** The body is empty — makes it hard to understand context or review intent without reading the full diff.

**Document quality**

- The appendix file ends without a newline (`\ No newline at end of file`) — minor but should be fixed.
- Section numbering may be off: §2.6 was originally §2.6 "Phase 2 Definition," which is now deleted. The manuscript should be checked for any downstream section number references that broke.
- `feature_6261_location and place.png` is added to `docs/` but is never referenced in the diff — likely an accidental addition.

### Summary

PR #32 adds genuinely valuable content (the 50-prompt scaling experiment is a strong result), but the deletion of §4 Discussion (4.1–4.5) appears to be an unintended loss of significant scientific content. Before merging:

1. **Confirm the §4 deletion is intentional** — if not, restore the removed subsections.
2. **Add experiment code** for the 50-prompt analysis so results are reproducible.
3. **Remove `feature_6261_location and place.png`** if it's not referenced anywhere.
4. **Add a PR description** explaining what changed and why.

---

## Files That Will Be Overwritten If PRs Are Merged

### PR #33 — 27 files total

**Modified (1 — existing file overwritten):**
- `docs/index.html` — §3.7 rewrite + cascading edits to §2.4, §3.3, §5, §6

**New files added (26):**

*Experiment 8 — 180-feature validation:*
- `experiments/exp8_180feature_validation/run_validation.py`
- `experiments/exp8_180feature_validation/cross_prompt_validation.py`
- `experiments/exp8_180feature_validation/deep_validation.py`
- `experiments/exp8_180feature_validation/null_control.py`
- `experiments/exp8_180feature_validation/retry_failed.py`
- `experiments/exp8_180feature_validation/analogical_circuit_180features.json`
- `experiments/exp8_180feature_validation/cross_prompt_results.json`
- `experiments/exp8_180feature_validation/null_control.json`
- `experiments/exp8_180feature_validation/results.json`
- `experiments/exp8_180feature_validation/failed_features.json`
- `experiments/exp8_180feature_validation/cross_prompt.log`
- `experiments/exp8_180feature_validation/null_control.log`
- `experiments/exp8_180feature_validation/retry.log`
- `experiments/exp8_180feature_validation/run.log`

*Experiment 9 — path patching scaffold:*
- `experiments/exp9_path_patching/path_patching.py`
- `experiments/exp9_path_patching/README.md`

*Experiment 4 — circuit JSON data files:*
- `experiments/exp4_scale/circuits/analogy_graph.json`
- `experiments/exp4_scale/circuits/antonym_graph.json`
- `experiments/exp4_scale/circuits/IOI_graph.json`
- `experiments/exp4_scale/circuits/factual_causal_graph.json`
- `experiments/exp4_scale/circuits/factual_chemistry_graph.json`
- `experiments/exp4_scale/circuits/factual_geography_graph.json`
- `experiments/exp4_scale/circuits/factual_science_graph.json`
- `experiments/exp4_scale/circuits/greater_than_graph.json`

*Experiment 2:*
- `experiments/exp2_steering/run_steering.py`

*Repo config:*
- `.gitignore`

> **Note:** `docs/index.html` does not currently exist in the local repo — it is generated at CI time by `build.py` from `docs/content.md`. So PR #33 technically adds a new file, but it will be overwritten on the next CI run.

### PR #32 — 14 files total

**Modified (1 — existing file overwritten):**
- `docs/autocircuit_manuscript_v5 - Olalekan.md` — 50-prompt scaling experiment added, §4 Discussion collapsed

**New files added (13):**

*UI screenshot appendix images:*
- `docs/UI-GRAPHS/13344_paris-is-to-france-as-berlin-is-to.png`
- `docs/UI-GRAPHS/13344_paris-is-to-france-as-rome-is-to.png`
- `docs/UI-GRAPHS/13344_paris-is-to-france-as-tokyo-is-to.png`
- `docs/UI-GRAPHS/13344_doctor-is-to-hospital-as-teacher-is-to.png`
- `docs/UI-GRAPHS/13344_fish-is-to-water-as-bird-is-to.png`
- `docs/UI-GRAPHS/13766_paris-is-to-france-as-berlin-is-to.png`
- `docs/UI-GRAPHS/13766_paris-is-to-france-as-rome-is-to.png`
- `docs/UI-GRAPHS/13766_paris-is-to-france-as-tokyo-is-to.png`
- `docs/UI-GRAPHS/13766_doctor-is-to-hospital-as-teacher-is-to.png`
- `docs/UI-GRAPHS/13766_fish-is-to-water-as-bird-is-to.png`
- `docs/UI-GRAPHS/feature_13344_all_prompts.png`
- `docs/scaling_curve.png`
- `docs/feature_6261_location and place.png` *(unreferenced — likely accidental)*

**No overlap between the two PRs** — they touch completely different files, so merging both will not produce conflicts.

---

## Scientific Contributions of PR #33

### 1. Cross-prompt circuit ablation (the core new experiment)

The original validation only tested the 180-feature circuit on a single prompt ("Paris is to France as Berlin is to"). PR #33 runs the same two-test design — **full circuit ablation vs. a size- and strength-matched random null** — on all five defining analogy prompts. The result is striking: ablating the circuit collapses every prompt to the template connective `" to"` at near-identical confidence (logprob −0.026 to −0.030, σ = 0.0015, 5/5), while the matched null produces idiosyncratic out-of-distribution tokens. This is the first cross-prompt evidence that the circuit is the shared causal mechanism, not a Berlin-specific artifact.

### 2. Matched-null control design

The null control is not just "ablate random features" — it samples 180 features **from that prompt's own attribution graph**, stratified to match the circuit's size and strength. This rules out the alternative explanation that ablation works simply because you're suppressing 180 features at −80 strength (a very blunt intervention). The contrast between structured collapse (circuit) vs. unstructured collapse (null) is what makes the result interpretable.

### 3. Layer-stratified ablation addressing the L0 token-deletion confound

`deep_validation.py` (Stage A, early results reported) tests whether the observed collapse is just "you erased the input token representations" — a legitimate objection since 40/180 circuit features are at L0 (the embedding layer). The key result: **ablating the circuit with L0 excluded (140 L1+ features) still collapses the answer on 4/5 prompts**, directly refuting the token-deletion explanation and localizing causal necessity to mid/late layers.

### 4. Epistemic rewrite of the causal validation section

The §3.7 rewrite is itself a scientific contribution: it introduces a **claim-by-claim verdict ledger** that explicitly categorizes each paper claim as *demonstrated*, *supported*, or *not tested*, and records sufficiency as not established. This is a methodological honesty standard that is uncommon in mechanistic interpretability papers and directly addresses the critique that steering evidence conflates necessity with sufficiency.

### 5. Scaffold for path patching (exp9)

`path_patching.py` lays out the design for **denoising path patching** (Wang et al. 2022 / Goldowsky-Dill et al. 2023) — the gold-standard causal test that would establish whether information actually *flows along* the Phase 1 → Phase 2 edge, rather than just showing each phase is individually necessary. This is marked untested (requires GPU + HuggingFace access), but the experimental design is complete and connects the paper's steering results to the broader mechanistic interpretability literature on mediation.

**In short:** PR #33 upgrades the paper from "we showed the circuit is necessary on one prompt" to "we showed necessity generalizes across all five prompts with a controlled null, and we've begun ruling out the two most serious confounds (token deletion and blunt ablation magnitude)."

---

## Website Build — How It Works

The GitHub Actions workflow (`.github/workflows/pages.yml`) triggers on pushes to `main` or `claude/neuronpedia-graph-agent-main` and:

1. Runs `python3 docs/build.py` — builds `docs/index.html` and `docs/presentation.html` from **`docs/content.md`**
2. Runs `python3 docs/build_pdf.py` — builds a PDF
3. Runs `python3 docs/build_supplementary.py` — builds supplementary material
4. Uploads the entire `docs/` folder as the deployed GitHub Pages artifact

**`docs/content.md` is the authoritative source for the website.** Neither PR #33 nor PR #32 touches it.

### Implications

- **PR #33** adds a pre-built `docs/index.html` directly. The next CI run will overwrite it by regenerating from `content.md`. To make PR #33's paper edits appear on the live site, they must be ported into `content.md`.
- **PR #32** edits `docs/autocircuit_manuscript_v5 - Olalekan.md`, which `build.py` does not read — it has no effect on the published website.

### How to incorporate PR #33's text changes into the website

The sections that need to be updated in `docs/content.md` are:

| Section | What changed in PR #33 |
|---|---|
| §2.4 | Rewritten to reflect corrected causal framing |
| §3.7 preamble | New governing-logic paragraph (necessity vs. sufficiency, attribution is correlational) |
| §3.7.1 | Rebuilt around the matched-contrast design + 5-prompt cross-prompt table |
| §3.7.2 / §3.7.3 | Reframed as validating the three-phase *architecture*, not the 180-feature circuit |
| §3.7.4 | Replaced with the claim-by-claim verdict ledger |
| §5 Limitations | Updated to reflect new framing |
| §6 Conclusions | Updated |
| Appendix A.3/A.4/A.6 | New appendix sections (sufficiency, specificity demoted; per-prompt circuit-vs-null detail) |

**Steps to apply:**
1. Check out PR #33 locally (or merge it) to get `index.html` as amhw460 wrote it.
2. For each changed section, read the HTML and write the equivalent markdown into `content.md`.
3. Commit the updated `content.md` — CI will regenerate `index.html` correctly from it.
