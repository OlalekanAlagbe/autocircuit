# Neuronpedia Circuit Analysis Pipeline

Automated pipeline for analyzing factual knowledge circuits in LLMs via SAE attribution graphs. Traces backward through Neuronpedia circuit graphs to identify bottleneck features, runs cross-domain statistical analysis, and validates causal influence via steering experiments.

Supported models: GEMMA-2-2B (26 layers), QWEN3-4B (36 layers).

## Key Findings

Analysis of 60 circuits (30 prompts × 2 models) across chemistry, geography, and history:

- **Bottleneck depth = architecture, not domain.** GEMMA bottlenecks cluster at L5-7 (~22%), QWEN at L22-25 (~65%), regardless of knowledge category.
- **94.1% circuit redundancy.** Minimal pathway extraction shows only 5.9% of nodes are essential for preserving predictions.
- **Bottleneck features are infrastructure, not knowledge.** CODE (20%) and LANGUAGE (20%) dominate; 54% of features appear in 2+ domains.
- **Steering reveals a three-tier dissociation.** 80 experiments show essential-pathway features produce strong distributional perturbations (mean KL=1.448) but circuit redundancy absorbs most before the output layer. Neither pathway topology nor cross-circuit frequency predicts text-level changes — output determinism does.

Full paper: `docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md`

## Quick Start

```bash
pip install -r config/requirements.txt
# Add your Neuronpedia API key to config/neuronpedia_config.yaml
python run_full_pipeline.py --prompt "The chemical symbol for Argon is" --model gemma-2-2b
```

## Supernode Pipeline

`scripts/supernode_pipeline.py` is a standalone end-to-end tool that takes a text prompt, generates an attribution graph, detects communities via multi-algorithm clustering, labels them with Neuronpedia explanations, and outputs an annotated JSON with cantor-paired features ready for direct upload to Neuronpedia.

```bash
# Single prompt
python scripts/supernode_pipeline.py "The capital of France is"

# With programmatic upload to Neuronpedia
python scripts/supernode_pipeline.py "The Titanic sank in the year" --upload

# Batch mode
python scripts/supernode_pipeline.py --batch-file prompts.txt --upload

# Upload existing graph
python scripts/supernode_pipeline.py --upload-only output/gemma-2-2b_my-prompt/annotated_graph.json
```

Output: `output/{model}_{prompt_slug}/annotated_graph.json`, `supernode_report.md`, `community_visualization.png`, `upload_result.json` (after `--upload`).

The standalone version with full documentation is at `../supernode_detector/`. See the [Supernode Detector tool paper](../supernode_detector/docs/SUPERNODE_PIPELINE_TOOL_PAPER.md) for algorithmic details.

## Pipeline Orchestrator

`run_full_pipeline.py` runs the analysis pipeline end-to-end.

```
--prompt TEXT       Prompt to analyze (required)
--model MODEL       gemma-2-2b | qwen3-4b | both (default: both)
--advanced          Also run advanced analysis steps 5-10
--steer-quick       Run Stage 3 steering validation (quick: top 5 features, ~20 min)
--steer-full        Run Stage 3 steering validation (full: all features, ~10 hours)
--steer-rate-delay  Seconds between steering API calls (default: 36 = 100/hr)
--skip-api          Skip Neuronpedia API queries in Stage 1.5
--api-limit N       Max API queries for Stage 1.5 (default: 100)
```

Execution order: Steps 1 > 2 > 3 > 3b > 4 > Stage 1.5 > Stage 2. With `--steer-quick/--steer-full`: also Stage 3. With `--advanced`: also Steps 5, 7, 8, 9, 10.

## Core Scripts (`scripts/`)

**`1_generate_graph.py`** -- Generates a raw SAE attribution graph via the Neuronpedia circuit tracer API.

**`2_convert_graph.py`** -- Converts raw Circuit Tracer JSON into the pipeline's standardized format.

**`3_analyze_circuit.py`** -- Structural analysis: Louvain community detection, betweenness centrality, optional Neuronpedia descriptions.

**`3b_traceback_paths.py`** -- Core innovation: backward BFS from output to input with geometric decay scoring (score^0.8 per hop). Identifies bottleneck features where 60%+ of critical paths converge.

**`4_visualize.py`** -- Generates 8 PNG visualizations per circuit (supernode overview, layer distribution, activation heatmap, etc.).

**`5_steering_validation.py`** -- Stage 3: causal validation via Neuronpedia Steering API. Amplifies/suppresses bottleneck features and measures output disruption.

**`supernode_pipeline.py`** -- Standalone end-to-end supernode detection (see above).

**`supernode_detector.py`** -- Community detection module using Louvain clustering with configurable cluster sizes.

**`annotate_features_v2.py`** -- Keyword-based semantic classifier (SYNTAX, SEMANTICS:CODE/CONCEPT/GEOGRAPHIC/ENTITY/TEMPORAL, POLYSEMANTIC).

**`feature_description_fetcher.py`** -- Neuronpedia API queries with ID mapping (`neuronpedia_id = circuit_tracer_id % 16384`) and rate limiting.

**`path_manager.py`** -- Centralized path resolution for the `data/prompts/<model>_<slug>/` directory structure.

**`pipeline_constants.py`** -- Shared constants: layer counts, SAE dictionary sizes, layer group boundaries, convergence threshold (0.6).

## Stage 2 Analysis Scripts (`scripts/stage_2_*.py`)

These scripts power the cross-domain analysis paper. Each reads from `data/` and writes results/figures to `data/stage_2_*/`.

| Script | Purpose |
|--------|---------|
| `stage_2_cross_category_analysis.py` | Main cross-domain statistical analysis (ANOVA, regression, effect sizes) |
| `stage_2_minimal_pathways.py` | Essential pathway extraction (94.1% redundancy finding) |
| `stage_2_multi_algorithm_validation.py` | Community detection with 4 algorithms (Louvain, Leiden, Infomap, Label Prop) |
| `stage_2_polysemanticity_analysis.py` | Semantic purity vs polysemanticity of bottleneck features |
| `stage_2_expanded_steering.py` | Expanded D5 steering experiments (50 experiments, 10 features) |
| `stage_2_essential_pathway_steering.py` | Essential-pathway steering validation (30 experiments, 5 features) |
| `stage_2_layer_energy_analysis.py` | Per-layer activation energy profiling |
| `stage_2_edge_flow_analysis.py` | Edge weight flow patterns across layers |
| `stage_2_feature_coactivation.py` | Feature co-activation network analysis |
| `stage_2_output_decomposition.py` | Output node contribution decomposition |
| `stage_2_enhanced_analysis.py` | Enhanced circuit metrics (8 categories, 16 metrics) |
| `stage_2_statistical_deepdive.py` | Extended statistical tests and outlier analysis |
| `stage_2_outlier_analysis.py` | Circuit outlier identification |
| `stage_2_batch_annotate.py` | Batch Neuronpedia annotation enrichment |
| `stage_2_paper_figures.py` | Publication-quality figures for the paper |
| `stage_2_*_figures.py` | Additional figure generation scripts |

## Advanced Analysis (`scripts/advanced_analysis/`)

Deeper per-circuit analysis. Run via `--advanced` flag or individually.

| Script | Purpose |
|--------|---------|
| `3_analyze_circuit_multi.py` | Batch analysis with multi-algorithm comparison |
| `5_compare_prompts.py` | Cross-prompt circuit structure comparison |
| `6_identify_targets.py` | High-leverage intervention target identification |
| `7_extract_minimal_pathways.py` | Minimum viable circuit extraction |
| `8_supernode_evolution.py` | Supernode composition evolution across layers |
| `9_steering_analysis.py` | Steering potential analysis at bottleneck points |
| `10_polysemanticity_analysis.py` | Feature polysemanticity measurement |

## Skills (`skills/`)

Claude Code skills for interactive pipeline usage:

- **neuronpedia-convert** -- Convert raw graphs to pipeline format
- **neuronpedia-compare** -- Compare circuits across prompts/models
- **neuronpedia-validate** -- Validate circuit analysis results
- **neuronpedia-visualize** -- Generate and inspect visualizations
- **neuronpedia-analyze** -- Run analysis steps on existing circuit data
- **neuronpedia-fetch** -- Fetch feature descriptions from Neuronpedia API
- **steering-validate** -- Validate bottleneck features via Neuronpedia Steering API
- **circuit-report** -- Run full pipeline and generate comprehensive analysis report

## Key Concepts

- **Traceback graphing**: Backward BFS from output to input with geometric decay (0.8) to score feature importance without exponential path explosion.
- **Bottleneck convergence**: Features appearing in 60%+ of critical paths act as information gates. Layer position determines what information survives to output.
- **Feature ID mapping**: Neuronpedia uses 16k SAE dictionaries, so `neuronpedia_id = circuit_tracer_id % 16384`. Only GEMMA features are currently queryable via the feature API.
- **Essential pathways**: Minimal subgraph preserving model prediction. Only 5.9% of nodes are essential; the rest provide redundancy.
- **Cross-circuit features**: Features appearing as bottlenecks across multiple prompts — universal circuit infrastructure rather than prompt-specific processing.
- **Steering validation**: Causal testing via Neuronpedia Steering API. Amplify/suppress features and measure KL divergence + text change rate.

## Output Structure

All generated data goes to `data/` (gitignored).

```
data/
  prompts/<model>_<prompt-slug>/
    1_generation/         Raw API graph
    2_conversion/         Standardized pipeline format
    3_analysis/           Circuit analysis + traceback paths
    4_visualizations/     PNG figures

  stage_1_5_bottleneck_library.json   Cross-circuit bottleneck library
  stage_2_analysis/                   Cross-domain statistical results
  stage_2_minimal_pathways/           Essential pathway data
  stage_2_multi_algorithm/            Community detection validation
  stage_2_polysemanticity/            Semantic purity analysis
  stage_2_steering_validation/        Expanded steering (D5)
  stage_2_essential_pathway_steering/  Pathway steering results
  stage_2_figures/                    Publication figures
  stage_3_steering/                   Original steering validation
```

## Documentation

- `docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md` -- Main paper: cross-domain circuit analysis (18 findings)
- `docs/papers/TRACEBACK_GRAPHING_PAPER.md` -- Original traceback graphing methodology paper
- `docs/papers/SEMANTIC_TAXONOMY_METHODOLOGY.md` -- Feature classification methodology
- `docs/papers/TOKEN_ATTRIBUTION_VALIDATION.md` -- Token attribution validation study
