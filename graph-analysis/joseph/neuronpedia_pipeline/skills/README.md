# Neuronpedia Pipeline Skills

**Claude Code skills for LLM circuit analysis using Neuronpedia attribution graphs**

---

## Overview

This folder contains 8 Claude Code skills that provide one-command access to the Neuronpedia analysis pipeline. Each skill corresponds to a script in the pipeline and handles a specific step of circuit analysis.

---

## Skills

### 1. `/neuronpedia-fetch`
Generate attribution graphs from Neuronpedia API

- **What**: Connects to Neuronpedia Circuit Tracer, generates graph, downloads from S3
- **Output**: `real_{prompt}.json` (4-6 MB, 500-1500 nodes)
- **Time**: ~10-15 seconds
- **Script**: `scripts/1_generate_graph.py`

### 2. `/neuronpedia-convert`
Convert raw graphs to pipeline format

- **What**: Filters nodes, converts to NetworkX, preserves metadata
- **Output**: `real_{prompt}_converted.json` (2-4 MB)
- **Time**: <1 second
- **Script**: `scripts/2_convert_graph.py`

### 3. `/neuronpedia-analyze`
Detect supernodes and analyze structure

- **What**: Louvain community detection, betweenness centrality, circuit stats
- **Output**: `real_{prompt}_supernodes.json` + console stats
- **Time**: ~5-10 seconds
- **Script**: `scripts/3_analyze_circuit.py`

### 4. `/neuronpedia-visualize`
Generate publication-quality visualizations

- **What**: Creates 5 PNG visualizations (supernode diagram, heatmap, distributions)
- **Output**: 5 PNG files (~5-8 MB total, 300 DPI)
- **Time**: ~10-20 seconds
- **Script**: `scripts/4_visualize.py`

### 5. `/neuronpedia-compare`
Compare graphs across prompts

- **What**: Feature overlap, shared circuits, complexity analysis
- **Output**: Console report + comparison PNG
- **Time**: ~5-10 seconds
- **Script**: `scripts/5_compare_prompts.py`

### 6. `/neuronpedia-validate`
Ensure data quality (no mock data)

- **What**: Validates filenames, sizes, metadata, node counts
- **Output**: Console validation report
- **Time**: <1 second per file
- **Script**: `scripts/validate_real_data.py`

### 7. `/circuit-report`
Run full pipeline and generate comprehensive analysis report

- **What**: Runs pipeline end-to-end, reads all output data, writes a detailed circuit analysis .md
- **Output**: `circuit_analysis_report.md` in the prompt's output directory
- **Time**: 3-8 minutes (pipeline) + analysis
- **Script**: `run_full_pipeline.py` + data analysis

### 8. `/steering-validate`
Validate bottleneck features via Neuronpedia Steering API

- **What**: Tests amplifying/suppressing bottleneck features, measures output changes, correlates cross-circuit frequency with steering effectiveness
- **Output**: `steering_validation_report.md`, `steering_analysis.json`, `steering_results.json`
- **Time**: ~20 min (quick) or ~10 hours (full)
- **Script**: `scripts/5_steering_validation.py`

---

## Complete Workflow

```bash
# Standard pipeline
/neuronpedia-fetch "The currency in Japan is"
/neuronpedia-convert
/neuronpedia-analyze
/neuronpedia-visualize
/neuronpedia-validate

# Multi-prompt comparison
/neuronpedia-fetch "The currency in Japan is"
/neuronpedia-convert
/neuronpedia-analyze

/neuronpedia-fetch "The capital of France is"
/neuronpedia-convert
/neuronpedia-analyze

/neuronpedia-compare real_japan_currency real_france_capital

# Steering validation
/steering-validate               # Quick test (~20 min)
/steering-validate --full         # Full run (~10 hours)
```

**Total time**: ~30-45 seconds per prompt (core pipeline)

---

## Key Features

- **Fast**: 60-120x faster than manual approach (30 sec vs 30-60 min)
- **Automated**: One command per step, auto-detects inputs
- **Quality**: Publication-ready 300 DPI visualizations
- **Validated**: Enforces real data only, no mock data
- **Discoveries**: Found L15_F376262 universal feature, 35.7% overlap

---

## Installation

Skills are already in the `neuronpedia_pipeline/skills/` folder.

To use:
1. Ensure pipeline dependencies installed: `pip install -r config/requirements.txt`
2. Add API key to `config/neuronpedia_config.yaml`
3. Skills will be available as `/neuronpedia-*` commands

---

## Real Data Discoveries

From testing on Japan (yen) and France (Paris) prompts:

- **L15_F376262**: Universal factual feature (150.290 activation in both)
- **35.7% overlap**: Shared circuits for factual retrieval
- **Layer 15 dominance**: 3 of top 5 features from Layer 15
- **42-43% inhibition**: Consistent suppression networks
- **Variable complexity**: France 50% more edges than Japan

---

## Documentation

Each skill has:
- Purpose and description
- When to use
- Step-by-step instructions for Claude
- Expected outputs
- Common issues and troubleshooting
- Next steps in workflow

See individual `SKILL.md` files for complete documentation. Two skills (`circuit-report`, `neuronpedia-analyze`) also have `references/` subdirectories with detailed templates and guidelines.

---

## Support

- **Pipeline docs**: `../README.md`, `../PIPELINE_STATUS.md`
- **Innovations**: `../INNOVATION_REPORT.md`
- **Discoveries**: `../COMPARISON_ANALYSIS_RESULTS.md`
- **Quick ref**: `../QUICK_REFERENCE.md`

---

**Status**: All 8 skills restructured with proper YAML front matter
**Format**: Claude Code SKILL.md format with `name` and `description` front matter
**Tested**: Windows 10, Python 3.11, real Neuronpedia data
**Archive**: Old skill files preserved in `_archive/`
