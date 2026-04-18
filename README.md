# AutoCircuit: Neural Circuit Analysis Tools

**AI Safety Camp 2025 — Project #24**

Automated tools for analyzing factual knowledge circuits in large language models using SAE attribution graphs from [Neuronpedia](https://neuronpedia.org).

---

## Projects

### 1. [Neuronpedia Circuit Analysis Pipeline](neuronpedia_pipeline/)

Research pipeline for cross-domain circuit analysis. Generates attribution graphs, traces critical paths via backward BFS with geometric decay, identifies bottleneck features, runs cross-circuit statistical comparisons, and validates causal influence through steering experiments.

**Key paper:** [Cross-Domain Circuit Analysis of Factual Knowledge Retrieval in LLMs](neuronpedia_pipeline/docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md) — 60 circuits, 3 domains, 2 architectures, 80 steering experiments.

**Main findings:**
- Architecture determines circuit structure (~14× more variance than domain)
- The "bottleneck tax": energy at GEMMA L6 negatively predicts confidence (r = -0.684)
- 94% circuit redundancy; universal features are routing infrastructure, not knowledge stores
- Three-tier causal dissociation: topology → perturbation, redundancy → absorption, determinism → text changes

### 2. [Supernode Detector](supernode_detector/)

Standalone tool: prompt in → annotated Neuronpedia graph out. Multi-algorithm community detection with penalty-based selection, influence-based trimming, semantic labeling, cantor-paired feature export, and programmatic upload to Neuronpedia.

```bash
# Generate and upload in one command
python scripts/supernode_pipeline.py "The Titanic sank in the year" --upload

# Batch mode
python scripts/supernode_pipeline.py --batch-file prompts.txt --upload
```

**Tool paper:** [Supernode Detector: Automated Community Detection for Neuronpedia Attribution Graphs](supernode_detector/docs/SUPERNODE_PIPELINE_TOOL_PAPER.md)

---

## Quick Start

```bash
# Clone
git clone https://github.com/J-Lawrence10/autocircuit.git
cd autocircuit

# Research pipeline
cd neuronpedia_pipeline
pip install -r config/requirements.txt
# Add API key to config/neuronpedia_config.yaml
python run_full_pipeline.py --prompt "The chemical symbol for gold is" --model gemma-2-2b

# Supernode detector
cd ../supernode_detector
pip install -r requirements.txt
# Add API key to config/neuronpedia_config.yaml
python scripts/supernode_pipeline.py "The capital of France is" --upload
```

---

## Documentation

### Papers
| Paper | Location | Description |
|-------|----------|-------------|
| Cross-Domain Circuit Paper | [`neuronpedia_pipeline/docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md`](neuronpedia_pipeline/docs/papers/CROSS_DOMAIN_CIRCUIT_PAPER.md) | Main research paper (18 findings, 9 results sections) |
| Supplementary Materials | [`neuronpedia_pipeline/docs/papers/SUPPLEMENTARY_MATERIALS.md`](neuronpedia_pipeline/docs/papers/SUPPLEMENTARY_MATERIALS.md) | Extended results (25 sections) |
| Data Availability | [`neuronpedia_pipeline/docs/papers/DATA_AVAILABILITY_AND_EXPERIMENTS.md`](neuronpedia_pipeline/docs/papers/DATA_AVAILABILITY_AND_EXPERIMENTS.md) | Full data inventory and reproducibility guide |
| Supernode Tool Paper | [`supernode_detector/docs/SUPERNODE_PIPELINE_TOOL_PAPER.md`](supernode_detector/docs/SUPERNODE_PIPELINE_TOOL_PAPER.md) | Tool paper for the supernode detector |
| Traceback Graphing Paper | [`neuronpedia_pipeline/docs/papers/TRACEBACK_GRAPHING_PAPER.md`](neuronpedia_pipeline/docs/papers/TRACEBACK_GRAPHING_PAPER.md) | Original traceback methodology |

### PDFs
Generated PDFs with embedded figures: [`neuronpedia_pipeline/docs/papers/pdf/`](neuronpedia_pipeline/docs/papers/pdf/)

---

## Project Structure

```
autocircuit/
├── neuronpedia_pipeline/           # Research pipeline
│   ├── scripts/                    # Core pipeline (Steps 1-5) + 22 Stage 2 analysis scripts
│   ├── docs/papers/                # Research papers + supplementary
│   ├── config/                     # API config + requirements
│   ├── data/                       # Generated data (~2 GB, gitignored)
│   └── skills/                     # 12 Claude Code skills
│
├── supernode_detector/             # Standalone upload tool
│   ├── scripts/supernode_pipeline.py   # End-to-end pipeline
│   ├── docs/                       # Tool paper + design docs
│   ├── config/                     # API config
│   └── output/                     # Generated graphs (gitignored)
│
└── Archive/                        # Legacy code from earlier phases
```

---

## Requirements

- Python 3.8+
- Neuronpedia API key ([neuronpedia.org/account](https://neuronpedia.org/account))
- Dependencies: `networkx`, `python-louvain`, `leidenalg`, `python-igraph`, `matplotlib`, `requests`, `pyyaml`, `numpy`, `scikit-learn`

## Models Supported

| Model | Graph Generation | Feature Labels | Steering | Notes |
|-------|-----------------|----------------|----------|-------|
| gemma-2-2b | Yes | Yes | Yes | Full support via gemmascope-transcoder-16k |
| qwen3-4b | Yes | Structural only | No | Feature API not yet available |

---

**Last Updated:** April 2026
