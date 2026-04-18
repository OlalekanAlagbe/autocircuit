#!/usr/bin/env python3
"""
Stage 2: Per-Layer Activation Energy Analysis

Computes detailed per-layer energy profiles for all 60 circuits to support
two paper angles:

  Angle 1 — "Architecture Over Knowledge":
    Shows that layer-by-layer energy profiles are determined by model
    architecture, not knowledge domain. Within-model domain profiles overlap;
    between-model profiles diverge even after depth normalization.

  Angle 3 — "Evidence Accumulation, Not Compression":
    Shows that confidence correlates with cumulative activation energy,
    not bottleneck convergence quality. Energy accumulates gradually through
    layers in a pattern consistent with drift-diffusion evidence accumulation.

Analyses:
  1. Per-circuit per-layer energy extraction (60 circuits × 26/36 layers)
  2. Domain-averaged layer energy curves per model (3 domains overlaid)
  3. Within-model vs between-model profile similarity (cosine + KS)
  4. Normalized depth comparison (GEMMA 26 → QWEN 36 alignment)
  5. Cumulative energy curves (evidence accumulation through layers)
  6. Energy-at-bottleneck decomposition (bottleneck vs distributed energy)
  7. Per-layer energy–confidence correlation (which layers drive confidence?)
  8. Drift-diffusion analogy metrics (accumulation rate, threshold, slope)

Usage:
    python stage_2_layer_energy_analysis.py
"""

import json
import sys
import io
import os
import math
import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import spearmanr, pearsonr, ks_2samp, mannwhitneyu

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==========================================================================
# Configuration
# ==========================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
ENHANCED_DIR = DATA_DIR / 'stage_2_enhanced_analysis'
DEEPDIVE_DIR = DATA_DIR / 'stage_2_statistical_deepdive'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_layer_energy'

MODELS = ['gemma-2-2b', 'qwen3-4b']

sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_constants import (
    BOTTLENECK_CONVERGENCE_THRESHOLD,
    GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS,
    GEMMA_LAYER_GROUPS, QWEN_LAYER_GROUPS,
)

MODEL_TOTAL_LAYERS = {
    'gemma-2-2b': GEMMA_TOTAL_LAYERS,
    'qwen3-4b': QWEN_TOTAL_LAYERS,
}
MODEL_LAYER_GROUPS = {
    'gemma-2-2b': GEMMA_LAYER_GROUPS,
    'qwen3-4b': QWEN_LAYER_GROUPS,
}

CATEGORIES = {
    'chemistry': [
        'the-chemical-symbol-for-gold-is', 'the-chemical-symbol-for-iron-is',
        'the-chemical-symbol-for-lead-is', 'the-chemical-symbol-for-mercury-is',
        'the-chemical-symbol-for-argon-is', 'the-chemical-symbol-for-sodium-is',
        'the-chemical-symbol-for-silver-is', 'the-chemical-symbol-for-potassium-is',
        'the-chemical-symbol-for-copper-is', 'the-chemical-symbol-for-tungsten-is',
    ],
    'geography': [
        'the-capital-of-france-is', 'the-capital-of-japan-is',
        'the-capital-of-germany-is', 'the-capital-of-egypt-is',
        'the-capital-of-italy-is', 'the-capital-of-spain-is',
        'the-capital-of-canada-is', 'the-capital-of-thailand-is',
        'the-capital-of-turkey-is', 'the-capital-of-south-korea-is',
    ],
    'history': [
        'world-war-ii-ended-in', 'the-first-moon-landing-was-in',
        'the-berlin-wall-fell-in', 'the-titanic-sank-in',
        'america-declared-independence-in', 'the-french-revolution-began-in',
        'world-war-i-started-in', 'columbus-reached-the-americas-in',
        'the-great-fire-of-london-occurred-in', 'nelson-mandela-was-released-from-prison-in',
    ],
}


# ==========================================================================
# Data Loading
# ==========================================================================

def load_converted_graph(model: str, prompt_slug: str) -> Optional[dict]:
    """Load converted_graph.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    conv_dir = PROMPTS_DIR / dir_name / '2_conversion'
    if not conv_dir.exists():
        return None
    matches = list(conv_dir.glob('*converted_graph.json'))
    if not matches:
        return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def load_traceback(model: str, prompt_slug: str) -> Optional[dict]:
    """Load traceback_paths.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    path = PROMPTS_DIR / dir_name / '3_analysis' / 'traceback_paths.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_data_table() -> List[dict]:
    """Load the flat 60-row data table from statistical deepdive."""
    path = DEEPDIVE_DIR / 'data_table.json'
    if not path.exists():
        print(f"  WARNING: data_table.json not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_category_for_slug(slug: str) -> str:
    """Return category name for a prompt slug."""
    for cat, slugs in CATEGORIES.items():
        if slug in slugs:
            return cat
    return 'unknown'


# ==========================================================================
# Analysis 1: Per-Circuit Per-Layer Energy Extraction
# ==========================================================================

def extract_layer_energy(graph_data: dict, total_layers: int) -> dict:
    """
    Compute per-layer activation energy for a single circuit.
    Returns: {
        'layer_energy': [float, ...],       # total activation per layer
        'layer_mean_act': [float, ...],      # mean activation per layer
        'layer_max_act': [float, ...],       # max activation per layer
        'layer_node_count': [int, ...],      # number of active nodes per layer
        'layer_energy_fraction': [float, ...], # fraction of total energy per layer
        'total_energy': float,
        'peak_layer': int,
        'peak_energy': float,
        'active_layers': int,                # layers with at least one node
    }
    """
    nodes = graph_data.get('nodes', [])
    if not nodes:
        return None

    layer_activations = defaultdict(list)
    for node in nodes:
        layer = node.get('layer', 0)
        act = node.get('activation', 0)
        layer_activations[layer].append(act)

    layer_energy = np.zeros(total_layers)
    layer_mean_act = np.zeros(total_layers)
    layer_max_act = np.zeros(total_layers)
    layer_node_count = np.zeros(total_layers, dtype=int)

    for layer_idx in range(total_layers):
        acts = layer_activations.get(layer_idx, [])
        if acts:
            layer_energy[layer_idx] = float(np.sum(acts))
            layer_mean_act[layer_idx] = float(np.mean(acts))
            layer_max_act[layer_idx] = float(np.max(acts))
            layer_node_count[layer_idx] = len(acts)

    total = float(np.sum(layer_energy))
    if total > 0:
        layer_frac = layer_energy / total
    else:
        layer_frac = np.zeros(total_layers)

    peak_layer = int(np.argmax(layer_energy))
    active_layers = int(np.sum(layer_node_count > 0))

    return {
        'layer_energy': layer_energy.tolist(),
        'layer_mean_act': layer_mean_act.tolist(),
        'layer_max_act': layer_max_act.tolist(),
        'layer_node_count': layer_node_count.tolist(),
        'layer_energy_fraction': layer_frac.tolist(),
        'total_energy': total,
        'peak_layer': peak_layer,
        'peak_energy': float(layer_energy[peak_layer]),
        'active_layers': active_layers,
    }


def extract_all_circuits() -> Dict[str, Dict[str, dict]]:
    """
    Extract per-layer energy for all 60 circuits.
    Returns: {model: {slug: layer_energy_dict}}
    """
    all_data = {}
    for model in MODELS:
        all_data[model] = {}
        total_layers = MODEL_TOTAL_LAYERS[model]
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                graph = load_converted_graph(model, slug)
                if graph is None:
                    print(f"  WARNING: No graph for {model}/{slug}")
                    continue
                energy = extract_layer_energy(graph, total_layers)
                if energy is not None:
                    energy['category'] = cat
                    energy['model'] = model
                    energy['slug'] = slug
                    # Add output probability from metadata
                    meta = graph.get('metadata', {})
                    energy['output_probability'] = meta.get('output_probability', 0.0)
                    all_data[model][slug] = energy
        print(f"  Extracted {len(all_data[model])} circuits for {model}")
    return all_data


# ==========================================================================
# Analysis 2: Domain-Averaged Layer Energy Curves
# ==========================================================================

def compute_domain_curves(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    Compute domain-averaged layer energy profiles per model.
    For each model+category: mean and std of layer_energy and layer_energy_fraction.
    """
    results = {}
    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        results[model] = {}
        for cat, slugs in CATEGORIES.items():
            energy_matrix = []
            frac_matrix = []
            mean_act_matrix = []
            for slug in slugs:
                if slug in circuit_data.get(model, {}):
                    c = circuit_data[model][slug]
                    energy_matrix.append(c['layer_energy'])
                    frac_matrix.append(c['layer_energy_fraction'])
                    mean_act_matrix.append(c['layer_mean_act'])

            if not energy_matrix:
                continue

            energy_arr = np.array(energy_matrix)
            frac_arr = np.array(frac_matrix)
            mean_act_arr = np.array(mean_act_matrix)

            results[model][cat] = {
                'n_circuits': len(energy_matrix),
                'mean_layer_energy': np.mean(energy_arr, axis=0).tolist(),
                'std_layer_energy': np.std(energy_arr, axis=0).tolist(),
                'mean_layer_fraction': np.mean(frac_arr, axis=0).tolist(),
                'std_layer_fraction': np.std(frac_arr, axis=0).tolist(),
                'mean_layer_mean_act': np.mean(mean_act_arr, axis=0).tolist(),
                'std_layer_mean_act': np.std(mean_act_arr, axis=0).tolist(),
                # Overall stats
                'mean_total_energy': float(np.mean([m[-1] for m in [np.cumsum(e) for e in energy_matrix]])),
                'mean_peak_layer': float(np.mean([np.argmax(e) for e in energy_matrix])),
                'std_peak_layer': float(np.std([np.argmax(e) for e in energy_matrix])),
            }

    return results


# ==========================================================================
# Analysis 3: Within-Model vs Between-Model Profile Similarity
# ==========================================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_profile_similarity(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    Compare energy profiles:
    1. Within-model, within-category (same model, same domain)
    2. Within-model, cross-category (same model, different domain)
    3. Between-model comparison (requires depth normalization)
    """
    results = {}

    for model in MODELS:
        model_data = circuit_data.get(model, {})
        if not model_data:
            continue

        # Collect profiles by category
        cat_profiles = defaultdict(list)
        cat_fracs = defaultdict(list)
        for slug, data in model_data.items():
            cat = data['category']
            cat_profiles[cat].append(data['layer_energy_fraction'])  # Use fractions for comparable profiles
            cat_fracs[cat].append({'slug': slug, 'frac': data['layer_energy_fraction']})

        # Within-category similarity
        within_sims = {}
        for cat in CATEGORIES:
            profiles = cat_profiles.get(cat, [])
            if len(profiles) < 2:
                continue
            sims = []
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    sims.append(cosine_similarity(profiles[i], profiles[j]))
            within_sims[cat] = {
                'mean_cosine': float(np.mean(sims)),
                'std_cosine': float(np.std(sims)),
                'min_cosine': float(np.min(sims)),
                'max_cosine': float(np.max(sims)),
                'n_pairs': len(sims),
            }

        # Cross-category similarity
        cross_sims = {}
        cats = sorted(CATEGORIES.keys())
        for ci in range(len(cats)):
            for cj in range(ci + 1, len(cats)):
                cat_a, cat_b = cats[ci], cats[cj]
                pa, pb = cat_profiles.get(cat_a, []), cat_profiles.get(cat_b, [])
                if not pa or not pb:
                    continue
                sims = []
                for a in pa:
                    for b in pb:
                        sims.append(cosine_similarity(a, b))
                cross_sims[f"{cat_a}_vs_{cat_b}"] = {
                    'mean_cosine': float(np.mean(sims)),
                    'std_cosine': float(np.std(sims)),
                    'n_pairs': len(sims),
                }

        # KS test: within-category vs cross-category cosines
        all_within = []
        all_cross = []
        for cat in CATEGORIES:
            profiles = cat_profiles.get(cat, [])
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    all_within.append(cosine_similarity(profiles[i], profiles[j]))

        for ci in range(len(cats)):
            for cj in range(ci + 1, len(cats)):
                pa = cat_profiles.get(cats[ci], [])
                pb = cat_profiles.get(cats[cj], [])
                for a in pa:
                    for b in pb:
                        all_cross.append(cosine_similarity(a, b))

        ks_result = {}
        if len(all_within) >= 5 and len(all_cross) >= 5:
            stat, pval = ks_2samp(all_within, all_cross)
            ks_result = {
                'ks_statistic': float(stat),
                'p_value': float(pval),
                'within_mean': float(np.mean(all_within)),
                'cross_mean': float(np.mean(all_cross)),
                'within_n': len(all_within),
                'cross_n': len(all_cross),
                'interpretation': 'significant' if pval < 0.05 else 'not significant',
            }

        results[model] = {
            'within_category': within_sims,
            'cross_category': cross_sims,
            'ks_within_vs_cross': ks_result,
        }

    # Between-model comparison (normalized depth)
    norm_profiles = {}
    n_bins = 20  # normalize both to 20 bins
    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_data = circuit_data.get(model, {})
        norm_profiles[model] = {}
        for slug, data in model_data.items():
            # Interpolate energy fraction to n_bins
            frac = np.array(data['layer_energy_fraction'])
            x_old = np.linspace(0, 1, len(frac))
            x_new = np.linspace(0, 1, n_bins)
            norm = np.interp(x_new, x_old, frac)
            norm_profiles[model][slug] = {
                'normalized_profile': norm.tolist(),
                'category': data['category'],
            }

    # Cross-model similarity (matched by category)
    cross_model_sims = {}
    for cat in CATEGORIES:
        gemma_profs = [v['normalized_profile'] for k, v in norm_profiles.get('gemma-2-2b', {}).items() if v['category'] == cat]
        qwen_profs = [v['normalized_profile'] for k, v in norm_profiles.get('qwen3-4b', {}).items() if v['category'] == cat]
        if gemma_profs and qwen_profs:
            sims = []
            for g in gemma_profs:
                for q in qwen_profs:
                    sims.append(cosine_similarity(g, q))
            cross_model_sims[cat] = {
                'mean_cosine': float(np.mean(sims)),
                'std_cosine': float(np.std(sims)),
                'n_pairs': len(sims),
            }

    # Overall within-model vs between-model
    all_within_model = []
    for model in MODELS:
        model_data = norm_profiles.get(model, {})
        profs = [v['normalized_profile'] for v in model_data.values()]
        for i in range(len(profs)):
            for j in range(i + 1, len(profs)):
                all_within_model.append(cosine_similarity(profs[i], profs[j]))

    all_between_model = []
    g_profs = [v['normalized_profile'] for v in norm_profiles.get('gemma-2-2b', {}).values()]
    q_profs = [v['normalized_profile'] for v in norm_profiles.get('qwen3-4b', {}).values()]
    for g in g_profs:
        for q in q_profs:
            all_between_model.append(cosine_similarity(g, q))

    cross_model_overall = {}
    if all_within_model and all_between_model:
        stat, pval = ks_2samp(all_within_model, all_between_model)
        # Mann-Whitney for stronger test
        u_stat, u_pval = mannwhitneyu(all_within_model, all_between_model, alternative='greater')
        cross_model_overall = {
            'within_model_mean': float(np.mean(all_within_model)),
            'within_model_std': float(np.std(all_within_model)),
            'between_model_mean': float(np.mean(all_between_model)),
            'between_model_std': float(np.std(all_between_model)),
            'ks_statistic': float(stat),
            'ks_p_value': float(pval),
            'mann_whitney_U': float(u_stat),
            'mann_whitney_p': float(u_pval),
            'within_model_n': len(all_within_model),
            'between_model_n': len(all_between_model),
        }

    results['cross_model'] = {
        'per_category': cross_model_sims,
        'overall': cross_model_overall,
        'normalized_profiles': {
            model: {
                slug: data['normalized_profile']
                for slug, data in norm_profiles.get(model, {}).items()
            }
            for model in MODELS
        },
    }

    return results


# ==========================================================================
# Analysis 4: Cumulative Energy Curves (Evidence Accumulation)
# ==========================================================================

def compute_cumulative_energy(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    Compute cumulative energy curves for each circuit.
    Analogous to drift-diffusion evidence accumulation:
    - Plot cumulative energy through layers
    - Measure accumulation rate (slope)
    - Detect thresholds (layer where 50%/80%/90% of energy is accumulated)
    """
    results = {}

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_data = circuit_data.get(model, {})
        results[model] = {}

        cat_curves = defaultdict(list)  # category -> list of cumulative curves
        cat_thresholds = defaultdict(list)  # category -> list of threshold dicts

        for slug, data in model_data.items():
            cat = data['category']
            energy = np.array(data['layer_energy'])
            total = data['total_energy']

            if total <= 0:
                continue

            # Cumulative sum (normalized to [0,1])
            cum_energy = np.cumsum(energy) / total

            # Accumulation thresholds (layer where X% of energy is reached)
            thresholds = {}
            for pct in [0.25, 0.50, 0.75, 0.90, 0.95]:
                layer_idx = np.searchsorted(cum_energy, pct)
                thresholds[f'layer_{int(pct*100)}pct'] = int(min(layer_idx, total_layers - 1))
                thresholds[f'depth_{int(pct*100)}pct'] = float(min(layer_idx, total_layers - 1)) / total_layers

            # Accumulation rate: fit linear to cumulative curve
            x = np.arange(total_layers)
            if len(cum_energy) == total_layers:
                slope, intercept = np.polyfit(x, cum_energy, 1)
            else:
                slope, intercept = 0.0, 0.0

            # Acceleration: second derivative of cumulative = first derivative of energy
            energy_norm = energy / total
            # Find the "burst" layer — where energy fraction is highest
            burst_layer = int(np.argmax(energy_norm))

            # Early vs late energy split
            midpoint = total_layers // 2
            early_energy = float(cum_energy[midpoint]) if midpoint < len(cum_energy) else 0.0
            late_energy = 1.0 - early_energy

            circuit_result = {
                'cumulative_curve': cum_energy.tolist(),
                'thresholds': thresholds,
                'accumulation_slope': float(slope),
                'burst_layer': burst_layer,
                'burst_layer_depth': float(burst_layer) / total_layers,
                'early_half_fraction': early_energy,
                'late_half_fraction': late_energy,
                'output_probability': data.get('output_probability', 0.0),
            }

            cat_curves[cat].append(cum_energy.tolist())
            cat_thresholds[cat].append(thresholds)
            results[model][slug] = circuit_result

        # Category-level aggregates
        results[model]['_category_summary'] = {}
        for cat in CATEGORIES:
            curves = cat_curves.get(cat, [])
            thresh_list = cat_thresholds.get(cat, [])
            if not curves:
                continue

            curves_arr = np.array(curves)
            results[model]['_category_summary'][cat] = {
                'n_circuits': len(curves),
                'mean_cumulative': np.mean(curves_arr, axis=0).tolist(),
                'std_cumulative': np.std(curves_arr, axis=0).tolist(),
                'mean_50pct_layer': float(np.mean([t['layer_50pct'] for t in thresh_list])),
                'std_50pct_layer': float(np.std([t['layer_50pct'] for t in thresh_list])),
                'mean_90pct_layer': float(np.mean([t['layer_90pct'] for t in thresh_list])),
                'std_90pct_layer': float(np.std([t['layer_90pct'] for t in thresh_list])),
                'mean_early_fraction': float(np.mean([
                    results[model][s]['early_half_fraction']
                    for s in CATEGORIES[cat] if s in results[model] and isinstance(results[model][s], dict) and 'early_half_fraction' in results[model][s]
                ])),
            }

    return results


# ==========================================================================
# Analysis 5: Energy-at-Bottleneck Decomposition
# ==========================================================================

def load_bottleneck_library() -> dict:
    """Load the bottleneck library from Stage 1.5 analysis."""
    path = DATA_DIR / 'stage_1_5_bottleneck_library.json'
    if not path.exists():
        print(f"  WARNING: bottleneck library not found at {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_bottleneck_layers_for_circuit(model: str, slug: str, bn_library: dict) -> set:
    """
    Extract bottleneck layers for a specific circuit.
    Uses multiple sources:
    1. Bottleneck library (Stage 1.5) — cross-circuit bottleneck features
    2. Traceback critical paths — nodes with high convergence
    """
    bn_layers = set()

    # Source 1: Bottleneck library — features tagged for this model
    # Library structure: list of features with circuit_appearances
    features = bn_library.get('features', bn_library.get('bottleneck_features', []))
    if isinstance(bn_library, list):
        features = bn_library

    for feat in features:
        if not isinstance(feat, dict):
            continue
        label = feat.get('label', feat.get('feature_label', ''))
        # Check if this feature is from the right model
        circuits = feat.get('circuit_appearances', feat.get('circuits', []))
        model_match = False
        if isinstance(circuits, list):
            for c in circuits:
                if isinstance(c, str) and model in c:
                    model_match = True
                    break
                elif isinstance(c, dict) and model in c.get('circuit_id', ''):
                    model_match = True
                    break
        elif isinstance(circuits, int) and circuits > 0:
            # Generic feature, check model from label
            model_match = True

        if model_match and label.startswith('L'):
            try:
                layer = int(label.split('_')[0][1:])
                bn_layers.add(layer)
            except (ValueError, IndexError):
                pass

    # Source 2: Traceback critical paths — find high-convergence intermediate nodes
    tb = load_traceback(model, slug)
    if tb is not None:
        critical_paths = tb.get('critical_paths', [])
        # Count node appearances across paths
        node_counts = Counter()
        total_paths = len(critical_paths)
        for path in critical_paths:
            path_nodes = path.get('path_nodes', [])
            for node in path_nodes:
                label = node.get('label', '')
                if label:
                    node_counts[label] += 1
        # Nodes appearing in 60%+ of paths are bottlenecks
        if total_paths > 0:
            for label, count in node_counts.items():
                if count / total_paths >= BOTTLENECK_CONVERGENCE_THRESHOLD:
                    if label.startswith('L'):
                        try:
                            layer = int(label.split('_')[0][1:])
                            bn_layers.add(layer)
                        except (ValueError, IndexError):
                            pass

    return bn_layers


def compute_bottleneck_energy(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    Decompose total energy into:
    - Energy at bottleneck layers
    - Energy at non-bottleneck layers
    Test: does bottleneck energy fraction predict confidence?
    """
    results = {}

    # Load bottleneck library once
    bn_library = load_bottleneck_library()
    print(f"  Loaded bottleneck library: {type(bn_library)}")

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_data = circuit_data.get(model, {})
        results[model] = {
            'per_circuit': {},
            'correlations': {},
        }

        bn_energy_fracs = []
        non_bn_energy_fracs = []
        confidences = []

        for slug, data in model_data.items():
            cat = data['category']
            energy = np.array(data['layer_energy'])
            total = data['total_energy']
            conf = data.get('output_probability', 0.0)

            if total <= 0:
                continue

            # Get bottleneck layers from multiple sources
            bn_layers = get_bottleneck_layers_for_circuit(model, slug, bn_library)

            if not bn_layers:
                # No identifiable bottleneck layers — use model-specific defaults
                # GEMMA bottlenecks at L5-7, QWEN at L22-25
                if model == 'gemma-2-2b':
                    bn_layers = {5, 6, 7}
                else:
                    bn_layers = {22, 23, 24, 25}
                # Mark as default
                using_default = True
            else:
                using_default = False

            bn_energy = sum(energy[l] for l in bn_layers if l < total_layers)
            non_bn_energy = total - bn_energy
            bn_frac = bn_energy / total
            non_bn_frac = non_bn_energy / total

            results[model]['per_circuit'][slug] = {
                'category': cat,
                'bottleneck_layers': sorted(bn_layers),
                'n_bottleneck_layers': len(bn_layers),
                'bottleneck_energy': float(bn_energy),
                'non_bottleneck_energy': float(non_bn_energy),
                'bottleneck_energy_fraction': float(bn_frac),
                'total_energy': total,
                'output_probability': conf,
                'used_default_layers': using_default if 'using_default' in dir() else False,
            }

            bn_energy_fracs.append(bn_frac)
            non_bn_energy_fracs.append(non_bn_frac)
            confidences.append(conf)

        # Correlations: bottleneck energy fraction vs confidence
        if len(bn_energy_fracs) >= 5:
            r_s, p_s = spearmanr(bn_energy_fracs, confidences)
            r_p, p_p = pearsonr(bn_energy_fracs, confidences)
            results[model]['correlations'] = {
                'bn_energy_frac_vs_confidence': {
                    'spearman_r': float(r_s),
                    'spearman_p': float(p_s),
                    'pearson_r': float(r_p),
                    'pearson_p': float(p_p),
                    'n': len(bn_energy_fracs),
                    'mean_bn_frac': float(np.mean(bn_energy_fracs)),
                    'std_bn_frac': float(np.std(bn_energy_fracs)),
                },
                'non_bn_energy_frac_vs_confidence': {
                    'spearman_r': float(-r_s),  # complementary
                    'spearman_p': float(p_s),
                    'interpretation': 'Higher bottleneck energy fraction predicts higher confidence' if r_s > 0 and p_s < 0.05 else 'No significant relationship',
                },
            }

    return results


# ==========================================================================
# Analysis 6: Per-Layer Energy–Confidence Correlation
# ==========================================================================

def compute_layer_confidence_correlation(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    For each layer: correlate its energy fraction with output confidence.
    Question: which layers' energy contributes most to confident predictions?
    This is the "causal tracing" analogy — which layers matter most?
    """
    results = {}

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_data = circuit_data.get(model, {})

        # Build matrix: rows=circuits, cols=layers (energy fraction)
        slugs = sorted(model_data.keys())
        if not slugs:
            continue

        n = len(slugs)
        energy_matrix = np.zeros((n, total_layers))
        confidences = np.zeros(n)

        for i, slug in enumerate(slugs):
            data = model_data[slug]
            energy_matrix[i, :] = data['layer_energy_fraction']
            confidences[i] = data.get('output_probability', 0.0)

        # Per-layer correlation with confidence
        layer_correlations = []
        for layer_idx in range(total_layers):
            col = energy_matrix[:, layer_idx]
            if np.std(col) < 1e-10:
                layer_correlations.append({
                    'layer': layer_idx,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0,
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'mean_energy_frac': float(np.mean(col)),
                    'active_circuits': int(np.sum(col > 0)),
                })
                continue

            r_s, p_s = spearmanr(col, confidences)
            r_p, p_p = pearsonr(col, confidences)
            layer_correlations.append({
                'layer': layer_idx,
                'spearman_r': float(r_s) if not np.isnan(r_s) else 0.0,
                'spearman_p': float(p_s) if not np.isnan(p_s) else 1.0,
                'pearson_r': float(r_p) if not np.isnan(r_p) else 0.0,
                'pearson_p': float(p_p) if not np.isnan(p_p) else 1.0,
                'mean_energy_frac': float(np.mean(col)),
                'active_circuits': int(np.sum(col > 0)),
            })

        # Top layers by |spearman_r|
        sorted_by_corr = sorted(layer_correlations, key=lambda x: abs(x['spearman_r']), reverse=True)
        top5_layers = sorted_by_corr[:5]

        # Significant layers (p < 0.05, Bonferroni corrected)
        bonferroni_alpha = 0.05 / total_layers
        sig_layers = [l for l in layer_correlations if l['spearman_p'] < bonferroni_alpha]

        results[model] = {
            'per_layer': layer_correlations,
            'top5_by_abs_correlation': top5_layers,
            'significant_after_bonferroni': sig_layers,
            'bonferroni_alpha': bonferroni_alpha,
            'n_circuits': n,
            'n_significant': len(sig_layers),
        }

    return results


# ==========================================================================
# Analysis 7: Drift-Diffusion Analogy Metrics
# ==========================================================================

def compute_drift_diffusion_metrics(circuit_data: Dict[str, Dict[str, dict]],
                                     cumulative_data: dict) -> dict:
    """
    Compute metrics inspired by drift-diffusion models:
    - Accumulation rate (how fast evidence accumulates)
    - Decision threshold (energy level at which model commits)
    - Variability (noise in the accumulation process)

    Test: do higher-confidence circuits accumulate evidence faster?
    """
    results = {}

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_cum = cumulative_data.get(model, {})
        model_data = circuit_data.get(model, {})

        accum_rates = []
        thresholds_50 = []
        thresholds_90 = []
        early_fracs = []
        confidences = []
        burst_layers = []

        for slug, cum_result in model_cum.items():
            if slug.startswith('_'):
                continue
            if not isinstance(cum_result, dict) or 'accumulation_slope' not in cum_result:
                continue

            accum_rates.append(cum_result['accumulation_slope'])
            thresholds_50.append(cum_result['thresholds']['layer_50pct'])
            thresholds_90.append(cum_result['thresholds']['layer_90pct'])
            early_fracs.append(cum_result['early_half_fraction'])
            confidences.append(cum_result['output_probability'])
            burst_layers.append(cum_result['burst_layer'])

        if len(confidences) < 5:
            continue

        # Correlations: accumulation metrics vs confidence
        corr_results = {}
        metrics_to_test = [
            ('accumulation_slope', accum_rates),
            ('layer_50pct_threshold', thresholds_50),
            ('layer_90pct_threshold', thresholds_90),
            ('early_half_fraction', early_fracs),
            ('burst_layer', burst_layers),
        ]

        for name, values in metrics_to_test:
            r_s, p_s = spearmanr(values, confidences)
            r_p, p_p = pearsonr(values, confidences)
            corr_results[name] = {
                'spearman_r': float(r_s) if not np.isnan(r_s) else 0.0,
                'spearman_p': float(p_s) if not np.isnan(p_s) else 1.0,
                'pearson_r': float(r_p) if not np.isnan(r_p) else 0.0,
                'pearson_p': float(p_p) if not np.isnan(p_p) else 1.0,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'n': len(values),
            }

        # Category comparison of accumulation metrics
        cat_comparison = {}
        for cat in CATEGORIES:
            cat_slugs = [s for s in CATEGORIES[cat] if s in model_cum and isinstance(model_cum.get(s, {}), dict) and 'accumulation_slope' in model_cum.get(s, {})]
            if not cat_slugs:
                continue
            cat_comparison[cat] = {
                'mean_accumulation_slope': float(np.mean([model_cum[s]['accumulation_slope'] for s in cat_slugs])),
                'mean_50pct_threshold': float(np.mean([model_cum[s]['thresholds']['layer_50pct'] for s in cat_slugs])),
                'mean_90pct_threshold': float(np.mean([model_cum[s]['thresholds']['layer_90pct'] for s in cat_slugs])),
                'mean_early_fraction': float(np.mean([model_cum[s]['early_half_fraction'] for s in cat_slugs])),
                'mean_burst_layer': float(np.mean([model_cum[s]['burst_layer'] for s in cat_slugs])),
                'n': len(cat_slugs),
            }

        results[model] = {
            'correlations_with_confidence': corr_results,
            'category_comparison': cat_comparison,
            'descriptive': {
                'mean_accumulation_slope': float(np.mean(accum_rates)),
                'std_accumulation_slope': float(np.std(accum_rates)),
                'mean_50pct_threshold': float(np.mean(thresholds_50)),
                'mean_90pct_threshold': float(np.mean(thresholds_90)),
                'mean_early_fraction': float(np.mean(early_fracs)),
                'n': len(confidences),
            },
        }

    return results


# ==========================================================================
# Analysis 8: Domain Overlap Test (Angle 1 Key Test)
# ==========================================================================

def compute_domain_overlap_test(circuit_data: Dict[str, Dict[str, dict]]) -> dict:
    """
    Key test for Angle 1:
    For each model, do domain-averaged energy curves overlap?
    Compute: maximum difference between domain curves at any layer,
    and compare to the variability WITHIN each domain.

    If max cross-domain difference < within-domain variability,
    then architecture determines energy profile, not domain.
    """
    results = {}

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        model_data = circuit_data.get(model, {})

        cat_frac_arrays = {}
        for cat in CATEGORIES:
            fracs = []
            for slug in CATEGORIES[cat]:
                if slug in model_data:
                    fracs.append(model_data[slug]['layer_energy_fraction'])
            if fracs:
                cat_frac_arrays[cat] = np.array(fracs)

        if len(cat_frac_arrays) < 2:
            continue

        # Per-layer: mean and std per category
        cat_means = {}
        cat_stds = {}
        for cat, arr in cat_frac_arrays.items():
            cat_means[cat] = np.mean(arr, axis=0)
            cat_stds[cat] = np.std(arr, axis=0)

        # Cross-domain maximum difference at each layer
        cats = sorted(cat_frac_arrays.keys())
        max_cross_diff_per_layer = np.zeros(total_layers)
        for layer_idx in range(total_layers):
            means = [cat_means[c][layer_idx] for c in cats]
            max_cross_diff_per_layer[layer_idx] = max(means) - min(means)

        # Within-domain average std at each layer
        mean_within_std_per_layer = np.mean([cat_stds[c] for c in cats], axis=0)

        # Signal-to-noise ratio: cross-domain difference / within-domain variability
        snr_per_layer = np.zeros(total_layers)
        for layer_idx in range(total_layers):
            if mean_within_std_per_layer[layer_idx] > 1e-10:
                snr_per_layer[layer_idx] = max_cross_diff_per_layer[layer_idx] / mean_within_std_per_layer[layer_idx]
            else:
                snr_per_layer[layer_idx] = 0.0

        # Overall: is cross-domain difference smaller than within-domain variability?
        # If most layers have SNR < 1, domains are indistinguishable
        n_snr_below_1 = int(np.sum(snr_per_layer < 1.0))
        n_active = int(np.sum(mean_within_std_per_layer > 1e-10))

        # Permutation test: are cross-domain differences significant?
        # Pool all circuits, shuffle labels, recompute mean cross-domain diff
        all_circuits = []
        all_labels = []
        for cat in cats:
            for slug in CATEGORIES[cat]:
                if slug in model_data:
                    all_circuits.append(model_data[slug]['layer_energy_fraction'])
                    all_labels.append(cat)

        observed_diff = float(np.mean(max_cross_diff_per_layer))

        n_perm = 5000
        perm_diffs = []
        np.random.seed(42)
        for _ in range(n_perm):
            shuffled = np.random.permutation(all_labels)
            perm_cat_means = {}
            for cat in cats:
                idxs = [i for i, l in enumerate(shuffled) if l == cat]
                if idxs:
                    perm_cat_means[cat] = np.mean([all_circuits[i] for i in idxs], axis=0)
            if len(perm_cat_means) >= 2:
                perm_max_diff = np.zeros(total_layers)
                for layer_idx in range(total_layers):
                    means = [perm_cat_means[c][layer_idx] for c in perm_cat_means]
                    perm_max_diff[layer_idx] = max(means) - min(means)
                perm_diffs.append(float(np.mean(perm_max_diff)))

        if perm_diffs:
            p_value = float(np.mean([1 if d >= observed_diff else 0 for d in perm_diffs]))
        else:
            p_value = 1.0

        results[model] = {
            'max_cross_diff_per_layer': max_cross_diff_per_layer.tolist(),
            'mean_within_std_per_layer': mean_within_std_per_layer.tolist(),
            'snr_per_layer': snr_per_layer.tolist(),
            'n_layers_snr_below_1': n_snr_below_1,
            'n_active_layers': n_active,
            'fraction_indistinguishable': float(n_snr_below_1 / max(n_active, 1)),
            'observed_mean_cross_diff': observed_diff,
            'permutation_p_value': p_value,
            'n_permutations': n_perm,
            'interpretation': (
                f"Domain energy profiles are {'indistinguishable' if p_value >= 0.05 else 'significantly different'} "
                f"(p={p_value:.4f}). {n_snr_below_1}/{n_active} active layers have SNR<1, "
                f"meaning within-domain variability exceeds cross-domain differences at "
                f"{100*n_snr_below_1/max(n_active,1):.0f}% of layers."
            ),
        }

    return results


# ==========================================================================
# Report Generation
# ==========================================================================

def generate_report(
    circuit_data: Dict[str, Dict[str, dict]],
    domain_curves: dict,
    profile_similarity: dict,
    cumulative_energy: dict,
    bottleneck_energy: dict,
    layer_confidence: dict,
    drift_diffusion: dict,
    domain_overlap: dict,
    output_dir: Path,
) -> str:
    """Generate comprehensive markdown report."""

    lines = []
    lines.append("# Per-Layer Activation Energy Analysis")
    lines.append(f"\n*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("---\n")

    # ---- Section 1: Data Overview ----
    lines.append("## 1. Data Overview\n")
    for model in MODELS:
        n = len(circuit_data.get(model, {}))
        lines.append(f"- **{model}**: {n} circuits extracted ({MODEL_TOTAL_LAYERS[model]} layers)")
    lines.append("")

    # ---- Section 2: Domain Energy Curves ----
    lines.append("## 2. Domain-Averaged Layer Energy Curves\n")
    lines.append("Key question: Do same-model circuits produce the same energy profile regardless of knowledge domain?\n")

    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        model_curves = domain_curves.get(model, {})
        total_layers = MODEL_TOTAL_LAYERS[model]

        for cat in ['chemistry', 'geography', 'history']:
            curve = model_curves.get(cat, {})
            if not curve:
                continue
            peak = curve.get('mean_peak_layer', 0)
            lines.append(f"- **{cat.title()}** (n={curve['n_circuits']}): Peak at L{peak:.1f} ({100*peak/total_layers:.0f}% depth)")

        lines.append("")

    # ---- Section 3: Profile Similarity ----
    lines.append("## 3. Energy Profile Similarity\n")
    lines.append("### 3a. Within-Model Comparison\n")

    for model in MODELS:
        sim = profile_similarity.get(model, {})
        lines.append(f"#### {model.upper()}\n")

        within = sim.get('within_category', {})
        cross = sim.get('cross_category', {})

        lines.append("| Comparison | Mean Cosine | Std | N pairs |")
        lines.append("|------------|-------------|-----|---------|")
        for cat in ['chemistry', 'geography', 'history']:
            w = within.get(cat, {})
            if w:
                lines.append(f"| Within {cat} | {w['mean_cosine']:.4f} | {w['std_cosine']:.4f} | {w['n_pairs']} |")
        for key, c in cross.items():
            lines.append(f"| {key.replace('_', ' ')} | {c['mean_cosine']:.4f} | {c['std_cosine']:.4f} | {c['n_pairs']} |")

        ks = sim.get('ks_within_vs_cross', {})
        if ks:
            lines.append(f"\n**KS test (within vs cross-category)**: D={ks.get('ks_statistic',0):.4f}, p={ks.get('p_value',1):.4f} ({ks.get('interpretation', 'n/a')})")
            lines.append(f"Within-category mean cosine: {ks.get('within_mean',0):.4f}, Cross-category: {ks.get('cross_mean',0):.4f}\n")

    lines.append("### 3b. Cross-Model Comparison (Normalized Depth)\n")
    cm = profile_similarity.get('cross_model', {}).get('overall', {})
    if cm:
        lines.append(f"- Within-model mean cosine: **{cm.get('within_model_mean',0):.4f}** (std={cm.get('within_model_std',0):.4f})")
        lines.append(f"- Between-model mean cosine: **{cm.get('between_model_mean',0):.4f}** (std={cm.get('between_model_std',0):.4f})")
        lines.append(f"- Mann-Whitney U={cm.get('mann_whitney_U',0):.1f}, p={cm.get('mann_whitney_p',1):.6f}")
        lines.append(f"- **Interpretation**: Within-model profiles are {'significantly' if cm.get('mann_whitney_p',1) < 0.001 else 'NOT significantly'} more similar than between-model profiles\n")

    # ---- Section 4: Domain Overlap Test (Angle 1) ----
    lines.append("## 4. Domain Overlap Test (Angle 1: Architecture Over Knowledge)\n")
    lines.append("Tests whether cross-domain energy differences exceed within-domain variability.\n")

    for model in MODELS:
        ov = domain_overlap.get(model, {})
        if not ov:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append(f"- Layers where within-domain variability > cross-domain difference (SNR<1): "
                     f"**{ov['n_layers_snr_below_1']}/{ov['n_active_layers']}** "
                     f"({ov['fraction_indistinguishable']*100:.0f}%)")
        lines.append(f"- Permutation test: p={ov['permutation_p_value']:.4f} (N={ov['n_permutations']})")
        lines.append(f"- **{ov['interpretation']}**\n")

    # ---- Section 5: Cumulative Energy (Angle 3) ----
    lines.append("## 5. Cumulative Energy Accumulation (Angle 3: Evidence Accumulation)\n")

    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        summary = cumulative_energy.get(model, {}).get('_category_summary', {})

        lines.append("| Category | Mean 50% Layer | Mean 90% Layer | Early-Half Fraction |")
        lines.append("|----------|---------------|---------------|---------------------|")
        for cat in ['chemistry', 'geography', 'history']:
            s = summary.get(cat, {})
            if s:
                lines.append(f"| {cat.title()} | L{s['mean_50pct_layer']:.1f} | L{s['mean_90pct_layer']:.1f} | {s.get('mean_early_fraction', 0)*100:.1f}% |")
        lines.append("")

    # ---- Section 6: Bottleneck Energy Decomposition ----
    lines.append("## 6. Bottleneck Energy Decomposition\n")
    lines.append("Does energy at bottleneck layers specifically predict confidence, or is it total distributed energy?\n")

    for model in MODELS:
        be = bottleneck_energy.get(model, {})
        corr = be.get('correlations', {})
        bn_corr = corr.get('bn_energy_frac_vs_confidence', {})
        if bn_corr:
            lines.append(f"### {model.upper()}\n")
            lines.append(f"- Bottleneck energy fraction vs confidence: r={bn_corr.get('spearman_r',0):.3f} (p={bn_corr.get('spearman_p',1):.4f})")
            lines.append(f"- Mean bottleneck energy fraction: {bn_corr.get('mean_bn_frac',0)*100:.1f}%")
            lines.append(f"- N circuits with identifiable bottleneck layers: {bn_corr.get('n',0)}\n")

    # ---- Section 7: Per-Layer Confidence Correlation ----
    lines.append("## 7. Per-Layer Energy-Confidence Correlation\n")
    lines.append("Which specific layers' energy predicts model confidence? (Analogous to causal tracing)\n")

    for model in MODELS:
        lc = layer_confidence.get(model, {})
        if not lc:
            continue
        lines.append(f"### {model.upper()}\n")
        lines.append(f"Bonferroni-corrected alpha: {lc.get('bonferroni_alpha',0):.5f}\n")

        top5 = lc.get('top5_by_abs_correlation', [])
        if top5:
            lines.append("**Top 5 layers by |correlation| with confidence:**\n")
            lines.append("| Layer | Spearman r | p-value | Mean Energy Frac | Significant? |")
            lines.append("|-------|-----------|---------|-----------------|-------------|")
            for l in top5:
                sig = "YES" if l['spearman_p'] < lc.get('bonferroni_alpha', 0.05) else "no"
                lines.append(f"| L{l['layer']} | {l['spearman_r']:.3f} | {l['spearman_p']:.4f} | {l['mean_energy_frac']:.4f} | {sig} |")
            lines.append("")

        sig_layers = lc.get('significant_after_bonferroni', [])
        n_sig = len(sig_layers)
        lines.append(f"**{n_sig} layer(s) survive Bonferroni correction** out of {MODEL_TOTAL_LAYERS[model]} tested.\n")

    # ---- Section 8: Drift-Diffusion Metrics ----
    lines.append("## 8. Drift-Diffusion Analogy\n")
    lines.append("Testing cognitive science analogy: does evidence accumulate faster in more confident circuits?\n")

    for model in MODELS:
        dd = drift_diffusion.get(model, {})
        if not dd:
            continue
        lines.append(f"### {model.upper()}\n")

        corrs = dd.get('correlations_with_confidence', {})
        if corrs:
            lines.append("| Metric | Spearman r | p-value | Mean | Interpretation |")
            lines.append("|--------|-----------|---------|------|---------------|")
            for name, c in corrs.items():
                interp = ""
                if c['spearman_p'] < 0.05:
                    direction = "positive" if c['spearman_r'] > 0 else "negative"
                    interp = f"Significant {direction}"
                else:
                    interp = "Not significant"
                lines.append(f"| {name} | {c['spearman_r']:.3f} | {c['spearman_p']:.4f} | {c['mean']:.3f} | {interp} |")
            lines.append("")

        # Category comparison
        cat_comp = dd.get('category_comparison', {})
        if cat_comp:
            lines.append("**Category comparison of accumulation metrics:**\n")
            lines.append("| Category | Accum. Slope | 50% Layer | 90% Layer | Early Fraction | Burst Layer |")
            lines.append("|----------|-------------|-----------|-----------|----------------|-------------|")
            for cat in ['chemistry', 'geography', 'history']:
                cc = cat_comp.get(cat, {})
                if cc:
                    lines.append(f"| {cat.title()} | {cc['mean_accumulation_slope']:.4f} | {cc['mean_50pct_threshold']:.1f} | {cc['mean_90pct_threshold']:.1f} | {cc['mean_early_fraction']:.3f} | {cc['mean_burst_layer']:.1f} |")
            lines.append("")

    # ---- Summary ----
    lines.append("## 9. Key Findings Summary\n")

    # Angle 1 findings
    lines.append("### Angle 1: Architecture Over Knowledge\n")
    for model in MODELS:
        ov = domain_overlap.get(model, {})
        sim = profile_similarity.get(model, {})
        if ov:
            pct = ov.get('fraction_indistinguishable', 0) * 100
            lines.append(f"- **{model.upper()}**: {pct:.0f}% of layers show within-domain variability exceeding cross-domain difference (permutation p={ov.get('permutation_p_value', 1):.4f})")

    cm = profile_similarity.get('cross_model', {}).get('overall', {})
    if cm:
        lines.append(f"- Cross-model profile similarity: {cm.get('between_model_mean',0):.4f} vs within-model: {cm.get('within_model_mean',0):.4f}")
    lines.append("")

    # Angle 3 findings
    lines.append("### Angle 3: Evidence Accumulation, Not Compression\n")
    for model in MODELS:
        dd = drift_diffusion.get(model, {})
        be = bottleneck_energy.get(model, {})
        if dd:
            slope_corr = dd.get('correlations_with_confidence', {}).get('accumulation_slope', {})
            if slope_corr:
                lines.append(f"- **{model.upper()}**: Accumulation slope vs confidence: r={slope_corr.get('spearman_r',0):.3f} (p={slope_corr.get('spearman_p',1):.4f})")
        if be:
            bn_corr = be.get('correlations', {}).get('bn_energy_frac_vs_confidence', {})
            if bn_corr:
                lines.append(f"- **{model.upper()}**: Bottleneck energy fraction vs confidence: r={bn_corr.get('spearman_r',0):.3f} (p={bn_corr.get('spearman_p',1):.4f})")

    lines.append("")
    return '\n'.join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 2: PER-LAYER ACTIVATION ENERGY ANALYSIS")
    print("=" * 70)
    print()

    # ---- Step 1: Extract all circuit data ----
    print("[1/8] Extracting per-layer energy for all circuits...")
    circuit_data = extract_all_circuits()
    total = sum(len(v) for v in circuit_data.values())
    print(f"  Total circuits: {total}")

    # Save per-circuit layer energy data
    # Convert for JSON serialization
    serializable_data = {}
    for model, slugs in circuit_data.items():
        serializable_data[model] = {}
        for slug, data in slugs.items():
            serializable_data[model][slug] = data

    with open(output_dir / 'per_circuit_layer_energy.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"  Saved per-circuit data to per_circuit_layer_energy.json")

    # ---- Step 2: Domain-averaged curves ----
    print("\n[2/8] Computing domain-averaged layer energy curves...")
    domain_curves = compute_domain_curves(circuit_data)
    print("  Done.")

    # ---- Step 3: Profile similarity ----
    print("\n[3/8] Computing within-model vs between-model profile similarity...")
    profile_similarity = compute_profile_similarity(circuit_data)
    cm = profile_similarity.get('cross_model', {}).get('overall', {})
    if cm:
        print(f"  Within-model cosine: {cm.get('within_model_mean',0):.4f}")
        print(f"  Between-model cosine: {cm.get('between_model_mean',0):.4f}")
        print(f"  Mann-Whitney p: {cm.get('mann_whitney_p',1):.6f}")

    # ---- Step 4: Cumulative energy ----
    print("\n[4/8] Computing cumulative energy curves (evidence accumulation)...")
    cumulative_energy = compute_cumulative_energy(circuit_data)
    for model in MODELS:
        summary = cumulative_energy.get(model, {}).get('_category_summary', {})
        for cat, s in summary.items():
            print(f"  {model}/{cat}: 50% energy by L{s['mean_50pct_layer']:.1f}, 90% by L{s['mean_90pct_layer']:.1f}")

    # ---- Step 5: Bottleneck energy decomposition ----
    print("\n[5/8] Computing bottleneck energy decomposition...")
    bottleneck_energy = compute_bottleneck_energy(circuit_data)
    for model in MODELS:
        corr = bottleneck_energy.get(model, {}).get('correlations', {}).get('bn_energy_frac_vs_confidence', {})
        if corr:
            print(f"  {model}: BN energy frac vs confidence: r={corr.get('spearman_r',0):.3f} (p={corr.get('spearman_p',1):.4f})")

    # ---- Step 6: Per-layer confidence correlation ----
    print("\n[6/8] Computing per-layer energy-confidence correlations...")
    layer_confidence = compute_layer_confidence_correlation(circuit_data)
    for model in MODELS:
        lc = layer_confidence.get(model, {})
        n_sig = lc.get('n_significant', 0)
        top5 = lc.get('top5_by_abs_correlation', [])
        if top5:
            top_layer = top5[0]
            print(f"  {model}: Top layer L{top_layer['layer']} (r={top_layer['spearman_r']:.3f}), {n_sig} layers survive Bonferroni")

    # ---- Step 7: Drift-diffusion metrics ----
    print("\n[7/8] Computing drift-diffusion analogy metrics...")
    drift_diffusion = compute_drift_diffusion_metrics(circuit_data, cumulative_energy)
    for model in MODELS:
        dd = drift_diffusion.get(model, {})
        slope_corr = dd.get('correlations_with_confidence', {}).get('accumulation_slope', {})
        if slope_corr:
            print(f"  {model}: Accumulation slope vs confidence: r={slope_corr.get('spearman_r',0):.3f} (p={slope_corr.get('spearman_p',1):.4f})")

    # ---- Step 8: Domain overlap test ----
    print("\n[8/8] Running domain overlap test (Angle 1 key test)...")
    domain_overlap = compute_domain_overlap_test(circuit_data)
    for model in MODELS:
        ov = domain_overlap.get(model, {})
        if ov:
            print(f"  {model}: {ov['fraction_indistinguishable']*100:.0f}% layers indistinguishable, permutation p={ov['permutation_p_value']:.4f}")

    # ---- Save results ----
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Remove numpy-unfriendly fields from cross_model normalized profiles (large)
    # Keep them in per_circuit file but not in main results
    profile_sim_save = {}
    for k, v in profile_similarity.items():
        if k == 'cross_model':
            cm_save = dict(v)
            cm_save.pop('normalized_profiles', None)  # Large, already have per_circuit file
            profile_sim_save[k] = cm_save
        else:
            profile_sim_save[k] = v

    all_results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_circuits': total,
            'models': MODELS,
            'categories': list(CATEGORIES.keys()),
        },
        'domain_curves': domain_curves,
        'profile_similarity': profile_sim_save,
        'cumulative_energy': cumulative_energy,
        'bottleneck_energy': bottleneck_energy,
        'layer_confidence_correlation': layer_confidence,
        'drift_diffusion': drift_diffusion,
        'domain_overlap': domain_overlap,
    }

    with open(output_dir / 'layer_energy_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results: {output_dir / 'layer_energy_results.json'}")

    # ---- Generate report ----
    report = generate_report(
        circuit_data, domain_curves, profile_similarity,
        cumulative_energy, bottleneck_energy, layer_confidence,
        drift_diffusion, domain_overlap, output_dir,
    )
    with open(output_dir / 'layer_energy_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {output_dir / 'layer_energy_report.md'}")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("ANGLE 1 (Architecture Over Knowledge):")
    for model in MODELS:
        ov = domain_overlap.get(model, {})
        if ov:
            print(f"  {model}: {ov['fraction_indistinguishable']*100:.0f}% of layers show domains are indistinguishable")
            print(f"           Permutation p={ov['permutation_p_value']:.4f}")
    cm = profile_similarity.get('cross_model', {}).get('overall', {})
    if cm:
        print(f"  Cross-model profiles are {'dissimilar' if cm.get('mann_whitney_p',1) < 0.001 else 'similar'} "
              f"(within={cm.get('within_model_mean',0):.4f} vs between={cm.get('between_model_mean',0):.4f})")
    print()
    print("ANGLE 3 (Evidence Accumulation):")
    for model in MODELS:
        dd = drift_diffusion.get(model, {})
        be = bottleneck_energy.get(model, {})
        if dd:
            slope = dd.get('correlations_with_confidence', {}).get('accumulation_slope', {})
            if slope:
                sig = "***" if slope.get('spearman_p', 1) < 0.01 else ("*" if slope.get('spearman_p', 1) < 0.05 else "n.s.")
                print(f"  {model}: Accumulation rate vs confidence: r={slope.get('spearman_r',0):.3f} {sig}")
        if be:
            bn = be.get('correlations', {}).get('bn_energy_frac_vs_confidence', {})
            if bn:
                sig = "***" if bn.get('spearman_p', 1) < 0.01 else ("*" if bn.get('spearman_p', 1) < 0.05 else "n.s.")
                print(f"  {model}: Bottleneck energy fraction vs confidence: r={bn.get('spearman_r',0):.3f} {sig}")

    print("\nDone!")


if __name__ == '__main__':
    main()
