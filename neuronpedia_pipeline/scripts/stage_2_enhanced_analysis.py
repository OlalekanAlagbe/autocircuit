#!/usr/bin/env python3
"""
Stage 2 Enhanced Analysis: Deep Cross-Domain Circuit Metrics

Extends the initial cross-category analysis with 8 categories of deeper metrics:
  A. Path Topology Metrics (compression, decay rate, branching)
  B. Statistical Significance (permutation tests, bootstrap CIs, chi-square)
  C. Prediction Confidence Correlation (output_probability vs circuit complexity)
  D. Output Node Sharing (final-layer feature convergence)
  E. Feature Semantic Clustering (bottleneck function classification)
  F. Community Structure Comparison (Louvain supernode similarity)
  G. Edge Weight Distribution (weight statistics, KS tests)
  H. Activation Profile Analysis (layer-by-layer fingerprints)

Usage:
    python stage_2_enhanced_analysis.py
    python stage_2_enhanced_analysis.py --output-dir data/stage_2_enhanced_analysis
"""

import json
import sys
import io
import os
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

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
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_enhanced_analysis'

MODELS = ['gemma-2-2b', 'qwen3-4b']

# Import constants
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

# Prompt categories (same as stage_2_cross_category_analysis.py)
CATEGORIES = {
    'chemistry': [
        'the-chemical-symbol-for-gold-is',
        'the-chemical-symbol-for-iron-is',
        'the-chemical-symbol-for-lead-is',
        'the-chemical-symbol-for-mercury-is',
        'the-chemical-symbol-for-argon-is',
        'the-chemical-symbol-for-sodium-is',
        'the-chemical-symbol-for-silver-is',
        'the-chemical-symbol-for-potassium-is',
        'the-chemical-symbol-for-copper-is',
        'the-chemical-symbol-for-tungsten-is',
    ],
    'geography': [
        'the-capital-of-france-is',
        'the-capital-of-japan-is',
        'the-capital-of-germany-is',
        'the-capital-of-egypt-is',
        'the-capital-of-italy-is',
        'the-capital-of-spain-is',
        'the-capital-of-canada-is',
        'the-capital-of-thailand-is',
        'the-capital-of-turkey-is',
        'the-capital-of-south-korea-is',
    ],
    'history': [
        'world-war-ii-ended-in',
        'the-first-moon-landing-was-in',
        'the-berlin-wall-fell-in',
        'the-titanic-sank-in',
        'america-declared-independence-in',
        'the-french-revolution-began-in',
        'world-war-i-started-in',
        'columbus-reached-the-americas-in',
        'the-great-fire-of-london-occurred-in',
        'nelson-mandela-was-released-from-prison-in',
    ],
}

BOTTLENECK_THRESHOLD = BOTTLENECK_CONVERGENCE_THRESHOLD

# Permutation / bootstrap iterations
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
RANDOM_SEED = 42


# ==========================================================================
# Data Loading
# ==========================================================================

def load_traceback(model: str, prompt_slug: str) -> Optional[dict]:
    """Load traceback_paths.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    path = PROMPTS_DIR / dir_name / '3_analysis' / 'traceback_paths.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def load_circuit_analysis(model: str, prompt_slug: str) -> Optional[dict]:
    """Load circuit_analysis.json for a given model+prompt."""
    dir_name = f"{model}_{prompt_slug}"
    analysis_dir = PROMPTS_DIR / dir_name / '3_analysis'
    if not analysis_dir.exists():
        return None
    matches = list(analysis_dir.glob('*circuit_analysis.json'))
    if not matches:
        return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def load_bottleneck_library() -> Optional[dict]:
    """Load the stage 1.5 bottleneck library."""
    path = DATA_DIR / 'stage_1_5_bottleneck_library.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================================================
# Utility functions (reused from stage_2_cross_category_analysis.py)
# ==========================================================================

def extract_bottlenecks(traceback_data: dict, threshold: float = BOTTLENECK_THRESHOLD) -> List[dict]:
    """Extract bottleneck features from traceback paths."""
    paths = traceback_data.get('critical_paths', [])
    if not paths:
        return []

    num_paths = len(paths)
    feature_appearances = Counter()
    feature_layers = {}
    feature_max_score = {}

    for path in paths:
        seen_in_path = set()
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label and label not in seen_in_path:
                feature_appearances[label] += 1
                seen_in_path.add(label)
                feature_layers[label] = node.get('layer', -1)
                current_max = feature_max_score.get(label, 0)
                feature_max_score[label] = max(current_max, node.get('score', 0))

    bottlenecks = []
    for feature, count in feature_appearances.most_common():
        convergence = count / num_paths
        if convergence >= threshold:
            bottlenecks.append({
                'label': feature,
                'layer': feature_layers[feature],
                'convergence': convergence,
                'path_count': count,
                'total_paths': num_paths,
                'max_score': feature_max_score[feature],
            })
    return bottlenecks


def extract_all_path_features(traceback_data: dict) -> Set[str]:
    """Extract ALL features from all paths (for Jaccard similarity)."""
    features = set()
    for path in traceback_data.get('critical_paths', []):
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label:
                features.add(label)
    return features


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ==========================================================================
# Analysis A: Path Topology Metrics
# ==========================================================================

def analyze_path_topology(traceback_data: dict, model: str) -> dict:
    """Compute path topology metrics for a single circuit."""
    total_layers = MODEL_TOTAL_LAYERS[model]
    paths = traceback_data.get('critical_paths', [])
    if not paths:
        return {}

    path_metrics = []
    for path in paths:
        nodes = path.get('path_nodes', [])
        path_length = path.get('path_length', len(nodes))
        layers_visited = path.get('layers_visited', len(set(n.get('layer', 0) for n in nodes)))

        # 1. Compression ratio
        compression_ratio = layers_visited / total_layers if total_layers > 0 else 0

        # 2. Score decay rate: slope of log(score) vs depth
        depths = []
        log_scores = []
        for node in nodes:
            d = node.get('depth', 0)
            s = node.get('score', 0)
            if s > 0:
                depths.append(d)
                log_scores.append(math.log(s))

        decay_rate = 0.0
        if len(depths) >= 2:
            try:
                slope, _ = np.polyfit(depths, log_scores, 1)
                decay_rate = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                decay_rate = 0.0

        # 3. Branching factor: avg unique nodes per depth level
        depth_nodes = defaultdict(set)
        for node in nodes:
            d = node.get('depth', 0)
            depth_nodes[d].add(node.get('node_id', ''))

        # Exclude depth 0 (always 1 node)
        branching_counts = [len(v) for k, v in depth_nodes.items() if k > 0]
        mean_branching = float(np.mean(branching_counts)) if branching_counts else 1.0

        # Layer span
        node_layers = [n.get('layer', 0) for n in nodes]
        layer_span = [min(node_layers), max(node_layers)] if node_layers else [0, 0]

        path_metrics.append({
            'rank': path.get('rank', 0),
            'path_length': path_length,
            'layers_visited': layers_visited,
            'compression_ratio': compression_ratio,
            'score_decay_rate': decay_rate,
            'mean_branching_factor': mean_branching,
            'max_depth': max(depths) if depths else 0,
            'layer_span': layer_span,
        })

    # Aggregate across paths
    return {
        'paths': path_metrics,
        'avg_compression_ratio': float(np.mean([p['compression_ratio'] for p in path_metrics])),
        'avg_score_decay_rate': float(np.mean([p['score_decay_rate'] for p in path_metrics])),
        'avg_branching_factor': float(np.mean([p['mean_branching_factor'] for p in path_metrics])),
        'avg_path_length': float(np.mean([p['path_length'] for p in path_metrics])),
        'avg_layers_visited': float(np.mean([p['layers_visited'] for p in path_metrics])),
    }


def aggregate_path_topology(per_circuit: Dict[str, dict]) -> dict:
    """Aggregate path topology metrics across circuits in a group."""
    if not per_circuit:
        return {}

    metrics = ['avg_compression_ratio', 'avg_score_decay_rate', 'avg_branching_factor',
               'avg_path_length', 'avg_layers_visited']
    result = {}
    for m in metrics:
        values = [c[m] for c in per_circuit.values() if m in c]
        if values:
            result[f'mean_{m}'] = float(np.mean(values))
            result[f'std_{m}'] = float(np.std(values))
    result['n_circuits'] = len(per_circuit)
    return result


# ==========================================================================
# Analysis B: Statistical Significance
# ==========================================================================

def run_permutation_test(
    feature_sets: Dict[str, Set[str]],
    labels: Dict[str, str],
    n_perms: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Permutation test: is within-category Jaccard significantly > cross-category?

    Args:
        feature_sets: {circuit_id: set of feature labels}
        labels: {circuit_id: category_name}
    """
    rng = np.random.RandomState(seed)
    circuit_ids = sorted(feature_sets.keys())
    n = len(circuit_ids)

    if n < 4:
        return {'error': 'Too few circuits for permutation test'}

    # Compute pairwise Jaccard for all pairs
    pair_jaccards = {}
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = circuit_ids[i], circuit_ids[j]
            pair_jaccards[(ci, cj)] = jaccard_similarity(feature_sets[ci], feature_sets[cj])

    def compute_diff(lab):
        """Compute mean_within - mean_cross given labels."""
        within_vals = []
        cross_vals = []
        for (ci, cj), jac in pair_jaccards.items():
            if lab[ci] == lab[cj]:
                within_vals.append(jac)
            else:
                cross_vals.append(jac)
        mean_within = np.mean(within_vals) if within_vals else 0
        mean_cross = np.mean(cross_vals) if cross_vals else 0
        return mean_within - mean_cross, mean_within, mean_cross

    # Observed difference
    observed_diff, obs_within, obs_cross = compute_diff(labels)

    # Permutation distribution
    perm_diffs = []
    category_values = [labels[cid] for cid in circuit_ids]
    for _ in range(n_perms):
        shuffled = list(category_values)
        rng.shuffle(shuffled)
        perm_labels = {circuit_ids[i]: shuffled[i] for i in range(n)}
        diff, _, _ = compute_diff(perm_labels)
        perm_diffs.append(diff)

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(perm_diffs >= observed_diff))

    return {
        'observed_difference': float(observed_diff),
        'within_category_mean_jaccard': float(obs_within),
        'cross_category_mean_jaccard': float(obs_cross),
        'p_value': p_value,
        'n_permutations': n_perms,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
    }


def run_bootstrap_ci(
    jaccard_values: List[float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """Bootstrap 95% CI for mean Jaccard similarity."""
    if not jaccard_values:
        return {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    rng = np.random.RandomState(seed)
    values = np.array(jaccard_values)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))

    boot_means = np.array(boot_means)
    return {
        'mean': float(np.mean(values)),
        'ci_lower': float(np.percentile(boot_means, 2.5)),
        'ci_upper': float(np.percentile(boot_means, 97.5)),
        'n_bootstrap': n_boot,
    }


def run_chi_square_layer_test(
    layer_distributions: Dict[str, Dict[int, int]],
    model: str,
) -> dict:
    """Chi-square test for independence of layer distribution across categories."""
    layer_groups = MODEL_LAYER_GROUPS[model]
    group_names = list(layer_groups.keys())
    categories = sorted(layer_distributions.keys())

    # Build contingency table: rows=categories, cols=layer_groups
    table = []
    for cat in categories:
        row = []
        for group_name in group_names:
            lo, hi = layer_groups[group_name]
            count = sum(
                v for k, v in layer_distributions[cat].items()
                if lo <= int(k) <= hi
            )
            row.append(count)
        table.append(row)

    table = np.array(table)

    # Need at least 2 rows and 2 cols with nonzero values
    if table.shape[0] < 2 or table.shape[1] < 2 or table.sum() == 0:
        return {'error': 'Insufficient data for chi-square test'}

    try:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(table)
        return {
            'chi2': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'significant_at_05': p_value < 0.05,
            'contingency_table': {
                'rows': categories,
                'cols': group_names,
                'observed': table.tolist(),
                'expected': expected.tolist(),
            },
        }
    except ValueError as e:
        return {'error': str(e)}


# ==========================================================================
# Analysis C: Prediction Confidence Correlation
# ==========================================================================

def extract_confidence_data(
    traceback_data: dict,
    graph_data: dict,
    model: str,
) -> Optional[dict]:
    """Extract prediction confidence and circuit complexity metrics."""
    if not graph_data or not traceback_data:
        return None

    metadata = graph_data.get('metadata', {})
    output_prob = metadata.get('output_probability', None)
    if output_prob is None:
        return None

    paths = traceback_data.get('critical_paths', [])
    path_lengths = [p.get('path_length', 0) for p in paths]
    layers_visited_list = [p.get('layers_visited', 0) for p in paths]

    bottlenecks = extract_bottlenecks(traceback_data)

    return {
        'output_probability': output_prob,
        'avg_path_length': float(np.mean(path_lengths)) if path_lengths else 0,
        'avg_layers_visited': float(np.mean(layers_visited_list)) if layers_visited_list else 0,
        'num_bottlenecks': len(bottlenecks),
        'total_nodes': metadata.get('converted_nodes', 0),
        'total_edges': metadata.get('converted_edges', 0),
        'model_output': metadata.get('model_output', ''),
    }


def compute_correlations(data_table: List[dict]) -> dict:
    """Compute correlation matrix between output_probability and circuit metrics."""
    if len(data_table) < 3:
        return {'error': 'Too few data points'}

    metrics = ['avg_path_length', 'avg_layers_visited', 'num_bottlenecks',
               'total_nodes', 'total_edges']
    probs = np.array([d['output_probability'] for d in data_table])

    results = {}
    for metric in metrics:
        values = np.array([d[metric] for d in data_table])
        if np.std(values) == 0 or np.std(probs) == 0:
            results[f'output_prob_vs_{metric}'] = {'error': 'No variance'}
            continue

        try:
            spearman_r, spearman_p = scipy_stats.spearmanr(probs, values)
            pearson_r, pearson_p = scipy_stats.pearsonr(probs, values)
            results[f'output_prob_vs_{metric}'] = {
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'n': len(data_table),
            }
        except Exception as e:
            results[f'output_prob_vs_{metric}'] = {'error': str(e)}

    return results


# ==========================================================================
# Analysis D: Output Node Sharing
# ==========================================================================

def extract_output_features(traceback_data: dict) -> Set[str]:
    """Extract the set of output feature labels from final_layer_nodes."""
    features = set()
    for node in traceback_data.get('final_layer_nodes', []):
        label = node.get('label', '')
        if label:
            features.add(label)
    return features


def analyze_output_sharing(
    output_feature_sets: Dict[str, Set[str]],
    path_feature_sets: Dict[str, Set[str]],
    category_labels: Dict[str, str],
) -> dict:
    """Compare output node sharing within/across categories."""
    circuit_ids = sorted(output_feature_sets.keys())
    categories = sorted(set(category_labels.values()))

    # Pairwise Jaccard for output features
    within_output = defaultdict(list)
    cross_output = defaultdict(list)
    within_path = defaultdict(list)
    cross_path = defaultdict(list)

    for i in range(len(circuit_ids)):
        for j in range(i + 1, len(circuit_ids)):
            ci, cj = circuit_ids[i], circuit_ids[j]
            cat_i = category_labels[ci]
            cat_j = category_labels[cj]

            out_jac = jaccard_similarity(output_feature_sets[ci], output_feature_sets[cj])
            path_jac = jaccard_similarity(path_feature_sets[ci], path_feature_sets[cj])

            if cat_i == cat_j:
                within_output[cat_i].append(out_jac)
                within_path[cat_i].append(path_jac)
            else:
                pair = tuple(sorted([cat_i, cat_j]))
                cross_output[f'{pair[0]}_vs_{pair[1]}'].append(out_jac)
                cross_path[f'{pair[0]}_vs_{pair[1]}'].append(path_jac)

    # Shared output features per category
    shared_output = {}
    for cat in categories:
        cat_circuits = [cid for cid in circuit_ids if category_labels[cid] == cat]
        feature_counts = Counter()
        for cid in cat_circuits:
            for f in output_feature_sets[cid]:
                feature_counts[f] += 1
        shared = [(f, c) for f, c in feature_counts.most_common() if c >= 2]
        shared_output[cat] = [{'label': f, 'circuit_count': c, 'total': len(cat_circuits),
                                'share_pct': c / len(cat_circuits) * 100}
                               for f, c in shared[:20]]

    result = {
        'within_category_output_jaccard': {},
        'within_category_path_jaccard': {},
        'cross_category_output_jaccard': {},
        'output_vs_path_comparison': {},
        'shared_output_features': shared_output,
    }

    for cat in categories:
        out_vals = within_output.get(cat, [])
        path_vals = within_path.get(cat, [])
        result['within_category_output_jaccard'][cat] = {
            'mean': float(np.mean(out_vals)) if out_vals else 0,
            'std': float(np.std(out_vals)) if out_vals else 0,
            'n_pairs': len(out_vals),
        }
        result['within_category_path_jaccard'][cat] = {
            'mean': float(np.mean(path_vals)) if path_vals else 0,
            'std': float(np.std(path_vals)) if path_vals else 0,
        }
        result['output_vs_path_comparison'][cat] = {
            'output_jaccard': float(np.mean(out_vals)) if out_vals else 0,
            'path_jaccard': float(np.mean(path_vals)) if path_vals else 0,
            'ratio': (float(np.mean(out_vals)) / float(np.mean(path_vals)))
                     if path_vals and np.mean(path_vals) > 0 else 0,
        }

    for pair, vals in cross_output.items():
        result['cross_category_output_jaccard'][pair] = {
            'mean': float(np.mean(vals)) if vals else 0,
            'n_pairs': len(vals),
        }

    return result


# ==========================================================================
# Analysis E: Feature Semantic Clustering
# ==========================================================================

# Import classify_from_explanation
try:
    from annotate_features_v2 import classify_from_explanation
    HAS_CLASSIFIER = True
except ImportError:
    HAS_CLASSIFIER = False
    def classify_from_explanation(explanation, layer=None):
        return 'UNKNOWN'


def analyze_feature_semantics(
    bottleneck_library: dict,
    universal_features: Dict[str, Set[str]],
    category_specific: Dict[str, Dict[str, Set[str]]],
) -> dict:
    """Classify bottleneck features by semantic function."""
    cross_circuit = bottleneck_library.get('cross_circuit_features', {})

    classified = {}
    category_profiles = defaultdict(lambda: Counter())
    universal_profile = Counter()
    specific_profiles = defaultdict(lambda: Counter())

    for label, info in cross_circuit.items():
        explanation = info.get('explanation', '')
        layer = info.get('layer', 0)

        # Classify
        if explanation and len(explanation.strip()) >= 5:
            sem_cat = classify_from_explanation(explanation, layer)
        else:
            # Heuristic: classify by layer position
            if layer <= 5:
                sem_cat = 'SYNTAX'
            elif layer <= 15:
                sem_cat = 'SEMANTICS:CONCEPT'
            else:
                sem_cat = 'UNKNOWN'

        # Determine which domain categories this feature appears in
        appearances = info.get('appearances', [])
        feature_categories = set()
        for app in appearances:
            prompt = app.get('prompt', '')
            for cat, slugs in CATEGORIES.items():
                if prompt in slugs:
                    feature_categories.add(cat)

        classified[label] = {
            'label': label,
            'layer': layer,
            'explanation': explanation[:200] if explanation else '',
            'semantic_category': sem_cat,
            'has_explanation': bool(explanation and len(explanation.strip()) >= 5),
            'domain_categories': list(feature_categories),
            'circuits_appeared_in': info.get('circuits_appeared_in', 0),
        }

        # Build profiles
        for cat in feature_categories:
            category_profiles[cat][sem_cat] += 1

        # Check if universal
        for model, uni_set in universal_features.items():
            if label in uni_set:
                universal_profile[sem_cat] += 1

        # Check if category-specific
        for model, cat_specs in category_specific.items():
            for cat, spec_set in cat_specs.items():
                if label in spec_set:
                    specific_profiles[cat][sem_cat] += 1

    # Summary statistics
    total_classified = len([c for c in classified.values() if c['has_explanation']])
    total_heuristic = len([c for c in classified.values() if not c['has_explanation']])

    return {
        'feature_classifications': classified,
        'category_semantic_profiles': {k: dict(v) for k, v in category_profiles.items()},
        'universal_semantic_profile': dict(universal_profile),
        'category_specific_profiles': {k: dict(v) for k, v in specific_profiles.items()},
        'classification_stats': {
            'total_features': len(classified),
            'with_explanation': total_classified,
            'heuristic_only': total_heuristic,
            'has_classifier': HAS_CLASSIFIER,
        },
    }


# ==========================================================================
# Analysis F: Community Structure Comparison
# ==========================================================================

def extract_supernode_stats(circuit_analysis: dict) -> Optional[dict]:
    """Extract supernode statistics from circuit_analysis.json."""
    supernodes_data = circuit_analysis.get('louvain_supernodes', {})
    supernodes = supernodes_data.get('supernodes', {})
    rankings = supernodes_data.get('rankings', [])

    if not supernodes:
        return None

    num_supernodes = len(supernodes)
    sizes = []
    layer_spans = {}
    activations = []

    for sn_id, sn in supernodes.items():
        sizes.append(sn.get('size', 0))
        layers = sn.get('layers', [])
        if layers:
            layer_spans[sn_id] = [min(layers), max(layers)]
        activations.append(sn.get('mean_activation', 0))

    # Size distribution entropy
    total_size = sum(sizes)
    if total_size > 0:
        probs = [s / total_size for s in sizes if s > 0]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0

    # Top-ranked supernode
    top_ranked = rankings[0] if rankings else {}

    return {
        'num_supernodes': num_supernodes,
        'sizes': sizes,
        'size_entropy': entropy,
        'layer_spans': layer_spans,
        'mean_activation_by_supernode': activations,
        'top_ranked': top_ranked,
    }


def compute_layer_partition(circuit_analysis: dict, total_layers: int) -> Optional[List[int]]:
    """
    Create a layer->supernode partition vector for ARI computation.
    partition[layer] = supernode_id that dominates that layer.
    """
    supernodes_data = circuit_analysis.get('louvain_supernodes', {})
    supernodes = supernodes_data.get('supernodes', {})

    if not supernodes:
        return None

    # For each layer, find which supernode has the most nodes there
    layer_to_supernode = {}
    for sn_id, sn in supernodes.items():
        layers = sn.get('layers', [])
        for layer in layers:
            if layer not in layer_to_supernode:
                layer_to_supernode[layer] = {}
            # Count nodes per supernode at this layer (approximate: just count presence)
            layer_to_supernode[layer][int(sn_id)] = layer_to_supernode[layer].get(int(sn_id), 0) + 1

    partition = []
    for layer in range(total_layers):
        if layer in layer_to_supernode:
            # Assign layer to supernode with most nodes
            best_sn = max(layer_to_supernode[layer], key=layer_to_supernode[layer].get)
            partition.append(best_sn)
        else:
            partition.append(-1)  # No supernode at this layer

    return partition


def adjusted_rand_index(partition_a: List[int], partition_b: List[int]) -> float:
    """Compute Adjusted Rand Index between two partitions."""
    n = len(partition_a)
    if n != len(partition_b) or n == 0:
        return 0.0

    # Build contingency table
    labels_a = set(partition_a)
    labels_b = set(partition_b)

    # Count pairs
    a_map = defaultdict(set)
    b_map = defaultdict(set)
    for i in range(n):
        a_map[partition_a[i]].add(i)
        b_map[partition_b[i]].add(i)

    # Contingency matrix
    nij_sum = 0
    ai_sum = 0
    bj_sum = 0

    for a_label in labels_a:
        a_set = a_map[a_label]
        ai = len(a_set)
        ai_sum += ai * (ai - 1) // 2
        for b_label in labels_b:
            b_set = b_map[b_label]
            nij = len(a_set & b_set)
            nij_sum += nij * (nij - 1) // 2

    for b_label in labels_b:
        bj = len(b_map[b_label])
        bj_sum += bj * (bj - 1) // 2

    n_choose_2 = n * (n - 1) // 2
    if n_choose_2 == 0:
        return 0.0

    expected = (ai_sum * bj_sum) / n_choose_2 if n_choose_2 > 0 else 0
    max_val = (ai_sum + bj_sum) / 2
    denominator = max_val - expected

    if denominator == 0:
        return 1.0 if nij_sum == expected else 0.0

    return (nij_sum - expected) / denominator


def analyze_community_structure(
    circuit_analyses: Dict[str, dict],
    category_labels: Dict[str, str],
    model: str,
) -> dict:
    """Compare community structure across circuits."""
    total_layers = MODEL_TOTAL_LAYERS[model]
    circuit_ids = sorted(circuit_analyses.keys())
    categories = sorted(set(category_labels.values()))

    # Extract stats and partitions
    stats = {}
    partitions = {}
    for cid in circuit_ids:
        s = extract_supernode_stats(circuit_analyses[cid])
        if s:
            stats[cid] = s
        p = compute_layer_partition(circuit_analyses[cid], total_layers)
        if p:
            partitions[cid] = p

    # Compute pairwise ARI
    within_ari = defaultdict(list)
    cross_ari = defaultdict(list)

    for i in range(len(circuit_ids)):
        for j in range(i + 1, len(circuit_ids)):
            ci, cj = circuit_ids[i], circuit_ids[j]
            if ci not in partitions or cj not in partitions:
                continue
            ari = adjusted_rand_index(partitions[ci], partitions[cj])
            cat_i = category_labels[ci]
            cat_j = category_labels[cj]
            if cat_i == cat_j:
                within_ari[cat_i].append(ari)
            else:
                pair = tuple(sorted([cat_i, cat_j]))
                cross_ari[f'{pair[0]}_vs_{pair[1]}'].append(ari)

    # Aggregate
    result = {
        'per_circuit_stats': {},
        'within_category_ari': {},
        'cross_category_ari': {},
        'summary': {},
    }

    for cid, s in stats.items():
        result['per_circuit_stats'][cid] = {
            'num_supernodes': s['num_supernodes'],
            'size_entropy': s['size_entropy'],
            'sizes': s['sizes'],
        }

    for cat in categories:
        vals = within_ari.get(cat, [])
        result['within_category_ari'][cat] = {
            'mean': float(np.mean(vals)) if vals else 0,
            'std': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }

    for pair, vals in cross_ari.items():
        result['cross_category_ari'][pair] = {
            'mean': float(np.mean(vals)) if vals else 0,
            'std': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }

    # Overall entropy stats per category
    for cat in categories:
        cat_circuits = [cid for cid in circuit_ids if category_labels[cid] == cat and cid in stats]
        if cat_circuits:
            entropies = [stats[cid]['size_entropy'] for cid in cat_circuits]
            n_sns = [stats[cid]['num_supernodes'] for cid in cat_circuits]
            result['summary'][cat] = {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
                'mean_num_supernodes': float(np.mean(n_sns)),
                'std_num_supernodes': float(np.std(n_sns)),
            }

    return result


# ==========================================================================
# Analysis G: Edge Weight Distribution
# ==========================================================================

def analyze_edge_weights(graph_data: dict) -> Optional[dict]:
    """Compute edge weight statistics for a single circuit."""
    edges = graph_data.get('edges', [])
    nodes = graph_data.get('nodes', [])
    if not edges:
        return None

    # Build node -> layer map
    node_layer = {}
    for node in nodes:
        node_layer[node['id']] = node.get('layer', 0)

    weights = np.array([e.get('weight', 0) for e in edges])
    abs_weights = np.abs(weights)

    # Forward/backward/same-layer classification
    forward_weights = []
    backward_weights = []
    same_layer_weights = []

    for e in edges:
        src_layer = node_layer.get(e['source'], 0)
        tgt_layer = node_layer.get(e['target'], 0)
        w = abs(e.get('weight', 0))

        if src_layer < tgt_layer:
            forward_weights.append(w)
        elif src_layer == tgt_layer:
            same_layer_weights.append(w)
        else:
            backward_weights.append(w)

    total = len(edges)
    forward_frac = len(forward_weights) / total if total > 0 else 0
    backward_frac = len(backward_weights) / total if total > 0 else 0
    same_frac = len(same_layer_weights) / total if total > 0 else 0

    # Heavy tail: what fraction of edges carry 50% of total abs weight
    sorted_abs = np.sort(abs_weights)[::-1]
    cumsum = np.cumsum(sorted_abs)
    total_weight = cumsum[-1] if len(cumsum) > 0 else 0

    if total_weight > 0:
        idx_50 = np.searchsorted(cumsum, total_weight * 0.5) + 1
        edges_for_50pct = int(idx_50)
        top_10pct_idx = max(1, int(len(sorted_abs) * 0.1))
        top_10pct_share = float(cumsum[top_10pct_idx - 1] / total_weight)
    else:
        edges_for_50pct = 0
        top_10pct_share = 0.0

    return {
        'total_edges': total,
        'weight_mean': float(np.mean(weights)),
        'weight_median': float(np.median(weights)),
        'weight_std': float(np.std(weights)),
        'weight_abs_mean': float(np.mean(abs_weights)),
        'weight_skewness': float(scipy_stats.skew(weights)),
        'weight_kurtosis': float(scipy_stats.kurtosis(weights)),
        'forward_edge_fraction': forward_frac,
        'backward_edge_fraction': backward_frac,
        'same_layer_edge_fraction': same_frac,
        'forward_edge_mean_weight': float(np.mean(forward_weights)) if forward_weights else 0,
        'backward_edge_mean_weight': float(np.mean(backward_weights)) if backward_weights else 0,
        'top_10pct_weight_share': top_10pct_share,
        'edges_for_50pct_weight': edges_for_50pct,
    }


def compute_edge_weight_ks_tests(
    edge_weights_per_circuit: Dict[str, np.ndarray],
    category_labels: Dict[str, str],
) -> dict:
    """KS tests comparing edge weight distributions within/across categories."""
    circuit_ids = sorted(edge_weights_per_circuit.keys())
    categories = sorted(set(category_labels.values()))

    within_ks = defaultdict(list)
    cross_ks = defaultdict(list)

    for i in range(len(circuit_ids)):
        for j in range(i + 1, len(circuit_ids)):
            ci, cj = circuit_ids[i], circuit_ids[j]
            # Subsample for speed (max 1000 edges each)
            w_i = edge_weights_per_circuit[ci]
            w_j = edge_weights_per_circuit[cj]
            if len(w_i) > 1000:
                w_i = np.random.choice(w_i, 1000, replace=False)
            if len(w_j) > 1000:
                w_j = np.random.choice(w_j, 1000, replace=False)

            try:
                ks_stat, ks_p = scipy_stats.ks_2samp(w_i, w_j)
            except Exception:
                continue

            cat_i = category_labels[ci]
            cat_j = category_labels[cj]
            if cat_i == cat_j:
                within_ks[cat_i].append(ks_stat)
            else:
                pair = tuple(sorted([cat_i, cat_j]))
                cross_ks[f'{pair[0]}_vs_{pair[1]}'].append(ks_stat)

    result = {'within_category': {}, 'cross_category': {}}
    for cat in categories:
        vals = within_ks.get(cat, [])
        result['within_category'][cat] = {
            'mean_ks_stat': float(np.mean(vals)) if vals else 0,
            'std_ks_stat': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }
    for pair, vals in cross_ks.items():
        result['cross_category'][pair] = {
            'mean_ks_stat': float(np.mean(vals)) if vals else 0,
            'std_ks_stat': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }
    return result


# ==========================================================================
# Analysis H: Activation Profile Analysis
# ==========================================================================

def compute_activation_profile(graph_data: dict, total_layers: int) -> Optional[dict]:
    """Compute layer-by-layer activation profile for a circuit."""
    nodes = graph_data.get('nodes', [])
    if not nodes:
        return None

    layer_activations = defaultdict(list)
    for node in nodes:
        layer = node.get('layer', 0)
        act = node.get('activation', 0)
        layer_activations[layer].append(act)

    per_layer = {}
    profile_vector = np.zeros(total_layers)

    for layer in range(total_layers):
        acts = layer_activations.get(layer, [])
        if acts:
            per_layer[layer] = {
                'mean_act': float(np.mean(acts)),
                'max_act': float(np.max(acts)),
                'std_act': float(np.std(acts)),
                'num_nodes': len(acts),
            }
            profile_vector[layer] = float(np.mean(acts))
        else:
            per_layer[layer] = {
                'mean_act': 0.0, 'max_act': 0.0, 'std_act': 0.0, 'num_nodes': 0,
            }

    # Peak layer
    peak_layer = int(np.argmax(profile_vector))

    # Total energy
    total_energy = float(np.sum([n.get('activation', 0) for n in nodes]))

    return {
        'per_layer': per_layer,
        'profile_vector': profile_vector.tolist(),
        'peak_layer': peak_layer,
        'total_energy': total_energy,
    }


def analyze_activation_profiles(
    profiles: Dict[str, dict],
    category_labels: Dict[str, str],
) -> dict:
    """Compare activation profiles across circuits."""
    circuit_ids = sorted(profiles.keys())
    categories = sorted(set(category_labels.values()))

    # Cosine similarity
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    within_cosine = defaultdict(list)
    cross_cosine = defaultdict(list)

    for i in range(len(circuit_ids)):
        for j in range(i + 1, len(circuit_ids)):
            ci, cj = circuit_ids[i], circuit_ids[j]
            sim = cosine_sim(profiles[ci]['profile_vector'], profiles[cj]['profile_vector'])
            cat_i = category_labels[ci]
            cat_j = category_labels[cj]
            if cat_i == cat_j:
                within_cosine[cat_i].append(sim)
            else:
                pair = tuple(sorted([cat_i, cat_j]))
                cross_cosine[f'{pair[0]}_vs_{pair[1]}'].append(sim)

    result = {
        'profile_similarity': {
            'within_category': {},
            'cross_category': {},
        },
        'peak_layer_distribution': {},
        'total_energy_comparison': {},
    }

    for cat in categories:
        vals = within_cosine.get(cat, [])
        result['profile_similarity']['within_category'][cat] = {
            'mean_cosine': float(np.mean(vals)) if vals else 0,
            'std_cosine': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }

        # Peak layer distribution
        cat_circuits = [cid for cid in circuit_ids if category_labels[cid] == cat]
        peaks = [profiles[cid]['peak_layer'] for cid in cat_circuits]
        energies = [profiles[cid]['total_energy'] for cid in cat_circuits]
        result['peak_layer_distribution'][cat] = {
            'mean_peak': float(np.mean(peaks)) if peaks else 0,
            'std_peak': float(np.std(peaks)) if peaks else 0,
            'peaks': peaks,
        }
        result['total_energy_comparison'][cat] = {
            'mean_energy': float(np.mean(energies)) if energies else 0,
            'std_energy': float(np.std(energies)) if energies else 0,
        }

    for pair, vals in cross_cosine.items():
        result['profile_similarity']['cross_category'][pair] = {
            'mean_cosine': float(np.mean(vals)) if vals else 0,
            'std_cosine': float(np.std(vals)) if vals else 0,
            'n_pairs': len(vals),
        }

    return result


# ==========================================================================
# Report Generation
# ==========================================================================

def generate_enhanced_report(results: dict) -> str:
    """Generate a comprehensive markdown report from enhanced analysis results."""
    lines = []
    lines.append("# Stage 2 Enhanced Analysis Report")
    lines.append(f"\n**Date**: {__import__('datetime').date.today()}")
    lines.append("**Analysis**: 8-category deep cross-domain circuit analysis")
    lines.append("**Circuits**: 60 (30 prompts x 2 models)")
    lines.append("")
    lines.append("---")

    # ---- A. Path Topology ----
    lines.append("\n## A. Path Topology Metrics\n")
    topo = results.get('path_topology', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        lines.append("| Category | Compression | Decay Rate | Branching | Path Length | Layers Visited |")
        lines.append("|----------|-------------|------------|-----------|-------------|----------------|")
        for cat in CATEGORIES:
            key = f"{model}_{cat}"
            agg = topo.get('aggregated', {}).get(key, {})
            lines.append(
                f"| {cat.title()} | "
                f"{agg.get('mean_avg_compression_ratio', 0):.3f}±{agg.get('std_avg_compression_ratio', 0):.3f} | "
                f"{agg.get('mean_avg_score_decay_rate', 0):.2f}±{agg.get('std_avg_score_decay_rate', 0):.2f} | "
                f"{agg.get('mean_avg_branching_factor', 0):.2f}±{agg.get('std_avg_branching_factor', 0):.2f} | "
                f"{agg.get('mean_avg_path_length', 0):.1f}±{agg.get('std_avg_path_length', 0):.1f} | "
                f"{agg.get('mean_avg_layers_visited', 0):.1f}±{agg.get('std_avg_layers_visited', 0):.1f} |"
            )
        lines.append("")

    # ---- B. Statistical Significance ----
    lines.append("\n## B. Statistical Significance\n")
    sig = results.get('statistical_significance', {})

    lines.append("### Permutation Tests (within vs cross-category Jaccard)\n")
    lines.append("| Model | Observed Diff | p-value | Significant (α=0.05) | Within Mean | Cross Mean |")
    lines.append("|-------|---------------|---------|----------------------|-------------|------------|")
    for model in MODELS:
        perm = sig.get('permutation_tests', {}).get(model, {})
        if 'error' not in perm:
            lines.append(
                f"| {model} | {perm.get('observed_difference', 0):.4f} | "
                f"{perm.get('p_value', 1):.4f} | "
                f"{'**YES**' if perm.get('significant_at_05') else 'no'} | "
                f"{perm.get('within_category_mean_jaccard', 0):.4f} | "
                f"{perm.get('cross_category_mean_jaccard', 0):.4f} |"
            )
    lines.append("")

    lines.append("### Bootstrap 95% Confidence Intervals (within-category Jaccard)\n")
    lines.append("| Model | Category | Mean Jaccard | 95% CI |")
    lines.append("|-------|----------|-------------|--------|")
    for model in MODELS:
        for cat in CATEGORIES:
            boot = sig.get('bootstrap_ci', {}).get(model, {}).get(cat, {})
            lines.append(
                f"| {model} | {cat.title()} | {boot.get('mean', 0):.4f} | "
                f"[{boot.get('ci_lower', 0):.4f}, {boot.get('ci_upper', 0):.4f}] |"
            )
    lines.append("")

    lines.append("### Chi-Square Tests (layer distribution independence)\n")
    lines.append("| Model | χ² | p-value | dof | Significant |")
    lines.append("|-------|-----|---------|-----|-------------|")
    for model in MODELS:
        chi = sig.get('chi_square', {}).get(model, {})
        if 'error' not in chi:
            lines.append(
                f"| {model} | {chi.get('chi2', 0):.2f} | "
                f"{chi.get('p_value', 1):.4f} | {chi.get('dof', 0)} | "
                f"{'**YES**' if chi.get('significant_at_05') else 'no'} |"
            )
    lines.append("")

    # ---- C. Prediction Confidence ----
    lines.append("\n## C. Prediction Confidence Correlation\n")
    conf = results.get('prediction_confidence', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        model_corr = conf.get('per_model', {}).get(model, {})
        lines.append("| Metric | Spearman r | Spearman p | Pearson r | Pearson p |")
        lines.append("|--------|-----------|-----------|----------|----------|")
        for key, vals in model_corr.items():
            if 'error' not in vals:
                metric_name = key.replace('output_prob_vs_', '')
                lines.append(
                    f"| {metric_name} | {vals.get('spearman_r', 0):.3f} | "
                    f"{vals.get('spearman_p', 1):.4f} | "
                    f"{vals.get('pearson_r', 0):.3f} | "
                    f"{vals.get('pearson_p', 1):.4f} |"
                )
        lines.append("")

    # ---- D. Output Node Sharing ----
    lines.append("\n## D. Output Node Sharing\n")
    out = results.get('output_node_sharing', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        model_out = out.get(model, {})
        lines.append("| Category | Output Jaccard | Path Jaccard | Output/Path Ratio |")
        lines.append("|----------|---------------|-------------|-------------------|")
        for cat in CATEGORIES:
            comp = model_out.get('output_vs_path_comparison', {}).get(cat, {})
            lines.append(
                f"| {cat.title()} | {comp.get('output_jaccard', 0):.4f} | "
                f"{comp.get('path_jaccard', 0):.4f} | "
                f"{comp.get('ratio', 0):.2f} |"
            )
        lines.append("")

    # ---- E. Feature Semantics ----
    lines.append("\n## E. Feature Semantic Clustering\n")
    sem = results.get('feature_semantics', {})
    stats_info = sem.get('classification_stats', {})
    lines.append(f"- Total features classified: {stats_info.get('total_features', 0)}")
    lines.append(f"- With Neuronpedia explanation: {stats_info.get('with_explanation', 0)}")
    lines.append(f"- Heuristic classification: {stats_info.get('heuristic_only', 0)}")
    lines.append("")

    lines.append("### Category Semantic Profiles\n")
    lines.append("| Category | SYNTAX | CONCEPT | GEOGRAPHIC | TEMPORAL | ENTITY | CODE | POLYSEMANTIC | UNKNOWN |")
    lines.append("|----------|--------|---------|------------|----------|--------|------|-------------|---------|")
    for cat in list(CATEGORIES.keys()) + ['universal']:
        if cat == 'universal':
            profile = sem.get('universal_semantic_profile', {})
        else:
            profile = sem.get('category_semantic_profiles', {}).get(cat, {})
        lines.append(
            f"| {cat.title()} | "
            f"{profile.get('SYNTAX', 0)} | "
            f"{profile.get('SEMANTICS:CONCEPT', 0)} | "
            f"{profile.get('SEMANTICS:GEOGRAPHIC', 0)} | "
            f"{profile.get('SEMANTICS:TEMPORAL', 0)} | "
            f"{profile.get('SEMANTICS:ENTITY', 0)} | "
            f"{profile.get('SEMANTICS:CODE', 0)} | "
            f"{profile.get('POLYSEMANTIC', 0)} | "
            f"{profile.get('UNKNOWN', 0)} |"
        )
    lines.append("")

    # ---- F. Community Structure ----
    lines.append("\n## F. Community Structure Comparison\n")
    comm = results.get('community_structure', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        model_comm = comm.get(model, {})

        lines.append("**Within-Category ARI (community partition similarity)**\n")
        lines.append("| Category | Mean ARI | Std ARI | Pairs |")
        lines.append("|----------|----------|---------|-------|")
        for cat in CATEGORIES:
            ari = model_comm.get('within_category_ari', {}).get(cat, {})
            lines.append(
                f"| {cat.title()} | {ari.get('mean', 0):.3f} | "
                f"{ari.get('std', 0):.3f} | {ari.get('n_pairs', 0)} |"
            )
        lines.append("")

        lines.append("**Supernode Summary**\n")
        lines.append("| Category | Mean # Supernodes | Mean Entropy |")
        lines.append("|----------|-------------------|-------------|")
        for cat in CATEGORIES:
            summ = model_comm.get('summary', {}).get(cat, {})
            lines.append(
                f"| {cat.title()} | {summ.get('mean_num_supernodes', 0):.1f}±{summ.get('std_num_supernodes', 0):.1f} | "
                f"{summ.get('mean_entropy', 0):.2f}±{summ.get('std_entropy', 0):.2f} |"
            )
        lines.append("")

    # ---- G. Edge Weight Distribution ----
    lines.append("\n## G. Edge Weight Distribution\n")
    edge = results.get('edge_weights', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        lines.append("| Category | Mean |w| | Std | Skew | Kurt | Fwd% | Bwd% | Top10% Share |")
        lines.append("|----------|---------|------|------|------|------|------|-------------|")
        agg = edge.get('aggregated', {})
        for cat in CATEGORIES:
            key = f"{model}_{cat}"
            a = agg.get(key, {})
            lines.append(
                f"| {cat.title()} | {a.get('mean_weight_abs_mean', 0):.4f} | "
                f"{a.get('mean_weight_std', 0):.4f} | "
                f"{a.get('mean_weight_skewness', 0):.2f} | "
                f"{a.get('mean_weight_kurtosis', 0):.1f} | "
                f"{a.get('mean_forward_edge_fraction', 0):.1%} | "
                f"{a.get('mean_backward_edge_fraction', 0):.1%} | "
                f"{a.get('mean_top_10pct_weight_share', 0):.1%} |"
            )
        lines.append("")

        # KS tests
        ks = edge.get('ks_tests', {}).get(model, {})
        lines.append("**KS Test (edge weight distribution similarity)**\n")
        lines.append("| Comparison | Mean KS Stat | Pairs |")
        lines.append("|-----------|-------------|-------|")
        for cat in CATEGORIES:
            vals = ks.get('within_category', {}).get(cat, {})
            lines.append(f"| Within {cat.title()} | {vals.get('mean_ks_stat', 0):.3f} | {vals.get('n_pairs', 0)} |")
        for pair, vals in ks.get('cross_category', {}).items():
            lines.append(f"| {pair} | {vals.get('mean_ks_stat', 0):.3f} | {vals.get('n_pairs', 0)} |")
        lines.append("")

    # ---- H. Activation Profiles ----
    lines.append("\n## H. Activation Profile Analysis\n")
    act = results.get('activation_profiles', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        model_act = act.get(model, {})

        lines.append("| Category | Mean Cosine (within) | Peak Layer | Total Energy |")
        lines.append("|----------|---------------------|------------|-------------|")
        for cat in CATEGORIES:
            cosine = model_act.get('profile_similarity', {}).get('within_category', {}).get(cat, {})
            peak = model_act.get('peak_layer_distribution', {}).get(cat, {})
            energy = model_act.get('total_energy_comparison', {}).get(cat, {})
            lines.append(
                f"| {cat.title()} | {cosine.get('mean_cosine', 0):.3f}±{cosine.get('std_cosine', 0):.3f} | "
                f"L{peak.get('mean_peak', 0):.1f}±{peak.get('std_peak', 0):.1f} | "
                f"{energy.get('mean_energy', 0):.0f}±{energy.get('std_energy', 0):.0f} |"
            )
        lines.append("")

        lines.append("**Cross-category profile similarity**\n")
        lines.append("| Comparison | Mean Cosine |")
        lines.append("|-----------|-------------|")
        for pair, vals in model_act.get('profile_similarity', {}).get('cross_category', {}).items():
            lines.append(f"| {pair} | {vals.get('mean_cosine', 0):.3f} |")
        lines.append("")

    return '\n'.join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2 Enhanced Circuit Analysis')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: ENHANCED DEEP ANALYSIS")
    print("  8 analysis categories across 60 circuits")
    print("=" * 70)

    # ================================================================
    # Phase 1: Load all data
    # ================================================================
    print("\n[Phase 1] Loading data...")

    traceback_data = {}     # (model, cat, slug) -> traceback
    graph_data = {}         # (model, cat, slug) -> converted_graph
    analysis_data = {}      # (model, cat, slug) -> circuit_analysis
    loaded_counts = {'traceback': 0, 'graph': 0, 'analysis': 0}

    for model in MODELS:
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                key = (model, cat, slug)

                tb = load_traceback(model, slug)
                if tb:
                    traceback_data[key] = tb
                    loaded_counts['traceback'] += 1

                gd = load_converted_graph(model, slug)
                if gd:
                    graph_data[key] = gd
                    loaded_counts['graph'] += 1

                ad = load_circuit_analysis(model, slug)
                if ad:
                    analysis_data[key] = ad
                    loaded_counts['analysis'] += 1

    bottleneck_lib = load_bottleneck_library()

    print(f"  Traceback: {loaded_counts['traceback']}/60")
    print(f"  Graphs: {loaded_counts['graph']}/60")
    print(f"  Analyses: {loaded_counts['analysis']}/60")
    print(f"  Bottleneck library: {'loaded' if bottleneck_lib else 'MISSING'}")

    # ================================================================
    # Phase 2: Run all analyses
    # ================================================================
    results = {}

    # ---- A. Path Topology ----
    print("\n[A] Analyzing path topology...")
    topo_per_circuit = {}
    topo_aggregated = {}

    for model in MODELS:
        for cat, slugs in CATEGORIES.items():
            group_circuits = {}
            for slug in slugs:
                key = (model, cat, slug)
                if key in traceback_data:
                    cid = f"{model}_{slug}"
                    topo = analyze_path_topology(traceback_data[key], model)
                    if topo:
                        topo_per_circuit[cid] = topo
                        group_circuits[cid] = topo

            agg_key = f"{model}_{cat}"
            topo_aggregated[agg_key] = aggregate_path_topology(group_circuits)
            agg = topo_aggregated[agg_key]
            print(f"  {model}/{cat}: compression={agg.get('mean_avg_compression_ratio', 0):.3f}, "
                  f"decay={agg.get('mean_avg_score_decay_rate', 0):.2f}")

    results['path_topology'] = {
        'per_circuit': topo_per_circuit,
        'aggregated': topo_aggregated,
    }

    # ---- B. Statistical Significance ----
    print("\n[B] Running statistical significance tests...")
    sig_results = {'permutation_tests': {}, 'bootstrap_ci': {}, 'chi_square': {}}

    for model in MODELS:
        # Build feature sets and labels for this model
        feature_sets = {}
        labels = {}
        jaccard_by_category = defaultdict(list)
        layer_dists_by_cat = {}

        for cat, slugs in CATEGORIES.items():
            cat_feature_sets = {}
            for slug in slugs:
                key = (model, cat, slug)
                if key in traceback_data:
                    cid = f"{model}_{slug}"
                    feature_sets[cid] = extract_all_path_features(traceback_data[key])
                    labels[cid] = cat
                    cat_feature_sets[slug] = feature_sets[cid]

            # Compute pairwise Jaccard for bootstrap
            slug_list = sorted(cat_feature_sets.keys())
            for i in range(len(slug_list)):
                for j in range(i + 1, len(slug_list)):
                    jac = jaccard_similarity(
                        cat_feature_sets[slug_list[i]],
                        cat_feature_sets[slug_list[j]]
                    )
                    jaccard_by_category[cat].append(jac)

            # Layer distribution for chi-square
            layer_dist = Counter()
            for slug in slugs:
                key = (model, cat, slug)
                if key in traceback_data:
                    for bn in extract_bottlenecks(traceback_data[key]):
                        layer_dist[bn['layer']] += 1
            layer_dists_by_cat[cat] = dict(layer_dist)

        # Permutation test
        print(f"  Running permutation test for {model}...")
        perm = run_permutation_test(feature_sets, labels)
        sig_results['permutation_tests'][model] = perm
        print(f"    Observed diff={perm.get('observed_difference', 0):.4f}, p={perm.get('p_value', 1):.4f}")

        # Bootstrap CIs
        sig_results['bootstrap_ci'][model] = {}
        for cat in CATEGORIES:
            boot = run_bootstrap_ci(jaccard_by_category[cat])
            sig_results['bootstrap_ci'][model][cat] = boot
            print(f"    {cat} Jaccard: {boot['mean']:.4f} [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

        # Chi-square
        chi = run_chi_square_layer_test(layer_dists_by_cat, model)
        sig_results['chi_square'][model] = chi
        if 'error' not in chi:
            print(f"    Chi-square: χ²={chi['chi2']:.2f}, p={chi['p_value']:.4f}")

    results['statistical_significance'] = sig_results

    # ---- C. Prediction Confidence ----
    print("\n[C] Analyzing prediction confidence correlations...")
    conf_results = {'per_model': {}, 'overall': {}, 'data_table': []}

    for model in MODELS:
        model_table = []
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                key = (model, cat, slug)
                if key in traceback_data and key in graph_data:
                    cd = extract_confidence_data(traceback_data[key], graph_data[key], model)
                    if cd:
                        cd['circuit'] = f"{model}_{slug}"
                        cd['model'] = model
                        cd['category'] = cat
                        model_table.append(cd)
                        conf_results['data_table'].append(cd)

        conf_results['per_model'][model] = compute_correlations(model_table)
        print(f"  {model}: {len(model_table)} circuits with confidence data")

    # Overall correlations
    conf_results['overall'] = compute_correlations(conf_results['data_table'])

    results['prediction_confidence'] = conf_results

    # ---- D. Output Node Sharing ----
    print("\n[D] Analyzing output node sharing...")
    output_sharing_results = {}

    for model in MODELS:
        output_sets = {}
        path_sets = {}
        cat_labels = {}

        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                key = (model, cat, slug)
                if key in traceback_data:
                    cid = f"{model}_{slug}"
                    output_sets[cid] = extract_output_features(traceback_data[key])
                    path_sets[cid] = extract_all_path_features(traceback_data[key])
                    cat_labels[cid] = cat

        sharing = analyze_output_sharing(output_sets, path_sets, cat_labels)
        output_sharing_results[model] = sharing

        for cat in CATEGORIES:
            comp = sharing.get('output_vs_path_comparison', {}).get(cat, {})
            print(f"  {model}/{cat}: output_jac={comp.get('output_jaccard', 0):.4f}, "
                  f"path_jac={comp.get('path_jaccard', 0):.4f}")

    results['output_node_sharing'] = output_sharing_results

    # ---- E. Feature Semantics ----
    print("\n[E] Classifying feature semantics...")
    if bottleneck_lib:
        # Get universal and category-specific feature sets from existing analysis
        universal_features = {}
        category_specific = {}

        # Load existing cross-category results
        existing_results_path = DATA_DIR / 'stage_2_analysis' / 'cross_category_results.json'
        if existing_results_path.exists():
            with open(existing_results_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for cr in existing.get('cross_category', []):
                model = cr['model']
                universal_features[model] = set(cr.get('universal_bottlenecks', []))
                category_specific[model] = {
                    cat: set(feats) for cat, feats in cr.get('category_specific', {}).items()
                }

        sem_results = analyze_feature_semantics(bottleneck_lib, universal_features, category_specific)
        results['feature_semantics'] = sem_results
        print(f"  Classified {sem_results['classification_stats']['total_features']} features")
        print(f"  Universal profile: {sem_results['universal_semantic_profile']}")
    else:
        results['feature_semantics'] = {'error': 'Bottleneck library not found'}
        print("  [WARN] Bottleneck library not available")

    # ---- F. Community Structure ----
    print("\n[F] Comparing community structure...")
    comm_results = {}

    for model in MODELS:
        model_analyses = {}
        cat_labels = {}

        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                key = (model, cat, slug)
                if key in analysis_data:
                    cid = f"{model}_{slug}"
                    model_analyses[cid] = analysis_data[key]
                    cat_labels[cid] = cat

        comm = analyze_community_structure(model_analyses, cat_labels, model)
        comm_results[model] = comm

        for cat in CATEGORIES:
            ari = comm.get('within_category_ari', {}).get(cat, {})
            print(f"  {model}/{cat}: ARI={ari.get('mean', 0):.3f}±{ari.get('std', 0):.3f}")

    results['community_structure'] = comm_results

    # ---- G. Edge Weight Distribution ----
    print("\n[G] Analyzing edge weight distributions...")
    edge_results = {'per_circuit': {}, 'aggregated': {}, 'ks_tests': {}}

    for model in MODELS:
        edge_weights_raw = {}
        cat_labels = {}

        for cat, slugs in CATEGORIES.items():
            group_stats = []
            for slug in slugs:
                key = (model, cat, slug)
                if key in graph_data:
                    cid = f"{model}_{slug}"
                    ew = analyze_edge_weights(graph_data[key])
                    if ew:
                        edge_results['per_circuit'][cid] = ew
                        group_stats.append(ew)

                        # Raw weights for KS test
                        weights = np.array([e.get('weight', 0) for e in graph_data[key].get('edges', [])])
                        edge_weights_raw[cid] = weights
                        cat_labels[cid] = cat

            # Aggregate per category
            agg_key = f"{model}_{cat}"
            if group_stats:
                agg = {}
                for metric in group_stats[0].keys():
                    if isinstance(group_stats[0][metric], (int, float)):
                        vals = [s[metric] for s in group_stats]
                        agg[f'mean_{metric}'] = float(np.mean(vals))
                        agg[f'std_{metric}'] = float(np.std(vals))
                edge_results['aggregated'][agg_key] = agg

        # KS tests
        if edge_weights_raw:
            ks = compute_edge_weight_ks_tests(edge_weights_raw, cat_labels)
            edge_results['ks_tests'][model] = ks

        print(f"  {model}: {len([k for k in edge_results['per_circuit'] if k.startswith(model)])} circuits analyzed")

    results['edge_weights'] = edge_results

    # ---- H. Activation Profiles ----
    print("\n[H] Computing activation profiles...")
    act_results = {}

    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        profiles = {}
        cat_labels = {}

        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                key = (model, cat, slug)
                if key in graph_data:
                    cid = f"{model}_{slug}"
                    profile = compute_activation_profile(graph_data[key], total_layers)
                    if profile:
                        profiles[cid] = profile
                        cat_labels[cid] = cat

        act = analyze_activation_profiles(profiles, cat_labels)
        act_results[model] = act

        for cat in CATEGORIES:
            cos = act.get('profile_similarity', {}).get('within_category', {}).get(cat, {})
            peak = act.get('peak_layer_distribution', {}).get(cat, {})
            print(f"  {model}/{cat}: cosine={cos.get('mean_cosine', 0):.3f}, "
                  f"peak=L{peak.get('mean_peak', 0):.1f}")

    results['activation_profiles'] = act_results

    # ================================================================
    # Phase 3: Save results
    # ================================================================
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    # Clean up non-serializable data from results before saving
    # Remove per-circuit path topology details to keep JSON manageable
    results_save = json.loads(json.dumps(results, default=str))

    json_path = output_dir / 'enhanced_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_save, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Generate and save report
    report = generate_enhanced_report(results)
    report_path = output_dir / 'enhanced_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # ================================================================
    # Phase 4: Key findings summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  KEY FINDINGS SUMMARY")
    print("=" * 70)

    # Significance
    for model in MODELS:
        perm = sig_results.get('permutation_tests', {}).get(model, {})
        p = perm.get('p_value', 1)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"\n  {model.upper()} permutation test: p={p:.4f} ({sig})")
        print(f"    Within-category Jaccard: {perm.get('within_category_mean_jaccard', 0):.4f}")
        print(f"    Cross-category Jaccard: {perm.get('cross_category_mean_jaccard', 0):.4f}")

    # Best correlations
    print(f"\n  Prediction Confidence Correlations:")
    for model in MODELS:
        model_corr = conf_results.get('per_model', {}).get(model, {})
        best_key = None
        best_r = 0
        for k, v in model_corr.items():
            if isinstance(v, dict) and 'spearman_r' in v:
                if abs(v['spearman_r']) > abs(best_r):
                    best_r = v['spearman_r']
                    best_key = k
        if best_key:
            print(f"    {model}: strongest = {best_key} (r={best_r:.3f})")

    # Activation profile insight
    print(f"\n  Activation Profile Cosine Similarity:")
    for model in MODELS:
        model_act = act_results.get(model, {})
        for cat in CATEGORIES:
            cos = model_act.get('profile_similarity', {}).get('within_category', {}).get(cat, {})
            print(f"    {model}/{cat}: {cos.get('mean_cosine', 0):.3f}")

    print(f"\n  Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
