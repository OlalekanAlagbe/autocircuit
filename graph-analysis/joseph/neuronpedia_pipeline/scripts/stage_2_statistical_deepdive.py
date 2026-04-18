#!/usr/bin/env python3
"""
Stage 2: Statistical Deepdive

Comprehensive statistical analysis of 60 circuits across 3 knowledge domains.
Builds a flat 60-row data table, then runs 8 analysis categories:

  1. Confidence vs Bottleneck Position
  2. Expanded Correlation Matrix (16 metrics, Bonferroni)
  3. Multiple Regression (OLS)
  4. ANOVA / Kruskal-Wallis (metric ~ category)
  5. Cross-Model Comparison (Mann-Whitney U)
  6. Effect Sizes (Cohen's d, eta-squared, rank-biserial)
  7. Bottleneck Convergence vs Confidence
  8. Information Concentration Index

Usage:
    python stage_2_statistical_deepdive.py
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
from scipy.stats import (
    spearmanr, pearsonr,
    mannwhitneyu,
    kruskal, f_oneway,
    shapiro,
)
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

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
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_statistical_deepdive'

MODELS = ['gemma-2-2b', 'qwen3-4b']

sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_constants import (
    BOTTLENECK_CONVERGENCE_THRESHOLD,
    GEMMA_TOTAL_LAYERS, QWEN_TOTAL_LAYERS,
)

MODEL_TOTAL_LAYERS = {
    'gemma-2-2b': GEMMA_TOTAL_LAYERS,
    'qwen3-4b': QWEN_TOTAL_LAYERS,
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
        'the-great-fire-of-london-occurred-in',
        'nelson-mandela-was-released-from-prison-in',
    ],
}

BOTTLENECK_THRESHOLD = BOTTLENECK_CONVERGENCE_THRESHOLD


# ==========================================================================
# Data Loading
# ==========================================================================

def load_traceback(model, slug):
    path = PROMPTS_DIR / f"{model}_{slug}" / '3_analysis' / 'traceback_paths.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_converted_graph(model, slug):
    conv_dir = PROMPTS_DIR / f"{model}_{slug}" / '2_conversion'
    if not conv_dir.exists():
        return None
    matches = list(conv_dir.glob('*converted_graph.json'))
    if not matches:
        return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_bottlenecks(traceback_data, threshold=BOTTLENECK_THRESHOLD):
    paths = traceback_data.get('critical_paths', [])
    if not paths:
        return []
    num_paths = len(paths)
    feature_appearances = Counter()
    feature_layers = {}
    feature_max_score = {}
    for path in paths:
        seen = set()
        for node in path.get('path_nodes', []):
            label = node.get('label', '')
            if label and label not in seen:
                feature_appearances[label] += 1
                seen.add(label)
                feature_layers[label] = node.get('layer', -1)
                feature_max_score[label] = max(feature_max_score.get(label, 0), node.get('score', 0))
    bottlenecks = []
    for feature, count in feature_appearances.most_common():
        convergence = count / num_paths
        if convergence >= threshold:
            bottlenecks.append({
                'label': feature, 'layer': feature_layers[feature],
                'convergence': convergence, 'max_score': feature_max_score[feature],
            })
    return bottlenecks


def load_enhanced_results():
    path = ENHANCED_DIR / 'enhanced_results.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==========================================================================
# Phase 0: Build Flat Data Table
# ==========================================================================

def build_data_table():
    """Assemble 60-row flat table from enhanced_results + raw data."""
    enhanced = load_enhanced_results()
    if not enhanced:
        print("  [ERROR] enhanced_results.json not found!")
        return []

    # Pre-index enhanced data
    topo_per_circuit = enhanced.get('path_topology', {}).get('per_circuit', {})
    edge_per_circuit = enhanced.get('edge_weights', {}).get('per_circuit', {})
    conf_table = enhanced.get('prediction_confidence', {}).get('data_table', [])
    conf_by_circuit = {d['circuit']: d for d in conf_table}

    # Community structure is nested by model
    comm_per_circuit = {}
    for model in MODELS:
        model_comm = enhanced.get('community_structure', {}).get(model, {}).get('per_circuit_stats', {})
        comm_per_circuit.update(model_comm)

    table = []
    for model in MODELS:
        total_layers = MODEL_TOTAL_LAYERS[model]
        for cat, slugs in CATEGORIES.items():
            for slug in slugs:
                cid = f"{model}_{slug}"
                row = {
                    'circuit_id': cid,
                    'model': model,
                    'category': cat,
                    'prompt_slug': slug,
                }

                # From prediction_confidence data_table
                cd = conf_by_circuit.get(cid, {})
                row['output_probability'] = cd.get('output_probability', 0)
                row['total_nodes'] = cd.get('total_nodes', 0)
                row['total_edges'] = cd.get('total_edges', 0)
                row['num_bottlenecks'] = cd.get('num_bottlenecks', 0)

                # From path_topology
                topo = topo_per_circuit.get(cid, {})
                row['avg_compression_ratio'] = topo.get('avg_compression_ratio', 0)
                row['avg_score_decay_rate'] = topo.get('avg_score_decay_rate', 0)
                row['avg_branching_factor'] = topo.get('avg_branching_factor', 0)
                row['avg_path_length'] = topo.get('avg_path_length', 0)
                row['avg_layers_visited'] = topo.get('avg_layers_visited', 0)

                # From edge_weights
                ew = edge_per_circuit.get(cid, {})
                row['weight_mean'] = ew.get('weight_mean', 0)
                row['weight_median'] = ew.get('weight_median', 0)
                row['weight_std'] = ew.get('weight_std', 0)
                row['weight_abs_mean'] = ew.get('weight_abs_mean', 0)
                row['weight_skewness'] = ew.get('weight_skewness', 0)
                row['weight_kurtosis'] = ew.get('weight_kurtosis', 0)
                row['top_10pct_weight_share'] = ew.get('top_10pct_weight_share', 0)
                row['edges_for_50pct_weight'] = ew.get('edges_for_50pct_weight', 0)

                # From community structure
                cs = comm_per_circuit.get(cid, {})
                row['num_supernodes'] = cs.get('num_supernodes', 0)
                row['size_entropy'] = cs.get('size_entropy', 0)

                # Derive per-circuit peak_layer and total_energy from raw graph
                gd = load_converted_graph(model, slug)
                if gd:
                    nodes = gd.get('nodes', [])
                    layer_acts = defaultdict(list)
                    total_energy = 0.0
                    for n in nodes:
                        layer_acts[n.get('layer', 0)].append(n.get('activation', 0))
                        total_energy += n.get('activation', 0)
                    profile = np.zeros(total_layers)
                    for layer in range(total_layers):
                        if layer in layer_acts:
                            profile[layer] = np.mean(layer_acts[layer])
                    row['peak_layer'] = int(np.argmax(profile))
                    row['total_energy'] = total_energy
                else:
                    row['peak_layer'] = 0
                    row['total_energy'] = 0

                # Derive per-circuit bottleneck position stats from raw traceback
                tb = load_traceback(model, slug)
                if tb:
                    bns = extract_bottlenecks(tb)
                    if bns:
                        bn_layers = [b['layer'] for b in bns]
                        bn_convergences = [b['convergence'] for b in bns]
                        row['avg_bottleneck_layer'] = float(np.mean(bn_layers))
                        row['median_bottleneck_layer'] = float(np.median(bn_layers))
                        row['min_bottleneck_layer'] = int(np.min(bn_layers))
                        row['max_bottleneck_layer'] = int(np.max(bn_layers))
                        row['bottleneck_layer_std'] = float(np.std(bn_layers))
                        row['mean_bottleneck_convergence'] = float(np.mean(bn_convergences))
                    else:
                        for k in ['avg_bottleneck_layer', 'median_bottleneck_layer',
                                   'min_bottleneck_layer', 'max_bottleneck_layer',
                                   'bottleneck_layer_std', 'mean_bottleneck_convergence']:
                            row[k] = 0
                else:
                    for k in ['avg_bottleneck_layer', 'median_bottleneck_layer',
                               'min_bottleneck_layer', 'max_bottleneck_layer',
                               'bottleneck_layer_std', 'mean_bottleneck_convergence']:
                        row[k] = 0

                # Composite: concentration index
                te = row['total_edges'] if row['total_edges'] > 0 else 1
                row['concentration_index'] = (
                    row['top_10pct_weight_share'] * row['weight_kurtosis'] / te
                )

                table.append(row)

    return table


# ==========================================================================
# Analysis 1: Confidence vs Bottleneck Position
# ==========================================================================

def analysis_1(data_table):
    """Correlate output_probability with bottleneck layer metrics."""
    bn_metrics = ['avg_bottleneck_layer', 'median_bottleneck_layer',
                  'bottleneck_layer_std', 'min_bottleneck_layer', 'max_bottleneck_layer']
    results = {}

    for model in MODELS + ['combined']:
        rows = data_table if model == 'combined' else [r for r in data_table if r['model'] == model]
        probs = np.array([r['output_probability'] for r in rows])
        model_results = {}

        for metric in bn_metrics:
            vals = np.array([r[metric] for r in rows])
            if np.std(vals) == 0 or np.std(probs) == 0:
                model_results[metric] = {'error': 'No variance'}
                continue
            sr, sp = spearmanr(probs, vals)
            pr, pp = pearsonr(probs, vals)
            model_results[metric] = {
                'spearman_r': float(sr), 'spearman_p': float(sp),
                'pearson_r': float(pr), 'pearson_p': float(pp),
                'n': len(rows),
            }
        results[model] = model_results

    # Scatter data
    results['scatter_data'] = [
        {'circuit_id': r['circuit_id'], 'model': r['model'], 'category': r['category'],
         'output_probability': r['output_probability'],
         'avg_bottleneck_layer': r['avg_bottleneck_layer']}
        for r in data_table
    ]
    return results


# ==========================================================================
# Analysis 2: Expanded Correlation Matrix
# ==========================================================================

def analysis_2(data_table):
    """Correlate output_probability against all 16 metrics with Bonferroni."""
    metrics = [
        'avg_compression_ratio', 'avg_score_decay_rate', 'avg_branching_factor',
        'avg_layers_visited', 'total_nodes', 'total_edges', 'num_bottlenecks',
        'weight_skewness', 'weight_kurtosis', 'top_10pct_weight_share',
        'num_supernodes', 'size_entropy', 'peak_layer', 'total_energy',
        'avg_bottleneck_layer', 'bottleneck_layer_std',
    ]
    n_tests = len(metrics)
    bonferroni_alpha = 0.05 / n_tests
    results = {}

    for model in MODELS + ['combined']:
        rows = data_table if model == 'combined' else [r for r in data_table if r['model'] == model]
        probs = np.array([r['output_probability'] for r in rows])
        corrs = {}
        sig_count = 0
        bonf_count = 0

        for metric in metrics:
            vals = np.array([r[metric] for r in rows])
            if np.std(vals) == 0 or np.std(probs) == 0:
                corrs[metric] = {'error': 'No variance'}
                continue
            sr, sp = spearmanr(probs, vals)
            pr, pp = pearsonr(probs, vals)
            survives = sp < bonferroni_alpha
            if sp < 0.05:
                sig_count += 1
            if survives:
                bonf_count += 1
            corrs[metric] = {
                'spearman_r': float(sr), 'spearman_p': float(sp),
                'pearson_r': float(pr), 'pearson_p': float(pp),
                'n': len(rows),
                'survives_bonferroni': survives,
            }

        results[model] = {
            'n_tests': n_tests,
            'bonferroni_alpha': bonferroni_alpha,
            'correlations': corrs,
            'significant_uncorrected': sig_count,
            'significant_bonferroni': bonf_count,
        }
    return results


# ==========================================================================
# Analysis 3: Multiple Regression (OLS)
# ==========================================================================

def ols_regression(y, X_raw, feature_names):
    """OLS regression with full diagnostics. Pure numpy."""
    n = len(y)
    X = np.column_stack([np.ones(n), X_raw])
    p = X.shape[1]

    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p) if n > p else 0

    k = p - 1
    ms_reg = (ss_tot - ss_res) / k if k > 0 else 0
    ms_res = ss_res / (n - p) if n > p else 1e-10
    f_stat = ms_reg / ms_res if ms_res > 0 else 0
    try:
        f_p_value = 1 - f_dist.cdf(f_stat, k, n - p) if n > p else 1.0
    except Exception:
        f_p_value = 1.0

    sigma_sq = ss_res / (n - p) if n > p else 0
    try:
        cov_beta = sigma_sq * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov_beta = np.zeros((p, p))

    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0))
    t_stats = np.where(se_beta > 0, beta / se_beta, 0)

    if n > p:
        t_p_values = [float(2 * (1 - t_dist.cdf(abs(t), n - p))) for t in t_stats]
    else:
        t_p_values = [1.0] * p

    coefficients = {}
    names = ['intercept'] + feature_names
    for i, name in enumerate(names):
        coefficients[name] = {
            'beta': float(beta[i]),
            'std_error': float(se_beta[i]),
            't_statistic': float(t_stats[i]),
            'p_value': float(t_p_values[i]),
            'significant_at_05': t_p_values[i] < 0.05,
        }

    warning = None
    if n < p * 3:
        warning = f"Low power: {n} observations for {k} predictors ({n - p} residual df)"

    return {
        'r_squared': float(r_squared),
        'adj_r_squared': float(adj_r_squared),
        'f_statistic': float(f_stat),
        'f_p_value': float(f_p_value),
        'n_observations': n,
        'n_predictors': k,
        'residual_std_error': float(np.sqrt(sigma_sq)),
        'coefficients': coefficients,
        'warning': warning,
    }


def analysis_3(data_table, corr_results):
    """Multiple regression: predict output_probability from circuit features."""
    full_features = [
        'total_nodes', 'num_bottlenecks', 'avg_layers_visited',
        'avg_compression_ratio', 'avg_score_decay_rate',
        'weight_kurtosis', 'size_entropy', 'total_energy',
        'avg_bottleneck_layer',
    ]

    results = {}
    for model in MODELS + ['combined']:
        rows = data_table if model == 'combined' else [r for r in data_table if r['model'] == model]
        y = np.array([r['output_probability'] for r in rows])

        # Full model
        X_full = np.column_stack([np.array([r[f] for r in rows]) for f in full_features])

        # Standardize predictors for numerical stability
        X_means = X_full.mean(axis=0)
        X_stds = X_full.std(axis=0)
        X_stds[X_stds == 0] = 1  # prevent div by zero
        X_norm = (X_full - X_means) / X_stds

        full_result = ols_regression(y, X_norm, full_features)

        # Reduced model: pick top 3 features by Spearman |r| from correlation matrix
        model_corrs = corr_results.get(model, {}).get('correlations', {})
        ranked = sorted(
            [(f, abs(model_corrs.get(f, {}).get('spearman_r', 0))) for f in full_features],
            key=lambda x: x[1], reverse=True
        )
        reduced_features = [f for f, _ in ranked[:3]]

        X_red = np.column_stack([np.array([r[f] for r in rows]) for f in reduced_features])
        X_red_means = X_red.mean(axis=0)
        X_red_stds = X_red.std(axis=0)
        X_red_stds[X_red_stds == 0] = 1
        X_red_norm = (X_red - X_red_means) / X_red_stds

        reduced_result = ols_regression(y, X_red_norm, reduced_features)
        reduced_result['features_used'] = reduced_features

        results[model] = {
            'full_model': full_result,
            'reduced_model': reduced_result,
        }

    return results


# ==========================================================================
# Analysis 4: ANOVA / Kruskal-Wallis
# ==========================================================================

def analysis_4(data_table):
    """Test whether each metric differs by category."""
    metrics = [
        'avg_compression_ratio', 'avg_score_decay_rate', 'avg_branching_factor',
        'num_bottlenecks', 'total_energy', 'avg_bottleneck_layer',
        'output_probability', 'weight_kurtosis', 'size_entropy',
    ]
    n_total_tests = len(MODELS) * len(metrics)
    bonferroni_alpha = 0.05 / n_total_tests
    results = {}
    sig_uncorr = 0
    sig_bonf = 0

    for model in MODELS:
        rows = [r for r in data_table if r['model'] == model]
        model_results = {}

        for metric in metrics:
            groups = {}
            for cat in CATEGORIES:
                groups[cat] = [r[metric] for r in rows if r['category'] == cat]

            group_list = [groups[cat] for cat in ['chemistry', 'geography', 'history']]

            # Shapiro-Wilk normality test
            normality = {}
            all_normal = True
            for cat in CATEGORIES:
                vals = groups[cat]
                if len(vals) >= 3:
                    try:
                        sw_stat, sw_p = shapiro(vals)
                        is_normal = sw_p > 0.05
                    except Exception:
                        is_normal = False
                        sw_p = 0
                else:
                    is_normal = False
                    sw_p = 0
                normality[cat] = {'shapiro_p': float(sw_p), 'is_normal': is_normal}
                if not is_normal:
                    all_normal = False

            # Choose test
            if all_normal:
                try:
                    stat, p = f_oneway(*group_list)
                    test_used = 'one_way_anova'
                except Exception:
                    stat, p = kruskal(*group_list)
                    test_used = 'kruskal_wallis'
            else:
                try:
                    stat, p = kruskal(*group_list)
                    test_used = 'kruskal_wallis'
                except Exception:
                    stat, p = 0, 1.0
                    test_used = 'error'

            # Effect size
            all_vals = sum(group_list, [])
            grand_mean = np.mean(all_vals)
            if test_used == 'one_way_anova':
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_list)
                ss_total = sum((v - grand_mean)**2 for v in all_vals)
                effect_size = ss_between / ss_total if ss_total > 0 else 0
                effect_name = 'eta_squared'
            else:
                n_total = len(all_vals)
                effect_size = (float(stat) - 3 + 1) / (n_total - 3) if n_total > 3 else 0
                effect_size = max(0, effect_size)
                effect_name = 'epsilon_squared'

            if float(p) < 0.05:
                sig_uncorr += 1
            if float(p) < bonferroni_alpha:
                sig_bonf += 1

            model_results[metric] = {
                'test_used': test_used,
                'statistic': float(stat),
                'p_value': float(p),
                'bonferroni_alpha': bonferroni_alpha,
                'survives_bonferroni': float(p) < bonferroni_alpha,
                effect_name: float(effect_size),
                'group_means': {cat: float(np.mean(groups[cat])) for cat in CATEGORIES},
                'group_stds': {cat: float(np.std(groups[cat])) for cat in CATEGORIES},
                'group_ns': {cat: len(groups[cat]) for cat in CATEGORIES},
                'normality': normality,
            }

        results[model] = model_results

    results['summary'] = {
        'total_tests': n_total_tests,
        'significant_uncorrected': sig_uncorr,
        'significant_bonferroni': sig_bonf,
    }
    return results


# ==========================================================================
# Analysis 5: Cross-Model Comparison (Mann-Whitney U)
# ==========================================================================

def analysis_5(data_table):
    """Compare GEMMA vs QWEN on each metric."""
    metrics = [
        'output_probability', 'avg_compression_ratio', 'avg_score_decay_rate',
        'avg_branching_factor', 'avg_layers_visited', 'total_nodes', 'total_edges',
        'num_bottlenecks', 'weight_skewness', 'weight_kurtosis',
        'top_10pct_weight_share', 'num_supernodes', 'size_entropy',
        'peak_layer', 'total_energy', 'avg_bottleneck_layer',
    ]
    n_tests = len(metrics)
    bonferroni_alpha = 0.05 / n_tests
    results = {'metrics': {}}
    sig_uncorr = 0
    sig_bonf = 0
    largest_effect = {'metric': '', 'r_rb': 0}

    gemma_rows = [r for r in data_table if r['model'] == 'gemma-2-2b']
    qwen_rows = [r for r in data_table if r['model'] == 'qwen3-4b']

    for metric in metrics:
        gvals = np.array([r[metric] for r in gemma_rows])
        qvals = np.array([r[metric] for r in qwen_rows])

        try:
            u_stat, p_val = mannwhitneyu(gvals, qvals, alternative='two-sided')
        except Exception:
            results['metrics'][metric] = {'error': 'Test failed'}
            continue

        # Rank-biserial correlation
        n1, n2 = len(gvals), len(qvals)
        r_rb = 1 - (2 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0

        # Interpretation
        abs_r = abs(r_rb)
        if abs_r < 0.1:
            interp = 'negligible'
        elif abs_r < 0.3:
            interp = 'small'
        elif abs_r < 0.5:
            interp = 'medium'
        else:
            interp = 'large'

        if p_val < 0.05:
            sig_uncorr += 1
        if p_val < bonferroni_alpha:
            sig_bonf += 1
        if abs_r > abs(largest_effect['r_rb']):
            largest_effect = {'metric': metric, 'r_rb': float(r_rb)}

        results['metrics'][metric] = {
            'gemma_mean': float(np.mean(gvals)),
            'gemma_std': float(np.std(gvals)),
            'gemma_median': float(np.median(gvals)),
            'qwen_mean': float(np.mean(qvals)),
            'qwen_std': float(np.std(qvals)),
            'qwen_median': float(np.median(qvals)),
            'u_statistic': float(u_stat),
            'p_value': float(p_val),
            'rank_biserial_r': float(r_rb),
            'effect_size_interpretation': interp,
            'survives_bonferroni': p_val < bonferroni_alpha,
            'bonferroni_alpha': bonferroni_alpha,
        }

    results['summary'] = {
        'total_tests': n_tests,
        'significant_uncorrected': sig_uncorr,
        'significant_bonferroni': sig_bonf,
        'largest_effect': largest_effect,
    }
    return results


# ==========================================================================
# Analysis 6: Effect Sizes
# ==========================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1 = np.std(group1, ddof=1) if n1 > 1 else 0
    s2 = np.std(group2, ddof=1) if n2 > 1 else 0
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if (n1 + n2) > 2 else 1
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def analysis_6(data_table, anova_results, mann_whitney_results):
    """Compute effect sizes for all comparisons."""
    anova_metrics = [
        'avg_compression_ratio', 'avg_score_decay_rate', 'avg_branching_factor',
        'num_bottlenecks', 'total_energy', 'avg_bottleneck_layer',
        'output_probability', 'weight_kurtosis', 'size_entropy',
    ]
    cat_pairs = [('chemistry', 'geography'), ('chemistry', 'history'), ('geography', 'history')]

    pairwise_d = {}
    for model in MODELS:
        rows = [r for r in data_table if r['model'] == model]
        model_d = {}
        for metric in anova_metrics:
            metric_d = {}
            for cat_a, cat_b in cat_pairs:
                va = [r[metric] for r in rows if r['category'] == cat_a]
                vb = [r[metric] for r in rows if r['category'] == cat_b]
                d = cohens_d(va, vb)
                abs_d = abs(d)
                if abs_d < 0.2:
                    interp = 'negligible'
                elif abs_d < 0.5:
                    interp = 'small'
                elif abs_d < 0.8:
                    interp = 'medium'
                else:
                    interp = 'large'
                metric_d[f'{cat_a}_vs_{cat_b}'] = {'d': float(d), 'interpretation': interp}
            model_d[metric] = metric_d
        pairwise_d[model] = model_d

    # Collect eta-squared from ANOVA results
    eta_sq = {}
    for model in MODELS:
        model_eta = {}
        model_anova = anova_results.get(model, {})
        for metric in anova_metrics:
            mr = model_anova.get(metric, {})
            eta = mr.get('eta_squared', mr.get('epsilon_squared', 0))
            model_eta[metric] = float(eta)
        eta_sq[model] = model_eta

    # Collect rank-biserial from Mann-Whitney
    rb_vals = {}
    mw_metrics = mann_whitney_results.get('metrics', {})
    for metric, vals in mw_metrics.items():
        if isinstance(vals, dict) and 'rank_biserial_r' in vals:
            rb_vals[metric] = float(vals['rank_biserial_r'])

    return {
        'pairwise_cohens_d': pairwise_d,
        'eta_squared_from_anova': eta_sq,
        'rank_biserial_from_mannwhitney': rb_vals,
    }


# ==========================================================================
# Analysis 7: Bottleneck Convergence vs Confidence
# ==========================================================================

def analysis_7(data_table):
    """Correlate mean_bottleneck_convergence with output_probability."""
    results = {}
    for model in MODELS + ['combined']:
        rows = data_table if model == 'combined' else [r for r in data_table if r['model'] == model]
        probs = np.array([r['output_probability'] for r in rows])
        convs = np.array([r['mean_bottleneck_convergence'] for r in rows])

        if np.std(convs) == 0 or np.std(probs) == 0:
            results[model] = {'error': 'No variance'}
            continue

        sr, sp = spearmanr(probs, convs)
        pr, pp = pearsonr(probs, convs)

        # By category
        cat_stats = {}
        for cat in CATEGORIES:
            cat_convs = [r['mean_bottleneck_convergence'] for r in rows if r['category'] == cat]
            cat_stats[cat] = {'mean': float(np.mean(cat_convs)), 'std': float(np.std(cat_convs))}

        # Kruskal-Wallis across categories
        groups = [[r['mean_bottleneck_convergence'] for r in rows if r['category'] == cat]
                  for cat in CATEGORIES]
        try:
            kw_stat, kw_p = kruskal(*groups)
        except Exception:
            kw_stat, kw_p = 0, 1.0

        results[model] = {
            'convergence_vs_confidence': {
                'spearman_r': float(sr), 'spearman_p': float(sp),
                'pearson_r': float(pr), 'pearson_p': float(pp),
                'n': len(rows),
            },
            'convergence_by_category': cat_stats,
            'kruskal_wallis': {'statistic': float(kw_stat), 'p_value': float(kw_p)},
        }
    return results


# ==========================================================================
# Analysis 8: Information Concentration Index
# ==========================================================================

def analysis_8(data_table):
    """Analyze the concentration_index composite metric."""
    results = {}
    for model in MODELS + ['combined']:
        rows = data_table if model == 'combined' else [r for r in data_table if r['model'] == model]
        probs = np.array([r['output_probability'] for r in rows])
        concs = np.array([r['concentration_index'] for r in rows])

        desc = {
            'mean': float(np.mean(concs)), 'std': float(np.std(concs)),
            'median': float(np.median(concs)),
            'min': float(np.min(concs)), 'max': float(np.max(concs)),
        }

        if np.std(concs) == 0 or np.std(probs) == 0:
            results[model] = {'descriptive_stats': desc, 'error': 'No variance'}
            continue

        sr, sp = spearmanr(probs, concs)
        pr, pp = pearsonr(probs, concs)

        cat_stats = {}
        for cat in CATEGORIES:
            cat_concs = [r['concentration_index'] for r in rows if r['category'] == cat]
            cat_stats[cat] = {'mean': float(np.mean(cat_concs)), 'std': float(np.std(cat_concs))}

        groups = [[r['concentration_index'] for r in rows if r['category'] == cat]
                  for cat in CATEGORIES]
        try:
            kw_stat, kw_p = kruskal(*groups)
        except Exception:
            kw_stat, kw_p = 0, 1.0

        results[model] = {
            'descriptive_stats': desc,
            'vs_output_probability': {
                'spearman_r': float(sr), 'spearman_p': float(sp),
                'pearson_r': float(pr), 'pearson_p': float(pp),
                'n': len(rows),
            },
            'by_category': cat_stats,
            'kruskal_wallis': {'statistic': float(kw_stat), 'p_value': float(kw_p)},
        }
    return results


# ==========================================================================
# Report Generation
# ==========================================================================

def generate_report(results):
    lines = []
    lines.append("# Stage 2: Statistical Deepdive Report")
    lines.append(f"\n**Date**: {datetime.date.today()}")
    lines.append("**Circuits**: 60 (30 prompts x 2 models)")
    lines.append("**Analyses**: 8 statistical categories")
    lines.append("")
    lines.append("---")

    # 1. Confidence vs Bottleneck Position
    lines.append("\n## 1. Confidence vs Bottleneck Position\n")
    a1 = results.get('confidence_vs_bottleneck_position', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        lines.append("| Metric | Spearman r | p-value | Pearson r | p-value | n |")
        lines.append("|--------|-----------|---------|----------|---------|---|")
        mr = a1.get(model, {})
        for metric, vals in mr.items():
            if isinstance(vals, dict) and 'spearman_r' in vals:
                lines.append(
                    f"| {metric} | {vals['spearman_r']:.3f} | {vals['spearman_p']:.4f} | "
                    f"{vals['pearson_r']:.3f} | {vals['pearson_p']:.4f} | {vals['n']} |"
                )
        lines.append("")

    # 2. Expanded Correlation Matrix
    lines.append("\n## 2. Expanded Correlation Matrix (vs output_probability)\n")
    a2 = results.get('expanded_correlation_matrix', {})
    for model in MODELS:
        mr = a2.get(model, {})
        lines.append(f"### {model.upper()} (Bonferroni α = {mr.get('bonferroni_alpha', 0):.4f})\n")
        lines.append(f"Significant uncorrected: {mr.get('significant_uncorrected', 0)} / {mr.get('n_tests', 0)}")
        lines.append(f"Survives Bonferroni: {mr.get('significant_bonferroni', 0)} / {mr.get('n_tests', 0)}\n")
        lines.append("| Metric | Spearman r | p-value | Bonferroni |")
        lines.append("|--------|-----------|---------|------------|")
        for metric, vals in mr.get('correlations', {}).items():
            if isinstance(vals, dict) and 'spearman_r' in vals:
                flag = "**YES**" if vals.get('survives_bonferroni') else ""
                lines.append(
                    f"| {metric} | {vals['spearman_r']:.3f} | {vals['spearman_p']:.4f} | {flag} |"
                )
        lines.append("")

    # 3. Multiple Regression
    lines.append("\n## 3. Multiple Regression (predicting output_probability)\n")
    a3 = results.get('multiple_regression', {})
    for model in MODELS:
        mr = a3.get(model, {})
        for mtype in ['full_model', 'reduced_model']:
            reg = mr.get(mtype, {})
            label = mtype.replace('_', ' ').title()
            lines.append(f"### {model.upper()} — {label}\n")
            if reg.get('warning'):
                lines.append(f"**Warning**: {reg['warning']}\n")
            if 'features_used' in reg:
                lines.append(f"**Features**: {', '.join(reg['features_used'])}\n")
            lines.append(f"- R² = {reg.get('r_squared', 0):.3f}, Adjusted R² = {reg.get('adj_r_squared', 0):.3f}")
            lines.append(f"- F({reg.get('n_predictors', 0)}, {reg.get('n_observations', 0) - reg.get('n_predictors', 0) - 1}) = {reg.get('f_statistic', 0):.2f}, p = {reg.get('f_p_value', 1):.4f}")
            lines.append("")
            lines.append("| Predictor | β | SE | t | p-value | Sig? |")
            lines.append("|-----------|-----|-----|-----|---------|------|")
            for name, coef in reg.get('coefficients', {}).items():
                sig = "**YES**" if coef.get('significant_at_05') else ""
                lines.append(
                    f"| {name} | {coef['beta']:.4f} | {coef['std_error']:.4f} | "
                    f"{coef['t_statistic']:.2f} | {coef['p_value']:.4f} | {sig} |"
                )
            lines.append("")

    # 4. ANOVA / Kruskal-Wallis
    lines.append("\n## 4. ANOVA / Kruskal-Wallis (metric ~ category)\n")
    a4 = results.get('anova_kruskal', {})
    summ = a4.get('summary', {})
    lines.append(f"Total tests: {summ.get('total_tests', 0)}, "
                 f"Significant uncorrected: {summ.get('significant_uncorrected', 0)}, "
                 f"Survives Bonferroni: {summ.get('significant_bonferroni', 0)}\n")
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        lines.append("| Metric | Test | Stat | p-value | Effect Size | Bonf? | Chem Mean | Geo Mean | Hist Mean |")
        lines.append("|--------|------|------|---------|-------------|-------|-----------|----------|-----------|")
        mr = a4.get(model, {})
        for metric, vals in mr.items():
            if not isinstance(vals, dict) or 'test_used' not in vals:
                continue
            es_val = vals.get('eta_squared', vals.get('epsilon_squared', 0))
            flag = "**YES**" if vals.get('survives_bonferroni') else ""
            gm = vals.get('group_means', {})
            lines.append(
                f"| {metric} | {vals['test_used'][:2].upper()} | {vals['statistic']:.2f} | "
                f"{vals['p_value']:.4f} | {es_val:.3f} | {flag} | "
                f"{gm.get('chemistry', 0):.3f} | {gm.get('geography', 0):.3f} | {gm.get('history', 0):.3f} |"
            )
        lines.append("")

    # 5. Cross-Model Comparison
    lines.append("\n## 5. Cross-Model Comparison (GEMMA vs QWEN)\n")
    a5 = results.get('cross_model_comparison', {})
    summ = a5.get('summary', {})
    lines.append(f"Total tests: {summ.get('total_tests', 0)}, "
                 f"Significant: {summ.get('significant_uncorrected', 0)} uncorrected, "
                 f"{summ.get('significant_bonferroni', 0)} Bonferroni\n")
    le = summ.get('largest_effect', {})
    lines.append(f"Largest effect: **{le.get('metric', '')}** (r = {le.get('r_rb', 0):.3f})\n")
    lines.append("| Metric | GEMMA Mean | QWEN Mean | U | p-value | r_rb | Effect | Bonf? |")
    lines.append("|--------|-----------|----------|---|---------|------|--------|-------|")
    for metric, vals in a5.get('metrics', {}).items():
        if not isinstance(vals, dict) or 'u_statistic' not in vals:
            continue
        flag = "**YES**" if vals.get('survives_bonferroni') else ""
        lines.append(
            f"| {metric} | {vals['gemma_mean']:.3f} | {vals['qwen_mean']:.3f} | "
            f"{vals['u_statistic']:.0f} | {vals['p_value']:.4f} | "
            f"{vals['rank_biserial_r']:.3f} | {vals['effect_size_interpretation']} | {flag} |"
        )
    lines.append("")

    # 6. Effect Sizes
    lines.append("\n## 6. Effect Sizes (Cohen's d)\n")
    a6 = results.get('effect_sizes', {})
    for model in MODELS:
        lines.append(f"### {model.upper()}\n")
        pairwise = a6.get('pairwise_cohens_d', {}).get(model, {})
        lines.append("| Metric | Chem vs Geo | Chem vs Hist | Geo vs Hist |")
        lines.append("|--------|-------------|-------------|-------------|")
        for metric, pairs in pairwise.items():
            cg = pairs.get('chemistry_vs_geography', {})
            ch = pairs.get('chemistry_vs_history', {})
            gh = pairs.get('geography_vs_history', {})
            lines.append(
                f"| {metric} | {cg.get('d', 0):.2f} ({cg.get('interpretation', '')[0]}) | "
                f"{ch.get('d', 0):.2f} ({ch.get('interpretation', '')[0]}) | "
                f"{gh.get('d', 0):.2f} ({gh.get('interpretation', '')[0]}) |"
            )
        lines.append("")

    # 7. Bottleneck Convergence
    lines.append("\n## 7. Bottleneck Convergence vs Confidence\n")
    a7 = results.get('bottleneck_convergence', {})
    lines.append("| Model | Spearman r | p-value | Pearson r | p-value |")
    lines.append("|-------|-----------|---------|----------|---------|")
    for model in MODELS:
        mr = a7.get(model, {})
        cc = mr.get('convergence_vs_confidence', {})
        if 'spearman_r' in cc:
            lines.append(
                f"| {model} | {cc['spearman_r']:.3f} | {cc['spearman_p']:.4f} | "
                f"{cc['pearson_r']:.3f} | {cc['pearson_p']:.4f} |"
            )
    lines.append("")

    # 8. Concentration Index
    lines.append("\n## 8. Information Concentration Index\n")
    lines.append("`concentration_index = top_10pct_weight_share * weight_kurtosis / total_edges`\n")
    a8 = results.get('concentration_index', {})
    lines.append("| Model | Mean ± Std | vs Confidence (r) | p-value |")
    lines.append("|-------|-----------|-------------------|---------|")
    for model in MODELS:
        mr = a8.get(model, {})
        ds = mr.get('descriptive_stats', {})
        vp = mr.get('vs_output_probability', {})
        lines.append(
            f"| {model} | {ds.get('mean', 0):.6f}±{ds.get('std', 0):.6f} | "
            f"{vp.get('spearman_r', 0):.3f} | {vp.get('spearman_p', 1):.4f} |"
        )
    lines.append("")

    # Power Limitations
    lines.append("\n## Statistical Power and Limitations\n")
    lines.append("- **Per-model sample size**: n=30 (10 per category)")
    lines.append("- **Correlation detection**: At n=30, α=0.05 has ~80% power for |r| ≥ 0.45")
    lines.append("- **After Bonferroni (α=0.003)**: Only |r| ≥ 0.60 reliably detected")
    lines.append("- **ANOVA with n=10/group**: ~80% power for large effects (f ≥ 0.40)")
    lines.append("- **Full regression (9 predictors, n=30)**: Only 21 residual df; reduced model preferred")
    lines.append("- **Mann-Whitney (n1=n2=30)**: ~80% power for medium-to-large effects")
    lines.append("")

    return '\n'.join(lines)


# ==========================================================================
# Main
# ==========================================================================

def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: STATISTICAL DEEPDIVE")
    print("  Comprehensive statistical analysis of 60 circuits")
    print("=" * 70)

    # Phase 0: Build data table
    print("\n[Phase 0] Building per-circuit data table...")
    data_table = build_data_table()
    print(f"  Assembled {len(data_table)} rows")

    if not data_table:
        print("  [ERROR] No data! Exiting.")
        return

    # Save data table
    dt_path = output_dir / 'data_table.json'
    with open(dt_path, 'w', encoding='utf-8') as f:
        json.dump(data_table, f, indent=2)
    print(f"  Saved data table: {dt_path}")

    results = {}

    # 1. Confidence vs Bottleneck Position
    print("\n[1/8] Confidence vs Bottleneck Position...")
    results['confidence_vs_bottleneck_position'] = analysis_1(data_table)
    for model in MODELS:
        mr = results['confidence_vs_bottleneck_position'].get(model, {})
        abl = mr.get('avg_bottleneck_layer', {})
        if 'spearman_r' in abl:
            print(f"  {model}: avg_bn_layer r={abl['spearman_r']:.3f}, p={abl['spearman_p']:.4f}")

    # 2. Expanded Correlation Matrix
    print("\n[2/8] Expanded Correlation Matrix...")
    results['expanded_correlation_matrix'] = analysis_2(data_table)
    for model in MODELS:
        mr = results['expanded_correlation_matrix'].get(model, {})
        print(f"  {model}: {mr.get('significant_uncorrected', 0)}/{mr.get('n_tests', 0)} sig (uncorrected), "
              f"{mr.get('significant_bonferroni', 0)} survive Bonferroni")

    # 3. Multiple Regression
    print("\n[3/8] Multiple Regression...")
    results['multiple_regression'] = analysis_3(data_table, results['expanded_correlation_matrix'])
    for model in MODELS:
        mr = results['multiple_regression'].get(model, {})
        full = mr.get('full_model', {})
        red = mr.get('reduced_model', {})
        print(f"  {model} full: R²={full.get('r_squared', 0):.3f}, adj={full.get('adj_r_squared', 0):.3f}, "
              f"F p={full.get('f_p_value', 1):.4f}")
        print(f"  {model} reduced: R²={red.get('r_squared', 0):.3f}, features={red.get('features_used', [])}")

    # 4. ANOVA / Kruskal-Wallis
    print("\n[4/8] ANOVA / Kruskal-Wallis...")
    results['anova_kruskal'] = analysis_4(data_table)
    summ = results['anova_kruskal'].get('summary', {})
    print(f"  {summ.get('significant_uncorrected', 0)}/{summ.get('total_tests', 0)} sig uncorrected, "
          f"{summ.get('significant_bonferroni', 0)} survive Bonferroni")

    # 5. Cross-Model Comparison
    print("\n[5/8] Cross-Model Comparison (Mann-Whitney)...")
    results['cross_model_comparison'] = analysis_5(data_table)
    summ = results['cross_model_comparison'].get('summary', {})
    le = summ.get('largest_effect', {})
    print(f"  {summ.get('significant_bonferroni', 0)}/{summ.get('total_tests', 0)} survive Bonferroni")
    print(f"  Largest effect: {le.get('metric', '')} (r={le.get('r_rb', 0):.3f})")

    # 6. Effect Sizes
    print("\n[6/8] Effect Sizes...")
    results['effect_sizes'] = analysis_6(
        data_table, results['anova_kruskal'], results['cross_model_comparison']
    )
    # Count large effects
    for model in MODELS:
        pd = results['effect_sizes'].get('pairwise_cohens_d', {}).get(model, {})
        large_count = sum(
            1 for metric_d in pd.values()
            for pair_d in metric_d.values()
            if isinstance(pair_d, dict) and pair_d.get('interpretation') == 'large'
        )
        print(f"  {model}: {large_count} large Cohen's d effects")

    # 7. Bottleneck Convergence
    print("\n[7/8] Bottleneck Convergence vs Confidence...")
    results['bottleneck_convergence'] = analysis_7(data_table)
    for model in MODELS:
        mr = results['bottleneck_convergence'].get(model, {})
        cc = mr.get('convergence_vs_confidence', {})
        if 'spearman_r' in cc:
            print(f"  {model}: r={cc['spearman_r']:.3f}, p={cc['spearman_p']:.4f}")

    # 8. Concentration Index
    print("\n[8/8] Concentration Index...")
    results['concentration_index'] = analysis_8(data_table)
    for model in MODELS:
        mr = results['concentration_index'].get(model, {})
        vp = mr.get('vs_output_probability', {})
        if 'spearman_r' in vp:
            print(f"  {model}: r={vp['spearman_r']:.3f}, p={vp['spearman_p']:.4f}")

    # Save results
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    results['metadata'] = {
        'timestamp': str(datetime.datetime.now()),
        'n_circuits': len(data_table),
        'models': MODELS,
        'categories': list(CATEGORIES.keys()),
    }

    json_path = output_dir / 'statistical_deepdive_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    report = generate_report(results)
    report_path = output_dir / 'statistical_deepdive_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # Key findings
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    # Confidence vs bottleneck layer
    for model in MODELS:
        a1 = results['confidence_vs_bottleneck_position'].get(model, {})
        abl = a1.get('avg_bottleneck_layer', {})
        if 'spearman_r' in abl:
            sig = "SIGNIFICANT" if abl['spearman_p'] < 0.05 else "not significant"
            print(f"\n  {model}: Confidence vs avg bottleneck layer: r={abl['spearman_r']:.3f} ({sig})")

    # Top correlations surviving Bonferroni
    for model in MODELS:
        corrs = results['expanded_correlation_matrix'].get(model, {}).get('correlations', {})
        surviving = [(m, v) for m, v in corrs.items()
                     if isinstance(v, dict) and v.get('survives_bonferroni')]
        if surviving:
            print(f"\n  {model} Bonferroni-surviving correlations:")
            for m, v in surviving:
                print(f"    {m}: r={v['spearman_r']:.3f}")

    print(f"\n  Complete! Results at {output_dir}")


if __name__ == '__main__':
    main()
