#!/usr/bin/env python3
"""Stage 2: Unused Columns Analysis - exploiting underused metrics in data_table.json.

Analyzes 8-10 data table columns that were never or barely used in prior analyses:
  1. Bottleneck span (max - min layer)
  2. Pareto edge efficiency (edges_for_50pct_weight)
  3. Weight variance (weight_std, weight_abs_mean)
  4. Bottleneck position consistency (bottleneck_layer_std)
  5. Concentration index deep dive
  6. Interaction regression (category x predictor)

Output: data/stage_2_unused_columns/unused_columns_results.json + report.md
"""

import json
import sys
import io
import math
import datetime
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_unused_columns'
DATA_TABLE = DATA_DIR / 'stage_2_statistical_deepdive' / 'data_table.json'

MODELS = ['gemma-2-2b', 'qwen3-4b']
CATEGORIES = ['chemistry', 'geography', 'history']


# ==========================================================================
# Statistics helpers (pure Python)
# ==========================================================================

def pearson_r(x, y):
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a-mx)**2 for a in x))
    dy = math.sqrt(sum((b-my)**2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0, 1.0
    r = num / (dx * dy)
    if abs(r) >= 1.0:
        return min(max(r, -1), 1), 0.0
    t = r * math.sqrt((n-2) / (1-r*r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def mean(x):
    return sum(x) / len(x) if x else 0


def std(x):
    if len(x) < 2:
        return 0
    m = mean(x)
    return math.sqrt(sum((v-m)**2 for v in x) / (len(x)-1))


def median(x):
    s = sorted(x)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n//2 - 1] + s[n//2]) / 2


def solve_linear(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        max_row = col
        for row in range(col+1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for row in range(col+1, n):
            f = M[row][col] / M[col][col]
            for j in range(col, n+1):
                M[row][j] -= f * M[col][j]
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            continue
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


def ols(X, y):
    """OLS regression. X includes intercept. Returns beta, y_hat, r_squared."""
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][j]*X[i][k] for i in range(n)) for k in range(p)] for j in range(p)]
    Xty = [sum(X[i][j]*y[i] for i in range(n)) for j in range(p)]
    beta = solve_linear(XtX, Xty)
    y_hat = [sum(X[i][j]*beta[j] for j in range(p)) for i in range(n)]
    y_mean = sum(y) / n
    ss_tot = sum((yi-y_mean)**2 for yi in y)
    ss_res = sum((yi-yhi)**2 for yi, yhi in zip(y, y_hat))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return beta, y_hat, r2


def cohen_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    pooled = math.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    return (mx - my) / pooled if pooled > 0 else 0.0


# ==========================================================================
# Analysis Functions
# ==========================================================================

def analyze_bottleneck_span(data, model_data):
    """D1.1: Bottleneck span = max_bottleneck_layer - min_bottleneck_layer."""
    print("\n  [D1.1] Bottleneck Span Analysis...")
    results = {}

    for model_name, rows in model_data.items():
        spans = [r['max_bottleneck_layer'] - r['min_bottleneck_layer'] for r in rows]
        confs = [r['output_probability'] for r in rows]

        r, p = pearson_r(spans, confs)
        avg_span = mean(spans)
        med_span = median(spans)

        # Split by median span
        med = median(spans)
        small = [(s, c) for s, c in zip(spans, confs) if s <= med]
        large = [(s, c) for s, c in zip(spans, confs) if s > med]

        small_conf = [c for _, c in small] if small else [0]
        large_conf = [c for _, c in large] if large else [0]
        d = cohen_d(small_conf, large_conf)

        # Per-category
        cat_stats = {}
        for cat in CATEGORIES:
            cat_rows = [r for r in rows if r['category'] == cat]
            cat_spans = [r['max_bottleneck_layer'] - r['min_bottleneck_layer'] for r in cat_rows]
            cat_confs = [r['output_probability'] for r in cat_rows]
            cr, cp = pearson_r(cat_spans, cat_confs) if len(cat_rows) >= 5 else (0, 1)
            cat_stats[cat] = {'n': len(cat_rows), 'avg_span': mean(cat_spans),
                              'avg_conf': mean(cat_confs), 'r': cr, 'p': cp}

        results[model_name] = {
            'n': len(rows), 'avg_span': avg_span, 'median_span': med_span,
            'r_vs_confidence': r, 'p_vs_confidence': p,
            'small_span_conf': mean(small_conf), 'large_span_conf': mean(large_conf),
            'cohen_d': d, 'per_category': cat_stats
        }
        sig = '*' if p < 0.05 else ''
        print(f"    {model_name}: avg span={avg_span:.1f}, r={r:.3f}{sig}, Cohen d={d:.2f}")

    return results


def analyze_pareto_edges(data, model_data):
    """D1.2: Pareto edge analysis using edges_for_50pct_weight."""
    print("\n  [D1.2] Pareto Edge Analysis...")
    results = {}

    for model_name, rows in model_data.items():
        edges_50 = [r['edges_for_50pct_weight'] for r in rows]
        total_edges = [r['total_edges'] for r in rows]
        confs = [r['output_probability'] for r in rows]

        # Efficiency ratio
        eff_ratios = [e50/te if te > 0 else 0 for e50, te in zip(edges_50, total_edges)]

        r_raw, p_raw = pearson_r(edges_50, confs)
        r_eff, p_eff = pearson_r(eff_ratios, confs)

        # Per-category
        cat_stats = {}
        for cat in CATEGORIES:
            cat_rows = [r for r in rows if r['category'] == cat]
            cat_eff = [r['edges_for_50pct_weight']/r['total_edges'] if r['total_edges']>0 else 0
                       for r in cat_rows]
            cat_stats[cat] = {'n': len(cat_rows), 'avg_efficiency': mean(cat_eff),
                              'avg_edges_50': mean([r['edges_for_50pct_weight'] for r in cat_rows])}

        results[model_name] = {
            'n': len(rows),
            'avg_edges_for_50pct': mean(edges_50), 'avg_efficiency_ratio': mean(eff_ratios),
            'raw_r': r_raw, 'raw_p': p_raw,
            'efficiency_r': r_eff, 'efficiency_p': p_eff,
            'per_category': cat_stats
        }
        sig_r = '*' if p_raw < 0.05 else ''
        sig_e = '*' if p_eff < 0.05 else ''
        print(f"    {model_name}: avg eff={mean(eff_ratios):.3f}, "
              f"r_raw={r_raw:.3f}{sig_r}, r_eff={r_eff:.3f}{sig_e}")

    return results


def analyze_weight_variance(data, model_data):
    """D1.3: Weight variance using weight_std and weight_abs_mean."""
    print("\n  [D1.3] Weight Variance Analysis...")
    results = {}

    for model_name, rows in model_data.items():
        w_std = [r['weight_std'] for r in rows]
        w_abs = [r['weight_abs_mean'] for r in rows]
        confs = [r['output_probability'] for r in rows]

        # Coefficient of variation
        cvs = [s/a if a > 0 else 0 for s, a in zip(w_std, w_abs)]

        r_std, p_std = pearson_r(w_std, confs)
        r_abs, p_abs = pearson_r(w_abs, confs)
        r_cv, p_cv = pearson_r(cvs, confs)

        results[model_name] = {
            'n': len(rows),
            'avg_weight_std': mean(w_std), 'avg_weight_abs_mean': mean(w_abs),
            'avg_cv': mean(cvs),
            'std_r': r_std, 'std_p': p_std,
            'abs_r': r_abs, 'abs_p': p_abs,
            'cv_r': r_cv, 'cv_p': p_cv,
        }
        sig_s = '*' if p_std < 0.05 else ''
        sig_c = '*' if p_cv < 0.05 else ''
        print(f"    {model_name}: avg CV={mean(cvs):.3f}, "
              f"r_std={r_std:.3f}{sig_s}, r_cv={r_cv:.3f}{sig_c}")

    return results


def analyze_bottleneck_consistency(data, model_data):
    """D1.4: Bottleneck position consistency using bottleneck_layer_std."""
    print("\n  [D1.4] Bottleneck Position Consistency...")
    results = {}

    for model_name, rows in model_data.items():
        bn_std = [r['bottleneck_layer_std'] for r in rows]
        confs = [r['output_probability'] for r in rows]

        r, p = pearson_r(bn_std, confs)

        # Per-category: do categories with higher Jaccard have lower bn_std?
        cat_stats = {}
        for cat in CATEGORIES:
            cat_rows = [r for r in rows if r['category'] == cat]
            cat_bn_std = [r['bottleneck_layer_std'] for r in cat_rows]
            cat_stats[cat] = {
                'n': len(cat_rows),
                'avg_bn_std': mean(cat_bn_std),
                'std_bn_std': std(cat_bn_std),
            }

        results[model_name] = {
            'n': len(rows), 'avg_bn_std': mean(bn_std),
            'r_vs_confidence': r, 'p_vs_confidence': p,
            'per_category': cat_stats,
        }
        sig = '*' if p < 0.05 else ''
        print(f"    {model_name}: avg bn_std={mean(bn_std):.2f}, r={r:.3f}{sig}")
        for cat, s in cat_stats.items():
            print(f"      {cat}: avg bn_std={s['avg_bn_std']:.2f}")

    return results


def analyze_concentration_index(data, model_data):
    """D1.5: Concentration index deep dive."""
    print("\n  [D1.5] Concentration Index Analysis...")
    results = {}

    for model_name, rows in model_data.items():
        ci = [r['concentration_index'] for r in rows]
        confs = [r['output_probability'] for r in rows]

        r, p = pearson_r(ci, confs)

        # Split by median
        med = median(ci)
        high_ci = [r for r in rows if r['concentration_index'] > med]
        low_ci = [r for r in rows if r['concentration_index'] <= med]

        high_conf = [r['output_probability'] for r in high_ci]
        low_conf = [r['output_probability'] for r in low_ci]
        d = cohen_d(high_conf, low_conf)

        # Correlate with other predictors
        predictor_corrs = {}
        for pred in ['total_energy', 'weight_kurtosis', 'total_nodes', 'weight_skewness']:
            pred_vals = [r[pred] for r in rows]
            pr, pp = pearson_r(ci, pred_vals)
            predictor_corrs[pred] = {'r': pr, 'p': pp}

        results[model_name] = {
            'n': len(rows), 'avg_ci': mean(ci), 'median_ci': med,
            'r_vs_confidence': r, 'p_vs_confidence': p,
            'high_ci_avg_conf': mean(high_conf) if high_conf else 0,
            'low_ci_avg_conf': mean(low_conf) if low_conf else 0,
            'cohen_d': d,
            'predictor_correlations': predictor_corrs,
        }
        sig = '*' if p < 0.05 else ''
        print(f"    {model_name}: r={r:.3f}{sig}, high_ci_conf={mean(high_conf):.3f}, "
              f"low_ci_conf={mean(low_conf):.3f}")

    return results


def analyze_interaction_regression(data, model_data):
    """D1.6: Interaction regression — category x predictor terms."""
    print("\n  [D1.6] Interaction Regression Analysis...")
    results = {}

    for model_name, rows in model_data.items():
        n = len(rows)
        y = [r['output_probability'] for r in rows]

        # Additive model: intercept + energy + kurtosis + nodes
        X_add = []
        for r in rows:
            X_add.append([1.0, r['total_energy'], r['weight_kurtosis'], float(r['total_nodes'])])

        beta_add, yhat_add, r2_add = ols(X_add, y)

        # Model with category dummies: intercept + energy + kurtosis + nodes + geo + hist
        # (chemistry is reference)
        X_cat = []
        for r in rows:
            geo = 1.0 if r['category'] == 'geography' else 0.0
            hist = 1.0 if r['category'] == 'history' else 0.0
            X_cat.append([1.0, r['total_energy'], r['weight_kurtosis'],
                         float(r['total_nodes']), geo, hist])

        beta_cat, yhat_cat, r2_cat = ols(X_cat, y)

        # Interaction model: additive + category dummies + energy×category interactions
        X_int = []
        for r in rows:
            e = r['total_energy']
            geo = 1.0 if r['category'] == 'geography' else 0.0
            hist = 1.0 if r['category'] == 'history' else 0.0
            X_int.append([1.0, e, r['weight_kurtosis'], float(r['total_nodes']),
                         geo, hist, e * geo, e * hist])

        beta_int, yhat_int, r2_int = ols(X_int, y)

        # Full interaction: also kurtosis×category, nodes×category
        X_full = []
        for r in rows:
            e = r['total_energy']
            k = r['weight_kurtosis']
            nd = float(r['total_nodes'])
            geo = 1.0 if r['category'] == 'geography' else 0.0
            hist = 1.0 if r['category'] == 'history' else 0.0
            X_full.append([1.0, e, k, nd, geo, hist,
                          e*geo, e*hist, k*geo, k*hist, nd*geo, nd*hist])

        beta_full, yhat_full, r2_full = ols(X_full, y)

        # Also test NEW predictors
        # Model with bottleneck_span + edges_for_50pct + weight_std added
        X_expanded = []
        for r in rows:
            bn_span = r['max_bottleneck_layer'] - r['min_bottleneck_layer']
            eff = r['edges_for_50pct_weight'] / r['total_edges'] if r['total_edges'] > 0 else 0
            X_expanded.append([1.0, r['total_energy'], r['weight_kurtosis'],
                              float(r['total_nodes']), bn_span, eff, r['weight_std']])

        beta_exp, yhat_exp, r2_exp = ols(X_expanded, y)

        pred_names_exp = ['intercept', 'total_energy', 'weight_kurtosis', 'total_nodes',
                          'bottleneck_span', 'edge_efficiency', 'weight_std']

        results[model_name] = {
            'n': n,
            'additive_r2': r2_add,
            'additive_plus_category_r2': r2_cat,
            'energy_interaction_r2': r2_int,
            'full_interaction_r2': r2_full,
            'expanded_predictors_r2': r2_exp,
            'expanded_coefficients': {name: beta_exp[i] for i, name in enumerate(pred_names_exp)},
            'r2_gain_category': r2_cat - r2_add,
            'r2_gain_interaction': r2_int - r2_cat,
            'r2_gain_full_interaction': r2_full - r2_cat,
            'r2_gain_new_predictors': r2_exp - r2_add,
        }

        print(f"    {model_name} R² comparison:")
        print(f"      Additive (3 pred):        {r2_add:.3f}")
        print(f"      + Category dummies:        {r2_cat:.3f} (+{r2_cat-r2_add:.3f})")
        print(f"      + Energy interaction:      {r2_int:.3f} (+{r2_int-r2_cat:.3f})")
        print(f"      + Full interactions:       {r2_full:.3f} (+{r2_full-r2_cat:.3f})")
        print(f"      Expanded (6 pred, no cat): {r2_exp:.3f} (+{r2_exp-r2_add:.3f})")

    return results


# ==========================================================================
# Comprehensive new correlations
# ==========================================================================

def correlate_all_unused(data, model_data):
    """Correlate ALL unused columns with confidence and each other."""
    print("\n  [BONUS] Comprehensive correlation matrix for unused columns...")
    unused_cols = ['edges_for_50pct_weight', 'weight_std', 'weight_abs_mean',
                   'weight_median', 'median_bottleneck_layer', 'min_bottleneck_layer',
                   'max_bottleneck_layer', 'bottleneck_layer_std', 'concentration_index']

    results = {}
    for model_name, rows in model_data.items():
        confs = [r['output_probability'] for r in rows]
        corr_with_conf = {}
        for col in unused_cols:
            vals = [r.get(col, 0) for r in rows]
            if all(v == vals[0] for v in vals):
                corr_with_conf[col] = {'r': 0, 'p': 1}
                continue
            r, p = pearson_r(vals, confs)
            corr_with_conf[col] = {'r': r, 'p': p}

        results[model_name] = corr_with_conf

        print(f"    {model_name} — unused columns vs confidence:")
        for col, vals in sorted(corr_with_conf.items(), key=lambda x: abs(x[1]['r']), reverse=True):
            sig = '*' if vals['p'] < 0.05 else ''
            print(f"      {col:30s}: r={vals['r']:.3f}{sig} (p={vals['p']:.4f})")

    return results


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Unused Columns Analysis')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: UNUSED COLUMNS DEEP DIVE")
    print("=" * 70)

    with open(DATA_TABLE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} circuits, {len(data[0])} columns each")

    model_data = {m: [r for r in data if r['model'] == m] for m in MODELS}
    for m in MODELS:
        print(f"  {m}: {len(model_data[m])} circuits")

    # Run all analyses
    r1 = analyze_bottleneck_span(data, model_data)
    r2 = analyze_pareto_edges(data, model_data)
    r3 = analyze_weight_variance(data, model_data)
    r4 = analyze_bottleneck_consistency(data, model_data)
    r5 = analyze_concentration_index(data, model_data)
    r6 = analyze_interaction_regression(data, model_data)
    r_corr = correlate_all_unused(data, model_data)

    results = {
        'metadata': {'timestamp': datetime.datetime.now().isoformat(), 'n_circuits': len(data)},
        'bottleneck_span': r1,
        'pareto_edges': r2,
        'weight_variance': r3,
        'bottleneck_consistency': r4,
        'concentration_index': r5,
        'interaction_regression': r6,
        'unused_correlations': r_corr,
    }

    # Save JSON
    json_path = output_dir / 'unused_columns_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Unused Columns Analysis Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Circuits**: {len(data)}\n")
    lines.append("---\n")

    # 1. Bottleneck Span
    lines.append("## 1. Bottleneck Span (max - min layer)\n")
    for m in MODELS:
        d = r1.get(m, {})
        if not d:
            continue
        sig = '**' if d['p_vs_confidence'] < 0.01 else '*' if d['p_vs_confidence'] < 0.05 else ''
        lines.append(f"### {m.upper()}\n")
        lines.append(f"- Avg span: **{d['avg_span']:.1f}** layers")
        lines.append(f"- Correlation with confidence: r={d['r_vs_confidence']:.3f}{sig}, p={d['p_vs_confidence']:.4f}")
        lines.append(f"- Small-span avg confidence: {d['small_span_conf']:.3f}")
        lines.append(f"- Large-span avg confidence: {d['large_span_conf']:.3f}")
        lines.append(f"- Cohen's d: {d['cohen_d']:.2f}\n")
        lines.append("| Category | N | Avg Span | Avg Conf | r | p |")
        lines.append("|----------|---|----------|----------|---|---|")
        for cat, s in d.get('per_category', {}).items():
            lines.append(f"| {cat} | {s['n']} | {s['avg_span']:.1f} | {s['avg_conf']:.3f} | {s['r']:.3f} | {s['p']:.4f} |")
        lines.append("")

    # 2. Pareto
    lines.append("## 2. Pareto Edge Efficiency\n")
    for m in MODELS:
        d = r2.get(m, {})
        if not d:
            continue
        lines.append(f"### {m.upper()}\n")
        lines.append(f"- Avg edges for 50% weight: **{d['avg_edges_for_50pct']:.0f}**")
        lines.append(f"- Avg efficiency ratio: **{d['avg_efficiency_ratio']:.3f}**")
        sig_r = '*' if d['raw_p'] < 0.05 else ''
        sig_e = '*' if d['efficiency_p'] < 0.05 else ''
        lines.append(f"- Raw count vs confidence: r={d['raw_r']:.3f}{sig_r}")
        lines.append(f"- Efficiency ratio vs confidence: r={d['efficiency_r']:.3f}{sig_e}\n")

    # 3. Weight variance
    lines.append("## 3. Weight Variance\n")
    for m in MODELS:
        d = r3.get(m, {})
        if not d:
            continue
        lines.append(f"### {m.upper()}\n")
        lines.append(f"- weight_std vs confidence: r={d['std_r']:.3f} (p={d['std_p']:.4f})")
        lines.append(f"- weight_abs_mean vs confidence: r={d['abs_r']:.3f} (p={d['abs_p']:.4f})")
        lines.append(f"- CV (std/abs_mean) vs confidence: r={d['cv_r']:.3f} (p={d['cv_p']:.4f})\n")

    # 4. Bottleneck consistency
    lines.append("## 4. Bottleneck Position Consistency\n")
    for m in MODELS:
        d = r4.get(m, {})
        if not d:
            continue
        lines.append(f"### {m.upper()}\n")
        lines.append(f"- Avg bottleneck_layer_std: **{d['avg_bn_std']:.2f}**")
        sig = '*' if d['p_vs_confidence'] < 0.05 else ''
        lines.append(f"- vs confidence: r={d['r_vs_confidence']:.3f}{sig}\n")
        for cat, s in d.get('per_category', {}).items():
            lines.append(f"- {cat}: avg bn_std={s['avg_bn_std']:.2f}")
        lines.append("")

    # 5. Concentration index
    lines.append("## 5. Concentration Index\n")
    for m in MODELS:
        d = r5.get(m, {})
        if not d:
            continue
        lines.append(f"### {m.upper()}\n")
        sig = '*' if d['p_vs_confidence'] < 0.05 else ''
        lines.append(f"- vs confidence: r={d['r_vs_confidence']:.3f}{sig}")
        lines.append(f"- High-CI avg conf: {d['high_ci_avg_conf']:.3f}")
        lines.append(f"- Low-CI avg conf: {d['low_ci_avg_conf']:.3f}")
        lines.append(f"- Cohen's d: {d['cohen_d']:.2f}\n")
        lines.append("**Collinearity with other predictors:**\n")
        for pred, vals in d.get('predictor_correlations', {}).items():
            lines.append(f"- {pred}: r={vals['r']:.3f}")
        lines.append("")

    # 6. Interaction regression
    lines.append("## 6. Interaction Regression\n")
    for m in MODELS:
        d = r6.get(m, {})
        if not d:
            continue
        lines.append(f"### {m.upper()}\n")
        lines.append("| Model | R² | Gain |")
        lines.append("|-------|-----|------|")
        lines.append(f"| Additive (3 predictors) | {d['additive_r2']:.3f} | — |")
        lines.append(f"| + Category dummies | {d['additive_plus_category_r2']:.3f} | +{d['r2_gain_category']:.3f} |")
        lines.append(f"| + Energy × Category | {d['energy_interaction_r2']:.3f} | +{d['r2_gain_interaction']:.3f} |")
        lines.append(f"| + Full interactions | {d['full_interaction_r2']:.3f} | +{d['r2_gain_full_interaction']:.3f} |")
        lines.append(f"| Expanded (6 pred, no category) | {d['expanded_predictors_r2']:.3f} | +{d['r2_gain_new_predictors']:.3f} |")
        lines.append("")

        if d.get('expanded_coefficients'):
            lines.append("**Expanded model coefficients:**\n")
            for name, coef in d['expanded_coefficients'].items():
                lines.append(f"- {name}: {coef:.6f}")
            lines.append("")

    # 7. Full correlation table
    lines.append("## 7. All Unused Columns vs Confidence\n")
    for m in MODELS:
        corrs = r_corr.get(m, {})
        if not corrs:
            continue
        lines.append(f"### {m.upper()}\n")
        lines.append("| Column | r | p | Significant? |")
        lines.append("|--------|---|---|-------------|")
        for col, vals in sorted(corrs.items(), key=lambda x: abs(x[1]['r']), reverse=True):
            sig = 'Yes' if vals['p'] < 0.05 else 'No'
            lines.append(f"| {col} | {vals['r']:.3f} | {vals['p']:.4f} | {sig} |")
        lines.append("")

    report_path = output_dir / 'unused_columns_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    for m in MODELS:
        reg = r6.get(m, {})
        if reg:
            print(f"  {m.upper()}: R² additive={reg['additive_r2']:.3f} → "
                  f"expanded={reg['expanded_predictors_r2']:.3f} → "
                  f"full_interaction={reg['full_interaction_r2']:.3f}")
    print("\nDone!")


if __name__ == '__main__':
    main()
