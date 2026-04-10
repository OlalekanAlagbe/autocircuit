#!/usr/bin/env python3
"""Stage 2: Outlier Analysis - Cook's distance and residual diagnostics.

Identifies outlier circuits using Cook's distance from the GEMMA regression model
(3-predictor: total_energy, weight_kurtosis, total_nodes -> output_probability).
Checks if outliers are domain-clustered.

Output: data/stage_2_outlier_analysis/outlier_results.json + outlier_report.md

Usage:
    python scripts/stage_2_outlier_analysis.py
"""

import json
import sys
import io
import os
import argparse
import math
import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_outlier_analysis'
DATA_TABLE = DATA_DIR / 'stage_2_statistical_deepdive' / 'data_table.json'


# ==========================================================================
# Linear Algebra (pure Python)
# ==========================================================================

def solve_linear_system(A, b):
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-12:
            continue

        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            x[i] = 0.0
            continue
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


def invert_matrix(M):
    """Invert a square matrix via Gauss-Jordan."""
    n = len(M)
    # Augment with identity
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]

    for col in range(n):
        # Pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-12:
            continue

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]


def mat_mult(A, B):
    """Matrix multiply A (m x n) * B (n x p) -> (m x p)."""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(M):
    """Transpose a matrix."""
    return [list(row) for row in zip(*M)]


def ols_regression(X, y):
    """OLS regression: beta = (X^T X)^-1 X^T y.

    X should include intercept column.
    Returns: beta, y_hat, residuals, hat_diag, mse
    """
    n = len(X)
    p = len(X[0])

    # X^T X
    XtX = [[sum(X[i][j] * X[i][k] for i in range(n)) for k in range(p)] for j in range(p)]
    Xty = [sum(X[i][j] * y[i] for i in range(n)) for j in range(p)]

    beta = solve_linear_system(XtX, Xty)

    # Fitted values and residuals
    y_hat = [sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    residuals = [y[i] - y_hat[i] for i in range(n)]

    # MSE
    sse = sum(r * r for r in residuals)
    mse = sse / (n - p) if n > p else sse

    # Hat matrix diagonal: h_ii = X_i (X^T X)^-1 X_i^T
    XtX_inv = invert_matrix(XtX)
    hat_diag = []
    for i in range(n):
        h_ii = sum(X[i][j] * sum(XtX_inv[j][k] * X[i][k] for k in range(p)) for j in range(p))
        hat_diag.append(h_ii)

    return beta, y_hat, residuals, hat_diag, mse


def cooks_distance(residuals, hat_diag, mse, p):
    """Compute Cook's distance for each observation."""
    n = len(residuals)
    D = []
    for i in range(n):
        h = hat_diag[i]
        e = residuals[i]
        if abs(1 - h) < 1e-12:
            D.append(0.0)
        else:
            D.append((e * e / (p * mse)) * (h / ((1 - h) * (1 - h))))
    return D


def r_squared(y, y_hat):
    """Compute R-squared."""
    n = len(y)
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Outlier Analysis')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  STAGE 2: OUTLIER ANALYSIS")
    print("=" * 70)

    # Load data table
    print("\n[1/4] Loading data table...")
    with open(DATA_TABLE, 'r', encoding='utf-8') as f:
        data_table = json.load(f)
    print(f"  {len(data_table)} circuits in table")

    # Split by model
    gemma = [r for r in data_table if r['model'] == 'gemma-2-2b']
    qwen = [r for r in data_table if r['model'] == 'qwen3-4b']
    print(f"  GEMMA: {len(gemma)}, QWEN: {len(qwen)}")

    results = {'metadata': {'timestamp': datetime.datetime.now().isoformat()}}

    # Process each model
    for model_name, data in [('gemma-2-2b', gemma), ('qwen3-4b', qwen)]:
        print(f"\n[2/4] Processing {model_name.upper()}...")
        n = len(data)
        if n < 5:
            print(f"  Too few circuits ({n}), skipping")
            continue

        # Predictors: total_energy, weight_kurtosis, total_nodes
        predictor_names = ['total_energy', 'weight_kurtosis', 'total_nodes']
        y = [r['output_probability'] for r in data]

        # Build X matrix with intercept
        X = []
        for r in data:
            row = [1.0]  # intercept
            for pred in predictor_names:
                val = r.get(pred, 0.0)
                row.append(float(val) if val is not None else 0.0)
            X.append(row)

        p = len(X[0])  # number of parameters (including intercept)

        # Fit regression
        beta, y_hat, residuals, hat_diag, mse = ols_regression(X, y)
        r2 = r_squared(y, y_hat)

        print(f"  R-squared: {r2:.3f}")
        print(f"  Coefficients: intercept={beta[0]:.4f}, "
              + ", ".join(f"{name}={beta[i+1]:.6f}" for i, name in enumerate(predictor_names)))

        # Cook's distance
        D = cooks_distance(residuals, hat_diag, mse, p)
        threshold = 4.0 / n

        # Identify outliers
        outliers = []
        for i in range(n):
            is_outlier = D[i] > threshold
            is_high_leverage = hat_diag[i] > 2 * p / n
            is_high_residual = abs(residuals[i]) > 2 * math.sqrt(mse)

            if is_outlier or is_high_leverage or is_high_residual:
                outliers.append({
                    'circuit_id': data[i]['circuit_id'],
                    'category': data[i]['category'],
                    'slug': data[i].get('prompt_slug', ''),
                    'cooks_d': D[i],
                    'residual': residuals[i],
                    'hat_diag': hat_diag[i],
                    'y_actual': y[i],
                    'y_predicted': y_hat[i],
                    'is_outlier': is_outlier,
                    'is_high_leverage': is_high_leverage,
                    'is_high_residual': is_high_residual,
                    'total_energy': data[i].get('total_energy', 0),
                    'weight_kurtosis': data[i].get('weight_kurtosis', 0),
                    'total_nodes': data[i].get('total_nodes', 0),
                })

        outliers.sort(key=lambda x: x['cooks_d'], reverse=True)
        n_outliers = sum(1 for o in outliers if o['is_outlier'])
        print(f"  Cook's D threshold: {threshold:.4f}")
        print(f"  Outliers (D > 4/n): {n_outliers}")
        print(f"  High-leverage: {sum(1 for o in outliers if o['is_high_leverage'])}")
        print(f"  High-residual: {sum(1 for o in outliers if o['is_high_residual'])}")

        # Residual-by-category
        cat_residuals = defaultdict(list)
        for i in range(n):
            cat_residuals[data[i]['category']].append(residuals[i])

        cat_stats = {}
        for cat, resids in sorted(cat_residuals.items()):
            mean_r = sum(resids) / len(resids)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in resids) / len(resids)) if len(resids) > 1 else 0
            cat_stats[cat] = {
                'n': len(resids),
                'mean_residual': mean_r,
                'std_residual': std_r,
                'min_residual': min(resids),
                'max_residual': max(resids),
            }
            print(f"    {cat}: mean_resid={mean_r:.4f}, std={std_r:.4f}")

        # Check if outliers cluster by category
        outlier_cats = Counter(o['category'] for o in outliers if o['is_outlier'])

        results[model_name] = {
            'regression': {
                'r_squared': r2,
                'mse': mse,
                'coefficients': {
                    'intercept': beta[0],
                    **{name: beta[i+1] for i, name in enumerate(predictor_names)},
                },
                'n': n,
                'p': p,
            },
            'cooks_threshold': threshold,
            'n_outliers': n_outliers,
            'outliers': outliers,
            'category_residuals': cat_stats,
            'outlier_category_distribution': dict(outlier_cats),
            'all_cooks_d': [
                {'circuit_id': data[i]['circuit_id'], 'cooks_d': D[i],
                 'category': data[i]['category']}
                for i in range(n)
            ],
        }

    # Save JSON
    json_path = output_dir / 'outlier_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Generate report
    print("\n[3/4] Generating report...")
    lines = []
    lines.append("# Stage 2: Outlier Analysis Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}\n")
    lines.append("---\n")

    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        mdata = results.get(model_name, {})
        if not mdata:
            continue

        reg = mdata['regression']
        lines.append(f"## {model_name.upper()}\n")
        lines.append(f"### Regression Model\n")
        lines.append(f"- **R-squared**: {reg['r_squared']:.3f}")
        lines.append(f"- **MSE**: {reg['mse']:.6f}")
        lines.append(f"- **N**: {reg['n']}, **p**: {reg['p']}")
        lines.append(f"- Predictors: total_energy, weight_kurtosis, total_nodes")
        lines.append(f"- Coefficients:")
        for name, val in reg['coefficients'].items():
            lines.append(f"  - {name}: {val:.6f}")
        lines.append("")

        lines.append(f"### Outlier Detection\n")
        lines.append(f"- Cook's D threshold (4/n): **{mdata['cooks_threshold']:.4f}**")
        lines.append(f"- Outliers detected: **{mdata['n_outliers']}**\n")

        if mdata['outliers']:
            lines.append("| Circuit | Category | Cook's D | Residual | Actual | Predicted | Type |")
            lines.append("|---------|----------|----------|----------|--------|-----------|------|")
            for o in mdata['outliers'][:15]:
                types = []
                if o['is_outlier']:
                    types.append('Outlier')
                if o['is_high_leverage']:
                    types.append('HighLev')
                if o['is_high_residual']:
                    types.append('HighRes')
                lines.append(f"| {o['slug'][:30]} | {o['category']} | {o['cooks_d']:.4f} | "
                             f"{o['residual']:.4f} | {o['y_actual']:.3f} | "
                             f"{o['y_predicted']:.3f} | {', '.join(types)} |")
            lines.append("")

        lines.append(f"### Residual by Category\n")
        lines.append("| Category | N | Mean Residual | Std | Min | Max |")
        lines.append("|----------|---|---------------|-----|-----|-----|")
        for cat, stats in sorted(mdata['category_residuals'].items()):
            lines.append(f"| {cat} | {stats['n']} | {stats['mean_residual']:.4f} | "
                         f"{stats['std_residual']:.4f} | {stats['min_residual']:.4f} | "
                         f"{stats['max_residual']:.4f} |")
        lines.append("")

        if mdata['outlier_category_distribution']:
            lines.append(f"### Outlier Category Distribution\n")
            for cat, count in sorted(mdata['outlier_category_distribution'].items()):
                lines.append(f"- {cat}: {count} outliers")
            lines.append("")

    report_path = output_dir / 'outlier_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    for model_name in ['gemma-2-2b', 'qwen3-4b']:
        mdata = results.get(model_name, {})
        if not mdata:
            continue
        print(f"  {model_name.upper()}: R2={mdata['regression']['r_squared']:.3f}, "
              f"outliers={mdata['n_outliers']}")
    print("\nDone!")


if __name__ == '__main__':
    main()
