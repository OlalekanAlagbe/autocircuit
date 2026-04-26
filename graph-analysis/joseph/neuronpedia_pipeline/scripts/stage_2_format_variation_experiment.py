#!/usr/bin/env python3
"""
Format Variation Experiment — Tests whether circuit similarity is driven by
prompt format or knowledge domain.

All original prompts use a fixed template per domain:
  Chemistry: "The chemical symbol for X is"
  Geography: "The capital of X is"
  History:   "X [verb] in [year]"

This experiment generates circuits for the SAME facts with DIFFERENT prompt formats,
then compares:
  - Same-fact, different-format Jaccard (should be HIGH if domain drives convergence)
  - Different-fact, same-format Jaccard (should be LOW if domain, not format, matters)

If format drives convergence: same-format pairs > same-fact pairs
If domain drives convergence: same-fact pairs > same-format pairs

Usage:
    python stage_2_format_variation_experiment.py              # Full run
    python stage_2_format_variation_experiment.py --analyze     # Analyze existing results
    python stage_2_format_variation_experiment.py --generate    # Only generate circuits (no analysis)
"""

import json
import sys
import io
import os
import time
import argparse
import yaml
import requests
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
CONFIG_PATH = BASE_DIR / 'config' / 'neuronpedia_config.yaml'
OUTPUT_DIR = DATA_DIR / 'stage_2_format_variation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Varied-format prompts: same facts, different sentence structures
# ============================================================================

# For each domain, we pick 3-4 facts that were in the original study,
# and write 2 alternative phrasings per fact.

VARIED_PROMPTS = {
    'chemistry': {
        'gold': {
            'original': 'The chemical symbol for gold is',
            'alt1': 'Gold is represented by the chemical symbol',
            'alt2': 'In the periodic table, gold has the symbol',
        },
        'sodium': {
            'original': 'The chemical symbol for sodium is',
            'alt1': 'Sodium is represented on the periodic table as',
            'alt2': 'The element sodium has the chemical symbol',
        },
        'iron': {
            'original': 'The chemical symbol for iron is',
            'alt1': 'Iron is denoted by the symbol',
            'alt2': 'On the periodic table, iron appears as',
        },
    },
    'geography': {
        'france': {
            'original': 'The capital of France is',
            'alt1': 'France has its capital city in',
            'alt2': 'The city that serves as the capital of France is',
        },
        'japan': {
            'original': 'The capital of Japan is',
            'alt1': 'Japan has its seat of government in',
            'alt2': 'The capital city of Japan is called',
        },
        'germany': {
            'original': 'The capital of Germany is',
            'alt1': 'Germany has its capital in the city of',
            'alt2': 'The seat of government in Germany is',
        },
    },
    'history': {
        'wwii': {
            'original': 'World War II ended in',
            'alt1': 'The year that World War II came to an end was',
            'alt2': 'World War II concluded in the year',
        },
        'moon': {
            'original': 'The first moon landing was in',
            'alt1': 'Humans first landed on the moon in the year',
            'alt2': 'The year of the first lunar landing was',
        },
        'titanic': {
            'original': 'The Titanic sank in',
            'alt1': 'The year the Titanic went down was',
            'alt2': 'The Titanic was lost at sea in',
        },
    },
}

MODELS = ['gemma-2-2b']  # QWEN steering not available; focus on GEMMA for causal comparison

# ============================================================================
# Circuit generation via Neuronpedia API
# ============================================================================

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def slugify(text):
    import re
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug)
    return slug[:60]

def generate_graph(prompt, model_id, api_key, rate_delay=15.0, max_retries=5):
    """Generate a circuit graph via Neuronpedia API. Gives up after max_retries rate limit hits."""
    url = 'https://www.neuronpedia.org/api/graph/generate'
    headers = {'Content-Type': 'application/json', 'X-Api-Key': api_key}
    payload = {
        'prompt': prompt,
        'modelId': model_id,
        'sourceSetName': 'gemmascope-transcoder-16k' if 'gemma' in model_id else None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    print(f'    Generating graph for "{prompt[:50]}"...', end='', flush=True)
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            graph_url = data.get('s3url', data.get('url', data.get('graphUrl', '')))
            if graph_url:
                time.sleep(2)
                graph_resp = requests.get(graph_url, timeout=60)
                if graph_resp.status_code == 200:
                    graph = graph_resp.json()
                    n_nodes = len(graph.get('nodes', []))
                    n_links = len(graph.get('links', graph.get('edges', [])))
                    print(f' OK ({n_nodes} nodes, {n_links} links)')
                    time.sleep(rate_delay)
                    return graph
            print(f' ERROR: no graph URL in response')
        elif resp.status_code == 429:
            if max_retries <= 0:
                print(f' RATE LIMITED (max retries exceeded, skipping)')
                return None
            print(f' RATE LIMITED, waiting 90s ({max_retries} retries left)...')
            time.sleep(90)
            return generate_graph(prompt, model_id, api_key, rate_delay, max_retries - 1)
        else:
            print(f' ERROR {resp.status_code}: {resp.text[:100]}')
    except Exception as e:
        print(f' ERROR: {e}')
    return None

def extract_features(graph):
    """Extract feature set from a raw graph for Jaccard comparison."""
    features = set()
    for node in graph.get('nodes', []):
        feature = node.get('feature', node.get('id', ''))
        layer = node.get('layer', node.get('layer_index', -1))
        try:
            layer = int(layer) if layer is not None else -1
        except (ValueError, TypeError):
            layer = -1
        if feature and layer >= 0:
            features.add(f'L{layer}_F{feature}')
    return features

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0

# ============================================================================
# Main experiment
# ============================================================================

def run_generation(config):
    """Generate all varied-format circuits."""
    api_key = config['api']['api_key']
    results = {}

    total_prompts = sum(len(facts) * 3 for facts in VARIED_PROMPTS.values())  # 3 versions each
    print(f'\n[1/3] Generating {total_prompts} circuits across {len(MODELS)} model(s)...')
    print(f'  Estimated time: ~{total_prompts * 10 / 60:.0f} min per model\n')

    for model in MODELS:
        results[model] = {}
        for domain, facts in VARIED_PROMPTS.items():
            results[model][domain] = {}
            for fact_key, versions in facts.items():
                results[model][domain][fact_key] = {}
                for version_name, prompt in versions.items():
                    slug = f'{model}_format_{slugify(prompt)}'
                    cache_dir = OUTPUT_DIR / slug
                    cache_file = cache_dir / 'raw_graph.json'

                    if cache_file.exists():
                        print(f'    [CACHED] {version_name}: "{prompt[:40]}..."')
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            graph = json.load(f)
                    else:
                        graph = generate_graph(prompt, model, api_key)
                        if graph:
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(graph, f)

                    if graph:
                        features = extract_features(graph)
                        results[model][domain][fact_key][version_name] = {
                            'prompt': prompt,
                            'slug': slug,
                            'n_nodes': len(graph.get('nodes', [])),
                            'n_features': len(features),
                            'features': sorted(features),
                        }

    # Save intermediate results
    with open(OUTPUT_DIR / 'generation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_analysis(results):
    """Compare same-fact-different-format vs different-fact-same-format Jaccard."""

    print(f'\n[2/3] Computing Jaccard comparisons...\n')

    analysis = {
        'same_fact_different_format': [],    # Key comparison: format doesn't matter if these are high
        'different_fact_same_format': [],     # Template-driven if these are high
        'different_fact_different_format': [], # Baseline
        'original_within_domain': [],         # Reference from main study
    }

    for model in MODELS:
        model_data = results.get(model, {})

        for domain, facts in model_data.items():
            # Same-fact, different-format comparisons
            for fact_key, versions in facts.items():
                version_names = list(versions.keys())
                for i in range(len(version_names)):
                    for j in range(i + 1, len(version_names)):
                        v1 = versions[version_names[i]]
                        v2 = versions[version_names[j]]
                        f1 = set(v1['features'])
                        f2 = set(v2['features'])
                        j_val = jaccard(f1, f2)
                        analysis['same_fact_different_format'].append({
                            'model': model,
                            'domain': domain,
                            'fact': fact_key,
                            'format1': version_names[i],
                            'format2': version_names[j],
                            'prompt1': v1['prompt'],
                            'prompt2': v2['prompt'],
                            'jaccard': j_val,
                            'n_shared': len(f1 & f2),
                            'n_union': len(f1 | f2),
                        })

            # Different-fact, same-format comparisons (within domain)
            fact_keys = list(facts.keys())
            for fmt in ['original', 'alt1', 'alt2']:
                for i in range(len(fact_keys)):
                    for j in range(i + 1, len(fact_keys)):
                        v1 = facts[fact_keys[i]].get(fmt)
                        v2 = facts[fact_keys[j]].get(fmt)
                        if v1 and v2:
                            f1 = set(v1['features'])
                            f2 = set(v2['features'])
                            j_val = jaccard(f1, f2)
                            analysis['different_fact_same_format'].append({
                                'model': model,
                                'domain': domain,
                                'fact1': fact_keys[i],
                                'fact2': fact_keys[j],
                                'format': fmt,
                                'jaccard': j_val,
                            })

            # Different-fact, different-format (cross everything within domain)
            for i in range(len(fact_keys)):
                for j in range(len(fact_keys)):
                    if i == j:
                        continue
                    for fmt1 in ['original', 'alt1', 'alt2']:
                        for fmt2 in ['original', 'alt1', 'alt2']:
                            if fmt1 == fmt2:
                                continue
                            v1 = facts[fact_keys[i]].get(fmt1)
                            v2 = facts[fact_keys[j]].get(fmt2)
                            if v1 and v2:
                                f1 = set(v1['features'])
                                f2 = set(v2['features'])
                                j_val = jaccard(f1, f2)
                                analysis['different_fact_different_format'].append({
                                    'model': model,
                                    'domain': domain,
                                    'jaccard': j_val,
                                })

    # Compute summary statistics
    summary = {}
    for key, comparisons in analysis.items():
        if not comparisons:
            continue
        jaccards = [c['jaccard'] for c in comparisons]
        summary[key] = {
            'n': len(jaccards),
            'mean': float(np.mean(jaccards)),
            'std': float(np.std(jaccards)),
            'median': float(np.median(jaccards)),
            'min': float(np.min(jaccards)),
            'max': float(np.max(jaccards)),
        }

    # Per-domain breakdown for same-fact-different-format
    domain_summary = {}
    for domain in ['chemistry', 'geography', 'history']:
        domain_pairs = [c for c in analysis['same_fact_different_format'] if c['domain'] == domain]
        if domain_pairs:
            jaccards = [c['jaccard'] for c in domain_pairs]
            domain_summary[domain] = {
                'n': len(jaccards),
                'mean': float(np.mean(jaccards)),
                'std': float(np.std(jaccards)),
            }

    # Print results
    print('=' * 70)
    print('FORMAT VARIATION EXPERIMENT RESULTS')
    print('=' * 70)

    print(f'\n  Comparison Type                        N     Mean J   Std     Median')
    print(f'  {"-"*68}')
    for key, stats in summary.items():
        label = key.replace('_', ' ').title()
        print(f'  {label:<40} {stats["n"]:<5} {stats["mean"]:.4f}   {stats["std"]:.4f}  {stats["median"]:.4f}')

    print(f'\n  Per-domain same-fact-different-format:')
    for domain, stats in domain_summary.items():
        print(f'    {domain:>12}: mean J = {stats["mean"]:.4f} (±{stats["std"]:.4f}, n={stats["n"]})')

    # Key interpretation
    sf_mean = summary.get('same_fact_different_format', {}).get('mean', 0)
    df_sf_mean = summary.get('different_fact_same_format', {}).get('mean', 0)
    df_df_mean = summary.get('different_fact_different_format', {}).get('mean', 0)

    print(f'\n  === KEY FINDING ===')
    if sf_mean > df_sf_mean:
        ratio = sf_mean / df_sf_mean if df_sf_mean > 0 else float('inf')
        print(f'  Same-fact-different-format ({sf_mean:.3f}) > Different-fact-same-format ({df_sf_mean:.3f})')
        print(f'  Ratio: {ratio:.2f}x')
        print(f'  -> DOMAIN drives convergence more than FORMAT')
        print(f'  -> Circuit similarity is a genuine knowledge-domain effect')
    else:
        ratio = df_sf_mean / sf_mean if sf_mean > 0 else float('inf')
        print(f'  Different-fact-same-format ({df_sf_mean:.3f}) > Same-fact-different-format ({sf_mean:.3f})')
        print(f'  Ratio: {ratio:.2f}x')
        print(f'  -> FORMAT drives convergence more than DOMAIN')
        print(f'  -> Circuit similarity is partially a prompt-template effect')

    print(f'  Baseline (different-fact, different-format): {df_df_mean:.3f}')

    # Save results
    output = {
        'metadata': {
            'experiment': 'format_variation',
            'timestamp': datetime.now().isoformat(),
            'models': MODELS,
            'n_domains': 3,
            'n_facts_per_domain': 3,
            'n_formats_per_fact': 3,
            'total_circuits': sum(
                sum(len(v) for v in facts.values())
                for domain_data in results.get(MODELS[0], {}).values()
                for facts in [domain_data]
            ),
        },
        'summary': summary,
        'domain_breakdown': domain_summary,
        'key_finding': {
            'same_fact_different_format_mean': sf_mean,
            'different_fact_same_format_mean': df_sf_mean,
            'different_fact_different_format_mean': df_df_mean,
            'domain_drives_convergence': sf_mean > df_sf_mean,
        },
        'comparisons': analysis,
    }

    out_file = OUTPUT_DIR / 'format_variation_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\n[3/3] Results saved to {out_file}')

    return output


def main():
    parser = argparse.ArgumentParser(description='Format variation experiment')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results only')
    parser.add_argument('--generate', action='store_true', help='Generate circuits only (no analysis)')
    args = parser.parse_args()

    config = load_config()

    if args.analyze:
        gen_file = OUTPUT_DIR / 'generation_results.json'
        if gen_file.exists():
            with open(gen_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            run_analysis(results)
        else:
            print(f'ERROR: {gen_file} not found. Run generation first.')
        return

    # Generate circuits
    results = run_generation(config)

    if not args.generate:
        # Analyze
        run_analysis(results)


if __name__ == '__main__':
    main()
