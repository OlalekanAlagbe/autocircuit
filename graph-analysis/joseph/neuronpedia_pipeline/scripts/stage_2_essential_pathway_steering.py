#!/usr/bin/env python3
"""
Essential-Pathway Steering Validation (Section 5.25)

Tests whether features on the minimal viable pathway (from D3/Section 5.20)
produce stronger steering effects than frequency-based features.

Research question: Does graph topology (being on the essential path) predict
causal influence better than cross-circuit frequency?

Approach:
  1. Extract features on essential pathways across all 30 GEMMA circuits
  2. Select 5 high-frequency essential-pathway features NOT already tested
  3. Run steering at ±20 across the same 3 circuits used in D5 expansion
  4. Compare change rate and KL divergence against frequency-only features

Usage:
  python stage_2_essential_pathway_steering.py                # Full run
  python stage_2_essential_pathway_steering.py --analyze-only  # Analyze existing results
  python stage_2_essential_pathway_steering.py --dry-run       # Show selected features only
"""

import json
import sys
import io
import time
import math
import argparse
import yaml
import requests
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from datetime import datetime

# Windows UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
CONFIG_PATH = BASE_DIR / 'config' / 'neuronpedia_config.yaml'
MINIMAL_PATHWAY_DIR = DATA_DIR / 'stage_2_minimal_pathways'
OUTPUT_DIR = DATA_DIR / 'stage_2_essential_pathway_steering'

# Constants
GEMMA_SAE_DICT_SIZE = 16384
GEMMA_MODEL_ID = 'gemma-2-2b'
SAE_PATH_TEMPLATE = '{layer}-gemmascope-transcoder-16k'
N_TOKENS = 10
TEMPERATURE = 0.0
SEED = 42

# Already-tested features (from D5 + D5 expansion)
ALREADY_TESTED = {
    'L0_F1813559', 'L1_F99962728', 'L2_F25751073', 'L3_F5150441',
    'L4_F110446948', 'L5_F7993995', 'L6_F2586668', 'L7_F4828270',
    'L9_F125286525', 'L24_F88478228',
}

# Same 3 target circuits as D5 expansion for fair comparison
TARGET_CIRCUITS = {
    'chemistry': 'gemma-2-2b_the-chemical-symbol-for-sodium-is',
    'geography': 'gemma-2-2b_the-capital-of-japan-is',
    'history':   'gemma-2-2b_world-war-ii-ended-in',
}

TARGET_PROMPTS = {
    'gemma-2-2b_the-chemical-symbol-for-sodium-is': 'The chemical symbol for sodium is',
    'gemma-2-2b_the-capital-of-japan-is': 'The capital of Japan is',
    'gemma-2-2b_world-war-ii-ended-in': 'World War II ended in',
}

STRENGTHS = [-20, 20]  # Same as D5: ±20


# ============================================================================
# Step 1: Extract essential-pathway features
# ============================================================================

def extract_essential_pathway_features() -> Dict[str, dict]:
    """
    Re-extract essential pathway features by running the minimal pathway algorithm
    on each GEMMA circuit's converted graph (same algorithm as stage_2_minimal_pathways.py).

    Returns dict of feature_label -> {layer, feature_id, circuits: [list], np_id}
    """
    import networkx as nx

    print('\n[1/5] Extracting essential-pathway features from converted graphs...')

    GEMMA_INPUT_MAX = 3    # layers 0-3 are "input" region
    GEMMA_OUTPUT_MIN = 21  # layers 21+ are "output" region

    feature_circuits = defaultdict(set)  # feature_label -> set of circuit_ids
    circuits_processed = 0

    for circuit_dir in sorted(PROMPTS_DIR.iterdir()):
        if not circuit_dir.is_dir() or 'gemma' not in circuit_dir.name:
            continue

        # Find converted graph
        conv_dir = circuit_dir / '2_conversion'
        if not conv_dir.exists():
            continue
        conv_files = list(conv_dir.glob('*converted*.json'))
        if not conv_files:
            continue

        try:
            with open(conv_files[0], 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # Build NetworkX graph
            # Node IDs in converted graphs are like "0_41_1", labels are "L0_F902"
            # We use the raw ID for graph edges but store label for feature extraction
            G = nx.DiGraph()
            id_to_label = {}
            for node in graph_data.get('nodes', []):
                nid = node.get('id', node.get('node_id', ''))
                label = node.get('label', '')
                layer = node.get('layer', -1)
                if nid and isinstance(layer, int) and layer >= 0:
                    G.add_node(nid, layer=layer, label=label)
                    if label and '_F' in label:
                        id_to_label[nid] = label

            for edge in graph_data.get('edges', []):
                src = edge.get('source', edge.get('src', ''))
                tgt = edge.get('target', edge.get('tgt', ''))
                w = edge.get('weight', 1.0)
                if src in G.nodes and tgt in G.nodes:
                    G.add_edge(src, tgt, weight=w)

            if G.number_of_nodes() < 10:
                continue

            # Find input and output features
            input_features = [(n, G.out_degree(n)) for n in G.nodes()
                              if G.nodes[n].get('layer', 999) <= GEMMA_INPUT_MAX and G.out_degree(n) > 0]
            input_features.sort(key=lambda x: x[1], reverse=True)
            input_features = input_features[:10]

            output_features = [(n, G.in_degree(n)) for n in G.nodes()
                               if G.nodes[n].get('layer', 0) >= GEMMA_OUTPUT_MIN and G.in_degree(n) > 0]
            output_features.sort(key=lambda x: x[1], reverse=True)
            output_features = output_features[:10]

            if not input_features or not output_features:
                continue

            # Find shortest paths
            sources = [n for n, _ in input_features]
            targets = [n for n, _ in output_features]

            all_paths = []
            for src in sources:
                for tgt in targets:
                    try:
                        gen = nx.shortest_simple_paths(G, src, tgt)
                        count = 0
                        for path in gen:
                            if count >= 3:
                                break
                            if len(path) <= 20:
                                all_paths.append(path)
                            count += 1
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            if not all_paths:
                continue

            # Extract essential edges (>= 20% of paths)
            edge_counts = defaultdict(int)
            for path in all_paths:
                for i in range(len(path) - 1):
                    edge_counts[(path[i], path[i + 1])] += 1

            threshold = max(1, int(0.2 * len(all_paths)))
            essential_edges = {e for e, c in edge_counts.items() if c >= threshold}

            if not essential_edges:
                sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
                n_keep = max(1, len(sorted_edges) // 2)
                essential_edges = {e for e, c in sorted_edges[:n_keep]}

            # Collect minimal nodes and map to feature labels
            minimal_node_ids = set()
            for s, t in essential_edges:
                minimal_node_ids.add(s)
                minimal_node_ids.add(t)

            # Record feature-circuit associations using labels
            circuit_id = circuit_dir.name
            for node_id in minimal_node_ids:
                label = id_to_label.get(node_id, '')
                if label and '_F' in label:
                    feature_circuits[label].add(circuit_id)

            circuits_processed += 1
            if circuits_processed % 10 == 0:
                print(f'  ... processed {circuits_processed} circuits')

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f'  [WARN] {circuit_dir.name}: {e}')
            continue

    print(f'  Processed {circuits_processed} GEMMA circuits')

    # Build feature info
    features = {}
    for label, circuits in feature_circuits.items():
        if '_F' not in label:
            continue
        parts = label.split('_F')
        layer = int(parts[0].replace('L', ''))
        feature_id = int(parts[1])
        np_id = feature_id % GEMMA_SAE_DICT_SIZE

        features[label] = {
            'label': label,
            'layer': layer,
            'feature_id': feature_id,
            'np_id': np_id,
            'circuit_count': len(circuits),
            'circuits': sorted(circuits),
        }

    print(f'  Total unique features on essential pathways: {len(features)}')

    # Show top features
    top = sorted(features.values(), key=lambda x: x['circuit_count'], reverse=True)[:15]
    print(f'\n  Top 15 essential-pathway features:')
    for f in top:
        tested = ' (ALREADY TESTED)' if f['label'] in ALREADY_TESTED else ''
        print(f'    {f["label"]:<20} {f["circuit_count"]:>2} circuits  NP={f["np_id"]}{tested}')

    return features


# ============================================================================
# Step 2: Select 5 best candidates
# ============================================================================

def select_candidates(features: Dict[str, dict], n: int = 5) -> List[dict]:
    """
    Select n essential-pathway features for steering, prioritizing:
    1. Not already tested
    2. High circuit frequency
    3. Layer diversity
    """
    print(f'\n[2/5] Selecting {n} candidate features...')

    # Filter out already-tested features
    candidates = {k: v for k, v in features.items() if k not in ALREADY_TESTED}
    print(f'  {len(features)} total -> {len(candidates)} after removing {len(ALREADY_TESTED)} already-tested')

    # Sort by circuit count (descending)
    sorted_candidates = sorted(candidates.values(), key=lambda x: x['circuit_count'], reverse=True)

    # Select with layer diversity: try to pick from different layers
    selected = []
    used_layers = set()

    # First pass: pick highest-frequency from each unique layer
    for feat in sorted_candidates:
        if feat['layer'] not in used_layers and len(selected) < n:
            selected.append(feat)
            used_layers.add(feat['layer'])

    # Second pass: fill remaining slots with highest-frequency regardless of layer
    if len(selected) < n:
        for feat in sorted_candidates:
            if feat not in selected and len(selected) < n:
                selected.append(feat)

    # Print selection
    print(f'\n  Selected {len(selected)} features:')
    print(f'  {"Feature":<20} {"Layer":<6} {"NP ID":<8} {"Circuits":<10} {"On Pathway?"}')
    print(f'  {"-"*60}')
    for feat in selected:
        print(f'  {feat["label"]:<20} L{feat["layer"]:<5} {feat["np_id"]:<8} {feat["circuit_count"]:<10} YES')

    # Also show the already-tested features for comparison
    print(f'\n  Previously tested (D5) features for comparison:')
    print(f'  {"Feature":<20} {"On Essential Pathway?"}')
    print(f'  {"-"*40}')
    for label in sorted(ALREADY_TESTED):
        on_pathway = label in features
        pathway_count = features[label]['circuit_count'] if on_pathway else 0
        print(f'  {label:<20} {"YES (" + str(pathway_count) + " circuits)" if on_pathway else "NO"}')

    return selected


# ============================================================================
# Step 3: Steering client
# ============================================================================

class SteeringClient:
    """Minimal steering client for the Neuronpedia API."""

    def __init__(self, api_key: str, rate_delay: float = 36.0):
        self.api_key = api_key
        self.base_url = 'https://www.neuronpedia.org/api'
        self.rate_delay = rate_delay
        self.call_count = 0
        self.last_call_time = 0

    def _wait(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_delay:
            wait = self.rate_delay - elapsed
            print(f'    [rate limit: {wait:.0f}s]', end='', flush=True)
            time.sleep(wait)

    def steer(self, prompt: str, layer: int, np_id: int, strength: float) -> dict:
        """Steer a single feature and return the response."""
        self._wait()

        sae_path = SAE_PATH_TEMPLATE.format(layer=layer)
        payload = {
            'prompt': prompt,
            'modelId': GEMMA_MODEL_ID,
            'features': [{
                'modelId': GEMMA_MODEL_ID,
                'layer': sae_path,
                'index': np_id,
                'strength': strength,
            }],
            'temperature': TEMPERATURE,
            'n_tokens': N_TOKENS,
            'seed': SEED,
            'freq_penalty': 0,
            'strength_multiplier': 1,
        }

        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key,
        }

        try:
            resp = requests.post(f'{self.base_url}/steer', json=payload,
                                 headers=headers, timeout=30)
            self.last_call_time = time.time()
            self.call_count += 1

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(' [RATE LIMITED, waiting 60s]', end='', flush=True)
                time.sleep(60)
                return self.steer(prompt, layer, np_id, strength)
            else:
                return {'error': resp.status_code, 'message': resp.text[:200]}
        except requests.exceptions.Timeout:
            print(' [TIMEOUT, retrying]', end='', flush=True)
            time.sleep(5)
            return self.steer(prompt, layer, np_id, strength)
        except Exception as e:
            return {'error': str(e)}

    def get_baseline(self, prompt: str) -> dict:
        """Get unsteered baseline (steer at strength=0)."""
        self._wait()

        payload = {
            'prompt': prompt,
            'modelId': GEMMA_MODEL_ID,
            'features': [{
                'modelId': GEMMA_MODEL_ID,
                'layer': '0-gemmascope-transcoder-16k',
                'index': 0,
                'strength': 0,
            }],
            'temperature': TEMPERATURE,
            'n_tokens': N_TOKENS,
            'seed': SEED,
            'freq_penalty': 0,
            'strength_multiplier': 1,
        }

        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key,
        }

        try:
            resp = requests.post(f'{self.base_url}/steer', json=payload,
                                 headers=headers, timeout=30)
            self.last_call_time = time.time()
            self.call_count += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                return {'error': resp.status_code}
        except Exception as e:
            return {'error': str(e)}


# ============================================================================
# Step 4: Run experiments
# ============================================================================

def compute_kl_divergence(baseline_logprobs: list, steered_logprobs: list) -> float:
    """Compute KL divergence between baseline and steered token distributions."""
    if not baseline_logprobs or not steered_logprobs:
        return 0.0

    kl_total = 0.0
    n_tokens = min(len(baseline_logprobs), len(steered_logprobs))

    for i in range(n_tokens):
        base_top = baseline_logprobs[i].get('topLogprobs', [])
        steer_top = steered_logprobs[i].get('topLogprobs', [])

        if not base_top or not steer_top:
            continue

        # Build probability dicts
        base_probs = {}
        for t in base_top:
            base_probs[t['token']] = math.exp(t['logprob'])

        steer_probs = {}
        for t in steer_top:
            steer_probs[t['token']] = math.exp(t['logprob'])

        # KL(baseline || steered) for overlapping tokens
        for token, p in base_probs.items():
            q = steer_probs.get(token, 1e-10)
            if p > 0 and q > 0:
                kl_total += p * math.log(p / q)

    return kl_total / max(n_tokens, 1)


def run_experiments(selected: List[dict], client: SteeringClient) -> List[dict]:
    """Run steering experiments: 5 features × 3 circuits × ±20 = 30 experiments."""

    total = len(selected) * len(TARGET_CIRCUITS) * len(STRENGTHS)
    print(f'\n[3/5] Running {total} steering experiments...')
    print(f'  Features: {len(selected)}')
    print(f'  Circuits: {", ".join(TARGET_CIRCUITS.keys())}')
    print(f'  Strengths: {STRENGTHS}')
    print(f'  Estimated time: ~{total * 36 / 60:.0f} min (at 36s rate limit)')

    # Collect baselines first
    print(f'\n  Collecting baselines...')
    baselines = {}
    for domain, circuit_id in TARGET_CIRCUITS.items():
        prompt = TARGET_PROMPTS[circuit_id]
        print(f'    {domain}: "{prompt}"...', end='', flush=True)
        result = client.get_baseline(prompt)
        if 'error' not in result:
            default_text = result.get('DEFAULT', '')
            baselines[circuit_id] = {
                'prompt': prompt,
                'response': result,
                'default_text': default_text,
                'default_logprobs': result.get('defaultLogProbs', result.get('DEFAULT_LOGPROBS', [])),
            }
            print(f' -> "{default_text[:50]}"')
        else:
            print(f' -> ERROR: {result}')

    # Run steering experiments
    results = []
    exp_num = 0

    for feat in selected:
        print(f'\n  Feature: {feat["label"]} (L{feat["layer"]}, NP={feat["np_id"]}, {feat["circuit_count"]} pathway circuits)')

        for domain, circuit_id in TARGET_CIRCUITS.items():
            prompt = TARGET_PROMPTS[circuit_id]
            baseline = baselines.get(circuit_id, {})
            baseline_text = baseline.get('default_text', '')
            baseline_logprobs = baseline.get('default_logprobs', [])

            for strength in STRENGTHS:
                exp_num += 1
                action = 'AMPLIFY' if strength > 0 else 'SUPPRESS'
                print(f'    [{exp_num}/{total}] {action} {strength:+d} on {domain}...', end='', flush=True)

                result = client.steer(prompt, feat['layer'], feat['np_id'], strength)

                if 'error' in result:
                    print(f' ERROR: {result}')
                    results.append({
                        'feature_label': feat['label'],
                        'layer': feat['layer'],
                        'np_id': feat['np_id'],
                        'feature_id': feat['feature_id'],
                        'pathway_circuits': feat['circuit_count'],
                        'circuit_id': circuit_id,
                        'domain': domain,
                        'prompt': prompt,
                        'strength': strength,
                        'error': str(result),
                    })
                    continue

                steered_text = result.get('STEERED', '')
                steered_logprobs = result.get('steeredLogProbs', [])
                text_changed = steered_text != baseline_text

                # Compute KL divergence
                kl_div = compute_kl_divergence(baseline_logprobs, steered_logprobs)

                # Compute logprob shift (mean difference in top token logprob)
                logprob_shift = 0.0
                if baseline_logprobs and steered_logprobs:
                    n = min(len(baseline_logprobs), len(steered_logprobs))
                    shifts = []
                    for i in range(n):
                        b_lp = baseline_logprobs[i].get('logprob', 0)
                        s_lp = steered_logprobs[i].get('logprob', 0)
                        shifts.append(abs(s_lp - b_lp))
                    logprob_shift = sum(shifts) / len(shifts) if shifts else 0.0

                exp_result = {
                    'feature_label': feat['label'],
                    'layer': feat['layer'],
                    'np_id': feat['np_id'],
                    'feature_id': feat['feature_id'],
                    'pathway_circuits': feat['circuit_count'],
                    'circuit_id': circuit_id,
                    'domain': domain,
                    'prompt': prompt,
                    'strength': strength,
                    'baseline_text': baseline_text,
                    'steered_text': steered_text,
                    'text_changed': text_changed,
                    'kl_divergence': kl_div,
                    'logprob_shift': logprob_shift,
                }
                results.append(exp_result)

                change_marker = ' CHANGED' if text_changed else ''
                print(f' KL={kl_div:.4f}, shift={logprob_shift:.4f}{change_marker}')
                print(f'        baseline: "{baseline_text[:60]}"')
                print(f'        steered:  "{steered_text[:60]}"')

    return results


# ============================================================================
# Step 5: Analysis and comparison
# ============================================================================

def analyze_results(results: List[dict], features: Dict[str, dict]):
    """Compare essential-pathway features vs frequency-only features from D5."""

    print(f'\n[4/5] Analyzing results...')

    # Load D5 results for comparison
    d5_file = DATA_DIR / 'stage_3_steering' / 'steering_results.json'
    d5_results = []
    if d5_file.exists():
        with open(d5_file, 'r', encoding='utf-8') as f:
            d5_data = json.load(f)
        d5_results = d5_data.get('results', [])

    # Classify D5 features as pathway vs non-pathway
    d5_on_pathway = []   # Features that ARE on essential pathways
    d5_off_pathway = []  # Features that are NOT on essential pathways

    for r in d5_results:
        label = r.get('feature_label', '')
        # Only include experiments on our 3 target circuits at ±20
        if r.get('circuit_id') not in TARGET_CIRCUITS.values():
            continue
        if abs(r.get('strength', 0)) != 20:
            continue

        if label in features:
            d5_on_pathway.append(r)
        else:
            d5_off_pathway.append(r)

    # Current results (essential-pathway features)
    valid_results = [r for r in results if 'error' not in r]

    print(f'\n  === ESSENTIAL-PATHWAY STEERING RESULTS ===')
    print(f'  New experiments: {len(valid_results)}')

    # Per-feature summary
    feature_stats = defaultdict(lambda: {'changes': 0, 'total': 0, 'kl_sum': 0, 'shift_sum': 0})
    for r in valid_results:
        label = r['feature_label']
        feature_stats[label]['total'] += 1
        if r.get('text_changed'):
            feature_stats[label]['changes'] += 1
        feature_stats[label]['kl_sum'] += r.get('kl_divergence', 0)
        feature_stats[label]['shift_sum'] += r.get('logprob_shift', 0)

    print(f'\n  Per-feature breakdown:')
    print(f'  {"Feature":<20} {"Changes":<12} {"Change%":<10} {"Mean KL":<10} {"Mean Shift":<12} {"Pathways"}')
    print(f'  {"-"*80}')

    total_changes = 0
    total_exps = 0
    total_kl = 0
    total_shift = 0

    for label, stats in sorted(feature_stats.items()):
        n = stats['total']
        ch = stats['changes']
        mean_kl = stats['kl_sum'] / n if n else 0
        mean_shift = stats['shift_sum'] / n if n else 0
        pathway_count = features.get(label, {}).get('circuit_count', '?')
        print(f'  {label:<20} {ch}/{n:<10} {ch/n*100:.0f}%{"":<7} {mean_kl:.4f}{"":<5} {mean_shift:.4f}{"":<7} {pathway_count}')
        total_changes += ch
        total_exps += n
        total_kl += stats['kl_sum']
        total_shift += stats['shift_sum']

    ep_change_rate = total_changes / total_exps if total_exps else 0
    ep_mean_kl = total_kl / total_exps if total_exps else 0
    ep_mean_shift = total_shift / total_exps if total_exps else 0

    print(f'\n  ESSENTIAL-PATHWAY AGGREGATE:')
    print(f'    Text change rate: {total_changes}/{total_exps} ({ep_change_rate*100:.1f}%)')
    print(f'    Mean KL divergence: {ep_mean_kl:.4f}')
    print(f'    Mean logprob shift: {ep_mean_shift:.4f}')

    # D5 comparison
    print(f'\n  === D5 COMPARISON (frequency-based features, same circuits, ±20) ===')

    if d5_on_pathway:
        d5_on_changes = sum(1 for r in d5_on_pathway if r.get('steered_text', '') != r.get('baseline_text', ''))
        # Approximate: check if STEERED != DEFAULT
        d5_on_text_changes = 0
        d5_on_kl = 0
        for r in d5_on_pathway:
            steered = r.get('response', {}).get('STEERED', '')
            default = r.get('response', {}).get('DEFAULT', '')
            if steered and default and steered != default:
                d5_on_text_changes += 1
        print(f'  D5 features ON pathway: {len(d5_on_pathway)} experiments, {d5_on_text_changes} text changes ({d5_on_text_changes/len(d5_on_pathway)*100:.1f}%)')

    if d5_off_pathway:
        d5_off_text_changes = 0
        for r in d5_off_pathway:
            steered = r.get('response', {}).get('STEERED', '')
            default = r.get('response', {}).get('DEFAULT', '')
            if steered and default and steered != default:
                d5_off_text_changes += 1
        print(f'  D5 features OFF pathway: {len(d5_off_pathway)} experiments, {d5_off_text_changes} text changes ({d5_off_text_changes/len(d5_off_pathway)*100:.1f}%)')

    # Domain breakdown
    print(f'\n  === DOMAIN BREAKDOWN ===')
    for domain in ['chemistry', 'geography', 'history']:
        domain_results = [r for r in valid_results if r.get('domain') == domain]
        domain_changes = sum(1 for r in domain_results if r.get('text_changed'))
        domain_kl = sum(r.get('kl_divergence', 0) for r in domain_results)
        n = len(domain_results)
        if n:
            print(f'  {domain:>12}: {domain_changes}/{n} changes ({domain_changes/n*100:.0f}%), mean KL={domain_kl/n:.4f}')

    # Key finding
    print(f'\n  === KEY FINDING ===')
    print(f'  Essential-pathway features: {ep_change_rate*100:.1f}% text change rate, {ep_mean_kl:.4f} mean KL')
    if d5_off_pathway:
        d5_off_rate = d5_off_text_changes / len(d5_off_pathway) if d5_off_pathway else 0
        if ep_change_rate > d5_off_rate:
            print(f'  -> Essential-pathway features show HIGHER steering effectiveness than frequency-only')
        elif ep_change_rate < d5_off_rate:
            print(f'  -> Essential-pathway features show LOWER steering effectiveness than frequency-only')
        else:
            print(f'  -> Essential-pathway and frequency-only features show SIMILAR steering effectiveness')

    return {
        'essential_pathway': {
            'experiments': total_exps,
            'text_changes': total_changes,
            'change_rate': ep_change_rate,
            'mean_kl': ep_mean_kl,
            'mean_logprob_shift': ep_mean_shift,
        },
        'd5_on_pathway': {
            'experiments': len(d5_on_pathway),
        },
        'd5_off_pathway': {
            'experiments': len(d5_off_pathway),
        },
    }


def save_results(results: List[dict], analysis: dict, selected: List[dict], features: Dict[str, dict]):
    """Save results to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'experiment': 'essential_pathway_steering',
            'description': 'Tests whether features on the minimal viable pathway produce stronger steering effects than frequency-based features',
            'timestamp': datetime.now().isoformat(),
            'target_circuits': TARGET_CIRCUITS,
            'strengths': STRENGTHS,
            'total_experiments': len(results),
            'already_tested_features': sorted(ALREADY_TESTED),
        },
        'selected_features': selected,
        'feature_pathway_status': {
            label: label in features
            for label in sorted(ALREADY_TESTED)
        },
        'results': results,
        'analysis': analysis,
    }

    out_file = OUTPUT_DIR / 'essential_pathway_steering_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    print(f'\n[5/5] Results saved to {out_file}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Essential-pathway steering validation')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Analyze existing results without running new experiments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Select features and show plan, but do not steer')
    parser.add_argument('--rate-delay', type=float, default=36.0,
                        help='Seconds between API calls (default: 36)')
    parser.add_argument('--n-features', type=int, default=5,
                        help='Number of features to test (default: 5)')
    args = parser.parse_args()

    print('=' * 70)
    print('ESSENTIAL-PATHWAY STEERING VALIDATION')
    print('Testing topology-based vs frequency-based feature selection')
    print('=' * 70)

    # Step 1: Extract features
    features = extract_essential_pathway_features()

    if not features:
        print('\nERROR: No essential-pathway features found. Cannot proceed.')
        sys.exit(1)

    # Step 2: Select candidates
    selected = select_candidates(features, n=args.n_features)

    if not selected:
        print('\nERROR: No candidate features available.')
        sys.exit(1)

    if args.dry_run:
        print('\n[DRY RUN] Would run:')
        total = len(selected) * len(TARGET_CIRCUITS) * len(STRENGTHS)
        print(f'  {total} experiments ({len(selected)} features × {len(TARGET_CIRCUITS)} circuits × {len(STRENGTHS)} strengths)')
        print(f'  Estimated time: ~{total * args.rate_delay / 60:.0f} min')
        return

    # Analyze-only mode
    if args.analyze_only:
        existing_file = OUTPUT_DIR / 'essential_pathway_steering_results.json'
        if existing_file.exists():
            with open(existing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            analyze_results(data['results'], features)
        else:
            print(f'No existing results at {existing_file}')
        return

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    api_key = config['api']['api_key']

    # Step 3-4: Run experiments
    client = SteeringClient(api_key, rate_delay=args.rate_delay)
    results = run_experiments(selected, client)

    # Step 5: Analyze
    analysis = analyze_results(results, features)

    # Save
    save_results(results, analysis, selected, features)

    print(f'\nTotal API calls: {client.call_count}')
    print(f'Done!')


if __name__ == '__main__':
    main()
