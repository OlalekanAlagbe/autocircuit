#!/usr/bin/env python3
"""Stage 2: Expanded GEMMA Steering (D5).

Tests 5 NEW bottleneck features across 3 circuits (one per category)
with strengths [-20, +20]. Complements the 20 existing experiments.

New features (by cross-circuit frequency):
  L2_F25751073 (freq=11, np_id=11809)
  L5_F7993995  (freq=9, np_id=14987) — "sky/heaven" filter
  L9_F125286525 (freq=9, np_id=14461)
  L1_F99962728 (freq=9, np_id=3944) — universal gateway
  L7_F4828270  (freq=9, np_id=11374)

Target circuits (one per category with analysis files):
  Chemistry: gemma-2-2b_the-chemical-symbol-for-sodium-is
  Geography: gemma-2-2b_the-capital-of-japan-is
  History:   gemma-2-2b_world-war-ii-ended-in

Estimated: 30 API calls × 36s = ~18 minutes

Usage:
    python scripts/stage_2_expanded_steering.py
    python scripts/stage_2_expanded_steering.py --rate-delay 20  # faster
"""

import json
import sys
import io
import time
import yaml
import math
import requests
import datetime
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
CONFIG_PATH = BASE_DIR / 'config' / 'neuronpedia_config.yaml'
OUTPUT_DIR = DATA_DIR / 'stage_3_steering'

GEMMA_MODEL_ID = 'gemma-2-2b'
SAE_PATH_TEMPLATE = '{layer}-gemmascope-transcoder-16k'

# Target features (layer, np_id, label, description)
TARGET_FEATURES = [
    (2, 11809, 'L2_F25751073', 'highest frequency untested (11 circuits)'),
    (5, 14987, 'L5_F7993995', 'sky/heaven filter (9 circuits)'),
    (9, 14461, 'L9_F125286525', 'deep mid-range (9 circuits)'),
    (1, 3944, 'L1_F99962728', 'universal gateway (9 circuits)'),
    (7, 11374, 'L7_F4828270', 'mid-range (9 circuits)'),
]

# Target circuits (one per category)
TARGET_CIRCUITS = {
    'gemma-2-2b_the-chemical-symbol-for-sodium-is': {
        'prompt': 'The chemical symbol for sodium is',
        'category': 'chemistry',
    },
    'gemma-2-2b_the-capital-of-japan-is': {
        'prompt': 'The capital of Japan is',
        'category': 'geography',
    },
    'gemma-2-2b_world-war-ii-ended-in': {
        'prompt': 'World War II ended in',
        'category': 'history',
    },
}

STRENGTHS = [-20, 20]


def steer_api_call(api_key, prompt, layer, np_id, strength, rate_delay=36.0):
    """Call the Neuronpedia steering API."""
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
        'temperature': 0.0,
        'n_tokens': 10,
        'seed': 42,
        'freq_penalty': 0,
        'strength_multiplier': 1,
    }
    headers = {
        'Content-Type': 'application/json',
        'X-Api-Key': api_key,
    }

    try:
        resp = requests.post(
            'https://www.neuronpedia.org/api/steer',
            json=payload, headers=headers, timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print(f'  [RATE LIMITED] Waiting 60s...')
            time.sleep(60)
            return steer_api_call(api_key, prompt, layer, np_id, strength, rate_delay)
        else:
            return {'error': resp.status_code, 'message': resp.text[:200]}
    except requests.exceptions.Timeout:
        print(f'  [TIMEOUT] Retrying...')
        time.sleep(5)
        return steer_api_call(api_key, prompt, layer, np_id, strength, rate_delay)
    except Exception as e:
        return {'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate-delay', type=float, default=36.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  D5: EXPANDED GEMMA STEERING")
    print("=" * 70)

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    api_key = config['api']['api_key']
    print(f"  API key loaded ({len(api_key)} chars)")
    print(f"  Rate delay: {args.rate_delay}s")

    # Load existing results to append to
    existing_results_path = OUTPUT_DIR / 'steering_results.json'
    if existing_results_path.exists():
        with open(existing_results_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing_results = existing.get('results', [])
        existing_keys = {r.get('experiment_key', '') for r in existing_results}
        print(f"  Existing experiments: {len(existing_results)}")
    else:
        existing_results = []
        existing_keys = set()

    # Load bottleneck library for convergence data
    with open(DATA_DIR / 'stage_1_5_bottleneck_library.json', encoding='utf-8') as f:
        lib = json.load(f)
    ccf = lib.get('cross_circuit_features', {})

    # Run experiments
    n_total = len(TARGET_FEATURES) * len(TARGET_CIRCUITS) * len(STRENGTHS)
    print(f"\n  Planned: {n_total} API calls")
    print(f"  Estimated time: {n_total * args.rate_delay / 60:.0f} minutes\n")

    new_results = []
    call_count = 0

    for layer, np_id, label, desc in TARGET_FEATURES:
        feat_data = ccf.get(label, {})
        circuits_appeared_in = feat_data.get('circuits_appeared_in', 0)
        avg_convergence = feat_data.get('avg_convergence', 0)

        print(f"\n  Feature: {label} (L{layer}, NP:{np_id}) — {desc}")

        for circuit_id, circuit_info in TARGET_CIRCUITS.items():
            prompt = circuit_info['prompt']
            category = circuit_info['category']

            for strength in STRENGTHS:
                experiment_key = f'{label}_{circuit_id}_{strength}'

                if experiment_key in existing_keys:
                    print(f"    [SKIP] {category:10s} str={strength:+3d} (already done)")
                    continue

                action = 'SUPPRESS' if strength < 0 else 'AMPLIFY'
                print(f"    [{call_count+1:2d}/{n_total}] {category:10s} {action} {abs(strength)}...",
                      end='', flush=True)

                # Rate limit
                if call_count > 0:
                    time.sleep(args.rate_delay)

                result = steer_api_call(api_key, prompt, layer, np_id, strength)
                call_count += 1

                if 'error' not in result:
                    steered = result.get('STEERED', '')[:50]
                    default = result.get('DEFAULT', '')[:50]
                    changed = steered.strip() != default.strip()
                    tag = 'CHANGED' if changed else 'same'
                    print(f' {tag} -> "{steered[:40]}"')

                    new_results.append({
                        'experiment_key': experiment_key,
                        'feature_label': label,
                        'layer': layer,
                        'np_id': np_id,
                        'circuit_id': circuit_id,
                        'prompt': prompt,
                        'category': category,
                        'strength': strength,
                        'response': result,
                        'circuits_appeared_in': circuits_appeared_in,
                        'avg_convergence': avg_convergence,
                        'is_universal': circuits_appeared_in >= 3,
                        'timestamp': datetime.datetime.now().isoformat(),
                    })
                else:
                    print(f' ERROR: {result.get("message", result.get("error", "unknown"))}')

    # Merge with existing
    all_results = existing_results + new_results
    print(f"\n  New experiments: {len(new_results)}")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  API calls made: {call_count}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'metadata': {
            'total_experiments': len(all_results),
            'total_api_calls': call_count + len(existing_results),
            'timestamp': datetime.datetime.now().isoformat(),
            'strengths_tested': [-50, -20, -10, 10, 20, 50],
            'd5_expansion': {
                'new_features': [label for _, _, label, _ in TARGET_FEATURES],
                'new_experiments': len(new_results),
                'api_calls': call_count,
            },
        },
        'results': all_results,
    }

    with open(existing_results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {existing_results_path}")

    # Quick analysis
    if new_results:
        print("\n  QUICK ANALYSIS:")
        for label in set(r['feature_label'] for r in new_results):
            feat_exps = [r for r in new_results if r['feature_label'] == label]
            changed = sum(1 for r in feat_exps if
                          r['response'].get('STEERED', '').strip() != r['response'].get('DEFAULT', '').strip())
            print(f"    {label:25s}: {changed}/{len(feat_exps)} changed output")

    print("\n" + "=" * 70)
    print("  D5 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
