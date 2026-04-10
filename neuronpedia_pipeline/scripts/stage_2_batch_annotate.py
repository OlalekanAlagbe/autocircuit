#!/usr/bin/env python3
"""Stage 2: Batch Feature Annotation (D6).

Fetches Neuronpedia explanations and activation examples for all
unannotated GEMMA cross-circuit bottleneck features (84 features).

QWEN features cannot be fetched (API not yet supported for QWEN SAEs).

Updates the bottleneck library in-place with new annotations.

Estimated: 84 API calls × 2s = ~3 minutes

Usage:
    python scripts/stage_2_batch_annotate.py
    python scripts/stage_2_batch_annotate.py --dry-run  # just count
"""

import json
import sys
import io
import time
import yaml
import urllib.request
import urllib.error
import datetime
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
CONFIG_PATH = BASE_DIR / 'config' / 'neuronpedia_config.yaml'
BOTTLENECK_LIBRARY = DATA_DIR / 'stage_1_5_bottleneck_library.json'
OUTPUT_DIR = DATA_DIR / 'stage_2_annotations'


def query_neuronpedia_feature(layer, np_id, api_key):
    """Query Neuronpedia API for a single GEMMA feature.

    Returns dict with explanation, examples, and activation_count, or None on failure.
    """
    sae_id = f"{layer}-gemmascope-transcoder-16k"
    url = f"https://neuronpedia.org/api/feature/gemma-2-2b/{sae_id}/{np_id}"

    req = urllib.request.Request(url)
    req.add_header('X-Api-Key', api_key)
    req.add_header('Accept', 'application/json')

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

            # Extract explanation
            explanations = data.get('explanations', [{}])
            explanation_text = explanations[0].get('description', '') if explanations else ''

            # Get activation examples (top tokens from activation samples)
            activations = data.get('activations', [])
            examples = []
            for act in activations[:10]:  # Up to 10 examples
                tokens = act.get('tokens', [])
                values = act.get('values', [])
                if tokens and values:
                    max_idx = values.index(max(values))
                    if max_idx < len(tokens):
                        examples.append(tokens[max_idx])

            return {
                'explanation': explanation_text,
                'examples': examples,
                'activation_count': len(activations),
                'has_data': True,
            }
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f' [RATE LIMITED] Waiting 60s...')
            time.sleep(60)
            return query_neuronpedia_feature(layer, np_id, api_key)
        return {'error': f'HTTP {e.code}', 'has_data': False}
    except urllib.error.URLError as e:
        return {'error': str(e), 'has_data': False}
    except Exception as e:
        return {'error': str(e), 'has_data': False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rate-delay', type=float, default=2.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  D6: BATCH FEATURE ANNOTATION")
    print("=" * 70)

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    api_key = config['api']['api_key']

    # Load bottleneck library
    print("\n[1/3] Loading bottleneck library...")
    with open(BOTTLENECK_LIBRARY, 'r', encoding='utf-8') as f:
        library = json.load(f)

    ccf = library.get('cross_circuit_features', {})
    print(f"  Total features: {len(ccf)}")

    # Find unannotated GEMMA features
    gemma_unannotated = []
    for label, feat in ccf.items():
        if feat.get('explanation'):
            continue  # already annotated
        models = set(a['model'] for a in feat.get('appearances', []))
        if 'gemma-2-2b' in models:
            gemma_unannotated.append((label, feat))

    # Also find QWEN-only features (can't annotate)
    qwen_only = []
    for label, feat in ccf.items():
        if feat.get('explanation'):
            continue
        models = set(a['model'] for a in feat.get('appearances', []))
        if 'qwen3-4b' in models and 'gemma-2-2b' not in models:
            qwen_only.append(label)

    print(f"  GEMMA unannotated (will fetch): {len(gemma_unannotated)}")
    print(f"  QWEN-only unannotated (cannot fetch): {len(qwen_only)}")
    print(f"  Already annotated: {len(ccf) - len(gemma_unannotated) - len(qwen_only)}")

    if args.dry_run:
        print("\n  [DRY RUN] Would make {len(gemma_unannotated)} API calls")
        return

    # Fetch annotations
    print(f"\n[2/3] Fetching annotations ({len(gemma_unannotated)} features)...")
    print(f"  Rate delay: {args.rate_delay}s")
    print(f"  Estimated time: {len(gemma_unannotated) * args.rate_delay / 60:.1f} minutes\n")

    successes = 0
    failures = 0
    newly_annotated = {}

    for idx, (label, feat) in enumerate(gemma_unannotated):
        layer = feat['layer']
        np_id = feat.get('np_id')
        if np_id is None:
            ct_id = feat.get('circuit_id', 0)
            np_id = ct_id % 16384

        if idx > 0:
            time.sleep(args.rate_delay)

        print(f"  [{idx+1:3d}/{len(gemma_unannotated)}] {label:25s} L{layer} NP:{np_id}...", end='', flush=True)

        result = query_neuronpedia_feature(layer, np_id, api_key)

        if result and result.get('has_data'):
            expl = result.get('explanation', '')
            examples = result.get('examples', [])
            n_act = result.get('activation_count', 0)

            # Update library in memory
            ccf[label]['explanation'] = expl
            ccf[label]['examples'] = examples

            newly_annotated[label] = {
                'explanation': expl,
                'examples': examples,
                'activation_count': n_act,
            }

            if expl:
                successes += 1
                print(f' OK: "{expl[:50]}" ({len(examples)} examples)')
            else:
                successes += 1
                print(f' OK but no explanation ({len(examples)} examples, {n_act} activations)')
        else:
            failures += 1
            err = result.get('error', 'unknown') if result else 'null response'
            print(f' FAIL: {err}')

    print(f"\n  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  With explanations: {sum(1 for v in newly_annotated.values() if v.get('explanation'))}")

    # Save updated library
    print(f"\n[3/3] Saving updated bottleneck library...")
    library['cross_circuit_features'] = ccf
    library['metadata']['last_annotation_update'] = datetime.datetime.now().isoformat()
    library['metadata']['annotation_stats'] = {
        'total_annotated': sum(1 for v in ccf.values() if v.get('explanation')),
        'total_with_examples': sum(1 for v in ccf.values() if v.get('examples')),
        'gemma_fetched_this_run': successes,
        'qwen_unavailable': len(qwen_only),
    }

    with open(BOTTLENECK_LIBRARY, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {BOTTLENECK_LIBRARY}")

    # Also save detailed annotation log
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        'timestamp': datetime.datetime.now().isoformat(),
        'successes': successes,
        'failures': failures,
        'newly_annotated': newly_annotated,
        'qwen_unavailable': qwen_only,
    }
    log_path = OUTPUT_DIR / 'annotation_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Log: {log_path}")

    # Summary
    total_annotated = sum(1 for v in ccf.values() if v.get('explanation'))
    total_with_examples = sum(1 for v in ccf.values() if v.get('examples'))

    print("\n" + "=" * 70)
    print("  D6 COMPLETE")
    print("=" * 70)
    print(f"  Total annotated: {total_annotated}/{len(ccf)} ({100*total_annotated/len(ccf):.0f}%)")
    print(f"  With examples: {total_with_examples}")
    print(f"  Still missing: {len(ccf) - total_annotated} (mostly QWEN)")
    print("\nDone!")


if __name__ == '__main__':
    main()
