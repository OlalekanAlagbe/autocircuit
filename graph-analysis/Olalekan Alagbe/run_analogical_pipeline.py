"""
Analogical Reasoning Circuit Discovery Pipeline
Generates 5 attribution graphs, compares them, labels top features,
validates with steering, saves the circuit.
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import time
import sys
import requests
from pathlib import Path

from autocircuit_tools import (
    generate_graph, load_graph, compare_graphs,
    label_node, steer_feature,
    save_circuit, build_prompt_dataset,
    GRAPHS_DIR, CIRCUITS_DIR, HEADERS
)


def label_feature_layered(layer: str, sae_idx: str) -> dict:
    """Label a feature using the layer-prefixed SAE ID (correct format for transcoders)."""
    url = f'https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{sae_idx}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {'explanation': None, 'error': f'HTTP {r.status_code}'}
        d = r.json()
        expls = d.get('explanations', [])
        return {
            'explanation': expls[0].get('description') if expls else None,
            'max_activation': d.get('maxActApprox'),
            'positive_tokens': d.get('pos_str', [])[:10],
            'examples': [
                {
                    'tokens': act.get('tokens'),
                    'max_value': act.get('maxValue'),
                    'peak_token': act['tokens'][act.get('maxValueTokenIndex', -1)]
                                  if act.get('tokens') else None,
                }
                for act in d.get('activations', [])[:3]
            ],
        }
    except Exception as e:
        return {'explanation': None, 'error': str(e)}


def safe_round(v, d=4):
    try:
        return round(float(v), d)
    except (TypeError, ValueError):
        return 0.0


def safe_get_top_nodes(G, n=20, exclude_types=None):
    exclude_types = exclude_types or []
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get('feature_type') in exclude_types:
            continue
        nodes.append({
            'node_id':         node_id,
            'feature':         attrs.get('feature'),
            'layer':           attrs.get('layer'),
            'ctx_idx':         attrs.get('ctx_idx'),
            'feature_type':    attrs.get('feature_type'),
            'influence':       safe_round(attrs.get('influence', 0.0)),
            'activation':      safe_round(attrs.get('activation', 0.0)),
            'is_target_logit': attrs.get('is_target_logit', False),
            'clerp':           attrs.get('clerp', ''),
        })
    nodes.sort(key=lambda x: x['influence'], reverse=True)
    return nodes[:n]


def safe_graph_summary(G, slug=''):
    influences = [safe_round(attrs.get('influence', 0.0))
                  for _, attrs in G.nodes(data=True)]
    layers = sorted(
        set(str(attrs.get('layer')) for _, attrs in G.nodes(data=True)
            if attrs.get('layer') is not None),
        key=lambda x: int(x) if x.lstrip('-').isdigit() else 999
    )
    return {
        'slug':           slug,
        'prompt':         G.graph.get('prompt', ''),
        'num_nodes':      G.number_of_nodes(),
        'num_edges':      G.number_of_edges(),
        'layers_present': layers,
        'max_influence':  round(max(influences), 4) if influences else 0,
        'avg_influence':  round(sum(influences) / len(influences), 4) if influences else 0,
        'top_5_nodes':    safe_get_top_nodes(G, n=5),
    }

PROMPTS = build_prompt_dataset()['analogical']

SLUGS = [
    'analog_berlin',
    'analog_rome',
    'analog_tokyo',
    'analog_teacher',
    'analog_bird',
]


def fetch_graph_from_s3(slug: str, force: bool = False) -> str:
    """
    Generate a graph (if needed) and download actual node/edge data from S3.
    Returns the path to the saved file.
    """
    path = GRAPHS_DIR / f'{slug}.json'
    # Check if it already has real graph data (nodes key)
    if path.exists() and not force:
        with open(path) as f:
            d = json.load(f)
        if 'nodes' in d:
            print(f"[skip] {slug} already has real graph data ({len(d['nodes'])} nodes)")
            return str(path)
        elif 's3url' in d:
            s3url = d['s3url']
            print(f"[fetch_s3] {slug}: downloading from S3...")
        else:
            # Need to regenerate
            force = True

    if force or not path.exists():
        # Generate via Neuronpedia API
        prompt = PROMPTS[SLUGS.index(slug)]
        print(f"[generate] Requesting graph for: '{prompt}'")
        resp = requests.post(
            'https://www.neuronpedia.org/api/graph/generate',
            json={
                'modelId': 'gemma-2-2b',
                'prompt': prompt,
                'slug': slug,
                'maxFeatureNodes': 3000,
                'desiredLogitProb': 0.95,
                'nodeThreshold': 0.8,
                'edgeThreshold': 0.85,
            },
            headers=HEADERS,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f'Generate failed: {resp.status_code} - {resp.text[:300]}')
        meta = resp.json()
        with open(path, 'w') as f:
            json.dump(meta, f)
        s3url = meta['s3url']
        time.sleep(2)

    # Download real data from S3
    print(f"[s3] Downloading: {s3url[:80]}...")
    r = requests.get(s3url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f'S3 download failed: {r.status_code}')
    graph_data = r.json()
    with open(path, 'w') as f:
        json.dump(graph_data, f)
    print(f"  -> Saved {len(graph_data.get('nodes', []))} nodes, "
          f"{len(graph_data.get('links', []))} links to {path}")
    return str(path)


# ── Step 1: Generate/fetch graphs ────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating/fetching 5 analogical attribution graphs")
print("=" * 60)

graph_paths = []
for prompt, slug in zip(PROMPTS, SLUGS):
    try:
        p = fetch_graph_from_s3(slug)
        graph_paths.append((slug, p))
    except Exception as e:
        print(f"  ERROR for '{slug}': {e}")
        sys.exit(1)

print(f"\nAll {len(graph_paths)} graphs ready.\n")

# ── Step 2: Load into NetworkX ───────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Loading graphs into NetworkX")
print("=" * 60)

graphs = []
summaries = []

for slug, path in graph_paths:
    G = load_graph(path)
    graphs.append(G)
    summary = safe_graph_summary(G, slug=slug)
    summaries.append(summary)
    print(f"  {slug}: {summary['num_nodes']} nodes, {summary['num_edges']} edges | "
          f"max_inf={summary['max_influence']}")

with open(GRAPHS_DIR / 'analog_summaries.json', 'w') as f:
    json.dump(summaries, f, indent=2)
print(f"\nSummaries saved.\n")

# ── Step 3: Compare graphs ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Comparing graphs - finding recurring features")
print("=" * 60)

# Start with 3/5 threshold, then tighten if we have too many
common_3 = compare_graphs(graphs, min_appearances=3)
common_4 = compare_graphs(graphs, min_appearances=4)
common_5 = compare_graphs(graphs, min_appearances=5)

print(f"  Features in >=3/5 graphs: {len(common_3)}")
print(f"  Features in >=4/5 graphs: {len(common_4)}")
print(f"  Features in all 5 graphs: {len(common_5)}")

# Use >=4/5 as core, show top 25
common = common_3  # use all for comprehensive analysis

if common:
    print("\nTop 25 recurring features (>=3/5):")
    for i, feat in enumerate(common[:25], 1):
        print(f"  {i:2d}. L{feat['layer']:>2} F{feat['feature']:>12} | "
              f"n={feat['appearances']}/5 | inf={feat['avg_influence']:.4f}")

# Full comparison data
comparison_data = {
    'threshold_3': common_3[:50],
    'threshold_4': common_4[:50],
    'threshold_5': common_5[:50],
}
with open(GRAPHS_DIR / 'analog_comparison.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)
print(f"\nComparison saved to graphs/analog_comparison.json\n")

# ── Step 4: Label top candidates ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Labeling top feature candidates")
print("=" * 60)

# Label top 25 from the >=3/5 pool, sorted by appearances then influence
label_pool = sorted(common_3, key=lambda x: (x['appearances'], x['avg_influence']), reverse=True)[:25]

print(f"Labeling {len(label_pool)} features...")

labeled = []
for i, feat in enumerate(label_pool):
    # node_id format: {layer}_{sae_idx}_{ctx_idx}
    node_id = feat['node_ids'][0]
    parts = node_id.split('_')
    layer_str = parts[0]
    sae_idx = parts[1] if len(parts) > 1 else ''
    print(f"  {i+1}/{len(label_pool)}: {node_id} (L{feat['layer']} SAE#{sae_idx})")
    result = label_feature_layered(layer_str, sae_idx)
    expl = result.get('explanation') or '(no explanation)'
    labeled.append({
        **feat,
        'node_id_sample': node_id,
        'sae_idx': sae_idx,
        'explanation': expl,
        'max_activation': result.get('max_activation'),
        'positive_tokens': result.get('positive_tokens', []),
        'examples': result.get('examples', []),
    })
    print(f"     -> {expl[:80]}")
    time.sleep(0.4)

with open(GRAPHS_DIR / 'analog_labeled_features.json', 'w') as f:
    json.dump(labeled, f, indent=2)
print(f"\nSaved {len(labeled)} labeled features.\n")

# ── Step 5: Validate with steer_feature() ────────────────────────────────────
print("=" * 60)
print("STEP 5: Validating with steer_feature()")
print("=" * 60)

# Pick top 5 by (appearances=5, then avg_influence) for steering
steer_candidates = sorted(
    [f for f in labeled if f['appearances'] >= 4],
    key=lambda x: x['avg_influence'],
    reverse=True
)[:5]

if not steer_candidates:
    steer_candidates = sorted(labeled, key=lambda x: x['avg_influence'], reverse=True)[:5]

print(f"Will steer {len(steer_candidates)} top features:")
for f in steer_candidates:
    print(f"  L{f['layer']} F{f['feature']} | {f['appearances']}/5 | {f.get('explanation','')[:60]}")

test_prompts = [
    PROMPTS[0],  # Paris-Berlin
    PROMPTS[3],  # Doctor-Teacher
    PROMPTS[4],  # Fish-Bird
]

steering_results = []

for feat in steer_candidates:
    layer = feat['layer']
    # Use SAE-local index (sae_idx from node_id), not the global feature ID
    sae_idx = int(feat.get('sae_idx', 0) or 0)
    feature_index = sae_idx  # steer with SAE-local index
    expl = feat.get('explanation') or 'unknown'
    print(f"\n--- Steering L{layer} SAE#{feature_index} | {expl[:50]} ---")

    for prompt in test_prompts:
        for strength, label in [(20.0, 'amplify'), (-20.0, 'suppress')]:
            try:
                res = steer_feature(
                    prompt=prompt,
                    layer=layer,
                    feature_index=feature_index,
                    strength=strength,
                    strength_multiplier=4.0,
                    n_tokens=12,
                )
                api_error = res.get('error', '')
                baseline_out = res.get('original_output', '')
                steered_out = res.get('steered_output', '')
                changed = res.get('changed', False)
                steering_results.append({
                    'layer': layer,
                    'feature_sae_idx': feature_index,
                    'feature_global': feat['feature'],
                    'explanation': expl,
                    'prompt': prompt,
                    'direction': label,
                    'baseline': baseline_out,
                    'steered': steered_out,
                    'changed': changed,
                    'api_status': 'error' if api_error else ('changed' if changed else 'unchanged'),
                    'api_error': api_error or None,
                })
                time.sleep(0.8)
            except Exception as e:
                print(f"  Steer error ({label}): {e}")
                steering_results.append({
                    'layer': layer,
                    'feature_sae_idx': feature_index,
                    'feature_global': feat['feature'],
                    'explanation': expl,
                    'prompt': prompt,
                    'direction': label,
                    'api_status': 'exception',
                    'api_error': str(e),
                })

with open(GRAPHS_DIR / 'analog_steering_validation.json', 'w') as f:
    json.dump(steering_results, f, indent=2)
print(f"\nSteering results saved ({len(steering_results)} tests).\n")

# Print summary of changes
changed = [r for r in steering_results if r.get('changed')]
print(f"Changed outputs: {len(changed)}/{len(steering_results)}")

# ── Step 6: Save the circuit ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Saving analogical reasoning circuit")
print("=" * 60)

circuit_nodes = []

# Core labeled features
for feat in labeled:
    circuit_nodes.append({
        'layer': feat['layer'],
        'feature': feat['feature'],
        'role': 'core_circuit' if feat['appearances'] == 5 else 'recurring',
        'appearances': feat['appearances'],
        'out_of': 5,
        'avg_influence': feat['avg_influence'],
        'avg_activation': feat['avg_activation'],
        'explanation': feat.get('explanation'),
        'positive_tokens': feat.get('positive_tokens', []),
    })

# Add top high-influence nodes from each graph (not in labeled list)
labeled_keys = {(n['layer'], str(n['feature'])) for n in circuit_nodes}
for G in graphs:
    top = safe_get_top_nodes(G, n=10)
    for n in top:
        k = (str(n['layer']), str(n['feature']))
        if k in labeled_keys:
            continue
        if n.get('feature_type') in ('cross layer transcoder', 'transcoder'):
            circuit_nodes.append({
                'layer': str(n['layer']),
                'feature': str(n['feature']),
                'role': 'high_influence_single',
                'appearances': 1,
                'out_of': 5,
                'avg_influence': n['influence'],
                'avg_activation': n['activation'],
                'explanation': n.get('clerp', ''),
                'positive_tokens': [],
            })
            labeled_keys.add(k)

core_count = len([n for n in circuit_nodes if n['role'] == 'core_circuit'])
recurring_count = len([n for n in circuit_nodes if n['role'] == 'recurring'])

circuit = {
    'name': 'Analogical Reasoning Circuit',
    'description': (
        f'Circuit for analogical reasoning in Gemma-2-2B. '
        f'Identified by analyzing 5 attribution graphs spanning capital-city analogies '
        f'(Paris:France::Berlin/Rome/Tokyo:?) and semantic role analogies '
        f'(Doctor:hospital::teacher:? and Fish:water::bird:?). '
        f'{core_count} features appear in all 5 graphs (core circuit). '
        f'{recurring_count} features appear in 3-4/5 graphs (recurring).'
    ),
    'prompt_category': 'analogical',
    'source_graphs': SLUGS,
    'num_source_graphs': len(SLUGS),
    'nodes': circuit_nodes,
    'validation_results': steering_results,
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
circuit_path = str(CIRCUITS_DIR / 'analogical_reasoning_circuit.json')
with open(circuit_path, 'w', encoding='utf-8') as f:
    json.dump(circuit, f, indent=2, ensure_ascii=False)
print(f"[save_circuit] Saved to {circuit_path}")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"  Graphs:              {len(graphs)}")
print(f"  Features (>=3/5):    {len(common_3)}")
print(f"  Features (all 5):    {len(common_5)}")
print(f"  Labeled:             {len(labeled)}")
print(f"  Steering tests:      {len(steering_results)}")
print(f"  Changed outputs:     {len(changed)}")
print(f"  Circuit nodes:       {len(circuit_nodes)}")
print(f"\nFiles:")
print(f"  graphs/analog_summaries.json")
print(f"  graphs/analog_comparison.json")
print(f"  graphs/analog_labeled_features.json")
print(f"  graphs/analog_steering_validation.json")
print(f"  circuits/analogical_reasoning_circuit.json")
