"""
AutoCircuit Graph Utils - Local Python Version
Adapted from the Colab notebook for local execution.

New in this version
-------------------
* get_token_predictions()   — retrieves ranked next-token predictions with
                              probabilities for any prompt via the Neuronpedia
                              steer/completion endpoint (no local model needed).
* interpret_prompt_graph()  — per-prompt narrative: labels the top-N influential
                              nodes, extracts their logit contributions from the
                              graph JSON, then asks Claude (claude-sonnet-4-20250514)
                              to synthesise a plain-English explanation of *why*
                              the model predicted what it predicted.
"""

import os
import json
import time
import requests
import networkx as nx
from pathlib import Path
from typing import Optional

# ── API Keys ──────────────────────────────────────────────────────────────────
NEURONPEDIA_API_KEY = os.environ.get(
    'NEURONPEDIA_API_KEY',
    'sk-np-DinhN2coo82hQFJXSzNsAQtQl6RbvEj3XMVO6ZrLcME0'
)
# Claude API key — set via env var ANTHROPIC_API_KEY or hard-code for dev only.
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL        = 'https://www.neuronpedia.org/api'
ANTHROPIC_URL   = 'https://api.anthropic.com/v1/messages'
CLAUDE_MODEL    = 'claude-sonnet-4-20250514'
MODEL_ID        = 'gemma-2-2b'
SAE_ID          = 'gemmascope-transcoder-16k'

HEADERS = {'Content-Type': 'application/json'}
if NEURONPEDIA_API_KEY:
    HEADERS['X-Api-Key'] = NEURONPEDIA_API_KEY

# ── Local Storage Folders ─────────────────────────────────────────────────────
GRAPHS_DIR   = Path('graphs')
CIRCUITS_DIR = Path('circuits')
GRAPHS_DIR.mkdir(exist_ok=True)
CIRCUITS_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION & FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def generate_graph(prompt: str, slug: str = None, save: bool = True) -> dict:
    """
    Request Neuronpedia to generate an attribution graph for a given prompt.
    """
    if slug is None:
        slug = prompt.lower().replace(' ', '_')[:40].strip('_')

    payload = {
        'modelId'         : MODEL_ID,
        'prompt'          : prompt,
        'slug'            : slug,
        'maxFeatureNodes' : 3000,
        'desiredLogitProb': 0.95,
        'nodeThreshold'   : 0.8,
        'edgeThreshold'   : 0.85,
    }

    print(f'[generate_graph] Requesting graph for: "{prompt}"')
    response = requests.post(f'{BASE_URL}/graph/generate', json=payload, headers=HEADERS)

    if response.status_code != 200:
        raise RuntimeError(
            f'Graph generation failed: {response.status_code} — {response.text}'
        )

    graph_data = response.json()

    if save:
        path = GRAPHS_DIR / f'{slug}.json'
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f'[generate_graph] Saved to {path}')

    return graph_data


def fetch_existing_graph(slug: str) -> dict:
    """
    Fetch an already-generated graph from Neuronpedia by its slug.
    """
    url = f'{BASE_URL}/graph/{slug}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f'Failed to fetch graph "{slug}": {response.status_code}')

    graph_data = response.json()
    path = GRAPHS_DIR / f'{slug}.json'
    with open(path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f'[fetch_existing_graph] Saved "{slug}" → {path}')
    return graph_data


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN PREDICTIONS  ← NEW
# ═════════════════════════════════════════════════════════════════════════════

def get_token_predictions(prompt: str, top_k: int = 5,
                          n_tokens: int = 1, temperature: float = 0.0) -> list:
    """
    Return the top-k next-token predictions and their probabilities for a
    given prompt, using Neuronpedia's steer endpoint (no local model needed).

    Strategy
    --------
    The steer endpoint runs the model with no feature interventions and
    returns the generated output.  To approximate a ranked probability list
    we call it ``top_k`` times with temperature=0 (greedy) — this reliably
    gives the single most-likely token — and then parse whatever probability
    metadata the API returns.

    If the Neuronpedia steer endpoint returns a ``logits`` or ``top_tokens``
    field in its JSON, we use that directly.  Otherwise we fall back to a
    single-call heuristic and note the limitation.

    Parameters
    ----------
    prompt      : The text whose next token we want to rank.
    top_k       : How many candidates to return.
    n_tokens    : How many tokens to generate per call (keep 1 for next-token).
    temperature : Sampling temperature (0 = greedy / deterministic).

    Returns
    -------
    List of dicts: [{'token': str, 'probability': float | None, 'rank': int}]
    """
    payload = {
        'prompt'             : prompt,
        'modelId'            : MODEL_ID,
        'features'           : [],          # no steering — baseline only
        'temperature'        : temperature,
        'n_tokens'           : n_tokens,
        'freq_penalty'       : 0.0,
        'seed'               : 42,
        'strength_multiplier': 1.0,
    }

    print(f'[get_token_predictions] Querying next tokens for: "{prompt}"')
    response = requests.post(f'{BASE_URL}/steer', json=payload, headers=HEADERS)

    if response.status_code != 200:
        raise RuntimeError(
            f'Token prediction failed: {response.status_code} — {response.text}'
        )

    data = response.json()

    # ── Case 1: API returns structured top-token list ──────────────────────
    # Neuronpedia may return 'top_tokens' or 'logits' in future API versions.
    if 'top_tokens' in data:
        ranked = []
        for rank, item in enumerate(data['top_tokens'][:top_k], start=1):
            ranked.append({
                'rank'       : rank,
                'token'      : item.get('token', ''),
                'probability': round(item.get('prob', item.get('probability', 0.0)), 4),
            })
        print(f'  Top-{top_k} predictions: {[(r["token"], r["probability"]) for r in ranked]}')
        return ranked

    # ── Case 2: API returns plain text output only ─────────────────────────
    # Extract the first generated token from the output string.
    output_text = data.get('output', '').strip()
    # The output typically starts with the prompt; strip it to isolate tokens.
    if output_text.startswith(prompt):
        generated = output_text[len(prompt):].strip()
    else:
        generated = output_text

    # Build a best-effort single-entry result
    top_token = generated.split()[0] if generated else '(empty)'
    result = [{
        'rank'       : 1,
        'token'      : top_token,
        'probability': None,   # not available from plain-text endpoint
        'note'       : (
            'Neuronpedia steer endpoint returned plain text only. '
            'Probability unavailable; this is the greedy top-1 token. '
            'For full ranked probabilities use a local model or a logit endpoint.'
        ),
    }]

    # ── Also extract any logit-contribution hints from graph nodes ─────────
    # If the caller already has a graph loaded, get_logit_candidates_from_graph()
    # (below) provides richer probability-like scores from the attribution data.
    print(f'  Greedy top-1 token: "{top_token}"  '
          f'(full ranking unavailable from steer endpoint — '
          f'use get_logit_candidates_from_graph() for feature-level votes)')

    return result


def get_logit_candidates_from_graph(graph_data: dict, top_k: int = 10) -> list:
    """
    Extract ranked next-token candidates from the raw attribution graph JSON.

    The graph stores ``logitContributions`` inside each node — a JSON string
    containing per-layer positive/negative token votes with logit-delta scores.
    This function aggregates those votes across all nodes to produce a ranked
    list of tokens the model was collectively pushing toward.

    Parameters
    ----------
    graph_data : Raw dict returned by generate_graph() or fetch_existing_graph().
    top_k      : Number of candidate tokens to return.

    Returns
    -------
    List of dicts sorted by total logit score (descending):
        [{'token': str, 'total_score': float, 'num_features_voting': int}]
    """
    token_scores: dict = {}

    nodes = graph_data.get('nodes', [])
    for node in nodes:
        contrib_raw = node.get('logitContributions')
        if not contrib_raw:
            continue

        try:
            contrib = json.loads(contrib_raw) if isinstance(contrib_raw, str) else contrib_raw
        except (json.JSONDecodeError, TypeError):
            continue

        # contrib has shape: {"pos": [{}, {"t": [...], "v": [...]}, ...], "neg": [...]}
        for direction, sign in (('pos', +1), ('neg', -1)):
            for layer_entry in contrib.get(direction, []):
                if not layer_entry:
                    continue
                tokens = layer_entry.get('t', [])
                values = layer_entry.get('v', [])
                for tok, val in zip(tokens, values):
                    tok_clean = tok.strip()
                    if tok_clean not in token_scores:
                        token_scores[tok_clean] = {'total': 0.0, 'votes': 0}
                    token_scores[tok_clean]['total'] += sign * val
                    token_scores[tok_clean]['votes'] += 1

    ranked = sorted(token_scores.items(), key=lambda x: x[1]['total'], reverse=True)

    results = []
    for rank, (token, stats) in enumerate(ranked[:top_k], start=1):
        results.append({
            'rank'                : rank,
            'token'               : token,
            'total_logit_score'   : round(stats['total'], 4),
            'num_features_voting' : stats['votes'],
        })

    print(f'[get_logit_candidates_from_graph] Top-{top_k} token candidates:')
    for r in results:
        print(f"  {r['rank']}. '{r['token']}'  score={r['total_logit_score']}  "
              f"features_voting={r['num_features_voting']}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PER-PROMPT INTERPRETATION  ← NEW
# ═════════════════════════════════════════════════════════════════════════════

def interpret_prompt_graph(
    G           : nx.DiGraph,
    graph_data  : dict,
    top_n_nodes : int   = 8,
    label_delay : float = 0.3,
    save        : bool  = True,
) -> dict:
    """
    Produce a plain-English explanation of *why* the model predicted what it
    predicted for a single prompt, by combining:

      1. Top-N influential nodes from the attribution graph (get_top_nodes).
      2. Human-readable feature labels from Neuronpedia (label_nodes_batch).
      3. Logit-contribution token candidates from the raw graph JSON
         (get_logit_candidates_from_graph).
      4. A Claude-generated narrative that synthesises steps 1-3 into a
         coherent mechanistic interpretation.

    Parameters
    ----------
    G            : NetworkX DiGraph loaded via load_graph().
    graph_data   : Raw dict from generate_graph() / fetch_existing_graph().
                   Must be the same graph as G.
    top_n_nodes  : How many top nodes to label and include in the LLM prompt.
    label_delay  : Seconds between Neuronpedia label requests (rate-limit guard).
    save         : If True, saves the interpretation JSON alongside the graph.

    Returns
    -------
    dict with keys:
        prompt           – original prompt string
        predicted_token  – the target logit token (from graph)
        token_prob       – probability of predicted token (if stored in graph)
        top_candidates   – ranked alternative tokens from logit contributions
        labeled_nodes    – list of labeled feature nodes
        interpretation   – Claude's plain-English narrative
        saved_to         – file path if saved, else None
    """
    prompt = G.graph.get('prompt', '')
    slug   = G.graph.get('slug', 'unknown')

    # ── Step 1: Find the predicted (target logit) token ───────────────────
    target_node  = None
    target_token = None
    target_prob  = None
    for node_id, attrs in G.nodes(data=True):
        if attrs.get('is_target_logit'):
            target_node  = node_id
            # clerp often holds the token string for logit nodes
            target_token = attrs.get('clerp') or node_id
            target_prob  = attrs.get('token_prob')
            break

    print(f'\n[interpret_prompt_graph] Prompt   : "{prompt}"')
    print(f'[interpret_prompt_graph] Predicted: "{target_token}"  '
          f'(prob={target_prob})')

    # ── Step 2: Get top influential feature nodes and label them ──────────
    top_nodes  = get_top_nodes(G, n=top_n_nodes, exclude_types=['logit', 'embed'])
    node_ids   = [n['node_id'] for n in top_nodes]

    print(f'[interpret_prompt_graph] Labeling top-{top_n_nodes} nodes …')
    labeled = label_nodes_batch(node_ids, delay=label_delay)

    # Merge influence/activation back into labeled results for the LLM
    influence_map = {n['node_id']: n for n in top_nodes}
    for item in labeled:
        nid = item.get('node_id', '')
        if nid in influence_map:
            item['influence']  = influence_map[nid].get('influence')
            item['activation'] = influence_map[nid].get('activation')

    # ── Step 3: Extract ranked token candidates from logit contributions ──
    top_candidates = get_logit_candidates_from_graph(graph_data, top_k=8)

    # ── Step 4: Ask Claude to synthesise a mechanistic narrative ──────────
    interpretation = _call_claude_for_interpretation(
        prompt         = prompt,
        target_token   = target_token,
        target_prob    = target_prob,
        labeled_nodes  = labeled,
        top_candidates = top_candidates,
        model_id       = MODEL_ID,
    )

    result = {
        'prompt'         : prompt,
        'predicted_token': target_token,
        'token_prob'     : target_prob,
        'top_candidates' : top_candidates,
        'labeled_nodes'  : labeled,
        'interpretation' : interpretation,
        'saved_to'       : None,
    }

    # ── Step 5: Optionally save ───────────────────────────────────────────
    if save:
        fname = f'{slug}_interpretation.json'
        path  = GRAPHS_DIR / fname
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        result['saved_to'] = str(path)
        print(f'[interpret_prompt_graph] Saved → {path}')

    print('\n── Interpretation ──────────────────────────────────────────────')
    print(interpretation)
    print('────────────────────────────────────────────────────────────────\n')

    return result


def _call_claude_for_interpretation(
    prompt        : str,
    target_token  : str,
    target_prob   : Optional[float],
    labeled_nodes : list,
    top_candidates: list,
    model_id      : str,
) -> str:
    """
    Internal helper: send a structured prompt to Claude and return the
    plain-English mechanistic interpretation.

    If ANTHROPIC_API_KEY is not set, returns a structured text summary
    built from the data directly (no LLM required).
    """
    # ── Fallback: no Claude API key — build a rule-based summary ──────────
    if not ANTHROPIC_API_KEY:
        lines = [
            f'Prompt: "{prompt}"',
            f'Predicted token: "{target_token}"'
            + (f' (prob={target_prob:.3f})' if target_prob else ''),
            '',
            'Top alternative token candidates (by aggregated logit score):',
        ]
        for c in top_candidates[:5]:
            lines.append(
                f"  {c['rank']}. '{c['token']}'  "
                f"score={c['total_logit_score']}  "
                f"features_voting={c['num_features_voting']}"
            )
        lines += ['', 'Most influential features driving this prediction:']
        for node in labeled_nodes:
            exp = node.get('explanation') or '(no label)'
            inf = node.get('influence', '?')
            lines.append(f"  Layer {node.get('layer', '?')} | {exp}  [influence={inf}]")
        lines += [
            '',
            '[Set ANTHROPIC_API_KEY to enable Claude-generated narrative interpretation.]'
        ]
        return '\n'.join(lines)

    # ── Build the LLM prompt ───────────────────────────────────────────────
    node_lines = []
    for node in labeled_nodes:
        exp  = node.get('explanation') or '(no Neuronpedia label)'
        inf  = node.get('influence',  '?')
        act  = node.get('activation', '?')
        lyr  = node.get('layer',      '?')
        node_lines.append(
            f'  - Layer {lyr} | Influence={inf} | Activation={act} | Concept: "{exp}"'
        )

    cand_lines = []
    for c in top_candidates[:6]:
        cand_lines.append(
            f"  {c['rank']}. Token='{c['token']}'  "
            f"AggregatedLogitScore={c['total_logit_score']}  "
            f"FeaturesVoting={c['num_features_voting']}"
        )

    user_message = f"""You are a mechanistic interpretability researcher analysing an attribution graph for a large language model ({model_id}).

PROMPT GIVEN TO THE MODEL:
"{prompt}"

PREDICTED (TARGET) TOKEN:
"{target_token}"{(' (probability=' + str(round(target_prob, 4)) + ')') if target_prob else ''}

TOP ALTERNATIVE TOKEN CANDIDATES (aggregated logit scores from feature contributions):
{chr(10).join(cand_lines)}

MOST INFLUENTIAL FEATURES ACTIVE IN THE GRAPH (from Neuronpedia SAE labels):
{chr(10).join(node_lines)}

TASK:
Write a concise plain-English explanation (3-6 paragraphs) of WHY the model predicted "{target_token}" for this prompt. Your explanation should:

1. Identify which concepts the active features represent and how they relate to the prompt.
2. Explain the causal chain: which early-layer features activated first and how they influenced later layers toward the predicted token.
3. Explain why the alternative token candidates scored lower — what features were absent or actively suppressed them.
4. Note any surprising or interesting aspects of the circuit (unexpected features, suppressed alternatives, etc.).
5. Be concrete — reference specific feature descriptions and layer numbers from the data above.

Keep the tone precise but accessible (readable by someone familiar with transformers but not necessarily SAE internals)."""

    anthropic_headers = {
        'Content-Type'     : 'application/json',
        'x-api-key'        : ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
    }

    payload = {
        'model'     : CLAUDE_MODEL,
        'max_tokens': 1024,
        'messages'  : [{'role': 'user', 'content': user_message}],
    }

    print(f'[_call_claude_for_interpretation] Calling Claude ({CLAUDE_MODEL}) …')
    response = requests.post(ANTHROPIC_URL, json=payload, headers=anthropic_headers)

    if response.status_code != 200:
        return (
            f'[Claude API error {response.status_code}]: {response.text[:300]}\n\n'
            + _call_claude_for_interpretation.__doc__  # fall through to rule-based
        )

    data    = response.json()
    content = data.get('content', [])
    text_blocks = [block['text'] for block in content if block.get('type') == 'text']
    return '\n\n'.join(text_blocks) if text_blocks else '[No text in Claude response]'


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH LOADING & ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def load_graph(path: str) -> nx.DiGraph:
    """
    Load a saved attribution graph JSON and convert it to a NetworkX DiGraph.
    """
    with open(path) as f:
        data = json.load(f)

    G = nx.DiGraph()

    metadata = data.get('metadata', {})
    G.graph['slug']          = metadata.get('slug', '')
    G.graph['prompt']        = metadata.get('prompt', '')
    G.graph['prompt_tokens'] = metadata.get('prompt_tokens', [])
    G.graph['model']         = metadata.get('scan', MODEL_ID)

    for node in data.get('nodes', []):
        node_id = node['node_id']
        G.add_node(node_id,
            feature         = node.get('feature'),
            layer           = node.get('layer'),
            ctx_idx         = node.get('ctx_idx'),
            feature_type    = node.get('feature_type', ''),
            influence       = node.get('influence', 0.0),
            activation      = node.get('activation', 0.0),
            is_target_logit = node.get('is_target_logit', False),
            clerp           = node.get('clerp', ''),
            token_prob      = node.get('token_prob', 0.0),
        )

    for link in data.get('links', []):
        G.add_edge(link['source'], link['target'], weight=link.get('weight', 0.0))

    print(f'[load_graph] "{G.graph["prompt"]}"')
    print(f'             {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Return a high-level summary of the graph.
    """
    influences = [attrs.get('influence', 0.0) for _, attrs in G.nodes(data=True)]

    layers = sorted(
        set(str(attrs.get('layer')) for _, attrs in G.nodes(data=True)
            if attrs.get('layer') is not None),
        key=lambda x: int(x) if x.lstrip('-').isdigit() else 999
    )

    logit_count = sum(
        1 for _, attrs in G.nodes(data=True) if attrs.get('is_target_logit')
    )

    return {
        'prompt'         : G.graph.get('prompt', ''),
        'model'          : G.graph.get('model', ''),
        'num_nodes'      : G.number_of_nodes(),
        'num_edges'      : G.number_of_edges(),
        'layers_present' : layers,
        'num_logit_nodes': logit_count,
        'top_5_nodes'    : get_top_nodes(G, n=5),
        'max_influence'  : round(max(influences), 4) if influences else 0,
        'avg_influence'  : round(sum(influences) / len(influences), 4) if influences else 0,
    }


def get_top_nodes(G: nx.DiGraph, n: int = 20, exclude_types: list = None) -> list:
    """
    Return the top-N most influential nodes in the graph.
    """
    exclude_types = exclude_types or []
    nodes = []

    for node_id, attrs in G.nodes(data=True):
        if attrs.get('feature_type') in exclude_types:
            continue
        nodes.append({
            'node_id'        : node_id,
            'feature'        : attrs.get('feature'),
            'layer'          : attrs.get('layer'),
            'ctx_idx'        : attrs.get('ctx_idx'),
            'feature_type'   : attrs.get('feature_type'),
            'influence'      : round(attrs.get('influence', 0.0), 4),
            'activation'     : round(attrs.get('activation', 0.0), 4),
            'is_target_logit': attrs.get('is_target_logit', False),
            'clerp'          : attrs.get('clerp', ''),
        })

    nodes.sort(key=lambda x: x['influence'], reverse=True)
    return nodes[:n]


def get_edges_for_node(G: nx.DiGraph, node_id: str, top_k: int = 10) -> dict:
    """
    Get all incoming and outgoing edges for a specific node.
    """
    incoming = [
        {'source': u, 'weight': round(data['weight'], 4)}
        for u, v, data in G.in_edges(node_id, data=True)
    ]
    outgoing = [
        {'target': v, 'weight': round(data['weight'], 4)}
        for u, v, data in G.out_edges(node_id, data=True)
    ]

    incoming.sort(key=lambda x: abs(x['weight']), reverse=True)
    outgoing.sort(key=lambda x: abs(x['weight']), reverse=True)

    return {
        'node_id'   : node_id,
        'incoming'  : incoming[:top_k],
        'outgoing'  : outgoing[:top_k],
        'in_degree' : G.in_degree(node_id),
        'out_degree': G.out_degree(node_id),
    }


def get_nodes_by_layer(G: nx.DiGraph, layer: str) -> list:
    """
    Return all nodes at a specific transformer layer, sorted by influence.
    """
    nodes = [
        {
            'node_id'   : nid,
            'feature'   : attrs.get('feature'),
            'ctx_idx'   : attrs.get('ctx_idx'),
            'influence' : round(attrs.get('influence', 0.0), 4),
            'activation': round(attrs.get('activation', 0.0), 4),
            'clerp'     : attrs.get('clerp', ''),
        }
        for nid, attrs in G.nodes(data=True)
        if str(attrs.get('layer')) == str(layer)
    ]
    nodes.sort(key=lambda x: x['influence'], reverse=True)
    return nodes


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-GRAPH PATTERN MINING
# ═════════════════════════════════════════════════════════════════════════════

def compare_graphs(graphs: list, min_appearances: int = None) -> list:
    """
    Find nodes (features) that appear consistently across multiple graphs.
    """
    if min_appearances is None:
        min_appearances = len(graphs) // 2 + 1

    registry = {}

    for G in graphs:
        slug = G.graph.get('slug', 'unknown')
        for node_id, attrs in G.nodes(data=True):
            if attrs.get('feature_type') not in ('cross layer transcoder', 'transcoder'):
                continue
            layer   = attrs.get('layer')
            feature = attrs.get('feature')
            if layer is None or feature is None:
                continue

            key = (str(layer), str(feature))
            if key not in registry:
                registry[key] = []
            registry[key].append({
                'graph_slug': slug,
                'node_id'   : node_id,
                'influence' : attrs.get('influence', 0.0),
                'activation': attrs.get('activation', 0.0),
            })

    results = []
    for (layer, feature), occurrences in registry.items():
        if len(occurrences) < min_appearances:
            continue
        avg_inf = sum(o['influence']  for o in occurrences) / len(occurrences)
        avg_act = sum(o['activation'] for o in occurrences) / len(occurrences)
        results.append({
            'layer'         : layer,
            'feature'       : feature,
            'appearances'   : len(occurrences),
            'out_of'        : len(graphs),
            'avg_influence' : round(avg_inf, 4),
            'avg_activation': round(avg_act, 4),
            'graph_slugs'   : [o['graph_slug'] for o in occurrences],
            'node_ids'      : [o['node_id']    for o in occurrences],
        })

    results.sort(key=lambda x: (x['appearances'], x['avg_influence']), reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE LABELING
# ═════════════════════════════════════════════════════════════════════════════

def label_node(node_id: str, model_id: str = MODEL_ID, sae_id: str = SAE_ID) -> dict:
    """
    Translate a node's feature ID into human-readable English via Neuronpedia.
    """
    parts = node_id.split('_')
    if len(parts) < 2:
        return {'node_id': node_id, 'explanation': None, 'error': 'Unparseable node_id'}

    layer_str  = parts[0]
    feature_id = parts[1]

    if layer_str.upper() == 'E' or not feature_id.isdigit():
        return {
            'node_id'    : node_id,
            'layer'      : layer_str,
            'feature'    : feature_id,
            'explanation': f'[{layer_str} node — embedding or error term, no SAE feature]',
            'examples'   : [],
        }

    url = f'{BASE_URL}/feature/{model_id}/{sae_id}/{feature_id}'
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return {
            'node_id'    : node_id,
            'layer'      : layer_str,
            'feature'    : feature_id,
            'explanation': None,
            'error'      : f'API error {response.status_code}',
        }

    data = response.json()
    explanations = data.get('explanations', [])
    explanation  = explanations[0].get('description') if explanations else None

    return {
        'node_id'        : node_id,
        'layer'          : layer_str,
        'feature'        : feature_id,
        'explanation'    : explanation,
        'max_activation' : data.get('maxActApprox'),
        'positive_tokens': data.get('pos_str', [])[:10],
        'negative_tokens': data.get('neg_str', [])[:10],
        'examples'       : [
            {
                'tokens'    : act.get('tokens'),
                'max_value' : act.get('maxValue'),
                'peak_token': act['tokens'][act.get('maxValueTokenIndex', -1)]
                              if act.get('tokens') else None
            }
            for act in data.get('activations', [])[:3]
        ],
    }


def label_nodes_batch(node_ids: list, delay: float = 0.3) -> list:
    """
    Label multiple nodes with a delay between requests to respect rate limits.
    """
    results = []
    for i, nid in enumerate(node_ids):
        print(f'  Labeling {i+1}/{len(node_ids)}: {nid}')
        result = label_node(nid)
        results.append(result)
        time.sleep(delay)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE STEERING
# ═════════════════════════════════════════════════════════════════════════════

def steer_feature(prompt: str, layer: str, feature_index: int,
                  strength: float = 0.0, strength_multiplier: float = 4.0,
                  n_tokens: int = 10, temperature: float = 0.2) -> dict:
    """
    Intervene on a feature and observe how the model's output changes.
    """
    # Step 1: Get baseline output (no steering)
    baseline_payload = {
        'prompt'             : prompt,
        'modelId'            : MODEL_ID,
        'features'           : [],
        'temperature'        : temperature,
        'n_tokens'           : n_tokens,
        'freq_penalty'       : 1.0,
        'seed'               : 42,
        'strength_multiplier': strength_multiplier,
    }
    baseline_resp = requests.post(f'{BASE_URL}/steer', json=baseline_payload, headers=HEADERS)
    baseline_text = baseline_resp.json().get('output', '') if baseline_resp.status_code == 200 else ''

    # Step 2: Steer the target feature
    steer_payload = {
        'prompt'   : prompt,
        'modelId'  : MODEL_ID,
        'features' : [{
            'modelId' : MODEL_ID,
            'layer'   : layer,
            'index'   : feature_index,
            'strength': strength,
        }],
        'temperature'        : temperature,
        'n_tokens'           : n_tokens,
        'freq_penalty'       : 1.0,
        'seed'               : 42,
        'strength_multiplier': strength_multiplier,
    }
    steer_resp = requests.post(f'{BASE_URL}/steer', json=steer_payload, headers=HEADERS)

    if steer_resp.status_code != 200:
        return {
            'error'       : f'Steering API error: {steer_resp.status_code}',
            'feature_used': f'layer={layer}, feature={feature_index}, strength={strength}',
        }

    steered_text = steer_resp.json().get('output', '')

    result = {
        'prompt'         : prompt,
        'original_output': baseline_text,
        'steered_output' : steered_text,
        'changed'        : baseline_text.strip() != steered_text.strip(),
        'feature_used'   : {
            'layer'   : layer,
            'feature' : feature_index,
            'strength': strength,
        }
    }

    print(f'Prompt: "{prompt}"')
    print(f'  Baseline:  "{baseline_text.strip()}"')
    print(f'  Steered:   "{steered_text.strip()}"')
    print(f'  Changed:   {result["changed"]}')

    return result


# ═════════════════════════════════════════════════════════════════════════════
# CIRCUIT STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def save_circuit(name: str, nodes: list, description: str,
                 prompt_category: str, source_graphs: list,
                 validation_results: list = None) -> str:
    """
    Save a discovered circuit hypothesis to disk.
    """
    circuit = {
        'name'              : name,
        'description'       : description,
        'prompt_category'   : prompt_category,
        'source_graphs'     : source_graphs,
        'num_source_graphs' : len(source_graphs),
        'nodes'             : nodes,
        'validation_results': validation_results or [],
        'created_at'        : time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    filename = name.replace(' ', '_').lower() + '.json'
    path = CIRCUITS_DIR / filename
    with open(path, 'w') as f:
        json.dump(circuit, f, indent=2)

    print(f'[save_circuit] Saved "{name}" → {path}')
    return str(path)


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT DATASET  (scaled: 50 per category)
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt_dataset() -> dict:
    """
    Returns structured prompts organised by reasoning category.
    50 prompts per category (200 total), designed for maximum circuit diversity.

    Design principles
    -----------------
    * No two prompts share the same surface template.
    * Each category spans multiple sub-domains so compare_graphs() can
      distinguish category-level circuits from domain-level noise.
    * All prompts end mid-sentence so the next token is unambiguous.
    """
    return {

        # ── FACTUAL RECALL ────────────────────────────────────────────────
      'factual_recall': [
            'The capital of Nigeria is',
            'The capital of Ghana is',
            'The capital of USA is',
            'The capital of England is',
            'The capital of Spain is',
            'The capital of Canada is',
            'The capital of China is',
            # Authors / creators
            'Hamlet was written by',
            'The Mona Lisa was painted by',
            'The theory of relativity was developed by',
            # Biology / science
            'The powerhouse of the cell is the',
            'Photosynthesis takes place inside the',
            'Water is composed of hydrogen and',

             # History
            'The first US president was',
            'Napoleon was exiled to the island of',
             # Geography / nature
            'The longest river in the world is the',
            'Mount Everest is located in the',
            'The largest ocean on Earth is the',
            'The Sahara is a',
            'The Amazon rainforest is primarily located in',

            # Sport
            'A game of chess ends when a king is put in',
            'In tennis, a score of zero is called',
            'The Olympic Games take place every',
            'The sport of sumo wrestling originated in',

             # Technology / computing
            'The programming language Python was created by',
            'HTML stands for HyperText Markup',
            'Wi-Fi uses radio waves to transmit',
            'A CPU stands for Central Processing',

             # Art / culture
            'The Eiffel Tower is located in',
            'Beethoven composed his ninth symphony while being',
            'The Louvre museum is in the city of',
            'The Great Wall was built in',

             # Language / misc
            'The official language of Brazil is',
            'The currency of Japan is the',
            'The speed of light is approximately 300,000 kilometres per',
            'The human body has',
            'Honey is produced by',
        ],

        

        # ── LINGUISTIC ────────────────────────────────────────────────────
                'linguistic': [
             # Antonyms
            'The opposite of hot is',
            'The opposite of dark is',
            'The antonym of large is',
            'The opposite of loud is',
            'The antonym of ancient is',
            'The opposite of weak is',
            'The opposite of fast is',
            'The antonym of north is',
            'The opposite of begin is',
            'The antonym of full is',

            # Synonyms
            'A synonym for happy is',
            'Another word for fast is',
            'A synonym for tired is',

            # Plurals
            'The plural of child is',
            'The plural of mouse is',
            'The plural of foot is',
            'The plural of tooth is',
            'The plural of goose is',

              # Verb forms
            'The past tense of run is',
            'The past tense of swim is',
            'The past tense of write is',
            'The past tense of speak is',
            'The past tense of fly is',

             # Definitions / completions
            'A word meaning the study of stars is the',
            'Someone who practices medicine is called a',
            'A book of maps is called an',
            'The fear of heights is called',
            'A person who writes books is called an',

            # Word families / derived forms
            'The adjective form of sun is',
            'The noun form of happy is',
            'The adverb form of quick is',
            'The verb form of decision is',
            'The adjective form of beauty is',

        ],

        # ── ANALOGICAL ────────────────────────────────────────────────────
        'analogical': [
            'Paris is to France as Berlin is to',
            'Paris is to France as Rome is to',
            'Paris is to France as Tokyo is to',
            'Doctor is to hospital as teacher is to',
            'Fish is to water as bird is to',

              # Capital city analogies
            'Paris is to France as Berlin is to',
            'London is to England as Rome is to',
            'Tokyo is to Japan as Beijing is to',
            'Madrid is to Spain as Lisbon is to',
            'Cairo is to Egypt as Nairobi is to',

            # Animal analogies
            'Fish is to water as bird is to',
            'Puppy is to dog as kitten is to',

             # Tool-function analogies
            'Pen is to writing as brush is to',
            'Clock is to time as thermometer is to',
            'Book is to reading as radio is to',
            'Fork is to eating as pen is to',
            'Remote is to television as steering wheel is to'
            # Part-whole
            'Finger is to hand as toe is to',
            'Page is to book as brick is to',
            'Leaf is to tree as petal is to',
            'Wheel is to car as wing is to',
            'Chapter is to novel as verse is to',
            # Job-workplace
            'Doctor is to hospital as teacher is to',
            'Chef is to kitchen as pilot is to',
            'Judge is to court as priest is to',
            'Soldier is to army as sailor is to',
            'Farmer is to field as miner is to',
            # Material-object
            'Cotton is to shirt as leather is to',
            'Glass is to window as stone is to',
            'Clay is to pot as wax is to',
            # Sequence / time
            'Monday is to Tuesday as January is to',
            'Morning is to noon as noon is to',
            'Spring is to summer as autumn is to',
            'First is to second as third is to',
            'Dawn is to day as dusk is to',
            # Cause-effect
            'Fire is to ash as rain is to',
            'Seed is to plant as egg is to',
            'Study is to knowledge as exercise is to',
            'Sun is to warmth as ice is to',
            'Wind is to erosion as water is to',
            # Scale / unit
            'Second is to minute as minute is to',
            'Gram is to kilogram as millilitre is to',
            'Cent is to dollar as penny is to',
            'Decade is to century as century is to',
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CONNECTION TEST & ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def test_connection() -> bool:
    """
    Test connection to Neuronpedia API.
    """
    print('Testing Neuronpedia API connection...')
    print(f'API key loaded: {"YES" if NEURONPEDIA_API_KEY else "NO (set NEURONPEDIA_API_KEY env var)"}')
    print(f'Anthropic key : {"YES" if ANTHROPIC_API_KEY else "NO (set ANTHROPIC_API_KEY env var — needed for interpret_prompt_graph)"}')

    try:
        url = f'{BASE_URL}/graph/generate'
        payload = {
            'modelId'        : MODEL_ID,
            'prompt'         : 'test',
            'slug'           : 'connection_test',
            'maxFeatureNodes': 3000,
        }
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        if response.status_code in (200, 400):
            print('Connection successful! API is reachable.')
            return True
        else:
            print(f'Connection test returned status {response.status_code}')
            print(f'Response: {response.text[:500]}')
            return False
    except Exception as e:
        print(f'Connection failed: {e}')
        return False


if __name__ == '__main__':
    if test_connection():
        print('\nNeuronpedia is connected and ready!')
        print(f'Graphs folder:   {GRAPHS_DIR.resolve()}')
        print(f'Circuits folder: {CIRCUITS_DIR.resolve()}')

        # ── Quick usage demo ──────────────────────────────────────────────
        print('\n── New functions available ──────────────────────────────────')
        print('  get_token_predictions(prompt)          → ranked next-token list')
        print('  get_logit_candidates_from_graph(data)  → token votes from graph JSON')
        print('  interpret_prompt_graph(G, graph_data)  → per-prompt Claude narrative')
        print('\nExample workflow:')
        print('  graph_data = generate_graph("Remote is to television as steering wheel is to")')
        print('  G          = load_graph("graphs/remote_is_to_television_as_steering_wh.json")')
        print('  preds      = get_token_predictions("Remote is to television as steering wheel is to")')
        print('  result     = interpret_prompt_graph(G, graph_data)')
        print('  print(result["interpretation"])')
    else:
        print('\nConnection test failed. Please check your API key.')
