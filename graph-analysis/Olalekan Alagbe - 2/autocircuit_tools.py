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
# Note: No ANTHROPIC_API_KEY needed. This script is intended to be run by
# Claude Code, which IS Claude — interpretations are produced by Claude Code
# reading the structured output of interpret_prompt_graph() directly.

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL        = 'https://www.neuronpedia.org/api'
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

# ── Interpretation state ───────────────────────────────────────────────────────
# Tracks whether the narration example has been printed yet this session.
# The example prints once (first call) so Claude Code learns the style,
# then is suppressed on all subsequent calls to keep output manageable.
_INTERPRETATION_EXAMPLE_PRINTED = False


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION & FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def generate_graph(prompt: str, slug: str = None, save: bool = True) -> dict:
    """
    Request Neuronpedia to generate an attribution graph for a given prompt,
    then download the full graph JSON from the returned S3 URL.

    The /api/graph/generate endpoint does NOT return the graph data directly.
    It returns metadata: { slug, s3url, url, numNodes, numLinks, ... }.
    The actual graph (nodes + links) must be fetched from the s3url.
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

    meta   = response.json()
    s3url  = meta.get('s3url')
    result_slug = meta.get('slug', slug)

    print(f'[generate_graph] Metadata received — slug="{result_slug}" '
          f'nodes={meta.get("numNodes")} links={meta.get("numLinks")}')
    print(f'[generate_graph] Downloading full graph from S3 ...')

    if not s3url:
        raise RuntimeError(
            f'[generate_graph] No s3url in response — cannot download graph data.\n'
            f'Response was: {meta}'
        )

    s3_response = requests.get(s3url, timeout=120)
    if s3_response.status_code != 200:
        raise RuntimeError(
            f'[generate_graph] S3 download failed: {s3_response.status_code} — {s3url}'
        )

    graph_data = s3_response.json()

    # Inject top-level metadata into the graph dict so load_graph() can find
    # the slug and prompt even if the graph JSON itself omits them.
    if 'metadata' not in graph_data:
        graph_data['metadata'] = {}
    graph_data['metadata'].setdefault('slug',   result_slug)
    graph_data['metadata'].setdefault('prompt', prompt)
    graph_data['metadata'].setdefault('scan',   MODEL_ID)
    # Store the Neuronpedia URL for reference
    graph_data['_neuronpedia_url'] = meta.get('url')

    if save:
        path = GRAPHS_DIR / f'{result_slug}.json'
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f'[generate_graph] Full graph saved to {path} '
              f'({len(graph_data.get("nodes", []))} nodes, '
              f'{len(graph_data.get("links", []))} links)')

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
    top_n_nodes : int   = 40,
    label_delay : float = 0.3,
    save        : bool  = True,
    paper_path  : str   = None,
) -> dict:
    """
    Produce a layer-by-layer mechanistic walkthrough of why the model predicted
    what it predicted for a single prompt.

    paper_path : Optional path to a markdown file. If given, this function
                 appends the interpretation for this prompt directly to that
                 file as it completes — so the research paper is written
                 incrementally across all 47 prompts rather than held in
                 memory until the end. This keeps Claude Code's working
                 context manageable.

    Rather than collecting all nodes and dumping them at once, this function
    works interactively — labeling and printing each layer's features as they
    are discovered, so Claude Code can narrate the reasoning as it builds up
    from early layers to the final prediction.

    top_n_nodes : Total number of influential nodes to label across all layers.
                  Default 40 — enough to cover the full causal chain without
                  hitting rate limits excessively. 8 was too few; the prediction
                  is driven by a layered chain, not a handful of isolated features.

    Returns
    -------
    dict with keys:
        prompt              – original prompt string
        predicted_token     – the target logit token
        token_prob          – probability of predicted token (if in graph)
        top_candidates      – ranked alternative tokens from logit contributions
        nodes_by_layer      – labeled nodes grouped by early / middle / late
        labeled_nodes       – flat list of all labeled nodes (for save/research paper)
        interpretation_data – the full structured walkthrough string printed to stdout
        saved_to            – file path if saved, else None
    """
    prompt = G.graph.get('prompt', '')
    slug   = G.graph.get('slug', 'unknown')

    print(f'\n{"="*66}')
    print(f'INTERPRETING: "{prompt}"')
    print(f'{"="*66}')

    # ── Step 1: Find the predicted (target logit) token ───────────────────
    target_node  = None
    target_token = None
    target_prob  = None
    for node_id, attrs in G.nodes(data=True):
        if attrs.get('is_target_logit'):
            target_node  = node_id
            target_token = attrs.get('clerp') or node_id
            target_prob  = attrs.get('token_prob')
            break

    # ── Fallback: if the graph's target logit node has no token string ────
    if not target_token and prompt:
        print('[interpret_prompt_graph] target_token missing from graph — '
              'falling back to get_token_predictions() ...')
        try:
            preds = get_token_predictions(prompt, top_k=1)
            if preds:
                target_token = preds[0].get('token')
                target_prob  = preds[0].get('probability')
        except Exception as _pred_err:
            print(f'[interpret_prompt_graph] token-prediction fallback failed: {_pred_err}')

    prob_str = f'  (prob={target_prob:.4f})' if target_prob else ''
    print(f'\n>> PREDICTED TOKEN: "{target_token}"{prob_str}')

    # ── Step 2: Get top-N influential nodes, then group them by layer ──────
    # 40 nodes covers the full causal chain: early recognisers, mid-layer
    # relational features, and late-layer token-pushing features.
    top_nodes = get_top_nodes(G, n=top_n_nodes, exclude_types=['logit', 'embed'])
    total     = G.number_of_nodes()
    print(f'\n>> Graph has {total} total nodes. Analysing top {len(top_nodes)} '
          f'by influence to trace the full causal chain.\n')

    # Determine layer boundaries from the actual data
    all_layers = []
    for n in top_nodes:
        lyr = n.get('layer')
        try:
            all_layers.append(int(lyr))
        except (TypeError, ValueError):
            pass
    if all_layers:
        min_layer = min(all_layers)
        max_layer = max(all_layers)
        mid_start = min_layer + (max_layer - min_layer) // 3
        late_start = min_layer + 2 * (max_layer - min_layer) // 3
    else:
        mid_start  = 6
        late_start = 16

    def layer_band(lyr):
        try:
            l = int(lyr)
        except (TypeError, ValueError):
            return 'other'
        if l < mid_start:
            return 'early'
        elif l < late_start:
            return 'middle'
        else:
            return 'late'

    # Separate nodes into bands, preserving influence rank within each
    bands = {'early': [], 'middle': [], 'late': [], 'other': []}
    for n in top_nodes:
        bands[layer_band(n.get('layer'))].append(n)

    # ── Step 3: Label and print each band interactively ───────────────────
    # This is the interactive part — Claude Code sees labels arriving
    # layer-band by layer-band as the reasoning builds up, not all at once.

    all_labeled  = []
    nodes_by_layer = {}

    band_titles = {
        'early' : f'EARLY LAYERS (≤{mid_start-1})  — input recognition & syntactic structure',
        'middle': f'MIDDLE LAYERS ({mid_start}–{late_start-1}) — relational knowledge & domain mapping',
        'late'  : f'LATE LAYERS (≥{late_start})   — token selection & final prediction push',
        'other' : 'OTHER NODES',
    }
    band_questions = {
        'early' : 'What did the model notice in the raw input?',
        'middle': 'What relational structure / world knowledge kicked in?',
        'late'  : 'What concepts are directly pushing toward the predicted token?',
        'other' : '',
    }

    for band in ('early', 'middle', 'late', 'other'):
        nodes_in_band = bands[band]
        if not nodes_in_band:
            continue

        print(f'\n{"─"*66}')
        print(f'  {band_titles[band]}')
        q = band_questions[band]
        if q:
            print(f'  >> {q}')
        print(f'{"─"*66}')
        print(f'  Labeling {len(nodes_in_band)} nodes ...\n')

        node_ids_band = [n['node_id'] for n in nodes_in_band]
        labeled_band  = label_nodes_batch(node_ids_band, delay=label_delay)

        # Merge influence/activation into labels and print immediately
        inf_map = {n['node_id']: n for n in nodes_in_band}
        for item in labeled_band:
            nid = item.get('node_id', '')
            if nid in inf_map:
                item['influence']  = inf_map[nid].get('influence')
                item['activation'] = inf_map[nid].get('activation')

            exp = item.get('explanation') or '(no label)'
            inf = item.get('influence',  '?')
            act = item.get('activation', '?')
            lyr = item.get('layer',      '?')
            # Print each feature as it arrives — this is the interactive build-up
            print(f'  Layer {lyr:>3} | inf={inf:>6} | act={act:>6} |  "{exp}"')

        all_labeled.extend(labeled_band)
        nodes_by_layer[band] = labeled_band

    # ── Step 4: Token candidates ───────────────────────────────────────────
    print(f'\n{"─"*66}')
    print('  TOKEN COMPETITION — what was the circuit collectively voting for?')
    print(f'{"─"*66}')
    top_candidates = get_logit_candidates_from_graph(graph_data, top_k=10)
    print(f'\n  Predicted winner: "{target_token}"{prob_str}')
    print('  Runner-up candidates:')
    for c in top_candidates[:6]:
        marker = '  **' if c['token'].strip() == str(target_token).strip() else '    '
        print(f"{marker}{c['rank']}. '{c['token']}'  "
              f"score={c['total_logit_score']}  "
              f"({c['num_features_voting']} features voting)")

    # ── Step 5: Build the full interpretation data block for Claude Code ──
    interpretation = _call_claude_for_interpretation(
        prompt         = prompt,
        target_token   = target_token,
        target_prob    = target_prob,
        labeled_nodes  = all_labeled,
        top_candidates = top_candidates,
        nodes_by_layer = nodes_by_layer,
        band_titles    = band_titles,
        model_id       = MODEL_ID,
    )

    result = {
        'prompt'             : prompt,
        'predicted_token'    : target_token,
        'token_prob'         : target_prob,
        'top_candidates'     : top_candidates,
        'nodes_by_layer'     : nodes_by_layer,
        'labeled_nodes'      : all_labeled,
        'interpretation_data': interpretation,
        'saved_to'           : None,
    }

    # ── Step 6: Append this prompt's results to the research paper ────────
    # Writing incrementally means Claude Code never has to hold 47
    # interpretations in memory simultaneously — each one is flushed to
    # disk as soon as it is complete.
    if paper_path:
        _append_prompt_to_paper(
            paper_path     = paper_path,
            prompt         = prompt,
            target_token   = target_token,
            target_prob    = target_prob,
            top_candidates = top_candidates,
            nodes_by_layer = nodes_by_layer,
            all_labeled    = all_labeled,
        )

    # ── Step 7: Optionally save the full result dict ───────────────────────
    if save:
        fname = f'{slug}_interpretation.json'
        path  = GRAPHS_DIR / fname
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        result['saved_to'] = str(path)
        print(f'\n[interpret_prompt_graph] Saved → {path}')

    return result


def _append_prompt_to_paper(paper_path: str, prompt: str, target_token: str,
                             target_prob, top_candidates: list,
                             nodes_by_layer: dict, all_labeled: list):
    """
    Append a single prompt's interpretation results to the research paper
    markdown file. Called by interpret_prompt_graph() after each prompt
    completes so the paper is built incrementally — one section per prompt.

    Claude Code reads each appended section and writes the mechanistic
    narrative for that prompt into the paper immediately, before moving
    to the next prompt. This keeps its working context small.
    """
    prob_str = f' (prob={target_prob:.4f})' if target_prob else ''
    slug     = prompt.lower().replace(' ', '_')[:40].strip('_')

    lines = [
        '',
        f'### Prompt: "{prompt}"',
        '',
        f'**Predicted token:** `{target_token}`{prob_str}',
        '',
        '**Token competition:**',
        '',
        '| Rank | Token | Logit score | Features voting |',
        '|------|-------|-------------|-----------------|',
    ]
    for c in top_candidates[:6]:
        winner = ' ✓' if c['token'].strip() == str(target_token).strip() else ''
        lines.append(
            f"| {c['rank']} | `{c['token'].strip()}`{winner} "
            f"| {c['total_logit_score']} | {c['num_features_voting']} |"
        )

    lines += ['', '**Circuit walkthrough:**', '']
    band_order  = ['early', 'middle', 'late', 'other']
    band_labels = {
        'early' : 'Early layers — input recognition',
        'middle': 'Middle layers — relational mapping',
        'late'  : 'Late layers — token selection',
        'other' : 'Other nodes',
    }
    for band in band_order:
        nodes = nodes_by_layer.get(band, [])
        if not nodes:
            continue
        lines += [f'*{band_labels[band]}*', '']
        lines += ['| Layer | Influence | Activation | Feature label |',
                  '|-------|-----------|------------|---------------|']
        for node in nodes:
            exp = (node.get('explanation') or '(no label)').replace('|', '/')
            lines.append(
                f"| {node.get('layer','?')} "
                f"| {node.get('influence','?')} "
                f"| {node.get('activation','?')} "
                f"| {exp} |"
            )
        lines.append('')

    lines += [
        '**[Claude Code — write mechanistic narrative here]**',
        '',
        '> *Narrate the full causal chain for this prompt: early recognition →*',
        '> *middle mapping → late push → predicted token. Use the table above.*',
        '> *Follow the style established in the first prompt narration.*',
        '',
        '---',
    ]

    with open(paper_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'[paper] Appended "{prompt[:50]}" → {paper_path}')


def _call_claude_for_interpretation(
    prompt        : str,
    target_token  : str,
    target_prob   : Optional[float],
    labeled_nodes : list,
    top_candidates: list,
    model_id      : str,
    nodes_by_layer: dict = None,
    band_titles   : dict = None,
) -> str:
    """
    Build and print a structured layer-by-layer interpretation data block
    that Claude Code reads to write its mechanistic narrative.

    First call: prints the full HOW-TO-NARRATE example so Claude Code
    learns the required voice and depth once.
    Subsequent calls: prints only the compact circuit data (no repeated
    example) to keep output manageable across 47 prompts.
    """
    global _INTERPRETATION_EXAMPLE_PRINTED
    nodes_by_layer = nodes_by_layer or {}
    band_titles    = band_titles    or {}
    prob_str = f'  (prob={target_prob:.4f})' if target_prob else ''

    lines = [
        '',
        '╔══════════════════════════════════════════════════════════════════╗',
        '║   CIRCUIT WALKTHROUGH  —  instructions for Claude Code           ║',
        '╚══════════════════════════════════════════════════════════════════╝',
        f'  Model  : {model_id}',
        f'  Prompt : "{prompt}"',
        f'  Answer : "{target_token}"{prob_str}',
        '',
    ]

    # ── Print the narration example only on the very first call ───────────
    if not _INTERPRETATION_EXAMPLE_PRINTED:
        lines += [
            '  ── HOW TO NARRATE (printed once — style guide for all prompts) ──',
            '  For each layer band below, write 2-4 sentences in the style of a',
            '  mechanistic interpretability researcher watching the circuit fire',
            '  in slow motion. Do NOT just list the feature labels. Instead,',
            '  explain what those activations MEAN for the computation at that',
            '  stage — what the model is building toward, what is being resolved,',
            '  and how it connects to the next stage.',
            '',
            '  EXAMPLE (for "Remote is to television as steering wheel is to"):',
            '',
            '  [EARLY] "At the earliest layers, the model is parsing the structural',
            '  frame of the analogy itself. The high activation on \'relational',
            '  mapping between object pairs\' at Layer 2 (inf=0.81) tells us the',
            '  model has already locked onto the A:B::C:? pattern before resolving',
            '  the domains. \'Tool or device token\' at Layer 4 is beginning to',
            '  register that both \'remote\' and \'steering wheel\' are physical',
            '  control interfaces."',
            '',
            '  [MIDDLE] "By the middle layers, domain knowledge takes over.',
            '  \'Vehicle components and controls\' at Layer 9 (inf=0.68) is the',
            '  pivotal moment — \'steering wheel\' is mapped into the vehicle',
            '  domain, mirroring how \'remote\' mapped to the television domain.',
            '  \'Navigation and direction\' at Layer 11 (inf=0.54) reinforces the',
            '  function-role relationship: just as a remote controls a TV,',
            '  a steering wheel controls direction of a vehicle."',
            '',
            '  [LATE] "The late layers crystallise the answer. \'Car or automobile',
            '  as a category\' fires at Layer 17 (inf=0.91) directly upstream of',
            '  the logit node — it is voting for the answer token, not describing',
            '  the steering wheel. The analogy is complete: remote→TV,',
            '  so steering wheel→car."',
            '',
            '  [TOKEN COMPETITION] "\'Car\' won over \'vehicle\' (score 4.2 vs 1.8)',
            '  because \'television\' is specific, so the model resolves toward the',
            '  equally specific \'car\' rather than the hypernym \'vehicle\'.',
            '  \'Truck\' was suppressed — \'steering wheel\' does not co-activate',
            '  with truck-specific features."',
            '',
            '  ── This style applies to ALL prompts. Example will not reprint. ──',
            '',
        ]
        _INTERPRETATION_EXAMPLE_PRINTED = True
    else:
        lines += [
            '  [Style guide already printed on first prompt — same voice applies.]',
            '',
        ]

    band_order = ['early', 'middle', 'late', 'other']
    band_questions = {
        'early' : 'What did the model notice in the raw input tokens?',
        'middle': 'What relational structure or world knowledge emerged?',
        'late'  : 'What is directly pushing toward the final predicted token?',
        'other' : 'Additional nodes:',
    }
    band_narrate = {
        'early' : (
            'Narrate what the model is recognising at this stage. '
            'Focus on: does it know this is an analogy? What semantic classes '
            'is it registering? Use the influence scores to judge what matters most.'
        ),
        'middle': (
            'Narrate the domain mapping. What world knowledge has activated? '
            'Which features represent the key relational insight that connects '
            'the A:B pair to the C:? resolution? Call out any surprising co-activations.'
        ),
        'late'  : (
            'Narrate the final push. Which features are directly upstream of the '
            'predicted token? Explain why their activation causally produces '
            f'"{target_token}" rather than the runner-up alternatives.'
        ),
        'other' : 'Narrate any other relevant nodes.',
    }

    for band in band_order:
        nodes = nodes_by_layer.get(band, [])
        if not nodes:
            continue
        title = band_titles.get(band, band.upper())
        lines += [
            f'  ┌─ {title}',
            f'  │  Question: {band_questions[band]}',
            '  │',
        ]
        for node in nodes:
            exp = node.get('explanation') or '(no label)'
            inf = node.get('influence',  '?')
            act = node.get('activation', '?')
            lyr = node.get('layer',      '?')
            lines.append(
                f'  │  Layer {lyr:>3} | inf={inf:>6} | act={act:>6} | "{exp}"'
            )
        lines += [
            '  │',
            f'  └─ NARRATE THIS BAND: {band_narrate[band]}',
            '',
        ]

    lines += [
        '  ┌─ TOKEN COMPETITION',
        '  │  Question: Why did the model choose this token over the alternatives?',
        '  │',
        f'  │  Predicted: "{target_token}"{prob_str}',
    ]
    for c in top_candidates[:6]:
        winner = ' ← WINNER' if c['token'].strip() == str(target_token).strip() else ''
        lines.append(
            f"  │  {c['rank']}. '{c['token']}'  "
            f"score={c['total_logit_score']}  "
            f"({c['num_features_voting']} features){winner}"
        )
    lines += [
        '  │',
        '  └─ NARRATE TOKEN COMPETITION: Explain the score gap. Which middle/late',
        '     features were present for the winner but absent for the runner-ups?',
        '     Is the margin large (confident) or small (ambiguous)?',
        '',
        '  ┌─ SYNTHESIS',
        '  │  Write one final paragraph tying the full causal chain together:',
        '  │    early recognition → middle mapping → late push → predicted token.',
        '  │  Mention anything surprising: features active for unexpected reasons,',
        '  │  alternative tokens that came close, or circuit behaviour that differs',
        '  │  from what you would expect for this type of analogy.',
        '  └─',
        '',
        '╔══════════════════════════════════════════════════════════════════╗',
        '║   END OF CIRCUIT DATA                                            ║',
        '╚══════════════════════════════════════════════════════════════════╝',
    ]

    block = '\n'.join(lines)
    print(block)
    return block


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
# RECURRING FEATURE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def extract_node_ids_from_recurring(recurring_features: list, top_n: int = None) -> list:
    """
    Given the list returned by compare_graphs(), extract one representative
    node_id per recurring feature so it can be passed directly to
    label_nodes_batch().

    compare_graphs() stores ALL node_ids from every graph in which the feature
    appeared.  We want exactly one canonical ID per (layer, feature) pair.
    Strategy: pick the node_id whose layer/feature matches the canonical key,
    i.e. the first node_id in the list (they all share the same layer+feature,
    only ctx_idx and graph_slug differ).

    Parameters
    ----------
    recurring_features : list returned by compare_graphs(), sorted by
                         (appearances, avg_influence) descending.
    top_n              : if given, only process the first top_n entries.

    Returns
    -------
    List of node_id strings, one per recurring feature, in the same order.
    """
    if top_n is not None:
        recurring_features = recurring_features[:top_n]

    node_ids = []
    for feat in recurring_features:
        ids = feat.get('node_ids', [])
        if ids:
            # All ids share the same layer+feature; take the first as canonical.
            node_ids.append(ids[0])
        else:
            # Reconstruct a minimal node_id from layer + feature
            node_ids.append(f"{feat['layer']}_{feat['feature']}_0")

    return node_ids


def get_subgraph_node_ids_for_feature(G: nx.DiGraph, layer: str,
                                      feature: str, depth: int = 1) -> list:
    """
    Given a graph G and a (layer, feature) pair, find the matching node(s)
    and return the node_ids of their immediate graph neighbours (predecessors
    + successors up to ``depth`` hops), excluding logit and embed nodes.

    This is used in Step 7 to label 'the nodes beneath' a steered feature —
    i.e. the circuit neighbourhood that feeds into it and that it feeds into.

    Parameters
    ----------
    G       : NetworkX DiGraph loaded via load_graph().
    layer   : layer string, e.g. '14'.
    feature : feature index string, e.g. '8021'.
    depth   : number of hops to traverse (default 1).

    Returns
    -------
    List of unique node_id strings (neighbours of the matched node).
    """
    # Find the seed node(s) that match this layer+feature
    seed_nodes = [
        nid for nid, attrs in G.nodes(data=True)
        if str(attrs.get('layer')) == str(layer)
        and str(attrs.get('feature')) == str(feature)
        and attrs.get('feature_type') not in ('logit', 'embed')
    ]

    if not seed_nodes:
        print(f'[get_subgraph_node_ids_for_feature] No node found for '
              f'layer={layer}, feature={feature}')
        return []

    neighbours = set()
    frontier   = set(seed_nodes)
    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            for pred in G.predecessors(nid):
                if pred not in seed_nodes:
                    next_frontier.add(pred)
            for succ in G.successors(nid):
                if succ not in seed_nodes:
                    next_frontier.add(succ)
        neighbours.update(next_frontier)
        frontier = next_frontier

    # Filter out logit / embed nodes — they don't have SAE feature labels
    result = [
        nid for nid in neighbours
        if G.nodes[nid].get('feature_type') not in ('logit', 'embed')
        and G.nodes[nid].get('feature') is not None
    ]

    print(f'[get_subgraph_node_ids_for_feature] layer={layer} feature={feature} '
          f'-> {len(seed_nodes)} seed(s), {len(result)} neighbour node(s)')
    return result


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE LABELING
# ═════════════════════════════════════════════════════════════════════════════

def label_node(node_id: str, model_id: str = MODEL_ID) -> dict:
    """
    Translate a node's feature ID into human-readable English via Neuronpedia.

    The SAE ID is derived automatically from the node's own layer using the
    layer-prefixed format required by the API:
        {layer}-gemmascope-transcoder-16k
    e.g. node_id '7_3099_1' → SAE ID '7-gemmascope-transcoder-16k'

    The old ``sae_id`` parameter has been removed.  Passing a bare
    'gemmascope-transcoder-16k' (without a layer prefix) returns HTTP 500
    from Neuronpedia for every request and was the source of all the
    '(no label)' results seen in earlier versions of this library.
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

    # Build the layer-prefixed SAE ID that Neuronpedia actually expects.
    # Using the bare SAE_ID constant (no layer prefix) always returns HTTP 500.
    sae_id = f'{layer_str}-{SAE_ID}'
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
        # NOTE: duplicates removed; a missing comma after "steering wheel is to"
        # was also fixed (it caused string concatenation with the next literal).
        'analogical': [
            # Capital city analogies
            'Paris is to France as Berlin is to',
            'Paris is to France as Rome is to',
            'Paris is to France as Tokyo is to',
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
            'Remote is to television as steering wheel is to',

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
    print('Note: No Anthropic API key needed — interpretations are written by Claude Code directly.')

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

        print('\n-- autocircuit_tools_2.py fixes over v1 --')
        print('  * Analogical prompt list deduplicated (no duplicate graphs)')
        print('  * Missing comma bug fixed ("steering wheel is to" entry)')
        print('  * interpret_prompt_graph() falls back to get_token_predictions()')
        print('    when the graph has no target-logit token string')
        print('  * extract_node_ids_from_recurring() — pick one representative')
        print('    node_id per compare_graphs() entry for label_nodes_batch()')
        print('  * get_subgraph_node_ids_for_feature() — label the circuit')
        print('    neighbourhood of a steered feature (Step 7 "nodes beneath")')

        print('\n-- New functions available --')
        print('  extract_node_ids_from_recurring(recurring, top_n)')
        print('  get_subgraph_node_ids_for_feature(G, layer, feature, depth=1)')
        print('  get_token_predictions(prompt)          -> ranked next-token list')
        print('  get_logit_candidates_from_graph(data)  -> token votes from graph JSON')
        print('  interpret_prompt_graph(G, graph_data)  -> per-prompt Claude narrative')

        print('\nExample workflow:')
        print('  recurring = compare_graphs(graphs, min_appearances=0.3)')
        print('  node_ids  = extract_node_ids_from_recurring(recurring, top_n=15)')
        print('  labels    = label_nodes_batch(node_ids, delay=0.5)')
    else:
        print('\nConnection test failed. Please check your API key.')
