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
    'sk-np-KIWbSmJKRUIbNq7gfOHK06GZN08x7Zkurig2kddEYJI0'
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

    # ── Step 5: Trace causal paths through edge weights ────────────────────
    # This is the structural complement to the band-by-band analysis above.
    # The band analysis tells you WHICH features mattered (ranked by influence).
    # The path analysis tells you HOW they are connected — the actual wiring
    # of the circuit from input tokens to the final logit.
    #
    # We run this AFTER labeling (steps 3-4) so that format_causal_paths_for_narration
    # can merge the human-readable labels into the path display.
    # Claude Code reads both the band output (above) and the path output (below)
    # together to write a complete mechanistic narrative.
    print(f'\n{"─"*66}')
    print('  CAUSAL PATH ANALYSIS — how did signals travel to the logit?')
    print(f'{"─"*66}')
    causal_paths = trace_causal_paths(
        G               = G,
        top_nodes       = top_nodes,
        max_paths       = 5,
        max_path_len    = 6,
        min_edge_weight = 0.05,
    )
    path_narration = format_causal_paths_for_narration(
        paths         = causal_paths,
        labeled_nodes = all_labeled,
        target_token  = target_token,
        target_prob   = target_prob,
    )

    # Render the causal paths as a matplotlib figure saved to the graphs folder.
    # This produces one PNG per prompt showing path nodes by layer position,
    # edge widths proportional to weight, and colour-coded by direction.
    # The returned path is passed to _append_prompt_to_paper so the paper
    # embeds the figure inline with a markdown image link.
    # If matplotlib is not installed the pipeline continues without error.
    causal_paths_figure = visualize_causal_paths(
        paths         = causal_paths,
        labeled_nodes = all_labeled,
        G             = G,
        prompt        = prompt,
        target_token  = target_token or '',
        save_path     = None,   # auto-saves to graphs/{slug}__{token}_causal_paths.png
    )

    # ── Step 6: Build the full interpretation data block for Claude Code ──
    interpretation = _call_claude_for_interpretation(
        prompt         = prompt,
        target_token   = target_token,
        target_prob    = target_prob,
        labeled_nodes  = all_labeled,
        top_candidates = top_candidates,
        nodes_by_layer = nodes_by_layer,
        band_titles    = band_titles,
        causal_paths   = causal_paths,
        model_id       = MODEL_ID,
    )

    result = {
        'prompt'              : prompt,
        'predicted_token'     : target_token,
        'token_prob'          : target_prob,
        'top_candidates'      : top_candidates,
        'nodes_by_layer'      : nodes_by_layer,
        'labeled_nodes'       : all_labeled,
        'causal_paths'        : causal_paths,
        'causal_paths_figure' : causal_paths_figure,
        'interpretation_data' : interpretation,
        'saved_to'            : None,
    }

    # ── Step 7: Append this prompt's results to the research paper ────────
    # Writing incrementally means Claude Code never has to hold 47
    # interpretations in memory simultaneously — each one is flushed to
    # disk as soon as it is complete.
    if paper_path:
        _append_prompt_to_paper(
            paper_path          = paper_path,
            prompt              = prompt,
            target_token        = target_token,
            target_prob         = target_prob,
            top_candidates      = top_candidates,
            nodes_by_layer      = nodes_by_layer,
            all_labeled         = all_labeled,
            causal_paths        = causal_paths,
            causal_paths_figure = causal_paths_figure,
        )

    # ── Step 8: Optionally save the full result dict ───────────────────────
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
                             nodes_by_layer: dict, all_labeled: list,
                             causal_paths: list = None,
                             causal_paths_figure: str = None):
    """
    Append a single prompt's interpretation results to the research paper
    markdown file. Called by interpret_prompt_graph() after each prompt
    completes so the paper is built incrementally — one section per prompt.

    Includes:
      - Token competition table
      - Circuit walkthrough table (band by band)
      - Causal paths table (edge chains to logit)
      - Inline figure link — the causal paths PNG is embedded using a
        markdown image tag so the figure appears directly in the paper
        when rendered, not just saved to disk.
      - [Claude Code — write narrative here] marker

    The figure path is relative to the paper file so the markdown renders
    correctly whether opened in VS Code, Obsidian, or any markdown viewer.
    """
    causal_paths        = causal_paths        or []
    causal_paths_figure = causal_paths_figure or None
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
        lines += ['| Layer | Feature | Node_id | Influence | Activation | Feature label |',
                  '|-------|---------|---------|-----------|------------|---------------|']
        for node in nodes:
            exp     = (node.get('explanation') or '(no label)').replace('|', '/')
            feature = node.get('feature', '?')
            node_id = node.get('node_id', '?')
            lines.append(
                f"| {node.get('layer','?')} "
                f"| {feature} "
                f"| {node_id} "
                f"| {node.get('influence','?')} "
                f"| {node.get('activation','?')} "
                f"| {exp} |"
            )
        lines.append('')

    # ── Causal paths section ───────────────────────────────────────────────
    if causal_paths:
        lines += ['**Causal paths (edge-weight chains to logit):**', '']
        label_map = {
            item.get('node_id', ''): item.get('explanation') or '(no label)'
            for item in all_labeled
        }
        # Build feature lookup from labeled nodes — feature ID is in the labeled
        # node dict, NOT parsed from node_id (middle number ≠ feature ID)
        feature_map = {
            item.get('node_id', ''): item.get('feature', '?')
            for item in all_labeled
        }
        for i, p in enumerate(causal_paths, 1):
            direction_str = {
                'excitatory': '(+) excitatory',
                'inhibitory': '(-) inhibitory',
                'mixed'     : '(±) mixed',
            }.get(p['direction'], p['direction'])
            lines.append(
                f'*Path {i} — {direction_str} | '
                f'weight={p["total_weight"]} | hops={len(p["path"])-1}*'
            )
            for j, nid in enumerate(p['path']):
                is_logit = (j == len(p['path']) - 1)
                lbl     = label_map.get(nid, '(no label)')
                feature = feature_map.get(nid, '?')
                # Layer is the first segment of node_id (this IS reliable)
                parts   = nid.split('_')
                layer_s = parts[0] if parts else '?'
                if is_logit:
                    lines.append(
                        f'→ **LOGIT** `node_id={nid}` Layer {layer_s} — `{target_token}`'
                    )
                else:
                    edge_w = p['edges'][j]
                    sign   = '+' if edge_w >= 0 else ''
                    lines.append(
                        f'→ `node_id={nid}` | `feature={feature}` | '
                        f'Layer {layer_s} | `{lbl}` '
                        f'— edge [{sign}{edge_w}]'
                    )
            lines.append('')

    # ── Embed the causal paths figure inline ──────────────────────────────
    # The PNG was saved to graphs/{slug}__{token}_causal_paths.png by
    # visualize_causal_paths().  We embed it using a relative path from the
    # paper file location so the figure renders in any markdown viewer.
    if causal_paths_figure:
        # Make the path relative to the paper file's directory
        import os
        paper_dir   = os.path.dirname(os.path.abspath(paper_path))
        fig_abs     = os.path.abspath(causal_paths_figure)
        try:
            fig_rel = os.path.relpath(fig_abs, paper_dir)
        except ValueError:
            # On Windows, relpath can fail across drives — fall back to absolute
            fig_rel = fig_abs
        fig_rel_md = fig_rel.replace('\\', '/')   # ensure forward slashes in MD

        token_display = (target_token or '').strip()
        lines += [
            '**Causal path diagram:**',
            '',
            f'![Causal paths for "{prompt[:50]}" → "{token_display}"]'
            f'({fig_rel_md})',
            '',
            f'*Figure: Causal paths from input features to predicted token '
            f'"{token_display}". '
            f'Y-axis = transformer layer. '
            f'Edge width = connection strength. '
            f'Green = excitatory, red = inhibitory, orange = mixed.*',
            '',
        ]

    lines += [
        '**[Claude Code — write mechanistic narrative here]**',
        '',
        '> *Using both the circuit walkthrough table AND the causal paths above*',
        '> *(including the figure), narrate the complete mechanistic story:*',
        '> *what did the model detect at each stage, how did signals travel*',
        '> *through the edges, which path was dominant, and why did the winner*',
        '> *beat the alternatives. Follow the style from the first prompt.*',
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
    causal_paths  : list = None,
) -> str:
    """
    Build and print a structured layer-by-layer interpretation data block
    that Claude Code reads to write its mechanistic narrative.

    First call: prints the full HOW-TO-NARRATE example so Claude Code
    learns the required voice and depth once.
    Subsequent calls: prints only the compact circuit data (no repeated
    example) to keep output manageable across 47 prompts.

    Now also includes a summary of causal paths so Claude Code can
    cross-reference the band-by-band analysis with the actual edge wiring.
    """
    global _INTERPRETATION_EXAMPLE_PRINTED
    nodes_by_layer = nodes_by_layer or {}
    band_titles    = band_titles    or {}
    causal_paths   = causal_paths   or []
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
    ]

    # ── Causal paths cross-reference ──────────────────────────────────────
    # Include a compact summary of paths so Claude Code can cross-reference
    # the band analysis (which features mattered) with the path analysis
    # (how they were connected). This is where the two analyses converge.
    if causal_paths:
        lines += [
            '  ┌─ CAUSAL PATHS SUMMARY (cross-reference with band analysis above)',
            '  │  These are the actual edge-weight chains from the graph JSON.',
            '  │  Use them to confirm or deepen your band narration:',
            '  │    - Does the dominant excitatory path match the early→middle→late',
            '  │      story you narrated above?',
            '  │    - Are the inhibitory paths consistent with the token competition',
            '  │      scores?  Do they explain why the runner-up tokens lost?',
            '  │    - Are there any features in the paths that were NOT in the top-40',
            '  │      band analysis?  If so, the path analysis has found a node that',
            '  │      influence-ranking missed — note it explicitly.',
            '  │',
        ]
        for i, p in enumerate(causal_paths, 1):
            dir_marker = {
                'excitatory': '(+)',
                'inhibitory': '(-)',
                'mixed'     : '(±)',
            }.get(p['direction'], '(?)')
            hop_count = len(p['path']) - 1
            lines.append(
                f"  │  Path {i} {dir_marker} weight={p['total_weight']}  "
                f"hops={hop_count}  type={p['path_type']}"
            )
            for j, (nid, lbl) in enumerate(
                zip(p['path'], p['node_labels'])
            ):
                is_last = j == len(p['path']) - 1
                parts   = nid.split('_')
                layer_s = parts[0] if parts else '?'
                if is_last:
                    lines.append(f'  │    └→ LOGIT L{layer_s}')
                else:
                    edge_w = p['edges'][j]
                    sign   = '+' if edge_w >= 0 else ''
                    lines.append(
                        f'  │    ├→ L{layer_s} [{sign}{edge_w}] {lbl}'
                    )
        lines += [
            '  │',
            '  └─ Cross-reference complete. Incorporate path insights into',
            '     your synthesis paragraph below.',
            '',
        ]

    lines += [
        '  ┌─ SYNTHESIS',
        '  │  Write one final paragraph tying the full causal chain together.',
        '  │  Use BOTH the band analysis and the causal paths:',
        '  │    early recognition → middle mapping → late push → predicted token.',
        '  │  Identify the single dominant pathway (the one excitatory path with',
        '  │  the highest weight) and name it explicitly.',
        '  │  Mention anything surprising: unexpected features, inhibitory paths',
        '  │  that reveal suppressed competitors, or path structure that differs',
        '  │  from what the influence rankings alone would suggest.',
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
    influences = [attrs.get('influence') or 0.0 for _, attrs in G.nodes(data=True)]

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
            'influence'      : round(attrs.get('influence') or 0.0, 4),
            'activation'     : round(attrs.get('activation') or 0.0, 4),
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
            'influence' : round(attrs.get('influence') or 0.0, 4),
            'activation': round(attrs.get('activation') or 0.0, 4),
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

def compare_graphs(graphs: list, min_appearances: int = None,
                   threshold: float = 0.50) -> list:
    """
    Find nodes (features) that appear consistently across multiple graphs.

    threshold : Fraction of graphs a feature must appear in to be considered
                a recurring circuit component (default 0.50 = 50%).

                With 12 prompts per category (the current dataset size),
                the values mean:
                  0.30 → appears in ≥ 4/10 graphs  (too permissive — noise risk)
                  0.50 → appears in ≥ 6/10 graphs  (majority — recommended)
                  0.67 → appears in ≥ 8/10 graphs  (strong consensus)

                The threshold is applied as ceil(threshold × N) so it always
                rounds up — e.g. 50% of 10 = 5.0 → 5, 50% of 7 = 3.5 → 4.

    min_appearances : If passed directly as an integer, overrides threshold.
                      Kept for backwards compatibility with old call sites
                      that passed a raw integer.
    """
    import math
    if min_appearances is None:
        min_appearances = math.ceil(threshold * len(graphs))
        print(f'[compare_graphs] threshold={threshold:.0%} × {len(graphs)} graphs '
              f'→ min_appearances={min_appearances}')
    else:
        print(f'[compare_graphs] min_appearances={min_appearances} '
              f'(passed directly, threshold parameter ignored)')

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
        # Pick the canonical (first) node_id as the representative for this feature
        canonical_node_id = occurrences[0]['node_id']
        results.append({
            'layer'         : layer,
            'feature'       : feature,
            'node_id'       : canonical_node_id,
            'appearances'   : len(occurrences),
            'out_of'        : len(graphs),
            'avg_influence' : round(avg_inf, 4),
            'avg_activation': round(avg_act, 4),
            'graph_slugs'   : [o['graph_slug'] for o in occurrences],
            'node_ids'      : [o['node_id']    for o in occurrences],
        })

    results.sort(key=lambda x: (x['appearances'], x['avg_influence']), reverse=True)

    # Print the recurring features table with Feature and Node_id columns
    print(f'\n[compare_graphs] Recurring features table '
          f'(threshold={threshold:.0%}, min_appearances={min_appearances}):')
    print(f'  {"Layer":<6} {"Feature":<12} {"Node_id":<18} '
          f'{"Appearances":<13} {"Avg Influence":<14} {"Avg Activation"}')
    print(f'  {"-"*6} {"-"*12} {"-"*18} {"-"*13} {"-"*14} {"-"*14}')
    for r in results:
        print(f'  {r["layer"]:<6} {r["feature"]:<12} {r["node_id"]:<18} '
              f'{r["appearances"]}/{r["out_of"]:<10} '
              f'{r["avg_influence"]:<14} {r["avg_activation"]}')

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


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL PATH TRACING  ← NEW
# ═════════════════════════════════════════════════════════════════════════════

def trace_causal_paths(
    G              : nx.DiGraph,
    top_nodes      : list,
    max_paths      : int   = 5,
    max_path_len   : int   = 6,
    min_edge_weight: float = 0.05,
) -> list:
    """
    Trace the strongest causal paths from influential feature nodes to the
    final logit (predicted token) node, following edge weights through the
    graph.

    WHY THIS MATTERS
    ----------------
    get_top_nodes() gives you a ranked LIST of important features — but a
    list has no structure.  It cannot tell you whether feature A caused
    feature B, or whether they are independent parallel contributors, or
    whether one suppresses the other.

    The edges in the graph carry exactly this structural information.  Each
    edge  source → target  with  weight W  means: "when source activates,
    it changes target's activation by W units."  A positive weight is
    excitatory (source amplifies target); a negative weight is inhibitory
    (source suppresses target).

    By following the highest-weight edges backwards from the logit node,
    we can reconstruct the actual COMPUTATIONAL PATHWAY the information
    travelled — not just a ranked list of individual actors, but the script
    that connects them.

    HOW THE ALGORITHM WORKS
    -----------------------
    1. Find the logit node (is_target_logit=True) — this is our destination.
    2. From the logit node, walk BACKWARDS along incoming edges, always
       following the edge with the highest absolute weight at each step.
    3. Continue until we reach a node with no influential predecessors
       (an input-level feature at an early layer) or until max_path_len
       is reached.
    4. Record the full path: [early_node → ... → mid_node → ... → logit].
    5. Repeat starting from each of the top_nodes to find multiple paths.
       Deduplicate paths that share the same sequence of node_ids.
    6. Sort paths by their total path weight (product of absolute edge
       weights along the path) — stronger paths rank higher.

    WHAT THE EDGE WEIGHT SIGN MEANS
    --------------------------------
    Positive weight: excitatory connection — this predecessor is PUSHING
        the target node toward higher activation, which contributes toward
        the predicted token.
    Negative weight: inhibitory connection — this predecessor is SUPPRESSING
        the target node.  A negative-weight path to the logit means this
        feature chain was actually working AGAINST the predicted token.
        This is important: it tells you which early-layer features were
        competing to produce a different answer.

    EXAMPLE OUTPUT (for "A book of maps is called an" → " atlas")
    -------------------------------------------------------------
    Path 1 (excitatory, weight=1.83):
      Layer 0 "books and reference materials" (inf=0.44)
        → [+0.31] →
      Layer 7 "reference codes and identifiers" (inf=0.81)
        → [+0.67] →
      Layer 14 "encyclopedic text patterns" (inf=1.23)
        → [+1.12] →
      Layer 22 "atlas and map-related terms" (inf=2.10)
        → [+1.89] →
      LOGIT: " atlas" (p=0.825)

    This tells you the MECHANISM: layer-0 surface recognition passed its
    signal to a reference-encoding feature at layer 7, which activated
    encyclopedic-register features at layer 14, which converged on the
    atlas-specific feature at layer 22 that directly pushed the logit.

    Path 2 (inhibitory, weight=-0.43):
      Layer 3 "index and catalogue terms" (inf=0.29)
        → [-0.43] →
      LOGIT: " atlas" (p=0.825)

    This tells you " index" was a suppressed competitor: a layer-3 feature
    was pushing toward " index" but lost the competition to Path 1.

    Parameters
    ----------
    G               : NetworkX DiGraph from load_graph().
    top_nodes       : List of top influential nodes from get_top_nodes().
                      Used as starting points for path search.
    max_paths       : Maximum number of distinct paths to return (default 5).
                      More paths = richer picture but more output to read.
    max_path_len    : Maximum number of hops in a single path (default 6).
                      Gemma-2-2B has 26 layers; a path longer than 6 hops
                      is likely traversing noise rather than true signal.
    min_edge_weight : Minimum absolute edge weight to follow (default 0.05).
                      Edges below this threshold are structural noise —
                      the source barely influences the target.

    Returns
    -------
    List of path dicts, sorted by abs(total_weight) descending:
    [
      {
        'path'        : [node_id, node_id, ..., logit_node_id],
        'edges'       : [weight, weight, ...],   # one per hop
        'total_weight': float,                   # product of abs(weights)
        'direction'   : 'excitatory' | 'inhibitory' | 'mixed',
        'node_labels' : [layer/feature/clerp per node],
        'path_type'   : 'direct' | 'multi-hop',
      }
    ]
    Returns empty list if no logit node found or no paths above threshold.
    """
    # ── Step 1: Find the logit node ───────────────────────────────────────
    logit_node = None
    for node_id, attrs in G.nodes(data=True):
        if attrs.get('is_target_logit'):
            logit_node = node_id
            break

    if logit_node is None:
        print('[trace_causal_paths] No logit node found in graph — cannot trace paths.')
        return []

    logit_clerp = G.nodes[logit_node].get('clerp', logit_node)
    print(f'\n[trace_causal_paths] Tracing causal paths to logit: "{logit_clerp}"')
    print(f'  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.')
    print(f'  Settings: max_paths={max_paths}, max_path_len={max_path_len}, '
          f'min_edge_weight={min_edge_weight}')

    # Build a quick lookup: node_id → (layer, feature, influence, clerp)
    def node_label(nid):
        attrs = G.nodes[nid]
        lyr   = attrs.get('layer', '?')
        feat  = attrs.get('feature', '?')
        clp   = attrs.get('clerp', '')
        inf   = attrs.get('influence')
        inf_s = f'inf={inf:.4f}' if inf is not None else 'inf=?'
        return f'L{lyr}/F{feat} [{inf_s}] clerp="{clp}"'

    # ── Step 2: For each top node, find the strongest path to the logit ───
    # Strategy: greedy forward walk from each top node following the
    # highest-weight outgoing edge at each step until we hit the logit node
    # or run out of hops.  This is not exhaustive (not all paths) but finds
    # the dominant causal pathway from each starting feature.

    found_paths = []
    seen_path_sigs = set()   # deduplicate by node sequence

    # Also do a backwards walk from the logit to find paths not starting
    # in top_nodes — sometimes the strongest path bypasses the top-40.
    start_nodes = [n['node_id'] for n in top_nodes] + [logit_node]

    for start_nid in start_nodes:
        if start_nid == logit_node:
            # Backwards walk from logit
            path    = [logit_node]
            weights = []
            current = logit_node
            for _hop in range(max_path_len - 1):
                in_edges = [
                    (u, data['weight'])
                    for u, v, data in G.in_edges(current, data=True)
                    if abs(data.get('weight', 0.0)) >= min_edge_weight
                    and u not in path   # no cycles
                ]
                if not in_edges:
                    break
                # Follow the strongest incoming edge
                best_pred, best_w = max(in_edges, key=lambda x: abs(x[1]))
                weights.insert(0, best_w)
                path.insert(0, best_pred)
                current = best_pred
            # Only keep if path has at least 2 nodes
            if len(path) >= 2:
                sig = tuple(path)
                if sig not in seen_path_sigs:
                    seen_path_sigs.add(sig)
                    found_paths.append((path, weights))
        else:
            # Forward walk from top node toward logit
            path    = [start_nid]
            weights = []
            current = start_nid
            reached_logit = False
            for _hop in range(max_path_len - 1):
                out_edges = [
                    (v, data['weight'])
                    for u, v, data in G.out_edges(current, data=True)
                    if abs(data.get('weight', 0.0)) >= min_edge_weight
                    and v not in path   # no cycles
                ]
                if not out_edges:
                    break
                # Follow the strongest outgoing edge
                best_succ, best_w = max(out_edges, key=lambda x: abs(x[1]))
                weights.append(best_w)
                path.append(best_succ)
                current = best_succ
                if best_succ == logit_node:
                    reached_logit = True
                    break
            # Only keep paths that reach the logit node
            if reached_logit and len(path) >= 2:
                sig = tuple(path)
                if sig not in seen_path_sigs:
                    seen_path_sigs.add(sig)
                    found_paths.append((path, weights))

    # ── Step 3: Score and sort paths ──────────────────────────────────────
    # Total weight = product of absolute edge weights.
    # A path where every edge is strong (e.g. 0.8 × 0.7 × 0.9) scores
    # higher than a path with one strong edge and several weak ones.
    def score_path(weights):
        if not weights:
            return 0.0
        score = 1.0
        for w in weights:
            score *= abs(w)
        return round(score, 6)

    def path_direction(weights):
        pos = sum(1 for w in weights if w > 0)
        neg = sum(1 for w in weights if w < 0)
        if neg == 0:
            return 'excitatory'
        elif pos == 0:
            return 'inhibitory'
        else:
            return 'mixed'

    results = []
    for path, weights in found_paths:
        total_w   = score_path(weights)
        direction = path_direction(weights)
        path_type = 'direct' if len(path) == 2 else 'multi-hop'
        results.append({
            'path'        : path,
            'edges'       : [round(w, 4) for w in weights],
            'total_weight': total_w,
            'direction'   : direction,
            'node_labels' : [node_label(nid) for nid in path],
            'path_type'   : path_type,
        })

    results.sort(key=lambda x: x['total_weight'], reverse=True)
    results = results[:max_paths]

    # ── Step 4: Print for Claude Code ─────────────────────────────────────
    print(f'\n[trace_causal_paths] Found {len(results)} causal path(s) '
          f'(showing top {max_paths}):')

    for i, p in enumerate(results, 1):
        direction_marker = {
            'excitatory': '(+) excitatory — pushing TOWARD predicted token',
            'inhibitory': '(-) inhibitory — pushing AGAINST predicted token',
            'mixed'     : '(±) mixed — partially excitatory, partially inhibitory',
        }[p['direction']]

        print(f'\n  Path {i} | {direction_marker}')
        print(f'  Total path weight: {p["total_weight"]}  '
              f'| Type: {p["path_type"]}  '
              f'| Hops: {len(p["path"]) - 1}')
        print(f'  ──')

        for j, (nid, label) in enumerate(zip(p['path'], p['node_labels'])):
            is_last = (j == len(p['path']) - 1)
            if is_last:
                print(f'  → LOGIT  {label}')
            else:
                edge_w   = p['edges'][j]
                sign_str = f'+{edge_w}' if edge_w >= 0 else str(edge_w)
                print(f'  → NODE   {label}')
                print(f'           ↓ edge weight [{sign_str}]')

    return results


def format_causal_paths_for_narration(
    paths        : list,
    labeled_nodes: list,
    target_token : str,
    target_prob  : float,
) -> str:
    """
    Format the output of trace_causal_paths() into a structured block that
    Claude Code reads to narrate the causal chain in the research paper.

    This function bridges the gap between raw path data and the mechanistic
    narrative.  It organises paths into three groups and gives Claude Code
    explicit instructions on what each group means and how to reason about it.

    THREE GROUPS
    ------------
    EXCITATORY PATHS: the dominant computation — features that collectively
        built the case for the predicted token.  Narrate these as the
        story of HOW the model arrived at its answer: what did it recognise
        first, how did that signal propagate, what feature made the final
        push to the logit.

    INHIBITORY PATHS: suppressed competitors — features that were pushing
        toward a different token but lost.  Narrate these as the story of
        WHY the alternatives failed: what features were active for " index"
        or " almanac" or whatever the runner-up was, and why they were
        overpowered.

    MIXED PATHS: ambiguous circuits where excitatory and inhibitory edges
        are interleaved.  These often represent feature interactions where
        one feature gates or modulates another.  Narrate these carefully —
        a mixed path often reveals the most interesting circuit behaviour.

    HOW PATHS RELATE TO labeled_nodes
    ----------------------------------
    The labeled_nodes list (from interpret_prompt_graph) gives you the
    human-readable Neuronpedia label for each node.  This function merges
    those labels into the path display so Claude Code sees, for each hop:

        Layer 7 | inf=0.81 | "reference codes and identifiers"
            ↓ edge weight [+0.67]
        Layer 14 | inf=1.23 | "encyclopedic text patterns"

    ...rather than raw node_ids.  The label is what makes the path
    mechanistically interpretable — without it you have a chain of numbers,
    with it you have a story.

    Parameters
    ----------
    paths         : List returned by trace_causal_paths().
    labeled_nodes : Flat list of labeled node dicts from interpret_prompt_graph().
                    Used to enrich path nodes with human-readable labels.
    target_token  : The predicted token string (e.g. ' atlas').
    target_prob   : Probability of the predicted token (e.g. 0.825).

    Returns
    -------
    A formatted string printed to stdout for Claude Code to read and narrate.
    """
    # Build lookup: node_id → explanation label
    label_map = {
        item.get('node_id', ''): item.get('explanation') or '(no label)'
        for item in labeled_nodes
    }

    prob_str = f'p={target_prob:.4f}' if target_prob else 'p=?'

    lines = [
        '',
        '╔══════════════════════════════════════════════════════════════════╗',
        '║   CAUSAL PATH ANALYSIS  —  instructions for Claude Code          ║',
        '╚══════════════════════════════════════════════════════════════════╝',
        f'  Predicted token : "{target_token}" ({prob_str})',
        f'  Paths found     : {len(paths)}',
        '',
        '  These paths show the ACTUAL WIRING of the circuit — how signals',
        '  travelled through the network from input features to the final',
        '  prediction.  Each hop is an edge in the attribution graph with a',
        '  real weight from the JSON.',
        '',
        '  HOW TO NARRATE PATHS (Claude Code instructions):',
        '',
        '  EXCITATORY paths (+): narrate as the dominant computational story.',
        '    "The circuit begins at layer X where feature Y detected [label].',
        '    This signal passed through layer Z [label] with edge weight +W,',
        '    amplifying the representation toward [target_token].  The chain',
        '    converges at layer N where [label] delivers the final push to',
        '    the logit with edge weight +W2."',
        '',
        '  INHIBITORY paths (-): narrate as the suppressed competitor story.',
        '    "A parallel pathway through layer X [label] was pushing toward',
        '    a competing token.  Its inhibitory connection (edge weight -W)',
        '    to the logit reveals it was actively working against [target_token]',
        '    — this is the mechanistic signature of token competition."',
        '',
        '  MIXED paths (±): narrate as a gating or modulation story.',
        '    "Feature [label] at layer X appears to gate downstream processing',
        '    — its positive edge to [B] amplifies the reference-document signal',
        '    while its negative edge to [C] suppresses a competing pathway.',
        '    This is a control feature, not a direct contributor."',
        '',
        '  KEY QUESTIONS TO ANSWER FOR EACH PATH:',
        '    1. What does the FIRST node in the path represent?',
        '       (This is what the model noticed in the input that started',
        '        this chain of computation.)',
        '    2. Does edge weight INCREASE or DECREASE along the path?',
        '       (Increasing weights = signal is being amplified and focused.',
        '        Decreasing weights = signal is diffusing and losing strength.)',
        '    3. Is the final edge to the logit strong or weak?',
        '       (Strong final edge = this path is a major contributor.',
        '        Weak final edge = this path exists but barely mattered.)',
        '    4. How does this path relate to the top_candidates token scores?',
        '       (The excitatory paths explain why the winner won.',
        '        The inhibitory paths explain why the runner-ups lost.)',
        '',
        '  ── PATHS ────────────────────────────────────────────────────────',
        '',
    ]

    # Group paths by direction
    excitatory = [p for p in paths if p['direction'] == 'excitatory']
    inhibitory = [p for p in paths if p['direction'] == 'inhibitory']
    mixed      = [p for p in paths if p['direction'] == 'mixed']

    def render_path(p, idx):
        """Render one path with labels merged in."""
        direction_str = {
            'excitatory': '(+) EXCITATORY — builds toward predicted token',
            'inhibitory': '(-) INHIBITORY — pushes against predicted token',
            'mixed'     : '(±) MIXED — gating / modulation behaviour',
        }[p['direction']]

        out = [
            f'  Path {idx}  {direction_str}',
            f'  Total weight={p["total_weight"]}  '
            f'Hops={len(p["path"])-1}  Type={p["path_type"]}',
            '  │',
        ]
        for j, nid in enumerate(p['path']):
            is_logit = (j == len(p['path']) - 1)
            attrs    = {}
            label    = label_map.get(nid, '(no label)')

            # Try to get layer/influence from node_id format L_layer
            # node_ids are like "7_3099_1" → layer=7
            parts = nid.split('_')
            layer_s = parts[0] if parts else '?'
            # Try to get influence from labeled_nodes
            inf_val = next(
                (item.get('influence') for item in labeled_nodes
                 if item.get('node_id') == nid),
                None
            )
            inf_s = f'inf={inf_val:.4f}' if inf_val is not None else ''

            if is_logit:
                out.append(f'  └→ LOGIT  Layer {layer_s} | "{target_token}" ({prob_str})')
            else:
                edge_w   = p['edges'][j]
                sign_str = f'+{edge_w}' if edge_w >= 0 else str(edge_w)
                out.append(f'  ├→ NODE   Layer {layer_s} {inf_s} | "{label}"')
                out.append(f'  │         ↓ edge [{sign_str}]  '
                           f'{"amplifies →" if edge_w > 0 else "suppresses ↓"}')
        out.append('')
        return out

    path_idx = 1
    if excitatory:
        lines += [
            '  ┌─ EXCITATORY PATHS — the dominant story of how the model',
            '  │  built its case for the predicted token',
            '  │',
        ]
        for p in excitatory:
            lines += render_path(p, path_idx)
            path_idx += 1
        lines += [
            '  └─ NARRATE EXCITATORY: Walk through each path step by step.',
            '     Explain what the first node detected, how the signal was',
            '     amplified or transformed at each hop, and what the final',
            '     node contributed directly to the logit.',
            '',
        ]

    if inhibitory:
        lines += [
            '  ┌─ INHIBITORY PATHS — the suppressed competitors',
            '  │  These features were pushing for a different token',
            '  │',
        ]
        for p in inhibitory:
            lines += render_path(p, path_idx)
            path_idx += 1
        lines += [
            '  └─ NARRATE INHIBITORY: Identify which competing token these',
            '     paths were supporting.  Explain why they lost — was the',
            '     inhibitory signal weak, or was it overpowered by stronger',
            '     excitatory paths?',
            '',
        ]

    if mixed:
        lines += [
            '  ┌─ MIXED PATHS — gating and modulation',
            '  │',
        ]
        for p in mixed:
            lines += render_path(p, path_idx)
            path_idx += 1
        lines += [
            '  └─ NARRATE MIXED: Identify the feature that acts as a gate.',
            '     Explain what it is modulating and whether this represents',
            '     feature competition, conditional routing, or noise.',
            '',
        ]

    if not paths:
        lines += [
            '  [No paths found above the minimum edge weight threshold.',
            '   This may indicate the graph edges are all below 0.05 or',
            '   that the logit node has no strong direct predecessors.',
            '   Consider reducing min_edge_weight and re-running.]',
            '',
        ]

    lines += [
        '  ┌─ SYNTHESIS TASK (causal paths)',
        '  │  After narrating each path above, write a paragraph that',
        '  │  answers: what is the COMPLETE causal story from input to',
        '  │  output for this prompt?',
        '  │',
        '  │  Specifically address:',
        '  │    - Which excitatory path was the dominant mechanism?',
        '  │    - Were there genuine competitors (inhibitory paths)?',
        '  │    - Do the paths confirm or contradict the layer-band',
        '  │      narrative from the band-by-band analysis above?',
        '  │    - Is this a clean convergent circuit (one dominant path,',
        '  │      weak competitors) or a genuinely ambiguous one (multiple',
        '  │      paths of similar strength, many mixed edges)?',
        '  └─',
        '',
        '╔══════════════════════════════════════════════════════════════════╗',
        '║   END OF CAUSAL PATH DATA                                        ║',
        '╚══════════════════════════════════════════════════════════════════╝',
    ]

    block = '\n'.join(lines)
    print(block)
    return block


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL PATH VISUALIZATION  ← NEW
# ═════════════════════════════════════════════════════════════════════════════

def visualize_causal_paths(
    paths        : list,
    labeled_nodes: list,
    G            : nx.DiGraph,
    prompt       : str,
    target_token : str,
    save_path    : str  = None,
    figsize      : tuple = (14, 9),
) -> None:
    """
    Render the causal paths from trace_causal_paths() as a publication-ready
    matplotlib figure and save it to disk.

    WHAT THE FIGURE SHOWS
    ---------------------
    Each path is drawn as a vertical chain of nodes connected by directed
    edges.  Multiple paths are drawn side by side in columns.

    Y-AXIS = transformer layer number (0 at top, max_layer at bottom).
    This mirrors how transformer computation actually flows: information
    enters at layer 0, is processed downward through the layers, and exits
    at the logit node at the final layer.  A node's vertical position is
    therefore its position in the causal chain — early-layer nodes sit near
    the top, late-layer nodes near the bottom, the logit node at the very
    bottom.

    X-AXIS = path index (one column per path).  Paths are sorted left to
    right by total weight (strongest path on the left).

    NODE COLOUR = path direction:
        Green  (#2ecc71) — excitatory node (part of a (+) path)
        Red    (#e74c3c) — inhibitory node (part of a (-) path)
        Orange (#f39c12) — mixed node (part of a (±) path)
        Gold   (#f1c40f) — logit node (always the bottom node, shared)

    EDGE WIDTH = proportional to absolute edge weight.
    Thick edges are strong connections; thin edges are weak ones.
    Positive edges are solid lines; negative edges are dashed lines
    (dashed = inhibitory — the source is suppressing the target).

    NODE LABEL = truncated Neuronpedia explanation (max 28 chars) + layer
    number + influence score.  If no label is available, the raw node_id
    is shown.  Labels are wrapped to avoid overlap.

    WHY THIS IS USEFUL FOR THE PAPER
    ---------------------------------
    The band-by-band table in the paper tells you which features mattered.
    The causal path figure shows HOW they were connected — it is the visual
    equivalent of the mechanistic narrative.  A reader can look at the figure
    and immediately see:
        - Which layer the computation started at
        - How many hops it took to reach the output
        - Whether the signal was amplified (increasing edge widths) or
          diffused (decreasing edge widths) along the way
        - Whether there were competing inhibitory paths that nearly won

    USAGE
    -----
    Call this immediately after trace_causal_paths() and
    format_causal_paths_for_narration() inside interpret_prompt_graph,
    or call it standalone after the fact using saved path data:

        paths  = trace_causal_paths(G, top_nodes)
        labels = label_nodes_batch(extract_node_ids_from_top(paths))
        visualize_causal_paths(
            paths         = paths,
            labeled_nodes = all_labeled,
            G             = G,
            prompt        = "A book of maps is called an",
            target_token  = " atlas",
            save_path     = "graphs/atlas_causal_paths.png",
        )

    Parameters
    ----------
    paths         : List returned by trace_causal_paths().
    labeled_nodes : Flat list of labeled node dicts from interpret_prompt_graph().
                    Used to get human-readable labels for each node.
    G             : The loaded NetworkX DiGraph (from load_graph()).
                    Used to get layer numbers for nodes not in labeled_nodes.
    prompt        : The original prompt string (for the figure title).
    target_token  : The predicted token string (shown at logit node).
    save_path     : File path to save the figure (PNG or PDF).
                    If None, saves to graphs/{slug}_causal_paths.png.
    figsize       : Matplotlib figure size tuple (width, height) in inches.
                    Default (14, 9) is good for up to 5 paths.  Increase
                    width for more paths, increase height for longer paths.

    Returns
    -------
    None.  Saves the figure to save_path and prints the file path.
    Requires matplotlib.  If matplotlib is not installed, prints a warning
    and returns without error so the pipeline continues uninterrupted.
    """
    # ── Dependency check ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend — safe for Claude Code
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines   import Line2D
    except ImportError:
        print('[visualize_causal_paths] matplotlib not installed. '
              'Run: pip install matplotlib --break-system-packages\n'
              'Skipping visualization and continuing pipeline.')
        return

    if not paths:
        print('[visualize_causal_paths] No paths to visualize — skipping.')
        return

    # ── Build label lookup ────────────────────────────────────────────────
    label_map = {
        item.get('node_id', ''): item.get('explanation') or ''
        for item in labeled_nodes
    }
    inf_map = {
        item.get('node_id', ''): item.get('influence')
        for item in labeled_nodes
    }
    # Feature lookup — use the feature field from labeled_nodes directly.
    # Do NOT parse from node_id: middle number in node_id ≠ feature ID.
    feature_map = {
        item.get('node_id', ''): item.get('feature', '?')
        for item in labeled_nodes
    }

    def get_layer(nid):
        """Extract integer layer from node_id or from graph attrs."""
        attrs = G.nodes.get(nid, {})
        lyr   = attrs.get('layer')
        if lyr is not None:
            try:
                return int(lyr)
            except (TypeError, ValueError):
                pass
        # Fall back to parsing node_id  "layer_feature_ctx"
        parts = nid.split('_')
        try:
            return int(parts[0])
        except (IndexError, ValueError):
            return 0

    def get_label(nid, max_chars=28):
        """Get truncated human-readable label for a node."""
        lbl = label_map.get(nid, '')
        if not lbl:
            # Fall back to clerp from graph
            lbl = G.nodes.get(nid, {}).get('clerp', '') or nid
        if len(lbl) > max_chars:
            lbl = lbl[:max_chars - 1] + '…'
        return lbl

    # ── Determine layer range across all paths ────────────────────────────
    all_layers_in_paths = []
    for p in paths:
        for nid in p['path']:
            all_layers_in_paths.append(get_layer(nid))
    min_lyr = min(all_layers_in_paths) if all_layers_in_paths else 0
    max_lyr = max(all_layers_in_paths) if all_layers_in_paths else 26

    # ── Colour scheme ─────────────────────────────────────────────────────
    COLOUR = {
        'excitatory': '#27ae60',   # green  — building toward predicted token
        'inhibitory': '#c0392b',   # red    — pushing against predicted token
        'mixed'     : '#d35400',   # orange — gating / modulation
        'logit'     : '#f39c12',   # amber  — final prediction node
        'edge_pos'  : '#2c3e50',   # dark   — excitatory edge
        'edge_neg'  : '#c0392b',   # red    — inhibitory edge
        'bg'        : '#fafafa',
        'title'     : '#2c3e50',
        'label_fg'  : '#ffffff',
    }

    n_paths  = len(paths)
    col_gap  = 1.0 / (n_paths + 1)   # horizontal spacing between path columns

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(max_lyr + 1, min_lyr - 1)   # y increases downward (layer 0 at top)
    ax.set_facecolor(COLOUR['bg'])
    fig.patch.set_facecolor(COLOUR['bg'])

    # Draw layer grid lines
    for lyr in range(min_lyr, max_lyr + 1):
        ax.axhline(lyr, color='#e0e0e0', linewidth=0.5, zorder=0)
        ax.text(-0.01, lyr, f'L{lyr}', fontsize=7, color='#aaaaaa',
                va='center', ha='right', transform=ax.get_yaxis_transform())

    # ── Draw each path as a column ────────────────────────────────────────
    max_edge_w = max(
        (abs(w) for p in paths for w in p['edges']),
        default=1.0
    )

    for col_idx, p in enumerate(paths):
        x = col_gap * (col_idx + 1)
        direction = p['direction']
        node_colour = COLOUR.get(direction, COLOUR['excitatory'])

        node_positions = {}   # nid → (x, y) for edge drawing

        for j, nid in enumerate(p['path']):
            is_logit = (j == len(p['path']) - 1)
            y = get_layer(nid)
            node_positions[nid] = (x, y)

            # Node circle
            colour = COLOUR['logit'] if is_logit else node_colour
            circle = plt.Circle(
                (x, y), radius=0.07,
                color=colour, zorder=3, clip_on=False
            )
            ax.add_patch(circle)

            # Node label — node_id + feature + short explanation + influence
            lbl     = get_label(nid)
            inf     = inf_map.get(nid)
            feature = feature_map.get(nid, G.nodes.get(nid, {}).get('feature', '?'))
            inf_s   = f'\ninf={inf:.3f}' if inf is not None else ''
            if is_logit:
                display = f'LOGIT\n"{target_token}"'
            else:
                display = f'L{y} | nid:{nid}\nF:{feature}\n{lbl}{inf_s}'

            ax.text(
                x + 0.09, y, display,
                fontsize  = 7,
                va        = 'center',
                ha        = 'left',
                color     = COLOUR['title'],
                zorder    = 4,
                wrap      = True,
                bbox      = dict(
                    boxstyle  = 'round,pad=0.2',
                    facecolor = '#ffffff',
                    edgecolor = colour,
                    linewidth = 0.8,
                    alpha     = 0.85,
                ),
            )

        # Draw edges between consecutive nodes
        for j in range(len(p['path']) - 1):
            src = p['path'][j]
            dst = p['path'][j + 1]
            w   = p['edges'][j]

            x0, y0 = node_positions[src]
            x1, y1 = node_positions[dst]

            # Edge width proportional to absolute weight
            lw        = 1.0 + 5.0 * (abs(w) / max(max_edge_w, 1e-6))
            linestyle = '--' if w < 0 else '-'
            ecolour   = COLOUR['edge_neg'] if w < 0 else COLOUR['edge_pos']

            ax.annotate(
                '',
                xy     = (x1, y1 - 0.07),
                xytext = (x0, y0 + 0.07),
                arrowprops = dict(
                    arrowstyle = '-|>',
                    color      = ecolour,
                    lw         = lw,
                    linestyle  = linestyle,
                ),
                zorder = 2,
            )

            # Edge weight label at midpoint
            mx = (x0 + x1) / 2 - 0.04
            my = (y0 + y1) / 2
            sign = '+' if w >= 0 else ''
            ax.text(
                mx, my, f'{sign}{w:.3f}',
                fontsize  = 6.5,
                color     = ecolour,
                ha        = 'right',
                va        = 'center',
                zorder    = 5,
                bbox      = dict(
                    boxstyle  = 'round,pad=0.1',
                    facecolor = COLOUR['bg'],
                    edgecolor = 'none',
                    alpha     = 0.7,
                ),
            )

        # Path header above each column
        dir_symbol = {'excitatory': '(+)', 'inhibitory': '(-)', 'mixed': '(±)'}.get(
            direction, ''
        )
        ax.text(
            x, min_lyr - 0.6,
            f'Path {col_idx+1} {dir_symbol}\nweight={p["total_weight"]:.4f}',
            fontsize  = 7.5,
            ha        = 'center',
            va        = 'bottom',
            color     = node_colour,
            fontweight= 'bold',
            zorder    = 5,
        )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLOUR['excitatory'], label='Excitatory (+)'),
        mpatches.Patch(color=COLOUR['inhibitory'], label='Inhibitory (-)'),
        mpatches.Patch(color=COLOUR['mixed'],      label='Mixed (±)'),
        mpatches.Patch(color=COLOUR['logit'],      label='Logit node'),
        Line2D([0], [0], color=COLOUR['edge_pos'], lw=2,
               linestyle='-',  label='Positive edge'),
        Line2D([0], [0], color=COLOUR['edge_neg'], lw=2,
               linestyle='--', label='Negative edge'),
    ]
    ax.legend(
        handles   = legend_handles,
        loc       = 'lower right',
        fontsize  = 7,
        framealpha= 0.9,
        edgecolor = '#cccccc',
    )

    # ── Axes and title ────────────────────────────────────────────────────
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Wrap long prompt strings
    prompt_display = prompt if len(prompt) <= 60 else prompt[:57] + '...'
    ax.set_title(
        f'Causal paths → "{target_token}"\n'
        f'Prompt: "{prompt_display}"',
        fontsize  = 11,
        color     = COLOUR['title'],
        pad       = 14,
        fontweight= 'normal',
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    # Filename includes both the prompt slug AND the predicted token so files
    # are immediately identifiable on disk without opening them.
    # e.g. paris_is_to_france_as_berlin_is_to__germany_causal_paths.png
    if save_path is None:
        slug_raw    = prompt.lower().replace(' ', '_')[:40].strip('_')
        slug        = ''.join(c if c.isalnum() or c == '_' else '_' for c in slug_raw)
        token_clean = (target_token or 'unknown').strip().lower()
        token_clean = ''.join(c if c.isalnum() or c == '_' else '_'
                              for c in token_clean).strip('_')[:20]
        save_path = str(GRAPHS_DIR / f'{slug}__{token_clean}_causal_paths.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLOUR['bg'])
    plt.close(fig)
    print(f'[visualize_causal_paths] Saved → {save_path}')
    return save_path   # return so caller can embed in paper


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
# CIRCUIT STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def save_circuit(name: str, nodes: list, description: str,
                 prompt_category: str, source_graphs: list) -> str:
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
        'created_at'        : time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    filename = name.replace(' ', '_').lower() + '.json'
    path = CIRCUITS_DIR / filename
    with open(path, 'w') as f:
        json.dump(circuit, f, indent=2)

    print(f'[save_circuit] Saved "{name}" → {path}')
    return str(path)


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT DATASET  (scaled: 10 per category)
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt_dataset() -> dict:
    """

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
            'Hamlet was written by',
            'The theory of relativity was developed by',
            'The powerhouse of the cell is the',
            'Water is composed of hydrogen and',
            'The first US president was',
            'Napoleon was exiled to the island of',
            'The longest river in the world is the',
            'Mount Everest is located in the', 
        ],

        # ── LINGUISTIC ────────────────────────────────────────────────────
                'linguistic': [
            'The opposite of hot is',
            'The antonym of ancient is',
            'A synonym for happy is',
            'Another word for fast is',
            'The plural of child is',
            'The plural of mouse is',
            'The past tense of swim is',
            'The past tense of write is',
            'A book of maps is called an',
            'A person who writes books is called an',
        ],

        # ── ANALOGICAL ────────────────────────────────────────────────────
        'analogical': [
            'Paris is to France as Berlin is to',
            'Cairo is to Egypt as Nairobi is to',
            'Fish is to water as bird is to',
            'Puppy is to dog as kitten is to',
            'Clock is to time as thermometer is to',
            'Book is to reading as radio is to',
            'Leaf is to tree as petal is to',
            'Wheel is to car as wing is to',
            'Judge is to court as priest is to',
            'Soldier is to army as sailor is to',
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

        print('\n-- New functions available --')
        print('  trace_causal_paths(G, top_nodes)            -> edge-weight paths to logit')
        print('  format_causal_paths_for_narration(...)       -> Claude Code narration block')
        print('  visualize_causal_paths(paths, labels, G,...) -> matplotlib PNG figure')
        print('  extract_node_ids_from_recurring(recurring, top_n)')
        print('  get_token_predictions(prompt)               -> ranked next-token list')
        print('  get_logit_candidates_from_graph(data)       -> token votes from graph JSON')
        print('  interpret_prompt_graph(G, graph_data)       -> full per-prompt analysis')

        print('\nExample workflow:')
        print('  recurring = compare_graphs(graphs, threshold=0.50)  # 50% of graphs')
        print('  node_ids  = extract_node_ids_from_recurring(recurring, top_n=15)')
        print('  labels    = label_nodes_batch(node_ids, delay=0.5)')
    else:
        print('\nConnection test failed. Please check your API key.')
