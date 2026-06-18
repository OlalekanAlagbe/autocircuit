#!/usr/bin/env python3
"""
Experiment 8: Causal Validation of the Full 180-Feature Analogical Circuit

Runs the original 4-test causal validation battery on the 180-feature shared
analogical circuit (features appearing in >=5 nodes across the 5 analogy
attribution graphs: Berlin/Rome/Tokyo/teacher/bird).

Differs from exp6/exp7 only in the feature set: this validates the actual
shared circuit cited in the paper (180 features), not the per-prompt
9-feature top-5-causal-path backbone.

Focal prompt: "Paris is to France as Berlin is to" -> "Germany"
Same parameters as exp6 (strength=-20, strength_multiplier=4, seed=42).

Tests:
  1. Necessity (individual): suppress each of 180 features one at a time.
  2. Full circuit suppression: suppress all 180 simultaneously.
  3. Sufficiency: boost top-1 hub feature on 3 altered prompts.
  4. Specificity: suppress 20 features in the Berlin graph that are NOT in
     the 180-feature circuit, sampled across activation bands.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# Reuse exp2's primitives (same API parameters as the rest of the paper)
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

# --- Configuration -------------------------------------------------

DELAY = 2.0  # seconds between API calls
STRENGTH_SUPPRESS = -20
STRENGTH_BOOST = 20
N_SPECIFICITY_SAMPLES = 20
SEED = 42

PROMPT = "Paris is to France as Berlin is to"
EXPECTED_TOKEN = "Germany"

# Sufficiency prompts match exp6's analog_berlin design
SUFFICIENCY_PROMPTS = [
    ("Cairo is to Egypt as Nairobi is to", "different analogy same structure"),
    ("Madrid is to Spain as Berlin is to", "different source pair (Berlin still present)"),
    ("The weather today is really", "irrelevant context"),
]

OUTPUT_DIR = Path(__file__).parent
CIRCUIT_JSON = Path("/Users/anish/Downloads/analogical_circuit_180features.json")
BERLIN_GRAPH = Path("/Users/anish/Desktop/aisc_project/experiments/exp4_scale/circuits/analogy_graph.json")
RESULTS_PATH = OUTPUT_DIR / "results.json"


# --- Helpers -------------------------------------------------------

def parse_node_id(node_id: str) -> tuple[int, int, int]:
    """'4_14857_1' -> (layer=4, local_index=14857, ctx_idx=1)"""
    parts = node_id.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_circuit_features(path: Path) -> list[dict]:
    """Load the 180-feature circuit. Returns list of {layer, index, name, avg_influence}."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for feat in d["features"]:
        # Extract layer + local index from any node_id (all node_ids for a feature share the local index)
        node_id = feat["node_ids"][0]
        layer, local_idx, _ = parse_node_id(node_id)
        out.append({
            "name": f"L{layer}_F{local_idx}",
            "layer": layer,
            "index": local_idx,
            "avg_influence": feat["avg_influence"],
            "avg_activation": feat["avg_activation"],
            "total_appearances": feat["total_appearances"],
        })
    return out


def sample_non_circuit_features(graph_path: Path, circuit_set: set[tuple[int, int]],
                                n: int, seed: int) -> list[dict]:
    """Sample features from analog_berlin graph that are NOT in the circuit.

    Stratified sampling: split high-activation non-circuit features into 4 bands,
    sample equally from each. Yields a representative spread, not just top features.
    """
    with open(graph_path) as f:
        g = json.load(f)
    candidates = []
    for node in g["nodes"]:
        if node.get("feature_type") != "cross layer transcoder":
            continue
        try:
            layer = int(node["layer"])
            idx = int(node["feature"])  # local index in this graph schema
        except (KeyError, ValueError, TypeError):
            continue
        # The 'feature' field in this graph is the layer-local index already
        # (cross-checked against node_id "L_idx_pos" pattern)
        node_id = node.get("node_id", "")
        try:
            l_check, idx_check, _ = parse_node_id(node_id)
            layer = l_check
            idx = idx_check
        except (ValueError, IndexError):
            pass
        if (layer, idx) in circuit_set:
            continue
        act = node.get("activation", 0.0)
        infl = node.get("influence", 0.0)
        candidates.append({
            "layer": layer,
            "index": idx,
            "activation": act,
            "influence": infl,
            "node_id": node_id,
        })

    # Deduplicate by (layer, index) — keep highest influence
    by_pair = {}
    for c in candidates:
        key = (c["layer"], c["index"])
        if key not in by_pair or c["influence"] > by_pair[key]["influence"]:
            by_pair[key] = c
    candidates = list(by_pair.values())

    candidates.sort(key=lambda x: x["activation"], reverse=True)
    if len(candidates) < n:
        return candidates

    rng = random.Random(seed)
    bands = 4
    band_size = len(candidates) // bands
    per_band = max(1, n // bands)
    sampled = []
    for b in range(bands):
        start = b * band_size
        end = start + band_size if b < bands - 1 else len(candidates)
        band_pop = candidates[start:end]
        sampled.extend(rng.sample(band_pop, min(per_band, len(band_pop))))
    return sampled[:n]


def predicts_target(result: dict, target: str) -> bool:
    """Did the steered output still predict the target token?"""
    if "error" in result:
        return False
    txt = result.get("steered_text", "")
    first = result.get("first_steered_token", "")
    return target.lower() in txt.lower() or target.lower() in first.lower()


def save_results(results: dict):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# --- Test runners --------------------------------------------------

def run_baseline() -> dict:
    print(f"\n[BASELINE] '{PROMPT}'")
    r = steer(PROMPT, [])
    print(f"  Default text: {r.get('default_text', '?')!r}")
    print(f"  First token: {r.get('first_default_token', '?')!r} "
          f"(logprob {r.get('first_default_logprob')})")
    return {
        "experiment": "baseline",
        "prompt": PROMPT,
        "expected": EXPECTED_TOKEN,
        "result": r,
    }


def run_necessity(circuit: list[dict]) -> list[dict]:
    """Suppress each circuit feature one at a time."""
    print(f"\n[NECESSITY] {len(circuit)} individual ablations at strength {STRENGTH_SUPPRESS}")
    out = []
    for i, feat in enumerate(circuit, 1):
        features = [make_feature(feat["layer"], feat["index"], STRENGTH_SUPPRESS)]
        r = steer(PROMPT, features)
        disrupted = not predicts_target(r, EXPECTED_TOKEN)
        rec = {
            "experiment": "necessity_individual",
            "feature": feat["name"],
            "layer": feat["layer"],
            "index": feat["index"],
            "avg_influence": feat["avg_influence"],
            "avg_activation": feat["avg_activation"],
            "total_appearances": feat["total_appearances"],
            "disrupted": disrupted,
            "result": r,
        }
        out.append(rec)
        marker = "DISRUPT" if disrupted else "       "
        print(f"  [{i:>3}/{len(circuit)}] {feat['name']:<14}  steered={r.get('first_steered_token','?'):<12}  {marker}")
        time.sleep(DELAY)
    n_disrupt = sum(1 for r in out if r["disrupted"])
    print(f"  -> {n_disrupt}/{len(out)} features individually disrupt the prediction.")
    return out


def run_full_suppression(circuit: list[dict]) -> dict:
    """Suppress all circuit features simultaneously."""
    print(f"\n[FULL SUPPRESSION] All {len(circuit)} features at strength {STRENGTH_SUPPRESS}")
    features = [make_feature(f["layer"], f["index"], STRENGTH_SUPPRESS) for f in circuit]
    r = steer(PROMPT, features)
    disrupted = not predicts_target(r, EXPECTED_TOKEN)
    print(f"  Default: {r.get('default_text', '?')!r}")
    print(f"  Steered: {r.get('steered_text', '?')!r}")
    print(f"  Disrupted: {'YES' if disrupted else 'no'}")
    return {
        "experiment": "necessity_full",
        "n_features": len(circuit),
        "disrupted": disrupted,
        "result": r,
    }


def run_sufficiency(circuit: list[dict]) -> list[dict]:
    """Boost the top-1 hub on altered prompts."""
    hub = max(circuit, key=lambda f: f["avg_influence"])
    print(f"\n[SUFFICIENCY] Boost hub {hub['name']} (avg_influence={hub['avg_influence']:.4f}) "
          f"at strength {STRENGTH_BOOST}")
    out = []
    for prompt, desc in SUFFICIENCY_PROMPTS:
        # Baseline for the altered prompt
        r_base = steer(prompt, [])
        time.sleep(DELAY)

        # Boost hub
        features = [make_feature(hub["layer"], hub["index"], STRENGTH_BOOST)]
        r_boost = steer(prompt, features)
        induced = EXPECTED_TOKEN.lower() in r_boost.get("steered_text", "").lower()
        rec = {
            "experiment": "sufficiency_hub_boost",
            "prompt": prompt,
            "description": desc,
            "hub": hub["name"],
            "baseline_text": r_base.get("default_text", "?"),
            "boosted_text": r_boost.get("steered_text", "?"),
            "induced_target": induced,
            "result_baseline": r_base,
            "result_boost": r_boost,
        }
        out.append(rec)
        print(f"  '{prompt[:50]}' ({desc})")
        print(f"    Baseline: {r_base.get('default_text', '?')!r}")
        print(f"    Boosted:  {r_boost.get('steered_text', '?')!r}")
        print(f"    Induces '{EXPECTED_TOKEN}': {'YES' if induced else 'no'}")
        time.sleep(DELAY)
    return out


def run_specificity(non_circuit: list[dict]) -> list[dict]:
    """Suppress non-circuit features sampled from the Berlin graph."""
    print(f"\n[SPECIFICITY] {len(non_circuit)} non-circuit ablations at strength {STRENGTH_SUPPRESS}")
    out = []
    for i, feat in enumerate(non_circuit, 1):
        features = [make_feature(feat["layer"], feat["index"], STRENGTH_SUPPRESS)]
        r = steer(PROMPT, features)
        disrupted = not predicts_target(r, EXPECTED_TOKEN)
        rec = {
            "experiment": "specificity_individual",
            "layer": feat["layer"],
            "index": feat["index"],
            "activation": feat["activation"],
            "influence": feat["influence"],
            "disrupted": disrupted,
            "result": r,
        }
        out.append(rec)
        marker = "DISRUPT" if disrupted else "       "
        print(f"  [{i:>2}/{len(non_circuit)}] L{feat['layer']:>2}_{feat['index']:<6} "
              f"act={feat['activation']:6.2f} infl={feat['influence']:.3f}  "
              f"steered={r.get('first_steered_token','?'):<12}  {marker}")
        time.sleep(DELAY)
    n_disrupt = sum(1 for r in out if r["disrupted"])
    print(f"  -> {n_disrupt}/{len(out)} non-circuit features disrupt the prediction.")
    return out


# --- Main ----------------------------------------------------------

def main():
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Loading 180-feature circuit from {CIRCUIT_JSON}")
    circuit = load_circuit_features(CIRCUIT_JSON)
    print(f"  -> Loaded {len(circuit)} features.")

    circuit_set = {(f["layer"], f["index"]) for f in circuit}
    print(f"Sampling {N_SPECIFICITY_SAMPLES} non-circuit features from {BERLIN_GRAPH.name}")
    non_circuit = sample_non_circuit_features(BERLIN_GRAPH, circuit_set,
                                              N_SPECIFICITY_SAMPLES, SEED)
    print(f"  -> Sampled {len(non_circuit)} (activation range "
          f"{min(f['activation'] for f in non_circuit):.2f}–"
          f"{max(f['activation'] for f in non_circuit):.2f}).")

    results = {
        "metadata": {
            "prompt": PROMPT,
            "expected_token": EXPECTED_TOKEN,
            "circuit_size": len(circuit),
            "n_specificity_samples": len(non_circuit),
            "strength_suppress": STRENGTH_SUPPRESS,
            "strength_boost": STRENGTH_BOOST,
            "seed": SEED,
            "circuit_source": str(CIRCUIT_JSON),
            "specificity_source": str(BERLIN_GRAPH),
        }
    }

    # Stage 0: baseline
    results["baseline"] = run_baseline()
    save_results(results)
    time.sleep(DELAY)

    # Stage 1: necessity (individual)
    results["necessity_individual"] = run_necessity(circuit)
    save_results(results)

    # Stage 2: full circuit suppression
    results["necessity_full"] = run_full_suppression(circuit)
    save_results(results)
    time.sleep(DELAY)

    # Stage 3: sufficiency
    results["sufficiency"] = run_sufficiency(circuit)
    save_results(results)

    # Stage 4: specificity
    results["specificity"] = run_specificity(non_circuit)
    save_results(results)

    # --- Final summary ---
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    n_nec = sum(1 for r in results["necessity_individual"] if r["disrupted"])
    print(f"Necessity (individual):   {n_nec}/{len(circuit)} features disrupt prediction")
    print(f"Necessity (full):         {'DISRUPTED' if results['necessity_full']['disrupted'] else 'intact'}")
    n_suf = sum(1 for r in results["sufficiency"] if r["induced_target"])
    print(f"Sufficiency:              hub boost induces '{EXPECTED_TOKEN}' on "
          f"{n_suf}/{len(results['sufficiency'])} altered prompts")
    n_spec = sum(1 for r in results["specificity"] if r["disrupted"])
    print(f"Specificity:              {n_spec}/{len(non_circuit)} non-circuit features disrupt "
          f"({'PASS' if n_spec == 0 else 'PARTIAL' if n_spec < len(non_circuit) // 2 else 'FAIL'})")
    print(f"\nFull results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
