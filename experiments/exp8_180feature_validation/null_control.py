#!/usr/bin/env python3
"""Null control for full-suppression test.

Question: is the " to to to to to" collapse under full 180-feature suppression
specific to the recurring circuit, or would suppressing any 180 features at
the same strength produce a similar collapse via norm-injection?

Procedure: sample 180 features from analog_berlin's attribution graph that are
NOT in the 180-feature circuit, ablate all of them simultaneously at strength
-20, compare the steered output to the circuit-suppression output.

Single API call. Result saved to null_control.json.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import make_feature, steer

from run_validation import sample_non_circuit_features, parse_node_id

CIRCUIT_JSON = Path("/Users/anish/Downloads/analogical_circuit_180features.json")
BERLIN_GRAPH = Path("/Users/anish/Desktop/aisc_project/experiments/exp4_scale/circuits/analogy_graph.json")
OUTPUT = Path(__file__).parent / "null_control.json"

PROMPT = "Paris is to France as Berlin is to"
STRENGTH = -20
N = 180
SEED = 7  # different from specificity seed (42) so we don't reuse those 20


def main():
    with open(CIRCUIT_JSON) as f:
        d = json.load(f)
    circuit_set = set()
    for feat in d["features"]:
        layer, idx, _ = parse_node_id(feat["node_ids"][0])
        circuit_set.add((layer, idx))
    print(f"Circuit: {len(circuit_set)} (layer, idx) pairs", flush=True)

    sampled = sample_non_circuit_features(BERLIN_GRAPH, circuit_set, n=N, seed=SEED)
    print(f"Sampled {len(sampled)} non-circuit features for ablation", flush=True)
    if len(sampled) < N:
        print(f"WARN: only {len(sampled)} non-circuit candidates available (asked {N})", flush=True)

    features = [make_feature(f["layer"], f["index"], STRENGTH) for f in sampled]
    print(f"Ablating {len(features)} non-circuit features at strength {STRENGTH}...", flush=True)
    t0 = time.time()
    r = steer(PROMPT, features)
    dt = time.time() - t0
    print(f"API call took {dt:.1f}s", flush=True)

    if "error" in r:
        print(f"ERROR: {r['error']}", flush=True)
    else:
        print(f"Default text:  {r.get('default_text', '?')!r}", flush=True)
        print(f"Steered text:  {r.get('steered_text', '?')!r}", flush=True)
        print(f"First default token: {r.get('first_default_token', '?')!r} "
              f"(logprob {r.get('first_default_logprob')})", flush=True)
        print(f"First steered token: {r.get('first_steered_token', '?')!r} "
              f"(logprob {r.get('first_steered_logprob')})", flush=True)

    out = {
        "prompt": PROMPT,
        "strength": STRENGTH,
        "n_features": len(features),
        "seed": SEED,
        "features": [{"layer": f["layer"], "index": f["index"],
                      "activation": f["activation"], "influence": f["influence"]}
                     for f in sampled],
        "result": r,
        "elapsed_s": dt,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
