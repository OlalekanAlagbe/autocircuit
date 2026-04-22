#!/usr/bin/env python3
"""
Experiment 7: Causal Validation of Remaining 4 Analogy Prompts

Runs necessity, sufficiency, and specificity steering tests on the
4 analogy circuits not covered in exp6:
  - analog_rome: "Paris is to France as Rome is to" → Italy (p=0.974)
  - analog_tokyo: "Paris is to France as Tokyo is to" → Japan (p=0.990)
  - analog_teacher: "Doctor is to hospital as teacher is to" → school (p=0.486)
  - analog_bird: "Fish is to water as bird is to" → air (p=0.117)

Backbone features extracted from Neuronpedia attribution graphs via
greedy causal path tracing (top-5 forward + top-5 backward paths).
We focus on late-layer (L17+) features for individual necessity testing
since those sit in the direct causal chain to the logit.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

DELAY = 2.5
OUTPUT_DIR = Path(__file__).parent

# Late-layer backbone features (L17+) for each circuit, plus key features
CIRCUITS = {
    "analog_rome": {
        "prompt": "Paris is to France as Rome is to",
        "expected_token": "Italy",
        "category": "analogical",
        "confidence": 0.974,
        "backbone": {
            # Backward-path features (directly traced from logit)
            "L20_relay": {"layer": 20, "index": 15360, "role": "Backward path from logit, relay"},
            "L24_gate": {"layer": 24, "index": 16122, "role": "Backward path from logit, late gate"},
            "L25_output_b": {"layer": 25, "index": 286, "role": "Backward path from logit, output driver"},
            # Other key late features
            "L25_amplifier": {"layer": 25, "index": 4717, "role": "Final amplifier (shared, act=265.2)"},
            "L25_output_c": {"layer": 25, "index": 10521, "role": "Tertiary output driver"},
            "L17_relay": {"layer": 17, "index": 14546, "role": "Mid-cascade relay (shared with Berlin)"},
            "L22_relay_a": {"layer": 22, "index": 12202, "role": "Late relay"},
            "L22_relay_b": {"layer": 22, "index": 14727, "role": "Late relay"},
            "L23_relay": {"layer": 23, "index": 5917, "role": "Late relay"},
            "L24_gate_b": {"layer": 24, "index": 13277, "role": "Secondary gate"},
        },
        "sufficiency_prompts": [
            "Paris is to France as Tokyo is to",
            "Madrid is to Spain as Rome is to",
        ],
        "non_backbone": [
            {"layer": 6, "index": 2267, "desc": "formal text/code (shared recurring)"},
            {"layer": 4, "index": 14857, "desc": "code snippets (shared recurring)"},
        ],
    },
    "analog_tokyo": {
        "prompt": "Paris is to France as Tokyo is to",
        "expected_token": "Japan",
        "category": "analogical",
        "confidence": 0.990,
        "backbone": {
            "L20_relay": {"layer": 20, "index": 15360, "role": "Backward path from logit, relay"},
            "L25_output_a": {"layer": 25, "index": 286, "role": "Backward path from logit, output driver"},
            "L25_output_b": {"layer": 25, "index": 12223, "role": "Backward path from logit, output driver"},
            "L17_relay": {"layer": 17, "index": 14546, "role": "Mid-cascade relay (shared)"},
            "L23_relay_a": {"layer": 23, "index": 850, "role": "Late relay"},
            "L23_relay_b": {"layer": 23, "index": 13914, "role": "Late relay (necessary in Cairo circuit)"},
            "L24_gate": {"layer": 24, "index": 13277, "role": "Late gate (shared with Rome)"},
            "L25_output_c": {"layer": 25, "index": 10152, "role": "Tertiary output"},
            "L20_hub": {"layer": 20, "index": 6648, "role": "L20 convergence hub"},
            "L21_relay": {"layer": 21, "index": 7764, "role": "Late integration"},
        },
        "sufficiency_prompts": [
            "Paris is to France as Rome is to",
            "Beijing is to China as Tokyo is to",
        ],
        "non_backbone": [
            {"layer": 6, "index": 2267, "desc": "formal text/code (shared recurring)"},
            {"layer": 3, "index": 10018, "desc": "early structural feature"},
        ],
    },
    "analog_teacher": {
        "prompt": "Doctor is to hospital as teacher is to",
        "expected_token": "school",
        "category": "analogical",
        "confidence": 0.486,
        "backbone": {
            # Backward-path features
            "L0_embed": {"layer": 0, "index": 17, "role": "Backward path, embedding-level"},
            "L18_gateway": {"layer": 18, "index": 6532, "role": "Backward path, mid-late gateway"},
            "L20_hub": {"layer": 20, "index": 6179, "role": "Backward path, convergence hub"},
            "L25_output": {"layer": 25, "index": 4975, "role": "Backward path, output driver"},
            # Other key late features
            "L25_amplifier": {"layer": 25, "index": 4717, "role": "Final amplifier (shared, act=135.6)"},
            "L22_relay": {"layer": 22, "index": 15670, "role": "Late relay (shared)"},
            "L18_relay_b": {"layer": 18, "index": 11952, "role": "Mid-late relay"},
            "L18_relay_c": {"layer": 18, "index": 13586, "role": "Legal docs feature"},
            "L21_convergence": {"layer": 21, "index": 2655, "role": "Late convergence hub"},
            "L24_gate": {"layer": 24, "index": 15259, "role": "Late suppression gate"},
        },
        "sufficiency_prompts": [
            "Nurse is to hospital as teacher is to",
            "Doctor is to hospital as chef is to",
        ],
        "non_backbone": [
            {"layer": 4, "index": 14857, "desc": "code snippets (shared recurring)"},
            {"layer": 8, "index": 13766, "desc": "analogies or comparisons"},
        ],
    },
    "analog_bird": {
        "prompt": "Fish is to water as bird is to",
        "expected_token": "air",
        "category": "analogical",
        "confidence": 0.117,
        "backbone": {
            # Backward-path features
            "L22_backward": {"layer": 22, "index": 4252, "role": "Backward path from logit"},
            "L24_backward": {"layer": 24, "index": 8106, "role": "Backward path from logit, late gate"},
            # Other key late features
            "L25_amplifier": {"layer": 25, "index": 4717, "role": "Final amplifier (shared, act=122.3)"},
            "L25_output": {"layer": 25, "index": 11801, "role": "Output driver"},
            "L22_relay_a": {"layer": 22, "index": 15670, "role": "Late relay (shared)"},
            "L22_relay_b": {"layer": 22, "index": 14727, "role": "Late relay"},
            "L22_relay_c": {"layer": 22, "index": 13619, "role": "Late relay"},
            "L24_gate_a": {"layer": 24, "index": 4383, "role": "Suppression gate"},
            "L24_gate_b": {"layer": 24, "index": 12559, "role": "Suppression gate"},
            "L20_hub": {"layer": 20, "index": 3094, "role": "Integration hub (shared with Puppy)"},
        },
        "sufficiency_prompts": [
            "Cat is to land as bird is to",
            "Fish is to water as eagle is to",
        ],
        "non_backbone": [
            {"layer": 6, "index": 2267, "desc": "formal text/code (shared recurring)"},
            {"layer": 5, "index": 5793, "desc": "analogies feature"},
        ],
    },
}


def run_steering_for_circuit(name, circuit):
    """Run all 4 experiment types for one circuit."""
    results = []
    prompt = circuit["prompt"]
    expected = circuit["expected_token"]
    backbone = circuit["backbone"]

    print(f"\n{'='*60}")
    print(f"Circuit: {name} ({prompt} → {expected}, p={circuit['confidence']})")
    print(f"{'='*60}")

    # A) Individual necessity — suppress each backbone feature
    for feat_name, feat in backbone.items():
        label = f"suppress_{feat_name}"
        print(f"  [{label}] L{feat['layer']}/{feat['index']}...", end=" ", flush=True)
        features = [make_feature(feat["layer"], feat["index"], -20)]
        result = steer(prompt, features)
        result["experiment"] = label
        result["feature"] = feat_name
        result["role"] = feat["role"]
        results.append(result)
        tok = result.get("first_steered_token", "?")
        disrupted = tok.strip().lower() != expected.lower()
        print(f"→ '{tok}' {'DISRUPTED' if disrupted else 'ok'}")
        time.sleep(DELAY)

    # B) Full late-layer backbone suppression
    print(f"  [full_backbone_suppress] all {len(backbone)} features...", end=" ", flush=True)
    features = [make_feature(f["layer"], f["index"], -20) for f in backbone.values()]
    result = steer(prompt, features)
    result["experiment"] = "full_backbone_suppress"
    results.append(result)
    print(f"→ '{result.get('steered_text', '?')[:40]}'")
    time.sleep(DELAY)

    # C) Sufficiency — boost highest-activation late hub
    # Pick the feature with highest activation from backward paths
    backward_feats = {k: v for k, v in backbone.items()
                      if "backward" in v.get("role", "").lower() or "output" in v.get("role", "").lower()}
    if not backward_feats:
        backward_feats = dict(list(backbone.items())[:1])
    hub_name = list(backward_feats.keys())[0]
    hub = backward_feats[hub_name]

    for alt_prompt, _ in [(p, "") for p in circuit.get("sufficiency_prompts", [])]:
        label = f"sufficiency_{hub_name}"
        print(f"  [{label}] boost on '{alt_prompt[:40]}'...", end=" ", flush=True)
        features = [make_feature(hub["layer"], hub["index"], 20)]
        result = steer(alt_prompt, features)
        result["experiment"] = label
        result["altered_prompt"] = alt_prompt
        results.append(result)
        tok = result.get("first_steered_token", "?")
        induced = tok.strip().lower() == expected.lower()
        print(f"→ '{tok}' {'INDUCED' if induced else 'no'}")
        time.sleep(DELAY)

    # D) Specificity — suppress non-backbone features
    for nb in circuit.get("non_backbone", []):
        label = f"specificity_L{nb['layer']}_{nb['index']}"
        print(f"  [{label}] {nb['desc']}...", end=" ", flush=True)
        features = [make_feature(nb["layer"], nb["index"], -20)]
        result = steer(prompt, features)
        result["experiment"] = label
        result["desc"] = nb["desc"]
        results.append(result)
        tok = result.get("first_steered_token", "?")
        disrupted = tok.strip().lower() != expected.lower()
        print(f"→ '{tok}' {'DISRUPTED!' if disrupted else 'pass'}")
        time.sleep(DELAY)

    return results


def main():
    load_api_key()
    results_file = OUTPUT_DIR / "steering_results.json"

    # Load existing results if any (crash-safe)
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for name, circuit in CIRCUITS.items():
        if name in all_results:
            print(f"\nSkipping {name} (already done)")
            continue
        results = run_steering_for_circuit(name, circuit)
        all_results[name] = results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved {len(results)} results for {name}")

    print(f"\nDone! {len(all_results)} circuits complete.")


if __name__ == "__main__":
    main()
