#!/usr/bin/env python3
"""Test Phase 1 and Phase 2 features for necessity across all 5 prompts."""

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

DELAY = 2.5
OUTPUT = Path(__file__).parent / "phase_steering_results.json"

PROMPTS = {
    "analog_berlin": {"prompt": "Paris is to France as Berlin is to", "expected": "Germany"},
    "analog_rome": {"prompt": "Paris is to France as Rome is to", "expected": "Italy"},
    "analog_tokyo": {"prompt": "Paris is to France as Tokyo is to", "expected": "Japan"},
    "analog_teacher": {"prompt": "Doctor is to hospital as teacher is to", "expected": "school"},
    "analog_bird": {"prompt": "Fish is to water as bird is to", "expected": "air"},
}

# The paper's key Phase 1, Phase 2, and formal-text features
FEATURES = {
    # Phase 1 - Structural Template Parsing
    "P1_L0_11651": {"layer": 0, "index": 11651, "label": "the word 'to'", "phase": 1},
    "P1_L1_11356": {"layer": 1, "index": 11356, "label": "'to' followed by a verb", "phase": 1},
    "P1_L4_10752": {"layer": 4, "index": 10752, "label": "'to be' preceded by 'to'", "phase": 1},
    "P1_L5_9672": {"layer": 5, "index": 9672, "label": "the phrase 'it is to'", "phase": 1},
    # Phase 2 - Analogy Recognition Hub
    "P2_L5_5793": {"layer": 5, "index": 5793, "label": "analogies", "phase": 2},
    "P2_L5_2141": {"layer": 5, "index": 2141, "label": "comparisons of public figures", "phase": 2},
    "P2_L8_13766": {"layer": 8, "index": 13766, "label": "analogies or comparisons", "phase": 2},
    "P2_L9_13344": {"layer": 9, "index": 13344, "label": "comparison between two things", "phase": 2},
    # Phase 3 - Relational Integration
    "P3_L13_10969": {"layer": 13, "index": 10969, "label": "comparisons between disciplines", "phase": 3},
}

def main():
    load_api_key()
    results = {}

    for pname, pdata in PROMPTS.items():
        results[pname] = []
        print(f"\n{'='*50}")
        print(f"{pname}: {pdata['prompt']} → {pdata['expected']}")
        print(f"{'='*50}")

        for fname, fdata in FEATURES.items():
            print(f"  suppress {fname} (L{fdata['layer']}/{fdata['index']} \"{fdata['label']}\")...", end=" ", flush=True)
            features = [make_feature(fdata["layer"], fdata["index"], -20)]
            r = steer(pdata["prompt"], features)
            tok = r.get("first_steered_token", "?").strip()
            disrupted = tok.lower() != pdata["expected"].lower()
            print(f"→ '{tok}' {'DISRUPTED' if disrupted else 'ok'}")
            results[pname].append({
                "feature": fname, "layer": fdata["layer"], "index": fdata["index"],
                "label": fdata["label"], "phase": fdata["phase"],
                "first_token": tok, "disrupted": disrupted,
            })
            time.sleep(DELAY)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for pname, res_list in results.items():
        disrupted = [r for r in res_list if r["disrupted"]]
        print(f"{pname}: {len(disrupted)}/{len(res_list)} disrupted")
        for r in disrupted:
            print(f"  {r['feature']} (Phase {r['phase']}, \"{r['label']}\") → '{r['first_token']}'")

if __name__ == "__main__":
    main()
