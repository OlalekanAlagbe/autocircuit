#!/usr/bin/env python3
"""Test collective Phase 2 suppression and all-Phase 1+2 suppression."""

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

DELAY = 2.5
OUTPUT = Path(__file__).parent / "collective_steering_results.json"

PROMPTS = {
    "analog_berlin": {"prompt": "Paris is to France as Berlin is to", "expected": "Germany"},
    "analog_rome": {"prompt": "Paris is to France as Rome is to", "expected": "Italy"},
    "analog_tokyo": {"prompt": "Paris is to France as Tokyo is to", "expected": "Japan"},
    "analog_teacher": {"prompt": "Doctor is to hospital as teacher is to", "expected": "school"},
    "analog_bird": {"prompt": "Fish is to water as bird is to", "expected": "air"},
}

PHASE1 = [
    {"layer": 0, "index": 11651},  # "the word 'to'"
    {"layer": 1, "index": 11356},  # "'to' followed by a verb"
    {"layer": 2, "index": 11475},  # "the word 'refers'"
    {"layer": 4, "index": 10752},  # "'to be' preceded by 'to'"
    {"layer": 5, "index": 9672},   # "the phrase 'it is to'"
]

PHASE2 = [
    {"layer": 5, "index": 5793},   # "analogies"
    {"layer": 5, "index": 2141},   # "comparisons of public figures"
    {"layer": 8, "index": 13766},  # "analogies or comparisons"
    {"layer": 9, "index": 13344},  # "comparison between two things"
]

PHASE3 = [
    {"layer": 13, "index": 10969}, # "comparisons between disciplines"
]

EXPERIMENTS = {
    "suppress_all_phase2": PHASE2,
    "suppress_all_phase1": PHASE1,
    "suppress_all_phase1_and_phase2": PHASE1 + PHASE2,
    "suppress_all_phase1_phase2_phase3": PHASE1 + PHASE2 + PHASE3,
}

def main():
    load_api_key()
    results = {}

    for pname, pdata in PROMPTS.items():
        results[pname] = {}
        print(f"\n{'='*50}")
        print(f"{pname}: {pdata['prompt']} → {pdata['expected']}")

        for ename, features_list in EXPERIMENTS.items():
            feats = [make_feature(f["layer"], f["index"], -20) for f in features_list]
            print(f"  {ename} ({len(feats)} features)...", end=" ", flush=True)
            r = steer(pdata["prompt"], feats)
            tok = r.get("first_steered_token", "?").strip()
            default_tok = r.get("default_text", "")[:50]
            steered_txt = r.get("steered_text", "")[:50]
            disrupted = tok.lower() != pdata["expected"].lower()
            print(f"→ '{tok}' {'DISRUPTED' if disrupted else 'ok'}")
            results[pname][ename] = {
                "first_token": tok, "disrupted": disrupted,
                "steered_text": steered_txt, "default_text": default_tok,
                "n_features": len(feats),
            }
            time.sleep(DELAY)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for ename in EXPERIMENTS:
        disrupted = [p for p, r in results.items() if r[ename]["disrupted"]]
        print(f"{ename}: {len(disrupted)}/5 disrupted — {disrupted}")

if __name__ == "__main__":
    main()
