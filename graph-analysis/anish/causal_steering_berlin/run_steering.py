#!/usr/bin/env python3
"""
Experiment 6: Causal Validation of PR #20 Circuits (Olalekan Alagbe)

Runs necessity, sufficiency, and specificity steering tests on circuits
discovered in KKrampis/autocircuit PR #20 across three task categories:
  - Analogical reasoning (Paris→Germany, Cairo→Kenya, Puppy→cat)
  - Factual recall (Water→oxygen, Napoleon→Elba)
  - Linguistic reasoning (hot→cold, maps→atlas, writes books→author)

Backbone features are extracted from per-prompt causal path analysis in
the PR's circuit papers. Each backbone node appears on at least one of
the top-5 causal paths traced from input to logit.

Methodology matches experiments/exp2_steering and exp4_scale/run_steering.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

DELAY = 2.5  # seconds between API calls (be gentle)
OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Circuit definitions — backbone features from PR #20 causal paths
# ---------------------------------------------------------------------------

CIRCUITS = {
    # === ANALOGICAL REASONING ===
    "analogical_paris_germany": {
        "prompt": "Paris is to France as Berlin is to",
        "expected_token": "Germany",
        "category": "analogical",
        "confidence": 0.973,
        "backbone": {
            # Path 1: L21→L22→L25→logit (weight=306, strongest path)
            "L21_science_hub": {
                "layer": 21, "index": 4827,
                "role": "Strongest path entry — scientific/medical research feature (act=45.0, edge +198.0)"
            },
            "L22_relay": {
                "layer": 22, "index": 15670,
                "role": "Path 1 relay (edge -1.36 to L25)"
            },
            "L25_output_driver_a": {
                "layer": 25, "index": 4717,
                "role": "Final amplifier to logit (edge +1.14)"
            },
            # Paths 2-4: L16/L17→L19→L21→L25→logit (shared late cascade)
            "L16_location_encoder": {
                "layer": 16, "index": 6491,
                "role": "Location/direction feature, path 2 entry (edge -0.74)"
            },
            "L17_relay": {
                "layer": 17, "index": 14546,
                "role": "Mid-cascade relay (edge +4.55)"
            },
            "L19_relay": {
                "layer": 19, "index": 5773,
                "role": "Late relay (edge +2.97)"
            },
            "L21_integrator": {
                "layer": 21, "index": 7482,
                "role": "Integration hub on paths 2-4 (edge +8.49 to L25)"
            },
            "L25_output_driver_b": {
                "layer": 25, "index": 2725,
                "role": "Secondary output driver (edge -2.09 to logit)"
            },
            # Path 5: L19→L24→L25→logit
            "L19_relation_applier": {
                "layer": 19, "index": 855,
                "role": "Relation application node (edge +13.50)"
            },
        },
        "sufficiency_prompts": [
            ("Cairo is to Egypt as Nairobi is to", "different analogy same structure"),
            ("Madrid is to Spain as Berlin is to", "different source pair"),
            ("The weather today is really", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 6, "index": 3335, "desc": "L6 'difficulty/challenges' (path 5 entry, low weight)"},
            {"layer": 13, "index": 4435, "desc": "L13 'opera-related terms' (not on Paris paths)"},
        ],
    },

    "analogical_cairo_kenya": {
        "prompt": "Cairo is to Egypt as Nairobi is to",
        "expected_token": "Kenya",
        "category": "analogical",
        "confidence": 0.963,
        "backbone": {
            # Path 1: L13→L15→L18→L24→L25→logit (weight=86)
            "L15_relay": {
                "layer": 15, "index": 15954,
                "role": "Path 1 mid relay (edge +6.74)"
            },
            "L18_legal_docs": {
                "layer": 18, "index": 13586,
                "role": "Legal docs feature on path 1 (edge +1.53)"
            },
            "L24_suppressor": {
                "layer": 24, "index": 1260,
                "role": "Competitor suppression (edge -8.50)"
            },
            "L25_output_a": {
                "layer": 25, "index": 15920,
                "role": "Output driver path 1 (edge -0.44 to logit)"
            },
            # Path 2: L5→L20→L23→L24→L25→logit (weight=67)
            "L20_relay": {
                "layer": 20, "index": 15360,
                "role": "Path 2 relay (edge +7.79)"
            },
            "L23_relay": {
                "layer": 23, "index": 13914,
                "role": "Late relay (edge +10.65)"
            },
            "L25_output_b": {
                "layer": 25, "index": 286,
                "role": "Secondary output driver (edge -1.07 to logit)"
            },
            # Path 5: L15→L20→L21→L25→logit (excitatory, weight=35)
            "L15_geography": {
                "layer": 15, "index": 5355,
                "role": "Geographical locations feature (edge +0.72)"
            },
            "L21_convergence": {
                "layer": 21, "index": 5251,
                "role": "Convergence hub (edge +9.67)"
            },
        },
        "sufficiency_prompts": [
            ("Paris is to France as Berlin is to", "different analogy same task"),
            ("Lagos is to Nigeria as Nairobi is to", "different source pair"),
            ("The temperature outside was around", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 5, "index": 5500, "desc": "L5 'profanity and comparisons' (weak path 4 entry)"},
            {"layer": 0, "index": 8, "desc": "L0 general 'are/research' feature"},
        ],
    },

    "analogical_puppy_cat": {
        "prompt": "Puppy is to dog as kitten is to",
        "expected_token": "cat",
        "category": "analogical",
        "confidence": 0.756,
        "backbone": {
            # Path 1: L9→L14→L20→L22→L25→logit (weight=552, very strong)
            "L14_relay": {
                "layer": 14, "index": 3704,
                "role": "Path 1 relay (edge +11.89)"
            },
            "L20_integration": {
                "layer": 20, "index": 3094,
                "role": "Integration hub (edge -12.76)"
            },
            "L22_relay": {
                "layer": 22, "index": 15670,
                "role": "Late relay shared with Paris circuit (edge -2.45)"
            },
            "L25_amplifier": {
                "layer": 25, "index": 4717,
                "role": "Final amplifier to logit (shared with Paris circuit)"
            },
        },
        "sufficiency_prompts": [
            ("Fish is to water as bird is to", "different analogy same structure"),
            ("Lamb is to sheep as kitten is to", "different source pair"),
            ("The weather today is really", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 9, "index": 2909, "desc": "L9 'formulas/ratios' (path 1 entry but general feature)"},
        ],
    },

    # === FACTUAL RECALL ===
    "factual_water_oxygen": {
        "prompt": "Water is composed of hydrogen and",
        "expected_token": "oxygen",
        "category": "factual_recall",
        "confidence": 0.978,
        "backbone": {
            # Paths 1-2: L19→L22→L24→L25→logit (dominant cascade, wt=357+263)
            "L19_hub": {
                "layer": 19, "index": 4647,
                "role": "Primary hub — feeds the dominant cascade (edge +5.81)"
            },
            "L22_bottleneck": {
                "layer": 22, "index": 3598,
                "role": "Circuit bottleneck (edge +43.25 — massive weight)"
            },
            "L24_suppressor": {
                "layer": 24, "index": 10522,
                "role": "Inhibitory suppressor (edge -13.13)"
            },
            "L25_gate": {
                "layer": 25, "index": 8017,
                "role": "Final gate to logit (edge -0.55)"
            },
            # Path 3: L0→L18→L19→logit (excitatory, wt=132)
            "L18_relay": {
                "layer": 18, "index": 2175,
                "role": "Excitatory relay (edge +17.50)"
            },
            "L19_secondary": {
                "layer": 19, "index": 13684,
                "role": "Secondary L19 output (edge +1.27 to logit)"
            },
            # Path 4: L9→L22→L23→L25→logit (chemistry path)
            "L9_chemistry": {
                "layer": 9, "index": 16135,
                "role": "Chemistry/chemical elements feature (edge +0.67)"
            },
            "L22_secondary": {
                "layer": 22, "index": 11135,
                "role": "Secondary L22 node (edge +5.11)"
            },
            "L25_amplifier": {
                "layer": 25, "index": 4717,
                "role": "Final amplifier shared across circuits"
            },
        },
        "sufficiency_prompts": [
            ("The chemical formula for methane is CH", "different molecule same task"),
            ("Water boils at a temperature of", "same entity different relation"),
            ("The weather today is quite", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 6, "index": 2238, "desc": "L6 'passive voice/designed systems' (path 1 entry but general)"},
            {"layer": 7, "index": 4310, "desc": "L7 'food/math' feature (path 5, low weight)"},
        ],
    },

    "factual_napoleon_elba": {
        "prompt": "Napoleon was exiled to the island of",
        "expected_token": "Elba",
        "category": "factual_recall",
        "confidence": 0.709,
        "backbone": {
            # Path 2: L4→L5→L21→L25→logit (mixed, wt=0.53)
            "L5_relay": {
                "layer": 5, "index": 5682,
                "role": "Early-mid relay (edge +0.86)"
            },
            "L21_convergence": {
                "layer": 21, "index": 2655,
                "role": "Late convergence hub (edge +2.17)"
            },
            "L25_output": {
                "layer": 25, "index": 15920,
                "role": "Output driver (edge -0.52 to logit)"
            },
            # Key late-layer features from narrative (highest activations)
            "L18_island_encoder": {
                "layer": 18, "index": 13286,
                "role": "Island/park names encoder (act=51.48 — highest late-layer activation)"
            },
            "L17_empire": {
                "layer": 17, "index": 8783,
                "role": "Empires/global power (act=19.18)"
            },
        },
        "sufficiency_prompts": [
            ("After his defeat, the emperor was sent to the island of", "rephrased same fact"),
            ("The French general was exiled to", "paraphrased"),
            ("The weather in the morning was", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 6, "index": 526, "desc": "L6 abbreviations/acronyms (inhibitory path 3)"},
            {"layer": 4, "index": 6280, "desc": "L4 legal text/processes (path 2 entry)"},
        ],
    },

    # === LINGUISTIC REASONING ===
    "linguistic_hot_cold": {
        "prompt": "The opposite of hot is",
        "expected_token": "cold",
        "category": "linguistic",
        "confidence": 0.566,
        "backbone": {
            # Path 1: L11→L18→L20→L25→logit (dominant, wt=314)
            "L11_systems": {
                "layer": 11, "index": 4301,
                "role": "Path 1 entry — systems description (edge -0.86)"
            },
            "L18_relay": {
                "layer": 18, "index": 1296,
                "role": "Mid-late relay (edge +11.56)"
            },
            "L20_hub": {
                "layer": 20, "index": 6666,
                "role": "Central convergence hub — on paths 1,2,3,5 (edge +5.31)"
            },
            "L25_output": {
                "layer": 25, "index": 16258,
                "role": "Output driver — on all multi-hop paths (edge -5.95)"
            },
            # Path 2: L7→L12→L14→L20→L25→logit (wt=55)
            "L7_comparison": {
                "layer": 7, "index": 4526,
                "role": "Comparisons/symmetry/reversals feature (edge -1.90)"
            },
            "L12_relay": {
                "layer": 12, "index": 95,
                "role": "Mid relay (edge +1.17)"
            },
            "L14_relay": {
                "layer": 14, "index": 1354,
                "role": "Late-mid relay (edge +0.78)"
            },
        },
        "sufficiency_prompts": [
            ("The opposite of fast is", "different adjective"),
            ("The meaning of hot is", "different relation"),
            ("The weather today is", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 2, "index": 8212, "desc": "L2 'compare/contrast' (path 3 entry, low weight)"},
            {"layer": 0, "index": 902, "desc": "L0 'the word is' (path 5, very low weight)"},
        ],
    },

    "linguistic_maps_atlas": {
        "prompt": "A book of maps is called an",
        "expected_token": "atlas",
        "category": "linguistic",
        "confidence": 0.825,
        "backbone": {
            # Paths 1-2: L8→L18→L19→L24→L25→logit (shared gateway, wt=53+30)
            "L18_gateway": {
                "layer": 18, "index": 10679,
                "role": "Central gateway — on paths 1,2,4,5 (edge +22.62)"
            },
            "L19_relay": {
                "layer": 19, "index": 8729,
                "role": "Post-gateway relay (edge +2.21)"
            },
            "L24_suppressor": {
                "layer": 24, "index": 11962,
                "role": "Competitor suppression bottleneck (edge -6.89)"
            },
            "L25_gate": {
                "layer": 25, "index": 8267,
                "role": "Final gate to logit (edge -0.27)"
            },
            # Path 4-5 alternate entry via L10→L12→L18
            "L10_references": {
                "layer": 10, "index": 6804,
                "role": "References feature (act=38.22, edge +3.31)"
            },
            # Late-layer features from narrative
            "L21_visual_info": {
                "layer": 21, "index": 3848,
                "role": "Visual representations feature (act=17.10)"
            },
            "L16_word_origins": {
                "layer": 16, "index": 16329,
                "role": "Word origins / etymology feature (act=7.30)"
            },
        },
        "sufficiency_prompts": [
            ("A collection of maps is called an", "rephrased same fact"),
            ("A book of recipes is called a", "different content same structure"),
            ("The temperature outside was around", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 8, "index": 1494, "desc": "L8 coding/GUI (path 1 entry but general feature)"},
            {"layer": 8, "index": 13791, "desc": "L8 web dev/URLs (path 2 entry but general feature)"},
        ],
    },

    "linguistic_author": {
        "prompt": "A person who writes books is called an",
        "expected_token": "author",
        "category": "linguistic",
        "confidence": 0.836,
        "backbone": {
            # Path 1: L0→L18→L23→L24→L25→logit (wt=176, strongest)
            "L24_bottleneck": {
                "layer": 24, "index": 1633,
                "role": "Inhibitory bottleneck (edge -2.56)"
            },
            "L25_output": {
                "layer": 25, "index": 14935,
                "role": "Output driver (edge -2.76 to logit)"
            },
            # Path 2: E→L22→logit (excitatory, wt=66)
            "L22_gateway": {
                "layer": 22, "index": 7158,
                "role": "Direct excitatory gateway (edge +3.79 to logit)"
            },
            # Path 3: L18→L22→L25→logit (wt=29)
            "L18_writing_projects": {
                "layer": 18, "index": 1943,
                "role": "Writing projects feature — most semantically precise (edge -1.38)"
            },
            # Path 5: L9→L19→L21→L25→logit (excitatory, wt=11.5)
            "L19_relay": {
                "layer": 19, "index": 7101,
                "role": "Relay (edge +7.15)"
            },
            "L21_relay": {
                "layer": 21, "index": 13124,
                "role": "Late relay (edge +4.38)"
            },
            "L25_amplifier": {
                "layer": 25, "index": 4717,
                "role": "Shared amplifier (edge +0.75)"
            },
            # Key late-layer features from narrative
            "L22_literature": {
                "layer": 22, "index": 9109,
                "role": "Literature/writing feature (act=8.55)"
            },
            "L24_creative_works": {
                "layer": 24, "index": 1735,
                "role": "Creative endeavors/authors (act=9.63)"
            },
        },
        "sufficiency_prompts": [
            ("Someone who writes novels is called an", "rephrased same fact"),
            ("A person who paints pictures is called an", "different domain same structure"),
            ("The weather in the morning was", "irrelevant context"),
        ],
        "non_backbone": [
            {"layer": 10, "index": 1037, "desc": "L10 'capitalized words' (path 4 entry but general)"},
            {"layer": 9, "index": 10769, "desc": "L9 'people/categories' (path 5 entry but general)"},
        ],
    },
}

STRENGTH = -20


def run_steering_for_circuit(circuit_name, config):
    """Run necessity, sufficiency, and specificity tests for one circuit."""
    prompt = config["prompt"]
    expected = config["expected_token"]
    backbone = config["backbone"]

    print(f"\n{'='*70}")
    print(f"Circuit: {circuit_name}")
    print(f"  Prompt: \"{prompt}\" → \"{expected}\" (p={config['confidence']})")
    print(f"  Category: {config['category']}")
    print(f"  Backbone features: {len(backbone)}")
    print(f"{'='*70}")

    results = []

    # --- A. Individual suppression (necessity) ---
    print("\n--- A. Individual Suppression (Necessity) ---")
    for name, feat in backbone.items():
        print(f"\n  Suppressing {name} ({feat['role'][:60]}...)...")
        features = [make_feature(feat["layer"], feat["index"], STRENGTH)]
        r = steer(prompt, features)
        r["experiment"] = f"suppress_{name}"
        r["circuit"] = circuit_name
        r["category"] = config["category"]
        r["description"] = f"Suppress {name} ({feat['role']})"
        results.append(r)

        if "error" not in r:
            steered = r.get("steered_text", "")
            disrupted = expected.lower() not in steered.lower()
            print(f"    Default: {r.get('first_default_token','?')} | "
                  f"Steered: {r.get('first_steered_token','?')}")
            print(f"    {expected} disrupted: {'YES' if disrupted else 'no'}")
        else:
            print(f"    ERROR: {r['error']}")
        time.sleep(DELAY)

    # --- B. Full backbone suppression ---
    print("\n--- B. Full Backbone Suppression ---")
    all_features = [make_feature(f["layer"], f["index"], STRENGTH)
                    for f in backbone.values()]
    r = steer(prompt, all_features)
    r["experiment"] = "suppress_all_backbone"
    r["circuit"] = circuit_name
    r["category"] = config["category"]
    r["description"] = f"Suppress all {len(backbone)} backbone features"
    results.append(r)
    if "error" not in r:
        steered = r.get("steered_text", "")
        disrupted = expected.lower() not in steered.lower()
        print(f"  Default: {r.get('default_text', '?')}")
        print(f"  Steered: {steered}")
        print(f"  {expected} disrupted: {'YES' if disrupted else 'no'}")
    time.sleep(DELAY)

    # --- C. Hub boost on altered prompts (sufficiency) ---
    print("\n--- C. Hub Sufficiency ---")
    # Use the first backbone feature as the "hub" (strongest path entry)
    hub_name = list(backbone.keys())[0]
    hub = backbone[hub_name]

    for alt_prompt, desc in config.get("sufficiency_prompts", []):
        print(f"\n  Prompt: '{alt_prompt}' ({desc})")

        # Baseline
        r_base = steer(alt_prompt, [])
        base_text = "?"
        if "error" not in r_base:
            base_text = r_base.get("steered_text", r_base.get("default_text", "?"))
            print(f"    Baseline: {base_text}")
        time.sleep(DELAY)

        # Boost hub
        features = [make_feature(hub["layer"], hub["index"], 20)]
        r_boost = steer(alt_prompt, features)
        r_boost["experiment"] = f"boost_hub_{alt_prompt[:25].replace(' ','_')}"
        r_boost["circuit"] = circuit_name
        r_boost["category"] = config["category"]
        r_boost["description"] = f"Boost {hub_name} on: {desc}"
        r_boost["baseline_text"] = base_text
        results.append(r_boost)
        if "error" not in r_boost:
            boosted = r_boost.get("steered_text", "?")
            print(f"    Boosted: {boosted}")
            has_expected = expected.lower() in boosted.lower()
            print(f"    Contains '{expected}': {'YES' if has_expected else 'no'}")
        time.sleep(DELAY)

    # --- D. Non-backbone suppression (specificity) ---
    print("\n--- D. Non-Backbone Suppression (Specificity) ---")
    for feat in config.get("non_backbone", []):
        print(f"\n  Suppressing {feat['desc']}...")
        features = [make_feature(feat["layer"], feat["index"], STRENGTH)]
        r = steer(prompt, features)
        r["experiment"] = f"suppress_nonbackbone_L{feat['layer']}_{feat['index']}"
        r["circuit"] = circuit_name
        r["category"] = config["category"]
        r["description"] = f"Suppress non-backbone: {feat['desc']}"
        results.append(r)
        if "error" not in r:
            steered = r.get("steered_text", "")
            disrupted = expected.lower() not in steered.lower()
            print(f"    Default: {r.get('first_default_token','?')} | "
                  f"Steered: {r.get('first_steered_token','?')}")
            print(f"    {expected} disrupted: {'YES' if disrupted else 'no'}")
        else:
            print(f"    ERROR: {r['error']}")
        time.sleep(DELAY)

    return results


def main():
    results_path = OUTPUT_DIR / "steering_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results for: {list(all_results.keys())}")
    else:
        all_results = {}

    # Only run circuits not already completed
    to_run = {k: v for k, v in CIRCUITS.items() if k not in all_results}
    if not to_run:
        print("All circuits already have results. Delete steering_results.json to re-run.")
        return

    print(f"Running causal validation on {len(to_run)} circuits: {list(to_run.keys())}")
    print(f"Total circuits defined: {len(CIRCUITS)}")

    for circuit_name, config in to_run.items():
        try:
            results = run_steering_for_circuit(circuit_name, config)
            all_results[circuit_name] = results

            # Save after each circuit (crash-safe)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  [Saved results for {circuit_name}]")

        except Exception as e:
            print(f"\n  ERROR on {circuit_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for circuit_name, results in all_results.items():
        config = CIRCUITS[circuit_name]
        expected = config["expected_token"]
        print(f"\n{circuit_name} ({config['category']}):")
        print(f"  Prompt: \"{config['prompt']}\" → \"{expected}\"")

        necessity_disrupted = 0
        necessity_total = 0
        for r in results:
            exp = r.get("experiment", "")
            if exp.startswith("suppress_") and exp != "suppress_all_backbone" and \
               not exp.startswith("suppress_nonbackbone") and "error" not in r:
                steered = r.get("steered_text", "")
                disrupted = expected.lower() not in steered.lower()
                necessity_total += 1
                if disrupted:
                    necessity_disrupted += 1
                    print(f"    {exp}: DISRUPTED → {r.get('first_steered_token', '?')}")
                else:
                    print(f"    {exp}: intact")

        if necessity_total > 0:
            print(f"  Necessity: {necessity_disrupted}/{necessity_total} features disrupt prediction")

        # Full suppression
        for r in results:
            if r.get("experiment") == "suppress_all_backbone" and "error" not in r:
                steered = r.get("steered_text", "")
                disrupted = expected.lower() not in steered.lower()
                print(f"  Full suppression: {'DISRUPTED' if disrupted else 'intact'}")

        # Specificity
        spec_disrupted = 0
        spec_total = 0
        for r in results:
            if r.get("experiment", "").startswith("suppress_nonbackbone") and "error" not in r:
                steered = r.get("steered_text", "")
                disrupted = expected.lower() not in steered.lower()
                spec_total += 1
                if disrupted:
                    spec_disrupted += 1
        if spec_total > 0:
            print(f"  Specificity: {spec_disrupted}/{spec_total} non-backbone features disrupt "
                  f"({'PASS' if spec_disrupted == 0 else 'FAIL'})")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
