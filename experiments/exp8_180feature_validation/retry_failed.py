#!/usr/bin/env python3
"""
Retry script for exp8 — picks up the calls that errored on the first run.

The original run hit Neuronpedia's 120/hour rate limit after ~119 successful
necessity calls. This script:
  1. Reads results.json to find errored calls in each stage.
  2. Sleeps until the rate-limit window has rolled over (65 min from the
     first call, computed from results.json's mtime as a proxy).
  3. Re-runs only the failed calls. Merges into results.json in place.
  4. If a 429 is encountered mid-run, sleeps 65 min and continues.

Safe to re-run: it skips anything that already succeeded.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_steering"))
from run_steering import api_post, make_feature, load_api_key, steer

DELAY = 2.0
STRENGTH_SUPPRESS = -20
STRENGTH_BOOST = 20
RATE_LIMIT_WINDOW = 65 * 60  # seconds

PROMPT = "Paris is to France as Berlin is to"
EXPECTED_TOKEN = "Germany"
SUFFICIENCY_PROMPTS = [
    ("Cairo is to Egypt as Nairobi is to", "different analogy same structure"),
    ("Madrid is to Spain as Berlin is to", "different source pair (Berlin still present)"),
    ("The weather today is really", "irrelevant context"),
]

OUTPUT_DIR = Path(__file__).parent
RESULTS_PATH = OUTPUT_DIR / "results.json"
CIRCUIT_JSON = Path("/Users/anish/Downloads/analogical_circuit_180features.json")
BERLIN_GRAPH = Path("/Users/anish/Desktop/aisc_project/experiments/exp4_scale/circuits/analogy_graph.json")


def is_rate_limited(result: dict) -> bool:
    return "error" in result and "429" in str(result.get("error", ""))


def is_error(result: dict) -> bool:
    return "error" in result


def predicts_target(result: dict, target: str) -> bool:
    if is_error(result):
        return False
    txt = result.get("steered_text", "")
    return target.lower() in txt.lower()


def safe_steer(prompt, features, label=""):
    """Steer with built-in 429 handling. Sleeps and retries indefinitely on rate limit."""
    while True:
        r = steer(prompt, features)
        if not is_rate_limited(r):
            return r
        wait = RATE_LIMIT_WINDOW
        print(f"  [{label}] 429 hit; sleeping {wait}s for window reset...", flush=True)
        time.sleep(wait)


def wait_for_reset():
    """Sleep until the original rate-limit window has rolled over."""
    if not RESULTS_PATH.exists():
        return
    first_call_time = RESULTS_PATH.stat().st_mtime
    target = first_call_time + RATE_LIMIT_WINDOW
    now = time.time()
    if now < target:
        wait = int(target - now)
        print(f"Sleeping {wait}s ({wait/60:.1f} min) for rate-limit window to roll over...", flush=True)
        time.sleep(wait)
    else:
        print(f"Window already rolled over ({(now - target)/60:.1f} min ago).", flush=True)


def parse_node_id(node_id):
    parts = node_id.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_circuit():
    with open(CIRCUIT_JSON) as f:
        d = json.load(f)
    out = []
    for feat in d["features"]:
        layer, idx, _ = parse_node_id(feat["node_ids"][0])
        out.append({
            "name": f"L{layer}_F{idx}",
            "layer": layer,
            "index": idx,
            "avg_influence": feat["avg_influence"],
            "avg_activation": feat["avg_activation"],
            "total_appearances": feat["total_appearances"],
        })
    return out


def main():
    print(f"Loading existing results from {RESULTS_PATH}", flush=True)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # --- Identify what needs retrying ---
    nec = results.get("necessity_individual", [])
    failed_nec_idx = [i for i, r in enumerate(nec) if is_error(r["result"])]
    print(f"  Necessity: {len(failed_nec_idx)}/{len(nec)} need retry", flush=True)

    full = results.get("necessity_full", {})
    full_needs_retry = is_error(full.get("result", {})) if full else True
    print(f"  Full suppression: {'needs retry' if full_needs_retry else 'OK'}", flush=True)

    suff = results.get("sufficiency", [])
    failed_suff_idx = [i for i, s in enumerate(suff)
                      if is_error(s.get("result_baseline", {})) or is_error(s.get("result_boost", {}))]
    print(f"  Sufficiency: {len(failed_suff_idx)}/{len(suff)} need retry", flush=True)

    spec = results.get("specificity", [])
    failed_spec_idx = [i for i, r in enumerate(spec) if is_error(r["result"])]
    print(f"  Specificity: {len(failed_spec_idx)}/{len(spec)} need retry", flush=True)

    baseline_needs_retry = is_error(results.get("baseline", {}).get("result", {}))
    print(f"  Baseline: {'needs retry' if baseline_needs_retry else 'OK'}", flush=True)

    total_retries = (len(failed_nec_idx) + (1 if full_needs_retry else 0) +
                     2 * len(failed_suff_idx) + len(failed_spec_idx) + (1 if baseline_needs_retry else 0))
    print(f"Total retry calls: {total_retries}", flush=True)

    # --- Wait for rate limit ---
    wait_for_reset()

    # --- Stage 0: baseline (if failed) ---
    if baseline_needs_retry:
        print("\n[BASELINE] retrying", flush=True)
        r = safe_steer(PROMPT, [], label="baseline")
        results["baseline"]["result"] = r
        if not is_error(r):
            print(f"  Default text: {r.get('default_text', '?')!r}", flush=True)
            print(f"  First token: {r.get('first_default_token', '?')!r} "
                  f"(logprob {r.get('first_default_logprob')})", flush=True)
        else:
            print(f"  Persistent error: {r.get('error')}", flush=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(DELAY)

    # --- Stage 1: necessity retries ---
    if failed_nec_idx:
        print(f"\n[NECESSITY RETRY] {len(failed_nec_idx)} features", flush=True)
        for n, i in enumerate(failed_nec_idx, 1):
            rec = nec[i]
            features = [make_feature(rec["layer"], rec["index"], STRENGTH_SUPPRESS)]
            r = safe_steer(PROMPT, features, label=f"nec[{rec['feature']}]")
            rec["result"] = r
            rec["disrupted"] = (not predicts_target(r, EXPECTED_TOKEN)) if not is_error(r) else None
            tok = r.get("first_steered_token", "?") if not is_error(r) else "ERR"
            mark = "DISRUPT" if rec["disrupted"] else ("ERR" if is_error(r) else "       ")
            print(f"  [{n:>3}/{len(failed_nec_idx)}] {rec['feature']:<14}  steered={tok!r:<14}  {mark}",
                  flush=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            time.sleep(DELAY)

    # --- Stage 2: full suppression ---
    if full_needs_retry:
        print("\n[FULL SUPPRESSION RETRY]", flush=True)
        circuit = load_circuit()
        features = [make_feature(f["layer"], f["index"], STRENGTH_SUPPRESS) for f in circuit]
        r = safe_steer(PROMPT, features, label="full_suppress")
        full["result"] = r
        full["disrupted"] = (not predicts_target(r, EXPECTED_TOKEN)) if not is_error(r) else None
        full["n_features"] = len(circuit)
        results["necessity_full"] = full
        if not is_error(r):
            print(f"  Default: {r.get('default_text', '?')!r}", flush=True)
            print(f"  Steered: {r.get('steered_text', '?')!r}", flush=True)
            print(f"  Disrupted: {full['disrupted']}", flush=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(DELAY)

    # --- Stage 3: sufficiency retries ---
    if failed_suff_idx:
        print(f"\n[SUFFICIENCY RETRY] {len(failed_suff_idx)}", flush=True)
        circuit = load_circuit()
        hub = max(circuit, key=lambda f: f["avg_influence"])
        for i in failed_suff_idx:
            s = suff[i]
            prompt = s["prompt"]
            print(f"  '{prompt[:50]}'", flush=True)

            # Baseline for altered prompt
            r_base = safe_steer(prompt, [], label=f"suff_base[{prompt[:20]}]")
            s["result_baseline"] = r_base
            if not is_error(r_base):
                s["baseline_text"] = r_base.get("default_text", "?")
                print(f"    Baseline: {s['baseline_text']!r}", flush=True)
            time.sleep(DELAY)

            # Boost hub
            features = [make_feature(hub["layer"], hub["index"], STRENGTH_BOOST)]
            r_boost = safe_steer(prompt, features, label=f"suff_boost[{prompt[:20]}]")
            s["result_boost"] = r_boost
            s["hub"] = hub["name"]
            if not is_error(r_boost):
                s["boosted_text"] = r_boost.get("steered_text", "?")
                s["induced_target"] = EXPECTED_TOKEN.lower() in s["boosted_text"].lower()
                print(f"    Boosted:  {s['boosted_text']!r}", flush=True)
                print(f"    Induces '{EXPECTED_TOKEN}': {s['induced_target']}", flush=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            time.sleep(DELAY)

    # --- Stage 4: specificity retries ---
    if failed_spec_idx:
        print(f"\n[SPECIFICITY RETRY] {len(failed_spec_idx)}", flush=True)
        for n, i in enumerate(failed_spec_idx, 1):
            rec = spec[i]
            features = [make_feature(rec["layer"], rec["index"], STRENGTH_SUPPRESS)]
            r = safe_steer(PROMPT, features, label=f"spec[L{rec['layer']}_{rec['index']}]")
            rec["result"] = r
            rec["disrupted"] = (not predicts_target(r, EXPECTED_TOKEN)) if not is_error(r) else None
            tok = r.get("first_steered_token", "?") if not is_error(r) else "ERR"
            mark = "DISRUPT" if rec["disrupted"] else ("ERR" if is_error(r) else "       ")
            print(f"  [{n:>2}/{len(failed_spec_idx)}] L{rec['layer']:>2}_{rec['index']:<6} "
                  f"act={rec['activation']:6.2f} infl={rec['influence']:.3f}  steered={tok!r:<14}  {mark}",
                  flush=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            time.sleep(DELAY)

    # --- Final summary ---
    print(f"\n{'='*70}\nRETRY COMPLETE\n{'='*70}", flush=True)
    nec_ok = [r for r in results["necessity_individual"] if not is_error(r["result"])]
    nec_disrupt = [r for r in nec_ok if r.get("disrupted")]
    print(f"Necessity:    {len(nec_disrupt)}/{len(nec_ok)} features disrupt "
          f"(of {len(results['necessity_individual'])} total)", flush=True)

    fs = results["necessity_full"]
    if not is_error(fs.get("result", {})):
        print(f"Full suppress: {'DISRUPTED' if fs.get('disrupted') else 'intact'}", flush=True)
        print(f"  steered: {fs['result'].get('steered_text', '?')!r}", flush=True)

    suff_ok = [s for s in results["sufficiency"]
               if not is_error(s.get("result_boost", {}))]
    n_induced = sum(1 for s in suff_ok if s.get("induced_target"))
    print(f"Sufficiency:  {n_induced}/{len(suff_ok)} altered prompts induce '{EXPECTED_TOKEN}'", flush=True)

    spec_ok = [r for r in results["specificity"] if not is_error(r["result"])]
    spec_disrupt = [r for r in spec_ok if r.get("disrupted")]
    print(f"Specificity:  {len(spec_disrupt)}/{len(spec_ok)} non-circuit features disrupt", flush=True)


if __name__ == "__main__":
    main()
