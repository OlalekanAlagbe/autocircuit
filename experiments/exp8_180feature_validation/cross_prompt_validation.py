#!/usr/bin/env python3
"""Cross-prompt replication of the two circuit-level validation tests.

The §3.7.1 Test 2 (full circuit suppression) and Appendix A.6 (null control)
were originally run on analog_berlin only. The 180-feature circuit is *defined*
by recurrence across all five analogy graphs, so the causal "shared circuit"
claim requires showing the same two-test contrast on the other four prompts.

For each of the five prompts this runs exactly two steering calls:

  (A) CIRCUIT ablation  — suppress all 180 shared-circuit features at strength
      -20 (effective -80). Tests whether removing the circuit breaks the
      prediction, and *how* it breaks (the failure mode).

  (B) NULL control      — suppress 180 RANDOM non-circuit features sampled from
      THAT prompt's own attribution graph (same stratified procedure as the
      specificity test, seed 7), at the same strength. Tests whether the
      breakage in (A) is circuit-specific or a generic consequence of ablating
      180 features at this magnitude.

The discriminating signal is the contrast between (A) and (B): if the circuit
is faithful, (A) produces a structured, mechanism-consistent collapse (toward a
prompt-template token) while (B) produces an unstructured collapse (garbage /
unrelated token), even though both may remove the expected answer.

Baselines are read from the `default_text` returned with every steering call
(the unsteered branch), so no separate baseline call is needed.

10 calls total (2 per prompt x 5 prompts) — well within the 120/hour limit.
Crash-safe: writes cross_prompt_results.json after every call.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "exp2_steering"))
from run_steering import make_feature, steer  # noqa: E402
from run_validation import sample_non_circuit_features, parse_node_id  # noqa: E402

STRENGTH = -20
NULL_SEED = 7
RATE_LIMIT_WINDOW = 65 * 60
DELAY = 2.0

CIRCUIT_JSON = HERE / "analogical_circuit_180features.json"
OUTPUT = HERE / "cross_prompt_results.json"

EXP4 = HERE.parent / "exp4_scale" / "circuits"
EXP7 = HERE.parent / "exp7_remaining_analogy_validation" / "graphs"

# prompt -> (prompt text, expected first token, baseline confidence p, graph path)
PROMPTS = {
    "analog_berlin":  ("Paris is to France as Berlin is to", "Germany", 0.958, EXP4 / "analogy_graph.json"),
    "analog_rome":    ("Paris is to France as Rome is to",   "Italy",   0.974, EXP7 / "analog_rome.json"),
    "analog_tokyo":   ("Paris is to France as Tokyo is to",  "Japan",   0.990, EXP7 / "analog_tokyo.json"),
    "analog_teacher": ("Doctor is to hospital as teacher is to", "school", 0.486, EXP7 / "analog_teacher.json"),
    "analog_bird":    ("Fish is to water as bird is to",     "air",     0.117, EXP7 / "analog_bird.json"),
}


def is_error(r):
    return isinstance(r, dict) and "error" in r


def is_rate_limited(r):
    return is_error(r) and "429" in str(r.get("error", ""))


def is_server_error(r):
    return is_error(r) and "500" in str(r.get("error", ""))


def safe_steer(prompt, features, label="", max_500_retries=6):
    """Retry 429 (rate limit) indefinitely and 500 (transient server error) a
    bounded number of times with short backoff. The 180-feature payload
    intermittently 500s under load even though the identical call succeeds on
    retry, so we re-issue rather than record a spurious error."""
    server_errors = 0
    while True:
        r = steer(prompt, features)
        if is_rate_limited(r):
            print(f"  [{label}] 429; sleeping {RATE_LIMIT_WINDOW}s for window reset...", flush=True)
            time.sleep(RATE_LIMIT_WINDOW)
            continue
        if is_server_error(r) and server_errors < max_500_retries:
            server_errors += 1
            wait = 10 * server_errors
            print(f"  [{label}] HTTP 500 (attempt {server_errors}/{max_500_retries}); "
                  f"retrying in {wait}s...", flush=True)
            time.sleep(wait)
            continue
        return r


def load_circuit_features():
    with open(CIRCUIT_JSON) as f:
        d = json.load(f)
    feats, pairs = [], set()
    for feat in d["features"]:
        layer, idx, _ = parse_node_id(feat["node_ids"][0])
        feats.append({"layer": layer, "index": idx})
        pairs.add((layer, idx))
    return feats, pairs


def predicts_expected(r, expected):
    """Expected answer survives if it is the first steered token (case-insensitive)."""
    if is_error(r):
        return None
    tok = (r.get("first_steered_token") or "").strip().lower()
    return tok == expected.strip().lower()


def save(results):
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)


def main():
    circuit_feats, circuit_set = load_circuit_features()
    print(f"Circuit: {len(circuit_feats)} features (shared across all prompts)\n", flush=True)

    results = {}
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            results = json.load(f)
        print(f"Resuming; {len(results)} prompts already recorded\n", flush=True)

    circuit_payload = [make_feature(f["layer"], f["index"], STRENGTH) for f in circuit_feats]

    for pname, (prompt, expected, conf, graph_path) in PROMPTS.items():
        if pname in results and all(
            k in results[pname] and not is_error(results[pname][k].get("result", {}))
            for k in ("circuit", "null")
        ):
            print(f"[{pname}] already complete; skipping", flush=True)
            continue

        print(f"=== {pname}: {prompt!r} -> {expected!r} (baseline p={conf}) ===", flush=True)
        rec = results.get(pname, {"prompt": prompt, "expected": expected, "baseline_p": conf})

        # ---- (A) circuit ablation ----
        print(f"  (A) suppressing all {len(circuit_payload)} circuit features...", flush=True)
        r_circ = safe_steer(prompt, circuit_payload, label=f"{pname}:circuit")
        rec["circuit"] = {
            "n_features": len(circuit_payload),
            "strength": STRENGTH,
            "result": r_circ,
            "disrupted": (None if is_error(r_circ) else not predicts_expected(r_circ, expected)),
        }
        if not is_error(r_circ):
            rec["baseline_text"] = r_circ.get("default_text", "?")
            rec["baseline_first_token"] = r_circ.get("first_default_token", "?")
            rec["baseline_first_logprob"] = r_circ.get("first_default_logprob")
            print(f"      default : {r_circ.get('default_text','?')!r}", flush=True)
            print(f"      steered : {r_circ.get('steered_text','?')!r}  "
                  f"(first={r_circ.get('first_steered_token','?')!r}, "
                  f"lp={r_circ.get('first_steered_logprob')})", flush=True)
            print(f"      disrupted={rec['circuit']['disrupted']}", flush=True)
        else:
            print(f"      ERROR: {r_circ.get('error')}", flush=True)
        results[pname] = rec
        save(results)
        time.sleep(DELAY)

        # ---- (B) null control ----
        sampled = sample_non_circuit_features(graph_path, circuit_set, n=180, seed=NULL_SEED)
        null_payload = [make_feature(f["layer"], f["index"], STRENGTH) for f in sampled]
        print(f"  (B) suppressing {len(null_payload)} random non-circuit features "
              f"(seed {NULL_SEED}, from {graph_path.name})...", flush=True)
        r_null = safe_steer(prompt, null_payload, label=f"{pname}:null")
        rec["null"] = {
            "n_features": len(null_payload),
            "n_requested": 180,
            "strength": STRENGTH,
            "seed": NULL_SEED,
            "sampled_features": [{"layer": f["layer"], "index": f["index"],
                                  "activation": f["activation"], "influence": f["influence"]}
                                 for f in sampled],
            "result": r_null,
            "disrupted": (None if is_error(r_null) else not predicts_expected(r_null, expected)),
        }
        if not is_error(r_null):
            print(f"      steered : {r_null.get('steered_text','?')!r}  "
                  f"(first={r_null.get('first_steered_token','?')!r}, "
                  f"lp={r_null.get('first_steered_logprob')})", flush=True)
            print(f"      disrupted={rec['null']['disrupted']}", flush=True)
        else:
            print(f"      ERROR: {r_null.get('error')}", flush=True)
        results[pname] = rec
        save(results)
        time.sleep(DELAY)
        print(flush=True)

    # ---- summary ----
    print(f"\n{'='*72}\nCROSS-PROMPT SUMMARY\n{'='*72}", flush=True)
    print(f"{'prompt':16s} {'expected':9s} {'circuit->':28s} {'null->':28s}", flush=True)
    for pname, (prompt, expected, conf, _) in PROMPTS.items():
        if pname not in results:
            continue
        rec = results[pname]
        c = rec.get("circuit", {}).get("result", {})
        n = rec.get("null", {}).get("result", {})
        c_tok = "ERR" if is_error(c) else repr(c.get("first_steered_token", "?"))
        n_tok = "ERR" if is_error(n) else repr(n.get("first_steered_token", "?"))
        c_lp = "" if is_error(c) else f"(lp {c.get('first_steered_logprob'):.2f})"
        n_lp = "" if is_error(n) else f"(lp {n.get('first_steered_logprob'):.2f})"
        print(f"{pname:16s} {expected:9s} {c_tok+' '+c_lp:28s} {n_tok+' '+n_lp:28s}", flush=True)


if __name__ == "__main__":
    main()
