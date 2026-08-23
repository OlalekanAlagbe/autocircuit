#!/usr/bin/env python3
"""Re-measure the unsteered baseline probability for the five analogy prompts.

The `baseline_p` field in cross_prompt_results.json was never measured — it was
a literal carried in the PROMPTS table of cross_prompt_validation.py (see the
0.958/0.974/0.99/0.486/0.117 constants there) and copied verbatim into the
output JSON, which made it look like a result. It disagrees with the
`first_default_logprob` the API actually returned on the same runs.

This script measures the baseline directly: one /steer call per prompt with an
empty feature list, so the DEFAULT branch is the unsteered model. Records the
full top-5 so the expected-answer probability can be read even when it is not
the argmax.

5 calls total — well within the 120/hour limit.
"""

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "exp2_steering"))
from run_steering import make_feature, steer  # noqa: E402

# /steer rejects an empty feature list with HTTP 500 (this is why the original
# dedicated baseline call in results.json failed). The DEFAULT branch returned
# by /steer is the unsteered model regardless of what is passed, so we send a
# single arbitrary feature at strength 0 — a no-op intervention — and read the
# DEFAULT branch. Confirmed no-op: STEERED and DEFAULT come back identical.
NOOP_FEATURE = [make_feature(0, 11651, 0)]

OUTPUT = HERE / "baseline_remeasured.json"
DELAY = 2.0

# prompt name -> (prompt text, expected answer token)
PROMPTS = {
    "analog_berlin":  ("Paris is to France as Berlin is to", " Germany"),
    "analog_rome":    ("Paris is to France as Rome is to", " Italy"),
    "analog_tokyo":   ("Paris is to France as Tokyo is to", " Japan"),
    "analog_teacher": ("Doctor is to hospital as teacher is to", " school"),
    "analog_bird":    ("Fish is to water as bird is to", " air"),
}


def p(lp):
    return None if lp is None else math.exp(lp)


def main():
    results = {}
    for name, (prompt, expected) in PROMPTS.items():
        print(f"=== {name}: {prompt!r} -> {expected!r}", flush=True)
        r = steer(prompt, NOOP_FEATURE, n_tokens=5)
        if "error" in r:
            print(f"  ERROR {r['error']}", flush=True)
            results[name] = {"prompt": prompt, "expected": expected, "error": r["error"]}
            time.sleep(DELAY)
            continue

        top5 = r.get("first_default_top5", []) or []
        # probability the model assigns to the expected answer, argmax or not
        expected_lp = next(
            (t["logprob"] for t in top5 if t.get("token") == expected), None
        )

        results[name] = {
            "prompt": prompt,
            "expected": expected,
            "argmax_token": r.get("first_default_token"),
            "argmax_p": p(r.get("first_default_logprob")),
            "expected_p": p(expected_lp),
            "expected_in_top5": expected_lp is not None,
            "top5": [
                {"token": t.get("token"), "p": p(t.get("logprob"))} for t in top5
            ],
            "default_text": r.get("default_text"),
            # sanity check that the strength-0 feature really was a no-op
            "noop_verified": r.get("default_text") == r.get("steered_text"),
        }
        rec = results[name]
        print(
            f"  argmax={rec['argmax_token']!r} p={rec['argmax_p']:.4f}"
            f"   expected {expected!r} p="
            + (f"{rec['expected_p']:.4f}" if rec["expected_p"] is not None else "not in top-5"),
            flush=True,
        )
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        time.sleep(DELAY)

    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
