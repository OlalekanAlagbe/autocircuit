#!/usr/bin/env python3
"""
Path patching for the analogical circuit — the gold-standard causal test that
the Neuronpedia steering API CANNOT do, because it requires patching internal
activations on the model's own weights.

STATUS: SCAFFOLD. This is structurally complete but UNTESTED — it has not been
run (the dev environment has no torch / transformer_lens / GPU / HF access).
Treat it as a starting point to debug on a GPU box, not turnkey code. Points
marked  # VERIFY  depend on exact library versions / release ids and must be
checked against the installed sae_lens registry.

WHAT THIS TESTS (the claim steering can only gesture at):
  The paper claims a three-phase architecture where Phase 1 (template parsing,
  L0-L4) feeds Phase 2 (relational transfer, L5-L9), which feeds the answer.
  Steering shows each phase is *necessary*, but necessity does not establish
  that information *flows along the edge* Phase 1 -> Phase 2. Path patching does:
  it patches the Phase-1 -> Phase-2 path specifically and measures whether the
  answer logit moves, while holding all other paths at their clean values.

METHOD (denoising path patching, Wang et al. 2022 / Goldowsky-Dill et al. 2023):
  clean   prompt: "Paris is to France as Berlin is to" -> " Germany"
  corrupt prompt: a structure-matched analogy with a different answer, e.g.
                  "Paris is to France as Madrid is to" -> " Spain"
  1. Cache all transcoder-feature activations on BOTH prompts.
  2. Run the corrupt prompt, but patch the Phase-1 circuit features' outputs to
     their CLEAN values, restricting the patch to the path that reaches the
     Phase-2 circuit features (i.e. patch Phase-1 -> Phase-2 receiver edges only,
     freezing everything else at corrupt).
  3. Measure recovery of the clean answer logit. Large recovery => the
     Phase-1 -> Phase-2 edge carries the analogical signal.

Requires (install on a CUDA box):
  pip install torch transformer_lens sae_lens
  huggingface-cli login         # Gemma is gated: accept the license first
"""

import json
from pathlib import Path

import torch  # noqa: E402  (install on GPU box)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = "google/gemma-2-2b"

# The exact Gemma Scope transcoder release used by Neuronpedia's
# "{layer}-gemmascope-transcoder-16k". # VERIFY against sae_lens' pretrained
# registry: `from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory`
TRANSCODER_RELEASE = "gemma-scope-2b-pt-transcoders"   # VERIFY exact id
TRANSCODER_SAE_ID  = "layer_{layer}/width_16k/canonical"  # VERIFY id template

CIRCUIT_JSON = Path(__file__).parents[1] / "exp8_180feature_validation" / "analogical_circuit_180features.json"

CLEAN_PROMPT   = "Paris is to France as Berlin is to"
CORRUPT_PROMPT = "Paris is to France as Madrid is to"   # same template, answer = Spain
ANSWER_CLEAN   = " Germany"
ANSWER_CORRUPT = " Spain"

PHASE1_LAYERS = range(0, 5)    # L0-L4
PHASE2_LAYERS = range(5, 10)   # L5-L9


def parse_node_id(nid):
    a, b, c = nid.split("_"); return int(a), int(b), int(c)


def load_circuit_by_layer():
    """Return {layer: set(feature_index)} for the 180-feature circuit."""
    d = json.load(open(CIRCUIT_JSON))
    out = {}
    for f in d["features"]:
        layer, idx, _ = parse_node_id(f["node_ids"][0])
        out.setdefault(layer, set()).add(idx)
    return out


# --------------------------------------------------------------------------
# Model + transcoders
# --------------------------------------------------------------------------
def load_model_and_transcoders():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE  # transcoders load through the SAE class in recent sae_lens  # VERIFY

    model = HookedTransformer.from_pretrained(MODEL, device=DEVICE)
    transcoders = {}
    for layer in list(PHASE1_LAYERS) + list(PHASE2_LAYERS):
        tc, cfg, _ = SAE.from_pretrained(
            release=TRANSCODER_RELEASE,
            sae_id=TRANSCODER_SAE_ID.format(layer=layer),
            device=DEVICE,
        )  # VERIFY return signature / loader for transcoders specifically
        transcoders[layer] = tc
    return model, transcoders


# --------------------------------------------------------------------------
# Core: cache transcoder feature activations, then path-patch
# --------------------------------------------------------------------------
def transcoder_acts(model, transcoders, prompt):
    """Run `prompt`, return {layer: feature_activations[pos, n_features]} and the cache."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    acts = {}
    for layer, tc in transcoders.items():
        # Transcoders read the MLP input (ln2 / pre-MLP residual) and predict MLP out.
        mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"]  # VERIFY hook name for transcoder input
        acts[layer] = tc.encode(mlp_in)  # feature activations
    return acts, cache, tokens


def answer_logit_diff(model, tokens, answer_clean, answer_corrupt):
    logits = model(tokens)[0, -1]
    ic = model.to_single_token(answer_clean)
    ik = model.to_single_token(answer_corrupt)
    return (logits[ic] - logits[ik]).item()


def path_patch_phase1_to_phase2(model, transcoders, circuit):
    """
    Denoising path patch: freeze everything at CORRUPT, restore the clean
    Phase-1 circuit-feature contribution ONLY along the edge into the Phase-2
    circuit features, and measure recovery of the clean-answer logit.

    Returns the fraction of the clean-vs-corrupt logit gap recovered.
    """
    clean_acts, clean_cache, clean_tok = transcoder_acts(model, transcoders, CLEAN_PROMPT)
    corr_acts,  corr_cache,  corr_tok  = transcoder_acts(model, transcoders, CORRUPT_PROMPT)

    baseline = answer_logit_diff(model, corr_tok, ANSWER_CLEAN, ANSWER_CORRUPT)
    clean_ref = answer_logit_diff(model, clean_tok, ANSWER_CLEAN, ANSWER_CORRUPT)

    # TODO (debug on GPU): implement the two-hook path patch.
    #  1. Forward CORRUPT with a hook that, at each Phase-1 layer, replaces the
    #     CIRCUIT features' decoded contribution with their CLEAN value while
    #     leaving non-circuit features at corrupt.
    #  2. Restrict the downstream effect to the Phase-2 circuit features by
    #     freezing all OTHER receivers (the "path" restriction) — standard
    #     two-pass path patching with sender=Phase1-circuit, receiver=Phase2-circuit.
    #  3. Read the resulting answer_logit_diff; recovery = (patched - baseline) /
    #     (clean_ref - baseline).
    raise NotImplementedError(
        "Path-patch hooks not implemented — finish on a GPU box. "
        f"baseline logit diff (corrupt) = {baseline:.3f}, clean = {clean_ref:.3f}"
    )


def main():
    circuit = load_circuit_by_layer()
    print("Circuit features per layer:", {l: len(s) for l, s in sorted(circuit.items())})
    model, transcoders = load_model_and_transcoders()
    recovery = path_patch_phase1_to_phase2(model, transcoders, circuit)
    print(f"Phase1 -> Phase2 path recovers {recovery:.1%} of the clean-answer logit gap.")


if __name__ == "__main__":
    main()
