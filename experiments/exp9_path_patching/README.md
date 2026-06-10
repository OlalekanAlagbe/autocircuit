# exp9 — Path Patching (true causal mediation)

**Why this exists.** The steering experiments (§3.7) establish *necessity* and
*failure modes* but cannot establish *information flow along an edge* — that the
Phase 1 (template, L0–L4) features feed the Phase 2 (relational, L5–L9) features.
Path patching is the field-standard test for that, and it requires patching
internal activations on the model's own weights, which the Neuronpedia steering
API does not expose.

**Status: scaffold, not run.** `path_patching.py` is structurally complete but
**untested** — this machine has no `torch`/`transformer_lens`/`sae_lens`, no GPU
(Apple-Silicon, no CUDA), and no Hugging Face token (Gemma is gated). Lines
marked `# VERIFY` depend on exact library/release ids and must be checked.

## To run (on a CUDA box with HF access)

```bash
pip install torch transformer_lens sae_lens
huggingface-cli login            # accept the google/gemma-2-2b license first
python path_patching.py
```

## What to finish

1. **Confirm the transcoder release id** for `gemmascope-transcoder-16k` in the
   installed `sae_lens` registry (the `TRANSCODER_RELEASE` / `TRANSCODER_SAE_ID`
   constants), plus the correct hook for the transcoder *input* (pre-MLP).
2. **Implement the two-pass path patch** in `path_patch_phase1_to_phase2`
   (sender = Phase-1 circuit features, receiver = Phase-2 circuit features),
   freezing all other paths at the corrupt activations.
3. Report recovery = (patched − corrupt) / (clean − corrupt) of the answer
   logit gap. High recovery ⇒ the Phase 1 → Phase 2 edge carries the
   analogical signal (genuine mediation, beyond co-necessity).

## Companion: what we *could* do via the API instead

`exp8/deep_validation.py` Stage C-proxy runs a **behavioral interaction** test
(does suppressing Phase 2 depend on Phase 1 being intact) as a weaker substitute.
It is reported as an interaction, **not** as path patching.
