---
name: supernode-upload
description: Use when the user wants to upload an annotated graph to Neuronpedia programmatically, publish a graph, or batch-process multiple prompts into published graphs. Wraps the 3-step Neuronpedia graph upload API (signed-put -> S3 PUT -> save-to-db).
---

# Supernode Upload to Neuronpedia

Push an annotated supernode graph to Neuronpedia's interactive viewer via the graph upload API. Supports single uploads, upload-only (skip pipeline), and batch mode.

## Instructions

1. **Run the pipeline and upload in one command:**

```bash
python scripts/supernode_pipeline.py "The capital of France is" --upload
```

This runs the full pipeline, then uploads the resulting `annotated_graph.json` to Neuronpedia and prints the viewable URL.

2. **Upload an existing annotated graph (skip pipeline):**

```bash
python scripts/supernode_pipeline.py --upload-only output/gemma-2-2b_my-prompt/annotated_graph.json
```

Useful when the pipeline already ran and you just want to publish or republish the result.

3. **Batch upload multiple prompts (inline):**

```bash
python scripts/supernode_pipeline.py "The capital of France is" "The capital of Japan is" --upload
```

4. **Batch upload from a prompt file:**

```bash
python scripts/supernode_pipeline.py --batch-file prompts.txt --upload
```

Prompt file format: one prompt per line. Lines starting with `#` are comments; blank lines are skipped.

```
# Geography set
The capital of France is
The capital of Japan is

# Chemistry set
The chemical symbol for gold is
```

## Output

**Single upload:** Creates `upload_result.json` next to the annotated graph:

```json
{
  "url": "https://www.neuronpedia.org/gemma-2-2b/graph?slug=...",
  "putRequestId": "abc123...",
  "timestamp": "2026-04-09T10:30:00"
}
```

**Batch upload:** Creates `output/batch_results.json` with per-prompt status, output directory, and upload URL, plus a summary table printed to stdout:

```
Prompt                                   | Status | Output Dir                        | URL
the-capital-of-france-is                 | ok     | output/gemma-2-2b_the-capital...  | https://neuronpedia.org/...
the-capital-of-japan-is                  | ok     | output/gemma-2-2b_the-capital...  | https://neuronpedia.org/...
```

## Example Interaction

**User:** Upload the Titanic graph to Neuronpedia

**Command:**

```bash
python scripts/supernode_pipeline.py --upload-only output/gemma-2-2b_bos-the-titanic-sank-in-the-year/annotated_graph.json
```

**Expected output:**

```
Loading: output/gemma-2-2b_bos-the-titanic-sank-in-the-year/annotated_graph.json
[1/3] Requesting signed S3 URL...
[2/3] Uploading JSON (387 KB)...
[3/3] Registering in database...
Uploaded: https://www.neuronpedia.org/gemma-2-2b/graph?slug=abc123
```

## Common Issues

- **401/403 on signed-put:** Verify `config/neuronpedia_config.yaml` has a valid `api_key`. Get one at `https://www.neuronpedia.org/account`.
- **Failed S3 PUT:** Rare but can happen with very large graphs. The tool preserves the local file and prints the manual fallback URL: `https://neuronpedia.org/graph/validator`.
- **One prompt fails mid-batch:** The batch loop catches exceptions per prompt, so other prompts continue. Check `batch_results.json` for which failed and their error messages.
- **"Upload-only" exits without running pipeline:** Expected — `--upload-only` explicitly skips the pipeline. Use `--upload` (without `-only`) if you want to run the pipeline too.

## When to Use

- Publishing a newly generated supernode graph for sharing or review
- Re-uploading after manual edits to an existing annotated graph
- Batch-publishing a set of circuits (e.g., the 30 prompts from the cross-domain study)
- Combining supernode detection and publication in a single command

## Related Skills

- `/supernode-detect` — generate the annotated graph (run this first unless using `--upload-only`)
- `/neuronpedia-validate` — sanity-check the annotated JSON before upload

## Prerequisites

- Annotated graph produced by `/supernode-detect` (unless using `--upload-only` with an existing file)
- Neuronpedia API key in `config/neuronpedia_config.yaml`
- Graph file must be between 1 KB and ~200 MB

## Upload API Flow (reference)

1. `POST /api/graph/signed-put` with `{filename, contentLength, contentType}` -> returns pre-signed S3 URL + putRequestId
2. `PUT` JSON bytes directly to S3 (no auth needed)
3. `POST /api/graph/save-to-db` with `{putRequestId}` -> returns viewable URL

Auth: `x-api-key` header on steps 1 and 3.
