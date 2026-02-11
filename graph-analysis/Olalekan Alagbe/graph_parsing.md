# Skill: Neuronpedia Graph Parsing

## When to Use
Use this skill whenever the agent is given a Neuronpedia Circuit Tracer graph
(JSON format) and needs to understand, analyze, or manipulate the graph.

This skill MUST run before:
- Circuit discovery
- CLT identification
- Supernode grouping
- Attribution analysis

---

## Goal
Parse a Neuronpedia attribution graph into a clean internal graph
representation with explicit nodes, edges, and Cross-Layer Transcoder (CLT)
identification.

---

## Expected Input
A JSON object with top-level keys:

- `metadata`
- `qParams`
- `nodes`
- `links`

Only `nodes` and `links` are used for graph construction.

---

## Parsing Instructions

### Step 1: Ignore Non-Graph Fields
Ignore the following keys entirely:
- `metadata`
- `qParams`

They do not affect graph topology.

---

### Step 2: Parse Nodes

For each object in `nodes[]`, extract the following fields:

- `node_id` (string, unique identifier)
- `feature` (integer feature index)
- `layer` (integer model layer)
- `ctx_idx` (integer token index)
- `feature_type` (string)
- `activation` (float)
- `influence` (float)

Convert `layer` to an integer if it is serialized as a string.

---

### Step 3: Classify Nodes

Classify each node using `feature_type`:

- If `feature_type == "cross layer transcoder"` → CLT node
- Otherwise → non-CLT feature node

Maintain:
- A dictionary mapping `node_id → node`
- A list of all CLT nodes
- Groupings by layer and token index

---

### Step 4: Parse Links

For each object in `links[]`, extract:

- `source` (string node ID)
- `target` (string node ID)
- `weight` (float)

Edges are **directed**:  
`source → target` indicates influence.

Interpret weight sign:
- Positive → excitatory influence
- Negative → inhibitory influence

---

### Step 5: Handle External Nodes

Some node IDs appear only in `links[]` and not in `nodes[]`.

Rules:
- Node IDs beginning with `E_` represent embedding or input nodes
- These nodes are valid graph sources
- Do NOT discard edges whose source is missing from `nodes[]`

---

### Step 6: Build Graph Indices

Construct the following internal structures:

- `nodes_by_id`
- `in_edges_by_target`
- `out_edges_by_source`
- `clt_nodes`

All further reasoning must use these structures, not the raw JSON.

---

## Output Guarantees

After this skill completes:

- All CLT nodes are explicitly identified
- Directed influence structure is preserved
- External input nodes are retained
- The graph is ready for circuit-level reasoning

---

## Notes

This skill defines a **parsing contract**.
Downstream skills may assume this contract has been satisfied.