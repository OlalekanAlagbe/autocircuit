I have autocircuit_tools.py in this folder.

# ── CONFIGURATION — change only this line to switch categories ───────────────
CATEGORY = "linguistic"   # options: "analogical" | "factual_recall" | "linguistic"
# ─────────────────────────────────────────────────────────────────────────────

# ── SECTION 0: READ AND UNDERSTAND THE CODE FIRST ────────────────────────────
#
# Before executing any step below, read autocircuit_tools.py in full.
# Do not skip this. Do not proceed to Step 1 until you have completed
# all four sub-tasks here.
#
# 0a. READ EVERY FUNCTION.
#     For each function, understand:
#       - What it does
#       - What arguments it takes and what it returns
#       - When in the pipeline it should be called
#       - Whether it is called automatically by another function, or whether
#         YOU must call it explicitly
#
#     Pay particular attention to these functions which are easy to miss
#     but must be called explicitly — nothing calls them for you:
#
#       get_graph_summary(G)
#           → Call immediately after every load_graph(). Prints node count,
#             layer distribution, top-5 nodes, max/avg influence. Gives you
#             a quick sanity check that the graph loaded correctly and tells
#             you which layers are most active before you dive into details.
#
#       get_edges_for_node(G, node_id)
#           → Call on the top 1-2 recurring features from compare_graphs()
#             in Step 5. Shows what feeds INTO the feature and what it feeds
#             INTO — the immediate wiring around the most important node.
#             The node_id to pass is the canonical node_id from the recurring
#             features table (now included in compare_graphs() output).
#
#       get_nodes_by_layer(G, layer)
#           → Call when a specific layer keeps appearing as a hub across
#             multiple prompts in Step 5. Lists every node at that layer
#             ranked by influence, so you can see the full picture of what
#             that layer is doing, not just the top recurring feature.
#
#       fetch_existing_graph(slug)
#           → Before calling generate_graph() for any prompt in Step 3,
#             check whether graphs/{slug}.json already exists on disk.
#             If it does, load it directly instead of re-generating.
#             This makes the pipeline resumable if it crashes mid-run.
#
# 0b. MAP THE CALL CHAIN.
#     Some functions are called automatically inside other functions.
#     Know which ones so you don't call them twice or miss them entirely.
#     Specifically, interpret_prompt_graph() automatically calls:
#       - get_top_nodes()
#       - label_nodes_batch()
#       - get_logit_candidates_from_graph()
#       - trace_causal_paths()
#       - format_causal_paths_for_narration()
#       - visualize_causal_paths()   ← saves the PNG automatically
#       - _call_claude_for_interpretation()
#       - _append_prompt_to_paper()  ← appends to paper automatically
#     You do NOT need to call any of these separately inside Step 6.
#
# 0c. UNDERSTAND THE DATA FLOW.
#     Trace the full pipeline in your head before writing a single line:
#       build_prompt_dataset()
#         → generate_graph() × N        (one per prompt)
#         → load_graph() × N            (returns G and graph_data)
#         → get_graph_summary() × N     (call explicitly after each load)
#         → compare_graphs([G, ...])    (cross-graph pattern mining)
#         → extract_node_ids_from_recurring()
#         → label_nodes_batch()
#         → get_edges_for_node()        (call explicitly on top features)
#         → get_nodes_by_layer()        (call explicitly on hub layers)
#         → interpret_prompt_graph() × N (per-prompt, writes to paper)
#         → save_circuit()
#
# 0d. CRITICAL: FEATURE IDs ARE NOT IN THE NODE_ID STRING.
#     A node_id looks like "8_13766_8". The three parts are:
#       - Part 0 ("8")     → the transformer layer. This IS reliable.
#       - Part 1 ("13766") → an internal graph index. NOT the feature ID.
#       - Part 2 ("8")     → the token context index (ctx_idx).
#     The actual feature ID (e.g. 94882191) is stored in the node's "feature"
#     field in the graph JSON. It bears NO relation to the middle number in
#     the node_id. Always read feature IDs from the node's data, never parse
#     them from the node_id string.
#
#     compare_graphs() and the per-prompt band tables now include both
#     "feature" and "node_id" columns explicitly so you always have both.
#     When calling label_node() or get_edges_for_node(), pass the node_id.
#     When reporting or referencing a feature, use the feature field value.
#
# 0e. CONFIRM YOUR UNDERSTANDING.
#     Before Step 1, print a one-line summary of what each function does,
#     in pipeline order. This is your confirmation that you have read and
#     understood the full file. If you cannot summarise a function, re-read it.
#
# ─────────────────────────────────────────────────────────────────────────────

1. Run `python autocircuit_tools.py` to confirm Neuronpedia is connected. Stop and
   report if it fails.

2. Import autocircuit_tools and call build_prompt_dataset(). Extract the prompt list
   for CATEGORY. Do NOT deduplicate — the list is already clean. Print the category
   name and the exact count of prompts before proceeding.

3. Generate attribution graphs for every prompt in the list using generate_graph().
   Before each call, check whether graphs/{slug}.json already exists on disk.
   If it does, skip generate_graph() and load from disk directly — do not
   re-generate. Sleep 2 seconds between each API call. If any single graph
   fails, print the error, skip it, and continue. Track which prompts succeeded
   and which failed. Print a summary at the end: N succeeded, M failed.

4. Load all successfully saved graphs from the graphs/ folder using load_graph().
   Match by slug (first 40 chars of prompt, lowercased, spaces → underscores).
   Store each as a tuple (G, graph_data).

   After each load_graph() call, immediately call get_graph_summary(G) and
   print the result. This confirms the graph loaded correctly and shows you
   which layers and nodes are most active before deeper analysis.

   Print the total count of graphs loaded once all are done.

5. ── ACROSS-GRAPHS ANALYSIS ──────────────────────────────────────────────────
   Run compare_graphs() on the loaded graphs with threshold=0.50 (50% of graph
   count). With 10 graphs this means a feature must appear in at least 5 graphs
   to qualify as a recurring circuit component. This distinguishes genuine shared
   circuit structure from prompt-specific noise.

   compare_graphs() prints the full recurring features table automatically,
   including Layer, Feature, Node_id, Appearances, Avg Influence, and Avg
   Activation columns. The Feature column is the actual feature ID read from
   the node's data field — not parsed from the node_id string.

   Then call extract_node_ids_from_recurring() with top_n=15 to get one
   representative node_id per recurring feature. Call label_nodes_batch() on
   those 15 node_ids with delay=0.5. Merge labels back into the recurring
   features list so each entry has: layer, feature, node_id, appearances,
   avg_influence, and human-readable label.

   Then:

   a. Call get_edges_for_node(G, node_id) on the top 2 recurring features by
      avg_influence. Use the node_id from the recurring features table and any
      graph in which they appear. Print the incoming and outgoing edges with
      their weights. For each connected node, note its feature ID (from the
      graph's node data, not from the node_id string) and its layer — this
      gives the full wiring picture: what feeds into these features and what
      they feed into, with proper feature IDs throughout.

   b. Identify which layer appears most frequently among the top 15 recurring
      features. Call get_nodes_by_layer(G, layer) on that layer (use any
      loaded graph). Print all nodes at that layer ranked by influence,
      including their node_id and feature ID. This gives the full picture of
      what that layer is doing across the circuit.

   c. Interpret the full recurring features table:
      - Which features appear most universally across prompts?
      - Do the labels cluster into recognisable functional groups — structural/
        syntactic features in early layers, domain knowledge in middle layers,
        answer-selection features in late layers?
      - What does the layer distribution suggest about how the model implements
        CATEGORY reasoning as a shared circuit?
      - Are there any features whose labels seem surprising or mismatched to
        their layer position? These often reveal that the model uses a feature
        for a different purpose than its label suggests.
      - What do the edge neighbourhoods from (a) reveal about how the top
        features are connected to the rest of the circuit?

6. ── PER-PROMPT INTERPRETATION ───────────────────────────────────────────────
   Before starting, create {CATEGORY}_circuit_paper.md (substituting the actual
   category name) and write the paper skeleton: Abstract (leave as placeholder),
   Introduction, Methods, then a Results section with:
     - The recurring features table from step 5 already filled in (with Layer,
       Feature, Node_id, Appearances, Avg Influence, Label columns)
     - The edge neighbourhood analysis from step 5a
     - The layer analysis from step 5b
     - A "Per-prompt circuit interpretations" sub-section header, left empty

   Then for each prompt's (G, graph_data) tuple, call:
     interpret_prompt_graph(G, graph_data, paper_path="CATEGORY_circuit_paper.md")
   substituting the actual category name in the paper_path.

   Each call does the following automatically — you do not need to call
   anything extra:
     a. Labels 40 nodes (top_n_nodes=40) grouped into early / middle / late
        bands, printing them as they arrive band by band. Each node in the
        band tables includes Layer, Feature, Node_id, Influence, Activation,
        and Feature label columns.
     b. Computes token competition — which tokens the circuit voted for.
     c. Traces causal paths through edge weights — the actual wiring from
        input features to the logit node, following the strongest edges at
        each hop. Paths are grouped into excitatory (pushing toward the
        predicted token), inhibitory (pushing against it), and mixed.
        Each node in a path shows its node_id, feature ID (read from graph
        node data, never parsed from node_id), layer, label, and edge weight.
     d. Saves a PNG figure to graphs/{slug}__{token}_causal_paths.png showing
        the paths as a layered diagram. Each node box in the figure displays
        its node_id, feature ID, label, and influence score. Edge widths are
        proportional to weight.
     e. Appends all data — band tables, token competition, causal paths, and
        an embedded figure link — to the paper file automatically.

   The narration style guide prints once on the first prompt only.

   After EACH call returns, immediately write the mechanistic narrative for
   that prompt into the paper (replacing the [Claude Code — write narrative
   here] marker). Do this ONE PROMPT AT A TIME — do not batch them.

   Your narrative must use BOTH the band analysis AND the causal paths:
     - The band analysis tells you which features mattered (ranked by influence)
     - The causal paths tell you how they were connected (the actual wiring)
     - Together they answer: what did the model detect, how did that signal
       travel through the network, and what made the final push to the output?

   When referencing any feature in your narrative, always cite both its
   feature ID and its node_id so the reader can locate it unambiguously.

   For each prompt your narrative must answer:
     1. What did the early-layer features detect in the raw input?
        (cite feature ID, node_id, and label for each)
     2. Which middle-layer feature was the pivotal domain-mapping moment?
        (cite feature ID, node_id, label, and its influence score)
     3. Which late-layer feature made the final push to the logit?
        (cite feature ID, node_id, label, and edge weight to logit)
     4. What was the dominant excitatory path?
        (list each node with node_id, feature ID, label, and edge weight)
     5. Were there any inhibitory paths revealing suppressed competitor tokens?
        (cite the suppressing feature's node_id, feature ID, and edge weight)
     6. Was the circuit convergent (one dominant path, high-confidence prediction)
        or ambiguous (multiple competing paths, low confidence)?

7. ── SAVE CIRCUIT ────────────────────────────────────────────────────────────
   Call save_circuit() with:
   - nodes: merged recurring-feature list from step 5 (with labels, feature
     IDs, and node_ids)
   - prompt_category: CATEGORY
   - source_graphs: slugs of all successfully loaded graphs
   - name and description summarising the discovered circuit

8. ── COMPLETE THE PAPER ──────────────────────────────────────────────────────
   Per-prompt interpretations were written incrementally in step 6. Now add:
   - Abstract: fill in with actual numbers (prompts run, graphs generated,
     recurring features found at 50% threshold, top feature IDs and labels,
     dominant causal paths identified)
   - Discussion: what do the recurring features AND causal paths together reveal
     about how Gemma-2-2B implements CATEGORY reasoning? Which excitatory paths
     appeared most consistently? Were inhibitory paths systematic (same competitor
     tokens suppressed across prompts) or random? How does the path structure
     relate to the layer distribution of recurring features? Any surprising
     findings — features active for unexpected reasons, mismatched labels,
     unusually long or short causal chains? Include insights from the edge
     neighbourhood analysis (step 5a) and layer analysis (step 5b).
   - Limitations and Conclusion
   Use actual numbers, feature IDs, and node_ids throughout. No placeholders.
