# Interpretation Guidelines

Follow these guidelines when writing the circuit analysis report:

1. **Feature descriptions from Neuronpedia are often misleading.** Activation examples like "Drupal views" or "HTML tags" on a universal bottleneck don't mean the feature literally encodes Drupal. Cross-circuit universality is the clue -- if a feature appears in 5+ circuits across different prompts, it encodes something abstract (relational structure, emphasis, routing) not the literal surface pattern.

2. **Look for directly semantic features.** These are features whose activation examples directly relate to the prompt topic (e.g., "Africa, Africa" or "river, streams, Mississippi" for a river prompt). These typically emerge in L16-L20 (LATE_PROC) and are the most interpretable.

3. **Traceback scores indicate information routing importance.** A score of 2 billion vs 500 million means that feature carries 4x more information toward the output. The highest-scoring feature in the traceback is the single most critical relay.

4. **Cross-circuit overlap rate tells you how unique a circuit is.** Chemistry prompts share 60-75% of bottlenecks. Unusual factual queries may share only 20-30%. Low overlap = specialized pathway.

5. **Betweenness centrality identifies chokepoints.** High betweenness = information must flow through this node. Best steering targets are high-betweenness, prompt-specific features (not universal ones that would disrupt everything).
