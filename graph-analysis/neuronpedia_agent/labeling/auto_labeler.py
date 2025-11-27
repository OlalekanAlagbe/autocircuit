"""Auto-labeling module using LLM to generate supernode labels"""

from typing import Dict
import anthropic
from ..analysis.grouping_engine import Supernode


LABELING_PROMPT = """
You are analyzing a group of features in a language model's computational pathway. Your task is to create a concise label (2-5 words) that describes what this group of features collectively does.

Context:
- Input prompt: "{prompt}"
- Target output: "{target_logit}"
- Functional role: {functional_role} (input_detector, relational_processor, or output_promoter)
- Layer range: {layer_min} to {layer_max}

Features in this group:
{feature_details}

Guidelines for labeling:
1. For input detectors: Describe what tokens/patterns they detect (e.g., "state names", "capital-of preposition")
2. For relational processors: Describe the relationship they encode (e.g., "capital-state mapping", "location relations")
3. For output promoters: Describe what they predict (e.g., "say [city name]", "promote Texas cities")
4. Be specific but concise (2-5 words)
5. Avoid technical jargon like "features" or "nodes"

Examples of good labels:
- "Texas detection" (input detector activating on Texas-related tokens)
- "capital-of relation" (processor encoding capital-state relationships)
- "say [capital city]" (output promoter for capital city names)

Generate a label:
"""


class AutoLabeler:
    """Generate human-readable labels for supernodes using an LLM"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_label(self, supernode: Supernode, node_data: Dict, prompt: str = "", target_logit: str = "") -> str:
        """
        Generate a concise, interpretable label for a supernode

        Inputs:
        - supernode: Supernode object with node_ids and functional_role
        - node_data: Full data for each node (explanations, max activations, top logits)
        - prompt: Original input prompt
        - target_logit: Target output token

        Returns: Label string (2-5 words)
        """
        labeling_prompt = self._create_labeling_prompt(supernode, node_data, prompt, target_logit)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": labeling_prompt}
                ]
            )

            label = message.content[0].text.strip()
            # Ensure label is concise (truncate if too long)
            words = label.split()
            if len(words) > 5:
                label = " ".join(words[:5])

            return label
        except Exception as e:
            # Fallback to generic label
            return f"{supernode.functional_role} (layers {supernode.layer_range[0]}-{supernode.layer_range[1]})"

    def _create_labeling_prompt(self, supernode: Supernode, node_data: Dict, prompt: str, target_logit: str) -> str:
        """
        Create a prompt that includes:
        - Functional role of the supernode
        - Individual node explanations
        - Max activating examples for each node
        - Top promoted/suppressed tokens
        - Position in the computational pathway

        Ask LLM to synthesize a concise label
        """
        # Gather feature details
        feature_details = []
        for node_id in supernode.node_ids:
            node = node_data.get(node_id, {})
            explanation = node.get('explanation', 'No explanation')
            layer = node.get('layer', '?')
            feature_idx = node.get('feature_index', '?')

            detail = f"- Layer {layer}, Feature {feature_idx}: {explanation}"

            # Add top logits if available
            top_logits = node.get('top_logits', [])
            if top_logits:
                logit_str = ", ".join([f"{t['token']} ({t['value']:.2f})" for t in top_logits[:3]])
                detail += f"\n  Top logits: {logit_str}"

            feature_details.append(detail)

        feature_details_str = "\n".join(feature_details)

        return LABELING_PROMPT.format(
            prompt=prompt,
            target_logit=target_logit,
            functional_role=supernode.functional_role,
            layer_min=supernode.layer_range[0],
            layer_max=supernode.layer_range[1],
            feature_details=feature_details_str
        )
