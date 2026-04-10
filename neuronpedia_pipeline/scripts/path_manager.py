"""
Path Manager - Centralized file organization for Neuronpedia Pipeline

This module provides a single source of truth for all file paths and folder structure.
All scripts should import and use PathManager instead of hardcoding paths.

Standard Structure:
    data/prompts/{prompt-slug}/
        1_generation/raw_graph.json, metadata.json
        2_conversion/converted_graph.json, conversion_stats.json
        3_analysis/circuit_analysis.json, supernodes.json, layer_groups.json
        4_visualizations/*.png
        7_minimal_pathways/minimal_circuit.json, pathway_comparison.png
        8_evolution/evolution_data.json, evolution_heatmap.png
        9_steering/steering_targets.json, targets_viz.png
        10_polysemanticity/purity_scores.json, purity_viz.png

    data/cross_analysis/{analysis-name}/
        metadata.json
        feature_overlap.json
        supernode_matches.json
        consensus_supernodes.json
        *.png
"""

from pathlib import Path
import re
import json
from datetime import datetime


class PathManager:
    """Centralized path management for the pipeline"""

    def __init__(self, base_dir=None):
        """
        Args:
            base_dir: Base directory for data (defaults to neuronpedia_pipeline/data)
        """
        if base_dir is None:
            # Auto-detect: assume we're in scripts/ folder
            script_dir = Path(__file__).parent
            self.base_dir = script_dir.parent / 'data'
        else:
            self.base_dir = Path(base_dir)

        self.prompts_dir = self.base_dir / 'prompts'
        self.cross_analysis_dir = self.base_dir / 'cross_analysis'

        # Ensure base directories exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.cross_analysis_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def slugify(text: str) -> str:
        """
        Convert prompt text to a clean slug

        Examples:
            "The capitol of Texas is" -> "the-capitol-of-texas-is"
            "Dallas is located in the state of" -> "dallas-is-located-in-the-state-of"
            "<bos>The capitol..." -> "the-capitol..."
        """
        # Remove <bos> and <|im_end|> tags
        text = text.replace('<bos>', '').replace('<|im_end|>', '').strip()

        # Convert to lowercase
        text = text.lower()

        # Replace spaces and special chars with hyphens
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)

        # Remove leading/trailing hyphens
        text = text.strip('-')

        # Limit length (max 50 chars)
        if len(text) > 50:
            text = text[:50].rstrip('-')

        return text

    def get_prompt_dir(self, prompt: str, model_id: str = None) -> Path:
        """
        Get main directory for a prompt

        Args:
            prompt: The prompt text
            model_id: Optional model ID to include in path (e.g., 'gemma-2-2b', 'qwen3-4b')
        """
        slug = self.slugify(prompt)

        # If model_id provided, prepend it to the slug
        if model_id:
            # Clean model_id (remove special chars)
            clean_model = re.sub(r'[^\w-]', '-', model_id.lower())
            slug = f"{clean_model}_{slug}"

        prompt_dir = self.prompts_dir / slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        return prompt_dir

    def get_step_dir(self, prompt: str, step: int, step_name: str, model_id: str = None) -> Path:
        """
        Get directory for a pipeline step

        Args:
            prompt: The prompt text
            step: Step number (1-10)
            step_name: Step name (e.g., 'generation', 'conversion')
            model_id: Optional model ID to include in path

        Returns:
            Path to step directory
        """
        prompt_dir = self.get_prompt_dir(prompt, model_id=model_id)
        step_dir = prompt_dir / f"{step}_{step_name}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def _get_file_prefix(self, prompt: str, model_id: str = None) -> str:
        """
        Generate a descriptive file prefix from model and prompt

        Examples:
            "gemma-2-2b_paris-is-capital"
            "qwen3-4b_5-plus-6"
        """
        # Get short prompt slug (first 3-4 words)
        words = re.sub(r'[^\w\s-]', '', prompt.lower()).split()[:4]
        short_slug = '-'.join(words)

        if model_id:
            clean_model = re.sub(r'[^\w-]', '-', model_id.lower())
            return f"{clean_model}_{short_slug}"
        return short_slug

    # ========== Step 1: Generation ==========
    def generation_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 1, 'generation', model_id=model_id)

    def raw_graph_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.generation_dir(prompt, model_id=model_id) / f'{prefix}_raw_graph.json'

    def metadata_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.generation_dir(prompt, model_id=model_id) / f'{prefix}_metadata.json'

    # ========== Step 2: Conversion ==========
    def conversion_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 2, 'conversion', model_id=model_id)

    def converted_graph_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.conversion_dir(prompt, model_id=model_id) / f'{prefix}_converted_graph.json'

    def conversion_stats_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.conversion_dir(prompt, model_id=model_id) / f'{prefix}_conversion_stats.json'

    # ========== Step 3: Analysis ==========
    def analysis_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 3, 'analysis', model_id=model_id)

    def circuit_analysis_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.analysis_dir(prompt, model_id=model_id) / f'{prefix}_circuit_analysis.json'

    def supernodes_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.analysis_dir(prompt, model_id=model_id) / f'{prefix}_supernodes.json'

    def layer_groups_path(self, prompt: str, model_id: str = None) -> Path:
        prefix = self._get_file_prefix(prompt, model_id)
        return self.analysis_dir(prompt, model_id=model_id) / f'{prefix}_layer_groups.json'

    # ========== Step 4: Visualizations ==========
    def visualizations_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 4, 'visualizations', model_id=model_id)

    def visualization_path(self, prompt: str, viz_name: str, model_id: str = None) -> Path:
        """
        Get path for a specific visualization

        Args:
            prompt: The prompt text
            viz_name: Name like 'supernode_overview', 'layer_distribution', etc.
            model_id: Optional model ID to include in path
        """
        return self.visualizations_dir(prompt, model_id=model_id) / f"{viz_name}.png"

    # ========== Step 7: Minimal Pathways ==========
    def minimal_pathways_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 7, 'minimal_pathways', model_id=model_id)

    def minimal_circuit_path(self, prompt: str, model_id: str = None) -> Path:
        return self.minimal_pathways_dir(prompt, model_id=model_id) / 'minimal_circuit.json'

    def pathway_comparison_path(self, prompt: str, model_id: str = None) -> Path:
        return self.minimal_pathways_dir(prompt, model_id=model_id) / 'pathway_comparison.png'

    def minimal_pathways_result_path(self, prompt: str, model_id: str = None) -> Path:
        return self.minimal_pathways_dir(prompt, model_id=model_id) / 'minimal_pathway.json'

    def minimal_pathways_viz_path(self, prompt: str, model_id: str = None) -> Path:
        return self.minimal_pathways_dir(prompt, model_id=model_id) / 'minimal_pathway_comparison.png'

    def minimal_pathways_structure_viz_path(self, prompt: str, model_id: str = None) -> Path:
        return self.minimal_pathways_dir(prompt, model_id=model_id) / 'minimal_pathway_structure.png'

    # ========== Step 8: Evolution ==========
    def evolution_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 8, 'evolution', model_id=model_id)

    def evolution_data_path(self, prompt: str, model_id: str = None) -> Path:
        return self.evolution_dir(prompt, model_id=model_id) / 'evolution_data.json'

    def evolution_report_path(self, prompt: str, model_id: str = None) -> Path:
        return self.evolution_dir(prompt, model_id=model_id) / 'evolution_report.txt'

    def evolution_viz_path(self, prompt: str, model_id: str = None) -> Path:
        return self.evolution_dir(prompt, model_id=model_id) / 'evolution_heatmap.png'

    # ========== Step 9: Steering ==========
    def steering_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 9, 'steering', model_id=model_id)

    def steering_targets_path(self, prompt: str, model_id: str = None) -> Path:
        return self.steering_dir(prompt, model_id=model_id) / 'steering_targets.json'

    def steering_viz_path(self, prompt: str, model_id: str = None) -> Path:
        return self.steering_dir(prompt, model_id=model_id) / 'targets_viz.png'

    # ========== Step 10: Polysemanticity ==========
    def polysemanticity_dir(self, prompt: str, model_id: str = None) -> Path:
        return self.get_step_dir(prompt, 10, 'polysemanticity', model_id=model_id)

    def polysemanticity_data_path(self, prompt: str, model_id: str = None) -> Path:
        return self.polysemanticity_dir(prompt, model_id=model_id) / 'purity_scores.json'

    def polysemanticity_report_path(self, prompt: str, model_id: str = None) -> Path:
        return self.polysemanticity_dir(prompt, model_id=model_id) / 'purity_report.txt'

    def polysemanticity_viz_path(self, prompt: str, model_id: str = None) -> Path:
        return self.polysemanticity_dir(prompt, model_id=model_id) / 'purity_viz.png'

    # ========== Cross-Analysis ==========
    def get_cross_analysis_dir(self, analysis_name: str) -> Path:
        """
        Get directory for cross-analysis

        Args:
            analysis_name: e.g., 'geography-domain-5prompts', 'politics-vs-geography'
        """
        cross_dir = self.cross_analysis_dir / analysis_name
        cross_dir.mkdir(parents=True, exist_ok=True)
        return cross_dir

    def cross_analysis_metadata_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'metadata.json'

    def cross_analysis_overlap_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'feature_overlap.json'

    def cross_analysis_matches_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'supernode_matches.json'

    def cross_analysis_consensus_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'consensus_supernodes.json'

    def cross_analysis_report_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'analysis_report.txt'

    def cross_analysis_heatmap_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'overlap_heatmap.png'

    def cross_analysis_network_path(self, analysis_name: str) -> Path:
        return self.get_cross_analysis_dir(analysis_name) / 'reuse_network.png'

    # ========== Utility Methods ==========
    def save_metadata(self, prompt: str, metadata: dict, model_id: str = None):
        """Save metadata for a prompt"""
        metadata['generated_at'] = datetime.now().isoformat()
        metadata['prompt'] = prompt
        metadata['slug'] = self.slugify(prompt)

        with open(self.metadata_path(prompt, model_id=model_id), 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, prompt: str) -> dict:
        """Load metadata for a prompt"""
        path = self.metadata_path(prompt)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def get_all_analyzed_prompts(self) -> list:
        """Get list of all prompts that have been analyzed"""
        analyzed = []
        for prompt_dir in self.prompts_dir.iterdir():
            if prompt_dir.is_dir():
                analysis_dir = prompt_dir / '3_analysis'
                if analysis_dir.exists():
                    analysis_files = list(analysis_dir.glob('*circuit_analysis.json'))
                    if analysis_files:
                        metadata = self.load_metadata_from_dir(prompt_dir)
                        if metadata:
                            analyzed.append(metadata['prompt'])
        return analyzed

    def load_metadata_from_dir(self, prompt_dir: Path) -> dict:
        """Load metadata from a specific directory"""
        gen_dir = prompt_dir / '1_generation'
        if gen_dir.exists():
            metadata_files = list(gen_dir.glob('*metadata.json'))
            if metadata_files:
                with open(metadata_files[0]) as f:
                    return json.load(f)
        return {}

    def find_analysis_for_prompts(self, prompts: list) -> list:
        """
        Find circuit_analysis.json files for given prompts

        Args:
            prompts: List of prompt strings

        Returns:
            List of Path objects to circuit_analysis.json files
        """
        paths = []
        for prompt in prompts:
            analysis_path = self.circuit_analysis_path(prompt)
            if analysis_path.exists():
                paths.append(analysis_path)
        return paths


# Global instance for easy importing
path_manager = PathManager()


if __name__ == "__main__":
    # Test the path manager
    pm = PathManager()

    test_prompts = [
        "The capitol of Texas is",
        "Dallas is located in the state of",
        "<bos>Houston is a city in"
    ]

    print("=== Testing Path Manager ===\n")

    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print(f"  Slug: {pm.slugify(prompt)}")
        print(f"  Prompt dir: {pm.get_prompt_dir(prompt)}")
        print(f"  Raw graph: {pm.raw_graph_path(prompt)}")
        print(f"  Analysis: {pm.circuit_analysis_path(prompt)}")
        print()

    print("=== Cross-Analysis Paths ===")
    analysis_name = "geography-domain-5prompts"
    print(f"Analysis: {analysis_name}")
    print(f"  Dir: {pm.get_cross_analysis_dir(analysis_name)}")
    print(f"  Report: {pm.cross_analysis_report_path(analysis_name)}")
    print(f"  Consensus: {pm.cross_analysis_consensus_path(analysis_name)}")
