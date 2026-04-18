"""
Steering Analysis - Circuit Intervention Planning
Identifies intervention targets and predicts their effects

Phase 1: Circuit-Level Prediction (Implemented Now)
- Rank features by predicted intervention impact
- Generate steering vectors from supernode activation patterns
- Predict output changes from graph structure

Phase 2: Model Execution (Design for Future)
- Actually run interventions on the model
- Measure real output changes
- Validate predictions
"""

import json
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

# Add scripts to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from path_manager import PathManager
from collections import defaultdict


class SteeringAnalyzer:
    """Analyze circuit intervention targets and predict effects"""

    def __init__(self, analysis_path):
        """
        Args:
            analysis_path: Path to analysis JSON file
        """
        with open(analysis_path) as f:
            self.analysis = json.load(f)

        self.metadata = self.analysis['metadata']
        self.prompt = self.metadata['prompt']

        # Get supernodes and features
        self.supernodes = self.analysis.get('louvain_supernodes', {}).get('supernodes', {})
        self.layer_groups = self.analysis.get('layer_groups', {})
        self.flow_analysis = self.analysis.get('flow_analysis', {})

        print(f"[OK] Loaded circuit: {self.prompt}")
        print(f"  Supernodes: {len(self.supernodes)}")
        print(f"  Layer groups: {len(self.layer_groups)}")

    def rank_intervention_targets(self, top_k=20):
        """
        Rank features by predicted intervention impact

        Impact score based on:
        1. Activation strength (higher = more active)
        2. Betweenness centrality (higher = more critical to information flow)
        3. Layer position (middle layers = knowledge retrieval)

        Returns:
            List of (feature_id, impact_score, metadata) tuples
        """
        targets = []

        # Get bottleneck features (high betweenness)
        bottleneck_features = self.flow_analysis.get('bottleneck_features', [])

        for feature_data in bottleneck_features[:top_k]:
            feature_id = feature_data.get('feature') or feature_data.get('node_id')
            betweenness = feature_data.get('betweenness', 0)
            layer = feature_data.get('layer', 0)

            # Impact score: betweenness * layer_weight
            # Middle layers (11-15) get higher weight
            if 11 <= layer <= 15:
                layer_weight = 2.0  # Knowledge retrieval layer
            elif 6 <= layer <= 20:
                layer_weight = 1.5  # Processing layers
            else:
                layer_weight = 1.0  # Input/output layers

            impact_score = betweenness * layer_weight * 100

            targets.append({
                'feature_id': feature_id,
                'impact_score': impact_score,
                'betweenness': betweenness,
                'layer': layer,
                'layer_weight': layer_weight
            })

        # Sort by impact score
        targets.sort(key=lambda x: x['impact_score'], reverse=True)

        return targets

    def rank_supernode_targets(self):
        """
        Rank supernodes by predicted intervention impact

        Returns:
            List of (supernode_id, impact_score, metadata) tuples
        """
        targets = []

        for sn_id, sn_data in self.supernodes.items():
            size = sn_data.get('size', 0)
            mean_activation = sn_data.get('mean_activation', 0)
            max_activation = sn_data.get('max_activation', 0)
            layers = sn_data.get('layers', [])

            # Skip empty or single-layer supernodes
            if not layers or len(layers) == 1:
                continue

            # Check if spans middle layers (knowledge retrieval)
            spans_middle = any(11 <= l <= 15 for l in layers)

            # Impact score: size * activation * layer_bonus
            layer_bonus = 2.0 if spans_middle else 1.0
            impact_score = size * mean_activation * layer_bonus

            targets.append({
                'supernode_id': int(sn_id),
                'impact_score': impact_score,
                'size': size,
                'mean_activation': mean_activation,
                'max_activation': max_activation,
                'layers': layers,
                'spans_middle': spans_middle
            })

        # Sort by impact score
        targets.sort(key=lambda x: x['impact_score'], reverse=True)

        return targets

    def generate_steering_vector(self, supernode_id):
        """
        Generate steering vector from supernode activation pattern

        Returns:
            dict with steering vector information
        """
        if str(supernode_id) not in self.supernodes:
            return None

        sn_data = self.supernodes[str(supernode_id)]
        nodes = sn_data.get('nodes', [])

        # Extract layer-wise activations
        layer_activations = defaultdict(list)

        for node_id in nodes:
            # Parse node format: "layer_feature_ctx"
            parts = node_id.split('_')
            if len(parts) >= 2:
                layer = int(parts[0])
                # We don't have actual activations in the analysis file
                # So we'll use mean_activation as a proxy
                layer_activations[layer].append(sn_data.get('mean_activation', 0))

        # Create activation pattern per layer
        activation_pattern = {}
        for layer, acts in layer_activations.items():
            activation_pattern[layer] = {
                'mean': np.mean(acts),
                'max': np.max(acts),
                'num_features': len(acts)
            }

        steering_vector = {
            'supernode_id': supernode_id,
            'num_features': len(nodes),
            'layers': sn_data.get('layers', []),
            'activation_pattern': activation_pattern,
            'mean_activation': sn_data.get('mean_activation', 0),
            'max_activation': sn_data.get('max_activation', 0)
        }

        return steering_vector

    def compute_intervention_confidence(self, target_type, target_id, has_description=False):
        """
        Compute confidence level for intervention prediction

        Args:
            target_type: 'feature' or 'supernode'
            target_id: ID of target
            has_description: Whether target has Neuronpedia description

        Returns:
            str: 'HIGH', 'MEDIUM', or 'LOW'
        """
        confidence_score = 0.0

        if target_type == 'supernode':
            sn_data = self.supernodes.get(str(target_id), {})
            layers = sn_data.get('layers', [])

            # Factor 1: Graph centrality (30 points)
            # Supernodes in middle layers (11-15) are more predictable
            if any(11 <= l <= 15 for l in layers):
                confidence_score += 0.30
            elif any(6 <= l <= 20 for l in layers):
                confidence_score += 0.15
            else:
                confidence_score += 0.05  # Input/output less predictable

            # Factor 2: Feature description available (30 points)
            if has_description:
                confidence_score += 0.30

            # Factor 3: Supernode size (consistency indicator, 20 points)
            size = sn_data.get('size', 0)
            if 10 <= size <= 50:  # Good size - not too small/large
                confidence_score += 0.20
            elif size > 5:
                confidence_score += 0.10

            # Factor 4: Activation strength (20 points)
            activation = sn_data.get('mean_activation', 0)
            if activation > 10:  # Strong activation = more reliable
                confidence_score += 0.20
            elif activation > 5:
                confidence_score += 0.10

        # Categorize confidence
        if confidence_score >= 0.65:
            return "HIGH"
        elif confidence_score >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_intervention_effect(self, target_type, target_id, intervention_type):
        """
        Predict effect of intervening on a target

        Args:
            target_type: 'feature' or 'supernode'
            target_id: ID of the target
            intervention_type: 'amplify' (2x), 'ablate' (0x), 'clamp' (fixed)

        Returns:
            dict with predicted effect information
        """
        if target_type == 'supernode':
            if str(target_id) not in self.supernodes:
                return None

            sn_data = self.supernodes[str(target_id)]
            size = sn_data.get('size', 0)
            mean_activation = sn_data.get('mean_activation', 0)
            layers = sn_data.get('layers', [])

            # Predict effect based on graph structure
            # Assumption: Middle-layer supernodes have higher impact
            if any(11 <= l <= 15 for l in layers):
                base_impact = 0.8  # High impact
            elif any(6 <= l <= 20 for l in layers):
                base_impact = 0.5  # Medium impact
            else:
                base_impact = 0.2  # Low impact

            # Adjust by size and activation
            normalized_size = min(size / 100, 1.0)  # Cap at 100 features
            normalized_activation = min(mean_activation / 50, 1.0)  # Cap at 50

            predicted_impact = base_impact * (0.5 + 0.25 * normalized_size + 0.25 * normalized_activation)

            # Intervention type modifies impact
            if intervention_type == 'amplify':
                multiplier = 1.5  # Amplifying has more effect
            elif intervention_type == 'ablate':
                multiplier = 2.0  # Ablating has strongest effect
            else:  # clamp
                multiplier = 1.0

            predicted_impact *= multiplier

            # Compute confidence based on structural properties
            # Check if description exists in analysis data
            has_description = False
            if 'feature_descriptions' in self.analysis:
                for desc_category in self.analysis['feature_descriptions'].values():
                    if isinstance(desc_category, dict) and str(target_id) in desc_category:
                        has_description = True
                        break

            confidence = self.compute_intervention_confidence(target_type, target_id, has_description)

            return {
                'target_type': target_type,
                'target_id': target_id,
                'intervention_type': intervention_type,
                'predicted_impact': predicted_impact,
                'confidence': confidence,
                'explanation': f"Predicted {predicted_impact:.1%} change in output probability based on graph structure ({confidence} confidence)"
            }

        return None

    def analyze_steering_targets(self, output_path):
        """Main method to analyze steering targets"""

        print("\n" + "="*60)
        print("STEP 1: Rank Feature-Level Targets")
        print("="*60)
        feature_targets = self.rank_intervention_targets(top_k=15)

        print("Top intervention targets (by predicted impact):")
        for i, target in enumerate(feature_targets[:5], 1):
            print(f"  {i}. {target['feature_id']} (L{target['layer']})")
            print(f"     Impact: {target['impact_score']:.2f}, Betweenness: {target['betweenness']:.4f}")

        print("\n" + "="*60)
        print("STEP 2: Rank Supernode-Level Targets")
        print("="*60)
        supernode_targets = self.rank_supernode_targets()

        print("Top supernode targets:")
        for i, target in enumerate(supernode_targets[:5], 1):
            print(f"  {i}. SN{target['supernode_id']} ({target['size']} features)")
            print(f"     Impact: {target['impact_score']:.2f}, Activation: {target['mean_activation']:.1f}")
            print(f"     Layers: {min(target['layers'])}-{max(target['layers'])}, Spans Middle: {target['spans_middle']}")

        print("\n" + "="*60)
        print("STEP 3: Generate Steering Vectors")
        print("="*60)
        steering_vectors = []

        for target in supernode_targets[:3]:
            sn_id = target['supernode_id']
            vector = self.generate_steering_vector(sn_id)
            if vector:
                steering_vectors.append(vector)
                print(f"[OK] Generated steering vector for SN{sn_id}")

        print("\n" + "="*60)
        print("STEP 4: Predict Intervention Effects")
        print("="*60)

        predictions = []
        for target in supernode_targets[:5]:
            sn_id = target['supernode_id']

            # Predict for different intervention types
            for intervention_type in ['amplify', 'ablate']:
                effect = self.predict_intervention_effect('supernode', sn_id, intervention_type)
                if effect:
                    predictions.append(effect)

        print("Predicted effects:")
        for pred in predictions[:5]:
            print(f"  {pred['intervention_type'].upper()} SN{pred['target_id']}: {pred['predicted_impact']:.1%} impact")

        # Save results
        result = {
            'metadata': self.metadata,
            'feature_targets': feature_targets,
            'supernode_targets': supernode_targets,
            'steering_vectors': steering_vectors,
            'intervention_predictions': predictions,
            'note': 'Phase 1: Circuit-level predictions. Phase 2 (model execution) not yet implemented.'
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n[OK] Steering analysis saved to: {output_path}")
        return result

    def visualize_intervention_targets(self, result, output_path):
        """Create visualization of intervention targets"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Chart 1: Top feature targets by impact
        feature_targets = result['feature_targets'][:10]
        features = [t['feature_id'].split('_')[1][:6] + f"..L{t['layer']}" for t in feature_targets]
        impacts = [t['impact_score'] for t in feature_targets]

        ax1.barh(range(len(features)), impacts, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=9)
        ax1.set_xlabel('Predicted Impact Score', fontsize=11, fontweight='bold')
        ax1.set_title('Top 10 Feature Intervention Targets', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Chart 2: Supernode targets by impact
        supernode_targets = result['supernode_targets'][:8]
        supernodes = [f"SN{t['supernode_id']}\n({t['size']} feat)" for t in supernode_targets]
        sn_impacts = [t['impact_score'] for t in supernode_targets]

        colors = ['coral' if t['spans_middle'] else 'lightblue' for t in supernode_targets]

        ax2.bar(range(len(supernodes)), sn_impacts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(supernodes)))
        ax2.set_xticklabels(supernodes, fontsize=9, rotation=45, ha='right')
        ax2.set_ylabel('Predicted Impact Score', fontsize=11, fontweight='bold')
        ax2.set_title('Top 8 Supernode Intervention Targets', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add legend for middle-layer highlight
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='coral', alpha=0.7, label='Spans Middle Layers (L11-15)'),
            Patch(facecolor='lightblue', alpha=0.7, label='Other Layers')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.suptitle(f'Steering Intervention Targets\n"{self.prompt}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Visualization saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze circuit steering targets')
    parser.add_argument('--analysis', type=str,
                       help='Path to analysis JSON file (if not provided, uses latest)')
    parser.add_argument('--prompt', type=str,
                       help='Prompt text to analyze')

    args = parser.parse_args()

    pm = PathManager()

    # Auto-discover analysis file
    if args.analysis:
        analysis_path = Path(args.analysis)
    elif args.prompt:
        analysis_path = pm.circuit_analysis_path(args.prompt)
        if not analysis_path.exists():
            print(f"[ERROR] Analysis not found for prompt: {args.prompt}")
            return
    else:
        # Find available analyses and let user choose
        prompts = pm.get_all_analyzed_prompts()
        if not prompts:
            print("[ERROR] No analysis files found!")
            return

        # Display available prompts
        print("\nAvailable circuit analyses:")
        print("=" * 60)
        for i, prompt_text in enumerate(prompts, 1):
            analysis_file = pm.circuit_analysis_path(prompt_text)
            size_mb = analysis_file.stat().st_size / (1024 * 1024)
            print(f"{i}. {prompt_text} ({size_mb:.2f} MB)")
        print("=" * 60)

        # Get user selection
        print("\nSelect analysis to process:")
        print("-" * 60)

        while True:
            try:
                choice = input("Enter number (or press Enter for latest): ").strip()

                if choice == "":
                    prompt = prompts[0]
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(prompts):
                    prompt = prompts[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(prompts)}")
            except ValueError:
                print("Please enter a valid number or press Enter for latest")

        analysis_path = pm.circuit_analysis_path(prompt)
        print(f"\n[INFO] Selected: {prompt}")

    # Extract prompt
    with open(analysis_path) as f:
        analysis = json.load(f)
        prompt = analysis['metadata']['prompt']

    # Create output directory
    output_dir = pm.steering_dir(prompt)
    print(f"\nOutput directory: {output_dir}")

    # Analyze steering targets
    analyzer = SteeringAnalyzer(analysis_path)

    analysis_path = output_dir / "steering_analysis.json"
    result = analyzer.analyze_steering_targets(analysis_path)

    # Create visualization
    viz_path = output_dir / "steering_targets_viz.png"
    analyzer.visualize_intervention_targets(result, viz_path)

    print("\n" + "="*60)
    print("STEERING ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analysis: {analysis_path}")
    print(f"Visualization: {viz_path}")
    print("\nPhase 1 (Circuit-Level Prediction): COMPLETE")
    print("Phase 2 (Model Execution): Not yet implemented")


if __name__ == "__main__":
    main()
