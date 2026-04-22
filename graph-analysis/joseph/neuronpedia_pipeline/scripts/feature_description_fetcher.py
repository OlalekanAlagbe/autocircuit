"""
Feature Description Fetcher
Fetches human-readable feature descriptions from Neuronpedia API
"""

import requests
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, List

class FeatureDescriptionFetcher:
    """Fetch feature descriptions from Neuronpedia API"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize fetcher with config

        Args:
            config_path: Path to neuronpedia_config.yaml (optional)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "neuronpedia_config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.api_key = self.config['api']['api_key']
        self.base_url = self.config['api']['base_url']
        self.rate_limit_delay = self.config['api']['rate_limit_delay']
        self.model_name = self.config['model']['name']

        # Cache to avoid duplicate API calls
        self.cache: Dict[str, Optional[str]] = {}

        print(f"[FeatureDescriptionFetcher] Initialized for model: {self.model_name}")

    def parse_feature_id(self, feature_id: str) -> tuple:
        """
        Parse feature ID from format 'layer_featurenum_activation' to (layer, feature)

        Args:
            feature_id: Feature ID like "0_41_1" or "15_376262_150"

        Returns:
            Tuple of (layer_number, feature_number)
        """
        parts = feature_id.split('_')
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"Invalid feature ID format: {feature_id}")

    def get_sae_layer_name(self, layer_num: int, use_transcoder: bool = True) -> str:
        """
        Get SAE/Transcoder layer name for Neuronpedia API

        Circuit Tracer data uses transcoders, not traditional SAEs!

        Args:
            layer_num: Layer number (0-25)
            use_transcoder: If True, use transcoder format (for Circuit Tracer data)
                          If False, use SAE res format (for traditional SAE features)

        Returns:
            Layer name like "0-gemmascope-transcoder-16k" or "0-gemmascope-res-16k"
        """
        layer_type = "transcoder" if use_transcoder else "res"
        return f"{layer_num}-gemmascope-{layer_type}-16k"

    def fetch_feature_description(self, layer: int, feature: int) -> Optional[str]:
        """
        Fetch description for a single feature from Neuronpedia API

        Args:
            layer: Layer number (0-25 for GEMMA, 0-35 for QWEN)
            feature: Raw feature index from circuit tracer (will be mapped via % 16384)

        Returns:
            Human-readable description string or None if not available
        """
        # CRITICAL: Map circuit tracer feature ID to Neuronpedia feature ID
        # The SAE has 16384 features, so: neuronpedia_id = circuit_tracer_id % 16384
        feature_np = feature % 16384

        cache_key = f"{layer}_{feature_np}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        try:
            sae_layer = self.get_sae_layer_name(layer)
            url = f"{self.base_url}/feature/{self.model_name}/{sae_layer}/{feature_np}"

            headers = {
                'x-api-key': self.api_key
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract description from explanations AND activation tokens
                # Priority: Use activation tokens (more concrete) over explanations (often generic)
                pos_str = data.get('pos_str', [])
                if pos_str and len(pos_str) >= 3:
                    # Clean up tokens by removing special characters
                    clean_tokens = []
                    for token in pos_str[:5]:
                        # Remove common prefixes
                        clean = token.replace('▁', ' ').replace('_', ' ').strip()
                        if clean and len(clean) > 1:
                            clean_tokens.append(clean)

                    if clean_tokens:
                        description = f"Activates on: {', '.join(clean_tokens[:3])}"
                        self.cache[cache_key] = description
                        return description

                # Fallback to explanation if tokens aren't useful
                explanations = data.get('explanations', [])
                if explanations and len(explanations) > 0:
                    description = explanations[0].get('description', None)
                    self.cache[cache_key] = description
                    return description
                else:
                    self.cache[cache_key] = None
                    return None
            else:
                print(f"[WARNING] API returned status {response.status_code} for L{layer}_F{feature} (NP: {feature_np})")
                self.cache[cache_key] = None
                return None

        except Exception as e:
            print(f"[ERROR] Failed to fetch L{layer}_F{feature}: {e}")
            self.cache[cache_key] = None
            return None

    def fetch_descriptions_for_features(self, feature_ids: List[str], max_features: int = 50) -> Dict[str, Optional[str]]:
        """
        Fetch descriptions for multiple features

        Args:
            feature_ids: List of feature IDs in format "layer_feature_activation"
            max_features: Maximum number of features to fetch (to avoid rate limits)

        Returns:
            Dictionary mapping feature_id -> description
        """
        descriptions = {}

        # Limit to max_features to avoid hitting rate limits
        features_to_fetch = feature_ids[:max_features]

        print(f"[FeatureDescriptionFetcher] Fetching descriptions for {len(features_to_fetch)} features...")

        for i, feature_id in enumerate(features_to_fetch, 1):
            try:
                layer, feature = self.parse_feature_id(feature_id)
                description = self.fetch_feature_description(layer, feature)
                descriptions[feature_id] = description

                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(features_to_fetch)} features fetched")

            except ValueError as e:
                print(f"[WARNING] Skipping invalid feature ID: {feature_id} ({e})")
                descriptions[feature_id] = None

        print(f"[FeatureDescriptionFetcher] Completed! Fetched {len([d for d in descriptions.values() if d])} descriptions")

        return descriptions

    def get_top_features_per_layer_group(self, layer_groups: Dict, top_n: int = 3) -> Dict[str, List[tuple]]:
        """
        Get top N features per layer group with descriptions

        Args:
            layer_groups: Layer groups dict from analysis
            top_n: Number of top features per group

        Returns:
            Dictionary mapping group_name -> [(feature_id, activation, description), ...]
        """
        result = {}

        for group_name, group_data in layer_groups.items():
            top_features = group_data.get('top_features', [])[:top_n]

            features_with_descriptions = []
            for feature_data in top_features:
                # Handle both 'feature' and 'node_id' keys for backward compatibility
                feature_id = feature_data.get('feature') or feature_data.get('node_id')
                activation = feature_data['activation']

                layer, feature = self.parse_feature_id(feature_id)
                description = self.fetch_feature_description(layer, feature)

                features_with_descriptions.append((feature_id, activation, description))

            result[group_name] = features_with_descriptions

        return result


if __name__ == "__main__":
    # Test the fetcher
    fetcher = FeatureDescriptionFetcher()

    # Test single feature
    print("\nTesting single feature fetch:")
    print(f"  Raw ID: 41, NP ID: {41 % 16384}")
    desc = fetcher.fetch_feature_description(0, 41)
    print(f"L0_F41: {desc}")

    # Test multiple features (raw circuit tracer IDs - will be mapped via % 16384)
    print("\nTesting multiple feature fetch:")
    test_features = ["0_41_1", "15_376262_150", "24_88478228_120"]
    for fid in test_features:
        parts = fid.split('_')
        raw = int(parts[1])
        print(f"  {fid}: raw={raw}, NP={raw % 16384}")
    descriptions = fetcher.fetch_descriptions_for_features(test_features)

    for fid, desc in descriptions.items():
        print(f"{fid}: {desc}")
