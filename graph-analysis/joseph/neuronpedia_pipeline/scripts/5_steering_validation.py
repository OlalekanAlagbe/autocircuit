#!/usr/bin/env python3
"""
Stage 3: Steering Validation of Cross-Circuit Bottleneck Features

Tests whether amplifying or suppressing bottleneck features (identified via
cross-circuit analysis in Stage 1.5) changes model output via the Neuronpedia
Steering API.

Research question: Do universal bottleneck features have predictable steering
effects, and does cross-circuit frequency predict steering impact?

Usage:
  python 5_steering_validation.py                         # Full run
  python 5_steering_validation.py --quick                 # Top 5 features, 2 circuits (~30 calls)
  python 5_steering_validation.py --feature L3_F5150441   # Single feature test
  python 5_steering_validation.py --analyze-only          # Analyze existing results
  python 5_steering_validation.py --resume                # Resume interrupted run
"""

import argparse
import json
import sys
import time
import yaml
import math
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Windows UTF-8
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
PROMPTS_DIR = DATA_DIR / 'prompts'
CONFIG_PATH = BASE_DIR / 'config' / 'neuronpedia_config.yaml'
BOTTLENECK_LIBRARY = DATA_DIR / 'stage_1_5_bottleneck_library.json'
OUTPUT_DIR = DATA_DIR / 'stage_3_steering'

# Constants
from pipeline_constants import GEMMA_SAE_DICT_SIZE, GEMMA_TOTAL_LAYERS

GEMMA_MODEL_ID = 'gemma-2-2b'
SAE_PATH_TEMPLATE = '{layer}-gemmascope-transcoder-16k'
STEERING_STRENGTHS = [-50, -20, -10, 10, 20, 50]
N_TOKENS = 10
TEMPERATURE = 0.0
SEED = 42


# ============================================================================
# SteeringClient: Wraps Neuronpedia /api/steer endpoint
# ============================================================================

class SteeringClient:
    """Calls the Neuronpedia Steering API with rate limiting."""

    def __init__(self, api_key: str, base_url: str = 'https://www.neuronpedia.org/api',
                 rate_limit_delay: float = 36.0):
        """
        Args:
            api_key: Neuronpedia API key
            base_url: API base URL
            rate_limit_delay: Seconds between requests (36s = 100/hr)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.call_count = 0
        self.last_call_time = 0

    def _wait_for_rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            wait = self.rate_limit_delay - elapsed
            time.sleep(wait)

    def steer(self, prompt: str, features: List[dict], model_id: str = GEMMA_MODEL_ID,
              n_tokens: int = N_TOKENS, temperature: float = TEMPERATURE,
              seed: int = SEED) -> dict:
        """
        Call the Neuronpedia steering API.

        Args:
            prompt: Input text to steer
            features: List of feature dicts with modelId, layer, index, strength
            model_id: Model identifier
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature (0 = deterministic)
            seed: Random seed for reproducibility

        Returns:
            API response dict with steered output
        """
        self._wait_for_rate_limit()

        payload = {
            'prompt': prompt,
            'modelId': model_id,
            'features': features,
            'temperature': temperature,
            'n_tokens': n_tokens,
            'seed': seed,
            'freq_penalty': 0,
            'strength_multiplier': 1,
        }

        headers = {
            'Content-Type': 'application/json',
        }
        if self.api_key:
            headers['X-Api-Key'] = self.api_key

        try:
            resp = requests.post(
                f'{self.base_url}/steer',
                json=payload,
                headers=headers,
                timeout=30
            )
            self.last_call_time = time.time()
            self.call_count += 1

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f'  [RATE LIMITED] Waiting 60s before retry...')
                time.sleep(60)
                return self.steer(prompt, features, model_id, n_tokens, temperature, seed)
            else:
                print(f'  [ERROR] Status {resp.status_code}: {resp.text[:200]}')
                return {'error': resp.status_code, 'message': resp.text[:200]}
        except requests.exceptions.Timeout:
            print(f'  [TIMEOUT] Request timed out, retrying...')
            time.sleep(5)
            return self.steer(prompt, features, model_id, n_tokens, temperature, seed)
        except Exception as e:
            print(f'  [ERROR] {e}')
            return {'error': str(e)}

    def get_baseline(self, prompt: str, model_id: str = GEMMA_MODEL_ID,
                     n_tokens: int = N_TOKENS) -> dict:
        """
        Get unsteered baseline output for a prompt.

        Note: The Neuronpedia API returns DEFAULT alongside every STEERED call,
        so baselines can be extracted from any steering result. This method
        steers a feature at strength=0 as a fallback to get a valid response.
        """
        # The API returns 500 on empty features array, so use a harmless
        # strength=0 steer to get the DEFAULT output
        sae_path = SAE_PATH_TEMPLATE.format(layer=0)
        features = [{
            'modelId': model_id,
            'layer': sae_path,
            'index': 0,
            'strength': 0,
        }]
        return self.steer(prompt, features, model_id, n_tokens)

    def steer_feature(self, prompt: str, layer: int, np_id: int, strength: float,
                      model_id: str = GEMMA_MODEL_ID) -> dict:
        """Convenience method to steer a single feature."""
        sae_path = SAE_PATH_TEMPLATE.format(layer=layer)
        features = [{
            'modelId': model_id,
            'layer': sae_path,
            'index': np_id,
            'strength': strength,
        }]
        return self.steer(prompt, features, model_id)


# ============================================================================
# ExperimentRunner: Orchestrates steering experiments
# ============================================================================

class ExperimentRunner:
    """Runs steering experiments across features and circuits."""

    def __init__(self, client: SteeringClient, quick: bool = False):
        self.client = client
        self.quick = quick
        self.bottleneck_library = {}
        self.circuit_prompts = {}  # circuit_id -> prompt text
        self.baselines = {}  # circuit_id -> baseline result
        self.results = []  # list of experiment result dicts
        self.checkpoint_path = OUTPUT_DIR / 'steering_checkpoint.json'

    def load_bottleneck_library(self):
        """Load the Stage 1.5 bottleneck library."""
        print(f'\n[1/5] Loading bottleneck library...')
        with open(BOTTLENECK_LIBRARY, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bottleneck_library = data.get('cross_circuit_features', {})
        metadata = data.get('metadata', {})
        print(f'  Loaded {len(self.bottleneck_library)} cross-circuit features')
        print(f'  Total circuits: {metadata.get("total_circuits_analyzed", "?")}')

    def load_circuit_prompts(self):
        """Extract prompts from circuit analysis files for all GEMMA circuits."""
        print(f'\n[2/5] Loading circuit prompts...')

        for circuit_dir in sorted(PROMPTS_DIR.iterdir()):
            if not circuit_dir.is_dir():
                continue
            if not circuit_dir.name.startswith('gemma-2-2b_'):
                continue

            analysis_dir = circuit_dir / '3_analysis'
            if not analysis_dir.exists():
                continue

            # Find circuit_analysis.json
            analysis_files = list(analysis_dir.glob('*_circuit_analysis.json'))
            if not analysis_files:
                continue

            try:
                with open(analysis_files[0], 'r', encoding='utf-8') as f:
                    analysis = json.load(f)

                prompt = analysis.get('metadata', {}).get('prompt', '')
                if prompt:
                    # Strip <bos> prefix for steering API
                    clean_prompt = prompt.replace('<bos>', '').strip()
                    self.circuit_prompts[circuit_dir.name] = {
                        'prompt': clean_prompt,
                        'top_prediction': analysis['metadata'].get('model_output', ''),
                        'top_probability': analysis['metadata'].get('output_probability', 0),
                        'top_predictions': analysis['metadata'].get('top_predictions', []),
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f'  [WARN] Could not load {circuit_dir.name}: {e}')

        print(f'  Found {len(self.circuit_prompts)} GEMMA circuits with prompts')
        for cid, info in self.circuit_prompts.items():
            print(f'    {cid}: "{info["prompt"]}" -> "{info["top_prediction"]}" ({info["top_probability"]*100:.1f}%)')

    def get_gemma_features(self) -> List[Tuple[str, dict]]:
        """Get GEMMA-only features sorted by cross-circuit frequency."""
        gemma_features = []
        for label, feat in self.bottleneck_library.items():
            # Check if this feature appears in any GEMMA circuit
            gemma_appearances = [a for a in feat['appearances'] if a['model'] == 'gemma-2-2b']
            if gemma_appearances:
                gemma_features.append((label, feat, gemma_appearances))

        # Sort by number of GEMMA circuits (descending)
        gemma_features.sort(key=lambda x: len(x[2]), reverse=True)
        return gemma_features

    def run_baselines(self):
        """Collect baseline (unsteered) outputs for all circuits."""
        print(f'\n[3/5] Collecting baselines ({len(self.circuit_prompts)} circuits)...')

        for circuit_id, info in self.circuit_prompts.items():
            if circuit_id in self.baselines:
                print(f'  [SKIP] {circuit_id} (already have baseline)')
                continue

            print(f'  Baseline: {circuit_id}...')
            result = self.client.get_baseline(info['prompt'])

            if 'error' not in result:
                self.baselines[circuit_id] = {
                    'prompt': info['prompt'],
                    'response': result,
                    'timestamp': datetime.now().isoformat(),
                }
                # API returns DEFAULT and STEERED keys
                output_text = result.get('DEFAULT', result.get('STEERED', result.get('output', '')))
                print(f'    -> "{output_text[:60]}"')
            else:
                print(f'    -> ERROR: {result}')

        print(f'  Baselines collected: {len(self.baselines)}/{len(self.circuit_prompts)}')

    def run_universal_bottleneck_experiments(self):
        """Test universal bottleneck features across their circuits."""
        gemma_features = self.get_gemma_features()

        if self.quick:
            # Quick mode: top 5 features, max 2 circuits each
            gemma_features = gemma_features[:5]
            max_circuits = 2
            strengths = [-20, 20]
            print(f'\n[4/5] Quick mode: {len(gemma_features)} features, {max_circuits} circuits, {len(strengths)} strengths...')
        else:
            max_circuits = None
            strengths = STEERING_STRENGTHS
            print(f'\n[4/5] Full run: {len(gemma_features)} features, all circuits, {len(strengths)} strengths...')

        total_calls = 0
        for i, (label, feat, gemma_appearances) in enumerate(gemma_features):
            layer = feat['layer']
            np_id = feat.get('np_id')
            if np_id is None:
                np_id = feat.get('circuit_id', 0) % GEMMA_SAE_DICT_SIZE

            # Get circuits this feature appears in
            circuits = [a['circuit'] for a in gemma_appearances
                        if a['circuit'] in self.circuit_prompts]

            if max_circuits:
                circuits = circuits[:max_circuits]

            if not circuits:
                continue

            print(f'\n  Feature {i+1}/{len(gemma_features)}: {label} (L{layer}, NP:{np_id}, {len(circuits)} circuits)')
            if feat.get('explanation'):
                print(f'    Explanation: {feat["explanation"][:80]}')

            for circuit_id in circuits:
                prompt = self.circuit_prompts[circuit_id]['prompt']

                for strength in strengths:
                    # Check if already done (for resume)
                    experiment_key = f'{label}_{circuit_id}_{strength}'
                    if any(r.get('experiment_key') == experiment_key for r in self.results):
                        continue

                    action = 'AMPLIFY' if strength > 0 else 'SUPPRESS'
                    print(f'    {action} {abs(strength):+d} on "{circuit_id}"...', end='', flush=True)

                    result = self.client.steer_feature(
                        prompt=prompt,
                        layer=layer,
                        np_id=np_id,
                        strength=strength,
                    )

                    if 'error' not in result:
                        output_text = result.get('STEERED', result.get('output', ''))
                        print(f' -> "{output_text[:50]}"')

                        self.results.append({
                            'experiment_key': experiment_key,
                            'feature_label': label,
                            'layer': layer,
                            'np_id': np_id,
                            'circuit_id': circuit_id,
                            'prompt': prompt,
                            'strength': strength,
                            'response': result,
                            'circuits_appeared_in': feat['circuits_appeared_in'],
                            'avg_convergence': feat.get('avg_convergence', 0),
                            'is_universal': len(gemma_appearances) >= 3,
                            'timestamp': datetime.now().isoformat(),
                        })
                        total_calls += 1
                    else:
                        print(f' -> ERROR')

            # Checkpoint after each feature
            self._save_checkpoint()

        print(f'\n  Total steering calls: {total_calls}')
        print(f'  Total API calls (including baselines): {self.client.call_count}')

    def run_prompt_specific_experiments(self):
        """Test prompt-specific features (those unique to single circuits)."""
        if self.quick:
            print(f'\n[5/5] Skipping prompt-specific experiments in quick mode')
            return

        print(f'\n[5/5] Testing prompt-specific steering targets...')

        for circuit_id, info in self.circuit_prompts.items():
            # Load this circuit's steering targets from its analysis
            circuit_dir = PROMPTS_DIR / circuit_id / '3_analysis'
            analysis_files = list(circuit_dir.glob('*_circuit_analysis.json'))
            if not analysis_files:
                continue

            with open(analysis_files[0], 'r', encoding='utf-8') as f:
                analysis = json.load(f)

            steering_targets = analysis.get('steering_targets', {})
            if not steering_targets:
                continue

            # Get top 3 by betweenness (from the node IDs in steering_targets)
            target_items = list(steering_targets.items())[:3]

            for node_id, description in target_items:
                parts = node_id.split('_')
                if len(parts) < 2:
                    continue

                layer = int(parts[0])
                raw_feature_id = int(parts[1])
                np_id = raw_feature_id % GEMMA_SAE_DICT_SIZE
                label = f'L{layer}_F{raw_feature_id}'

                # Skip if this is already a cross-circuit feature we've tested
                if label in self.bottleneck_library:
                    continue

                print(f'  Circuit-specific: {label} in {circuit_id}')
                print(f'    {description[:80]}')

                for strength in [-20, 20]:
                    experiment_key = f'specific_{label}_{circuit_id}_{strength}'
                    if any(r.get('experiment_key') == experiment_key for r in self.results):
                        continue

                    action = 'AMPLIFY' if strength > 0 else 'SUPPRESS'
                    print(f'    {action} {abs(strength):+d}...', end='', flush=True)

                    result = self.client.steer_feature(
                        prompt=info['prompt'],
                        layer=layer,
                        np_id=np_id,
                        strength=strength,
                    )

                    if 'error' not in result:
                        output_text = result.get('output', result.get('text', ''))
                        print(f' -> "{output_text[:50]}"')
                        self.results.append({
                            'experiment_key': experiment_key,
                            'feature_label': label,
                            'layer': layer,
                            'np_id': np_id,
                            'circuit_id': circuit_id,
                            'prompt': info['prompt'],
                            'strength': strength,
                            'response': result,
                            'circuits_appeared_in': 1,
                            'avg_convergence': 0,
                            'is_universal': False,
                            'description': description,
                            'timestamp': datetime.now().isoformat(),
                        })
                    else:
                        print(f' -> ERROR')

            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save progress for resume capability."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'baselines': self.baselines,
            'results': self.results,
            'call_count': self.client.call_count,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self) -> bool:
        """Load previous checkpoint for resume."""
        if not self.checkpoint_path.exists():
            return False

        with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        self.baselines = checkpoint.get('baselines', {})
        self.results = checkpoint.get('results', [])
        print(f'  Resumed from checkpoint: {len(self.baselines)} baselines, {len(self.results)} results')
        return True

    def save_results(self):
        """Save final results to output directory."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save baselines
        with open(OUTPUT_DIR / 'steering_baselines.json', 'w', encoding='utf-8') as f:
            json.dump(self.baselines, f, indent=2, ensure_ascii=False)

        # Save all results
        with open(OUTPUT_DIR / 'steering_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_experiments': len(self.results),
                    'total_api_calls': self.client.call_count,
                    'timestamp': datetime.now().isoformat(),
                    'strengths_tested': STEERING_STRENGTHS,
                },
                'results': self.results,
            }, f, indent=2, ensure_ascii=False)

        print(f'\n  Results saved to: {OUTPUT_DIR}')
        print(f'  - steering_baselines.json ({len(self.baselines)} baselines)')
        print(f'  - steering_results.json ({len(self.results)} experiments)')


# ============================================================================
# SteeringAnalyzer: Computes metrics and generates report
# ============================================================================

class SteeringAnalyzer:
    """Analyzes steering results and generates report."""

    def __init__(self):
        self.baselines = {}
        self.results = []
        self.analysis = {}

    def load_results(self):
        """Load saved results from disk."""
        baselines_path = OUTPUT_DIR / 'steering_baselines.json'
        results_path = OUTPUT_DIR / 'steering_results.json'

        if not baselines_path.exists() or not results_path.exists():
            # Try checkpoint
            checkpoint_path = OUTPUT_DIR / 'steering_checkpoint.json'
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.baselines = checkpoint.get('baselines', {})
                self.results = checkpoint.get('results', [])
            else:
                print('[ERROR] No results found. Run experiments first.')
                return False
        else:
            with open(baselines_path, 'r', encoding='utf-8') as f:
                self.baselines = json.load(f)
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.results = data.get('results', [])

        print(f'Loaded {len(self.baselines)} baselines, {len(self.results)} experiments')
        return True

    def _extract_steered_output(self, response: dict) -> str:
        """Extract steered text from steering API response."""
        if not response or 'error' in response:
            return ''
        # API returns STEERED and DEFAULT keys
        return response.get('STEERED', response.get('output', response.get('text', '')))

    def _extract_default_output(self, response: dict) -> str:
        """Extract default (unsteered) text from steering API response."""
        if not response or 'error' in response:
            return ''
        return response.get('DEFAULT', response.get('output', response.get('text', '')))

    def _extract_first_token_prob(self, response: dict, steered: bool = True) -> Tuple[str, float]:
        """Extract the first predicted token and its probability from log probs."""
        if not response or 'error' in response:
            return ('', 0.0)
        key = 'steeredLogProbs' if steered else 'defaultLogProbs'
        log_probs = response.get(key, [])
        if log_probs and len(log_probs) > 0:
            first = log_probs[0]
            token = first.get('token', '')
            logprob = first.get('logprob', -100)
            prob = math.exp(logprob)
            return (token, prob)
        return ('', 0.0)

    def compute_metrics(self):
        """Compute Steering Impact Score (SIS) and Disruption Score (DS) per feature."""
        print('\nComputing metrics...')

        feature_metrics = {}  # label -> {amplify_outputs, suppress_outputs, ...}

        for result in self.results:
            label = result['feature_label']
            circuit = result['circuit_id']
            strength = result['strength']
            response = result.get('response', {})

            # API returns both STEERED and DEFAULT in each response
            steered_output = self._extract_steered_output(response)
            default_output = self._extract_default_output(response)

            # Also extract from baselines if available (for pre-existing baselines)
            baseline_data = self.baselines.get(circuit, {})
            baseline_response = baseline_data.get('response', {})
            if default_output:
                baseline_output = default_output
            elif baseline_response:
                baseline_output = self._extract_default_output(baseline_response)
                if not baseline_output:
                    baseline_output = self._extract_steered_output(baseline_response)
            else:
                baseline_output = ''

            # Extract first token probabilities
            steered_token, steered_prob = self._extract_first_token_prob(response, steered=True)
            default_token, default_prob = self._extract_first_token_prob(response, steered=False)

            if label not in feature_metrics:
                feature_metrics[label] = {
                    'label': label,
                    'layer': result['layer'],
                    'np_id': result['np_id'],
                    'circuits_appeared_in': result.get('circuits_appeared_in', 1),
                    'avg_convergence': result.get('avg_convergence', 0),
                    'is_universal': result.get('is_universal', False),
                    'experiments': [],
                }

            # Did the output change?
            output_changed = steered_output.strip() != baseline_output.strip()

            # Did the first token change?
            first_token_changed = steered_token.strip() != default_token.strip()

            # Probability shift
            prob_shift = steered_prob - default_prob if (steered_prob > 0 and default_prob > 0) else 0.0

            # Record experiment
            feature_metrics[label]['experiments'].append({
                'circuit': circuit,
                'strength': strength,
                'baseline_output': baseline_output[:100],
                'steered_output': steered_output[:100],
                'output_changed': output_changed,
                'first_token_changed': first_token_changed,
                'default_token': default_token,
                'steered_token': steered_token,
                'default_prob': default_prob,
                'steered_prob': steered_prob,
                'prob_shift': prob_shift,
            })

        # Compute aggregate scores per feature
        for label, metrics in feature_metrics.items():
            amplify_exps = [e for e in metrics['experiments'] if e['strength'] > 0]
            suppress_exps = [e for e in metrics['experiments'] if e['strength'] < 0]

            # Steering Impact Score: fraction of amplify experiments that changed output
            if amplify_exps:
                metrics['amplify_change_rate'] = sum(1 for e in amplify_exps if e['output_changed']) / len(amplify_exps)
                metrics['amplify_token_change_rate'] = sum(1 for e in amplify_exps if e['first_token_changed']) / len(amplify_exps)
                metrics['mean_amplify_prob_shift'] = sum(e['prob_shift'] for e in amplify_exps) / len(amplify_exps)
            else:
                metrics['amplify_change_rate'] = 0
                metrics['amplify_token_change_rate'] = 0
                metrics['mean_amplify_prob_shift'] = 0

            # Disruption Score: fraction of suppress experiments that changed output
            if suppress_exps:
                metrics['suppress_change_rate'] = sum(1 for e in suppress_exps if e['output_changed']) / len(suppress_exps)
                metrics['suppress_token_change_rate'] = sum(1 for e in suppress_exps if e['first_token_changed']) / len(suppress_exps)
                metrics['mean_suppress_prob_shift'] = sum(e['prob_shift'] for e in suppress_exps) / len(suppress_exps)
            else:
                metrics['suppress_change_rate'] = 0
                metrics['suppress_token_change_rate'] = 0
                metrics['mean_suppress_prob_shift'] = 0

            # Combined steering effectiveness
            all_exps = metrics['experiments']
            metrics['overall_change_rate'] = sum(1 for e in all_exps if e['output_changed']) / len(all_exps) if all_exps else 0
            metrics['overall_token_change_rate'] = sum(1 for e in all_exps if e['first_token_changed']) / len(all_exps) if all_exps else 0
            metrics['mean_prob_shift'] = sum(e['prob_shift'] for e in all_exps) / len(all_exps) if all_exps else 0
            metrics['total_experiments'] = len(all_exps)

        self.analysis = {
            'feature_metrics': feature_metrics,
            'summary': self._compute_summary(feature_metrics),
        }

        return feature_metrics

    def _compute_summary(self, feature_metrics: dict) -> dict:
        """Compute summary statistics and correlations."""
        features = list(feature_metrics.values())
        if not features:
            return {}

        universal = [f for f in features if f['is_universal']]
        specific = [f for f in features if not f['is_universal']]

        # Correlation: cross-circuit frequency vs disruption
        freq_scores = [(f['circuits_appeared_in'], f['suppress_change_rate']) for f in features
                       if f['suppress_change_rate'] is not None and f['circuits_appeared_in'] > 0]

        correlation = self._pearson_correlation(freq_scores) if len(freq_scores) >= 3 else None

        # Layer analysis
        early_features = [f for f in features if f['layer'] <= 10]
        late_features = [f for f in features if f['layer'] > 10]

        summary = {
            'total_features_tested': len(features),
            'total_experiments': sum(f['total_experiments'] for f in features),
            'universal_features': len(universal),
            'specific_features': len(specific),
            'mean_amplify_change_rate': sum(f['amplify_change_rate'] for f in features) / len(features) if features else 0,
            'mean_suppress_change_rate': sum(f['suppress_change_rate'] for f in features) / len(features) if features else 0,
            'frequency_disruption_correlation': correlation,
            'early_layer_mean_change': sum(f['overall_change_rate'] for f in early_features) / len(early_features) if early_features else 0,
            'late_layer_mean_change': sum(f['overall_change_rate'] for f in late_features) / len(late_features) if late_features else 0,
        }

        return summary

    def _pearson_correlation(self, pairs: List[Tuple[float, float]]) -> Optional[float]:
        """Compute Pearson correlation coefficient."""
        n = len(pairs)
        if n < 3:
            return None

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def save_analysis(self):
        """Save analysis results."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save JSON analysis
        serializable = {
            'summary': self.analysis.get('summary', {}),
            'feature_metrics': {
                label: {k: v for k, v in metrics.items() if k != 'experiments'}
                for label, metrics in self.analysis.get('feature_metrics', {}).items()
            },
        }

        with open(OUTPUT_DIR / 'steering_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f'  Analysis saved to: {OUTPUT_DIR / "steering_analysis.json"}')

    def generate_report(self):
        """Generate human-readable steering validation report."""
        feature_metrics = self.analysis.get('feature_metrics', {})
        summary = self.analysis.get('summary', {})

        if not feature_metrics:
            print('[ERROR] No analysis data. Run compute_metrics() first.')
            return

        # Sort features by overall change rate (most effective first)
        sorted_features = sorted(
            feature_metrics.values(),
            key=lambda f: f['overall_change_rate'],
            reverse=True
        )

        lines = []
        lines.append('# Stage 3: Steering Validation Report')
        lines.append('')
        lines.append(f'**Date:** {datetime.now().strftime("%Y-%m-%d")}')
        lines.append(f'**Features Tested:** {summary.get("total_features_tested", 0)}')
        lines.append(f'**Total Experiments:** {summary.get("total_experiments", 0)}')
        lines.append(f'**Universal Features:** {summary.get("universal_features", 0)}')
        lines.append(f'**Circuit-Specific Features:** {summary.get("specific_features", 0)}')
        lines.append('')
        lines.append('---')
        lines.append('')

        # Executive Summary
        lines.append('## 1. Executive Summary')
        lines.append('')
        corr = summary.get('frequency_disruption_correlation')
        if corr is not None:
            direction = 'positive' if corr > 0 else 'negative'
            strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
            lines.append(f'Cross-circuit frequency shows a **{strength} {direction} correlation** (r={corr:.3f}) with steering disruption score.')
        else:
            lines.append('Insufficient data to compute frequency-disruption correlation.')
        lines.append('')
        lines.append(f'- Mean amplification change rate: {summary.get("mean_amplify_change_rate", 0)*100:.1f}%')
        lines.append(f'- Mean suppression change rate: {summary.get("mean_suppress_change_rate", 0)*100:.1f}%')
        lines.append(f'- Early layers (L0-L10) mean change: {summary.get("early_layer_mean_change", 0)*100:.1f}%')
        lines.append(f'- Late layers (L11+) mean change: {summary.get("late_layer_mean_change", 0)*100:.1f}%')
        lines.append('')

        # Feature Results Table
        lines.append('## 2. Feature Steering Results')
        lines.append('')
        lines.append('| Feature | Layer | Circuits | Convergence | Amplify Rate | Suppress Rate | Overall | Prob Shift | Universal |')
        lines.append('|---------|-------|----------|-------------|-------------|--------------|---------|-----------|-----------|')

        for f in sorted_features:
            prob_shift = f.get('mean_prob_shift', 0)
            lines.append(
                f'| {f["label"]} | L{f["layer"]} | {f["circuits_appeared_in"]} | '
                f'{f["avg_convergence"]*100:.0f}% | '
                f'{f["amplify_change_rate"]*100:.0f}% | '
                f'{f["suppress_change_rate"]*100:.0f}% | '
                f'{f["overall_change_rate"]*100:.0f}% | '
                f'{prob_shift:+.4f} | '
                f'{"Yes" if f["is_universal"] else "No"} |'
            )

        lines.append('')

        # Detailed experiments for top features
        lines.append('## 3. Detailed Results (Top 10 Features)')
        lines.append('')

        for f in sorted_features[:10]:
            lines.append(f'### {f["label"]} (L{f["layer"]}, {f["circuits_appeared_in"]} circuits)')
            lines.append('')
            lines.append(f'| Circuit | Strength | Default Token | Steered Token | Prob Shift | Output Changed |')
            lines.append(f'|---------|----------|---------------|---------------|-----------|----------------|')

            for exp in f.get('experiments', []):
                changed = 'YES' if exp['output_changed'] else 'no'
                default_tok = exp.get('default_token', '?').strip()
                steered_tok = exp.get('steered_token', '?').strip()
                prob_shift = exp.get('prob_shift', 0)
                circuit_short = exp["circuit"].replace('gemma-2-2b_', '')[:30]
                lines.append(
                    f'| {circuit_short} | {exp["strength"]:+d} | '
                    f'`{default_tok}` ({exp.get("default_prob", 0)*100:.1f}%) | '
                    f'`{steered_tok}` ({exp.get("steered_prob", 0)*100:.1f}%) | '
                    f'{prob_shift:+.4f} | {changed} |'
                )

            lines.append('')

        # Key Findings
        lines.append('## 4. Key Findings')
        lines.append('')

        effective_features = [f for f in sorted_features if f['overall_change_rate'] > 0.5]
        ineffective_features = [f for f in sorted_features if f['overall_change_rate'] == 0]

        lines.append(f'- **{len(effective_features)} features** changed output in >50% of experiments')
        lines.append(f'- **{len(ineffective_features)} features** had no steering effect at all')

        if summary.get('early_layer_mean_change', 0) != summary.get('late_layer_mean_change', 0):
            early = summary.get('early_layer_mean_change', 0)
            late = summary.get('late_layer_mean_change', 0)
            if early > late:
                lines.append(f'- Early-layer bottlenecks ({early*100:.0f}%) are MORE steerable than late-layer ones ({late*100:.0f}%)')
            else:
                lines.append(f'- Late-layer bottlenecks ({late*100:.0f}%) are MORE steerable than early-layer ones ({early*100:.0f}%)')

        lines.append('')

        report_text = '\n'.join(lines)
        report_path = OUTPUT_DIR / 'steering_validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f'  Report saved to: {report_path}')
        return report_text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Steering Validation')
    parser.add_argument('--quick', action='store_true', help='Quick test (5 features, 2 circuits)')
    parser.add_argument('--feature', type=str, help='Test single feature (e.g., L3_F5150441)')
    parser.add_argument('--analyze-only', action='store_true', help='Skip API calls, analyze existing results')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--rate-delay', type=float, default=36.0,
                        help='Seconds between API calls (default: 36 = 100/hr)')
    args = parser.parse_args()

    print('=' * 70)
    print('  STAGE 3: STEERING VALIDATION')
    print('=' * 70)

    # Analyze-only mode
    if args.analyze_only:
        analyzer = SteeringAnalyzer()
        if not analyzer.load_results():
            sys.exit(1)
        analyzer.compute_metrics()
        analyzer.save_analysis()
        analyzer.generate_report()
        return

    # Load config
    if not CONFIG_PATH.exists():
        print(f'[ERROR] Config not found: {CONFIG_PATH}')
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['api']['api_key']

    # Initialize client and runner
    client = SteeringClient(api_key=api_key, rate_limit_delay=args.rate_delay)
    runner = ExperimentRunner(client, quick=args.quick)

    # Resume from checkpoint
    if args.resume:
        print('\nResuming from checkpoint...')
        runner.load_checkpoint()

    # Load data
    runner.load_bottleneck_library()
    runner.load_circuit_prompts()

    # Single feature mode
    if args.feature:
        print(f'\nSingle feature mode: {args.feature}')
        if args.feature not in runner.bottleneck_library:
            print(f'[ERROR] Feature {args.feature} not found in bottleneck library')
            sys.exit(1)

        feat = runner.bottleneck_library[args.feature]
        layer = feat['layer']
        np_id = feat.get('np_id', feat.get('circuit_id', 0) % GEMMA_SAE_DICT_SIZE)
        gemma_circuits = [a['circuit'] for a in feat['appearances']
                          if a['model'] == 'gemma-2-2b' and a['circuit'] in runner.circuit_prompts]

        print(f'  Layer: {layer}, NP ID: {np_id}, Circuits: {len(gemma_circuits)}')

        # Get baselines
        for circuit_id in gemma_circuits:
            if circuit_id not in runner.baselines:
                print(f'  Baseline: {circuit_id}...')
                result = client.get_baseline(runner.circuit_prompts[circuit_id]['prompt'])
                if 'error' not in result:
                    runner.baselines[circuit_id] = {
                        'prompt': runner.circuit_prompts[circuit_id]['prompt'],
                        'response': result,
                    }
                    output = result.get('DEFAULT', result.get('STEERED', ''))
                    print(f'    -> "{output[:60]}"')

        # Steer
        for circuit_id in gemma_circuits:
            prompt = runner.circuit_prompts[circuit_id]['prompt']
            for strength in STEERING_STRENGTHS:
                action = 'AMPLIFY' if strength > 0 else 'SUPPRESS'
                print(f'  {action} {abs(strength):+d} on {circuit_id}...', end='', flush=True)
                result = client.steer_feature(prompt, layer, np_id, strength)
                if 'error' not in result:
                    output = result.get('STEERED', result.get('output', ''))
                    print(f' -> "{output[:50]}"')
                    runner.results.append({
                        'feature_label': args.feature,
                        'layer': layer,
                        'np_id': np_id,
                        'circuit_id': circuit_id,
                        'prompt': prompt,
                        'strength': strength,
                        'response': result,
                        'circuits_appeared_in': feat['circuits_appeared_in'],
                        'avg_convergence': feat.get('avg_convergence', 0),
                        'is_universal': len(gemma_circuits) >= 3,
                    })
                else:
                    print(f' -> ERROR')

        runner.save_results()
    else:
        # Full experiment run
        runner.run_baselines()
        runner.run_universal_bottleneck_experiments()
        runner.run_prompt_specific_experiments()
        runner.save_results()

    # Analyze
    print('\n' + '=' * 70)
    print('  ANALYZING RESULTS')
    print('=' * 70)

    analyzer = SteeringAnalyzer()
    analyzer.baselines = runner.baselines
    analyzer.results = runner.results
    analyzer.compute_metrics()
    analyzer.save_analysis()
    analyzer.generate_report()

    print('\n' + '=' * 70)
    print('  STAGE 3 COMPLETE')
    print('=' * 70)
    print(f'\n  Output: {OUTPUT_DIR}')
    print(f'  Total API calls: {client.call_count}')


if __name__ == '__main__':
    main()
