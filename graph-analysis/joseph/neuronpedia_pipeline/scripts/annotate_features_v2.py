#!/usr/bin/env python3
"""
Improved Feature Annotation Tool v2

Key improvements over v1:
  1. Uses Neuronpedia explanations as PRIMARY signal (not just tokens)
  2. Unicode script detection for multilingual awareness
  3. Structural pattern detection (camelCase, snake_case)
  4. Two-pass classification: classify each token, then vote
  5. Layer-aware Bayesian priors
  6. Explanation keyword mapping

Usage:
    python annotate_features_v2.py --input ../semantic_taxonomy_annotations.csv --output ../semantic_taxonomy_annotations_v2.csv
    python annotate_features_v2.py --test  # Compare v2 against validated CSV
"""

import argparse
import csv
import re
import sys
import io
import json
import time
import unicodedata
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE = Path(__file__).parent.parent
CONFIG_PATH = BASE / "config" / "neuronpedia_config.yaml"

# ── Standalone keyword map (importable without class instantiation) ──────────
_EXPLANATION_KEYWORDS = {
    'CODE': [
        'code', 'programming', 'html', 'function', 'variable',
        'method', 'class', 'api', 'framework', 'library',
        'syntax', 'parameter', 'module', 'script', 'tag',
        'javascript', 'python', 'java', 'css', 'sql',
        'web app', 'software', 'drupal', 'android', 'flutter',
    ],
    'CONCEPT': [
        'word', 'phrase', 'language', 'concept', 'idea',
        'abstract', 'meaning', 'topic', 'theme', 'related to',
        'inhibition', 'biological', 'process', 'action',
        'planning', 'future', 'possibility', 'problem',
        'help', 'question', 'uncertainty',
    ],
    'GEOGRAPHIC': [
        'place', 'city', 'country', 'capital', 'geographic',
        'location', 'region', 'state', 'continent', 'border',
        'north', 'south', 'east', 'west', 'latitude',
    ],
    'ENTITY': [
        'name', 'person', 'people', 'organization', 'company',
        'staff', 'leader', 'president', 'gender', 'male',
        'female', 'human',
    ],
    'TEMPORAL': [
        'time', 'date', 'year', 'day', 'month', 'period',
        'century', 'decade', 'before', 'after', 'historical',
    ],
    'SYNTAX': [
        'punctuation', 'formatting', 'token', 'delimiter',
        'bracket', 'indentation', 'whitespace', 'special token',
        'formula', 'expression', 'mathematical',
    ],
}


def classify_from_explanation(explanation: str, layer: int = None) -> str:
    """Classify a feature's semantic category from its NP explanation string.

    Standalone function — can be imported without instantiating FeatureClassifierV2.
    Uses keyword scoring with polysemantic detection.

    Args:
        explanation: Neuronpedia explanation text for the feature.
        layer: Optional layer number (reserved for future Bayesian priors).

    Returns:
        Category string: SYNTAX, SEMANTICS:CODE, SEMANTICS:CONCEPT,
        SEMANTICS:GEOGRAPHIC, SEMANTICS:TEMPORAL, SEMANTICS:ENTITY,
        POLYSEMANTIC, or UNKNOWN.
    """
    if not explanation or len(explanation.strip()) < 5:
        return 'UNKNOWN'

    exp_lower = explanation.lower()

    # Score each category by keyword matches
    scores = {}
    for cat, keywords in _EXPLANATION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in exp_lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return 'SEMANTICS:CONCEPT'  # Has explanation but no keyword match

    # Best match
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Check for mixed signals (polysemantic)
    runner_up = sorted(scores.values(), reverse=True)
    if len(runner_up) >= 2 and runner_up[1] >= runner_up[0] * 0.7:
        top_cats = [c for c, s in scores.items() if s >= best_score * 0.7]
        if len(top_cats) >= 2 and set(top_cats) != {'CODE', 'SYNTAX'}:
            return 'POLYSEMANTIC'

    if best_cat in ('CODE', 'CONCEPT', 'GEOGRAPHIC', 'ENTITY', 'TEMPORAL'):
        return f'SEMANTICS:{best_cat}'
    else:
        return best_cat


class FeatureClassifierV2:
    """Improved feature classifier with NP explanations + multilingual awareness."""

    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self.api_key = None
        self.api_base = None
        self._load_config()

        # ── Explanation keyword map ──────────────────────────────────
        self.explanation_keywords = {
            'CODE': [
                'code', 'programming', 'html', 'function', 'variable',
                'method', 'class', 'api', 'framework', 'library',
                'syntax', 'parameter', 'module', 'script', 'tag',
                'javascript', 'python', 'java', 'css', 'sql',
                'web app', 'software', 'drupal', 'android', 'flutter',
            ],
            'CONCEPT': [
                'word', 'phrase', 'language', 'concept', 'idea',
                'abstract', 'meaning', 'topic', 'theme', 'related to',
                'inhibition', 'biological', 'process', 'action',
                'planning', 'future', 'possibility', 'problem',
                'help', 'question', 'uncertainty',
            ],
            'GEOGRAPHIC': [
                'place', 'city', 'country', 'capital', 'geographic',
                'location', 'region', 'state', 'continent', 'border',
                'north', 'south', 'east', 'west', 'latitude',
            ],
            'ENTITY': [
                'name', 'person', 'people', 'organization', 'company',
                'staff', 'leader', 'president', 'gender', 'male',
                'female', 'human',
            ],
            'TEMPORAL': [
                'time', 'date', 'year', 'day', 'month', 'period',
                'century', 'decade', 'before', 'after', 'historical',
            ],
            'SYNTAX': [
                'punctuation', 'formatting', 'token', 'delimiter',
                'bracket', 'indentation', 'whitespace', 'special token',
                'formula', 'expression', 'mathematical',
            ],
        }

    def _load_config(self):
        """Load API config."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            self.api_key = config.get('api', {}).get('api_key')
            self.api_base = config.get('api', {}).get('base_url')

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 1: Neuronpedia Explanation (highest priority)
    # ══════════════════════════════════════════════════════════════════
    def fetch_np_explanation(self, layer: int, feature_id: int) -> Optional[Dict]:
        """
        Fetch explanation + activation examples from Neuronpedia.
        Returns dict with 'explanation', 'top_tokens', or None.
        """
        if not self.use_api or not self.api_key:
            return None

        np_id = feature_id % 16384
        sae_name = f"{layer}-gemmascope-transcoder-16k"
        url = f"{self.api_base}/feature/gemma-2-2b/{sae_name}/{np_id}"
        headers = {'x-api-key': self.api_key}

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()

            # Extract explanation
            explanation = ''
            explanations = data.get('explanations', [])
            if explanations:
                explanation = explanations[0].get('description', '')

            # Extract top activation tokens
            top_tokens = []
            if 'activations' in data:
                for act in data['activations'][:10]:
                    tokens = act.get('tokens', [])
                    values = act.get('values', [])
                    if tokens and values:
                        max_idx = values.index(max(values))
                        if max_idx < len(tokens):
                            top_tokens.append(tokens[max_idx])

            return {
                'explanation': explanation,
                'top_tokens': top_tokens[:10],
            }
        except Exception:
            return None

    def classify_from_explanation(self, explanation: str) -> Optional[Tuple[str, str]]:
        """
        Classify feature based on NP explanation text.
        Returns (category, confidence) or None if no clear match.
        """
        if not explanation or len(explanation) < 5:
            return None

        exp_lower = explanation.lower()

        # Score each category by keyword matches
        scores = {}
        for cat, keywords in self.explanation_keywords.items():
            score = sum(1 for kw in keywords if kw in exp_lower)
            if score > 0:
                scores[cat] = score

        if not scores:
            return None

        # Best match
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        # Check for mixed signals (polysemantic)
        runner_up = sorted(scores.values(), reverse=True)
        if len(runner_up) >= 2 and runner_up[1] >= runner_up[0] * 0.7:
            # Two categories are close -- check if it's CODE+something
            top_cats = [c for c, s in scores.items() if s >= best_score * 0.7]
            if len(top_cats) >= 2 and set(top_cats) != {'CODE', 'SYNTAX'}:
                return ('POLYSEMANTIC', 'MEDIUM')

        confidence = 'HIGH' if best_score >= 3 else 'MEDIUM' if best_score >= 2 else 'LOW'

        if best_cat in ('CODE', 'CONCEPT', 'GEOGRAPHIC', 'ENTITY', 'TEMPORAL'):
            return (f'SEMANTICS:{best_cat}', confidence)
        else:
            return (best_cat, confidence)

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 2: Unicode Script Detection
    # ══════════════════════════════════════════════════════════════════
    def detect_scripts(self, text: str) -> set:
        """Detect which Unicode scripts are present in text."""
        scripts = set()
        for char in text:
            if char.isalpha():
                try:
                    name = unicodedata.name(char, '')
                    if 'LATIN' in name:
                        scripts.add('Latin')
                    elif 'ARABIC' in name:
                        scripts.add('Arabic')
                    elif 'HEBREW' in name:
                        scripts.add('Hebrew')
                    elif 'CJK' in name or 'HANGUL' in name:
                        scripts.add('CJK')
                    elif 'CYRILLIC' in name:
                        scripts.add('Cyrillic')
                    elif 'DEVANAGARI' in name or 'BENGALI' in name:
                        scripts.add('Indic')
                    elif 'GREEK' in name:
                        scripts.add('Greek')
                    elif 'THAI' in name:
                        scripts.add('Thai')
                    elif 'GEORGIAN' in name:
                        scripts.add('Georgian')
                    elif 'KATAKANA' in name or 'HIRAGANA' in name:
                        scripts.add('Japanese')
                    else:
                        scripts.add('Other')
                except ValueError:
                    pass
        return scripts

    def multilingual_score(self, examples: List[str]) -> Tuple[int, set]:
        """
        Count how many distinct scripts appear across examples.
        Returns (num_scripts, set_of_scripts).
        """
        all_scripts = set()
        for ex in examples:
            all_scripts.update(self.detect_scripts(ex))
        return len(all_scripts), all_scripts

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL 3: Structural Pattern Detection (per-token)
    # ══════════════════════════════════════════════════════════════════
    def classify_single_token(self, token: str) -> str:
        """
        Classify a single token as 'code', 'syntax', 'nl' (natural language),
        or 'fragment' (subword artifact).
        """
        token = token.strip().replace('▁', '')
        if not token:
            return 'fragment'

        # Special tokens
        if token in ('<bos>', '<eos>', '<pad>', '<unk>', '<s>', '</s>'):
            return 'syntax'

        # Common function words (articles, prepositions, conjunctions)
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'it', 'this', 'that', 'not', 'no', 'if', 'so', 'as', 'from',
            'use', 'using', 'into', 'also',
        }
        if token.lower() in function_words:
            return 'syntax'

        # Pure punctuation / operators
        if all(not c.isalnum() for c in token):
            return 'syntax'

        # Very short fragments (< 3 chars, no vowels)
        if len(token) < 3 and not any(c in 'aeiouAEIOU' for c in token):
            return 'fragment'

        # Code identifiers: camelCase, PascalCase, snake_case, UPPER_CASE
        if re.match(r'^[a-z]+[A-Z]', token):  # camelCase
            return 'code'
        if re.match(r'^[A-Z][a-z]+[A-Z]', token):  # PascalCase
            return 'code'
        if '_' in token and any(c.isalpha() for c in token):  # snake_case
            return 'code'
        if re.match(r'^[A-Z]{2,}[A-Z_]*$', token):  # UPPER_CASE
            return 'code'

        # Code patterns: contains dots, brackets, or semicolons mixed with letters
        if re.search(r'[a-zA-Z]+\.[a-zA-Z]', token):  # foo.bar
            return 'code'
        if re.search(r'[a-zA-Z]+\(', token):  # func(
            return 'code'
        if token.endswith(');') or token.endswith('};'):
            return 'code'

        # Known code suffixes
        code_suffixes = ('Builder', 'Helper', 'Writer', 'Reader', 'Handler',
                         'Listener', 'Provider', 'Factory', 'Manager', 'Service',
                         'Controller', 'Adapter', 'Iterator', 'Exception',
                         'Interface', 'Impl', 'Config', 'Init', 'Async',
                         'Style', 'Layout', 'Widget', 'View', 'Activity',
                         'Fragment', 'Component', 'Module', 'Plugin')
        if any(token.endswith(s) for s in code_suffixes):
            return 'code'

        code_prefixes = ('get', 'set', 'is', 'has', 'on', 'add', 'remove',
                         'create', 'delete', 'update', 'find', 'fetch', 'parse',
                         'render', 'handle', 'init', 'load', 'save', 'validate')
        token_lower = token.lower()
        if any(token_lower.startswith(p) and len(token) > len(p) + 2 and
               token[len(p)].isupper() for p in code_prefixes):
            return 'code'

        # Non-Latin script = natural language
        scripts = self.detect_scripts(token)
        if scripts - {'Latin'}:  # Has non-Latin characters
            return 'nl'

        # Regular lowercase word = natural language
        if token.islower() and token.isalpha() and len(token) >= 3:
            return 'nl'

        # Mixed case single word without code patterns = probably NL
        if token[0].isupper() and token[1:].islower() and len(token) >= 3:
            return 'nl'

        # Default: fragment
        return 'fragment'

    # ══════════════════════════════════════════════════════════════════
    # MAIN CLASSIFIER: Two-pass with voting
    # ══════════════════════════════════════════════════════════════════
    def classify_feature(self, layer: int, feature_id: int,
                         activation_examples: str, activation_count: int,
                         np_data: Optional[Dict] = None) -> Dict:
        """
        Full classification pipeline.

        Priority order:
        1. NP explanation (if available and clear)
        2. Token-level voting with script awareness
        3. Layer-aware priors as tiebreaker

        Returns dict with category, confidence, reasoning, method.
        """
        reasoning = []

        # ── Handle empty ────────────────────────────────────────────
        if not activation_examples and activation_count == 0:
            return {
                'category': 'UNKNOWN',
                'confidence': 'MEDIUM',
                'reasoning': ['No activation examples available'],
                'method': 'empty',
            }

        # ── PASS 1: NP Explanation ──────────────────────────────────
        if np_data and np_data.get('explanation'):
            explanation = np_data['explanation']
            exp_result = self.classify_from_explanation(explanation)
            if exp_result:
                cat, conf = exp_result
                reasoning.append(f'NP explanation: "{explanation[:60]}"')
                reasoning.append(f'Explanation-based classification: {cat} ({conf})')

                # Cross-validate with tokens if we have them
                examples = self._parse_examples(activation_examples, np_data)
                token_result = self._vote_on_tokens(examples)
                if token_result:
                    reasoning.append(f'Token vote: {token_result["vote_summary"]}')
                    # If tokens strongly disagree, downgrade confidence
                    if token_result['dominant'] != self._simplify_cat(cat):
                        if conf == 'HIGH':
                            conf = 'MEDIUM'
                        reasoning.append('Token vote disagrees - confidence reduced')

                return {
                    'category': cat,
                    'confidence': conf,
                    'reasoning': reasoning,
                    'method': 'explanation',
                }

        # ── PASS 2: Token-level voting ──────────────────────────────
        examples = self._parse_examples(activation_examples, np_data)

        if not examples:
            return {
                'category': 'UNKNOWN',
                'confidence': 'LOW',
                'reasoning': ['No usable examples'],
                'method': 'empty',
            }

        token_result = self._vote_on_tokens(examples)
        reasoning.append(f'Token classifications: {token_result["vote_summary"]}')

        # Check multilingual
        num_scripts, scripts = self.multilingual_score(examples)
        if num_scripts >= 3:
            reasoning.append(f'Multilingual: {num_scripts} scripts ({", ".join(sorted(scripts))})')
            return {
                'category': 'POLYSEMANTIC',
                'confidence': 'HIGH',
                'reasoning': reasoning,
                'method': 'multilingual',
            }
        elif num_scripts == 2 and 'Latin' in scripts:
            reasoning.append(f'Bilingual: {", ".join(sorted(scripts))}')
            # Only flag as polysemantic if NL tokens outnumber or equal code tokens
            if token_result['nl_pct'] >= token_result['code_pct'] and token_result['nl_pct'] > 0.3:
                return {
                    'category': 'POLYSEMANTIC',
                    'confidence': 'MEDIUM',
                    'reasoning': reasoning,
                    'method': 'multilingual+mixed',
                }
            # If code dominates even with bilingual, still call it CODE
            elif token_result['code_pct'] >= 0.5:
                return {
                    'category': 'SEMANTICS:CODE',
                    'confidence': 'MEDIUM',
                    'reasoning': reasoning + ['Code-dominant despite bilingual tokens'],
                    'method': 'code_dominant_bilingual',
                }

        # Apply voting thresholds
        dominant = token_result['dominant']
        dom_pct = token_result['dominant_pct']

        if dominant == 'syntax' and dom_pct >= 0.5:
            # Special tokens + common words = syntax
            conf = 'HIGH' if dom_pct >= 0.9 else 'MEDIUM'
            return {
                'category': 'SYNTAX',
                'confidence': conf,
                'reasoning': reasoning,
                'method': 'token_vote',
            }

        if dominant == 'code' and dom_pct >= 0.5:
            # With few examples, 2/3 code is enough (67%)
            conf = 'HIGH' if dom_pct >= 0.9 else 'MEDIUM'
            return {
                'category': 'SEMANTICS:CODE',
                'confidence': conf,
                'reasoning': reasoning,
                'method': 'token_vote',
            }

        if dominant == 'nl' and dom_pct >= 0.7:
            # Try to infer subcategory from layer priors
            subcat = self._infer_subcategory(examples, layer)
            return {
                'category': f'SEMANTICS:{subcat}',
                'confidence': 'MEDIUM',
                'reasoning': reasoning + [f'Subcategory inferred: {subcat} (layer {layer} prior)'],
                'method': 'token_vote+prior',
            }

        # Mixed: check if it's genuinely mixed or just noise
        if token_result['code_pct'] > 0.2 and token_result['nl_pct'] > 0.2:
            # If multilingual (3+ scripts), definitely polysemantic
            if num_scripts >= 2:
                return {
                    'category': 'POLYSEMANTIC',
                    'confidence': 'MEDIUM',
                    'reasoning': reasoning + [f'Mixed code + NL across {num_scripts} scripts'],
                    'method': 'token_vote_mixed',
                }
            # If code is dominant (even slightly), lean CODE with lower confidence
            if token_result['code_pct'] > token_result['nl_pct']:
                return {
                    'category': 'SEMANTICS:CODE',
                    'confidence': 'MEDIUM',
                    'reasoning': reasoning + ['Code-dominant mixed pattern'],
                    'method': 'token_vote_mixed_code',
                }
            return {
                'category': 'POLYSEMANTIC',
                'confidence': 'MEDIUM',
                'reasoning': reasoning + ['Mixed code + natural language tokens'],
                'method': 'token_vote_mixed',
            }

        # Mostly fragments
        if token_result['fragment_pct'] > 0.5:
            return {
                'category': 'UNKNOWN',
                'confidence': 'LOW',
                'reasoning': reasoning + ['Majority subword fragments'],
                'method': 'fragments',
            }

        # Default fallback: POLYSEMANTIC with LOW confidence
        return {
            'category': 'POLYSEMANTIC',
            'confidence': 'LOW',
            'reasoning': reasoning + ['No clear dominant pattern'],
            'method': 'fallback',
        }

    def _parse_examples(self, activation_examples: str,
                        np_data: Optional[Dict] = None) -> List[str]:
        """Parse examples from CSV field + NP API data."""
        examples = []
        if activation_examples:
            examples = [ex.strip() for ex in activation_examples.split(';') if ex.strip()]
        # Add NP tokens if available (gives us more data points)
        if np_data and np_data.get('top_tokens'):
            for tok in np_data['top_tokens']:
                clean = tok.replace('▁', '').strip()
                if clean and clean not in examples:
                    examples.append(clean)
        return examples

    def _vote_on_tokens(self, examples: List[str]) -> Dict:
        """Classify each token and aggregate votes."""
        votes = Counter()
        for ex in examples:
            vote = self.classify_single_token(ex)
            votes[vote] += 1

        total = sum(votes.values()) or 1
        dominant = votes.most_common(1)[0][0] if votes else 'fragment'
        dom_pct = votes[dominant] / total

        return {
            'votes': dict(votes),
            'dominant': dominant,
            'dominant_pct': dom_pct,
            'code_pct': votes.get('code', 0) / total,
            'nl_pct': votes.get('nl', 0) / total,
            'syntax_pct': votes.get('syntax', 0) / total,
            'fragment_pct': votes.get('fragment', 0) / total,
            'vote_summary': ', '.join(f'{k}={v}' for k, v in votes.most_common()),
        }

    def _simplify_cat(self, category: str) -> str:
        """Simplify category to match token vote labels."""
        if 'CODE' in category:
            return 'code'
        if 'SYNTAX' in category:
            return 'syntax'
        if 'SEMANTICS' in category:
            return 'nl'
        return 'fragment'

    def _infer_subcategory(self, examples: List[str], layer: int) -> str:
        """Use layer priors + example content to infer semantic subcategory."""
        # Layer priors (from our validated data)
        if layer <= 3:
            default = 'CODE'  # Early layers: lots of code tokens
        elif layer <= 10:
            default = 'CONCEPT'
        elif layer <= 20:
            default = 'CONCEPT'
        else:
            default = 'CONCEPT'

        # Try simple keyword matching on examples
        all_text = ' '.join(examples).lower()
        if any(kw in all_text for kw in ('city', 'country', 'capital', 'north', 'south', 'east', 'west')):
            return 'GEOGRAPHIC'
        if any(kw in all_text for kw in ('person', 'staff', 'people', 'name')):
            return 'ENTITY'
        if any(kw in all_text for kw in ('time', 'year', 'day', 'month', 'date')):
            return 'TEMPORAL'

        return default

    def generate_notes(self, result: Dict, auto_category: str) -> str:
        """Generate concise notes from classification result."""
        parts = []
        parts.append(f'Method: {result["method"]}.')
        if result['reasoning']:
            parts.append(result['reasoning'][0])
        if result['category'] != auto_category:
            parts.append(f'Disagree with auto ({auto_category}).')
        return ' '.join(parts)


# ══════════════════════════════════════════════════════════════════════════
# MAIN: Annotate CSV or Test against validated data
# ══════════════════════════════════════════════════════════════════════════
def annotate_csv(input_file: Path, output_file: Path, use_api: bool = False):
    """Run v2 classifier on all features in CSV."""
    classifier = FeatureClassifierV2(use_api=use_api)

    with open(input_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"V2 Classifier: annotating {len(rows)} features (API={'ON' if use_api else 'OFF'})")
    print("=" * 70)

    for idx, row in enumerate(rows, 1):
        layer = int(row['layer'])
        fid = int(row['feature_id'])
        examples = row['activation_examples']
        count = int(row['activation_count']) if row['activation_count'] else 0

        # Fetch NP data if API enabled
        np_data = None
        if use_api:
            np_data = classifier.fetch_np_explanation(layer, fid)
            time.sleep(0.5)

        result = classifier.classify_feature(layer, fid, examples, count, np_data)

        row['manual_category'] = result['category']
        row['confidence'] = result['confidence']
        row['notes'] = classifier.generate_notes(result, row.get('auto_category', ''))

        symbol = '✓' if result['confidence'] == 'HIGH' else '~' if result['confidence'] == 'MEDIUM' else '?'
        print(f"  [{idx:2d}/{len(rows)}] {symbol} {row['feature_label']:12s} -> {result['category']:20s} ({result['confidence']}) [{result['method']}]")

    # Save
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {output_file}")


def test_against_validated(validated_file: Path, use_api: bool = False):
    """
    Test v2 classifier against human-validated annotations.
    Computes agreement rate and shows disagreements.
    """
    classifier = FeatureClassifierV2(use_api=use_api)

    with open(validated_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Testing V2 classifier against {len(rows)} validated features")
    print(f"API: {'ON' if use_api else 'OFF'}")
    print("=" * 70)

    exact_match = 0
    broad_match = 0  # Match at top level (SEMANTICS vs SYNTAX vs POLYSEMANTIC)
    disagreements = []
    v2_cats = Counter()
    val_cats = Counter()

    for idx, row in enumerate(rows, 1):
        layer = int(row['layer'])
        fid = int(row['feature_id'])
        examples = row['activation_examples']
        count = int(row['activation_count']) if row['activation_count'] else 0
        validated = row['manual_category']

        np_data = None
        if use_api:
            np_data = classifier.fetch_np_explanation(layer, fid)
            time.sleep(0.5)

        result = classifier.classify_feature(layer, fid, examples, count, np_data)
        predicted = result['category']

        v2_cats[predicted] += 1
        val_cats[validated] += 1

        # Exact match
        if predicted == validated:
            exact_match += 1
        else:
            # Broad match (top-level)
            pred_top = predicted.split(':')[0] if ':' in predicted else predicted
            val_top = validated.split(':')[0] if ':' in validated else validated
            if pred_top == val_top:
                broad_match += 1
            else:
                disagreements.append({
                    'feature': row['feature_label'],
                    'validated': validated,
                    'predicted': predicted,
                    'method': result['method'],
                    'examples': examples,
                })

    total = len(rows)
    exact_pct = exact_match / total * 100
    broad_pct = (exact_match + broad_match) / total * 100

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Exact match:   {exact_match:3d}/{total} ({exact_pct:.1f}%)")
    print(f"Broad match:   {exact_match + broad_match:3d}/{total} ({broad_pct:.1f}%)")
    print(f"Disagree:      {len(disagreements):3d}/{total} ({len(disagreements)/total*100:.1f}%)")

    print(f"\nV2 distribution:")
    for cat, cnt in v2_cats.most_common():
        print(f"  {cat:25s}: {cnt:3d}")

    print(f"\nValidated distribution:")
    for cat, cnt in val_cats.most_common():
        print(f"  {cat:25s}: {cnt:3d}")

    if disagreements:
        print(f"\nTop-level disagreements ({len(disagreements)}):")
        for d in disagreements[:15]:
            print(f"  {d['feature']:12s}: validated={d['validated']:20s} v2={d['predicted']:20s} [{d['method']}]")
            print(f"    examples: {d['examples'][:80]}")

    return exact_pct, broad_pct


def main():
    parser = argparse.ArgumentParser(description='V2 Feature Classifier')
    parser.add_argument('--input', type=Path, help='Input CSV')
    parser.add_argument('--output', type=Path, help='Output CSV')
    parser.add_argument('--test', action='store_true', help='Test against validated CSV')
    parser.add_argument('--api', action='store_true', help='Use Neuronpedia API for explanations')

    args = parser.parse_args()

    if args.test:
        validated = BASE / "semantic_taxonomy_annotations_auto.csv"
        test_against_validated(validated, use_api=args.api)
    elif args.input and args.output:
        annotate_csv(args.input, args.output, use_api=args.api)
    else:
        # Default: test mode
        validated = BASE / "semantic_taxonomy_annotations_auto.csv"
        test_against_validated(validated, use_api=False)


if __name__ == '__main__':
    main()
