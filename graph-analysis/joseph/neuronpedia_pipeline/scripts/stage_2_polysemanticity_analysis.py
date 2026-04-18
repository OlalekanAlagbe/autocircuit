#!/usr/bin/env python3
"""Stage 2: Polysemanticity Analysis (D7).

Analyzes the 130 annotated bottleneck features for polysemantic patterns:
  1. Semantic category classification from NP explanations
  2. Cross-domain feature behavior (does a feature serve different roles?)
  3. Layer distribution of semantic categories
  4. Model comparison of semantic organization

Uses the updated bottleneck library with 130 annotated features (84 GEMMA + 46 original).

Output: data/stage_2_polysemanticity/polysemanticity_results.json + report.md

Usage:
    python scripts/stage_2_polysemanticity_analysis.py
"""

import json
import sys
import io
import math
import re
import datetime
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
BOTTLENECK_LIBRARY = DATA_DIR / 'stage_1_5_bottleneck_library.json'
DEFAULT_OUTPUT_DIR = DATA_DIR / 'stage_2_polysemanticity'

# Semantic categories and their keyword patterns
SEMANTIC_CATEGORIES = {
    'CODE': [
        r'\bcode\b', r'\bprogram', r'\bJava\b', r'\bPython\b', r'\bHTML\b',
        r'\bCSS\b', r'\bSQL\b', r'\bJavaScript\b', r'\bfunction\b',
        r'\bsnippet', r'\bimport\b', r'\bregex', r'\bsyntax\b',
        r'\bvariable', r'\btag\b', r'\bxml\b', r'\bformat',
    ],
    'SCIENCE': [
        r'\bscientific\b', r'\bbiology\b', r'\bchemistry\b', r'\bphysics\b',
        r'\bmedical\b', r'\bbiolog', r'\bcell\b', r'\bmolecu',
        r'\bstudy\b', r'\bstudies\b', r'\bexperiment', r'\bjournal\b',
        r'\bpublication', r'\bcitation', r'\bresearch\b',
    ],
    'LANGUAGE': [
        r'\bword\b', r'\bphrase\b', r'\bsentence\b', r'\btoken\b',
        r'\bpreposition', r'\bconjunction', r'\bpronoun', r'\bnoun\b',
        r'\bverb\b', r'\badjective', r'\blanguage\b', r'\bgramm',
        r'\bletter\b', r'\babbrev',
    ],
    'MATH': [
        r'\bmath', r'\bnumber', r'\bdigit', r'\bdecimal', r'\bequation',
        r'\bexpression\b', r'\bcalcul', r'\bpower\b.*ten\b',
    ],
    'DOMAIN_SPECIFIC': [
        r'\bgeograph', r'\bplace\b', r'\bcity\b', r'\bcountry\b',
        r'\bcapital\b', r'\bhistor', r'\bwar\b', r'\bpolitical\b',
        r'\bmilitary\b', r'\blegal\b', r'\bcourt\b', r'\blaw\b',
    ],
    'CONCEPTS': [
        r'\bemotion', r'\bfairness', r'\bimportanc', r'\bjudg',
        r'\bdesire\b', r'\bwarn', r'\bobligation', r'\bvisib',
    ],
    'ENTITIES': [
        r'\bname\b', r'\bperson\b', r'\bpeople\b', r'\borgani',
        r'\btitle\b', r'\bacron', r'\bauthor\b', r'\bbook\b',
    ],
}

# Unicode detection for multilingual content
MULTILINGUAL_PATTERNS = [
    r'\bIceland', r'\bGerman\b', r'\bVietnamese', r'\bFilipino',
    r'\bnon-English\b', r'\blanguage\b.*\b(possib|likely)',
    r'\bArabic\b', r'\bHebrew\b', r'\bKorean\b', r'\bChinese\b',
    r'\bJapanese\b', r'\bFrench\b', r'\bSpanish\b',
]


def classify_explanation(explanation):
    """Classify a feature's explanation into semantic categories.

    Returns list of matching categories (a feature can be multi-category = polysemantic).
    """
    if not explanation:
        return ['UNKNOWN']

    matches = []
    explanation_lower = explanation.lower()

    for category, patterns in SEMANTIC_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, explanation, re.IGNORECASE):
                matches.append(category)
                break  # One match per category is enough

    # Check multilingual
    is_multilingual = any(re.search(p, explanation, re.IGNORECASE) for p in MULTILINGUAL_PATTERNS)
    if is_multilingual:
        matches.append('MULTILINGUAL')

    if not matches:
        matches = ['OTHER']

    return matches


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  D7: POLYSEMANTICITY ANALYSIS")
    print("=" * 70)

    # Load bottleneck library
    print("\n[1/4] Loading annotated bottleneck library...")
    with open(BOTTLENECK_LIBRARY, 'r', encoding='utf-8') as f:
        library = json.load(f)

    ccf = library.get('cross_circuit_features', {})
    annotated = {k: v for k, v in ccf.items() if v.get('explanation')}
    unannotated = {k: v for k, v in ccf.items() if not v.get('explanation')}

    print(f"  Total: {len(ccf)}")
    print(f"  Annotated: {len(annotated)}")
    print(f"  Unannotated: {len(unannotated)}")

    # Classify each annotated feature
    print("\n[2/4] Classifying features...")
    feature_classes = {}
    category_counts = Counter()
    polysemantic_count = 0
    monosemantic_count = 0

    for label, feat in annotated.items():
        expl = feat.get('explanation', '')
        categories = classify_explanation(expl)
        is_poly = len(categories) > 1

        feature_classes[label] = {
            'categories': categories,
            'is_polysemantic': is_poly,
            'explanation': expl[:100],
            'layer': feat['layer'],
            'n_categories': len(categories),
        }

        for cat in categories:
            category_counts[cat] += 1

        if is_poly:
            polysemantic_count += 1
        else:
            monosemantic_count += 1

    poly_rate = polysemantic_count / len(annotated) if annotated else 0
    print(f"  Monosemantic: {monosemantic_count} ({100*(1-poly_rate):.0f}%)")
    print(f"  Polysemantic: {polysemantic_count} ({100*poly_rate:.0f}%)")
    print(f"\n  Category counts:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:20s}: {count:3d} ({100*count/len(annotated):.0f}%)")

    # Analyze by layer
    print("\n[3/4] Layer-level analysis...")
    layer_categories = defaultdict(lambda: Counter())
    layer_poly_rate = defaultdict(lambda: {'poly': 0, 'mono': 0})

    for label, cls in feature_classes.items():
        layer = cls['layer']
        for cat in cls['categories']:
            layer_categories[layer][cat] += 1
        if cls['is_polysemantic']:
            layer_poly_rate[layer]['poly'] += 1
        else:
            layer_poly_rate[layer]['mono'] += 1

    print(f"  {'Layer':>5s} {'Total':>5s} {'Poly%':>6s} {'Top Category':>20s}")
    print(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*20}")
    for layer in sorted(layer_categories.keys()):
        total = layer_poly_rate[layer]['poly'] + layer_poly_rate[layer]['mono']
        poly_pct = 100 * layer_poly_rate[layer]['poly'] / total if total > 0 else 0
        top_cat = layer_categories[layer].most_common(1)[0][0] if layer_categories[layer] else '?'
        print(f"  L{layer:>3d} {total:5d} {poly_pct:5.0f}% {top_cat:>20s}")

    # Model comparison
    print("\n  Model comparison:")
    gemma_features = {k: v for k, v in feature_classes.items()
                      if any(a['model'] == 'gemma-2-2b'
                             for a in ccf[k].get('appearances', []))}
    qwen_features = {k: v for k, v in feature_classes.items()
                     if any(a['model'] == 'qwen3-4b'
                            for a in ccf[k].get('appearances', []))}

    for model_name, feats in [('GEMMA', gemma_features), ('QWEN', qwen_features)]:
        if not feats:
            continue
        n_poly = sum(1 for v in feats.values() if v['is_polysemantic'])
        model_cats = Counter()
        for v in feats.values():
            for c in v['categories']:
                model_cats[c] += 1
        print(f"  {model_name}: {len(feats)} features, {n_poly} polysemantic ({100*n_poly/len(feats):.0f}%)")
        print(f"    Top 3: {', '.join(f'{c}({n})' for c, n in model_cats.most_common(3))}")

    # Cross-domain features: features that appear in multiple knowledge categories
    print("\n  Cross-domain feature analysis:")
    cross_domain_features = []
    for label, feat in ccf.items():
        appearances = feat.get('appearances', [])
        # Determine which prompt categories this feature appears in
        prompt_categories = set()
        for app in appearances:
            circuit = app.get('circuit', '')
            if 'chemical-symbol' in circuit:
                prompt_categories.add('chemistry')
            elif 'capital-of' in circuit:
                prompt_categories.add('geography')
            elif any(h in circuit for h in ['war', 'moon', 'berlin', 'titanic', 'independence',
                                              'revolution', 'columbus', 'fire', 'mandela']):
                prompt_categories.add('history')

        if len(prompt_categories) >= 2:
            cross_domain_features.append({
                'label': label,
                'prompt_categories': list(prompt_categories),
                'n_categories': len(prompt_categories),
                'semantic_class': feature_classes.get(label, {}).get('categories', ['?']),
                'explanation': feat.get('explanation', '')[:80],
            })

    print(f"  Features in 2+ knowledge domains: {len(cross_domain_features)}")
    print(f"  Features in all 3 domains: {sum(1 for f in cross_domain_features if f['n_categories'] == 3)}")

    # What semantic types are most common among cross-domain features?
    cross_domain_cats = Counter()
    for f in cross_domain_features:
        for c in f['semantic_class']:
            cross_domain_cats[c] += 1
    print(f"  Cross-domain semantic types: {', '.join(f'{c}({n})' for c, n in cross_domain_cats.most_common(5))}")

    # Build results
    print("\n[4/4] Saving results...")
    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_annotated': len(annotated),
            'n_unannotated': len(unannotated),
            'n_classified': len(feature_classes),
        },
        'overall': {
            'polysemantic_count': polysemantic_count,
            'monosemantic_count': monosemantic_count,
            'polysemantic_rate': poly_rate,
            'category_counts': dict(category_counts),
        },
        'layer_analysis': {
            str(layer): {
                'categories': dict(cats),
                'poly_rate': layer_poly_rate[layer]['poly'] / (layer_poly_rate[layer]['poly'] + layer_poly_rate[layer]['mono'])
                if (layer_poly_rate[layer]['poly'] + layer_poly_rate[layer]['mono']) > 0 else 0,
            }
            for layer, cats in layer_categories.items()
        },
        'cross_domain': {
            'count': len(cross_domain_features),
            'in_all_3': sum(1 for f in cross_domain_features if f['n_categories'] == 3),
            'semantic_types': dict(cross_domain_cats),
            'features': cross_domain_features[:20],  # Top 20
        },
        'feature_classifications': feature_classes,
    }

    json_path = output_dir / 'polysemanticity_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # Generate report
    lines = []
    lines.append("# Stage 2: Polysemanticity Analysis Report\n")
    lines.append(f"**Generated**: {results['metadata']['timestamp']}")
    lines.append(f"**Features analyzed**: {len(annotated)}/{len(ccf)} (annotated/total)\n")
    lines.append("---\n")

    lines.append("## Overall Classification\n")
    lines.append(f"- **Monosemantic**: {monosemantic_count} ({100*(1-poly_rate):.0f}%)")
    lines.append(f"- **Polysemantic** (2+ categories): {polysemantic_count} ({100*poly_rate:.0f}%)\n")

    lines.append("## Semantic Category Distribution\n")
    lines.append("| Category | Count | % of Annotated |")
    lines.append("|----------|-------|----------------|")
    for cat, count in category_counts.most_common():
        lines.append(f"| {cat} | {count} | {100*count/len(annotated):.0f}% |")
    lines.append("")

    lines.append("## Layer Distribution\n")
    lines.append("| Layer | Total | Poly% | Top Category |")
    lines.append("|-------|-------|-------|-------------|")
    for layer in sorted(layer_categories.keys()):
        total = layer_poly_rate[layer]['poly'] + layer_poly_rate[layer]['mono']
        poly_pct = 100 * layer_poly_rate[layer]['poly'] / total if total > 0 else 0
        top_cat = layer_categories[layer].most_common(1)[0][0] if layer_categories[layer] else '?'
        lines.append(f"| L{layer} | {total} | {poly_pct:.0f}% | {top_cat} |")
    lines.append("")

    lines.append("## Cross-Domain Features\n")
    lines.append(f"- Features in 2+ knowledge domains: {len(cross_domain_features)}")
    lines.append(f"- Features in all 3 domains: {sum(1 for f in cross_domain_features if f['n_categories'] == 3)}\n")

    lines.append("## Sample Cross-Domain Features\n")
    lines.append("| Feature | Domains | Semantic Type | Explanation |")
    lines.append("|---------|---------|---------------|-------------|")
    for f in cross_domain_features[:15]:
        domains = ', '.join(f['prompt_categories'])
        sem = ', '.join(f['semantic_class'])
        lines.append(f"| {f['label']} | {domains} | {sem} | {f['explanation'][:50]} |")
    lines.append("")

    report_path = output_dir / 'polysemanticity_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report: {report_path}")

    print("\n" + "=" * 70)
    print("  D7 COMPLETE")
    print("=" * 70)
    print(f"  Polysemantic: {polysemantic_count}/{len(annotated)} ({100*poly_rate:.0f}%)")
    print(f"  Cross-domain: {len(cross_domain_features)} features")
    print(f"  Top category: {category_counts.most_common(1)[0][0]} ({category_counts.most_common(1)[0][1]})")
    print("\nDone!")


if __name__ == '__main__':
    main()
