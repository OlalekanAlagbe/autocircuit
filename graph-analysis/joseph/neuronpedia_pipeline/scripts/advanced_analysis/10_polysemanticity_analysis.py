"""
Polysemanticity Analysis - Measure Supernode Semantic Purity
Determines if supernodes are monosemantic (one concept) or polysemantic (multiple concepts)

This analysis reveals:
- How "pure" each supernode is semantically
- Whether supernodes handle multiple unrelated concepts
- Sub-theme clusters within supernodes
- Which supernodes are best intervention targets (monosemantic = more predictable)
"""

import json
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import sys

# Add scripts to path for imports (parent.parent because we're in scripts/advanced_analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from path_manager import PathManager


class PolysemanticityAnalyzer:
    """Analyze semantic purity of supernodes"""

    def __init__(self, analysis_path):
        """
        Args:
            analysis_path: Path to analysis JSON file
        """
        with open(analysis_path) as f:
            self.analysis = json.load(f)

        self.metadata = self.analysis['metadata']
        self.prompt = self.metadata['prompt']
        self.supernodes = self.analysis.get('louvain_supernodes', {}).get('supernodes', {})
        self.feature_descriptions = self.analysis.get('feature_descriptions', {})

        print(f"[OK] Loaded circuit: {self.prompt}")
        print(f"  Supernodes: {len(self.supernodes)}")
        print(f"  Feature descriptions: {len(self.feature_descriptions)}")

    def extract_supernode_descriptions(self, supernode_id):
        """
        Extract feature descriptions for a supernode

        Returns:
            List of description strings
        """
        if str(supernode_id) not in self.supernodes:
            return []

        sn_data = self.supernodes[str(supernode_id)]
        nodes = sn_data.get('nodes', [])

        descriptions = []

        for node_id in nodes:
            # Look up description
            desc = self.feature_descriptions.get(node_id, '')

            if desc and isinstance(desc, str) and len(desc) > 5:
                # Clean description
                desc = desc.replace('Activates on:', '').replace('activates on', '').strip()
                descriptions.append(desc)

        return descriptions

    def extract_keywords(self, descriptions):
        """
        Extract keywords from descriptions using simple NLP

        Returns:
            Counter of keywords
        """
        # Combine all descriptions
        text = ' '.join(descriptions).lower()

        # Remove common prefixes and noise
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation

        # Extract words
        words = text.split()

        # Filter stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'your', 'he', 'she', 'him', 'her', 'i', 'my', 'me'
        }

        # Keep only meaningful words (4+ chars, not stop words)
        keywords = [w for w in words if len(w) >= 4 and w not in stop_words]

        return Counter(keywords)

    def compute_purity_score(self, keyword_counter, top_n=5):
        """
        Compute semantic purity score

        High purity = dominant theme (top keywords account for most occurrences)
        Low purity = mixed themes (keywords evenly distributed)

        Returns:
            float between 0 (polysemantic) and 1 (monosemantic)
        """
        if not keyword_counter:
            return 0.0

        total_count = sum(keyword_counter.values())
        top_keywords = keyword_counter.most_common(top_n)
        top_count = sum(count for _, count in top_keywords)

        # Purity = concentration of top keywords
        purity = top_count / total_count if total_count > 0 else 0.0

        return purity

    def identify_themes(self, keyword_counter, min_count=2):
        """
        Identify semantic themes from keywords

        Returns:
            List of (theme, keywords) tuples
        """
        # Predefined theme patterns
        theme_patterns = {
            'Geographic': ['city', 'cities', 'state', 'states', 'country', 'capitol',
                          'capital', 'located', 'location', 'place', 'region'],
            'Political': ['president', 'political', 'party', 'republican', 'democratic',
                         'election', 'government', 'congress', 'politics'],
            'Numerical': ['number', 'numbers', 'digit', 'count', 'quantity', 'amount'],
            'Temporal': ['time', 'date', 'year', 'month', 'when', 'period'],
            'Entity': ['name', 'person', 'people', 'entity', 'who'],
            'Syntactic': ['token', 'word', 'syntax', 'grammar', 'parsing'],
            'Technical': ['function', 'class', 'code', 'programming', 'variable']
        }

        # Match keywords to themes
        theme_matches = defaultdict(list)

        for keyword, count in keyword_counter.most_common(20):
            if count < min_count:
                continue

            matched = False
            for theme, patterns in theme_patterns.items():
                if keyword in patterns or any(pattern in keyword for pattern in patterns):
                    theme_matches[theme].append((keyword, count))
                    matched = True
                    break

            if not matched:
                theme_matches['Other'].append((keyword, count))

        # Convert to sorted list
        themes = [(theme, keywords) for theme, keywords in theme_matches.items()]
        themes.sort(key=lambda x: sum(c for _, c in x[1]), reverse=True)

        return themes

    def analyze_all_supernodes(self):
        """Analyze polysemanticity for all supernodes"""

        poly_data = {}

        for sn_id in self.supernodes.keys():
            sn_id_int = int(sn_id)

            # Extract descriptions
            descriptions = self.extract_supernode_descriptions(sn_id_int)

            if not descriptions or len(descriptions) < 3:
                # Skip supernodes with too few descriptions
                continue

            # Extract keywords
            keywords = self.extract_keywords(descriptions)

            # Compute purity
            purity = self.compute_purity_score(keywords, top_n=5)

            # Identify themes
            themes = self.identify_themes(keywords)

            poly_data[sn_id_int] = {
                'num_descriptions': len(descriptions),
                'num_unique_keywords': len(keywords),
                'top_keywords': keywords.most_common(10),
                'purity_score': purity,
                'themes': themes,
                'classification': 'monosemantic' if purity > 0.7 else 'polysemantic' if purity < 0.4 else 'mixed'
            }

        return poly_data

    def generate_polysemanticity_report(self, poly_data, output_path):
        """Generate text report of polysemanticity analysis"""

        report = []
        report.append("=" * 80)
        report.append("POLYSEMANTICITY ANALYSIS")
        report.append("=" * 80)
        report.append("")
        report.append(f"Circuit: {self.prompt}")
        report.append(f"Analyzed Supernodes: {len(poly_data)}")
        report.append("")

        # Overall statistics
        report.append("=" * 80)
        report.append("OVERALL PURITY DISTRIBUTION")
        report.append("=" * 80)

        if not poly_data:
            report.append("No supernodes with sufficient feature descriptions (minimum 3 required)")
            report.append("Note: Feature descriptions may need to be fetched from Neuronpedia API")
            report.append("")
        else:
            classifications = Counter([data['classification'] for data in poly_data.values()])
            total = sum(classifications.values())

            if total > 0:
                report.append(f"Monosemantic (purity > 0.7): {classifications['monosemantic']} ({classifications['monosemantic']/total*100:.1f}%)")
                report.append(f"Mixed (purity 0.4-0.7): {classifications['mixed']} ({classifications['mixed']/total*100:.1f}%)")
                report.append(f"Polysemantic (purity < 0.4): {classifications['polysemantic']} ({classifications['polysemantic']/total*100:.1f}%)")
            report.append("")

        # Supernode-by-supernode analysis
        report.append("=" * 80)
        report.append("SUPERNODE PURITY SCORES (sorted by purity)")
        report.append("=" * 80)

        sorted_supernodes = sorted(poly_data.items(), key=lambda x: x[1]['purity_score'], reverse=True)

        for sn_id, data in sorted_supernodes:
            report.append(f"\nSupernode {sn_id}: {data['classification'].upper()} (purity={data['purity_score']:.2f})")
            report.append(f"  Descriptions: {data['num_descriptions']}, Unique Keywords: {data['num_unique_keywords']}")

            # Top keywords
            top_kw = [f"{kw}({count})" for kw, count in data['top_keywords'][:5]]
            report.append(f"  Top Keywords: {', '.join(top_kw)}")

            # Themes
            if data['themes']:
                theme_str = ', '.join([f"{theme}({len(kws)})" for theme, kws in data['themes'][:3]])
                report.append(f"  Themes: {theme_str}")

        # Key findings
        report.append("")
        report.append("=" * 80)
        report.append("KEY INSIGHTS")
        report.append("=" * 80)

        if poly_data:
            # Most monosemantic
            monosemantic = [(sn_id, data) for sn_id, data in sorted_supernodes if data['classification'] == 'monosemantic']
            if monosemantic:
                report.append("\n1. MOST MONOSEMANTIC (purest supernodes):")
                for sn_id, data in monosemantic[:3]:
                    top_theme = data['themes'][0][0] if data['themes'] else 'Unknown'
                    report.append(f"   SN{sn_id}: {data['purity_score']:.2f} purity, Theme: {top_theme}")
            else:
                report.append("\n1. MOST MONOSEMANTIC: None found")

            # Most polysemantic
            polysemantic = [(sn_id, data) for sn_id, data in sorted_supernodes if data['classification'] == 'polysemantic']
            if polysemantic:
                report.append("\n2. MOST POLYSEMANTIC (mixed concepts):")
                polysemantic_sorted = sorted(polysemantic, key=lambda x: x[1]['purity_score'])
                for sn_id, data in polysemantic_sorted[:3]:
                    num_themes = len(data['themes'])
                    report.append(f"   SN{sn_id}: {data['purity_score']:.2f} purity, {num_themes} themes detected")
            else:
                report.append("\n2. MOST POLYSEMANTIC: None found")

            # Best intervention targets
            report.append("\n3. BEST INTERVENTION TARGETS (monosemantic = more predictable):")
            intervention_targets = sorted(monosemantic, key=lambda x: x[1]['num_descriptions'], reverse=True)
            for sn_id, data in intervention_targets[:3]:
                report.append(f"   SN{sn_id}: {data['purity_score']:.2f} purity, {data['num_descriptions']} features")

        report.append("")
        report.append("=" * 80)

        # Write report
        output_text = "\n".join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"[OK] Polysemanticity report saved to: {output_path}")
        return output_text

    def visualize_polysemanticity(self, poly_data, output_path):
        """Create visualization of polysemanticity scores"""

        if not poly_data:
            # Create empty placeholder visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available\nInsufficient feature descriptions (minimum 3 per supernode)',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            plt.suptitle(f'Polysemanticity Analysis\n"{self.prompt}"',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Polysemanticity visualization saved to: {output_path}")
            plt.close()
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Chart 1: Purity score distribution
        supernode_ids = sorted(poly_data.keys())
        purity_scores = [poly_data[sn_id]['purity_score'] for sn_id in supernode_ids]
        classifications = [poly_data[sn_id]['classification'] for sn_id in supernode_ids]

        colors = {'monosemantic': 'green', 'mixed': 'orange', 'polysemantic': 'red'}
        bar_colors = [colors[c] for c in classifications]

        ax1.bar(range(len(supernode_ids)), purity_scores, color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0.7, color='green', linestyle='--', label='Monosemantic threshold')
        ax1.axhline(y=0.4, color='red', linestyle='--', label='Polysemantic threshold')
        ax1.set_xticks(range(len(supernode_ids)))
        ax1.set_xticklabels([f"SN{sn_id}" for sn_id in supernode_ids], fontsize=9, rotation=45, ha='right')
        ax1.set_ylabel('Purity Score', fontsize=11, fontweight='bold')
        ax1.set_title('Semantic Purity by Supernode', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Chart 2: Classification pie chart
        classification_counts = Counter(classifications)

        ax2.pie(classification_counts.values(),
               labels=[f"{k}\n({v})" for k, v in classification_counts.items()],
               colors=[colors[k] for k in classification_counts.keys()],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Supernode Classification Distribution', fontsize=12, fontweight='bold')

        plt.suptitle(f'Polysemanticity Analysis\n"{self.prompt}"',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Polysemanticity visualization saved to: {output_path}")
        plt.close()

    def analyze_polysemanticity(self, output_dir):
        """Main method to analyze polysemanticity"""

        print("\n" + "="*60)
        print("STEP 1: Extract Feature Descriptions")
        print("="*60)
        print(f"Found {len(self.feature_descriptions)} feature descriptions")

        print("\n" + "="*60)
        print("STEP 2: Analyze Semantic Purity")
        print("="*60)

        poly_data = self.analyze_all_supernodes()
        print(f"Analyzed {len(poly_data)} supernodes with sufficient descriptions")

        print("\n" + "="*60)
        print("STEP 3: Generate Report")
        print("="*60)

        report_path = output_dir / "polysemanticity_report.txt"
        self.generate_polysemanticity_report(poly_data, report_path)

        print("\n" + "="*60)
        print("STEP 4: Create Visualizations")
        print("="*60)

        viz_path = output_dir / "polysemanticity_viz.png"
        self.visualize_polysemanticity(poly_data, viz_path)

        # Save data
        data_path = output_dir / "polysemanticity_data.json"
        with open(data_path, 'w') as f:
            json.dump(poly_data, f, indent=2, default=str)
        print(f"[OK] Polysemanticity data saved to: {data_path}")

        return poly_data


def main():
    parser = argparse.ArgumentParser(description='Analyze supernode polysemanticity')
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
    output_dir = pm.polysemanticity_dir(prompt)
    print(f"\nOutput directory: {output_dir}")

    # Analyze polysemanticity
    analyzer = PolysemanticityAnalyzer(analysis_path)
    poly_data = analyzer.analyze_polysemanticity(output_dir)

    print("\n" + "="*60)
    print("POLYSEMANTICITY ANALYSIS COMPLETE")
    print("="*60)
    print(f"Report: {output_dir / 'polysemanticity_report.txt'}")
    print(f"Visualization: {output_dir / 'polysemanticity_viz.png'}")
    print(f"Data: {output_dir / 'polysemanticity_data.json'}")


if __name__ == "__main__":
    main()
