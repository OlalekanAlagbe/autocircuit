#!/usr/bin/env python3
"""
Neuronpedia Pipeline Orchestrator

Runs the full circuit analysis pipeline for one or more models:
  Steps 1-4:    Per-model graph generation, conversion, analysis, visualization
  Stage 1.5:    Cross-circuit bottleneck comparison
  Stage 2:      Enhanced visualizations
  Stage 3:      Steering validation (optional, --steer-quick or --steer-full flag)
  Steps 5-10:   Advanced analysis (optional, --advanced flag)

Usage examples:
  python run_full_pipeline.py --prompt "The chemical symbol for Argon is"
  python run_full_pipeline.py --prompt "Paris is the capital of" --model gemma-2-2b
  python run_full_pipeline.py --prompt "Water boils at 100 degrees" --model both --advanced
  python run_full_pipeline.py --prompt "The president lives in" --skip-api --api-limit 50
  python run_full_pipeline.py --prompt "The longest river in Africa is" --steer-quick
"""

import argparse
import glob
import io
import os
import re
import subprocess
import sys
import time

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# Path helpers (mirror PathManager logic without importing it, so this script
# stays self-contained and works from the repo root)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, 'scripts')
ADVANCED_DIR = os.path.join(SCRIPTS_DIR, 'advanced_analysis')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PROMPTS_DIR = os.path.join(DATA_DIR, 'prompts')

MODELS = ['gemma-2-2b', 'qwen3-4b']


def slugify(text: str) -> str:
    """Convert prompt text to a URL-safe slug (matches PathManager.slugify)."""
    text = text.replace('<bos>', '').replace('<|im_end|>', '').strip()
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = text.strip('-')
    if len(text) > 50:
        text = text[:50].rstrip('-')
    return text


def short_slug(prompt: str) -> str:
    """First 4 words of the prompt, hyphen-joined (matches PathManager._get_file_prefix)."""
    words = re.sub(r'[^\w\s-]', '', prompt.lower()).split()[:4]
    return '-'.join(words)


def circuit_dir_name(model: str, prompt: str) -> str:
    """Directory name under data/prompts/ for a given model + prompt."""
    clean_model = re.sub(r'[^\w-]', '-', model.lower())
    return f"{clean_model}_{slugify(prompt)}"


def find_raw_graph(model: str, prompt: str) -> str:
    """Find the raw graph JSON produced by Step 1 (glob-based, resilient to naming)."""
    circ_dir = os.path.join(PROMPTS_DIR, circuit_dir_name(model, prompt))
    gen_dir = os.path.join(circ_dir, '1_generation')
    pattern = os.path.join(gen_dir, '*_raw_graph.json')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        return matches[0]
    # Fallback: any JSON in the generation directory
    fallback = sorted(glob.glob(os.path.join(gen_dir, '*.json')), key=os.path.getmtime, reverse=True)
    if fallback:
        return fallback[0]
    return None


def find_converted_graph(model: str, prompt: str) -> str:
    """Find the converted graph JSON produced by Step 2 (glob-based)."""
    circ_dir = os.path.join(PROMPTS_DIR, circuit_dir_name(model, prompt))
    conv_dir = os.path.join(circ_dir, '2_conversion')
    pattern = os.path.join(conv_dir, '*_converted_graph.json')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        return matches[0]
    fallback = sorted(glob.glob(os.path.join(conv_dir, '*.json')), key=os.path.getmtime, reverse=True)
    if fallback:
        return fallback[0]
    return None


def find_analysis_file(model: str, prompt: str) -> str:
    """Find the circuit analysis JSON produced by Step 3."""
    circ_dir = os.path.join(PROMPTS_DIR, circuit_dir_name(model, prompt))
    analysis_dir = os.path.join(circ_dir, '3_analysis')
    pattern = os.path.join(analysis_dir, '*_circuit_analysis.json')
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        return matches[0]
    return None


# ============================================================================
# Execution helpers
# ============================================================================

def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def run_step(cmd: list, description: str, stdin_text: str = None) -> None:
    """
    Run a subprocess step with clear status output and timing.

    Args:
        cmd: Command as a list (first element should be sys.executable).
        description: Human-readable step description printed before running.
        stdin_text: Optional text to pipe to the process's stdin.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    # Show the command being run (abbreviated for readability)
    display_cmd = ' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd)
    print(f"  CMD: {display_cmd}")
    print()

    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            input=stdin_text,
            text=True if stdin_text is not None else None,
            stdin=None if stdin_text is not None else subprocess.PIPE,
            timeout=1800,  # 30 minute timeout per step
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n[TIMEOUT] {description} exceeded 30-minute limit ({format_duration(elapsed)})")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Command not found: {e}")
        sys.exit(1)

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAILED] {description}")
        print(f"  Exit code: {result.returncode}")
        print(f"  Duration:  {format_duration(elapsed)}")
        sys.exit(1)

    print(f"\n  [OK] {description} ({format_duration(elapsed)})")


# ============================================================================
# Pipeline phases
# ============================================================================

def run_core_pipeline(model: str, prompt: str) -> str:
    """
    Run Steps 1-4 for a single model.

    Returns:
        The circuit directory name (e.g. 'gemma-2-2b_the-chemical-symbol-for-argon-is').
    """
    circ_name = circuit_dir_name(model, prompt)
    print(f"\n{'#' * 70}")
    print(f"#  MODEL: {model}")
    print(f"#  PROMPT: \"{prompt}\"")
    print(f"#  CIRCUIT: {circ_name}")
    print(f"{'#' * 70}")

    # -- Step 1: Generate graph --
    run_step(
        [sys.executable, os.path.join(SCRIPTS_DIR, '1_generate_graph.py'),
         '--prompt', prompt, '--model', model],
        f"Step 1: Generate attribution graph [{model}]",
    )

    raw_graph = find_raw_graph(model, prompt)
    if not raw_graph:
        print(f"[ERROR] No raw graph found after Step 1 for {model}. Expected in:")
        print(f"  {os.path.join(PROMPTS_DIR, circ_name, '1_generation')}")
        sys.exit(1)
    print(f"  Raw graph: {os.path.basename(raw_graph)}")

    # -- Step 2: Convert graph --
    run_step(
        [sys.executable, os.path.join(SCRIPTS_DIR, '2_convert_graph.py'),
         '--file', raw_graph],
        f"Step 2: Convert graph to pipeline format [{model}]",
    )

    converted = find_converted_graph(model, prompt)
    if not converted:
        print(f"[ERROR] No converted graph found after Step 2 for {model}. Expected in:")
        print(f"  {os.path.join(PROMPTS_DIR, circ_name, '2_conversion')}")
        sys.exit(1)
    print(f"  Converted graph: {os.path.basename(converted)}")

    # -- Step 3: Analyze circuit --
    # Pipe "3" to stdin to auto-select the "Smart" feature-fetch strategy
    run_step(
        [sys.executable, os.path.join(SCRIPTS_DIR, '3_analyze_circuit.py'),
         '--file', converted],
        f"Step 3: Analyze circuit and detect supernodes [{model}]",
        stdin_text="3\n",
    )

    # -- Step 3b: Traceback paths --
    run_step(
        [sys.executable, os.path.join(SCRIPTS_DIR, '3b_traceback_paths.py'),
         '--file', converted],
        f"Step 3b: Traceback attribution paths [{model}]",
    )

    # -- Step 4: Visualize --
    run_step(
        [sys.executable, os.path.join(SCRIPTS_DIR, '4_visualize.py'),
         '--file', converted],
        f"Step 4: Generate visualizations [{model}]",
    )

    return circ_name


def run_cross_circuit(circuit_dirs: list, skip_api: bool, api_limit: int) -> None:
    """Run Stage 1.5 (cross-circuit bottlenecks) and Stage 2 (enhanced viz)."""

    # -- Stage 1.5: Cross-circuit bottleneck comparison --
    stage15_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, 'stage_1_5_cross_circuit_bottlenecks.py'),
    ]
    if skip_api:
        stage15_cmd.append('--skip-api')
    stage15_cmd.extend(['--api-limit', str(api_limit)])

    run_step(
        stage15_cmd,
        "Stage 1.5: Cross-circuit bottleneck comparison",
    )

    # -- Stage 2: Enhanced visualizations --
    stage2_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, 'stage_2_enhanced_visualizations.py'),
    ]
    for circ_dir in circuit_dirs:
        stage2_cmd.extend(['--circuit', circ_dir])

    run_step(
        stage2_cmd,
        f"Stage 2: Enhanced visualizations ({len(circuit_dirs)} circuits)",
    )


def run_steering_validation(quick: bool = True, rate_delay: float = 36.0) -> None:
    """Run Stage 3: Steering validation of cross-circuit bottleneck features."""

    print(f"\n{'#' * 70}")
    print(f"#  STAGE 3: STEERING VALIDATION")
    print(f"{'#' * 70}")

    mode = 'quick' if quick else 'full'
    steer_cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, '5_steering_validation.py'),
    ]
    if quick:
        steer_cmd.append('--quick')
    steer_cmd.extend(['--rate-delay', str(rate_delay)])

    run_step(
        steer_cmd,
        f"Stage 3: Steering validation ({mode} mode)",
    )


def run_advanced_pipeline(models_run: list, prompt: str) -> None:
    """Run Steps 5-10 (advanced analysis) for each model that was processed."""

    print(f"\n{'#' * 70}")
    print(f"#  ADVANCED ANALYSIS (Steps 5-10)")
    print(f"{'#' * 70}")

    # -- Step 5: Compare prompts (runs once, not per-model) --
    run_step(
        [sys.executable, os.path.join(ADVANCED_DIR, '5_compare_prompts.py')],
        "Step 5: Compare prompts across models",
    )

    # Steps 7-10 run per-model
    for model in models_run:
        converted = find_converted_graph(model, prompt)
        analysis = find_analysis_file(model, prompt)

        if not converted:
            print(f"[WARNING] No converted graph for {model}, skipping advanced steps 7-10.")
            continue

        # Use --analysis if available, else fall back to --prompt
        analysis_flag = ['--analysis', analysis] if analysis else ['--prompt', prompt]

        # -- Step 7: Extract minimal pathways --
        run_step(
            [sys.executable, os.path.join(ADVANCED_DIR, '7_extract_minimal_pathways.py')]
            + analysis_flag,
            f"Step 7: Extract minimal pathways [{model}]",
        )

        # -- Step 8: Supernode evolution --
        run_step(
            [sys.executable, os.path.join(ADVANCED_DIR, '8_supernode_evolution.py')]
            + analysis_flag,
            f"Step 8: Supernode evolution [{model}]",
        )

        # -- Step 9: Steering analysis --
        run_step(
            [sys.executable, os.path.join(ADVANCED_DIR, '9_steering_analysis.py')]
            + analysis_flag,
            f"Step 9: Steering analysis [{model}]",
        )

        # -- Step 10: Polysemanticity analysis --
        run_step(
            [sys.executable, os.path.join(ADVANCED_DIR, '10_polysemanticity_analysis.py')]
            + analysis_flag,
            f"Step 10: Polysemanticity analysis [{model}]",
        )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Neuronpedia Pipeline Orchestrator - run the full circuit analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --prompt "The chemical symbol for Argon is"
  %(prog)s --prompt "Paris is the capital of" --model gemma-2-2b
  %(prog)s --prompt "Water boils at 100 degrees" --model both --advanced
  %(prog)s --prompt "The president lives in" --skip-api --api-limit 50
        ''',
    )
    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Prompt text to analyze (e.g. "The chemical symbol for Argon is")',
    )
    parser.add_argument(
        '--model', type=str, default='both',
        choices=['gemma-2-2b', 'qwen3-4b', 'both'],
        help='Model to run (default: both)',
    )
    parser.add_argument(
        '--advanced', action='store_true',
        help='Also run advanced analysis steps 5-10',
    )
    parser.add_argument(
        '--skip-api', action='store_true',
        help='Skip Neuronpedia API queries in Stage 1.5',
    )
    parser.add_argument(
        '--api-limit', type=int, default=100,
        help='Maximum API queries for Stage 1.5 (default: 100)',
    )
    parser.add_argument(
        '--steer-quick', action='store_true',
        help='Run Stage 3 steering validation in quick mode (top 5 features, ~20 min)',
    )
    parser.add_argument(
        '--steer-full', action='store_true',
        help='Run Stage 3 steering validation in full mode (all features, ~10 hours)',
    )
    parser.add_argument(
        '--steer-rate-delay', type=float, default=36.0,
        help='Seconds between steering API calls (default: 36 = 100/hr)',
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.model == 'both':
        models_to_run = list(MODELS)
    else:
        models_to_run = [args.model]

    # Print banner
    pipeline_start = time.time()
    print("=" * 70)
    print("  NEURONPEDIA PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"  Prompt:    \"{args.prompt}\"")
    print(f"  Models:    {', '.join(models_to_run)}")
    print(f"  Advanced:  {'Yes' if args.advanced else 'No'}")
    steer_mode = 'Quick' if args.steer_quick else ('Full' if args.steer_full else 'No')
    print(f"  Steering:  {steer_mode}")
    print(f"  Skip API:  {'Yes' if args.skip_api else 'No'}")
    print(f"  API limit: {args.api_limit}")
    print(f"  Python:    {sys.executable}")
    print("=" * 70)

    # ---- Phase 1: Core pipeline per model (Steps 1-4) ----
    circuit_dirs = []
    for model in models_to_run:
        circ_name = run_core_pipeline(model, args.prompt)
        circuit_dirs.append(circ_name)

    # ---- Phase 2: Cross-circuit analysis (Stage 1.5 + Stage 2) ----
    print(f"\n{'#' * 70}")
    print(f"#  CROSS-CIRCUIT ANALYSIS")
    print(f"{'#' * 70}")
    run_cross_circuit(circuit_dirs, args.skip_api, args.api_limit)

    # ---- Phase 3: Steering validation (Stage 3, optional) ----
    if args.steer_quick or args.steer_full:
        run_steering_validation(
            quick=args.steer_quick,
            rate_delay=args.steer_rate_delay,
        )

    # ---- Phase 4: Advanced analysis (Steps 5-10, optional) ----
    if args.advanced:
        run_advanced_pipeline(models_to_run, args.prompt)

    # ---- Summary ----
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE  ({format_duration(total_elapsed)} total)")
    print(f"{'=' * 70}")
    print()
    print("Output directories:")
    for circ_dir in circuit_dirs:
        full_path = os.path.join(PROMPTS_DIR, circ_dir)
        print(f"  {full_path}")
    print()
    print("Cross-circuit outputs:")
    print(f"  {os.path.join(DATA_DIR, 'stage_1_5_bottleneck_library.json')}")
    print(f"  {os.path.join(DATA_DIR, 'stage_1_5_cross_circuit_report.md')}")
    print(f"  {os.path.join(DATA_DIR, 'stage_2_visualizations')}")
    if args.steer_quick or args.steer_full:
        print()
        print("Steering validation outputs:")
        print(f"  {os.path.join(DATA_DIR, 'stage_3_steering', 'steering_validation_report.md')}")
        print(f"  {os.path.join(DATA_DIR, 'stage_3_steering', 'steering_analysis.json')}")
    if args.advanced:
        print()
        print("Advanced analysis outputs are in each circuit's sub-directories (steps 7-10).")


if __name__ == '__main__':
    main()
