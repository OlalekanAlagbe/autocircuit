"""
Generate Attribution Graph from Neuronpedia Circuit Tracer API
Supports interactive prompt input or command-line argument
"""

import requests
import yaml
import json
import argparse
import re
from pathlib import Path
import sys

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from path_manager import PathManager

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config" / "neuronpedia_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

API_KEY = config['api']['api_key']
BASE_URL = config['api']['base_url']

def test_authentication():
    """Test if API key is valid by listing user graphs"""
    print("Testing API authentication...")

    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.get(
        f"{BASE_URL}/graph/list",
        headers=headers
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response text: {response.text[:200]}")

    if response.status_code == 200:
        print("[OK] Authentication successful!")
        try:
            data = response.json() if response.text else {}
            print(f"Found {len(data.get('graphs', []))} existing graphs")
            return True, data
        except Exception as e:
            print(f"[WARNING] Could not parse JSON: {e}")
            print("Response might be empty or in different format")
            return True, {}
    else:
        print(f"[ERROR] Authentication failed: {response.text}")
        return False, None

def generate_attribution_graph(prompt_text, model_id="gemma-2-2b"):
    """Generate a new attribution graph for a prompt"""
    print(f"\nGenerating attribution graph for: '{prompt_text}'")

    # POST endpoint doesn't require authentication according to docs
    payload = {
        "prompt": prompt_text,
        "modelId": model_id
    }

    response = requests.post(
        f"{BASE_URL}/graph/generate",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response text preview: {response.text[:500]}")

    if response.status_code == 200:
        try:
            data = response.json()
            print("[OK] Graph generated successfully!")
            print(f"Graph URL: {data.get('url', 'N/A')}")
            print(f"Nodes: {data.get('numNodes', 'N/A')}")
            print(f"Links: {data.get('numLinks', 'N/A')}")
            # Extract slug from URL
            url = data.get('url', '')
            slug = url.split('slug=')[-1] if 'slug=' in url else 'N/A'
            print(f"Slug: {slug}")
            print(f"S3 Location: {data.get('s3url', 'N/A')}")
            data['slug'] = slug  # Add slug to data
            data['s3Location'] = data.get('s3url')  # Normalize key name
            return True, data
        except Exception as e:
            print(f"[ERROR] Could not parse response: {e}")
            return False, None
    else:
        print(f"[ERROR] Graph generation failed: {response.text[:500]}")
        return False, None

def get_graph_data(model_id, slug):
    """Fetch graph data by model ID and slug"""
    print(f"\nFetching graph data for {model_id}/{slug}...")

    response = requests.get(
        f"{BASE_URL}/graph/{model_id}/{slug}",
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response preview: {response.text[:300]}")

    if response.status_code == 200:
        try:
            data = response.json()
            print("[OK] Graph metadata retrieved!")
            return True, data
        except Exception as e:
            print(f"[ERROR] Could not parse JSON: {e}")
            return False, None
    else:
        print(f"[ERROR] Failed to fetch graph: {response.text[:300]}")
        return False, None

def fetch_s3_graph_json(s3_url):
    """Fetch the actual graph JSON from S3"""
    print(f"\nFetching graph JSON from S3: {s3_url[:100]}...")

    response = requests.get(s3_url)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            print("[OK] Graph JSON downloaded!")
            print(f"Nodes: {len(data.get('nodes', []))}")
            print(f"Links: {len(data.get('links', []))}")
            return True, data
        except Exception as e:
            print(f"[ERROR] Could not parse S3 JSON: {e}")
            return False, None
    else:
        print(f"[ERROR] S3 fetch failed: {response.status_code}")
        return False, None

def get_prompt_from_user():
    """Get prompt from user interactively"""
    print("\n" + "=" * 60)
    print("Enter the prompt you want to analyze:")
    print("=" * 60)
    prompt = input("> ").strip()

    if not prompt:
        print("[ERROR] Prompt cannot be empty!")
        return None

    return prompt

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate Circuit Tracer attribution graph for a prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python 1_generate_graph.py --prompt "The currency in Japan is"
  python 1_generate_graph.py  (interactive mode)
        '''
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='The prompt to analyze (if not provided, will prompt interactively)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemma-2-2b',
        help='Model ID (default: gemma-2-2b)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CIRCUIT TRACER - GRAPH GENERATION")
    print("=" * 60)

    # Get prompt from command-line or interactively
    if args.prompt:
        test_prompt = args.prompt
        print(f"\nUsing prompt from command-line: '{test_prompt}'")
    else:
        test_prompt = get_prompt_from_user()
        if not test_prompt:
            print("\n[ERROR] No prompt provided. Exiting.")
            exit(1)

    # Test 1: Authentication
    auth_success, user_graphs = test_authentication()

    if auth_success and user_graphs and user_graphs.get('graphs'):
        print(f"\nYou have {len(user_graphs.get('graphs', []))} existing graph(s)")
        print("Recent graphs:")
        for graph in user_graphs.get('graphs', [])[:3]:  # Show first 3
            print(f"  - {graph.get('prompt', 'N/A')[:60]}...")

    # Test 2: Generate new graph
    gen_success, graph_data = generate_attribution_graph(test_prompt, model_id=args.model)

    if gen_success and graph_data:
        # Test 3: Fetch the generated graph metadata
        slug = graph_data.get('slug')
        model_id = graph_data.get('modelId', args.model)
        s3_location = graph_data.get('s3Location')

        if slug:
            fetch_success, metadata = get_graph_data(model_id, slug)

            if fetch_success and metadata:
                print(f"\nMetadata retrieved: {len(metadata.keys())} keys")

        # Test 4: Fetch actual graph JSON from S3
        if s3_location:
            s3_success, graph_json = fetch_s3_graph_json(s3_location)

            if s3_success and graph_json:
                # Use PathManager for consistent file organization
                pm = PathManager()

                # Save raw graph JSON (pass model_id for proper directory structure)
                output_path = pm.raw_graph_path(test_prompt, model_id=args.model)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_json, f, indent=2)

                # Save metadata (pass model_id for proper directory structure)
                metadata = {
                    'prompt': test_prompt,
                    'model': args.model,
                    'slug': slug,
                    'num_nodes': len(graph_json.get('nodes', [])),
                    'num_links': len(graph_json.get('links', [])),
                    's3_url': s3_location,
                    'neuronpedia_url': graph_data.get('url', '')
                }
                pm.save_metadata(test_prompt, metadata, model_id=args.model)

                print(f"\n{'=' * 60}")
                print("[SUCCESS] Graph saved successfully!")
                print(f"{'=' * 60}")
                print(f"Model: {args.model}")
                print(f"File: {output_path.name}")
                print(f"Location: {output_path}")
                print(f"Prompt directory: {pm.get_prompt_dir(test_prompt, model_id=args.model)}")
                print(f"Nodes: {len(graph_json.get('nodes', []))}")
                print(f"Links: {len(graph_json.get('links', []))} ")
                print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
                print(f"\nNext step: Run script 2 to convert the graph")
        else:
            print("\n[ERROR] No S3 location returned. Graph may not have been generated.")
    else:
        print("\n[ERROR] Graph generation failed. Check your API key and internet connection.")

    print("\n" + "=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)
