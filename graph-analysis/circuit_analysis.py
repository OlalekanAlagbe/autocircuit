import json
from collections import defaultdict
import statistics

# Load graph data
with open('/tmp/neuronpedia_graph_data.json', 'r') as f:
    data = json.load(f)

print("=" * 100)
print("CIRCUIT ANALYSIS: DNA Stands for Deoxyribonucleic")
print("=" * 100)

# Group nodes by layer and context
layer_ctx_groups = defaultdict(list)
for node in data['nodes']:
    if node['feature_type'] not in ['embedding', 'logit']:
        key = (node['layer'], node['ctx_idx'])
        layer_ctx_groups[key].append(node)

# Calculate statistics for each group
print("\n1. GROUPING FEATURES INTO SUPERNODES")
print("=" * 100)
print("\nProposed Supernode Structure (Layer-Context Groups):\n")

supernode_summary = []
for (layer, ctx_idx), nodes in sorted(layer_ctx_groups.items()):
    influences = [n['influence'] for n in nodes if n.get('influence') is not None]
    activations = [n['activation'] for n in nodes if n.get('activation') is not None]
    
    if influences and activations:
        avg_influence = statistics.mean(influences)
        avg_activation = statistics.mean(activations)
        max_influence = max(influences)
        
        supernode_summary.append({
            'layer': layer,
            'ctx_idx': ctx_idx,
            'count': len(nodes),
            'avg_influence': avg_influence,
            'max_influence': max_influence,
            'avg_activation': avg_activation
        })

# Print supernodes by layer
current_layer = None
for sn in sorted(supernode_summary, key=lambda x: (int(x['layer']), x['ctx_idx'])):
    if sn['layer'] != current_layer:
        current_layer = sn['layer']
        print(f"\n--- Layer {current_layer} ---")
    
    print(f"  Supernode L{sn['layer']}_C{sn['ctx_idx']}: "
          f"{sn['count']:3d} features, "
          f"AvgInf={sn['avg_influence']:.3f}, "
          f"MaxInf={sn['max_influence']:.3f}")

# Analyze information flow by layer transitions
print("\n\n2. MAIN INFORMATION FLOWS (Input → Output)")
print("=" * 100)

# Track connections between layer groups
layer_transitions = defaultdict(lambda: {'count': 0, 'total_weight': 0, 'max_weight': 0})

for link in data['links']:
    source = link['source']
    target = link['target']
    weight = abs(link['weight'])
    
    # Get layer info
    if source.startswith('E_'):
        src_layer = 'E'
    else:
        src_layer = source.split('_')[0]
    
    if target == '27_6898_6':
        tgt_layer = '27'
    else:
        tgt_layer = target.split('_')[0]
    
    if src_layer != tgt_layer:
        key = (src_layer, tgt_layer)
        layer_transitions[key]['count'] += 1
        layer_transitions[key]['total_weight'] += weight
        layer_transitions[key]['max_weight'] = max(layer_transitions[key]['max_weight'], weight)

# Sort by total weight
sorted_transitions = sorted(layer_transitions.items(), 
                           key=lambda x: x[1]['total_weight'], 
                           reverse=True)[:30]

print("\nTop 30 Layer-to-Layer Information Flows (by total edge weight):\n")
for (src, tgt), stats in sorted_transitions:
    avg_weight = stats['total_weight'] / stats['count']
    print(f"  Layer {src:>2} → Layer {tgt:>2}: "
          f"{stats['count']:4d} edges, "
          f"TotalWeight={stats['total_weight']:8.1f}, "
          f"AvgWeight={avg_weight:6.2f}, "
          f"MaxWeight={stats['max_weight']:7.2f}")

# Analyze context position flow
print("\n\n3. CONTEXT POSITION FLOW (Token-level processing)")
print("=" * 100)

ctx_flow = defaultdict(lambda: {'count': 0, 'total_weight': 0})
nodes_dict = {n['node_id']: n for n in data['nodes']}

for link in data['links']:
    source_node = nodes_dict.get(link['source'])
    target_node = nodes_dict.get(link['target'])
    
    if source_node and target_node:
        src_ctx = source_node.get('ctx_idx', -1)
        tgt_ctx = target_node.get('ctx_idx', -1)
        if src_ctx != -1 and tgt_ctx != -1:
            key = (src_ctx, tgt_ctx)
            ctx_flow[key]['count'] += 1
            ctx_flow[key]['total_weight'] += abs(link['weight'])

print("\nToken-to-Token Flow (by edge count):\n")
for (src, tgt), stats in sorted(ctx_flow.items(), key=lambda x: x[1]['count'], reverse=True)[:20]:
    tokens = ["<bos>", "DNA", " stands", " for", " deoxy", "ri", "bonucleic"]
    src_token = tokens[src] if src < len(tokens) else f"ctx{src}"
    tgt_token = tokens[tgt] if tgt < len(tokens) else f"ctx{tgt}"
    print(f"  C{src} ({src_token:>12}) → C{tgt} ({tgt_token:>12}): "
          f"{stats['count']:5d} edges, TotalWeight={stats['total_weight']:8.1f}")

