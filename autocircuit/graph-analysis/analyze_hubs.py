import json
from collections import defaultdict

# Load graph data
with open('/tmp/neuronpedia_graph_data.json', 'r') as f:
    data = json.load(f)

# Calculate node degrees
in_degree = defaultdict(int)
out_degree = defaultdict(int)
weighted_in = defaultdict(float)
weighted_out = defaultdict(float)

for link in data['links']:
    source = link['source']
    target = link['target']
    weight = abs(link['weight'])
    
    out_degree[source] += 1
    in_degree[target] += 1
    weighted_out[source] += weight
    weighted_in[target] += weight

# Create node lookup
nodes_dict = {n['node_id']: n for n in data['nodes']}

# Find hub nodes (high degree)
all_nodes = set(list(in_degree.keys()) + list(out_degree.keys()))
node_degrees = []
for node_id in all_nodes:
    total_degree = in_degree[node_id] + out_degree[node_id]
    node_info = nodes_dict.get(node_id, {})
    node_degrees.append({
        'node_id': node_id,
        'in_degree': in_degree[node_id],
        'out_degree': out_degree[node_id],
        'total_degree': total_degree,
        'weighted_in': weighted_in[node_id],
        'weighted_out': weighted_out[node_id],
        'layer': node_info.get('layer', 'unknown'),
        'ctx_idx': node_info.get('ctx_idx', -1),
        'influence': node_info.get('influence')
    })

# Sort by total degree
node_degrees.sort(key=lambda x: x['total_degree'], reverse=True)

print("Top 20 Hub Nodes (by total degree):")
print("="*80)
for i, node in enumerate(node_degrees[:20], 1):
    print(f"{i}. {node['node_id']:<20} Layer:{node['layer']:<3} Ctx:{node['ctx_idx']} "
          f"In:{node['in_degree']:<4} Out:{node['out_degree']:<4} "
          f"Total:{node['total_degree']:<4} Influence:{node['influence']}")

print("\n\nTop 20 Nodes by Weighted In-degree:")
print("="*80)
node_degrees.sort(key=lambda x: x['weighted_in'], reverse=True)
for i, node in enumerate(node_degrees[:20], 1):
    print(f"{i}. {node['node_id']:<20} Layer:{node['layer']:<3} Ctx:{node['ctx_idx']} "
          f"WeightedIn:{node['weighted_in']:.2f} Influence:{node['influence']}")

print("\n\nTop 20 Nodes by Weighted Out-degree:")
print("="*80)
node_degrees.sort(key=lambda x: x['weighted_out'], reverse=True)
for i, node in enumerate(node_degrees[:20], 1):
    print(f"{i}. {node['node_id']:<20} Layer:{node['layer']:<3} Ctx:{node['ctx_idx']} "
          f"WeightedOut:{node['weighted_out']:.2f} Influence:{node['influence']}")
