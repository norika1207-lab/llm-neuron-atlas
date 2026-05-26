#!/usr/bin/env python3
"""Extract neuron-as-city / weight-as-road graph from Qwen 2.5 3B.

Output:
  graph/nodes.json    -- 73,728 neuron-cities (36 layers x 2048 dims)
  graph/edges.json    -- top-K strongest connections per neuron
  graph/highways.json -- residual stream + 11 Mercury anchor pathways
  graph/meta.json     -- model meta + Mercury named neurons

Connections semantics:
  For each layer L, dim D:
    outgoing edges = top-K columns in weight matrices going from dim D in L
                     to layer L's q_proj / k_proj / v_proj / gate / up rows
    incoming edges = top-K rows in o_proj / down going INTO dim D of next residual

  Simplification for Phase 1:
    edge = (src_layer, src_dim) -> (dst_layer, dst_dim) with weight magnitude
    We extract from gate_proj (most informative MLP gate) and q_proj (attention routing)
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True, help='dir with safetensors shards')
ap.add_argument('--out', required=True, help='output graph dir')
ap.add_argument('--top-k', type=int, default=10, help='top-K edges per neuron')
args = ap.parse_args()

WEIGHTS = Path(args.weights)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

N_LAYERS = 36
HIDDEN = 2048
INTERMEDIATE = 11008
TOP_K = args.top_k  # top-K strongest outgoing edges per neuron

# Mercury named neurons (from paper-B-functional-control)
MERCURY_NAMES = {
    11:  {'name': 'anchor-11',  'role': 'cross-Qwen-family conserved'},
    25:  {'name': 'anchor-25',  'role': 'cross-Qwen-family conserved'},
    279: {'name': 'pathway-A',  'role': 'Pathway A (style transfer)'},
    382: {'name': 'pathway-B',  'role': 'Pathway B (functional control)'},
    476: {'name': 'pathway-A2', 'role': 'Pathway A secondary'},
    481: {'name': 'style-destab', 'role': 'destabilization marker'},
    715: {'name': 'INHIBITOR', 'role': 'selective suppression (Mercury Paper B)'},
    758: {'name': 'CONTROLLER', 'role': 'single-dim style controller'},
}

def load_layer_weights(handles, layer):
    """Return dict of {short_name: torch.Tensor float32 numpy} for layer."""
    tensors = {}
    targets = {
        'q':    f'model.layers.{layer}.self_attn.q_proj.weight',
        'k':    f'model.layers.{layer}.self_attn.k_proj.weight',
        'v':    f'model.layers.{layer}.self_attn.v_proj.weight',
        'o':    f'model.layers.{layer}.self_attn.o_proj.weight',
        'gate': f'model.layers.{layer}.mlp.gate_proj.weight',
        'up':   f'model.layers.{layer}.mlp.up_proj.weight',
        'down': f'model.layers.{layer}.mlp.down_proj.weight',
    }
    for short, key in targets.items():
        for f in handles:
            if key in f.keys():
                t = f.get_tensor(key).to(dtype=torch.float32).cpu().numpy()
                tensors[short] = t
                break
    return tensors

def top_k_edges_from_layer(layer, tensors, k=TOP_K):
    """For each dim D in layer L, find top-K strongest outgoing connections.

    The flow: residual at layer L (HIDDEN dims) -> q/k/v/gate/up projections
    of layer L (which then feed into layer L+1's residual via o_proj/down).

    We approximate 'where dim D's influence goes' as:
      effective_weight[D -> D'] = max over {q, gate} of
        sum(input_proj[:, D] * output_proj[D', :])

    Simpler & faster Phase 1: use gate_proj * down_proj composition.
      gate is (INTERMEDIATE, HIDDEN), down is (HIDDEN, INTERMEDIATE)
      effective_HIDDEN_to_HIDDEN[D, D'] = down[D', :] @ gate[:, D]
      = sum over intermediate i of down[D', i] * gate[i, D]

    This gives the linear contribution of L's dim D to L+1's dim D' via MLP.
    """
    edges = []  # list of (src_dim, dst_dim, weight)
    gate = tensors['gate']        # (11008, 2048)
    down = tensors['down']        # (2048, 11008)

    # Compute D x D effective transfer matrix via MLP
    # M[D', D] = down[D', :] @ gate[:, D]  -- (2048, 2048)
    # Too big to store as 2048x2048 dense (16MB), but trivially compute per column

    M = down @ gate  # (2048, 11008) @ (11008, 2048) = (2048, 2048)
    # M[dst, src] = signed effective contribution of src dim -> dst dim via MLP
    M_abs = np.abs(M)

    for src in range(HIDDEN):
        # top-K destinations from src
        col = M_abs[:, src]
        top_idx = np.argpartition(col, -k)[-k:]
        # sort by magnitude desc
        top_idx = top_idx[np.argsort(-col[top_idx])]
        for dst in top_idx:
            edges.append({
                'src_layer': layer,
                'src_dim': int(src),
                'dst_layer': layer + 1,
                'dst_dim': int(dst),
                'weight': float(M[dst, src]),  # signed
                'abs': float(col[dst]),
                'via': 'mlp',
            })
    return edges

def main():
    print('=== Phase 1: extracting neuron graph from Qwen 2.5 3B ===')
    shards = sorted(WEIGHTS.glob('*.safetensors'))
    handles = [safe_open(s, framework='pt') for s in shards]

    nodes = []
    edges = []
    highways = []
    t0 = time.time()

    # Pre-build node list (73,728 neurons)
    for L in range(N_LAYERS):
        for D in range(HIDDEN):
            node = {
                'id': f'L{L}D{D}',
                'layer': L,
                'dim': D,
            }
            if D in MERCURY_NAMES:
                node.update(MERCURY_NAMES[D])
                node['named'] = True
            else:
                node['named'] = False
            nodes.append(node)
    print(f'  nodes: {len(nodes)}')

    # Extract edges per layer (skip last layer, no outgoing to L+1)
    for L in range(N_LAYERS - 1):
        t_layer = time.time()
        tensors = load_layer_weights(handles, L)
        layer_edges = top_k_edges_from_layer(L, tensors)
        edges.extend(layer_edges)
        print(f'  L{L}: {len(layer_edges)} edges ({time.time()-t_layer:.1f}s)')

    # Residual highways: each dim D as a vertical line through all 36 layers
    for D in range(HIDDEN):
        is_anchor = D in MERCURY_NAMES
        highways.append({
            'kind': 'residual',
            'dim': D,
            'layers': list(range(N_LAYERS)),
            'named': is_anchor,
            'name': MERCURY_NAMES.get(D, {}).get('name'),
        })

    # Mercury anchor pathways (subset of residuals, but flagged as fragile-core)
    for D, info in MERCURY_NAMES.items():
        highways.append({
            'kind': 'mercury_anchor',
            'dim': D,
            'name': info['name'],
            'role': info['role'],
            'layers': list(range(N_LAYERS)),
        })

    meta = {
        'model': 'Qwen/Qwen2.5-3B',
        'n_layers': N_LAYERS,
        'hidden': HIDDEN,
        'intermediate': INTERMEDIATE,
        'top_k_per_neuron': TOP_K,
        'mercury_named': MERCURY_NAMES,
        'extract_time_sec': round(time.time() - t0, 1),
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'total_highways': len(highways),
    }

    # Write outputs
    (OUT / 'nodes.json').write_text(json.dumps(nodes))
    (OUT / 'edges.json').write_text(json.dumps(edges))
    (OUT / 'highways.json').write_text(json.dumps(highways))
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

    # Sizes
    for f in ['nodes.json', 'edges.json', 'highways.json', 'meta.json']:
        size = (OUT / f).stat().st_size
        print(f'  {f}: {size/1024:.1f} KB')

    print(f'=== done in {meta["extract_time_sec"]}s ===')

if __name__ == '__main__':
    main()
