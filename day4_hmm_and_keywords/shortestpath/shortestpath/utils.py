from scipy.io import mmread
from collections import defaultdict

def load_sparse_graph_and_index(header):
    x = mmread(header+'.mtx').tocsr()
    with open('{}.vocab'.format(header), encoding='utf-8') as f:
        idx2vocab = [vocab.strip() for vocab in f]
    return x, idx2vocab

def sparse_to_dict_graph(x, idx2vocab):
    def to_csr(i):
        return idx2vocab[i]

    rows, cols = x.nonzero()
    data = x.data
    d = defaultdict(lambda: {})
    for i, j, w in zip(rows, cols, data):
        d[to_csr(i)][to_csr(j)] = w
    return dict(d)

def list_to_dict_graph(list_form):
    g = defaultdict(lambda: {})
    for from_, to_, weight in list_form:
        g[from_][to_] = weight
    return dict(g)

def dict_to_list_graph(g):
    edges = []
    nodes = set()
    for from_node, to_node_weight in g.items():
        nodes.add(from_node)
        for to_node, weight in to_node_weight.items():
            nodes.add(to_node)
            edges.append((from_node, to_node, weight))
    return edges, nodes

def _set_nodes_dict(g):
    from_nodes = set(g)
    to_nodes = {node for nw in g.values() for node in nw.keys()}
    return from_nodes, to_nodes

def _initialize_dict(g, start):
    nodes = set(g.keys())
    nodes.update(set({n for nw in g.values() for n in nw.keys()}))
    n_nodes = len(nodes)
    n_edges = sum((len(nw) for nw in g.values()))
    max_weight = max(w for nw in g.values() for w in nw.values())

    init_cost = n_nodes * (max_weight + 1)
    cost = {node:(0 if node == start else init_cost) for node in nodes}
    return n_nodes, n_edges, cost

def _print_changing(from_, to_, before, after):
    print('cost[{} -> {}] = {} -> {}'.format(
        from_, to_, before, after))

def _find_shortest_path_dict(g, start, end, cost, n_nodes):
    immatures = [[start]]
    mature = []
    n_iter = 0
    for _ in range(n_nodes):
        immatures_ = []
        for path in immatures:
            last = path[-1]
            for adjacent, c in g[last].items():
                if cost[adjacent] == cost[last] + c:
                    if adjacent == end:
                        mature.append([p for p in path] + [adjacent])
                    else:
                        immatures_.append([p for p in path] + [adjacent])
        immatures = immatures_
    return mature