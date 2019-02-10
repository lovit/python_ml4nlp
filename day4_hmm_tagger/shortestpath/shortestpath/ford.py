from .utils import _set_nodes_dict
from .utils import _initialize_dict
from .utils import _print_changing
from .utils import _find_shortest_path_dict

def ford(g, start, end, debug=False):
    if isinstance(g, dict):
        return _ford_dict(g, start, end, debug)
    if isinstance(g, tuple) and (isinstance(g[0], list) and isinstance(g[1], set)):
        return _ford_list(*g, start, end)
    raise NotImplemented

def _ford_dict(g, start, end, debug=False):
    from_nodes, to_nodes = _set_nodes_dict(g)
    if not ((start in from_nodes) and (end in to_nodes)):
        raise ValueError('There is no path {} - ... - {}'.format(start, end))

    n_nodes, n_edges, cost = _initialize_dict(g, start)

    for _iter in range(n_nodes * n_edges):
        cost, changed = _update_ford_dict(g, cost, debug)
        if not changed:
            break
    paths = _find_shortest_path_dict(g, start, end, cost, n_nodes)
    return {'paths': paths, 'cost': cost[end]}

def _update_ford_dict(g, cost, debug=False):
    changed = False
    for from_, to_weight in g.items():
        for to_, weight in to_weight.items():
            if cost[to_] > cost[from_] + weight:
                before = cost[to_]
                after = cost[from_] + weight
                cost[to_] = after
                changed = True
                if debug:
                    _print_changing(from_, to_, before, after)
    return cost, changed

def _ford_list(E, V, S, T):

    ## Initialize ##
    # (max weight + 1) * num of nodes
    inf = (max((weight for from_, to_, weight in E)) + 1) * len(V)

    # distance
    d = {node:0 if node == S else inf for node in V}
    # previous node
    prev = {node:None for node in V}

    ## Iteration ##
    # preventing infinite loop
    for _ in range(len(V)):
        # for early stop
        changed = False
        for u, v, Wuv in E:
            d_new = d[u] + Wuv
            if d_new < d[v]:
                d[v] = d_new
                prev[v] = u
                changed = True
        if not changed:
            break

    # Checking negative cycle loop
    for u, v, Wuv in E:
        if d[u] + Wuv < d[v]:
            raise ValueError('Negative cycle exists')

    # Finding path
    prev_ = prev[T]
    if prev_ == S:
        return {'paths':[[prev_, S][::-1]], 'cost':d[T]}

    path = []
    while prev_ != S:
        path.append(prev_)
        prev_ = prev[prev_]
    path.append(S)

    return {'paths':[path[::-1]], 'cost':d[T]}