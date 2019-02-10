from .utils import _set_nodes_dict
from .utils import _initialize_dict
from .utils import _print_changing
from .utils import _find_shortest_path_dict

def dijkstra(g, start, end, debug=False):
    if isinstance(g, dict):
        return _dijkstra_dict(g, start, end, debug)
    raise NotImplemented

def _dijkstra_dict(g, start, end, debug=False):
    from_nodes, to_nodes = _set_nodes_dict(g)
    if not ((start in from_nodes) and (end in to_nodes)):
        raise ValueError('There is no path {} - ... - {}'.format(start, end))

    n_nodes, n_edges, cost = _initialize_dict(g, start)
    unvisited = {node:None for node in from_nodes}
    unvisited.update({node:None for node in to_nodes})

    while unvisited:
        current = sorted(unvisited, key=lambda x:cost[x])[0]
        unvisited.pop(current)
        cost, changed = _update_dijkstra_dict(g, cost, current, debug)
        if debug:
            print()
        if (current == end) or (not changed):
            break

    paths = _find_shortest_path_dict(g, start, end, cost, n_nodes)
    return {'cost': cost[end], 'paths': paths}

def _update_dijkstra_dict(g, cost, from_, debug=False):
    changed = False
    for to_, weight in g.get(from_, {}).items():
        if cost[to_] > cost[from_] + weight:
            before = cost[to_]
            after = cost[from_] + weight
            cost[to_] = after
            changed = True
            if debug:
                _print_changing(from_, to_, before, after)
    return cost, changed