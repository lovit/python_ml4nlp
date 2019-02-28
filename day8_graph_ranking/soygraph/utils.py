import os
import psutil
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix

def get_available_memory():
    """It returns remained memory as percentage"""

    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def bow_to_graph(x):
    """It transform doc-term sparse matrix to graph.
    Vertex = [doc_0, doc_1, ..., doc_{n-1}|term_0, term_1, ..., term_{m-1}]

    Arguments
    ---------
    x: scipy.sparse

    Returns
    -------
    g: scipy.sparse.csr_matrix
        V` = x.shape[0] + x.shape[1]
        its shape = (V`, V`)
    """
    x = x.tocsr()
    x_ = x.transpose().tocsr()
    data = np.concatenate((x.data, x_.data))
    indices = np.concatenate(
        (x.indices + x.shape[0] , x_.indices))
    indptr = np.concatenate(
        (x.indptr, x_.indptr[1:] + len(x.data)))
    return csr_matrix((data, indices, indptr))

def matrix_to_dict(m):
    """It transform sparse matrix (scipy.sparse.matrix) to dictdict"""
    d = defaultdict(lambda: {})
    for f, (idx_b, idx_e) in enumerate(zip(m.indptr, m.indptr[1:])):
        for idx in range(idx_b, idx_e):
            d[f][m.indices[idx]] = m.data[idx]
    return dict(d)

def dict_to_matrix(dd):
    rows = []
    cols = []
    data = []
    for d1, d2s in dd.items():
        for d2, w in d2s.items():
            rows.append(d1)
            cols.append(d2)
            data.append(w)
    n_nodes = max(max(rows), max(cols)) + 1
    x = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return x

def is_dict_dict(dd):
    if not type(dd) == dict:
        return False
    value_item = list(dd.values())[0]
    return type(value_item) == dict

def is_numeric_dict_dict(dd):
    if not is_dict_dict(dd):
        return False
    key0 = list(dd.keys())[0]
    key1, value1 = list(list(dd.values())[0].items())[0]
    if not type(key0) == int or not type(key1) == int:
        return False
    return type(value1) == int or type(value1) == float