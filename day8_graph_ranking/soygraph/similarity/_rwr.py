import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from soygraph import is_numeric_dict_dict
from soygraph import dict_to_matrix
from soygraph.ranking._rank import _update_pagerank


class RandomWalkWithRestart:
    def __init__(self, outbound_matrix, bias=None,
        max_iter=10, decaying_factor=0.85, verbose=False):

        self.outbound_matrix, self.bias, self.n_nodes = self._initialize(
            outbound_matrix, bias)
        self.max_iter = max_iter
        self.df = decaying_factor
        self.verbose = verbose

    def _initialize(self, outbound_matrix, bias):
        # check outbound matrix type
        if is_numeric_dict_dict(outbound_matrix):
            x = dict_to_matrix(outbound_matrix)
        elif not sp.issparse(outbound_matrix):
            raise ValueError('inboud_matrix type should be sparse matrix')
        else:
            x = outbound_matrix

        # check bias type and shape
        n_nodes = max(x.shape)
        initial_weight = 1 / n_nodes
        if not bias:
            bias = np.asarray([initial_weight] * n_nodes)
        else:
            if not isinstance(bias, np.ndarray):
                raise ValueError('user specific bias type should be numpy.ndarray')
            if bias.shape[0] != n_nodes:
                raise ValueError('user specific bias should have same length of outbound matrix')

        # normalize. outbound matrix should be transaction matrix
        x = normalize(x, norm='l1')
        return x, bias, n_nodes

    def most_similar(self, query, topk=10, max_iter=-1, df=-1):
        if (type(query) == int):
            if not (0 <= query < self.n_nodes):
                raise ValueError(
                    'query index is out of bound. It should be int in [0, {})'.format(
                        self.n_nodes))
            query_vec = np.zeros(self.n_nodes)
            query_vec[query] = 1
        elif not isinstance(query, np.ndarray):
            raise ValueError('query type should be int or numpy.ndarray')
        else:
            if query.shape[0] != self.n_nodes:
                raise ValueError('query length should be {}'.format(self.n_nodes))
            query_vec = query

        idxs, prob = random_walk_with_restart(
            self.outbound_matrix,
            query_vec,
            self.bias,
            max_iter if max_iter > 0 else self.max_iter,
            df if 0 <= df < 1 else self.df,
            self.verbose,
            0
        )

        if topk > 0:
            idxs = idxs[:topk]
            prob = prob[:topk]
        most_similars = [(idx, p) for idx, p in zip(idxs, prob)]

        return most_similars

def random_walk_with_restart(outbound_matrix, query_vec, bias,
    max_iter=10, df=0.85, verbose=False, converge_threshold=0.0001):

    for n_iter in range(1, max_iter + 1):
        query_vec_new = _update_pagerank(outbound_matrix, query_vec, bias, df)
        diff = np.sqrt(((query_vec - query_vec_new) **2).sum())
        query_vec = query_vec_new

        if diff <= converge_threshold:
            if verbose:
                print('Early stop. because it already converged.')
            break
        if verbose:
            print('iter {} : diff = {}'.format(n_iter, diff))

    idxs = query_vec.argsort()[::-1]
    prob = query_vec[idxs]

    return idxs, prob