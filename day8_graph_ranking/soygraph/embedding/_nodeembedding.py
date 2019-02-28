import numpy as np

class NodeEmbedding:
    
    def __init__(self, graph, embedding_dimension=50, learning_rate=0.01, max_iter=5):
        self._g = graph
        self._e = embedding_dimension
        self._lr = learning_rate
        self._max_iter = max_iter

    def train(self):
        n = len(graph.nodes())
        self.v = (np.random.random((n, self._e)) - 0.5)