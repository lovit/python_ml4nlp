import sys
sys.path.insert(0, '../')
from shortestpath import ford
from shortestpath import list_to_dict_graph

from .cohesion import CohesionScore
from .graph import WordSequenceGraph

class CohesionGraphWordSegmentor:
    def __init__(self, cohesion=None, max_l_length=10, edge_weight_func=None):
        self.cohesion = None
        if isinstance(cohesion, CohesionScore):
            self.cohesion = cohesion
        self.max_l_length = max_l_length
        self.edge_weight_func = edge_weight_func

    @property
    def is_trained(self):
        return hasattr(self, 'graph_generator')

    def train(self, sentences, min_count=10):
        self.cohesion = CohesionScore(self.max_l_length)
        self.cohesion.train(sentences, min_count)
        if self.edge_weight_func is None:
            self.graph_generator = WordSequenceGraph(
                self.cohesion
            )
        else:
            self.graph_generator = WordSequenceGraph(
                self.cohesion,
                self.edge_weight_func
            )

    def tokenize(self, sentence):
        edges = self.graph_generator.as_graph(sentence)
        graph = list_to_dict_graph(edges)
        return ford(graph, ('BOS',), ('EOS',))
        