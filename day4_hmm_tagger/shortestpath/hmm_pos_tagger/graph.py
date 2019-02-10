from collections import defaultdict

class WordSequenceGraph:

    def __init__(self, dictionary, hmm_model):
        self.dictionary = dictionary
        self.hmm_model = hmm_model

    def as_graph(self, sentence):
        edges, sent = self._as_edges(sentence)
        graph = self._add_weight(edges)
        idx2node = defaultdict(lambda: len(idx2node))
        encoded_graph = [(idx2node[n0], idx2node[n1], w) for n0, n1, w in graph]
        idx2node = dict(idx2node)
        return encoded_graph, idx2node, graph, sent

    def _lookup(self, sentence):

        def word_lookup(eojeol, offset):
            n = len(eojeol)
            words = [[] for _ in range(n)]
            for b in range(n):
                for r in range(1, self.dictionary.max_len+1):
                    e = b+r
                    if e > n:
                        continue
                    sub = eojeol[b:e]
                    for pos in self.dictionary.get_pos(sub):
                        words[b].append((sub, pos, b+offset, e+offset))
            return words

        sent = []
        for eojeol in sentence.split():
            sent += word_lookup(eojeol, offset=len(sent))
        return sent

    def _as_edges(self, sentence):

        def get_nonempty_first(sent, end, offset=0):
            for i in range(offset, end):
                if sent[i]:
                    return i
            return offset

        chars = sentence.replace(' ', '')
        sent = self._lookup(sentence)
        n_char = len(sent) + 1
        sent.append([('EOS', 'EOS', n_char, n_char)])

        nonempty_first = get_nonempty_first(sent, n_char)
        if nonempty_first > 0:
            sent[0].append((chars[:nonempty_first], 'Unk', 0, nonempty_first))

        graph = []
        for words in sent[:-1]:
            for word in words:
                begin = word[2]
                end = word[3]
                if not sent[end]:
                    b = get_nonempty_first(sent, n_char, end)
                    unk = (chars[end:b], 'Unk', end, b)
                    graph.append((word, unk))
                for adjacent in sent[end]:
                    graph.append((word, adjacent))

        unks = {node for _, node in graph if node[1] == 'Unk'}
        for unk in unks:
            for adjacent in sent[unk[3]]:
                graph.append((unk, adjacent))
        bos = ('BOS', 'BOS', 0, 0)
        for word in sent[0]:
            graph.append((bos, word))
        graph = sorted(graph, key=lambda x:(x[0][2], x[1][3]))

        return graph, sent

    def _add_weight(self, edges):
        graph = [(edge[0], edge[1], self.hmm_model(edge)) for edge in edges]
        return graph