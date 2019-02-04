class WordSequenceGraph:

    def __init__(self, cohesion, cost=lambda word, word_score: len(word) * (1 - word_score) ):
        self.cohesion = cohesion
        self.cost = cost

    def as_graph(self, sentence):
        sent = self._sentence_lookup(sentence)
        edges = self._link_adjacent_nodes(sent)
        return edges

    def _sentence_lookup(self, sentence):
        offset = 0
        sent = []
        for eojeol in sentence.split():
            sent += self._word_lookup(eojeol, offset)
            offset += len(eojeol)
        return sent

    def _word_lookup(self, eojeol, offset=0):
        n = len(eojeol)
        # (word, score, begin, end)
        words = [[(eojeol[i], offset + i, offset + i + 1)] for i in range(n)]
        for b in range(n):
            for r in range(2, self.cohesion.max_l_length + 1):
                e = b + r
                if e > n:
                    continue
                sub = eojeol[b:e]
                score = self.cohesion[sub]
                if score > 0:
                    words[b].append((sub, offset + b, offset + e))
        return words

    def _link_adjacent_nodes(self, sent):

        bos = ('BOS',)
        eos = ('EOS',)

        sent += [[eos]]

        edges = [
            (bos, first, self.cost(first[0], self.cohesion[first[0], 0]))
             for first in sent[0]
        ]
        for words in sent[:-1]:
            for word in words:
                end = word[2]
                for adjacent in sent[end]:
                    edges.append(
                        (word,
                         adjacent,
                         self.cost(adjacent[0], self.cohesion[adjacent[0], 0])
                        )
                    )
        return edges