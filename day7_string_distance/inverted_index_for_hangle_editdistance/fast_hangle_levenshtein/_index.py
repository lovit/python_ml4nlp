from collections import defaultdict
from time import time
from ._string_distance import jamo_levenshtein
from ._string_distance import levenshtein
from ._hangle import decompose
from ._hangle import character_is_korean

class LevenshteinIndex:
    def __init__(self, words=None, levenshtein_distance=levenshtein, verbose=False):
        self._words = {}
        self._index = {} # character to words
        self._cho_index, self._jung_index, self._jong_index = {}, {}, {}
        if words:
            self.indexing(words)
        self.verbose = verbose
        self.jamo_base_distance = 1/3

    def indexing(self, words):
        self._words = words if type(words) == dict else {w:0 for w in words if w}
        self._index = defaultdict(lambda: set())
        self._cho_index = defaultdict(lambda: set())
        self._jung_index = defaultdict(lambda: set())
        self._jong_index = defaultdict(lambda: set())

        for word in words:
            # Indexing for levenshtein
            for c in word:
                self._index[c].add(word)
            # Indexing for jamo_levenshtein
            for c in word:
                if not character_is_korean(c):
                    continue
                cho, jung, jong = decompose(c)
                self._cho_index[cho].add(word)
                self._jung_index[jung].add(word)
                self._jong_index[jong].add(word)

        self._index = dict(self._index)
        self._cho_index = dict(self._cho_index)
        self._jung_index = dict(self._jung_index)
        self._jong_index = dict(self._jong_index)

    def levenshtein_search(self, query, max_distance=1):
        search_time = time()
        similars = defaultdict(int)
        (n, nc) = (len(query), len(set(query)))
        for c in set(query):
            for item in self._index.get(c, {}):
                similars[item] += 1

        if self.verbose:
            print('query={}, candidates={} '.format(query, len(similars)), end='')

        similars = {word for word,f in similars.items()
                    if (abs(n-len(word)) <= max_distance) and (abs(nc - f) <= max_distance)}
        if self.verbose:
            print('-> {}'.format(len(similars)), end='')

        dist = {}
        for word in similars:
            dist[word] = levenshtein(word, query)

        if self.verbose:
            search_time = time() - search_time
            print(', time={:.3} sec.'.format(search_time))

        return sorted(filter(lambda x:x[1] <= max_distance, dist.items()), key=lambda x:x[1])

    def jamo_levenshtein_search(self, query, max_distance=1):
        search_time = time()
        similars = defaultdict(lambda: 0)
        (n, nc) = (len(query), len(set(query)))
        for c in set(query):
            if not character_is_korean(c):
                for item in self._index.get(c, {}):
                    similars[item] += 1
                continue
            cho, jung, jong = decompose(c)
            for item in self._cho_index.get(cho, {}):
                similars[item] += self.jamo_base_distance
            for item in self._jung_index.get(jung, {}):
                similars[item] += self.jamo_base_distance
            for item in self._jong_index.get(jong, {}):
                similars[item] += self.jamo_base_distance

        if self.verbose:
            print('query={}, candidates={} '.format(query, len(similars)), end='')

        similars = {word for word,f in similars.items()
                    if (abs(n-len(word)) <= max_distance) and (abs(nc - f) <= max_distance)}
        if self.verbose:
            print('-> {}'.format(len(similars)), end='')

        dist = {}
        for word in similars:
            dist[word] = jamo_levenshtein(word, query)

        if self.verbose:
            search_time = time() - search_time
            print(', time={:.3} sec.'.format(search_time))

        return sorted(filter(lambda x:x[1] <= max_distance, dist.items()), key=lambda x:x[1])