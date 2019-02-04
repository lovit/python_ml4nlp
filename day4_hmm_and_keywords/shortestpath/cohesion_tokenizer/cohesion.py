class CohesionScore:
    def __init__(self, max_l_length=10):
        self.max_l_length = max_l_length
        self._scores = {}
        self._subword_count = {}

    def __getitem__(self, args):
        if isinstance(args, str):
            word = args
            default = 0
        else:
            word, default = args
        return self.get_cohesion(word, default)

    def train(self, sentences, min_count=10):
        self._subword_count = {}
        for num_sent, sent in enumerate(sentences):
            if num_sent % 5000 == 0:
                print('\rcounting subwords from %d sents... ' % num_sent, end='', flush=True)
            for token in sent.split():
                for e in range(1, min(self.max_l_length, len(token)) + 1):
                    subword = token[:e]
                    self._subword_count[subword] = self._subword_count.get(subword,0) + 1

        words = {word for word, count in self._subword_count.items()
                 if (count >= min_count) and (len(word) > 1)}

        self._scores = {word:self.compute_cohesion(word) for word in words}
        print('\rtraining was done with %d sents' % (num_sent + 1))

    def get_cohesion(self, word, default=0):
        return self._scores.get(word, default)

    def compute_cohesion(self, word):

        # 글자가 아니거나 공백, 혹은 희귀한 단어인 경우
        if (not word) or ((word in self._subword_count) == False):
            return 0.0

        if len(word) == 1:
            return 0

        word_freq = self._subword_count.get(word, 0)
        base_freq = self._subword_count.get(word[:1], 0)

        if base_freq == 0:
            return 0.0
        else:
            return (word_freq / base_freq) ** (1 / (len(word) - 1))