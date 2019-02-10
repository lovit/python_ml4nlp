class Dictionary:
    """Usage
    ---------
    pos2words = {
        'Noun': set('아이 아이오 아이오아이 청하 출신'.split()),
        'Josa': set('은 는 이 가 의 를 을'.split()),
        'Verb': set('청하 이 있 하 했 입'.split()),
        'Eomi': set('다 었다 는 니다'.split())
    }

    dictionary = Dictionary(pos2words)
    dictionary.get_pos('아이오아이')
    # ['Noun']
    """

    def __init__(self, pos2words=None):
        self.pos2words = pos2words if pos2words else {}
        self.max_len = self._set_max_len()

    def _set_max_len(self):
        if not self.pos2words: return 0
        return max((len(word) for words in self.pos2words.values() for word in words))

    def get_pos(self, word):
        return [pos for pos, words in self.pos2words.items() if word in words]