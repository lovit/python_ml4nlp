from collections import Counter
from scipy.sparse import csr_matrix
from collections import defaultdict


def scan_vocabulary(sents, tokenizer, min_count):
    """
    :param sents: list of str
        Sentences
    :param tokenizer: callable
        tokenizer(str) = list of str
    :param min_count: int
        Minumum term frequency

    It returns
    ----------
    idx_to_vocab : list of str
        Vocabulary list
    vocab_to_idx : {str:int}
        Vocabulary to index
    idx_to_count : list of int
        Index to count

    Usage
    -----
        >>> tokenizer = lambda x:x.split()
        >>> idx_to_vocab, vocab_to_idx, idx_to_count = scan_vocabulary(
                sents, tokenizer, min_count)
    """

    vocab_counter = Counter(vocab for sent in sents for vocab in tokenizer(sent))
    vocab_counter = {vocab:count for vocab, count in vocab_counter.items()
                     if count >= min_count}
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(sorted(
        vocab_counter, key=lambda x:-vocab_counter[x]))}
    idx_to_vocab = [vocab for vocab in sorted(vocab_to_idx, key=lambda x:vocab_to_idx[x])]
    idx_to_count = [vocab_counter[vocab] for vocab in idx_to_vocab]
    return idx_to_vocab, vocab_to_idx, idx_to_count

def adjacent_cooccurrence_graph(sents, tokenizer, vocab_to_idx, verbose=False):
    """
    :param sents: list of str
        Sentences
    :param tokenizer: callable
        tokenizer(str) = list of str
    :param vocab_to_idx: {str:int}
        Vocabulary to index
    :param verbose: Boolean
        If True, verbose on

    It returns
    ----------
    C : {str:{str:int}}
        Adjacent co-occurrence matrix

    Usage
    -----
        >>> tokenizer = lambda x:x.split()
        >>> idx_to_vocab, vocab_to_idx, idx_to_count = scan_vocabulary(
                sents, tokenizer, min_count)
        >>> C = adjacent_cooccurrence_graph(sents, tokenizer, vocab_to_idx)
    """
    C = defaultdict(lambda: defaultdict(int))
    n_sents = len(sents)

    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 100 == 0:
            print('\rconstruct adjacent graph {} / {}'.format(i_sent, n_sents), end='')
        words = tokenizer(sent)
        n = len(words)
        if n < 2:
            continue

        for left, right in zip(words, words[1:]):
            if not (left in vocab_to_idx) or not (right in vocab_to_idx):
                continue
            C[left][right] += 1
            C[right][left] += 1

    if verbose:
        print('\rconstruct adjacent graph was done{}'.format(' '*20))
    return C

def c_to_x(C, vocab_to_idx, min_cooccurrence=1, verbose=False):
    """
    :param C: {str:{str:int}}
        Co-occurrence matrix
    :param vocab_to_idx: {str:int}
        Vocabulary to index
    :param min_cooccurrence: int
        Minumum co-occurrence frequency
    :param verbose: Boolean
        If True, verbose on

    It returns
    ----------
    X : scipy.sparse.csr_matrix
        sparse matrix format

    Usage
    -----
        >>> tokenizer = lambda x:x.split()
        >>> idx_to_vocab, vocab_to_idx, idx_to_count = scan_vocabulary(
                sents, tokenizer, min_count)
        >>> C = adjacent_cooccurrence_graph(sents, tokenizer, vocab_to_idx)
        >>> X = c_to_x(C, vocab_to_idx)
    """
    rows = []
    cols = []
    data = []
    n = len(C)
    for i, (vocab1, vocab2s) in enumerate(C.items()):
        if verbose and i % 1000 == 0:
            print('\rtransforming dict to sparse {} / {}'.format(i, n), end='')
        vocab1 = vocab_to_idx[vocab1]
        for vocab2, count in vocab2s.items():
            if count < min_cooccurrence:
                continue
            vocab2 = vocab_to_idx[vocab2]
            rows.append(vocab1)
            cols.append(vocab2)
            data.append(count)
    if verbose:
        print('\rtransforming dict to sparse was done{}'.format(' '*20))
    n_vocabs = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_vocabs, n_vocabs))

def sents_to_adjacent_graph(sents, tokenizer=lambda s:s.split(),
    min_count=2, min_cooccurrence=1, verbose=False):
    """
    :param sents: list of str
        Sentences
    :param tokenizer: callable
        tokenizer(str) = list of str
    :param min_count: int
        Minumum term frequency
    :param min_cooccurrence: int
        Minumum co-occurrence frequency
    :param verbose: Boolean
        If True, verbose on

    It returns
    ----------
    X : scipy.sparse.csr_matrix
        sparse matrix format
    idx_to_vocab : list of str
        Vocabulary list

    Usage
    -----
        >>> X, idx_to_vocab = sents_to_adjacent_graph(sents)
    """

    idx_to_vocab, vocab_to_idx, idx_to_count = scan_vocabulary(
        sents, tokenizer, min_count)

    C = adjacent_cooccurrence_graph(sents, tokenizer, vocab_to_idx, verbose)

    X = c_to_x(C, vocab_to_idx, min_cooccurrence, verbose)

    return X, idx_to_vocab
