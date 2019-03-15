from collections import Counter
import torch


def scan_vocabulary(sents, min_count=1, padding=None):
    """
    :param sents: list of list of str
        Sentences. Each sentence is tokenized.
    :param min_count: int
        Minimum frequency of words
    :param padding: str or None
        Use padding as pecial token if padding is not None

    Usage
    -----
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sent, min_count=1, padding='<padding>')
    """

    counter = Counter(term for sent in sents for term in sent)
    counter = {vocab:count for vocab, count in counter.items() if count >= min_count}
    idx_to_vocab = [vocab for vocab in sorted(counter, key=lambda x:-counter[x])]
    if padding is not None:
        idx_to_vocab.append(padding)
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def encode_sequence(sequence, vocab_to_idx, max_len=-1):
    """
    :param sequence: sequence of vocabs
    :param vocab_to_idx: {str:int}
    :param max_len: int
        If max_len is positive, it fills padding idx to end of sequence
        If max_len is positive and the length of sequence is longer than max_len,
        it uses first max_len terms

    Notes
    -----
    `padding idx` is len(vocab_to_idx) - 1
    `unknown_idx` is len(vocab_to_idx)
    """

    padding_idx = len(vocab_to_idx) - 1
    unknown_idx = len(vocab_to_idx)

    sequence_idx = [vocab_to_idx.get(vocab, unknown_idx) for vocab in sequence]
    seq_len = len(sequence_idx)
    if max_len > 0 and seq_len < max_len:
        sequence_idx += [padding_idx] * (max_len - seq_len)
    elif max_len > 0:
        sequence_idx = sequence_idx[:max_len]

    sequence_idx = torch.LongTensor(sequence_idx)
    return sequence_idx

def decode_sequence(sequence_idx, idx_to_sequence):
    """
    :param sequence_idx: encoded id sequence
    :param idx_to_sequence: list of str
        Vocabulary index

    Usage
    -----
        seq = decode_sequence(seq_idx, idx_to_vocab)
    """

    def get(idx, n_vocab):
        if not (0 <= idx < n_vocab):
            return None
        return idx_to_sequence[idx]

    n_vocab = len(idx_to_sequence)
    sequence = [get(idx, n_vocab) for idx in sequence_idx]
    return sequence
