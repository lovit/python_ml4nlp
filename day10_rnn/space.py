from collections import Counter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def correct(sent, char_to_idx, model):
    """
    :param sent: str
        Input sentence
    :param char_to_idx: dict
        Mapper from character to index
    :param model: torch.nn.Module
        Trained space correction model

    It returns
    ----------
    sent_ : str
        Space corrected sentence
    """

    x, y = sent_to_xy(sent, char_to_idx)
    tags = torch.argmax(model(x), dim=1).numpy()
    chars = sent.replace(' ','')
    sent_ = ''.join([c if t == 0 else (c + ' ') for c, t in zip(chars, tags)]).strip()
    return sent_

def space_tag(sent, nonspace=0, space=1):
    """
    :param sent: str
        Input sentence
    :param nonspace: Object
        Non-space tag. Default is 0, int type
    :param space: Object
        Space tag. Default is 1, int type

    It returns
    ----------
    chars : list of character
    tags : list of tag

    (example)
        sent  = '이건 예시문장입니다'
        chars = list('이건예시문장입니다')
        tags  = [0,1,0,0,0,0,0,0,1]
    """

    sent = sent.strip()
    chars = list(sent.replace(' ',''))
    tags = [nonspace]*(len(chars) - 1) + [space]
    idx = 0

    for c in sent:
        if c == ' ':
            tags[idx-1] = space
        else:
            idx += 1

    return chars, tags

def to_idx(item, mapper, unknown=None):
    """
    :param item: Object
        Object to be encoded
    :param mapper: dict
        Dictionary from item to idx
    :param unknown: int
        Index of unknown item. If None, use len(mapper)

    It returns
    ----------
    idx : int
        Index of item
    """

    if unknown is None:
        unknown = len(mapper)
    return mapper.get(item, unknown)

def sent_to_xy(sent, char_to_idx):
    """
    :param sent: str
        Input sentence
    :param char_to_idx: dict
        Dictionary from character to index

    It returns
    ----------
    idxs : torch.LongTensor
        Encoded character sequence
    tags : torch.LongTensor
        Space tag sequence
    """

    chars, tags = space_tag(sent)
    idxs = torch.LongTensor(
        [to_idx(c, char_to_idx) for c in chars])
    tags = torch.LongTensor(tags)
    return idxs, tags


class GRUSpace(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
        bias=True, dropout=0, bidirectional=False):

        num_layers = 1 # ignore

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
            num_layers = num_layers,
            bias = bias,
            dropout = dropout,
            bidirectional=bidirectional)
        self.hidden2tag = nn.Linear(hidden_dim * (1 + self.bidirectional), tagset_size)

    def forward(self, char_idxs):
        embeds = self.embeddings(char_idxs)
        hidden = self.init_hidden()
        lstm_out, hidden = self.gru(embeds.view(len(char_idxs), 1, -1), hidden)
        tag_space = self.hidden2tag(lstm_out.view(char_idxs.size()[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(1 + self.bidirectional, 1, self.hidden_dim)