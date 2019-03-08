import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, embedding_dim, num_words, num_filters,
                 num_classes, pretrained_wordvec, dropout_ratio=0.5):

        super(TextCNN, self).__init__()

        self.embed = nn.Embedding(num_words, embedding_dim)

        # check word embedding vector shape
        n, m = pretrained_wordvec.shape
        assert n == num_words and m == embedding_dim

        self.embed.weight.data.copy_(torch.from_numpy(pretrained_wordvec))

        # in_channels, out_channels,
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(2, embedding_dim), bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(3, embedding_dim), bias=False)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(4, embedding_dim), bias=False)

        self.dropout = nn.Dropout(dropout_ratio)
        # three type convolution. each conv. has num_filters
        self.fc1 = nn.Linear(3 * num_filters, num_classes)

    def forward(self, x):
        """x: sentence image
                torch.LongTensor"""

        out = self.embed(x) # (batch, sent_len, embed_dim)
        out = out.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)

        # three convolution filter
        out = [F.relu(self.conv1(out)).squeeze(3),
               F.relu(self.conv2(out)).squeeze(3),
               F.relu(self.conv3(out)).squeeze(3)]

        # 1 - max pooling for each conv
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]

        # concatenation
        out = torch.cat(out, 1)

        # dropout
        out = self.dropout(out)

        # fully connected neural network
        logit = self.fc1(out) # (batch, target_size)

        return logit
